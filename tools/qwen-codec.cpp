// qwen-codec.cpp: codec CLI for Qwen3-TTS.
//
// Encode a 24 kHz mono WAV into RVQ codes (.rvq), or decode RVQ codes
// back into a 24 kHz mono float32 WAV. Mode is inferred from the input
// file extension: .wav in -> encode, .rvq in -> decode. Output is
// auto-named next to the input file by swapping the extension.
//
// Encode truncates the input to the hop boundary, strictly conforming
// to the qwen-tts --ref-wav ICL path, so a .rvq produced here feeds
// qwen-tts --ref-rvq directly. Passing --talker additionally runs the
// speaker encoder from the talker GGUF on the full input and writes
// the x-vector embedding next to the .rvq as a .spk file (raw f32,
// enc_dim values), feeding qwen-tts --ref-spk.
//
// File format (.rvq): flat code stream packed at 11 bits per code,
// LSB-first, no header. Layout is [K, T] row-major. K is fixed by the
// codec config in the GGUF (16 codebooks for the 12Hz tokenizer,
// codebook_size = 2048). T is the frame count derived from filesize.

#include "audio-io.h"
#include "backend.h"
#include "gguf-weights.h"
#include "pipeline-codec.h"
#include "rvq-file.h"
#include "speaker-encoder-extract.h"
#include "speaker-encoder-weights.h"
#include "utf8.h"
#include "version.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> [-i <input>] [--talker <gguf>] [--format <fmt>]\n\n"
            "Required:\n"
            "  --model <gguf>          Codec GGUF (qwen-tokenizer-12hz-*.gguf)\n\n"
            "Optional:\n"
            "  -i <path>               Input. WAV -> encode, .rvq -> decode\n"
            "  --talker <gguf>         Talker GGUF (Base only). Encode also extracts the speaker\n"
            "                          embedding and writes it next to the .rvq as a .spk file\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.\n"
            "Encode truncates to the hop boundary, conforming to the qwen-tts --ref-wav path:\n"
            "the .rvq feeds qwen-tts --ref-rvq, the .spk feeds qwen-tts --ref-spk.\n"
            "When -i is omitted, runs a load self-test of the codec GGUF.\n",
            prog);
}

// Replace or append extension on a path string.
static std::string swap_ext(const std::string & path, const char * ext) {
    size_t dot = path.find_last_of('.');
    size_t sep = path.find_last_of("/\\");
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) {
        return path.substr(0, dot) + ext;
    }
    return path + ext;
}

// 0: unsupported, 1: encode (.wav in), 2: decode (.rvq in).
static int infer_mode(const char * path) {
    size_t n = strlen(path);
    if (n >= 4 && strcmp(path + n - 4, ".wav") == 0) {
        return 1;
    }
    if (n >= 4 && strcmp(path + n - 4, ".rvq") == 0) {
        return 2;
    }
    return 0;
}

// Load the speaker encoder from the talker GGUF, run it on the full input
// buffer and write the embedding as a raw f32 .spk file (enc_dim values,
// validated by filesize on the qwen-tts side). Returns 0 on success.
static int extract_spk(const char *  talker_path,
                       BackendPair   bp,
                       const float * audio,
                       int           n_samples,
                       const char *  out_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, talker_path)) {
        fprintf(stderr, "[Codec] FATAL: cannot open talker GGUF %s\n", talker_path);
        return 1;
    }

    SpeakerEncoderWeights sw = {};
    if (!speaker_encoder_weights_load(&sw, gf, bp.backend)) {
        fprintf(stderr, "[Codec] FATAL: speaker encoder load failed from %s\n", talker_path);
        gf_close(&gf);
        return 1;
    }
    gf_close(&gf);
    if (sw.weight_buf == NULL) {
        fprintf(stderr, "[Codec] FATAL: %s has no speaker encoder (Base only)\n", talker_path);
        return 1;
    }

    ggml_backend_sched_t sched   = backend_sched_new(bp, 4096);
    const int            enc_dim = sw.enc_dim;
    std::vector<float>   emb;
    bool                 ok = speaker_encoder_extract(&sw, sched, audio, n_samples, emb);
    ggml_backend_sched_free(sched);
    speaker_encoder_weights_free(&sw);
    if (!ok || (int) emb.size() != enc_dim) {
        fprintf(stderr, "[Codec] FATAL: speaker embedding extraction failed (%zu values, enc_dim %d)\n", emb.size(),
                enc_dim);
        return 1;
    }

    FILE * f = utf8_fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "[Codec] FATAL: cannot open %s for write\n", out_path);
        return 1;
    }
    if (fwrite(emb.data(), sizeof(float), emb.size(), f) != emb.size()) {
        fprintf(stderr, "[Codec] FATAL: short write on %s\n", out_path);
        fclose(f);
        return 1;
    }
    fclose(f);

    fprintf(stderr, "[Codec] Wrote %s: %zu f32 values (%zu bytes)\n", out_path, emb.size(), emb.size() * sizeof(float));
    return 0;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    if (argc <= 1) {
        print_usage(argv[0]);
        return 1;
    }

    const char * model_path  = NULL;
    const char * input_path  = NULL;
    const char * talker_path = NULL;
    WavFormat    wav_fmt     = WAV_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--talker") == 0 && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            if (!audio_parse_format(argv[++i], wav_fmt)) {
                fprintf(stderr, "[CLI] ERROR: unknown format: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    int mode = 0;
    if (input_path) {
        mode = infer_mode(input_path);
        if (mode == 0) {
            fprintf(stderr, "[CLI] ERROR: %s: unsupported extension (expect .wav or .rvq)\n", input_path);
            return 1;
        }
    }

    BackendPair bp = backend_init("Codec");
    if (!bp.backend) {
        fprintf(stderr, "[Codec] FATAL: backend init failed\n");
        return 1;
    }

    PipelineCodec pc = {};
    if (!pipeline_codec_load(&pc, model_path, bp)) {
        backend_release(bp.backend, bp.cpu_backend);
        return 1;
    }

    int rc = 0;

    if (!input_path) {
        fprintf(stderr, "[Codec] Load self-test passed\n");
    } else if (mode == 1) {
        // Encode .wav -> .rvq (+ .spk with --talker)
        const std::string out_str = swap_ext(input_path, ".rvq");

        int     T_in     = 0;
        float * audio_in = audio_read_mono(input_path, TOKENIZER_SAMPLE_RATE, &T_in);
        if (!audio_in || T_in < TOKENIZER_HOP_LENGTH) {
            fprintf(stderr, "[Codec] FATAL: cannot read %s or input shorter than one hop (%d samples)\n", input_path,
                    TOKENIZER_HOP_LENGTH);
            free(audio_in);
            rc = 1;
        } else {
            // Truncate to a multiple of HOP_LENGTH, strictly conforming to
            // the qwen-tts --ref-wav ICL path.
            int hop       = TOKENIZER_HOP_LENGTH;
            int T_aligned = (T_in / hop) * hop;
            int T_frames  = T_aligned / hop;

            fprintf(stderr, "[Codec] Encode: %s, %d samples @ %d Hz, truncated to %d (%d frames @ 12.5 Hz, %.2f s)\n",
                    input_path, T_in, TOKENIZER_SAMPLE_RATE, T_aligned, T_frames,
                    (double) T_aligned / (double) TOKENIZER_SAMPLE_RATE);

            std::vector<int32_t> codes = pipeline_codec_encode(&pc, audio_in, T_aligned);
            if (codes.empty()) {
                fprintf(stderr, "[Codec] FATAL: encode failed\n");
                rc = 1;
            } else if (!rvq_write_file(out_str.c_str(), codes, TOKENIZER_CODE_BITS)) {
                rc = 1;
            } else {
                fprintf(stderr, "[Codec] Wrote %s: K=%d T=%d, %zu codes -> %zu packed bytes\n", out_str.c_str(),
                        TOKENIZER_NUM_CODEBOOKS, T_frames, codes.size(),
                        (codes.size() * (size_t) TOKENIZER_CODE_BITS + 7) / 8);
            }

            // Speaker embedding extraction, conforming to the qwen-tts
            // --ref-wav mode A path: the encoder consumes the FULL input
            // buffer, never the hop-truncated one.
            if (rc == 0 && talker_path) {
                rc = extract_spk(talker_path, bp, audio_in, T_in, swap_ext(input_path, ".spk").c_str());
            }
            free(audio_in);
        }
    } else {
        // Decode .rvq -> .wav
        const std::string out_str = swap_ext(input_path, ".wav");

        std::vector<int32_t> codes;
        int                  T = 0;
        if (!rvq_read_file(input_path, TOKENIZER_NUM_CODEBOOKS, TOKENIZER_CODE_BITS, codes, &T)) {
            rc = 1;
        } else {
            fprintf(stderr, "[Codec] Decode: %s, K=%d T=%d (%.2f s)\n", input_path, TOKENIZER_NUM_CODEBOOKS, T,
                    (double) (T * TOKENIZER_HOP_LENGTH) / (double) TOKENIZER_SAMPLE_RATE);

            std::vector<float> audio = pipeline_codec_decode(&pc, codes.data(), TOKENIZER_NUM_CODEBOOKS, T);
            if (audio.empty()) {
                fprintf(stderr, "[Codec] FATAL: decode failed\n");
                rc = 1;
            } else if (!audio_write_wav(out_str.c_str(), audio.data(), (int) audio.size(), TOKENIZER_SAMPLE_RATE,
                                        wav_fmt)) {
                fprintf(stderr, "[Codec] FATAL: cannot write %s\n", out_str.c_str());
                rc = 1;
            } else {
                fprintf(stderr, "[Codec] Wrote %s: %d samples @ %d Hz, %.2f s\n", out_str.c_str(), (int) audio.size(),
                        TOKENIZER_SAMPLE_RATE, (double) audio.size() / (double) TOKENIZER_SAMPLE_RATE);
            }
        }
    }

    pipeline_codec_free(&pc);
    backend_release(bp.backend, bp.cpu_backend);
    return rc;
}
