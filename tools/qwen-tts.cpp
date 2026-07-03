// qwen-tts.cpp: thin CLI wrapper around the qwentts.cpp public ABI.
// Parses arguments, reads the optional reference WAV plus transcript,
// hands off to qt_synthesize and writes the resulting waveform as a
// WAV file. All synthesis logic, mode validation and seed resolution
// live behind the qwen_* facade declared in qwen.h.
//
// Talker variants: 0.6B-Base / 0.6B-CustomVoice / 1.7B-Base /
// 1.7B-CustomVoice / 1.7B-VoiceDesign. The decoder path is selected
// from GGUF metadata at qt_init time. The CLI surface mirrors the
// omnivoice.cpp tooling: kebab-case flags, --format wav16/wav24/wav32,
// -o '-' streams to stdout, --seed -1 means non deterministic
// (resolved inside qt_synthesize), the utterance text is read from stdin.

#include "audio-io.h"
#include "qwen.h"
#include "rvq-file.h"
#include "utf8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", qt_version());
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV (24 kHz mono). '-' streams to stdout (pipe friendly).\n\n"
            "Input:\n"
            "  stdin                   Target text to synthesise. Read fully then synthesised in one\n"
            "                          shot, or line by line with --stream-by-line.\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --instruct <str>        Style instruction. Required for VoiceDesign, optional for\n"
            "                          CustomVoice, rejected for Base\n"
            "  --speaker <name>        Speaker name (CustomVoice only)\n"
            "  --ref-wav <path>        Reference WAV for voice cloning (Base only)\n"
            "  --ref-spk <path>        Pre-extracted speaker embedding from qwen-codec --talker\n"
            "                          (replaces --ref-wav, Base only)\n"
            "  --ref-rvq <path>        Pre-encoded reference codes from qwen-codec (requires\n"
            "                          --ref-spk and --ref-text, enables ICL clone mode)\n"
            "  --ref-text <path>       Transcript file for the reference (enables ICL clone mode)\n"
            "  --max-new <n>           Max new audio frames (default: 2048)\n"
            "  --codec-chunk-dur <f>   Codec decode chunk duration in seconds (default: 24.0)\n"
            "  --codec-left-dur <f>    Codec decode left context duration in seconds (default: 2.0)\n"
            "  --stream-by-line        Flush synthesis at each newline, one WAV header per line (-o '-')\n\n"
            "Sampling:\n"
            "  --seed <int>            Sampling seed (default: -1 for random)\n"
            "  --greedy                Disable stochastic sampling on both stacks\n"
            "  --temp <f>              Talker temperature (default: 0.9)\n"
            "  --top-k <n>             Talker top-k (default: 50, 0 disables)\n"
            "  --top-p <f>             Talker top-p (default: 1.0)\n"
            "  --rep-pen <f>           Talker repetition penalty (default: 1.05)\n"
            "  --sub-temp <f>          Sub-talker temperature (default: 0.9)\n"
            "  --sub-top-k <n>         Sub-talker top-k (default: 50)\n"
            "  --sub-top-p <f>         Sub-talker top-p (default: 1.0)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors (f32) to <dir>\n",
            prog);
}

struct Args {
    const char * model;
    const char * codec;
    const char * lang;
    const char * instruct;
    const char * speaker;
    const char * ref_wav;
    const char * ref_spk;
    const char * ref_rvq;
    const char * ref_text_path;
    const char * dump_dir;
    const char * out_wav;
    const char * format;
    int          max_new_tokens;
    int64_t      seed;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    int          subtalker_top_k;
    float        subtalker_top_p;
    float        subtalker_temperature;
    bool         subtalker_do_sample;
    bool         use_fa;
    bool         clamp_fp16;
    bool         stream_by_line;
    float        codec_chunk_sec;
    float        codec_left_context_sec;
};

// Read all of stdin into a string. Binary mode on Windows so UTF-16 input
// survives CRLF translation, then normalised to UTF-8. Trims trailing
// newlines so a piped text file behaves like clean utterance input.
static std::string read_stdin_text() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    utf8_normalize(s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file into a string, normalised to UTF-8.
// Trims trailing newlines.
// 11 bits per code (V <= 2048), matching qwen-codec.
static const int RVQ_CODE_BITS = 11;

// Read a .spk file: raw f32 values, the count IS the embedding dimension.
static bool read_spk_file(const char * path, std::vector<float> & emb) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CLI] ERROR: cannot open --ref-spk '%s'\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % (long) sizeof(float)) != 0) {
        fprintf(stderr, "[CLI] ERROR: --ref-spk '%s' size %ld is not a positive multiple of 4\n", path, sz);
        fclose(f);
        return false;
    }
    emb.resize((size_t) sz / sizeof(float));
    if (fread(emb.data(), sizeof(float), emb.size(), f) != emb.size()) {
        fprintf(stderr, "[CLI] ERROR: short read on --ref-spk '%s'\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

static bool read_text_file(const char * path, std::string & out) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CLI] FATAL: cannot open '%s'\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: ftell failed on '%s'\n", path);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: fread failed on '%s'\n", path);
        return false;
    }
    fclose(f);
    utf8_normalize(out);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

static bool parse_args(int argc, char ** argv, Args & a) {
    a                        = {};
    a.lang                   = "auto";
    a.format                 = "wav16";
    a.max_new_tokens         = 2048;
    a.seed                   = -1;
    a.do_sample              = true;
    a.temperature            = 0.9f;
    a.top_k                  = 50;
    a.top_p                  = 1.0f;
    a.repetition_penalty     = 1.05f;
    a.subtalker_do_sample    = true;
    a.subtalker_top_k        = 50;
    a.subtalker_top_p        = 1.0f;
    a.subtalker_temperature  = 0.9f;
    a.use_fa                 = true;
    a.clamp_fp16             = false;
    a.stream_by_line         = false;
    a.codec_chunk_sec        = 24.0f;
    a.codec_left_context_sec = 2.0f;
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            return false;
        }
        if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            a.model = argv[++i];
        } else if (std::strcmp(arg, "--codec") == 0 && i + 1 < argc) {
            a.codec = argv[++i];
        } else if (std::strcmp(arg, "--lang") == 0 && i + 1 < argc) {
            a.lang = argv[++i];
        } else if (std::strcmp(arg, "--instruct") == 0 && i + 1 < argc) {
            a.instruct = argv[++i];
        } else if (std::strcmp(arg, "--speaker") == 0 && i + 1 < argc) {
            a.speaker = argv[++i];
        } else if (std::strcmp(arg, "--ref-wav") == 0 && i + 1 < argc) {
            a.ref_wav = argv[++i];
        } else if (std::strcmp(arg, "--ref-spk") == 0 && i + 1 < argc) {
            a.ref_spk = argv[++i];
        } else if (std::strcmp(arg, "--ref-rvq") == 0 && i + 1 < argc) {
            a.ref_rvq = argv[++i];
        } else if (std::strcmp(arg, "--ref-text") == 0 && i + 1 < argc) {
            a.ref_text_path = argv[++i];
        } else if (std::strcmp(arg, "--format") == 0 && i + 1 < argc) {
            a.format = argv[++i];
        } else if (std::strcmp(arg, "--dump") == 0 && i + 1 < argc) {
            a.dump_dir = argv[++i];
        } else if (std::strcmp(arg, "--max-new") == 0 && i + 1 < argc) {
            a.max_new_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            a.seed = (int64_t) std::atoll(argv[++i]);
        } else if (std::strcmp(arg, "--greedy") == 0) {
            // Greedy mode: argmax sampling on both stacks. The sampling
            // fast path in sampling.h uses temperature <= 0 to short
            // circuit to argmax, bypassing rep penalty and top-k/p
            // truncation, which exactly mirrors the Python reference
            // greedy behaviour used by tests/debug-tts-cossim.py.
            a.do_sample           = false;
            a.subtalker_do_sample = false;
        } else if (std::strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            a.temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            a.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            a.top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--rep-pen") == 0 && i + 1 < argc) {
            a.repetition_penalty = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-temp") == 0 && i + 1 < argc) {
            a.subtalker_temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-k") == 0 && i + 1 < argc) {
            a.subtalker_top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-p") == 0 && i + 1 < argc) {
            a.subtalker_top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--no-fa") == 0) {
            a.use_fa = false;
        } else if (std::strcmp(arg, "--clamp-fp16") == 0) {
            a.clamp_fp16 = true;
        } else if (std::strcmp(arg, "--stream-by-line") == 0) {
            a.stream_by_line = true;
        } else if (std::strcmp(arg, "--codec-chunk-dur") == 0 && i + 1 < argc) {
            a.codec_chunk_sec = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--codec-left-dur") == 0 && i + 1 < argc) {
            a.codec_left_context_sec = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "-o") == 0 && i + 1 < argc) {
            a.out_wav = argv[++i];
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown or incomplete argument: %s\n", arg);
            return false;
        }
    }
    return a.model && a.codec;
}

static int run(const Args & a) {
    // Init the facade. The seven mode validations
    // (base / custom_voice / voice_design rules) and the BPE tokenizer
    // load live inside qt_init / qt_synthesize; the CLI just hands
    // off the two GGUF paths and reports qt_last_error on failure.
    qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = a.model;
    iparams.codec_path  = a.codec;
    iparams.use_fa      = a.use_fa;
    iparams.clamp_fp16  = a.clamp_fp16;

    qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        return 1;
    }

    // Load the reference transcript from the file passed via --ref-text.
    // Empty content is rejected since the ICL prompt needs it non empty.
    std::string  ref_text_buf;
    const char * ref_text = NULL;
    if (a.ref_text_path) {
        if (!read_text_file(a.ref_text_path, ref_text_buf)) {
            qt_free(q);
            return 1;
        }
        if (ref_text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: --ref-text file '%s' is empty\n", a.ref_text_path);
            qt_free(q);
            return 1;
        }
        ref_text = ref_text_buf.c_str();
    }

    // Decode the reference WAV once, mono at the codec sample rate. The
    // facade consumes the buffer directly so the WAV is read exactly
    // once regardless of how many encoders need the audio (speaker
    // encoder embedding + codec encoder RVQ codes for ICL mode B).
    std::unique_ptr<float, void (*)(void *)> raw_holder(NULL, std::free);
    const float *                            ref_audio_24k = NULL;
    int                                      ref_n_samples = 0;
    if (a.ref_wav) {
        int     T_in = 0;
        float * raw  = audio_read_mono(a.ref_wav, 24000, &T_in);
        if (!raw || T_in <= 0) {
            fprintf(stderr, "[CLI] ERROR: cannot read --ref-wav '%s'\n", a.ref_wav);
            if (raw) {
                std::free(raw);
            }
            qt_free(q);
            return 1;
        }
        raw_holder.reset(raw);
        ref_audio_24k = raw;
        ref_n_samples = T_in;
    }

    // Latent reference files. The .spk holds raw f32 values whose count
    // IS the embedding dimension; the .rvq holds the packed ICL code
    // matrix. The facade validates the structural constraints (mutual
    // exclusions, dim match against the talker hidden size).
    std::vector<float>   ref_spk_emb;
    std::vector<int32_t> ref_codes;
    int                  ref_T = 0;
    if (a.ref_spk) {
        if (!read_spk_file(a.ref_spk, ref_spk_emb)) {
            qt_free(q);
            return 1;
        }
        fprintf(stderr, "[CLI] Reference SPK: %s, %zu f32 values\n", a.ref_spk, ref_spk_emb.size());
    }
    if (a.ref_rvq) {
        const int K = qt_num_codebooks(q);
        if (!rvq_read_file(a.ref_rvq, K, RVQ_CODE_BITS, ref_codes, &ref_T)) {
            qt_free(q);
            return 1;
        }
        fprintf(stderr, "[CLI] Reference RVQ: %s, K=%d T=%d\n", a.ref_rvq, K, ref_T);
    }

    // Resolve output WAV format string: wav16 / wav24 / wav32. Default
    // wav16 mirrors the omnivoice.cpp default.
    WavFormat wav_fmt;
    if (!audio_parse_format(a.format, wav_fmt)) {
        fprintf(stderr, "[CLI] ERROR: invalid --format '%s' (expected wav16, wav24, wav32)\n", a.format);
        qt_free(q);
        return 1;
    }

    // Streaming detection : -o '-' writes a wide RIFF header to stdout
    // up front and pipes encoded samples chunk by chunk through the
    // on_chunk callback as the AR loop produces frames. Any other path
    // (or no -o) takes the buffered route so the file gets accurate
    // sizes in its header.
    const char * out_path         = a.out_wav ? a.out_wav : "out.wav";
    const bool   stream_to_stdout = (out_path[0] == '-' && out_path[1] == '\0');
    const bool   line_mode        = a.stream_by_line && stream_to_stdout;

    // Resolve utterance text: read stdin fully. Empty stdin triggers a
    // clean error. Inline text on the command line is intentionally not
    // supported: shell quoting and special characters belong in a file
    // or piped input, never in argv. Line mode reads stdin line by line
    // in the streaming loop below instead.
    std::string text_buf;
    if (!line_mode) {
        text_buf = read_stdin_text();
        if (text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: stdin is empty, nothing to synthesise\n");
            qt_free(q);
            return 1;
        }
    }
    const char * text = text_buf.c_str();

    // Translate CLI args into the facade params. Seed -1 is forwarded
    // verbatim and resolved by qt_synthesize via std::random_device.
    qt_tts_params params;
    qt_tts_default_params(&params);
    params.text                   = text;
    params.lang                   = a.lang;
    params.instruct               = a.instruct;
    params.speaker                = a.speaker;
    params.ref_audio_24k          = ref_audio_24k;
    params.ref_n_samples          = ref_n_samples;
    params.ref_text               = ref_text;
    params.ref_spk_emb            = ref_spk_emb.empty() ? NULL : ref_spk_emb.data();
    params.ref_spk_dim            = (int) ref_spk_emb.size();
    params.ref_codes              = ref_codes.empty() ? NULL : ref_codes.data();
    params.ref_T                  = ref_T;
    params.seed                   = a.seed;
    params.max_new_tokens         = a.max_new_tokens;
    params.do_sample              = a.do_sample;
    params.temperature            = a.temperature;
    params.top_k                  = a.top_k;
    params.top_p                  = a.top_p;
    params.repetition_penalty     = a.repetition_penalty;
    params.subtalker_do_sample    = a.subtalker_do_sample;
    params.subtalker_temperature  = a.subtalker_temperature;
    params.subtalker_top_k        = a.subtalker_top_k;
    params.subtalker_top_p        = a.subtalker_top_p;
    params.dump_dir               = a.dump_dir;
    params.codec_chunk_sec        = a.codec_chunk_sec;
    params.codec_left_context_sec = a.codec_left_context_sec;

    if (stream_to_stdout) {
        wav_stream ws = {};
        if (!wav_stream_open_stdout(&ws, 24000, wav_fmt)) {
            qt_free(q);
            return 1;
        }
        params.on_chunk = [](const float * s, int n, void * ud) -> bool {
            return wav_stream_write((wav_stream *) ud, s, n);
        };
        params.on_chunk_user_data = &ws;

        if (line_mode) {
            // Line oriented streaming: every stdin line synthesises
            // immediately as its own utterance, model and speaker staying
            // resident across lines. Each utterance after the first opens
            // with a fresh RIFF header so a client can split the stream
            // into one standalone WAV per line on the RIFF magic. The
            // header is armed when a line completes and consumed lazily at
            // the next synthesis, so empty lines and a trailing newline
            // never emit an orphan header. Lines are text, an embedded NUL
            // truncates the read at strlen.
            char        buf[4096];
            std::string line;
            bool        need_header = false;
            int         n_lines     = 0;
#if defined(_WIN32)
            _setmode(_fileno(stdin), _O_BINARY);
#endif
            while (true) {
                const bool eof = (fgets(buf, sizeof(buf), stdin) == nullptr);
                if (!eof) {
                    line += buf;
                }
                const bool complete = !line.empty() && line.back() == '\n';
                if (complete || (eof && !line.empty())) {
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                        line.pop_back();
                    }
                    if (!line.empty()) {
                        if (need_header) {
                            if (!wav_stream_write_header(&ws)) {
                                qt_free(q);
                                return 1;
                            }
                            need_header = false;
                        }
                        params.text      = line.c_str();
                        qt_audio  audio  = {};
                        qt_status status = qt_synthesize(q, &params, &audio);
                        qt_audio_free(&audio);
                        if (status != QT_STATUS_OK) {
                            fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
                            wav_stream_close(&ws);
                            qt_free(q);
                            return 1;
                        }
                        n_lines++;
                        need_header = true;
                    }
                    line.clear();
                }
                if (eof) {
                    break;
                }
            }
            wav_stream_close(&ws);
            qt_free(q);
            fprintf(stderr, "[Pipeline] Streamed %d lines to <stdout>\n", n_lines);
            return 0;
        }

        qt_audio  audio  = {};
        qt_status status = qt_synthesize(q, &params, &audio);
        wav_stream_close(&ws);
        if (status != QT_STATUS_OK) {
            fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
            qt_audio_free(&audio);
            qt_free(q);
            return 1;
        }
        qt_audio_free(&audio);
        qt_free(q);
        fprintf(stderr, "[Pipeline] Streamed to <stdout>\n");
        return 0;
    }

    qt_audio  audio  = {};
    qt_status status = qt_synthesize(q, &params, &audio);
    if (status != QT_STATUS_OK) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        qt_audio_free(&audio);
        qt_free(q);
        return 1;
    }

    if (audio.n_samples > 0) {
        if (!audio_write_wav(out_path, audio.samples, audio.n_samples, audio.sample_rate, wav_fmt)) {
            fprintf(stderr, "[Pipeline] FATAL: WAV write failed for %s\n", out_path);
            qt_audio_free(&audio);
            qt_free(q);
            return 1;
        }
        fprintf(stderr, "[Pipeline] Wrote %d samples (%.2f s) -> %s\n", audio.n_samples,
                (double) audio.n_samples / (double) audio.sample_rate, out_path);
    }

    qt_audio_free(&audio);
    qt_free(q);
    return 0;
}

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    // The facade absorbs every std::exception thrown deep in the load
    // and synthesis chains, converting them into qt_status + a
    // qt_last_error message. No top-level try / catch needed here :
    // the CLI just reads the status returned by qt_init /
    // qt_synthesize and renders qt_last_error to stderr.
    return run(a);
}
