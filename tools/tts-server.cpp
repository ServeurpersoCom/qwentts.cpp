// tts-server.cpp: OpenAI-compatible HTTP server backed by the qwentts
// ABI. Loads a talker + codec once, GPU resident, and serves synthesis over
// POST /v1/audio/speech. The shared core lives in src/tts-server.h ; this
// file only wires the qt_* ABI into the generic adapter.

#include "tts-server.h"

#include "qwen.h"
#include "rvq-file.h"
#include "version.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// Packed .rvq code width, fixed by the Qwen3-TTS 12 Hz codec (2048
// entries per codebook).
static const int RVQ_CODE_BITS = 11;

// One registered cloned voice: the extraction latents in the ABI
// ownership contract (malloc-owned, released by qt_voice_ref_free) plus
// the reference transcript that enables ICL clone mode when present.
struct voice_entry {
    struct qt_voice_ref ref;
    std::string         ref_text;
};

// Registered voices, name keyed. Every access happens under
// g_synth_mutex: registration touches the GPU through the extraction
// path and lookups run inside the already serialized synthesize.
static std::unordered_map<std::string, voice_entry> g_voices;

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n",
            prog);
}

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

int main(int argc, char ** argv) {
    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   lang        = "auto";
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!talker_path || !codec_path) {
        print_usage(argv[0]);
        return 0;
    }

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = talker_path;
    iparams.codec_path  = codec_path;
    iparams.use_fa      = use_fa;
    iparams.clamp_fp16  = clamp_fp16;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    tts_backend be;
    be.model_id = basename_of(talker_path);
    int n       = qt_n_speakers(q);
    for (int i = 0; i < n; i++) {
        be.voices.push_back(qt_speaker_name(q, i));
    }

    // Voice registry: POST /v1/voices stores a cloned voice either from a
    // WAV (server side extraction through qt_extract_voice_ref) or from
    // pre-extracted .spk / .rvq payloads. Re-registering a name replaces
    // the previous entry.
    be.register_voice = [q](const tts_voice_upload & up, std::string & err) -> bool {
        voice_entry entry;
        entry.ref      = {};
        entry.ref_text = up.ref_text;

        if (!up.wav.empty()) {
            int     T   = 0;
            float * pcm = audio_read_mono_buf((const uint8_t *) up.wav.data(), up.wav.size(), 24000, &T);
            if (!pcm) {
                err = "cannot decode the WAV payload";
                return false;
            }
            enum qt_status rc;
            {
                std::lock_guard<std::mutex> lock(g_synth_mutex);
                rc = qt_extract_voice_ref(q, pcm, T, &entry.ref);
            }
            free(pcm);
            if (rc != QT_STATUS_OK) {
                err = qt_last_error();
                return false;
            }
        } else {
            if (up.spk.size() % sizeof(float) != 0 || up.spk.empty()) {
                err = "'spk_b64' must decode to a positive multiple of 4 bytes";
                return false;
            }
            std::vector<int32_t> codes;
            int                  ref_T = 0;
            const int            K     = qt_num_codebooks(q);
            if (!rvq_read_buf((const uint8_t *) up.rvq.data(), up.rvq.size(), K, RVQ_CODE_BITS, codes, &ref_T)) {
                err = "'rvq_b64' does not decode to a valid packed code stream";
                return false;
            }
            entry.ref.ref_spk_dim = (int) (up.spk.size() / sizeof(float));
            entry.ref.ref_spk_emb = (float *) malloc(up.spk.size());
            std::memcpy(entry.ref.ref_spk_emb, up.spk.data(), up.spk.size());
            entry.ref.ref_T         = ref_T;
            entry.ref.num_codebooks = K;
            entry.ref.ref_codes     = (int32_t *) malloc(codes.size() * sizeof(int32_t));
            std::memcpy(entry.ref.ref_codes, codes.data(), codes.size() * sizeof(int32_t));
        }

        std::lock_guard<std::mutex> lock(g_synth_mutex);
        auto                        it = g_voices.find(up.name);
        if (it != g_voices.end()) {
            qt_voice_ref_free(&it->second.ref);
            g_voices.erase(it);
        }
        fprintf(stderr, "[Server] voice '%s' registered (T=%d, ref_text=%s)\n", up.name.c_str(), entry.ref.ref_T,
                entry.ref_text.empty() ? "no" : "yes");
        g_voices.emplace(up.name, std::move(entry));
        return true;
    };

    be.remove_voice = [](const std::string & name) -> bool {
        std::lock_guard<std::mutex> lock(g_synth_mutex);
        auto                        it = g_voices.find(name);
        if (it == g_voices.end()) {
            return false;
        }
        qt_voice_ref_free(&it->second.ref);
        g_voices.erase(it);
        return true;
    };

    be.registered_voices = []() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(g_synth_mutex);
        std::vector<std::string>    names;
        names.reserve(g_voices.size());
        for (const auto & kv : g_voices) {
            names.push_back(kv.first);
        }
        return names;
    };

    // pcm drives the streaming pipeline : on_chunk routes each decoded
    // frame to the shared sink for lowest latency. wav leaves on_chunk
    // unset so the pipeline runs the buffered chunked codec path (batch
    // decode, talker uninterrupted) and the whole utterance pushes to
    // the sink once. A registered voice wins over a model speaker of
    // the same name and injects the pre-extracted reference latents. A
    // name matching neither is rejected instead of silently generating
    // voiceless.
    be.synthesize = [q, &lang](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = lang.c_str();

        auto vit = req.voice.empty() ? g_voices.end() : g_voices.find(req.voice);
        if (vit != g_voices.end()) {
            const voice_entry & v = vit->second;
            p.ref_spk_emb         = v.ref.ref_spk_emb;
            p.ref_spk_dim         = v.ref.ref_spk_dim;
            if (!v.ref_text.empty() && v.ref.ref_codes) {
                p.ref_codes = v.ref.ref_codes;
                p.ref_T     = v.ref.ref_T;
                p.ref_text  = v.ref_text.c_str();
            }
        } else if (!req.voice.empty() && qt_n_speakers(q) > 0) {
            p.speaker = req.voice.c_str();
        } else if (!req.voice.empty()) {
            err = "unknown voice '" + req.voice + "'";
            return (int) QT_STATUS_INVALID_PARAMS;
        }
        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }

        // Sampling overrides ride straight into the ABI; the subtalker
        // mirrors the talker knobs so the HTTP surface stays a single
        // coherent set. A temperature of zero selects greedy decoding
        // on both.
        p.seed = req.seed;
        if (req.max_new_tokens != -1) {
            p.max_new_tokens = req.max_new_tokens;
        }
        if (req.top_k != -1) {
            p.top_k           = req.top_k;
            p.subtalker_top_k = req.top_k;
        }
        if (!std::isnan(req.temperature)) {
            if (req.temperature == 0.0f) {
                p.do_sample           = false;
                p.subtalker_do_sample = false;
            } else {
                p.temperature           = req.temperature;
                p.subtalker_temperature = req.temperature;
            }
        }
        if (!std::isnan(req.top_p)) {
            p.top_p           = req.top_p;
            p.subtalker_top_p = req.top_p;
        }
        if (!std::isnan(req.repetition_penalty)) {
            p.repetition_penalty = req.repetition_penalty;
        }

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        if (req.format == "pcm") {
            p.on_chunk = [](const float * s, int ns, void * u) -> bool {
                return (*static_cast<const tts_sink *>(u))(s, ns);
            };
            p.on_chunk_user_data = (void *) sink_ptr;
        }

        struct qt_audio out = {};
        enum qt_status  rc  = qt_synthesize(q, &p, &out);
        if (rc == QT_STATUS_OK && p.on_chunk == NULL && out.n_samples > 0) {
            sink(out.samples, out.n_samples);
        }
        qt_audio_free(&out);
        if (rc != QT_STATUS_OK) {
            err = qt_last_error();
            return (int) rc;
        }
        return 0;
    };

    int rc = tts_server_run(be, cfg);
    qt_free(q);
    return rc;
}
