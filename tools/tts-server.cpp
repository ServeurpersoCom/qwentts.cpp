// tts-server.cpp: OpenAI-compatible HTTP server backed by the qwentts
// ABI. Loads a talker + codec once, GPU resident, and serves synthesis over
// POST /v1/audio/speech. The shared core lives in src/tts-server.h ; this
// file only wires the qt_* ABI into the generic adapter.

#include "tts-server.h"

#include "qwen.h"
#include "version.h"
#include "utf8.h"

#include <cstdio>
#include <cstring>
#include <string>

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
            "  --ref-spk <path>        Pre-extracted speaker embedding from qwen-codec --talker (Base only)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n",
            prog);
}

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

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

int main(int argc, char ** argv) {
    typedef std::map<std::string, std::vector<float>> spk_map;
    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   lang        = "auto";
    spk_map       spks;
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
        } else if (!std::strcmp(arg, "--ref-spk") && i + 1 < argc) {
            spks[argv[++i]] = std::vector<float>();
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

    for (spk_map::iterator i = spks.begin(); i != spks.end(); ++i) {
        if (!read_spk_file(i->first.c_str(), i->second)) {
            return 1;
        }
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
    for (spk_map::iterator i = spks.begin(); i != spks.end(); ++i) {
        be.voices.push_back(i->first.c_str());
    }

    // The adapter always drives the streaming pipeline : on_chunk routes to
    // the shared sink, which either streams to the socket (pcm) or fills a
    // one-shot buffer (wav). Either way the audio path is identical.
    be.synthesize = [q, &lang, &spks](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.text = req.input.c_str();
        p.lang = lang.c_str();
        if (!req.voice.empty()) {
            if (spks.find(req.voice) != spks.end()) {
                std::vector<float> &ref_spk_emb = spks[req.voice];
                p.ref_spk_emb = ref_spk_emb.empty() ? NULL : ref_spk_emb.data();
                p.ref_spk_dim = (int) ref_spk_emb.size();
            } else if (qt_n_speakers(q) > 0) {
                p.speaker = req.voice.c_str();
            }
        }
        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        p.on_chunk                = [](const float * s, int ns, void * u) -> bool {
            return (*static_cast<const tts_sink *>(u))(s, ns);
        };
        p.on_chunk_user_data = (void *) sink_ptr;

        struct qt_audio out = {};
        enum qt_status  rc  = qt_synthesize(q, &p, &out);
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
