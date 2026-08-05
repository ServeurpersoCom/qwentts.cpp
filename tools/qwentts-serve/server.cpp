#include "server.h"
#include "server-http.h"

#include "qwen.h"
#include "version.h"
#include "audio-io.h"
#include "yyjson.h"

#include <atomic>
#include <clocale>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;
static std::atomic<bool> g_cancel_requested{false};

static void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        _Exit(1);
    }
    shutdown_handler(signal);
}

static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = "{\"error\":{\"message\":\"" + std::string(e.what()) + "\",\"type\":\"invalid_request_error\",\"code\":400}}";
            return res;
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 500;
            res->data = "{\"error\":{\"message\":\"" + std::string(e.what()) + "\",\"type\":\"server_error\",\"code\":500}}";
            return res;
        } catch (...) {
            auto res = std::make_unique<server_http_res>();
            res->status = 500;
            res->data = "{\"error\":{\"message\":\"unknown error\",\"type\":\"server_error\",\"code\":500}}";
            return res;
        }
    };
}

static std::string basename_of(const std::string & path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Server:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n\n"
            "Voice / Input:\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --instruct <str>        Default style instruction (overridable per request)\n"
            "  --speaker <name>        Default speaker name (overridable per request)\n"
            "  --ref-wav <path>        Reference WAV for voice cloning (Base only)\n"
            "  --ref-text <path>       Transcript for the reference (ICL clone mode)\n\n"
            "Sampling:\n"
            "  --seed <int>            Sampling seed (default: -1 for random)\n"
            "  --greedy                Disable stochastic sampling on both stacks\n"
            "  --temp <f>              Talker temperature (default: 0.9)\n"
            "  --top-k <n>             Talker top-k (default: 50)\n"
            "  --top-p <f>             Talker top-p (default: 1.0)\n"
            "  --rep-pen <f>           Talker repetition penalty (default: 1.05)\n"
            "  --sub-temp <f>          Sub-talker temperature (default: 0.9)\n"
            "  --sub-top-k <n>         Sub-talker top-k (default: 50)\n"
            "  --sub-top-p <f>         Sub-talker top-p (default: 1.0)\n\n"
            "Codec:\n"
            "  --codec-chunk-dur <f>   Codec decode chunk duration in seconds (default: 5.0)\n\n"
             "Debug:\n"
             "  --device <name>         Force a specific backend device (default: auto).\n"
             "                          Use \"cpu\" or \"none\" for CPU only.\n"
             "  --list-devices          List available devices and exit\n"
             "  --dump <dir>            Dump intermediate tensors to <dir>\n"
             "  --no-fa                 Disable flash attention\n"
             "  --clamp-fp16            Clamp hidden states to FP16 range\n",
            prog);
}

// --- JSON helpers ---

static std::string json_error_body(int status, const char * type, const char * message) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * err = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, err, "message", message);
    yyjson_mut_obj_add_str(doc, err, "type", type);
    yyjson_mut_obj_add_val(doc, root, "error", err);
    char * json = yyjson_mut_write(doc, 0, NULL);
    std::string result = json ? json : "{}";
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
    return result;
}

// --- File I/O helpers ---

static bool read_text_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return false; }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f); return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

// --- Synthesis state shared by all route handlers ---

struct synth_state {
    qt_context * q;
    std::string  lang;
    std::string  model_id;
    // Default synthesis params (from CLI, overridable per request)
    std::string  default_instruct;
    std::string  default_speaker;
    int64_t      seed;
    int          max_new_tokens;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    bool         subtalker_do_sample;
    float        subtalker_temperature;
    int          subtalker_top_k;
    float        subtalker_top_p;
    std::string  dump_dir;
    // Reference audio (loaded at startup from --ref-wav / --ref-text)
    float *      ref_audio_24k;
    int          ref_n_samples;
    std::string  ref_text;
};
static std::mutex g_synth_mutex;

// --- Audio helpers ---

static inline int16_t f32_to_s16(float x) {
    float v = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    return (int16_t) lrintf(v * 32767.0f);
}

static void append_s16le(std::string & out, const float * samples, int n_samples) {
    size_t base = out.size();
    out.resize(base + (size_t) n_samples * 2);
    char * p = &out[base];
    for (int i = 0; i < n_samples; i++) {
        int16_t s = f32_to_s16(samples[i]);
        *p++      = (char) ((uint16_t) s & 0xff);
        *p++      = (char) (((uint16_t) s >> 8) & 0xff);
    }
}

// --- Route handler implementations ---

static server_http_res_ptr handle_health(const server_http_req &) {
    auto res = std::make_unique<server_http_res>();
    res->data = R"({"status":"ok"})";
    return res;
}

static server_http_res_ptr handle_models(const synth_state & st, const server_http_req &) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "object", "list");
    yyjson_mut_val * data = yyjson_mut_arr(doc);
    yyjson_mut_val * one  = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, one, "id", st.model_id.c_str());
    yyjson_mut_obj_add_str(doc, one, "object", "model");
    yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
    yyjson_mut_arr_add_val(data, one);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    char * json = yyjson_mut_write(doc, 0, NULL);
    auto res = std::make_unique<server_http_res>();
    res->data = json ? json : "{}";
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
    return res;
}

static server_http_res_ptr handle_voices(const synth_state & st, const server_http_req &) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    int n = qt_n_speakers(st.q);
    for (int i = 0; i < n; i++) {
        yyjson_mut_val * one = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, one, "name", qt_speaker_name(st.q, i));
        yyjson_mut_arr_add_val(arr, one);
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);
    char * json = yyjson_mut_write(doc, 0, NULL);
    auto res = std::make_unique<server_http_res>();
    res->data = json ? json : "{}";
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
    return res;
}

// --- TTS request parsing ---

struct tts_request {
    std::string input;
    std::string voice;
    std::string instructions;
    std::string format;
    float       speed;
};

static bool parse_tts_request(const std::string & body, tts_request & req, std::string & err) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        err = "request body is not valid JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        err = "request body must be a JSON object";
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * input = yyjson_obj_get(root, "input");
    if (!yyjson_is_str(input) || yyjson_get_len(input) == 0) {
        err = "'input' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    req.input = yyjson_get_str(input);

    yyjson_val * model = yyjson_obj_get(root, "model");
    (void)model;

    yyjson_val * voice = yyjson_obj_get(root, "voice");
    req.voice          = yyjson_is_str(voice) ? yyjson_get_str(voice) : "";

    yyjson_val * instructions = yyjson_obj_get(root, "instructions");
    req.instructions          = yyjson_is_str(instructions) ? yyjson_get_str(instructions) : "";

    yyjson_val * fmt = yyjson_obj_get(root, "response_format");
    req.format       = yyjson_is_str(fmt) ? yyjson_get_str(fmt) : "pcm";

    yyjson_val * speed = yyjson_obj_get(root, "speed");
    req.speed          = yyjson_is_num(speed) ? (float) yyjson_get_num(speed) : 1.0f;

    yyjson_doc_free(doc);

    if (req.format != "pcm" && req.format != "wav") {
        req.format = "wav";
    }
    return true;
}

static int run_synthesis(const synth_state & st, const tts_request & req,
                         const std::function<bool(const float *, int)> & sink, std::string & err) {
    struct qt_tts_params p;
    qt_tts_default_params(&p);
    p.text = req.input.c_str();
    p.lang = st.lang.c_str();

    // Speaker: request body overrides CLI default
    if (!req.voice.empty() && qt_n_speakers(st.q) > 0) {
        p.speaker = req.voice.c_str();
    } else if (!st.default_speaker.empty()) {
        p.speaker = st.default_speaker.c_str();
    }

    // Instructions: request body overrides CLI default
    if (!req.instructions.empty()) {
        p.instruct = req.instructions.c_str();
    } else if (!st.default_instruct.empty()) {
        p.instruct = st.default_instruct.c_str();
    }

    // Sampling params (server-wide from CLI)
    p.seed               = st.seed;
    p.max_new_tokens     = st.max_new_tokens;
    p.do_sample          = st.do_sample;
    p.temperature        = st.temperature;
    p.top_k              = st.top_k;
    p.top_p              = st.top_p;
    p.repetition_penalty = st.repetition_penalty;
    p.subtalker_do_sample    = st.subtalker_do_sample;
    p.subtalker_temperature  = st.subtalker_temperature;
    p.subtalker_top_k        = st.subtalker_top_k;
    p.subtalker_top_p        = st.subtalker_top_p;

    // Reference audio (loaded once at startup)
    if (st.ref_audio_24k && st.ref_n_samples > 0) {
        p.ref_audio_24k = st.ref_audio_24k;
        p.ref_n_samples  = st.ref_n_samples;
        p.ref_text       = st.ref_text.empty() ? nullptr : st.ref_text.c_str();
    }

    // Dump directory
    p.dump_dir = st.dump_dir.empty() ? nullptr : st.dump_dir.c_str();

    // Cancel callback
    g_cancel_requested = false;
    p.cancel            = [](void *) -> bool { return g_cancel_requested.load(); };

    // Streaming sink
    const auto * sink_ptr = &sink;
    p.on_chunk            = [](const float * s, int ns, void * u) -> bool {
        return (*static_cast<const std::function<bool(const float *, int)> *>(u))(s, ns);
    };
    p.on_chunk_user_data = (void *) sink_ptr;

    struct qt_audio out = {};
    enum qt_status  rc  = qt_synthesize(st.q, &p, &out);
    qt_audio_free(&out);
    if (rc != QT_STATUS_OK) {
        err = qt_last_error();
        return (int) rc;
    }
    return 0;
}

static int status_to_http(int rc) {
    if (rc == 0) return 200;
    if (rc == -1 || rc == -2) return 400;
    if (rc == -5) return 499;
    return 502;
}

static server_http_res_ptr handle_speech(const synth_state & st, const server_http_req & http_req) {
    tts_request req;
    std::string parse_err;
    if (!parse_tts_request(http_req.body, req, parse_err)) {
        auto res = std::make_unique<server_http_res>();
        res->status = 400;
        res->data = json_error_body(400, "invalid_request_error", parse_err.c_str());
        return res;
    }

    if (req.format == "wav") {
        std::vector<float> buf;
        auto sink = [&buf](const float * s, int n) -> bool {
            buf.insert(buf.end(), s, s + n);
            return true;
        };
        std::string synth_err;
        int rc;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            rc = run_synthesis(st, req, sink, synth_err);
        }
        if (rc != 0) {
            auto res = std::make_unique<server_http_res>();
            res->status = status_to_http(rc);
            res->data = json_error_body(res->status, "server_error",
                                        synth_err.empty() ? "synthesis failed" : synth_err.c_str());
            return res;
        }
        auto res = std::make_unique<server_http_res>();
        res->data = audio_encode_wav(buf.data(), (int) buf.size(), 24000, WAV_S16);
        res->content_type = "audio/wav";
        return res;
    }

    // Streaming PCM — use the generator pattern via res->next
    auto res = std::make_unique<server_http_res>();
    res->content_type = "audio/pcm";
    res->headers["Cache-Control"] = "no-cache";
    res->headers["X-Accel-Buffering"] = "no";

    auto req_ptr = std::make_shared<tts_request>(std::move(req));
    auto done = std::make_shared<bool>(false);
    res->next = [st, req_ptr, done](std::string & chunk) -> bool {
        if (*done) {
            return false;
        }
        *done = true;

        std::string buf;
        auto sink = [&buf](const float * s, int n) -> bool {
            append_s16le(buf, s, n);
            return true;
        };

        std::string synth_err;
        int rc;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            rc = run_synthesis(st, *req_ptr, sink, synth_err);
        }

        if (rc != 0) {
            chunk = std::move(buf);
            return !chunk.empty();
        }

        chunk = std::move(buf);
        return true;
    };
    return res;
}

static server_http_res_ptr handle_cancel(const server_http_req &) {
    g_cancel_requested.store(true);
    auto res = std::make_unique<server_http_res>();
    res->data = R"({"success":true})";
    return res;
}

// --- Main entry point ---

int qwentts_server_main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    server_params params;
    bool show_help = false;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            params.codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            params.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            params.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            params.lang = argv[++i];
        } else if (!std::strcmp(arg, "--instruct") && i + 1 < argc) {
            params.instruct = argv[++i];
        } else if (!std::strcmp(arg, "--speaker") && i + 1 < argc) {
            params.speaker = argv[++i];
        } else if (!std::strcmp(arg, "--seed") && i + 1 < argc) {
            params.seed = (int64_t) std::atoll(argv[++i]);
        } else if (!std::strcmp(arg, "--max-new") && i + 1 < argc) {
            params.max_new_tokens = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--greedy")) {
            params.do_sample = false;
            params.subtalker_do_sample = false;
        } else if (!std::strcmp(arg, "--temp") && i + 1 < argc) {
            params.temperature = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--top-k") && i + 1 < argc) {
            params.top_k = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--top-p") && i + 1 < argc) {
            params.top_p = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--rep-pen") && i + 1 < argc) {
            params.repetition_penalty = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--sub-temp") && i + 1 < argc) {
            params.subtalker_temperature = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--sub-top-k") && i + 1 < argc) {
            params.subtalker_top_k = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--sub-top-p") && i + 1 < argc) {
            params.subtalker_top_p = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--codec-chunk-dur") && i + 1 < argc) {
            params.codec_chunk_sec = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--dump") && i + 1 < argc) {
            params.dump_dir = argv[++i];
        } else if (!std::strcmp(arg, "--ref-wav") && i + 1 < argc) {
            params.ref_wav = argv[++i];
        } else if (!std::strcmp(arg, "--ref-text") && i + 1 < argc) {
            params.ref_text = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            params.use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            params.clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--device") && i + 1 < argc) {
            params.device = argv[++i];
        } else if (!std::strcmp(arg, "--list-devices")) {
            qt_list_devices(stderr);
            return 0;
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            show_help = true;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (show_help || params.model_path.empty() || params.codec_path.empty()) {
        print_usage(argv[0]);
        return show_help ? 0 : 1;
    }

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = params.model_path.c_str();
    iparams.codec_path  = params.codec_path.c_str();
    iparams.device      = params.device.empty() ? nullptr : params.device.c_str();
    iparams.use_fa      = params.use_fa;
    iparams.clamp_fp16  = params.clamp_fp16;
    iparams.codec_chunk_sec = params.codec_chunk_sec;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    // Load reference audio / transcript (validated lazily by qt_synthesize)
    float *     ref_audio = NULL;
    int         ref_n     = 0;
    std::string ref_text_buf;
    if (!params.ref_wav.empty()) {
        ref_audio = audio_read_mono(params.ref_wav.c_str(), 24000, &ref_n);
        if (!ref_audio || ref_n <= 0) {
            fprintf(stderr, "[Server] WARNING: cannot read --ref-wav '%s', ignoring\n",
                    params.ref_wav.c_str());
            ref_audio = NULL;
            ref_n     = 0;
        }
    }
    if (!params.ref_text.empty()) {
        if (!read_text_file(params.ref_text.c_str(), ref_text_buf)) {
            fprintf(stderr, "[Server] WARNING: cannot read --ref-text '%s', ignoring\n",
                    params.ref_text.c_str());
        }
    }

    synth_state st;
    st.q                      = q;
    st.lang                   = params.lang;
    st.model_id               = basename_of(params.model_path);
    st.default_instruct       = params.instruct;
    st.default_speaker        = params.speaker;
    st.seed                   = params.seed;
    st.max_new_tokens         = params.max_new_tokens;
    st.do_sample              = params.do_sample;
    st.temperature            = params.temperature;
    st.top_k                  = params.top_k;
    st.top_p                  = params.top_p;
    st.repetition_penalty     = params.repetition_penalty;
    st.subtalker_do_sample    = params.subtalker_do_sample;
    st.subtalker_temperature  = params.subtalker_temperature;
    st.subtalker_top_k        = params.subtalker_top_k;
    st.subtalker_top_p        = params.subtalker_top_p;
    st.dump_dir               = params.dump_dir;
    st.ref_audio_24k          = ref_audio;
    st.ref_n_samples          = ref_n;
    st.ref_text               = ref_text_buf;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        fprintf(stderr, "[Server] FATAL: failed to initialize HTTP server\n");
        std::free(ref_audio);
        qt_free(q);
        return 1;
    }

    // Build route handlers (wrapped for exception safety)
    server_routes routes;
    routes.get_health  = ex_wrapper(handle_health);
    routes.get_models  = ex_wrapper([st](const server_http_req & req) { return handle_models(st, req); });
    routes.get_voices  = ex_wrapper([st](const server_http_req & req) { return handle_voices(st, req); });
    routes.post_speech = ex_wrapper([st](const server_http_req & req) { return handle_speech(st, req); });
    routes.post_cancel = ex_wrapper(handle_cancel);

    // Register routes (already wrapped, pass directly)
    ctx_http.get("/health",      routes.get_health);
    ctx_http.get("/v1/models",   routes.get_models);
    ctx_http.get("/v1/voices",   routes.get_voices);
    ctx_http.post("/v1/audio/speech",      routes.post_speech);
    ctx_http.post("/v1/audio/speech/cancel", routes.post_cancel);

    // Start HTTP server
    if (!ctx_http.start()) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", params.host.c_str(), params.port);
        std::free(ref_audio);
        qt_free(q);
        return 1;
    }
    ctx_http.is_ready.store(true);

    fprintf(stderr, "[Server] model %s\n", st.model_id.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", params.host.c_str(), params.port);

    // Signal handling
    shutdown_handler = [&](int) {
        ctx_http.stop();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }

    std::free(ref_audio);
    qt_free(q);
    return 0;
}
