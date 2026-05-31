// qwen-tts-server.cpp: OpenAI-compatible HTTP server for qwentts.cpp.
//
// Loads the talker + codec GGUFs once at startup, keeps them GPU-resident,
// and serves synthesis requests via POST /v1/audio/speech.
//
// Usage:
//   qwen-tts-server --model <talker.gguf> --codec <codec.gguf> [options]
//
// Options:
//   --host <addr>           Bind address (default: 0.0.0.0)
//   --port <n>              Bind port (default: 8080)
//   --api-key <key>         API key for Authorization header (can repeat)
//   --no-fa                 Disable flash attention
//   --clamp-fp16            Clamp hidden states to FP16 range

#include "server-tts.h"
#include "audio-io.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

// If *arg* names a readable file, return its contents (trailing newline
// stripped).  Otherwise return *arg* verbatim so the caller can treat it
// as literal text.
static std::string resolve_text_or_file(const std::string & arg) {
    if (arg.empty()) return arg;
    if (access(arg.c_str(), R_OK) == 0) {
        std::FILE * f = std::fopen(arg.c_str(), "r");
        if (f) {
            std::string content;
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
                content.append(buf, n);
            }
            std::fclose(f);
            // Strip a single trailing newline (convenience)
            if (!content.empty() && content.back() == '\n') {
                content.pop_back();
            }
            fprintf(stderr, "[INFO] Loaded ref-transcript from file: %s (%zu chars)\n", arg.c_str(), content.size());
            return content;
        }
    }
    return arg;
}

/* ------------------------------------------------------------------ */
/*  Signal handling                                                    */
/* ------------------------------------------------------------------ */

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running.store(false);
}

/* ------------------------------------------------------------------ */
/*  Audio encoding via miniaudio                                       */
/* ------------------------------------------------------------------ */

std::string encode_audio(
    const float * samples, int n_samples, int sample_rate,
    const std::string & format,
    std::string & err_msg)
{
    // WAV — use existing qwentts encoder (fast, no deps)
    if (format == "wav" || format == "pcm") {
        return audio_encode_wav(samples, n_samples, sample_rate, WAV_S16);
    }

    // FLAC / MP3 / OGG — use miniaudio encoder to a temp file, then read back
    ma_encoding_format enc_format = ma_encoding_format_unknown;
    std::string ext;

    if (format == "flac") {
        enc_format = ma_encoding_format_flac;
        ext = ".flac";
    }
#ifdef MA_ENABLE_MP3
    else if (format == "mp3") {
        enc_format = ma_encoding_format_mp3;
        ext = ".mp3";
    }
#endif
#ifdef MA_ENABLE_VORBIS
    else if (format == "ogg") {
        enc_format = ma_encoding_format_vorbis;
        ext = ".ogg";
    }
#endif
    else {
        err_msg = "Unsupported response format: " + format +
                  " (supported: wav, flac"
#ifdef MA_ENABLE_MP3
                  ", mp3"
#endif
#ifdef MA_ENABLE_VORBIS
                  ", ogg"
#endif
                  ")";
        return "";
    }

    // Create temp file
    std::string tmp_path = "/tmp/qwen-tts-XXXXXX" + ext;
    int fd = open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_EXCL, 0600);
    if (fd < 0) {
        err_msg = "Failed to create temp file for encoding";
        return "";
    }

    ma_encoder encoder;
    ma_encoder_config config = ma_encoder_config_init(
        enc_format, ma_format_f32, 1, (ma_uint32)sample_rate);

    ma_result result = ma_encoder_init_file(tmp_path.c_str(), &config, &encoder);
    if (result != MA_SUCCESS) {
        close(fd);
        unlink(tmp_path.c_str());
        err_msg = "miniaudio: failed to init " + format + " encoder";
        return "";
    }

    ma_uint64 frames_written = 0;
    result = ma_encoder_write_pcm_frames(&encoder, samples, (ma_uint64)n_samples, &frames_written);
    ma_encoder_uninit(&encoder);
    close(fd);

    if (result != MA_SUCCESS) {
        unlink(tmp_path.c_str());
        err_msg = "miniaudio: failed to write " + format + " frames";
        return "";
    }

    // Read back the encoded file
    std::FILE * f = std::fopen(tmp_path.c_str(), "rb");
    unlink(tmp_path.c_str()); // unlink immediately, file is still open
    if (!f) {
        err_msg = "Failed to read back encoded " + format + " file";
        return "";
    }

    std::string encoded;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        encoded.append(buf, n);
    }
    std::fclose(f);

    if (encoded.empty()) {
        err_msg = "Encoded " + format + " file is empty";
        return "";
    }

    return encoded;
}

/* ------------------------------------------------------------------ */
/*  Model metadata                                                     */
/* ------------------------------------------------------------------ */

void TTSModel::populate_metadata()
{
    size_t pos = talker_path.find("qwen-talker-");
    if (pos != std::string::npos) {
        std::string fname = talker_path.substr(pos);
        if (fname.find("0.6b") != std::string::npos || fname.find("0.6B") != std::string::npos)
            model_size = "0.6B";
        else if (fname.find("1.7b") != std::string::npos || fname.find("1.7B") != std::string::npos)
            model_size = "1.7B";

        if (fname.find("customvoice") != std::string::npos || fname.find("custom_voice") != std::string::npos)
            model_type = "custom_voice";
        else if (fname.find("voicedesign") != std::string::npos || fname.find("voice_design") != std::string::npos)
            model_type = "voice_design";
        else
            model_type = "base";
    }
}

/* ------------------------------------------------------------------ */
/*  HTTP handlers                                                      */
/* ------------------------------------------------------------------ */

static json make_error(const std::string & message, const std::string & type = "server_error") {
    return json{
        {"error", json{
            {"message", message},
            {"type", type},
            {"param", nullptr},
            {"code", nullptr}
        }}
    };
}

// Load a reference audio file into mono float PCM at 24kHz.
// Supports WAV and FLAC via miniaudio.
static std::vector<float> load_ref_audio(const std::string & path, std::string & err_msg) {
    std::vector<float> samples;
    ma_decoder decoder;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 24000);

    ma_result result = ma_decoder_init_file(path.c_str(), &config, &decoder);
    if (result != MA_SUCCESS) {
        err_msg = "Failed to load reference audio: " + path;
        return samples;
    }

    ma_uint64 total_frames = 0;
    ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    if (total_frames == 0) {
        ma_decoder_uninit(&decoder);
        err_msg = "Reference audio is empty: " + path;
        return samples;
    }

    samples.resize(total_frames);
    ma_uint64 frames_read = 0;
    result = ma_decoder_read_pcm_frames(&decoder, samples.data(), total_frames, &frames_read);
    ma_decoder_uninit(&decoder);

    if (result != MA_SUCCESS || frames_read == 0) {
        samples.clear();
        err_msg = "Failed to read reference audio: " + path;
        return samples;
    }

    return samples;
}

static void handle_speech(
    const TTSModel & model,
    const ServerConfig & config,
    const std::vector<float> & ref_audio,
    const httplib::Request & req,
    httplib::Response & res)
{
    // Parse JSON body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        res.status = 400;
        res.set_content(make_error(std::string("Invalid JSON: ") + e.what(), "invalid_request_error").dump(), "application/json");
        return;
    }

    // Required: input (text)
    if (!body.contains("input") || !body["input"].is_string() || body["input"].get<std::string>().empty()) {
        res.status = 400;
        res.set_content(make_error("'input' must be a non-empty string", "invalid_request_error").dump(), "application/json");
        return;
    }

    // Optional params — CLI defaults overridden by request body
    std::string voice       = body.value("voice", config.default_speaker.empty() ? config.default_instruct : std::string(""));
    std::string format      = body.value("response_format", std::string("wav"));
    float       speed       = body.value("speed", 1.0f);

    // Validate format
    bool format_ok = (format == "wav" || format == "flac" || format == "pcm"
#ifdef MA_ENABLE_MP3
                      || format == "mp3"
#endif
#ifdef MA_ENABLE_VORBIS
                      || format == "ogg"
#endif
    );
    if (!format_ok) {
        res.status = 400;
        res.set_content(make_error(
            "Invalid response_format. Supported: wav, flac"
#ifdef MA_ENABLE_MP3
            ", mp3"
#endif
#ifdef MA_ENABLE_VORBIS
            ", ogg"
#endif
            , "invalid_request_error").dump(), "application/json");
        return;
    }

    // Speed: warn if not 1.0 (not natively supported)
    if (speed < 0.25f || speed > 4.0f) {
        res.status = 400;
        res.set_content(make_error("speed must be between 0.25 and 4.0", "invalid_request_error").dump(), "application/json");
        return;
    }
    if (std::abs(speed - 1.0f) > 0.001f) {
        fprintf(stderr, "[WARN] speed=%.2f is not natively supported, using 1.0\n", speed);
    }

    // Build tts params
    struct qt_tts_params params;
    qt_tts_default_params(&params);

    // Keep strings alive for the duration of qt_synthesize (params holds raw pointers)
    std::string input_text = body["input"].get<std::string>();
    params.text = input_text.c_str();

    // lang: qwentts pipeline passes null lang to prompt_builder_build which
    // expects std::string& — construction from null throws. Use empty string.
    static const char lang_auto[] = "auto";
    params.lang = lang_auto;

    // Voice → speaker (for custom_voice) or instruct (for voice_design)
    if (!voice.empty()) {
        if (model.model_type == "voice_design") {
            params.instruct = voice.c_str();
        } else if (model.model_type == "custom_voice") {
            // Validate speaker name against loaded speaker table
            void * names_void = NULL;
            int n = qt_get_speakers(model.ctx, &names_void);
            const char ** names = (const char **) names_void;
            bool found = false;
            for (int i = 0; i < n; i++) {
                if (voice == names[i]) { found = true; break; }
            }
            if (!found) {
                // Build list of valid names for the error message
                std::string valid_list;
                for (int i = 0; i < n; i++) {
                    if (i > 0) valid_list += ", ";
                    valid_list += names[i];
                }
                res.status = 400;
                res.set_content(make_error(
                    "Unknown voice: '" + voice + "'. Available voices: " + valid_list,
                    "invalid_request_error").dump(), "application/json");
                return;
            }
            params.speaker = voice.c_str();
        }
        // base mode: voice param ignored (use ref_audio instead)
    } else if (model.model_type == "custom_voice" && !config.default_speaker.empty()) {
        params.speaker = config.default_speaker.c_str();
    } else if (model.model_type == "voice_design" && !config.default_instruct.empty()) {
        params.instruct = config.default_instruct.c_str();
    }

    // Voice cloning: reference audio (base mode)
    if (!ref_audio.empty()) {
        params.ref_audio_24k = ref_audio.data();
        params.ref_n_samples = (int)ref_audio.size();
        if (!config.default_ref_transcript.empty()) {
            params.ref_text = config.default_ref_transcript.c_str();
        }
        // ref_text left as default (nullptr OK when ref_audio not set,
        // but when ref_audio IS set, qwentts needs it — omit it and let
        // qt_tts_default_params keep the safe default)
    }

    // Log request
    auto t0 = std::chrono::steady_clock::now();
    std::string input_preview = body["input"].get<std::string>();
    if (input_preview.size() > 80) input_preview = input_preview.substr(0, 80) + "...";
    fprintf(stderr, "[REQ] text=%d chars format=%s voice=%s\n",
            (int)body["input"].get<std::string>().size(), format.c_str(), voice.empty() ? "(default)" : voice.c_str());

    // Synthesize
    struct qt_audio audio = {};
    audio.samples = nullptr;
    audio.n_samples = 0;
    audio.sample_rate = 0;
    audio.channels = 0;

    enum qt_status status = qt_synthesize(model.ctx, &params, &audio);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (status != QT_STATUS_OK) {
        std::string err = qt_last_error() ? qt_last_error() : "Synthesis failed";
        fprintf(stderr, "[ERR] %s (%.0fms)\n", err.c_str(), elapsed_ms);
        res.status = 500;
        res.set_content(make_error(err, "server_error").dump(), "application/json");
        return;
    }

    // Encode
    std::string err_msg;
    std::string encoded = encode_audio(audio.samples, audio.n_samples, audio.sample_rate, format, err_msg);
    qt_audio_free(&audio);

    if (encoded.empty()) {
        fprintf(stderr, "[ERR] %s (%.0fms)\n", err_msg.c_str(), elapsed_ms);
        res.status = 500;
        res.set_content(make_error(err_msg, "server_error").dump(), "application/json");
        return;
    }

    fprintf(stderr, "[OK] %d samples %.0fms %.1fkB %s\n",
            audio.n_samples, elapsed_ms, (double)encoded.size() / 1024, format.c_str());

    // Set content type
    std::string content_type;
    if (format == "wav" || format == "pcm") content_type = "audio/wav";
    else if (format == "flac") content_type = "audio/flac";
#ifdef MA_ENABLE_MP3
    else if (format == "mp3") content_type = "audio/mpeg";
#endif
#ifdef MA_ENABLE_VORBIS
    else if (format == "ogg") content_type = "audio/ogg";
#endif
    else content_type = "application/octet-stream";

    res.set_content(encoded, content_type);
}

static void handle_models(
    const TTSModel & model,
    const httplib::Request & req,
    httplib::Response & res)
{
    (void)req;

    std::string model_id = "qwen3-tts-" + model.model_size + "-" + model.model_type;

    json models = json{
        {"object", "list"},
        {"data", json{
            json{
                {"id", model_id},
                {"object", "model"},
                {"created", 1700000000},
                {"owned_by", "qwentts"},
                {"permission", json::array()},
                {"root", model_id},
                {"parent", nullptr}
            }
        }}
    };

    res.set_content(models.dump(), "application/json");
}

static void handle_health(
    const httplib::Request & req,
    httplib::Response & res)
{
    (void)req;
    res.set_content(R"({"status":"ok","model_loaded":true})", "application/json");
}

static void handle_voices(
    const TTSModel & model,
    const httplib::Request & req,
    httplib::Response & res)
{
    (void)req;

    json voices = json::array();

    if (model.model_type == "custom_voice") {
        void * names_void = NULL;
        int n = qt_get_speakers(model.ctx, &names_void);
        const char ** names = (const char **) names_void;
        for (int i = 0; i < n; i++) {
            voices.push_back(json{
                {"id", names[i]},
                {"name", names[i]},
                {"type", "built_in"}
            });
        }
    } else if (model.model_type == "voice_design") {
        voices.push_back(json{
            {"id", "custom"},
            {"name", "custom"},
            {"type", "text_description"},
            {"description", "Pass any voice description in the 'voice' field"}
        });
    } else if (model.model_type == "base") {
        voices.push_back(json{
            {"id", "cloned"},
            {"name", "cloned"},
            {"type", "reference_audio"},
            {"description", "Voice cloned from --ref-audio at startup"}
        });
    }

    json result = json{
        {"object", "list"},
        {"data", voices}
    };

    res.set_content(result.dump(), "application/json");
}

/* ------------------------------------------------------------------ */
/*  Server bootstrap                                                   */
/* ------------------------------------------------------------------ */

int run_server(ServerConfig config)
{
    // Load model
    fprintf(stderr, "Loading talker: %s\n", config.talker_path.c_str());
    fprintf(stderr, "Loading codec:  %s\n", config.codec_path.c_str());

    struct qt_init_params init_params;
    qt_init_default_params(&init_params);
    init_params.talker_path = config.talker_path.c_str();
    init_params.codec_path  = config.codec_path.c_str();
    init_params.use_fa      = config.use_fa;
    init_params.clamp_fp16  = config.clamp_fp16;

    struct qt_context * ctx = qt_init(&init_params);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", qt_last_error());
        return -1;
    }

    TTSModel model;
    model.ctx         = ctx;
    model.talker_path = config.talker_path;
    model.codec_path  = config.codec_path;
    model.populate_metadata();

    fprintf(stderr, "Model loaded: %s %s\n", model.model_size.c_str(), model.model_type.c_str());

    // Load reference audio (voice cloning)
    std::vector<float> ref_audio;
    if (!config.default_ref_audio.empty()) {
        std::string err_msg;
        ref_audio = load_ref_audio(config.default_ref_audio, err_msg);
        if (ref_audio.empty()) {
            fprintf(stderr, "Error: %s\n", err_msg.c_str());
            qt_free(ctx);
            return -1;
        }
        fprintf(stderr, "Loaded reference audio: %zu samples @ 24kHz\n", ref_audio.size());
    }

    // Start HTTP server
    httplib::Server svr;

    // API key middleware
    if (!config.api_keys.empty()) {
        svr.set_pre_routing_handler([&config](const httplib::Request & req, httplib::Response & res) {
            // Public endpoints
            if (req.path == "/health" || req.path == "/v1/health" ||
                req.path == "/models" || req.path == "/v1/models" ||
                req.path == "/voices" || req.path == "/v1/voices") {
                return httplib::Server::HandlerResponse::Unhandled;
            }

            std::string auth = req.get_header_value("Authorization");
            if (auth.size() > 7 && auth.substr(0, 7) == "Bearer ") {
                auth = auth.substr(7);
            }

            bool valid = false;
            for (const auto & key : config.api_keys) {
                if (auth == key) { valid = true; break; }
            }
            if (!valid) {
                res.status = 401;
                res.set_content(R"({"error":{"message":"Invalid API key","type":"auth_error"}})", "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // CORS preflight
    svr.Options("/.*", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
    });

    // Routes
    svr.Post ("/v1/audio/speech",        [&model, &config, &ref_audio](const auto & req, auto & res) { handle_speech(model, config, ref_audio, req, res); });
    svr.Get  ("/v1/models",              [&model](const auto & req, auto & res) { handle_models(model, req, res); });
    svr.Get  ("/models",                 [&model](const auto & req, auto & res) { handle_models(model, req, res); });
    svr.Get  ("/v1/voices",              [&model](const auto & req, auto & res) { handle_voices(model, req, res); });
    svr.Get  ("/voices",                 [&model](const auto & req, auto & res) { handle_voices(model, req, res); });
    svr.Get  ("/health",                 [](const auto & req, auto & res) { handle_health(req, res); });
    svr.Get  ("/v1/health",              [](const auto & req, auto & res) { handle_health(req, res); });

    // Graceful shutdown via signal
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Shutdown endpoint
    svr.Get("/_shutdown", [](const auto &, auto &) { g_running.store(false); });

    fprintf(stderr, "Server listening on %s:%d\n", config.host.c_str(), config.port);

    // Start server in a thread so we can poll g_running
    std::thread server_thread([&svr, &config]() {
        if (!svr.listen(config.host.c_str(), config.port)) {
            fprintf(stderr, "Failed to start server on %s:%d\n", config.host.c_str(), config.port);
            g_running.store(false);
        }
    });

    // Poll for shutdown
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    svr.stop();
    server_thread.join();
    qt_free(ctx);
    fprintf(stderr, "Server shut down.\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/*  CLI                                                                */
/* ------------------------------------------------------------------ */

static void print_usage(const char * prog) {
    fprintf(stderr, "qwen-tts-server %s\n\n", qt_version());
    fprintf(stderr,
            "Usage: %s --model <talker.gguf> --codec <codec.gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Server:\n"
            "  --host <addr>           Bind address (default: 0.0.0.0)\n"
            "  --port <n>              Bind port (default: 8080)\n"
            "  --api-key <key>         Require API key (can repeat)\n\n"
            "Voice (defaults, overridden per-request):\n"
            "  --speaker <name>        Default speaker name (custom_voice)\n"
            "  --instruct <text>       Default voice instruction (voice_design)\n"
            "  --ref-audio <file>      Reference audio for voice cloning (base mode)\n"
            "  --ref-transcript <text|file>  Transcript of ref audio (string or file path)\n\n"
            "Inference:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n",
            prog);
}

int main(int argc, char ** argv)
{
    ServerConfig config;
    std::string talker_path, codec_path;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            config.host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            config.api_keys.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--speaker") == 0 && i + 1 < argc) {
            config.default_speaker = argv[++i];
        } else if (strcmp(argv[i], "--instruct") == 0 && i + 1 < argc) {
            config.default_instruct = argv[++i];
        } else if (strcmp(argv[i], "--ref-audio") == 0 && i + 1 < argc) {
            config.default_ref_audio = argv[++i];
        } else if (strcmp(argv[i], "--ref-transcript") == 0 && i + 1 < argc) {
            config.default_ref_transcript = resolve_text_or_file(argv[++i]);
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            config.use_fa = false;
        } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
            config.clamp_fp16 = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (talker_path.empty() || codec_path.empty()) {
        fprintf(stderr, "Error: --model and --codec are required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    config.talker_path = talker_path;
    config.codec_path  = codec_path;

    return run_server(config);
}
