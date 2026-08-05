#pragma once
// tts-server.h: shared OpenAI-compatible TTS HTTP core for the *.cpp ports.
//
// One synthesis context lives GPU resident for the process lifetime. The
// project tool fills a tts_backend adapter that wires its own ABI
// (qt_synthesize / ov_synthesize) into the generic sink, then calls
// tts_server_run. The HTTP layer, tuning, OAI parsing and audio framing
// are identical across projects ; only the adapter differs.
//
// Endpoints:
//   POST   /v1/audio/speech         OAI text-to-speech
//   GET    /v1/models               single loaded model
//   GET    /v1/audio/voices         model speakers plus registered cloned voices
//   POST   /v1/audio/voices         register a cloned voice: {name, ref_text,
//                                   wav_b64} extracts server side, {name,
//                                   ref_text, spk_b64, rvq_b64} takes
//                                   pre-extracted latents verbatim
//   DELETE /v1/audio/voices/{name}  drop a registered voice
//   GET    /health                  liveness probe
//
// Audio out: response_format "pcm" streams s16le 24 kHz mono chunked as it
// is generated (real time), "wav" returns a one-shot RIFF file. pcm is the
// default so streaming is on unless the client asks for a file.

#include "../vendor/cpp-httplib/httplib.h"
#include "audio-io.h"
#include "yyjson.h"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// One synthesis request parsed from the OAI JSON body.
struct tts_request {
    std::string input;         // text to speak
    std::string voice;         // OAI voice, mapped to a speaker by the adapter
    std::string instructions;  // OAI instructions, mapped to the ABI instruct field
    std::string format;        // "pcm" (stream) or "wav" (one-shot)
    float       speed;         // OAI speed, parsed then ignored (no time stretch in the ABI)

    // Optional sampling overrides. -1 (ints) and NaN (floats) mark a
    // field the client left unset, keeping the engine defaults.
    int64_t seed;                // forwarded verbatim, -1 draws a random seed
    int     max_new_tokens;      // strictly positive
    int     top_k;               // 0 disables the top-k filter
    float   temperature;         // 0 selects greedy decoding
    float   top_p;               // in (0, 1]
    float   repetition_penalty;  // strictly positive
};

// One voice registration parsed from the POST /v1/audio/voices JSON body.
// Exactly one payload form is present: wav holds decoded base64 WAV
// bytes for server side extraction, or spk plus rvq hold the raw
// contents of pre-extracted .spk and .rvq files. ref_text carries the
// reference transcript enabling ICL clone mode when present.
struct tts_voice_upload {
    std::string name;
    std::string ref_text;
    std::string wav;  // WAV file bytes
    std::string spk;  // .spk file bytes, raw f32 values
    std::string rvq;  // .rvq file bytes, packed codes
};

// The adapter pushes mono f32 24 kHz audio here. Returns false to abort the
// synthesis (client gone or cancellation), which propagates into the ABI
// on_chunk and stops generation.
using tts_sink = std::function<bool(const float * samples, int n_samples)>;

// Adapter implemented by each project tool.
struct tts_backend {
    std::string              model_id;  // reported by GET /v1/models
    std::vector<std::string> voices;    // reported by GET /v1/audio/voices, may be empty
    // Run synthesis. When the request streams, the adapter routes the ABI
    // on_chunk to sink ; otherwise it pushes the whole buffer once. Returns
    // the ABI status (0 on success), and fills err with the ABI message on
    // failure. The shared layer maps the status to an HTTP code.
    std::function<int(const tts_request & req, const tts_sink & sink, std::string & err)> synthesize;
    // Voice registry hooks, all optional: a null hook answers 501 on the
    // matching route. register_voice stores or replaces a cloned voice,
    // remove_voice drops one (false when absent), registered_voices lists
    // the current names for GET /v1/audio/voices alongside the model speakers.
    std::function<bool(const tts_voice_upload & up, std::string & err)>                   register_voice;
    std::function<bool(const std::string & name)>                                         remove_voice;
    std::function<std::vector<std::string>()>                                             registered_voices;
};

struct server_config {
    std::string host = "127.0.0.1";
    int         port = 8080;
};

// Concurrency lives behind the ABI: qt_synthesize is thread safe and
// batches concurrent requests when the context was initialised with
// max_batch > 1, so connection threads call the backend directly.
static httplib::Server * g_svr = nullptr;

static void tts_on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// Clamp to [-1, 1] and scale to s16. lrintf rounds to nearest, ties to even.
static inline int16_t tts_f32_to_s16(float x) {
    float v = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    return (int16_t) lrintf(v * 32767.0f);
}

// Append a mono f32 block as s16le bytes onto out.
static void tts_append_s16le(std::string & out, const float * samples, int n_samples) {
    size_t base = out.size();
    out.resize(base + (size_t) n_samples * 2);
    char * p = &out[base];
    for (int i = 0; i < n_samples; i++) {
        int16_t s = tts_f32_to_s16(samples[i]);
        *p++      = (char) ((uint16_t) s & 0xff);
        *p++      = (char) (((uint16_t) s >> 8) & 0xff);
    }
}

// Write a JSON error body in the OAI error envelope and set the status.
static void tts_json_error(httplib::Response & res, int status, const char * type, const char * message) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * err = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, err, "message", message);
    yyjson_mut_obj_add_str(doc, err, "type", type);
    yyjson_mut_obj_add_val(doc, root, "error", err);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.status  = status;
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// Parse the OAI body into req. Returns false and fills err on bad input.
static bool tts_parse_request(const std::string & body, tts_request & req, std::string & err) {
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

    yyjson_val * voice = yyjson_obj_get(root, "voice");
    req.voice          = yyjson_is_str(voice) ? yyjson_get_str(voice) : "";

    yyjson_val * instructions = yyjson_obj_get(root, "instructions");
    req.instructions          = yyjson_is_str(instructions) ? yyjson_get_str(instructions) : "";

    yyjson_val * fmt = yyjson_obj_get(root, "response_format");
    req.format       = yyjson_is_str(fmt) ? yyjson_get_str(fmt) : "pcm";

    yyjson_val * speed = yyjson_obj_get(root, "speed");
    req.speed          = yyjson_is_num(speed) ? (float) yyjson_get_num(speed) : 1.0f;

    // Optional sampling overrides. A missing field keeps its unset
    // marker; a present field must be well typed and in domain.
    req.seed               = -1;
    req.max_new_tokens     = -1;
    req.top_k              = -1;
    req.temperature        = NAN;
    req.top_p              = NAN;
    req.repetition_penalty = NAN;

    auto opt_int = [&](const char * key, int64_t lo, int64_t hi, int64_t & out) -> bool {
        yyjson_val * v = yyjson_obj_get(root, key);
        if (!v) {
            return true;
        }
        if (!yyjson_is_int(v) || yyjson_get_sint(v) < lo || yyjson_get_sint(v) > hi) {
            err = std::string("'") + key + "' is out of domain";
            return false;
        }
        out = yyjson_get_sint(v);
        return true;
    };
    auto opt_num = [&](const char * key, double lo, double hi, float & out) -> bool {
        yyjson_val * v = yyjson_obj_get(root, key);
        if (!v) {
            return true;
        }
        if (!yyjson_is_num(v) || yyjson_get_num(v) < lo || yyjson_get_num(v) > hi) {
            err = std::string("'") + key + "' is out of domain";
            return false;
        }
        out = (float) yyjson_get_num(v);
        return true;
    };

    int64_t max_new = -1;
    int64_t top_k   = -1;
    bool    ok = opt_int("seed", INT64_MIN, INT64_MAX, req.seed) && opt_int("max_new_tokens", 1, INT32_MAX, max_new) &&
              opt_int("top_k", 0, INT32_MAX, top_k) && opt_num("temperature", 0.0, FLT_MAX, req.temperature) &&
              opt_num("top_p", DBL_MIN, 1.0, req.top_p) &&
              opt_num("repetition_penalty", DBL_MIN, FLT_MAX, req.repetition_penalty);
    req.max_new_tokens = (int) max_new;
    req.top_k          = (int) top_k;

    yyjson_doc_free(doc);

    if (!ok) {
        return false;
    }
    if (req.format != "pcm" && req.format != "wav") {
        err = "response_format must be 'pcm' or 'wav'";
        return false;
    }
    return true;
}

// Map an ABI status to an HTTP code. The two ABIs share numeric values:
// -1 invalid params, -2 mode/instruct invalid -> client error ; the rest
// are server side failures.
static int tts_status_to_http(int rc) {
    if (rc == 0) {
        return 200;
    }
    if (rc == -1 || rc == -2) {
        return 400;
    }
    return 502;
}

static void tts_handle_speech(const tts_backend & be, const httplib::Request & http_req, httplib::Response & res) {
    tts_request req;
    std::string err;
    if (!tts_parse_request(http_req.body, req, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }

    if (req.format == "wav") {
        // One-shot : collect the whole utterance, then emit a RIFF
        // file. The backend call blocks this connection thread; in
        // batch mode the chunks arrive from the ABI worker thread while
        // this thread waits inside qt_synthesize, so the plain append
        // stays single producer and the blocking return synchronizes.
        std::vector<float> buf;
        tts_sink           sink = [&buf](const float * s, int n) {
            buf.insert(buf.end(), s, s + n);
            return true;
        };
        std::string synth_err;
        int         rc = be.synthesize(req, sink, synth_err);
        if (rc != 0) {
            tts_json_error(res, tts_status_to_http(rc), "server_error",
                           synth_err.empty() ? "synthesis failed" : synth_err.c_str());
            return;
        }
        std::string wav = audio_encode_wav(buf.data(), (int) buf.size(), 24000, WAV_S16);
        res.set_content(std::move(wav), "audio/wav");
        return;
    }

    // Streaming : the synthesis runs on its own thread pushing s16le
    // bytes into a queue; the connection thread drains the queue into
    // the chunked sink. The decoupling keeps a slow client from
    // stalling the ABI worker (head of line blocking across the whole
    // batch), and a client disconnect flips client_gone so the next
    // chunk callback aborts generation and frees the GPU instead of
    // finishing a stream nobody reads. Backpressure is the utterance
    // itself: pending grows at most to the full PCM of one synthesis.
    struct stream_state {
        std::mutex              mu;
        std::condition_variable cv;
        std::string             pending;
        bool                    done = false;
        std::atomic<bool>       client_gone{ false };
        std::thread             th;
    };

    auto st = std::make_shared<stream_state>();

    st->th = std::thread([&be, req, st]() {
        tts_sink push = [st](const float * s, int n) {
            if (st->client_gone.load(std::memory_order_acquire)) {
                return false;
            }
            std::string bytes;
            tts_append_s16le(bytes, s, n);
            std::lock_guard<std::mutex> lk(st->mu);
            st->pending += bytes;
            st->cv.notify_all();
            return true;
        };
        std::string synth_err;
        be.synthesize(req, push, synth_err);
        std::lock_guard<std::mutex> lk(st->mu);
        st->done = true;
        st->cv.notify_all();
    });

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "audio/pcm",
        [st](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            {
                std::unique_lock<std::mutex> lk(st->mu);
                st->cv.wait(lk, [&] { return st->done || !st->pending.empty(); });
                chunk.swap(st->pending);
            }
            if (!chunk.empty()) {
                if (!sink.write(chunk.data(), chunk.size())) {
                    st->client_gone.store(true, std::memory_order_release);
                    return false;
                }
                return true;
            }
            sink.done();
            return true;
        },
        [st](bool) {
            st->client_gone.store(true, std::memory_order_release);
            if (st->th.joinable()) {
                st->th.join();
            }
        });
}

// Decode standard base64 (with optional padding) into out. Returns
// false on any character outside the alphabet.
static bool tts_b64_decode(const std::string & in, std::string & out) {
    static int8_t table[256];
    static bool   init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) {
            table[i] = -1;
        }
        const char * alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) {
            table[(uint8_t) alpha[i]] = (int8_t) i;
        }
        init = true;
    }

    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc  = 0;
    int      bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') {
            continue;
        }
        int8_t v = table[(uint8_t) c];
        if (v < 0) {
            return false;
        }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xff));
        }
    }
    return true;
}

// Parse the POST /v1/audio/voices body: name plus either wav_b64 or the
// spk_b64 / rvq_b64 pair, ref_text optional (enables ICL clone mode).
static bool tts_parse_voice_upload(const std::string & body, tts_voice_upload & up, std::string & err) {
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

    yyjson_val * name = yyjson_obj_get(root, "name");
    if (!yyjson_is_str(name) || yyjson_get_len(name) == 0) {
        err = "'name' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    up.name = yyjson_get_str(name);

    yyjson_val * ref_text = yyjson_obj_get(root, "ref_text");
    up.ref_text           = yyjson_is_str(ref_text) ? yyjson_get_str(ref_text) : "";

    yyjson_val * wav = yyjson_obj_get(root, "wav_b64");
    yyjson_val * spk = yyjson_obj_get(root, "spk_b64");
    yyjson_val * rvq = yyjson_obj_get(root, "rvq_b64");

    const bool has_wav = yyjson_is_str(wav) && yyjson_get_len(wav) > 0;
    const bool has_spk = yyjson_is_str(spk) && yyjson_get_len(spk) > 0;
    const bool has_rvq = yyjson_is_str(rvq) && yyjson_get_len(rvq) > 0;

    const bool has_latents = has_spk && has_rvq;
    if (has_spk != has_rvq || has_wav == has_latents) {
        err = "provide either 'wav_b64' or both 'spk_b64' and 'rvq_b64'";
        yyjson_doc_free(doc);
        return false;
    }
    if ((has_wav && !tts_b64_decode(yyjson_get_str(wav), up.wav)) ||
        (has_spk && !tts_b64_decode(yyjson_get_str(spk), up.spk)) ||
        (has_rvq && !tts_b64_decode(yyjson_get_str(rvq), up.rvq))) {
        err = "invalid base64 payload";
        yyjson_doc_free(doc);
        return false;
    }
    yyjson_doc_free(doc);
    return true;
}

static void tts_handle_voice_register(const tts_backend &      be,
                                      const httplib::Request & http_req,
                                      httplib::Response &      res) {
    if (!be.register_voice) {
        tts_json_error(res, 501, "not_implemented", "this backend has no voice registry");
        return;
    }
    tts_voice_upload up;
    std::string      err;
    if (!tts_parse_voice_upload(http_req.body, up, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }
    if (!be.register_voice(up, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.empty() ? "voice registration failed" : err.c_str());
        return;
    }
    std::string body = "{\"name\":\"" + up.name + "\",\"status\":\"registered\"}";
    res.set_content(body, "application/json");
}

static void tts_handle_voice_delete(const tts_backend &      be,
                                    const httplib::Request & http_req,
                                    httplib::Response &      res) {
    if (!be.remove_voice) {
        tts_json_error(res, 501, "not_implemented", "this backend has no voice registry");
        return;
    }
    const std::string name = http_req.matches[1];
    if (!be.remove_voice(name)) {
        tts_json_error(res, 404, "not_found_error", "no registered voice with this name");
        return;
    }
    res.set_content("{\"status\":\"deleted\"}", "application/json");
}

static void tts_handle_models(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "object", "list");
    yyjson_mut_val * data = yyjson_mut_arr(doc);
    yyjson_mut_val * one  = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, one, "id", be.model_id.c_str());
    yyjson_mut_obj_add_str(doc, one, "object", "model");
    yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
    yyjson_mut_arr_add_val(data, one);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_voices(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const std::string & v : be.voices) {
        yyjson_mut_val * one = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, one, "name", v.c_str());
        yyjson_mut_obj_add_str(doc, one, "kind", "speaker");
        yyjson_mut_arr_add_val(arr, one);
    }
    if (be.registered_voices) {
        for (const std::string & v : be.registered_voices()) {
            yyjson_mut_val * one = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_val(doc, one, "name", yyjson_mut_strcpy(doc, v.c_str()));
            yyjson_mut_obj_add_str(doc, one, "kind", "registered");
            yyjson_mut_arr_add_val(arr, one);
        }
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static int tts_server_run(const tts_backend & be, const server_config & cfg) {
    httplib::Server svr;
    g_svr = &svr;

    // per-operation socket idle timeouts. read is small (text in), write is
    // generous to cover a long streamed utterance without tripping on a slow
    // client.
    svr.set_read_timeout(60);
    svr.set_write_timeout(120);

    // reject oversized bodies. text plus an optional reference clip stays
    // well under this.
    svr.set_payload_max_length(32 * 1024 * 1024);

    // Nagle coalescing holds small packets back for tens of ms ; streamed
    // PCM chunks must leave the socket the moment they are written.
    svr.set_tcp_nodelay(true);

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set : a second instance on the same
    // port then fails with EADDRINUSE instead of silently sharing the socket
    // and splitting traffic between two daemons.
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    // permissive CORS so a browser client can call the API directly.
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });
    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.Post("/v1/audio/speech",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Get("/v1/audio/voices",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices(be, req, res); });
    svr.Post("/v1/audio/voices",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voice_register(be, req, res); });
    svr.Delete(R"(/v1/audio/voices/(.+))",
               [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voice_delete(be, req, res); });
    svr.Get("/health", tts_handle_health);

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    fprintf(stderr, "[Server] model %s\n", be.model_id.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
