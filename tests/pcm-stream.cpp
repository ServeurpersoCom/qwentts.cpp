// pcm-stream.cpp: wire contract and cancellation tests for
// POST /v1/audio/speech/pcm-stream.
//
// A fake backend stands in for the ABI: it polls the abort predicate at the
// top of every simulated frame and emits one 80 ms block per chunk, the way
// the real pipeline does, and it admits one request at a time. The tests
// drive it over real loopback sockets and read the raw HTTP body, so they
// observe exactly what a client sees. No model is loaded, no GPU is touched
// and the binary does not link qwen-core.

#include "tts-server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// --- library diagnostics stand-in ------------------------------------------
// tts-server.h logs through qt_log; the real definition lives in qwen.cpp,
// which would drag the model library in. Keep the tests model-free.
void qt_log(enum qt_log_level level, const char * fmt, ...) {
    if (level < QT_LOG_INFO) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

namespace {

constexpr int  kFrameSamples = 1920;   // 80 ms at 24 kHz: one codec frame
constexpr int  kFrameBytes   = kFrameSamples * 2;
constexpr int  kFrameMs      = 10;     // stands in for GPU compute per frame
constexpr int  kLongFrames   = 400;    // 32 s of audio, far past any buffer
constexpr int  kNormalFrames = 12;
constexpr int  kPort         = 18099;

// --- fake backend ---------------------------------------------------------

enum fake_voice {
    FV_NORMAL,
    FV_LONG,
    FV_FAIL,
    FV_LATE,
    FV_COUNT,
};

struct voice_stats {
    std::atomic<int> frames{ 0 };
    std::atomic<int> cancels{ 0 };
    std::atomic<int> cancel_frame{ -1 };
};

voice_stats g_stats[FV_COUNT];
std::atomic<int> g_in_flight{ 0 };
std::atomic<int> g_max_in_flight{ 0 };

// The ABI admits one sequence at a time with max_batch 1, and the adapter
// gates on it as well; the fake does the same so overlap is observable.
std::mutex g_admit;

void reset_stats() {
    for (voice_stats & s : g_stats) {
        s.frames.store(0);
        s.cancels.store(0);
        s.cancel_frame.store(-1);
    }
    g_in_flight.store(0);
    g_max_in_flight.store(0);
}

int voice_index(const std::string & name) {
    if (name == "fake") {
        return FV_NORMAL;
    }
    if (name == "long") {
        return FV_LONG;
    }
    if (name == "fail") {
        return FV_FAIL;
    }
    if (name == "late") {
        return FV_LATE;
    }
    return -1;
}

// Mirrors the ABI: abort is polled at the top of each decode step, one block
// leaves per step, and a sink that refuses cancels the run.
int fake_synthesize(const tts_request & req, const tts_sink & sink, const tts_abort & abort, std::string & err) {
    const int id = voice_index(req.voice);
    if (id < 0) {
        err = "unknown voice '" + req.voice + "'";
        return -1;
    }

    std::lock_guard<std::mutex> gate(g_admit);

    const int now = g_in_flight.fetch_add(1) + 1;
    for (int prev = g_max_in_flight.load(); now > prev && !g_max_in_flight.compare_exchange_weak(prev, now);) {
    }
    struct leave {
        ~leave() { g_in_flight.fetch_sub(1); }
    } guard;

    if (id == FV_FAIL) {
        err = "fake failure";
        return -3;
    }

    const int total = id == FV_LONG ? kLongFrames : kNormalFrames;
    for (int f = 0; f < total; f++) {
        if (abort && abort()) {
            g_stats[id].cancels++;
            g_stats[id].cancel_frame.store(f);
            return -5;
        }
        if (id == FV_LATE && f == 2) {
            err = "fake failure";
            return -3;
        }
        std::vector<float> block(kFrameSamples);
        for (int i = 0; i < kFrameSamples; i++) {
            const double t = (double) (f * kFrameSamples + i) / 24000.0;
            block[(size_t) i] = 0.25f * (float) std::sin(2.0 * 3.14159265358979323846 * 220.0 * t);
        }
        if (!sink(block.data(), kFrameSamples)) {
            g_stats[id].cancels++;
            g_stats[id].cancel_frame.store(f);
            return -5;
        }
        g_stats[id].frames.store(f + 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(kFrameMs));
    }
    return 0;
}

tts_backend make_backend() {
    tts_backend be;
    be.model_id  = "fake-model";
    be.voices    = { "fake" };
    be.synthesize = fake_synthesize;
    return be;
}

// --- minimal HTTP client --------------------------------------------------

int failures = 0;

void check(bool cond, const char * what, const char * detail = "") {
    if (cond) {
        fprintf(stderr, "  ok   %s\n", what);
        return;
    }
    failures++;
    fprintf(stderr, "  FAIL %s%s%s\n", what, *detail ? ": " : "", detail);
}

std::string json_body(const std::string & input, const std::string & voice) {
    return "{\"input\":\"" + input + "\",\"voice\":\"" + voice + "\"}";
}

// One request on one connection: headers then the de-chunked body.
struct exchange {
    int         fd     = -1;
    std::string buf;
    int         status = 0;
    std::string headers;

    bool open(int port) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons((uint16_t) port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        return ::connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0;
    }

    bool send(const std::string & path, const std::string & body) {
        const std::string req = "POST " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
        size_t off = 0;
        while (off < req.size()) {
            const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
            if (n <= 0) {
                return false;
            }
            off += (size_t) n;
        }
        return true;
    }

    // Reads whatever arrives. Returns bytes read, 0 on EOF, -1 on timeout.
    ssize_t fill(int timeout_ms) {
        char          tmp[65536];
        struct timeval tv{ timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n > 0) {
            buf.append(tmp, (size_t) n);
        }
        return n;
    }

    bool read_headers(int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (buf.find("\r\n\r\n") == std::string::npos) {
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            if (fill(50) == 0) {
                return false;
            }
        }
        const size_t end = buf.find("\r\n\r\n") + 4;
        headers          = buf.substr(0, end);
        buf.erase(0, end);
        status = std::atoi(headers.c_str() + headers.find(' ') + 1);
        return true;
    }

    // Next chunk of the chunked body. false at the terminating chunk, on EOF
    // or on a truncated stream; done tells the two apart.
    bool read_chunk(std::string & out, bool & done, int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        done                = false;
        for (;;) {
            const size_t eol = buf.find("\r\n");
            if (eol != std::string::npos) {
                const size_t size = (size_t) std::strtoul(buf.c_str(), nullptr, 16);
                if (size == 0) {
                    done = true;
                    return false;
                }
                if (buf.size() >= eol + 2 + size + 2) {
                    out = buf.substr(eol + 2, size);
                    buf.erase(0, eol + 2 + size + 2);
                    return true;
                }
            }
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            if (fill(50) == 0) {
                return false;
            }
        }
    }

    void close_now() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    // Closes leaving no unread data behind. shutdown(SHUT_RD) discards the
    // receive queue, so close() puts a bare FIN on the wire instead of the
    // RST an unread queue would trigger. That FIN-without-RST peer is the
    // case where a write-only cancel detector keeps succeeding and the
    // utterance runs to completion.
    void close_quiet() {
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RD);
            ::close(fd);
            fd = -1;
        }
    }
};

// Parses [u32 big endian length][payload] frames out of a body, the way the
// Thalvi Qwen3 client does.
struct frame_reader {
    std::string buf;
    bool        truncated = false;

    std::vector<std::string> frames;
    bool                     ended = false;

    void append(const std::string & bytes) {
        buf += bytes;
        for (;;) {
            if (buf.size() < 4) {
                return;
            }
            const uint32_t len = ((uint32_t) (uint8_t) buf[0] << 24) | ((uint32_t) (uint8_t) buf[1] << 16) |
                                 ((uint32_t) (uint8_t) buf[2] << 8) | (uint32_t) (uint8_t) buf[3];
            if (len == 0) {
                ended = true;
                buf.erase(0, 4);
                return;
            }
            if (buf.size() < 4 + len) {
                return;
            }
            frames.push_back(buf.substr(4, len));
            buf.erase(0, 4 + len);
        }
    }
};

// Runs a whole body through the frame reader.
frame_reader read_frames(exchange & ex, int timeout_ms) {
    frame_reader                       fr;
    std::string                        chunk;
    bool                               done = false;
    while (ex.read_chunk(chunk, done, timeout_ms)) {
        fr.append(chunk);
    }
    fr.truncated = !done;
    return fr;
}

// --- tests ---------------------------------------------------------------

void test_framed_contract() {
    fprintf(stderr, "framed pcm-stream contract\n");
    reset_stats();
    exchange ex;
    check(ex.open(kPort), "connect");
    check(ex.send("/v1/audio/speech/pcm-stream", json_body("Hello there.", "fake")), "send");
    check(ex.read_headers(5000) && ex.status == 200, "200 response");
    check(ex.headers.find("application/octet-stream") != std::string::npos, "content-type is octet-stream");
    check(ex.headers.find("Transfer-Encoding: chunked") != std::string::npos, "chunked transfer");

    const frame_reader fr = read_frames(ex, 5000);
    check(!fr.truncated, "body ends with the HTTP terminator");
    check(fr.ended, "zero length frame ends the stream");
    check(fr.buf.empty(), "nothing after the end frame");
    check(fr.frames.size() == (size_t) kNormalFrames, "one frame per decoded block");
    bool sized = !fr.frames.empty();
    bool audio = false;
    for (const std::string & f : fr.frames) {
        sized = sized && f.size() == (size_t) kFrameBytes;
        for (size_t i = 0; i + 1 < f.size() && !audio; i += 2) {
            audio = f[i] != 0 || f[i + 1] != 0;
        }
    }
    check(sized, "every frame is 80 ms of s16le");
    check(audio, "frames carry audio samples");
    check(g_stats[FV_NORMAL].frames.load() == kNormalFrames, "backend produced every frame");
    ex.close_now();
}

void test_disconnect_cancels() {
    fprintf(stderr, "disconnect cancels at the next frame\n");
    reset_stats();
    exchange ex;
    check(ex.open(kPort), "connect");
    check(ex.send("/v1/audio/speech/pcm-stream", json_body("Long line.", "long")), "send");
    check(ex.read_headers(5000) && ex.status == 200, "200 response");

    std::string chunk;
    bool        done = false;
    check(ex.read_chunk(chunk, done, 5000), "first body chunk arrives");
    frame_reader first;
    first.append(chunk);
    check(first.frames.size() >= 1, "first frame is one decoded block");
    ex.close_quiet();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (g_stats[FV_LONG].cancels.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(g_stats[FV_LONG].cancels.load() == 1, "disconnect reached the cancel callback");
    check(g_stats[FV_LONG].cancel_frame.load() <= 60, "generation stopped one frame in, not at the end");
    check(g_stats[FV_LONG].frames.load() < kLongFrames, "the utterance did not run to completion");
}

void test_next_request_recovers() {
    fprintf(stderr, "next request after a cancel\n");
    reset_stats();
    exchange first;
    check(first.open(kPort), "connect");
    check(first.send("/v1/audio/speech/pcm-stream", json_body("Long line.", "long")), "send");
    check(first.read_headers(5000) && first.status == 200, "200 response");
    std::string chunk;
    bool        done = false;
    check(first.read_chunk(chunk, done, 5000), "first body chunk arrives");
    first.close_quiet();

    // Issued while the cancelled worker is still winding down: it must wait
    // for that worker to exit instead of running beside it.
    exchange second;
    check(second.open(kPort), "second connect");
    check(second.send("/v1/audio/speech/pcm-stream", json_body("Short line.", "fake")), "second send");
    check(second.read_headers(5000) && second.status == 200, "second 200 response");
    const frame_reader fr = read_frames(second, 5000);
    check(fr.ended && !fr.truncated, "second stream ends cleanly");
    check(fr.frames.size() == (size_t) kNormalFrames, "second stream is its own utterance");
    check(g_stats[FV_NORMAL].frames.load() == kNormalFrames, "second request generated its own frames only");
    check(g_max_in_flight.load() == 1, "never two generations at once");
    check(g_stats[FV_LONG].cancels.load() == 1, "the cancelled worker did cancel");
    second.close_now();
}

void test_errors_before_audio() {
    fprintf(stderr, "errors before the first frame\n");
    struct error_case {
        const char * voice;
        int          status;
        const char * needle;
    };
    const error_case cases[] = {
        { "fail", 502, "fake failure" },
        { "ghost", 400, "unknown voice" },
    };
    for (const error_case & c : cases) {
        reset_stats();
        exchange ex;
        check(ex.open(kPort), "connect");
        check(ex.send("/v1/audio/speech/pcm-stream", json_body("Hello.", c.voice)), "send");
        check(ex.read_headers(5000), "response arrives");
        check(ex.status == c.status, "error status");
        check(ex.headers.find("application/json") != std::string::npos, "json error envelope");
        std::string body = ex.buf;
        ex.buf.clear();
        while (ex.fill(500) > 0) {
            body += ex.buf;
            ex.buf.clear();
        }
        check(body.rfind("{\"error\"", 0) == 0, "error body is the json envelope");
        check(body.find(c.needle) != std::string::npos, "error message reaches the client");
        ex.close_now();
    }
}

void test_error_mid_stream_is_truncated() {
    fprintf(stderr, "failure after the first frame\n");
    reset_stats();
    exchange ex;
    check(ex.open(kPort), "connect");
    check(ex.send("/v1/audio/speech/pcm-stream", json_body("Hello.", "late")), "send");
    check(ex.read_headers(5000) && ex.status == 200, "200 response");
    const frame_reader fr = read_frames(ex, 5000);
    check(fr.frames.size() == 2, "frames before the failure were delivered");
    check(!fr.ended, "no zero length end frame on a failed stream");
    check(fr.truncated, "body ends without the HTTP terminator");
    ex.close_now();
}

void test_health() {
    fprintf(stderr, "health liveness\n");
    exchange ex;
    check(ex.open(kPort), "connect");
    const std::string req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    check(::send(ex.fd, req.data(), req.size(), MSG_NOSIGNAL) == (ssize_t) req.size(), "send");
    check(ex.read_headers(5000) && ex.status == 200, "200 response");
    while (ex.fill(500) > 0) {
    }
    check(ex.buf.find("\"status\":\"ok\"") != std::string::npos, "reports ok");
    ex.close_now();
}

void test_legacy_route_still_raw() {
    fprintf(stderr, "legacy /v1/audio/speech pcm is unchanged\n");
    reset_stats();
    exchange ex;
    check(ex.open(kPort), "connect");
    check(ex.send("/v1/audio/speech", "{\"input\":\"Hello.\",\"voice\":\"fake\",\"response_format\":\"pcm\"}"), "send");
    check(ex.read_headers(5000) && ex.status == 200, "200 response");
    check(ex.headers.find("audio/pcm") != std::string::npos, "content-type is audio/pcm");
    std::string body;
    bool        done = false;
    std::string chunk;
    while (ex.read_chunk(chunk, done, 5000)) {
        body += chunk;
    }
    check(done, "body ends with the HTTP terminator");
    check(body.size() == (size_t) (kNormalFrames * kFrameBytes), "bare s16le, no length prefixes");
    check(g_stats[FV_NORMAL].frames.load() == kNormalFrames, "backend produced every frame");
    ex.close_now();
}

}  // namespace

int main() {
    tts_backend be = make_backend();

    server_config cfg;
    cfg.host = "127.0.0.1";
    cfg.port = kPort;

    std::thread server([&be, &cfg]() { tts_server_run(be, cfg); });

    // Wait for the port, bounded, on the health route the server owns.
    bool up = false;
    for (int attempt = 0; attempt < 100 && !up; attempt++) {
        exchange probe;
        if (probe.open(kPort)) {
            const std::string req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
            ::send(probe.fd, req.data(), req.size(), MSG_NOSIGNAL);
            up = probe.read_headers(500) && probe.status == 200;
        }
        probe.close_now();
        if (!up) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    if (!up) {
        fprintf(stderr, "FAIL server did not come up on port %d\n", kPort);
        return 1;
    }

    test_framed_contract();
    test_disconnect_cancels();
    test_next_request_recovers();
    test_errors_before_audio();
    test_error_mid_stream_is_truncated();
    test_health();
    test_legacy_route_still_raw();

    kill(getpid(), SIGTERM);
    server.join();

    fprintf(stderr, failures == 0 ? "\nall pcm-stream tests passed\n" : "\n%d pcm-stream checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
