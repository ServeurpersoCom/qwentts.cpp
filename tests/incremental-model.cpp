// Optional CPU smoke: test-incremental-model TALKER CODEC OUTPUT_PREFIX.
// Writes raw 24 kHz mono float32 audio for local listening/analysis.
#include "qwen.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

struct Probe {
    int calls = 0;
    int chunks = 0;
    bool eof = false;
    bool prefinish_audio = false;
    bool cancel = false;
    std::vector<float> audio;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
static long long milliseconds(Probe * p) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - p->start).count();
}
static bool audio(const float * samples, int n, void * user) {
    auto * p = static_cast<Probe *>(user);
    if (!p->chunks++) std::printf("first_audio_ms=%lld input_finished=%d\n", milliseconds(p), (int)p->eof);
    p->prefinish_audio |= !p->eof;
    for (int i = 0; i < n; i++) if (!std::isfinite(samples[i])) return false;
    p->audio.insert(p->audio.end(), samples, samples + n);
    return true;
}
static int text(void * user, const char ** bytes, size_t * n) {
    auto * p = static_cast<Probe *>(user);
    static const char * parts[] = {"Hello there. ", "This is a streaming test. "};
    if (p->calls == 1 && p->cancel) return -1;
    if (p->calls < 2) {
        *bytes = parts[p->calls++]; *n = std::strlen(*bytes); return 1;
    }
    p->eof = true;
    std::printf("input_end_ms=%lld audio_chunks=%d\n", milliseconds(p), p->chunks);
    return 0;
}
static bool save(const char * prefix, const char * suffix, const std::vector<float> & samples) {
    char name[4096];
    std::snprintf(name, sizeof(name), "%s-%s.f32", prefix, suffix);
    FILE * f = std::fopen(name, "wb");
    if (!f) return false;
    bool ok = std::fwrite(samples.data(), sizeof(float), samples.size(), f) == samples.size();
    return std::fclose(f) == 0 && ok;
}
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    qt_init_params ip; qt_init_default_params(&ip);
    ip.talker_path = argv[1]; ip.codec_path = argv[2]; ip.device = "CPU";
    qt_context * q = qt_init(&ip);
    if (!q) { std::fprintf(stderr, "%s\n", qt_last_error()); return 3; }
    qt_tts_params p; qt_tts_default_params(&p);
    p.lang = "English"; p.speaker = "ryan"; p.seed = 42; p.max_new_tokens = 160;
    Probe baseline;
    p.text = "Hello there. This is a streaming test. "; p.on_chunk = audio; p.on_chunk_user_data = &baseline;
    auto baseline_status = qt_synthesize(q, &p, nullptr);
    std::printf("baseline_status=%d samples=%zu\n", (int)baseline_status, baseline.audio.size());
    save(argv[3], "baseline", baseline.audio);
    Probe incremental;
    p.on_chunk_user_data = &incremental;
    auto status = qt_synthesize_incremental(q, &p, text, &incremental);
    std::printf("incremental_status=%d samples=%zu prefinish_audio=%d error=%s\n", (int)status,
                incremental.audio.size(), (int)incremental.prefinish_audio, qt_last_error());
    save(argv[3], "incremental", incremental.audio);
    Probe cancelled; cancelled.cancel = true;
    p.on_chunk_user_data = &cancelled;
    auto cancel_status = qt_synthesize_incremental(q, &p, text, &cancelled);
    std::printf("cancel_status=%d chunks=%d\n", (int)cancel_status, cancelled.chunks);
    qt_free(q);
    return baseline_status == QT_STATUS_OK && status == QT_STATUS_OK && incremental.prefinish_audio &&
           incremental.eof && incremental.audio.size() > 0 && cancel_status == QT_STATUS_CANCELLED ? 0 : 1;
}
