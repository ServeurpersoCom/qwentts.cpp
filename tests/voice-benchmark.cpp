#include "qwen.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

template<typename T> static std::vector<T> read_binary(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error(std::string("无法读取：") + path);
    const auto size = input.tellg();
    if (size <= 0 || size % sizeof(T) != 0) throw std::runtime_error("二进制材料尺寸非法");
    std::vector<T> result((size_t) size / sizeof(T));
    input.seekg(0);
    if (!input.read((char *) result.data(), size)) throw std::runtime_error("材料读取不完整");
    return result;
}

static std::string read_text(const char * path) {
    auto bytes = read_binary<char>(path);
    std::string result(bytes.begin(), bytes.end());
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
}

struct Capture {
    Clock::time_point start;
    double first_ms = -1;
    std::vector<float> samples;
};

static bool receive_chunk(const float * samples, int count, void * user) {
    auto * capture = (Capture *) user;
    if (capture->first_ms < 0 && count > 0) {
        capture->first_ms = std::chrono::duration<double, std::milli>(Clock::now() - capture->start).count();
    }
    capture->samples.insert(capture->samples.end(), samples, samples + count);
    return true;
}

static void receive_log(enum qt_log_level, const char * message, void * user) {
    std::fprintf(stderr, "%s\n", message);
    const char * ttfa = std::strstr(message, "[Perf] TTFA ");
    if (ttfa) std::sscanf(ttfa, "[Perf] TTFA %lf", (double *) user);
}

int main(int argc, char ** argv) {
    if (argc != 10 && argc != 11) {
        std::fprintf(stderr, "用法：voice-benchmark model codec spk codes-i32 ref-text text output-f32 device threads\n"
                             "现场提取时用 reference-f32 extract 代替 spk codes-i32；末尾加 buffered 测批量解码。\n");
        return 2;
    }
    struct qt_context * context = nullptr;
    double first_codes_ms = -1;
    qt_log_set(receive_log, &first_codes_ms);
    try {
        const bool extract = std::string(argv[4]) == "extract";
        std::vector<float> speaker;
        std::vector<int32_t> codes;
        if (!extract) {
            speaker = read_binary<float>(argv[3]);
            codes = read_binary<int32_t>(argv[4]);
        }
        auto reference_text = read_text(argv[5]);
        auto text = read_text(argv[6]);
        struct qt_init_params init;
        qt_init_default_params(&init);
        init.talker_path = argv[1];
        init.codec_path = argv[2];
        init.backend = argv[8];
        init.n_threads = std::stoi(argv[9]);
        context = qt_init(&init);
        if (!context) throw std::runtime_error(qt_last_error());
        if (extract) {
            auto reference = read_binary<float>(argv[3]);
            struct qt_voice_ref ref = {};
            if (qt_extract_voice_ref(context, reference.data(), (int) reference.size(), &ref) != QT_STATUS_OK) {
                throw std::runtime_error(qt_last_error());
            }
            if (ref.ref_spk_dim != 1024 || ref.num_codebooks != 16 || ref.ref_T <= 0) {
                qt_voice_ref_free(&ref);
                throw std::runtime_error("提取结果尺寸非法");
            }
            speaker.assign(ref.ref_spk_emb, ref.ref_spk_emb + ref.ref_spk_dim);
            codes.assign(ref.ref_codes, ref.ref_codes + ref.ref_T * ref.num_codebooks);
            qt_voice_ref_free(&ref);
            for (const auto & item : {std::make_pair(".spk", std::make_pair((const char *) speaker.data(), speaker.size() * sizeof(float))),
                                     std::make_pair(".codes-i32", std::make_pair((const char *) codes.data(), codes.size() * sizeof(int32_t)))}) {
                std::ofstream file(std::string(argv[7]) + item.first, std::ios::binary);
                file.write(item.second.first, (std::streamsize) item.second.second);
                if (!file) throw std::runtime_error("无法保存提取结果");
            }
        }
        if (speaker.size() != 1024 || codes.empty() || codes.size() % 16 != 0) throw std::runtime_error("声学条件尺寸非法");
        struct qt_backend_info backend;
        if (qt_backend_info(context, &backend) != QT_STATUS_OK) throw std::runtime_error(qt_last_error());
        if (backend.n_threads != init.n_threads || std::string(backend.device_name) != init.backend) {
            throw std::runtime_error("实际后端或线程数与请求不一致");
        }
        struct qt_tts_params params;
        qt_tts_default_params(&params);
        params.text = text.c_str();
        params.lang = "Chinese";
        params.seed = 42;
        params.max_new_tokens = 512;
        params.ref_spk_emb = speaker.data();
        params.ref_spk_dim = (int) speaker.size();
        params.ref_codes = codes.data();
        params.ref_T = (int) (codes.size() / 16);
        params.ref_text = reference_text.c_str();
        Capture capture;
        capture.start = Clock::now();
        if (argc == 10) {
            params.on_chunk = receive_chunk;
            params.on_chunk_user_data = &capture;
        } else if (std::string(argv[10]) != "buffered") {
            throw std::runtime_error("未知输出模式");
        }
        struct qt_audio output = {};
        const auto status = qt_synthesize(context, &params, &output);
        const double elapsed = std::chrono::duration<double>(Clock::now() - capture.start).count();
        if (status == QT_STATUS_OK && argc == 11 && output.samples) {
            capture.samples.assign(output.samples, output.samples + output.n_samples);
        }
        qt_audio_free(&output);
        if (status != QT_STATUS_OK) throw std::runtime_error(qt_last_error());
        if (capture.samples.empty()) throw std::runtime_error("没有生成音频");
        for (const float sample : capture.samples) {
            if (!std::isfinite(sample)) throw std::runtime_error("输出包含非有限采样值");
        }
        std::ofstream file(argv[7], std::ios::binary);
        file.write((const char *) capture.samples.data(), (std::streamsize) (capture.samples.size() * sizeof(float)));
        if (!file) throw std::runtime_error("无法保存音频");
        const double seconds = (double) capture.samples.size() / 24000;
        std::printf("{\"device\":\"%s\",\"backend\":\"%s\",\"n_threads\":%d,\"samples\":%zu,\"audio_s\":%.6f,\"compute_s\":%.6f,\"rtf\":%.6f,\"first_ms\":%.6f,\"first_codes_ms\":%.6f}\n",
                    backend.device_name, backend.backend_name, backend.n_threads, capture.samples.size(), seconds,
                    elapsed, elapsed / seconds, capture.first_ms, first_codes_ms);
        qt_free(context);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        qt_free(context);
        return 1;
    }
}
