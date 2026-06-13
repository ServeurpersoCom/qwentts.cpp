#pragma once

#include "server-http.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct server_params {
    std::string model_path;
    std::string codec_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string lang = "auto";
    bool use_fa = true;
    bool clamp_fp16 = false;
    std::string device;
    // TTS defaults (CLI-configured, per-request overridable via JSON body)
    std::string instruct;
    std::string speaker;
    int64_t seed = -1;
    bool do_sample = true;
    bool subtalker_do_sample = true;
    int max_new_tokens = 2048;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    float subtalker_temperature = 0.9f;
    int subtalker_top_k = 50;
    float subtalker_top_p = 1.0f;
    float codec_chunk_sec = 5.0f;
    float codec_left_context_sec = 2.0f;
    std::string dump_dir;
    std::string ref_wav;
    std::string ref_text;
};

struct server_routes {
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_models;
    server_http_context::handler_t get_voices;
    server_http_context::handler_t post_speech;
    server_http_context::handler_t post_cancel;
};

int qwentts_server_main(int argc, char ** argv);
