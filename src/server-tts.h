#pragma once
// server-tts.h: OpenAI-compatible /v1/audio/speech endpoint for qwentts.cpp.
//
// Loads talker + codec GGUFs once at startup (GPU resident), then serves
// synthesis requests over HTTP.  Audio encoding via miniaudio (WAV/FLAC
// built-in, MP3/OGG optional via LAME/libvorbis).
//
// Endpoints:
//   POST /v1/audio/speech   — OpenAI TTS compat
//   GET  /v1/models          — list loaded model
//   GET  /health             — health check

#include "qwen.h"

// Must be defined BEFORE the first miniaudio include.
#define MINIAUDIO_IMPLEMENTATION
#define MA_ENABLE_DECODER_WRAPPERS 0
#define MA_ENABLE_RESOURCE_MANAGER 0

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include "../vendor/miniaudio/miniaudio.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "../vendor/cpp-httplib/httplib.h"
#include "../vendor/nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

/* ------------------------------------------------------------------ */
/*  Audio encoding — float PCM @ 24 kHz → WAV / FLAC / MP3 / OGG     */
/* ------------------------------------------------------------------ */

// Encode `n_samples` mono float PCM samples at `sample_rate` Hz into
// the requested container format.  Returns the binary blob on success
// or an empty string with an error message in `err_msg`.
//
// Supported formats (always):  "wav", "flac"
// Supported formats (optional, compile-time):  "mp3" (LAME), "ogg" (libvorbis)
std::string encode_audio(
    const float * samples, int n_samples, int sample_rate,
    const std::string & format,
    std::string & err_msg);

/* ------------------------------------------------------------------ */
/*  Loaded model — initialised once, shared across requests           */
/* ------------------------------------------------------------------ */

struct TTSModel {
    struct qt_context * ctx = nullptr;
    std::string         talker_path;
    std::string         codec_path;
    std::string         model_type;   // "base" / "custom_voice" / "voice_design"
    std::string         model_size;   // "0.6B" / "1.7B"
    std::vector<std::string> speakers;
    std::vector<std::string> languages;

    // Probe GGUF metadata and fill model_type / model_size / speakers / languages
    void populate_metadata();
};

/* ------------------------------------------------------------------ */
/*  Server config & bootstrap                                         */
/* ------------------------------------------------------------------ */

struct ServerConfig {
    std::string host = "0.0.0.0";
    int         port = 8080;
    std::string talker_path;
    std::string codec_path;
    bool        use_fa       = true;
    bool        clamp_fp16   = false;
    std::vector<std::string> api_keys;
    // Voice cloning defaults (can be overridden per-request in JSON body)
    std::string default_speaker;       // named speaker for custom_voice
    std::string default_instruct;      // style instruction for voice_design
    std::string default_ref_audio;     // reference audio file for base mode cloning
    std::string default_ref_transcript;// transcript of reference audio
};

// Load model, start HTTP server, block until SIGINT/SIGTERM.
// Returns 0 on success, -1 on failure.
int run_server(ServerConfig config);
