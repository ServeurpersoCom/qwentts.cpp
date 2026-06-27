#pragma once
// qwen.h: public ABI for qwentts.cpp.
//
// Single-header public API. Pure C99, consumable from C and C++ alike.
// Bindings (Python ctypes, Rust bindgen, Go cgo) parse this file directly.
// Style follows whisper.h / llama.h / omnivoice.h: extern "C" linkage on
// every entry, POD structs only, const char * UTF-8 strings, qt_status
// enum returns.
//
// The opaque qt_context handle aggregates every module the synthesis
// path needs (Talker LM weights, code predictor MTP head, optional
// speaker encoder, 12 Hz audio tokenizer codec, BPE tokenizer, GGML
// backend pair). One init, one free, one synthesize call covers the
// full TTS path. The lower-level pipeline_tts_* / pipeline_codec_*
// entries declared in pipeline-tts.h / pipeline-codec.h stay available
// for tooling that needs partial init, but they are intentionally not
// part of this public ABI.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Symbol visibility. Three Windows cases: building the SHARED target
// (QWEN_BUILD set, dllexport), consuming the SHARED target from
// outside (nothing set, dllimport), consuming the STATIC archive
// (QWEN_STATIC set by the static target's INTERFACE definitions,
// empty so the linker resolves the symbol directly without dllimport).
// On GCC/Clang the default-visibility attribute is harmless on static
// builds and required on shared builds.
#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(QWEN_STATIC)
#        define QT_API
#    elif defined(QWEN_BUILD)
#        define QT_API __declspec(dllexport)
#    else
#        define QT_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define QT_API __attribute__((visibility("default")))
#else
#    define QT_API
#endif

// Struct ABI version. Incremented every time a public POD struct grows a
// new field at the end. Callers fill `.abi_version = QT_ABI_VERSION`
// (or let qwen_*_default_params set it). Entries that consume those
// structs reject inputs whose abi_version exceeds the build-time
// constant: this guards a binary built against vN from receiving a
// struct laid out for vN+1 by a freshly compiled binding. Adding fields
// stays backward compat because the new tail is zero init in older
// callers and the lib reads only what its abi_version permits.
//
// There is no separate semver triple. The runtime build identity is the
// git short hash + commit date string returned by qt_version(); for
// binding compat checks, QT_ABI_VERSION is the only number that
// matters.
#define QT_ABI_VERSION 2

// Returns a static string of the form "<git-hash> (<date>)" identifying
// the exact commit this binary was built from. Safe to call from any
// thread, no allocation. Pointer stays valid for the process lifetime.
QT_API const char * qt_version(void);

// Status code returned by every fallible entry. QT_STATUS_OK is always
// zero so `if (rc)` reads as `if (rc != QT_STATUS_OK)`.
enum qt_status {
    QT_STATUS_OK              = 0,
    QT_STATUS_INVALID_PARAMS  = -1,
    QT_STATUS_MODE_INVALID    = -2,
    QT_STATUS_GENERATE_FAILED = -3,
    QT_STATUS_OOM             = -4,
    QT_STATUS_CANCELLED       = -5,
};

// Returns the last error message produced on the calling thread by any
// qwen_* entry, as a NUL terminated UTF-8 string. errno-style semantics:
// the pointer is only meaningful right after a failure (qt_init
// returning NULL, or any qwen_* entry returning a negative qt_status);
// calling it after a successful entry yields the previous message or an
// empty string. Storage is thread local so two threads running
// qt_synthesize concurrently never race on each other's diagnostics.
// The pointer stays valid until the next failing qwen_* entry on the
// same thread.
QT_API const char * qt_last_error(void);

// Output audio buffer. Plain POD: the samples pointer is malloc
// allocated by qt_synthesize, owned by the struct, released by
// qt_audio_free. Do not free samples directly nor reassign without
// freeing first. Zero initialise before the first use:
// `struct qt_audio a = {0};`.
struct qt_audio {
    float * samples;      // mono PCM, malloc allocated
    int     n_samples;    // length in samples
    int     sample_rate;  // 24000 (codec rate)
    int     channels;     // 1 (mono)
};

// Release the samples buffer and reset the struct to empty. Safe on a
// zero initialised struct (no double free, no NULL deref).
QT_API void qt_audio_free(struct qt_audio * a);

// Opaque handle. Definition lives in qwen.cpp. Use qt_init / qt_free.
struct qt_context;

// Initialisation parameters. Both GGUF paths are required: the talker
// GGUF holds the LM weights, the code predictor MTP head and (for
// custom_voice / voice_design checkpoints) the speaker encoder; the
// codec GGUF holds the 12 Hz audio tokenizer. abi_version stays first
// so a future struct growth keeps reading the version field at offset
// 0. use_fa enables fused flash attention in the Talker and Code
// Predictor forwards when a GPU backend is present (CPU always uses the
// F32 manual chain); clamp_fp16 inserts ggml_clamp(-65504, 65504) on V
// before attention and on the residual stream between blocks to guard
// FP16 matmul accumulation on sub Ampere CUDA targets.
struct qt_init_params {
    int          abi_version;
    const char * talker_path;
    const char * codec_path;
    bool         use_fa;
    bool         clamp_fp16;
};

// Initialise to the standard defaults: both paths NULL (caller must set
// them before calling qt_init), use_fa true, clamp_fp16 false.
QT_API void qt_init_default_params(struct qt_init_params * p);

// Allocate every module described by params. Returns NULL on any
// failure after releasing whatever it has allocated so far. The
// returned handle owns its GGML backend pair and must be released with
// qt_free.
QT_API struct qt_context * qt_init(const struct qt_init_params * params);

// Release every module owned by the handle and free the handle itself.
// Safe on NULL.
QT_API void qt_free(struct qt_context * q);

// Precomputed Base-model voice reference latents. Plain POD: both
// pointers are malloc allocated by qt_extract_voice_ref, owned by the
// struct, released by qt_voice_ref_free. Do not free either pointer
// directly nor reassign without freeing first. Zero initialise before
// first use: `struct qt_voice_ref ref = {0};`.
//
// ref_spk_emb is the speaker embedding equivalent to a raw .spk file.
// ref_codes is the RVQ code matrix equivalent to a raw .rvq file,
// laid out [num_codebooks, ref_T] row-major (T fastest).
struct qt_voice_ref {
    float *   ref_spk_emb;
    int       ref_spk_dim;
    int32_t * ref_codes;
    int       ref_T;
    int       num_codebooks;
};

// Extract reusable voice-clone conditioning from a decoded reference
// .wav/audio buffer: mono float32 PCM at 24 kHz. Requires a loaded Base
// model with speaker encoder weights. The speaker embedding consumes the
// full input buffer, matching --ref-wav clone mode A. RVQ encoding
// truncates to the codec hop boundary, matching qwen-codec --talker
// ref.wav / --ref-rvq.
// For reference-WAV-plus-transcription ICL mode, pass the returned
// ref_spk_emb and ref_codes back to qt_synthesize together with the
// transcript in qt_tts_params.ref_text.
//
// On success fills out with malloc-owned buffers. On failure leaves out
// empty and stores a diagnostic in qt_last_error().
QT_API enum qt_status qt_extract_voice_ref(struct qt_context *   q,
                                           const float *         ref_audio_24k,
                                           int                   ref_n_samples,
                                           struct qt_voice_ref * out);

// Release the speaker embedding and RVQ code buffers and reset the
// struct to empty. Safe on a zero initialised struct.
QT_API void qt_voice_ref_free(struct qt_voice_ref * ref);

// Cooperative cancellation callback. Returns true to request the
// synthesis to abort. Polled at the top of every Talker decode step in
// the autoregressive loop, so the cancel granularity is roughly one
// audio frame, i.e. 1 / 12 Hz ~ 83 ms.
typedef bool (*qt_cancel_cb)(void * user_data);

// Streaming output callback. When set on qt_tts_params, the synth
// pipeline runs in streaming mode: audio is decoded chunk by chunk from
// the AR codec frames and emitted through this callback rather than
// accumulated into the `out` buffer of qt_synthesize. Returning false
// aborts the synthesis with QT_STATUS_CANCELLED, identical to the
// qt_cancel_cb behaviour. The samples pointer is mono float PCM at
// 24 kHz; valid only for the duration of the call.
// user_data is forwarded verbatim from on_chunk_user_data.
//
// The chunk granularity is driven by chunk_duration_sec in qt_tts_params:
// once the AR loop has produced enough frames to cover that duration,
// the codec decodes that bundle and emits it. The last chunk on EOS /
// max_new flushes whatever frames remain.
typedef bool (*qt_audio_chunk_cb)(const float * samples, int n_samples, void * user_data);

// Log severity. Numerically ordered so a callback can filter with a
// simple `if (level < threshold) return;`. ERROR is reserved for
// failure reports that the lib also surfaces via qt_status /
// qt_last_error; WARN for recoverable surprises; INFO for the
// normal load and synthesis cadence; DEBUG for tensor-level cossim
// diagnostics.
enum qt_log_level {
    QT_LOG_DEBUG = 0,
    QT_LOG_INFO  = 1,
    QT_LOG_WARN  = 2,
    QT_LOG_ERROR = 3,
};

// Logging callback. msg is a NUL terminated UTF-8 string already
// formatted by the lib, with no trailing newline (the callback is free
// to add one). user_data is forwarded verbatim from qt_log_set.
// Called from any thread the lib runs on: the callback must be
// reentrant.
typedef void (*qt_log_cb)(enum qt_log_level level, const char * msg, void * user_data);

// Install a global log callback. Passing cb == NULL restores the
// default behaviour (write to stderr). Safe to call at any point;
// takes effect immediately on subsequent log emissions across every
// thread. Storage is process wide, not per handle, matching
// whisper_log_set / llama_log_set / ov_log_set.
QT_API void qt_log_set(qt_log_cb cb, void * user_data);

// Synthesis parameters. Strings are NULL terminated UTF-8; NULL maps
// to empty where the underlying pipeline accepts it. The selection
// between base / custom_voice / voice_design synthesis mode is driven
// by the model_type read from the talker GGUF at qt_init time, not
// by an explicit flag here; the seven mode rules are enforced inside
// qt_synthesize and surface as QT_STATUS_MODE_INVALID with a
// descriptive qt_last_error(). abi_version stays first so the lib
// can route on it before reading any field that may have shifted in a
// future minor.
struct qt_tts_params {
    int abi_version;

    // Input text and language hint. text is required and non empty.
    // lang accepts the upstream qwen3-tts language names ("english",
    // "chinese", "auto", ...). NULL selects "auto": the prompt carries
    // no language id and the model infers it from the text.
    // instruct is the style instruction string; required for
    // voice_design, optional for custom_voice, rejected for base.
    // speaker is the named speaker for custom_voice models, rejected
    // for the other two modes.
    const char * text;
    const char * lang;
    const char * instruct;
    const char * speaker;

    // Optional voice reference for base mode voice cloning. Mode A
    // (x_vector_only) sets ref_audio_24k only; mode B (ICL) sets
    // both ref_audio_24k and ref_text. ref_audio_24k is a mono float
    // PCM buffer sampled at 24 kHz. Mutually exclusive
    // with speaker. Rejected for custom_voice / voice_design.
    const float * ref_audio_24k;
    int           ref_n_samples;
    const char *  ref_text;

    // Sampling configuration. seed == -1 is resolved by qt_synthesize
    // to a hardware random seed via std::random_device, anything else
    // is forwarded verbatim for deterministic replay across runs.
    // Defaults match the upstream Python reference: do_sample true,
    // temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05,
    // subtalker mirrors talker, max_new_tokens 2048.
    int64_t seed;
    int     max_new_tokens;
    bool    do_sample;
    float   temperature;
    int     top_k;
    float   top_p;
    float   repetition_penalty;
    bool    subtalker_do_sample;
    float   subtalker_temperature;
    int     subtalker_top_k;
    float   subtalker_top_p;

    // Intermediate tensor dump directory. NULL disables dumps. Debug
    // only, slows the run.
    const char * dump_dir;

    // Cooperative cancellation. cancel NULL disables the feature.
    // cancel_user_data is forwarded to the callback verbatim. Polled
    // at the top of every Talker decode step (~83 ms granularity).
    qt_cancel_cb cancel;
    void *       cancel_user_data;

    // Streaming output. When on_chunk is non NULL, qt_synthesize runs
    // the streaming pipeline: audio chunks emit through on_chunk and
    // `out` stays empty on success. on_chunk NULL keeps the buffered
    // path. The last chunk on EOS or max_new flushes whatever frames
    // remain.
    qt_audio_chunk_cb on_chunk;
    void *            on_chunk_user_data;

    // Codec decode framing. Applied to both the streaming path (chunk
    // by chunk emission) and the buffered path (one shot decode at the
    // end) : the chunked decode rolls a left context window across the
    // codec frames to avoid edge artefacts at chunk boundaries. The
    // first chunk has its left context collapsed to whatever is
    // available, matching the upstream Qwen3-TTS 12 Hz tokenizer
    // chunked_decode rule. Defaults match the upstream reference :
    // codec_chunk_sec 24.0 (300 frames at 12.5 Hz) and
    // codec_left_context_sec 2.0 (25 frames at 12.5 Hz). Values are
    // converted internally to integer frame counts via the codec frame
    // rate ; codec_chunk_sec clamps to >= 1 frame, codec_left_context_sec
    // clamps to >= 0 frames.
    float codec_chunk_sec;
    float codec_left_context_sec;

    // ABI v2. Pre-encoded voice reference, the latent counterpart of
    // ref_audio_24k. ref_spk_emb is the speaker embedding produced by
    // the speaker encoder (ref_spk_dim f32 values, must equal the
    // talker hidden size). ref_codes is the ICL code matrix produced
    // by the codec encoder, [num_codebooks, ref_T] row-major.
    // ref_spk_emb alone selects clone mode A; ref_spk_emb + ref_codes
    // + ref_text selects mode B, mirroring the raw constraints.
    // Mutually exclusive with ref_audio_24k and speaker.
    const float *   ref_spk_emb;
    int             ref_spk_dim;
    const int32_t * ref_codes;
    int             ref_T;
};

// Initialise to the standard defaults. Strings NULL, seed -1,
// max_new_tokens 2048, do_sample true, temperature 0.9, top_k 50,
// top_p 1.0, repetition_penalty 1.05, subtalker mirrors talker,
// dump_dir NULL, cancel NULL, on_chunk NULL, codec_chunk_sec 24.0,
// codec_left_context_sec 2.0.
QT_API void qt_tts_default_params(struct qt_tts_params * p);

// Number of RVQ codebooks (K) of the loaded codec. Pre-encoded ICL
// reference codes passed via ref_codes are laid out [K, ref_T]
// row-major; callers reading a packed .rvq stream need K to derive
// ref_T from the code count. Returns 0 on a NULL handle.
QT_API int qt_num_codebooks(const struct qt_context * q);

// Run the full TTS synthesis. Validates the params against the loaded
// model_type (the seven base / custom_voice / voice_design rules),
// resolves the seed, hands off to pipeline_tts_synthesize and fills
// `out` with mono float PCM at 24 kHz in buffered mode.
// In streaming mode (params->on_chunk != NULL), audio is emitted
// through the callback and `out` stays empty. Returns QT_STATUS_OK on
// success; on any failure returns a negative qt_status describing the
// cause and leaves `out` empty.
QT_API enum qt_status qt_synthesize(struct qt_context * q, const struct qt_tts_params * params, struct qt_audio * out);

// Convert a duration in seconds to a frame count using the codec
// frame rate (24000 / TOKENIZER_HOP_LENGTH = 12.5 Hz).
// Clamps to a minimum of one frame.
QT_API int qt_duration_sec_to_tokens(const struct qt_context * q, float duration_sec);

// Number of named speakers in the loaded model. custom_voice carries a
// speaker table ; base and voice_design return 0.
QT_API int qt_n_speakers(const struct qt_context * q);

// Name of speaker i, valid for i in [0, qt_n_speakers). Returns NULL when
// i is out of range. The pointer stays valid until qt_free. UTF-8.
QT_API const char * qt_speaker_name(const struct qt_context * q, int i);

#ifdef __cplusplus
}
#endif
