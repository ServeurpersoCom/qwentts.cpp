// qwen.cpp: public ABI implementation.
//
// Every entry declared in qwen.h lives here under one extern "C" block
// so the symbols carry C linkage and are linkable from C, Rust, Go,
// Python ctypes and any other binding generator. The struct
// qt_context opaque handle owns one BackendPair, one PipelineTTS
// (which already embeds the PipelineCodec) and one BPETokenizer.
// qt_init walks the load chain in dependency order and unwinds
// whatever it already allocated when any step fails. qt_free mirrors
// that order in reverse.
//
// This translation unit also absorbs the internal qt_set_error /
// qt_throw / qt_log helpers that the rest of the codebase calls. The
// log callback installed via qt_log_set routes every diagnostic from
// any caller, internal or public.

#include "qwen.h"

#include "backend.h"
#include "bpe.h"
#include "pipeline-tts.h"
#include "qt-error.h"
#include "speaker-encoder-extract.h"
#include "version.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Internal definition of the opaque handle. C++ types are fine here
// because nothing in this struct ever crosses the public ABI boundary :
// callers only ever see `struct qt_context *`. PipelineTTS already
// embeds the PipelineCodec, so no separate codec field is needed.
struct qt_context {
    BackendPair  bp;
    PipelineTTS  pt;
    BPETokenizer tok;
};

// Thread-local backing store for qt_last_error(). std::string sized once
// per thread, grows on demand, never freed across calls: the std runtime
// reclaims it on thread exit. An empty string means "no error recorded
// on this thread yet", which qt_last_error() exposes as "".
static thread_local std::string g_last_error;

void qt_set_error_v(const char * fmt, va_list ap) {
    if (!fmt) {
        g_last_error.clear();
        return;
    }
    // Two-pass vsnprintf: first call sizes the buffer, second writes the
    // message. va_copy keeps the original ap valid for the second pass.
    va_list ap2;
    va_copy(ap2, ap);
    int needed = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0) {
        g_last_error = "qt_set_error: vsnprintf failed";
        return;
    }
    g_last_error.resize(static_cast<size_t>(needed));
    std::vsnprintf(g_last_error.data(), static_cast<size_t>(needed) + 1, fmt, ap);
}

void qt_set_error(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    qt_set_error_v(fmt, ap);
    va_end(ap);
}

// Formats a message with printf semantics and throws std::runtime_error.
// The catch site at the binary entry inspects the what() string and feeds
// it into qt_set_error so the user-visible diagnostic is identical
// whether the failure used the bool-return path or the throw path.
void qt_throw(const char * fmt, ...) {
    char buf[1024];
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
    } else {
        buf[0] = '\0';
    }
    throw std::runtime_error(buf);
}

// Process-wide log callback. Atomic so qt_log_set can replace it without
// locking: write happens with memory_order_release, every reader sees a
// fully published callback pointer paired with its user_data slot.
// std::atomic on a function pointer is lock-free on every platform we
// target. user_data is a plain pointer because it is only ever published
// alongside cb under the same release ordering.
static std::atomic<qt_log_cb> g_log_cb{ nullptr };
static void *                 g_log_cb_user = nullptr;

// Routes one log line to the installed callback or to stderr. Two-pass
// vsnprintf sizes the heap buffer when the message exceeds the stack
// scratchpad, which keeps the common case allocation-free.
void qt_log(qt_log_level level, const char * fmt, ...) {
    if (!fmt) {
        return;
    }
    char    stackbuf[512];
    char *  buf    = stackbuf;
    int     needed = 0;
    va_list ap;
    va_start(ap, fmt);
    {
        va_list ap2;
        va_copy(ap2, ap);
        needed = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap2);
        va_end(ap2);
    }
    if (needed < 0) {
        va_end(ap);
        return;
    }
    std::string heapbuf;
    if ((size_t) needed >= sizeof(stackbuf)) {
        heapbuf.resize((size_t) needed);
        std::vsnprintf(heapbuf.data(), (size_t) needed + 1, fmt, ap);
        buf = heapbuf.data();
    }
    va_end(ap);

    qt_log_cb cb = g_log_cb.load(std::memory_order_acquire);
    if (cb) {
        cb(level, buf, g_log_cb_user);
    } else {
        std::fprintf(stderr, "%s\n", buf);
    }
}

// Resolve a -1 seed to a hardware random 64-bit value. Anything else is
// forwarded verbatim, so reproducibility is one explicit seed away. The
// resolved value travels into pipeline_tts_synthesize so the dump traces
// log the exact seed that drove the sampler, even when the caller asked
// for non determinism.
static int64_t qt_resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    std::random_device rd;
    return (int64_t) (((uint64_t) rd() << 32) ^ (uint64_t) rd());
}

extern "C" {

const char * qt_version(void) {
    // QWEN_VERSION is a string literal injected by tools/version.cmake
    // ("<git-hash> (<date>)"), so its storage already has process
    // lifetime and no formatting wrapper is needed.
    return QWEN_VERSION;
}

const char * qt_last_error(void) {
    // c_str() on an empty std::string is guaranteed to point to a NUL
    // byte by C++11, so callers never have to NULL-check the result.
    return g_last_error.c_str();
}

void qt_audio_free(struct qt_audio * a) {
    if (!a) {
        return;
    }
    if (a->samples) {
        std::free(a->samples);
    }
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void qt_log_set(qt_log_cb cb, void * user_data) {
    g_log_cb_user = user_data;
    g_log_cb.store(cb, std::memory_order_release);
}

void qt_init_default_params(struct qt_init_params * p) {
    p->abi_version = QT_ABI_VERSION;
    p->talker_path = nullptr;
    p->codec_path  = nullptr;
    p->use_fa      = true;
    p->clamp_fp16  = false;
}

void qt_tts_default_params(struct qt_tts_params * p) {
    p->abi_version            = QT_ABI_VERSION;
    p->text                   = nullptr;
    p->lang                   = nullptr;
    p->instruct               = nullptr;
    p->speaker                = nullptr;
    p->ref_audio_24k          = nullptr;
    p->ref_n_samples          = 0;
    p->ref_text               = nullptr;
    p->seed                   = -1;
    p->max_new_tokens         = 2048;
    p->do_sample              = true;
    p->temperature            = 0.9f;
    p->top_k                  = 50;
    p->top_p                  = 1.0f;
    p->repetition_penalty     = 1.05f;
    p->subtalker_do_sample    = true;
    p->subtalker_temperature  = 0.9f;
    p->subtalker_top_k        = 50;
    p->subtalker_top_p        = 1.0f;
    p->dump_dir               = nullptr;
    p->cancel                 = nullptr;
    p->cancel_user_data       = nullptr;
    p->on_chunk               = nullptr;
    p->on_chunk_user_data     = nullptr;
    p->codec_chunk_sec        = 24.0f;
    p->codec_left_context_sec = 2.0f;
    p->ref_spk_emb            = nullptr;
    p->ref_spk_dim            = 0;
    p->ref_codes              = nullptr;
    p->ref_T                  = 0;
}

int qt_num_codebooks(const struct qt_context * q) {
    if (!q) {
        qt_set_error("qt_num_codebooks: q is NULL");
        return 0;
    }
    return q->pt.num_code_groups;
}

struct qt_context * qt_init(const struct qt_init_params * params) {
    if (!params || !params->talker_path || !params->codec_path) {
        qt_set_error("qt_init: params, talker_path or codec_path is NULL");
        qt_log(QT_LOG_ERROR, "[Qwen] qt_init requires talker_path and codec_path");
        return nullptr;
    }
    if (params->abi_version > QT_ABI_VERSION) {
        qt_set_error("qt_init: params->abi_version %d > QT_ABI_VERSION %d (binding compiled against a newer header)",
                     params->abi_version, QT_ABI_VERSION);
        qt_log(QT_LOG_ERROR, "[Qwen] qt_init params struct is from a newer ABI (%d > %d)", params->abi_version,
               QT_ABI_VERSION);
        return nullptr;
    }

    qt_log(QT_LOG_INFO, "[Qwen] qwentts.cpp %s", qt_version());

    // new qt_context() value-initialises every field: POD aggregates
    // (BackendPair, PipelineTTS) are zero-init, std containers in
    // BPETokenizer construct empty.
    qt_context * q = new qt_context();

    // The load chain runs inside a try block. Any failure deep in the
    // GGUF reader, the codec load or the LM weight load throws via
    // qt_throw; the catch funnels every variant into one cleanup via
    // qt_free, which is idempotent on partial state (NULL-safe sched,
    // NULL GGUF handles, refcount-correct backend release).
    try {
        q->bp = backend_init("Talker");
        if (!q->bp.backend) {
            qt_throw("qt_init: backend_init failed (no GGML backend available)");
        }

        if (!pipeline_tts_load(&q->pt, params->talker_path, params->codec_path, q->bp, params->use_fa,
                               params->clamp_fp16)) {
            qt_throw("qt_init: pipeline_tts_load failed for '%s' / '%s'", params->talker_path, params->codec_path);
        }

        // BPE tokenizer payload lives inside the talker GGUF. Load the
        // base vocab + the qwen3-tts text specials in one shot. The
        // specials key list matches the keys written by the conversion
        // script under the qwen3-tts.text.* namespace.
        if (!load_bpe_from_gguf(&q->tok, params->talker_path)) {
            qt_throw("qt_init: load_bpe_from_gguf failed for '%s'", params->talker_path);
        }
        const char * specials_keys[] = {
            "qwen3-tts.text.im_start_id", "qwen3-tts.text.im_end_id",  "qwen3-tts.text.tts_pad_id",
            "qwen3-tts.text.tts_bos_id",  "qwen3-tts.text.tts_eos_id",
        };
        bpe_load_specials_from_keys(&q->tok, params->talker_path, specials_keys, 5);
    } catch (const std::exception & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "[Qwen] %s", e.what());
        qt_free(q);
        return nullptr;
    }

    return q;
}

void qt_free(struct qt_context * q) {
    if (!q) {
        return;
    }
    pipeline_tts_free(&q->pt);
    backend_release(q->bp.backend, q->bp.cpu_backend);
    delete q;
}

void qt_voice_ref_free(struct qt_voice_ref * ref) {
    if (!ref) {
        return;
    }
    if (ref->ref_spk_emb) {
        std::free(ref->ref_spk_emb);
    }
    if (ref->ref_codes) {
        std::free(ref->ref_codes);
    }
    ref->ref_spk_emb   = nullptr;
    ref->ref_spk_dim   = 0;
    ref->ref_codes     = nullptr;
    ref->ref_T         = 0;
    ref->num_codebooks = 0;
}

enum qt_status qt_extract_voice_ref(struct qt_context *   q,
                                    const float *         ref_audio_24k,
                                    int                   ref_n_samples,
                                    struct qt_voice_ref * out) {
    if (out) {
        qt_voice_ref_free(out);
    }
    if (!q || !ref_audio_24k || !out) {
        qt_set_error("qt_extract_voice_ref: q, ref_audio_24k or out is NULL");
        return QT_STATUS_INVALID_PARAMS;
    }
    if (ref_n_samples < TOKENIZER_HOP_LENGTH) {
        qt_set_error("qt_extract_voice_ref: ref_audio_24k too short for RVQ encode (%d samples, need at least %d)",
                     ref_n_samples, TOKENIZER_HOP_LENGTH);
        return QT_STATUS_INVALID_PARAMS;
    }

    const std::string & mt = q->pt.model_type;
    if (mt != "base") {
        qt_set_error("qt_extract_voice_ref: voice references are only valid for base models (loaded: %s)", mt.c_str());
        return QT_STATUS_MODE_INVALID;
    }
    if (!q->pt.has_speaker_encoder) {
        qt_set_error("qt_extract_voice_ref: loaded base model has no speaker encoder");
        return QT_STATUS_GENERATE_FAILED;
    }
    if (q->pt.num_code_groups <= 0) {
        qt_set_error("qt_extract_voice_ref: invalid codebook count %d", q->pt.num_code_groups);
        return QT_STATUS_GENERATE_FAILED;
    }

    try {
        std::vector<float> emb;
        if (!speaker_encoder_extract(&q->pt.speaker_encoder, q->pt.sched, ref_audio_24k, ref_n_samples, emb)) {
            qt_set_error("qt_extract_voice_ref: speaker embedding extraction failed");
            return QT_STATUS_GENERATE_FAILED;
        }
        if ((int) emb.size() != q->pt.talker.hidden_size) {
            qt_set_error("qt_extract_voice_ref: speaker embedding size %zu mismatches talker hidden %d", emb.size(),
                         q->pt.talker.hidden_size);
            return QT_STATUS_GENERATE_FAILED;
        }

        const int            aligned_n = (ref_n_samples / TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH;
        const int            ref_T     = aligned_n / TOKENIZER_HOP_LENGTH;
        std::vector<int32_t> codes     = pipeline_codec_encode(&q->pt.codec, ref_audio_24k, aligned_n);
        if (codes.empty()) {
            qt_set_error("qt_extract_voice_ref: pipeline_codec_encode returned empty codes");
            return QT_STATUS_GENERATE_FAILED;
        }
        const int num_codebooks = q->pt.num_code_groups;
        if ((codes.size() % (size_t) num_codebooks) != 0) {
            qt_set_error("qt_extract_voice_ref: encoded code count %zu is not divisible by %d", codes.size(),
                         num_codebooks);
            return QT_STATUS_GENERATE_FAILED;
        }
        const int codes_T = (int) (codes.size() / (size_t) num_codebooks);
        if (codes_T != ref_T) {
            qt_set_error("qt_extract_voice_ref: encoded frame count %d mismatches aligned frame count %d", codes_T,
                         ref_T);
            return QT_STATUS_GENERATE_FAILED;
        }

        const size_t emb_bytes   = emb.size() * sizeof(float);
        const size_t codes_bytes = codes.size() * sizeof(int32_t);
        float *      emb_copy    = (float *) std::malloc(emb_bytes);
        int32_t *    codes_copy  = (int32_t *) std::malloc(codes_bytes);
        if (!emb_copy || !codes_copy) {
            std::free(emb_copy);
            std::free(codes_copy);
            qt_set_error("qt_extract_voice_ref: malloc failed for %zu emb bytes and %zu code bytes", emb_bytes,
                         codes_bytes);
            return QT_STATUS_OOM;
        }
        std::memcpy(emb_copy, emb.data(), emb_bytes);
        std::memcpy(codes_copy, codes.data(), codes_bytes);

        out->ref_spk_emb   = emb_copy;
        out->ref_spk_dim   = (int) emb.size();
        out->ref_codes     = codes_copy;
        out->ref_T         = ref_T;
        out->num_codebooks = num_codebooks;

        qt_log(QT_LOG_INFO, "[Qwen] Extracted voice ref: spk_dim=%d K=%d T=%d (%d/%d samples)", out->ref_spk_dim,
               out->num_codebooks, out->ref_T, aligned_n, ref_n_samples);
        return QT_STATUS_OK;
    } catch (const std::bad_alloc &) {
        qt_set_error("qt_extract_voice_ref: out of memory");
        qt_voice_ref_free(out);
        return QT_STATUS_OOM;
    } catch (const std::exception & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "[Qwen] %s", e.what());
        qt_voice_ref_free(out);
        return QT_STATUS_GENERATE_FAILED;
    }
}

enum qt_status qt_synthesize(struct qt_context * q, const struct qt_tts_params * params, struct qt_audio * out) {
    if (!q || !params) {
        qt_set_error("qt_synthesize: q or params is NULL");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_INVALID_PARAMS;
    }
    // Streaming mode (on_chunk non NULL) emits through the callback and
    // leaves out unused, so out=NULL is valid there. Buffered mode
    // requires out to receive the synthesised waveform.
    if (!params->on_chunk && !out) {
        qt_set_error("qt_synthesize: out is NULL in buffered mode");
        return QT_STATUS_INVALID_PARAMS;
    }
    if (params->abi_version > QT_ABI_VERSION) {
        qt_set_error(
            "qt_synthesize: params->abi_version %d > QT_ABI_VERSION %d (binding compiled against a newer header)",
            params->abi_version, QT_ABI_VERSION);
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_INVALID_PARAMS;
    }

    if (!params->text || !params->text[0]) {
        qt_set_error("qt_synthesize: params->text is NULL or empty");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_INVALID_PARAMS;
    }

    // Mode validation. Mirrors the upstream Python which raises
    // ValueError when generate_voice_design is called on a non
    // voice_design model and the same shape applies to
    // generate_custom_voice. Explicit and KISS, so the caller never
    // gets a silently wrong synthesis. Messages preserved verbatim
    // from the previous CLI-side checks.
    const std::string & mt = q->pt.model_type;
    if (params->speaker && mt != "custom_voice") {
        qt_set_error("--speaker is only valid for custom_voice models (loaded: %s)", mt.c_str());
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_MODE_INVALID;
    }
    if (params->instruct && mt == "base") {
        qt_set_error("--instruct is not supported for base models");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_MODE_INVALID;
    }
    if (mt == "custom_voice" && !params->speaker) {
        qt_set_error("custom_voice models require --speaker");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_MODE_INVALID;
    }
    if (mt == "voice_design" && (!params->instruct || params->instruct[0] == '\0')) {
        qt_set_error("voice_design models require --instruct");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_MODE_INVALID;
    }
    // ABI v2 latent reference fields, same gate as the pipeline.
    const bool has_lat_spk   = params->abi_version >= 2 && params->ref_spk_emb && params->ref_spk_dim > 0;
    const bool has_lat_codes = params->abi_version >= 2 && params->ref_codes && params->ref_T > 0;

    if ((params->ref_audio_24k || has_lat_spk) && mt != "base") {
        qt_set_error("--ref-wav / --ref-spk is only valid for base models (loaded: %s)", mt.c_str());
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_MODE_INVALID;
    }
    if (params->speaker && (params->ref_audio_24k || has_lat_spk)) {
        qt_set_error("--speaker and --ref-wav / --ref-spk are mutually exclusive");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_INVALID_PARAMS;
    }
    if (params->ref_text && !params->ref_audio_24k && !has_lat_codes) {
        qt_set_error("--ref-text requires --ref-wav or --ref-rvq");
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_INVALID_PARAMS;
    }

    // Defense in depth: the synthesis path normally reports failures
    // via qt_status return + qt_set_error. A future load-style throw or
    // any std::bad_alloc deep inside the GGML backend is caught here
    // and converted to QT_STATUS_GENERATE_FAILED so an exception never
    // crosses the extern "C" boundary.
    try {
        const int64_t resolved_seed = qt_resolve_seed(params->seed);
        return pipeline_tts_synthesize(&q->pt, &q->tok, params, resolved_seed, out);
    } catch (const std::exception & e) {
        qt_set_error("%s", e.what());
        qt_log(QT_LOG_ERROR, "[Qwen] %s", e.what());
        if (out) {
            qt_audio_free(out);
        }
        return QT_STATUS_GENERATE_FAILED;
    }
}

int qt_duration_sec_to_tokens(const struct qt_context * q, float duration_sec) {
    if (!q) {
        qt_set_error("qt_duration_sec_to_tokens: q is NULL");
        qt_log(QT_LOG_ERROR, "[Qwen] qt_duration_sec_to_tokens requires a valid handle");
        return 1;
    }
    return pipeline_tts_duration_sec_to_tokens(&q->pt, duration_sec);
}

int qt_n_speakers(const struct qt_context * q) {
    if (!q) {
        qt_set_error("qt_n_speakers: q is NULL");
        return 0;
    }
    return (int) q->pt.speakers.size();
}

const char * qt_speaker_name(const struct qt_context * q, int i) {
    if (!q || i < 0 || i >= (int) q->pt.speakers.size()) {
        return NULL;
    }
    return q->pt.speakers[(size_t) i].name.c_str();
}

}  // extern "C"
