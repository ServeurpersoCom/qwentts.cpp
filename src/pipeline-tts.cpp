// pipeline-tts.cpp: load and verify both GGUF files (talker + codec)
// onto the same shared backend, parse all metadata into typed structs,
// and provide a structured load-time summary for --load-only mode.

#include "pipeline-tts.h"

#include "audio-io.h"
#include "bpe.h"
#include "code-predictor-forward.h"
#include "codec-chunked-decode.h"
#include "debug.h"
#include "ggml.h"
#include "pipeline-codec.h"
#include "prompt-builder.h"
#include "qt-error.h"
#include "sampling.h"
#include "speaker-encoder-extract.h"
#include "talker-forward.h"
#include "timer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void parse_codec_specials(const GGUFModel & gf, CodecSpecials & cs) {
    cs.pad_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.pad_id");
    cs.bos_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.bos_id");
    cs.eos_id       = (int) gf_get_u32(gf, "qwen3-tts.codec.eos_id");
    cs.think_id     = (int) gf_get_u32(gf, "qwen3-tts.codec.think_id");
    cs.nothink_id   = (int) gf_get_u32(gf, "qwen3-tts.codec.nothink_id");
    cs.think_bos_id = (int) gf_get_u32(gf, "qwen3-tts.codec.think_bos_id");
    cs.think_eos_id = (int) gf_get_u32(gf, "qwen3-tts.codec.think_eos_id");
}

static void parse_text_specials(const GGUFModel & gf, TextSpecials & ts) {
    ts.im_start_id = (int) gf_get_u32(gf, "qwen3-tts.text.im_start_id");
    ts.im_end_id   = (int) gf_get_u32(gf, "qwen3-tts.text.im_end_id");
    ts.tts_pad_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_pad_id");
    ts.tts_bos_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_bos_id");
    ts.tts_eos_id  = (int) gf_get_u32(gf, "qwen3-tts.text.tts_eos_id");
}

static void parse_languages(const GGUFModel & gf, std::vector<LanguageEntry> & out) {
    int64_t name_idx = gguf_find_key(gf.gguf, "qwen3-tts.codec.language_names");
    int64_t id_idx   = gguf_find_key(gf.gguf, "qwen3-tts.codec.language_ids");
    if (name_idx < 0 || id_idx < 0) {
        return;
    }
    size_t n_names = gguf_get_arr_n(gf.gguf, name_idx);
    size_t n_ids   = gguf_get_arr_n(gf.gguf, id_idx);
    if (n_names != n_ids) {
        qt_log(QT_LOG_WARN, "[Pipeline] language arrays size mismatch (names=%zu, ids=%zu)", n_names, n_ids);
        return;
    }
    const uint32_t * ids = (const uint32_t *) gguf_get_arr_data(gf.gguf, id_idx);
    out.reserve(n_names);
    for (size_t i = 0; i < n_names; i++) {
        LanguageEntry e;
        e.name = gguf_get_arr_str(gf.gguf, name_idx, i);
        e.id   = (int) ids[i];
        out.push_back(e);
    }
}

// Parse the speaker table for CustomVoice variants. Three parallel arrays
// produced by convert.py: speaker_names, speaker_ids, speaker_dialects.
// Empty dialect string means the speaker keeps the user supplied language.
// Skipped silently when the GGUF carries no speaker table (Base / VoiceDesign).
static void parse_speakers(const GGUFModel & gf, std::vector<SpeakerEntry> & out) {
    int64_t name_idx    = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_names");
    int64_t id_idx      = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_ids");
    int64_t dialect_idx = gguf_find_key(gf.gguf, "qwen3-tts.codec.speaker_dialects");
    if (name_idx < 0 || id_idx < 0 || dialect_idx < 0) {
        return;
    }
    size_t n_names    = gguf_get_arr_n(gf.gguf, name_idx);
    size_t n_ids      = gguf_get_arr_n(gf.gguf, id_idx);
    size_t n_dialects = gguf_get_arr_n(gf.gguf, dialect_idx);
    if (n_names != n_ids || n_names != n_dialects) {
        qt_log(QT_LOG_WARN, "[Pipeline] speaker arrays size mismatch (names=%zu, ids=%zu, dialects=%zu)", n_names,
               n_ids, n_dialects);
        return;
    }
    const uint32_t * ids = (const uint32_t *) gguf_get_arr_data(gf.gguf, id_idx);
    out.reserve(n_names);
    for (size_t i = 0; i < n_names; i++) {
        SpeakerEntry e;
        e.name    = gguf_get_arr_str(gf.gguf, name_idx, i);
        e.id      = (int) ids[i];
        e.dialect = gguf_get_arr_str(gf.gguf, dialect_idx, i);
        out.push_back(e);
    }
}

static void parse_generation_defaults(const GGUFModel & gf, GenerationDefaults & g) {
    g.do_sample             = gf_get_bool(gf, "generation.do_sample");
    g.top_k                 = (int) gf_get_u32(gf, "generation.top_k");
    g.top_p                 = gf_get_f32(gf, "generation.top_p");
    g.temperature           = gf_get_f32(gf, "generation.temperature");
    g.repetition_penalty    = gf_get_f32(gf, "generation.repetition_penalty");
    g.subtalker_do_sample   = gf_get_bool(gf, "generation.subtalker_do_sample");
    g.subtalker_top_k       = (int) gf_get_u32(gf, "generation.subtalker_top_k");
    g.subtalker_top_p       = gf_get_f32(gf, "generation.subtalker_top_p");
    g.subtalker_temperature = gf_get_f32(gf, "generation.subtalker_temperature");
    g.max_new_tokens        = (int) gf_get_u32(gf, "generation.max_new_tokens");
}

// Ensure the static predictor graph set for batch width N exists:
// prefill (T=2 through lm_head[0]) plus one T=1 step per acoustic
// codebook after the first. Built lazily on the first frame at a given
// width, then replayed for the process lifetime.
static bool pipeline_tts_cp_graphs_ensure(PipelineTTS * pt, int N) {
    if ((int) pt->cp_graphs.size() < N) {
        pt->cp_graphs.resize((size_t) N);
    }
    CodePredGraphSet & s = pt->cp_graphs[(size_t) (N - 1)];
    if (s.prefill.ctx) {
        return true;
    }
    if (!code_predictor_graph_build(&pt->code_predictor, &pt->code_predictor_kv, pt->backend,
                                    pt->talker.codec_embedding, pt->hidden_bridge, 0, N, pt->use_flash_attn,
                                    pt->clamp_fp16, &s.prefill)) {
        return false;
    }
    s.steps.resize((size_t) (pt->num_code_groups - 2));
    for (size_t g = 0; g < s.steps.size(); g++) {
        if (!code_predictor_graph_build(&pt->code_predictor, &pt->code_predictor_kv, pt->backend,
                                        pt->code_predictor.codec_embedding[g], NULL, (int) g + 1, N, pt->use_flash_attn,
                                        pt->clamp_fp16, &s.steps[g])) {
            return false;
        }
    }
    return true;
}

bool pipeline_tts_load(PipelineTTS * pt,
                       const char *  talker_gguf_path,
                       const char *  codec_gguf_path,
                       BackendPair   bp,
                       bool          use_fa,
                       bool          clamp_fp16,
                       int           max_batch) {
    pt->bp                  = bp;
    pt->backend             = bp.backend;
    pt->sched               = NULL;
    pt->has_speaker_encoder = false;
    pt->bridge_ctx          = NULL;
    pt->bridge_buf          = NULL;
    pt->hidden_bridge       = NULL;
    pt->max_batch           = max_batch > 1 ? max_batch : 1;

    // Fused flash attention needs a GPU kernel; CPU only backends fall
    // back to the F32 manual chain automatically. clamp_fp16 is forwarded
    // verbatim: a no op on backends that already accumulate in F32, an
    // FP16 overflow guard on sub Ampere CUDA tensor cores.
    pt->use_flash_attn = use_fa && bp.has_gpu;
    pt->clamp_fp16     = clamp_fp16;

    if (!gf_load(&pt->gguf_talker, talker_gguf_path)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] failed to load talker GGUF: %s", talker_gguf_path);
        return false;
    }

    const char * arch = gf_get_str(pt->gguf_talker, "general.architecture");
    if (!arch || std::strcmp(arch, "qwen3-tts") != 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] talker GGUF has wrong architecture '%s', expected 'qwen3-tts'",
               arch ? arch : "");
        gf_close(&pt->gguf_talker);
        return false;
    }

    pt->tokenizer_type  = gf_get_str(pt->gguf_talker, "qwen3-tts.tokenizer_type");
    pt->model_size      = gf_get_str(pt->gguf_talker, "qwen3-tts.model_size");
    pt->model_type      = gf_get_str(pt->gguf_talker, "qwen3-tts.model_type");
    pt->num_code_groups = (int) gf_get_u32(pt->gguf_talker, "qwen3-tts.num_code_groups");

    parse_codec_specials(pt->gguf_talker, pt->codec_specials);
    parse_text_specials(pt->gguf_talker, pt->text_specials);
    parse_languages(pt->gguf_talker, pt->languages);
    parse_speakers(pt->gguf_talker, pt->speakers);
    parse_generation_defaults(pt->gguf_talker, pt->gen_defaults);

    if (!talker_weights_load(&pt->talker, pt->gguf_talker, pt->backend)) {
        gf_close(&pt->gguf_talker);
        return false;
    }

    if (!code_predictor_weights_load(&pt->code_predictor, pt->gguf_talker, pt->backend)) {
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Speaker encoder tensors are only present in Base checkpoints. The
    // weights load lazily on the first --ref-wav request: synthesis from
    // pre extracted embeddings never pays for them. has_speaker_encoder
    // advertises the capability, spk_enc_loaded tracks residency.
    pt->has_speaker_encoder = (pt->model_type == "base");
    pt->spk_enc_loaded      = false;

    if (!pipeline_codec_load(&pt->codec, codec_gguf_path, bp)) {
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Scheduler shared by talker_forward_* and code_predictor_step.
    // Routes ops the GPU backend cannot run (typical case: K-quant
    // get_rows on CUDA) to the CPU backend. 4096 nodes covers the 28L
    // Qwen3 talker graph (~48 ops per layer with KV cache writes) with
    // headroom; the 5L code predictor uses a fraction of that.
    pt->sched = backend_sched_new(bp, 4096);
    if (!pt->sched) {
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Prompt cache: special embeds projected once on the backend, prefix
    // cache primed empty. Requires the sched, so it runs after sched_new.
    if (!prompt_cache_load(pt)) {
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // KV caches, one set per slot: the talker holds the LM context up
    // to 4096 positions (the longest ICL prompt observed is ~250 +
    // max_new_tokens ~ 1500, so 4096 has 60% headroom). Predictor holds
    // one frame of 16 sub-steps per slot.
    if (!kv_cache_init(&pt->talker_kv, pt->talker.num_hidden_layers, pt->talker.num_key_value_heads,
                       pt->talker.head_dim, 4096, pt->max_batch, pt->backend)) {
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }
    if (!kv_cache_init(&pt->code_predictor_kv, pt->code_predictor.num_hidden_layers,
                       pt->code_predictor.num_key_value_heads, pt->code_predictor.head_dim, pt->num_code_groups,
                       pt->max_batch, pt->backend)) {
        kv_cache_free(&pt->talker_kv);
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Hidden bridge: one [talker_hidden, max_batch] f32 tensor resident
    // on the backend, columns written by the talker graphs and read by
    // the code predictor prefill graph. Cleared once so debug dumps
    // never see stale bytes before the first talker forward.
    {
        struct ggml_init_params gp = { ggml_tensor_overhead() * 2, NULL, true };
        pt->bridge_ctx             = ggml_init(gp);
        pt->hidden_bridge =
            pt->bridge_ctx ? ggml_new_tensor_2d(pt->bridge_ctx, GGML_TYPE_F32, pt->talker.hidden_size, pt->max_batch) :
                             NULL;
        if (pt->hidden_bridge) {
            ggml_set_name(pt->hidden_bridge, "talker_hidden_bridge");
            pt->bridge_buf = ggml_backend_alloc_ctx_tensors(pt->bridge_ctx, pt->backend);
        }
        if (!pt->hidden_bridge || !pt->bridge_buf) {
            qt_log(QT_LOG_ERROR, "[Pipeline] hidden bridge allocation failed");
            if (pt->bridge_ctx) {
                ggml_free(pt->bridge_ctx);
                pt->bridge_ctx = NULL;
            }
            pt->hidden_bridge = NULL;
            kv_cache_free(&pt->code_predictor_kv);
            kv_cache_free(&pt->talker_kv);
            ggml_backend_sched_free(pt->sched);
            pt->sched = NULL;
            pipeline_codec_free(&pt->codec);
            code_predictor_weights_free(&pt->code_predictor);
            talker_weights_free(&pt->talker);
            gf_close(&pt->gguf_talker);
            return false;
        }
        ggml_backend_buffer_clear(pt->bridge_buf, 0);
    }

    // Talker graph arena plus the batch width 1 predictor graph set:
    // the talker keeps one arena per shape class for the CUDA graph
    // cache, the predictor builds one static graph per flavor (prefill
    // + one per acoustic step) with its lm_head and embedding table
    // fixed and the positions, kv rows, and mask baked in. Wider
    // predictor sets and the batched talker decode graphs build lazily
    // on first use.
    pt->talker_decode_graphs.resize(((size_t) pt->talker_kv.max_seq_len + 255) / 256);
    bool graphs_ok = graph_arena_init(&pt->talker_arena, talker_graph_max_nodes(pt->talker.num_hidden_layers)) &&
                     pipeline_tts_cp_graphs_ensure(pt, 1);
    if (!graphs_ok) {
        for (size_t n = 0; n < pt->cp_graphs.size(); n++) {
            code_predictor_graph_free(&pt->cp_graphs[n].prefill);
            for (size_t g = 0; g < pt->cp_graphs[n].steps.size(); g++) {
                code_predictor_graph_free(&pt->cp_graphs[n].steps[g]);
            }
        }
        pt->cp_graphs.clear();
        pt->talker_decode_graphs.clear();
        graph_arena_free(&pt->talker_arena);
        ggml_backend_buffer_free(pt->bridge_buf);
        pt->bridge_buf = NULL;
        ggml_free(pt->bridge_ctx);
        pt->bridge_ctx    = NULL;
        pt->hidden_bridge = NULL;
        kv_cache_free(&pt->code_predictor_kv);
        kv_cache_free(&pt->talker_kv);
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    qt_log(QT_LOG_INFO,
           "[Pipeline] Loaded: arch=%s variant=%s tokenizer=%s codebooks=%d speaker_encoder=%s speakers=%zu fa=%s "
           "clamp_fp16=%s max_batch=%d",
           pt->model_size.c_str(), pt->model_type.c_str(), pt->tokenizer_type.c_str(), pt->num_code_groups,
           pt->has_speaker_encoder ? "deferred" : "absent", pt->speakers.size(), pt->use_flash_attn ? "on" : "off",
           pt->clamp_fp16 ? "on" : "off", pt->max_batch);
    return true;
}

void pipeline_tts_free(PipelineTTS * pt) {
    for (size_t n = 0; n < pt->cp_graphs.size(); n++) {
        code_predictor_graph_free(&pt->cp_graphs[n].prefill);
        for (size_t g = 0; g < pt->cp_graphs[n].steps.size(); g++) {
            code_predictor_graph_free(&pt->cp_graphs[n].steps[g]);
        }
    }
    pt->cp_graphs.clear();
    for (size_t g = 0; g < pt->talker_decode_graphs.size(); g++) {
        talker_decode_graph_free(&pt->talker_decode_graphs[g]);
    }
    pt->talker_decode_graphs.clear();
    graph_arena_free(&pt->talker_arena);
    if (pt->bridge_buf) {
        ggml_backend_buffer_free(pt->bridge_buf);
        pt->bridge_buf = NULL;
    }
    if (pt->bridge_ctx) {
        ggml_free(pt->bridge_ctx);
        pt->bridge_ctx = NULL;
    }
    pt->hidden_bridge = NULL;
    kv_cache_free(&pt->code_predictor_kv);
    kv_cache_free(&pt->talker_kv);
    if (pt->sched) {
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
    }
    pipeline_codec_free(&pt->codec);
    if (pt->spk_enc_loaded) {
        speaker_encoder_weights_free(&pt->speaker_encoder);
        pt->spk_enc_loaded = false;
    }
    code_predictor_weights_free(&pt->code_predictor);
    talker_weights_free(&pt->talker);
    gf_close(&pt->gguf_talker);
    pt->prompt_cache        = {};
    pt->backend             = NULL;
    pt->bp                  = {};
    pt->has_speaker_encoder = false;
}

// Pull one row of an embedding table directly from the GGUF mmap. Used
// in the generation loop to assemble the next-token embedding (sum of 16
// codebook embeddings) without paying for a backend round-trip per row.
static void embed_row_from_gguf(const GGUFModel & gf, const char * tensor_name, int row_id, int hidden, float * dst) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, tensor_name);
    if (!src) {
        qt_throw("[Pipeline] tensor not found in GGUF: %s", tensor_name);
    }
    const uint8_t * base = (const uint8_t *) gf_get_data(gf, tensor_name);
    if (!base) {
        qt_throw("[Pipeline] tensor data missing in GGUF: %s", tensor_name);
    }
    const size_t row_bytes = ggml_row_size(src->type, hidden);
    const void * row       = base + (size_t) row_id * row_bytes;
    if (src->type == GGML_TYPE_F32) {
        std::memcpy(dst, row, (size_t) hidden * sizeof(float));
        return;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(src->type);
    if (!tt || !tt->to_float) {
        qt_throw("[Pipeline] unsupported codec_embedding dtype %d for %s", (int) src->type, tensor_name);
    }
    tt->to_float(row, dst, hidden);
}

// Helper: malloc a heap copy of a float vector and hand it off into the
// public qt_audio struct. Returns true on success; on OOM sets the
// error string and leaves out untouched. Empty vectors land as an
// allocation of size 0 with a stub malloc to keep the free path simple.
static bool fill_qt_audio(const std::vector<float> & audio, qt_audio * out) {
    const size_t n     = audio.size();
    const size_t bytes = n * sizeof(float);
    float *      buf   = (float *) std::malloc(bytes > 0 ? bytes : 1);
    if (!buf) {
        qt_set_error("pipeline_tts_synthesize: malloc failed for %zu samples", n);
        return false;
    }
    if (n > 0) {
        std::memcpy(buf, audio.data(), bytes);
    }
    out->samples     = buf;
    out->n_samples   = (int) n;
    out->sample_rate = TOKENIZER_SAMPLE_RATE;
    out->channels    = 1;
    return true;
}

int pipeline_tts_duration_sec_to_tokens(const PipelineTTS * /*pt*/, float duration_sec) {
    // The 12 Hz Qwen3-TTS tokenizer has a fixed hop of 1920 samples at
    // 24 kHz, so the frame rate is 24000 / 1920 = 12.5 Hz regardless of
    // the variant loaded. Clamp to a minimum of one frame so a zero or
    // negative duration still picks up one decoder step.
    const float fps      = (float) TOKENIZER_SAMPLE_RATE / (float) TOKENIZER_HOP_LENGTH;
    int         n_frames = (int) (duration_sec * fps + 0.5f);
    if (n_frames < 1) {
        n_frames = 1;
    }
    return n_frames;
}

// Per stage wall clock for one synthesis. Every span is measured around
// a call that ends on a device readback, so the GPU work is included.
struct TtsPerf {
    double build_ms;      // prompt builder
    double prefill_ms;    // talker prefill over T_ctx
    double ttfa_ms;       // entry to first frame codes ready
    double talker_ms;     // talker decode, summed over frames > 0
    double predictor_ms;  // code predictor step, summed over frames
    double host_ms;       // c0 sampling + next emb composition, summed
    double codec_ms;      // codec decode, streaming chunks + tail or buffered
    double total_ms;      // entry to return
    int    n_frames;      // emitted audio frames
};

static void tts_log_perf(const TtsPerf & p) {
    const double audio_sec = (double) p.n_frames * (double) TOKENIZER_HOP_LENGTH / (double) TOKENIZER_SAMPLE_RATE;
    const double rtf       = audio_sec > 0.0 ? (p.total_ms / 1000.0) / audio_sec : 0.0;
    const double per_frame = p.n_frames > 0 ? (p.talker_ms + p.predictor_ms + p.host_ms) / (double) p.n_frames : 0.0;

    qt_log(QT_LOG_INFO, "[Perf] PromptBuild %.1f ms", p.build_ms);
    qt_log(QT_LOG_INFO, "[Perf] Prefill %.1f ms (T_ctx prefill)", p.prefill_ms);
    qt_log(QT_LOG_INFO, "[Perf] TTFA %.1f ms (first frame codes)", p.ttfa_ms);
    qt_log(QT_LOG_INFO, "[Perf] TalkerDecode %.1f ms (%d frames, %.2f ms/frame)", p.talker_ms, p.n_frames,
           p.n_frames > 0 ? p.talker_ms / (double) p.n_frames : 0.0);
    qt_log(QT_LOG_INFO, "[Perf] CodePredictor %.1f ms (%.2f ms/frame)", p.predictor_ms,
           p.n_frames > 0 ? p.predictor_ms / (double) p.n_frames : 0.0);
    qt_log(QT_LOG_INFO, "[Perf] HostCompose %.1f ms (c0 sample + next emb)", p.host_ms);
    qt_log(QT_LOG_INFO, "[Perf] CodecDecode %.1f ms", p.codec_ms);
    qt_log(QT_LOG_INFO, "[Perf] Total %.1f ms (%d frames, %.2f ms/frame AR, audio %.2f s, RTF %.3f)", p.total_ms,
           p.n_frames, per_frame, audio_sec, rtf);
}

// ---------------------------------------------------------------------------
// Batch engine: up to pt->max_batch concurrent synthesis slots in
// lockstep. Slot i owns talker KV set i, predictor KV set i and bridge
// column i; the active range is always [0, N) so the batched decode
// and predictor graphs view consecutive sets. A retirement compacts
// the range by moving the tail slot (host state plus one device side
// talker KV set copy) into the freed index; the bridge and the
// predictor sets rewrite every frame so only the talker cache moves.
// Single threaded: every entry runs on the thread that owns the GPU.
// ---------------------------------------------------------------------------

struct TtsSlot {
    TtsJob * job;
    int64_t  serial;  // stable identity across compaction swaps

    PromptBuilderOutput prompt;

    // ICL reference codes kept for the codec: stream seeding at admit,
    // buffered decode left context at completion. ref_codes_ptr aims at
    // ref_codes_store or at the caller's latent buffer.
    std::vector<int32_t> ref_codes_store;
    const int32_t *      ref_codes_ptr;
    int                  ref_codes_T;

    // Resolved sampling temperatures (0 selects greedy).
    float talker_T;
    float subtk_T;

    // AR state, one to one with the single sequence loop.
    int                  step;            // frames emitted so far
    int64_t              subseq_counter;  // Philox subsequence cursor
    std::vector<int32_t> talker_history;  // emitted c0, feeds repetition penalty
    std::vector<int32_t> prev_ids;        // previous frame codes [num_code_groups]
    const float *        prev_overlay;    // trailing text row or tts_pad row
    std::vector<float>   logits;          // pending c0 logits [vocab]
    int                  pending_c0;      // c0 of the frame in flight
    bool                 has_frame;       // slot emits a frame this engine step

    // Streaming state: the per slot codec stream mirror parks in snap
    // whenever another slot takes the live codec state.
    bool                              streaming;
    codec_stream_decoder              stream;
    CodecStateSnap                    snap;
    std::vector<std::vector<int32_t>> all_codes;

    bool      finished;
    qt_status fin_status;
    TtsPerf   perf;
    Timer     t_total;
};

struct TtsEngine {
    PipelineTTS *        pt;
    BPETokenizer *       tok;
    std::vector<TtsSlot> slots;
    int64_t              next_serial;
    int64_t              stream_owner;  // serial of the slot holding the live codec state, -1 none
};

TtsEngine * tts_engine_new(PipelineTTS * pt, BPETokenizer * tok) {
    TtsEngine * e   = new TtsEngine();
    e->pt           = pt;
    e->tok          = tok;
    e->next_serial  = 0;
    e->stream_owner = -1;
    e->slots.reserve((size_t) pt->max_batch);
    return e;
}

void tts_engine_free(TtsEngine * e) {
    if (!e) {
        return;
    }
    for (TtsSlot & s : e->slots) {
        pipeline_codec_snap_free(&s.snap);
    }
    delete e;
}

int tts_engine_active(const TtsEngine * e) {
    return (int) e->slots.size();
}

// Make `s` the owner of the live codec stream state: park the current
// owner's state into its mirror, then restore this slot's. A slot with
// no mirror yet keeps whatever is live (its admit resets the state
// right after). With one streaming slot the ownership never moves and
// no copy is paid.
static bool tts_engine_codec_own(TtsEngine * e, TtsSlot * s) {
    if (e->stream_owner == s->serial) {
        return true;
    }
    if (e->stream_owner >= 0) {
        for (TtsSlot & o : e->slots) {
            if (o.serial == e->stream_owner) {
                if (!pipeline_codec_stream_save(&e->pt->codec, &o.snap)) {
                    return false;
                }
                break;
            }
        }
    }
    e->stream_owner = -1;
    if (s->snap.ctx && !pipeline_codec_stream_load(&e->pt->codec, &s->snap)) {
        return false;
    }
    e->stream_owner = s->serial;
    return true;
}

static bool tts_admit_fail(TtsJob * job, qt_status st) {
    job->status = st;
    job->error  = qt_last_error();
    return false;
}

bool tts_engine_admit(TtsEngine * e, TtsJob * job) {
    PipelineTTS *                pt     = e->pt;
    const struct qt_tts_params * params = job->params;
    job->status                         = QT_STATUS_OK;
    job->error.clear();

    if ((int) e->slots.size() >= pt->max_batch) {
        qt_set_error("tts_engine_admit: no free slot (%d active, max_batch %d)", (int) e->slots.size(), pt->max_batch);
        return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
    }

    const std::string instruct = params->instruct ? params->instruct : "";
    const std::string speaker  = params->speaker ? params->speaker : "";
    const std::string ref_text = params->ref_text ? params->ref_text : "";

    // ABI v2 latent reference fields. Callers compiled against ABI 1
    // never set them; the abi_version gate keeps their uninitialised
    // tail bytes out of the read path.
    const float *   lat_spk_emb = (params->abi_version >= 2) ? params->ref_spk_emb : NULL;
    const int       lat_spk_dim = (params->abi_version >= 2) ? params->ref_spk_dim : 0;
    const int32_t * lat_codes   = (params->abi_version >= 2) ? params->ref_codes : NULL;
    const int       lat_T       = (params->abi_version >= 2) ? params->ref_T : 0;

    const bool has_ref_audio = (params->ref_audio_24k != NULL) && (params->ref_n_samples > 0);
    const bool has_lat_spk   = (lat_spk_emb != NULL) && (lat_spk_dim > 0);
    const bool has_lat_codes = (lat_codes != NULL) && (lat_T > 0);

    // Raw waveform and pre-encoded latents are mutually exclusive: the
    // caller is told immediately rather than picking a winner silently.
    if (has_ref_audio && (has_lat_spk || has_lat_codes)) {
        qt_set_error("pipeline_tts_synthesize: ref_audio_24k and ref_spk_emb / ref_codes are mutually exclusive");
        qt_log(QT_LOG_ERROR, "[Pipeline] ref_audio_24k and ref_spk_emb / ref_codes are mutually exclusive");
        return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
    }
    // Latent ICL codes ride on top of the speaker embedding and need the
    // transcript, mirroring the raw path where mode B implies mode A.
    if (has_lat_codes && (!has_lat_spk || ref_text.empty())) {
        qt_set_error("pipeline_tts_synthesize: ref_codes requires ref_spk_emb and ref_text");
        qt_log(QT_LOG_ERROR, "[Pipeline] ref_codes requires ref_spk_emb and ref_text");
        return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
    }

    // Voice clone mode A: a pre-extracted latent embedding feeds the
    // prompt builder directly; otherwise, if ref_audio_24k is given, run
    // the speaker encoder on the pre-decoded mono buffer. Mutually
    // exclusive with --speaker.
    std::vector<float> ref_spk_emb;
    const float *      ref_spk_emb_ptr = NULL;
    if (has_lat_spk) {
        if (lat_spk_dim != pt->talker.hidden_size) {
            qt_set_error("pipeline_tts_synthesize: ref_spk_dim %d mismatches talker hidden %d", lat_spk_dim,
                         pt->talker.hidden_size);
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_spk_dim %d mismatches talker hidden %d", lat_spk_dim,
                   pt->talker.hidden_size);
            return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
        }
        ref_spk_emb_ptr = lat_spk_emb;
        qt_log(QT_LOG_INFO, "[Pipeline] Latent speaker embedding: %d values", lat_spk_dim);
    } else if (has_ref_audio) {
        if (!pt->has_speaker_encoder) {
            qt_set_error("pipeline_tts_synthesize: --ref-wav requires a model with a speaker encoder (Base only)");
            qt_log(QT_LOG_ERROR, "[Pipeline] --ref-wav requires a model with a speaker encoder (Base only)");
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        // Lazy residency: the first reference audio request pays the
        // weight load once, pre extracted paths never do.
        if (!pt->spk_enc_loaded) {
            Timer t_spk_load;
            if (!speaker_encoder_weights_load(&pt->speaker_encoder, pt->gguf_talker, pt->backend) ||
                pt->speaker_encoder.weight_buf == NULL) {
                pt->has_speaker_encoder = false;
                qt_set_error("pipeline_tts_synthesize: speaker encoder load failed");
                qt_log(QT_LOG_ERROR, "[Pipeline] speaker encoder load failed");
                return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
            }
            pt->spk_enc_loaded = true;
            qt_log(QT_LOG_INFO, "[Pipeline] Speaker encoder lazy loaded in %.0f ms", t_spk_load.ms());
        }
        if (!speaker_encoder_extract(&pt->speaker_encoder, pt->sched, params->ref_audio_24k, params->ref_n_samples,
                                     ref_spk_emb, params->dump_dir)) {
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        if ((int) ref_spk_emb.size() != pt->talker.hidden_size) {
            qt_set_error("pipeline_tts_synthesize: speaker embedding size %zu mismatches talker hidden %d",
                         ref_spk_emb.size(), pt->talker.hidden_size);
            qt_log(QT_LOG_ERROR, "[Pipeline] speaker embedding size %zu mismatches talker hidden %d",
                   ref_spk_emb.size(), pt->talker.hidden_size);
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        ref_spk_emb_ptr = ref_spk_emb.data();
    }

    // Slot construction: everything below fills the tail slot; a
    // failure pops it and reports through the job.
    e->slots.emplace_back();
    TtsSlot & s        = e->slots.back();
    const int slot_idx = (int) e->slots.size() - 1;
    s.job              = job;
    s.serial           = e->next_serial++;
    s.ref_codes_ptr    = NULL;
    s.ref_codes_T      = 0;
    s.step             = 0;
    s.subseq_counter   = 0;
    s.prev_overlay     = NULL;
    s.pending_c0       = -1;
    s.has_frame        = false;
    s.streaming        = (params->on_chunk != NULL);
    s.snap             = {};
    s.finished         = false;
    s.fin_status       = QT_STATUS_OK;
    s.perf             = {};
    s.t_total.reset();

    // Voice clone mode B: pre-encoded latent codes feed the ICL prompt
    // directly; otherwise, if ref_text is given, encode the reference
    // audio into 16 codebook indices via the codec encoder. Layout is
    // [num_codebooks, T_codec] row major in both cases, matching what
    // the prompt builder expects for the ICL sum loop.
    if (has_lat_codes) {
        s.ref_codes_ptr = lat_codes;
        s.ref_codes_T   = lat_T;
        qt_log(QT_LOG_INFO, "[Pipeline] Latent ICL ref_codes: %d frames at 12.5 Hz", s.ref_codes_T);
    } else if (!ref_text.empty()) {
        if (!has_ref_audio) {
            qt_set_error("pipeline_tts_synthesize: ref_text requires ref_audio_24k or latent ref_codes");
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_text requires ref_audio_24k or latent ref_codes");
            e->slots.pop_back();
            return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
        }
        // The codec hop is 1920 samples at 24 kHz so n_samples must be
        // a multiple of 1920. Truncate to the nearest hop boundary.
        if (params->ref_n_samples < TOKENIZER_HOP_LENGTH) {
            qt_set_error("pipeline_tts_synthesize: ref_wav too short for ICL (%d samples)", params->ref_n_samples);
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_wav too short for ICL (%d samples)", params->ref_n_samples);
            e->slots.pop_back();
            return tts_admit_fail(job, QT_STATUS_INVALID_PARAMS);
        }
        int aligned_T     = (params->ref_n_samples / TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH;
        s.ref_codes_store = pipeline_codec_encode(&pt->codec, params->ref_audio_24k, aligned_T, params->dump_dir);
        if (s.ref_codes_store.empty()) {
            qt_set_error("pipeline_tts_synthesize: pipeline_codec_encode returned empty codes");
            qt_log(QT_LOG_ERROR, "[Pipeline] pipeline_codec_encode returned empty codes");
            e->slots.pop_back();
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        s.ref_codes_ptr = s.ref_codes_store.data();
        s.ref_codes_T   = (int) s.ref_codes_store.size() / pt->num_code_groups;
        qt_log(QT_LOG_INFO, "[Pipeline] ICL ref_codes: %d frames at 12.5 Hz (%d audio samples)", s.ref_codes_T,
               aligned_T);
    }

    // NULL lang selects automatic language: the prompt carries no
    // language id and the model infers it from the text.
    const char * lang = params->lang ? params->lang : "auto";

    Timer t_build;
    if (!prompt_builder_build(pt, e->tok, params->text, lang, instruct, speaker, ref_spk_emb_ptr, ref_text,
                              s.ref_codes_ptr, s.ref_codes_T, &s.prompt)) {
        e->slots.pop_back();
        return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
    }
    s.perf.build_ms = t_build.ms();

    if (params->dump_dir) {
        DebugDumper d;
        debug_init(&d, params->dump_dir);
        std::vector<int32_t> ids32(s.prompt.prompt_ids.begin(), s.prompt.prompt_ids.end());
        int                  n_ids = (int) ids32.size();
        debug_dump_i32_as_f32(&d, "prompt-ids", ids32.data(), &n_ids, 1);
        debug_dump_2d(&d, "talker-input-embed", s.prompt.input_embed.data(), s.prompt.T_ctx, s.prompt.hidden);
        debug_dump_2d(&d, "trailing-text-hidden", s.prompt.trailing_text_hidden.data(), s.prompt.T_trailing,
                      s.prompt.hidden);
        debug_dump_1d(&d, "tts-pad-embed", s.prompt.tts_pad_embed.data(), s.prompt.hidden);

        // Voice clone dumps: spk-emb fires when ref_wav is set
        // (modes A and B), ref-codes fires only when ref_text is also set
        // (mode B ICL). Both are no-ops in base / tts / customvoice modes,
        // the dump files simply do not appear in those runs.
        if (ref_spk_emb_ptr != NULL) {
            debug_dump_1d(&d, "spk-emb", ref_spk_emb_ptr, pt->talker.hidden_size);
        }
        if (s.ref_codes_T > 0) {
            const int shape[2] = { pt->num_code_groups, s.ref_codes_T };
            debug_dump_i32_as_f32(&d, "ref-codes", s.ref_codes_ptr, shape, 2);
        }
    }

    s.talker_T = params->do_sample ? params->temperature : 0.0f;
    s.subtk_T  = params->subtalker_do_sample ? params->subtalker_temperature : 0.0f;
    s.prev_ids.assign((size_t) pt->num_code_groups, 0);
    s.all_codes.reserve((size_t) params->max_new_tokens);
    s.talker_history.reserve((size_t) params->max_new_tokens);

    // Stateful streaming decoder: every generated frame decodes
    // immediately through the persistent codec state and emits its
    // samples, so the first audio callback fires with the first frame.
    // ICL clone priming runs the full reference through the same state,
    // matching the upstream reference plus generated decode exactly.
    // Taking ownership parks the previous streaming slot's state first.
    if (s.streaming) {
        if (!tts_engine_codec_own(e, &s)) {
            qt_set_error("pipeline_tts_synthesize: codec stream state park failed");
            e->slots.pop_back();
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        if (!s.stream.init(&pt->codec, pt->num_code_groups)) {
            qt_set_error("pipeline_tts_synthesize: codec stream state init failed");
            e->stream_owner = -1;
            e->slots.pop_back();
            return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
        }
        if (s.ref_codes_ptr != NULL) {
            Timer t_seed;
            if (!s.stream.seed_reference(&pt->codec, s.ref_codes_ptr, s.ref_codes_T)) {
                qt_set_error("pipeline_tts_synthesize: codec stream reference priming failed");
                e->stream_owner = -1;
                e->slots.pop_back();
                return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
            }
            s.perf.codec_ms += t_seed.ms();
        }
    }

    // Talker prefill into KV set slot_idx: the joining request stalls
    // every already active slot for the duration of one prefill, so the
    // measured span is the batch's join cost.
    TalkerForwardOutput fw;
    Timer               t_prefill;
    if (!talker_forward_prefill(&pt->talker, &pt->talker_kv, slot_idx, pt->sched, &pt->talker_arena, pt->hidden_bridge,
                                s.prompt.input_embed.data(), s.prompt.T_ctx, pt->use_flash_attn, pt->clamp_fp16,
                                params->dump_dir, &fw)) {
        if (e->stream_owner == s.serial) {
            e->stream_owner = -1;
        }
        e->slots.pop_back();
        return tts_admit_fail(job, QT_STATUS_GENERATE_FAILED);
    }
    s.perf.prefill_ms = t_prefill.ms();
    s.logits          = std::move(fw.logits_last);
    qt_log(QT_LOG_INFO, "[Batch] Admit slot=%d T_ctx=%d prefill=%.1f ms build=%.1f ms (stall for %d active slots)",
           slot_idx, s.prompt.T_ctx, s.perf.prefill_ms, s.perf.build_ms, slot_idx);
    return true;
}

// Retire one finished slot: streaming drain or buffered codec decode,
// perf accounting, job status and worker side error capture. The codec
// stream mirror releases here.
static void tts_slot_complete(TtsEngine * e, TtsSlot & s) {
    PipelineTTS *                pt     = e->pt;
    TtsJob *                     job    = s.job;
    const struct qt_tts_params * params = job->params;
    qt_status                    st     = s.fin_status;

    if (st == QT_STATUS_OK) {
        qt_log(QT_LOG_INFO, "[Pipeline] Generation done : %zu frames", s.all_codes.size());
        s.perf.n_frames = (int) s.all_codes.size();

        const int num_codebooks = pt->num_code_groups;
        if (params->dump_dir && !s.all_codes.empty()) {
            DebugDumper d;
            debug_init(&d, params->dump_dir);
            int                  T_frames = (int) s.all_codes.size();
            std::vector<int32_t> flat((size_t) T_frames * (size_t) num_codebooks);
            for (int t = 0; t < T_frames; t++) {
                for (int k = 0; k < num_codebooks; k++) {
                    flat[(size_t) t * (size_t) num_codebooks + (size_t) k] = s.all_codes[(size_t) t][(size_t) k];
                }
            }
            int shape[2] = { T_frames, num_codebooks };
            debug_dump_i32_as_f32(&d, "codes-full", flat.data(), shape, 2);
        }

        if (s.streaming) {
            // Streaming tail: drain the sub chunk remainder of the
            // ramp, then finish with an empty buffered output.
            if (!tts_engine_codec_own(e, &s)) {
                qt_set_error("pipeline_tts_synthesize: codec stream state park failed");
                st = QT_STATUS_GENERATE_FAILED;
            } else if (!s.stream.drain(&pt->codec, params->on_chunk, params->on_chunk_user_data)) {
                if (s.stream.cancelled) {
                    qt_log(QT_LOG_INFO, "[Pipeline] on_chunk callback aborted the synthesis");
                    st = QT_STATUS_CANCELLED;
                } else {
                    qt_set_error("pipeline_tts_synthesize: streaming codec drain failed");
                    qt_log(QT_LOG_ERROR, "[Pipeline] streaming codec drain failed");
                    st = QT_STATUS_GENERATE_FAILED;
                }
            } else {
                if (job->out) {
                    job->out->samples     = NULL;
                    job->out->n_samples   = 0;
                    job->out->sample_rate = TOKENIZER_SAMPLE_RATE;
                    job->out->channels    = 1;
                }
                s.perf.total_ms = s.t_total.ms();
                tts_log_perf(s.perf);
            }
        } else if (s.all_codes.empty()) {
            // Buffered path: empty all_codes means EOS at step 0 with
            // no audio. Return success and an empty qt_audio struct;
            // the facade leaves it to the caller to decide what to do
            // with a zero sample synthesis.
            job->out->samples     = NULL;
            job->out->n_samples   = 0;
            job->out->sample_rate = TOKENIZER_SAMPLE_RATE;
            job->out->channels    = 1;
            s.perf.total_ms       = s.t_total.ms();
            tts_log_perf(s.perf);
        } else {
            // Buffered codec decode through the chunked path : same framing as
            // the streaming branch (chunk_frames + left_ctx_frames), bit perfect
            // equivalent to a single pipeline_codec_decode call when T_frames
            // fits in one chunk, bounded VRAM beyond that. Transpose codes from
            // [T_frames, K] to [K, T_frames] because codec_chunked_decode
            // expects K major layout. On the ICL clone path the tail of the
            // reference codes prepends the buffer so the onset is voiced with
            // the reference's causal state, mirroring the upstream pipeline
            // which decodes reference plus generated then trims; the seeded
            // samples strip from the front afterwards. Raising
            // codec_left_context_sec past the reference duration reproduces the
            // upstream full reference decode exactly.
            const float chunk_sec    = params->codec_chunk_sec > 0.0f ? params->codec_chunk_sec : 24.0f;
            const float left_ctx_sec = params->codec_left_context_sec >= 0.0f ? params->codec_left_context_sec : 2.0f;
            const int   chunk_frames = pipeline_tts_duration_sec_to_tokens(pt, chunk_sec);
            const int   left_ctx_frames = pipeline_tts_duration_sec_to_tokens(pt, left_ctx_sec);

            const int T_frames = (int) s.all_codes.size();
            int       seed     = 0;
            if (s.ref_codes_ptr != NULL) {
                seed = s.ref_codes_T < left_ctx_frames ? s.ref_codes_T : left_ctx_frames;
            }
            const int            T_dec = seed + T_frames;
            std::vector<int32_t> codes_kt((size_t) num_codebooks * (size_t) T_dec);
            for (int k = 0; k < num_codebooks; k++) {
                int32_t * row = codes_kt.data() + (size_t) k * (size_t) T_dec;
                if (seed > 0) {
                    std::memcpy(row,
                                s.ref_codes_ptr + (size_t) k * (size_t) s.ref_codes_T + (size_t) (s.ref_codes_T - seed),
                                (size_t) seed * sizeof(int32_t));
                }
                for (int t = 0; t < T_frames; t++) {
                    row[(size_t) (seed + t)] = s.all_codes[(size_t) t][(size_t) k];
                }
            }
            Timer              t_codec;
            std::vector<float> audio =
                codec_chunked_decode(&pt->codec, codes_kt.data(), num_codebooks, T_dec, chunk_frames, left_ctx_frames);
            s.perf.codec_ms += t_codec.ms();
            if (audio.empty()) {
                qt_set_error("pipeline_tts_synthesize: codec decode returned no audio");
                qt_log(QT_LOG_ERROR, "[Pipeline] codec decode returned no audio");
                st = QT_STATUS_GENERATE_FAILED;
            } else {
                if (seed > 0) {
                    audio.erase(audio.begin(), audio.begin() + (size_t) seed * (size_t) TOKENIZER_HOP_LENGTH);
                }
                if (params->dump_dir) {
                    DebugDumper d;
                    debug_init(&d, params->dump_dir);
                    debug_dump_1d(&d, "output-audio", audio.data(), (int) audio.size());
                }
                if (!fill_qt_audio(audio, job->out)) {
                    st = QT_STATUS_OOM;
                } else {
                    s.perf.total_ms = s.t_total.ms();
                    tts_log_perf(s.perf);
                }
            }
        }
    }

    if (e->stream_owner == s.serial) {
        e->stream_owner = -1;
    }
    pipeline_codec_snap_free(&s.snap);

    if (st != QT_STATUS_OK) {
        job->error = qt_last_error();
    }
    job->status = st;
}

void tts_engine_step(TtsEngine * e, std::vector<TtsJob *> * retired) {
    PipelineTTS * pt = e->pt;
    const int     N  = (int) e->slots.size();
    if (N == 0) {
        return;
    }
    const int hidden        = pt->talker.hidden_size;
    const int vocab         = pt->talker.vocab_size;
    const int num_codebooks = pt->num_code_groups;
    const int codec_eos_id  = pt->codec_specials.eos_id;
    const int n_acoustic    = pt->code_predictor.num_acoustic_codebooks;

    // 1) Batched talker decode over the slots past their prefill. The
    // freshly admitted slots form a contiguous tail (step == 0) and
    // consume their prefill logits instead; every retirement happens at
    // frame end when all survivors carry step >= 1, so the decode span
    // [0, N_dec) stays consecutive by construction.
    int N_dec = 0;
    while (N_dec < N && e->slots[(size_t) N_dec].step > 0) {
        N_dec++;
    }
    bool any_dump = false;
    for (int i = 0; i < N; i++) {
        any_dump = any_dump || (e->slots[(size_t) i].job->params->dump_dir != NULL);
    }
    if (N_dec > 0) {
        std::vector<int32_t> ids((size_t) num_codebooks * (size_t) N_dec);
        std::vector<float>   overlays((size_t) hidden * (size_t) N_dec);
        for (int i = 0; i < N_dec; i++) {
            TtsSlot & s = e->slots[(size_t) i];
            for (int g = 0; g < num_codebooks; g++) {
                ids[(size_t) g * (size_t) N_dec + (size_t) i] = s.prev_ids[(size_t) g];
            }
            std::memcpy(overlays.data() + (size_t) i * (size_t) hidden, s.prev_overlay,
                        (size_t) hidden * sizeof(float));
        }
        TalkerForwardOutput fw;
        Timer               t_talker;
        if (!talker_forward_decode(&pt->talker, &pt->talker_kv, pt->backend, pt->talker_decode_graphs,
                                   pt->hidden_bridge, ids.data(), pt->code_predictor.codec_embedding.data(), n_acoustic,
                                   overlays.data(), N_dec, pt->use_flash_attn, pt->clamp_fp16, any_dump, &fw)) {
            qt_set_error("pipeline_tts_synthesize: talker decode failed");
            for (TtsSlot & s : e->slots) {
                s.finished   = true;
                s.fin_status = QT_STATUS_GENERATE_FAILED;
            }
        } else {
            const double ms = t_talker.ms();
            for (int i = 0; i < N_dec; i++) {
                TtsSlot & s = e->slots[(size_t) i];
                s.perf.talker_ms += ms;
                s.logits.assign(fw.logits_last.begin() + (size_t) i * (size_t) vocab,
                                fw.logits_last.begin() + (size_t) (i + 1) * (size_t) vocab);

                // Bisection dump: the talker hidden_last at step 1 is
                // the input the code predictor consumes after consuming
                // the next-emb of step 0. Pairing it byte for byte with
                // the Python hook tells us whether the next-emb
                // composition + talker decode round trip is bit exact
                // end to end.
                if (s.job->params->dump_dir && s.step == 1) {
                    DebugDumper d;
                    debug_init(&d, s.job->params->dump_dir);
                    debug_dump_1d(&d, "talker-hidden-step1", fw.hidden_last.data() + (size_t) i * (size_t) hidden,
                                  hidden);
                }
            }
        }
    }

    // 2) Cancel poll and per slot c0 sampling: suppression, repetition
    // penalty over the slot's own history, its own Philox stream.
    for (int i = 0; i < N; i++) {
        TtsSlot & s = e->slots[(size_t) i];
        s.has_frame = false;
        if (s.finished) {
            continue;
        }
        const struct qt_tts_params * p = s.job->params;

        // Cooperative cancellation, polled at every step. Granularity is
        // one AR frame = 1 / 12.5 Hz ~ 83 ms of audio, which is well
        // below any reasonable UX cancel latency target.
        if (p->cancel && p->cancel(p->cancel_user_data)) {
            qt_log(QT_LOG_INFO, "[Pipeline] cancelled at step %d (slot %d)", s.step, i);
            s.finished   = true;
            s.fin_status = QT_STATUS_CANCELLED;
            continue;
        }

        // Apply codec suppression: forbid [vocab - 1024, vocab) except
        // codec_eos. Then run the upstream sampling chain.
        Timer t_host;
        apply_suppress(s.logits.data(), vocab, vocab - 1024, vocab, codec_eos_id);
        float u_c0 = 0.0f;
        int   c0   = sample_top_k_p(s.logits.data(), vocab, s.talker_T, p->top_k, p->top_p, p->repetition_penalty,
                                    s.talker_history.data(), (int) s.talker_history.size(), s.job->resolved_seed,
                                    s.subseq_counter, &u_c0);
        s.perf.host_ms += t_host.ms();
        s.subseq_counter++;
        if (c0 < 0) {
            qt_set_error("pipeline_tts_synthesize: c0 sample returned no candidate");
            qt_log(QT_LOG_ERROR, "[Pipeline] c0 sample returned no candidate");
            s.finished   = true;
            s.fin_status = QT_STATUS_GENERATE_FAILED;
            continue;
        }

        // Trace the first 32 samples unconditionally so [Sample] lines
        // up with [Sample-PY] / [Sample-CP] across the 16 codes of step
        // 0 and step 1 the Python harness emits.
        if ((s.subseq_counter - 1) < 32) {
            qt_log(QT_LOG_DEBUG, "[Sample] step=%d c0=%d u=%.10f subseq=%lld", s.step, c0, (double) u_c0,
                   (long long) (s.subseq_counter - 1));
        }

        if (c0 == codec_eos_id) {
            qt_log(QT_LOG_INFO, "[Pipeline] EOS at step %d, stopping (slot %d)", s.step, i);
            s.finished = true;
            continue;
        }
        s.pending_c0 = c0;
        s.has_frame  = true;
    }

    // 3) Batched code predictor over all N lanes in lockstep. Lanes
    // whose slot finished this frame ride along with a zero id and get
    // discarded; the live lanes each consume their own Philox stream so
    // per slot outputs stay identical to a single sequence run.
    bool any_live = false;
    for (int i = 0; i < N; i++) {
        any_live = any_live || e->slots[(size_t) i].has_frame;
    }
    if (any_live) {
        CodePredictorOutput cp;
        if (!pipeline_tts_cp_graphs_ensure(pt, N)) {
            qt_set_error("pipeline_tts_synthesize: code predictor graph build failed (N=%d)", N);
            for (TtsSlot & s : e->slots) {
                s.finished   = true;
                s.fin_status = QT_STATUS_GENERATE_FAILED;
                s.has_frame  = false;
            }
        } else {
            CodePredGraphSet &   gs = pt->cp_graphs[(size_t) (N - 1)];
            std::vector<int32_t> c0s((size_t) N, 0);
            std::vector<float>   temps((size_t) N, 0.0f);
            std::vector<int>     top_ks((size_t) N, 0);
            std::vector<float>   top_ps((size_t) N, 1.0f);
            std::vector<int64_t> seeds((size_t) N, 0);
            std::vector<int64_t> subseqs((size_t) N, 0);
            const char *         cp_dump = NULL;
            for (int i = 0; i < N; i++) {
                TtsSlot & s = e->slots[(size_t) i];
                if (!s.has_frame) {
                    continue;
                }
                const struct qt_tts_params * p = s.job->params;
                c0s[(size_t) i]                = s.pending_c0;
                temps[(size_t) i]              = s.subtk_T;
                top_ks[(size_t) i]             = p->subtalker_top_k;
                top_ps[(size_t) i]             = p->subtalker_top_p;
                seeds[(size_t) i]              = s.job->resolved_seed;
                subseqs[(size_t) i]            = s.subseq_counter - 1;
                if (N == 1 && s.step == 0 && p->dump_dir) {
                    cp_dump = p->dump_dir;
                }
            }
            Timer t_pred;
            if (!code_predictor_step(&pt->code_predictor, pt->backend, &gs.prefill, gs.steps.data(), c0s.data(), N,
                                     temps.data(), top_ks.data(), top_ps.data(), seeds.data(), subseqs.data(), cp_dump,
                                     &cp)) {
                for (TtsSlot & s : e->slots) {
                    s.finished   = true;
                    s.fin_status = QT_STATUS_GENERATE_FAILED;
                    s.has_frame  = false;
                }
            } else {
                const double ms = t_pred.ms();

                // 4) Per slot frame post: history, codec streaming,
                // next decode inputs.
                for (int i = 0; i < N; i++) {
                    TtsSlot & s = e->slots[(size_t) i];
                    if (!s.has_frame) {
                        continue;
                    }
                    const struct qt_tts_params * p = s.job->params;
                    s.perf.predictor_ms += ms;
                    if (s.step == 0) {
                        s.perf.ttfa_ms = s.t_total.ms();
                    }
                    // Predictor consumed (num_codebooks - 1) subsequences
                    // after the c0 one (subseq_base + 1 .. subseq_base + 15).
                    s.subseq_counter += (num_codebooks - 1);

                    std::vector<int32_t> codes(cp.codes.begin() + (size_t) i * (size_t) num_codebooks,
                                               cp.codes.begin() + (size_t) (i + 1) * (size_t) num_codebooks);
                    s.all_codes.push_back(codes);
                    s.talker_history.push_back(s.pending_c0);

                    if (s.streaming) {
                        Timer t_codec;
                        bool  pushed = tts_engine_codec_own(e, &s) &&
                                      s.stream.push_frame(&pt->codec, codes.data(), p->on_chunk, p->on_chunk_user_data);
                        s.perf.codec_ms += t_codec.ms();
                        if (!pushed) {
                            if (s.stream.cancelled) {
                                qt_log(QT_LOG_INFO, "[Pipeline] on_chunk callback aborted the synthesis (slot %d)", i);
                                s.finished   = true;
                                s.fin_status = QT_STATUS_CANCELLED;
                            } else {
                                qt_set_error("pipeline_tts_synthesize: streaming codec decode failed at frame %d",
                                             s.step);
                                qt_log(QT_LOG_ERROR, "[Pipeline] streaming codec decode failed at frame %d (slot %d)",
                                       s.step, i);
                                s.finished   = true;
                                s.fin_status = QT_STATUS_GENERATE_FAILED;
                            }
                            continue;
                        }
                    }

                    // Next decode input: the 16 frame codes gather and sum
                    // in graph (codebook 0 from talker.codec_embedding, the
                    // 15 acoustic groups from the predictor's private
                    // tables). The overlay row adds the next utterance text
                    // hidden while any remains, the tts_pad embedding
                    // afterwards.
                    for (int g = 0; g < num_codebooks; g++) {
                        s.prev_ids[(size_t) g] = codes[(size_t) g];
                    }
                    s.prev_overlay = (s.step < s.prompt.T_trailing) ?
                                         s.prompt.trailing_text_hidden.data() + (size_t) s.step * (size_t) hidden :
                                         s.prompt.tts_pad_embed.data();

                    // Bisection dump: reproduce the in graph composition on
                    // host so the step 0 next embedding stays byte
                    // comparable against the Python hook (codebook sums plus
                    // trailing text overlay).
                    if (p->dump_dir && s.step == 0) {
                        std::vector<float> next_emb((size_t) hidden, 0.0f);
                        std::vector<float> tmp((size_t) hidden);
                        embed_row_from_gguf(pt->gguf_talker, "talker.codec_embd.weight", s.pending_c0, hidden,
                                            tmp.data());
                        for (int j = 0; j < hidden; j++) {
                            next_emb[(size_t) j] += tmp[(size_t) j];
                        }
                        for (int g = 0; g < num_codebooks - 1; g++) {
                            int  cg = codes[(size_t) (g + 1)];
                            char name[64];
                            snprintf(name, sizeof(name), "code_pred.codec_embd.%d.weight", g);
                            embed_row_from_gguf(pt->gguf_talker, name, cg, hidden, tmp.data());
                            for (int j = 0; j < hidden; j++) {
                                next_emb[(size_t) j] += tmp[(size_t) j];
                            }
                        }
                        for (int j = 0; j < hidden; j++) {
                            next_emb[(size_t) j] += s.prev_overlay[(size_t) j];
                        }
                        DebugDumper d;
                        debug_init(&d, p->dump_dir);
                        debug_dump_1d(&d, "next-emb-step0", next_emb.data(), hidden);
                    }

                    s.step++;
                    if ((s.step % 8) == 0) {
                        qt_log(QT_LOG_INFO, "[Pipeline] Generated %d frames (slot %d)", s.step, i);
                    }
                    if (s.step >= p->max_new_tokens) {
                        s.finished = true;
                    }
                }
            }
        }
    }

    // Fresh slots that got no frame this step (their very first frame
    // ended in EOS or cancel) still advanced past prefill conceptually;
    // slots that emitted advanced in the loop above. Slots neither
    // finished nor advanced cannot exist: every live slot either emits
    // or finishes.

    // 5) Retirement: swap-remove keeps the active range consecutive.
    // The tail slot's talker KV set copies device side into the freed
    // index; the bridge column and the predictor set rewrite next frame
    // before any read, so only the talker cache moves.
    for (int i = 0; i < (int) e->slots.size();) {
        if (!e->slots[(size_t) i].finished) {
            i++;
            continue;
        }
        tts_slot_complete(e, e->slots[(size_t) i]);
        if (retired) {
            retired->push_back(e->slots[(size_t) i].job);
        }
        const int last = (int) e->slots.size() - 1;
        if (i != last) {
            kv_cache_copy_set(&pt->talker_kv, last, i);
            e->slots[(size_t) i] = std::move(e->slots[(size_t) last]);
        }
        e->slots.pop_back();
    }
}

qt_status pipeline_tts_synthesize(PipelineTTS *                pt,
                                  BPETokenizer *               tok,
                                  const struct qt_tts_params * params,
                                  int64_t                      resolved_seed,
                                  struct qt_audio *            out) {
    TtsEngine * e = tts_engine_new(pt, tok);
    TtsJob      job;
    job.params        = params;
    job.resolved_seed = resolved_seed;
    job.out           = out;
    job.status        = QT_STATUS_OK;
    job.done          = false;
    if (tts_engine_admit(e, &job)) {
        while (tts_engine_active(e) > 0) {
            tts_engine_step(e, NULL);
        }
    }
    tts_engine_free(e);
    return job.status;
}
