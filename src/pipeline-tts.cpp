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

bool pipeline_tts_load(PipelineTTS * pt,
                       const char *  talker_gguf_path,
                       const char *  codec_gguf_path,
                       BackendPair   bp,
                       bool          use_fa,
                       bool          clamp_fp16) {
    pt->bp                  = bp;
    pt->backend             = bp.backend;
    pt->sched               = NULL;
    pt->has_speaker_encoder = false;
    pt->bridge_ctx          = NULL;
    pt->bridge_buf          = NULL;
    pt->hidden_bridge       = NULL;

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

    // KV caches: talker holds the LM context up to 4096 positions (the
    // longest ICL prompt observed is ~250 + max_new_tokens ~ 1500, so
    // 4096 has 60% headroom). Predictor holds one frame of 16 sub-steps.
    if (!kv_cache_init(&pt->talker_kv, pt->talker.num_hidden_layers, pt->talker.num_key_value_heads,
                       pt->talker.head_dim, 4096, pt->backend)) {
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
                       pt->backend)) {
        kv_cache_free(&pt->talker_kv);
        ggml_backend_sched_free(pt->sched);
        pt->sched = NULL;
        pipeline_codec_free(&pt->codec);
        code_predictor_weights_free(&pt->code_predictor);
        talker_weights_free(&pt->talker);
        gf_close(&pt->gguf_talker);
        return false;
    }

    // Hidden bridge: one [talker_hidden] f32 tensor resident on the
    // backend, written by the talker graph and read by the code
    // predictor prefill graph. Cleared once so debug dumps never see
    // stale bytes before the first talker forward.
    {
        struct ggml_init_params gp = { ggml_tensor_overhead() * 2, NULL, true };
        pt->bridge_ctx             = ggml_init(gp);
        pt->hidden_bridge =
            pt->bridge_ctx ? ggml_new_tensor_1d(pt->bridge_ctx, GGML_TYPE_F32, pt->talker.hidden_size) : NULL;
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

    // Talker graph arena plus the static predictor graphs: the talker
    // keeps one arena per shape class for the CUDA graph cache, the
    // predictor builds one static graph per flavor (prefill + one per
    // acoustic step), each with its lm_head and embedding table fixed
    // and the positions, kv rows, and mask baked in.
    bool graphs_ok =
        graph_arena_init(&pt->talker_arena, talker_graph_max_nodes(pt->talker.num_hidden_layers)) &&
        code_predictor_graph_build(&pt->code_predictor, &pt->code_predictor_kv, pt->backend, pt->talker.codec_embedding,
                                   pt->hidden_bridge, 0, pt->use_flash_attn, pt->clamp_fp16, &pt->cp_prefill_graph);
    pt->talker_decode_graphs.resize(((size_t) pt->talker_kv.max_seq_len + 255) / 256);
    pt->cp_step_graphs.resize((size_t) (pt->num_code_groups - 2));
    for (size_t g = 0; graphs_ok && g < pt->cp_step_graphs.size(); g++) {
        graphs_ok = code_predictor_graph_build(&pt->code_predictor, &pt->code_predictor_kv, pt->backend,
                                               pt->code_predictor.codec_embedding[g], NULL, (int) g + 1,
                                               pt->use_flash_attn, pt->clamp_fp16, &pt->cp_step_graphs[g]);
    }
    if (!graphs_ok) {
        for (size_t g = 0; g < pt->cp_step_graphs.size(); g++) {
            code_predictor_graph_free(&pt->cp_step_graphs[g]);
        }
        code_predictor_graph_free(&pt->cp_prefill_graph);
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
           "clamp_fp16=%s",
           pt->model_size.c_str(), pt->model_type.c_str(), pt->tokenizer_type.c_str(), pt->num_code_groups,
           pt->has_speaker_encoder ? "deferred" : "absent", pt->speakers.size(), pt->use_flash_attn ? "on" : "off",
           pt->clamp_fp16 ? "on" : "off");
    return true;
}

void pipeline_tts_free(PipelineTTS * pt) {
    for (size_t g = 0; g < pt->cp_step_graphs.size(); g++) {
        code_predictor_graph_free(&pt->cp_step_graphs[g]);
    }
    pt->cp_step_graphs.clear();
    code_predictor_graph_free(&pt->cp_prefill_graph);
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

qt_status pipeline_tts_synthesize(PipelineTTS *                pt,
                                  BPETokenizer *               tok,
                                  const struct qt_tts_params * params,
                                  int64_t                      resolved_seed,
                                  struct qt_audio *            out) {
    PromptBuilderOutput prompt;
    const std::string   instruct = params->instruct ? params->instruct : "";
    const std::string   speaker  = params->speaker ? params->speaker : "";
    const std::string   ref_text = params->ref_text ? params->ref_text : "";

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
        return QT_STATUS_INVALID_PARAMS;
    }
    // Latent ICL codes ride on top of the speaker embedding and need the
    // transcript, mirroring the raw path where mode B implies mode A.
    if (has_lat_codes && (!has_lat_spk || ref_text.empty())) {
        qt_set_error("pipeline_tts_synthesize: ref_codes requires ref_spk_emb and ref_text");
        qt_log(QT_LOG_ERROR, "[Pipeline] ref_codes requires ref_spk_emb and ref_text");
        return QT_STATUS_INVALID_PARAMS;
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
            return QT_STATUS_INVALID_PARAMS;
        }
        ref_spk_emb_ptr = lat_spk_emb;
        qt_log(QT_LOG_INFO, "[Pipeline] Latent speaker embedding: %d values", lat_spk_dim);
    } else if (has_ref_audio) {
        if (!pt->has_speaker_encoder) {
            qt_set_error("pipeline_tts_synthesize: --ref-wav requires a model with a speaker encoder (Base only)");
            qt_log(QT_LOG_ERROR, "[Pipeline] --ref-wav requires a model with a speaker encoder (Base only)");
            return QT_STATUS_GENERATE_FAILED;
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
                return QT_STATUS_GENERATE_FAILED;
            }
            pt->spk_enc_loaded = true;
            qt_log(QT_LOG_INFO, "[Pipeline] Speaker encoder lazy loaded in %.0f ms", t_spk_load.ms());
        }
        if (!speaker_encoder_extract(&pt->speaker_encoder, pt->sched, params->ref_audio_24k, params->ref_n_samples,
                                     ref_spk_emb, params->dump_dir)) {
            return QT_STATUS_GENERATE_FAILED;
        }
        if ((int) ref_spk_emb.size() != pt->talker.hidden_size) {
            qt_set_error("pipeline_tts_synthesize: speaker embedding size %zu mismatches talker hidden %d",
                         ref_spk_emb.size(), pt->talker.hidden_size);
            qt_log(QT_LOG_ERROR, "[Pipeline] speaker embedding size %zu mismatches talker hidden %d",
                   ref_spk_emb.size(), pt->talker.hidden_size);
            return QT_STATUS_GENERATE_FAILED;
        }
        ref_spk_emb_ptr = ref_spk_emb.data();
    }

    // Voice clone mode B: pre-encoded latent codes feed the ICL prompt
    // directly; otherwise, if ref_text is given, encode the reference
    // audio into 16 codebook indices via the codec encoder. Layout is
    // [num_codebooks, T_codec] row major in both cases, matching what
    // the prompt builder expects for the ICL sum loop.
    std::vector<int32_t> ref_codes;
    const int32_t *      ref_codes_ptr = NULL;
    int                  ref_codes_T   = 0;
    if (has_lat_codes) {
        ref_codes_ptr = lat_codes;
        ref_codes_T   = lat_T;
        qt_log(QT_LOG_INFO, "[Pipeline] Latent ICL ref_codes: %d frames at 12.5 Hz", ref_codes_T);
    } else if (!ref_text.empty()) {
        if (!has_ref_audio) {
            qt_set_error("pipeline_tts_synthesize: ref_text requires ref_audio_24k or latent ref_codes");
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_text requires ref_audio_24k or latent ref_codes");
            return QT_STATUS_INVALID_PARAMS;
        }
        // The codec hop is 1920 samples at 24 kHz so n_samples must be
        // a multiple of 1920. Truncate to the nearest hop boundary.
        if (params->ref_n_samples < TOKENIZER_HOP_LENGTH) {
            qt_set_error("pipeline_tts_synthesize: ref_wav too short for ICL (%d samples)", params->ref_n_samples);
            qt_log(QT_LOG_ERROR, "[Pipeline] ref_wav too short for ICL (%d samples)", params->ref_n_samples);
            return QT_STATUS_INVALID_PARAMS;
        }
        int aligned_T = (params->ref_n_samples / TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH;
        ref_codes     = pipeline_codec_encode(&pt->codec, params->ref_audio_24k, aligned_T, params->dump_dir);
        if (ref_codes.empty()) {
            qt_set_error("pipeline_tts_synthesize: pipeline_codec_encode returned empty codes");
            qt_log(QT_LOG_ERROR, "[Pipeline] pipeline_codec_encode returned empty codes");
            return QT_STATUS_GENERATE_FAILED;
        }
        ref_codes_ptr = ref_codes.data();
        ref_codes_T   = (int) ref_codes.size() / pt->num_code_groups;
        qt_log(QT_LOG_INFO, "[Pipeline] ICL ref_codes: %d frames at 12.5 Hz (%d audio samples)", ref_codes_T,
               aligned_T);
    }

    TtsPerf perf = {};
    Timer   t_total;

    // NULL lang selects automatic language: the prompt carries no
    // language id and the model infers it from the text.
    const char * lang = params->lang ? params->lang : "auto";

    Timer t_build;
    if (!prompt_builder_build(pt, tok, params->text, lang, instruct, speaker, ref_spk_emb_ptr, ref_text, ref_codes_ptr,
                              ref_codes_T, &prompt)) {
        return QT_STATUS_GENERATE_FAILED;
    }
    perf.build_ms = t_build.ms();

    if (params->dump_dir) {
        DebugDumper d;
        debug_init(&d, params->dump_dir);
        std::vector<int32_t> ids32(prompt.prompt_ids.begin(), prompt.prompt_ids.end());
        int                  n_ids = (int) ids32.size();
        debug_dump_i32_as_f32(&d, "prompt-ids", ids32.data(), &n_ids, 1);
        debug_dump_2d(&d, "talker-input-embed", prompt.input_embed.data(), prompt.T_ctx, prompt.hidden);
        debug_dump_2d(&d, "trailing-text-hidden", prompt.trailing_text_hidden.data(), prompt.T_trailing, prompt.hidden);
        debug_dump_1d(&d, "tts-pad-embed", prompt.tts_pad_embed.data(), prompt.hidden);

        // Voice clone dumps: spk-emb fires when ref_wav is set
        // (modes A and B), ref-codes fires only when ref_text is also set
        // (mode B ICL). Both are no-ops in base / tts / customvoice modes,
        // the dump files simply do not appear in those runs.
        if (ref_spk_emb_ptr != NULL) {
            debug_dump_1d(&d, "spk-emb", ref_spk_emb_ptr, pt->talker.hidden_size);
        }
        if (ref_codes_T > 0) {
            const int shape[2] = { pt->num_code_groups, ref_codes_T };
            debug_dump_i32_as_f32(&d, "ref-codes", ref_codes_ptr, shape, 2);
        }
    }

    // Generation loop: step 0 prefills the talker over the full prompt
    // and writes T_ctx positions into the KV cache. Subsequent steps
    // feed one next_emb at a time and append one position. The code
    // predictor maintains its own per-frame cache that gets reset at
    // every step.
    const int   hidden        = prompt.hidden;
    const int   codec_eos_id  = pt->codec_specials.eos_id;
    const int   num_codebooks = pt->num_code_groups;
    const int   talker_vocab  = pt->talker.vocab_size;
    const bool  use_fa        = pt->use_flash_attn;
    const bool  clamp_fp16    = pt->clamp_fp16;
    const float talker_T      = params->do_sample ? params->temperature : 0.0f;
    const float subtk_T       = params->subtalker_do_sample ? params->subtalker_temperature : 0.0f;
    const float talker_rp     = params->repetition_penalty;

    // Codec decode framing. The buffered path routes through
    // codec_chunked_decode with a rolling left context window mirroring
    // the upstream Qwen3-TTS 12 Hz tokenizer chunked_decode rule: every
    // chunk re uses up to left_ctx_frames previously decoded frames as
    // left context, then the matching audio samples are stripped from
    // the head of the decoded chunk. The streaming path decodes frame
    // by frame through the stateful codec and ignores both knobs.
    const bool  streaming       = (params->on_chunk != NULL);
    const float chunk_sec       = params->codec_chunk_sec > 0.0f ? params->codec_chunk_sec : 24.0f;
    const float left_ctx_sec    = params->codec_left_context_sec >= 0.0f ? params->codec_left_context_sec : 2.0f;
    const int   chunk_frames    = pipeline_tts_duration_sec_to_tokens(pt, chunk_sec);
    const int   left_ctx_frames = pipeline_tts_duration_sec_to_tokens(pt, left_ctx_sec);

    std::vector<std::vector<int32_t>> all_codes;
    all_codes.reserve((size_t) params->max_new_tokens);

    // c0 codes already emitted, fed to repetition penalty.
    std::vector<int32_t> talker_history;
    talker_history.reserve((size_t) params->max_new_tokens);

    // Global Philox subsequence counter advances once per primitive
    // sample (one for c0 of each step, then 15 for the predictor codes).
    int64_t subseq_counter = 0;

    // Decode input state: the codes sampled at the previous frame plus
    // the trailing text / pad overlay row for that frame. The talker
    // decode graph gathers and sums the 16 embeddings on device.
    std::vector<int32_t> prev_ids((size_t) num_codebooks, 0);
    const float *        prev_overlay = NULL;

    // Stateful streaming decoder: every generated frame decodes
    // immediately through the persistent codec state and emits its
    // samples, so the first audio callback fires with the first frame.
    // ICL clone priming runs the full reference through the same state,
    // matching the upstream reference plus generated decode exactly.
    codec_stream_decoder stream;
    if (streaming) {
        if (!stream.init(&pt->codec, num_codebooks)) {
            qt_set_error("pipeline_tts_synthesize: codec stream state init failed");
            return QT_STATUS_GENERATE_FAILED;
        }
        if (ref_codes_ptr != NULL) {
            Timer t_seed;
            if (!stream.seed_reference(&pt->codec, ref_codes_ptr, ref_codes_T)) {
                qt_set_error("pipeline_tts_synthesize: codec stream reference priming failed");
                return QT_STATUS_GENERATE_FAILED;
            }
            perf.codec_ms += t_seed.ms();
        }
    }

    for (int step = 0; step < params->max_new_tokens; step++) {
        // Cooperative cancellation, polled at every step. Granularity is
        // one AR frame = 1 / 12.5 Hz ~ 83 ms of audio, which is well
        // below any reasonable UX cancel latency target.
        if (params->cancel && params->cancel(params->cancel_user_data)) {
            qt_log(QT_LOG_INFO, "[Pipeline] cancelled at step %d", step);
            return QT_STATUS_CANCELLED;
        }

        TalkerForwardOutput fw;
        const char *        step_dump = (params->dump_dir && step == 0) ? params->dump_dir : NULL;
        bool                ok;
        Timer               t_talker;
        if (step == 0) {
            ok = talker_forward_prefill(&pt->talker, &pt->talker_kv, pt->sched, &pt->talker_arena, pt->hidden_bridge,
                                        prompt.input_embed.data(), prompt.T_ctx, use_fa, clamp_fp16, step_dump, &fw);
        } else {
            ok = talker_forward_decode(&pt->talker, &pt->talker_kv, pt->backend, pt->talker_decode_graphs.data(),
                                       pt->hidden_bridge, prev_ids.data(), pt->code_predictor.codec_embedding.data(),
                                       pt->code_predictor.num_acoustic_codebooks, prev_overlay, use_fa, clamp_fp16,
                                       params->dump_dir != NULL, &fw);
        }
        if (!ok) {
            return QT_STATUS_GENERATE_FAILED;
        }
        if (step == 0) {
            perf.prefill_ms = t_talker.ms();
        } else {
            perf.talker_ms += t_talker.ms();
        }

        // Bisection dump: the talker hidden_last at step 1 is the input
        // the code predictor consumes after consuming the next-emb of
        // step 0. Pairing it byte for byte with the Python hook tells us
        // whether the next-emb composition + talker decode round trip
        // is bit exact end to end.
        if (params->dump_dir && step == 1) {
            DebugDumper d;
            debug_init(&d, params->dump_dir);
            debug_dump_1d(&d, "talker-hidden-step1", fw.hidden_last.data(), hidden);
        }

        // Apply codec suppression: forbid [vocab - 1024, vocab) except
        // codec_eos. Then run the upstream sampling chain.
        Timer t_host;
        apply_suppress(fw.logits_last.data(), talker_vocab, talker_vocab - 1024, talker_vocab, codec_eos_id);
        float u_c0 = 0.0f;
        int   c0 =
            sample_top_k_p(fw.logits_last.data(), talker_vocab, talker_T, params->top_k, params->top_p, talker_rp,
                           talker_history.data(), (int) talker_history.size(), resolved_seed, subseq_counter, &u_c0);
        perf.host_ms += t_host.ms();
        subseq_counter++;
        if (c0 < 0) {
            qt_log(QT_LOG_ERROR, "[Pipeline] c0 sample returned no candidate");
            return QT_STATUS_GENERATE_FAILED;
        }

        // Trace the first 32 samples unconditionally so [Sample] lines
        // up with [Sample-PY] / [Sample-CP] across the 16 codes of step
        // 0 and step 1 the Python harness emits.
        if ((subseq_counter - 1) < 32) {
            qt_log(QT_LOG_DEBUG, "[Sample] step=%d c0=%d u=%.10f subseq=%lld", step, c0, (double) u_c0,
                   (long long) (subseq_counter - 1));
        }

        if (c0 == codec_eos_id) {
            qt_log(QT_LOG_INFO, "[Pipeline] EOS at step %d, stopping", step);
            break;
        }

        CodePredictorOutput cp;
        const char *        cp_dump = (params->dump_dir && step == 0) ? params->dump_dir : NULL;
        Timer               t_pred;
        if (!code_predictor_step(&pt->code_predictor, pt->backend, &pt->cp_prefill_graph, pt->cp_step_graphs.data(), c0,
                                 subtk_T, params->subtalker_top_k, params->subtalker_top_p, resolved_seed,
                                 subseq_counter - 1, cp_dump, &cp)) {
            return QT_STATUS_GENERATE_FAILED;
        }
        perf.predictor_ms += t_pred.ms();
        if (step == 0) {
            perf.ttfa_ms = t_total.ms();
        }
        // Predictor consumed (num_codebooks - 1) subsequences after the
        // c0 one (subseq_base + 1 .. subseq_base + 15).
        subseq_counter += (num_codebooks - 1);

        all_codes.push_back(cp.codes);
        talker_history.push_back(c0);
        if (streaming) {
            Timer t_codec;
            bool  pushed = stream.push_frame(&pt->codec, cp.codes.data(), params->on_chunk, params->on_chunk_user_data);
            perf.codec_ms += t_codec.ms();
            if (!pushed) {
                if (stream.cancelled) {
                    qt_log(QT_LOG_INFO, "[Pipeline] on_chunk callback aborted the synthesis");
                    return QT_STATUS_CANCELLED;
                }
                qt_set_error("pipeline_tts_synthesize: streaming codec decode failed at frame %d", step);
                qt_log(QT_LOG_ERROR, "[Pipeline] streaming codec decode failed at frame %d", step);
                return QT_STATUS_GENERATE_FAILED;
            }
        }

        // Next decode input: the 16 frame codes gather and sum in graph
        // (codebook 0 from talker.codec_embedding, the 15 acoustic
        // groups from the predictor's private tables). The overlay row
        // adds the next utterance text hidden while any remains, the
        // tts_pad embedding afterwards.
        prev_ids[0] = c0;
        for (int g = 1; g < num_codebooks; g++) {
            prev_ids[(size_t) g] = cp.codes[(size_t) g];
        }
        prev_overlay = (step < prompt.T_trailing) ?
                           prompt.trailing_text_hidden.data() + (size_t) step * (size_t) hidden :
                           prompt.tts_pad_embed.data();

        // Bisection dump: reproduce the in graph composition on host so
        // the step 0 next embedding stays byte comparable against the
        // Python hook (codebook sums plus trailing text overlay).
        if (params->dump_dir && step == 0) {
            std::vector<float> next_emb((size_t) hidden, 0.0f);
            std::vector<float> tmp((size_t) hidden);
            embed_row_from_gguf(pt->gguf_talker, "talker.codec_embd.weight", c0, hidden, tmp.data());
            for (int i = 0; i < hidden; i++) {
                next_emb[(size_t) i] += tmp[(size_t) i];
            }
            for (int g = 0; g < num_codebooks - 1; g++) {
                int  cg = cp.codes[(size_t) (g + 1)];
                char name[64];
                snprintf(name, sizeof(name), "code_pred.codec_embd.%d.weight", g);
                embed_row_from_gguf(pt->gguf_talker, name, cg, hidden, tmp.data());
                for (int i = 0; i < hidden; i++) {
                    next_emb[(size_t) i] += tmp[(size_t) i];
                }
            }
            for (int i = 0; i < hidden; i++) {
                next_emb[(size_t) i] += prev_overlay[(size_t) i];
            }
            DebugDumper d;
            debug_init(&d, params->dump_dir);
            debug_dump_1d(&d, "next-emb-step0", next_emb.data(), hidden);
        }

        if (((step + 1) % 8) == 0) {
            qt_log(QT_LOG_INFO, "[Pipeline] Generated %d frames", step + 1);
        }
    }

    qt_log(QT_LOG_INFO, "[Pipeline] Generation done : %zu frames", all_codes.size());
    perf.n_frames = (int) all_codes.size();

    if (params->dump_dir && !all_codes.empty()) {
        DebugDumper d;
        debug_init(&d, params->dump_dir);
        int                  T_frames = (int) all_codes.size();
        std::vector<int32_t> flat((size_t) T_frames * (size_t) num_codebooks);
        for (int t = 0; t < T_frames; t++) {
            for (int k = 0; k < num_codebooks; k++) {
                flat[(size_t) t * (size_t) num_codebooks + (size_t) k] = all_codes[(size_t) t][(size_t) k];
            }
        }
        int shape[2] = { T_frames, num_codebooks };
        debug_dump_i32_as_f32(&d, "codes-full", flat.data(), shape, 2);
    }

    // Streaming tail: nothing to drain, every frame already emitted at
    // generation time through the stateful decoder. The buffered output
    // stays empty in this branch.
    if (streaming) {
        out->samples     = NULL;
        out->n_samples   = 0;
        out->sample_rate = TOKENIZER_SAMPLE_RATE;
        out->channels    = 1;
        perf.total_ms    = t_total.ms();
        tts_log_perf(perf);
        return QT_STATUS_OK;
    }

    // Buffered path: empty all_codes means EOS at step 0 with no audio.
    // Return success and an empty qt_audio struct; the facade leaves it
    // to the caller to decide what to do with a zero sample synthesis.
    if (all_codes.empty()) {
        out->samples     = NULL;
        out->n_samples   = 0;
        out->sample_rate = TOKENIZER_SAMPLE_RATE;
        out->channels    = 1;
        perf.total_ms    = t_total.ms();
        tts_log_perf(perf);
        return QT_STATUS_OK;
    }

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
    const int T_frames = (int) all_codes.size();
    int       seed     = 0;
    if (ref_codes_ptr != NULL) {
        seed = ref_codes_T < left_ctx_frames ? ref_codes_T : left_ctx_frames;
    }
    const int            T_dec = seed + T_frames;
    std::vector<int32_t> codes_kt((size_t) num_codebooks * (size_t) T_dec);
    for (int k = 0; k < num_codebooks; k++) {
        int32_t * row = codes_kt.data() + (size_t) k * (size_t) T_dec;
        if (seed > 0) {
            std::memcpy(row, ref_codes_ptr + (size_t) k * (size_t) ref_codes_T + (size_t) (ref_codes_T - seed),
                        (size_t) seed * sizeof(int32_t));
        }
        for (int t = 0; t < T_frames; t++) {
            row[(size_t) (seed + t)] = all_codes[(size_t) t][(size_t) k];
        }
    }
    Timer              t_codec;
    std::vector<float> audio =
        codec_chunked_decode(&pt->codec, codes_kt.data(), num_codebooks, T_dec, chunk_frames, left_ctx_frames);
    perf.codec_ms += t_codec.ms();
    if (audio.empty()) {
        qt_set_error("pipeline_tts_synthesize: codec decode returned no audio");
        qt_log(QT_LOG_ERROR, "[Pipeline] codec decode returned no audio");
        return QT_STATUS_GENERATE_FAILED;
    }
    if (seed > 0) {
        audio.erase(audio.begin(), audio.begin() + (size_t) seed * (size_t) TOKENIZER_HOP_LENGTH);
    }

    if (params->dump_dir) {
        DebugDumper d;
        debug_init(&d, params->dump_dir);
        debug_dump_1d(&d, "output-audio", audio.data(), (int) audio.size());
    }

    if (!fill_qt_audio(audio, out)) {
        return QT_STATUS_OOM;
    }
    perf.total_ms = t_total.ms();
    tts_log_perf(perf);
    return QT_STATUS_OK;
}
