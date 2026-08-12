#pragma once
// prompt-builder.h: assemble the talker prefix input embedding from
// a tokenized text plus a language tag, mirroring the upstream
// generate() function of Qwen3-TTS.
//
// Output shape: [T_ctx, hidden_size] f32 row-major. Two pad-aligned
// streams (text and codec) are summed at the granularity of single
// vectors. The trailing text hidden buffer is also produced for the
// streaming-text overlay used during generation.
//
// Modes:
//   base          text only, no instruct, no speaker
//   voice_design  text + instruct (style description), no speaker
//   custom_voice  text + speaker, optional instruct
//
// Empty / NULL strings disable the corresponding stream. Text projection
// and reference codebook embeddings run on the loaded backend tensors
// through the pipeline scheduler. The special embeds (tts_bos, tts_eos,
// tts_pad, codec_pad, codec_bos) are projected once at load into the
// prompt cache.
//
// Two streams are aligned then summed:
//   text  stream:  text_projection(text_embedding(ids))   151936 -> 2048 -> 1024
//   codec stream:  codec_embedding(ids)                   3072 -> 1024
//
// Layout (lang_id != none, no speaker, no instruct):
//
//   role          text(input_id[0:3])                                3 vecs
//   prefill_lhs   tts_pad x4 + tts_bos                              5 vecs
//                 + codec_emb([think, think_bos, lang_id, think_eos, codec_pad])
//   trailing_lhs  text(input_id[3:-5]) + tts_eos                    N_text + 1 vecs
//                 + codec_emb([codec_pad x (N_text + 1)])
//   trailing_rhs  tts_pad + codec_emb([codec_bos])                  1 vec
//
// CustomVoice inserts the speaker codec embedding row between think_eos
// and codec_pad in the prefill, growing the prefill by one vector and
// substituting one tts_pad with another in the text stream alignment.
//
// VoiceDesign / CustomVoice may also prepend an instruct segment built
// from text_projection(text_embedding(<|im_start|>user\n{instruct}<|im_end|>\n))
// laid out as N_instruct standalone vectors before the role.
//
// Prompt text projection runs on the loaded GGML backend tensors through
// the pipeline scheduler. Backend failure is fatal: no host path exists.

#include "bpe.h"
#include "ggml.h"
#include "pipeline-tts.h"
#include "qt-error.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct PromptBuilderOutput {
    // Final talker input embedding [T_ctx, hidden] f32 row-major
    std::vector<float> input_embed;
    int                T_ctx;
    int                hidden;

    // Trailing text overlay: added on top of the next-token-input during
    // the autoregressive loop, one vector per generated frame until
    // exhausted, then tts_pad_embed for every following frame.
    std::vector<float> trailing_text_hidden;
    int                T_trailing;

    // tts_pad_embed [hidden] f32, kept around so the generation loop
    // can fall back on it once the trailing text runs out.
    std::vector<float> tts_pad_embed;

    // Token ids fed through the tokenizer (kept for debug parity with
    // the Python prompt-ids.bin dump).
    std::vector<int32_t> prompt_ids;

    // Length of the text segment used as utterance text, ie input_id[3:-5].
    int N_text;
};

// Convert one row of an embedding matrix W [vocab, dim] to f32. Uses the
// ggml type traits to_float dispatch so every dtype shipped by the
// quantizer is supported (F32, BF16, F16, Q8_0, Q4_K_M, etc). The row
// stride is the type block size, computed via ggml_row_size.
static void embed_row_to_f32(const GGUFModel & gf, const char * tensor_name, int row_id, int dim, float * dst) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, tensor_name);
    if (!src) {
        qt_throw("[Prompt] tensor '%s' not in meta context", tensor_name);
    }
    if (src->ne[0] != dim) {
        qt_throw("[Prompt] tensor '%s' dim mismatch %lld vs %d", tensor_name, (long long) src->ne[0], dim);
    }
    if (row_id < 0 || row_id >= (int) src->ne[1]) {
        qt_throw("[Prompt] row %d out of range for '%s' (vocab=%lld)", row_id, tensor_name, (long long) src->ne[1]);
    }

    const uint8_t * base = (const uint8_t *) gf_get_data(gf, tensor_name);
    if (!base) {
        qt_throw("[Prompt] tensor '%s' has no data", tensor_name);
    }

    const size_t row_bytes = ggml_row_size(src->type, dim);
    const void * row       = base + (size_t) row_id * row_bytes;

    if (src->type == GGML_TYPE_F32) {
        std::memcpy(dst, row, (size_t) dim * sizeof(float));
        return;
    }

    const struct ggml_type_traits * tt = ggml_get_type_traits(src->type);
    if (!tt || !tt->to_float) {
        qt_throw("[Prompt] unsupported dtype %d for '%s'", (int) src->type, tensor_name);
    }
    tt->to_float(row, dst, dim);
}

static bool project_text_ids_backend(PipelineTTS * pt, const int32_t * ids, int count, float * dst) {
    if (count <= 0) {
        return true;
    }
    if (!pt->sched || !pt->talker.text_embedding || !pt->talker.text_proj_fc1_w || !pt->talker.text_proj_fc1_b ||
        !pt->talker.text_proj_fc2_w || !pt->talker.text_proj_fc2_b) {
        return false;
    }

    const int    hidden    = pt->talker.hidden_size;
    const int    max_nodes = 64;
    const size_t graph_arena_bytes =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);

    struct ggml_init_params gparams = {
        graph_arena_bytes,
        NULL,
        true,
    };
    struct ggml_context * gctx = ggml_init(gparams);
    if (!gctx) {
        return false;
    }

    struct ggml_tensor * ids_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, count);
    ggml_set_name(ids_in, "prompt_text_ids");
    ggml_set_input(ids_in);

    struct ggml_tensor * h = ggml_get_rows(gctx, pt->talker.text_embedding, ids_in);
    ggml_set_name(h, "prompt_text_embeds");
    h = ggml_mul_mat(gctx, pt->talker.text_proj_fc1_w, h);
    ggml_mul_mat_set_prec(h, GGML_PREC_F32);
    h = ggml_add(gctx, h, pt->talker.text_proj_fc1_b);
    h = ggml_silu(gctx, h);

    struct ggml_tensor * out = ggml_mul_mat(gctx, pt->talker.text_proj_fc2_w, h);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    out = ggml_add(gctx, out, pt->talker.text_proj_fc2_b);
    ggml_set_name(out, "prompt_text_projection");
    ggml_set_output(out);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, max_nodes, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_sched_reset(pt->sched);
    if (!ggml_backend_sched_alloc_graph(pt->sched, graph)) {
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        return false;
    }

    ggml_backend_tensor_set(ids_in, ids, 0, (size_t) count * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(pt->sched, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        return false;
    }

    ggml_backend_tensor_get(out, dst, 0, (size_t) count * (size_t) hidden * sizeof(float));
    ggml_backend_sched_reset(pt->sched);
    ggml_free(gctx);
    return true;
}

static void project_text_range(PipelineTTS * pt, const int * ids, int start, int end, float * dst) {
    const int count = end - start;
    if (count <= 0) {
        return;
    }

    std::vector<int32_t> ids_i32((size_t) count);
    for (int i = 0; i < count; i++) {
        ids_i32[(size_t) i] = (int32_t) ids[start + i];
    }
    if (!project_text_ids_backend(pt, ids_i32.data(), count, dst)) {
        qt_throw("[Prompt] backend text projection failed (%d ids)", count);
    }
}

// Project a flat list of text token ids -> [count, hidden] f32 on the
// backend in one pass. dst is row major. Empty list is a no op.
static void project_text_ids(PipelineTTS * pt, const std::vector<int32_t> & ids, float * dst) {
    if (ids.empty()) {
        return;
    }
    if (!project_text_ids_backend(pt, ids.data(), (int) ids.size(), dst)) {
        qt_throw("[Prompt] backend text projection failed (%d ids)", (int) ids.size());
    }
}

// Sum the num_code_groups codebook embeddings of the reference codes on the
// backend. ref_codes is [num_code_groups, ref_codes_T] row major i32.
// dst receives [ref_codes_T, hidden] f32 row major, one summed embedding per
// reference frame. Codebook 0 reads talker.codec_embedding, codebooks 1..N
// read code_predictor.codec_embedding[k - 1], all dequantized by get_rows.
static void project_ref_codes_backend(PipelineTTS * pt, const int32_t * ref_codes, int ref_codes_T, float * dst) {
    const int hidden = pt->talker.hidden_size;
    const int groups = pt->num_code_groups;

    const int    max_nodes = groups * 4 + 16;
    const size_t arena =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);

    struct ggml_init_params gp   = { arena, NULL, true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        qt_throw("[Prompt] ref codes graph init failed");
    }

    struct ggml_tensor * ids_all = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t) groups * ref_codes_T);
    ggml_set_name(ids_all, "ref_code_ids");
    ggml_set_input(ids_all);

    struct ggml_tensor * sum = NULL;
    for (int k = 0; k < groups; k++) {
        struct ggml_tensor * table =
            (k == 0) ? pt->talker.codec_embedding : pt->code_predictor.codec_embedding[(size_t) (k - 1)];
        struct ggml_tensor * ids_k =
            ggml_view_1d(gctx, ids_all, ref_codes_T, (size_t) k * (size_t) ref_codes_T * sizeof(int32_t));
        struct ggml_tensor * rows = ggml_get_rows(gctx, table, ids_k);
        sum                       = sum ? ggml_add(gctx, sum, rows) : rows;
    }
    ggml_set_name(sum, "ref_codes_sum");
    ggml_set_output(sum);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, max_nodes, false);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_sched_reset(pt->sched);
    if (!ggml_backend_sched_alloc_graph(pt->sched, graph)) {
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        qt_throw("[Prompt] ref codes graph alloc failed");
    }

    ggml_backend_tensor_set(ids_all, ref_codes, 0, (size_t) groups * (size_t) ref_codes_T * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(pt->sched, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(pt->sched);
        ggml_free(gctx);
        qt_throw("[Prompt] ref codes graph compute failed");
    }

    ggml_backend_tensor_get(sum, dst, 0, (size_t) ref_codes_T * (size_t) hidden * sizeof(float));
    ggml_backend_sched_reset(pt->sched);
    ggml_free(gctx);
}

static bool prompt_cache_load(PipelineTTS * pt) {
    const int hidden = pt->talker.hidden_size;

    PromptCache & pc = pt->prompt_cache;
    pc.initialized   = false;
    pc.prefix_entries.clear();
    pc.max_prefix_entries = 16;

    // Special text embeds: tts_bos, tts_eos, tts_pad projected in a single
    // backend pass then split into their slots.
    pc.tts_bos_emb.assign((size_t) hidden, 0.0f);
    pc.tts_eos_emb.assign((size_t) hidden, 0.0f);
    pc.tts_pad_emb.assign((size_t) hidden, 0.0f);
    {
        const int32_t      ids[3] = { (int32_t) pt->text_specials.tts_bos_id, (int32_t) pt->text_specials.tts_eos_id,
                                      (int32_t) pt->text_specials.tts_pad_id };
        std::vector<float> proj((size_t) 3 * (size_t) hidden);
        if (!project_text_ids_backend(pt, ids, 3, proj.data())) {
            qt_log(QT_LOG_ERROR, "[Prompt] FATAL: special text projection failed");
            return false;
        }
        std::memcpy(pc.tts_bos_emb.data(), proj.data() + (size_t) 0 * hidden, (size_t) hidden * sizeof(float));
        std::memcpy(pc.tts_eos_emb.data(), proj.data() + (size_t) 1 * hidden, (size_t) hidden * sizeof(float));
        std::memcpy(pc.tts_pad_emb.data(), proj.data() + (size_t) 2 * hidden, (size_t) hidden * sizeof(float));
    }

    // Codec specials are direct embedding lookups, no projection.
    pc.codec_pad_emb.assign((size_t) hidden, 0.0f);
    pc.codec_bos_emb.assign((size_t) hidden, 0.0f);
    embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", pt->codec_specials.pad_id, hidden,
                     pc.codec_pad_emb.data());
    embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", pt->codec_specials.bos_id, hidden,
                     pc.codec_bos_emb.data());

    pc.initialized = true;
    return true;
}

// Vector add: a += b, length n.
static void vec_add(float * a, const float * b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

static void append_ints_to_key(std::string & key, const char * label, const std::vector<int> & values) {
    key += label;
    key += '=';
    for (int v : values) {
        key += std::to_string(v);
        key += ',';
    }
    key += ';';
}

static std::string prompt_prefix_cache_key(const std::vector<int> & instruct_ids,
                                           const int *              role_ids,
                                           const std::vector<int> & codec_left) {
    std::string key;
    key.reserve(128 + instruct_ids.size() * 8 + codec_left.size() * 8);
    append_ints_to_key(key, "instruct", instruct_ids);
    std::vector<int> role(role_ids, role_ids + 3);
    append_ints_to_key(key, "role", role);
    append_ints_to_key(key, "codec", codec_left);
    return key;
}

static PromptPrefixCacheEntry * prompt_prefix_cache_find(PromptCache & pc, const std::string & key) {
    for (PromptPrefixCacheEntry & entry : pc.prefix_entries) {
        if (entry.key == key) {
            return &entry;
        }
    }
    return NULL;
}

static void prompt_prefix_cache_store(PromptCache &       pc,
                                      const std::string & key,
                                      int                 rows,
                                      int                 hidden,
                                      const float *       data) {
    if (pc.max_prefix_entries == 0 || rows <= 0) {
        return;
    }
    if (PromptPrefixCacheEntry * existing = prompt_prefix_cache_find(pc, key)) {
        existing->rows = rows;
        existing->input_embed_prefix.assign(data, data + (size_t) rows * (size_t) hidden);
        return;
    }
    while (pc.prefix_entries.size() >= pc.max_prefix_entries) {
        pc.prefix_entries.erase(pc.prefix_entries.begin());
    }
    PromptPrefixCacheEntry entry;
    entry.key  = key;
    entry.rows = rows;
    entry.input_embed_prefix.assign(data, data + (size_t) rows * (size_t) hidden);
    pc.prefix_entries.push_back(std::move(entry));
}

static bool prompt_builder_build(PipelineTTS *         pt,
                                 const BPETokenizer *  tok,
                                 const std::string &   utterance_text,
                                 const std::string &   language,
                                 const std::string &   instruct_text,
                                 const std::string &   speaker_name,
                                 const float *         ref_spk_emb,
                                 const std::string &   ref_text,
                                 const int32_t *       ref_codes,
                                 int                   ref_codes_T,
                                 PromptBuilderOutput * out) {
    const int hidden = pt->talker.hidden_size;

    if (!speaker_name.empty() && ref_spk_emb != NULL) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: speaker_name and ref_spk_emb are mutually exclusive");
        return false;
    }

    // Voice clone mode B: ref_text and ref_codes drive an ICL prefix.
    // Mode B requires ref_spk_emb so the speaker slot is also filled.
    const bool icl = !ref_text.empty() && ref_codes != NULL && ref_codes_T > 0;
    if (icl && ref_spk_emb == NULL) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: ICL mode requires ref_spk_emb (no --ref-wav?)");
        return false;
    }

    // Build the chat-templated prompt fed to the BPE tokenizer.
    // Same wrap as the upstream demos: assistant role + utterance +
    // im_end + newline + assistant role.
    std::string full_text;
    full_text.reserve(utterance_text.size() + 64);
    full_text = "<|im_start|>assistant\n";
    full_text += utterance_text;
    full_text += "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> ids = bpe_encode(tok, full_text, /*add_eos=*/false);
    if ((int) ids.size() < 8) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: tokenized prompt too short (%d tokens)", (int) ids.size());
        return false;
    }

    out->prompt_ids.assign(ids.begin(), ids.end());
    const int N      = (int) ids.size();
    const int N_text = N - 3 - 5;
    if (N_text <= 0) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: no utterance text in prompt (N=%d)", N);
        return false;
    }

    // Resolve language: "auto" -> no language id, prefill is 3 codec
    // tokens (nothink, think_bos, think_eos). Otherwise insert the
    // configured language id between think_bos and think_eos.
    int language_id = -1;
    {
        std::string lang_lc = language;
        for (char & c : lang_lc) {
            c = (char) std::tolower((unsigned char) c);
        }
        if (lang_lc != "auto") {
            for (const LanguageEntry & e : pt->languages) {
                if (e.name == lang_lc) {
                    language_id = e.id;
                    break;
                }
            }
            if (language_id < 0) {
                qt_log(QT_LOG_ERROR, "[Prompt] FATAL: unknown language '%s'", language.c_str());
                return false;
            }
        }
    }

    // Resolve speaker: empty name -> no speaker. Otherwise lookup case
    // insensitively in pt->speakers and override the language id with the
    // dialect entry when the user supplied language is chinese or auto,
    // mirroring modeling_qwen3_tts.py lines 2118 to 2122.
    int speaker_id = -1;
    if (!speaker_name.empty()) {
        std::string spk_lc = speaker_name;
        for (char & c : spk_lc) {
            c = (char) std::tolower((unsigned char) c);
        }
        const SpeakerEntry * found = NULL;
        for (const SpeakerEntry & e : pt->speakers) {
            if (e.name == spk_lc) {
                found = &e;
                break;
            }
        }
        if (!found) {
            qt_log(QT_LOG_ERROR, "[Prompt] FATAL: unknown speaker '%s'", speaker_name.c_str());
            return false;
        }
        speaker_id = found->id;

        // Dialect override: applied only when the user supplied language
        // is chinese or auto, the dialect string is non empty, and the
        // dialect resolves to a known language id.
        if (!found->dialect.empty()) {
            std::string lang_lc = language;
            for (char & c : lang_lc) {
                c = (char) std::tolower((unsigned char) c);
            }
            if (lang_lc == "chinese" || lang_lc == "auto") {
                int dialect_id = -1;
                for (const LanguageEntry & e : pt->languages) {
                    if (e.name == found->dialect) {
                        dialect_id = e.id;
                        break;
                    }
                }
                if (dialect_id < 0) {
                    qt_log(QT_LOG_ERROR, "[Prompt] FATAL: dialect '%s' not in language table", found->dialect.c_str());
                    return false;
                }
                language_id = dialect_id;
            }
        }
    }

    if (!pt->prompt_cache.initialized) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: prompt cache is not initialized");
        return false;
    }
    PromptCache & pc = pt->prompt_cache;

    const std::vector<float> & tts_bos_emb   = pc.tts_bos_emb;
    const std::vector<float> & tts_eos_emb   = pc.tts_eos_emb;
    const std::vector<float> & tts_pad_emb   = pc.tts_pad_emb;
    const std::vector<float> & codec_pad_emb = pc.codec_pad_emb;
    const std::vector<float> & codec_bos_emb = pc.codec_bos_emb;

    // Codec prefill list: 3 ids if auto (no language), 4 otherwise.
    // Speaker insertion: if a speaker id is set, the codec embedding row
    // for that speaker slips between think_eos and codec_pad in the codec
    // stream, mirroring modeling_qwen3_tts.py lines 2167 to 2172.
    std::vector<int> codec_prefill;
    if (language_id < 0) {
        codec_prefill = { pt->codec_specials.nothink_id, pt->codec_specials.think_bos_id,
                          pt->codec_specials.think_eos_id };
    } else {
        codec_prefill = { pt->codec_specials.think_id, pt->codec_specials.think_bos_id, language_id,
                          pt->codec_specials.think_eos_id };
    }
    if (speaker_id >= 0) {
        codec_prefill.push_back(speaker_id);
    } else if (ref_spk_emb != NULL) {
        // Sentinel: the codec_left builder below copies ref_spk_emb in
        // place of an embedding lookup whenever it sees -2.
        codec_prefill.push_back(-2);
    }
    const int        n_prefill      = (int) codec_prefill.size();
    const int        T_codec_prefix = n_prefill + 2;  // + codec_pad + codec_bos
    const int        n_pad_pre      = T_codec_prefix - 2;
    std::vector<int> codec_left     = codec_prefill;
    codec_left.push_back(pt->codec_specials.pad_id);

    // Tokenize the instruct segment when non empty. The wrapper mirrors
    // _build_instruct_text upstream: <|im_start|>user\n{instruct}<|im_end|>\n
    // The result is a flat list of text token ids that will be projected
    // and placed as standalone vectors at the head of the input embed,
    // with no codec stream contribution.
    std::vector<int> instruct_ids;
    if (!instruct_text.empty()) {
        std::string wrapped;
        wrapped.reserve(instruct_text.size() + 32);
        wrapped = "<|im_start|>user\n";
        wrapped += instruct_text;
        wrapped += "<|im_end|>\n";
        instruct_ids = bpe_encode(tok, wrapped, /*add_eos=*/false);
    }
    const int N_instruct = (int) instruct_ids.size();

    // Tokenize the reference utterance when ICL is active. The wrap is
    // identical to the main utterance: assistant role + ref_text +
    // im_end + newline + assistant role. We slice [3:-5] later to keep
    // only the inner text body, mirroring input_id[:, 3:-5] upstream.
    std::vector<int> ref_ids;
    int              N_ref_text = 0;
    if (icl) {
        std::string ref_full;
        ref_full.reserve(ref_text.size() + 64);
        ref_full = "<|im_start|>assistant\n";
        ref_full += ref_text;
        ref_full += "<|im_end|>\n<|im_start|>assistant\n";
        ref_ids = bpe_encode(tok, ref_full, /*add_eos=*/false);
        if ((int) ref_ids.size() < 8) {
            qt_log(QT_LOG_ERROR, "[Prompt] FATAL: ref_text tokenized too short (%d tokens)", (int) ref_ids.size());
            return false;
        }
        // ref_ids[3: -5] is the inner ref text body without role tokens
        N_ref_text = (int) ref_ids.size() - 3 - 5;
        if (N_ref_text <= 0) {
            qt_log(QT_LOG_ERROR, "[Prompt] FATAL: empty ref_text body");
            return false;
        }
    }

    // ICL geometry. text_lens = N_ref_text + N_text + 1 (tts_eos).
    // codec_lens = 1 (codec_bos) + ref_codes_T. The non_streaming_mode
    // branch upstream pads the shorter stream so they end up the same
    // length, except when text > codec where trailing_text_hidden carries
    // the leftover text rows.
    const int text_lens_icl  = icl ? (N_ref_text + N_text + 1) : 0;
    const int codec_lens_icl = icl ? (1 + ref_codes_T) : 0;
    const int icl_T          = icl ? (text_lens_icl > codec_lens_icl ? codec_lens_icl : codec_lens_icl) : 0;

    // Allocate the full output buffer.
    // Standard layout    : N_instruct + 3 (role) + (n_pad_pre + 1) + N_text + 1 (eos) + 1 (final)
    // ICL layout         : N_instruct + 3 (role) + (n_pad_pre + 1) + icl_T
    const int T_ctx =
        icl ? (N_instruct + 3 + (n_pad_pre + 1) + icl_T) : (N_instruct + 3 + (n_pad_pre + 1) + N_text + 1 + 1);
    out->T_ctx  = T_ctx;
    out->hidden = hidden;
    out->input_embed.assign((size_t) T_ctx * (size_t) hidden, 0.0f);
    out->N_text = N_text;

    int  row     = 0;
    auto row_ptr = [&](int r) {
        return out->input_embed.data() + (size_t) r * (size_t) hidden;
    };

    const int                prefix_rows      = N_instruct + 3 + (int) codec_left.size();
    const bool               cacheable_prefix = !icl && ref_spk_emb == NULL;
    std::string              prefix_key;
    PromptPrefixCacheEntry * prefix_hit = NULL;
    if (cacheable_prefix) {
        prefix_key = prompt_prefix_cache_key(instruct_ids, ids.data(), codec_left);
        prefix_hit = prompt_prefix_cache_find(pc, prefix_key);
    }

    if (prefix_hit) {
        std::memcpy(row_ptr(0), prefix_hit->input_embed_prefix.data(),
                    (size_t) prefix_hit->rows * (size_t) hidden * sizeof(float));
        row = prefix_hit->rows;
    } else {
        // Instruct prefix + role: text_proj over [instruct_ids ; ids[0:3]]
        // in one backend pass. These occupy the contiguous head rows
        // [0, N_instruct + 3) of the input embed, no codec stream added.
        {
            std::vector<int32_t> head_ids;
            head_ids.reserve((size_t) N_instruct + 3);
            for (int v : instruct_ids) {
                head_ids.push_back((int32_t) v);
            }
            head_ids.push_back((int32_t) ids[0]);
            head_ids.push_back((int32_t) ids[1]);
            head_ids.push_back((int32_t) ids[2]);
            project_text_ids(pt, head_ids, row_ptr(0));
            row = N_instruct + 3;
        }

        // Codec prefix: tts_pad x n_pad_pre + tts_bos, summed with
        // codec_emb([codec_prefill_list[:-1]] + codec_pad). The Python code
        // takes codec_input_embedding[:, :-1] which drops the codec_bos,
        // leaving [codec_prefill_list..., codec_pad].
        for (int i = 0; i < (int) codec_left.size(); i++) {
            float *       r        = row_ptr(row + i);
            // text stream: tts_pad * (n - 1) then tts_bos at the end
            const float * text_vec = (i == (int) codec_left.size() - 1) ? tts_bos_emb.data() : tts_pad_emb.data();
            std::memcpy(r, text_vec, (size_t) hidden * sizeof(float));
            // codec stream: either an embedding lookup or, when the
            // sentinel -2 marks the speaker slot, a direct copy of the
            // user supplied ref_spk_emb (voice clone mode A).
            std::vector<float> ce((size_t) hidden);
            if (codec_left[(size_t) i] == -2) {
                std::memcpy(ce.data(), ref_spk_emb, (size_t) hidden * sizeof(float));
            } else {
                embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", codec_left[(size_t) i], hidden,
                                 ce.data());
            }
            vec_add(r, ce.data(), hidden);
        }
        row += (int) codec_left.size();

        if (cacheable_prefix && row == prefix_rows) {
            prompt_prefix_cache_store(pc, prefix_key, prefix_rows, hidden, row_ptr(0));
        }
    }

    // From here, two paths: standard (no ICL) builds the trailing
    // utterance text + tts_eos + final_pad, ICL builds an aligned
    // text/codec block that replaces those rows entirely.
    if (!icl) {
        // Standard layout: trailing utterance text + tts_eos rows summed
        // with codec_pad, then a final tts_pad + codec_bos row.
        project_text_range(pt, ids.data(), 3, 3 + N_text, row_ptr(row));
        for (int i = 0; i < N_text; i++) {
            float * r = row_ptr(row + i);
            vec_add(r, codec_pad_emb.data(), hidden);
        }
        row += N_text;
        {
            float * r = row_ptr(row);
            std::memcpy(r, tts_eos_emb.data(), (size_t) hidden * sizeof(float));
            vec_add(r, codec_pad_emb.data(), hidden);
            row++;
        }
        {
            float * r = row_ptr(row);
            std::memcpy(r, tts_pad_emb.data(), (size_t) hidden * sizeof(float));
            vec_add(r, codec_bos_emb.data(), hidden);
            row++;
        }
    } else {
        // ICL layout: compute the text stream and the codec stream
        // separately then add them. The text stream is text_proj of
        // [ref_text_ids; utterance_text_ids] followed by tts_eos. The
        // codec stream is codec_emb(codec_bos) followed by sum over the
        // 16 codebook embeddings of ref_codes[i, t] for each frame t.
        // Both streams are aligned to length icl_T per the upstream
        // non_streaming_mode=False branch of generate_icl_prompt.
        const int T_icl = codec_lens_icl;  // text_lens > codec: truncate to codec, else pad text up to codec

        // Build the codec stream [T_icl, hidden]. Row 0 is codec_bos, rows
        // 1..ref_codes_T are the per frame sum over the num_code_groups
        // codebook embeddings, computed on the backend in one pass.
        std::vector<float> codec_stream((size_t) T_icl * (size_t) hidden, 0.0f);
        std::memcpy(codec_stream.data(), codec_bos_emb.data(), (size_t) hidden * sizeof(float));
        project_ref_codes_backend(pt, ref_codes, ref_codes_T, codec_stream.data() + (size_t) hidden);

        // Build the text stream [text_lens_icl, hidden] = text_proj of
        // [ref_text ; utterance_text] then tts_eos. The ref and utterance
        // bodies are contiguous, projected in one backend pass.
        std::vector<float> text_stream((size_t) text_lens_icl * (size_t) hidden, 0.0f);
        {
            std::vector<int32_t> text_ids;
            text_ids.reserve((size_t) N_ref_text + (size_t) N_text);
            for (int i = 0; i < N_ref_text; i++) {
                text_ids.push_back((int32_t) ref_ids[3 + i]);
            }
            for (int i = 0; i < N_text; i++) {
                text_ids.push_back((int32_t) ids[3 + i]);
            }
            project_text_ids(pt, text_ids, text_stream.data());
        }
        // Append tts_eos at the end of the text stream.
        std::memcpy(text_stream.data() + (size_t) (text_lens_icl - 1) * (size_t) hidden, tts_eos_emb.data(),
                    (size_t) hidden * sizeof(float));

        // Align the two streams to T_icl. text_lens > codec: truncate
        // text and stash the leftover into trailing_text_hidden. text_lens
        // <= codec: pad text with tts_pad up to codec, trailing reduces
        // to tts_pad.
        std::vector<float> aligned_text((size_t) T_icl * (size_t) hidden, 0.0f);
        if (text_lens_icl >= T_icl) {
            // truncate text to T_icl rows, leftover goes into trailing
            std::memcpy(aligned_text.data(), text_stream.data(), (size_t) T_icl * (size_t) hidden * sizeof(float));
            const int trailing_n = text_lens_icl - T_icl;
            out->T_trailing      = trailing_n > 0 ? trailing_n : 1;
            out->trailing_text_hidden.assign((size_t) out->T_trailing * (size_t) hidden, 0.0f);
            if (trailing_n > 0) {
                std::memcpy(out->trailing_text_hidden.data(), text_stream.data() + (size_t) T_icl * (size_t) hidden,
                            (size_t) trailing_n * (size_t) hidden * sizeof(float));
            } else {
                std::memcpy(out->trailing_text_hidden.data(), tts_pad_emb.data(), (size_t) hidden * sizeof(float));
            }
        } else {
            // pad text with tts_pad up to T_icl, trailing = single tts_pad row
            std::memcpy(aligned_text.data(), text_stream.data(),
                        (size_t) text_lens_icl * (size_t) hidden * sizeof(float));
            for (int i = text_lens_icl; i < T_icl; i++) {
                std::memcpy(aligned_text.data() + (size_t) i * (size_t) hidden, tts_pad_emb.data(),
                            (size_t) hidden * sizeof(float));
            }
            out->T_trailing = 1;
            out->trailing_text_hidden.assign((size_t) hidden, 0.0f);
            std::memcpy(out->trailing_text_hidden.data(), tts_pad_emb.data(), (size_t) hidden * sizeof(float));
        }

        // Sum aligned_text + codec_stream into the input embed at the
        // current row offset.
        for (int i = 0; i < T_icl; i++) {
            float * r = row_ptr(row + i);
            std::memcpy(r, aligned_text.data() + (size_t) i * (size_t) hidden, (size_t) hidden * sizeof(float));
            vec_add(r, codec_stream.data() + (size_t) i * (size_t) hidden, hidden);
        }
        row += T_icl;
    }

    if (row != T_ctx) {
        qt_log(QT_LOG_ERROR, "[Prompt] FATAL: layout error row=%d expected T_ctx=%d", row, T_ctx);
        return false;
    }

    // Trailing text hidden: non streaming mode (no ICL) collapses the
    // overlay to a single row equal to tts_pad_embed
    // (modeling_qwen3_tts.py line 2227). The full utterance text is
    // already integrated into the prefill above as codec_pad summed text
    // rows + tts_eos, so the overlay loop only ever needs tts_pad: step
    // 0 reads trailing_text_hidden[0] which is tts_pad, every later step
    // falls through to the else branch and reads tts_pad_embed. One row,
    // bit exact with the Python hook dump.
    //
    // ICL mode populates out->trailing_text_hidden directly inside the
    // ICL branch above, so we only set the default here for non ICL.
    if (!icl) {
        out->T_trailing = 1;
        out->trailing_text_hidden.assign((size_t) hidden, 0.0f);
        std::memcpy(out->trailing_text_hidden.data(), tts_pad_emb.data(), (size_t) hidden * sizeof(float));
    }

    out->tts_pad_embed = tts_pad_emb;

    qt_log(QT_LOG_INFO,
           "[Prompt] Built: %d ids, N_text=%d, N_instruct=%d, T_ctx=%d, hidden=%d, lang=%s (id=%d), speaker=%s "
           "(id=%d) ref_spk_emb=%s icl=%s",
           N, N_text, N_instruct, T_ctx, hidden, language.c_str(), language_id,
           speaker_name.empty() ? "none" : speaker_name.c_str(), speaker_id, ref_spk_emb ? "yes" : "no",
           icl ? "yes" : "no");

    return true;
}
