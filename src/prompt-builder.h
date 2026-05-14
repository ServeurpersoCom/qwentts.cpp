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
// Empty / NULL strings disable the corresponding stream. The builder runs
// CPU-side using the BF16 weight blocks mmapped from the talker GGUF, no
// backend allocation, no graph compute.
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
// All math is f32. text_embedding and codec_embedding are read from
// the mmapped GGUF in their stored dtype (bf16 by default) and cast
// row by row. The 2-layer ResizeMLP runs as two GEMMs with a SiLU in
// between, with bias on both linear layers.

#include "bpe.h"
#include "ggml.h"
#include "pipeline-tts.h"
#include "qt-error.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Read a full small tensor (bias, projection weight) into an f32 buffer.
// Allocates dst.resize internally. Routed through ggml_get_type_traits so
// quants are accepted, same as embed_row_to_f32 above.
static void read_tensor_f32(const GGUFModel & gf, const char * tensor_name, std::vector<float> & dst) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, tensor_name);
    if (!src) {
        qt_throw("[Prompt] tensor '%s' not in meta context", tensor_name);
    }
    int64_t         n    = ggml_nelements(src);
    const uint8_t * base = (const uint8_t *) gf_get_data(gf, tensor_name);
    dst.resize((size_t) n);

    if (src->type == GGML_TYPE_F32) {
        std::memcpy(dst.data(), base, (size_t) n * sizeof(float));
        return;
    }

    const struct ggml_type_traits * tt = ggml_get_type_traits(src->type);
    if (!tt || !tt->to_float) {
        qt_throw("[Prompt] unsupported dtype %d for '%s'", (int) src->type, tensor_name);
    }
    tt->to_float(base, dst.data(), (int64_t) n);
}

// y = W @ x + b
//   x [in_dim] f32, W [out_dim, in_dim] row-major f32, b [out_dim] f32
//   y [out_dim] f32
// Naive dot-product GEMV, fine for small (≤2048) inputs at build time.
static void linear_f32(const float * x, const float * W, const float * b, int in_dim, int out_dim, float * y) {
    for (int o = 0; o < out_dim; o++) {
        const float * row = W + (size_t) o * (size_t) in_dim;
        float         acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

static inline float silu(float v) {
    return v / (1.0f + std::exp(-v));
}

// Apply text_projection: F1 (text_hidden -> text_hidden) -> SiLU -> F2
// (text_hidden -> hidden), both with bias.
struct TextProjection {
    int                in_dim;   // text_hidden_size
    int                hid_dim;  // intermediate (= text_hidden_size in 0.6B)
    int                out_dim;  // hidden_size
    std::vector<float> fc1_w;    // [hid_dim, in_dim]
    std::vector<float> fc1_b;    // [hid_dim]
    std::vector<float> fc2_w;    // [out_dim, hid_dim]
    std::vector<float> fc2_b;    // [out_dim]
};

static void text_projection_load(TextProjection * tp, const GGUFModel & gf, int text_hidden_size, int hidden_size) {
    tp->in_dim  = text_hidden_size;
    tp->hid_dim = text_hidden_size;
    tp->out_dim = hidden_size;
    read_tensor_f32(gf, "talker.text_proj.fc1.weight", tp->fc1_w);
    read_tensor_f32(gf, "talker.text_proj.fc1.bias", tp->fc1_b);
    read_tensor_f32(gf, "talker.text_proj.fc2.weight", tp->fc2_w);
    read_tensor_f32(gf, "talker.text_proj.fc2.bias", tp->fc2_b);
}

static void text_projection_apply(const TextProjection * tp, const float * x, float * y) {
    std::vector<float> h((size_t) tp->hid_dim);
    linear_f32(x, tp->fc1_w.data(), tp->fc1_b.data(), tp->in_dim, tp->hid_dim, h.data());
    for (int i = 0; i < tp->hid_dim; i++) {
        h[(size_t) i] = silu(h[(size_t) i]);
    }
    linear_f32(h.data(), tp->fc2_w.data(), tp->fc2_b.data(), tp->hid_dim, tp->out_dim, y);
}

// Compute text_proj(text_embedding(ids[start:end])) row by row, append
// to dst (which already holds previous rows). Each output row is one
// hidden-dim vector.
static void embed_text_range(const GGUFModel &      gf,
                             const TextProjection * tp,
                             const int32_t *        ids,
                             int                    start,
                             int                    end,
                             int                    text_hidden_size,
                             int                    hidden_size,
                             std::vector<float> &   dst) {
    std::vector<float> e((size_t) text_hidden_size);
    std::vector<float> y((size_t) hidden_size);
    for (int i = start; i < end; i++) {
        embed_row_to_f32(gf, "talker.text_embd.weight", ids[i], text_hidden_size, e.data());
        text_projection_apply(tp, e.data(), y.data());
        dst.insert(dst.end(), y.begin(), y.end());
    }
}

// Append codec_embedding(id) to dst (one hidden-dim vector).
static void embed_codec(const GGUFModel & gf, int id, int hidden_size, std::vector<float> & dst) {
    size_t old = dst.size();
    dst.resize(old + (size_t) hidden_size);
    embed_row_to_f32(gf, "talker.codec_embd.weight", id, hidden_size, dst.data() + old);
}

// Vector add: a += b, length n.
static void vec_add(float * a, const float * b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

static bool prompt_builder_build(const PipelineTTS *   pt,
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
    const int hidden   = pt->talker.hidden_size;
    const int text_hid = pt->talker.text_hidden_size;

    if (!speaker_name.empty() && ref_spk_emb != NULL) {
        fprintf(stderr, "[Prompt] FATAL: speaker_name and ref_spk_emb are mutually exclusive\n");
        return false;
    }

    // Voice clone mode B: ref_text and ref_codes drive an ICL prefix.
    // Mode B requires ref_spk_emb so the speaker slot is also filled.
    const bool icl = !ref_text.empty() && ref_codes != NULL && ref_codes_T > 0;
    if (icl && ref_spk_emb == NULL) {
        fprintf(stderr, "[Prompt] FATAL: ICL mode requires ref_spk_emb (no --ref-wav?)\n");
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
        fprintf(stderr, "[Prompt] FATAL: tokenized prompt too short (%d tokens)\n", (int) ids.size());
        return false;
    }

    out->prompt_ids.assign(ids.begin(), ids.end());
    const int N      = (int) ids.size();
    const int N_text = N - 3 - 5;
    if (N_text <= 0) {
        fprintf(stderr, "[Prompt] FATAL: no utterance text in prompt (N=%d)\n", N);
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
                fprintf(stderr, "[Prompt] FATAL: unknown language '%s'\n", language.c_str());
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
            fprintf(stderr, "[Prompt] FATAL: unknown speaker '%s'\n", speaker_name.c_str());
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
                    fprintf(stderr, "[Prompt] FATAL: dialect '%s' not in language table\n", found->dialect.c_str());
                    return false;
                }
                language_id = dialect_id;
            }
        }
    }

    // Load the small tensors needed for the builder onto the host side.
    TextProjection tp;
    text_projection_load(&tp, pt->gguf_talker, text_hid, hidden);

    // Special embeds (tts_bos, tts_eos, tts_pad, codec_pad, codec_bos)
    // computed once.
    std::vector<float> tts_bos_emb((size_t) hidden);
    std::vector<float> tts_eos_emb((size_t) hidden);
    std::vector<float> tts_pad_emb((size_t) hidden);
    {
        std::vector<float> e((size_t) text_hid);
        embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", pt->text_specials.tts_bos_id, text_hid, e.data());
        text_projection_apply(&tp, e.data(), tts_bos_emb.data());
        embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", pt->text_specials.tts_eos_id, text_hid, e.data());
        text_projection_apply(&tp, e.data(), tts_eos_emb.data());
        embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", pt->text_specials.tts_pad_id, text_hid, e.data());
        text_projection_apply(&tp, e.data(), tts_pad_emb.data());
    }

    std::vector<float> codec_pad_emb((size_t) hidden);
    embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", pt->codec_specials.pad_id, hidden,
                     codec_pad_emb.data());

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
    const int n_prefill      = (int) codec_prefill.size();
    const int T_codec_prefix = n_prefill + 2;  // + codec_pad + codec_bos
    const int n_pad_pre      = T_codec_prefix - 2;

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
            fprintf(stderr, "[Prompt] FATAL: ref_text tokenized too short (%d tokens)\n", (int) ref_ids.size());
            return false;
        }
        // ref_ids[3: -5] is the inner ref text body without role tokens
        N_ref_text = (int) ref_ids.size() - 3 - 5;
        if (N_ref_text <= 0) {
            fprintf(stderr, "[Prompt] FATAL: empty ref_text body\n");
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

    // Instruct prefix: text_proj(text_embed(instruct_ids)). Standalone
    // vectors with no codec stream (zero pad_id sum, ie nothing added).
    if (N_instruct > 0) {
        std::vector<float> dst;
        embed_text_range(pt->gguf_talker, &tp, instruct_ids.data(), 0, N_instruct, text_hid, hidden, dst);
        std::memcpy(row_ptr(row), dst.data(), dst.size() * sizeof(float));
        row += N_instruct;
    }

    // Role: text_proj(text_embed(ids[0:3]))
    {
        std::vector<float> dst;
        dst.reserve((size_t) 3 * (size_t) hidden);
        embed_text_range(pt->gguf_talker, &tp, ids.data(), 0, 3, text_hid, hidden, dst);
        std::memcpy(row_ptr(row), dst.data(), dst.size() * sizeof(float));
        row += 3;
    }

    // Codec prefix: tts_pad x n_pad_pre + tts_bos, summed with
    // codec_emb([codec_prefill_list[:-1]] + codec_pad). The Python code
    // takes codec_input_embedding[:, :-1] which drops the codec_bos,
    // leaving [codec_prefill_list..., codec_pad].
    {
        std::vector<int> codec_left = codec_prefill;
        codec_left.push_back(pt->codec_specials.pad_id);
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
    }

    // From here, two paths: standard (no ICL) builds the trailing
    // utterance text + tts_eos + final_pad, ICL builds an aligned
    // text/codec block that replaces those rows entirely.
    if (!icl) {
        // Standard layout: trailing utterance text + tts_eos rows summed
        // with codec_pad, then a final tts_pad + codec_bos row.
        for (int i = 0; i < N_text; i++) {
            std::vector<float> e((size_t) text_hid);
            std::vector<float> y((size_t) hidden);
            embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", ids[3 + i], text_hid, e.data());
            text_projection_apply(&tp, e.data(), y.data());
            float * r = row_ptr(row);
            std::memcpy(r, y.data(), (size_t) hidden * sizeof(float));
            vec_add(r, codec_pad_emb.data(), hidden);
            row++;
        }
        {
            float * r = row_ptr(row);
            std::memcpy(r, tts_eos_emb.data(), (size_t) hidden * sizeof(float));
            vec_add(r, codec_pad_emb.data(), hidden);
            row++;
        }
        {
            float * r = row_ptr(row);
            std::memcpy(r, tts_pad_emb.data(), (size_t) hidden * sizeof(float));
            std::vector<float> ce((size_t) hidden);
            embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", pt->codec_specials.bos_id, hidden, ce.data());
            vec_add(r, ce.data(), hidden);
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

        // Build the codec stream [T_icl, hidden]. Row 0: codec_emb(codec_bos).
        // Row 1..ref_codes_T: sum over k=0..15 of codebook_k_emb(ref_codes[k, t]).
        std::vector<float> codec_stream((size_t) T_icl * (size_t) hidden, 0.0f);
        {
            // Row 0: codec_bos lookup.
            embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", pt->codec_specials.bos_id, hidden,
                             codec_stream.data());
            // Row 1..ref_codes_T: sum over codebooks.
            std::vector<float> tmp((size_t) hidden);
            for (int t = 0; t < ref_codes_T; t++) {
                float * dst   = codec_stream.data() + (size_t) (1 + t) * (size_t) hidden;
                // codebook 0 lives in talker.codec_embd
                int     code0 = ref_codes[(size_t) 0 * (size_t) ref_codes_T + (size_t) t];
                embed_row_to_f32(pt->gguf_talker, "talker.codec_embd.weight", code0, hidden, dst);
                // codebooks 1..15 live in code_pred.codec_embd.{i-1}
                for (int i = 1; i < pt->num_code_groups; i++) {
                    int  code = ref_codes[(size_t) i * (size_t) ref_codes_T + (size_t) t];
                    char tname[64];
                    std::snprintf(tname, sizeof(tname), "code_pred.codec_embd.%d.weight", i - 1);
                    embed_row_to_f32(pt->gguf_talker, tname, code, hidden, tmp.data());
                    vec_add(dst, tmp.data(), hidden);
                }
            }
        }

        // Build the text stream [text_lens_icl, hidden] = text_proj of
        // [ref_text; utterance_text] then tts_eos.
        std::vector<float> text_stream((size_t) text_lens_icl * (size_t) hidden, 0.0f);
        for (int i = 0; i < N_ref_text; i++) {
            std::vector<float> e((size_t) text_hid);
            float *            r = text_stream.data() + (size_t) i * (size_t) hidden;
            embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", ref_ids[3 + i], text_hid, e.data());
            text_projection_apply(&tp, e.data(), r);
        }
        for (int i = 0; i < N_text; i++) {
            std::vector<float> e((size_t) text_hid);
            float *            r = text_stream.data() + (size_t) (N_ref_text + i) * (size_t) hidden;
            embed_row_to_f32(pt->gguf_talker, "talker.text_embd.weight", ids[3 + i], text_hid, e.data());
            text_projection_apply(&tp, e.data(), r);
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
        fprintf(stderr, "[Prompt] FATAL: layout error row=%d expected T_ctx=%d\n", row, T_ctx);
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

    fprintf(stderr,
            "[Prompt] Built: %d ids, N_text=%d, N_instruct=%d, T_ctx=%d, hidden=%d, lang=%s (id=%d), speaker=%s "
            "(id=%d) ref_spk_emb=%s icl=%s\n",
            N, N_text, N_instruct, T_ctx, hidden, language.c_str(), language_id,
            speaker_name.empty() ? "none" : speaker_name.c_str(), speaker_id, ref_spk_emb ? "yes" : "no",
            icl ? "yes" : "no");

    return true;
}
