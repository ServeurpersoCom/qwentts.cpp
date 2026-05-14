#pragma once
// code-predictor-forward.h: run the 5-layer Qwen3 code predictor over a
// growing context to produce the 15 acoustic codes of one audio frame,
// KV cached.
//
// Input:
//   talker_hidden_last [hidden] f32   -- last position hidden state from
//                                        the Talker forward (post final norm)
//   c0                                -- semantic code sampled from the
//                                        Talker codec_head (codebook 0)
// Output:
//   codes[16] = [c0, c1, ..., c15]    -- the full set of codes for one
//                                        frame, ready for decode through
//                                        the codec
//
// The predictor cache is local to a single frame: we reset it at every
// frame, prefill the first two positions (talker_hidden + embed(c0)),
// then decode 14 single-token steps. Total work drops from
// O(sum_{g=0..14} (g+2)^2) = O(1496 token-steps) to O(16) per frame,
// roughly 90x for the inner loop.
//
// Architecture mirrors the Talker block, only differences are:
//   - 5 layers instead of 28
//   - plain 1D RoPE (no multimodal sections)
//   - one private embedding table and one private linear head per
//     acoustic codebook (1..15)
//
// The single-frame loop here recomputes the full graph at every step g
// (0..14) over a sequence of length g+2. With 5 layers and at most 16
// tokens per recompute this is sub-millisecond on modern GPUs.

#include "code-predictor-weights.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "qt-error.h"
#include "sampling.h"
#include "talker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct CodePredictorOutput {
    // Sixteen codes: c0 from the talker plus c1..c15 from the predictor.
    std::vector<int32_t> codes;
};

// One Qwen3 decoder block, KV cached. K and V for the T fresh positions
// are written into the cache at [n_past, n_past+T) on dim 1; the
// attention reads the contiguous slice [0, n_past+T). Returns the layer
// output [hidden, T].
static struct ggml_tensor * code_predictor_layer_forward(struct ggml_context *        ctx,
                                                         const CodePredictorWeights * cw,
                                                         const TalkerLayer &          layer,
                                                         struct ggml_tensor *         x,
                                                         struct ggml_tensor *         positions,
                                                         struct ggml_tensor *         mask,
                                                         struct ggml_tensor *         k_cache,
                                                         struct ggml_tensor *         v_cache,
                                                         int                          n_past,
                                                         int                          T,
                                                         struct ggml_cgraph *         gf) {
    const int   n_q_heads = cw->num_attention_heads;
    const int   n_kv      = cw->num_key_value_heads;
    const int   hd        = cw->head_dim;
    const float eps       = cw->rms_norm_eps;

    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Write the fresh positions into the cache.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, T, n_kv]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    size_t k_off = (size_t) n_past * k_cache->nb[1];
    size_t v_off = (size_t) n_past * v_cache->nb[1];

    struct ggml_tensor * k_dst = ggml_view_3d(ctx, k_cache, hd, T, n_kv, k_cache->nb[1], k_cache->nb[2], k_off);
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, v_cache, hd, T, n_kv, v_cache->nb[1], v_cache->nb[2], v_off);

    struct ggml_tensor * k_cpy = ggml_cpy(ctx, k_perm, k_dst);
    struct ggml_tensor * v_cpy = ggml_cpy(ctx, v_perm, v_dst);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    const int            T_full = n_past + T;
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, T_full, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, T_full, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    // Q permute [hd, n_q_heads, T] -> [hd, T, n_q_heads] for flash_attn_ext.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    // Fused flash attention. Matches the working acestep qw3lm_build_attn
    // pattern, fixes the Vulkan autoregressive decode bug.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn  = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);
    x                      = ggml_add(ctx, x, o);

    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    return x;
}

// Run one predictor pass: feed `T` fresh embeddings starting at cache
// position `n_past`, run all 5 layers, and pull the logits for the last
// position through lm_head[g_head]. The cache is written as a side
// effect so subsequent decode steps can append a single token.
static bool code_predictor_run(const CodePredictorWeights * cw,
                               KVCache *                    kv,
                               ggml_backend_sched_t         sched,
                               const float *                fresh_input,
                               int                          T,
                               int                          n_past,
                               int                          talker_hidden,
                               int                          g_head,
                               std::vector<float> *         logits_out) {
    const int vocab    = cw->vocab_size;
    const int n_layers = cw->num_hidden_layers;
    const int T_full   = n_past + T;

    const int    max_nodes   = 48 * n_layers + 64;
    const size_t arena_bytes = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);

    struct ggml_init_params gp   = { arena_bytes, NULL, true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        fprintf(stderr, "[CodePredictor] FATAL: ggml_init failed\n");
        return false;
    }

    // Inputs: fresh embeddings (talker_hidden), positions, attention mask
    struct ggml_tensor * x_in    = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, talker_hidden, T);
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_full, T);
    ggml_set_name(x_in, "sub_input");
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);

    // small_to_mtp projection: Linear(talker_hidden -> hidden) with bias.
    // When absent (Identity case) the input is already at predictor hidden.
    struct ggml_tensor * h = x_in;
    if (cw->mtp_proj_w) {
        h = ggml_mul_mat(gctx, cw->mtp_proj_w, h);
        if (cw->mtp_proj_b) {
            h = ggml_add(gctx, h, cw->mtp_proj_b);
        }
        ggml_set_name(h, "mtp_proj_out");
    }

    for (int l = 0; l < n_layers; l++) {
        h = code_predictor_layer_forward(gctx, cw, cw->layers[(size_t) l], h, pos_in, mask_in, kv->k[(size_t) l],
                                         kv->v[(size_t) l], n_past, T, gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, cw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, cw->norm_w);

    struct ggml_tensor * logits = ggml_mul_mat(gctx, cw->lm_head[(size_t) g_head], h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "[CodePredictor] FATAL: graph allocation failed\n");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }

    ggml_backend_tensor_set(x_in, fresh_input, 0, (size_t) T * (size_t) talker_hidden * sizeof(float));

    {
        std::vector<int32_t> pos((size_t) T);
        for (int i = 0; i < T; i++) {
            pos[(size_t) i] = n_past + i;
        }
        ggml_backend_tensor_set(pos_in, pos.data(), 0, (size_t) T * sizeof(int32_t));
    }

    {
        std::vector<ggml_fp16_t> mask((size_t) T * (size_t) T_full);
        const ggml_fp16_t        zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); i++) {
            mask[i] = neg_inf;
        }
        for (int q = 0; q < T; q++) {
            const int q_pos = n_past + q;
            for (int k = 0; k <= q_pos; k++) {
                mask[(size_t) q * (size_t) T_full + (size_t) k] = zero;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[CodePredictor] FATAL: graph compute failed\n");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }

    logits_out->resize((size_t) vocab);
    size_t row_bytes = (size_t) vocab * sizeof(float);
    ggml_backend_tensor_get(logits, logits_out->data(), (size_t) (T - 1) * row_bytes, row_bytes);

    kv->cur_len = T_full;

    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return true;
}

// Read one row of an embedding table to f32. Reads from the backend
// (the predictor weights live there) via ggml_backend_tensor_get,
// dispatched through ggml_get_type_traits so quants are accepted.
static void embed_row_from_backend(struct ggml_tensor * t, int row_id, int dim, float * dst) {
    if (t->ne[0] != dim) {
        qt_throw("[CodePredictor] embed dim mismatch %lld vs %d", (long long) t->ne[0], dim);
    }
    if (row_id < 0 || row_id >= (int) t->ne[1]) {
        qt_throw("[CodePredictor] row %d out of range (vocab=%lld)", row_id, (long long) t->ne[1]);
    }
    const size_t row_bytes = ggml_row_size(t->type, dim);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst, (size_t) row_id * row_bytes, row_bytes);
        return;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) {
        qt_throw("[CodePredictor] unsupported embed dtype %d", (int) t->type);
    }
    std::vector<uint8_t> tmp(row_bytes);
    ggml_backend_tensor_get(t, tmp.data(), (size_t) row_id * row_bytes, row_bytes);
    tt->to_float(tmp.data(), dst, dim);
}

// Run the predictor for one audio frame. Caller passes the talker hidden
// state for the current frame and the already-sampled c0. Sampling
// parameters control greedy (temperature <= 0) vs stochastic. subseq_base
// is the Philox subsequence of the c0 sample for this step; the 15
// acoustic samples consume subseq_base + 1 .. subseq_base + 15.
// Returns the full vector of 16 codes. dump_dir may be NULL.
static bool code_predictor_step(const TalkerWeights *        tw,
                                const CodePredictorWeights * cw,
                                KVCache *                    kv,
                                ggml_backend_sched_t         sched,
                                const float *                talker_hidden_last,
                                int                          c0,
                                float                        temperature,
                                int                          top_k,
                                float                        top_p,
                                int64_t                      seed,
                                int64_t                      subseq_base,
                                const char *                 dump_dir,
                                CodePredictorOutput *        out) {
    // sub_input slots live at the talker hidden dimension: both the
    // talker last hidden and the codec_embedding rows feeding the sub
    // network are talker sized in the upstream checkpoint. The graph's
    // mtp_proj brings them down to predictor hidden when present.
    const int talker_hidden = tw->hidden_size;
    const int n_acoustic    = cw->num_acoustic_codebooks;

    if (n_acoustic + 1 > kv->max_seq_len) {
        fprintf(stderr, "[CodePredictor] FATAL: frame width %d exceeds cache max_seq_len %d\n", n_acoustic + 1,
                kv->max_seq_len);
        return false;
    }

    out->codes.assign((size_t) (n_acoustic + 1), 0);
    out->codes[0] = c0;

    // Prefill: two positions, talker_hidden_last and embed_talker(c0).
    kv_cache_reset(kv);
    std::vector<float> prefill_input((size_t) 2 * (size_t) talker_hidden, 0.0f);
    std::memcpy(prefill_input.data(), talker_hidden_last, (size_t) talker_hidden * sizeof(float));
    embed_row_from_backend(tw->codec_embedding, c0, talker_hidden, prefill_input.data() + (size_t) talker_hidden);

    std::vector<float> logits;
    if (!code_predictor_run(cw, kv, sched, prefill_input.data(), 2, 0, talker_hidden, 0, &logits)) {
        return false;
    }
    {
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1, &u_g);
        if (subseq_base + 1 < 32) {
            fprintf(stderr, "[Sample-CP] g=0 c=%d u=%.10f subseq=%lld\n", cg, (double) u_g,
                    (long long) (subseq_base + 1));
        }
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=0\n");
            return false;
        }
        out->codes[1] = cg;
    }

    // Decode loop: 14 single-token steps. At step g (g=1..14) we feed
    // the embedding of the code we just sampled and read lm_head[g].
    std::vector<float> step_input((size_t) talker_hidden);
    for (int g = 1; g < n_acoustic; g++) {
        embed_row_from_backend(cw->codec_embedding[(size_t) (g - 1)], out->codes[(size_t) g], talker_hidden,
                               step_input.data());
        if (!code_predictor_run(cw, kv, sched, step_input.data(), 1, kv->cur_len, talker_hidden, g, &logits)) {
            return false;
        }
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1 + g, &u_g);
        if (subseq_base + 1 + g < 32) {
            fprintf(stderr, "[Sample-CP] g=%d c=%d u=%.10f subseq=%lld\n", g, cg, (double) u_g,
                    (long long) (subseq_base + 1 + g));
        }
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=%d\n", g);
            return false;
        }
        out->codes[(size_t) (g + 1)] = cg;
    }

    if (dump_dir) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        std::vector<int32_t> codes32(out->codes.begin(), out->codes.end());
        int                  n = (int) codes32.size();
        debug_dump_i32_as_f32(&d, "codes-step0", codes32.data(), &n, 1);
    }

    return true;
}
