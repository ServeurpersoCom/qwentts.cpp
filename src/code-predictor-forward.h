#pragma once
// code-predictor-forward.h: run the 5-layer Qwen3 code predictor over a
// growing context to produce the 15 acoustic codes of one audio frame,
// KV cached.
//
// Input:
//   hidden_bridge [hidden] f32        -- persistent backend tensor holding
//                                        the talker last position hidden
//                                        (post final norm), written on
//                                        device by the talker graph and
//                                        read here as a graph leaf
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
// Graph metadata lives in caller owned static graphs, one for the T=2
// prefill and one per T=1 step: each graph is built and allocated once
// at load, then replayed directly on the backend with a 4 byte code id
// upload per call, the positions, kv rows, and causal mask baked in.

#include "code-predictor-graph.h"
#include "code-predictor-weights.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "qt-error.h"
#include "sampling.h"

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

// Manual F32 attention chain for the code predictor block. Same shape
// contract as talker_attn_f32: q [hd, T, n_q_heads], k/v [hd, T_full,
// n_kv], output [hd, n_q_heads, T]. Used when use_flash_attn is false.
static struct ggml_tensor * code_predictor_attn_f32(struct ggml_context * ctx,
                                                    struct ggml_tensor *  q,
                                                    struct ggml_tensor *  k,
                                                    struct ggml_tensor *  v,
                                                    struct ggml_tensor *  mask,
                                                    float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Node budget for one predictor graph, same accounting as the talker.
static int code_predictor_graph_max_nodes(int n_layers) {
    return 48 * n_layers + 64;
}

// One Qwen3 decoder block, KV cached. K and V for the T fresh positions
// are written into the cache at [n_past, n_past+T) on dim 1; the
// attention reads the fixed [0, n_kv_pad) window with the mask carrying
// neg inf beyond n_past+T. Returns the layer output [hidden, T].
// use_flash_attn and clamp_fp16 follow the same contract as in
// talker-forward.h.
static struct ggml_tensor * code_predictor_layer_forward(struct ggml_context *        ctx,
                                                         const CodePredictorWeights * cw,
                                                         const TalkerLayer &          layer,
                                                         struct ggml_tensor *         x,
                                                         struct ggml_tensor *         positions,
                                                         struct ggml_tensor *         mask,
                                                         struct ggml_tensor *         kv_rows,
                                                         struct ggml_tensor *         k_cache,
                                                         struct ggml_tensor *         v_cache,
                                                         int                          T,
                                                         int                          n_kv_pad,
                                                         bool                         use_flash_attn,
                                                         bool                         clamp_fp16,
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

    // Write the fresh positions into the cache via set_rows: positions
    // travel as data so every step keeps an identical topology and the
    // captured CUDA graph replays without an update.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, T, n_kv]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_perm, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_perm, kv_rows));

    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, n_kv_pad, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, n_kv_pad, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    // Q permute [hd, n_q_heads, T] -> [hd, T, n_q_heads] for flash_attn_ext.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    // Clamp V before attention when clamp_fp16 is set, same rationale
    // as the talker block: sub Ampere CUDA tensor cores accumulate in
    // FP16 and a V projection overflow corrupts everything downstream.
    if (clamp_fp16) {
        v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
    }

    // Attention: fused flash kernel or manual F32 chain. Matches the
    // working acestep qw3lm_build_attn pattern on the fused branch and
    // the omnivoice qwen3_attn_f32 helper on the manual one.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = code_predictor_attn_f32(ctx, q_p, k_full, v_full, mask, scale);
    }

    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);
    x                      = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

// Build one static predictor graph. A non NULL hidden_bridge selects
// the T=2 prefill flavor reading [talker_hidden, embed(c0)] through
// lm_head[0]; otherwise the graph is the single token step for g_head,
// appending at the fixed cache row g_head + 1. The logits node holds
// the last position only, so every flavor reads back one row at
// offset zero. use_flash_attn / clamp_fp16 apply to every layer.
static bool code_predictor_graph_build(const CodePredictorWeights * cw,
                                       KVCache *                    kv,
                                       ggml_backend_t               backend,
                                       struct ggml_tensor *         embd_table,
                                       struct ggml_tensor *         hidden_bridge,
                                       int                          g_head,
                                       bool                         use_flash_attn,
                                       bool                         clamp_fp16,
                                       CodePredGraph *              cp) {
    const int T        = hidden_bridge ? 2 : 1;
    const int n_past   = hidden_bridge ? 0 : g_head + 1;
    const int n_layers = cw->num_hidden_layers;

    // The attention window spans the whole frame cache (16 slots): a
    // constant width keeps every flavor at the same mask shape.
    const int n_kv_pad = kv->max_seq_len;

    const int    max_nodes = code_predictor_graph_max_nodes(n_layers);
    const size_t bytes =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);
    struct ggml_init_params gp = { bytes, NULL, true };
    cp->ctx                    = ggml_init(gp);
    if (!cp->ctx) {
        fprintf(stderr, "[CodePredictor] FATAL: graph ctx allocation failed\n");
        return false;
    }
    struct ggml_context * gctx = cp->ctx;

    // Inputs: one code id gathered in graph from embd_table, positions,
    // attention mask. The prefill path (T == 2, hidden_bridge non NULL)
    // concats the resident talker hidden ahead of embed(c0), both on
    // device: the sequence is [talker_hidden, embed(c0)] with zero row
    // upload. Steps (T == 1) are pure gathers: the only per step upload
    // is 4 bytes of code id.
    struct ggml_tensor * ids_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_kv_pad, T);
    struct ggml_tensor * rows_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I64, T);
    ggml_set_name(ids_in, "sub_code_id");
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");
    ggml_set_name(rows_in, "kv_rows");
    // ids uploads before every replay; pos, rows, and mask bake once,
    // so they also carry the output flag: the allocator never frees an
    // output, which keeps their slots out of the intermediate reuse
    // pool across replays.
    ggml_set_input(ids_in);
    ggml_set_input(pos_in);
    ggml_set_output(pos_in);
    ggml_set_input(mask_in);
    ggml_set_output(mask_in);
    ggml_set_input(rows_in);
    ggml_set_output(rows_in);

    struct ggml_tensor * x_in = ggml_get_rows(gctx, embd_table, ids_in);
    if (T == 2) {
        x_in = ggml_concat(gctx, hidden_bridge, x_in, 1);
    }
    ggml_set_name(x_in, "sub_input");

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
        h = code_predictor_layer_forward(gctx, cw, cw->layers[(size_t) l], h, pos_in, mask_in, rows_in,
                                         kv->k[(size_t) l], kv->v[(size_t) l], T, n_kv_pad, use_flash_attn, clamp_fp16,
                                         gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, cw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, cw->norm_w);
    if (T > 1) {
        // Last position only: the prefill pays a single lm_head row.
        h_final = ggml_cont(
            gctx, ggml_view_2d(gctx, h_final, h_final->ne[0], 1, h_final->nb[1], (size_t) (T - 1) * h_final->nb[1]));
    }

    struct ggml_tensor * logits = ggml_mul_mat(gctx, cw->lm_head[(size_t) g_head], h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    cp->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!cp->galloc || !ggml_gallocr_alloc_graph(cp->galloc, gf)) {
        fprintf(stderr, "[CodePredictor] FATAL: graph allocation failed\n");
        code_predictor_graph_free(cp);
        return false;
    }

    {
        std::vector<int32_t> pos((size_t) T);
        for (int i = 0; i < T; i++) {
            pos[(size_t) i] = n_past + i;
        }
        ggml_backend_tensor_set(pos_in, pos.data(), 0, (size_t) T * sizeof(int32_t));

        std::vector<int64_t> rows((size_t) T);
        for (int i = 0; i < T; i++) {
            rows[(size_t) i] = (int64_t) (n_past + i);
        }
        ggml_backend_tensor_set(rows_in, rows.data(), 0, (size_t) T * sizeof(int64_t));
    }

    {
        std::vector<ggml_fp16_t> mask((size_t) T * (size_t) n_kv_pad);
        const ggml_fp16_t        zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); i++) {
            mask[i] = neg_inf;
        }
        for (int q = 0; q < T; q++) {
            const int q_pos = n_past + q;
            for (int k = 0; k <= q_pos; k++) {
                mask[(size_t) q * (size_t) n_kv_pad + (size_t) k] = zero;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    cp->gf     = gf;
    cp->ids_in = ids_in;
    cp->logits = logits;
    return true;
}

// Replay one static predictor graph: upload the code id, run the graph
// directly on the backend, read the single logits row back.
static bool code_predictor_replay(CodePredGraph *      cp,
                                  ggml_backend_t       backend,
                                  int32_t              code_id,
                                  std::vector<float> * logits_out) {
    ggml_backend_tensor_set(cp->ids_in, &code_id, 0, sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, cp->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[CodePredictor] FATAL: graph compute failed\n");
        return false;
    }
    logits_out->resize((size_t) cp->logits->ne[0]);
    ggml_backend_tensor_get(cp->logits, logits_out->data(), 0, (size_t) cp->logits->ne[0] * sizeof(float));
    return true;
}

// Run the predictor for one audio frame over the static graphs. The
// prefill replay consumes the persistent hidden bridge already written
// by the talker graph and the sampled c0; the 14 step replays each
// feed the code sampled just before. Sampling parameters control
// greedy (temperature <= 0) vs stochastic. subseq_base is the Philox
// subsequence of the c0 sample for this step; the 15 acoustic samples
// consume subseq_base + 1 .. subseq_base + 15. Returns the full vector
// of 16 codes. dump_dir may be NULL.
static bool code_predictor_step(const CodePredictorWeights * cw,
                                ggml_backend_t               backend,
                                CodePredGraph *              prefill_graph,
                                CodePredGraph *              step_graphs,
                                int                          c0,
                                float                        temperature,
                                int                          top_k,
                                float                        top_p,
                                int64_t                      seed,
                                int64_t                      subseq_base,
                                const char *                 dump_dir,
                                CodePredictorOutput *        out) {
    const int n_acoustic = cw->num_acoustic_codebooks;

    out->codes.assign((size_t) (n_acoustic + 1), 0);
    out->codes[0] = c0;

    std::vector<float> logits;
    if (!code_predictor_replay(prefill_graph, backend, c0, &logits)) {
        return false;
    }
    {
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1, &u_g);
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=0\n");
            return false;
        }
        out->codes[1] = cg;
    }

    // Decode loop: 14 single-token replays. At step g (g=1..14) we feed
    // the id of the code we just sampled, gathered in graph from the
    // group's private embedding table, and read lm_head[g].
    for (int g = 1; g < n_acoustic; g++) {
        if (!code_predictor_replay(&step_graphs[(size_t) (g - 1)], backend, out->codes[(size_t) g], &logits)) {
            return false;
        }
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1 + g, &u_g);
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
