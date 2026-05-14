#pragma once
// talker-forward.h: prefill + decode forwards of the Talker LM, KV
// cached. Eager prefill graph for the Talker LM.
//
// Both entry points run the same 28-layer Qwen3 decoder stack with
// multimodal RoPE collapsed to 1D NEOX, GQA attention with per-head
// QK-norm, and SwiGLU MLP. The final hidden state is RMS-normalised
// and projected through codec_head to produce codebook 0 logits over a
// 3072-entry vocab.
//
//   talker_forward_prefill: feeds a [T_ctx, hidden] input embedding,
//   rewinds the KV cache to 0 and writes T_ctx positions into it. Used
//   once per utterance at the start of generation, and re-runnable for
//   bisect dumps. Optional dump_dir captures L0/7/14/21/27 hidden taps
//   plus the final hidden and logits.
//
//   talker_forward_decode: feeds a single [1, hidden] embedding,
//   appends one position to the cache at index kv->cur_len, attends to
//   the [0, cur_len+1) window. Called once per generated frame after
//   the predictor has produced its 15 acoustic codes and the loop has
//   summed the codec embeddings into next_emb.
//
// Mirrors Qwen3TTSTalkerDecoderLayer for TTS-only operation:
//   pre-norm, GQA attention with per-head QK-norm, mrope collapsed to
//   1D NEOX (since the three multimodal axes share position ids in TTS
//   mode), SwiGLU MLP, two residuals, repeated 28 times, then final
//   RMSNorm and codec_head.
//
// Tensor shapes follow ggml row-major convention: ne[0] is the fastest
// axis. Our input embedding lives as [hidden, T] inside the graph and
// the loader feeds it from a [T, hidden] f32 row-major host buffer
// (which becomes [hidden, T] in ggml after a 2d view because rows on
// the host are contiguous along the hidden axis).
//
// The Python reference uses pure causal attention with no sliding
// window, so the cache is a plain causal ring.

#include "backend.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "talker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct TalkerForwardOutput {
    // Final hidden state for the last position [hidden] f32 (post final norm).
    std::vector<float> hidden_last;

    // Codec head logits for the last position [vocab] f32.
    std::vector<float> logits_last;

    int hidden;
    int vocab;
};

// Bisect layers dumped when a dump_dir is set. Match the Python hook list
// in tests/debug-tts-cossim.py: 0, 7, 14, 21, 27.
static const int TALKER_BISECT_LAYERS[] = { 0, 7, 14, 21, 27 };
static const int TALKER_N_BISECT_LAYERS = (int) (sizeof(TALKER_BISECT_LAYERS) / sizeof(TALKER_BISECT_LAYERS[0]));

static bool talker_is_bisect_layer(int l) {
    for (int i = 0; i < TALKER_N_BISECT_LAYERS; i++) {
        if (TALKER_BISECT_LAYERS[i] == l) {
            return true;
        }
    }
    return false;
}

// Build the per-layer block, KV cached. K and V for the T fresh
// positions get computed normally, then written into the cache at
// [n_past, n_past+T) on dim 1. The attention reads the contiguous slice
// [0, n_past+T) on the same dim, which covers the full causal context
// in one tensor view. Returns the layer output [hidden, T].
static struct ggml_tensor * talker_layer_forward(struct ggml_context * ctx,
                                                 const TalkerWeights * tw,
                                                 const TalkerLayer &   layer,
                                                 struct ggml_tensor *  x,
                                                 struct ggml_tensor *  positions,
                                                 struct ggml_tensor *  mask,
                                                 struct ggml_tensor *  k_cache,
                                                 struct ggml_tensor *  v_cache,
                                                 int                   n_past,
                                                 int                   T,
                                                 struct ggml_cgraph *  gf) {
    const int   n_q_heads = tw->num_attention_heads;
    const int   n_kv      = tw->num_key_value_heads;
    const int   hd        = tw->head_dim;
    const float eps       = tw->rms_norm_eps;

    // Pre-norm
    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    // Q/K/V projections
    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);  // [n_q_heads*hd, T]
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);  // [n_kv*hd, T]
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);  // [n_kv*hd, T]

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);                       // [hd, n_q_heads, T]
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    // Per-head QK-norm: RMS over hd, then multiply by [hd] gain. The
    // norm operates on ne[0] = hd, identical layout for q (16 heads) and
    // k (8 heads), so the same code path covers both.
    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    // RoPE NEOX (half-split). In TTS-only mode the three mrope axes share
    // position ids, so the multimodal interleaved cos/sin collapses to
    // plain 1D rotate_half with the same freq base.
    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Write the T fresh positions of K and V into the cache. K and V are
    // [hd, n_kv, T] at this point and the cache lives as [hd, max_T, n_kv]
    // so we permute to [hd, T, n_kv] before the cpy. The destination view
    // covers [hd, T, n_kv] at dim 1 offset n_past * nb1.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, T, n_kv]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [hd, T, n_kv]

    size_t k_off = (size_t) n_past * k_cache->nb[1];
    size_t v_off = (size_t) n_past * v_cache->nb[1];

    struct ggml_tensor * k_dst = ggml_view_3d(ctx, k_cache, hd, T, n_kv, k_cache->nb[1], k_cache->nb[2], k_off);
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, v_cache, hd, T, n_kv, v_cache->nb[1], v_cache->nb[2], v_off);

    struct ggml_tensor * k_cpy = ggml_cpy(ctx, k_perm, k_dst);
    struct ggml_tensor * v_cpy = ggml_cpy(ctx, v_perm, v_dst);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    // Read the [0, n_past + T) window for attention. The cache slice is
    // already in the [hd, T_full, n_kv] layout flash_attn_ext expects for
    // K and V (n_embd, n_kv, n_head_kv, ne3), passed directly as views.
    const int            T_full = n_past + T;
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, T_full, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, T_full, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    // Q permute [hd, n_q_heads, T] -> [hd, T, n_q_heads]. flash_attn_ext
    // expects ne[1] = n_batch and ne[2] = n_head; the GQA broadcast
    // check ggml_can_mul_mat is n_head % n_head_kv == 0, matching K layout.
    // No cont: flash_attn_ext takes the view directly, like acestep does.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    // Fused flash attention. The manual mul_mat + soft_max_ext + mul_mat
    // path it replaces had a Vulkan bug in autoregressive decode (T=1)
    // where the KV cache view stride on dim 2 (= max_T * hd, non
    // contiguous with dim 1 of length T_full) caused the second mul_mat
    // to diverge silently. Flash attention has a backend-tested kernel
    // on every target (CPU, CUDA, Vulkan), matches the working acestep
    // qw3lm_build_attn pattern.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn  = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    // Flash attention output is [hd, n_q_heads, T], flatten heads for o_proj.
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);

    x = ggml_add(ctx, x, o);

    // MLP block: pre-norm + SwiGLU + residual
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

// Shared core that builds the graph, allocates, uploads inputs, runs
// it and pulls out the last position hidden + logits. T tokens are
// appended to the cache starting at n_past. When n_past == 0 and
// dump_dir is set, the bisect taps fire on the prefill path.
static bool talker_forward_core(const TalkerWeights * tw,
                                KVCache *             kv,
                                ggml_backend_sched_t  sched,
                                const float *         input_embed,
                                int                   T,
                                int                   n_past,
                                const char *          dump_dir,
                                TalkerForwardOutput * out) {
    const int hidden   = tw->hidden_size;
    const int n_layers = tw->num_hidden_layers;
    const int vocab    = tw->vocab_size;
    const int T_full   = n_past + T;

    // Dedicated context for graph + IO tensors. Counts approximate:
    //   per layer: ~38 ops (added 2 cpy + 4 views per layer for KV)
    //   IO        : 4 tensors (input embed, positions, mask, output norm)
    //   final     : norm + codec_head + dump branches
    const int    max_nodes         = 48 * n_layers + 256;
    const size_t graph_arena_bytes = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);

    struct ggml_init_params gparams = {
        graph_arena_bytes,
        NULL,
        true,
    };
    struct ggml_context * gctx = ggml_init(gparams);
    if (!gctx) {
        fprintf(stderr, "[TalkerForward] FATAL: ggml_init failed\n");
        return false;
    }

    // IO tensors: input embedding, positions, causal mask. The mask
    // spans [T_full, T]: for each fresh query q in [0, T) we allow
    // keys k in [0, n_past + q]. In the decode case (T=1) this is a
    // single row of zeros of length T_full.
    struct ggml_tensor * x_in    = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, T);
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_full, T);
    ggml_set_name(x_in, "input_embed");
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);

    // Build the layer stack. Bisect taps fire on prefill only.
    const bool                        record_taps = (dump_dir != NULL) && (n_past == 0);
    struct ggml_tensor *              h           = x_in;
    std::vector<struct ggml_tensor *> taps(TALKER_N_BISECT_LAYERS, NULL);
    for (int l = 0; l < n_layers; l++) {
        h = talker_layer_forward(gctx, tw, tw->layers[(size_t) l], h, pos_in, mask_in, kv->k[(size_t) l],
                                 kv->v[(size_t) l], n_past, T, gf);
        if (record_taps && talker_is_bisect_layer(l)) {
            for (int i = 0; i < TALKER_N_BISECT_LAYERS; i++) {
                if (TALKER_BISECT_LAYERS[i] == l) {
                    char tap_name[64];
                    snprintf(tap_name, sizeof(tap_name), "tap_l%d", l);
                    ggml_set_name(h, tap_name);
                    ggml_set_output(h);
                    taps[(size_t) i] = h;
                    break;
                }
            }
        }
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, tw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, tw->norm_w);
    ggml_set_name(h_final, "hidden_final");
    ggml_set_output(h_final);

    // codec_head: [hidden, vocab]. ggml_mul_mat returns [vocab, T].
    struct ggml_tensor * logits = ggml_mul_mat(gctx, tw->codec_head_w, h_final);
    ggml_set_name(logits, "logits");

    if (record_taps) {
        for (int i = 0; i < TALKER_N_BISECT_LAYERS; i++) {
            if (taps[(size_t) i]) {
                ggml_build_forward_expand(gf, taps[(size_t) i]);
            }
        }
        ggml_build_forward_expand(gf, h_final);
    }
    ggml_build_forward_expand(gf, logits);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "[TalkerForward] FATAL: graph allocation failed\n");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }

    // Upload input embedding (host [T, hidden] -> ggml [hidden, T]).
    ggml_backend_tensor_set(x_in, input_embed, 0, (size_t) T * (size_t) hidden * sizeof(float));

    // Positions: n_past .. n_past + T - 1
    {
        std::vector<int32_t> pos((size_t) T);
        for (int i = 0; i < T; i++) {
            pos[(size_t) i] = n_past + i;
        }
        ggml_backend_tensor_set(pos_in, pos.data(), 0, (size_t) T * sizeof(int32_t));
    }

    // Causal mask: 0 where k <= n_past + q, -inf otherwise. Stored
    // row-major [T_q, T_k] with T_k as the fast axis (ne[0]). F16 dtype
    // matches ggml_flash_attn_ext convention used by the attention path.
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
        fprintf(stderr, "[TalkerForward] FATAL: graph compute failed\n");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }

    // Bisect dumps: pull each tap [hidden, T] back to host as [T, hidden].
    if (record_taps) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        std::vector<float> buf((size_t) T * (size_t) hidden);
        for (int i = 0; i < TALKER_N_BISECT_LAYERS; i++) {
            if (!taps[(size_t) i]) {
                continue;
            }
            ggml_backend_tensor_get(taps[(size_t) i], buf.data(), 0, buf.size() * sizeof(float));
            char name[64];
            snprintf(name, sizeof(name), "talker-hidden-prefill-l%d", TALKER_BISECT_LAYERS[i]);
            debug_dump_2d(&d, name, buf.data(), T, hidden);
        }
        ggml_backend_tensor_get(h_final, buf.data(), 0, buf.size() * sizeof(float));
        debug_dump_2d(&d, "talker-hidden-prefill-final", buf.data(), T, hidden);
    }

    // Pull the last position: final hidden + logits
    out->hidden = hidden;
    out->vocab  = vocab;
    out->hidden_last.assign((size_t) hidden, 0.0f);
    out->logits_last.assign((size_t) vocab, 0.0f);
    {
        size_t row_bytes = (size_t) vocab * sizeof(float);
        ggml_backend_tensor_get(logits, out->logits_last.data(), (size_t) (T - 1) * row_bytes, row_bytes);

        size_t hrow_bytes = (size_t) hidden * sizeof(float);
        ggml_backend_tensor_get(h_final, out->hidden_last.data(), (size_t) (T - 1) * hrow_bytes, hrow_bytes);
    }

    if (record_taps) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        debug_dump_1d(&d, "talker-logits-prefill", out->logits_last.data(), vocab);
    }

    // Advance the cache write head. The graph already executed the cpy
    // nodes so positions [n_past, n_past + T) are now populated.
    kv->cur_len = T_full;

    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return true;
}

// Prefill: reset the cache and write T_ctx positions in one shot.
// input_embed is [T, hidden] f32 row-major. dump_dir may be NULL.
static bool talker_forward_prefill(const TalkerWeights * tw,
                                   KVCache *             kv,
                                   ggml_backend_sched_t  sched,
                                   const float *         input_embed,
                                   int                   T,
                                   const char *          dump_dir,
                                   TalkerForwardOutput * out) {
    kv_cache_reset(kv);
    if (T > kv->max_seq_len) {
        fprintf(stderr, "[TalkerForward] FATAL: prefill T=%d exceeds cache max_seq_len=%d\n", T, kv->max_seq_len);
        return false;
    }
    return talker_forward_core(tw, kv, sched, input_embed, T, 0, dump_dir, out);
}

// Decode: feed exactly one embedding and append one position to the
// cache. Reads positions [0, kv->cur_len + 1). Caller is responsible
// for ensuring kv->cur_len + 1 <= kv->max_seq_len.
static bool talker_forward_decode(const TalkerWeights * tw,
                                  KVCache *             kv,
                                  ggml_backend_sched_t  sched,
                                  const float *         input_embed_1,
                                  TalkerForwardOutput * out) {
    if (kv->cur_len + 1 > kv->max_seq_len) {
        fprintf(stderr, "[TalkerForward] FATAL: decode would overflow cache (%d + 1 > %d)\n", kv->cur_len,
                kv->max_seq_len);
        return false;
    }
    return talker_forward_core(tw, kv, sched, input_embed_1, 1, kv->cur_len, NULL, out);
}
