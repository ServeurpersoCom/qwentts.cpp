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
#include "graph-arena.h"
#include "kv-cache.h"
#include "talker-decode-graph.h"
#include "talker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct TalkerForwardOutput {
    // Final hidden state for the last position [hidden] f32 (post final
    // norm). Filled only when read_hidden_host is set (dump paths); the
    // hot loop consumes the on device hidden bridge instead.
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

// Node budget for one talker graph. Counts approximate: ~38 ops per
// layer plus 2 cpy and 4 views for the KV write, IO tensors, final
// norm, codec_head and bisect dump branches. Sizes both the persistent
// arena in the pipeline and the graph allocated per forward.
static int talker_graph_max_nodes(int n_layers) {
    return 48 * n_layers + 256;
}

// Manual F32 attention chain. Used when use_flash_attn is false: GQA
// scaled dot product with explicit mul_mat / soft_max_ext / mul_mat,
// FP32 accumulators end to end. Mirrors the qwen3_attn_f32 helper in
// omnivoice.cpp/src/qwen3-enc.h. Inputs and output stay in the same
// layout flash_attn_ext expects: q [hd, T, n_q_heads], k/v [hd, T_full,
// n_kv], output [hd, n_q_heads, T]; the caller reshapes to
// [n_q_heads * hd, T] before o_proj exactly as on the FA path.
static struct ggml_tensor * talker_attn_f32(struct ggml_context * ctx,
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

// Build the per-layer block, KV cached. K and V for the T fresh
// positions get computed normally, then written into the cache at the
// rows carried by kv_rows via set_rows. The attention reads the fixed
// [0, n_kv_pad) window on the same dim, with the mask killing
// everything past the causal context. Returns the layer output
// [hidden, T].
//
// use_flash_attn picks between the fused ggml_flash_attn_ext kernel
// (GPU only, FP16 accumulation guarded with set_prec(F32)) and the
// manual F32 chain. clamp_fp16 inserts ggml_clamp(-65504, 65504) on V
// before attention and on the residual stream after the attention and
// MLP adds, protecting sub Ampere CUDA tensor cores that accumulate in
// FP16 from overflow.
static struct ggml_tensor * talker_layer_forward(struct ggml_context * ctx,
                                                 const TalkerWeights * tw,
                                                 const TalkerLayer &   layer,
                                                 struct ggml_tensor *  x,
                                                 struct ggml_tensor *  positions,
                                                 struct ggml_tensor *  mask,
                                                 struct ggml_tensor *  kv_rows,
                                                 struct ggml_tensor *  k_cache,
                                                 struct ggml_tensor *  v_cache,
                                                 int                   T,
                                                 int                   n_kv_pad,
                                                 bool                  use_flash_attn,
                                                 bool                  clamp_fp16,
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

    // Write the T fresh positions of K and V into the cache via
    // set_rows: kv_rows carries the destination positions as data, so
    // the graph topology stays identical across decode steps and the
    // captured CUDA graph replays without an update. K and V are
    // [hd, n_kv, T] at this point and the cache lives as
    // [hd, max_T, n_kv] so we permute to [hd, T, n_kv]; the row ids
    // broadcast across the n_kv head dim.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, T, n_kv]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [hd, T, n_kv]

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_perm, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_perm, kv_rows));

    // Read the [0, n_kv_pad) window for attention. n_kv_pad covers the
    // causal context and rounds it up so the view shape stays constant
    // across consecutive decode steps: the mask carries neg inf beyond
    // n_past + T and the cache buffer is zero initialized, so the padded
    // tail contributes nothing. The cache slice is already in the
    // [hd, n_kv_pad, n_kv] layout flash_attn_ext expects for K and V
    // (n_embd, n_kv, n_head_kv, ne3), passed directly as views.
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, n_kv_pad, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, n_kv_pad, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    // Q permute [hd, n_q_heads, T] -> [hd, T, n_q_heads]. flash_attn_ext
    // expects ne[1] = n_batch and ne[2] = n_head; the GQA broadcast
    // check ggml_can_mul_mat is n_head % n_head_kv == 0, matching K layout.
    // No cont: flash_attn_ext takes the view directly, like acestep does.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    // Clamp V before attention. Sub Ampere tensor cores accumulate in
    // FP16 and the V projection can overflow to inf, which corrupts
    // every subsequent attention. ggml_clamp is a no op on the F32
    // manual path but stays here to keep both branches numerically
    // aligned when the user opts into clamp_fp16.
    if (clamp_fp16) {
        v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
    }

    // Attention: fused flash kernel (FP16 accumulation, set_prec(F32)
    // promotes the softmax / V matmul accumulator back to F32) or
    // manual F32 chain. The fused path replaces a mul_mat + soft_max_ext
    // + mul_mat sequence that had a Vulkan bug in autoregressive decode
    // (T=1) where the KV cache view stride on dim 2 (= max_T * hd, non
    // contiguous with dim 1 of length T_full) caused the second mul_mat
    // to diverge silently. The manual path is the F32 reference, used
    // when use_flash_attn is false (CPU only backends, or explicit
    // request from the user).
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = talker_attn_f32(ctx, q_p, k_full, v_full, mask, scale);
    }

    // Flash attention output is [hd, n_q_heads, T], flatten heads for o_proj.
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);

    x = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    // MLP block: pre-norm + SwiGLU + residual
    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    // gate and up mul_mats adjacent to the GLU node: the CUDA backend
    // fuses the three into one kernel (ggml_cuda_should_fuse_mul_mat).
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    struct ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

// Prefill core: builds the graph in the caller owned arena, allocates
// through the sched, uploads the raw embedding, runs it and pulls out
// the last position logits. T tokens are appended to KV set `kv_set`
// starting at n_past. The last position hidden copies in graph into
// column `slot` of the caller owned persistent hidden_bridge tensor
// [hidden, max_batch] the code predictor prefill reads on device; the
// host copy in out->hidden_last fills only under read_hidden_host.
// When n_past == 0 and dump_dir is set, the bisect taps fire. use_fa /
// clamp_fp16 are forwarded as is to every layer. The decode hot path
// runs on the static batched graphs below instead.
static bool talker_forward_core(const TalkerWeights * tw,
                                KVCache *             kv,
                                int                   kv_set,
                                ggml_backend_sched_t  sched,
                                GraphArena *          arena,
                                struct ggml_tensor *  hidden_bridge,
                                const float *         input_embed,
                                int                   T,
                                int                   n_past,
                                bool                  use_flash_attn,
                                bool                  clamp_fp16,
                                bool                  read_hidden_host,
                                const char *          dump_dir,
                                TalkerForwardOutput * out) {
    const int hidden   = tw->hidden_size;
    const int n_layers = tw->num_hidden_layers;
    const int vocab    = tw->vocab_size;
    const int T_full   = n_past + T;

    // Attention window rounded up to 256 and clamped to the cache size.
    // Fixed shapes over spans of 256 decode steps let the CUDA graph
    // executable update in place (pointer patch) instead of rebuilding.
    // Ternary instead of std::min: windows.h min/max macros break the
    // latter in headers on MSVC.
    const int kv_pad_raw = (int) GGML_PAD(T_full, 256);
    const int n_kv_pad   = kv_pad_raw < kv->max_seq_len ? kv_pad_raw : kv->max_seq_len;

    const int             max_nodes = talker_graph_max_nodes(n_layers);
    struct ggml_context * gctx      = graph_arena_begin(arena);

    // IO tensors: positions and causal mask. The mask spans
    // [n_kv_pad, T]: for each fresh query q in [0, T) keys k in
    // [0, n_past + q] carry 0 and every other slot carries neg inf,
    // including the padded tail beyond T_full.
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_kv_pad, T);
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");

    // KV write positions as data: identical topology at every step,
    // pure CUDA graph replay across the decode loop.
    struct ggml_tensor * rows_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I64, T);
    ggml_set_name(rows_in, "kv_rows");
    ggml_set_input(rows_in);

    // Input: the raw prefill embedding uploaded from host.
    struct ggml_tensor * x_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, T);
    ggml_set_input(x_in);
    ggml_set_name(x_in, "input_embed");

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);

    // Build the layer stack. Bisect taps fire on prefill only.
    const bool                        record_taps = (dump_dir != NULL) && (n_past == 0);
    struct ggml_tensor *              h           = x_in;
    std::vector<struct ggml_tensor *> taps(TALKER_N_BISECT_LAYERS, NULL);
    for (int l = 0; l < n_layers; l++) {
        h = talker_layer_forward(gctx, tw, tw->layers[(size_t) l], h, pos_in, mask_in, rows_in,
                                 kv_cache_k(kv, kv_set, l), kv_cache_v(kv, kv_set, l), T, n_kv_pad, use_flash_attn,
                                 clamp_fp16, gf);
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
    if (record_taps || read_hidden_host) {
        ggml_set_output(h_final);
    }

    // Bridge: the last position hidden copies on device into column
    // `kv_set` of the persistent [hidden, max_batch] tensor. Constant
    // destination address across steps, so the decode graph topology
    // stays replayable.
    struct ggml_tensor * h_last =
        ggml_view_1d(gctx, h_final, hidden, (size_t) (T - 1) * (size_t) hidden * sizeof(float));
    struct ggml_tensor * bridge_col = ggml_view_1d(gctx, hidden_bridge, hidden, (size_t) kv_set * hidden_bridge->nb[1]);
    struct ggml_tensor * bridge_cpy = ggml_cpy(gctx, h_last, bridge_col);

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
    ggml_build_forward_expand(gf, bridge_cpy);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "[TalkerForward] FATAL: graph allocation failed\n");
        ggml_backend_sched_reset(sched);
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

        std::vector<int64_t> rows((size_t) T);
        for (int i = 0; i < T; i++) {
            rows[(size_t) i] = (int64_t) (n_past + i);
        }
        ggml_backend_tensor_set(rows_in, rows.data(), 0, (size_t) T * sizeof(int64_t));
    }

    // Causal mask: 0 where k <= n_past + q, neg inf otherwise. Stored
    // row major [T_q, n_kv_pad] with n_kv_pad as the fast axis (ne[0]).
    // F16 dtype matches the ggml_flash_attn_ext convention used by the
    // attention path.
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

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[TalkerForward] FATAL: graph compute failed\n");
        ggml_backend_sched_reset(sched);
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

    // Pull the last position logits, the only per step readback on the
    // hot path. The hidden row comes back to host only for dumps.
    out->hidden = hidden;
    out->vocab  = vocab;
    out->logits_last.assign((size_t) vocab, 0.0f);
    {
        size_t row_bytes = (size_t) vocab * sizeof(float);
        ggml_backend_tensor_get(logits, out->logits_last.data(), (size_t) (T - 1) * row_bytes, row_bytes);
    }
    if (read_hidden_host) {
        out->hidden_last.assign((size_t) hidden, 0.0f);
        size_t hrow_bytes = (size_t) hidden * sizeof(float);
        ggml_backend_tensor_get(h_final, out->hidden_last.data(), (size_t) (T - 1) * hrow_bytes, hrow_bytes);
    }

    if (record_taps) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        debug_dump_1d(&d, "talker-logits-prefill", out->logits_last.data(), vocab);
    }

    // Advance the cache write head. The graph already executed the cpy
    // nodes so positions [n_past, n_past + T) are now populated. The
    // arena and the sched allocation persist into the next forward.
    kv->cur_len[(size_t) kv_set] = T_full;
    return true;
}

// Prefill: reset KV set `kv_set` and write T_ctx positions in one
// shot. input_embed is [T, hidden] f32 row-major. dump_dir may be NULL.
static bool talker_forward_prefill(const TalkerWeights * tw,
                                   KVCache *             kv,
                                   int                   kv_set,
                                   ggml_backend_sched_t  sched,
                                   GraphArena *          arena,
                                   struct ggml_tensor *  hidden_bridge,
                                   const float *         input_embed,
                                   int                   T,
                                   bool                  use_flash_attn,
                                   bool                  clamp_fp16,
                                   const char *          dump_dir,
                                   TalkerForwardOutput * out) {
    kv_cache_reset(kv, kv_set);
    if (T > kv->max_seq_len) {
        fprintf(stderr, "[TalkerForward] FATAL: prefill T=%d exceeds cache max_seq_len=%d\n", T, kv->max_seq_len);
        return false;
    }
    return talker_forward_core(tw, kv, kv_set, sched, arena, hidden_bridge, input_embed, T, 0, use_flash_attn,
                               clamp_fp16, dump_dir != NULL, dump_dir, out);
}

// Build the batched per-layer block over KV sets [0, N), one fresh
// token per set. Projections, norms, RoPE and the MLP run on [*, N]
// with per column math identical to the single sequence layer. The
// attention is per set: fresh K,V write into the 4D cache at the rows
// carried by kv_rows via set_rows, the read views the padded causal
// window with ne3 = N, and the mask kills everything past each set's
// own context. Returns the layer output [hidden, N].
static struct ggml_tensor * talker_layer_forward_batch(struct ggml_context * ctx,
                                                       const TalkerWeights * tw,
                                                       const TalkerLayer &   layer,
                                                       struct ggml_tensor *  x,
                                                       struct ggml_tensor *  positions,
                                                       struct ggml_tensor *  mask,
                                                       struct ggml_tensor *  kv_rows,
                                                       struct ggml_tensor *  k4,
                                                       struct ggml_tensor *  v4,
                                                       int                   N,
                                                       int                   n_kv_pad,
                                                       bool                  use_flash_attn,
                                                       bool                  clamp_fp16,
                                                       struct ggml_cgraph *  gf) {
    const int   n_q_heads = tw->num_attention_heads;
    const int   n_kv      = tw->num_key_value_heads;
    const int   hd        = tw->head_dim;
    const float eps       = tw->rms_norm_eps;

    // Pre-norm
    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    // Q/K/V projections, batched over the N columns
    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);  // [n_q_heads*hd, N]
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);  // [n_kv*hd, N]
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);  // [n_kv*hd, N]

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, N);                       // [hd, n_q_heads, N]
    k = ggml_reshape_3d(ctx, k, hd, n_kv, N);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, N);

    // Per-head QK-norm: RMS over hd, then multiply by [hd] gain.
    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    // RoPE NEOX: positions [N] map to dim 2 of [hd, heads, N], one
    // absolute position per set.
    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Write the fresh K,V into the 4D cache via set_rows over the N
    // consecutive sets: src reshapes [hd, n_kv, N] to [hd, 1, n_kv, N]
    // (same layout) and kv_rows [1, 1, N] carries one destination row
    // per set, broadcast across the n_kv head dim. The row ids travel
    // as data so every step keeps an identical topology and the
    // captured CUDA graph replays without an update.
    struct ggml_tensor * k_sets = ggml_view_4d(ctx, k4, hd, k4->ne[1], n_kv, N, k4->nb[1], k4->nb[2], k4->nb[3], 0);
    struct ggml_tensor * v_sets = ggml_view_4d(ctx, v4, hd, v4->ne[1], n_kv, N, v4->nb[1], v4->nb[2], v4->nb[3], 0);
    struct ggml_tensor * k_new  = ggml_reshape_4d(ctx, k, hd, 1, n_kv, N);
    struct ggml_tensor * v_new  = ggml_reshape_4d(ctx, v, hd, 1, n_kv, N);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_sets, k_new, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_sets, v_new, kv_rows));

    // Batched read: [hd, n_kv_pad, n_kv, N] views of the 4D cache. The
    // window covers every set's causal context and rounds up so the
    // shape stays constant across 256 consecutive decode steps; the
    // mask carries neg inf past each set's own context and the cache
    // buffer is zero initialized, so the padded tail contributes
    // nothing.
    struct ggml_tensor * k_batch = ggml_view_4d(ctx, k4, hd, n_kv_pad, n_kv, N, k4->nb[1], k4->nb[2], k4->nb[3], 0);
    struct ggml_tensor * v_batch = ggml_view_4d(ctx, v4, hd, n_kv_pad, n_kv, N, v4->nb[1], v4->nb[2], v4->nb[3], 0);

    // Q: [hd, n_q_heads, N] -> [hd, 1, n_q_heads, N] (n_batch=1, ne3=N
    // for batched flash_attn).
    struct ggml_tensor * q4 = ggml_reshape_4d(ctx, q, hd, 1, n_q_heads, N);

    // Clamp V before attention, same rationale as the prefill block:
    // sub Ampere CUDA tensor cores accumulate in FP16 and a V overflow
    // corrupts every subsequent attention.
    if (clamp_fp16) {
        v_batch = ggml_clamp(ctx, v_batch, -65504.0f, 65504.0f);
    }

    // Attention: fused flash kernel (set_prec(F32) promotes the
    // accumulator) or the manual F32 chain, which broadcasts its
    // mul_mat over dims 2 and 3 so the same helper covers 4D.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q4, k_batch, v_batch, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = talker_attn_f32(ctx, q4, k_batch, v_batch, mask, scale);
    }

    // Output [hd, n_q_heads, 1, N] -> [n_q_heads*hd, N], flatten heads
    // for o_proj.
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, N);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);

    x = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    // MLP block: pre-norm + SwiGLU + residual
    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    // gate and up mul_mats adjacent to the GLU node: the CUDA backend
    // fuses the three into one kernel (ggml_cuda_should_fuse_mul_mat).
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    struct ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

// Build one static batched decode graph for the given attention window
// and batch width N. Each slot's previous frame code ids gather and
// sum in graph with its overlay row on top; the batch of last position
// hiddens copies into columns [0, N) of hidden_bridge and the codec
// head logits [vocab, N] are the single output. Positions, kv rows,
// and masks stay plain inputs re-uploaded before every replay since
// each slot's n_past moves every step.
static bool talker_decode_graph_build(const TalkerWeights *        tw,
                                      KVCache *                    kv,
                                      ggml_backend_t               backend,
                                      struct ggml_tensor *         hidden_bridge,
                                      struct ggml_tensor * const * acoustic_embd,
                                      int                          n_acoustic,
                                      int                          n_kv_pad,
                                      int                          N,
                                      bool                         use_flash_attn,
                                      bool                         clamp_fp16,
                                      TalkerDecodeGraph *          tg) {
    const int hidden   = tw->hidden_size;
    const int n_layers = tw->num_hidden_layers;

    const int    max_nodes = talker_graph_max_nodes(n_layers);
    const size_t bytes =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);
    struct ggml_init_params gp = { bytes, NULL, true };
    tg->ctx                    = ggml_init(gp);
    if (!tg->ctx) {
        fprintf(stderr, "[TalkerForward] FATAL: decode graph ctx allocation failed\n");
        return false;
    }
    struct ggml_context * gctx = tg->ctx;

    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, N);
    struct ggml_tensor * mask_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_kv_pad, 1, 1, N);
    struct ggml_tensor * rows_in = ggml_new_tensor_3d(gctx, GGML_TYPE_I64, 1, 1, N);
    struct ggml_tensor * ids_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (1 + n_acoustic) * N);
    struct ggml_tensor * overlay = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, N);
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");
    ggml_set_name(rows_in, "kv_rows");
    ggml_set_name(ids_in, "frame_code_ids");
    ggml_set_name(overlay, "overlay_rows");
    ggml_set_input(pos_in);
    ggml_set_input(mask_in);
    ggml_set_input(rows_in);
    ggml_set_input(ids_in);
    ggml_set_input(overlay);

    // ids_in is group major: entry g * N + slot. Each group slices a
    // contiguous [N] view and gathers its own table, summed across the
    // 16 codebooks, plus the per slot overlay row.
    struct ggml_tensor * id0  = ggml_view_1d(gctx, ids_in, N, 0);
    struct ggml_tensor * x_in = ggml_get_rows(gctx, tw->codec_embedding, id0);
    for (int g = 0; g < n_acoustic; g++) {
        struct ggml_tensor * idg = ggml_view_1d(gctx, ids_in, N, (size_t) (g + 1) * (size_t) N * sizeof(int32_t));
        x_in                     = ggml_add(gctx, x_in, ggml_get_rows(gctx, acoustic_embd[g], idg));
    }
    x_in = ggml_add(gctx, x_in, overlay);
    ggml_set_name(x_in, "input_embed");

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);

    struct ggml_tensor * h = x_in;
    for (int l = 0; l < n_layers; l++) {
        h = talker_layer_forward_batch(gctx, tw, tw->layers[(size_t) l], h, pos_in, mask_in, rows_in,
                                       kv->k4[(size_t) l], kv->v4[(size_t) l], N, n_kv_pad, use_flash_attn, clamp_fp16,
                                       gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, tw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, tw->norm_w);
    ggml_set_name(h_final, "hidden_final");

    struct ggml_tensor * bridge_cols = ggml_view_2d(gctx, hidden_bridge, hidden, N, hidden_bridge->nb[1], 0);
    struct ggml_tensor * bridge_cpy  = ggml_cpy(gctx, h_final, bridge_cols);

    struct ggml_tensor * logits = ggml_mul_mat(gctx, tw->codec_head_w, h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, bridge_cpy);

    tg->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!tg->galloc || !ggml_gallocr_alloc_graph(tg->galloc, gf)) {
        fprintf(stderr, "[TalkerForward] FATAL: decode graph allocation failed\n");
        talker_decode_graph_free(tg);
        return false;
    }

    tg->gf      = gf;
    tg->ids_in  = ids_in;
    tg->overlay = overlay;
    tg->pos_in  = pos_in;
    tg->rows_in = rows_in;
    tg->mask_in = mask_in;
    tg->logits  = logits;
    tg->mask.resize((size_t) n_kv_pad * (size_t) N);
    tg->pos_data.resize((size_t) N);
    tg->rows_data.resize((size_t) N);
    tg->n_kv_pad = n_kv_pad;
    tg->N        = N;
    return true;
}

// Batched decode: append one position per KV set over the static graph
// of the current window class and batch width, built lazily on the
// first step entering the (span, N) pair. frame_ids holds the previous
// frame codes of every slot, group major: entry g * N + slot.
// acoustic_embd holds the 15 group tables owned by the code predictor,
// overlays the [hidden, N] trailing text / pad rows summed on top. The
// window covers max over sets of (cur_len + 1); caller ensures every
// cur_len + 1 <= max_seq_len holds by cache sizing. Logits fill
// out->logits_last as [vocab, N] column blocks; read_hidden_host pulls
// the bridge back as [hidden, N] for dump paths.
static bool talker_forward_decode(const TalkerWeights *            tw,
                                  KVCache *                        kv,
                                  ggml_backend_t                   backend,
                                  std::vector<TalkerDecodeGraph> & graphs,
                                  struct ggml_tensor *             hidden_bridge,
                                  const int32_t *                  frame_ids,
                                  struct ggml_tensor * const *     acoustic_embd,
                                  int                              n_acoustic,
                                  const float *                    overlays,
                                  int                              N,
                                  bool                             use_flash_attn,
                                  bool                             clamp_fp16,
                                  bool                             read_hidden_host,
                                  TalkerForwardOutput *            out) {
    int max_len = 0;
    for (int i = 0; i < N; i++) {
        const int len = kv->cur_len[(size_t) i] + 1;
        if (len > kv->max_seq_len) {
            fprintf(stderr, "[TalkerForward] FATAL: decode would overflow cache (%d > %d, set %d)\n", len,
                    kv->max_seq_len, i);
            return false;
        }
        if (len > max_len) {
            max_len = len;
        }
    }
    const int kv_pad_raw = (int) GGML_PAD(max_len, 256);
    const int n_kv_pad   = kv_pad_raw < kv->max_seq_len ? kv_pad_raw : kv->max_seq_len;

    TalkerDecodeGraph * tg = &graphs[(size_t) ((n_kv_pad + 255) / 256 - 1)];
    if (tg->ctx && tg->N != N) {
        talker_decode_graph_free(tg);
    }
    if (!tg->ctx && !talker_decode_graph_build(tw, kv, backend, hidden_bridge, acoustic_embd, n_acoustic, n_kv_pad, N,
                                               use_flash_attn, clamp_fp16, tg)) {
        return false;
    }

    ggml_backend_tensor_set(tg->ids_in, frame_ids, 0, (size_t) (1 + n_acoustic) * (size_t) N * sizeof(int32_t));
    ggml_backend_tensor_set(tg->overlay, overlays, 0, (size_t) tw->hidden_size * (size_t) N * sizeof(float));

    for (int i = 0; i < N; i++) {
        tg->pos_data[(size_t) i]  = kv->cur_len[(size_t) i];
        tg->rows_data[(size_t) i] = (int64_t) kv->cur_len[(size_t) i];
    }
    ggml_backend_tensor_set(tg->pos_in, tg->pos_data.data(), 0, (size_t) N * sizeof(int32_t));
    ggml_backend_tensor_set(tg->rows_in, tg->rows_data.data(), 0, (size_t) N * sizeof(int64_t));

    // Causal masks: for slot i keys [0, cur_len[i]] carry 0, the rest
    // of the padded window neg inf.
    {
        std::vector<ggml_fp16_t> & mask    = tg->mask;
        const ggml_fp16_t          zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t          neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < N; i++) {
            const int n_past = kv->cur_len[(size_t) i];
            for (int j = 0; j < n_kv_pad; j++) {
                mask[(size_t) i * (size_t) n_kv_pad + (size_t) j] = j <= n_past ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(tg->mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(backend, tg->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[TalkerForward] FATAL: decode graph compute failed\n");
        return false;
    }

    out->hidden = tw->hidden_size;
    out->vocab  = tw->vocab_size;
    out->logits_last.assign((size_t) tw->vocab_size * (size_t) N, 0.0f);
    ggml_backend_tensor_get(tg->logits, out->logits_last.data(), 0,
                            (size_t) tw->vocab_size * (size_t) N * sizeof(float));
    if (read_hidden_host) {
        out->hidden_last.assign((size_t) tw->hidden_size * (size_t) N, 0.0f);
        ggml_backend_tensor_get(hidden_bridge, out->hidden_last.data(), 0,
                                (size_t) tw->hidden_size * (size_t) N * sizeof(float));
    }

    for (int i = 0; i < N; i++) {
        kv->cur_len[(size_t) i]++;
    }
    return true;
}
