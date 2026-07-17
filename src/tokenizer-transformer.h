#pragma once
// tokenizer-transformer.h: 8-layer Qwen3-style local-causal transformer
// for the Qwen3-TTS 12Hz tokenizer decoder.
//
// Hidden size 512, head_dim 64, 16 query and 16 KV heads (no GQA), FFN
// intermediate 1024, RoPE NEOX style with theta 10000, RMSNorm eps 1e-5,
// sliding window 72 frames causal attention, LayerScale post-attention
// and post-MLP. SwiGLU MLP. No biases on q/k/v/o or gate/up/down
// projections. Top-level input_proj 1024 -> 512 (with bias) and
// output_proj 512 -> 1024 (with bias) bracket the transformer stack and
// connect to the latent stream of the codec.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "kv-cache.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define TOK_TRANS_MAX_LAYERS 16

struct QwenTransformerAttention {
    struct ggml_tensor * q_proj_w;  // [hidden, num_q_heads * head_dim]
    struct ggml_tensor * k_proj_w;  // [hidden, num_kv_heads * head_dim]
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;  // [num_q_heads * head_dim, hidden]
};

struct QwenTransformerMLP {
    struct ggml_tensor * gate_proj_w;  // [hidden, intermediate]
    struct ggml_tensor * up_proj_w;
    struct ggml_tensor * down_proj_w;  // [intermediate, hidden]
};

struct QwenTransformerLayer {
    struct ggml_tensor *     input_norm_w;  // RMSNorm gain [hidden]
    QwenTransformerAttention attn;
    struct ggml_tensor *     attn_scale;    // LayerScale per-channel [hidden]
    struct ggml_tensor *     post_attn_norm_w;
    QwenTransformerMLP       mlp;
    struct ggml_tensor *     mlp_scale;
};

struct QwenTokenizerTransformer {
    int   hidden_size;
    int   latent_dim;
    int   num_layers;
    int   num_attention_heads;
    int   num_kv_heads;
    int   head_dim;
    int   intermediate_size;
    int   sliding_window;
    float rope_theta;
    float rms_norm_eps;

    struct ggml_tensor * input_proj_w;   // [latent_dim, hidden]
    struct ggml_tensor * input_proj_b;   // [hidden]
    QwenTransformerLayer layers[TOK_TRANS_MAX_LAYERS];
    struct ggml_tensor * norm_w;         // [hidden]
    struct ggml_tensor * output_proj_w;  // [hidden, latent_dim]
    struct ggml_tensor * output_proj_b;  // [latent_dim]

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Read decoder hyperparameters from GGUF metadata, allocate every weight
// tensor on the backend, and bind tensor pointers in the struct. Returns
// true on success.
static bool tok_trans_load(QwenTokenizerTransformer * tr, const GGUFModel & gf, ggml_backend_t backend) {
    tr->hidden_size         = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.hidden_size");
    tr->latent_dim          = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.latent_dim");
    uint32_t num_layers     = gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.num_hidden_layers");
    tr->num_layers          = (int) num_layers;
    tr->num_attention_heads = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.num_attention_heads");
    tr->num_kv_heads        = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.num_key_value_heads");
    tr->head_dim            = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.head_dim");
    tr->intermediate_size   = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.intermediate_size");
    tr->sliding_window      = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.sliding_window");
    tr->rope_theta          = gf_get_f32(gf, "qwen3-tts-tokenizer.decoder.rope_theta");
    tr->rms_norm_eps        = gf_get_f32(gf, "qwen3-tts-tokenizer.decoder.rms_norm_eps");

    if (num_layers == 0 || num_layers > (uint32_t) TOK_TRANS_MAX_LAYERS) {
        fprintf(stderr, "[Transformer] FATAL: invalid layer count %u (max %d)\n", num_layers, TOK_TRANS_MAX_LAYERS);
        return false;
    }

    int n_tensors = 6                      // input_proj wb + norm + output_proj wb
                    + tr->num_layers * 11  // 2 norms + 4 attn + 3 mlp + 2 layer scales
                    + 4;                   // headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    tr->input_proj_w  = gf_load_tensor(&wctx, gf, "tok_dec.pre_tfm.input_proj.weight");
    tr->input_proj_b  = gf_load_tensor(&wctx, gf, "tok_dec.pre_tfm.input_proj.bias");
    tr->norm_w        = gf_load_tensor(&wctx, gf, "tok_dec.pre_tfm.norm.weight");
    tr->output_proj_w = gf_load_tensor(&wctx, gf, "tok_dec.pre_tfm.output_proj.weight");
    tr->output_proj_b = gf_load_tensor(&wctx, gf, "tok_dec.pre_tfm.output_proj.bias");

    for (int l = 0; l < tr->num_layers; l++) {
        QwenTransformerLayer & layer = tr->layers[l];
        char                   name[160];

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_norm.weight", l);
        layer.input_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_q.weight", l);
        layer.attn.q_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_k.weight", l);
        layer.attn.k_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_v.weight", l);
        layer.attn.v_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_output.weight", l);
        layer.attn.o_proj_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.attn_scale", l);
        layer.attn_scale = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.ffn_norm.weight", l);
        layer.post_attn_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.ffn_gate.weight", l);
        layer.mlp.gate_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.ffn_up.weight", l);
        layer.mlp.up_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.ffn_down.weight", l);
        layer.mlp.down_proj_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_dec.pre_tfm.blk.%d.ffn_scale", l);
        layer.mlp_scale = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[Transformer] FATAL: backend allocation failed\n");
        return false;
    }
    tr->weight_ctx = wctx.ctx;
    tr->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[Transformer] Loaded: %d layers, hidden %d, heads %d/%d, head_dim %d, "
            "FFN %d, RoPE theta %.0f, sliding window %d\n",
            tr->num_layers, tr->hidden_size, tr->num_attention_heads, tr->num_kv_heads, tr->head_dim,
            tr->intermediate_size, tr->rope_theta, tr->sliding_window);
    return true;
}

static void tok_trans_free(QwenTokenizerTransformer * tr) {
    if (tr->weight_buf) {
        ggml_backend_buffer_free(tr->weight_buf);
        tr->weight_buf = NULL;
    }
    if (tr->weight_ctx) {
        ggml_free(tr->weight_ctx);
        tr->weight_ctx = NULL;
    }
}

// Fill a [T, T] f32 mask with 0 where attention is allowed and -inf
// elsewhere. Storage is row-major with k (key index) on the fast axis :
// dst[q * T + k] is the additive bias for query q attending to key k.
// Causal sliding window: mask[k, q] = 0 if (k <= q AND q - k < window),
// else -inf.
static void tok_trans_build_causal_sliding_mask(int T, int sliding_window, std::vector<float> & dst) {
    dst.assign((size_t) T * (size_t) T, -INFINITY);
    for (int q = 0; q < T; q++) {
        int k_min = q - sliding_window + 1;
        if (k_min < 0) {
            k_min = 0;
        }
        for (int k = k_min; k <= q; k++) {
            dst[(size_t) q * (size_t) T + (size_t) k] = 0.0f;
        }
    }
}

static void tok_trans_build_positions(int T, std::vector<int32_t> & dst) {
    dst.resize((size_t) T);
    for (int i = 0; i < T; i++) {
        dst[i] = i;
    }
}

// One transformer layer: attention block then MLP block, both with
// pre-RMSNorm, post-LayerScale and residual connection.
static struct ggml_tensor * tok_trans_layer_forward(struct ggml_context *            ctx,
                                                    const QwenTokenizerTransformer * tr,
                                                    const QwenTransformerLayer &     layer,
                                                    struct ggml_tensor *             x,
                                                    struct ggml_tensor *             positions,
                                                    struct ggml_tensor *             mask,
                                                    int                              T) {
    int hidden    = tr->hidden_size;
    int n_q_heads = tr->num_attention_heads;
    int n_kv      = tr->num_kv_heads;
    int hd        = tr->head_dim;

    // Attention block: pre-RMSNorm + project Q/K/V + RoPE + scaled dot product
    // + softmax with causal sliding mask + V combine + o_proj.
    struct ggml_tensor * ln1 = ggml_rms_norm(ctx, x, tr->rms_norm_eps);
    ln1                      = ggml_mul(ctx, ln1, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, ln1);  // [n_q_heads*hd, T]
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, ln1);  // [n_kv*hd, T]
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, ln1);  // [n_kv*hd, T]

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);                         // [hd, n_q_heads, T]
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Permute to head-as-batch layout: [hd, T, n_heads]
    struct ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    struct ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor * v_p = ggml_permute(ctx, v, 1, 2, 0, 3);  // [T, n_kv, hd] for V mul_mat
    v_p                      = ggml_cont(ctx, v_p);

    // Scores: mul_mat(K, Q) -> [T_k, T_q, n_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);

    float scale = 1.0f / sqrtf((float) hd);
    scores      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    // Attention output: mul_mat(V_T, scores) -> [hd, T_q, n_heads]
    // V_T has T_k as ne[0], hd as ne[1], n_heads as ne[2].
    struct ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);

    // Permute back to [hd, n_heads, T] then reshape to [n_heads*hd, T]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);  // [hidden, T]

    // LayerScale + residual
    o = ggml_mul(ctx, o, layer.attn_scale);
    x = ggml_add(ctx, x, o);

    // MLP block: pre-RMSNorm + SwiGLU + LayerScale + residual.
    struct ggml_tensor * ln2 = ggml_rms_norm(ctx, x, tr->rms_norm_eps);
    ln2                      = ggml_mul(ctx, ln2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, ln2);  // [intermediate, T]
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, ln2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);  // [hidden, T]

    mlp = ggml_mul(ctx, mlp, layer.mlp_scale);
    x   = ggml_add(ctx, x, mlp);

    (void) hidden;
    return x;
}

// Full forward pass: input_proj, 8 layers, final norm, output_proj.
//
// x         : [latent_dim, T] f32
// positions: [T] i32
// mask      : [T, T] f32, additive (-inf where masked)
// returns   : [latent_dim, T] f32
static struct ggml_tensor * tok_trans_forward(struct ggml_context *            ctx,
                                              const QwenTokenizerTransformer * tr,
                                              struct ggml_tensor *             x,
                                              struct ggml_tensor *             positions,
                                              struct ggml_tensor *             mask) {
    int T = (int) x->ne[1];

    // input_proj: [latent_dim, T] -> [hidden, T]
    struct ggml_tensor * h = ggml_mul_mat(ctx, tr->input_proj_w, x);
    h                      = ggml_add(ctx, h, tr->input_proj_b);

    for (int l = 0; l < tr->num_layers; l++) {
        h = tok_trans_layer_forward(ctx, tr, tr->layers[l], h, positions, mask, T);
    }

    h = ggml_rms_norm(ctx, h, tr->rms_norm_eps);
    h = ggml_mul(ctx, h, tr->norm_w);

    // output_proj: [hidden, T] -> [latent_dim, T]
    h = ggml_mul_mat(ctx, tr->output_proj_w, h);
    h = ggml_add(ctx, h, tr->output_proj_b);

    return h;
}

// Streaming layer forward: the fresh K and V rows write into a
// persistent ring cache via set_rows at the slots carried by kv_rows,
// and the attention reads the whole ring with the mask killing every
// slot outside the sliding window. Ring slots hold RoPE rotated keys at
// absolute positions, so relative attention falls out as usual. The
// graph topology is constant across steps: pure CUDA graph replay.
static struct ggml_tensor * tok_trans_layer_forward_stream(struct ggml_context *            ctx,
                                                           struct ggml_cgraph *             gf,
                                                           const QwenTokenizerTransformer * tr,
                                                           const QwenTransformerLayer &     layer,
                                                           struct ggml_tensor *             x,
                                                           struct ggml_tensor *             positions,
                                                           struct ggml_tensor *             mask,
                                                           struct ggml_tensor *             kv_rows,
                                                           struct ggml_tensor *             k_cache,
                                                           struct ggml_tensor *             v_cache,
                                                           int                              T,
                                                           int                              ring) {
    int n_q_heads = tr->num_attention_heads;
    int n_kv      = tr->num_kv_heads;
    int hd        = tr->head_dim;

    struct ggml_tensor * ln1 = ggml_rms_norm(ctx, x, tr->rms_norm_eps);
    ln1                      = ggml_mul(ctx, ln1, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, ln1);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, ln1);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, ln1);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Ring write: [hd, T, n_kv] rows land at kv_rows, ids broadcast
    // across the head dim.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_perm, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_perm, kv_rows));

    struct ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, ring, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, ring, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    struct ggml_tensor * scores = ggml_mul_mat(ctx, k_full, q_p);  // [ring, T, n_q_heads]

    float scale = 1.0f / sqrtf((float) hd);
    scores      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    struct ggml_tensor * vt   = ggml_cont(ctx, ggml_transpose(ctx, v_full));  // [ring, hd, n_kv]
    struct ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);                // [hd, T, n_q_heads]

    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);

    o = ggml_mul(ctx, o, layer.attn_scale);
    x = ggml_add(ctx, x, o);

    struct ggml_tensor * ln2 = ggml_rms_norm(ctx, x, tr->rms_norm_eps);
    ln2                      = ggml_mul(ctx, ln2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, ln2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, ln2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    mlp = ggml_mul(ctx, mlp, layer.mlp_scale);
    x   = ggml_add(ctx, x, mlp);

    return x;
}

// Streaming transformer forward over a persistent KV ring. kv holds one
// [hd, ring, n_kv] pair per layer; kv_rows carries the ring slots the T
// fresh positions land in; mask is [ring, T] f32 with 0 on the slots
// inside the causal sliding window and neg inf elsewhere.
static struct ggml_tensor * tok_trans_forward_stream(struct ggml_context *            ctx,
                                                     struct ggml_cgraph *             gf,
                                                     const QwenTokenizerTransformer * tr,
                                                     struct ggml_tensor *             x,
                                                     struct ggml_tensor *             positions,
                                                     struct ggml_tensor *             mask,
                                                     struct ggml_tensor *             kv_rows,
                                                     KVCache *                        kv) {
    int T    = (int) x->ne[1];
    int ring = kv->max_seq_len;

    struct ggml_tensor * h = ggml_mul_mat(ctx, tr->input_proj_w, x);
    h                      = ggml_add(ctx, h, tr->input_proj_b);

    for (int l = 0; l < tr->num_layers; l++) {
        h = tok_trans_layer_forward_stream(ctx, gf, tr, tr->layers[l], h, positions, mask, kv_rows, kv->k[(size_t) l],
                                           kv->v[(size_t) l], T, ring);
    }

    h = ggml_rms_norm(ctx, h, tr->rms_norm_eps);
    h = ggml_mul(ctx, h, tr->norm_w);

    h = ggml_mul_mat(ctx, tr->output_proj_w, h);
    h = ggml_add(ctx, h, tr->output_proj_b);

    return h;
}

// Ring mask for the streaming path: [ring, T] f32, row q carries 0 on
// the ring slots holding positions inside [pos_q - window + 1, pos_q]
// and neg inf everywhere else, padded and future slots included.
static void tok_trans_build_stream_mask(int pos0, int T, int ring, int sliding_window, std::vector<float> & dst) {
    dst.assign((size_t) ring * (size_t) T, -INFINITY);
    for (int q = 0; q < T; q++) {
        int pos_q = pos0 + q;
        int k_min = pos_q - sliding_window + 1;
        if (k_min < 0) {
            k_min = 0;
        }
        for (int p = k_min; p <= pos_q; p++) {
            dst[(size_t) q * (size_t) ring + (size_t) (p % ring)] = 0.0f;
        }
    }
}
