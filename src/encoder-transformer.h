#pragma once
// encoder-transformer.h: 8-layer Mimi-style transformer for the Qwen3-TTS
// encoder. Operates on the SEANet output stream at 25 Hz, 512 channels.
//
// Differs from the Qwen3-style decoder transformer on several points:
//   - LayerNorm with bias (not RMSNorm)
//   - Plain MLP fc1 -> GELU -> fc2 (not SwiGLU)
//   - No biases on q/k/v/o projections
//   - 8 attention heads instead of 16
//   - intermediate_size 2048 instead of 1024
//   - Pure causal attention, full T x T mask (the upstream config carries
//     a sliding_window field but Mimi never applies it)
//   - No top-level input_proj / output_proj brackets: the SEANet output
//     already has hidden_size channels
//
// Common with the decoder side:
//   - RoPE NEOX style with theta 10000
//   - LayerScale per channel post-attention and post-MLP
//   - Pre-norm residual on both attention and MLP

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define ENC_TRANS_MAX_LAYERS 16

struct QwenEncoderTransformerLayer {
    // Pre-attention LayerNorm
    struct ggml_tensor * input_norm_w;
    struct ggml_tensor * input_norm_b;
    // Attention projections, no bias
    struct ggml_tensor * q_proj_w;
    struct ggml_tensor * k_proj_w;
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;
    // Post-attention LayerScale per channel
    struct ggml_tensor * attn_scale;
    // Post-attention pre-MLP LayerNorm
    struct ggml_tensor * post_attn_norm_w;
    struct ggml_tensor * post_attn_norm_b;
    // MLP
    struct ggml_tensor * fc1_w;
    struct ggml_tensor * fc2_w;
    // Post-MLP LayerScale per channel
    struct ggml_tensor * mlp_scale;
};

struct QwenEncoderTransformer {
    int   hidden_size;
    int   num_layers;
    int   num_attention_heads;
    int   num_kv_heads;
    int   head_dim;
    int   intermediate_size;
    float rope_theta;
    float norm_eps;

    QwenEncoderTransformerLayer layers[ENC_TRANS_MAX_LAYERS];

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool enc_trans_load(QwenEncoderTransformer * tr, const GGUFModel & gf, ggml_backend_t backend) {
    tr->hidden_size         = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.hidden_size");
    uint32_t num_layers     = gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.num_hidden_layers");
    tr->num_layers          = (int) num_layers;
    tr->num_attention_heads = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.num_attention_heads");
    tr->num_kv_heads        = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.num_key_value_heads");
    tr->head_dim            = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.head_dim");
    tr->intermediate_size   = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.intermediate_size");
    tr->rope_theta          = gf_get_f32(gf, "qwen3-tts-tokenizer.encoder.rope_theta");
    tr->norm_eps            = gf_get_f32(gf, "qwen3-tts-tokenizer.encoder.norm_eps");

    if (num_layers == 0 || num_layers > (uint32_t) ENC_TRANS_MAX_LAYERS) {
        fprintf(stderr, "[EncTransformer] FATAL: invalid layer count %u (max %d)\n", num_layers, ENC_TRANS_MAX_LAYERS);
        return false;
    }

    int       n_tensors = tr->num_layers * 12 + 4;  // 12 tensors per layer + headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    for (int l = 0; l < tr->num_layers; l++) {
        QwenEncoderTransformerLayer & ly = tr->layers[l];
        char                          prefix[96];
        snprintf(prefix, sizeof(prefix), "tok_enc.blk.%d", l);
        std::string p(prefix);

        ly.input_norm_w     = gf_load_tensor(&wctx, gf, p + ".attn_norm.weight");
        ly.input_norm_b     = gf_load_tensor(&wctx, gf, p + ".attn_norm.bias");
        ly.q_proj_w         = gf_load_tensor(&wctx, gf, p + ".attn_q.weight");
        ly.k_proj_w         = gf_load_tensor(&wctx, gf, p + ".attn_k.weight");
        ly.v_proj_w         = gf_load_tensor(&wctx, gf, p + ".attn_v.weight");
        ly.o_proj_w         = gf_load_tensor(&wctx, gf, p + ".attn_output.weight");
        ly.attn_scale       = gf_load_tensor(&wctx, gf, p + ".attn_scale");
        ly.post_attn_norm_w = gf_load_tensor(&wctx, gf, p + ".ffn_norm.weight");
        ly.post_attn_norm_b = gf_load_tensor(&wctx, gf, p + ".ffn_norm.bias");
        ly.fc1_w            = gf_load_tensor(&wctx, gf, p + ".ffn_up.weight");
        ly.fc2_w            = gf_load_tensor(&wctx, gf, p + ".ffn_down.weight");
        ly.mlp_scale        = gf_load_tensor(&wctx, gf, p + ".ffn_scale");
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[EncTransformer] FATAL: backend allocation failed\n");
        return false;
    }
    tr->weight_ctx = wctx.ctx;
    tr->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[EncTransformer] Loaded: %d layers, hidden %d, heads %d/%d, head_dim %d, "
            "FFN %d, RoPE theta %.0f\n",
            tr->num_layers, tr->hidden_size, tr->num_attention_heads, tr->num_kv_heads, tr->head_dim,
            tr->intermediate_size, tr->rope_theta);
    return true;
}

static void enc_trans_free(QwenEncoderTransformer * tr) {
    if (tr->weight_buf) {
        ggml_backend_buffer_free(tr->weight_buf);
        tr->weight_buf = NULL;
    }
    if (tr->weight_ctx) {
        ggml_free(tr->weight_ctx);
        tr->weight_ctx = NULL;
    }
}

// Build a [T, T] additive causal mask (0 where allowed, -inf where masked).
// Pure causal: k <= q. The upstream config carries a sliding_window
// value but neither MimiAttention's eager forward nor MimiTransformerModel
// (create_causal_mask) ever apply it. The Qwen3TTS encoder inherits this
// convention, so we mirror it bit for bit here.
static void enc_trans_build_causal_mask(int T, std::vector<float> & dst) {
    dst.assign((size_t) T * (size_t) T, -INFINITY);
    for (int q = 0; q < T; q++) {
        for (int k = 0; k <= q; k++) {
            dst[(size_t) q * (size_t) T + (size_t) k] = 0.0f;
        }
    }
}

static void enc_trans_build_positions(int T, std::vector<int32_t> & dst) {
    dst.resize((size_t) T);
    for (int i = 0; i < T; i++) {
        dst[i] = i;
    }
}

// One Mimi transformer layer. Pre-LayerNorm with bias, attention without
// q/k/v/o biases, MLP with fc1 -> GELU -> fc2 (no SwiGLU), LayerScale on
// both residual paths.
//   x         : [hidden, T] f32 C-first
//   positions: [T] i32
//   mask      : [T, T] f32 additive
// Returns [hidden, T] f32 C-first.
static struct ggml_tensor * enc_trans_layer_forward(struct ggml_context *               ctx,
                                                    const QwenEncoderTransformer *      tr,
                                                    const QwenEncoderTransformerLayer & layer,
                                                    struct ggml_tensor *                x,
                                                    struct ggml_tensor *                positions,
                                                    struct ggml_tensor *                mask,
                                                    int                                 T) {
    int hidden    = tr->hidden_size;
    int n_q_heads = tr->num_attention_heads;
    int n_kv      = tr->num_kv_heads;
    int hd        = tr->head_dim;

    // Pre-LayerNorm with affine (weight + bias). ggml_norm normalizes on ne[0]
    // which is `hidden` here, matching PyTorch nn.LayerNorm(hidden).
    struct ggml_tensor * ln1 = ggml_norm(ctx, x, tr->norm_eps);
    ln1                      = ggml_mul(ctx, ln1, layer.input_norm_w);
    ln1                      = ggml_add(ctx, ln1, layer.input_norm_b);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.q_proj_w, ln1);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.k_proj_w, ln1);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.v_proj_w, ln1);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tr->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    struct ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    struct ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    struct ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    float                scale  = 1.0f / sqrtf((float) hd);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    struct ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    attn                      = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn                      = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.o_proj_w, attn);

    o = ggml_mul(ctx, o, layer.attn_scale);
    x = ggml_add(ctx, x, o);

    // MLP block: pre-LayerNorm (with bias) + fc1 -> GELU(erf) -> fc2 + LayerScale + residual.
    struct ggml_tensor * ln2 = ggml_norm(ctx, x, tr->norm_eps);
    ln2                      = ggml_mul(ctx, ln2, layer.post_attn_norm_w);
    ln2                      = ggml_add(ctx, ln2, layer.post_attn_norm_b);

    struct ggml_tensor * mlp = ggml_mul_mat(ctx, layer.fc1_w, ln2);
    mlp                      = ggml_gelu_erf(ctx, mlp);
    mlp                      = ggml_mul_mat(ctx, layer.fc2_w, mlp);

    mlp = ggml_mul(ctx, mlp, layer.mlp_scale);
    x   = ggml_add(ctx, x, mlp);

    (void) hidden;
    return x;
}

// Full encoder transformer forward. No top-level input_proj or output_proj
// brackets: the SEANet output already has hidden_size channels.
//   x         : [hidden, T] f32 C-first
//   positions: [T] i32
//   mask      : [T, T] f32 additive
// Returns [hidden, T] f32 C-first.
static struct ggml_tensor * enc_trans_forward(struct ggml_context *          ctx,
                                              const QwenEncoderTransformer * tr,
                                              struct ggml_tensor *           x,
                                              struct ggml_tensor *           positions,
                                              struct ggml_tensor *           mask) {
    int T = (int) x->ne[1];
    for (int l = 0; l < tr->num_layers; l++) {
        x = enc_trans_layer_forward(ctx, tr, tr->layers[l], x, positions, mask, T);
    }
    return x;
}
