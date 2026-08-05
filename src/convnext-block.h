#pragma once
// convnext-block.h: 2-block upsample stage for the Qwen3-TTS 12Hz
// tokenizer decoder.
//
// Each block is a CausalTransConv1d (kernel 2, stride 2) followed by a
// ConvNeXt block. The ConvNeXt block is :
//   x = x + gamma * pwconv2(gelu(pwconv1(layernorm(dwconv(x)))))
// where dwconv is a depthwise causal Conv1d (kernel 7, dilation 1),
// pwconv1 / pwconv2 are pointwise Linears (1024 -> 4096 -> 1024), and
// gamma is a per-channel LayerScale parameter. The two blocks together
// upsample the temporal axis by 4x while keeping the channel count at
// latent_dim (1024).

#include "causal-trans-conv.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define UPSAMPLE_MAX_BLOCKS 2

struct QwenConvNeXtBlock {
    struct ggml_tensor * dwconv_w;   // [K=7, 1, C] depthwise weight
    struct ggml_tensor * dwconv_b;   // [C]
    struct ggml_tensor * norm_w;     // LayerNorm gain [C]
    struct ggml_tensor * norm_b;     // LayerNorm bias [C]
    struct ggml_tensor * pwconv1_w;  // [C, 4*C] (in, out)
    struct ggml_tensor * pwconv1_b;  // [4*C]
    struct ggml_tensor * pwconv2_w;  // [4*C, C]
    struct ggml_tensor * pwconv2_b;  // [C]
    struct ggml_tensor * gamma;      // LayerScale [C]
};

struct QwenUpsampleStage {
    int num_blocks;                                         // 2
    int channels;                                           // 1024 (= latent_dim)
    int upsample_ratio;                                     // 2 per block, 4x total
    int dwconv_kernel;                                      // 7

    struct ggml_tensor * transconv_w[UPSAMPLE_MAX_BLOCKS];  // pre-permuted [IC, K*OC]
    struct ggml_tensor * transconv_b[UPSAMPLE_MAX_BLOCKS];  // [OC]
    QwenConvNeXtBlock    convnext[UPSAMPLE_MAX_BLOCKS];

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Read upsample hyperparameters from GGUF metadata, allocate every weight
// tensor on the backend, and bind tensor pointers in the struct.
static bool upsample_stage_load(QwenUpsampleStage * stage, const GGUFModel & gf, ggml_backend_t backend) {
    stage->channels       = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.latent_dim");
    stage->dwconv_kernel  = 7;
    stage->upsample_ratio = 2;
    stage->num_blocks     = 2;

    if (stage->num_blocks > UPSAMPLE_MAX_BLOCKS) {
        fprintf(stderr, "[Upsample] FATAL: %d blocks exceeds compile-time max %d\n", stage->num_blocks,
                UPSAMPLE_MAX_BLOCKS);
        return false;
    }

    int       n_tensors = stage->num_blocks * 11 + 4;  // 2 transconv + 9 convnext per block, plus headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    for (int i = 0; i < stage->num_blocks; i++) {
        char name[160];

        snprintf(name, sizeof(name), "tok_dec.upsample.%d.conv.weight", i);
        stage->transconv_w[i] = qwen_load_ctw_f32(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.conv.bias", i);
        stage->transconv_b[i] = gf_load_tensor(&wctx, gf, name);

        QwenConvNeXtBlock & cn = stage->convnext[i];
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.dwconv.weight", i);
        cn.dwconv_w = gf_load_conv(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.dwconv.bias", i);
        cn.dwconv_b = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.norm.weight", i);
        cn.norm_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.norm.bias", i);
        cn.norm_b = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.pwconv1.weight", i);
        cn.pwconv1_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.pwconv1.bias", i);
        cn.pwconv1_b = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.pwconv2.weight", i);
        cn.pwconv2_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.pwconv2.bias", i);
        cn.pwconv2_b = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_dec.upsample.%d.gamma", i);
        cn.gamma = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[Upsample] FATAL: backend allocation failed\n");
        return false;
    }
    stage->weight_ctx = wctx.ctx;
    stage->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[Upsample] Loaded: %d blocks (%dx ratio per block), channels %d, "
            "dwconv kernel %d\n",
            stage->num_blocks, stage->upsample_ratio, stage->channels, stage->dwconv_kernel);
    return true;
}

static void upsample_stage_free(QwenUpsampleStage * stage) {
    if (stage->weight_buf) {
        ggml_backend_buffer_free(stage->weight_buf);
        stage->weight_buf = NULL;
    }
    if (stage->weight_ctx) {
        ggml_free(stage->weight_ctx);
        stage->weight_ctx = NULL;
    }
}

// One ConvNeXt block forward.
//   x: [T, C] f32 T-first
// returns [T, C] f32 T-first
static struct ggml_tensor * convnext_block_forward(struct ggml_context *     ctx,
                                                   const QwenConvNeXtBlock & block,
                                                   struct ggml_tensor *      x,
                                                   int                       kernel) {
    int T = (int) x->ne[0];
    int C = (int) x->ne[1];

    struct ggml_tensor * residual = x;

    // dwconv: depthwise causal Conv1d. ggml_conv_1d_dw expects [T, C, B=1].
    // Pre-pad left by (kernel-1) zeros for causal behavior, no internal padding.
    struct ggml_tensor * y = ggml_reshape_3d(ctx, x, T, C, 1);
    y                      = ggml_pad_ext(ctx, y, kernel - 1, 0, 0, 0, 0, 0, 0, 0);
    y                      = ggml_conv_1d_dw(ctx, block.dwconv_w, y, 1, 0, 1);  // [T, C, 1]
    y                      = ggml_reshape_2d(ctx, y, T, C);
    if (block.dwconv_b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, block.dwconv_b, 1, C);  // (1, C) broadcasts on T
        y                        = ggml_add(ctx, y, b2d);
    }

    // LayerNorm wants the channel dim on ne[0]: transpose to [C, T].
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    y = ggml_norm(ctx, y, 1e-6f);
    y = ggml_mul(ctx, y, block.norm_w);
    y = ggml_add(ctx, y, block.norm_b);

    // pwconv1: Linear C -> 4*C. mul_mat contracts ne[0]=C of weight against
    // ne[0]=C of input.
    y = ggml_mul_mat(ctx, block.pwconv1_w, y);
    y = ggml_add(ctx, y, block.pwconv1_b);

    y = ggml_gelu(ctx, y);

    // pwconv2: Linear 4*C -> C
    y = ggml_mul_mat(ctx, block.pwconv2_w, y);
    y = ggml_add(ctx, y, block.pwconv2_b);

    // LayerScale gamma broadcast over T axis (ne[1]).
    y = ggml_mul(ctx, y, block.gamma);

    // Back to T-first [T, C] to match the residual layout.
    y = ggml_cont(ctx, ggml_transpose(ctx, y));

    y = ggml_add(ctx, y, residual);
    return y;
}

// Full upsample stage forward: 2 (CausalTransConv + ConvNeXt) blocks.
//   x: [T, C] f32 T-first
// returns [T * 4, C] f32 T-first
//
// The top-level upsample stage uses kernel == stride (no causal trim).
// The DAC decoder blocks (separate header) use kernel == 2 * stride
// with a stride-frame causal trim.
static struct ggml_tensor * upsample_stage_forward(struct ggml_context *     ctx,
                                                   const QwenUpsampleStage * stage,
                                                   struct ggml_tensor *      x) {
    int kernel = stage->upsample_ratio;
    for (int i = 0; i < stage->num_blocks; i++) {
        x = qwen_causal_trans_conv1d(ctx, stage->transconv_w[i], stage->transconv_b[i], x, stage->upsample_ratio,
                                     kernel, stage->channels);
        x = convnext_block_forward(ctx, stage->convnext[i], x, stage->dwconv_kernel);
    }
    return x;
}

// Streaming state for the upsample stage: one depthwise conv left
// context per ConvNeXt block, [dwconv_kernel - 1, C] f32 at the block's
// own rate. The transposed convs have kernel == stride, so they carry
// nothing and run the offline helper unchanged.
struct QwenUpsampleStreamState {
    struct ggml_tensor * dw[UPSAMPLE_MAX_BLOCKS];
};

// Streaming ConvNeXt block, batched over N lanes: the depthwise causal
// conv reads its left context from the persistent state instead of a
// zero pad. Everything else is pointwise and stateless.
//
// ggml_conv_1d_dw carries no batch dim, but a depthwise conv treats
// every channel independently, so the N lanes fold into the channel
// axis: x_ext [L + T, C, N] reshapes to [L + T, C*N], the kernel tiles
// N times to [k, 1, C*N] (channel' = c + n*C matches the reshape
// order), and the [T, C*N] output reshapes back to [T, C, N]. The
// N == 1 branch keeps the fold free chain.
static struct ggml_tensor * convnext_block_forward_stream(struct ggml_context *     ctx,
                                                          struct ggml_cgraph *      gf,
                                                          const QwenConvNeXtBlock & block,
                                                          struct ggml_tensor *      x,
                                                          int                       kernel,
                                                          struct ggml_tensor *      dw_state) {
    int C = (int) x->ne[1];
    int N = (int) x->ne[2];

    struct ggml_tensor * residual = x;

    // dwconv: concat the state ahead of the fresh rows, refresh it with
    // the tail, run the depthwise conv pad free.
    struct ggml_tensor * x_ext = ggml_concat(ctx, dw_state, x, 0);  // [k-1 + T, C, N]
    struct ggml_tensor * tail  = ggml_view_3d(ctx, x_ext, kernel - 1, C, N, x_ext->nb[1], x_ext->nb[2],
                                              (size_t) (x_ext->ne[0] - (kernel - 1)) * x_ext->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, dw_state));

    struct ggml_tensor * y;
    if (N == 1) {
        y = ggml_conv_1d_dw(ctx, block.dwconv_w, x_ext, 1, 0, 1);  // [T, C, 1]
    } else {
        struct ggml_tensor * w_tmpl =
            ggml_new_tensor_4d(ctx, block.dwconv_w->type, block.dwconv_w->ne[0], block.dwconv_w->ne[1], C, N);
        struct ggml_tensor * w_n    = ggml_repeat(ctx, block.dwconv_w, w_tmpl);         // [k, 1, C, N]
        w_n                         = ggml_reshape_3d(ctx, w_n, w_n->ne[0], 1, C * N);  // [k, 1, C*N]
        struct ggml_tensor * folded = ggml_reshape_3d(ctx, x_ext, x_ext->ne[0], C * N, 1);
        y                           = ggml_conv_1d_dw(ctx, w_n, folded, 1, 0, 1);       // [T, C*N, 1]
        y                           = ggml_reshape_3d(ctx, y, y->ne[0], C, N);
    }
    if (block.dwconv_b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, block.dwconv_b, 1, C);
        y                        = ggml_add(ctx, y, b2d);
    }

    // LayerNorm wants the channel dim on ne[0]: transpose to [C, T, N].
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    y = ggml_norm(ctx, y, 1e-6f);
    y = ggml_mul(ctx, y, block.norm_w);
    y = ggml_add(ctx, y, block.norm_b);

    y = ggml_mul_mat(ctx, block.pwconv1_w, y);
    y = ggml_add(ctx, y, block.pwconv1_b);

    y = ggml_gelu(ctx, y);

    y = ggml_mul_mat(ctx, block.pwconv2_w, y);
    y = ggml_add(ctx, y, block.pwconv2_b);

    y = ggml_mul(ctx, y, block.gamma);

    y = ggml_cont(ctx, ggml_transpose(ctx, y));

    y = ggml_add(ctx, y, residual);
    return y;
}

// Streaming upsample stage: transposed convs run the offline helper
// (kernel == stride, exact under chunking), ConvNeXt blocks thread
// their depthwise state.
static struct ggml_tensor * upsample_stage_forward_stream(struct ggml_context *           ctx,
                                                          struct ggml_cgraph *            gf,
                                                          const QwenUpsampleStage *       stage,
                                                          struct ggml_tensor *            x,
                                                          const QwenUpsampleStreamState * st) {
    int kernel = stage->upsample_ratio;
    for (int i = 0; i < stage->num_blocks; i++) {
        x = qwen_causal_trans_conv1d(ctx, stage->transconv_w[i], stage->transconv_b[i], x, stage->upsample_ratio,
                                     kernel, stage->channels);
        x = convnext_block_forward_stream(ctx, gf, stage->convnext[i], x, stage->dwconv_kernel, st->dw[i]);
    }
    return x;
}
