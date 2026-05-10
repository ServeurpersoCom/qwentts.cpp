#pragma once
// seanet-encoder.h: SEANet conv stack for the Qwen3-TTS encoder side.
//
// Mimi-style structure adapted to T-first ggml. The audio waveform enters
// at 24 kHz mono, gets downsampled by [4, 5, 6, 8] (cumulative 960x), and
// exits at 512 channels @ 25 Hz. A final downsample conv (k=4 stride=2)
// lives in encoder-downsample.h and brings the rate to 12.5 Hz.
//
// Structure:
//   init       : MimiConv1d k=7, 1   -> 64,  causal stride=1
//   for ratio in reversed(upsampling_ratios) i.e. iter [4, 5, 6, 8] :
//     resnet block: ELU -> Conv1d k=3 d=1 dim/2 -> ELU -> Conv1d k=1 dim
//     ELU
//     Conv1d k=2*ratio, stride=ratio, channels x2
//   last       : MimiConv1d k=3, 1024 -> 512, causal stride=1
//
// Apply order on a 24 kHz waveform (matches Python MimiEncoder forward):
//   audio -> init (1->64) -> stage 0 (4x, 64->128) -> stage 1 (5x, 128->256)
//         -> stage 2 (6x, 256->512) -> stage 3 (8x, 512->1024) -> last (1024->512)
// Total downsample = 4 * 5 * 6 * 8 = 960. The 12.5 Hz rate is reached after
// the final downsample conv (factor 2 more in encoder-downsample.h).

#include "causal-trans-conv.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define QWEN_SEANET_NUM_STAGES 4

struct QwenSEANetResNet {
    // First conv inside the residual: depthwise reduction by config.compress
    //   block_0: k=residual_kernel_size (3), stride=1, dilation=1, dim -> dim/2
    struct ggml_tensor * c0_w;
    struct ggml_tensor * c0_b;
    // Second conv: pointwise back to dim
    //   block_1: k=1, dim/2 -> dim
    struct ggml_tensor * c1_w;
    struct ggml_tensor * c1_b;
};

struct QwenSEANetStage {
    QwenSEANetResNet     resnet;
    // Downsampling conv: k=2*ratio, stride=ratio, dim -> dim*2
    struct ggml_tensor * down_w;
    struct ggml_tensor * down_b;
    int                  ratio;
    int                  in_ch;
    int                  out_ch;
};

struct QwenSEANetEncoder {
    // Initial conv: k=7, 1 -> num_filters (64), causal stride=1
    struct ggml_tensor * init_w;
    struct ggml_tensor * init_b;

    QwenSEANetStage stages[QWEN_SEANET_NUM_STAGES];

    // Last conv: k=last_kernel_size (3), final_dim -> hidden_size (512)
    struct ggml_tensor * last_w;
    struct ggml_tensor * last_b;

    int kernel_size;           // 7
    int residual_kernel_size;  // 3
    int last_kernel_size;      // 3
    int num_filters;           // 64
    int compress;              // 2
    int hidden_size;           // 512

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Read encoder hyperparameters and bind every SEANet tensor on the backend.
static bool qwen_seanet_encoder_load(QwenSEANetEncoder * s, const GGUFModel & gf, ggml_backend_t backend) {
    s->kernel_size          = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.kernel_size");
    s->residual_kernel_size = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.residual_kernel_size");
    s->last_kernel_size     = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.last_kernel_size");
    s->num_filters          = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.num_filters");
    s->compress             = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.compress");
    s->hidden_size          = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.hidden_size");

    // The config stores upsampling_ratios in upsample order [8, 6, 5, 4].
    // Python downsampling iterates `reversed(upsampling_ratios)` = [4, 5, 6, 8],
    // so stage 0 applies ratio 4, stage 1 ratio 5, stage 2 ratio 6, stage 3
    // ratio 8. Cumulative downsample is 4*5*6*8 = 960.
    int ratios[QWEN_SEANET_NUM_STAGES];
    {
        const auto & arr = gf_get_array_u32(gf, "qwen3-tts-tokenizer.encoder.upsampling_ratios");
        if ((int) arr.size() != QWEN_SEANET_NUM_STAGES) {
            fprintf(stderr, "[SEANet] FATAL: upsampling_ratios has %d entries, expected %d\n", (int) arr.size(),
                    QWEN_SEANET_NUM_STAGES);
            return false;
        }
        for (int i = 0; i < QWEN_SEANET_NUM_STAGES; i++) {
            ratios[i] = (int) arr[QWEN_SEANET_NUM_STAGES - 1 - i];
        }
    }

    int n_tensors = 4                             // init wb + last wb
                    + QWEN_SEANET_NUM_STAGES * 6  // 4 resnet wb + 2 down wb per stage
                    + 4;                          // headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    s->init_w = gf_load_tensor(&wctx, gf, "tok_enc.conv.0.weight");
    s->init_b = gf_load_tensor(&wctx, gf, "tok_enc.conv.0.bias");

    // Stage indexing follows the Python ModuleList layout :
    //   res block at py idx {1, 4, 7, 10}, two convs at sub-idx 1 and 3
    //   downsample conv at py idx {3, 6, 9, 12}
    static const int RES_PY_IDX[]  = { 1, 4, 7, 10 };
    static const int DOWN_PY_IDX[] = { 3, 6, 9, 12 };

    int dim = s->num_filters;
    for (int i = 0; i < QWEN_SEANET_NUM_STAGES; i++) {
        QwenSEANetStage & stg = s->stages[i];
        stg.ratio             = ratios[i];
        stg.in_ch             = dim;
        stg.out_ch            = dim * 2;

        char name[80];
        snprintf(name, sizeof(name), "tok_enc.res.%d.blk.1.weight", RES_PY_IDX[i]);
        stg.resnet.c0_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_enc.res.%d.blk.1.bias", RES_PY_IDX[i]);
        stg.resnet.c0_b = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_enc.res.%d.blk.3.weight", RES_PY_IDX[i]);
        stg.resnet.c1_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_enc.res.%d.blk.3.bias", RES_PY_IDX[i]);
        stg.resnet.c1_b = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "tok_enc.conv.%d.weight", DOWN_PY_IDX[i]);
        stg.down_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "tok_enc.conv.%d.bias", DOWN_PY_IDX[i]);
        stg.down_b = gf_load_tensor(&wctx, gf, name);

        dim = stg.out_ch;
    }

    s->last_w = gf_load_tensor(&wctx, gf, "tok_enc.conv.14.weight");
    s->last_b = gf_load_tensor(&wctx, gf, "tok_enc.conv.14.bias");

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[SEANet] FATAL: backend allocation failed\n");
        return false;
    }
    s->weight_ctx = wctx.ctx;
    s->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[SEANet] Loaded: 4 stages (ratios %d/%d/%d/%d), %d -> %d channels, "
            "weights %.1f MB\n",
            ratios[0], ratios[1], ratios[2], ratios[3], s->num_filters, s->hidden_size,
            (float) ggml_backend_buffer_get_size(s->weight_buf) / (1024.0f * 1024.0f));
    return true;
}

static void qwen_seanet_encoder_free(QwenSEANetEncoder * s) {
    if (s->weight_buf) {
        ggml_backend_buffer_free(s->weight_buf);
        s->weight_buf = NULL;
    }
    if (s->weight_ctx) {
        ggml_free(s->weight_ctx);
        s->weight_ctx = NULL;
    }
}

// SEANet ResNet forward: skip; ELU; conv k=3,s=1,d=1, dim->dim/2; ELU;
// conv k=1, dim/2->dim; add(skip).
static struct ggml_tensor * qwen_seanet_resnet_forward(struct ggml_context *    ctx,
                                                       const QwenSEANetResNet * ru,
                                                       struct ggml_tensor *     x,
                                                       int                      residual_kernel_size) {
    struct ggml_tensor * skip = x;
    x                         = ggml_elu(ctx, x);
    x                         = qwen_causal_conv1d(ctx, ru->c0_w, ru->c0_b, x, residual_kernel_size, 1, 1);
    x                         = ggml_elu(ctx, x);
    x                         = qwen_causal_conv1d(ctx, ru->c1_w, ru->c1_b, x, 1, 1, 1);
    return ggml_add(ctx, skip, x);
}

// Full SEANet forward.
//   x: [T_audio, 1] f32 T-first (mono waveform)
// Optional out-params capture intermediate stage outputs in T-first ggml
// layout ne=(C, T_out) for debug bisection. Each is NULL by default and
// the caller decides whether to mark them as graph outputs.
//   init_out    : post init MimiConv1d k=7, [T_audio, 64]
//   resnet0_out : post stage 0 resnet block, before ELU+downsample, [T_audio, 64]
//   stage0_out  : post stage 0 (resnet + ELU + downsample 4x), [T_audio/4, 128]
//   stage1_out  : post stage 1 (resnet + ELU + downsample 5x), [T_audio/20, 256]
//   stage3_out  : post stage 3 (resnet + ELU + downsample 8x), [T_audio/960, 1024]
// Returns [T_audio / 960, 512] f32 T-first.
static struct ggml_tensor * qwen_seanet_encoder_forward(struct ggml_context *     ctx,
                                                        const QwenSEANetEncoder * s,
                                                        struct ggml_tensor *      x,
                                                        struct ggml_tensor **     init_out    = NULL,
                                                        struct ggml_tensor **     resnet0_out = NULL,
                                                        struct ggml_tensor **     stage0_out  = NULL,
                                                        struct ggml_tensor **     stage1_out  = NULL,
                                                        struct ggml_tensor **     stage3_out  = NULL) {
    x = qwen_causal_conv1d(ctx, s->init_w, s->init_b, x, s->kernel_size, 1, 1);
    if (init_out) {
        *init_out = x;
    }

    for (int i = 0; i < QWEN_SEANET_NUM_STAGES; i++) {
        const QwenSEANetStage & stg = s->stages[i];
        x                           = qwen_seanet_resnet_forward(ctx, &stg.resnet, x, s->residual_kernel_size);
        if (i == 0 && resnet0_out) {
            *resnet0_out = x;
        }
        x = ggml_elu(ctx, x);
        x = qwen_causal_conv1d(ctx, stg.down_w, stg.down_b, x, 2 * stg.ratio, 1, stg.ratio);
        if (i == 0 && stage0_out) {
            *stage0_out = x;
        }
        if (i == 1 && stage1_out) {
            *stage1_out = x;
        }
        if (i == 3 && stage3_out) {
            *stage3_out = x;
        }
    }

    x = ggml_elu(ctx, x);
    x = qwen_causal_conv1d(ctx, s->last_w, s->last_b, x, s->last_kernel_size, 1, 1);
    return x;
}
