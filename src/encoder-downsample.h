#pragma once
// encoder-downsample.h: final conv k=4 stride=2 between the encoder
// transformer output and the RVQ. Brings the rate from 25 Hz to 12.5 Hz
// while preserving the 512-channel hidden dimension.
//
// The Python upstream defines this as a MimiConv1d with use_causal_conv,
// no bias, kernel 4, stride 2. The shape in the GGUF is (4, 512, 512).

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cstdlib>

struct QwenEncoderDownsample {
    struct ggml_tensor *  weight;  // [4, 512, 512] f32
    int                   in_ch;
    int                   out_ch;
    int                   kernel;
    int                   stride;
    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool enc_down_load(QwenEncoderDownsample * d, const GGUFModel & gf, ggml_backend_t backend) {
    d->kernel = 4;
    d->stride = 2;

    WeightCtx wctx;
    wctx_init(&wctx, 4);
    d->weight = gf_load_conv(&wctx, gf, "tok_enc.downsample.weight");
    if (!wctx_alloc(&wctx, backend)) {
        qt_log(QT_LOG_ERROR, "[EncDownsample] FATAL: backend allocation failed");
        return false;
    }
    d->weight_ctx = wctx.ctx;
    d->weight_buf = wctx.buffer;

    d->in_ch  = (int) d->weight->ne[1];
    d->out_ch = (int) d->weight->ne[2];

    qt_log(QT_LOG_INFO, "[EncDownsample] Loaded: k=%d stride=%d, %d -> %d channels, weights %.1f MB", d->kernel,
           d->stride, d->in_ch, d->out_ch, (float) ggml_backend_buffer_get_size(d->weight_buf) / (1024.0f * 1024.0f));
    return true;
}

static void enc_down_free(QwenEncoderDownsample * d) {
    if (d->weight_buf) {
        ggml_backend_buffer_free(d->weight_buf);
        d->weight_buf = NULL;
    }
    if (d->weight_ctx) {
        ggml_free(d->weight_ctx);
        d->weight_ctx = NULL;
    }
}

// Forward: causal Conv1d k=4 stride=2, no bias. The Mimi downsample is
// the only conv on this side that ships with pad_mode="replicate"
// hardcoded upstream (transformers MimiModel.__init__), unlike SEANet
// and the encoder transformer which inherit config.pad_mode='constant'.
//   x: [T, 512] f32 T-first
// Returns [ceil(T/2), 512] f32 T-first.
static struct ggml_tensor * enc_down_forward(struct ggml_context *         ctx,
                                             const QwenEncoderDownsample * d,
                                             struct ggml_tensor *          x) {
    return qwen_causal_conv1d(ctx, d->weight, NULL, x, d->kernel, 1, d->stride, CTC_PAD_REPLICATE);
}
