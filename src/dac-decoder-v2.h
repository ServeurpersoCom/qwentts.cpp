#pragma once
// dac-decoder-v2.h: DAC acoustic decoder for the Qwen3-TTS 12Hz tokenizer.
//
// Layout: T-first [T, C] from conv_pre to conv_post. The fused SNAKE op
// requires ne[0]=T and ne[1]=C, so the whole DAC pipeline matches that
// convention. ggml_conv_1d and ggml_conv_1d_dw are T-first natively, and
// the only mul_mat lives inside qwen_causal_trans_conv1d which transposes
// internally.
//
// Pipeline (input [T, 1024] -> audio [T*1920, 1] @ 24 kHz mono):
//   conv_pre k=7 (1024 -> 1536, causal)
//   4 blocks: SnakeBeta -> CausalTransConv k=2*stride -> 3 ResUnits
//     strides 8 / 5 / 4 / 3, channels 1536 -> 768 -> 384 -> 192 -> 96
//     ResUnit: SnakeBeta -> conv k=7 dilation -> SnakeBeta -> conv k=1 + skip
//     dilations 1 / 3 / 9
//   snake_post (96)
//   conv_post k=7 (96 -> 1, causal)
//
// SnakeBeta: the Qwen3-TTS reference applies exp() to alpha and beta on
// every forward. Both factors are precomputed CPU-side at load time and
// stored as a = exp(alpha) and inv_b = 1 / (exp(beta) + 1e-9), so the
// runtime kernel stays a single fused ggml_snake op with semantics
// y = x + sin^2(a * x) * inv_b.

#include "causal-trans-conv.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#define DAC_NUM_BLOCKS 4
#define DAC_RES_UNITS  3

// SnakeBeta runtime parameters with exp() folded in: a holds exp(alpha)
// and inv_b holds 1 / (exp(beta) + 1e-9). Layout [1, C] f32 matches the
// broadcast convention of ggml_snake.
struct QwenDACSnake {
    struct ggml_tensor * a;      // [1, C] f32
    struct ggml_tensor * inv_b;  // [1, C] f32
};

// One residual unit: act1 -> conv1 (k=7, dilation d, causal) -> act2 ->
// conv2 (k=1) + skip.
struct QwenDACResUnit {
    QwenDACSnake         act1;
    QwenDACSnake         act2;
    struct ggml_tensor * c1w;  // [7, C, C] f32, stored (K, IC, OC)
    struct ggml_tensor * c1b;  // [C] f32
    struct ggml_tensor * c2w;  // [1, C, C] f32
    struct ggml_tensor * c2b;  // [C] f32
    int                  dilation;
};

// One DAC block: snake1 -> causal transposed conv (upsample) -> 3 res
// units. The transconv weight is pre-permuted to [IC, K*OC] at load time
// so the col2im_1d path stays a fused mul_mat.
struct QwenDACBlock {
    QwenDACSnake         snake1;
    struct ggml_tensor * tcw;  // [IC, K*OC] f32, pre-permuted from (IC, OC, K)
    struct ggml_tensor * tcb;  // [OC] f32
    QwenDACResUnit       ru[DAC_RES_UNITS];
    int                  in_ch;
    int                  out_ch;
    int                  stride;
    int                  kernel;  // 2 * stride
};

struct QwenDACDecoder {
    // initial conv: 1024 -> 1536, k=7, causal
    struct ggml_tensor * conv_pre_w;  // [7, 1024, 1536] f32
    struct ggml_tensor * conv_pre_b;  // [1536] f32

    QwenDACBlock blk[DAC_NUM_BLOCKS];

    QwenDACSnake snake_post;  // 96 channels

    // final conv: 96 -> 1, k=7, causal
    struct ggml_tensor * conv_post_w;  // [7, 96, 1] f32
    struct ggml_tensor * conv_post_b;  // [1] f32

    int channels[DAC_NUM_BLOCKS + 1];  // 1536, 768, 384, 192, 96

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Read alpha and beta from the GGUF, fold exp() and reciprocal CPU-side,
// and bind two [1, C] f32 tensors on the backend ctx as a and inv_b.
static void dac_load_snakebeta(WeightCtx *         wctx,
                               const GGUFModel &   gf,
                               QwenDACSnake *      s,
                               const std::string & alpha_name,
                               const std::string & beta_name) {
    struct ggml_tensor * alpha_meta = ggml_get_tensor(gf.meta, alpha_name.c_str());
    struct ggml_tensor * beta_meta  = ggml_get_tensor(gf.meta, beta_name.c_str());
    if (!alpha_meta || !beta_meta) {
        qt_throw("[DAC] snake tensor '%s' or '%s' not found", alpha_name.c_str(), beta_name.c_str());
    }
    if (alpha_meta->type != GGML_TYPE_F32 || beta_meta->type != GGML_TYPE_F32) {
        qt_throw("[DAC] snake '%s' expects F32 alpha/beta", alpha_name.c_str());
    }
    int C = (int) alpha_meta->ne[0];
    if ((int) beta_meta->ne[0] != C) {
        qt_throw("[DAC] snake '%s' alpha/beta size mismatch (%d vs %d)", alpha_name.c_str(), C, (int) beta_meta->ne[0]);
    }

    s->a     = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, 1, C);
    s->inv_b = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, 1, C);
    ggml_set_name(s->a, alpha_name.c_str());
    ggml_set_name(s->inv_b, beta_name.c_str());

    const float * alpha_src = (const float *) gf_get_data(gf, alpha_name.c_str());
    const float * beta_src  = (const float *) gf_get_data(gf, beta_name.c_str());

    auto a_buf     = std::make_unique<float[]>((size_t) C);
    auto inv_b_buf = std::make_unique<float[]>((size_t) C);
    for (int c = 0; c < C; c++) {
        a_buf[c]     = expf(alpha_src[c]);
        inv_b_buf[c] = 1.0f / (expf(beta_src[c]) + 1e-9f);
    }

    wctx->pending.push_back({ s->a, a_buf.get(), (size_t) C * sizeof(float), 0 });
    wctx->pending.push_back({ s->inv_b, inv_b_buf.get(), (size_t) C * sizeof(float), 0 });
    wctx->staging.push_back(std::move(a_buf));
    wctx->staging.push_back(std::move(inv_b_buf));
}

// Allocate every weight tensor on the backend, copy from the GGUF mapping
// with per-tensor transforms (snake exp/reciprocal, transconv permute).
static bool dac_decoder_load(QwenDACDecoder * d, const GGUFModel & gf, ggml_backend_t backend) {
    static const int strides[DAC_NUM_BLOCKS]  = { 8, 5, 4, 3 };
    static const int chs[DAC_NUM_BLOCKS + 1]  = { 1536, 768, 384, 192, 96 };
    static const int dilations[DAC_RES_UNITS] = { 1, 3, 9 };

    for (int i = 0; i <= DAC_NUM_BLOCKS; i++) {
        d->channels[i] = chs[i];
    }

    int       n_tensors = 132;  // 118 actual + headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    d->conv_pre_w = gf_load_conv(&wctx, gf, "tok_dec.dec.0.conv.weight");
    d->conv_pre_b = gf_load_tensor(&wctx, gf, "tok_dec.dec.0.conv.bias");

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        QwenDACBlock & b = d->blk[i];
        b.in_ch          = chs[i];
        b.out_ch         = chs[i + 1];
        b.stride         = strides[i];
        b.kernel         = 2 * strides[i];

        // Python ModuleList places blocks at indices 1, 2, 3, 4. Index 0
        // is the entry conv (loaded above), indices 5 and 6 are the post
        // snake and final conv (loaded below).
        int  py_idx = i + 1;
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "tok_dec.dec.%d", py_idx);

        dac_load_snakebeta(&wctx, gf, &b.snake1, std::string(prefix) + ".snake.alpha",
                           std::string(prefix) + ".snake.beta");

        b.tcw = qwen_load_ctw_f32(&wctx, gf, std::string(prefix) + ".conv_t.weight");
        b.tcb = gf_load_tensor(&wctx, gf, std::string(prefix) + ".conv_t.bias");

        for (int r = 0; r < DAC_RES_UNITS; r++) {
            QwenDACResUnit & ru = b.ru[r];
            ru.dilation         = dilations[r];

            char rp[96];
            snprintf(rp, sizeof(rp), "%s.res.%d", prefix, r);

            dac_load_snakebeta(&wctx, gf, &ru.act1, std::string(rp) + ".act1.alpha", std::string(rp) + ".act1.beta");
            ru.c1w = gf_load_conv(&wctx, gf, std::string(rp) + ".conv1.weight");
            ru.c1b = gf_load_tensor(&wctx, gf, std::string(rp) + ".conv1.bias");
            dac_load_snakebeta(&wctx, gf, &ru.act2, std::string(rp) + ".act2.alpha", std::string(rp) + ".act2.beta");
            ru.c2w = gf_load_conv(&wctx, gf, std::string(rp) + ".conv2.weight");
            ru.c2b = gf_load_tensor(&wctx, gf, std::string(rp) + ".conv2.bias");
        }
    }

    dac_load_snakebeta(&wctx, gf, &d->snake_post, "tok_dec.dec.5.snake.alpha", "tok_dec.dec.5.snake.beta");
    d->conv_post_w = gf_load_conv(&wctx, gf, "tok_dec.dec.6.conv.weight");
    d->conv_post_b = gf_load_tensor(&wctx, gf, "tok_dec.dec.6.conv.bias");

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[DAC] FATAL: backend allocation failed\n");
        return false;
    }
    d->weight_ctx = wctx.ctx;
    d->weight_buf = wctx.buffer;

    fprintf(stderr, "[DAC] Loaded: %d blocks (strides 8/5/4/3), 24 kHz mono out, weights %.1f MB\n", DAC_NUM_BLOCKS,
            (float) ggml_backend_buffer_get_size(d->weight_buf) / (1024.0f * 1024.0f));
    return true;
}

static void dac_decoder_free(QwenDACDecoder * d) {
    if (d->weight_buf) {
        ggml_backend_buffer_free(d->weight_buf);
        d->weight_buf = NULL;
    }
    if (d->weight_ctx) {
        ggml_free(d->weight_ctx);
        d->weight_ctx = NULL;
    }
}

// SnakeBeta forward: y = x + sin^2(a * x) * inv_b. Written as primitive
// ops so the backend graph optimiser fuses them into a dedicated snake
// kernel where one is available, and falls back to a plain CPU/GPU op
// chain otherwise. x [T, C] T-first, a and inv_b broadcast on the C axis.
static struct ggml_tensor * dac_snake(struct ggml_context * ctx, struct ggml_tensor * x, const QwenDACSnake & s) {
    struct ggml_tensor * t = ggml_mul(ctx, x, s.a);      // a * x  (broadcast over T)
    t                      = ggml_sin(ctx, t);           // sin(a * x)
    t                      = ggml_sqr(ctx, t);           // sin^2(a * x)
    t                      = ggml_mul(ctx, t, s.inv_b);  // sin^2(a * x) * inv_b
    return ggml_add(ctx, x, t);                          // x + sin^2(a * x) * inv_b
}

// Causal Conv1d helper qwen_causal_conv1d lives in causal-trans-conv.h
// alongside qwen_causal_trans_conv1d so that pre_conv (in pipeline-codec)
// and the DAC share the same primitive.

// Residual unit forward: skip + conv2(snake(conv1(snake(x)))).
static struct ggml_tensor * dac_res_unit(struct ggml_context * ctx, const QwenDACResUnit * ru, struct ggml_tensor * x) {
    struct ggml_tensor * skip = x;
    x                         = dac_snake(ctx, x, ru->act1);
    x                         = qwen_causal_conv1d(ctx, ru->c1w, ru->c1b, x, 7, ru->dilation);
    x                         = dac_snake(ctx, x, ru->act2);
    x                         = qwen_causal_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 1);
    return ggml_add(ctx, skip, x);
}

// Full DAC forward graph.
//   x: [T, 1024] f32 T-first
// returns [T * 1920, 1] f32 T-first (raw audio samples @ 24 kHz mono,
// without final clamp; the orchestration layer clips to [-1, 1]).
static struct ggml_tensor * dac_decoder_forward(struct ggml_context *  ctx,
                                                const QwenDACDecoder * d,
                                                struct ggml_tensor *   x) {
    x = qwen_causal_conv1d(ctx, d->conv_pre_w, d->conv_pre_b, x, 7, 1);

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const QwenDACBlock & b = d->blk[i];
        x                      = dac_snake(ctx, x, b.snake1);
        x                      = qwen_causal_trans_conv1d(ctx, b.tcw, b.tcb, x, b.stride, b.kernel, b.out_ch);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            x = dac_res_unit(ctx, &b.ru[r], x);
        }
    }

    x = dac_snake(ctx, x, d->snake_post);
    x = qwen_causal_conv1d(ctx, d->conv_post_w, d->conv_post_b, x, 7, 1);
    return x;
}

// Streaming state for the DAC decoder: one left context per stride 1
// causal conv ([(k-1)*d, IC] f32 at the conv's own rate) and one
// overlap carry per transposed conv ([kernel - stride, out_ch] f32,
// bias free). The k=1 res unit convs are pointwise and stateless.
struct QwenDACStreamState {
    struct ggml_tensor * pre;                                // conv_pre k=7, [6, 1024]
    struct ggml_tensor * carry[DAC_NUM_BLOCKS];              // transconv tails, [stride, out_ch]
    struct ggml_tensor * ru[DAC_NUM_BLOCKS][DAC_RES_UNITS];  // conv1 k=7 dilated, [6*d, ch]
    struct ggml_tensor * post;                               // conv_post k=7, [6, 96]
};

// Streaming residual unit, batched over N lanes: conv1 reads its left
// context from the persistent state, conv2 is pointwise (k=1, pad
// free) and runs a direct batched conv.
static struct ggml_tensor * dac_res_unit_stream(struct ggml_context *  ctx,
                                                struct ggml_cgraph *   gf,
                                                const QwenDACResUnit * ru,
                                                struct ggml_tensor *   x,
                                                struct ggml_tensor *   state) {
    struct ggml_tensor * skip = x;
    x                         = dac_snake(ctx, x, ru->act1);
    x                         = qwen_causal_conv1d_stream(ctx, gf, ru->c1w, ru->c1b, x, 7, ru->dilation, state);
    x                         = dac_snake(ctx, x, ru->act2);
    x                         = ggml_conv_1d(ctx, ru->c2w, x, 1, 0, 1);
    if (ru->c2b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, ru->c2b, 1, ru->c2b->ne[0]);
        x                        = ggml_add(ctx, x, b2d);
    }
    return ggml_add(ctx, skip, x);
}

// Streaming DAC forward graph: same stack as dac_decoder_forward with
// every stateful op threaded through the persistent stream state.
//   x: [T, 1024] f32 T-first
// returns [T * 1920, 1] f32 T-first, unclamped.
static struct ggml_tensor * dac_decoder_forward_stream(struct ggml_context *      ctx,
                                                       struct ggml_cgraph *       gf,
                                                       const QwenDACDecoder *     d,
                                                       struct ggml_tensor *       x,
                                                       const QwenDACStreamState * st) {
    x = qwen_causal_conv1d_stream(ctx, gf, d->conv_pre_w, d->conv_pre_b, x, 7, 1, st->pre);

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const QwenDACBlock & b = d->blk[i];
        x                      = dac_snake(ctx, x, b.snake1);
        x = qwen_causal_trans_conv1d_stream(ctx, gf, b.tcw, b.tcb, x, b.stride, b.kernel, b.out_ch, st->carry[i]);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            x = dac_res_unit_stream(ctx, gf, &b.ru[r], x, st->ru[i][r]);
        }
    }

    x = dac_snake(ctx, x, d->snake_post);
    x = qwen_causal_conv1d_stream(ctx, gf, d->conv_post_w, d->conv_post_b, x, 7, 1, st->post);
    return x;
}
