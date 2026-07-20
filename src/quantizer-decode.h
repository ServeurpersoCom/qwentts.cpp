#pragma once
// quantizer-decode.h: split RVQ decode for the Qwen3-TTS 12Hz tokenizer
// (GGML).
// Reads 16 codebooks (1 semantic + 15 acoustic) of 2048 entries with
// internal dim 256, and produces a 512-channel hidden representation.
//
// Decode side: codes [T, 16] i32 -> hidden [T, 512] f32 by summing
// F.embedding(codes[:, k], codebook_k) within each split, then applying
// a per-split output_proj 1x1 conv (256 -> 512), then summing the two
// splits.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define RVQ_MAX_CODEBOOKS_PER_GROUP 15

struct QwenRVQGroup {
    int                  num_codebooks;
    struct ggml_tensor * embed[RVQ_MAX_CODEBOOKS_PER_GROUP];  // each [256, 2048] f32
    struct ggml_tensor * out_proj_w;                          // [256, 512] f32 (Conv1d 1x1 reshaped)
};

struct QwenQuantizerDecoder {
    int num_quantizers;           // total RVQ stages, 16
    int num_semantic_quantizers;  // 1
    int num_acoustic_quantizers;  // 15
    int codebook_size;            // 2048
    int codebook_dim_internal;    // 256
    int hidden;                   // 512

    QwenRVQGroup semantic;
    QwenRVQGroup acoustic;

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Load a Conv1d 1x1 weight stored on disk as (1, in, out) and present it
// as a 2D [in, out] f32 tensor suitable for ggml_mul_mat.
static struct ggml_tensor * qwen_load_proj_1x1(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        qt_throw("[Quantizer] tensor '%s' not found", name.c_str());
    }
    if (src->ne[0] != 1) {
        qt_throw("[Quantizer] '%s' expected kernel=1 on ne[0], got %lld", name.c_str(), (long long) src->ne[0]);
    }
    int64_t shape2d[2] = { src->ne[1], src->ne[2] };  // (in_dim, out_dim) in ggml row-major
    return gf_load_tensor(wctx, gf, name, shape2d, 2);
}

// Build the on-backend weights of the split RVQ decoder from a loaded GGUF.
// Mutates dec->weight_ctx and dec->weight_buf, and binds every group
// tensor pointer to a backend allocation.
static bool quant_decoder_load(QwenQuantizerDecoder * dec, const GGUFModel & gf, ggml_backend_t backend) {
    dec->num_quantizers          = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.num_quantizers");
    dec->num_semantic_quantizers = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.num_semantic_quantizers");
    dec->num_acoustic_quantizers = dec->num_quantizers - dec->num_semantic_quantizers;
    dec->codebook_size           = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.codebook_size");
    dec->codebook_dim_internal   = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.codebook_dim_internal");
    dec->hidden                  = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.decoder.vector_quantization_hidden_dim");

    if (dec->num_acoustic_quantizers > RVQ_MAX_CODEBOOKS_PER_GROUP) {
        fprintf(stderr, "[Quantizer] FATAL: %d acoustic codebooks exceeds compile-time max %d\n",
                dec->num_acoustic_quantizers, RVQ_MAX_CODEBOOKS_PER_GROUP);
        return false;
    }

    int n_tensors = (dec->num_semantic_quantizers + 1)    // semantic codebooks + out_proj
                    + (dec->num_acoustic_quantizers + 1)  // acoustic codebooks + out_proj
                    + 4;                                  // headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    dec->semantic.num_codebooks = dec->num_semantic_quantizers;
    dec->semantic.out_proj_w    = qwen_load_proj_1x1(&wctx, gf, "tok_dec.vq_first.output_proj.weight");
    for (int k = 0; k < dec->semantic.num_codebooks; k++) {
        char name[128];
        snprintf(name, sizeof(name), "tok_dec.vq_first.%d.codebook", k);
        dec->semantic.embed[k] = gf_load_tensor(&wctx, gf, name);
    }

    dec->acoustic.num_codebooks = dec->num_acoustic_quantizers;
    dec->acoustic.out_proj_w    = qwen_load_proj_1x1(&wctx, gf, "tok_dec.vq_rest.output_proj.weight");
    for (int k = 0; k < dec->acoustic.num_codebooks; k++) {
        char name[128];
        snprintf(name, sizeof(name), "tok_dec.vq_rest.%d.codebook", k);
        dec->acoustic.embed[k] = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[Quantizer] FATAL: backend allocation failed\n");
        return false;
    }
    dec->weight_ctx = wctx.ctx;
    dec->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[Quantizer] Loaded: %d codebooks (%d semantic + %d acoustic), "
            "%d entries x %d dim, hidden %d\n",
            dec->num_quantizers, dec->num_semantic_quantizers, dec->num_acoustic_quantizers, dec->codebook_size,
            dec->codebook_dim_internal, dec->hidden);
    return true;
}

static void quant_decoder_free(QwenQuantizerDecoder * dec) {
    if (dec->weight_buf) {
        ggml_backend_buffer_free(dec->weight_buf);
        dec->weight_buf = NULL;
    }
    if (dec->weight_ctx) {
        ggml_free(dec->weight_ctx);
        dec->weight_ctx = NULL;
    }
}

// Sum F.embedding(codes[:, k], embed_k) across the codebooks of one
// split, then project from internal_dim (256) to hidden (512) via a
// Conv1d 1x1 (mat_mul against out_proj_w).
//
// codes_split: [T, K] i32, K is the codebook count of this split
// returns     : [hidden, T] f32
static struct ggml_tensor * rvq_group_decode(struct ggml_context * ctx,
                                             const QwenRVQGroup &  g,
                                             struct ggml_tensor *  codes_split,
                                             int                   T) {
    struct ggml_tensor * sum = NULL;
    for (int k = 0; k < g.num_codebooks; k++) {
        struct ggml_tensor * idx = ggml_view_1d(ctx, codes_split, T, (size_t) k * codes_split->nb[1]);
        struct ggml_tensor * emb = ggml_get_rows(ctx, g.embed[k], idx);
        sum                      = (sum == NULL) ? emb : ggml_add(ctx, sum, emb);
    }
    // sum: [internal_dim=256, T]
    // out_proj_w: [internal_dim=256, hidden=512]
    // ggml_mul_mat returns [hidden=512, T]
    return ggml_mul_mat(ctx, g.out_proj_w, sum);
}

// codes: [T, num_quantizers=16] i32
// returns: [hidden=512, T] f32
static struct ggml_tensor * quant_decode(struct ggml_context *        ctx,
                                         const QwenQuantizerDecoder * dec,
                                         struct ggml_tensor *         codes) {
    int T = (int) codes->ne[0];
    if ((int) codes->ne[1] != dec->num_quantizers) {
        fprintf(stderr, "[Quantizer] FATAL: codes ne[1]=%lld != num_quantizers=%d\n", (long long) codes->ne[1],
                dec->num_quantizers);
        return NULL;
    }

    struct ggml_tensor * codes_sem = ggml_view_2d(ctx, codes, T, dec->num_semantic_quantizers, codes->nb[1], 0);
    size_t               aco_off   = (size_t) dec->num_semantic_quantizers * codes->nb[1];
    struct ggml_tensor * codes_aco = ggml_view_2d(ctx, codes, T, dec->num_acoustic_quantizers, codes->nb[1], aco_off);

    struct ggml_tensor * h_sem = rvq_group_decode(ctx, dec->semantic, codes_sem, T);
    struct ggml_tensor * h_aco = rvq_group_decode(ctx, dec->acoustic, codes_aco, T);

    return ggml_add(ctx, h_sem, h_aco);
}

// Streaming variant for the static frame graph computed directly on the
// backend, without the scheduler's per view input duplication: every
// codebook id passes through a ggml_cont so the get_rows sources land
// on allocator aligned tensors, which the Vulkan get_rows path
// requires. Same math and structure as quant_decode.
static struct ggml_tensor * rvq_group_decode_stream(struct ggml_context * ctx,
                                                    const QwenRVQGroup &  g,
                                                    struct ggml_tensor *  codes_split,
                                                    int                   T) {
    struct ggml_tensor * sum = NULL;
    for (int k = 0; k < g.num_codebooks; k++) {
        struct ggml_tensor * idx = ggml_cont(ctx, ggml_view_1d(ctx, codes_split, T, (size_t) k * codes_split->nb[1]));
        struct ggml_tensor * emb = ggml_get_rows(ctx, g.embed[(size_t) k], idx);
        sum                      = (sum == NULL) ? emb : ggml_add(ctx, sum, emb);
    }
    return ggml_mul_mat(ctx, g.out_proj_w, sum);
}

static struct ggml_tensor * quant_decode_stream(struct ggml_context *        ctx,
                                                const QwenQuantizerDecoder * dec,
                                                struct ggml_tensor *         codes) {
    int T = (int) codes->ne[0];

    struct ggml_tensor * codes_sem = ggml_view_2d(ctx, codes, T, dec->num_semantic_quantizers, codes->nb[1], 0);
    size_t               aco_off   = (size_t) dec->num_semantic_quantizers * codes->nb[1];
    struct ggml_tensor * codes_aco = ggml_view_2d(ctx, codes, T, dec->num_acoustic_quantizers, codes->nb[1], aco_off);

    struct ggml_tensor * h_sem = rvq_group_decode_stream(ctx, dec->semantic, codes_sem, T);
    struct ggml_tensor * h_aco = rvq_group_decode_stream(ctx, dec->acoustic, codes_aco, T);

    return ggml_add(ctx, h_sem, h_aco);
}

// Batched variant over M lanes: codes is [T, K, M] lane major. Every
// codebook's T ids of every lane gather flat as [T * M] through a
// strided view + cont (T contiguous inside a lane, lanes strided by
// K * T), keeping the same alignment safe get_rows chain as the single
// lane path. Returns [hidden, T, M] C-first.
static struct ggml_tensor * rvq_group_decode_stream_batch(struct ggml_context * ctx,
                                                          const QwenRVQGroup &  g,
                                                          struct ggml_tensor *  codes,
                                                          int                   k0,
                                                          int                   T,
                                                          int                   M) {
    struct ggml_tensor * sum = NULL;
    for (int k = 0; k < g.num_codebooks; k++) {
        // Lane strided [T, M] view of codebook k0 + k, made flat [T * M].
        struct ggml_tensor * idx2 = ggml_view_2d(ctx, codes, T, M, codes->nb[2], (size_t) (k0 + k) * codes->nb[1]);
        struct ggml_tensor * idx  = ggml_reshape_1d(ctx, ggml_cont(ctx, idx2), (int64_t) T * M);
        struct ggml_tensor * emb  = ggml_get_rows(ctx, g.embed[(size_t) k], idx);  // [hidden_in, T * M]
        sum                       = (sum == NULL) ? emb : ggml_add(ctx, sum, emb);
    }
    return ggml_mul_mat(ctx, g.out_proj_w, sum);  // [hidden, T * M]
}

static struct ggml_tensor * quant_decode_stream_batch(struct ggml_context *        ctx,
                                                      const QwenQuantizerDecoder * dec,
                                                      struct ggml_tensor *         codes) {
    int T = (int) codes->ne[0];
    int M = (int) codes->ne[2];

    struct ggml_tensor * h_sem = rvq_group_decode_stream_batch(ctx, dec->semantic, codes, 0, T, M);
    struct ggml_tensor * h_aco =
        rvq_group_decode_stream_batch(ctx, dec->acoustic, codes, dec->num_semantic_quantizers, T, M);

    struct ggml_tensor * h = ggml_add(ctx, h_sem, h_aco);
    return ggml_reshape_3d(ctx, h, h->ne[0], T, M);
}
