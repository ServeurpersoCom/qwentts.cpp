#pragma once
// quantizer-encode.h: RVQ encode for the Qwen3-TTS 12Hz tokenizer.
//
// Inputs: hidden state [T, 512] f32 T-first, output of the encoder
// downsample. Outputs: codes [16, T] i32, with codebook 0 carrying the
// semantic stream and codebooks 1..15 carrying the acoustic residual
// stream.
//
// Each side has the same shape:
//   input_proj  : Conv1d k=1, 512 -> 256 (linear projection on channels)
//   codebooks   : list of [2048, 256] f32 entries used as kNN centroids
//   output_proj: Conv1d k=1, 256 -> 512 (used only inside the residual loop)
//
// At encode time we run, for each side:
//   y     = input_proj(x)
//   res   = y
//   codes = []
//   for layer in layers:
//     idx     = argmin_e ||res - codebook_e||^2
//     q       = codebook[idx]
//     res     = res - q
//     codes  += [idx]
//
// The semantic side has 1 codebook, the acoustic side has 15 codebooks,
// concatenated to produce the final 16-codebook stream.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cstdlib>
#include <string>

#define QUANT_ENC_NUM_SEMANTIC 1
#define QUANT_ENC_NUM_ACOUSTIC 15
#define QUANT_ENC_TOTAL        (QUANT_ENC_NUM_SEMANTIC + QUANT_ENC_NUM_ACOUSTIC)

struct QwenQuantizerEncodeSide {
    struct ggml_tensor * input_proj_w;                       // [1, 512, 256] f32, k=1 conv
    struct ggml_tensor * output_proj_w;                      // [1, 256, 512] f32, k=1 conv
    int                  num_layers;
    struct ggml_tensor * codebooks[QUANT_ENC_NUM_ACOUSTIC];  // [256, 2048] each
};

struct QwenQuantizerEncode {
    QwenQuantizerEncodeSide semantic;  // 1 codebook
    QwenQuantizerEncodeSide acoustic;  // 15 codebooks

    int codebook_size;                 // 2048
    int codebook_dim;                  // 256
    int hidden_size;                   // 512

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool quant_encode_load(QwenQuantizerEncode * q, const GGUFModel & gf, ggml_backend_t backend) {
    q->codebook_size = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.codebook_size");
    q->codebook_dim  = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.vector_quantization_hidden_dim");
    q->hidden_size   = (int) gf_get_u32(gf, "qwen3-tts-tokenizer.encoder.hidden_size");

    int       n_tensors = 4 + QUANT_ENC_TOTAL + 4;  // 4 proj + 16 codebooks + headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    q->semantic.num_layers    = QUANT_ENC_NUM_SEMANTIC;
    q->semantic.input_proj_w  = gf_load_tensor(&wctx, gf, "tok_enc.vq_semantic.input_proj.weight");
    q->semantic.output_proj_w = gf_load_tensor(&wctx, gf, "tok_enc.vq_semantic.output_proj.weight");
    for (int i = 0; i < QUANT_ENC_NUM_SEMANTIC; i++) {
        char name[96];
        snprintf(name, sizeof(name), "tok_enc.vq_semantic.%d.codebook", i);
        q->semantic.codebooks[i] = gf_load_tensor(&wctx, gf, name);
    }

    q->acoustic.num_layers    = QUANT_ENC_NUM_ACOUSTIC;
    q->acoustic.input_proj_w  = gf_load_tensor(&wctx, gf, "tok_enc.vq_acoustic.input_proj.weight");
    q->acoustic.output_proj_w = gf_load_tensor(&wctx, gf, "tok_enc.vq_acoustic.output_proj.weight");
    for (int i = 0; i < QUANT_ENC_NUM_ACOUSTIC; i++) {
        char name[96];
        snprintf(name, sizeof(name), "tok_enc.vq_acoustic.%d.codebook", i);
        q->acoustic.codebooks[i] = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        qt_log(QT_LOG_ERROR, "[EncQuantizer] FATAL: backend allocation failed");
        return false;
    }
    q->weight_ctx = wctx.ctx;
    q->weight_buf = wctx.buffer;

    qt_log(QT_LOG_INFO,
           "[EncQuantizer] Loaded: %d codebooks (%d semantic + %d acoustic), %d entries x %d dim, "
           "hidden %d, weights %.1f MB",
           QUANT_ENC_TOTAL, QUANT_ENC_NUM_SEMANTIC, QUANT_ENC_NUM_ACOUSTIC, q->codebook_size, q->codebook_dim,
           q->hidden_size, (float) ggml_backend_buffer_get_size(q->weight_buf) / (1024.0f * 1024.0f));
    return true;
}

static void quant_encode_free(QwenQuantizerEncode * q) {
    if (q->weight_buf) {
        ggml_backend_buffer_free(q->weight_buf);
        q->weight_buf = NULL;
    }
    if (q->weight_ctx) {
        ggml_free(q->weight_ctx);
        q->weight_ctx = NULL;
    }
}

// CPU-side host buffers for the RVQ encode loop. Read once from the
// backend at first use and reused across encode calls. The codebooks
// stay on the backend for any GPU-side use, this is just a CPU mirror
// for the per-frame argmin.
struct QwenQuantizerEncodeHost {
    int                             num_layers;
    int                             codebook_size;
    int                             codebook_dim;
    int                             hidden_size;
    // input_proj as a row-major [in=hidden_size, out=codebook_dim] Linear.
    std::vector<float>              input_proj;
    // codebooks[l]: [codebook_size, codebook_dim] row-major
    std::vector<std::vector<float>> codebooks;
    // sqsum[l]: [codebook_size], precomputed ||c_e||^2 for the argmin trick
    std::vector<std::vector<float>> sqsum;
    // output_proj as a row-major [in=codebook_dim, out=hidden_size] Linear.
    std::vector<float>              output_proj;
};

static void quant_encode_host_load(QwenQuantizerEncodeHost *       h,
                                   const QwenQuantizerEncodeSide & side,
                                   int                             codebook_size,
                                   int                             codebook_dim,
                                   int                             hidden_size) {
    h->num_layers    = side.num_layers;
    h->codebook_size = codebook_size;
    h->codebook_dim  = codebook_dim;
    h->hidden_size   = hidden_size;

    // input_proj weight has ggml shape [1, hidden_size, codebook_dim] (k=1 conv).
    // The contiguous memory is [in=hidden_size, out=codebook_dim] row-major.
    h->input_proj.resize((size_t) hidden_size * (size_t) codebook_dim);
    ggml_backend_tensor_get(side.input_proj_w, h->input_proj.data(), 0, h->input_proj.size() * sizeof(float));

    h->output_proj.resize((size_t) codebook_dim * (size_t) hidden_size);
    ggml_backend_tensor_get(side.output_proj_w, h->output_proj.data(), 0, h->output_proj.size() * sizeof(float));

    h->codebooks.resize((size_t) side.num_layers);
    h->sqsum.resize((size_t) side.num_layers);
    for (int l = 0; l < side.num_layers; l++) {
        h->codebooks[l].resize((size_t) codebook_size * (size_t) codebook_dim);
        ggml_backend_tensor_get(side.codebooks[l], h->codebooks[l].data(), 0, h->codebooks[l].size() * sizeof(float));

        h->sqsum[l].resize((size_t) codebook_size);
        for (int e = 0; e < codebook_size; e++) {
            float         s = 0.0f;
            const float * c = h->codebooks[l].data() + (size_t) e * (size_t) codebook_dim;
            for (int d = 0; d < codebook_dim; d++) {
                s += c[d] * c[d];
            }
            h->sqsum[l][e] = s;
        }
    }
}

// Apply a Conv1d k=1 weight (PyTorch shape [out, in, 1], ggml ne=(1, in, out))
// to a [N, in] row-major buffer, producing a [N, out] row-major buffer.
// The contiguous memory of the ggml weight walks `in` fast and `out` slow,
// matching the numpy view as [out, in] row-major. So `w[o*in + i]` selects
// row o, column i of the underlying [out, in] matrix.
static void quant_encode_linear(const float * w, int in_dim, int out_dim, const float * x, int N, float * y) {
    for (int n = 0; n < N; n++) {
        const float * xn = x + (size_t) n * (size_t) in_dim;
        float *       yn = y + (size_t) n * (size_t) out_dim;
        for (int o = 0; o < out_dim; o++) {
            float         acc   = 0.0f;
            const float * w_row = w + (size_t) o * (size_t) in_dim;
            for (int i = 0; i < in_dim; i++) {
                acc += xn[i] * w_row[i];
            }
            yn[o] = acc;
        }
    }
}

// One RVQ side encode loop. Mutates `res` in-place as the residual stream.
// Appends T frames of codebook indices for each of side.num_layers, in
// the order: layer_0[0..T], layer_1[0..T], ..., layer_{L-1}[0..T].
static void quant_encode_side_loop(const QwenQuantizerEncodeHost * h,
                                   std::vector<float> &            res,
                                   int                             T,
                                   std::vector<int32_t> &          codes_out) {
    int D = h->codebook_dim;
    int E = h->codebook_size;

    std::vector<int32_t> layer_codes((size_t) T);
    for (int l = 0; l < h->num_layers; l++) {
        const float * codebook = h->codebooks[l].data();
        const float * cb_sqsum = h->sqsum[l].data();

        // For each frame t: idx = argmin_e (||c_e||^2 - 2 <res_t, c_e>)
        for (int t = 0; t < T; t++) {
            const float * r       = res.data() + (size_t) t * (size_t) D;
            int           best_e  = 0;
            float         best_sc = INFINITY;
            for (int e = 0; e < E; e++) {
                const float * c   = codebook + (size_t) e * (size_t) D;
                float         dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += r[d] * c[d];
                }
                float sc = cb_sqsum[e] - 2.0f * dot;
                if (sc < best_sc) {
                    best_sc = sc;
                    best_e  = e;
                }
            }
            layer_codes[t]  = best_e;
            // Subtract centroid from residual in place
            const float * c = codebook + (size_t) best_e * (size_t) D;
            for (int d = 0; d < D; d++) {
                res[(size_t) t * (size_t) D + (size_t) d] -= c[d];
            }
        }
        codes_out.insert(codes_out.end(), layer_codes.begin(), layer_codes.end());
    }
}

// Full RVQ encode. Takes the post-downsample hidden [T, hidden_size] f32
// row-major buffer and returns flat codes [K, T] row-major, where K is
// QUANT_ENC_TOTAL = 16.
//   hidden: [T, hidden_size] f32 row-major (T fast in pseudo, but here
//            row-major means index = t*hidden + c, t slow, c fast)
//
// Returns codes flat as [16, T] row-major: codes[k*T + t].
static std::vector<int32_t> quant_encode_cpu(const QwenQuantizerEncodeHost * sem,
                                             const QwenQuantizerEncodeHost * aco,
                                             const float *                   hidden,
                                             int                             T) {
    std::vector<int32_t> codes;
    codes.reserve((size_t) QUANT_ENC_TOTAL * (size_t) T);

    // Project hidden to codebook_dim for each side independently.
    int D = sem->codebook_dim;

    // Semantic side
    std::vector<float> proj_sem((size_t) T * (size_t) D);
    quant_encode_linear(sem->input_proj.data(), sem->hidden_size, D, hidden, T, proj_sem.data());
    quant_encode_side_loop(sem, proj_sem, T, codes);

    // Acoustic side
    std::vector<float> proj_aco((size_t) T * (size_t) D);
    quant_encode_linear(aco->input_proj.data(), aco->hidden_size, D, hidden, T, proj_aco.data());
    quant_encode_side_loop(aco, proj_aco, T, codes);

    return codes;
}
