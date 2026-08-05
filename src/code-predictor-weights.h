#pragma once
// code-predictor-weights.h: 5-layer Qwen3 stack that predicts the
// acoustic codebooks 1..15 of every audio frame conditioned on the
// Talker hidden state and the codebook 0 token just sampled.
//
// Architecture mirrors the Talker block (pre-norm, GQA attention with
// QK-norm, SwiGLU MLP) with one important difference: RoPE is plain
// 1D (half-split, neox-style in GGUF terms) at freq base 1e6, not the
// multimodal interleaved variant the Talker uses.
//
// The MTP head carries fifteen private embedding tables and fifteen
// private linear heads, one pair per acoustic codebook. The talker
// codebook 0 stays handled by talker.codec_embedding and talker.codec_head.
//
// Tensor naming (convert.py output) :
//   code_predictor.norm.weight                                [hidden]
//   code_predictor.codec_embedding.{0..14}.weight             [vocab, hidden]
//   code_predictor.lm_head.{0..14}.weight                     [vocab, hidden]
//   code_predictor.layers.{0..N-1}.input_layernorm.weight     [hidden]
//   code_predictor.layers.{0..N-1}.post_attention_layernorm.weight    [hidden]
//   code_predictor.layers.{0..N-1}.attn.{q,k,v,o}_proj.weight
//   code_predictor.layers.{0..N-1}.attn.{q,k}_norm.weight     [head_dim]
//   code_predictor.layers.{0..N-1}.mlp.{gate,up,down}_proj.weight

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "talker-weights.h"
#include "weight-ctx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct CodePredictorWeights {
    int   hidden_size;
    int   intermediate_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   head_dim;
    int   vocab_size;
    int   num_acoustic_codebooks;  // num_code_groups - 1
    float rope_theta;
    float rms_norm_eps;

    struct ggml_tensor *              norm_w;
    std::vector<struct ggml_tensor *> codec_embedding;  // size num_acoustic_codebooks
    std::vector<struct ggml_tensor *> lm_head;          // size num_acoustic_codebooks

    // Optional small_to_mtp projection that brings the talker hidden
    // dimension down to the predictor hidden dimension when the two
    // differ (1.7B-base case: 2048 -> 1024). Both tensors are NULL when
    // the upstream sets nn.Identity() i.e. talker_hidden == predictor_hidden
    // (0.6B case). Loaded with gf_try_load_tensor so absence is silent.
    struct ggml_tensor * mtp_proj_w;
    struct ggml_tensor * mtp_proj_b;

    // Layers reuse the same TalkerLayer struct since the per-layer
    // tensor set is identical. Only the model-level wiring differs.
    std::vector<TalkerLayer> layers;

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool code_predictor_weights_load(CodePredictorWeights * cw, const GGUFModel & gf, ggml_backend_t backend) {
    cw->hidden_size         = (int) gf_get_u32(gf, "qwen3-tts.code_pred.embedding_length");
    cw->intermediate_size   = (int) gf_get_u32(gf, "qwen3-tts.code_pred.feed_forward_length");
    cw->num_hidden_layers   = (int) gf_get_u32(gf, "qwen3-tts.code_pred.block_count");
    cw->num_attention_heads = (int) gf_get_u32(gf, "qwen3-tts.code_pred.attention.head_count");
    cw->num_key_value_heads = (int) gf_get_u32(gf, "qwen3-tts.code_pred.attention.head_count_kv");
    cw->head_dim            = (int) gf_get_u32(gf, "qwen3-tts.code_pred.attention.key_length");
    cw->vocab_size          = (int) gf_get_u32(gf, "qwen3-tts.code_pred.vocab_size");
    cw->rope_theta          = gf_get_f32(gf, "qwen3-tts.code_pred.rope.freq_base");
    cw->rms_norm_eps        = gf_get_f32(gf, "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon");

    int num_code_groups = (int) gf_get_u32(gf, "qwen3-tts.num_code_groups");
    if (num_code_groups <= 1) {
        fprintf(stderr, "[CodePredictor] FATAL: invalid num_code_groups=%d\n", num_code_groups);
        return false;
    }
    cw->num_acoustic_codebooks = num_code_groups - 1;

    if (cw->num_hidden_layers <= 0 || cw->hidden_size <= 0) {
        fprintf(stderr, "[CodePredictor] FATAL: invalid hyperparameters (layers=%d hidden=%d)\n", cw->num_hidden_layers,
                cw->hidden_size);
        return false;
    }

    cw->layers.resize((size_t) cw->num_hidden_layers);
    cw->codec_embedding.resize((size_t) cw->num_acoustic_codebooks);
    cw->lm_head.resize((size_t) cw->num_acoustic_codebooks);

    int n_tensors = 1                                 // final norm
                    + 2                               // mtp_proj weight + bias (when present)
                    + 2 * cw->num_acoustic_codebooks  // 15 embeds + 15 heads
                    + cw->num_hidden_layers * 11      // 11 per layer
                    + 8;                              // headroom
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    cw->norm_w = gf_load_tensor(&wctx, gf, "code_pred.output_norm.weight");

    // Optional projection talker_hidden -> predictor_hidden. Absent in
    // checkpoints where talker.hidden_size == code_pred.hidden_size
    // because upstream uses nn.Identity in that case.
    cw->mtp_proj_w = gf_try_load_tensor(&wctx, gf, "code_pred.mtp_proj.weight");
    cw->mtp_proj_b = gf_try_load_tensor(&wctx, gf, "code_pred.mtp_proj.bias");

    for (int g = 0; g < cw->num_acoustic_codebooks; g++) {
        char name[160];
        snprintf(name, sizeof(name), "code_pred.codec_embd.%d.weight", g);
        cw->codec_embedding[(size_t) g] = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.lm_head.%d.weight", g);
        cw->lm_head[(size_t) g] = gf_load_tensor(&wctx, gf, name);
    }

    for (int l = 0; l < cw->num_hidden_layers; l++) {
        TalkerLayer & layer = cw->layers[(size_t) l];
        char          name[160];

        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_norm.weight", l);
        layer.input_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "code_pred.blk.%d.ffn_norm.weight", l);
        layer.post_attn_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_q.weight", l);
        layer.attn.q_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_k.weight", l);
        layer.attn.k_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_v.weight", l);
        layer.attn.v_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_output.weight", l);
        layer.attn.o_proj_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_q_norm.weight", l);
        layer.attn.q_norm_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.attn_k_norm.weight", l);
        layer.attn.k_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "code_pred.blk.%d.ffn_gate.weight", l);
        layer.mlp.gate_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.ffn_up.weight", l);
        layer.mlp.up_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "code_pred.blk.%d.ffn_down.weight", l);
        layer.mlp.down_proj_w = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[CodePredictor] FATAL: backend allocation failed\n");
        return false;
    }
    cw->weight_ctx = wctx.ctx;
    cw->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[CodePredictor] Loaded: %d layers, hidden %d, heads %d/%d, head_dim %d, "
            "FFN %d, RoPE theta %.0f, %d acoustic codebooks (vocab %d each), mtp_proj %s\n",
            cw->num_hidden_layers, cw->hidden_size, cw->num_attention_heads, cw->num_key_value_heads, cw->head_dim,
            cw->intermediate_size, (double) cw->rope_theta, cw->num_acoustic_codebooks, cw->vocab_size,
            cw->mtp_proj_w ? "linear" : "identity");
    return true;
}

static void code_predictor_weights_free(CodePredictorWeights * cw) {
    if (cw->weight_buf) {
        ggml_backend_buffer_free(cw->weight_buf);
        cw->weight_buf = NULL;
    }
    if (cw->weight_ctx) {
        ggml_free(cw->weight_ctx);
        cw->weight_ctx = NULL;
    }
    cw->layers.clear();
    cw->codec_embedding.clear();
    cw->lm_head.clear();
}
