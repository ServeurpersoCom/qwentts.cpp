#pragma once
// talker-weights.h: Qwen3-style autoregressive Talker LM weights.
//
// Carries 28 decoder layers in 0.6B (36 in 1.7B), each with pre-norm
// attention plus pre-norm SwiGLU MLP. Attention is multi-head with GQA
// (16 query heads, 8 kv heads, head_dim 128) and per-head QK-norm. RoPE
// is mrope-interleaved with sections [24, 20, 20] and freq base 1e6;
// in TTS-only mode this collapses to plain interleaved 1D RoPE since
// the three multimodal axes carry the same position index.
//
// Top-level the talker holds two embedding tables (codec vocab 3072,
// text vocab 151936 with hidden 2048), a 2-layer ResizeMLP that
// projects text embeddings down to hidden 1024, the final RMSNorm and
// a codec_head Linear 1024 -> 3072 that emits the codebook 0 logits.
//
// Tensor naming follows convert.py output (flat, talker.*) :
//   talker.codec_embedding.weight              [3072, 1024]
//   talker.text_embedding.weight               [151936, 2048]
//   talker.text_projection.fc1.{weight,bias}   [2048, 2048] / [2048]
//   talker.text_projection.fc2.{weight,bias}   [2048, 1024] / [1024]
//   talker.codec_head.weight                   [3072, 1024]
//   talker.norm.weight                         [1024]
//   talker.layers.{0..N-1}.input_layernorm.weight              [hidden]
//   talker.layers.{0..N-1}.post_attention_layernorm.weight     [hidden]
//   talker.layers.{0..N-1}.attn.q_proj.weight                  [hidden, n_heads*head_dim]
//   talker.layers.{0..N-1}.attn.k_proj.weight                  [hidden, n_kv_heads*head_dim]
//   talker.layers.{0..N-1}.attn.v_proj.weight                  [hidden, n_kv_heads*head_dim]
//   talker.layers.{0..N-1}.attn.o_proj.weight                  [n_heads*head_dim, hidden]
//   talker.layers.{0..N-1}.attn.q_norm.weight                  [head_dim]
//   talker.layers.{0..N-1}.attn.k_norm.weight                  [head_dim]
//   talker.layers.{0..N-1}.mlp.gate_proj.weight                [hidden, intermediate]
//   talker.layers.{0..N-1}.mlp.up_proj.weight                  [hidden, intermediate]
//   talker.layers.{0..N-1}.mlp.down_proj.weight                [intermediate, hidden]

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct TalkerAttention {
    struct ggml_tensor * q_proj_w;
    struct ggml_tensor * k_proj_w;
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;
    struct ggml_tensor * q_norm_w;
    struct ggml_tensor * k_norm_w;
};

struct TalkerMLP {
    struct ggml_tensor * gate_proj_w;
    struct ggml_tensor * up_proj_w;
    struct ggml_tensor * down_proj_w;
};

struct TalkerLayer {
    struct ggml_tensor * input_norm_w;
    TalkerAttention      attn;
    struct ggml_tensor * post_attn_norm_w;
    TalkerMLP            mlp;
};

struct TalkerWeights {
    int   hidden_size;
    int   intermediate_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   head_dim;
    int   vocab_size;
    float rope_theta;
    float rms_norm_eps;
    int   mrope_section_t;
    int   mrope_section_h;
    int   mrope_section_w;
    bool  mrope_interleaved;

    struct ggml_tensor * codec_embedding;
    struct ggml_tensor * text_embedding;
    struct ggml_tensor * text_proj_fc1_w;
    struct ggml_tensor * text_proj_fc1_b;
    struct ggml_tensor * text_proj_fc2_w;
    struct ggml_tensor * text_proj_fc2_b;
    struct ggml_tensor * codec_head_w;
    struct ggml_tensor * norm_w;

    std::vector<TalkerLayer> layers;

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

static bool talker_weights_load(TalkerWeights * tw, const GGUFModel & gf, ggml_backend_t backend) {
    tw->hidden_size         = (int) gf_get_u32(gf, "qwen3-tts.talker.embedding_length");
    tw->intermediate_size   = (int) gf_get_u32(gf, "qwen3-tts.talker.feed_forward_length");
    tw->num_hidden_layers   = (int) gf_get_u32(gf, "qwen3-tts.talker.block_count");
    tw->num_attention_heads = (int) gf_get_u32(gf, "qwen3-tts.talker.attention.head_count");
    tw->num_key_value_heads = (int) gf_get_u32(gf, "qwen3-tts.talker.attention.head_count_kv");
    tw->head_dim            = (int) gf_get_u32(gf, "qwen3-tts.talker.attention.key_length");
    tw->vocab_size          = (int) gf_get_u32(gf, "qwen3-tts.talker.vocab_size");
    tw->rope_theta          = gf_get_f32(gf, "qwen3-tts.talker.rope.freq_base");
    tw->rms_norm_eps        = gf_get_f32(gf, "qwen3-tts.talker.attention.layer_norm_rms_epsilon");
    tw->mrope_interleaved   = gf_get_bool(gf, "qwen3-tts.talker.rope.mrope_interleaved");

    std::vector<uint32_t> mrope = gf_get_array_u32(gf, "qwen3-tts.talker.rope.mrope_section");
    if (mrope.size() == 3) {
        tw->mrope_section_t = (int) mrope[0];
        tw->mrope_section_h = (int) mrope[1];
        tw->mrope_section_w = (int) mrope[2];
    } else {
        tw->mrope_section_t = tw->mrope_section_h = tw->mrope_section_w = 0;
    }

    if (tw->num_hidden_layers <= 0 || tw->hidden_size <= 0) {
        fprintf(stderr, "[Talker] FATAL: invalid hyperparameters in GGUF (layers=%d hidden=%d)\n",
                tw->num_hidden_layers, tw->hidden_size);
        return false;
    }

    tw->layers.resize((size_t) tw->num_hidden_layers);

    // 8 top-level + per layer (2 norms + 4 attn + 2 qk norms + 3 mlp) = 11
    int       n_tensors = 8 + tw->num_hidden_layers * 11 + 8;
    WeightCtx wctx;
    wctx_init(&wctx, n_tensors);

    tw->codec_embedding = gf_load_tensor(&wctx, gf, "talker.codec_embd.weight");
    tw->text_embedding  = gf_load_tensor(&wctx, gf, "talker.text_embd.weight");
    tw->text_proj_fc1_w = gf_load_tensor(&wctx, gf, "talker.text_proj.fc1.weight");
    tw->text_proj_fc1_b = gf_load_tensor(&wctx, gf, "talker.text_proj.fc1.bias");
    tw->text_proj_fc2_w = gf_load_tensor(&wctx, gf, "talker.text_proj.fc2.weight");
    tw->text_proj_fc2_b = gf_load_tensor(&wctx, gf, "talker.text_proj.fc2.bias");
    tw->codec_head_w    = gf_load_tensor(&wctx, gf, "talker.codec_head.weight");
    tw->norm_w          = gf_load_tensor(&wctx, gf, "talker.output_norm.weight");

    for (int l = 0; l < tw->num_hidden_layers; l++) {
        TalkerLayer & layer = tw->layers[(size_t) l];
        char          name[160];

        snprintf(name, sizeof(name), "talker.blk.%d.attn_norm.weight", l);
        layer.input_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "talker.blk.%d.ffn_norm.weight", l);
        layer.post_attn_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "talker.blk.%d.attn_q.weight", l);
        layer.attn.q_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.attn_k.weight", l);
        layer.attn.k_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.attn_v.weight", l);
        layer.attn.v_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.attn_output.weight", l);
        layer.attn.o_proj_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "talker.blk.%d.attn_q_norm.weight", l);
        layer.attn.q_norm_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.attn_k_norm.weight", l);
        layer.attn.k_norm_w = gf_load_tensor(&wctx, gf, name);

        snprintf(name, sizeof(name), "talker.blk.%d.ffn_gate.weight", l);
        layer.mlp.gate_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.ffn_up.weight", l);
        layer.mlp.up_proj_w = gf_load_tensor(&wctx, gf, name);
        snprintf(name, sizeof(name), "talker.blk.%d.ffn_down.weight", l);
        layer.mlp.down_proj_w = gf_load_tensor(&wctx, gf, name);
    }

    if (!wctx_alloc(&wctx, backend)) {
        fprintf(stderr, "[Talker] FATAL: backend allocation failed\n");
        return false;
    }
    tw->weight_ctx = wctx.ctx;
    tw->weight_buf = wctx.buffer;

    fprintf(stderr,
            "[Talker] Loaded: %d layers, hidden %d, heads %d/%d, head_dim %d, "
            "FFN %d, RoPE theta %.0f, mrope sections [%d,%d,%d] interleaved=%d\n",
            tw->num_hidden_layers, tw->hidden_size, tw->num_attention_heads, tw->num_key_value_heads, tw->head_dim,
            tw->intermediate_size, (double) tw->rope_theta, tw->mrope_section_t, tw->mrope_section_h,
            tw->mrope_section_w, (int) tw->mrope_interleaved);
    return true;
}

static void talker_weights_free(TalkerWeights * tw) {
    if (tw->weight_buf) {
        ggml_backend_buffer_free(tw->weight_buf);
        tw->weight_buf = NULL;
    }
    if (tw->weight_ctx) {
        ggml_free(tw->weight_ctx);
        tw->weight_ctx = NULL;
    }
    tw->layers.clear();
}
