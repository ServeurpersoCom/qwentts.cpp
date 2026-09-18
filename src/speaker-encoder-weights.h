#pragma once
// speaker-encoder-weights.h: ECAPA-TDNN x-vector extractor used by the
// Base checkpoint to condition the Talker on a reference voice.
//
// Topology (from qwen_tts.core.models.modeling_qwen3_tts) :
//
//   conv0          TDNN k=5,  128 -> 512    (initial frontend)
//   blk[1..3]      SE-Res2Net (TDNN1 + Res2Net 8-branch + TDNN2 + SE)
//   mfa            TDNN k=1, 1536 -> 1536   (cat of blk1..3)
//   asp            attentive statistical pooling, 1536 -> 3072
//   fc             Conv1d k=1, 3072 -> 2048
//
// Weights live on the talker backend buffer next to the talker LM.
// All tensors are stored F32 in the source GGUF and stay F32 when
// quantizing because should_quantize keeps spk_enc as is (small
// channel counts make quantization meaningless here).
//
// Constants: enc_dim 2048 (size of the speaker embedding fed into
// the codec_prefill slot), input mel_dim 128, ECAPA hidden 512,
// res2net scale 8 -> 7 dilated TDNN branches, se hidden 128,
// asp attention 128.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Initial TDNN block: Conv1d(in=128, out=512, k=5, padding=same, reflect)
// followed by ReLU. Stored as 3D tensor [k, in_c, out_c] in the GGUF.
struct SpkEncTDNN {
    struct ggml_tensor * weight;  // [k, in_c, out_c]
    struct ggml_tensor * bias;    // [out_c]
    int                  k;
    int                  in_c;
    int                  out_c;
};

// Squeeze-Excitation attention: conv1 (out -> se), conv2 (se -> out),
// k=1 padding=same. Operates on the temporal mean of the input then
// broadcasts a sigmoid gate back over the time axis.
struct SpkEncSE {
    struct ggml_tensor * conv1_w;  // [1, out_c, se_c]
    struct ggml_tensor * conv1_b;  // [se_c]
    struct ggml_tensor * conv2_w;  // [1, se_c, out_c]
    struct ggml_tensor * conv2_b;  // [out_c]
};

// Res2Net branch: 7 dilated TDNN k=3 conv1d, dilation comes from the
// parent SE-Res2Net block. We keep flat arrays since enc_res2net_scale
// is 8 (which yields scale - 1 = 7 branches).
struct SpkEncRes2Net {
    struct ggml_tensor * weight[7];  // each [3, in_c/8, out_c/8]
    struct ggml_tensor * bias[7];    // each [out_c/8]
};

// SE-Res2Net block: tdnn1 (k=1) -> Res2Net (k=3, dil=d) -> tdnn2 (k=1)
// -> SE attention, plus a residual add over the whole stack.
struct SpkEncBlock {
    SpkEncTDNN    tdnn1;
    SpkEncRes2Net res2net;
    SpkEncTDNN    tdnn2;
    SpkEncSE      se;
    int           dilation;
};

// Attentive Statistical Pooling: tdnn maps from 3*1536 to 128 (channels
// concat of [x, mean, std]), conv maps 128 back to 1536. The mask
// branch reduces to a no-op for unbatched single-utterance inference,
// which is the only path the C++ side exposes.
struct SpkEncASP {
    SpkEncTDNN           tdnn;    // [1, 3*hidden, attn]
    struct ggml_tensor * conv_w;  // [1, attn, hidden]
    struct ggml_tensor * conv_b;  // [hidden]
};

struct SpeakerEncoderWeights {
    // Topology constants, sourced from upstream Qwen3TTSSpeakerEncoderConfig.
    int enc_dim;        // 2048
    int sample_rate;    // 24000
    int mel_dim;        // 128
    int hidden;         // 512
    int mfa_hidden;     // 1536
    int asp_attn;       // 128
    int se_channels;    // 128
    int res2net_scale;  // 8

    // Forward path tensors. blocks[0] is the conv0 TDNN frontend held in
    // its own slot for clarity. The three SE-Res2Net stacks live in
    // blocks[1..3].
    SpkEncTDNN           conv0;      // [5, 128, 512]
    SpkEncBlock          blocks[3];  // SE-Res2Net at dilations 2, 3, 4
    SpkEncTDNN           mfa;        // [1, 1536, 1536]
    SpkEncASP            asp;
    struct ggml_tensor * fc_w;       // [1, 3072, 2048]
    struct ggml_tensor * fc_b;       // [2048]

    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
};

// Helpers to load each tensor by upstream name. The generic gf_load_tensor
// pulls the row major data, the conv weights are 3D so they keep their
// natural [k, in_c, out_c] layout which matches what ggml_im2col + matmul
// expects when we treat ne[0]=k as the spatial filter axis.
static struct ggml_tensor * spk_load(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    return gf_load_tensor(wctx, gf, name);
}

static void spk_load_tdnn(WeightCtx * wctx, const GGUFModel & gf, const std::string & prefix, SpkEncTDNN & t) {
    t.weight = spk_load(wctx, gf, prefix + ".weight");
    t.bias   = spk_load(wctx, gf, prefix + ".bias");
    t.k      = (int) t.weight->ne[0];
    t.in_c   = (int) t.weight->ne[1];
    t.out_c  = (int) t.weight->ne[2];
}

static void spk_load_se(WeightCtx * wctx, const GGUFModel & gf, const std::string & prefix, SpkEncSE & se) {
    se.conv1_w = spk_load(wctx, gf, prefix + ".conv1.weight");
    se.conv1_b = spk_load(wctx, gf, prefix + ".conv1.bias");
    se.conv2_w = spk_load(wctx, gf, prefix + ".conv2.weight");
    se.conv2_b = spk_load(wctx, gf, prefix + ".conv2.bias");
}

static void spk_load_res2net(WeightCtx * wctx, const GGUFModel & gf, const std::string & prefix, SpkEncRes2Net & rn) {
    for (int i = 0; i < 7; i++) {
        char p[64];
        std::snprintf(p, sizeof(p), "%s.%d", prefix.c_str(), i);
        rn.weight[i] = spk_load(wctx, gf, std::string(p) + ".weight");
        rn.bias[i]   = spk_load(wctx, gf, std::string(p) + ".bias");
    }
}

static void spk_load_block(WeightCtx * wctx, const GGUFModel & gf, int idx, int dilation, SpkEncBlock & blk) {
    char p[64];
    std::snprintf(p, sizeof(p), "spk_enc.blk.%d", idx);
    spk_load_tdnn(wctx, gf, std::string(p) + ".tdnn1", blk.tdnn1);
    spk_load_res2net(wctx, gf, std::string(p) + ".res2net", blk.res2net);
    spk_load_tdnn(wctx, gf, std::string(p) + ".tdnn2", blk.tdnn2);
    spk_load_se(wctx, gf, std::string(p) + ".se", blk.se);
    blk.dilation = dilation;
}

static bool speaker_encoder_weights_load(SpeakerEncoderWeights * sw, const GGUFModel & gf, ggml_backend_t backend) {
    sw->enc_dim       = (int) gf_get_u32(gf, "qwen3-tts.spk_enc.embedding_length");
    sw->sample_rate   = (int) gf_get_u32(gf, "qwen3-tts.spk_enc.sample_rate");
    sw->mel_dim       = 128;
    sw->hidden        = 512;
    sw->mfa_hidden    = 1536;
    sw->asp_attn      = 128;
    sw->se_channels   = 128;
    sw->res2net_scale = 8;

    // Probe: Base GGUFs ship the speaker encoder, CustomVoice and
    // VoiceDesign do not. A missing conv0.weight aborts cleanly.
    if (gguf_find_tensor(gf.gguf, "spk_enc.conv0.weight") < 0) {
        qt_log(QT_LOG_WARN, "[SpeakerEncoder] No spk_enc.conv0.weight, base/clone mode unavailable");
        sw->weight_ctx = NULL;
        sw->weight_buf = NULL;
        return true;
    }

    // Roughly 80 tensors total: 1 conv0 + 3 * (2 tdnn + 7 res2net + 4 se) + 1 mfa
    // + asp.tdnn + asp.conv + fc, with weight + bias each. Allocate 100 slots
    // for safety.
    WeightCtx wctx;
    wctx_init(&wctx, 100);

    spk_load_tdnn(&wctx, gf, "spk_enc.conv0", sw->conv0);
    spk_load_block(&wctx, gf, 1, 2, sw->blocks[0]);
    spk_load_block(&wctx, gf, 2, 3, sw->blocks[1]);
    spk_load_block(&wctx, gf, 3, 4, sw->blocks[2]);
    spk_load_tdnn(&wctx, gf, "spk_enc.mfa", sw->mfa);
    spk_load_tdnn(&wctx, gf, "spk_enc.asp.tdnn", sw->asp.tdnn);
    sw->asp.conv_w = spk_load(&wctx, gf, "spk_enc.asp.conv.weight");
    sw->asp.conv_b = spk_load(&wctx, gf, "spk_enc.asp.conv.bias");
    sw->fc_w       = spk_load(&wctx, gf, "spk_enc.fc.weight");
    sw->fc_b       = spk_load(&wctx, gf, "spk_enc.fc.bias");

    if (!wctx_alloc(&wctx, backend)) {
        qt_log(QT_LOG_ERROR, "[SpeakerEncoder] FATAL: backend allocation failed");
        return false;
    }
    sw->weight_ctx = wctx.ctx;
    sw->weight_buf = wctx.buffer;

    qt_log(QT_LOG_INFO,
           "[SpeakerEncoder] Loaded: enc_dim=%d sr=%d mel_dim=%d hidden=%d mfa=%d asp_attn=%d se=%d scale=%d",
           sw->enc_dim, sw->sample_rate, sw->mel_dim, sw->hidden, sw->mfa_hidden, sw->asp_attn, sw->se_channels,
           sw->res2net_scale);
    return true;
}

static void speaker_encoder_weights_free(SpeakerEncoderWeights * sw) {
    if (sw->weight_buf) {
        ggml_backend_buffer_free(sw->weight_buf);
        sw->weight_buf = NULL;
    }
    if (sw->weight_ctx) {
        ggml_free(sw->weight_ctx);
        sw->weight_ctx = NULL;
    }
}
