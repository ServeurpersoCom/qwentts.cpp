#pragma once
// speaker-encoder-forward.h : ECAPA-TDNN forward graph in GGML.
//
// Mirrors qwen_tts.core.models.modeling_qwen3_tts.Qwen3TTSSpeakerEncoder
// for the single utterance unbatched path. The forward fuses the mel
// spectrogram extraction so the whole pipeline lives in one graph :
//
//   audio [T_pad] f32
//     -> mel  [128, T_frames]              (audio-mel.h)
//     -> conv0 TDNN k=5 + ReLU             [512, T_frames]
//     -> SE-Res2Net dil=2                  [512, T_frames]
//     -> SE-Res2Net dil=3                  [512, T_frames]
//     -> SE-Res2Net dil=4                  [512, T_frames]
//     -> cat blk[1..3] + MFA k=1 + ReLU    [1536, T_frames]
//     -> ASP attentive pooling             [3072, 1]
//     -> FC k=1                            [2048, 1]
//     -> squeeze                           [2048]
//
// Tensor convention : [C, T] inside the graph (ne[0]=C, ne[1]=T) so that
// ggml_im2col reads each Conv1d along the time axis and ggml_mul_mat
// contracts over the input channel axis. This matches the layout the
// upstream PyTorch code uses after its (1, 2) transpose.

#include "audio-mel.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "speaker-encoder-weights.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Conv1d k=K with padding="same" mode="reflect" + bias add. The weight
// tensor lives in upstream layout [K, in_c, out_c]. We implement it with
// reflect pad + im2col + matmul.
//
//   x        [in_c, T]                input
//   w        [K, in_c, out_c]         weights
//   b        [out_c]                  bias (broadcast over T)
// Returns    [out_c, T]
//
// Padding for "same" with kernel K and dilation d is (K - 1) * d / 2 on
// each side (PyTorch convention, kernel size always odd here so the
// division is exact).
static struct ggml_tensor * spk_conv1d_same(struct ggml_context * ctx,
                                            struct ggml_tensor *  x,
                                            struct ggml_tensor *  w,
                                            struct ggml_tensor *  b,
                                            int                   dilation) {
    const int K   = (int) w->ne[0];
    const int IC  = (int) w->ne[1];
    const int OC  = (int) w->ne[2];
    const int pad = ((K - 1) * dilation) / 2;

    // ggml_pad_reflect_1d pads the innermost axis ne[0]. Our temporal
    // axis is ne[1], so we transpose to bring T to ne[0], pad, and keep
    // it that way : the im2col downstream expects ne[0]=T_pad, ne[1]=IC,
    // which is exactly the layout we end up with here.
    struct ggml_tensor * x_t = ggml_cont(ctx, ggml_transpose(ctx, x));  // ne=(T, IC)
    if (pad > 0) {
        x_t = ggml_pad_reflect_1d(ctx, x_t, pad, pad);                  // ne=(T+2*pad, IC)
    }

    // Reshape to 4D for ggml_im2col 1D : ne=(T_pad, IC, 1, 1).
    struct ggml_tensor * x4d = ggml_reshape_4d(ctx, x_t, x_t->ne[0], IC, 1, 1);

    // Dummy F32 kernel with the (K, IC) shape ggml_im2col needs to read
    // off both axes. Borrowing only the ne and not the real weight data
    // avoids the impl's src0 type assert when w is quantized.
    struct ggml_tensor * dummy = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, IC, 1, 1);
    ggml_set_name(dummy, "spk.im2col_kernel");

    // im2col with is_2D=false : the constructor declares ne[0]=IC*K and
    // ne[1]=OW, and the impl writes the buffer in (k inner, ic middle,
    // t outer) order, which matches that ne directly. A reshape_2d to
    // (IC*K, T_out) reads col_2d[ic*K + k, t], lining up with the
    // weight reshape w_2d[ic*K + k, oc] for the mul_mat below.
    struct ggml_tensor * col   = ggml_im2col(ctx, dummy, x4d, 1, 1, 0, 0, dilation, 1, false, GGML_TYPE_F32);
    int                  T_out = (int) col->ne[1];
    col                        = ggml_reshape_2d(ctx, col, K * IC, T_out);

    // Weight reshape : [K, IC, OC] -> [K*IC, OC]. mul_mat returns [OC, T_out].
    struct ggml_tensor * w2d = ggml_reshape_2d(ctx, w, K * IC, OC);
    struct ggml_tensor * y   = ggml_mul_mat(ctx, w2d, col);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);

    // Add bias broadcast over T_out. b is [out_c], reshape [out_c, 1].
    struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, OC, 1);
    y                        = ggml_add(ctx, y, b2d);
    return y;
}

// TDNN block : Conv1d(same, reflect) + ReLU. Used both as the conv0
// frontend (k=5) and inside SE-Res2Net (k=1) and the MFA / ASP TDNNs.
static struct ggml_tensor * spk_tdnn(struct ggml_context * ctx,
                                     const SpkEncTDNN &    t,
                                     struct ggml_tensor *  x,
                                     int                   dilation) {
    struct ggml_tensor * y = spk_conv1d_same(ctx, x, t.weight, t.bias, dilation);
    y                      = ggml_relu(ctx, y);
    return y;
}

// Res2Net block : split the channel axis in 8 chunks. chunk 0 passes
// through, chunk 1 goes through TDNN[0], chunks 2..7 mix with the
// previous chunk output before going through TDNN[i-1]. The 7 TDNN
// branches share dilation but operate on hidden / 8 channels each.
//
//   x      [C, T]
// Returns  [C, T]
static struct ggml_tensor * spk_res2net(struct ggml_context * ctx,
                                        const SpkEncRes2Net & rn,
                                        struct ggml_tensor *  x,
                                        int                   dilation,
                                        int                   scale) {
    const int C  = (int) x->ne[0];
    const int T  = (int) x->ne[1];
    const int Cs = C / scale;

    std::vector<struct ggml_tensor *> outs;
    outs.reserve(scale);

    // chunk i is the slice along ne[0] of width Cs starting at i * Cs.
    auto chunk = [&](int i) -> struct ggml_tensor * {
        return ggml_view_2d(ctx, x, Cs, T, x->nb[1], (size_t) (i * Cs) * x->nb[0]);
    };

    struct ggml_tensor * prev = NULL;
    for (int i = 0; i < scale; i++) {
        struct ggml_tensor * c = ggml_cont(ctx, chunk(i));
        if (i == 0) {
            outs.push_back(c);
            continue;
        }
        struct ggml_tensor * inp = c;
        if (i >= 2) {
            inp = ggml_add(ctx, c, prev);
        }
        struct ggml_tensor * y = spk_conv1d_same(ctx, inp, rn.weight[i - 1], rn.bias[i - 1], dilation);
        y                      = ggml_relu(ctx, y);
        outs.push_back(y);
        prev = y;
    }

    // Concat along ne[0]. ggml_concat with dim=0 stacks along the
    // fastest axis. Build the concat tree iteratively.
    struct ggml_tensor * acc = outs[0];
    for (int i = 1; i < scale; i++) {
        acc = ggml_concat(ctx, acc, outs[i], 0);
    }
    return acc;
}

// Squeeze and Excitation : compute the temporal mean per channel,
// project down to se_c with a 1x1 conv + ReLU, project back up to
// out_c with a 1x1 conv + sigmoid, then scale the input by the gate
// broadcast over T.
static struct ggml_tensor * spk_se(struct ggml_context * ctx, const SpkEncSE & se, struct ggml_tensor * x) {
    const int T = (int) x->ne[1];

    // Mean over T, keep dim. ggml_mean reduces along ne[0], so transpose
    // to put T on ne[0], reduce, transpose back.
    struct ggml_tensor * x_t  = ggml_cont(ctx, ggml_transpose(ctx, x));     // [T, C]
    struct ggml_tensor * mean = ggml_mean(ctx, x_t);                        // [1, C]
    mean                      = ggml_cont(ctx, ggml_transpose(ctx, mean));  // [C, 1]

    // conv1 1x1 reduces C -> se_c. dilation 1, padding "same" trivial
    // since k=1.
    struct ggml_tensor * h = spk_conv1d_same(ctx, mean, se.conv1_w, se.conv1_b, 1);
    h                      = ggml_relu(ctx, h);
    h                      = spk_conv1d_same(ctx, h, se.conv2_w, se.conv2_b, 1);
    // Sigmoid over [out_c, 1].
    h                      = ggml_sigmoid(ctx, h);

    // Scale x by the gate. h is [C, 1], x is [C, T]. ggml_mul broadcasts
    // ne[1]=1 to T.
    struct ggml_tensor * y = ggml_mul(ctx, x, h);
    (void) T;
    return y;
}

// SE-Res2Net block : tdnn1 (1x1) -> Res2Net -> tdnn2 (1x1) -> SE plus
// a residual add over the whole stack.
static struct ggml_tensor * spk_block(struct ggml_context * ctx,
                                      const SpkEncBlock &   blk,
                                      struct ggml_tensor *  x,
                                      int                   res2net_scale) {
    struct ggml_tensor * residual = x;
    struct ggml_tensor * h        = spk_tdnn(ctx, blk.tdnn1, x, 1);
    h                             = spk_res2net(ctx, blk.res2net, h, blk.dilation, res2net_scale);
    h                             = spk_tdnn(ctx, blk.tdnn2, h, 1);
    h                             = spk_se(ctx, blk.se, h);
    return ggml_add(ctx, h, residual);
}

// Attentive Statistical Pooling : compute global mean and std along T,
// concat with x, run an attention TDNN + tanh + 1x1 conv, softmax along
// T, recompute weighted mean and std, return the [2C, 1] concat.
//
//   x      [C, T]
// Returns  [2C, 1]
static struct ggml_tensor * spk_asp(struct ggml_context * ctx, const SpkEncASP & asp, struct ggml_tensor * x) {
    const int C = (int) x->ne[0];
    const int T = (int) x->ne[1];

    // Mean and std over T axis. The mask reduction is uniform 1/T.
    // mean : [C, 1]
    struct ggml_tensor * x_t  = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * mean = ggml_mean(ctx, x_t);
    mean                      = ggml_cont(ctx, ggml_transpose(ctx, mean));

    // var = mean( (x - mean)^2 ) over T, then std = sqrt(clamp(var, eps)).
    // Broadcasting (x - mean) requires mean repeated to T. ggml_repeat
    // handles this when shapes are compatible.
    struct ggml_tensor * mean_T   = ggml_repeat(ctx, mean, x);
    struct ggml_tensor * centered = ggml_sub(ctx, x, mean_T);
    struct ggml_tensor * var_t    = ggml_cont(ctx, ggml_transpose(ctx, ggml_sqr(ctx, centered)));
    struct ggml_tensor * var      = ggml_mean(ctx, var_t);
    var                           = ggml_cont(ctx, ggml_transpose(ctx, var));
    var                           = ggml_scale_bias(ctx, var, 1.0f, 1e-12f);
    struct ggml_tensor * std      = ggml_sqrt(ctx, var);

    // Build [x, mean_repeat, std_repeat] concat along channel axis.
    struct ggml_tensor * std_T = ggml_repeat(ctx, std, x);
    struct ggml_tensor * cat   = ggml_concat(ctx, x, mean_T, 0);
    cat                        = ggml_concat(ctx, cat, std_T, 0);  // [3C, T]

    // Attention TDNN : 3C -> attn_c, ReLU, then tanh, then 1x1 conv
    // attn_c -> C. Upstream applies tanh on the TDNN output before the
    // second conv ; the TDNN itself already runs ReLU so the order is
    // ReLU then tanh which is unusual but mirrored faithfully.
    struct ggml_tensor * a = spk_tdnn(ctx, asp.tdnn, cat, 1);
    a                      = ggml_tanh(ctx, a);
    a                      = spk_conv1d_same(ctx, a, asp.conv_w, asp.conv_b, 1);

    // Softmax along T axis. ggml_soft_max reduces ne[0], transpose first.
    struct ggml_tensor * a_t = ggml_cont(ctx, ggml_transpose(ctx, a));    // [T, C]
    struct ggml_tensor * w_t = ggml_soft_max(ctx, a_t);
    struct ggml_tensor * w   = ggml_cont(ctx, ggml_transpose(ctx, w_t));  // [C, T]

    // Weighted mean : sum(w * x) over T, w already sums to 1 over T.
    struct ggml_tensor * wx     = ggml_mul(ctx, w, x);
    struct ggml_tensor * wx_t   = ggml_cont(ctx, ggml_transpose(ctx, wx));
    // ggml_mean averages over ne[0]=T, giving 1/T scaling. We want the
    // un-normalized sum since w already encodes the soft selection
    // probability, so multiply back by T.
    struct ggml_tensor * w_mean = ggml_mean(ctx, wx_t);
    w_mean                      = ggml_scale(ctx, w_mean, (float) T);
    w_mean                      = ggml_cont(ctx, ggml_transpose(ctx, w_mean));  // [C, 1]

    // Weighted std : sum(w * (x - w_mean)^2) over T.
    struct ggml_tensor * w_mean_T = ggml_repeat(ctx, w_mean, x);
    struct ggml_tensor * dev      = ggml_sub(ctx, x, w_mean_T);
    struct ggml_tensor * w_var_in = ggml_mul(ctx, w, ggml_sqr(ctx, dev));
    struct ggml_tensor * w_var_t  = ggml_cont(ctx, ggml_transpose(ctx, w_var_in));
    struct ggml_tensor * w_var    = ggml_mean(ctx, w_var_t);
    w_var                         = ggml_scale(ctx, w_var, (float) T);
    w_var                         = ggml_cont(ctx, ggml_transpose(ctx, w_var));
    w_var                         = ggml_scale_bias(ctx, w_var, 1.0f, 1e-12f);
    struct ggml_tensor * w_std    = ggml_sqrt(ctx, w_var);

    // Stack [w_mean, w_std] along channel -> [2C, 1]. Time axis already
    // collapsed.
    struct ggml_tensor * stats = ggml_concat(ctx, w_mean, w_std, 0);
    (void) C;
    return stats;
}

// Full speaker encoder forward graph. Assumes the audio waveform has
// already been resampled to sr=24000 and reflect padded by
// (n_fft - hop) / 2 on each side. The padded buffer must outlive the
// graph compute call.
//
// Inputs :
//   audio_padded   [T_pad]    f32, host or backend tensor
//   mel constants  hann/dft_real/dft_imag/mel_basis backend tensors
//   mel_out        optional out param. When non NULL, receives the post
//                  mel_spectrogram tensor [n_mels, T_frames] so the caller
//                  can mark it as a graph output and pull its values back.
//   mag_out        optional out param. When non NULL, receives the post
//                  STFT magnitude tensor [n_freq, T_frames] for debug
//                  bisection between the STFT and the mel filtering.
//   frontend_out   optional. Post conv0 TDNN k=5 + ReLU output [512, T].
//   block3_out     optional. Post third SE-Res2Net block output [512, T].
//   mfa_out        optional. Post multi-layer feature aggregation [1536, T].
//   asp_out        optional. Post attentive statistical pooling [3072, 1].
// Output : [enc_dim] f32, the speaker embedding (typically 2048 dims).
static struct ggml_tensor * speaker_encoder_forward(struct ggml_context *         ctx,
                                                    const SpeakerEncoderWeights * sw,
                                                    struct ggml_tensor *          audio_padded,
                                                    struct ggml_tensor *          hann,
                                                    struct ggml_tensor *          dft_real,
                                                    struct ggml_tensor *          dft_imag,
                                                    struct ggml_tensor *          mel_basis,
                                                    const AudioMelConfig &        mel_cfg,
                                                    struct ggml_tensor **         mel_out      = NULL,
                                                    struct ggml_tensor **         mag_out      = NULL,
                                                    struct ggml_tensor **         frontend_out = NULL,
                                                    struct ggml_tensor **         block3_out   = NULL,
                                                    struct ggml_tensor **         mfa_out      = NULL,
                                                    struct ggml_tensor **         asp_out      = NULL) {
    // Mel : [n_mels=128, T_frames]
    struct ggml_tensor * mel =
        audio_mel_build_graph(ctx, audio_padded, hann, dft_real, dft_imag, mel_basis, mel_cfg, mag_out);
    if (mel_out) {
        *mel_out = mel;
    }

    // Frontend conv0 TDNN k=5 + ReLU : 128 -> 512, T preserved.
    struct ggml_tensor * h = spk_tdnn(ctx, sw->conv0, mel, 1);
    if (frontend_out) {
        *frontend_out = h;
    }

    // Three SE-Res2Net blocks at dilations 2, 3, 4.
    struct ggml_tensor * b1 = spk_block(ctx, sw->blocks[0], h, sw->res2net_scale);
    struct ggml_tensor * b2 = spk_block(ctx, sw->blocks[1], b1, sw->res2net_scale);
    struct ggml_tensor * b3 = spk_block(ctx, sw->blocks[2], b2, sw->res2net_scale);
    if (block3_out) {
        *block3_out = b3;
    }

    // Multi-layer feature aggregation : cat blk1..3 then 1x1 TDNN + ReLU.
    struct ggml_tensor * cat = ggml_concat(ctx, b1, b2, 0);
    cat                      = ggml_concat(ctx, cat, b3, 0);    // [1536, T]
    struct ggml_tensor * mfa = spk_tdnn(ctx, sw->mfa, cat, 1);  // [1536, T]
    if (mfa_out) {
        *mfa_out = mfa;
    }

    // Attentive statistical pooling : [1536, T] -> [3072, 1].
    struct ggml_tensor * stats = spk_asp(ctx, sw->asp, mfa);
    if (asp_out) {
        *asp_out = stats;
    }

    // Final FC k=1 : [3072, 1] -> [enc_dim, 1].
    struct ggml_tensor * emb = spk_conv1d_same(ctx, stats, sw->fc_w, sw->fc_b, 1);

    // Squeeze T axis, return [enc_dim]. ggml_cont is required so the sched
    // assigns a fresh backend buffer to the graph output rather than
    // forwarding a view of the FC bias add.
    emb = ggml_reshape_1d(ctx, emb, sw->enc_dim);
    emb = ggml_cont(ctx, emb);
    ggml_set_name(emb, "spk.embedding");
    return emb;
}
