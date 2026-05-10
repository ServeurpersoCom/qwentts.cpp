#pragma once
// audio-mel.h : log mel spectrogram extractor matching Qwen3TTS upstream.
//
// Pipeline mirrored from qwen_tts/core/models/modeling_qwen3_tts.py
// mel_spectrogram() at lines 399 to 464 :
//
//   pad reflect by (n_fft - hop) / 2
//   torch.stft(n_fft, hop, win=n_fft, hann_periodic, center=False)
//   mag = sqrt(real^2 + imag^2 + 1e-9)
//   mel = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) @ mag
//   log_mel = log(max(mel, 1e-5))
//
// Spec for the speaker encoder path :
//   sr=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000
//
// GGML strategy : no native FFT op, so the DFT is folded into two
// real matmuls. We precompute on CPU two F32 matrices :
//   dft_real [n_freq, n_fft]  with cos(2 pi k n / n_fft)
//   dft_imag [n_freq, n_fft]  with -sin(2 pi k n / n_fft)
// where n_freq = n_fft / 2 + 1. The framing uses ggml_im2col on the
// padded signal viewed as a 1D conv input. See audio_mel_build_graph
// for the graph topology, and audio_mel_compute_constants for the
// cos/sin and mel basis baking.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

struct AudioMelConfig {
    int   sample_rate;
    int   n_fft;
    int   hop;
    int   n_mels;
    float fmin;
    float fmax;
};

// CPU side constants : Hann window, DFT real/imag matrices, mel filter.
// Allocated once per AudioMelConfig and uploaded to the backend as
// regular ggml tensors during graph build.
struct AudioMelConstants {
    AudioMelConfig     cfg;
    int                n_freq;
    std::vector<float> hann;       // [n_fft]
    std::vector<float> dft_real;   // [n_freq, n_fft] row major  ne=(n_fft, n_freq)
    std::vector<float> dft_imag;   // [n_freq, n_fft] row major
    std::vector<float> mel_basis;  // [n_mels, n_freq] row major  ne=(n_freq, n_mels)
};

// Slaney mel scale, the default of librosa.filters.mel.
static inline float audio_mel_hz_to_mel(float hz) {
    // Slaney : linear below 1000 Hz, log above.
    const float f_min       = 0.0f;
    const float f_sp        = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (hz < min_log_hz) {
        return (hz - f_min) / f_sp;
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

static inline float audio_mel_mel_to_hz(float mel) {
    const float f_min       = 0.0f;
    const float f_sp        = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (mel < min_log_mel) {
        return f_min + f_sp * mel;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

// Bake CPU constants once. Reproduces librosa.filters.mel(slaney) and
// torch.hann_window(periodic=True) bit-for-bit on F32, with the cos/sin
// DFT matrix evaluated at double precision then cast to float.
static void audio_mel_compute_constants(const AudioMelConfig & cfg, AudioMelConstants & c) {
    c.cfg    = cfg;
    c.n_freq = cfg.n_fft / 2 + 1;

    // Hann periodic : 0.5 * (1 - cos(2 pi i / N)) for i in [0, N).
    c.hann.assign(cfg.n_fft, 0.0f);
    for (int i = 0; i < cfg.n_fft; i++) {
        c.hann[i] = 0.5f * (1.0f - (float) std::cos(2.0 * M_PI * (double) i / (double) cfg.n_fft));
    }

    // DFT matrices, real and imag part, F32. Computed in F64 to keep
    // the trig roundoff below the F32 ULP threshold.
    c.dft_real.assign((size_t) c.n_freq * (size_t) cfg.n_fft, 0.0f);
    c.dft_imag.assign((size_t) c.n_freq * (size_t) cfg.n_fft, 0.0f);
    for (int k = 0; k < c.n_freq; k++) {
        for (int n = 0; n < cfg.n_fft; n++) {
            double th = 2.0 * M_PI * (double) k * (double) n / (double) cfg.n_fft;
            c.dft_real[(size_t) k * (size_t) cfg.n_fft + (size_t) n] = (float) std::cos(th);
            c.dft_imag[(size_t) k * (size_t) cfg.n_fft + (size_t) n] = (float) (-std::sin(th));
        }
    }

    // Slaney mel filterbank : n_mels triangular filters between fmin and
    // fmax, normalized by 2 / (mel_freqs[i+2] - mel_freqs[i]). Matches
    // librosa.filters.mel(htk=False, norm='slaney') byte for byte.
    const float fmin = cfg.fmin;
    const float fmax = (cfg.fmax <= 0.0f) ? (float) cfg.sample_rate * 0.5f : cfg.fmax;
    const float mmin = audio_mel_hz_to_mel(fmin);
    const float mmax = audio_mel_hz_to_mel(fmax);

    std::vector<float> mel_pts((size_t) cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; i++) {
        mel_pts[(size_t) i] = mmin + (mmax - mmin) * (float) i / (float) (cfg.n_mels + 1);
    }
    std::vector<float> hz_pts((size_t) cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; i++) {
        hz_pts[(size_t) i] = audio_mel_mel_to_hz(mel_pts[(size_t) i]);
    }
    std::vector<float> fft_freqs((size_t) c.n_freq);
    for (int k = 0; k < c.n_freq; k++) {
        fft_freqs[(size_t) k] = (float) k * (float) cfg.sample_rate / (float) cfg.n_fft;
    }

    c.mel_basis.assign((size_t) cfg.n_mels * (size_t) c.n_freq, 0.0f);
    for (int m = 0; m < cfg.n_mels; m++) {
        const float lo = hz_pts[(size_t) m];
        const float md = hz_pts[(size_t) m + 1];
        const float hi = hz_pts[(size_t) m + 2];
        for (int k = 0; k < c.n_freq; k++) {
            const float f    = fft_freqs[(size_t) k];
            float       up   = (f - lo) / (md - lo);
            float       down = (hi - f) / (hi - md);
            float       w    = std::fmin(up, down);
            if (w < 0.0f) {
                w = 0.0f;
            }
            c.mel_basis[(size_t) m * (size_t) c.n_freq + (size_t) k] = w;
        }
        // Slaney area normalization : 2 / (hi - lo).
        const float enorm = 2.0f / (hi - lo);
        for (int k = 0; k < c.n_freq; k++) {
            c.mel_basis[(size_t) m * (size_t) c.n_freq + (size_t) k] *= enorm;
        }
    }
}

// Build the GGML graph that turns a [T_in] f32 audio waveform into
// a [n_mels, T_frames] f32 log mel spectrogram. The signal is reflect
// padded by (n_fft - hop) / 2 on each side before framing, mirroring
// torch.nn.functional.pad(mode="reflect") used upstream.
//
// Inputs :
//   audio        [T_padded]                f32, already reflect padded by the caller
//   hann         [n_fft]                   f32, host constant
//   dft_real     [n_fft, n_freq]           f32, host constant
//   dft_imag     [n_fft, n_freq]           f32, host constant
//   mel_basis    [n_freq, n_mels]          f32, host constant
//   mag_out      optional pointer that receives the post-STFT magnitude
//                tensor [n_freq, T_frames] for debug bisection. NULL by
//                default. Caller marks it as graph output if needed.
//
// Output : [n_mels, T_frames] f32 log mel.
//
// The im2col path produces frames [n_fft, T_frames] T-fastest, which is
// the layout ggml_mul_mat expects on the right operand (ne[0] = K = n_fft,
// ne[1] = M = T_frames).
static struct ggml_tensor * audio_mel_build_graph(struct ggml_context *  ctx,
                                                  struct ggml_tensor *   audio_padded,
                                                  struct ggml_tensor *   hann,
                                                  struct ggml_tensor *   dft_real,
                                                  struct ggml_tensor *   dft_imag,
                                                  struct ggml_tensor *   mel_basis,
                                                  const AudioMelConfig & cfg,
                                                  struct ggml_tensor **  mag_out = NULL) {
    const int n_fft = cfg.n_fft;
    const int hop   = cfg.hop;

    // Shape audio as [T_padded, 1, 1, 1] so im2col reads it as a 1D
    // conv input with C_in = 1. ggml_im2col expects [IW, IH=1, IC, N=1]
    // and returns [K_w, IC * K_h, OW, N]. With IH = 1 and K_h = 1, the
    // output collapses to [n_fft, 1, T_frames, 1] which is what we need.
    struct ggml_tensor * a4d   = ggml_reshape_4d(ctx, audio_padded, audio_padded->ne[0], 1, 1, 1);
    struct ggml_tensor * dummy = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_fft, 1, 1, 1);
    ggml_set_name(dummy, "mel.im2col_dummy_kernel");

    // im2col(s0=hop, s1=1, p0=0, p1=0, d0=1, d1=1, is_2D=false).
    // Output dtype F32 to match audio dtype. Output shape with is_2D=false
    // is [K * IC, OW, IC_outer, 1] which collapses to [n_fft, T_frames, 1, 1]
    // since our IC = 1 and IC_outer = 1.
    struct ggml_tensor * frames = ggml_im2col(ctx, dummy, a4d, hop, 1, 0, 0, 1, 1, false, GGML_TYPE_F32);
    // frames has ne=(n_fft, T_frames, 1, 1). Reshape to [n_fft, T_frames].
    frames                      = ggml_reshape_2d(ctx, frames, n_fft, frames->ne[1]);

    // Multiply by hann window, broadcast over T_frames. ggml_mul
    // broadcasts ne[0]-equal operands when one has ne[1]=1. hann is
    // already [n_fft, 1].
    struct ggml_tensor * hann_2d = ggml_reshape_2d(ctx, hann, n_fft, 1);
    frames                       = ggml_mul(ctx, frames, hann_2d);

    // STFT real and imag parts via two F32 matmuls. dft_* have layout
    // [n_fft, n_freq] in ggml notation, so mul_mat returns [n_freq, T_frames].
    struct ggml_tensor * spec_re = ggml_mul_mat(ctx, dft_real, frames);
    struct ggml_tensor * spec_im = ggml_mul_mat(ctx, dft_imag, frames);
    ggml_mul_mat_set_prec(spec_re, GGML_PREC_F32);
    ggml_mul_mat_set_prec(spec_im, GGML_PREC_F32);

    // Magnitude with the same eps as torch upstream (1e-9 added to the
    // power, then sqrt). ggml_scale_bias does s*a + b so we add the eps
    // without conjuring a backend dependent constant tensor.
    struct ggml_tensor * mag2 = ggml_add(ctx, ggml_sqr(ctx, spec_re), ggml_sqr(ctx, spec_im));
    mag2                      = ggml_scale_bias(ctx, mag2, 1.0f, 1e-9f);
    struct ggml_tensor * mag  = ggml_sqrt(ctx, mag2);
    if (mag_out) {
        *mag_out = mag;
    }

    // mel_basis [n_freq, n_mels] @ mag [n_freq, T_frames] -> [n_mels, T_frames].
    struct ggml_tensor * mel = ggml_mul_mat(ctx, mel_basis, mag);
    ggml_mul_mat_set_prec(mel, GGML_PREC_F32);

    // log(max(mel, 1e-5)). ggml has clamp + log primitives.
    mel = ggml_clamp(ctx, mel, 1e-5f, 1e30f);
    mel = ggml_log(ctx, mel);
    ggml_set_name(mel, "mel.log_mel");

    return mel;
}
