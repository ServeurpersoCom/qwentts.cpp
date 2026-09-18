#pragma once
// audio-resample.h: torchaudio.functional.resample compatible reimplementation.
// Hann-windowed sinc interpolation with rolloff=0.99 and lowpass_filter_width=6,
// matching torchaudio defaults bit for bit.
//
// Reference: torchaudio/functional/functional.py, _get_sinc_resample_kernel
// and _apply_sinc_resample_kernel.
//
// Algorithm:
//   gcd                 = gcd(sr_in, sr_out)
//   orig                = sr_in / gcd
//   new                 = sr_out / gcd
//   base                = min(orig, new) * rolloff
//   width               = ceil(lpfw * orig / base)
//   kernel_size         = 2 * width + orig
//   kernel[j, k]        = sinc(t * pi) * hann(t)^2 * (base / orig)
//                          with t = clamp(((k - width) / orig - j / new) * base,
//                                         -lpfw, lpfw)
//   target_length       = ceil(sr_out * n_in / sr_in)
//
// Apply: pad (width, width + orig), strided conv1d, transpose, truncate.

#include "qt-error.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define AUDIO_RESAMPLE_LPFW    6
#define AUDIO_RESAMPLE_ROLLOFF 0.99

static int audio_resample_gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b     = a % b;
        a     = t;
    }
    return a;
}

// Build the Hann-sinc polyphase kernel [new_freq_red, kernel_size] in row major.
// new_freq_red and orig_freq_red are sr_out/gcd and sr_in/gcd respectively.
static std::vector<float> audio_resample_build_kernel(int orig, int newf, int * out_width, int * out_kernel_size) {
    int    base_int = (orig < newf) ? orig : newf;
    double base     = (double) base_int * AUDIO_RESAMPLE_ROLLOFF;
    int    width    = (int) std::ceil((double) AUDIO_RESAMPLE_LPFW * (double) orig / base);
    int    K        = 2 * width + orig;

    std::vector<float> ker((size_t) newf * (size_t) K);

    double scale = base / (double) orig;
    double inv_o = 1.0 / (double) orig;
    double inv_n = 1.0 / (double) newf;
    double pi    = M_PI;

    for (int j = 0; j < newf; j++) {
        double t_off = (double) (-j) * inv_n;
        for (int k = 0; k < K; k++) {
            double idx_k = (double) (k - width) * inv_o;
            double t     = (t_off + idx_k) * base;
            if (t < -AUDIO_RESAMPLE_LPFW) {
                t = -AUDIO_RESAMPLE_LPFW;
            }
            if (t > AUDIO_RESAMPLE_LPFW) {
                t = AUDIO_RESAMPLE_LPFW;
            }

            double w = std::cos(t * pi / (double) AUDIO_RESAMPLE_LPFW / 2.0);
            w        = w * w;

            double tp   = t * pi;
            double sinc = (tp == 0.0) ? 1.0 : std::sin(tp) / tp;

            ker[(size_t) j * (size_t) K + (size_t) k] = (float) (sinc * w * scale);
        }
    }

    *out_width       = width;
    *out_kernel_size = K;
    return ker;
}

// Resample one mono channel from sr_in to sr_out. Returns ceil(sr_out*n_in/sr_in)
// samples. Caller passes the kernel + width built once via audio_resample_build_kernel.
static void audio_resample_apply_mono(const float * in,
                                      int           n_in,
                                      int           orig,
                                      int           newf,
                                      int           width,
                                      int           kernel_size,
                                      const float * kernel,
                                      float *       out,
                                      long long     target_length) {
    int                K  = kernel_size;
    int                Np = n_in + 2 * width + orig;
    std::vector<float> padded((size_t) Np, 0.0f);
    std::memcpy(padded.data() + width, in, (size_t) n_in * sizeof(float));

    int       n_per_chan = (Np - K) / orig + 1;
    long long total      = (long long) n_per_chan * (long long) newf;
    long long out_len    = (target_length < total) ? target_length : total;

    for (long long t_out = 0; t_out < out_len; t_out++) {
        int           chan = (int) (t_out % (long long) newf);
        int           pos  = (int) (t_out / (long long) newf);
        const float * w    = kernel + (size_t) chan * (size_t) K;
        const float * x    = padded.data() + (size_t) pos * (size_t) orig;
        float         sum  = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += x[k] * w[k];
        }
        out[(size_t) t_out] = sum;
    }
}

// Public API: resample a planar (or mono) f32 buffer from sr_in to sr_out.
// in:     float buffer with channels stored planar [ch0: n_in][ch1: n_in][...].
// n_in:   per-channel input sample count.
// nch:    number of channels.
// n_out:  receives the per-channel output sample count.
//
// Returns a malloc'd planar buffer [ch0: *n_out][ch1: *n_out][...].
// Caller must free() the result. NULL on error.
static float * audio_resample(const float * in, int n_in, int sr_in, int sr_out, int nch, int * n_out) {
    if (!in || n_in <= 0 || sr_in <= 0 || sr_out <= 0 || nch <= 0) {
        *n_out = 0;
        return NULL;
    }

    // Passthrough when source and target rates match.
    if (sr_in == sr_out) {
        size_t  sz  = (size_t) n_in * (size_t) nch * sizeof(float);
        float * out = (float *) malloc(sz);
        if (!out) {
            qt_log(QT_LOG_ERROR, "[Audio-Resample] OOM passthrough buffer (%zu bytes)", sz);
            *n_out = 0;
            return NULL;
        }
        *n_out = n_in;
        memcpy(out, in, sz);
        return out;
    }

    int g    = audio_resample_gcd(sr_in, sr_out);
    int orig = sr_in / g;
    int newf = sr_out / g;

    int                width = 0, kernel_size = 0;
    std::vector<float> kernel = audio_resample_build_kernel(orig, newf, &width, &kernel_size);

    long long target = (long long) std::ceil((double) sr_out * (double) n_in / (double) sr_in);
    if (target <= 0) {
        *n_out = 0;
        return NULL;
    }

    *n_out = (int) target;

    float * out = (float *) malloc((size_t) target * (size_t) nch * sizeof(float));
    if (!out) {
        qt_log(QT_LOG_ERROR, "[Audio-Resample] OOM output buffer");
        *n_out = 0;
        return NULL;
    }

    for (int ch = 0; ch < nch; ch++) {
        const float * src = in + (size_t) ch * (size_t) n_in;
        float *       dst = out + (size_t) ch * (size_t) target;
        audio_resample_apply_mono(src, n_in, orig, newf, width, kernel_size, kernel.data(), dst, target);
    }

    return out;
}
