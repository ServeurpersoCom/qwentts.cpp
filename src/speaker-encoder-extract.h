#pragma once
// speaker-encoder-extract.h: end to end speaker embedding extraction
// from a WAV path. Loads, mono-mixes and resamples to 24 kHz, reflect
// pads by (n_fft - hop) / 2 = 384 samples, builds the fused mel + ECAPA
// graph and returns the f32 [enc_dim] embedding.
//
// Mirrors qwen_tts.core.models.modeling_qwen3_tts.extract_speaker_embedding :
//
//   audio = librosa.load(path, sr=None, mono=True)[0]
//   audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)
//   mels = mel_spectrogram(audio_24k, n_fft=1024, n_mels=128, sr=24000,
//                          hop=256, win=1024, fmin=0, fmax=12000, center=False)
//   spk_emb = speaker_encoder(mels)[0]
//
// Memory layout: the audio waveform input is passed as a regular ggml
// input tensor [T_pad] f32 living on the talker backend. Caller owns the
// returned vector. The graph context is freed after each call.

#include "audio-io.h"
#include "audio-mel.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "qt-error.h"
#include "speaker-encoder-forward.h"
#include "speaker-encoder-weights.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// Public entry point. Returns true on success, fills emb_out with the
// 2048-dim f32 embedding. The audio buffer must already be mono at
// sw->sample_rate (24 kHz). When dump_dir is non NULL, also writes the
// post mel_spectrogram tensor to mel-spk.bin under that directory using
// the debug.h header format. Quiet otherwise.
static bool speaker_encoder_extract(const SpeakerEncoderWeights * sw,
                                    ggml_backend_sched_t          sched,
                                    const float *                 audio,
                                    int                           n_samples,
                                    std::vector<float> &          emb_out,
                                    const char *                  dump_dir = NULL) {
    if (sw->weight_buf == NULL) {
        qt_log(QT_LOG_ERROR, "[SpkExtract] FATAL: speaker encoder weights not loaded");
        return false;
    }
    if (!audio || n_samples <= 0) {
        qt_log(QT_LOG_ERROR, "[SpkExtract] FATAL: empty audio buffer");
        return false;
    }

    AudioMelConfig mel_cfg;
    mel_cfg.sample_rate = sw->sample_rate;
    mel_cfg.n_fft       = 1024;
    mel_cfg.hop         = 256;
    mel_cfg.n_mels      = sw->mel_dim;
    mel_cfg.fmin        = 0.0f;
    mel_cfg.fmax        = 12000.0f;

    const int     T_in = n_samples;
    const float * raw  = audio;

    const int pad   = (mel_cfg.n_fft - mel_cfg.hop) / 2;  // 384
    const int T_pad = T_in + 2 * pad;
    if (T_in < pad + 1) {
        qt_log(QT_LOG_ERROR, "[SpkExtract] FATAL: audio too short (%d samples) for reflect pad %d", T_in, pad);
        return false;
    }

    // Reflect pad on the host so the graph just consumes a flat [T_pad]
    // input. PyTorch reflect copies samples [1..pad] then [T-2..T-pad-1]
    // into the padded edges (the boundary sample itself is not duplicated).
    std::vector<float> audio_padded((size_t) T_pad);
    for (int i = 0; i < pad; i++) {
        audio_padded[(size_t) i] = raw[pad - i];
    }
    std::memcpy(audio_padded.data() + pad, raw, (size_t) T_in * sizeof(float));
    for (int i = 0; i < pad; i++) {
        audio_padded[(size_t) (pad + T_in + i)] = raw[T_in - 2 - i];
    }

    // Bake CPU constants once per call: Hann, DFT, mel basis. The cost
    // is dominated by the DFT precompute which is 524 KB of f32.
    AudioMelConstants mel_c;
    audio_mel_compute_constants(mel_cfg, mel_c);

    // Build the graph context. mel + ECAPA accounts for ~150 nodes per
    // SE-Res2Net block + 30 for the mel front end + 60 for ASP and FC.
    // 2048 nodes is a comfortable upper bound.
    const size_t     mem_size  = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(2048, false);
    ggml_init_params init      = {};
    init.mem_size              = mem_size;
    init.mem_buffer            = NULL;
    init.no_alloc              = true;
    struct ggml_context * gctx = ggml_init(init);

    // Graph inputs: audio waveform and 4 mel constants.
    struct ggml_tensor * audio_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, T_pad);
    struct ggml_tensor * hann_in   = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, mel_cfg.n_fft);
    struct ggml_tensor * dft_re_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, mel_cfg.n_fft, mel_c.n_freq);
    struct ggml_tensor * dft_im_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, mel_cfg.n_fft, mel_c.n_freq);
    struct ggml_tensor * mel_b_in  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, mel_c.n_freq, mel_cfg.n_mels);
    ggml_set_name(audio_in, "spk.audio_padded");
    ggml_set_name(hann_in, "spk.hann");
    ggml_set_name(dft_re_in, "spk.dft_real");
    ggml_set_name(dft_im_in, "spk.dft_imag");
    ggml_set_name(mel_b_in, "spk.mel_basis");
    // Mark as graph inputs so the scheduler routes them through the
    // CPU backend then uploads on demand to the compute backend.
    ggml_set_input(audio_in);
    ggml_set_input(hann_in);
    ggml_set_input(dft_re_in);
    ggml_set_input(dft_im_in);
    ggml_set_input(mel_b_in);

    struct ggml_tensor * mel_t      = NULL;
    struct ggml_tensor * mel_dump   = NULL;
    struct ggml_tensor * mag_t      = NULL;
    struct ggml_tensor * mag_dump   = NULL;
    struct ggml_tensor * front_t    = NULL;
    struct ggml_tensor * front_dump = NULL;
    struct ggml_tensor * blk3_t     = NULL;
    struct ggml_tensor * blk3_dump  = NULL;
    struct ggml_tensor * mfa_t      = NULL;
    struct ggml_tensor * mfa_dump   = NULL;
    struct ggml_tensor * asp_t      = NULL;
    struct ggml_tensor * asp_dump   = NULL;
    struct ggml_tensor * emb = speaker_encoder_forward(gctx, sw, audio_in, hann_in, dft_re_in, dft_im_in, mel_b_in,
                                                       mel_cfg, &mel_t, &mag_t, &front_t, &blk3_t, &mfa_t, &asp_t);
    ggml_set_output(emb);
    if (dump_dir && mel_t) {
        // mel_t has ggml ne=(n_mels, T_frames), which streams row-major
        // as numpy [T_frames, n_mels], the exact layout the upstream
        // mel_spectrogram() exposes after its .transpose(1, 2). The
        // cont call forces materialization in case the scheduler fuses
        // ggml_log with anything downstream.
        mel_dump = ggml_cont(gctx, mel_t);
        ggml_set_output(mel_dump);
        ggml_set_name(mel_dump, "spk.mel_dump");
    }
    if (dump_dir && mag_t) {
        // mag_t has ggml ne=(n_freq, T_frames), which streams row-major
        // as numpy [T_frames, n_freq], matching the upstream STFT
        // magnitude after its transpose. Cont forces materialization.
        mag_dump = ggml_cont(gctx, mag_t);
        ggml_set_output(mag_dump);
        ggml_set_name(mag_dump, "spk.mag_dump");
    }
    if (dump_dir && front_t) {
        // front_t has ggml ne=(512, T_frames), reads row-major as numpy
        // [T_frames, 512]. Same layout the Python forward sees after
        // blocks[0] (TimeDelayNetBlock) when transposed (1, 2).
        front_dump = ggml_cont(gctx, front_t);
        ggml_set_output(front_dump);
        ggml_set_name(front_dump, "spk.frontend_dump");
    }
    if (dump_dir && blk3_t) {
        // blk3_t has ggml ne=(512, T_frames) = numpy [T_frames, 512],
        // matching Python blocks[3] output transposed (1, 2).
        blk3_dump = ggml_cont(gctx, blk3_t);
        ggml_set_output(blk3_dump);
        ggml_set_name(blk3_dump, "spk.block3_dump");
    }
    if (dump_dir && mfa_t) {
        // mfa_t has ggml ne=(1536, T_frames) = numpy [T_frames, 1536],
        // matching Python mfa output transposed (1, 2).
        mfa_dump = ggml_cont(gctx, mfa_t);
        ggml_set_output(mfa_dump);
        ggml_set_name(mfa_dump, "spk.mfa_dump");
    }
    if (dump_dir && asp_t) {
        // asp_t has ggml ne=(3072, 1), reads row-major as numpy [1, 3072].
        // Python asp returns [B=1, 3072, 1] which we slice with [0].T to
        // obtain [1, 3072] for a direct shape-aligned compare.
        asp_dump = ggml_cont(gctx, asp_t);
        ggml_set_output(asp_dump);
        ggml_set_name(asp_dump, "spk.asp_dump");
    }

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, 2048, false);
    ggml_build_forward_expand(graph, emb);
    if (mel_dump) {
        ggml_build_forward_expand(graph, mel_dump);
    }
    if (mag_dump) {
        ggml_build_forward_expand(graph, mag_dump);
    }
    if (front_dump) {
        ggml_build_forward_expand(graph, front_dump);
    }
    if (blk3_dump) {
        ggml_build_forward_expand(graph, blk3_dump);
    }
    if (mfa_dump) {
        ggml_build_forward_expand(graph, mfa_dump);
    }
    if (asp_dump) {
        ggml_build_forward_expand(graph, asp_dump);
    }

    // Reset the shared sched before allocating: the talker may have left
    // a residual graph state from a previous synthesis call.
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        qt_log(QT_LOG_ERROR, "[SpkExtract] FATAL: graph allocation failed");
        ggml_free(gctx);
        return false;
    }

    // Upload inputs to backend.
    ggml_backend_tensor_set(audio_in, audio_padded.data(), 0, (size_t) T_pad * sizeof(float));
    ggml_backend_tensor_set(hann_in, mel_c.hann.data(), 0, mel_c.hann.size() * sizeof(float));
    ggml_backend_tensor_set(dft_re_in, mel_c.dft_real.data(), 0, mel_c.dft_real.size() * sizeof(float));
    ggml_backend_tensor_set(dft_im_in, mel_c.dft_imag.data(), 0, mel_c.dft_imag.size() * sizeof(float));
    ggml_backend_tensor_set(mel_b_in, mel_c.mel_basis.data(), 0, mel_c.mel_basis.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[SpkExtract] FATAL: graph compute failed");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }

    emb_out.assign((size_t) sw->enc_dim, 0.0f);
    ggml_backend_tensor_get(emb, emb_out.data(), 0, (size_t) sw->enc_dim * sizeof(float));

    if (dump_dir && mel_dump) {
        size_t             n = ggml_nelements(mel_dump);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(mel_dump, buf.data(), 0, n * sizeof(float));
        DebugDumper d;
        debug_init(&d, dump_dir);
        // mel_dump has ggml ne=(n_mels, T_frames). debug_dump_2d takes
        // (rows, cols) in numpy convention so we pass (ne[1]=T_frames,
        // ne[0]=n_mels), which writes shape [T_frames, n_mels] over the
        // raw ggml memory layout, matching the Python side dump.
        debug_dump_2d(&d, "mel-spk", buf.data(), (int) mel_dump->ne[1], (int) mel_dump->ne[0]);

        // CPU side mel constants: audit against torch.hann_window and
        // librosa.filters.mel produced by the Python upstream. Layouts
        // are kept as numpy [n_fft] for hann and [n_mels, n_freq] for
        // mel_basis, matching the librosa convention.
        debug_dump_1d(&d, "mel-hann", mel_c.hann.data(), mel_cfg.n_fft);
        debug_dump_2d(&d, "mel-basis", mel_c.mel_basis.data(), mel_cfg.n_mels, mel_c.n_freq);

        if (mag_dump) {
            size_t             nm = ggml_nelements(mag_dump);
            std::vector<float> bm(nm);
            ggml_backend_tensor_get(mag_dump, bm.data(), 0, nm * sizeof(float));
            // mag_dump has ggml ne=(n_freq, T_frames). Same dumping
            // convention as mel-spk: passing (ne[1], ne[0]) writes
            // shape [T_frames, n_freq] over the raw memory layout.
            debug_dump_2d(&d, "mel-mag", bm.data(), (int) mag_dump->ne[1], (int) mag_dump->ne[0]);
        }

        // ECAPA forward bisection points. Each tensor has ggml ne=(C, T)
        // and dumps as numpy [T, C] using the (ne[1], ne[0]) convention.
        // ASP collapses T to 1 so it lands as [1, 3072].
        if (front_dump) {
            size_t             nf = ggml_nelements(front_dump);
            std::vector<float> bf(nf);
            ggml_backend_tensor_get(front_dump, bf.data(), 0, nf * sizeof(float));
            debug_dump_2d(&d, "spk-frontend", bf.data(), (int) front_dump->ne[1], (int) front_dump->ne[0]);
        }
        if (blk3_dump) {
            size_t             nb = ggml_nelements(blk3_dump);
            std::vector<float> bb(nb);
            ggml_backend_tensor_get(blk3_dump, bb.data(), 0, nb * sizeof(float));
            debug_dump_2d(&d, "spk-block3", bb.data(), (int) blk3_dump->ne[1], (int) blk3_dump->ne[0]);
        }
        if (mfa_dump) {
            size_t             nf = ggml_nelements(mfa_dump);
            std::vector<float> bf(nf);
            ggml_backend_tensor_get(mfa_dump, bf.data(), 0, nf * sizeof(float));
            debug_dump_2d(&d, "spk-mfa", bf.data(), (int) mfa_dump->ne[1], (int) mfa_dump->ne[0]);
        }
        if (asp_dump) {
            size_t             na = ggml_nelements(asp_dump);
            std::vector<float> ba(na);
            ggml_backend_tensor_get(asp_dump, ba.data(), 0, na * sizeof(float));
            debug_dump_2d(&d, "spk-asp", ba.data(), (int) asp_dump->ne[1], (int) asp_dump->ne[0]);
        }
    }

    ggml_backend_sched_reset(sched);
    ggml_free(gctx);

    qt_log(QT_LOG_INFO, "[SpkExtract] Extracted %d-dim embedding (%d samples, padded %d)", sw->enc_dim, T_in, T_pad);
    return true;
}
