#pragma once
// codec-chunked-decode.h: bounded VRAM codec decode with rolling left
// context. Strict equivalent of the upstream Qwen3-TTS 12 Hz tokenizer
// chunked_decode entry
// (qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
// line 886).
//
// The codec decoder runs a causal Conv1d, a sliding window causal
// transformer, an upsample stage and a DAC decoder. Decoding a chunk
// of frames in isolation introduces edge artefacts at the chunk
// boundary because the causal conv kernel and the transformer attention
// have no left context to draw from. Prepending left_ctx_frames
// previously decoded frames and stripping the resulting samples after
// the decode restores continuity.
//
// Two entry points:
//
//   codec_chunked_decode : one shot decode of a full codes buffer.
//     Bit perfect equivalent of pipeline_codec_decode when the audio
//     fits in a single chunk_frames sized window. Bounds VRAM beyond
//     that, mirrors the upstream chunked_decode loop frame for frame.
//
//   codec_stream_decoder : stateful frame by frame AR streaming.
//     The pipeline pushes one frame at a time as the talker produces
//     them ; push_frame decodes and emits a fresh chunk_frames sized
//     audio block through the on_chunk callback as soon as enough new
//     frames have accumulated. flush drains the tail at EOS.

#include "pipeline-codec.h"
#include "qwen.h"

#include <cstdint>
#include <cstring>
#include <vector>

// One shot chunked decode. codes is K major [K, T] row major (T fastest).
// Returns audio of length T * TOKENIZER_HOP_LENGTH on success, empty on
// failure. chunk_frames clamps to 1, left_ctx_frames clamps to 0.
static inline std::vector<float> codec_chunked_decode(PipelineCodec * pc,
                                                      const int32_t * codes,
                                                      int             K,
                                                      int             T,
                                                      int             chunk_frames,
                                                      int             left_ctx_frames) {
    std::vector<float> out;
    if (T <= 0) {
        return out;
    }
    if (chunk_frames < 1) {
        chunk_frames = 1;
    }
    if (left_ctx_frames < 0) {
        left_ctx_frames = 0;
    }
    out.reserve((size_t) T * (size_t) TOKENIZER_HOP_LENGTH);

    int start = 0;
    while (start < T) {
        int end = start + chunk_frames;
        if (end > T) {
            end = T;
        }
        // Upstream rule : context_size collapses to start when
        // left_ctx_frames would underflow before frame 0.
        int ctx         = (start - left_ctx_frames > 0) ? left_ctx_frames : start;
        int slice_start = start - ctx;
        int slice_T     = end - slice_start;

        std::vector<int32_t> slice((size_t) K * (size_t) slice_T);
        for (int k = 0; k < K; k++) {
            std::memcpy(slice.data() + (size_t) k * (size_t) slice_T,
                        codes + (size_t) k * (size_t) T + (size_t) slice_start, (size_t) slice_T * sizeof(int32_t));
        }
        std::vector<float> wav = pipeline_codec_decode(pc, slice.data(), K, slice_T);
        if (wav.empty()) {
            return std::vector<float>();
        }
        const size_t drop = (size_t) ctx * (size_t) TOKENIZER_HOP_LENGTH;
        if (wav.size() > drop) {
            out.insert(out.end(), wav.begin() + drop, wav.end());
        }
        start = end;
    }
    return out;
}

// Stateful streaming decoder over pipeline_codec_decode_stream: every
// pushed frame decodes immediately through the persistent codec state
// and emits its TOKENIZER_HOP_LENGTH samples, no buffering, no left
// context re-decode, no tail to drain at EOS. ICL priming feeds the
// full reference through the same state with the audio discarded,
// which matches the upstream reference plus generated decode exactly.
struct codec_stream_decoder {
    int  K;
    // Set true when push_frame returned false because the on_chunk
    // callback requested a cancel. Stays false on decode failures so
    // the caller can route to QT_STATUS_CANCELLED vs
    // QT_STATUS_GENERATE_FAILED on a negative return.
    bool cancelled;

    std::vector<float> frame;

    // Reset the persistent codec state to the zero context. Returns
    // false when the state allocation fails.
    bool init(PipelineCodec * pc, int K_) {
        K         = K_;
        cancelled = false;
        frame.assign((size_t) TOKENIZER_HOP_LENGTH, 0.0f);
        return pipeline_codec_stream_reset(pc);
    }

    // Prime the codec state with the full ICL reference: every frame
    // runs through the streaming decode with the audio discarded, so
    // the first generated frame sees the reference's exact causal
    // state. A reference already primed restores its snapshot device
    // to device instead of re-decoding; a fresh one saves its primed
    // state into the LRU. ref_kt is K major [K, ref_T]. Call once,
    // after init and before any push_frame.
    bool seed_reference(PipelineCodec * pc, const int32_t * ref_kt, int ref_T) {
        const uint64_t key = pipeline_codec_ref_key(ref_kt, K, ref_T);
        if (pipeline_codec_stream_restore(pc, key)) {
            return true;
        }
        std::vector<int32_t> codes((size_t) K);
        for (int t = 0; t < ref_T; t++) {
            for (int k = 0; k < K; k++) {
                codes[(size_t) k] = ref_kt[(size_t) k * (size_t) ref_T + (size_t) t];
            }
            if (!pipeline_codec_decode_stream(pc, codes.data(), NULL)) {
                return false;
            }
        }
        return pipeline_codec_stream_snapshot(pc, key);
    }

    // Decode one frame (K int32 codes, one per codebook) and emit its
    // samples through the callback. Returns false on decode failure or
    // when cb returns false (cancellation).
    bool push_frame(PipelineCodec * pc, const int32_t * frame_codes, qt_audio_chunk_cb cb, void * cb_ud) {
        if (!pipeline_codec_decode_stream(pc, frame_codes, frame.data())) {
            return false;
        }
        if (!cb(frame.data(), TOKENIZER_HOP_LENGTH, cb_ud)) {
            cancelled = true;
            return false;
        }
        return true;
    }
};
