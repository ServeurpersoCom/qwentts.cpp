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
// One entry point:
//
//   codec_chunked_decode : one shot decode of a full codes buffer.
//     Bit perfect equivalent of pipeline_codec_decode when the audio
//     fits in a single chunk_frames sized window. Bounds VRAM beyond
//     that, mirrors the upstream chunked_decode loop frame for frame.
//
// Frame by frame AR streaming lives in the batch engine
// (pipeline-tts.cpp), which drives the persistent multi lane stream
// state of pipeline-codec directly.

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
