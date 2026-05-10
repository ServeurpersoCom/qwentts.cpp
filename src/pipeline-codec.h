#pragma once
// pipeline-codec.h: codec decode pipeline for the Qwen3-TTS 12Hz tokenizer.
// Loads a codec GGUF (quantizer + pre_conv + pre_transformer + upsample +
// DAC), holds every weight on the backend, and exposes a one-shot decode:
//
//   codes [num_codebooks, T] i32  ->  audio [T * 1920] f32 mono 24 kHz
//
// Layout flow inside the graph:
//   codes     [T, K] i32        T-first
//     v quantizer.decode
//   hidden    [512, T] f32       C-first
//     v transpose to T-first
//     v pre_conv (causal Conv1d k=3, 512 -> 1024)
//   hidden    [T, 1024] f32      T-first
//     v transpose to C-first
//     v pre_transformer (8 layers Qwen3, sliding window 72 causal)
//   hidden    [1024, T] f32      C-first
//     v transpose to T-first
//     v upsample stage (4x)
//   hidden    [T*4, 1024] f32    T-first
//     v DAC decoder (480x)
//   audio     [T*1920, 1] f32    T-first
//     v ggml_clamp(-1, 1)
//   audio_out [T*1920, 1] f32    T-first

#include "backend.h"
#include "convnext-block.h"
#include "dac-decoder-v2.h"
#include "encoder-downsample.h"
#include "encoder-transformer.h"
#include "ggml-backend.h"
#include "gguf-weights.h"
#include "quantizer-decode.h"
#include "quantizer-encode.h"
#include "seanet-encoder.h"
#include "tokenizer-transformer.h"
#include "weight-ctx.h"

#include <cstdint>
#include <vector>

#define QWEN_TOKENIZER_HOP_LENGTH    1920
#define QWEN_TOKENIZER_SAMPLE_RATE   24000
#define QWEN_TOKENIZER_NUM_CODEBOOKS 16
#define QWEN_TOKENIZER_CODE_BITS     11

struct PipelineCodec {
    GGUFModel gguf;

    // Decode side modules
    QwenQuantizerDecoder     qdec;
    QwenTokenizerTransformer transformer;
    QwenUpsampleStage        upsample;
    QwenDACDecoder           dac;

    // pre_conv: causal Conv1d k=3, 512 -> 1024. Loaded into a dedicated
    // weight ctx because it is the only module that does not own one.
    struct ggml_tensor *  pre_conv_w;  // [3, 512, 1024] f32
    struct ggml_tensor *  pre_conv_b;  // [1024] f32
    struct ggml_context * pre_conv_ctx;
    ggml_backend_buffer_t pre_conv_buf;

    // Encode side modules
    QwenSEANetEncoder      seanet;
    QwenEncoderTransformer enc_transformer;
    QwenEncoderDownsample  enc_downsample;
    QwenQuantizerEncode    qenc;

    // CPU mirror of the RVQ encode side, lazy-loaded on first encode call.
    QwenQuantizerEncodeHost qenc_sem_host;
    QwenQuantizerEncodeHost qenc_aco_host;
    bool                    qenc_host_ready;

    BackendPair          bp;
    ggml_backend_t       backend;
    ggml_backend_sched_t sched;
};

// Open the GGUF, load every module on the backend, build the scheduler.
// On failure leaves the struct in a clean state and returns false.
bool pipeline_codec_load(PipelineCodec * pc, const char * gguf_path, BackendPair bp);

// Decode RVQ codes into a 24 kHz mono waveform.
//   codes: flat int32 buffer, [K, T] row-major (T fastest).
// Returns audio of length T * QWEN_TOKENIZER_HOP_LENGTH, empty on failure.
std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int K, int T);

// Encode a 24 kHz mono waveform into RVQ codes.
//   audio    : [n_samples] f32 mono 24 kHz. Must be a multiple of
//              QWEN_TOKENIZER_HOP_LENGTH (1920); the caller is expected
//              to pad with zeros if needed.
//   dump_dir : optional path. When non NULL, dumps the SEANet, encoder
//              transformer and post-downsample (pre-FSQ latents) buffers
//              into seanet-out.bin, enc-transformer-out.bin and
//              codec-pre-fsq.bin under that directory. Quiet otherwise.
// Returns codes flat as [K, T] row-major, K = QWEN_TOKENIZER_NUM_CODEBOOKS,
// T = n_samples / 1920. Empty on failure.
std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc,
                                           const float *   audio,
                                           int             n_samples,
                                           const char *    dump_dir = NULL);

// Free every backend buffer and ggml context. Safe to call on a zeroed struct.
void pipeline_codec_free(PipelineCodec * pc);
