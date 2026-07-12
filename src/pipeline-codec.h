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
#include "graph-arena.h"
#include "kv-cache.h"
#include "quantizer-decode.h"
#include "quantizer-encode.h"
#include "seanet-encoder.h"
#include "tokenizer-transformer.h"
#include "weight-ctx.h"

#include <cstdint>
#include <vector>

#define TOKENIZER_HOP_LENGTH    1920
#define TOKENIZER_SAMPLE_RATE   24000
#define TOKENIZER_NUM_CODEBOOKS 16
#define TOKENIZER_CODE_BITS     11

// Primed stream state snapshots kept per reference, LRU evicted.
static const int CODEC_SNAP_SLOTS = 8;

// Chunk width classes of the streaming decode: T in {1, 2, 4, 8}.
static const int CODEC_STREAM_CLASSES = 4;

// One static stream graph of chunk width T: built once, allocated
// once, replayed with per chunk uploads of codes, positions, ring
// rows, and the sliding window mask.
struct CodecStreamGraph {
    struct ggml_context * ctx    = nullptr;
    struct ggml_cgraph *  gf     = nullptr;
    ggml_gallocr_t        galloc = nullptr;
    struct ggml_tensor *  codes  = nullptr;
    struct ggml_tensor *  pos    = nullptr;
    struct ggml_tensor *  rows   = nullptr;
    struct ggml_tensor *  mask   = nullptr;
    struct ggml_tensor *  out    = nullptr;
    std::vector<int32_t>  pos_buf;
    std::vector<int64_t>  rows_buf;
    std::vector<float>    mask_buf;
};

// One primed stream state snapshot: a mirror of every conv context and
// KV ring tensor in a single backend buffer, plus the host position
// cursor. key is the content hash of the reference codes, stamp orders
// the slots for LRU eviction, stamp zero marks an empty slot. Mirrors
// allocate lazily on the slot's first save.
struct CodecStateSnap {
    uint64_t              key;
    uint64_t              stamp;
    int                   pos;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buf;
};

struct PipelineCodec {
    GGUFModel gguf;

    // Decode side modules
    QwenQuantizerDecoder     qdec;
    QwenTokenizerTransformer transformer;
    QwenUpsampleStage        upsample;
    QwenDACDecoder           dac;

    // pre_conv: causal Conv1d k=3, 512 -> 1024. Owns a dedicated
    // WeightCtx because it is the only module without one.
    struct ggml_tensor * pre_conv_w;  // [3, 512, 1024] f32
    struct ggml_tensor * pre_conv_b;  // [1024] f32
    WeightCtx            pre_conv_wctx;

    // Encode side modules
    QwenSEANetEncoder      seanet;
    QwenEncoderTransformer enc_transformer;
    QwenEncoderDownsample  enc_downsample;

    // Encoder weights (seanet, enc_transformer, enc_downsample, qenc)
    // load lazily on the first pipeline_codec_encode call: synthesis
    // from pre encoded reference codes never pays for them.
    bool                enc_loaded;
    QwenQuantizerEncode qenc;

    // Persistent graph arena for pipeline_codec_decode: stable node
    // addresses across rebuilds keep the backend CUDA graph cache hot,
    // so constant size streaming slices replay a captured executable.
    GraphArena dec_arena;

    // Stateful streaming decoder: every causal conv left context, every
    // transposed conv overlap carry, and the transformer sliding window
    // KV ring live as backend resident tensors, so a T=1 frame decode
    // reproduces the offline full decode exactly with zero re-decoded
    // context. Loaded lazily on the first pipeline_codec_stream_reset:
    // the buffered chunked path never pays for it.
    bool                    stream_ready;
    struct ggml_context *   stream_ctx;
    ggml_backend_buffer_t   stream_buf;
    struct ggml_tensor *    stream_pre_conv;  // pre_conv k=3, [2, 512]
    QwenUpsampleStreamState stream_up;
    QwenDACStreamState      stream_dac;
    KVCache                 stream_kv;   // tok transformer ring, [hd, ring, n_kv] per layer
    int                     stream_pos;  // absolute frame position, drives RoPE and ring slots

    // Static frame graph: the T=1 topology and every tensor address are
    // constant, so the graph builds and allocates once and every frame
    // is input uploads + one backend compute + one readback. The single
    // backend runs the whole graph, no scheduler involved.
    // Static stream graphs, one per chunk width class (T = 1 << cls).
    // Every class shares the stream state and KV ring tensors above;
    // only the input shapes and the intermediates differ. Classes
    // build lazily on their first decode.
    CodecStreamGraph stream_graphs[CODEC_STREAM_CLASSES];

    // Snapshot LRU over the stream state: after an ICL reference
    // priming the conv contexts, KV ring, and position copy device to
    // device into the slot keyed by the reference content hash, so a
    // repeated reference restores in one pass of tensor copies instead
    // of re-decoding every reference frame.
    CodecStateSnap snaps[CODEC_SNAP_SLOTS];
    uint64_t       snap_stamp;

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

// Load the encoder weights on demand. Idempotent, called by
// pipeline_codec_encode; harmless to call when already resident.
bool pipeline_codec_ensure_encoder(PipelineCodec * pc);

// Decode RVQ codes into a 24 kHz mono waveform.
//   codes: flat int32 buffer, [K, T] row-major (T fastest).
// Returns audio of length T * TOKENIZER_HOP_LENGTH, empty on failure.
std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int K, int T);

// Reset the stateful streaming decoder to the zero context: allocates
// the state tensors on first call, clears every conv left context,
// transposed conv carry, and the transformer KV ring, and rewinds the
// absolute position. Call once before each streamed utterance.
bool pipeline_codec_stream_reset(PipelineCodec * pc);

// Decode one frame through the stateful streaming path. codes holds the
// K codebook entries of a single frame; the persistent state advances
// as a side effect. When audio_out is non NULL the frame's
// TOKENIZER_HOP_LENGTH samples copy into it; a NULL audio_out primes
// the state without a readback (ICL reference priming).
// Decode one chunk of T frames (T in {1, 2, 4, 8}) through the
// persistent stream state. codes is [T, K] with T contiguous per
// codebook; audio_out receives T * TOKENIZER_HOP_LENGTH samples, NULL
// discards the audio (reference priming).
bool pipeline_codec_decode_stream(PipelineCodec * pc, const int32_t * codes, int T, float * audio_out);

// Content hash of an ICL reference, the snapshot LRU key.
//   codes: flat int32, [K, T] row-major.
uint64_t pipeline_codec_ref_key(const int32_t * codes, int K, int T);

// Restore the stream state from the snapshot slot matching key.
// Returns false on a miss; the caller then primes and snapshots.
bool pipeline_codec_stream_restore(PipelineCodec * pc, uint64_t key);

// Save the current stream state into the LRU slot for key, evicting
// the least recently used slot when all are taken.
bool pipeline_codec_stream_snapshot(PipelineCodec * pc, uint64_t key);

// Encode a 24 kHz mono waveform into RVQ codes.
//   audio    : [n_samples] f32 mono 24 kHz. Must be a multiple of
//              TOKENIZER_HOP_LENGTH (1920); the caller is expected
//              to pad with zeros if needed.
//   dump_dir: optional path. When non NULL, dumps the SEANet, encoder
//              transformer and post-downsample (pre-FSQ latents) buffers
//              into seanet-out.bin, enc-transformer-out.bin and
//              codec-pre-fsq.bin under that directory. Quiet otherwise.
// Returns codes flat as [K, T] row-major, K = TOKENIZER_NUM_CODEBOOKS,
// T = n_samples / 1920. Empty on failure.
std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc,
                                           const float *   audio,
                                           int             n_samples,
                                           const char *    dump_dir = NULL);

// Free every backend buffer and ggml context. Safe to call on a zeroed struct.
void pipeline_codec_free(PipelineCodec * pc);
