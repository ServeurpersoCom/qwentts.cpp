#pragma once
// pipeline-tts.h: full TTS pipeline composition (Talker LM + code
// predictor MTP head + optional speaker encoder + 12Hz codec decoder).
//
// pipeline_tts_load opens the talker GGUF and the codec GGUF, parses
// every typed metadata block (specials, languages, speakers,
// generation defaults), loads every weight tensor on the shared
// backend and initialises both KV caches. pipeline_tts_synthesize
// runs the prompt assembly, the autoregressive frame loop and the
// codec decode in one pass; it fills the public qt_audio struct
// directly so the facade in qwen.cpp stays a thin wrapper.

#include "backend.h"
#include "code-predictor-graph.h"
#include "code-predictor-weights.h"
#include "ggml-backend.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "kv-cache.h"
#include "pipeline-codec.h"
#include "qwen.h"
#include "speaker-encoder-weights.h"
#include "talker-decode-graph.h"
#include "talker-weights.h"

#include <cstdint>
#include <string>
#include <vector>

struct CodecSpecials {
    int pad_id;
    int bos_id;
    int eos_id;
    int think_id;
    int nothink_id;
    int think_bos_id;
    int think_eos_id;
};

struct TextSpecials {
    int im_start_id;
    int im_end_id;
    int tts_pad_id;
    int tts_bos_id;
    int tts_eos_id;
};

struct LanguageEntry {
    std::string name;
    int         id;
};

// Speaker entry for CustomVoice variants. id is the codec embedding row id
// inserted in the talker prefix, dialect is empty unless the speaker
// overrides the user supplied language with a dialect lang_id (eric ->
// sichuan_dialect, dylan -> beijing_dialect on the upstream checkpoint).
struct SpeakerEntry {
    std::string name;
    int         id;
    std::string dialect;
};

struct GenerationDefaults {
    bool  do_sample;
    int   top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    bool  subtalker_do_sample;
    int   subtalker_top_k;
    float subtalker_top_p;
    float subtalker_temperature;
    int   max_new_tokens;
};

struct PromptPrefixCacheEntry {
    std::string        key;
    int                rows;
    std::vector<float> input_embed_prefix;
};

struct PromptCache {
    bool                                initialized;
    std::vector<float>                  tts_bos_emb;
    std::vector<float>                  tts_eos_emb;
    std::vector<float>                  tts_pad_emb;
    std::vector<float>                  codec_pad_emb;
    std::vector<float>                  codec_bos_emb;
    std::vector<PromptPrefixCacheEntry> prefix_entries;
    size_t                              max_prefix_entries;
};

// One set of static predictor graphs for a given batch width: the T=2
// prefill plus one T=1 step per acoustic codebook after the first.
struct CodePredGraphSet {
    CodePredGraph              prefill;
    std::vector<CodePredGraph> steps;
};

struct PipelineTTS {
    GGUFModel             gguf_talker;
    TalkerWeights         talker;
    CodePredictorWeights  code_predictor;
    SpeakerEncoderWeights speaker_encoder;
    bool                  has_speaker_encoder;

    // Speaker encoder weights residency: loaded lazily on the first
    // reference audio request, see pipeline-tts.cpp.
    bool spk_enc_loaded;

    PipelineCodec codec;

    std::string tokenizer_type;
    std::string model_size;
    std::string model_type;
    int         num_code_groups;

    // Batch capacity: number of KV sets, bridge columns and maximum
    // concurrent slots the batch engine drives. 1 keeps the exact
    // single sequence layout and behavior.
    int max_batch;

    CodecSpecials              codec_specials;
    TextSpecials               text_specials;
    std::vector<LanguageEntry> languages;
    std::vector<SpeakerEntry>  speakers;
    GenerationDefaults         gen_defaults;
    PromptCache                prompt_cache;

    BackendPair          bp;
    ggml_backend_t       backend;
    ggml_backend_sched_t sched;

    // Attention path config, set at load and forwarded to every
    // talker / code predictor forward. use_flash_attn collapses to
    // false on CPU only backends (fused FA needs a GPU kernel).
    // clamp_fp16 inserts ggml_clamp on V before attention and on the
    // residual stream between blocks to guard FP16 matmul accumulation
    // on sub Ampere CUDA targets.
    bool use_flash_attn;
    bool clamp_fp16;

    // Persistent KV caches, one set per slot: the talker holds the LM
    // contexts, the predictor holds one frame's 16 sub-steps per slot,
    // rewritten every frame at baked rows.
    KVCache talker_kv;
    KVCache code_predictor_kv;

    // Hidden bridge: the talker last position hidden of every slot
    // stays resident on device as one column of [talker_hidden,
    // max_batch] f32. The talker graphs copy their columns in, the code
    // predictor prefill graph reads them as a leaf, so the AR hot loop
    // never round trips the rows through the host.
    struct ggml_context * bridge_ctx;
    ggml_backend_buffer_t bridge_buf;
    struct ggml_tensor *  hidden_bridge;

    // Persistent graph arena for the talker prefill (T_ctx varies per
    // request, rebuilt through the sched). The talker decode and the
    // whole predictor run on static batched graphs instead: the decode
    // keeps one graph per attention window class built lazily and
    // rebuilt when the batch width changes, the predictor one graph
    // set (prefill T=2 plus one per acoustic step) per batch width
    // built lazily, all replayed directly on the backend.
    GraphArena                     talker_arena;
    std::vector<TalkerDecodeGraph> talker_decode_graphs;  // one per 256 step window class, lazy
    std::vector<CodePredGraphSet>  cp_graphs;             // index N - 1, lazy per batch width
};

// Open the talker GGUF and the codec GGUF, load every module on the
// shared backend. Aborts with a logged error on any missing tensor or
// invalid metadata. use_fa is gated on bp.has_gpu inside the load:
// CPU only runs always use the manual F32 attention chain. clamp_fp16
// is forwarded as is. max_batch sizes the KV sets, the bridge columns
// and the maximum concurrent slots (minimum 1). max_prefill_tokens > 0
// runs one throwaway talker prefill of that length (clamped to the KV
// cache's max_seq_len) before returning, forcing ggml_backend_sched to
// reserve the compute buffer for that prefill bucket up front instead
// of growing it on a live request; failure is fatal (returns false), by
// design -- a deployment that can't afford its configured worst case
// should refuse to start. 0 skips the reservation. Caller frees with
// pipeline_tts_free.
bool pipeline_tts_load(PipelineTTS * pt,
                       const char *  talker_gguf_path,
                       const char *  codec_gguf_path,
                       BackendPair   bp,
                       bool          use_fa,
                       bool          clamp_fp16,
                       int           max_batch,
                       int           max_prefill_tokens);

void pipeline_tts_free(PipelineTTS * pt);

struct BPETokenizer;

// Run the full TTS pipeline: prompt assembly, prefill, frame loop with
// sampling, codec decode, fill qt_audio. Reads every knob (text,
// references, sampling, cancel, on_chunk, ...) straight from the
// public qt_tts_params struct so the facade in qwen.cpp can hand it
// off verbatim after the mode validation and seed resolve.
//
// Returns QT_STATUS_OK on success. On any failure returns a negative
// qt_status with a diagnostic already routed through qt_log /
// qt_set_error and leaves `out` empty. QT_STATUS_CANCELLED is returned
// when params->cancel or params->on_chunk returns true / false
// respectively during the AR loop.
//
// In buffered mode (params->on_chunk == NULL) the synthesised waveform
// is malloc allocated into out->samples; the caller releases it with
// qt_audio_free. In streaming mode (params->on_chunk != NULL) audio is
// emitted through the callback as decoded chunks and out->samples
// stays NULL on success.
//
// resolved_seed is the seed actually used for sampling: qt_synthesize
// hands over the same value it logged so dump traces and replays line
// up across runs even when params->seed was -1.
qt_status pipeline_tts_synthesize(PipelineTTS *                pt,
                                  BPETokenizer *               tok,
                                  const struct qt_tts_params * params,
                                  int64_t                      resolved_seed,
                                  struct qt_audio *            out);

// Convert a duration in seconds to a frame count at the codec frame
// rate (24000 / TOKENIZER_HOP_LENGTH). Clamps to a
// minimum of one frame.
int pipeline_tts_duration_sec_to_tokens(const PipelineTTS * pt, float duration_sec);

// One synthesis request driven by the batch engine. The caller owns
// params / out for the whole lifetime of the job; status and error
// fill at retirement. error carries the qt_last_error() text captured
// on the thread that ran the engine, so a scheduler on a worker thread
// can replay it into the caller's thread local slot. done is reserved
// for the owner's completion signaling; the engine never touches it.
struct TtsJob {
    const struct qt_tts_params * params;
    int64_t                      resolved_seed;
    struct qt_audio *            out;
    qt_status                    status;
    std::string                  error;
    bool                         done;
};

// Batch engine: drives up to pt->max_batch concurrent synthesis slots
// in lockstep over the batched talker decode and code predictor
// graphs. Slots always occupy KV sets [0, N); a retirement compacts
// the range with one device side set copy so the batched views stay
// consecutive. Single threaded: every call runs on the thread that
// owns the GPU. pipeline_tts_synthesize drives a transient engine
// synchronously for the one request case; the facade scheduler keeps a
// long lived one on its worker thread.
struct TtsEngine;

TtsEngine * tts_engine_new(PipelineTTS * pt, BPETokenizer * tok);
void        tts_engine_free(TtsEngine * e);

// Admit one job: prompt assembly, reference handling and talker
// prefill into the next free slot (KV set N). Returns false when the
// admit fails; job->status and job->error are then final and the job
// never occupies a slot. Logs the prefill stall the join imposes on
// the already active slots.
bool tts_engine_admit(TtsEngine * e, TtsJob * job);

// Run one frame for every active slot: batched talker decode for the
// slots past their prefill, per slot c0 sampling, batched code
// predictor, per slot codec streaming, then retirement of finished
// slots (EOS, max_new_tokens, cancel, error) including their buffered
// codec decode or streaming drain. Retired jobs append to *retired
// with status, error and out final.
void tts_engine_step(TtsEngine * e, std::vector<TtsJob *> * retired);

// Number of currently occupied slots.
int tts_engine_active(const TtsEngine * e);
