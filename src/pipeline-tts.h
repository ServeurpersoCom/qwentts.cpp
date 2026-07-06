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

    // Persistent KV caches: the talker holds the LM context, the
    // predictor holds one frame's 16 sub-steps and gets reset every
    // frame in code_predictor_step.
    KVCache talker_kv;
    KVCache code_predictor_kv;

    // Hidden bridge: the talker last position hidden stays resident on
    // device. The talker graph copies it in, the code predictor prefill
    // graph reads it as a leaf, so the AR hot loop never round trips
    // the row through the host. [talker_hidden] f32 on `backend`.
    struct ggml_context * bridge_ctx;
    ggml_backend_buffer_t bridge_buf;
    struct ggml_tensor *  hidden_bridge;

    // Persistent graph arena for the talker prefill (T_ctx varies per
    // request, rebuilt through the sched). The talker decode and the
    // whole predictor run on static graphs instead: the decode keeps
    // one graph per attention window class built lazily, the predictor
    // one prefill (T=2) plus one per acoustic step built at load, all
    // replayed directly on the backend.
    GraphArena                     talker_arena;
    std::vector<TalkerDecodeGraph> talker_decode_graphs;  // one per 256 step window class, lazy
    CodePredGraph                  cp_prefill_graph;
    std::vector<CodePredGraph>     cp_step_graphs;
};

// Open the talker GGUF and the codec GGUF, load every module on the
// shared backend. Aborts with a logged error on any missing tensor or
// invalid metadata. use_fa is gated on bp.has_gpu inside the load:
// CPU only runs always use the manual F32 attention chain. clamp_fp16
// is forwarded as is. Caller frees with pipeline_tts_free.
bool pipeline_tts_load(PipelineTTS * pt,
                       const char *  talker_gguf_path,
                       const char *  codec_gguf_path,
                       BackendPair   bp,
                       bool          use_fa,
                       bool          clamp_fp16);

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
