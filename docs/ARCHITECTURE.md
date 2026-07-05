# Architecture

Technical reference for qwentts.cpp, the GGML port of Qwen3-TTS 12 Hz
(Qwen team, Alibaba). This document covers the model, the conversion to
GGUF, the inference pipeline, the GGML graph conventions, and the CLI
tools.

## Upstream model

Qwen3-TTS 12 Hz (Qwen team / Alibaba, Apache 2.0) is a multilingual
zero shot text-to-speech system covering 11 languages with Mandarin
dialect support. It targets three checkpoint families :

  base          plain synthesis with an auto-picked voice, plus zero
                shot voice cloning from a reference clip
  custom_voice  named speakers selected by name, some carrying a
                dialect override
  voice_design  a synthesised speaker driven by a free text attribute
                instruction

The system is autoregressive. Two language models run in series : a
Talker that emits the semantic codebook one frame at a time, and a Code
Predictor MTP head that expands each semantic token into the 15
acoustic codes of that frame. The codes are turned into a waveform by a
separate audio tokenizer, the Qwen3-TTS-Tokenizer-12Hz (Mimi-style
SEANet plus a transformer, residual vector quantiser, a ConvNeXt
upsampler and a DAC decoder, the Descript Audio Codec family), running
at 12.5 frames per second over 24 kHz mono audio.

Public checkpoints, two talker sizes :

  Talker          Qwen3 0.6B or 1.7B decoder, codec_head over 3072
  Code Predictor  5-layer Qwen3, 15 acoustic codebooks of 2048 each
  Speaker encoder ECAPA-TDNN, base checkpoints only (x-vector cloning)
  Audio codebooks 16 residual (1 semantic + 15 acoustic), 2048 each
  Audio framerate 12.5 Hz
  Hop length      1920 samples
  Sample rate     24 kHz mono
  Semantic SR     24 kHz (SEANet input, no separate semantic rate)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
cd qwentts.cpp
./buildcuda.sh      # NVIDIA GPU
./buildvulkan.sh    # AMD/Intel GPU (Vulkan)
./buildcpu.sh       # CPU only
./buildall.sh       # all backends, runtime DL loading
```

The GGML submodule lives at `https://github.com/ServeurpersoCom/ggml.git`
and provides the two custom ops the codec needs : `GGML_OP_SNAKE` and
`GGML_OP_COL2IM_1D`. Both have CPU, CUDA, Metal, and Vulkan kernels.

## Model conversion

```
./checkpoints.sh    # hf download Qwen/Qwen3-TTS-12Hz-* -> checkpoints/
./convert.py        # F32 GGUFs (one talker per mode/size + tokenizer) -> models/
./quantize.sh       # BF16 / Q8_0 / Q4_K_M derived from each F32 source
```

`convert.py` writes the F32 source of truth. Each talker checkpoint
produces one `qwen-talker-{size}-{mode}-{variant}.gguf`, and the shared
tokenizer produces one `qwen-tokenizer-12hz-{variant}.gguf`. The talker
and the Code Predictor share the same Qwen3 layout so a single tensor
renamer covers both.

Quantisation policy, centralised in `tools/quantize.cpp` should_quantize
and mirrored in `quantize.sh` :

  RVQ codebooks (`quantizer.quantizers.*`), the input_proj / output_proj
  that wrap them, and the speaker encoder stay at F32 in every variant.
  Nearest-neighbour lookup is sensitive to per-row quantisation noise ;
  even BF16 mantissa truncation drifts codes enough to break voice
  fidelity.

  1D tensors (LayerScale gamma, biases, norms, snake alpha and beta)
  stay at F32.

  Conv kernels with non-alignable rows (K = 7, 3, 1) never divide a
  K-quant block size, so the quantiser lands on F16 directly. F16 has no
  block size and matches the runtime target dtype on every backend.

  The Talker LM (hidden divisible by 256) follows standard llama.cpp
  K-quant. The Code Predictor MTP head and the speaker encoder live in
  the talker GGUF and inherit its quantisation.

## GGUF layout

`qwen-talker-{size}-{mode}-{variant}.gguf` (arch `qwen3-tts`, 404
tensors on the 1.7B voice_design build) :

```
metadata
  general.architecture                          qwen3-tts
  qwen3-tts.tokenizer_type                       qwen3_tts_tokenizer_12hz
  qwen3-tts.model_size                           0.6b | 1.7b
  qwen3-tts.model_type                           base | custom_voice | voice_design
  qwen3-tts.num_code_groups                      16

  qwen3-tts.talker.embedding_length              1024 (0.6B) | 2048 (1.7B)
  qwen3-tts.talker.feed_forward_length           3072 (0.6B) | 6144 (1.7B)
  qwen3-tts.talker.block_count                    28
  qwen3-tts.talker.attention.head_count           16
  qwen3-tts.talker.attention.head_count_kv         8     (GQA 2:1)
  qwen3-tts.talker.attention.key_length          128
  qwen3-tts.talker.vocab_size                    3072   (codec_head)
  qwen3-tts.talker.text_vocab_size               151936
  qwen3-tts.talker.text_hidden_size              2048   (text-embedding width, both sizes)
  qwen3-tts.talker.context_length                32768
  qwen3-tts.talker.rope.freq_base                1e6
  qwen3-tts.talker.attention.layer_norm_rms_epsilon  1e-6
  qwen3-tts.talker.position_id_per_seconds       13
  qwen3-tts.talker.rope.mrope_section            [24, 20, 20]
  qwen3-tts.talker.mrope_interleaved             false

  qwen3-tts.code_pred.embedding_length           1024   (both sizes)
  qwen3-tts.code_pred.feed_forward_length        3072
  qwen3-tts.code_pred.block_count                 5
  qwen3-tts.code_pred.attention.head_count        16
  qwen3-tts.code_pred.attention.head_count_kv      8
  qwen3-tts.code_pred.attention.key_length       128
  qwen3-tts.code_pred.vocab_size                 2048
  qwen3-tts.code_pred.context_length             65536
  qwen3-tts.code_pred.attention.layer_norm_rms_epsilon  1e-6
  qwen3-tts.code_pred.rope.freq_base             1e6

  qwen3-tts.spk_enc.embedding_length             2048   (base only)
  qwen3-tts.spk_enc.sample_rate                  16000  (base only)

  qwen3-tts.codec.{pad,bos,eos,think,nothink,think_bos,think_eos}_id
  qwen3-tts.codec.language_names / language_ids
  qwen3-tts.codec.speaker_names / speaker_ids / speaker_dialects  (custom_voice)
  qwen3-tts.text.{im_start,im_end,tts_pad,tts_bos,tts_eos}_id
  generation.*                                   sampling defaults
  tokenizer (Qwen2 BPE, 151676 vocab, 151291 merges, eos 151643)

tensors
  talker.text_embd.weight                        text token embedding
  talker.codec_embd.weight                       (3072, hidden) codec/code embedding
  talker.text_proj.fc1.{weight,bias}             text-embedding -> hidden, 2-layer
  talker.text_proj.fc2.{weight,bias}
  talker.codec_head.weight                       hidden -> 3072, codebook 0 logits
  talker.output_norm.weight                      final RMSNorm
  talker.blk.0..27.attn_q / attn_k / attn_v / attn_o          GQA, no bias
  talker.blk.0..27.attn_q.q_norm / attn_k.k_norm              per-head RMSNorm (128,)
  talker.blk.0..27.attn_norm / ffn_norm                       RMSNorm
  talker.blk.0..27.ffn.{gate,up,down}_proj                    SwiGLU, no bias
  code_pred.blk.0..4.*                           same layout, 5 layers
  code_pred.output_norm.weight
  code_pred.mtp_proj.{weight,bias}               talker_hidden -> code_pred hidden (1.7B only, identity on 0.6B)
  spk_enc.*                                      ECAPA-TDNN (base only)
```

`qwen-tokenizer-12hz-{variant}.gguf` (arch `qwen3-tts-tokenizer`, 398
tensors) :

```
metadata
  qwen3-tts-tokenizer.input_sample_rate          24000
  qwen3-tts-tokenizer.output_sample_rate         24000
  qwen3-tts-tokenizer.decode_upsample_rate       1920
  qwen3-tts-tokenizer.encode_downsample_rate     1920
  qwen3-tts-tokenizer.encoder_valid_num_quantizers  16

  qwen3-tts-tokenizer.decoder.latent_dim         1024
  qwen3-tts-tokenizer.decoder.codebook_size      2048
  qwen3-tts-tokenizer.decoder.codebook_dim_internal  256
  qwen3-tts-tokenizer.decoder.hidden_size        512
  qwen3-tts-tokenizer.decoder.intermediate_size  1024
  qwen3-tts-tokenizer.decoder.head_dim            64
  qwen3-tts-tokenizer.decoder.num_attention_heads 16
  qwen3-tts-tokenizer.decoder.num_key_value_heads 16    (no GQA)
  qwen3-tts-tokenizer.decoder.num_hidden_layers    8
  qwen3-tts-tokenizer.decoder.num_quantizers      16
  qwen3-tts-tokenizer.decoder.num_semantic_quantizers 1
  qwen3-tts-tokenizer.decoder.rope_theta         10000
  qwen3-tts-tokenizer.decoder.sliding_window      72
  qwen3-tts-tokenizer.decoder.upsampling_ratios  [2, 2]   (ConvNeXt upsample, DAC strides 8/5/4/3 are internal)
  qwen3-tts-tokenizer.decoder.layer_scale_initial_scale

  qwen3-tts-tokenizer.encoder.num_filters         64
  qwen3-tts-tokenizer.encoder.upsampling_ratios  [8, 6, 5, 4]   (SEANet, reversed at encode)
  qwen3-tts-tokenizer.encoder.hidden_size        512
  qwen3-tts-tokenizer.encoder.intermediate_size  2048
  qwen3-tts-tokenizer.encoder.num_attention_heads 8
  qwen3-tts-tokenizer.encoder.num_hidden_layers    8
  qwen3-tts-tokenizer.encoder.rope_theta         10000
  qwen3-tts-tokenizer.encoder.codebook_size      2048
  qwen3-tts-tokenizer.encoder.num_quantizers      16

tensors
  tok_enc.*                  SEANet conv stack, encoder transformer,
                             downsample conv, RVQ encode (vq_first / vq_rest)
  tok_dec.pre_conv.*         conv_pre into the decoder transformer
  tok_dec.pre_tfm.input_proj / output_proj / norm + transformer blocks
  tok_dec.upsample.*         ConvNeXt upsample (2 blocks, 4x)
  tok_dec.<dac>.*            DAC decoder chain
  tok_dec.vq_first.output_proj / vq_rest.output_proj
  tok_{enc,dec}.{vq_*}.<idx>.codebook            RVQ codebook entries
```

## Component architecture

### Talker LM

Standard Qwen3 decoder, KV cached. 28 layers, 16 query heads, 8 KV
heads (GQA 2:1), head_dim 128, RoPE theta 1e6, per-head RMSNorm on Q and
K before RoPE, SwiGLU MLP, no bias on Q/K/V/O/MLP, RMS eps 1e-6. The two
sizes differ only in width : hidden 1024 / FFN 3072 on the 0.6B,
hidden 2048 / FFN 6144 on the 1.7B. Context length 32768, text vocab
151936, text embedding width 2048.

Two input streams, pad-aligned and summed into one embedding sequence :

```
text  stream : text_proj(text_embd(text_ids))      text vocab -> hidden
codec stream : codec_embd(codec_ids)               3072 -> hidden
input        : text_stream + codec_stream          [T_ctx, hidden]
```

The reference multimodal RoPE carries three sections (`mrope_section
[24, 20, 20]`), but a TTS prompt is a single text-plus-codec timeline,
so the forward collapses the sections to a plain 1D NEOX rope.

The final hidden state is RMS-normalised through `output_norm` and
projected through `codec_head` to the 3072-entry codebook 0 logits. The
sampler masks the reserved top range `[vocab - 1024, vocab)` except the
codec EOS before applying the sampling chain. There is no text `lm_head`
on this path.

### Code Predictor MTP head

A 5-layer Qwen3 stack, hidden 1024, heads 16/8, head_dim 128, FFN 3072,
RoPE 1e6, context length 65536, with its own KV cache. The predictor
hidden is 1024 on both talker sizes, so `mtp_proj` (which maps the
talker hidden onto the predictor hidden) is an identity on the 0.6B
(1024 == 1024) and a learned linear on the 1.7B (2048 -> 1024). The load
log prints `mtp_proj identity` or `mtp_proj linear` accordingly.

```
input : talker_hidden_last [hidden] -- last position from the Talker, post final norm
        c0                  -- semantic code sampled from codec_head (codebook 0)
output: codes[16] = [c0, c1, ..., c15] -- full code set for one frame
```

The predictor cache is local to a single frame. It is reset every
frame, prefilled with the first two positions (`mtp_proj(talker_hidden)`
and `embed(c0)`), then decodes 14 single-token steps. That drops the
inner work from `O(sum_{g=0..14} (g+2)^2)` to `O(16)` per frame, about
90x for the inner loop. Each of the 15 acoustic heads samples over its
own 2048-entry codebook.

### Speaker encoder (base checkpoints only)

An ECAPA-TDNN, present only in the base checkpoints, where it extracts
a fixed x-vector from the reference clip for voice cloning. The
custom_voice checkpoints carry no speaker encoder : their named speakers
are precomputed codec-embedding rows. voice_design carries neither.

```
audio [T_pad] -> mel [128, T] -> conv0 TDNN k=5 + ReLU [512, T]
              -> SE-Res2Net dil 2 / 3 / 4
              -> cat blk[1..3] + MFA k=1 + ReLU [1536, T]
              -> ASP attentive pooling [3072, 1]
              -> FC k=1 [2048, 1] -> squeeze [2048]
```

The forward fuses mel extraction so the whole speaker path is one graph.
Tensors live under `spk_enc.*` (conv0, blk.N, mfa, asp.tdnn, asp.conv,
fc) and stay at F32 in every quant.

### Audio tokenizer encoder

`omnivoice-codec` style round-trip, here `qwen-codec`. The encode path
turns 24 kHz audio into 16 RVQ codes at 12.5 Hz :

```
audio 24 kHz mono
  -> SEANet : init Conv1d k=7 (1 -> 64), 4 stages ratios 4/5/6/8
             (cumulative 960x, 64 -> 512 ch), last Conv1d k=3
  -> encoder transformer : 8 layers, hidden 512, heads 8/8, head_dim 64,
     FFN 2048, RoPE 10000, LayerNorm with bias, plain GELU MLP, LayerScale,
     pure causal attention (no GQA, no sliding window applied)
  -> downsample conv k=4 stride=2 (512 -> 512), 25 Hz -> 12.5 Hz
  -> RVQ encode : input_proj 1x1 (512 -> 256), 16 codebooks of 2048 x 256,
     argmin over the residual, codebook 0 semantic + 1..15 acoustic
  -> codes [16, T] i32
```

### Audio tokenizer decoder

The decode path is the inverse, bounded in VRAM by a chunked roll :

```
codes [T, 16] i32
  -> RVQ decode : F.embedding per codebook, per-split output_proj 1x1
     (256 -> 512), sum the semantic and acoustic splits -> hidden [T, 512]
  -> conv_pre -> decoder transformer : 8 layers, hidden 512, heads 16/16,
     head_dim 64, FFN 1024, RoPE 10000, RMSNorm, SwiGLU, sliding window 72
     causal, LayerScale, input_proj 1024 -> 512 / output_proj 512 -> 1024
  -> ConvNeXt upsample : 2 blocks, each CausalTransConv k=2 stride=2 then a
     ConvNeXt block (depthwise causal conv k=7, pointwise 1024 -> 4096 -> 1024,
     LayerScale gamma); together 4x on the time axis at channels 1024
  -> DAC : conv_pre k=7 (1024 -> 1536), 4 blocks strides 8/5/4/3
     channels 1536 -> 768 -> 384 -> 192 -> 96, each block SnakeBeta then
     CausalTransConv then 3 ResUnits (dilations 1/3/9), snake_post,
     conv_post k=7 (96 -> 1)
  -> audio [T * 1920, 1] @ 24 kHz mono
```

SnakeBeta applies `exp()` to alpha and beta on every forward in the
reference ; both factors are precomputed CPU-side at load time so the
graph multiplies plain F32 buffers. The whole DAC pipeline runs T-first
(`ne[0] = T`, `ne[1] = C`) so the fused SNAKE op and `ggml_conv_1d`
share one layout.

### Chunked decode (buffered path)

A standalone codec decode of an isolated window shows edge artefacts at
the chunk boundary, because the causal conv kernels and the sliding
window attention have no left context. `codec_chunked_decode` prepends
`codec_left_context_sec` worth of previously decoded frames, decodes,
then strips the samples that belong to the left context. Defaults match
the upstream tokenizer : `codec_chunk_sec` 24.0 (300 frames at 12.5 Hz)
and `codec_left_context_sec` 2.0 (25 frames). The first chunk collapses
its left context to whatever is available. This routine serves the
buffered one-shot decode only.

### Streaming decode (stateful path)

Every module in the decoder is causal, so a stateful decode of one
frame at a time reproduces the offline full decode exactly, with no
re-decoded context and no chunk seams. Each stride-1 causal conv keeps
its left context ((k-1)*d input rows) in a persistent backend tensor
that the graph concats ahead of the fresh rows and refreshes in graph;
each DAC transposed conv carries its col2im overlap tail (kernel -
stride rows, bias free) into the next frame; the decoder transformer
attends over a 128-slot sliding window KV ring written through
set_rows, with the ring slot and the absolute RoPE position carried as
input data. All state clears to zero at reset, which matches the
offline zero left pads bit for bit.

The whole T=1 frame graph builds and allocates once
(`pipeline_codec_stream_ensure`, lazy) and computes directly on the
backend without the scheduler : a frame decode is four input uploads
(16 codes, position, ring slot, window mask), one graph compute, and a
1920-sample readback. The constant topology and tensor addresses keep
the CUDA graph cache in pure replay. ICL cloning primes the state by
running the full reference codes through the same path with the
readback skipped, matching the upstream reference-plus-generated
decode; the transformer receptive field (8 layers x window 72) exceeds
any reference length, so the full prime is the exact one.

The primed state feeds a per-reference snapshot LRU
(`CODEC_SNAP_SLOTS`): after a fresh prime the conv contexts, KV ring,
and position copy device to device into a slot keyed by the FNV-1a
hash of the reference codes, and a later request with the same
reference restores the exact state in one pass of tensor copies
instead of re-decoding every reference frame. Slots allocate lazily
and evict least recently used, so the priming cost amortizes across
repeated cloned-voice requests without any registry coupling.

## Inference pipeline

### Prompt assembly

The talker prefix mirrors the upstream `generate()`. Two pad-aligned
streams (text and codec) are summed at single-vector granularity on the
CPU using the mmapped weight blocks, no backend allocation :

```
role          text(input_id[0:3])                          3 vecs
prefill_lhs   tts_pad x4 + tts_bos
              + codec_emb([think, think_bos, lang_id, think_eos, codec_pad])
trailing_lhs  text(input_id[3:-5]) + tts_eos
              + codec_emb([codec_pad x (N_text + 1)])
trailing_rhs  tts_pad + codec_emb([codec_bos])             1 vec
```

custom_voice inserts the speaker codec-embedding row between `think_eos`
and `codec_pad`. voice_design and custom_voice may prepend an instruct
segment built from `text_proj(text_embd(<|im_start|>user\n{instruct}<|im_end|>\n))`
before the role. Base voice cloning fills the speaker slot with the
x-vector from the speaker encoder (mode A) or, when a reference
transcript is supplied, builds an in-context prefix from the reference
text and the reference codes (mode B, ICL).

### Modes and the validation rules

The synthesis mode is read from the talker `model_type` at load, not
from a CLI flag. `qt_synthesize` validates the params against that
`model_type` before any compute and emits a verbatim `qt_last_error()`.
The checks split across two return codes.

Five conditions mean the flag set does not match the loaded checkpoint
family, returning `QT_STATUS_MODE_INVALID` :

```
--speaker  given but model_type != custom_voice
--instruct given but model_type == base
model_type == custom_voice but no --speaker
model_type == voice_design but --instruct empty or missing
--ref-wav  given but model_type != base
```

Two conditions mean the flag combination is self-contradictory whatever
the model_type, returning `QT_STATUS_INVALID_PARAMS` :

```
--speaker and --ref-wav both given (mutually exclusive)
--ref-text given without --ref-wav
```

### Frame loop

```
prefill the Talker on the prompt prefix                      writes T_ctx into talker_kv
for frame in 0..max_new_tokens-1 :
    poll cancel
    c0 = sample(codec_head(last logits))                     codebook 0, top-k/top-p
    codes[1..15] = code_predictor_step(hidden_bridge, c0)    reads the device bridge
    if c0 == codec_eos : break
    streaming : decode the frame through the stateful codec, emit 1920 samples
    talker_forward_decode(codes, overlay)                    gathers next_emb in graph
buffered : chunked codec decode of the gathered frames
```

The talker's last-position hidden never round-trips through the host on
the hot path : the talker graph copies it into a persistent device
tensor (the hidden bridge) that the code predictor prefill reads as a
graph leaf. The only per-frame host traffic is the code ids and overlay
row up, and the logits down for sampling.

Sampling matches the HuggingFace `generate()` chain in F32 :
`repetition_penalty -> temperature -> top_k -> top_p -> softmax ->
multinomial`, the uniform draw coming from `philox_uniform_fill` so a
fixed seed replays byte for byte across runs.

## Public API

### Top-level public ABI : src/qwen.h

Single-header, plain C99, `extern "C"`. The opaque `qt_context` handle
aggregates the GGML backend pair, the Talker LM, the Code Predictor, the
optional speaker encoder, the 12 Hz codec, the BPE tokenizer and the
language / speaker tables. One init, one free, one synthesize call
covers the full TTS path, consumable identically from C, C++, Python
ctypes, Rust bindgen or Go cgo.

```c
#include "qwen.h"

struct qt_init_params iparams;
qt_init_default_params(&iparams);
iparams.talker_path = "models/qwen-talker-1.7b-base-Q8_0.gguf";
iparams.codec_path  = "models/qwen-tokenizer-12hz-Q8_0.gguf";

struct qt_context * q = qt_init(&iparams);

struct qt_tts_params params;
qt_tts_default_params(&params);
params.text = "Hello world.";
params.lang = "English";

struct qt_audio audio = { 0 };
enum qt_status rc = qt_synthesize(q, &params, &audio);
if (rc == QT_STATUS_OK) {
    /* audio.samples : malloc'd mono float PCM, audio.n_samples,
       audio.sample_rate = 24000, audio.channels = 1 */
}
qt_audio_free(&audio);
qt_free(q);
```

Status codes :

```
QT_STATUS_OK               0
QT_STATUS_INVALID_PARAMS  -1
QT_STATUS_MODE_INVALID    -2   (the seven mode rules)
QT_STATUS_GENERATE_FAILED -3
QT_STATUS_OOM             -4
QT_STATUS_CANCELLED       -5
```

`qt_tts_params` exposes `cancel` (polled at the top of every Talker
decode step, ~83 ms granularity) and `on_chunk`. With `on_chunk` set,
synthesis runs in streaming mode : every generated frame emits its
1920 samples immediately through the stateful codec and `out` stays
empty on success. `codec_chunk_sec` / `codec_left_context_sec` drive
the chunk framing of the buffered path only; the streaming path
ignores both.

`QT_ABI_VERSION` guards struct growth : callers set `abi_version` (or
let the default-params helpers do it) and the lib rejects a struct laid
out for a newer header. `qt_version()` returns the git short hash and
commit date.

### Low-level API : src/pipeline-tts.h, src/pipeline-codec.h

Direct access to `pipeline_tts_load` / `pipeline_tts_synthesize`,
`pipeline_codec_encode` / `pipeline_codec_decode`,
`pipeline_codec_stream_reset` / `pipeline_codec_decode_stream`,
`codec_chunked_decode`, and the talker / predictor forwards. Used by the
`qwen-codec` round-trip and the Python cossim harness through dump
files. C++ types in the signatures, not part of the public ABI.

### ABI guarantee

`tests/abi-c.c` is built on every build as the `test-abi-c` target with
`-std=c99 -Wall -Werror -pedantic`. It includes the public header, calls
every entry through its early-return path, and never loads a model. Any
regression that breaks plain C consumability fails the main build.

The static `libqwen-core.a` is the default artefact and the one the CLI
tools link. For binding consumers, configure with `-DQWEN_SHARED=ON` to
add `libqwen.so` (or `.dll` / `.dylib`) exporting only the `qt_*`
symbols ; every internal `pipeline_*` and `backend_*` stays hidden
behind `-fvisibility=hidden`.

## CLI tools

### qwen-tts

Verbatim `--help` (the binary also prints a `qwentts.cpp <hash> (<date>)`
banner line first) :

```
Usage: ./build/qwen-tts --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt

Required:
  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)
  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)
  -o <path>               Output WAV (24 kHz mono). '-' streams to stdout (pipe friendly).

Input:
  stdin                   Target text to synthesise. Read fully then synthesised in one
                          shot, or line by line with --stream-by-line.

Optional:
  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)
  --lang <name>           Language label (default: auto)
  --instruct <str>        Style instruction. Required for VoiceDesign, optional for
                          CustomVoice, rejected for Base
  --speaker <name>        Speaker name (CustomVoice only)
  --ref-wav <path>        Reference WAV for voice cloning (Base only)
  --ref-spk <path>        Pre-extracted speaker embedding from qwen-codec --talker
                          (replaces --ref-wav, Base only)
  --ref-rvq <path>        Pre-encoded reference codes from qwen-codec (requires
                          --ref-spk and --ref-text, enables ICL clone mode)
  --ref-text <path>       Transcript file for the reference (enables ICL clone mode)
  --max-new <n>           Max new audio frames (default: 2048)
  --codec-chunk-dur <f>   Codec decode chunk duration in seconds (default: 24.0)
  --codec-left-dur <f>    Codec decode left context duration in seconds (default: 2.0)
  --stream-by-line        Flush synthesis at each newline, one WAV header per line (-o '-')

Sampling:
  --seed <int>            Sampling seed (default: -1 for random)
  --greedy                Disable stochastic sampling on both stacks
  --temp <f>              Talker temperature (default: 0.9)
  --top-k <n>             Talker top-k (default: 50, 0 disables)
  --top-p <f>             Talker top-p (default: 1.0)
  --rep-pen <f>           Talker repetition penalty (default: 1.05)
  --sub-temp <f>          Sub-talker temperature (default: 0.9)
  --sub-top-k <n>         Sub-talker top-k (default: 50)
  --sub-top-p <f>         Sub-talker top-p (default: 1.0)

Debug:
  --no-fa                 Disable flash attention
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors (f32) to <dir>
```

### qwen-codec

Verbatim `--help` :

```
Usage: ./build/qwen-codec --model <gguf> [-i <input>] [--talker <gguf>] [--format <fmt>]

Required:
  --model <gguf>          Codec GGUF (qwen-tokenizer-12hz-*.gguf)

Optional:
  -i <path>               Input. WAV -> encode, .rvq -> decode
  --talker <gguf>         Talker GGUF (Base only). Encode also extracts the speaker
                          embedding and writes it next to the .rvq as a .spk file
  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)

Output is auto-named next to input : clip.wav -> clip.rvq, clip.rvq -> clip.wav.
Encode truncates to the hop boundary, conforming to the qwen-tts --ref-wav path:
the .rvq feeds qwen-tts --ref-rvq, the .spk feeds qwen-tts --ref-spk.
When -i is omitted, runs a load self-test of the codec GGUF.
```

The `.rvq` container packs the 16 codes per frame at 11 bits LSB-first.

### tts-server

OpenAI-compatible HTTP server over the public ABI, one GPU-resident
context, synthesis serialized FIFO across connections. The shared HTTP
core lives in `src/tts-server.h` (also consumed by the sibling *.cpp
ports); `tools/tts-server.cpp` wires the `qt_*` ABI into it. Verbatim
`--help` :

```
Usage: ./build/tts-server --model <gguf> --codec <gguf> [options]

Required:
  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)
  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)

Optional:
  --host <ip>             Listen address (default: 127.0.0.1)
  --port <n>              Listen port (default: 8080)
  --lang <n>              Language label (default: auto)
  --no-fa                 Disable flash attention
  --clamp-fp16            Clamp hidden states to FP16 range
```

Endpoints :

```
POST   /v1/audio/speech   OAI text-to-speech; response_format "pcm"
                          streams s16le 24 kHz mono chunked as it is
                          generated, "wav" returns a one-shot RIFF file
GET    /v1/models         single loaded model
GET    /v1/voices         model speakers plus registered cloned voices
POST   /v1/voices         register a cloned voice: {name, ref_text,
                          wav_b64} extracts server side through
                          qt_extract_voice_ref, {name, ref_text,
                          spk_b64, rvq_b64} takes the pre-extracted
                          latents verbatim
DELETE /v1/voices/{name}  drop a registered voice
GET    /health            liveness probe
```

A registered voice wins over a model speaker of the same name and
injects the reference latents into `qt_tts_params` : `ref_text` present
selects ICL clone mode, absent selects the x-vector-only mode. The
registry lives in process RAM and every access shares the synthesis
mutex, so registration (which runs the extraction on the GPU) and
lookups never race a running synthesis.

## Module map

```
src/
  backend.h                 GGML backend init, scheduler factory, env override
  weight-ctx.h              Generic weight context for GGUF loaders
  gguf-weights.h            mmap GGUF, gf_load_tensor, gf_load_conv_f16, gf_get_*
  kv-cache.h                Persistent per-layer KV cache, fixed max len, rewind reset
  audio-io.h / wav.h        WAV read, mono write (S16 / S24 / F32)
  audio-resample.h          Kaiser polyphase resampler
  audio-mel.h               Mel spectrogram for the speaker encoder
  philox.h                  Philox4x32-10 counter-based PRNG
  sampling.h                Talker / CodePredictor sampling chain
  debug.h                   Tensor dumper for cossim tests
  bpe.h                     Qwen2 byte-level BPE, GGUF loader

  talker-weights.h          Talker GGUF weights
  talker-forward.h          Talker prefill + decode forwards, KV cached
  code-predictor-weights.h  Code Predictor MTP weights
  code-predictor-forward.h  Per-frame 16-code expansion, cache reset per frame
  speaker-encoder-weights.h ECAPA-TDNN weights (base only)
  speaker-encoder-forward.h ECAPA-TDNN forward, fused mel
  speaker-encoder-extract.h x-vector extraction entry

  seanet-encoder.h          SEANet conv stack (4 stages, 960x)
  encoder-transformer.h     8-layer Mimi-style encoder transformer
  encoder-downsample.h      25 Hz -> 12.5 Hz downsample conv
  quantizer-encode.h        RVQ encode (16 codebooks, split semantic/acoustic)
  quantizer-decode.h        RVQ decode, per-split output_proj
  tokenizer-transformer.h   8-layer local-causal decoder transformer (sw 72), KV ring stream variant
  convnext-block.h          ConvNeXt upsample stage (2 blocks, 4x), depthwise stream states
  causal-trans-conv.h       Causal Conv1d / ConvTranspose1d, offline and stateful stream variants
  dac-decoder-v2.h          DAC decoder (Descript Audio Codec; strides 8/5/4/3, SnakeBeta), stream states
  codec-chunked-decode.h    Buffered chunked decode plus the stateful frame-by-frame stream decoder
  rvq-file.h                Packed .rvq code stream IO, file and buffer readers

  prompt-builder.h          Talker prefix assembly, modes, ICL geometry
  pipeline-codec.{h,cpp}    Audio tokenizer end-to-end, persistent stream state, static frame graph
  pipeline-tts.{h,cpp}      Full TTS orchestration, prefill, frame loop, decode
  tts-server.h              Shared OAI HTTP core : routes, parsing, voice registry hooks
  qwen.{h,cpp}              Public ABI : opaque qt_context, plain C99 header

tools/
  qwen-tts.cpp              CLI : text to WAV
  qwen-codec.cpp            CLI : codes <-> WAV
  tts-server.cpp            OAI HTTP server : qt_* adapter, cloned voice registry
  quantize.cpp              GGUF requantizer with the codec-aware policy
  version.cmake             Embeds the git short hash into the binary

tests/
  debug-{tts,base,clone,customvoice}-cossim.py   Per-stage cossim vs PyTorch
  cossim_common.py          Shared comparison helpers
  abi-c.c                   Plain C99 smoke test for the public ABI
```

## GGML conventions

### Tensor shape and layout

PyTorch `(out, in)` for a Linear stores as ggml `ne[0]=in, ne[1]=out`.
`ggml_mul_mat(A, B)` with `A.ne[0]=K`, `A.ne[1]=M`, `B.ne[1]=N` gives
output `(N, M)`, equal to `A @ B^T` in PyTorch terms.

The codec runs T-first : `ne[0]=T`, `ne[1]=C`. `ggml_conv_1d` and
`ggml_conv_1d_dw` are T-first natively, the fused SNAKE op requires it,
and the only mul_mat in the convtranspose primitive transposes
internally.

For `ConvTranspose1d` weight `(IC, OC, K)`, the convert-time permutation
rearranges to ggml `(IC, K*OC)` with k varying faster than oc, so
`ggml_col2im_1d` receives the correct column matrix. The fork folds the
padding crop into the op via `p0`, removing a follow-up `ggml_view`.

### Custom GGML ops

Provided by the `ServeurpersoCom/ggml` fork :

`ggml_snake(ctx, x, a, inv_b)` : `y = x + sin^2(a*x) * inv_b`. The
SnakeBeta `exp()` on alpha and beta is folded CPU-side at load.

`ggml_col2im_1d(ctx, a, s0, oc, p0)` : scatter-add `[K*OC, T_in]`
columns into `[T_out, OC]` with `T_out = (T_in-1)*s0 + K - 2*p0`. Used
by every CausalTransConv in the ConvNeXt upsample and the DAC decoder.

### Conv weight dtype

Conv kernels are cast to F16 at load by `gf_load_conv_f16` (ARM im2col
is strict on the kernel dtype), independent of the GGUF storage dtype.

### Backend lifecycle

`backend_init("MOD")` then `backend_sched_new(bp, max_nodes)`. Backend
handles are shared across modules, refcounted. `use_fa` collapses to
false on CPU-only backends. `clamp_fp16` inserts `ggml_clamp(-65504,
65504)` on V before attention and on the residual stream between blocks
to guard FP16 matmul accumulation on sub-Ampere CUDA targets.

## Validation

The harness is `tests/debug-{tts,base,clone,customvoice}-cossim.py`. It
runs the same input through the PyTorch reference (TF32 disabled, eager
attention) and through the C++ binary, dumps each stage with `--dump`,
and reports cosine similarity per stage. Latest run, voice_design 1.7B,
Q8_0 on CUDA0, greedy :

```
Forward fidelity (single pass, the correctness signal)
  PromptIDs exact   100.00%
  Embed       cos   0.999988
  L0 .. L27   cos   >= 0.999893
  Final       cos   0.999401
  Logits      cos   0.999824
  NextEmbStep0 cos  0.999990

Free-running generation
  CodesFull exact   4.96%
  Audio       cos   0.095
  WAV stft    cos   0.360
```

Read this the way an autoregressive sampler demands. The forward graph
matches the reference closely at every stage : embeddings, all 28 hidden
taps, the final norm and the codebook 0 logits all sit at cosine
0.9994 and above. The large per-channel max-abs values at L27 are the
usual pre-final-norm outlier channels ; cosine stays high because the
direction is preserved.

The low CodesFull and Audio numbers are not a defect. A single argmax
tie at the FP epsilon between the GGML and cuBLAS kernels flips one
token, and because each frame conditions the next, the two runs walk
different sampling trajectories from that point on. The MaskGIT path in
omnivoice.cpp re-converges over its 32 refinement steps, so it can be
checked bit for bit ; an AR sampler has no such contraction, and the two
runs simply produce different but equally valid utterances of the same
text (here 64 frames against the reference 63). End-to-end token or
waveform equality is therefore the wrong metric. The meaningful check is
the per-stage forward fidelity above, plus listening ; the `--dump` taps
exist precisely to bisect that forward path stage by stage.

## Glossary

  MTP        Multi-Token Prediction head. Here the Code Predictor that
              expands one semantic token into 15 acoustic codes per
              frame in a short cached inner loop.

  RVQ        Residual Vector Quantisation. Stack of codebooks, each
              quantising the residual of the previous reconstruction.
              16 here : 1 semantic + 15 acoustic.

  SEANet     Convolutional audio encoder/decoder backbone (Mimi style),
              strided convs for down/up sampling.

  DAC        Descript Audio Codec. Convolutional decoder over the
              quantised latent, here with SnakeBeta activations.

  SnakeBeta  Periodic activation `x + (1/beta) * sin^2(alpha*x)` with
              `exp()` applied to alpha and beta, folded at load.

  ConvNeXt   Depthwise conv plus pointwise MLP block with a LayerScale
              residual, used here for the 4x temporal upsample.

  ECAPA-TDNN Time-delay speaker embedding network with SE-Res2Net blocks
              and attentive statistics pooling. Base checkpoints only.

  ASP        Attentive Statistics Pooling. Mean and std over time
              weighted by a learned attention, the ECAPA pooling head.

  GQA        Grouped Query Attention. Fewer KV heads than query heads
              (16/8 on the Talker, 2:1).

  mrope      Multimodal RoPE with per-section position axes. Collapsed
              to 1D NEOX for the single TTS timeline.

  ICL        In-Context Learning. Voice clone mode B : the reference
              text and codes prefix the prompt so the model continues
              the speaker.

  Philox     Counter-based PRNG used by PyTorch CUDA. Skip-ahead
              friendly, aligns the multinomial draw across runs.
