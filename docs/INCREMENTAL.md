# Incremental text and audio

`qt_synthesize_incremental` accepts UTF-8 text while it emits 24 kHz mono
float32 audio. One call owns one Talker KV cache and one codec stream until
completion. It does not synthesize separate utterances for incoming chunks.

The function uses the original Qwen3-TTS streaming text alignment: the first
text token joins codec BOS in the prefill. Each subsequent acoustic frame
consumes one projected text token. Input EOF supplies text EOS once, followed
by text padding until the model emits codec EOS. The existing `qt_synthesize`
path keeps its full-text prompt and output behavior. Public parameter
structures and their initializers retain the original ABI 6 layout.
Incremental callbacks live in the internal request, not the public struct.

## C API

Initialize `qt_tts_params` with `qt_tts_default_params`, set voice/language and
sampling options, and install `on_chunk`. Then call:

```c
qt_status status = qt_synthesize_incremental(q, &params, next_text, user_data);
```

The call blocks until audio drains or an error/cancellation occurs.
`params.text` is ignored. The compute worker invokes both callbacks
sequentially. Input callback signature:

```c
int next_text(void *user_data, const char **text, size_t *n_bytes);
```

Return `1` with a nonempty chunk, `0` for EOF, or `-1` for cancellation.
The library copies returned bytes immediately. Keep the pointer valid until
the next callback. Return at most 65536 bytes per invocation. The library
holds an unfinished word and its preceding whitespace until another
whitespace boundary or EOF arrives. CJK, Hangul, Kana, and full-width
punctuation commit complete codepoints immediately. These canonical units
produce the same token sequence for every partition of the input bytes.
CJK units disable cross-character BPE merges; quality must be checked
separately from the full-text tokenizer. The unfinished tail has a 65536-byte
limit. One unread chunk may coexist with that tail; complete units drain
before the next pull. UTF-8 codepoints may cross chunks; invalid committed UTF-8, NUL bytes,
and reserved `<|` token delimiters fail the request. Other scripts without whitespace wait until a word boundary or EOF.

When text runs out, the input callback can block waiting for the producer.
KV and codec state remain intact. **The integration must wake the input
callback and return `-1` when the client disconnects or cancels.** The normal
`cancel` callback cannot interrupt a blocked application callback.
Returning false from `on_chunk` also cancels. Neither callback may invoke a
blocking `qt_*` operation on the same context. Callback backpressure bounds
both input and output; the native runtime has no unbounded input queue.

Every generated frame flushes codec audio before the worker asks for more
input. This avoids keeping audio behind a pending input read. Slow text
arrival can introduce audible gaps; the API does not generate speculative
speech during starvation.

## Current limits

- Initialize the context with `max_batch=1`. A blocked producer owns the
  compute worker, so concurrent jobs on that context wait.
- ICL voice cloning (`ref_text`/`ref_codes`) is rejected. Named speakers,
  voice design, and speaker-vector references use the existing validations.
- `max_new_tokens` and the context capacity still bound one utterance.
  Reaching the generation cap returns a failure rather than silent success.
- Original checkpoints need quality evaluation. Producing finite audio and
  retaining state do not establish intelligibility or speaker similarity.

## Validation

A normal build compiles the public header as C99. Run `ctest` for the ABI
and pure input reducer traces, including cancellation, late events,
UTF-8 validation, and the tail limit.

The optional model test runs a complete-text baseline, incremental input,
and cancellation with the same loaded checkpoint:

```sh
build/test-incremental-model TALKER.gguf CODEC.gguf result
```

It uses CustomVoice speaker `ryan`, English, upstream sampling defaults,
seed 42 and a 160-frame cap. It writes local float32 PCM files for listening.
Success requires model EOS, finite audio before input EOF, input completion,
and cancellation returning `QT_STATUS_CANCELLED`. Output markers
`first_audio_ms`, `input_end_ms`, and `prefinish_audio` make the timing
check machine-readable. Keep quality claims separate from this smoke test.

Reference implementation:
https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py

The initial CPU smoke used the original 0.6B CustomVoice BF16 Talker and
Q8 codec from `Serveurperso/Qwen3-TTS-GGUF`, revision
`b7ee2e8c7459c3bea99da23e3d178125a7d1713c`:

| Check | Result |
| --- | --- |
| Full-text baseline | Model EOS after 47 frames; 90,240 finite samples |
| Incremental input | Model EOS after 60 frames; 115,200 finite samples |
| First incremental audio | 333 ms; input still open |
| Input EOF | 2,247 ms; nine audio chunks already emitted |
| Input cancellation | `QT_STATUS_CANCELLED` after three audio chunks |

These timings describe one smoke run, not a performance guarantee.
A local Parakeet TDT-CTC 110M Q8 transcription recovered the expected words
from both baseline and incremental audio: “Hello there This is a streaming
test”. This one short English sample establishes a smoke check only.
Speaker similarity, prosody, long inputs and multilingual speech quality
remain pending evaluation.
