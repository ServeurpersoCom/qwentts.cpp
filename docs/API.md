# HTTP Server API Reference

OpenAI-compatible TTS HTTP server provided by `tts-server`. The server
loads a Talker language model and a Codec model once (GPU resident), and
serves text-to-speech synthesis over a REST API.

## Quick start

### Start the server

```bash
./build/tts-server \
    --model models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --host 127.0.0.1 --port 8080 --lang auto
```

### Command-line arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model <gguf>` | Yes | — | Talker LM GGUF file (`qwen-talker-*.gguf`) |
| `--codec <gguf>` | Yes | — | Codec GGUF file (`qwen-tokenizer-*.gguf`) |
| `--host <ip>` | No | `127.0.0.1` | Listen address |
| `--port <n>` | No | `8080` | Listen port |
| `--lang <name>` | No | `auto` | Language label |
| `--cache-dir <dir>` | No | `qwentts_server_cache` | Reference audio cache directory (for zero-shot caching) |
| `--no-fa` | No | — | Disable flash attention |
| `--clamp-fp16` | No | — | Clamp hidden states to FP16 range |
| `--help`, `-h` | No | — | Print usage and exit |

### Server defaults

| Setting | Value |
|---|---|
| Read timeout | 60 s |
| Write timeout | 120 s |
| Max request body | 32 MB |
| TCP_NODELAY | Enabled |
| SO_REUSEADDR | Enabled |
| CORS | `*` (all origins) |

---

## Model types

The server supports three model types, determined at init time by the loaded
GGUF checkpoint. Each type accepts a different set of synthesis parameters.

| Model type | `voice` | `instructions` | `ref_wav_b64` / `ref_wav_path` | Description |
|---|---|---|---|---|
| **base** | Ignored | Ignored | Supported | Zero-shot voice cloning via reference audio. No preset speakers. |
| **custom_voice** | Required | Optional | Rejected | Fixed speaker voices. `voice` must be a name from `GET /v1/voices`. |
| **voice_design** | Rejected | Required | Rejected | Style-controlled synthesis via `instructions`. No speaker selection. |

> **Note**: The `voice` parameter is **only effective** with `custom_voice` models.
> With a base model, `voice` is silently ignored — use `ref_wav_b64` / `ref_wav_path`
> for zero-shot cloning instead. Use `GET /v1/voices` to check which speakers
> are available (empty array = base or voice_design model).

## Sampling

TTS synthesis uses autoregressive sampling. Even with identical inputs, each
call produces slightly different audio (natural prosody variation). Sampling
behavior can be tuned via the ABI parameters:

| Parameter | Default | Description |
|---|---|---|
| `temperature` | `0.9` | Sampling temperature. Lower = more deterministic. |
| `top_k` | `50` | Top-k sampling. |
| `top_p` | `1.0` | Nucleus sampling. |
| `seed` | `-1` | Random seed. `-1` uses hardware randomness (`std::random_device`). Set a fixed value for reproducible output. |

These parameters are not exposed in the HTTP API (server uses ABI defaults).
To customize them, use the C ABI or Python bindings directly.

---

## Endpoints

### POST /v1/audio/speech

Synthesize speech from text. Compatible with the OpenAI audio speech
API.

#### Request

```
POST /v1/audio/speech
Content-Type: application/json
```

**Body (JSON)**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `input` | string | Yes | — | Text to speak. Must be non-empty. |
| `voice` | string | No | `""` | Speaker name. **custom_voice models only**: maps to a loaded speaker from `GET /v1/voices`. Silently ignored for base and voice_design models (returns empty array on `GET /v1/voices`). Also ignored in zero-shot cloning mode. |
| `instructions` | string | No | `""` | Voice style instructions (e.g. tone, emotion). Maps to the instruct field in the ABI. Ignored in zero-shot cloning mode. |
| `response_format` | string | No | `"pcm"` | Output format. `"pcm"` for streaming s16le audio, `"wav"` for a one-shot RIFF WAV file. |
| `speed` | number | No | `1.0` | Speed multiplier. Parsed for compatibility but currently ignored (no time-stretch in the ABI). |
| `ref_wav_b64` | string | No | `""` | Base64-encoded reference WAV for zero-shot voice cloning (base model only). Mutually exclusive with `ref_wav_path`. Requires `ref_text`. |
| `ref_wav_path` | string | No | `""` | Absolute file path to reference WAV for zero-shot voice cloning (base model only). Mutually exclusive with `ref_wav_b64`. Requires `ref_text`. |
| `ref_text` | string | No | `""` | Transcript of the reference audio. Required when `ref_wav_b64` or `ref_wav_path` is provided. |

#### Response

**When `response_format` = `"pcm"` (default)**

Streams audio in real time via HTTP chunked transfer encoding.

| Property | Value |
|---|---|
| Content-Type | `audio/pcm` |
| Transfer-Encoding | `chunked` |
| Cache-Control | `no-cache` |
| X-Accel-Buffering | `no` |
| Format | s16le (signed 16-bit little-endian) |
| Sample rate | 24000 Hz |
| Channels | mono |

The audio chunks are pushed as soon as the codec produces them, enabling
real-time playback before the full utterance is finished.

**When `response_format` = `"wav"`**

Returns the complete audio as a single RIFF WAV file after synthesis
finishes.

| Property | Value |
|---|---|
| Content-Type | `audio/wav` |
| Format | RIFF WAV, s16 PCM |
| Sample rate | 24000 Hz |
| Channels | mono |

#### Zero-shot Voice Cloning Constraints

When `ref_wav_b64` or `ref_wav_path` is provided:

1. **Model requirement**: Only base models support zero-shot cloning. Custom voice and voice design models return HTTP 400.
2. **Mutual exclusions**:
   - `voice` parameter is ignored (speaker selection not supported in zero-shot mode)
   - `instructions` parameter is ignored (voice design not supported in zero-shot mode)
3. **Transcript requirement**: `ref_text` must be provided and non-empty
4. **Reference format**:
   - `ref_wav_b64`: Mono or stereo WAV in any sample rate (will be resampled to 24 kHz)
   - `ref_wav_path`: File accessible from the server's working directory (absolute path required)
5. **Exclusive modes**: Cannot use `voice`, `ref_wav_b64`, and `ref_wav_path` simultaneously
6. **Caching**: Reference audio encodings are cached by SHA256 hash in `--cache-dir` to accelerate repeated requests

**Mode A (x-vector only)**: Provide `ref_wav_b64`/`ref_wav_path` without pre-encoded codes
**Mode B (ICL)**: Provide `ref_wav_b64`/`ref_wav_path` with `ref_text` for in-context learning

#### Errors

Returns a JSON error envelope with the appropriate HTTP status code.

```json
{
  "error": {
    "message": "description of the error",
    "type": "error_type"
  }
}
```

| Status | Condition |
|---|---|
| 400 | `input` missing or empty |
| 400 | `response_format` is not `"pcm"` or `"wav"` |
| 400 | `ref_wav_b64` and `ref_wav_path` both provided |
| 400 | `ref_wav_b64` or `ref_wav_path` without `ref_text` |
| 400 | Invalid Base64 in `ref_wav_b64` |
| 400 | Reference WAV format invalid or corrupted |
| 400 | Model does not support zero-shot cloning (use base model) |
| 400 | Invalid request JSON |
| 400 | Invalid parameters or mode (ABI status -1 or -2) |
| 502 | Synthesis failed (server-side / GPU error) |

#### Examples

**Stream PCM and play in real time:**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"input":"The quick brown fox jumps over the lazy dog."}' \
    | ffplay -f s16le -ar 24000 -ch_layout mono -nodisp -autoexit -i -
```

**Save as WAV file:**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"input":"This one is written to a file.","response_format":"wav"}' \
    --output out.wav
```

**With voice and instructions:**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{
      "input": "Hello, how are you today?",
      "voice": "freeman",
      "instructions": "Speak in a warm and friendly tone.",
      "response_format": "wav"
    }' \
    --output out.wav
```

**Zero-shot voice cloning with Base64 reference (Mode B: ICL):**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{
      "input": "I am very happy to see you.",
      "ref_wav_b64": "'$(base64 -w 0 ./examples/freeman.wav)'",
      "ref_text": "Reference speaker: Freeman, a cheerful character.",
      "response_format": "wav"
    }' \
    --output cloned.wav
```

**Zero-shot voice cloning with absolute file path (Mode B: ICL):**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{
      "input": "Synthesize in the cloned voice.",
      "ref_wav_path": "/absolute/path/to/speaker_reference.wav",
      "ref_text": "This is the reference speaker transcript.",
      "response_format": "wav"
    }' \
    --output cloned.wav
```

**Python with zero-shot cloning:**

```python
import requests
import base64

# Base64 mode
with open("ref.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode('utf-8')

resp = requests.post(
    "http://127.0.0.1:8080/v1/audio/speech",
    json={
        "input": "Synthesize this text in the reference voice.",
        "ref_wav_b64": ref_b64,
        "ref_text": "Reference speaker transcript goes here.",
        "response_format": "wav"
    }
)
with open("cloned.wav", "wb") as f:
    f.write(resp.content)

# File path mode
resp = requests.post(
    "http://127.0.0.1:8080/v1/audio/speech",
    json={
        "input": "Another zero-shot synthesis.",
        "ref_wav_path": "/absolute/path/to/reference.wav",
        "ref_text": "Transcript of the reference audio.",
        "response_format": "wav"
    }
)
with open("cloned2.wav", "wb") as f:
    f.write(resp.content)
```

**Performance note**: Reference audio encodings are automatically cached by SHA256 hash. Repeated requests with the same reference audio will retrieve pre-encoded embeddings from cache (`--cache-dir`), avoiding re-encoding and accelerating synthesis.


**Python (requests):**

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8080/v1/audio/speech",
    json={"input": "Hello world.", "response_format": "wav"},
)
with open("out.wav", "wb") as f:
    f.write(resp.content)
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
response = client.audio.speech.create(
    model="qwen-tts",
    voice="freeman",
    input="Hello from the OpenAI SDK!",
    response_format="wav",
)
response.stream_to_file("out.wav")
```

---

### POST /v1/audio/encode

Pre-encode a reference audio into speaker embedding and RVQ codes. The
encoding is cached by content hash — repeated calls with the same audio
return the cached result instantly without re-encoding.

Use this endpoint to pre-encode a reference audio once, then subsequent
`/v1/audio/speech` requests with the same audio will automatically hit
the cache and skip encoding.

#### Request

```
POST /v1/audio/encode
Content-Type: application/json
```

**Body (JSON)**

| Field | Type | Required | Description |
|---|---|---|---|
| `ref_wav_b64` | string | Yes* | Base64-encoded reference WAV. Mutually exclusive with `ref_wav_path`. |
| `ref_wav_path` | string | Yes* | Absolute file path to reference WAV. Mutually exclusive with `ref_wav_b64`. |
| `ref_text` | string | Yes | Transcript of the reference audio. Saved alongside the cached encoding. |

\* Exactly one of `ref_wav_b64` or `ref_wav_path` must be provided.

#### Response

```json
{
  "hash": "a1b2c3d4e5f6...",
  "ref_text": "Reference speaker transcript.",
  "spk": "<base64 speaker embedding>",
  "rvq": "<base64 RVQ codes>",
  "cached": false
}
```

| Field | Type | Description |
|---|---|---|
| `hash` | string | Content hash of the reference audio, used as cache key. |
| `ref_text` | string | The reference transcript provided in the request. |
| `spk` | string | Base64-encoded speaker embedding (float32 vector). |
| `rvq` | string | Base64-encoded RVQ codebook. Binary layout: `[K:int32][ref_T:int32][codes:int32[]]`. |
| `cached` | boolean | `true` if served from cache, `false` if freshly encoded. |

#### Cache behavior

- **Directory**: `qwentts_server_cache/` (configurable via `--cache-dir`)
- **Files**: `{hash}.spk` (speaker embedding), `{hash}.rvq` (RVQ codes), `{hash}.txt` (reference text)
- **Hit**: Returns pre-encoded data without GPU computation
- **Miss**: Encodes the audio, saves to cache, returns the result

#### Errors

| Status | Condition |
|---|---|
| 400 | Neither `ref_wav_b64` nor `ref_wav_path` provided |
| 400 | Both `ref_wav_b64` and `ref_wav_path` provided |
| 400 | `ref_text` missing or empty |
| 400 | Invalid Base64 or corrupted WAV |
| 501 | Backend does not support encoding (non-base model) |
| 502 | Encoding failed (GPU error) |

#### Example

**Encode reference audio (bash):**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/audio/encode" \
    -H "Content-Type: application/json" \
    -d '{
      "ref_wav_b64": "'$(base64 -w 0 ./asset/ref.wav)'",
      "ref_text": "Reference speaker transcript."
    }'
```

**Encode reference audio (Python):**

```python
import requests, base64

with open("ref.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://127.0.0.1:8080/v1/audio/encode",
    json={"ref_wav_b64": ref_b64, "ref_text": "Reference transcript."}
)
result = resp.json()
print(f"Hash: {result['hash']}")
print(f"Cached: {result['cached']}")
```

---

### GET /v1/models

List the loaded model.

#### Request

```
GET /v1/models
```

No parameters.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-talker-1.7b-base-Q8_0.gguf",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `object` | string | Always `"list"` |
| `data` | array | Array of model objects |
| `data[].id` | string | Model filename (basename of the `--model` path) |
| `data[].object` | string | Always `"model"` |
| `data[].owned_by` | string | Always `"local"` |

---

### GET /v1/voices

List available speaker voices. Returns an empty array when the model has
no embedded speakers.

#### Request

```
GET /v1/voices
```

No parameters.

#### Response

```json
{
  "voices": [
    {"name": "freeman"},
    {"name": "alice"}
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `voices` | array | Array of voice objects |
| `voices[].name` | string | Speaker name, usable as the `voice` field in `/v1/audio/speech` |

---

### GET /health

Liveness probe. Returns immediately to confirm the server is running.

#### Request

```
GET /health
```

No parameters.

#### Response

```json
{"status": "ok"}
```

---

## Concurrency

All synthesis requests are serialized through a global mutex (FIFO
order). The server handles one GPU synthesis at a time. HTTP connections
are accepted concurrently, but only one synthesis runs on the GPU at any
given moment.

If a streaming client disconnects mid-synthesis, the generation is
aborted immediately to free the GPU for the next request.

## CORS

The server allows all origins (`Access-Control-Allow-Origin: *`). The
`OPTIONS` preflight handler permits `GET`, `POST`, and `OPTIONS` methods
with the `Content-Type` header.
