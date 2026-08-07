# Docker

Pre-built images: `ghcr.io/serveurpersocom/qwentts.cpp:cpu`,
`:cuda12`, `:cuda13` and `:vulkan` (also tagged per release, e.g.
`:cuda12-v1.2.3`). All four run `tts-server`; `qwen-tts` and
`qwen-codec` are included in the same image at `/app/`.

`:cuda12` (CUDA 12.9.x) is built against the widest arch range,
Pascal (sm_61) through Blackwell Ultra (120a/121a); `:cuda13`
(CUDA 13.3.x) covers Turing and newer only -- upstream dropped
offline compilation for pre-Turing architectures in CUDA 13, so a
Pascal/Maxwell card needs `:cuda12`. See `CMakeLists.txt` for the
full per-toolkit-version arch table.

```
docker run --rm -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -e MODEL_PATH=/models/qwen-talker-1.7b-base-Q8_0.gguf \
    -e CODEC_PATH=/models/qwen-tokenizer-12hz-Q8_0.gguf \
    ghcr.io/serveurpersocom/qwentts.cpp:cpu
```

CUDA image, with GPU access and a directory of reference WAVs to
auto-register as cloned voices on startup:

```
docker run --rm --gpus all -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -v /path/to/voices:/voices:ro \
    -e MODEL_PATH=/models/qwen-talker-1.7b-base-Q8_0.gguf \
    -e CODEC_PATH=/models/qwen-tokenizer-12hz-Q8_0.gguf \
    ghcr.io/serveurpersocom/qwentts.cpp:cuda12
```

Use `:cuda13` instead of `:cuda12` for a newer CUDA toolkit if your
card is Turing (sm_75) or newer -- see the note above.

Vulkan image (AMD/Intel GPUs), passing through the DRI device node:

```
docker run --rm --device /dev/dri -p 8080:8080 \
    -v /path/to/models:/models:ro \
    -e MODEL_PATH=/models/qwen-talker-1.7b-base-Q8_0.gguf \
    -e CODEC_PATH=/models/qwen-tokenizer-12hz-Q8_0.gguf \
    ghcr.io/serveurpersocom/qwentts.cpp:vulkan
```

The `:vulkan` image bundles Mesa's Vulkan drivers (AMD/Intel). On an
NVIDIA GPU, prefer `:cuda12` / `:cuda13`; running `:vulkan` there would
additionally need the host's proprietary NVIDIA Vulkan ICD mounted in,
which the image does not provide.

## Entrypoint environment variables

| Variable          | Default                                    |
|--------------------|---------------------------------------------|
| `MODEL_PATH`       | `/models/qwen-talker-1.7b-base-Q8_0.gguf`    |
| `CODEC_PATH`       | `/models/qwen-tokenizer-12hz-Q8_0.gguf`      |
| `TTS_LANG`         | `auto`                                       |
| `HOST`             | `0.0.0.0`                                    |
| `PORT`             | `8080`                                       |
| `MODEL_ALIAS`      | unset (reports the GGUF file name)           |
| `CODEC_CHUNK_DUR`  | unset (server default: `24.0`)               |
| `CODEC_LEFT_DUR`   | unset (server default: `2.0`)                 |
| `MAX_BATCH`        | unset (server default: `1`)                  |
| `MAX_PREFILL_TOKENS` | unset (server default: `0`, disabled)      |
| `NO_FA`            | unset; set to `1` to disable flash attention  |
| `CLAMP_FP16`       | unset; set to `1` to clamp hidden states      |
| `WARMUP_VOICE`     | unset; set to a registered voice name to enable the startup warmup below |
| `WARMUP_MAX_NEW_TOKENS` | `750` (only used when `WARMUP_VOICE` is set) |
| `WARMUP_TEXT`      | a generic filler sentence (only used when `WARMUP_VOICE` is set) |

Every `*.wav` placed in `/voices` is registered as a cloned voice
under its filename stem (e.g. `/voices/freeman.wav` -> voice
`freeman`) once `/health` responds. A same-stem `.txt` file (e.g.
`/voices/freeman.txt`) supplies that voice's `ref_text` -- the
transcript of the reference clip -- which enables higher-fidelity ICL
clone mode instead of the x_vector_only fallback used when no
transcript is given.

### Worst-case VRAM warmup

`ggml_backend_sched` grows its compute buffer to fit the largest graph
it has ever built and never shrinks it back, so VRAM use can ratchet
up the first time a long prompt or a long reply arrives on live
traffic. Setting `WARMUP_VOICE` runs one real synthesis at container
startup, capped at `WARMUP_MAX_NEW_TOKENS` frames, to force that
worst-case buffer growth to happen up front instead of on a live
request. Combine with `--max-prefill-tokens` (`MAX_PREFILL_TOKENS`
above) for the input side of the same problem. If the warmup
synthesis fails (most likely out of VRAM), the container exits
non-zero rather than come up healthy and fail unpredictably later --
by design, since a deployment that can't afford its own configured
worst case should know that at startup.

## Building locally

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
cd qwentts.cpp
docker build --target cpu    -t qwentts.cpp:cpu    .
docker build --target cuda   -t qwentts.cpp:cuda   .
docker build --target vulkan -t qwentts.cpp:vulkan .
```

`--target` is required to pick a variant; without it, `docker build`
uses the last stage in the `Dockerfile` (`cuda`).

### Older GPUs (pre-Pascal)

`docker build` never has GPU device access (unlike `docker run
--gpus`), so CMake's CUDA-architecture autodetection has nothing to
detect against. This project's own `CMakeLists.txt` already handles
that by defaulting `CMAKE_CUDA_ARCHITECTURES` to a fixed Pascal-and-newer
list (`61-real;75-virtual;80-virtual;86-real;89-real`, plus Blackwell
`120a-real` with CUDA 12.8+ and Blackwell Ultra `121a-real` with CUDA
12.9+) when the variable isn't set, so Pascal cards (sm_61, e.g. the
GTX 10-series) work out of the box with no override -- as long as the
toolkit is 12.x (CUDA 13 dropped Pascal offline compilation entirely,
see `:cuda13` above). GPUs older than Pascal (Maxwell and earlier)
still need the architecture passed explicitly:

```
docker build --target cuda -t qwentts.cpp:cuda \
    --build-arg CMAKE_CUDA_ARCHITECTURES=50 .   # Maxwell
```

Find your GPU's compute capability at
https://developer.nvidia.com/cuda-gpus.

### CUDA driver stub at link time

The CUDA build links against `libcuda.so` (the driver API, used by
ggml's VMM pool allocator) at build time even though no driver is
present. The `Dockerfile` already points the linker at the devel
image's `lib64/stubs/libcuda.so` for this; it's mentioned here only in
case you customize `CUDA_BUILD_IMAGE` to a base that ships that stub
somewhere else.
