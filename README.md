# qwentts.cpp

Local AI text-to-speech with named speakers, voice cloning and voice
design, powered by GGML. C++17 port of Qwen3-TTS 12 Hz (Qwen team,
Alibaba). 11 languages with Mandarin dialects, 24 kHz mono output,
runs on CPU, CUDA, Metal, Vulkan.

## Features

- Named speakers from the CustomVoice checkpoints, with per-speaker
  Mandarin dialect overrides (eric -> sichuan, dylan -> beijing)
- Zero shot voice cloning from a reference clip, x-vector only or
  in-context with a matching transcript
- Voice design from a free text attribute instruction (gender, age,
  pitch, style)
- Streaming synthesis : autoregressive frame loop with chunked codec
  decode over a rolling left context, low latency chunk callback API
- Two stage generation : the Talker LM emits the semantic codebook, a
  code predictor MTP head emits the 15 acoustic codes per frame, both
  KV cached
- Seedable Philox PRNG and an HF aligned sampling chain
  (repetition penalty -> temperature -> top-k -> top-p -> multinomial)
- Q8_0 and Q4_K_M quantisation of the Qwen3 talker backbone (0.6B and
  1.7B), the RVQ codec paths kept at F32
- Two CLI tools : `qwen-tts` (text -> WAV) and `qwen-codec`
  (WAV <-> RVQ codes)

## Use

> Suggest you use this forked version of the compiled build

### [Download the built version](https://github.com/furinasdog/qwentts.cpp/releases)

## Prerequisites & Hardware Compatibility

### 1. Runtime Environment & Package Selection
All software packages require the corresponding version of the Microsoft Visual C++ (VC) Redistributable and CUDA Toolkit installed on your system. The application will fail to launch without the correct environment.

Please choose the correct package based on your system configuration:

- **`qwen-tts-windows-vc18-cuda13.2.0-vulkan.zip`**
  - Requires: **VC18** runtime.
  - Requires: **CUDA Toolkit 13.2.0** for NVIDIA GPU acceleration.

- **`qwen-tts-windows-vc17-cuda12.9.0-vulkan.zip`**
  - Requires: **VC17** runtime.
  - Requires: **CUDA Toolkit 12.9.0** for NVIDIA GPU acceleration.

- **`qwen-tts-windows-vc18-nocuda-vulkan.zip`**
  - Requires: **VC18** runtime.

- **`qwen-tts-windows-vc17-nocuda-vulkan.zip`**
  - Requires: **VC17** runtime.

### 2. GPU Acceleration & Drivers
The application automatically detects your GPU and attempts to utilize the available hardware acceleration. Proper driver support is mandatory.

- **NVIDIA GPUs (CUDA):** Ensure the installed NVIDIA driver supports the CUDA version specified in your chosen package.
- **AMD / Intel / Other GPUs (Vulkan):** For non-NVIDIA hardware, the application utilizes Vulkan acceleration.
  > **Note:** Vulkan support is implemented via third-party tools rather than official SDKs. If your GPU does not support the latest Vulkan specifications, hardware acceleration may be unavailable.
- **CPU Fallback:** If no compatible GPU or appropriate drivers are detected, the application will default to CPU execution.

## Quick Start Guide

### Step 1: Download Models
Before launching the server, the required models must be downloaded to your local environment.
1. Execute `download-models.exe` to download the default models.
2. Run `download-models.exe --help` in your terminal for advanced configuration and download options.

### Step 2: Run the Services

#### HTTP Server (For Integration)
The primary component `tts-server.exe` launches the core HTTP service and loads the default models upon initialization.
```bash
tts-server.exe
```
*Refer to `tts-server.exe --help` for detailed execution parameters, configuration flags, and API documentation.*

#### WebUI (For Testing)
A basic WebUI (`tts-webui.exe`) is provided for convenient testing of the TTS capabilities.
```bash
tts-webui.exe
```
*Refer to `tts-webui.exe --help` for detailed usage instructions.*

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git
cd qwentts.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Model conversion

Pre-converted GGUFs are available on Hugging Face :

  https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF

Drop them in `models/` and skip to the quick start. To convert from
the original checkpoints :

```
./checkpoints.sh      # hf download Qwen/Qwen3-TTS-12Hz-* -> checkpoints/
./convert.py          # F32 GGUFs (one talker per mode/size + tokenizer) -> models/
./quantize.sh         # BF16 / Q8_0 / Q4_K_M ; RVQ codebooks and projections stay F32
```

Two GGUFs load together : a talker
(`qwen-talker-{size}-{mode}-{variant}.gguf`, LM plus code predictor MTP
head plus optional speaker encoder) and a shared tokenizer
(`qwen-tokenizer-12hz-{variant}.gguf`, SEANet + ConvNeXt + DAC v2 +
RVQ). Modes are `base`, `customvoice` and `voicedesign` ; sizes are
0.6B and 1.7B (voicedesign is 1.7B only).

## Quick start

Each block is the command run by the matching script in `examples/`.

Default voice (`base.sh`) :

```
./build/qwen-tts \
    --model models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --lang English -o out.wav < prompt.txt
```

Voice cloning (`clone.sh`, Base, reference WAV plus its transcript) :

```
./build/qwen-tts \
    --model models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --ref-wav ref.wav --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

Pre-encoded reference (`clone.sh`): `qwen-codec --talker` encodes a reference
WAV into two compact latents in one pass, the `.spk` speaker embedding and
the `.rvq` ICL codes, bit-identical to what the `--ref-wav` path computes
internally. Passing them via `--ref-spk` / `--ref-rvq` skips the speaker
encoder and the codec encode on every synthesis:

```
build/qwen-codec --model models/qwen-tokenizer-12hz-Q8_0.gguf \
    --talker models/qwen-talker-1.7b-base-Q8_0.gguf -i ref.wav
build/qwen-tts \
    --model models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --ref-spk ref.spk --ref-rvq ref.rvq --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

Named speaker (`customvoice.sh`, CustomVoice) :

```
./build/qwen-tts \
    --model models/qwen-talker-1.7b-customvoice-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --speaker vivian \
    --lang English -o out.wav < prompt.txt
```

Speakers : serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee,
eric (sichuan dialect), dylan (beijing dialect).

Voice design (`tts.sh`, VoiceDesign, attribute instruction) :

```
./build/qwen-tts \
    --model models/qwen-talker-1.7b-voicedesign-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --instruct "male, young adult, moderate pitch" \
    --lang English -o out.wav < prompt.txt
```

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

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
qt_synthesize(q, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
qt_audio_free(&audio);
qt_free(q);
```

Base voice-clone latents can also be precomputed in-process, replacing
the `qwen-codec --talker ref.wav` shell-out: `qt_extract_voice_ref`
takes the decoded `.wav` contents as mono float32 PCM at 24 kHz and
fills a `struct qt_voice_ref` with the `.spk`-equivalent speaker
embedding plus the `.rvq`-equivalent `[num_codebooks, ref_T]` code
matrix. Pass those buffers back through `qt_tts_params.ref_spk_emb` /
`ref_codes`, and for reference-WAV-plus-transcription ICL mode keep
passing the transcript as `qt_tts_params.ref_text`. Release the buffers
with `qt_voice_ref_free`.

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build (the `test-abi-c` target), so any regression that breaks
plain C consumability fails the build, not just an opt-in target.

For a binding-friendly shared library (libqwen.so / .dll / .dylib),
configure with `cmake -DQWEN_SHARED=ON ...`. The shared target exports
only the `qt_*` symbols ; every internal `pipeline_*` and `backend_*`
stays hidden inside the .so. The static `libqwen-core.a` is the default
build artefact and the one the bundled CLI tools link against.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline, every CLI flag, the public API
reference and the validation results.

## License

MIT. See [LICENSE](LICENSE).

Upstream model : Qwen3-TTS by Alibaba / Qwen team, Apache 2.0.
Audio codec : Qwen3-TTS-Tokenizer-12Hz (Qwen team), Apache 2.0.
