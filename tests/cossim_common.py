#!/usr/bin/env python3
"""Shared helpers for the qwentts.cpp cossim debug scripts.

Provides Philox uniform stream, dump load and save, install_hooks for the
talker submodel, the standard stage list and the metric helpers used by
debug-base / debug-tts / debug-customvoice / debug-clone cossim scripts.

Importing this module patches sys.path so qwen_tts upstream loads without
the V1 25Hz tokenizer (sox dependency stubbed out), and forces TF32 off on
every torch CUDA matmul path so Python results stay bit comparable across
runs and across machines.
"""

import os
import struct
import sys
import types

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32                             = False
torch.backends.cudnn.allow_tf32                                   = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")

# Stub the V1 25Hz tokenizer so qwen_tts.core imports without sox.
UPSTREAM_ROOT = "../../Qwen3-TTS"
sys.path.insert(0, UPSTREAM_ROOT)

class _StubV1Config:
    pass

class _StubV1Model:
    pass

_qwen_pkg          = types.ModuleType("qwen_tts")
_qwen_pkg.__path__ = [os.path.join(UPSTREAM_ROOT, "qwen_tts")]
sys.modules["qwen_tts"] = _qwen_pkg

_core_pkg          = types.ModuleType("qwen_tts.core")
_core_pkg.__path__ = [os.path.join(UPSTREAM_ROOT, "qwen_tts", "core")]
# Inject the stubbed core module before any submodule import so the real
# qwen_tts/core/__init__.py never runs : it pulls the V1 25Hz tokenizer that
# imports whisper_encoder, which prints a flash-attn warning at module load.
sys.modules["qwen_tts.core"] = _core_pkg
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2    import Qwen3TTSTokenizerV2Model
_core_pkg.Qwen3TTSTokenizerV1Config = _StubV1Config
_core_pkg.Qwen3TTSTokenizerV1Model  = _StubV1Model
_core_pkg.Qwen3TTSTokenizerV2Config = Qwen3TTSTokenizerV2Config
_core_pkg.Qwen3TTSTokenizerV2Model  = Qwen3TTSTokenizerV2Model

from qwen_tts.core.models.modeling_qwen3_tts      import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.processing_qwen3_tts    import Qwen3TTSProcessor
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.utils import logging as hf_logging

# Silence the GenerationConfig validator that warns "flags are not valid
# and may be ignored" for temperature / top_k / top_p when do_sample=False.
# Those flags ride along inside the checkpoint generation_config and the
# greedy path drops them on purpose, the warning is just noise here.
hf_logging.set_verbosity_error()

# Register the Qwen3-TTS classes once per process. Calling twice raises a
# ValueError inside transformers, hence the guard.
_REGISTERED = {"done": False}

def register_qwen3_tts():
    if _REGISTERED["done"]:
        return
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    _REGISTERED["done"] = True

# Path to the C++ qwen-tts binary, relative to tests/.
BIN = "../build/qwen-tts"

# Standard stage list shared by every cossim script. Mode-specific scripts
# may extend this list before iterating (eg. clone adds SpeakerEmb / RefCodes).
STAGES_STANDARD = [
    ("Embed",              "talker-input-embed.bin"),
    ("TrailingText",       "trailing-text-hidden.bin"),
    ("TTSPadEmbed",        "tts-pad-embed.bin"),
    ("L0",                 "talker-hidden-prefill-l0.bin"),
    ("L7",                 "talker-hidden-prefill-l7.bin"),
    ("L14",                "talker-hidden-prefill-l14.bin"),
    ("L21",                "talker-hidden-prefill-l21.bin"),
    ("L27",                "talker-hidden-prefill-l27.bin"),
    ("Final",              "talker-hidden-prefill-final.bin"),
    ("Logits",             "talker-logits-prefill.bin"),
    ("NextEmbStep0",       "next-emb-step0.bin"),
    ("TalkerHiddenStep1",  "talker-hidden-step1.bin"),
]

# Philox4x32-10 mirror of src/philox.h. Returns the same float u that
# philox_uniform_fill(seed, subseq, ctr_lo=0) gives for n=1.
PHILOX_M0     = np.uint32(0xD2511F53)
PHILOX_M1     = np.uint32(0xCD9E8D57)
PHILOX_W0     = np.uint32(0x9E3779B9)
PHILOX_W1     = np.uint32(0xBB67AE85)
TWO_POW32_INV = np.float32(2.3283064365386963e-10)

def _mulhilo32(a, b):
    p  = np.uint64(a) * np.uint64(b)
    lo = np.uint32(p & np.uint64(0xFFFFFFFF))
    hi = np.uint32(p >> np.uint64(32))
    return hi, lo

def _philox_round(ctr, k0, k1):
    hi0, lo0 = _mulhilo32(PHILOX_M0, ctr[0])
    hi1, lo1 = _mulhilo32(PHILOX_M1, ctr[2])
    return (np.uint32(hi1 ^ ctr[1] ^ k0),
            np.uint32(lo1),
            np.uint32(hi0 ^ ctr[3] ^ k1),
            np.uint32(lo0))

def _philox4x32_10(ctr, k0, k1):
    mask = np.uint64(0xFFFFFFFF)
    for _ in range(9):
        ctr = _philox_round(ctr, k0, k1)
        k0  = np.uint32((np.uint64(k0) + np.uint64(PHILOX_W0)) & mask)
        k1  = np.uint32((np.uint64(k1) + np.uint64(PHILOX_W1)) & mask)
    ctr = _philox_round(ctr, k0, k1)
    return ctr

def philox_uniform(seed, subseq, ctr_lo=0):
    slo = np.uint32(np.uint64(seed) & np.uint64(0xFFFFFFFF))
    shi = np.uint32(np.uint64(seed) >> np.uint64(32))
    ctr = (np.uint32(ctr_lo),
           np.uint32(0),
           np.uint32(np.uint64(subseq) & np.uint64(0xFFFFFFFF)),
           np.uint32(np.uint64(subseq) >> np.uint64(32)))
    r = _philox4x32_10(ctr, slo, shi)
    return (np.float32(r[0]) + np.float32(0.5)) * TWO_POW32_INV

# Globals advanced exactly once per multinomial sample, mirroring the C++
# side which advances subseq_counter at every sample_top_k_p call.
_subseq_counter = [0]
_seed           = [42]
_trace_samples  = [False]

def reset_philox(seed):
    _subseq_counter[0] = 0
    _seed[0]           = int(seed)

def set_trace(flag):
    _trace_samples[0] = bool(flag)

def patched_multinomial(input, num_samples, replacement=False, generator=None, out=None):
    """Drop in replacement for torch.multinomial(num_samples=1) that pulls
    the uniform draw from our Philox stream and walks the F32 cumulative
    sum the same way src/sampling.h does."""
    assert num_samples == 1, "patched_multinomial only handles num_samples=1"
    probs = input
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    bsz, vocab = probs.shape
    out_ids = torch.zeros((bsz, 1), dtype=torch.long, device=probs.device)
    for b in range(bsz):
        u   = philox_uniform(_seed[0], _subseq_counter[0], 0)
        seq = _subseq_counter[0]
        _subseq_counter[0] += 1
        row = probs[b].to(torch.float32).cpu().numpy()
        s   = float(row.sum())
        # The C++ sampler draws u in [0, 1) and compares against acc/sum
        # implicitly via acc >= u*sum. We replicate that exact arithmetic.
        target = float(u) * s
        acc    = 0.0
        idx    = vocab - 1
        for i in range(vocab):
            acc += float(row[i])
            if acc >= target:
                idx = i
                break
        out_ids[b, 0] = idx
        if _trace_samples[0] and seq < 32:
            print(f"[Sample-PY] subseq={seq} u={float(u):.10f} idx={idx} top_prob={float(row.max()):.6f}")
    if input.dim() == 1:
        return out_ids.squeeze(0)
    return out_ids

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dump(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.float32).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(data.tobytes())

def save_dump_i32(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.int64).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.int64))
    shape = data.shape
    fdata = data.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(fdata.tobytes())

def load_dump(path):
    raw   = np.fromfile(path, dtype=np.uint8)
    ndim  = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * ndim], dtype=np.int32))
    body  = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.float32)
    return body.reshape(shape), shape

def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def stft_cos(a, b, win=2048, hop=512):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    window = np.hanning(win)
    frames = (n - win) // hop + 1
    if frames <= 0:
        return 0.0
    sa = np.zeros((frames, win // 2 + 1))
    sb = np.zeros((frames, win // 2 + 1))
    for i in range(frames):
        s = i * hop
        sa[i] = np.abs(np.fft.rfft(a[s:s + win] * window))
        sb[i] = np.abs(np.fft.rfft(b[s:s + win] * window))
    return cos(sa.ravel(), sb.ravel())

def install_hooks(model, dump_dir, bisect_layers=(0, 7, 14, 21, 27)):
    """Capture every intermediate tensor we can pin against the C++ side.
    Layout : bisection layers, final norm, prefill logits, the input embed
    fed to the talker and the trailing-text overlay tensors that drive the
    next-token embedding sums during generation. Counters track how many
    times the talker submodel has run so step 1 (first single-token forward
    after prefill) gets its own dump."""
    seen_prefill = {"done": False}
    seen_codes   = {"done": False}
    # talker_step counts how many times talker_model.forward has been called
    # after the prefill. Prefill itself is recorded as 0, the first single
    # token forward is 1, and so on. Single token forwards are detected by
    # inputs_embeds.shape[1] == 1 in the pre hook.
    talker_step  = {"n": 0}

    talker_model = model.talker.model
    talker_lm    = model.talker

    seen_layers = {idx: False for idx in bisect_layers}
    def make_layer_hook(layer_idx):
        def hook(module, inputs, output):
            if seen_layers[layer_idx]:
                return
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                save_dump(os.path.join(dump_dir, f"talker-hidden-prefill-l{layer_idx}.bin"), h[0])
                seen_layers[layer_idx] = True
        return hook
    for layer_idx in bisect_layers:
        talker_model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))

    seen_norm = {"done": False}
    def norm_hook(module, inputs, output):
        if seen_norm["done"]:
            return
        if output.dim() == 3 and output.shape[1] > 1:
            save_dump(os.path.join(dump_dir, "talker-hidden-prefill-final.bin"), output[0])
            seen_norm["done"] = True
    talker_model.norm.register_forward_hook(norm_hook)

    # Pre-hook on the talker submodel : sees inputs_embeds whether the outer
    # talker forward was invoked with input_ids (single token step) or
    # inputs_embeds (prefill). The submodel always receives inputs_embeds
    # because the wrapper rebuilds it before calling self.model.
    def talker_model_pre_hook(module, args, kwargs):
        ie = kwargs.get("inputs_embeds", None)
        if ie is None:
            return
        if ie.dim() != 3:
            return
        if ie.shape[1] > 1:
            return
        if talker_step["n"] == 0:
            save_dump(os.path.join(dump_dir, "next-emb-step0.bin"), ie[0, 0])
        talker_step["n"] += 1
    talker_model.register_forward_pre_hook(talker_model_pre_hook, with_kwargs=True)

    # Post-hook on the talker submodel : captures last_hidden_state at step
    # 1 (first single token forward). That tensor is what feeds the code
    # predictor at step 1, so any drift between Python and C++ tells us the
    # next-emb-step0 changed the talker forward result.
    talker_post_step = {"n": 0}
    def talker_model_post_hook(module, inputs, output):
        last = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        if last.dim() != 3 or last.shape[1] != 1:
            return
        if talker_post_step["n"] == 0:
            save_dump(os.path.join(dump_dir, "talker-hidden-step1.bin"), last[0, -1])
        talker_post_step["n"] += 1
    talker_model.register_forward_hook(talker_model_post_hook)

    # Talker LM wrapper hook : captures the prefill input embed (the talker
    # codec_embedding sum + text projection that mirrors what
    # prompt_builder_build produces in C++), the prefill logits, and the
    # trailing_text_hidden / tts_pad_embed overlay tensors carried by the
    # output dataclass at every step (we only dump them once).
    seen_overlay = {"done": False}
    orig_talker_forward = talker_lm.forward
    def hooked_talker_forward(*args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if (inputs_embeds is not None and inputs_embeds.dim() == 3
                and inputs_embeds.shape[1] > 1 and not seen_prefill["done"]):
            save_dump(os.path.join(dump_dir, "talker-input-embed.bin"), inputs_embeds[0])
            seen_prefill["done"] = True
        out = orig_talker_forward(*args, **kwargs)
        if (out is not None and getattr(out, "logits", None) is not None
                and not seen_codes["done"]):
            logits = out.logits
            if logits.dim() == 3 and logits.shape[1] > 1:
                save_dump(os.path.join(dump_dir, "talker-logits-prefill.bin"), logits[0, -1])
                seen_codes["done"] = True
        if (out is not None and not seen_overlay["done"]
                and getattr(out, "trailing_text_hidden", None) is not None
                and getattr(out, "tts_pad_embed", None) is not None):
            tth = out.trailing_text_hidden
            tpe = out.tts_pad_embed
            if tth.dim() == 3 and tpe.dim() == 3:
                save_dump(os.path.join(dump_dir, "trailing-text-hidden.bin"), tth[0])
                save_dump(os.path.join(dump_dir, "tts-pad-embed.bin"), tpe[0, 0])
                seen_overlay["done"] = True
        return out
    talker_lm.forward = hooked_talker_forward

def pair(name, dump_cpp, dump_pt):
    a, _ = load_dump(os.path.join(dump_cpp, name))
    b, _ = load_dump(os.path.join(dump_pt,  name))
    return a, b

def metric(a, b):
    n     = min(a.size, b.size)
    af    = a.astype(np.float64).ravel()[:n]
    bf    = b.astype(np.float64).ravel()[:n]
    d     = np.abs(af - bf)
    nrm_a = float(np.linalg.norm(af))
    nrm_b = float(np.linalg.norm(bf))
    c     = float(np.dot(af, bf) / (nrm_a * nrm_b)) if nrm_a > 1e-10 and nrm_b > 1e-10 else 0.0
    return c, float(d.max()), float(d.mean())

def compare_stages(stages, dump_cpp, dump_pt):
    """Iterate the stages list and print one line per pair. Skips silently
    when a dump file is missing (eg. a mode that does not produce a given
    intermediate)."""
    for label, name in stages:
        try:
            a, b = pair(name, dump_cpp, dump_pt)
        except FileNotFoundError:
            print(f"[Cossim] {label} skipped (missing dump)")
            continue
        c, mx, mn = metric(a, b)
        print(f"[Cossim] {label} cos: {c:.6f} max: {mx:.4e} mean: {mn:.4e}")

def compare_exact_i32(name, dump_cpp, dump_pt, label):
    """Compare two int dumps stored as f32 (the encoding path used by both
    save_dump_i32 in Python and debug_dump_i32_as_f32 in C++). Prints an
    exact match percentage. Returns the percentage as a float."""
    a, b = pair(name, dump_cpp, dump_pt)
    ai   = a.astype(np.int64).ravel()
    bi   = b.astype(np.int64).ravel()
    n    = min(ai.size, bi.size)
    pct  = 100.0 * float(np.mean(ai[:n] == bi[:n]))
    print(f"[Cossim] {label} exact: {pct:.2f}% ({n} values)")
    return pct

# Greedy generation kwargs shared by every cossim script. do_sample=False
# alone selects argmax, top_k / top_p / temperature are intentionally
# omitted because GenerationConfig flags them as "not valid" warnings when
# do_sample=False. The subtalker_* keys are custom kwargs forwarded to the
# talker forward, the talker validator is bypassed by the script main.
GEN_KWARGS_GREEDY = dict(
    do_sample             = False,
    subtalker_dosample    = False,
    repetition_penalty    = 1.0,
)
