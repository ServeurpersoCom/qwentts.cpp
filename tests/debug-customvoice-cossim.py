#!/usr/bin/env python3
"""Cossim debug : C++ qwen-tts vs Python Qwen3-TTS on the CustomVoice 1.7B path.

Inputs (relative to CWD = tests/) :
    ../examples/prompt.txt       target text fed to both pipelines
    --speaker flag               speaker preset key, default mirrors customvoice.sh

Default mode is greedy (do_sample=False on both sides). The speaker
preset is passed straight through model.generate as `speakers=[name]`
on the Python side, mirroring qwen_tts.inference.qwen3_tts_model.
generate_custom_voice. The speaker codec embedding row slips between
think_eos and codec_pad in the prefill, growing the prefill by one
codec vector. Cote C++ the same insertion happens inside prompt_builder.

Optional --instruct adds a style instruction in front of the prompt.
The 1.7B CustomVoice accepts it, the 0.6B does not.

Dumps land in cpp/customvoice/ (C++) and python/customvoice/ (Python).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

import cossim_common as cc

MODEL_T     = "../models/qwen-talker-1.7b-customvoice-{q}.gguf"
MODEL_CDC_T = "../models/qwen-tokenizer-12hz-{q}.gguf"
CKPT        = "../checkpoints/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DUMP_CPP    = "cpp/customvoice"
DUMP_PT     = "python/customvoice"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",         default="../examples/prompt.txt")
    ap.add_argument("--speaker",        default="vivian",
                    help="speaker preset key (lowercase), validated by the model")
    ap.add_argument("--instruct",       default="",
                    help="optional style instruction, empty disables the instruct prefix")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--lang",           default="english")
    ap.add_argument("--quant",          default="F32",
                    help="GGUF quantization suffix (F32, BF16, Q8_0, Q4_K_M)")
    ap.add_argument("--out-pt",         default=os.path.join(DUMP_PT,  "customvoice-python.wav"))
    ap.add_argument("--out-cpp",        default=os.path.join(DUMP_CPP, "customvoice-cpp.wav"))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--trace",          action="store_true",
                    help="print per sample u and idx for the first 32 samples")
    args = ap.parse_args()

    cc.ensure_dir(DUMP_PT)
    cc.ensure_dir(DUMP_CPP)
    os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] Speaker: {args.speaker}")
    if args.instruct:
        print(f"[Input] Instruct: {args.instruct}")
    print(f"[Input] Lang: {args.lang} Seed: {args.seed} MaxNewTokens: {args.max_new_tokens}")
    print(f"[Input] Mode: greedy")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cc.set_trace(args.trace)

    cc.register_qwen3_tts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Python] Device: {device}")
    model = cc.AutoModel.from_pretrained(
        CKPT,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    processor = cc.AutoProcessor.from_pretrained(CKPT, fix_mistral_regex=True)

    # Utterance text wrapped as assistant role.
    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inp_utt   = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids = inp_utt["input_ids"].to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    print(f"[Python] InputIds shape: {tuple(input_ids.shape)}")
    cc.save_dump_i32(os.path.join(DUMP_PT, "prompt-ids.bin"), input_ids[0])

    # Optional instruct, None when empty so the talker forward keeps the
    # standard CustomVoice prefill without any instruct prefix.
    instruct_ids_arg = None
    if args.instruct:
        instruct_text = f"<|im_start|>user\n{args.instruct}<|im_end|>\n"
        inp_ins      = processor(text=instruct_text, return_tensors="pt", padding=True)
        instruct_ids = inp_ins["input_ids"].to(device)
        if instruct_ids.dim() == 1:
            instruct_ids = instruct_ids.unsqueeze(0)
        print(f"[Python] InstructIds shape: {tuple(instruct_ids.shape)}")
        cc.save_dump_i32(os.path.join(DUMP_PT, "instruct-ids.bin"), instruct_ids[0])
        instruct_ids_arg = [instruct_ids]

    cc.install_hooks(model, DUMP_PT)

    # Custom subtalker_* kwargs are forwarded to talker.forward but not
    # declared on GenerationMixin, so transformers 4.57 rejects them under
    # the strict validator. Disable it on the talker only.
    model.talker._validate_model_kwargs = lambda *a, **k: None

    talker_codes_list, _ = model.generate(
        input_ids=[input_ids],
        instruct_ids=instruct_ids_arg,
        languages=[args.lang],
        speakers=[args.speaker],
        non_streaming_mode=True,
        max_new_tokens=args.max_new_tokens,
        **cc.GEN_KWARGS_GREEDY,
    )
    codes = talker_codes_list[0]
    print(f"[Python] Codes shape: {tuple(codes.shape)} (T_frames, num_code_groups)")
    cc.save_dump_i32(os.path.join(DUMP_PT, "codes-full.bin"),  codes)
    cc.save_dump_i32(os.path.join(DUMP_PT, "codes-step0.bin"), codes[0])

    wavs, fs = model.speech_tokenizer.decode([{"audio_codes": codes}])
    audio_pt = np.asarray(wavs[0], dtype=np.float32)
    sf.write(args.out_pt, audio_pt, fs, subtype="FLOAT")
    cc.save_dump(os.path.join(DUMP_PT, "output-audio.bin"), audio_pt)
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {fs} Hz {audio_pt.shape[0]/fs:.2f}s -> {args.out_pt}")

    if not os.path.isfile(cc.BIN):
        print(f"[Cossim] FATAL: {cc.BIN} not found, build qwen-tts first")
        sys.exit(1)
    model_lm  = MODEL_T.format(q=args.quant)
    model_cdc = MODEL_CDC_T.format(q=args.quant)
    for p in (model_lm, model_cdc):
        if not os.path.isfile(p):
            print(f"[Cossim] FATAL: GGUF not found: {p}")
            sys.exit(1)
    print(f"[Quant] {args.quant} -> {model_lm} + {model_cdc}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    cmd = [
        cc.BIN,
        "--model",   model_lm,
        "--codec",   model_cdc,
        "--seed",    str(args.seed),
        "--speaker", args.speaker,
        "--lang",    args.lang,
        "--max-new", str(args.max_new_tokens),
        "--dump",    DUMP_CPP,
        "-o",        args.out_cpp,
        "--greedy",
    ]
    if args.instruct:
        cmd[-1:-1] = ["--instruct", args.instruct]
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, input=text, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

    audio_cpp, sr = sf.read(args.out_cpp)
    if audio_cpp.ndim > 1:
        audio_cpp = audio_cpp[:, 0]
    audio_cpp = audio_cpp.astype(np.float32)
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0]/sr:.2f}s -> {args.out_cpp}")

    cc.compare_exact_i32("prompt-ids.bin", DUMP_CPP, DUMP_PT, "PromptIDs")
    cc.compare_stages(cc.STAGES_STANDARD, DUMP_CPP, DUMP_PT)
    cc.compare_exact_i32("codes-full.bin", DUMP_CPP, DUMP_PT, "CodesFull")

    aa, ab = cc.pair("output-audio.bin", DUMP_CPP, DUMP_PT)
    print(f"[Cossim] Audio cos: {cc.cos(aa, ab):.6f}")

    n = min(aa.size, ab.size)
    print(f"[Cossim] WAV stft_cos: {cc.stft_cos(aa.ravel()[:n], ab.ravel()[:n]):.6f} samples: {n}")


if __name__ == "__main__":
    main()
