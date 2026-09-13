#!/bin/bash
# Download pre-quantized Qwen3-TTS GGUF models from HuggingFace.
# Usage: ./models.sh

set -eu

REPO="Serveurperso/Qwen3-TTS-GGUF"
DIR="models"
mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

dl "qwen-tokenizer-12hz-Q8_0.gguf"
dl "qwen-talker-1.7b-customvoice-Q8_0.gguf"
dl "qwen-talker-1.7b-customvoice-Q4_K_M.gguf"
