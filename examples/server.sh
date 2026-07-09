#!/bin/bash
# Start the qwentts OpenAI-compatible TTS server.

./build/tts-server \
    --model models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
    --alias qwen3-tts-base \
    --host 127.0.0.1 --port 8080 --lang auto
