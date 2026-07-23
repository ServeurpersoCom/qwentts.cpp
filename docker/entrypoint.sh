#!/bin/bash
# tts-server entrypoint: starts the server, then registers every reference
# voice found in /voices (each *.wav registers under its filename stem)
# once the server is ready to accept requests.
set -e

MODEL=${MODEL_PATH:-/models/qwen-talker-1.7b-base-Q8_0.gguf}
CODEC=${CODEC_PATH:-/models/qwen-tokenizer-12hz-Q8_0.gguf}
LANG=${TTS_LANG:-auto}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
ALIAS=${MODEL_ALIAS:-}

extra_args=()
[ -n "$ALIAS" ] && extra_args+=(--alias "$ALIAS")
[ -n "$CODEC_CHUNK_DUR" ] && extra_args+=(--codec-chunk-dur "$CODEC_CHUNK_DUR")
[ -n "$CODEC_LEFT_DUR" ] && extra_args+=(--codec-left-dur "$CODEC_LEFT_DUR")
[ -n "$MAX_BATCH" ] && extra_args+=(--max-batch "$MAX_BATCH")
[ "$NO_FA" = "1" ] && extra_args+=(--no-fa)
[ "$CLAMP_FP16" = "1" ] && extra_args+=(--clamp-fp16)

/app/tts-server \
    --model "$MODEL" \
    --codec "$CODEC" \
    --lang "$LANG" \
    --host "$HOST" \
    --port "$PORT" \
    "${extra_args[@]}" &
SERVER_PID=$!

until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "tts-server exited before becoming healthy" >&2; wait "$SERVER_PID"; }
    sleep 1
done

for wav in /voices/*.wav; do
    [ -f "$wav" ] || continue
    name=$(basename "$wav" .wav)
    b64=$(base64 -w0 "$wav")
    printf '{"name":"%s","wav_b64":"%s"}' "$name" "$b64" > /tmp/voice_payload.json
    echo "Registering voice '$name' from $wav"
    curl -sf -X POST "http://localhost:${PORT}/v1/audio/voices" \
        -H "Content-Type: application/json" \
        -d @/tmp/voice_payload.json \
        && echo " -> ok" || echo " -> FAILED"
    rm -f /tmp/voice_payload.json
done

wait "$SERVER_PID"
