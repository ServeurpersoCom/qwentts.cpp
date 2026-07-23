#!/bin/bash
# tts-server entrypoint: starts the server, then registers every reference
# voice found in /voices (each *.wav registers under its filename stem,
# optionally paired with a same-stem .txt for ref_text ICL cloning) once
# the server is ready to accept requests.
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
[ -n "$MAX_PREFILL_TOKENS" ] && extra_args+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
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
    # base64 payload can be multiple MB -- too large for an argv string
    # (ARG_MAX), so it's written to a temp file and read in via jq's
    # --rawfile rather than passed as a --arg.
    b64_file=$(mktemp)
    base64 -w0 "$wav" > "$b64_file"
    txt="${wav%.wav}.txt"
    # A same-stem .txt supplies the reference transcript, enabling ICL
    # clone mode (higher fidelity) instead of the x_vector_only fallback
    # used when no ref_text is sent. jq handles JSON-escaping arbitrary
    # transcript text safely (quotes, backslashes, ...).
    if [ -f "$txt" ]; then
        echo "Registering voice '$name' from $wav (with ref_text from $txt)"
        jq -n --arg name "$name" --rawfile wav_b64 "$b64_file" --rawfile ref_text "$txt" \
            '{name: $name, wav_b64: $wav_b64, ref_text: ($ref_text | sub("\n+$"; ""))}' \
            > /tmp/voice_payload.json
    else
        echo "Registering voice '$name' from $wav"
        jq -n --arg name "$name" --rawfile wav_b64 "$b64_file" \
            '{name: $name, wav_b64: $wav_b64}' \
            > /tmp/voice_payload.json
    fi
    curl -sf -X POST "http://localhost:${PORT}/v1/audio/voices" \
        -H "Content-Type: application/json" \
        -d @/tmp/voice_payload.json \
        && echo " -> ok" || echo " -> FAILED"
    rm -f /tmp/voice_payload.json "$b64_file"
done

# Optional fatal worst-case warmup: exercises one real end-to-end synthesis
# capped at WARMUP_MAX_NEW_TOKENS frames, so the decode and codec-decode
# compute buffers (the part that scales with output AUDIO length) get
# reserved up front -- complementing --max-prefill-tokens above, which
# only reserves for input TEXT length. Off by default: only runs when
# WARMUP_VOICE names an already-registered voice. Failure is fatal (kills
# the server and exits non-zero) by design: a deployment that can't
# afford its own configured worst case should refuse to come up healthy,
# not fail unpredictably on a live request later. The exact wording of
# WARMUP_TEXT doesn't matter -- only that it's long enough the model
# doesn't hit EOS on its own before WARMUP_MAX_NEW_TOKENS does, so the
# warmup actually exercises the same truncation path a real over-length
# response would hit in production.
if [ -n "$WARMUP_VOICE" ]; then
    WARMUP_MAX_NEW_TOKENS=${WARMUP_MAX_NEW_TOKENS:-750}
    WARMUP_TEXT=${WARMUP_TEXT:-"This is a warmup sentence used to reserve worst case memory usage before accepting real requests. This is a warmup sentence used to reserve worst case memory usage before accepting real requests. This is a warmup sentence used to reserve worst case memory usage before accepting real requests."}
    echo "Warming up with a ${WARMUP_MAX_NEW_TOKENS}-frame-capped synthesis to reserve worst-case VRAM..."
    if ! curl -sf -X POST "http://localhost:${PORT}/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg voice "$WARMUP_VOICE" --arg input "$WARMUP_TEXT" --argjson max_new_tokens "$WARMUP_MAX_NEW_TOKENS" \
            '{voice: $voice, input: $input, response_format: "pcm", max_new_tokens: $max_new_tokens}')" \
        -o /dev/null; then
        echo " -> FATAL: worst-case warmup synthesis failed (likely out of VRAM) -- refusing to come up healthy"
        kill "$SERVER_PID" 2>/dev/null
        exit 1
    fi
    echo " -> ok"
fi

wait "$SERVER_PID"
