#!/usr/bin/env python3
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="QwenTTS WebUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_BASE = "http://127.0.0.1:8000"
DEBUG = False
INSTRUCT = ""
ONESHOT = False


def dbg(*args):
    if DEBUG:
        print("[dbg]", *args, file=sys.stderr)


def split_sentences(text: str):
    # Protect a run of three or more dots (or the unicode ellipsis) so it
    # survives as a single token when told "take a look at the whole
    # sentence" concerns would otherwise split it across multiple shots.
    ellipsis_tokens = []

    def protect(m):
        ellipsis_tokens.append(m.group(0))
        return "\x00E%d\x00" % (len(ellipsis_tokens) - 1)

    protected = re.sub(r"\.\.\.+|\u2026", protect, text)
    parts = protected.split(".")
    sentences = []
    for i, part in enumerate(parts):
        s = part.strip()
        if not s:
            continue
        if i < len(parts) - 1 or protected.endswith("."):
            s = s + "."
        for j in range(len(ellipsis_tokens) - 1, -1, -1):
            s = s.replace("\x00E%d\x00" % j, ellipsis_tokens[j])
        sentences.append(s)
    return sentences


async def _emit_sentence(cl, request, index, sentence, voice, instructions):
    """POST one text unit to the backend and yield its SSE events.
    Yields 'done' on success, 'error' on failure, 'abort' on disconnect."""
    if request and await request.is_disconnected():
        dbg("Client disconnected, stopping stream")
        yield f"event: abort\ndata: {json.dumps({'index': index})}\n\n"
        return
    yield f"event: start\ndata: {json.dumps({'index': index, 'sentence': sentence})}\n\n"
    try:
        payload = {"input": sentence, "response_format": "wav"}
        if voice:
            payload["voice"] = voice
        if instructions:
            payload["instructions"] = instructions
        dbg("POST", f"{TTS_BASE}/v1/audio/speech", payload)
        resp = await cl.post(f"{TTS_BASE}/v1/audio/speech", json=payload)
        resp.raise_for_status()
        dbg("OK", resp.status_code, len(resp.content), "bytes")
        b64 = base64.b64encode(resp.content).decode()
        yield f"event: audio\ndata: {json.dumps({'index': index, 'data': b64})}\n\n"
        yield f"event: done\ndata: {json.dumps({'index': index})}\n\n"
    except httpx.HTTPStatusError as e:
        body_text = e.response.text[:500]
        detail = body_text
        try:
            detail = e.response.json().get("error", {}).get("message", body_text)
        except Exception:
            pass
        dbg("HTTP", e.response.status_code, detail)
        yield f"event: error\ndata: {json.dumps({'index': index, 'message': f'TTS {e.response.status_code}: {detail}'})}\n\n"
    except httpx.RequestError as e:
        dbg("REQ_ERROR", str(e))
        yield f"event: error\ndata: {json.dumps({'index': index, 'message': 'TTS server unreachable — is qwentts-serve running?'})}\n\n"


def build_event_stream(text: str, voice: str, instructions: str = "", request: Request = None, oneshot: bool = None):
    if not instructions:
        instructions = INSTRUCT
    if oneshot is None:
        oneshot = ONESHOT
    text = text.strip()
    if not text:
        return

    async def gen():
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as cl:
            if oneshot:
                # One shot : try the whole input as a single utterance so
                # the model sees the complete context. If the talker KV
                # cache overflows (long prompts), fall back to per
                # sentence streaming automatically.
                yield f"event: meta\ndata: {json.dumps({'total': 1})}\n\n"
                ok = True
                async for ev in _emit_sentence(cl, request, 0, text, voice, instructions):
                    yield ev
                    if ev.startswith("event: done"):
                        ok = True
                    elif ev.startswith("event: error"):
                        ok = False
                if not ok:
                    sentences = split_sentences(text)
                    if len(sentences) <= 1:
                        yield "event: complete\ndata: {}\n\n"
                        return
                    dbg("one-shot failed, falling back to per-sentence")
                    yield f"event: notice\ndata: {json.dumps({'message': 'One-shot overflowed the context cache; synthesizing per sentence instead.'})}\n\n"
                    yield f"event: meta\ndata: {json.dumps({'total': len(sentences)})}\n\n"
                    for i, sentence in enumerate(sentences):
                        async for ev in _emit_sentence(cl, request, i, sentence, voice, instructions):
                            yield ev
            else:
                sentences = split_sentences(text)
                yield f"event: meta\ndata: {json.dumps({'total': len(sentences)})}\n\n"
                for i, sentence in enumerate(sentences):
                    async for ev in _emit_sentence(cl, request, i, sentence, voice, instructions):
                        yield ev
        yield "event: complete\ndata: {}\n\n"

    return gen()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/v1/voices")
async def get_voices():
    async with httpx.AsyncClient() as cl:
        resp = await cl.get(f"{TTS_BASE}/v1/voices", timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch voices")
        return resp.json()


@app.get("/speak")
async def speak_get(request: Request, text: str = Query(...), voice: str = "", instructions: str = "", oneshot: bool = Query(None)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    stream = build_event_stream(text, voice, instructions, request=request, oneshot=oneshot)
    if stream is None:
        raise HTTPException(status_code=400, detail="No sentences found in text")
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/speak")
async def speak_post(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    voice = body.get("voice", "")
    instructions = body.get("instructions", "")
    oneshot = bool(body.get("oneshot", ONESHOT))
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    stream = build_event_stream(text, voice, instructions, request=request, oneshot=oneshot)
    if stream is None:
        raise HTTPException(status_code=400, detail="No sentences found in text")
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/cancel")
async def cancel():
    async with httpx.AsyncClient() as cl:
        resp = await cl.post(f"{TTS_BASE}/v1/audio/speech/cancel", timeout=10)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


def main():
    parser = argparse.ArgumentParser(description="QwenTTS WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8000", help="TTS server URL")
    parser.add_argument("--instruct", default="", help="Default style instructions for voice design models")
    parser.add_argument("--one-shot", action="store_true",
                        help="Synthesise the whole input in a single pass instead of per sentence (streaming)")
    parser.add_argument("--debug", action="store_true", help="Print debug logs to stderr")
    args = parser.parse_args()

    global TTS_BASE, DEBUG, INSTRUCT, ONESHOT
    TTS_BASE = args.tts_url
    INSTRUCT = args.instruct
    ONESHOT  = args.one_shot
    if args.debug:
        DEBUG = True

    print(f"  QwenTTS WebUI", file=sys.stderr)
    print(f"  TTS backend: {TTS_BASE}", file=sys.stderr)
    print(f"  Mode:        {'one shot (whole input)' if ONESHOT else 'streaming (per sentence)'}", file=sys.stderr)
    print(f"  Listening:   http://{args.host}:{args.port}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Make sure qwentts-serve is running:", file=sys.stderr)
    print(f"    qwentts-serve --model <talker> --codec <codec>", file=sys.stderr)
    print(file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
