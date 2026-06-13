#!/usr/bin/env python3
import argparse
import base64
import json
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


def dbg(*args):
    if DEBUG:
        print("[dbg]", *args, file=sys.stderr)


def split_sentences(text: str):
    parts = text.split(".")
    sentences = []
    for i, part in enumerate(parts):
        s = part.strip()
        if s:
            if i < len(parts) - 1 or text.endswith("."):
                sentences.append(s + ".")
            else:
                sentences.append(s)
    return sentences


def build_event_stream(text: str, voice: str, instructions: str = "", request: Request = None):
    if not instructions:
        instructions = INSTRUCT
    sentences = split_sentences(text.strip())
    if not sentences:
        return

    async def gen():
        total = len(sentences)
        yield f"event: meta\ndata: {json.dumps({'total': total})}\n\n"

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as cl:
            for i, sentence in enumerate(sentences):
                if request and await request.is_disconnected():
                    dbg("Client disconnected, stopping stream")
                    yield f"event: abort\ndata: {json.dumps({'index': i})}\n\n"
                    return

                yield f"event: start\ndata: {json.dumps({'index': i, 'sentence': sentence})}\n\n"

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
                    yield f"event: audio\ndata: {json.dumps({'index': i, 'data': b64})}\n\n"
                    yield f"event: done\ndata: {json.dumps({'index': i})}\n\n"

                except httpx.HTTPStatusError as e:
                    body_text = e.response.text[:500]
                    detail = body_text
                    try:
                        detail = e.response.json().get("error", {}).get("message", body_text)
                    except Exception:
                        pass
                    dbg("HTTP", e.response.status_code, detail)
                    yield f"event: error\ndata: {json.dumps({'index': i, 'message': f'TTS {e.response.status_code}: {detail}'})}\n\n"
                except httpx.RequestError as e:
                    dbg("REQ_ERROR", str(e))
                    yield f"event: error\ndata: {json.dumps({'index': i, 'message': 'TTS server unreachable — is qwentts-serve running?'})}\n\n"

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
async def speak_get(request: Request, text: str = Query(...), voice: str = "", instructions: str = ""):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    stream = build_event_stream(text, voice, instructions, request=request)
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
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    stream = build_event_stream(text, voice, instructions, request=request)
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
    parser.add_argument("--debug", action="store_true", help="Print debug logs to stderr")
    args = parser.parse_args()

    global TTS_BASE, DEBUG, INSTRUCT
    TTS_BASE = args.tts_url
    INSTRUCT = args.instruct
    if args.debug:
        DEBUG = True

    print(f"  QwenTTS WebUI", file=sys.stderr)
    print(f"  TTS backend: {TTS_BASE}", file=sys.stderr)
    print(f"  Listening:   http://{args.host}:{args.port}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Make sure qwentts-serve is running:", file=sys.stderr)
    print(f"    qwentts-serve --model <talker> --codec <codec>", file=sys.stderr)
    print(file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
