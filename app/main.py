"""
FastAPI application entry point.

Routes
------
GET  /                   Serves the browser voice UI (static/index.html)
POST /incoming-call      Twilio webhook — returns TwiML connecting to /media-stream
WS   /media-stream       Twilio Media Stream WebSocket (one session per call)
WS   /ws                 Browser WebSocket — mic PCM in, μ-law audio out
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.agent import ConversationSession
from app.config import settings
from app.telephony.twilio_handler import (
    build_clear_message,
    build_media_message,
    decode_audio,
    encode_audio,
    generate_twiml,
    parse_twilio_message,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("app").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Phone Call Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ------------------------------------------------------------------ #
# Twilio                                                              #
# ------------------------------------------------------------------ #

@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    host = request.headers.get("host", f"{settings.host}:{settings.port}")
    twiml = generate_twiml(f"wss://{host}/media-stream")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_sid: str = ""

    async def send_audio(audio_bytes: bytes) -> None:
        if not stream_sid:
            return
        chunk_size = 160
        for i in range(0, len(audio_bytes), chunk_size):
            msg = build_media_message(
                encode_audio(audio_bytes[i : i + chunk_size]), stream_sid
            )
            await websocket.send_text(msg)
            await asyncio.sleep(0.02)

    async def send_clear() -> None:
        if stream_sid:
            await websocket.send_text(build_clear_message(stream_sid))

    session = ConversationSession(send_audio=send_audio, send_clear=send_clear)

    try:
        async for raw in websocket.iter_text():
            msg = parse_twilio_message(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                logger.info("Twilio call started, streamSid=%s", stream_sid)
                await session.start()
            elif event == "media":
                await session.feed_audio(decode_audio(msg["media"]["payload"]))
            elif event == "stop":
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    finally:
        await session.stop()


# ------------------------------------------------------------------ #
# Browser WebSocket                                                   #
# ------------------------------------------------------------------ #

@app.websocket("/ws")
async def browser_ws(websocket: WebSocket) -> None:
    """
    Protocol (binary frames only):
      Browser → Server : raw PCM 16-bit little-endian, 16000 Hz, mono
      Server → Browser : raw μ-law 8000 Hz mono (each message = one TTS blob)

    The browser converts μ-law back to PCM and plays it via AudioContext.
    """
    await websocket.accept()
    logger.info("Browser WebSocket connected")

    async def send_audio(mulaw_bytes: bytes) -> None:
        await websocket.send_bytes(mulaw_bytes)

    async def send_clear() -> None:
        # Signal the browser to stop any queued playback
        await websocket.send_text("clear")

    session = ConversationSession(send_audio=send_audio, send_clear=send_clear)
    await session.start()

    try:
        async for data in websocket.iter_bytes():
            await session.feed_audio(_pcm16k_to_mulaw8k(data))
    except WebSocketDisconnect:
        logger.info("Browser WebSocket disconnected")
    finally:
        await session.stop()


# ------------------------------------------------------------------ #
# Audio conversion (browser PCM → Deepgram μ-law)                    #
# ------------------------------------------------------------------ #

import audioop  # noqa: E402 — stdlib, no install needed


def _pcm16k_to_mulaw8k(pcm_16k: bytes) -> bytes:
    """Downsample browser PCM (16-bit 16kHz) to μ-law 8kHz for Deepgram."""
    remainder = len(pcm_16k) % 2
    if remainder:
        pcm_16k = pcm_16k[:-remainder]
    pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)
