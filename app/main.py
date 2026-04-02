"""
FastAPI application entry point.

Routes
------
POST /incoming-call
    Twilio hits this when a call arrives. Returns TwiML that instructs
    Twilio to open a Media Stream WebSocket to /media-stream.

WebSocket /media-stream
    Bidirectional audio stream with Twilio. One ConversationSession is
    created per call and lives for the duration of the WebSocket connection.
"""

import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

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
logger = logging.getLogger(__name__)

app = FastAPI(title="Phone Call Agent")


@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    """Return TwiML to connect the call to our WebSocket media stream."""
    host = request.headers.get("host", f"{settings.host}:{settings.port}")
    ws_url = f"wss://{host}/media-stream"
    twiml = generate_twiml(ws_url)
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    """Handle a Twilio Media Stream WebSocket for one call."""
    await websocket.accept()
    stream_sid: str = ""

    async def send_audio(audio_bytes: bytes) -> None:
        if stream_sid:
            msg = build_media_message(encode_audio(audio_bytes), stream_sid)
            await websocket.send_text(msg)

    async def send_clear() -> None:
        if stream_sid:
            msg = build_clear_message(stream_sid)
            await websocket.send_text(msg)

    session = ConversationSession(send_audio=send_audio, send_clear=send_clear)

    try:
        async for raw in websocket.iter_text():
            msg = parse_twilio_message(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                logger.info("Call started, streamSid=%s", stream_sid)
                await session.start()

            elif event == "media":
                audio = decode_audio(msg["media"]["payload"])
                await session.feed_audio(audio)

            elif event == "stop":
                logger.info("Call stopped, streamSid=%s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        await session.stop()
