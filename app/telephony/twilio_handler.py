"""
Twilio telephony helpers.

Twilio Media Streams send audio as μ-law encoded, 8 kHz, mono, base64-encoded
chunks inside JSON messages over a WebSocket.

Inbound message shapes
----------------------
connected : {"event": "connected", "protocol": "Call", "version": "1.0.0"}
start     : {"event": "start", "start": {"streamSid": ..., "callSid": ..., ...}}
media     : {"event": "media", "media": {"payload": "<base64>"}}
stop      : {"event": "stop"}

Outbound message shape (to play audio back to caller)
------------------------------------------------------
{"event": "media", "streamSid": "<sid>", "media": {"payload": "<base64>"}}

To stop any queued audio:
{"event": "clear", "streamSid": "<sid>"}
"""

import base64
import json


def generate_twiml(websocket_url: str) -> str:
    """Return TwiML that connects the call to a Media Stream WebSocket."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{websocket_url}"/>'
        "</Connect>"
        "</Response>"
    )


def parse_twilio_message(raw: str) -> dict:
    """Parse a raw Twilio WebSocket message string into a dict."""
    return json.loads(raw)


def build_media_message(audio_b64: str, stream_sid: str) -> str:
    """Wrap base64 audio in a Twilio outbound media message."""
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": audio_b64},
        }
    )


def build_clear_message(stream_sid: str) -> str:
    """Tell Twilio to discard any buffered audio (used for interruptions)."""
    return json.dumps({"event": "clear", "streamSid": stream_sid})


def decode_audio(payload_b64: str) -> bytes:
    """Decode a Twilio media payload into raw μ-law bytes."""
    return base64.b64decode(payload_b64)


def encode_audio(audio_bytes: bytes) -> str:
    """Encode raw μ-law bytes to base64 for a Twilio media message."""
    return base64.b64encode(audio_bytes).decode("utf-8")
