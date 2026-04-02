from __future__ import annotations

import audioop
import base64
import html


def build_twiml(stream_url: str) -> str:
    safe_url = html.escape(stream_url, quote=True)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{safe_url}" />'
        "</Connect>"
        "</Response>"
    )


def decode_mulaw(payload: str) -> bytes:
    return base64.b64decode(payload)


def encode_mulaw(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


def detect_voice_activity(audio: bytes, *, threshold: int = 300) -> bool:
    if not audio:
        return False
    pcm = audioop.ulaw2lin(audio, 2)
    return audioop.rms(pcm, 2) >= threshold
