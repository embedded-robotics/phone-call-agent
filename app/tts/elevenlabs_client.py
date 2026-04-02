"""
ElevenLabs Text-to-Speech client.

Requests audio in ulaw_8000 format directly (μ-law 8 kHz mono) — exactly what
Twilio expects — so no conversion is needed.

IMPORTANT: output_format is a query parameter on the ElevenLabs API, not a
JSON body field. Passing it in the body is silently ignored and MP3 is returned.
"""

import logging
from io import BytesIO

import httpx

logger = logging.getLogger(__name__)

_ELEVENLABS_TTS_URL = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    "?output_format=ulaw_8000"
)


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str) -> None:
        self._api_key = api_key
        self._url = _ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to μ-law 8 kHz bytes ready to send to Twilio.
        Returns empty bytes if synthesis fails.
        """
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream(
                    "POST", self._url, headers=headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    buf = BytesIO()
                    async for chunk in resp.aiter_bytes():
                        buf.write(chunk)
                    return buf.getvalue()
        except Exception:
            logger.exception("ElevenLabs TTS request failed")
            return b""
