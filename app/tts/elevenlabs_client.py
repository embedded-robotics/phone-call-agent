"""
ElevenLabs Text-to-Speech client.

Converts text to μ-law 8 kHz audio suitable for Twilio Media Streams.

Flow:
  text → ElevenLabs streaming API (mp3/pcm) → convert to μ-law 8 kHz → bytes

The ElevenLabs SDK streaming endpoint returns PCM 16-bit 22050 Hz by default
when output_format="pcm_22050". We downsample to 8000 Hz then encode to μ-law
using the standard library `audioop` (Python ≤ 3.12) or `audioop-lts` (3.13+).
"""

import audioop
import logging
from io import BytesIO

import httpx

logger = logging.getLogger(__name__)

_ELEVENLABS_TTS_URL = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
)

# ElevenLabs PCM output sample rate we request
_EL_SAMPLE_RATE = 22050
_TWILIO_SAMPLE_RATE = 8000


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._url = _ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to μ-law 8 kHz bytes ready to send to Twilio.
        Returns empty bytes if synthesis fails.
        """
        pcm = await self._fetch_pcm(text)
        if not pcm:
            return b""
        return _pcm_to_mulaw(pcm)

    async def _fetch_pcm(self, text: str) -> bytes:
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "output_format": "pcm_22050",
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


def _pcm_to_mulaw(pcm_22050: bytes) -> bytes:
    """Downsample PCM 16-bit 22050 Hz → 8000 Hz, then encode to μ-law."""
    # Downsample: audioop.ratecv signature:
    # ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0)
    pcm_8000, _ = audioop.ratecv(
        pcm_22050, 2, 1, _EL_SAMPLE_RATE, _TWILIO_SAMPLE_RATE, None
    )
    mulaw = audioop.lin2ulaw(pcm_8000, 2)
    return mulaw
