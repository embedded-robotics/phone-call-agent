"""
Local microphone test — no Twilio required.

Captures audio from the default microphone, streams it to Deepgram STT,
and feeds the transcript through the same sentence-level pipelined
ConversationSession used in production (GPT-4o → ElevenLabs → speakers).

Requirements: pyaudio  (pip install pyaudio)
Usage:        python scripts/run_local_mic.py
Press Ctrl+C to quit.
"""

import asyncio
import audioop
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyaudio
except ImportError:
    print("pyaudio is required: pip install pyaudio")
    sys.exit(1)

from app.agent import ConversationSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Microphone capture settings
MIC_RATE = 16000   # capture at 16 kHz, downsample to 8 kHz for Deepgram
MIC_CHUNK = 320    # 20 ms @ 16 kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Playback sample rate (most speakers prefer 44100)
PLAYBACK_RATE = 44100


async def main() -> None:
    pa = pyaudio.PyAudio()

    # --- Audio callbacks injected into ConversationSession ---

    async def send_audio(mulaw_bytes: bytes) -> None:
        """Convert μ-law 8 kHz back to PCM and play through speakers."""
        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
        pcm_out, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, PLAYBACK_RATE, None)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=PLAYBACK_RATE,
            output=True,
        )
        stream.write(pcm_out)
        stream.stop_stream()
        stream.close()

    async def send_clear() -> None:
        """No-op for local mode (no Twilio buffer to flush)."""
        pass

    session = ConversationSession(send_audio=send_audio, send_clear=send_clear)
    await session.start()

    mic_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_CHUNK,
    )

    print("Listening... (Ctrl+C to stop)\n")

    try:
        while True:
            pcm_16k = mic_stream.read(MIC_CHUNK, exception_on_overflow=False)
            # Downsample 16 kHz → 8 kHz, encode to μ-law for Deepgram
            pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, MIC_RATE, 8000, None)
            mulaw = audioop.lin2ulaw(pcm_8k, 2)
            await session.feed_audio(mulaw)
            await asyncio.sleep(0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()
        await session.stop()


if __name__ == "__main__":
    asyncio.run(main())
