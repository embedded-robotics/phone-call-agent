"""
Local microphone test — no Twilio required.

Captures audio from the default microphone, sends it through Deepgram STT,
feeds the transcript to GPT-4o, synthesizes the reply via ElevenLabs, and
plays it back through the default speakers.

Requirements: pyaudio  (pip install pyaudio)
Usage:        python scripts/run_local_mic.py
Press Ctrl+C to quit.
"""

import asyncio
import audioop
import logging
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyaudio
except ImportError:
    print("pyaudio is required: pip install pyaudio")
    sys.exit(1)

from app.config import settings
from app.llm.openai_client import OpenAIAgent
from app.stt.deepgram_client import DeepgramSTT
from app.tts.elevenlabs_client import ElevenLabsTTS, _pcm_to_mulaw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAudio settings — capture at 16 kHz, convert to 8 kHz μ-law for Deepgram
MIC_RATE = 16000
MIC_CHUNK = 320  # 20 ms @ 16 kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16


class LocalSession:
    def __init__(self) -> None:
        self._llm = OpenAIAgent(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            system_prompt=settings.system_prompt,
        )
        self._tts = ElevenLabsTTS(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )
        self._transcript_buffer: list[str] = []
        self._pa = pyaudio.PyAudio()

        self._stt = DeepgramSTT(
            api_key=settings.deepgram_api_key,
            on_transcript=self._on_transcript,
            on_utterance_end=self._on_utterance_end,
        )

    def _on_transcript(self, text: str, is_final: bool) -> None:
        print(f"[STT {'FINAL' if is_final else 'interim'}] {text}")
        if is_final:
            self._transcript_buffer.append(text)

    def _on_utterance_end(self) -> None:
        if not self._transcript_buffer:
            return
        user_text = " ".join(self._transcript_buffer).strip()
        self._transcript_buffer.clear()
        if user_text:
            asyncio.create_task(self._respond(user_text))

    async def _respond(self, user_text: str) -> None:
        print(f"[USER] {user_text}")
        chunks = []
        async for chunk in self._llm.respond(user_text):
            chunks.append(chunk)
        reply = "".join(chunks).strip()
        print(f"[AGENT] {reply}")
        if reply:
            await self._play(reply)

    async def _play(self, text: str) -> None:
        audio = await self._tts.synthesize(text)
        if not audio:
            return
        # μ-law → PCM 16-bit for playback
        pcm = audioop.ulaw2lin(audio, 2)
        # upsample 8000 → 44100 for typical speaker output
        pcm_44100, _ = audioop.ratecv(pcm, 2, 1, 8000, 44100, None)
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            output=True,
        )
        stream.write(pcm_44100)
        stream.stop_stream()
        stream.close()

    async def run(self) -> None:
        await self._stt.connect()
        print("Listening... (Ctrl+C to stop)")

        mic_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )

        try:
            while True:
                pcm_16k = mic_stream.read(MIC_CHUNK, exception_on_overflow=False)
                # Downsample 16kHz → 8kHz for Deepgram
                pcm_8k, _ = audioop.ratecv(pcm_16k, 2, 1, MIC_RATE, 8000, None)
                mulaw = audioop.lin2ulaw(pcm_8k, 2)
                await self._stt.send_audio(mulaw)
                await asyncio.sleep(0)  # yield to event loop
        except asyncio.CancelledError:
            pass
        finally:
            mic_stream.stop_stream()
            mic_stream.close()
            self._pa.terminate()
            await self._stt.close()


async def main() -> None:
    session = LocalSession()
    try:
        await session.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
