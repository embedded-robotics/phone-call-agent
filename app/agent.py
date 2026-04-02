"""
ConversationSession — orchestrates one phone call end-to-end.

Lifecycle
---------
1. Call starts  → connect Deepgram, play greeting
2. Audio in     → Deepgram STT emits transcripts
3. Silence      → utterance_end fires → send buffered transcript to GPT-4o
4. GPT streams  → text chunks accumulate → synthesize full response via ElevenLabs
5. Audio out    → send μ-law chunks to Twilio via the WebSocket send callback
6. Interruption → if new speech detected while agent is speaking, cancel TTS
7. Call ends    → close Deepgram, reset state

The session is intentionally not aware of the WebSocket protocol details;
it communicates outbound audio via the injected `send_audio` coroutine.
"""

import asyncio
import logging

from app.config import settings
from app.llm.openai_client import OpenAIAgent
from app.stt.deepgram_client import DeepgramSTT
from app.tts.elevenlabs_client import ElevenLabsTTS

logger = logging.getLogger(__name__)

GREETING = "Hello! How can I help you today?"


class ConversationSession:
    def __init__(self, send_audio, send_clear) -> None:
        """
        Parameters
        ----------
        send_audio : async callable(audio_bytes: bytes) -> None
            Coroutine that sends μ-law audio bytes back to the caller.
        send_clear : async callable() -> None
            Coroutine that tells Twilio to flush buffered audio (interruption).
        """
        self._send_audio = send_audio
        self._send_clear = send_clear

        self._llm = OpenAIAgent(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            system_prompt=settings.system_prompt,
        )
        self._tts = ElevenLabsTTS(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )
        self._stt = DeepgramSTT(
            api_key=settings.deepgram_api_key,
            on_transcript=self._on_transcript,
            on_utterance_end=self._on_utterance_end,
        )

        self._transcript_buffer: list[str] = []
        self._agent_speaking = False
        self._speak_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Connect to Deepgram and play the greeting."""
        await self._stt.connect()
        await self._speak(GREETING)

    async def feed_audio(self, audio: bytes) -> None:
        """Forward raw μ-law audio from Twilio to Deepgram."""
        await self._stt.send_audio(audio)

    async def stop(self) -> None:
        """Clean up resources when the call ends."""
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
        await self._stt.close()
        self._llm.reset()
        logger.info("ConversationSession stopped")

    # ------------------------------------------------------------------ #
    # STT callbacks (called from Deepgram event loop — schedule coroutines)#
    # ------------------------------------------------------------------ #

    def _on_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            self._transcript_buffer.append(text)
            # Interrupt agent if it was speaking
            if self._agent_speaking:
                asyncio.create_task(self._interrupt())

    def _on_utterance_end(self) -> None:
        if not self._transcript_buffer:
            return
        user_text = " ".join(self._transcript_buffer).strip()
        self._transcript_buffer.clear()
        if user_text:
            asyncio.create_task(self._respond(user_text))

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _interrupt(self) -> None:
        """Cancel ongoing TTS and clear Twilio's audio buffer."""
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            try:
                await self._speak_task
            except asyncio.CancelledError:
                pass
        await self._send_clear()
        self._agent_speaking = False

    async def _respond(self, user_text: str) -> None:
        """Run LLM → TTS → send audio pipeline for one user turn."""
        logger.info("User said: %s", user_text)
        full_text_chunks: list[str] = []

        async for chunk in self._llm.respond(user_text):
            full_text_chunks.append(chunk)

        full_text = "".join(full_text_chunks).strip()
        if full_text:
            await self._speak(full_text)

    async def _speak(self, text: str) -> None:
        """Synthesize text and stream audio to the caller."""
        self._speak_task = asyncio.create_task(self._stream_tts(text))
        try:
            await self._speak_task
        except asyncio.CancelledError:
            logger.debug("TTS cancelled (interrupted by user)")

    async def _stream_tts(self, text: str) -> None:
        self._agent_speaking = True
        try:
            audio = await self._tts.synthesize(text)
            if audio:
                # Send in 160-byte chunks (20 ms of μ-law @ 8 kHz)
                chunk_size = 160
                for i in range(0, len(audio), chunk_size):
                    await self._send_audio(audio[i : i + chunk_size])
                    await asyncio.sleep(0.02)  # pace at real-time
        finally:
            self._agent_speaking = False
