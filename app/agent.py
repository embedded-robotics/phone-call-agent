"""
ConversationSession with sentence-level LLM→TTS pipelining.

Three concurrent asyncio tasks run for every agent turn:

  LLM producer     streams GPT-4o tokens, splits at sentence boundaries (.?!),
                   pushes complete sentences into sentence_queue.

  TTS synthesizer  dequeues sentences, calls ElevenLabs immediately (no waiting
                   for the full LLM response), pushes audio bytes into audio_queue.

  Audio player     dequeues audio blobs, passes them to the injected send_audio callback.

Interruption via generation counter
------------------------------------
Every pipeline run is stamped with a monotonically increasing _generation number.
_interrupt() increments the counter before cancelling the asyncio task.
Any audio blob that arrives in _audio_player after the generation has changed is
silently dropped — even if httpx finished delivering the response before the
asyncio CancelledError propagated to the synthesizer coroutine.

This makes interruption reliable regardless of httpx cancellation timing.
"""

import asyncio
import logging
import time
from typing import Optional

from app.config import settings
from app.llm.openai_client import OpenAIAgent
from app.stt.deepgram_client import DeepgramSTT
from app.tts.elevenlabs_client import ElevenLabsTTS

logger = logging.getLogger(__name__)

GREETING = "Hello! How can I help you today?"
_SENTENCE_ENDS = frozenset(".?!")
_SENTINEL = None


def _split_at_boundary(buffer: str) -> tuple[str, str]:
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_ENDS:
            return buffer[: i + 1].strip(), buffer[i + 1 :].lstrip()
    return "", buffer


class ConversationSession:
    def __init__(self, send_audio, send_clear, send_metrics=None) -> None:
        """
        Parameters
        ----------
        send_audio : async callable(audio_bytes: bytes) -> None
            Sends a μ-law audio blob to the client (browser WS or Twilio WS).
        send_clear : async callable() -> None
            Signals the client to discard buffered audio on interruption.
        send_metrics : async callable(metrics: dict) -> None, optional
            Receives per-turn latency metrics: stt_ms, llm_ms, tts_ms.
        """
        self._send_audio = send_audio
        self._send_clear = send_clear
        self._send_metrics = send_metrics

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
        self._agent_speaking = False   # True while LLM/TTS pipeline is running
        self._playback_active = False  # True while browser still has audio queued
        self._pipeline_task: Optional[asyncio.Task] = None
        self._generation: int = 0      # incremented on every interrupt
        self._utterance_end_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        await self._stt.connect()
        await self._run_pipeline(GREETING, is_greeting=True)

    async def feed_audio(self, audio: bytes) -> None:
        await self._stt.send_audio(audio)

    async def stop(self) -> None:
        await self._cancel_pipeline()
        await self._stt.close()
        self._llm.reset()
        logger.info("ConversationSession stopped")

    # ------------------------------------------------------------------ #
    # STT callbacks                                                        #
    # ------------------------------------------------------------------ #

    def notify_playback_done(self) -> None:
        """Called when the browser finishes playing all queued audio."""
        self._playback_active = False

    def _on_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            self._transcript_buffer.append(text)
            if self._agent_speaking or self._playback_active:
                # Increment generation immediately (sync — no await) so the
                # audio player drops all in-flight blobs right now, before any
                # coroutine gets a chance to run.
                self._generation += 1
                self._playback_active = False
                logger.info("Interrupted — generation now %d", self._generation)
                # Schedule the browser/Twilio clear signal — fire and forget.
                asyncio.create_task(self._cancel_pipeline())
                asyncio.create_task(self._send_clear())

    def _on_utterance_end(self) -> None:
        if not self._transcript_buffer:
            return
        user_text = " ".join(self._transcript_buffer).strip()
        self._transcript_buffer.clear()
        if user_text:
            self._utterance_end_time = time.monotonic()
            asyncio.create_task(self._respond(user_text))

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    async def _respond(self, user_text: str) -> None:
        logger.info("User said: %s", user_text)
        await self._run_pipeline(user_text, is_greeting=False)

    async def _run_pipeline(self, text_or_prompt: str, *, is_greeting: bool) -> None:
        await self._cancel_pipeline()

        sentence_queue: asyncio.Queue = asyncio.Queue()
        audio_queue: asyncio.Queue = asyncio.Queue()
        generation = self._generation          # capture current generation

        if is_greeting:
            producer = self._greeting_producer(text_or_prompt, sentence_queue)
            timing = None
        else:
            timing: Optional[dict] = {}
            producer = self._llm_producer(text_or_prompt, sentence_queue, timing)

        self._pipeline_task = asyncio.create_task(
            self._pipeline(producer, sentence_queue, audio_queue, generation, timing)
        )
        try:
            await self._pipeline_task
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled (gen=%d)", generation)

    async def _pipeline(
        self,
        producer,
        sentence_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
        generation: int,
        timing: Optional[dict],
    ) -> None:
        self._agent_speaking = True
        try:
            await asyncio.gather(
                producer,
                self._tts_synthesizer(sentence_queue, audio_queue, timing),
                self._audio_player(audio_queue, generation),
            )
        finally:
            self._agent_speaking = False

    # Producer variants ---------------------------------------------------

    async def _greeting_producer(
        self, greeting: str, sentence_queue: asyncio.Queue
    ) -> None:
        await sentence_queue.put(greeting)
        await sentence_queue.put(_SENTINEL)

    async def _llm_producer(
        self, user_text: str, sentence_queue: asyncio.Queue, timing: dict
    ) -> None:
        t_llm_start = time.monotonic()
        timing["stt_ms"] = round((t_llm_start - self._utterance_end_time) * 1000)

        buffer = ""
        first_token = True
        async for token in self._llm.respond(user_text):
            if first_token:
                timing["llm_ms"] = round((time.monotonic() - t_llm_start) * 1000)
                first_token = False
            buffer += token
            while True:
                sentence, remainder = _split_at_boundary(buffer)
                if not sentence:
                    break
                logger.info("Sentence ready: %s", sentence)
                await sentence_queue.put(sentence)
                buffer = remainder

        if buffer.strip():
            await sentence_queue.put(buffer.strip())
        await sentence_queue.put(_SENTINEL)

    # Synthesizer ---------------------------------------------------------

    async def _tts_synthesizer(
        self,
        sentence_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
        timing: Optional[dict],
    ) -> None:
        first = True
        while True:
            sentence = await sentence_queue.get()
            if sentence is _SENTINEL:
                await audio_queue.put(_SENTINEL)
                break
            logger.info("Synthesizing: %s", sentence)
            t0 = time.monotonic()
            audio = await self._tts.synthesize(sentence)
            if first:
                first = False
                if timing is not None:
                    timing["tts_ms"] = round((time.monotonic() - t0) * 1000)
                    if self._send_metrics and all(k in timing for k in ("stt_ms", "llm_ms", "tts_ms")):
                        asyncio.create_task(self._send_metrics(timing.copy()))
            if audio:
                await audio_queue.put(audio)

    # Player --------------------------------------------------------------

    async def _audio_player(
        self, audio_queue: asyncio.Queue, generation: int
    ) -> None:
        """
        Send audio blobs to the client.
        Checks generation before every send — blobs produced by a cancelled
        pipeline are dropped even if httpx delivered them after cancellation.
        """
        while True:
            audio = await audio_queue.get()
            if audio is _SENTINEL:
                break
            if self._generation != generation:
                logger.info("Dropping stale audio blob (gen %d != %d)", generation, self._generation)
                continue
            self._playback_active = True
            await self._send_audio(audio)

    # ------------------------------------------------------------------ #
    # Interruption                                                         #
    # ------------------------------------------------------------------ #

    async def _cancel_pipeline(self) -> None:
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        self._pipeline_task = None
        self._agent_speaking = False
        self._playback_active = False
