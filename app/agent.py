"""
ConversationSession with sentence-level LLM→TTS pipelining.

Three concurrent asyncio tasks run for every agent turn:

  LLM producer     streams GPT-4o tokens, splits at sentence boundaries (.?!),
                   pushes complete sentences into sentence_queue.

  TTS synthesizer  dequeues sentences, calls ElevenLabs immediately (no waiting
                   for the full LLM response), pushes audio bytes into audio_queue.

  Audio player     dequeues audio chunks, sends 160-byte frames to Twilio in order.

Timeline example for "Sure, I can help. What's your account number?":

  t=0ms   GPT starts streaming
  t=240ms sentence 1 complete → TTS synthesis starts immediately
  t=540ms sentence 1 audio ready → playback starts
  t=560ms sentence 2 complete → TTS synthesis starts (overlaps playback)
  t=860ms sentence 2 audio ready → plays right after sentence 1 finishes

Without pipelining the total wait would be ~1500ms. With pipelining the caller
hears the first words at ~540ms.
"""

import asyncio
import logging
from typing import Optional

from app.config import settings
from app.llm.openai_client import OpenAIAgent
from app.stt.deepgram_client import DeepgramSTT
from app.tts.elevenlabs_client import ElevenLabsTTS

logger = logging.getLogger(__name__)

GREETING = "Hello! How can I help you today?"
_SENTENCE_ENDS = frozenset(".?!")
_SENTINEL = None  # queue termination marker


def _split_at_boundary(buffer: str) -> tuple[str, str]:
    """
    Split buffer at the first sentence-ending character (.?!).
    Returns (sentence_including_punctuation, remaining_text).
    Returns ("", buffer) if no boundary found.
    """
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_ENDS:
            sentence = buffer[: i + 1].strip()
            remainder = buffer[i + 1 :].lstrip()
            return sentence, remainder
    return "", buffer


class ConversationSession:
    def __init__(self, send_audio, send_clear) -> None:
        """
        Parameters
        ----------
        send_audio : async callable(audio_bytes: bytes) -> None
            Sends μ-law audio bytes to the caller via Twilio WebSocket.
        send_clear : async callable() -> None
            Flushes Twilio's audio buffer (used on interruption).
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
        self._pipeline_task: Optional[asyncio.Task] = None

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

    def _on_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            self._transcript_buffer.append(text)
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
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    async def _respond(self, user_text: str) -> None:
        logger.info("User said: %s", user_text)
        await self._run_pipeline(user_text, is_greeting=False)

    async def _run_pipeline(self, text_or_prompt: str, *, is_greeting: bool) -> None:
        """
        Launch the 3-task pipeline and track it so it can be cancelled on
        interruption. For the greeting, text_or_prompt is the literal string
        to speak. For LLM turns, it is the user utterance.
        """
        await self._cancel_pipeline()

        sentence_queue: asyncio.Queue = asyncio.Queue()
        audio_queue: asyncio.Queue = asyncio.Queue()

        if is_greeting:
            producer = self._greeting_producer(text_or_prompt, sentence_queue)
        else:
            producer = self._llm_producer(text_or_prompt, sentence_queue)

        self._pipeline_task = asyncio.create_task(
            self._pipeline(producer, sentence_queue, audio_queue)
        )
        try:
            await self._pipeline_task
        except asyncio.CancelledError:
            logger.debug("Pipeline cancelled")

    async def _pipeline(self, producer, sentence_queue, audio_queue) -> None:
        """Run producer, synthesizer, and player concurrently."""
        self._agent_speaking = True
        try:
            await asyncio.gather(
                producer,
                self._tts_synthesizer(sentence_queue, audio_queue),
                self._audio_player(audio_queue),
            )
        finally:
            self._agent_speaking = False

    # Producer variants ---------------------------------------------------

    async def _greeting_producer(
        self, greeting: str, sentence_queue: asyncio.Queue
    ) -> None:
        """Push the greeting as a single sentence then signal done."""
        await sentence_queue.put(greeting)
        await sentence_queue.put(_SENTINEL)

    async def _llm_producer(
        self, user_text: str, sentence_queue: asyncio.Queue
    ) -> None:
        """
        Stream GPT-4o tokens, split at sentence boundaries, enqueue sentences.
        Flushes any remaining text after the stream ends.
        """
        buffer = ""
        async for token in self._llm.respond(user_text):
            buffer += token
            while True:
                sentence, remainder = _split_at_boundary(buffer)
                if not sentence:
                    break
                logger.debug("Sentence ready: %s", sentence)
                await sentence_queue.put(sentence)
                buffer = remainder

        # Flush leftover text (no trailing punctuation)
        if buffer.strip():
            await sentence_queue.put(buffer.strip())

        await sentence_queue.put(_SENTINEL)

    # Synthesizer ---------------------------------------------------------

    async def _tts_synthesizer(
        self,
        sentence_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
    ) -> None:
        """
        Dequeue sentences and synthesize them as fast as ElevenLabs allows.
        Runs concurrently with the LLM producer, so synthesis of sentence N
        overlaps with GPT generation of sentence N+1.
        """
        while True:
            sentence = await sentence_queue.get()
            if sentence is _SENTINEL:
                await audio_queue.put(_SENTINEL)
                break
            logger.debug("Synthesizing: %s", sentence)
            audio = await self._tts.synthesize(sentence)
            if audio:
                await audio_queue.put(audio)

    # Player --------------------------------------------------------------

    async def _audio_player(self, audio_queue: asyncio.Queue) -> None:
        """
        Dequeue synthesized audio blobs and send them to Twilio in 160-byte
        frames (20 ms of μ-law @ 8 kHz) at real-time pace.
        """
        while True:
            audio = await audio_queue.get()
            if audio is _SENTINEL:
                break
            chunk_size = 160
            for i in range(0, len(audio), chunk_size):
                await self._send_audio(audio[i : i + chunk_size])
                await asyncio.sleep(0.02)

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

    async def _interrupt(self) -> None:
        await self._cancel_pipeline()
        await self._send_clear()
