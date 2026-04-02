"""
Deepgram streaming Speech-to-Text client.

Connects to Deepgram's live transcription WebSocket and exposes:
  - send_audio(chunk: bytes)  — feed raw μ-law audio from Twilio
  - close()                   — gracefully shut down the connection

Callbacks (injected at construction time):
  - on_transcript(text: str, is_final: bool)
  - on_utterance_end()        — fires after utterance_end_ms of silence
"""

import asyncio
import logging
from collections.abc import Callable

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

logger = logging.getLogger(__name__)


class DeepgramSTT:
    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str, bool], None],
        on_utterance_end: Callable[[], None],
        utterance_end_ms: int = 1000,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._utterance_end_ms = utterance_end_ms

        config = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(api_key, config)
        self._connection = None

    async def connect(self) -> None:
        """Open the Deepgram streaming connection."""
        options = LiveOptions(
            model="nova-2",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            punctuate=True,
            interim_results=True,
            utterance_end_ms=str(self._utterance_end_ms),
            vad_events=True,
        )

        self._connection = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents.Transcript, self._handle_transcript
        )
        self._connection.on(
            LiveTranscriptionEvents.UtteranceEnd, self._handle_utterance_end
        )
        self._connection.on(LiveTranscriptionEvents.Error, self._handle_error)

        started = await self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram connection")
        logger.info("Deepgram STT connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection:
            await self._connection.send(chunk)

    async def close(self) -> None:
        if self._connection:
            await self._connection.finish()
            self._connection = None
            logger.info("Deepgram STT closed")

    # ------------------------------------------------------------------ #
    # Internal event handlers                                              #
    # ------------------------------------------------------------------ #

    async def _handle_transcript(self, *args, **kwargs) -> None:
        result = kwargs.get("result")
        if result is None:
            return
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return
            is_final = result.is_final
            self._on_transcript(text, is_final)
        except (AttributeError, IndexError):
            pass

    async def _handle_utterance_end(self, *args, **kwargs) -> None:
        self._on_utterance_end()

    async def _handle_error(self, *args, **kwargs) -> None:
        error = kwargs.get("error")
        logger.error("Deepgram error: %s", error)
