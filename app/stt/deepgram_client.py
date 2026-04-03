import asyncio
import logging
from collections.abc import Callable

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types import ListenV1Results, ListenV1UtteranceEnd

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
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._socket = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()

    async def connect(self) -> None:
        self._task = asyncio.create_task(self._run())
        # Wait for either _ready (success) or the task completing early (failure).
        ready_waiter = asyncio.create_task(self._ready.wait())
        done, _ = await asyncio.wait(
            [self._task, ready_waiter],
            timeout=5.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Always cancel the ready_waiter — it's a helper, not the main task.
        ready_waiter.cancel()

        if not done:
            self._task.cancel()
            raise RuntimeError("Deepgram connection timed out")

        if self._task in done and not self._ready.is_set():
            exc = self._task.exception()
            raise RuntimeError(f"Deepgram connection failed: {exc}")

        logger.info("Deepgram STT connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._socket:
            await self._socket.send_media(chunk)

    async def close(self) -> None:
        if self._socket:
            try:
                await self._socket.send_close_stream()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Deepgram STT closed")

    async def _run(self) -> None:
        try:
            async with self._client.listen.v1.connect(
                model="nova-2",
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
                punctuate="true",
                interim_results="true",
                utterance_end_ms=self._utterance_end_ms,
                vad_events="true",
            ) as ws:
                self._socket = ws
                self._ready.set()
                async for msg in ws:
                    if isinstance(msg, ListenV1Results):
                        try:
                            text = msg.channel.alternatives[0].transcript.strip()
                            if text:
                                self._on_transcript(text, bool(msg.is_final))
                        except (AttributeError, IndexError):
                            pass
                    elif isinstance(msg, ListenV1UtteranceEnd):
                        self._on_utterance_end()
        finally:
            self._socket = None
