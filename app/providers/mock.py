from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from app.models import CallSession, TranscriptEvent
from app.providers.base import SpeechToTextProvider, TextToSpeechProvider


class MockSpeechToTextProvider(SpeechToTextProvider):
    def __init__(self) -> None:
        self.callbacks: dict[str, Callable[[TranscriptEvent], object]] = {}

    async def start_stream(
        self,
        session: CallSession,
        on_transcript: Callable[[TranscriptEvent], object],
    ) -> None:
        self.callbacks[session.call_sid] = on_transcript
        session.stt_started = True

    async def send_audio(self, session: CallSession, chunk: bytes) -> None:
        return None

    async def stop(self, session: CallSession) -> None:
        self.callbacks.pop(session.call_sid, None)

    async def emit_transcript(
        self,
        session: CallSession,
        text: str,
        *,
        is_final: bool = True,
    ) -> None:
        callback = self.callbacks[session.call_sid]
        result = callback(TranscriptEvent(speaker="user", text=text, is_final=is_final))
        if asyncio.iscoroutine(result):
            await result


class MockTextToSpeechProvider(TextToSpeechProvider):
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, int]] = []

    async def synthesize_stream(
        self,
        session: CallSession,
        text: str,
        generation_id: int,
    ) -> AsyncIterator[bytes]:
        for _ in range(2):
            if (session.call_sid, generation_id) in self.cancelled:
                return
            yield b"\xff" * 160
            await asyncio.sleep(0)

    async def cancel(self, session: CallSession, generation_id: int) -> None:
        self.cancelled.append((session.call_sid, generation_id))
