from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

from app.models import CallSession, TranscriptEvent


TranscriptCallback = Callable[[TranscriptEvent], "object"]


class SpeechToTextProvider(ABC):
    @abstractmethod
    async def start_stream(
        self,
        session: CallSession,
        on_transcript: Callable[[TranscriptEvent], "object"],
    ) -> None:
        """Initialize any remote streaming state for the call."""

    @abstractmethod
    async def send_audio(self, session: CallSession, chunk: bytes) -> None:
        """Forward one audio chunk to the STT provider."""

    @abstractmethod
    async def stop(self, session: CallSession) -> None:
        """Tear down the stream for a call."""


class TextToSpeechProvider(ABC):
    @abstractmethod
    async def synthesize_stream(
        self,
        session: CallSession,
        text: str,
        generation_id: int,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks encoded as 8k mu-law for Twilio."""

    @abstractmethod
    async def cancel(self, session: CallSession, generation_id: int) -> None:
        """Cancel any in-flight audio generation for a call."""


class LlmProvider(ABC):
    @abstractmethod
    async def generate_reply(self, session: CallSession, user_text: str) -> str:
        """Generate a concise reply for the live phone call."""
