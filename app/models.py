from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic


class CallState(StrEnum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    ENDED = "ended"


@dataclass(slots=True)
class TranscriptEvent:
    speaker: str
    text: str
    is_final: bool
    start_ms: int | None = None
    end_ms: int | None = None
    provider_latency_ms: float | None = None


@dataclass(slots=True)
class AssistantTurn:
    generation_id: int
    text: str
    created_at: float = field(default_factory=monotonic)


@dataclass(slots=True)
class CallSession:
    call_sid: str
    stream_sid: str | None = None
    caller_id: str | None = None
    transport: str = "twilio"
    state: CallState = CallState.LISTENING
    transcript_buffer: list[TranscriptEvent] = field(default_factory=list)
    assistant_turn_id: int = 0
    playback_generation: int = 0
    timestamps: dict[str, float] = field(default_factory=dict)
    tool_context: dict[str, str] = field(default_factory=dict)
    pending_user_text: str = ""
    current_assistant_text: str = ""
    stt_started: bool = False
    speaking: bool = False
    barge_in_frame_count: int = 0

    def mark(self, name: str) -> None:
        self.timestamps[name] = monotonic()

    def next_generation(self) -> int:
        self.assistant_turn_id += 1
        self.playback_generation = self.assistant_turn_id
        return self.assistant_turn_id
