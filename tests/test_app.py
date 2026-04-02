import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import CallState, CallSession, TranscriptEvent
from app.providers.llm import RuleBasedPhoneAgent
from app.providers.mock import MockSpeechToTextProvider, MockTextToSpeechProvider
from app.services.call_artifacts import CallArtifactWriter
from app.services.session_store import SessionStore
from app.services.voice_agent import RealtimeConnection, VoiceAgentService


class DeterministicAgent(RuleBasedPhoneAgent):
    async def generate_reply(self, session: CallSession, user_text: str) -> str:
        return f"Replying to: {user_text}"


async def _build_service() -> tuple[VoiceAgentService, MockSpeechToTextProvider, MockTextToSpeechProvider]:
    session_store = SessionStore()
    stt = MockSpeechToTextProvider()
    tts = MockTextToSpeechProvider()
    service = VoiceAgentService(session_store, stt, DeterministicAgent(), tts)
    return service, stt, tts


def test_inbound_voice_returns_twiml() -> None:
    settings = Settings(
        transport_mode="twilio",
        public_base_url="https://voice.example.com",
        deepgram_api_key=None,
        openai_api_key=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/twilio/voice/inbound",
        data={"CallSid": "CA123", "From": "+15555551212"},
    )

    assert response.status_code == 200
    assert "<Stream url=\"wss://voice.example.com/twilio/media\"" in response.text


def test_microphone_config_exposes_microphone_websocket() -> None:
    settings = Settings(
        transport_mode="microphone",
        public_base_url="https://voice.example.com",
        deepgram_api_key=None,
        openai_api_key=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/microphone/config")

    assert response.status_code == 200
    data = response.json()
    assert data["transport_mode"] == "microphone"
    assert data["websocket_url"] == "wss://voice.example.com/microphone/stream"


def test_microphone_test_page_is_served() -> None:
    settings = Settings(
        transport_mode="microphone",
        public_base_url="https://voice.example.com",
        deepgram_api_key=None,
        openai_api_key=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/microphone/test")

    assert response.status_code == 200
    assert "Microphone Test Console" in response.text
    assert "mic-select" in response.text
    assert "Live Transcript" in response.text
    assert "Live Metrics" in response.text


def test_logs_endpoint_returns_paths(tmp_path: Path) -> None:
    settings = Settings(
        transport_mode="microphone",
        log_dir=str(tmp_path / "logs"),
        call_artifact_dir=str(tmp_path / "logs" / "calls"),
        public_base_url="https://voice.example.com",
        deepgram_api_key=None,
        openai_api_key=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/logs")

    assert response.status_code == 200
    data = response.json()
    assert data["text_log_file"].endswith("logs/app.log")
    assert data["structured_log_file"].endswith("logs/app.jsonl")
    assert data["call_artifact_dir"].endswith("logs/calls")


def test_twilio_endpoint_is_disabled_in_microphone_mode() -> None:
    settings = Settings(
        transport_mode="microphone",
        public_base_url="https://voice.example.com",
        deepgram_api_key=None,
        openai_api_key=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/twilio/voice/inbound",
        data={"CallSid": "CA123", "From": "+15555551212"},
    )

    assert response.status_code == 409
    assert "Twilio transport is disabled" in response.text


@pytest.mark.asyncio
async def test_final_transcript_triggers_reply_and_audio() -> None:
    service, _, _ = await _build_service()

    class StubSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.messages.append(payload)

    socket = StubSocket()
    session = await service.session_store.get_or_create("CA123", stream_sid="MZ123")
    service._connections["CA123"] = RealtimeConnection(websocket=socket, transport="twilio")

    await service.on_transcript(session, TranscriptEvent("user", "hello", True), socket)
    await service._speaker_tasks["CA123"]

    assert session.state == CallState.LISTENING
    assert any(json.loads(message)["event"] == "media" for message in socket.messages)
    assert any("Replying to: hello" == event.text for event in session.transcript_buffer if event.speaker == "assistant")


@pytest.mark.asyncio
async def test_interrupt_cancels_current_generation() -> None:
    service, _, tts = await _build_service()

    class StubSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.messages.append(payload)

    socket = StubSocket()
    session = await service.session_store.get_or_create("CA321", stream_sid="MZ321")
    service._connections["CA321"] = RealtimeConnection(websocket=socket, transport="twilio")

    await service.on_transcript(session, TranscriptEvent("user", "status", True), socket)
    await service.interrupt(session)

    assert session.state == CallState.INTERRUPTED
    assert tts.cancelled


@pytest.mark.asyncio
async def test_microphone_transcript_reply_uses_microphone_message_shape() -> None:
    service, _, _ = await _build_service()

    class StubSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.messages.append(payload)

    socket = StubSocket()
    session = await service.session_store.get_or_create(
        "mic-123",
        stream_sid="mic-123",
        caller_id="local-user",
        transport="microphone",
    )
    service._connections["mic-123"] = RealtimeConnection(
        websocket=socket,
        transport="microphone",
    )

    await service.on_transcript(session, TranscriptEvent("user", "hello", True), socket)
    await service._speaker_tasks["mic-123"]

    transcript_messages = [
        json.loads(message)
        for message in socket.messages
        if json.loads(message)["event"] == "transcript"
    ]
    assert transcript_messages[0]["transcript"]["speaker"] == "user"
    assert transcript_messages[0]["transcript"]["text"] == "hello"
    assert transcript_messages[1]["transcript"]["speaker"] == "assistant"

    metrics_message = next(
        json.loads(message)
        for message in socket.messages
        if json.loads(message)["event"] == "metrics"
    )
    assert metrics_message["metrics"]["generationId"] == 1

    first_media = json.loads(next(message for message in socket.messages if json.loads(message)["event"] == "media"))
    assert first_media["callSid"] == "mic-123"
    assert "streamSid" not in first_media


@pytest.mark.asyncio
async def test_call_artifact_is_written_on_end_call(tmp_path: Path) -> None:
    session_store = SessionStore()
    stt = MockSpeechToTextProvider()
    tts = MockTextToSpeechProvider()
    settings = Settings(
        log_dir=str(tmp_path / "logs"),
        call_artifact_dir=str(tmp_path / "logs" / "calls"),
        deepgram_api_key=None,
        openai_api_key=None,
    )
    service = VoiceAgentService(
        session_store,
        stt,
        DeterministicAgent(),
        tts,
        artifact_writer=CallArtifactWriter(settings),
    )
    session = await service.session_store.get_or_create(
        "CA999",
        stream_sid="MZ999",
        caller_id="+15550001111",
        transport="twilio",
    )
    session.transcript_buffer.append(TranscriptEvent("user", "hello", True))

    await service.end_call("CA999")

    artifact = tmp_path / "logs" / "calls" / "CA999.json"
    assert artifact.exists()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["call_sid"] == "CA999"
    assert data["transcript"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_barge_in_requires_multiple_voiced_frames() -> None:
    service, _, tts = await _build_service()
    service.barge_in_rms_threshold = 0
    service.barge_in_consecutive_frames = 3

    class StubSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.messages.append(payload)

    socket = StubSocket()
    session = await service.session_store.get_or_create("CA777", stream_sid="MZ777")
    service._connections["CA777"] = RealtimeConnection(websocket=socket, transport="twilio")
    session.speaking = True
    session.playback_generation = 1

    assert service._should_barge_in(session, b"\xff" * 160) is False
    assert service._should_barge_in(session, b"\xff" * 160) is False
    assert service._should_barge_in(session, b"\xff" * 160) is True

    await service.interrupt(session)
    assert tts.cancelled
