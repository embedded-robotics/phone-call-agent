from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.config import Settings, get_settings
from app.logging_utils import configure_logging
from app.providers.deepgram import (
    DeepgramSpeechToTextProvider,
    DeepgramTextToSpeechProvider,
)
from app.providers.llm import OpenAiChatProvider, RuleBasedPhoneAgent
from app.providers.mock import MockSpeechToTextProvider, MockTextToSpeechProvider
from app.services.call_artifacts import CallArtifactWriter
from app.services.session_store import SessionStore
from app.services.twilio_media import build_twiml
from app.services.voice_agent import VoiceAgentService


def create_app(
    settings: Settings | None = None,
    *,
    voice_agent: VoiceAgentService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(title="Phone Conversation Agent", version="0.1.0")

    if voice_agent is None:
        session_store = SessionStore()
        if settings.deepgram_api_key:
            stt_provider = DeepgramSpeechToTextProvider(settings)
            tts_provider = DeepgramTextToSpeechProvider(settings)
        else:
            stt_provider = MockSpeechToTextProvider()
            tts_provider = MockTextToSpeechProvider()
        llm_provider = (
            OpenAiChatProvider(settings) if settings.openai_api_key else RuleBasedPhoneAgent()
        )
        voice_agent = VoiceAgentService(
            session_store,
            stt_provider,
            llm_provider,
            tts_provider,
            barge_in_rms_threshold=settings.barge_in_rms_threshold,
            barge_in_consecutive_frames=settings.barge_in_consecutive_frames,
            artifact_writer=CallArtifactWriter(settings),
        )

    app.state.settings = settings
    app.state.voice_agent = voice_agent

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.app_env,
            "transport_mode": settings.transport_mode,
        }

    @app.get("/logs")
    async def log_paths() -> dict[str, str]:
        return {
            "log_dir": settings.log_dir,
            "text_log_file": f"{settings.log_dir}/{settings.text_log_file}",
            "structured_log_file": f"{settings.log_dir}/{settings.structured_log_file}",
            "call_artifact_dir": settings.call_artifact_dir,
        }

    @app.get("/transport")
    async def transport() -> dict[str, str]:
        return {
            "transport_mode": settings.transport_mode,
            "media_stream_url": settings.media_stream_url,
        }

    @app.post("/twilio/voice/inbound", response_class=PlainTextResponse)
    async def inbound_voice(
        request: Request,
        call_sid: Annotated[str | None, Form(alias="CallSid")] = None,
        from_number: Annotated[str | None, Form(alias="From")] = None,
    ) -> str:
        if settings.transport_mode != "twilio":
            return PlainTextResponse(
                "Twilio transport is disabled. Set TRANSPORT_MODE=twilio to use this endpoint.",
                status_code=409,
            )
        agent: VoiceAgentService = request.app.state.voice_agent
        if call_sid:
            await agent.session_store.get_or_create(
                call_sid,
                caller_id=from_number,
                transport="twilio",
            )
        return build_twiml(settings.media_stream_url)

    @app.websocket("/twilio/media")
    async def twilio_media(websocket: WebSocket) -> None:
        if settings.transport_mode != "twilio":
            await websocket.close(code=1008, reason="Twilio transport disabled.")
            return
        await app.state.voice_agent.handle_twilio_ws(websocket)

    @app.get("/microphone/config")
    async def microphone_config() -> JSONResponse:
        if settings.transport_mode != "microphone":
            return JSONResponse(
                {
                    "error": "Microphone transport is disabled. Set TRANSPORT_MODE=microphone to use this endpoint."
                },
                status_code=409,
            )
        return JSONResponse(
            {
                "transport_mode": settings.transport_mode,
                "websocket_url": settings.media_stream_url,
                "sample_rate": 8000,
                "encoding": "mulaw",
                "protocol": {
                    "client_events": ["start", "media", "stop"],
                    "server_events": ["ready", "media", "mark", "clear", "transcript", "metrics"],
                },
            }
        )

    @app.get("/microphone/test")
    async def microphone_test_page() -> FileResponse:
        return FileResponse("app/static/microphone_test.html")

    @app.websocket("/microphone/stream")
    async def microphone_stream(
        websocket: WebSocket,
        session_id: str | None = None,
        caller_id: str | None = None,
    ) -> None:
        if settings.transport_mode != "microphone":
            await websocket.close(code=1008, reason="Microphone transport disabled.")
            return
        await app.state.voice_agent.handle_microphone_ws(
            websocket,
            session_id=session_id,
            caller_id=caller_id,
        )

    return app


app = create_app()
