from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.models import CallState, CallSession, TranscriptEvent
from app.providers.base import LlmProvider, SpeechToTextProvider, TextToSpeechProvider
from app.services.call_artifacts import CallArtifactWriter
from app.services.session_store import SessionStore
from app.services.twilio_media import decode_mulaw, detect_voice_activity, encode_mulaw

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RealtimeConnection:
    websocket: WebSocket
    transport: str


class VoiceAgentService:
    def __init__(
        self,
        session_store: SessionStore,
        stt_provider: SpeechToTextProvider,
        llm_provider: LlmProvider,
        tts_provider: TextToSpeechProvider,
        *,
        barge_in_rms_threshold: int = 900,
        barge_in_consecutive_frames: int = 3,
        artifact_writer: CallArtifactWriter | None = None,
    ) -> None:
        self.session_store = session_store
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.barge_in_consecutive_frames = max(1, barge_in_consecutive_frames)
        self.artifact_writer = artifact_writer
        self._speaker_tasks: dict[str, asyncio.Task[None]] = {}
        self._connections: dict[str, RealtimeConnection] = {}

    async def handle_twilio_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        call_sid: str | None = None
        session: CallSession | None = None
        try:
            while True:
                message = await websocket.receive_text()
                payload = json.loads(message)
                event = payload.get("event")
                if event == "connected":
                    logger.info("Twilio media stream connected.", extra={"event_name": "connected"})
                elif event == "start":
                    start = payload.get("start", {})
                    call_sid = start.get("callSid") or start.get("call_sid")
                    stream_sid = start.get("streamSid") or start.get("stream_sid")
                    custom = start.get("customParameters") or {}
                    caller_id = custom.get("from")
                    if not call_sid:
                        raise ValueError("Missing callSid in Twilio start event.")
                    session = await self.session_store.get_or_create(
                        call_sid,
                        stream_sid=stream_sid,
                        caller_id=caller_id,
                        transport="twilio",
                    )
                    self._connections[call_sid] = RealtimeConnection(
                        websocket=websocket,
                        transport="twilio",
                    )
                    session.mark("call_started")
                    logger.info(
                        "Call session started.",
                        extra={
                            "call_sid": session.call_sid,
                            "stream_sid": session.stream_sid,
                            "transport": session.transport,
                            "state": session.state.value,
                            "event_name": "start",
                        },
                    )
                    await self.stt_provider.start_stream(
                        session,
                        lambda transcript: self.on_transcript(session, transcript, websocket),
                    )
                elif event == "media":
                    media = payload.get("media", {})
                    stream_sid = media.get("streamSid") or payload.get("streamSid")
                    audio = decode_mulaw(media.get("payload", ""))
                    if not call_sid:
                        call_sid = await self._resolve_call_sid(stream_sid)
                    if not call_sid:
                        continue
                    session = await self.session_store.get(call_sid)
                    if not session:
                        continue
                    session.mark("audio_in")
                    if session.speaking and self._should_barge_in(session, audio):
                        await self.interrupt(session)
                    await self.stt_provider.send_audio(session, audio)
                elif event == "mark":
                    if call_sid:
                        session = await self.session_store.get(call_sid)
                        if session:
                            session.mark("mark_received")
                elif event == "stop":
                    stop = payload.get("stop", {})
                    call_sid = stop.get("callSid") or call_sid
                    if call_sid:
                        await self.end_call(call_sid)
                    break
        except WebSocketDisconnect:
            logger.info("Twilio media stream disconnected.", extra={"call_sid": call_sid})
        except Exception:
            logger.exception("Voice WebSocket loop failed.", extra={"call_sid": call_sid})
            if call_sid and session is None:
                session = await self.session_store.get(call_sid)
            if session:
                session.state = CallState.ERROR
            raise
        finally:
            if call_sid:
                await self.end_call(call_sid)
            await websocket.close()

    async def handle_microphone_ws(
        self,
        websocket: WebSocket,
        session_id: str | None = None,
        caller_id: str | None = None,
    ) -> None:
        await websocket.accept()
        call_sid = session_id or f"mic-{uuid4().hex}"
        session: CallSession | None = None
        try:
            session = await self.session_store.get_or_create(
                call_sid,
                stream_sid=call_sid,
                caller_id=caller_id or "microphone",
                transport="microphone",
            )
            self._connections[call_sid] = RealtimeConnection(
                websocket=websocket,
                transport="microphone",
            )
            session.mark("call_started")
            logger.info(
                "Microphone session started.",
                extra={
                    "call_sid": session.call_sid,
                    "stream_sid": session.stream_sid,
                    "transport": session.transport,
                    "state": session.state.value,
                    "event_name": "start",
                },
            )
            await self.stt_provider.start_stream(
                session,
                lambda transcript: self.on_transcript(session, transcript, websocket),
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "ready",
                        "callSid": call_sid,
                        "sampleRate": 8000,
                        "encoding": "mulaw",
                        "transport": "microphone",
                    }
                )
            )
            while True:
                message = await websocket.receive_text()
                payload = json.loads(message)
                event = payload.get("event")
                if event == "start":
                    session.caller_id = payload.get("callerId") or session.caller_id
                elif event == "media":
                    media = payload.get("media", {})
                    encoded = media.get("payload") or payload.get("payload") or ""
                    audio = base64.b64decode(encoded) if encoded else b""
                    session.mark("audio_in")
                    if session.speaking and self._should_barge_in(session, audio):
                        await self.interrupt(session)
                    await self.stt_provider.send_audio(session, audio)
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            logger.info("Microphone stream disconnected.", extra={"call_sid": call_sid})
        except Exception:
            logger.exception("Microphone WebSocket loop failed.", extra={"call_sid": call_sid})
            if session is not None:
                session.state = CallState.ERROR
            raise
        finally:
            await self.end_call(call_sid)
            await websocket.close()

    async def on_transcript(
        self,
        session: CallSession,
        event: TranscriptEvent,
        websocket: WebSocket,
    ) -> None:
        try:
            session.transcript_buffer.append(event)
            if "first_stt_interim" not in session.timestamps:
                session.mark("first_stt_interim")
            await self._send_transcript_event(session, event)
            logger.info(
                "Transcript received.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "state": session.state.value,
                    "event_name": "transcript",
                },
            )
            if not event.is_final:
                session.pending_user_text = event.text
                if session.speaking:
                    await self.interrupt(session)
                return

            user_text = event.text.strip() or session.pending_user_text.strip()
            session.pending_user_text = ""
            if not user_text:
                return

            session.state = CallState.THINKING
            session.mark("end_of_user_utterance")
            reply = await self.llm_provider.generate_reply(session, user_text)
            session.mark("first_llm_token")
            assistant_event = TranscriptEvent(speaker="assistant", text=reply, is_final=True)
            session.transcript_buffer.append(assistant_event)
            await self._send_transcript_event(session, assistant_event)
            generation_id = session.next_generation()
            session.current_assistant_text = reply
            logger.info(
                "Assistant reply generated.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "generation_id": generation_id,
                    "state": session.state.value,
                    "event_name": "assistant_reply",
                },
            )
            self._speaker_tasks[session.call_sid] = asyncio.create_task(
                self._speak_reply(session, websocket, reply, generation_id)
            )
        except Exception:
            session.state = CallState.ERROR
            logger.exception(
                "Transcript handling failed.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "state": session.state.value,
                    "event_name": "transcript_error",
                },
            )
            raise

    async def interrupt(self, session: CallSession) -> None:
        generation_id = session.playback_generation
        if generation_id <= 0:
            return
        session.state = CallState.INTERRUPTED
        session.speaking = False
        session.barge_in_frame_count = 0
        logger.info(
            "Assistant playback interrupted.",
            extra={
                "call_sid": session.call_sid,
                "transport": session.transport,
                "generation_id": generation_id,
                "state": session.state.value,
                "event_name": "interrupt",
            },
        )
        await self.tts_provider.cancel(session, generation_id)
        speaker_task = self._speaker_tasks.get(session.call_sid)
        if speaker_task is not None:
            speaker_task.cancel()
            with suppress(asyncio.CancelledError):
                await speaker_task
        await self._send_clear_event(session)
        session.playback_generation += 1

    async def end_call(self, call_sid: str) -> None:
        speaker_task = self._speaker_tasks.pop(call_sid, None)
        self._connections.pop(call_sid, None)
        if speaker_task is not None:
            speaker_task.cancel()
            with suppress(asyncio.CancelledError):
                await speaker_task
        session = await self.session_store.remove(call_sid)
        if session is None:
            return
        session.state = CallState.ENDED
        await self.tts_provider.cancel(session, session.playback_generation)
        await self.stt_provider.stop(session)

        artifact_path = None
        if self.artifact_writer is not None:
            artifact_path = await self.artifact_writer.write(session)
            logger.info(
                "Call artifact written.",
                extra={
                    "call_sid": session.call_sid,
                    "stream_sid": session.stream_sid,
                    "transport": session.transport,
                    "state": session.state.value,
                    "event_name": "artifact_written",
                    "artifact_path": str(artifact_path),
                },
            )
        logger.info(
            "Call finished.",
            extra={
                "call_sid": session.call_sid,
                "stream_sid": session.stream_sid,
                "transport": session.transport,
                "state": session.state.value,
                "event_name": "call_finished",
                "artifact_path": str(artifact_path) if artifact_path else "",
            },
        )

    async def _speak_reply(
        self,
        session: CallSession,
        websocket: WebSocket,
        reply: str,
        generation_id: int,
    ) -> None:
        try:
            session.state = CallState.SPEAKING
            session.speaking = True
            session.barge_in_frame_count = 0
            session.mark("tts_requested")
            logger.info(
                "Assistant playback started.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "generation_id": generation_id,
                    "state": session.state.value,
                    "event_name": "tts_start",
                },
            )
            sent_first_chunk = False
            async for chunk in self.tts_provider.synthesize_stream(session, reply, generation_id):
                if generation_id != session.playback_generation:
                    break
                if not sent_first_chunk:
                    session.mark("first_tts_chunk")
                    sent_first_chunk = True
                await self._send_media_event(session, chunk)
                if "first_audio_out" not in session.timestamps:
                    session.mark("first_audio_out")
            await self._send_mark_event(session, generation_id)
            session.speaking = False
            session.barge_in_frame_count = 0
            session.state = CallState.LISTENING
            tts_provider_first_byte_ms = _elapsed_ms(
                session.timestamps,
                "tts_requested",
                "tts_provider_first_byte",
            )
            tts_first_audio_out_ms = _elapsed_ms(
                session.timestamps,
                "tts_requested",
                "first_audio_out",
            )
            logger.info(
                "Assistant playback finished.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "generation_id": generation_id,
                    "state": session.state.value,
                    "event_name": "tts_finished",
                    "tts_provider_first_byte_ms": tts_provider_first_byte_ms,
                    "tts_first_audio_out_ms": tts_first_audio_out_ms,
                },
            )
            await self._send_metrics_event(session, generation_id)
        except Exception:
            session.speaking = False
            session.state = CallState.ERROR
            logger.exception(
                "Assistant playback failed.",
                extra={
                    "call_sid": session.call_sid,
                    "transport": session.transport,
                    "generation_id": generation_id,
                    "state": session.state.value,
                    "event_name": "tts_error",
                },
            )
            raise

    def _should_barge_in(self, session: CallSession, audio: bytes) -> bool:
        is_voiced = detect_voice_activity(
            audio,
            threshold=self.barge_in_rms_threshold,
        )
        if is_voiced:
            session.barge_in_frame_count += 1
        else:
            session.barge_in_frame_count = 0
        return session.barge_in_frame_count >= self.barge_in_consecutive_frames

    async def _resolve_call_sid(self, stream_sid: str | None) -> str | None:
        if not stream_sid:
            return None
        for session in list(self.session_store._sessions.values()):
            if session.stream_sid == stream_sid:
                return session.call_sid
        return None

    async def _send_media_event(self, session: CallSession, chunk: bytes) -> None:
        connection = self._connections.get(session.call_sid)
        if connection is None:
            return
        encoded = encode_mulaw(chunk)
        if connection.transport == "twilio":
            payload = {
                "event": "media",
                "streamSid": session.stream_sid,
                "media": {"payload": encoded},
            }
        else:
            payload = {
                "event": "media",
                "callSid": session.call_sid,
                "media": {"payload": encoded},
            }
        await connection.websocket.send_text(json.dumps(payload))

    async def _send_mark_event(self, session: CallSession, generation_id: int) -> None:
        connection = self._connections.get(session.call_sid)
        if connection is None:
            return
        if connection.transport == "twilio":
            payload = {
                "event": "mark",
                "streamSid": session.stream_sid,
                "mark": {"name": f"assistant-turn-{generation_id}"},
            }
        else:
            payload = {
                "event": "mark",
                "callSid": session.call_sid,
                "mark": {"name": f"assistant-turn-{generation_id}"},
            }
        await connection.websocket.send_text(json.dumps(payload))

    async def _send_clear_event(self, session: CallSession) -> None:
        connection = self._connections.get(session.call_sid)
        if connection is None:
            return
        payload = {"event": "clear"}
        if connection.transport == "twilio":
            if session.stream_sid is None:
                return
            payload["streamSid"] = session.stream_sid
        else:
            payload["callSid"] = session.call_sid
        await connection.websocket.send_text(json.dumps(payload))

    async def _send_transcript_event(
        self,
        session: CallSession,
        event: TranscriptEvent,
    ) -> None:
        connection = self._connections.get(session.call_sid)
        if connection is None:
            return
        payload = {
            "event": "transcript",
            "callSid": session.call_sid,
            "transcript": {
                "speaker": event.speaker,
                "text": event.text,
                "isFinal": event.is_final,
                "startMs": event.start_ms,
                "endMs": event.end_ms,
                "providerLatencyMs": event.provider_latency_ms,
            },
        }
        if connection.transport == "twilio" and session.stream_sid is not None:
            payload["streamSid"] = session.stream_sid
        await connection.websocket.send_text(json.dumps(payload))

    async def _send_metrics_event(
        self,
        session: CallSession,
        generation_id: int,
    ) -> None:
        connection = self._connections.get(session.call_sid)
        if connection is None:
            return
        payload = {
            "event": "metrics",
            "callSid": session.call_sid,
            "metrics": {
                "generationId": generation_id,
                "sttFirstInterimMs": _elapsed_ms(session.timestamps, "audio_in", "first_stt_interim"),
                "llmFirstTokenMs": _elapsed_ms(session.timestamps, "end_of_user_utterance", "first_llm_token"),
                "ttsProviderFirstByteMs": _elapsed_ms(session.timestamps, "tts_requested", "tts_provider_first_byte"),
                "ttsFirstChunkMs": _elapsed_ms(session.timestamps, "tts_requested", "first_tts_chunk"),
                "ttsFirstAudioOutMs": _elapsed_ms(session.timestamps, "tts_requested", "first_audio_out"),
                "ttsStreamCompleteMs": _elapsed_ms(session.timestamps, "tts_requested", "tts_provider_stream_complete"),
            },
        }
        if connection.transport == "twilio" and session.stream_sid is not None:
            payload["streamSid"] = session.stream_sid
        await connection.websocket.send_text(json.dumps(payload))


def _elapsed_ms(
    timestamps: dict[str, float],
    start_key: str,
    end_key: str,
) -> float | None:
    start = timestamps.get(start_key)
    end = timestamps.get(end_key)
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 2)
