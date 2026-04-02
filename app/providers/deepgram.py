from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from time import monotonic

import httpx
import websockets

from app.config import Settings
from app.models import CallSession, TranscriptEvent
from app.providers.base import SpeechToTextProvider, TextToSpeechProvider


class DeepgramSpeechToTextProvider(SpeechToTextProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._receivers: dict[str, asyncio.Task[None]] = {}

    async def start_stream(
        self,
        session: CallSession,
        on_transcript: Callable[[TranscriptEvent], object],
    ) -> None:
        if not self.settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for live STT.")
        if session.call_sid in self._connections:
            return

        query = (
            f"model={self.settings.deepgram_stt_model}"
            f"&encoding=mulaw"
            f"&sample_rate=8000"
            f"&channels=1"
            f"&interim_results=true"
            f"&endpointing=300"
            f"&utterance_end_ms={self.settings.utterance_end_ms}"
            f"&smart_format=true"
        )
        ws = await websockets.connect(
            f"wss://api.deepgram.com/v1/listen?{query}",
            additional_headers={"Authorization": f"Token {self.settings.deepgram_api_key}"},
            max_queue=None,
        )
        self._connections[session.call_sid] = ws
        self._receivers[session.call_sid] = asyncio.create_task(
            self._receive_loop(session, ws, on_transcript)
        )
        session.stt_started = True

    async def send_audio(self, session: CallSession, chunk: bytes) -> None:
        ws = self._connections.get(session.call_sid)
        if ws is None:
            return
        await ws.send(chunk)

    async def stop(self, session: CallSession) -> None:
        ws = self._connections.pop(session.call_sid, None)
        receiver = self._receivers.pop(session.call_sid, None)
        if ws is not None:
            with suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
            with suppress(Exception):
                await ws.close()
        if receiver is not None:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver

    async def _receive_loop(
        self,
        session: CallSession,
        ws: websockets.WebSocketClientProtocol,
        on_transcript: Callable[[TranscriptEvent], object],
    ) -> None:
        async for raw_message in ws:
            payload = json.loads(raw_message)
            if payload.get("type") == "UtteranceEnd":
                text = session.pending_user_text.strip()
                if text:
                    await _maybe_await(
                        on_transcript(
                            TranscriptEvent(
                                speaker="user",
                                text=text,
                                is_final=True,
                            )
                        )
                    )
                continue
            channel = payload.get("channel") or {}
            alternatives = channel.get("alternatives") or []
            if not alternatives:
                continue
            transcript = (alternatives[0].get("transcript") or "").strip()
            if not transcript:
                continue
            started = payload.get("start")
            duration = payload.get("duration")
            provider_latency_ms = None
            if duration is not None:
                provider_latency_ms = max(0.0, (monotonic() - session.timestamps.get("audio_in", monotonic())) * 1000)
            event = TranscriptEvent(
                speaker="user",
                text=transcript,
                is_final=bool(payload.get("is_final")),
                start_ms=int(started * 1000) if started is not None else None,
                end_ms=int((started + duration) * 1000) if started is not None and duration is not None else None,
                provider_latency_ms=provider_latency_ms,
            )
            await _maybe_await(on_transcript(event))


class DeepgramTextToSpeechProvider(TextToSpeechProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cancelled: set[tuple[str, int]] = set()
        self._client = httpx.AsyncClient(timeout=20.0)

    async def synthesize_stream(
        self,
        session: CallSession,
        text: str,
        generation_id: int,
    ) -> AsyncIterator[bytes]:
        if not self.settings.deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for live TTS.")

        url = (
            "https://api.deepgram.com/v1/speak"
            f"?model={self.settings.deepgram_tts_model}"
            "&encoding=mulaw&sample_rate=8000&container=none"
        )
        frame_size = 160
        buffer = bytearray()
        session.timestamps.pop("tts_provider_first_byte", None)
        session.timestamps.pop("tts_provider_stream_complete", None)
        first_byte_recorded = False
        async with self._client.stream(
            "POST",
            url,
            headers={"Authorization": f"Token {self.settings.deepgram_api_key}"},
            json={"text": text},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if (session.call_sid, generation_id) in self._cancelled:
                    break
                if not chunk:
                    continue
                if not first_byte_recorded:
                    session.timestamps["tts_provider_first_byte"] = monotonic()
                    first_byte_recorded = True
                buffer.extend(chunk)
                while len(buffer) >= frame_size:
                    if (session.call_sid, generation_id) in self._cancelled:
                        break
                    yield bytes(buffer[:frame_size])
                    del buffer[:frame_size]
                    await asyncio.sleep(0.02)

        if buffer and (session.call_sid, generation_id) not in self._cancelled:
            yield bytes(buffer)
        session.timestamps["tts_provider_stream_complete"] = monotonic()
        self._cancelled.discard((session.call_sid, generation_id))

    async def cancel(self, session: CallSession, generation_id: int) -> None:
        self._cancelled.add((session.call_sid, generation_id))


async def _maybe_await(value: object) -> None:
    if asyncio.iscoroutine(value):
        await value
