from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models import CallSession


class CallArtifactWriter:
    def __init__(self, settings: Settings) -> None:
        self.base_dir = Path(settings.call_artifact_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def write(self, session: CallSession) -> Path:
        transcript = [
            {
                "speaker": event.speaker,
                "text": event.text,
                "is_final": event.is_final,
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
                "provider_latency_ms": event.provider_latency_ms,
            }
            for event in session.transcript_buffer
        ]
        timestamps = session.timestamps
        latency_metrics = {
            "stt_first_interim_ms": _elapsed_ms(timestamps, "audio_in", "first_stt_interim"),
            "llm_first_token_ms": _elapsed_ms(timestamps, "end_of_user_utterance", "first_llm_token"),
            "tts_provider_first_byte_ms": _elapsed_ms(timestamps, "tts_requested", "tts_provider_first_byte"),
            "tts_first_chunk_ms": _elapsed_ms(timestamps, "tts_requested", "first_tts_chunk"),
            "tts_first_audio_out_ms": _elapsed_ms(timestamps, "tts_requested", "first_audio_out"),
            "tts_stream_complete_ms": _elapsed_ms(timestamps, "tts_requested", "tts_provider_stream_complete"),
        }
        payload = {
            "call_sid": session.call_sid,
            "stream_sid": session.stream_sid,
            "caller_id": session.caller_id,
            "transport": session.transport,
            "state": session.state.value,
            "assistant_turn_id": session.assistant_turn_id,
            "playback_generation": session.playback_generation,
            "timestamps": timestamps,
            "latency_metrics_ms": latency_metrics,
            "transcript": transcript,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.base_dir / f"{session.call_sid}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path


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
