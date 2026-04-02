from functools import lru_cache
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    transport_mode: Literal["twilio", "microphone"] = "twilio"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "logs"
    structured_log_file: str = "app.jsonl"
    text_log_file: str = "app.log"
    call_artifact_dir: str = "logs/calls"
    public_base_url: str = "http://localhost:8000"
    twilio_stream_path: str = "/twilio/media"
    microphone_stream_path: str = "/microphone/stream"
    default_system_prompt: str = (
        "You are a helpful phone support agent. Keep replies concise, natural, "
        "and safe for a live phone conversation."
    )
    deepgram_api_key: str | None = None
    deepgram_stt_model: str = "nova-3"
    deepgram_tts_model: str = "aura-2-thalia-en"
    utterance_end_ms: int = 1000
    barge_in_rms_threshold: int = 900
    barge_in_consecutive_frames: int = 3
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 12.0
    openai_max_completion_tokens: int = 60
    llm_history_events: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def media_stream_url(self) -> str:
        base = self.public_base_url.rstrip("/")
        path = (
            self.twilio_stream_path
            if self.transport_mode == "twilio"
            else self.microphone_stream_path
        )
        if not path.startswith("/"):
            path = f"/{path}"
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = base
        return f"{ws_base}{path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
