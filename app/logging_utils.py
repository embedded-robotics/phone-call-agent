from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "call_sid",
            "stream_sid",
            "transport",
            "event_name",
            "state",
            "generation_id",
            "artifact_path",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(settings: Settings) -> None:
    log_dir = Path(settings.log_dir)
    call_dir = Path(settings.call_artifact_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    call_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root.handlers.clear()

    text_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    json_formatter = JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(text_formatter)

    text_file = RotatingFileHandler(
        log_dir / settings.text_log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    text_file.setFormatter(text_formatter)

    json_file = RotatingFileHandler(
        log_dir / settings.structured_log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    json_file.setFormatter(json_formatter)

    root.addHandler(console)
    root.addHandler(text_file)
    root.addHandler(json_file)
