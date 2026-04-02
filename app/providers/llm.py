from __future__ import annotations

import httpx

from app.config import Settings
from app.models import CallSession
from app.providers.base import LlmProvider


class OpenAiChatProvider(LlmProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.openai_timeout_seconds)

    async def generate_reply(self, session: CallSession, user_text: str) -> str:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI replies.")
        messages = _build_messages(
            self.settings.default_system_prompt,
            session,
            user_text,
            self.settings.llm_history_events,
        )
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "temperature": 0.2,
                "max_completion_tokens": self.settings.openai_max_completion_tokens,
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text or "Could you repeat that for me?"


class RuleBasedPhoneAgent(LlmProvider):
    async def generate_reply(self, session: CallSession, user_text: str) -> str:
        lowered = user_text.lower()
        if any(word in lowered for word in ["agent", "human", "representative"]):
            return "I can connect you to a human agent. Please hold while I arrange that."
        if "hours" in lowered:
            return "Our standard hours are nine a m to five p m, Monday through Friday."
        if "status" in lowered or "order" in lowered:
            return "I can help with status checks. Please share your order or reference number."
        if "hello" in lowered or "hi" in lowered:
            return "Hello. How can I help you today?"
        if "thank" in lowered:
            return "You are welcome. Is there anything else I can help with?"
        return (
            "I heard you say "
            f"{user_text[:120]}. "
            "Could you share one more detail so I can help accurately?"
        )


def _build_messages(
    system_prompt: str,
    session: CallSession,
    user_text: str,
    history_events: int,
) -> list[dict[str, str]]:
    concise_system_prompt = (
        f"{system_prompt} "
        "Reply in one or two short sentences. Prefer under 30 words unless clarification is required."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": concise_system_prompt}]
    history = session.transcript_buffer[-history_events:]
    for event in history:
        role = "assistant" if event.speaker == "assistant" else "user"
        messages.append({"role": role, "content": event.text})
    messages.append({"role": "user", "content": user_text})
    return messages
