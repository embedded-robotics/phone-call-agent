"""
OpenAI GPT-4o streaming chat client.

Maintains conversation history across turns for a single call session.
Each call to `respond()` appends the user message, streams the assistant
reply, and appends the completed reply to history before returning.
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIAgent:
    def __init__(self, api_key: str, model: str, system_prompt: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]

    async def respond(self, user_text: str) -> AsyncIterator[str]:
        """
        Append user_text to history, stream GPT reply, yield text chunks.
        The completed reply is appended to history after streaming finishes.
        """
        self._messages.append({"role": "user", "content": user_text})
        full_reply = []

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply.append(delta)
                yield delta

        assistant_text = "".join(full_reply)
        if assistant_text:
            self._messages.append({"role": "assistant", "content": assistant_text})
        logger.info("OpenAI reply: %s", assistant_text)

    def reset(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self._messages = [self._messages[0]]
