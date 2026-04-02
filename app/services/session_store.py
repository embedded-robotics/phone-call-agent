from __future__ import annotations

import asyncio

from app.models import CallSession


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, CallSession] = {}

    async def get_or_create(
        self,
        call_sid: str,
        *,
        stream_sid: str | None = None,
        caller_id: str | None = None,
        transport: str | None = None,
    ) -> CallSession:
        async with self._lock:
            session = self._sessions.get(call_sid)
            if session is None:
                session = CallSession(
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    caller_id=caller_id,
                    transport=transport or "twilio",
                )
                self._sessions[call_sid] = session
            else:
                if stream_sid:
                    session.stream_sid = stream_sid
                if caller_id:
                    session.caller_id = caller_id
                if transport:
                    session.transport = transport
            return session

    async def get(self, call_sid: str) -> CallSession | None:
        async with self._lock:
            return self._sessions.get(call_sid)

    async def remove(self, call_sid: str) -> CallSession | None:
        async with self._lock:
            return self._sessions.pop(call_sid, None)
