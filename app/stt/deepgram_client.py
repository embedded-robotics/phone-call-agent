"""
Deepgram streaming STT client — compatible with deepgram-sdk v6.

SDK v6 is synchronous: it uses a blocking context manager and start_listening()
that loops over WebSocket frames. We run the connection in a daemon thread and
bridge transcript/utterance_end events back to the asyncio event loop via
loop.call_soon_threadsafe(), so agent.py callbacks can safely call
asyncio.create_task().

Public API (all coroutines):
  connect()            — start the background thread and wait for connection
  send_audio(bytes)    — forward μ-law audio to Deepgram
  close()              — send CloseStream and join the background thread
"""

import asyncio
import importlib
import logging
import threading
from collections.abc import Callable
from deepgram import DeepgramClient

# Imported at runtime — these submodules exist in deepgram-sdk v6 but may not
# be resolvable by static type checkers if the env is not configured in the IDE.
_events = importlib.import_module("deepgram.core.events")
_v1_types = importlib.import_module("deepgram.listen.v1.types")
EventType = _events.EventType
ListenV1Results = _v1_types.ListenV1Results
ListenV1UtteranceEnd = _v1_types.ListenV1UtteranceEnd

logger = logging.getLogger(__name__)


class DeepgramSTT:
    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str, bool], None],
        on_utterance_end: Callable[[], None],
        utterance_end_ms: int = 1000,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._utterance_end_ms = utterance_end_ms

        self._client = DeepgramClient(api_key=api_key)
        self._socket = None          # V1SocketClient set inside the thread
        self._loop = None            # asyncio event loop, captured on connect()
        self._thread = None
        self._ready = threading.Event()   # set when socket is open
        self._error = None               # captures thread startup errors

    # ------------------------------------------------------------------ #
    # Public async API                                                     #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait up to 5 s for the connection to be established
        connected = await asyncio.to_thread(self._ready.wait, 5.0)
        if not connected or self._error:
            raise RuntimeError(
                f"Deepgram connection failed: {self._error or 'timeout'}"
            )
        logger.info("Deepgram STT connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._socket:
            await asyncio.to_thread(self._socket.send_media, chunk)

    async def close(self) -> None:
        if self._socket:
            try:
                await asyncio.to_thread(self._socket.send_close_stream)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, 3.0)
        self._socket = None
        logger.info("Deepgram STT closed")

    # ------------------------------------------------------------------ #
    # Background thread                                                    #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Runs in a daemon thread; owns the blocking Deepgram WS connection."""
        try:
            with self._client.listen.v1.connect(
                model="nova-2",
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
                punctuate="true",
                interim_results="true",
                utterance_end_ms=self._utterance_end_ms,
                vad_events="true",
            ) as ws:
                self._socket = ws
                ws.on(EventType.MESSAGE, self._handle_message)
                ws.on(EventType.ERROR, self._handle_error)
                self._ready.set()      # signal connect() that we're up
                ws.start_listening()   # blocks until connection closes
        except Exception as exc:
            self._error = exc
            self._ready.set()          # unblock connect() with an error

    # ------------------------------------------------------------------ #
    # Callbacks (called from the background thread)                       #
    # ------------------------------------------------------------------ #

    def _handle_message(self, msg) -> None:
        if isinstance(msg, ListenV1Results):
            try:
                text = msg.channel.alternatives[0].transcript.strip()
                if text:
                    # Schedule callback on the event loop thread
                    self._loop.call_soon_threadsafe(
                        self._on_transcript, text, msg.is_final
                    )
            except (AttributeError, IndexError):
                pass
        elif isinstance(msg, ListenV1UtteranceEnd):
            self._loop.call_soon_threadsafe(self._on_utterance_end)

    def _handle_error(self, error) -> None:
        logger.error("Deepgram error: %s", error)
