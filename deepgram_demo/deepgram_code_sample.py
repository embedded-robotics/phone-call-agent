import os
import httpx
import threading

from deepgram import DeepgramClient
from deepgram.core.events import EventType

# Note: This is an English stream, update accordingly for other languages
STREAM_URL = "https://playerservices.streamtheworld.com/api/livestream-redirect/CSPANRADIOAAC.aac"

client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])

with client.listen.v1.connect(
    model="nova-3",
    language="en",
) as connection:
    ready = threading.Event()

    def on_message(result):
        event_type = getattr(result, "type", None)
        channel = getattr(result, "channel", None)
        if channel and hasattr(channel, "alternatives"):
            transcript = channel.alternatives[0].transcript
            is_final = getattr(result, "is_final", True)
            if transcript:
                print(transcript)

    connection.on(EventType.OPEN, lambda _: ready.set())
    connection.on(EventType.MESSAGE, on_message)

    def stream():
        ready.wait()
        with httpx.stream("GET", STREAM_URL, follow_redirects=True) as response:
            for chunk in response.iter_bytes():
                connection.send_media(chunk)

    threading.Thread(target=stream, daemon=True).start()

    print(f"Transcribing {STREAM_URL}...")
    connection.start_listening()
