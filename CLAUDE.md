# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Open the browser UI (after server is running)
open http://localhost:8000

# Expose to Twilio via ngrok
ngrok http 8000
# Set Twilio webhook: POST https://<ngrok-url>/incoming-call
```

## Architecture

```
Browser (WebRTC AEC mic)          Twilio phone call
        │ PCM 16-bit 16kHz               │ μ-law 8kHz
        │ via WebSocket /ws              │ via WebSocket /media-stream
        └──────────────┬─────────────────┘
                       ▼
              ConversationSession
                       │
              DeepgramSTT (streaming, mulaw 8kHz)
                       │ utterance_end
              OpenAIAgent (GPT-4o, streaming, conversation history)
                       │ sentence-by-sentence
              ElevenLabsTTS (ulaw_8000 query param)
                       │ μ-law audio blob
                       ▼
        Browser plays via AudioContext   Twilio sends to caller
```

**One `ConversationSession` per connection** (browser tab or Twilio call). Created in the WebSocket handler, torn down on disconnect.

**Sentence-level pipelining**: three concurrent asyncio tasks per turn — LLM producer splits tokens at `.?!` boundaries, TTS synthesizer starts immediately per sentence, audio player streams blobs to the client. First audio reaches the user ~500ms after they stop speaking.

## Key files

| File | Purpose |
|---|---|
| [app/main.py](app/main.py) | FastAPI: `GET /`, `POST /incoming-call`, `WS /media-stream`, `WS /ws` |
| [app/agent.py](app/agent.py) | `ConversationSession` — sentence-pipelined STT→LLM→TTS orchestrator |
| [app/stt/deepgram_client.py](app/stt/deepgram_client.py) | Deepgram v6 streaming STT; runs in a daemon thread, bridges to asyncio |
| [app/llm/openai_client.py](app/llm/openai_client.py) | `OpenAIAgent` — GPT-4o streaming with conversation history |
| [app/tts/elevenlabs_client.py](app/tts/elevenlabs_client.py) | ElevenLabs TTS — `output_format=ulaw_8000` as **query param** (not body) |
| [app/telephony/twilio_handler.py](app/telephony/twilio_handler.py) | TwiML generation + Twilio WS message parsing/encoding |
| [app/config.py](app/config.py) | `Settings` via pydantic-settings, reads from `.env` |
| [static/index.html](static/index.html) | Single-page browser UI: WebRTC mic → WS → AudioContext playback |

## Audio pipeline

| Path | Format at each stage |
|---|---|
| Browser mic → server | PCM 16-bit 16kHz (WebRTC), downsampled to μ-law 8kHz in `main.py:_pcm16k_to_mulaw8k()` |
| Twilio → server | μ-law 8kHz base64 JSON, decoded in `twilio_handler.decode_audio()` |
| Server → Deepgram | μ-law 8kHz raw bytes via `DeepgramSTT.send_audio()` |
| ElevenLabs response | μ-law 8kHz natively (`?output_format=ulaw_8000` query param) |
| Server → browser | μ-law bytes as binary WebSocket frames; browser decodes with inline G.711 table |
| Server → Twilio | μ-law 8kHz in 160-byte chunks (20ms), base64 JSON, paced with `asyncio.sleep(0.02)` |

## Important gotchas

- **Deepgram SDK v6** is synchronous — `DeepgramSTT` runs the connection in a daemon thread and uses `loop.call_soon_threadsafe()` to fire callbacks into the asyncio loop. Boolean params must be passed as `"true"`/`"false"` strings, not Python `True`/`False`.
- **ElevenLabs `output_format`** must be a **query parameter** in the URL, not a JSON body field — the body field is silently ignored and MP3 is returned.
- **Echo cancellation** is handled by the browser's WebRTC AEC (`echoCancellation: true` in `getUserMedia`). The Twilio path has no echo issue since speaker and mic are on separate physical phones.

## Environment variables

Copy `.env.example` to `.env`. Required: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`.
