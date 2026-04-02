# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (requires .env with all API keys)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Expose locally to Twilio (requires ngrok)
ngrok http 8000
# Then set Twilio webhook: POST https://<ngrok-url>/incoming-call

# Local microphone test (no Twilio needed)
python scripts/run_local_mic.py
```

## Architecture

```
Twilio call → POST /incoming-call → TwiML response (points to /media-stream WS)
           → WebSocket /media-stream → ConversationSession
                                             │
                                    DeepgramSTT (streaming WS, mulaw 8kHz)
                                             │ utterance_end
                                    OpenAIAgent (GPT-4o, streaming, conversation history)
                                             │ text
                                    ElevenLabsTTS (PCM 22050Hz → mulaw 8kHz)
                                             │ audio bytes
                                    back to Twilio WebSocket → caller hears response
```

**One `ConversationSession` per call.** Created in `app/main.py:media_stream()` and torn down when the WebSocket closes.

## Key files

| File | Purpose |
|---|---|
| [app/main.py](app/main.py) | FastAPI app: `/incoming-call` (TwiML) + `/media-stream` (WebSocket) |
| [app/agent.py](app/agent.py) | `ConversationSession` — orchestrates STT→LLM→TTS per call |
| [app/stt/deepgram_client.py](app/stt/deepgram_client.py) | Deepgram streaming STT; `utterance_end` triggers LLM turn |
| [app/llm/openai_client.py](app/llm/openai_client.py) | `OpenAIAgent` — GPT-4o streaming with conversation history |
| [app/tts/elevenlabs_client.py](app/tts/elevenlabs_client.py) | ElevenLabs TTS → μ-law conversion via `audioop` |
| [app/telephony/twilio_handler.py](app/telephony/twilio_handler.py) | TwiML generation, Twilio WS message parsing/encoding |
| [app/config.py](app/config.py) | `Settings` (pydantic-settings, reads from `.env`) |
| [scripts/run_local_mic.py](scripts/run_local_mic.py) | Local mic test: PyAudio → Deepgram → GPT-4o → ElevenLabs → speakers |

## Audio format

- Twilio sends/receives **μ-law, 8 kHz, mono, base64-encoded** inside JSON WebSocket messages.
- ElevenLabs returns **PCM 16-bit, 22050 Hz**; `_pcm_to_mulaw()` in `elevenlabs_client.py` downsamples and encodes via `audioop`.
- Audio is sent back to Twilio in **160-byte chunks** (20 ms frames) with a 20 ms sleep to pace at real-time.

## Turn detection & interruption

- Deepgram's `utterance_end_ms=1000` fires after 1 s of silence → triggers `_on_utterance_end()` in `ConversationSession`.
- If new `is_final` speech arrives while the agent is speaking, `_interrupt()` cancels the TTS `asyncio.Task` and sends a Twilio `clear` event.

## Environment variables

Copy `.env.example` to `.env` and fill in all keys before running. Required: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`.
