# Phone Call Agent

A real-time conversational AI agent for phone calls. Customers call a Twilio number, speak naturally, and receive intelligent responses — all with sub-second latency.

## How it works

```
Customer call
     │
  Twilio (WebSocket Media Stream)
     │  μ-law audio @ 8 kHz
     ▼
  FastAPI server
     │
  Deepgram STT ──► utterance_end (1 s silence)
                          │
                   OpenAI GPT-4o (streaming, conversation history)
                          │
                   ElevenLabs TTS (PCM → μ-law)
                          │
                   back to Twilio ──► caller hears response
```

1. Twilio receives an inbound call and POSTs to `/incoming-call`. The server returns TwiML that opens a WebSocket Media Stream to `/media-stream`.
2. Raw μ-law audio from the caller is forwarded to **Deepgram** for streaming transcription.
3. When Deepgram fires an `utterance_end` event (1 second of silence), the accumulated transcript is sent to **GPT-4o**.
4. The LLM reply is synthesized by **ElevenLabs** and streamed back to the caller in 20 ms audio chunks.
5. If the caller speaks while the agent is talking, the TTS is cancelled immediately and the agent responds to the new input.

---

## Prerequisites

- Python 3.11+
- API keys for:
  - [Deepgram](https://console.deepgram.com/) — speech-to-text
  - [OpenAI](https://platform.openai.com/) — GPT-4o
  - [ElevenLabs](https://elevenlabs.io/) — text-to-speech (note your Voice ID)
  - [Twilio](https://console.twilio.com/) — phone number with Voice capability
- [ngrok](https://ngrok.com/) (for local development with Twilio)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`pyaudio` is only needed for the local microphone test script. If you skip it, everything else still works.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...          # found in ElevenLabs dashboard
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
SYSTEM_PROMPT=You are a helpful customer service agent. Be concise and friendly. Keep responses short since this is a phone call.
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Expose to Twilio via ngrok

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL (e.g. `https://abc123.ngrok.io`) and set it as the Voice webhook for your Twilio number:

- **URL:** `https://abc123.ngrok.io/incoming-call`
- **Method:** HTTP POST

### 5. Call your Twilio number

The agent will greet you and you can have a full conversation.

---

## Local microphone test (no Twilio needed)

Test the full STT → LLM → TTS pipeline directly from your computer's microphone:

```bash
python scripts/run_local_mic.py
```

Speak into your microphone. Transcripts, LLM replies, and audio playback all happen locally. Press `Ctrl+C` to stop.

---

## Project structure

```
phone-call-agent/
├── app/
│   ├── main.py                     # FastAPI app — HTTP and WebSocket routes
│   ├── config.py                   # Settings loaded from .env
│   ├── agent.py                    # ConversationSession — per-call orchestrator
│   ├── stt/
│   │   └── deepgram_client.py      # Deepgram streaming WebSocket client
│   ├── llm/
│   │   └── openai_client.py        # GPT-4o streaming chat with history
│   ├── tts/
│   │   └── elevenlabs_client.py    # ElevenLabs TTS → μ-law conversion
│   └── telephony/
│       └── twilio_handler.py       # TwiML generation + Twilio WS message helpers
├── scripts/
│   └── run_local_mic.py            # Standalone local mic test
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

---

## Architecture details

### ConversationSession (`app/agent.py`)

One session is created per call and destroyed when the call ends. It owns:

- A **Deepgram** connection (STT)
- An **OpenAIAgent** instance with its full conversation history
- An **ElevenLabsTTS** instance

Turn flow:

```
feed_audio(chunk)
  └─► Deepgram WS
         ├─► on_transcript(text, is_final=True)  → buffer transcript
         └─► on_utterance_end()                  → flush buffer → LLM → TTS → send audio
```

Interruption flow:

```
on_transcript(is_final=True) while agent_speaking=True
  └─► cancel asyncio TTS task
  └─► send Twilio "clear" event (flushes Twilio's audio buffer)
  └─► respond to new input
```

### Audio pipeline

| Stage | Format |
|---|---|
| Twilio inbound | μ-law, 8 kHz, mono, base64 |
| Deepgram input | μ-law, 8 kHz, mono, raw bytes |
| ElevenLabs output | PCM 16-bit, 22050 Hz |
| After conversion | μ-law, 8 kHz (via `audioop.ratecv` + `audioop.lin2ulaw`) |
| Twilio outbound | μ-law, 8 kHz, mono, base64, 160-byte chunks (20 ms frames) |

### Latency budget

| Component | Typical latency |
|---|---|
| Deepgram STT | ~300 ms |
| GPT-4o first token | ~400 ms |
| ElevenLabs TTS | ~300 ms |
| **Total to first audio** | **~1 s** |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | — | Deepgram API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model ID |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | — | ElevenLabs voice to use |
| `TWILIO_ACCOUNT_SID` | — | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | — | Twilio auth token |
| `SYSTEM_PROMPT` | (see .env.example) | Agent persona / instructions |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
