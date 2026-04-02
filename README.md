# Phone Call Agent

A real-time conversational AI agent that works both via a browser UI and Twilio phone calls. Customers speak naturally and receive intelligent responses with sub-second latency.

## How it works

```
Browser (WebRTC AEC mic)          Twilio phone call
        │ PCM 16kHz                       │ μ-law 8kHz
        │ WebSocket /ws                   │ WebSocket /media-stream
        └──────────────┬──────────────────┘
                       ▼
              ConversationSession
                       │
              Deepgram STT ──► utterance_end (1s silence)
                                      │
                               OpenAI GPT-4o (streaming)
                                      │ sentence by sentence
                               ElevenLabs TTS (ulaw_8000)
                                      │
        Browser AudioContext     Twilio caller hears response
```

**Sentence-level pipelining** — GPT tokens are split at `.?!` boundaries. TTS synthesis of each sentence starts immediately, overlapping with generation of the next. First audio reaches the user ~500ms after they stop speaking.

**Echo cancellation** — the browser UI uses WebRTC's built-in AEC (`echoCancellation: true`), so the agent never hears itself through the mic.

---

## Prerequisites

- Python 3.11
- API keys for [Deepgram](https://console.deepgram.com/), [OpenAI](https://platform.openai.com/), [ElevenLabs](https://elevenlabs.io/)
- (For Twilio calls) A Twilio account with a Voice-enabled phone number and [ngrok](https://ngrok.com/)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
SYSTEM_PROMPT=You are a helpful customer service agent. Be concise and friendly.
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the browser UI

```
http://localhost:8000
```

Click **START**, allow microphone access, and start talking. The agent will greet you and you can have a full conversation. No Twilio setup needed for this.

---

## Twilio phone call setup

```bash
ngrok http 8000
```

Set your Twilio number's Voice webhook:
- **URL:** `https://<ngrok-url>/incoming-call`
- **Method:** HTTP POST

Call the number — the same agent handles the call.

---

## Project structure

```
phone-call-agent/
├── app/
│   ├── main.py                     # FastAPI: GET /, WS /ws, POST /incoming-call, WS /media-stream
│   ├── config.py                   # Settings from .env
│   ├── agent.py                    # ConversationSession — sentence-pipelined STT→LLM→TTS
│   ├── stt/
│   │   └── deepgram_client.py      # Deepgram v6 streaming (daemon thread + asyncio bridge)
│   ├── llm/
│   │   └── openai_client.py        # GPT-4o streaming with conversation history
│   ├── tts/
│   │   └── elevenlabs_client.py    # ElevenLabs → native μ-law 8kHz
│   └── telephony/
│       └── twilio_handler.py       # TwiML + Twilio WebSocket message helpers
├── static/
│   └── index.html                  # Browser UI: WebRTC mic → WebSocket → AudioContext
├── requirements.txt
└── .env.example
```

---

## Architecture details

### ConversationSession (`app/agent.py`)

One session per connection (browser tab or Twilio call). Per turn, three asyncio tasks run concurrently:

```
_llm_producer      streams GPT tokens, splits at sentence boundaries → sentence_queue
_tts_synthesizer   dequeues sentences, synthesizes immediately       → audio_queue
_audio_player      dequeues blobs, sends to client
```

Interruption: if the user speaks while the agent is talking, the pipeline task is cancelled and Twilio's buffer is cleared via the `clear` event.

### Audio formats

| Stage | Format |
|---|---|
| Browser mic | PCM 16-bit 16kHz (WebRTC) |
| Server → Deepgram | μ-law 8kHz (downsampled in `main.py`) |
| ElevenLabs output | μ-law 8kHz natively (`?output_format=ulaw_8000`) |
| Server → browser | μ-law binary WebSocket frames (browser decodes with G.711 table) |
| Server → Twilio | μ-law 8kHz, 160-byte chunks, base64 JSON, paced at 20ms intervals |

### Latency budget

| Component | Typical |
|---|---|
| Deepgram STT | ~300ms |
| GPT-4o first sentence | ~300ms |
| ElevenLabs TTS | ~300ms |
| **Total to first audio** | **~900ms** |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | — | Deepgram API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model ID |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | — | Voice ID from ElevenLabs dashboard |
| `TWILIO_ACCOUNT_SID` | — | Twilio account SID (Twilio calls only) |
| `TWILIO_AUTH_TOKEN` | — | Twilio auth token (Twilio calls only) |
| `SYSTEM_PROMPT` | (see .env.example) | Agent persona and instructions |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
