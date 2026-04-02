# Phone Conversation Agent

Low-latency inbound phone voice agent built around `Twilio Media Streams`, `Deepgram` for speech, and a pluggable LLM layer. The service is designed so a human can call a phone number, speak naturally, interrupt the assistant mid-reply, and receive short spoken responses with low turnaround time.

## Overview

This project implements a production-minded scaffold for a phone-call AI agent with these goals:

- accept inbound phone calls from Twilio
- stream caller audio to the backend in real time
- transcribe the caller while they are speaking
- generate a concise assistant reply
- synthesize that reply into phone-compatible audio
- stream the audio back into the same live call
- stop speaking quickly when the caller barges in

The current implementation is intentionally modular:

- telephony is handled through Twilio
- speech-to-text and text-to-speech are provider abstractions
- the language model is also abstracted behind a provider interface
- call/session state is maintained separately from the transport layer

That structure makes it easier to benchmark providers, change prompt behavior, and add tools or business logic later.

## Architecture

### High-Level Flow

1. A caller dials a Twilio phone number.
2. Twilio sends an HTTP webhook to `POST /twilio/voice/inbound`.
3. The app responds with TwiML instructing Twilio to open a bidirectional media stream to `WS /twilio/media`.
4. Twilio starts sending `mulaw/8000 Hz` audio frames over WebSocket.
5. The backend forwards those audio frames to the streaming STT provider.
6. Interim and final transcript events are passed into the voice-agent orchestrator.
7. The orchestrator decides when the caller has finished enough of an utterance to respond.
8. The LLM provider generates a short reply.
9. The TTS provider converts that reply into `mulaw/8000 Hz` audio.
10. The backend streams synthesized audio frames back to Twilio.
11. If the caller starts speaking while TTS is playing, the service cancels the current generation and sends a Twilio `clear` event to flush buffered outbound audio.

### Main Runtime Components

#### 1. Telephony Ingress

Implemented in [app/main.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/main.py).

- `POST /twilio/voice/inbound`
  - receives the inbound voice webhook
  - creates or updates the session shell if `CallSid` is present
  - returns TwiML that tells Twilio to connect the call to the media WebSocket
- `WS /twilio/media`
  - receives Twilio stream lifecycle events such as `connected`, `start`, `media`, `mark`, and `stop`
  - forwards incoming media into the conversation pipeline
  - streams synthesized media back to Twilio

#### 2. Session Store

Implemented in [app/services/session_store.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/services/session_store.py).

Each live call gets a `CallSession` object with the data needed to manage turn-taking and cleanup:

- `call_sid`
- `stream_sid`
- `caller_id`
- `state`
- `transcript_buffer`
- `assistant_turn_id`
- `playback_generation`
- latency timestamps
- temporary tool context
- partial user transcript state

#### 3. Voice Agent Orchestrator

Implemented in [app/services/voice_agent.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/services/voice_agent.py).

This is the center of the realtime behavior:

- starts provider streams when Twilio sends the `start` event
- forwards incoming audio to STT
- records transcript events
- handles end-of-utterance behavior
- calls the LLM provider
- starts a TTS stream for the assistant reply
- tracks assistant generations so stale audio is not played
- supports barge-in by cancelling TTS and clearing Twilio playback
- tears everything down when the call ends

#### 4. Provider Abstractions

Interfaces are defined in [app/providers/base.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/providers/base.py).

Current implementations:

- STT: [app/providers/deepgram.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/providers/deepgram.py)
- TTS: [app/providers/deepgram.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/providers/deepgram.py)
- LLM: [app/providers/llm.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/providers/llm.py)
- Local testing mocks: [app/providers/mock.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/providers/mock.py)

The service uses:

- `DeepgramSpeechToTextProvider` for streaming STT
- `DeepgramTextToSpeechProvider` for TTS audio generation
- `OpenAiChatProvider` when `OPENAI_API_KEY` is configured
- `RuleBasedPhoneAgent` as a safe fallback when no LLM API key is configured

### Call States

The call state enum is defined in [app/models.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/models.py).

The session moves through:

- `listening`
- `thinking`
- `speaking`
- `interrupted`
- `error`
- `ended`

These states make the behavior easier to debug and are useful later for metrics, dashboards, and human handoff logic.

## Repository Layout

```text
app/
  main.py                  FastAPI entrypoint
  config.py                environment-driven settings
  models.py                call/session models and call states
  providers/
    base.py                provider interfaces
    deepgram.py            Deepgram STT/TTS implementations
    llm.py                 OpenAI + fallback rule-based agent
    mock.py                test-time providers
  services/
    session_store.py       in-memory live call store
    twilio_media.py        Twilio media helpers and VAD helper
    voice_agent.py         realtime orchestration logic
tests/
  test_app.py              core integration-style tests
.env.example               sample runtime configuration
pyproject.toml             dependencies and pytest config
```

## Environment Variables

Copy `.env.example` to `.env` before running.

### Required For A Live Call

- `PUBLIC_BASE_URL`
  - public HTTPS URL that Twilio or your microphone client can reach
  - example: `https://my-agent.ngrok-free.app`
- `DEEPGRAM_API_KEY`
  - required for live STT and TTS

### Recommended

- `OPENAI_API_KEY`
  - enables model-generated replies
  - if omitted, the app falls back to a simple rule-based phone agent

### Important Settings

- `TRANSPORT_MODE`
  - supported values: `twilio`, `microphone`
  - default: `twilio`
  - use `twilio` for PSTN phone calls through Twilio
  - use `microphone` for a direct microphone client over WebSocket
- `TWILIO_STREAM_PATH`
  - default: `/twilio/media`
- `MICROPHONE_STREAM_PATH`
  - default: `/microphone/stream`
- `DEFAULT_SYSTEM_PROMPT`
  - baseline system instruction for the LLM
- `DEEPGRAM_STT_MODEL`
  - default: `nova-3`
- `DEEPGRAM_TTS_MODEL`
  - default: `aura-2-thalia-en`
- `UTTERANCE_END_MS`
  - default: `700`
  - helps tune end-of-utterance responsiveness
- `OPENAI_MODEL`
  - default: `gpt-4o-mini`

## Detailed Setup

### 1. Create And Activate The Conda Environment

This has already been done once locally, but the standard commands are:

```bash
conda create -y -n phone-agent python=3.11
conda activate phone-agent
```

### 2. Install Dependencies

From the project root:

```bash
conda activate phone-agent
python -m pip install -e '.[dev]'
```

This installs:

- the FastAPI app
- runtime dependencies
- test dependencies
- the project itself in editable mode

### 3. Create The Local Environment File

```bash
cp .env.example .env
```

Then edit `.env` and provide at least:

```env
TRANSPORT_MODE=twilio
PUBLIC_BASE_URL=https://your-public-domain.example
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
```

If you omit `OPENAI_API_KEY`, the server still runs, but replies come from the fallback rule-based agent.

Choose one runtime mode:

- `TRANSPORT_MODE=twilio`
  - enables `POST /twilio/voice/inbound` and `WS /twilio/media`
  - disables microphone transport endpoints
- `TRANSPORT_MODE=microphone`
  - enables `GET /microphone/config` and `WS /microphone/stream`
  - disables Twilio transport endpoints

### 4. Start The API Server

```bash
conda activate phone-agent
uvicorn app.main:app --reload
```

By default the service runs on:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected result:

```json
{"status":"ok","environment":"development"}
```

### 5. Expose The Service Publicly

Twilio must be able to reach your app over the public internet.

For local development, use a tunnel such as `ngrok`:

```bash
ngrok http 8000
```

Take the generated HTTPS URL and update:

- `PUBLIC_BASE_URL` in `.env`
- the Twilio phone number voice webhook

If your tunnel URL changes, restart the app after updating `.env`.

### 6. Configure Twilio

In the Twilio console, configure your phone number voice settings:

- webhook method: `POST`
- webhook URL: `https://<your-public-host>/twilio/voice/inbound`

When a call arrives:

- Twilio hits the inbound webhook
- your app returns TwiML
- Twilio opens the media stream to `wss://<your-public-host>/twilio/media`

### 6B. Configure Microphone Mode

If you set `TRANSPORT_MODE=microphone`, a direct microphone client should connect to:

- `GET /microphone/config`
  - returns the active WebSocket URL and expected protocol
- `GET /microphone/test`
  - serves a built-in browser test page for quick manual testing
- `WS /microphone/stream`
  - accepts microphone audio frames

Microphone client protocol:

- client sends `start`
- client sends `media` events with base64-encoded `mulaw/8000 Hz` audio payloads
- client sends `stop` to end the session
- server responds with `ready`, `media`, `mark`, and `clear`

Example `media` message from a microphone client:

```json
{
  "event": "media",
  "media": {
    "payload": "<base64 mulaw audio>"
  }
}
```

### 6C. Test In The Browser

Once the server is running in microphone mode, open:

```text
http://127.0.0.1:8000/microphone/test
```

Then:

1. click `Connect`
2. allow microphone access in the browser
3. start speaking
4. listen for assistant audio coming back through the page

The page handles:

- microphone capture
- PCM to `mu-law / 8000 Hz` conversion
- base64 audio framing
- WebSocket streaming to the backend
- assistant audio playback in the browser

If you are exposing the app remotely over HTTPS, open the same `/microphone/test` path on the public host so the browser can access the secure WebSocket endpoint.

### 7. Place A Test Call Or Start A Microphone Session

For Twilio mode, call the Twilio number and verify:

- the call connects
- the assistant replies
- the assistant keeps replies short
- speaking over the assistant interrupts playback

For microphone mode, connect a WebSocket client and verify:

- the client receives a `ready` event
- microphone audio produces an assistant response
- the assistant audio comes back as `media` events
- speaking over the assistant triggers a `clear` event

## How The Audio Path Works

Twilio media streams use `mulaw` audio at `8000 Hz`. This project keeps that format through the realtime path to reduce unnecessary transcoding.

The main path is:

1. Twilio sends a base64 audio payload.
2. The app decodes it in [app/services/twilio_media.py](/Users/awais/Documents/Upwork Data/Phone Conversation Agent/app/services/twilio_media.py).
3. Audio frames are forwarded to Deepgram STT.
4. Transcript events trigger reply generation.
5. TTS audio is generated as `mulaw/8000 Hz`.
6. The app base64-encodes those chunks and streams them back to Twilio.

### Barge-In Behavior

When the assistant is speaking and caller speech is detected:

- the current TTS generation is cancelled
- the current speaker task is cancelled
- a Twilio `clear` event is sent
- the old assistant generation is invalidated

This prevents stale assistant audio from continuing after the human starts talking.

## Running The Tests

From the project root:

```bash
conda activate phone-agent
pytest -q
```

Current tests cover:

- TwiML generation for inbound voice webhook
- transcript-to-reply flow
- interruption / cancellation behavior

## Logs And Call Artifacts

When you run the app, logs are written to both the terminal and disk.

Default locations:

- text logs: `logs/app.log`
- structured JSON logs: `logs/app.jsonl`
- per-call transcript and timing artifacts: `logs/calls/<call_sid>.json`

You can also inspect the active paths at runtime:

```bash
curl http://127.0.0.1:8000/logs
```

Relevant environment variables:

- `LOG_DIR`
- `TEXT_LOG_FILE`
- `STRUCTURED_LOG_FILE`
- `CALL_ARTIFACT_DIR`

The per-call artifact file includes:

- call identifiers
- transport type
- final session state
- transcript history
- captured timestamps useful for latency analysis

## Current Limitations

This scaffold is a strong starting point, but there are still production upgrades you would likely want next:

- add Twilio request signature validation
- add structured metrics export
- replace the temporary RMS-based VAD helper with a more robust approach
- persist call logs and transcripts
- add human handoff integration
- add domain-specific tools or backend integrations
- support horizontal scaling with shared session/state infrastructure
- add provider retry logic and circuit breaking

## Important Notes

- The previous exploratory scripts are still in the repo for reference.
- The hard-coded Deepgram key was removed from code; runtime secrets now come from environment variables.
- Python raises a deprecation warning for `audioop`, which is currently used in the simple voice-activity helper. It works today, but should be replaced before moving to Python 3.13+.

## Quick Start Summary

If you just want the shortest path:

```bash
conda activate phone-agent
python -m pip install -e '.[dev]'
cp .env.example .env
# fill in PUBLIC_BASE_URL, DEEPGRAM_API_KEY, and optionally OPENAI_API_KEY
uvicorn app.main:app --reload
```

Then:

1. expose `localhost:8000` publicly
2. set your Twilio phone number webhook to `POST /twilio/voice/inbound`
3. call the number
