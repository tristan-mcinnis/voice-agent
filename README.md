# Real-Time VoiceBot — Soniox STT/TTS + DeepSeek + Pipecat (local)

A multimodal local voice agent built on Pipecat. Soniox handles real-time
speech-to-text and text-to-speech; DeepSeek (`deepseek-v4-flash` with thinking
disabled) is the LLM. Audio runs over your laptop's mic and speakers — no
browser, no WebRTC.

The bot ships with a wake-word gate ("hey ava" by default), plus a rich tool
surface (clipboard, file/folder reads, Finder & browser scraping via
AppleScript, screenshot / window / region / display / webcam capture, terminal
output, web search) and a vision-describer fallback chain (local MLX
`Qwen3-VL-2B` first, Kimi/Moonshot `kimi-k2.6` fallback). All providers,
models, voices, the wake word, and the system prompt live in
`server/config.yaml` — no Python edits to swap them.

## Quick Start

```bash
cd server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
# Required: SONIOX_API_KEY, DEEPSEEK_API_KEY
# Vision:   MOONSHOT_API_KEY   (fallback after local MLX; required off Apple Silicon)
# Optional: OPENAI_API_KEY (alt vision), SERPER_API_KEY (web_search),
#           VOICE_BOT_LOG_DIR (per-session JSONL log dir)

python local_bot.py
```

Standalone smoke tests:

```bash
python tests/test_tts.py    # writes test_tts.wav (Soniox TTS WebSocket)
python tests/test_stt.py    # transcribes test_tts.wav (run test_tts.py first)
python tests/test_tools.py  # exercises every tool in tools.TOOLS
```

## Wake word

By default the agent starts **awake** (so the intro plays), then sleeps after
30 seconds of user silence. While asleep, transcripts are dropped and the LLM
is never invoked. Say **"hey ava"** to wake it; it replies "Ready." and resumes
listening normally. Say **"go to sleep"** to put it back to sleep manually.

Tune in `config.yaml`:

```yaml
wake_word:
  enabled: true
  phrase: "hey ava"
  sleep_phrase: "go to sleep"
  idle_timeout_seconds: 30
  ack_text: "Ready."
  start_awake: true
```

Set `enabled: false` to disable the gate entirely; set `start_awake: false` to
require the wake word for the very first turn.

## Configuration

Everything swap-able lives in `server/config.yaml`: LLM provider/model/extras,
STT/TTS provider + voice, the ordered `vision:` fallback chain (local MLX →
Kimi by default), wake-word settings, and the system prompt. Secrets stay in
`.env` and are referenced by name (e.g. `api_key_env: DEEPSEEK_API_KEY`). To
swap LLM provider, edit the YAML and add the matching env var to `.env` — no
Python changes needed.

## Project Structure

```
server/
├── local_bot.py        # Entry point: mic in, speakers out
├── voice_bot.py        # Shared STT+TTS+LLM+tools builder
├── local_audio.py      # Mic/speakers concerns: mute strategies, turn strategies
├── wake_word.py        # Wake-word gate (FrameProcessor)
├── tools.py            # Tool registry (clipboard, files, browser, capture, search, ...)
├── config.py           # Loads config.yaml into typed dataclasses
├── config.yaml         # Provider/model/voice/prompt/vision/wake-word config
├── mlx_vision.py       # In-process MLX vision describer (Apple-Silicon only)
├── session_log.py      # Per-session kebab-case JSONL event logger
├── tests/              # Standalone smoke tests
│   ├── test_stt.py     #   Soniox STT WebSocket test
│   ├── test_tts.py     #   Soniox TTS WebSocket test
│   └── test_tools.py   #   Tool registry smoke test
└── requirements.txt
```
