# Voice Agent Server

A local voice bot built on [Pipecat](https://github.com/pipecat-ai/pipecat) — mic → STT → LLM → TTS → speakers. No browser, no WebRTC, no cloud transport.

## Setup

```bash
cd server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

## Environment variables (`server/.env`)

| Variable | Required | Purpose |
|---|---|---|
| `SONIOX_API_KEY` | yes | STT + TTS |
| `DEEPSEEK_API_KEY` | yes | LLM (DeepSeek chat completions) |
| `MOONSHOT_API_KEY` | yes* | Kimi vision fallback (`kimi-k2.6`) |
| `OPENAI_API_KEY` | no | Only if you add an OpenAI vision chain entry |
| `SERPER_API_KEY` | no | `web_search` tool (Google via Serper) |
| `VOICE_BOT_LOG_DIR` | no | Session log directory (default `./logs`) |

\* Required on non-Apple-Silicon hosts where MLX is unavailable.

## Running

```bash
python local_bot.py          # mic/speakers voice loop
python test_stt.py           # Soniox STT WebSocket smoke test (needs test_tts.wav)
python test_tts.py           # Soniox TTS WebSocket smoke test (writes test_tts.wav)
python test_tools.py         # Smoke-tests every registered tool
```

## Configuration (`config.yaml`)

Edit `config.yaml` to swap providers, models, voice, or wake word — no Python changes needed.

- **`llm`** — any OpenAI-compatible chat completions endpoint (DeepSeek, Kimi, OpenAI)
- **`stt`** / **`tts`** — Soniox only today; add a builder in `voice_bot._build_stt/tts` for others
- **`vision`** — fallback chain; first provider that returns text wins. Default: MLX local (`mlx-community/Qwen3-VL-2B-Instruct-4bit`) → Kimi `kimi-k2.6`
- **`wake_word`** — phrase, sleep phrase, idle timeout, ack text
- **`system_prompt`** — LLM system message; use `{tool_capabilities}` to auto-inject a tool inventory

## Pipeline

```
LocalAudioTransport.input → VADProcessor → Soniox STT → EchoSuppressor →
  WakeWordGate → SessionLogProcessor → user context → DeepSeek LLM →
  Soniox TTS → LocalAudioTransport.output → assistant context
```

## Hotkey interrupt

Press **⌘⇧I** (anywhere on the desktop) to interrupt the bot mid-sentence. Override with the `HOTKEY_INTERRUPT` env var (pynput syntax, e.g. `<ctrl>+<alt>+i`). Requires macOS Accessibility permission on first run.

## Adding a tool

Create a `BaseTool` subclass in `server/tools.py` and decorate it with `@REGISTRY.register`:

```python
@REGISTRY.register
class MyTool(BaseTool):
    name = "my_tool"
    category = "system"          # groups tool in the capability summary
    speak_text = "On it."        # optional spoken filler while executing
    description = "Does a thing."
    parameters = {
        "arg": {"type": "string", "description": "The argument."},
    }
    required = ["arg"]

    def execute(self, arg: str) -> str:
        return f"did a thing with {arg}"
```

That's it — the tool is automatically wired to the LLM and appears in the capability summary injected into the system prompt.

## Session logs

Per-session JSONL logs are written to `logs/session-<ts>.jsonl` as the session runs. Vision captures are saved to `logs/captures/` as timestamped JPEGs, with `image_path` recorded in the tool-result event.
