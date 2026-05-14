# Voice Agent Server

A local voice bot built on [Pipecat](https://github.com/pipecat-ai/pipecat) — mic → STT → LLM → TTS → speakers. No browser, no WebRTC, no cloud transport.

## Setup

```bash
cd server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env            # fill in your API keys
cp config.example.yaml config.yaml  # or customise from the example
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
python local_bot.py            # mic/speakers voice loop
python tests/test_stt.py       # Soniox STT WebSocket smoke test (needs test_tts.wav)
python tests/test_tts.py       # Soniox TTS WebSocket smoke test (writes test_tts.wav)
python tests/test_tools.py     # Smoke-tests every registered tool (pass --vision for captures)
python -m pytest tests/unit/ -v  # Pure-function unit tests (48 tests, no I/O)
```

## Package layout

```
server/
  local_bot.py          # main entry point — pipeline assembly, mute/turn strategies, lifecycle
  voice_bot.py          # shared component construction (STT, TTS, LLM, context)
  config.py             # typed config dataclasses, YAML loader
  config.example.yaml   # example configuration (copy to config.yaml)
  hotkey_interrupt.py   # global hotkey (⌘⇧I) for mid-sentence interruption
  CONTEXT.md            # domain glossary

  tools/                # LLM-callable tools — registry + implementations
    registry.py         #   BaseTool, ToolRegistry, REGISTRY, handler wiring
    vision.py           #   image description fallback chain (MLX → Kimi → …)
    mlx_vision.py       #   in-process MLX vision model loader (internal adapter)
    files.py            #   file system ops (impl in BaseTool.execute())
    desktop.py          #   macOS automation + capture path helpers
    web.py              #   web search (Serper), weather demo
    memory.py           #   memory tool — add/replace/list facts to USER.md / MEMORY.md
    search_history.py   #   search past session JSONL logs for conversation recall

  processors/           # pipeline FrameProcessor stages
    echo_suppressor.py  #   drops STT frames while bot speaks (no-AEC workaround)
    wake_word.py        #   wake-word gate (asleep/awake state machine)
    session_log.py      #   per-session JSONL logger + SessionLogProcessor

  tests/                # tests
    test_stt.py         #   Soniox STT WebSocket smoke test
    test_tts.py         #   Soniox TTS WebSocket smoke test
    test_tools.py       #   tool registry smoke test
    unit/               #   pytest unit tests for pure functions (48 tests)

  connection_rendezvous.py  # dual-connection coordination for bot introduction
  docs/adr/             # architecture decision records
  experiments/aec/      # archived Speex AEC experiment (voice barge-in)
  logs/                 # session log output (gitignored)
```

## Configuration (`config.example.yaml`)

Edit `config.yaml` (copied from `config.example.yaml`) to swap providers, models, voice, or wake word — no Python changes needed.

- **`llm`** — any OpenAI-compatible chat completions endpoint (DeepSeek, Kimi, OpenAI)
- **`stt`** / **`tts`** — Soniox only today; add a builder in `voice_bot._build_stt/tts` for others
- **`vision`** — fallback chain; first provider that returns text wins. Default: MLX local (`Qwen3-VL-2B-Instruct-4bit`) → Kimi `kimi-k2.6`. Set to `[]` to disable vision. See [ADR-0002](docs/adr/0002-vision-fallback-chain.md).
- **`wake_word`** — phrase, sleep phrase, idle timeout, ack text
- **`system_prompt`** — LLM system message. The `PromptBuilder` renders the tool inventory in its own layer, so identity text shouldn't list tools manually.

Secrets stay in `.env`; `config.yaml` references them by name (`api_key_env: DEEPSEEK_API_KEY`).

## Pipeline

```
LocalAudioTransport.input → VADProcessor → Soniox STT → EchoSuppressor →
  WakeWordGate → SessionLogProcessor → user context → DeepSeek LLM →
  Soniox TTS → LocalAudioTransport.output → assistant context
```

The **EchoSuppressor** drops transcription frames while the bot's TTS is playing — without hardware AEC, the mic picks up speaker output. The bot mutes the mic during its own turns to prevent self-interruption. See [ADR-0003](docs/adr/0003-deepseek-thinking-disabled.md) for the LLM latency decision.

## Wake word

Configured in `config.yaml` under `wake_word`. While asleep, all transcription frames are dropped — the LLM is never invoked. On hearing the wake phrase (e.g. "Hey Ava"), the bot speaks an acknowledgement and begins processing normally. It returns to sleep on the sleep phrase ("Go to sleep") or after `idle_timeout_seconds` of silence.

## Hotkey interrupt

Press **⌘⇧I** (anywhere on the desktop) to interrupt the bot mid-sentence. Override with the `HOTKEY_INTERRUPT` env var (pynput syntax, e.g. `<ctrl>+<alt>+i`). Requires macOS Accessibility permission on first run. Voice barge-in (interrupting by speaking) requires hardware AEC; see `experiments/aec/`.

## Adding a tool

Create a `BaseTool` subclass with the full implementation in `execute()`. Decorate with `@REGISTRY.register` and add it to the matching domain file in `tools/`:

```python
# In tools/desktop.py (macOS tools) or tools/files.py (file ops) or tools/web.py (network)

from tools.registry import REGISTRY, BaseTool

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
        # Put the full implementation here — no separate helper function needed.
        return f"did a thing with {arg}"
```

The tool is automatically wired to the LLM and appears in the capability summary that `PromptBuilder` renders into the `# Available Tools` layer of the system prompt. `tools.my_tool(...)` resolves to `REGISTRY.get("my_tool").execute` automatically via `tools/__init__.py`'s module-level `__getattr__`.

## Cognitive-stack memory files

The agent's frozen system prompt is assembled once per session by `agent/prompt_builder.py` from layered files in `.voice-agent/` (gitignored, project-root):

```
.voice-agent/
  SOUL.md              # agent identity (falls back to config.system_prompt if missing)
  memories/
    USER.md            # user preferences (~1,375 chars soft cap)
    MEMORY.md          # project facts (~2,200 chars soft cap)
  skills/              # one dir per skill, each with a SKILL.md
```

The repo ships `.example` templates for each. **First-time setup** — copy them and edit:

```bash
cp .voice-agent/SOUL.md.example                .voice-agent/SOUL.md
cp .voice-agent/memories/USER.md.example       .voice-agent/memories/USER.md
cp .voice-agent/memories/MEMORY.md.example     .voice-agent/memories/MEMORY.md
```

The agent maintains USER.md and MEMORY.md at runtime via the `memory` tool (`add` / `replace` / `list`), but you can edit them directly any time. Snapshots are loaded once per session — mid-session edits don't re-inject until the next run, which keeps provider-side prompt caches warm.

## Session logs

Per-session JSONL logs are written to `logs/session-<ts>.jsonl` as the session runs. Vision captures are saved to `logs/captures/` as timestamped JPEGs, with `image_path` recorded in the tool-result event. Override the directory with `VOICE_BOT_LOG_DIR`.

## Architecture decisions

See [`docs/adr/`](docs/adr/) for recorded architectural decisions:
- [ADR-0001](docs/adr/0001-soniox-stt-tts.md) — Soniox for STT and TTS
- [ADR-0002](docs/adr/0002-vision-fallback-chain.md) — Vision as a fallback chain
- [ADR-0003](docs/adr/0003-deepseek-thinking-disabled.md) — DeepSeek with thinking disabled

Domain terms are defined in [`CONTEXT.md`](CONTEXT.md).
