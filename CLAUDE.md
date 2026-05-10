# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repo layout

Single Python app under `server/` — a Pipecat voice bot that runs locally on
mic + speakers (no browser, no Daily/WebRTC).

```
server/
  local_bot.py          main entry point + mute/turn/lifecycle
  voice_bot.py          shared STT/TTS/LLM/context construction
  config.py             typed config dataclasses, YAML loader
  config.example.yaml   example config (copy to config.yaml)
  hotkey_interrupt.py   global hotkey (⌘⇧I)

  tools/                LLM-callable tool implementations
    __init__.py         re-exports REGISTRY, all tool functions
    registry.py         BaseTool, ToolRegistry, REGISTRY, _make_handler
    vision.py           image description fallback chain
    mlx_vision.py       in-process MLX vision (internal adapter)
    files.py            file ops + tool classes
    desktop.py          macOS automation + tool classes
    web.py              web search, weather demo + tool classes

  processors/           pipeline FrameProcessor stages
    echo_suppressor.py  drops STT frames while bot speaks
    wake_word.py        wake-word gate (asleep/awake state machine)
    session_log.py      per-session JSONL logger + SessionLogProcessor

  docs/adr/             architecture decision records
  experiments/aec/      archived Speex AEC experiment
  CONTEXT.md            domain glossary
```

## Common commands

```bash
cd server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # first-time setup
python local_bot.py                  # mic/speakers loop
python tests/test_stt.py               # Soniox STT WebSocket smoke test (needs test_tts.wav)
python tests/test_tts.py               # Soniox TTS WebSocket smoke test (writes test_tts.wav)
python tests/test_tools.py             # Smoke-tests every tool in tools.REGISTRY
```

There is no pytest suite — `tests/test_stt.py` / `tests/test_tts.py` / `tests/test_tools.py` are
run directly as scripts.

## Architecture

### Configuration (`server/config.example.yaml`)

`config.example.yaml` is the reference config. Copy it to `config.yaml` (gitignored)
to customise. It drives `voice_bot.build_components()`, `tools.vision.describe_image()`,
and the wake-word gate.

- `llm.{provider, base_url, model, api_key_env, extra}` — any OpenAI-compatible
  chat-completions endpoint (DeepSeek, Kimi/Moonshot, OpenAI). `extra` carries
  provider-specific knobs (e.g. DeepSeek's `thinking: disabled`). See ADR-0003.
- `stt.{provider, api_key_env, language_hints}` — only `soniox` is wired up
  today. Add a builder in `voice_bot._build_stt` to support more. See ADR-0001.
- `tts.{provider, api_key_env, voice}` — same shape as STT.
- `vision` — a **list (chain)** of providers tried in order; first one to
  return text wins. Each entry has `kind: mlx | openai`. Default chain:
  `mlx-local` (in-process `mlx_vlm`, Apple-Silicon-only) → `kimi` (Moonshot
  `kimi-k2.6` reasoning model). A provider is skipped if its `api_key_env` is
  set but missing in env, or if its call raises any error. Set to `[]` to
  disable vision. See ADR-0002.
- `wake_word.{enabled, phrase, sleep_phrase, idle_timeout_seconds, ack_text,
  start_awake}` — see Wake word section below.
- `system_prompt` — multiline LLM system message; the `{tool_capabilities}`
  placeholder is auto-filled with a category-grouped tool inventory.

`config.py` parses `config.yaml` into typed dataclasses; lookup is `lru_cache`d.

Secrets stay in `.env`; `config.yaml` references them by name (e.g.
`api_key_env: DEEPSEEK_API_KEY`). Never commit `config.yaml` or `.env`.

### Shared bot construction (`server/voice_bot.py`)

`build_components()` reads `config.yaml` and produces the STT + TTS + LLM +
context + tool stack. To change voice/model/prompt/provider, edit
`config.yaml` — not Python.

### Pipeline (`server/local_bot.py`)

```
LocalAudioTransport.input → VADProcessor → Soniox STT → EchoSuppressor →
  WakeWordGate → SessionLogProcessor → user context → DeepSeek LLM →
  Soniox TTS → LocalAudioTransport.output → assistant context
```

Key facts:

- **LLM is DeepSeek**, not OpenAI, even though it uses `OpenAILLMService`. The
  service points at `https://api.deepseek.com/v1` with
  `model="deepseek-v4-flash"` and reads `DEEPSEEK_API_KEY`. Thinking is
  explicitly disabled via `extra={"extra_body": {"thinking": {"type":
  "disabled"}}}` — leave it off unless you want first-token-latency cost
  (kills voice UX). See ADR-0003.
- **TTS is Soniox**, not Cartesia. See ADR-0001.
- **STT is Soniox in-pipeline.**
- **No AEC.** `LocalAudioTransport` has no echo cancellation, so the bot
  installs mute strategies + VAD-only turn-start + a connection rendezvous —
  all inlined in `local_bot.py` with the *why* documented inline. The
  experimental Speex AEC is archived in `experiments/aec/`.
- **Echo suppressor.** `EchoSuppressor` (`processors/echo_suppressor.py`) sits
  between STT and WakeWordGate. It drops `TranscriptionFrame` and
  `InterimTranscriptionFrame` while the bot is speaking and for
  `holdoff_seconds` (default 1 s) after it stops. Pure logic extracted as
  `should_suppress(now, bot_speaking, suppress_until) -> bool` for testability.
- **Hotkey interrupt.** `install_interrupt_hotkey` (`hotkey_interrupt.py`)
  registers a global pynput hotkey (default ⌘⇧I, override with
  `HOTKEY_INTERRUPT` env var using pynput syntax, e.g. `<ctrl>+<alt>+i`).
  Pressing it pushes an `InterruptionTaskFrame` into the pipeline, cancelling
  any in-flight LLM or TTS. Requires macOS Accessibility permission on first
  run.
- **Wake word.** `WakeWordGate` (`processors/wake_word.py`) sits after
  EchoSuppressor. While asleep it drops transcription frames so the LLM is
  never invoked. On wake it speaks `ack_text`; on sleep phrase it speaks
  `sleep_ack_text`. Both fields live in `config.yaml`. Pure logic extracted as
  `normalize_phrase(text) -> str` and `check_transition(awake, wake_phrase,
  sleep_phrase, text, is_final) -> (bool, str|None)` for testability.
- **Tool registry lives in `tools/registry.py`.** Adding a tool = write a
  `BaseTool` subclass in the matching domain file (`tools/files.py`,
  `tools/desktop.py`, or `tools/web.py`) and decorate with
  `@REGISTRY.register`. Set `name`, `description`, `parameters`, `required`,
  `speak_text` (optional spoken filler), `category` (groups tools in the
  capability summary), and implement `execute(**kwargs)`. Sync `execute`
  returning `dict` passes through unchanged; any other return type is wrapped
  in `{"result": value}`. Blocking I/O runs in `asyncio.to_thread`. Heavy
  imports (`pyperclip`, `PIL`, `cv2`, `mlx_vlm`) are lazy so a missing dep
  disables only that tool.
- **`{tool_capabilities}` placeholder.** `REGISTRY.capabilities_summary()`
  produces a category-grouped tool inventory that `voice_bot.build_components`
  substitutes into the `{tool_capabilities}` slot in the system prompt. Keep
  that slot in the prompt so the LLM stays aware of available tools without
  the prompt hard-coding names.
- **Vision is a fallback chain**, not a single provider.
  `tools.vision.describe_image()` walks the `vision:` list and returns the
  first non-empty description. DeepSeek's chat API rejects `image_url` blocks,
  so vision stays separate from the text LLM. Primary provider is in-process
  MLX (`mlx-community/Qwen3-VL-2B-Instruct-4bit`) via `tools/mlx_vision.py`;
  fallback is Kimi `kimi-k2.6` (a reasoning model — answer may arrive in
  `reasoning_content`; `tools.vision.strip_reasoning()` handles both). All
  vision capture tools accept an optional `question:` arg. See ADR-0002.
- **Session logs** are kebab-case JSONL written to `logs/session-<ts>.jsonl`
  by `processors/session_log.py`. Image captures are saved to
  `logs/captures/` as timestamped JPEGs; the session log records the
  `image_path` alongside the tool result. Override the directory with
  `VOICE_BOT_LOG_DIR`.

### Domain glossary (`server/CONTEXT.md`)

Defines 11 domain terms: Turn, Wake Word, Echo Suppression, Mute Strategy,
Vision Chain, Session Log, Tool, Pipeline Processor, Connection Rendezvous,
Interruption, AEC, Provider. Use these terms in code and docs.

### ADRs (`server/docs/adr/`)

- ADR-0001 — Soniox for STT and TTS
- ADR-0002 — Vision as a fallback chain
- ADR-0003 — DeepSeek with thinking disabled for voice latency

### Required env vars (`server/.env`)

- `SONIOX_API_KEY`, `DEEPSEEK_API_KEY` — required.
- `MOONSHOT_API_KEY` — required for the `kimi` vision fallback (and any
  vision use on non-Apple-Silicon hosts where MLX is skipped).
- `OPENAI_API_KEY` — only if you swap a vision chain entry to OpenAI.
- `SERPER_API_KEY` — only for the `web_search` tool.
- `VOICE_BOT_LOG_DIR` — optional override for session-log JSONL directory
  (default `./logs`).
