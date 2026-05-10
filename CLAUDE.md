# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repo layout

Single Python app under `server/` — a Pipecat voice bot that runs locally on
mic + speakers (no browser, no Daily/WebRTC).

## Common commands

```bash
cd server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python local_bot.py               # mic/speakers loop
python test_stt.py                # Soniox STT WebSocket smoke test (needs test_tts.wav)
python test_tts.py                # Soniox TTS WebSocket smoke test (writes test_tts.wav)
python test_tools.py              # Smoke-tests every tool in tools.TOOLS
```

There is no pytest suite — `test_stt.py` / `test_tts.py` / `test_tools.py` are
run directly as scripts.

## Architecture

### Configuration (`server/config.yaml`)

`config.yaml` is the swap-providers-without-touching-code surface. It drives
`voice_bot.build_components()`, `tools._describe_image()`, and the wake-word
gate.

- `llm.{provider, base_url, model, api_key_env, extra}` — any OpenAI-compatible
  chat-completions endpoint (DeepSeek, Kimi/Moonshot, OpenAI). `extra` carries
  provider-specific knobs (e.g. DeepSeek's `thinking: disabled`).
- `stt.{provider, api_key_env, language_hints}` — only `soniox` is wired up
  today. Add a builder in `voice_bot._build_stt` to support more.
- `tts.{provider, api_key_env, voice}` — same shape as STT.
- `vision` — a **list (chain)** of providers tried in order; first one to
  return text wins. Each entry has `kind: mlx | openai`. Default chain:
  `mlx-local` (in-process `mlx_vlm`, Apple-Silicon-only) → `kimi` (Moonshot
  `kimi-k2.6` reasoning model). A provider is skipped if its `api_key_env` is
  set but missing in env, or if its call raises any error. Set to `[]` to
  disable vision.
- `wake_word.{enabled, phrase, sleep_phrase, idle_timeout_seconds, ack_text,
  start_awake}` — see Wake word section below.
- `system_prompt` — multiline LLM system message.

`config.py` parses `config.yaml` into typed dataclasses; lookup is `lru_cache`d.

Secrets stay in `.env`; `config.yaml` references them by name.

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
  (kills voice UX).
- **TTS is Soniox**, not Cartesia.
- **STT is Soniox in-pipeline.**
- **No AEC.** `LocalAudioTransport` has no echo cancellation, so the bot
  installs mute strategies + VAD-only turn-start + a connection rendezvous —
  all bundled in `local_audio.py` with the *why* documented inline.
- **Echo suppressor.** `EchoSuppressor` (`server/echo_suppressor.py`) sits
  between STT and WakeWordGate. It drops `TranscriptionFrame` and
  `InterimTranscriptionFrame` while the bot is speaking and for
  `holdoff_seconds` (default 1 s) after it stops. This prevents TTS bleed-
  through from poisoning the LLM context.
- **Hotkey interrupt.** `install_interrupt_hotkey` (`server/hotkey_interrupt.py`)
  registers a global pynput hotkey (default ⌘⇧I, override with
  `HOTKEY_INTERRUPT` env var using pynput syntax, e.g. `<ctrl>+<alt>+i`).
  Pressing it pushes an `InterruptionTaskFrame` into the pipeline, cancelling
  any in-flight LLM or TTS. Requires macOS Accessibility permission on first
  run.
- **Wake word.** `WakeWordGate` (`server/wake_word.py`) sits after
  EchoSuppressor. While asleep it drops transcription frames so the LLM is
  never invoked. On wake it speaks `ack_text`; on sleep phrase it speaks
  `sleep_ack_text`. Both fields live in `config.yaml`.
- **Tool registry lives in `server/tools.py`.** Adding a tool = write a
  `BaseTool` subclass decorated with `@REGISTRY.register`. Set `name`,
  `description`, `parameters`, `required`, `speak_text` (optional spoken
  filler), `category` (groups tools in the capability summary), and implement
  `execute(**kwargs)`. Sync `execute` returning `dict` passes through unchanged;
  any other return type is wrapped in `{"result": value}`. Blocking I/O runs
  in `asyncio.to_thread`. Heavy imports (`pyperclip`, `PIL`, `cv2`, `mlx_vlm`)
  are lazy so a missing dep disables only that tool.
- **`{tool_capabilities}` placeholder.** `REGISTRY.capabilities_summary()`
  produces a category-grouped tool inventory that `voice_bot.build_components`
  substitutes into the `{tool_capabilities}` slot in the system prompt. Keep
  that slot in the prompt so the LLM stays aware of available tools without
  the prompt hard-coding names.
- **Vision is a fallback chain**, not a single provider.
  `tools._describe_image()` walks the `vision:` list and returns the first
  non-empty description. DeepSeek's chat API rejects `image_url` blocks, so
  vision stays separate from the text LLM. Primary provider is in-process MLX
  (`mlx-community/Qwen3-VL-2B-Instruct-4bit`); fallback is Kimi `kimi-k2.6`
  (a reasoning model — answer may arrive in `reasoning_content`; the code
  handles both). All vision capture tools accept an optional `question:` arg.
- **Session logs** are kebab-case JSONL written to `logs/session-<ts>.jsonl`
  by `session_log.py`. Image captures are saved to `logs/captures/` as
  timestamped JPEGs; the session log records the `image_path` alongside the
  tool result. Override the directory with `VOICE_BOT_LOG_DIR`.

### Required env vars (`server/.env`)

- `SONIOX_API_KEY`, `DEEPSEEK_API_KEY` — required.
- `MOONSHOT_API_KEY` — required for the `kimi` vision fallback (and any
  vision use on non-Apple-Silicon hosts where MLX is skipped).
- `OPENAI_API_KEY` — only if you swap a vision chain entry to OpenAI.
- `SERPER_API_KEY` — only for the `web_search` tool.
- `VOICE_BOT_LOG_DIR` — optional override for session-log JSONL directory
  (default `./logs`).
