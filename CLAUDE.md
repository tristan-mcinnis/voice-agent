# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repo layout

Single Python app under `server/` — a Pipecat voice bot that runs locally on
mic + speakers (no browser, no Daily/WebRTC).

```
server/
  local_bot.py            main entry point + mute/turn/lifecycle
  voice_bot.py            shared STT/TTS/LLM/context construction
  config.py               typed config dataclasses, YAML loader
  config.example.yaml     example config (copy to config.yaml)
  hotkey_interrupt.py     global hotkey (⌘⇧I)
  connection_rendezvous.py  wait-for-both-connections coordinator

  agent/                  cognitive-stack prompt assembly
    prompt_builder.py     multi-layer frozen system-prompt builder
    memory_store.py       read-only USER.md / MEMORY.md snapshots

  tools/                  LLM-callable tool implementations
    __init__.py           re-exports REGISTRY + compat callables
    registry.py           BaseTool, ToolRegistry, REGISTRY, _make_handler
    vision.py             image description fallback chain
    mlx_vision.py         in-process MLX vision (internal adapter)
    files.py              file ops + tool classes (impl in execute())
    desktop.py            macOS automation + tool classes (impl in execute())
    web.py                web search, weather demo + tool classes (impl in execute())
    memory.py             memory tool (add/replace/list USER.md, MEMORY.md)
    search_history.py     search past session logs for conversation recall
    agent_runner.py       background subprocess engine (spawn/tail/cancel)
    external_agents.py    consult + spawn tools for Kimi/DeepSeek/Gemini/Claude Code/Codex/Hermes

  processors/             pipeline FrameProcessor stages
    echo_suppressor.py    drops STT frames while bot speaks
    wake_word.py          wake-word gate (asleep/awake state machine)
    session_log.py        per-session JSONL logger + SessionLogProcessor

  tests/                  tests
    test_stt.py           Soniox STT WebSocket smoke test
    test_tts.py           Soniox TTS WebSocket smoke test
    test_tools.py         tool registry smoke test
    unit/                 pytest unit tests for pure functions
      test_echo_suppressor.py
      test_wake_word.py
      test_session_log.py
      test_vision.py
      test_files.py
      test_connection_rendezvous.py

  docs/adr/               architecture decision records
  experiments/aec/        archived Speex AEC experiment
  CONTEXT.md              domain glossary
  .voice-agent/           agent data (SOUL.md, memories, skills) — gitignored
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
python -m pytest tests/unit/ -v        # Pure-function unit tests (142 tests, no I/O)
python -m scripts.summarize_session logs/session-*.jsonl  # Aggregate per-session latency + cache stats
```

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
- `system_prompt` — fallback identity when `SOUL.md` is missing or empty. The
  actual system prompt is assembled once per session by `PromptBuilder` from
  the cognitive stack (Soul → Memory → User → Rules → Tools → Skills).

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
  (kills voice UX). See ADR-0003. When `llm.provider == "deepseek"`,
  `voice_bot.build_llm` instantiates `services.deepseek_llm.DeepSeekLLMService`
  (a thin subclass) which wraps the chunk stream to promote DeepSeek's flat
  `usage.prompt_cache_hit_tokens` into the OpenAI-standard
  `prompt_tokens_details.cached_tokens` — without it, the `llm-usage` session
  event would always report `cache_read_input_tokens=null` even when caching
  is working.
- **TTS is Soniox**, not Cartesia. See ADR-0001. When
  `tts.stream_clauses: true`, the service's internal text aggregator is
  swapped (in `voice_bot.build_tts`) for `ClauseTextAggregator`
  (`processors/clause_aggregator.py`), which releases at clause boundaries
  (`,;:—`) gated by `clause_min_words`. Pure logic
  `next_release_index(text, min_words)` is unit-tested.
- **STT is Soniox in-pipeline.**
- **No AEC.** `LocalAudioTransport` has no echo cancellation, so the bot
  installs mute strategies + VAD-only turn-start + a connection rendezvous —
  all inlined in `local_bot.py` with the *why* documented inline. The
  experimental Speex AEC is archived in `experiments/aec/`.
- **Turn-stop strategy is config-switched.** Default is
  `SpeechTimeoutUserTurnStopStrategy` (fixed-silence endpointer). When
  `turn.smart_turn_enabled: true`, `turn_policy._build_stop_strategy()`
  lazy-loads `LocalSmartTurnAnalyzerV3` (ONNX) wrapped in
  `TurnAnalyzerUserTurnStopStrategy`. Any import or model-load failure
  logs a warning and falls back to speech-timeout — the bot stays usable
  without `pipecat-ai[local-smart-turn]`.
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
- **Cognitive Stack Prompt Builder.** `agent.prompt_builder.PromptBuilder`
  assembles the frozen system prompt once per session from layered snapshots:
  `SOUL.md` (identity), `MEMORY.md` (project facts), `USER.md` (preferences),
  project rule files (`.voice-agent.md`, `AGENTS.md`, `CLAUDE.md`,
  `.cursorrules`), the tool registry inventory, and a skills index. The stable
  prefix (Soul + Memory + User) stays warm in provider-side caches because it
  never changes mid-session.
- **`memory` tool.** The agent manages two persistent memory files:
  `USER.md` (user preferences, max 1,375 chars) and `MEMORY.md` (project
  facts, max 2,200 chars). Entries are `§`-delimited for high-density recall.
  Actions: `add` (with near-duplicate detection and pressure checks),
  `replace` (substring match-and-swap), `list` (indexed inventory).
  The `patch_memory` name is kept as a backward-compat alias in `__init__.py`.
  Memory guidance instructions are injected into the system prompt by
  `PromptBuilder` so the agent knows *how* to use its own memory.
- **`search_history` tool.** Searches raw session JSONL logs for past
  conversations. Use when the user asks "what did we talk about last time?"
  — the curated memory files hold facts; this tool finds transcripts.
- **External-agent tools.** `tools/external_agents.py` exposes other models
  and CLIs as voice-callable tools, grouped under the `agents` category.
  Three tiers: (1) **consult** — sync `ask_kimi`, `ask_deepseek_reasoner`,
  `ask_gemini` make a single OpenAI-compatible chat call and return text;
  (2) **spawn** — `spawn_claude_code`, `spawn_codex`, `spawn_gemini_cli` fork
  a CLI subprocess and return a `task_id` immediately, with state tracked by
  `tools/agent_runner.py` in `.voice-agent/agent-tasks/<task_id>/`;
  (3) **hermes** — `hermes_chat` / `hermes_spawn` talk to the user's local
  Hermes agent over HTTP. All endpoints, binaries, and concurrency caps
  live under `agents:` in `config.yaml`. Missing keys or missing binaries
  degrade per-tool (return an error string) — the bot stays usable.
  `list_agent_tasks`, `get_agent_task`, and `cancel_agent_task` work
  across every spawn tool. Status reconciliation is lazy: a `running`
  task whose pid is dead gets marked `orphaned` on next read, so bot
  restarts don't leak stale state.
- **Memory architecture.** Follows the Hermes agent pattern: frozen curated
  facts (USER.md/MEMORY.md, loaded once per session for prefix caching)
  + fluid session logs (JSONL, searchable) + optional pluggable providers.
  The loop: agent observes a fact → calls `memory(action="add", …)` →
  file written to disk → next session's `PromptBuilder` reads the frozen
  snapshot → agent "just knows" without the user repeating themselves.
- **`{tool_capabilities}` placeholder.** `REGISTRY.capabilities_summary()`
  produces a category-grouped tool inventory. When the fallback identity
  (config `system_prompt`) still contains this placeholder, the PromptBuilder
  substitutes it inline for backward compatibility.
- **Vision is a fallback chain**, not a single provider.
  `tools.vision.describe_image()` walks the `vision:` list and returns the
  first non-empty description. DeepSeek's chat API rejects `image_url` blocks,
  so vision stays separate from the text LLM. Primary provider is in-process
  MLX (`mlx-community/Qwen3-VL-2B-Instruct-4bit`) via `tools/mlx_vision.py`;
  fallback is Kimi `kimi-k2.6` (a reasoning model — answer may arrive in
  `reasoning_content`; `tools.vision.strip_reasoning()` handles both). All
  vision capture tools accept an optional `question:` arg. See ADR-0002.
- **Session summarizer.** `scripts/summarize_session.py` is a CLI that
  reduces a session JSONL into one report: turn count, p50/p95/max for each
  latency phase, total prompt/completion tokens, cache-hit rate, *and* the
  active config (model, voice, stream_clauses, smart_turn, user-speech
  timeout) — so a diff across two reports tells you what changed
  *between runs* and what the latency difference was. Use it to A/B
  `turn.smart_turn_enabled` or `tts.stream_clauses` — flip one flag, run
  a session, diff the reports. Invoke:
  `python -m scripts.summarize_session logs/session-*.jsonl`.
- **Session-config snapshot.** `SessionLog.record_config(config)` writes one
  `session-config` event at startup capturing every dial that matters for
  latency/cache/turn-taking comparisons (providers, model, voice,
  stream_clauses, smart_turn_enabled, user_speech_timeout,
  echo_holdoff_seconds, connection_timeout_seconds, wake_word_enabled).
  `local_bot.main` calls it right after `SessionLog.for_now()`. The
  summarizer reads this and pins it at the top of each report.
- **LLM cache-hit logging.** The post-LLM `SessionLogProcessor`
  (constructed with `track_usage=True` in `turn_policy`) listens for
  `MetricsFrame`s carrying `LLMUsageMetricsData` and emits one
  `llm-usage` event per LLM call with `prompt_tokens`, `completion_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`,
  `reasoning_tokens`. Only one instance tracks usage to avoid double-count.
  This is the metric that tells you whether `PromptBuilder`'s frozen prefix
  is actually being cached by DeepSeek/Kimi/etc.
- **Latency tracer.** `LatencyTracer` (`processors/latency_tracer.py`) sits
  alongside the post-STT SessionLogProcessor. It stamps each phase of a turn
  (`user_stopped → llm_started → first_text → tts_started → first_audio →
  bot_started`) and emits one `turn-latency` event per turn into the session
  log, plus a `⏱` summary line to stderr. Pure logic
  `compute_turn_latency(stamps) -> dict[str, float|None]` returns ms-deltas
  and is unit-tested. Use these numbers to decide whether to invest in
  smart-turn detection, clause-streaming TTS, or a faster LLM — don't
  optimise without them.
- **Session logs** are kebab-case JSONL written to `logs/session-<ts>.jsonl`
  by `processors/session_log.py`. Image captures are saved to
  `logs/captures/` as timestamped JPEGs; the session log records the
  `image_path` alongside the tool result. Override the directory with
  `VOICE_BOT_LOG_DIR`.

### Domain glossary (`server/CONTEXT.md`)

Defines 11 domain terms: Turn, Wake Word, Echo Suppression, Mute Strategy,
Vision Chain, Session Log, Tool, Pipeline Processor, Connection Rendezvous,
Interruption, AEC, Provider. Use these terms in code and docs.

### Computer-use backends

Actuation tools in `tools/computer_use.py` (`click_at`, `type_text`,
`press_key`, `scroll`, `mouse_move`, and the click side of `click_element`)
have a swappable backend, controlled by `computer_use.backend` in
`config.yaml`:

- `native` (default) — pyautogui. Works everywhere; steals cursor/focus.
- `cua` — routes calls through `cua-driver` (trycua/cua), a Swift MCP
  server that drives macOS without stealing cursor/focus. One-time install:

      brew tap trycua/cua && brew install cua-driver
      which cua-driver   # verify on PATH

  Call the `cua_status` tool from a session to confirm the handshake and
  inspect the live MCP tool inventory. If `backend: cua` is set but the
  binary is missing or the handshake fails, each call logs a warning and
  falls back to pyautogui — the bot stays usable.

The integration lives entirely in `tools/cua_backend.py` (stdlib-only
MCP-stdio client) plus a small `_try_cua_backend()` resolver at the top of
`tools/computer_use.py`. Read-only desktop tools in `tools/desktop.py` are
unaffected — they stay on AppleScript regardless of backend. Removal steps
are documented in the `cua_backend.py` module docstring.

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
