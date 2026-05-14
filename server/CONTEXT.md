# CONTEXT.md — Voice Agent Domain Glossary

Shared vocabulary for this codebase. Module docstrings reference these terms
instead of re-explaining them. Architecture reviews (e.g. the `improve-codebase-architecture`
skill) use this glossary as the domain language.

## Terms

**Turn**
A complete user-speech-to-bot-response cycle. A turn starts when the user
begins speaking (VAD detects speech) and ends when the bot finishes producing
TTS audio. The `UserTurnStrategies` govern when turns start and stop.

**Wake Word**
A spoken phrase (e.g. "Hey Ava") that toggles the bot from asleep to awake.
While asleep, all transcription frames are dropped — the LLM is never invoked.
Controlled by `wake_word.py` / `WakeWordGate`, configured in `config.yaml`
under `wake_word`.

**Echo Suppression**
Dropping STT transcription frames while the bot is speaking (and for a brief
holdoff after). Without AEC, the bot's TTS output bleeds into the mic and
Soniox transcribes it. `EchoSuppressor` in `echo_suppressor.py`.

**Mute Strategy**
A Pipecat `UserMuteStrategy` that controls when the mic is muted. The local
audio transport uses `AlwaysUserMuteStrategy` (mute during bot turns) +
`MuteUntilFirstBotCompleteUserMuteStrategy` (mute until the first intro
completes). Configured in `local_bot.py`.

**Vision Chain**
A fallback list of image-description providers tried in order. The first
provider that returns text wins. Configured in `config.yaml` under `vision`.
Implemented in `tools/vision.py`. Called by capture tools (screenshot,
webcam, window, region, display).

**Session Log**
Per-session structured JSONL log written to `logs/session-<ts>.jsonl`.
Records user-spoke, bot-spoke, tool-called, tool-result, and other events
with timestamps. Implemented in `session_log.py`.

**Tool**
A declarative LLM-callable function. Each tool is a `BaseTool` subclass with
name, description, JSON-schema parameters, optional spoken filler, and an
`execute()` method. Tools register on the global `REGISTRY` via the
`@REGISTRY.register` decorator. Implemented in `tools/registry.py` with
domain-specific tools in `tools/files.py`, `tools/desktop.py`, `tools/web.py`.

**Pipeline Processor**
A Pipecat `FrameProcessor` subclass placed in the pipeline between transport,
STT, LLM, and TTS. Examples: `EchoSuppressor`, `WakeWordGate`,
`SessionLogProcessor`, `VADProcessor`. Each processes frames in order and
can drop, modify, or inject frames.

**Connection Rendezvous**
The point where both STT and TTS WebSocket connections are established, after
which the bot pushes its seed context frame (triggering the introduction).
Without this, the bot's intro is lost because it fires before one of the
connections is ready. Implemented in `local_bot.py`.

**Interruption**
Cancelling an in-flight LLM call and TTS output. In the local-audio transport,
interruption is triggered by a global hotkey (default ⌘⇧I) because the mic is
muted during bot turns. With AEC, interruption is triggered by
`UserStartedSpeakingFrame` mid-bot-turn (voice barge-in). Implemented in
`hotkey_interrupt.py`.

**AEC (Acoustic Echo Cancellation)**
Speex software echo cancellation that subtracts the bot's TTS output from the
mic input. Enables voice barge-in by keeping the mic open during bot turns.
Experimental — archived in `experiments/aec/`.

**TurnPolicy**
The cohesive bundle of mute strategies, turn-start/stop strategies, echo
suppressor, wake-word gate, VAD processor placement, and connection
rendezvous that together define how a turn starts and ends on a given
transport. Local-audio has its own (no-AEC) policy in `turn_policy.py`;
hardware-AEC transports would have a sibling policy.

**Provider**
An external API backend (Soniox, DeepSeek, Kimi/Moonshot, OpenAI). Each
provider is configured in `config.yaml` with a name, base URL, model, and
API key env var reference. Swapping providers means editing `config.yaml`,
not Python code.
