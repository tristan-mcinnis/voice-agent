# Context

Domain glossary for the server/ Python codebase. Names introduced or sharpened during architecture work belong here.

**BotComponents**: the dataclass returned by `voice_bot.build_components()`. Bundles the STT, TTS, LLM, `LLMContext`, and `LLMContextAggregatorPair` that any transport plugs into a Pipecat `Pipeline`.

**Tool**: a frozen dataclass in `tools.py` binding one LLM-callable function to its `FunctionSchema` and an optional spoken filler phrase. The `TOOLS` list is the registry — adding a tool means adding one entry.

**LocalAudioConcerns**: the bundle of mic-and-speakers-only pipeline concerns in `local_audio.py` — mute strategies, turn-start/stop strategies, STT/TTS connection rendezvous, and diagnostic event handlers. Exists because `LocalAudioTransport` has no AEC; Daily/WebRTC does not install any of this because the browser handles echo cancellation.

**Sprite loop**: the 25-frame robot animation in `server/assets/`. `talking_animation.load_sprite_loop()` builds a `(quiet_frame, talking_frame)` pair where `talking_frame` plays forward then back so the loop appears continuous.

**Connection rendezvous** _(local-audio only)_: both Soniox STT and Soniox TTS must fire `on_connected` before pushing the seed context frame; otherwise the bot's intro is lost in the gap. Implemented inside `install_local_audio_lifecycle`.
