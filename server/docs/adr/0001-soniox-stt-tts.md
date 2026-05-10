# ADR-0001: Soniox for STT and TTS

**Status:** accepted

**Date:** 2025-05-10

**Context:** The voice bot needs both speech-to-text and text-to-speech.
Alternatives considered: Whisper (local or API) for STT; ElevenLabs,
Cartesia, or piper-tts for TTS.

**Decision:** Use Soniox for both STT and TTS, accessed via Pipecat's
`SonioxSTTService` and `SonioxTTSService`.

Reasons:
- Single API key for both services — simpler secret management.
- WebSocket protocol — lower latency than REST polling for STT.
- No browser dependency (unlike Daily/WebRTC transports).
- Pipecat has first-class Soniox integration (no custom service wrapper needed).

**Consequences:**
- STT and TTS are coupled to Soniox's infrastructure. An outage affects both.
- Adding a new STT or TTS provider requires implementing a builder in
  `voice_bot._build_stt` / `_build_tts`, but the `config.yaml` swap-surface
  is already in place.
- No offline fallback — if Soniox is unreachable, the bot is silent.
