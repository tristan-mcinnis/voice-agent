# ADR-0003: DeepSeek with thinking disabled for voice latency

**Status:** accepted

**Date:** 2025-05-10

**Context:** Voice interaction requires low first-token latency. The user
speaks and expects the bot to begin replying within ~1s. Reasoning models
that emit chain-of-thought before the visible response add unacceptable delay.

**Decision:** Use DeepSeek's chat API (`deepseek-v4-flash`) with `thinking`
explicitly disabled via `extra_body: {thinking: {type: disabled}}`. The LLM
produces no reasoning tokens — the first token emitted is the visible response.

**Consequences:**
- First-token latency is consistently under 1s — viable for voice.
- The bot loses reasoning capability on DeepSeek models. For tasks that
  benefit from chain-of-thought, the system prompt must compensate with
  explicit instructions.
- The `thinking: disabled` flag is DeepSeek-specific. If the LLM provider
  is swapped (e.g. to Kimi/Moonshot or OpenAI), the `extra` block in
  `config.yaml` must be updated or removed.
- The LLM is accessed via `OpenAILLMService` pointing at
  `https://api.deepseek.com/v1` — any OpenAI-compatible endpoint works by
  changing `config.yaml`.
