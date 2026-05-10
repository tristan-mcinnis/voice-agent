# ADR-0002: Vision as a fallback chain

**Status:** accepted

**Date:** 2025-05-10

**Context:** The bot's vision tools (screenshot, webcam capture, window capture)
need to describe images. A single vision provider creates a single point of
failure — if that provider is down, rate-limited, or unconfigured, all vision
tools break.

**Decision:** Configure vision as an ordered list (chain) of providers in
`config.yaml`. The first provider that returns text wins. A provider is skipped
if its `api_key_env` is unset in the environment, or if its call raises any
error (timeout, 503, 429, model crash, overload).

Default chain:
1. `mlx-local` — in-process MLX model (`Qwen3-VL-2B-Instruct-4bit`). No HTTP,
   no rate limits. Apple-Silicon-only.
2. `kimi` — Kimi/Moonshot `kimi-k2.6` via HTTP. Acts as fallback for non-Apple
   hosts, and as overload protection when MLX is OOM.

**Consequences:**
- Vision is resilient to individual provider failures. Adding a third provider
  (e.g. OpenAI) is a one-line config entry.
- The MLX model loads on first call (~10s) and stays in RAM/VRAM. On 8GB
  machines this may cause OOM with other ML workloads.
- Reasoning models (`kimi-k2.6`) can dump chain-of-thought into `content`
  instead of `reasoning_content`. The `strip_reasoning()` heuristic in
  `tools/vision.py` mitigates this but isn't perfect.
- DeepSeek's chat API rejects `image_url` blocks — empirically verified.
  Vision stays separate from the text LLM.
