"""Local MLX vision-language describer.

Loads an MLX-quantised vision-language model (e.g. `mlx-community/Qwen3-VL-2B-Instruct-4bit`)
in-process via `mlx_vlm`, caches it module-level after first load, and exposes
`describe(model_id, image_path, prompt, max_tokens) -> str`.

No LM Studio, no HTTP server, no rate limits. Apple-Silicon-only (MLX requires it).
First call loads weights from `~/.cache/huggingface` (auto-downloaded on cache miss);
subsequent calls reuse the in-memory model.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# (model, processor, config) tuples cached per model id so we pay the load cost
# exactly once per process. Keep weights in RAM/VRAM across all describe calls.
_LOADED: dict[str, tuple[Any, Any, Any]] = {}


def is_available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def _load(model_id: str):
    if model_id in _LOADED:
        return _LOADED[model_id]
    logger.info(f"mlx_vlm: loading {model_id} (first call — may take ~10s)")
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    config = load_config(model_id)
    _LOADED[model_id] = (model, processor, config)
    logger.info(f"mlx_vlm: {model_id} loaded")
    return _LOADED[model_id]


def describe(
    model_id: str,
    image_path: str,
    prompt: str,
    max_tokens: int = 150,
) -> str:
    """Run a single image+prompt → text completion against the local MLX model."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load(model_id)
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )
    output = generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path],
        max_tokens=max_tokens,
        verbose=False,
    )
    # mlx_vlm.generate returns a GenerationResponse object in 0.5+;
    # earlier versions returned a string. Handle both.
    if hasattr(output, "text"):
        return (output.text or "").strip()
    return str(output).strip()
