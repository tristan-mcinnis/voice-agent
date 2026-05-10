"""Vision description chain — image preparation, provider dispatch, reasoning stripping.

Walk the `vision:` list from `config.yaml` in order; the first provider that
returns text wins. A provider is skipped if its `api_key_env` is set but the
env var is missing, or if its call raises any error.

No tool classes here — tools in `desktop.py` and `capture.py` call
`describe_image()` and `no_vision_message()` through this module.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Vision chain — the fallback list of providers as a proper object, not module-level
# mutable state. Constructed once at startup; passed explicitly or accessed via
# the module-level singleton. Tests construct their own chain; no temporal coupling.
# ---------------------------------------------------------------------------

class VisionChain:
    """Ordered list of vision providers. First success wins.

    Each provider is a ``config.VisionProvider``. The chain is immutable
    after construction — no more ``set_vision_config()`` mutating module state.
    """

    def __init__(self, providers: list[Any]) -> None:
        self._providers = list(providers)

    @property
    def providers(self) -> list[Any]:
        return list(self._providers)

    def describe(self, image_path: str, prompt: str) -> str | None:
        """Walk the chain; return the first provider's description, or None."""
        if not self._providers:
            return None
        for provider in self._providers:
            result = try_describe_with(provider, image_path, prompt)
            if result:
                return result
        return None

    def no_vision_message(self) -> str:
        """Build a spoken-friendly explanation when vision fails."""
        if not self._providers:
            return (
                "I captured the image but vision is disabled in config.yaml. "
                "Configure at least one vision provider there to enable describe."
            )
        names = ", ".join(p.name for p in self._providers)
        missing = [
            p.api_key_env for p in self._providers
            if p.api_key_env and not os.getenv(p.api_key_env)
        ]
        if missing:
            return (
                f"I captured the image but every vision provider failed. "
                f"Missing env vars: {', '.join(missing)}. "
                f"Tried providers: {names}."
            )
        return (
            f"I captured the image but every vision provider failed. "
            f"Tried: {names}. Check the server logs for the exact errors."
        )


# Module-level singleton — set once at startup.
_chain: VisionChain | None = None


def set_vision_config(providers: list[Any]) -> None:
    """Initialise the vision chain from config at startup.

    Called by ``voice_bot.build_components()``. Must be called before
    ``describe_image()`` or ``no_vision_message()``.
    """
    global _chain
    _chain = VisionChain(providers)


def _get_chain() -> VisionChain:
    """Return the vision chain, raising if not yet configured.

    Replaces the old silent-None-on-unconfigured behaviour with a loud
    failure — temporal coupling made visible.
    """
    if _chain is None:
        raise RuntimeError(
            "Vision chain not configured. "
            "Call set_vision_config() at startup before using describe_image()."
        )
    return _chain


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _prepare_image(image_path: str, max_width: int) -> tuple[str, str]:
    """Read the image, optionally downscale, return (base64, ext)."""
    if max_width and max_width > 0:
        try:
            from PIL import Image
            from io import BytesIO

            with Image.open(image_path) as img:
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.convert("RGB").resize(new_size, Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, "JPEG", quality=85)
                    return base64.b64encode(buf.getvalue()).decode(), "jpeg"
        except Exception as exc:
            logger.warning(f"Image downscale failed, sending original: {exc}")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "png" if image_path.endswith(".png") else "jpeg"
    return b64, ext


def downscaled_image_path(image_path: str, max_width: int) -> str:
    """Write a downscaled JPEG next to the original; return its path.

    Used by the in-process MLX path which takes a file path, not bytes.
    """
    if not max_width or max_width <= 0:
        return image_path
    try:
        from PIL import Image as PILImage

        with PILImage.open(image_path) as img:
            if img.width <= max_width:
                return image_path
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            resized = img.convert("RGB").resize(new_size, PILImage.LANCZOS)
            small_path = image_path.rsplit(".", 1)[0] + ".small.jpg"
            resized.save(small_path, "JPEG", quality=85)
            return small_path
    except Exception as exc:
        logger.warning(f"Image downscale failed, sending original: {exc}")
        return image_path


# ---------------------------------------------------------------------------
# Reasoning-stripping heuristic
# ---------------------------------------------------------------------------

_REASONING_MARKERS = (
    "wait,", "actually,", "let me think", "let's think", "let's keep",
    "let's go", "possible answer", "or simpler", "or even shorter",
    "the user", "the question", "i need to", "key elements",
)


def strip_reasoning(text: str) -> str:
    """Drop chain-of-thought preamble from reasoning-model output."""
    if not text:
        return text
    lower = text.lower()
    if not any(m in lower for m in _REASONING_MARKERS):
        return text.strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    tail: list[str] = []
    for line in reversed(lines):
        if any(m in line.lower() for m in _REASONING_MARKERS):
            break
        tail.append(line)
    if tail:
        result = " ".join(reversed(tail)).strip()
    else:
        result = lines[-1] if lines else text
    return result.strip().strip('"').strip("'").strip()


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def try_describe_with(provider: Any, image_path: str, prompt: str) -> str | None:
    """Attempt one vision provider. Returns text on success, None on any failure.

    `provider` is a `config.VisionProvider` instance.
    """
    full_prompt = (
        f"{prompt}\n\n{provider.brevity_suffix}".strip()
        if provider.brevity_suffix else prompt
    )

    if provider.kind == "mlx":
        try:
            import tools.mlx_vision as mlx_vision
        except ImportError:
            logger.info(f"vision: skipping {provider.name} — mlx_vlm not installed")
            return None
        small_path = downscaled_image_path(image_path, provider.max_image_width)
        try:
            t0 = time.monotonic()
            text = strip_reasoning(mlx_vision.describe(
                provider.model, small_path, full_prompt,
                max_tokens=provider.max_tokens,
            ))
            elapsed = time.monotonic() - t0
            kb = os.path.getsize(small_path) // 1024
            logger.info(
                f"vision: {provider.name}/{provider.model} "
                f"img_kb={kb} out_chars={len(text)} elapsed={elapsed:.1f}s"
            )
            return text or None
        except Exception as exc:
            logger.warning(f"vision: {provider.name} failed: {exc}")
            return None

    # kind == "openai" (default): OpenAI-compatible HTTP endpoint.
    from openai import OpenAI

    if provider.api_key_env:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            logger.info(
                f"vision: skipping {provider.name} — {provider.api_key_env} unset"
            )
            return None
    else:
        api_key = "no-auth"

    b64, ext = _prepare_image(image_path, provider.max_image_width)

    client = OpenAI(
        api_key=api_key, base_url=provider.base_url,
        max_retries=0, timeout=provider.timeout,
    )

    last_exc = None
    for attempt in range(2):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=provider.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{ext};base64,{b64}",
                        }},
                        {"type": "text", "text": full_prompt},
                    ],
                }],
                max_tokens=provider.max_tokens,
            )
            elapsed = time.monotonic() - t0
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
            text = strip_reasoning(content or reasoning)
            logger.info(
                f"vision: {provider.name}/{provider.model} "
                f"img_kb={len(b64)*3//4//1024} out_chars={len(text)} "
                f"src={'content' if content else 'reasoning'} "
                f"elapsed={elapsed:.1f}s"
            )
            return text or None
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and "overload" in str(exc).lower():
                time.sleep(2.0)
                continue
            break
    logger.warning(f"vision: {provider.name} failed: {last_exc}")
    return None


def describe_image(image_path: str, prompt: str) -> str | None:
    """Walk the vision provider chain; first success wins.

    Delegates to the module-level ``VisionChain`` singleton, which must
    be initialised via ``set_vision_config()`` at startup.
    """
    return _get_chain().describe(image_path, prompt)


def no_vision_message() -> str:
    """Build a spoken-friendly message explaining why vision didn't work.

    Delegates to the module-level ``VisionChain`` singleton.
    """
    return _get_chain().no_vision_message()
