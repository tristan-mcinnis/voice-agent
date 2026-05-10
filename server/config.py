"""Voice-bot runtime configuration loader.

Reads `config.yaml` (next to this file by default) into typed dataclasses.
Lookup is cached so the file is parsed once per process.

Schema mirrors `config.yaml`:

    llm:    LLMConfig       # OpenAI-compatible chat completions endpoint
    stt:    STTConfig       # speech-to-text
    tts:    TTSConfig       # text-to-speech
    vision: VisionConfig?   # optional image describer (None disables vision)
    system_prompt: str      # the LLM's system message at conversation start
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STTConfig:
    provider: str
    api_key_env: str
    language_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TTSConfig:
    provider: str
    api_key_env: str
    voice: str


@dataclass(frozen=True)
class VisionProvider:
    """One entry in the vision chain. Fields match config.yaml exactly.

    `kind` selects the backend:
      - "openai" (default): OpenAI-compatible HTTP endpoint at `base_url`
        (works for OpenAI, Kimi, LM Studio's HTTP server, etc.)
      - "mlx": load the model directly in-process via `mlx_vlm`. No HTTP,
        no rate limits, no external app. `base_url` is ignored. `model` is
        the Hugging Face repo id, e.g. `mlx-community/Qwen3-VL-2B-Instruct-4bit`.
    """
    name: str                      # short identifier for logging (e.g. 'kimi')
    model: str
    kind: str = "openai"           # "openai" | "mlx"
    base_url: str = ""             # required when kind == "openai"
    api_key_env: str = ""          # "" means no key required (LM Studio etc.)
    max_tokens: int = 150
    max_image_width: int = 0       # 0 disables downscale
    brevity_suffix: str = ""        # appended to caller prompt; "" disables
    timeout: float = 60.0


@dataclass(frozen=True)
class ShortcatConfig:
    """ShortCat command-palette driver. See ADR-0005."""
    enabled: bool = False
    hotkey: str = "cmd+alt+space"   # pyautogui combo string
    palette_delay_ms: int = 250     # wait after hotkey before typing
    settle_ms: int = 400            # wait after Enter before returning


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool = False
    phrase: str = "hey ava"
    sleep_phrase: str = "go to sleep"
    idle_timeout_seconds: float = 60.0
    ack_text: str = "Ready."
    sleep_ack_text: str = "Sleeping."
    start_awake: bool = True


@dataclass(frozen=True)
class TurnConfig:
    """Conversation-pacing knobs.

    These are the dials you reach for when the bot feels too eager (cuts
    you off) or too sluggish (long awkward pauses).
    """
    # How long the bot waits after you stop talking before it decides your
    # turn is over and sends to the LLM. Lower = snappier; too low = bot
    # interrupts your hesitation pauses.
    user_speech_timeout: float = 0.6
    # How long the EchoSuppressor keeps dropping STT frames after the bot
    # stops speaking. Compensates for speaker-bleed-into-mic without
    # hardware AEC. Raise if the bot keeps "hearing itself".
    echo_holdoff_seconds: float = 1.0


@dataclass(frozen=True)
class HotkeyConfig:
    """Global desktop hotkeys. Uses pynput combo syntax (e.g. <cmd>+<shift>+i)."""
    interrupt: str = "<cmd>+<shift>+i"


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    stt: STTConfig
    tts: TTSConfig
    vision: list[VisionProvider]
    system_prompt: str
    wake_word: WakeWordConfig
    shortcat: ShortcatConfig
    turn: TurnConfig
    hotkey: HotkeyConfig


@lru_cache(maxsize=1)
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}

    vision_raw = data.get("vision") or []
    # Backwards-compat: accept either a list of providers or a single dict.
    if isinstance(vision_raw, dict):
        vision_raw = [vision_raw]
    vision = [VisionProvider(**vp) for vp in vision_raw]

    wake_word_raw = data.get("wake_word") or {}
    wake_word = WakeWordConfig(**wake_word_raw)

    shortcat_raw = data.get("shortcat") or {}
    shortcat = ShortcatConfig(**shortcat_raw)

    turn_raw = data.get("turn") or {}
    turn = TurnConfig(**turn_raw)

    hotkey_raw = data.get("hotkey") or {}
    hotkey = HotkeyConfig(**hotkey_raw)

    return Config(
        llm=LLMConfig(**data["llm"]),
        stt=STTConfig(**data["stt"]),
        tts=TTSConfig(**data["tts"]),
        vision=vision,
        system_prompt=data["system_prompt"].strip(),
        wake_word=wake_word,
        shortcat=shortcat,
        turn=turn,
        hotkey=hotkey,
    )


def require_api_key(env_var: str, *, for_what: str) -> str:
    """Read a required API key from env. Fail loudly with a useful message."""
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"{for_what} requires {env_var} in environment (check server/.env). "
            f"See server/config.yaml to swap providers."
        )
    return key
