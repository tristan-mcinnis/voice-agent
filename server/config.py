"""Voice-bot runtime configuration loader.

Reads `config.yaml` (next to this file by default) into typed dataclasses.

Usage::

    from config import init_config, get_config

    init_config()                    # called once at startup
    cfg = get_config()               # anywhere after init

    # Or parse a file without caching (tests, tools that need a fresh read):
    from config import load_config
    cfg = load_config(Path("test_config.yaml"))

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
    # When true, TTS starts speaking at clause boundaries (,;:—) instead of
    # waiting for full sentences. Cuts first-audio latency on long answers
    # at the cost of slightly choppier mid-sentence prosody.
    stream_clauses: bool = False
    # Minimum word count before a clause is released. Stops one-word clauses
    # like "Hi," from being spoken in isolation.
    clause_min_words: int = 3


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
    # When true, use Pipecat's smart-turn-v3 ONNX model to detect end of
    # user turn instead of fixed silence timeout. Cuts ~300–500ms of dead
    # air on definite-end utterances. Requires:
    #   pip install pipecat-ai[local-smart-turn]
    # Falls back to speech-timeout if the model fails to load.
    smart_turn_enabled: bool = False
    # How long to wait for both Soniox WebSockets (STT + TTS) to connect
    # before logging which side(s) failed. Without this the bot can hang
    # silently forever on a bad API key or network blip. The bot still
    # logs the failure but continues; this is observability, not
    # auto-recovery.
    connection_timeout_seconds: float = 15.0


@dataclass(frozen=True)
class HotkeyConfig:
    """Global desktop hotkeys. Uses pynput combo syntax (e.g. <cmd>+<shift>+i)."""
    interrupt: str = "<cmd>+<shift>+i"


@dataclass(frozen=True)
class MCPServerConfig:
    """One MCP-over-stdio server to expose to the tool registry.

    At startup, ``tools.mcp_tools`` spawns each enabled entry, runs the
    JSON-RPC handshake, calls ``tools/list``, and registers each discovered
    tool as ``{name}__{tool_name}``. Connection failures are non-fatal —
    the bot logs a warning and stays usable.

    Fields:
      name      Short identifier; used as the registry-tool prefix.
                Must be a valid Python identifier slice (letters / digits
                / underscores) so the resulting tool names match the
                OpenAI function-name regex.
      command   Executable name (resolved against PATH) or absolute path.
      args      Extra argv tokens passed to the server.
      env       Additional environment variables merged on top of os.environ
                when spawning the server (e.g. API keys, config paths).
      enabled   Skip this server when false. Use to keep config templates
                ready-to-go without paying spawn cost.
      category  Registry category for the auto-generated capability summary
                in the system prompt.
      timeout   Seconds per RPC round-trip.
      include   If non-empty, only register these tool names.
      exclude   Never register these tool names. Takes precedence over include.
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    category: str = "misc"
    timeout: float = 30.0
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComputerUseConfig:
    """Actuation backend for click/type/key/scroll/move.

    `native` (default) uses pyautogui — works everywhere, steals cursor/focus.
    `cua` routes through `cua-driver`, a Swift MCP server from trycua/cua that
    drives macOS without stealing cursor or focus. Install with:
        brew tap trycua/cua && brew install cua-driver

    If `backend: cua` is set but the binary is missing or the handshake fails,
    each tool logs a warning and falls back to native — the bot stays usable.
    """
    backend: str = "native"            # "native" | "cua"
    cua_binary: str = "cua-driver"     # name on $PATH, or absolute path
    cua_timeout: float = 10.0          # seconds per cua-driver RPC


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
    computer_use: ComputerUseConfig
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)


def _parse_config(path: Path) -> Config:
    """Parse a YAML file into a Config. Pure parser — no caching."""
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

    computer_use_raw = data.get("computer_use") or {}
    computer_use = ComputerUseConfig(**computer_use_raw)

    mcp_servers_raw = data.get("mcp_servers") or []
    mcp_servers = [MCPServerConfig(**entry) for entry in mcp_servers_raw]

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
        computer_use=computer_use,
        mcp_servers=mcp_servers,
    )


# ---------------------------------------------------------------------------
# Explicit single-load: init_config() once at startup, get_config() everywhere else.
# No lru_cache footgun — the cache lifetime is visible.
# ---------------------------------------------------------------------------

_config: Config | None = None


def init_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load config once at startup. Idempotent — subsequent calls return the first result.

    Call this in ``main()`` before anything touches config-dependent code.
    """
    global _config
    if _config is None:
        _config = _parse_config(path)
    return _config


def get_config() -> Config:
    """Return the config loaded by ``init_config()``.

    Raises RuntimeError if ``init_config()`` hasn't been called yet.
    """
    if _config is None:
        raise RuntimeError(
            "Config not initialised. Call init_config() at startup first."
        )
    return _config


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Parse a config file. Stateless — no caching.

    Prefer ``init_config()`` + ``get_config()`` for the normal startup path.
    Use this directly for tests, tools, or alternate config files.
    """
    return _parse_config(path)


def require_api_key(env_var: str, *, for_what: str) -> str:
    """Read a required API key from env. Fail loudly with a useful message."""
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"{for_what} requires {env_var} in environment (check server/.env). "
            f"See server/config.yaml to swap providers."
        )
    return key
