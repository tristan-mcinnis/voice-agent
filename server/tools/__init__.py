"""Voice-bot tools — multimodal function-call capabilities.

The registry, base class, and handler wiring live in `registry.py`.
Domain-specific tool implementations live in:
  - files.py     — file system operations, shell, search
  - capture.py   — vision capture (screenshot, webcam, window, region, display)
  - desktop.py   — macOS desktop automation (clipboard, browser, terminal, system info)
  - computer_use.py — computer-use tools (AX listing, clicks, keys, Shortcat)
  - grounding_vision.py — vision-based UI element grounding
  - web.py       — web search and external APIs
  - memory.py    — persistent memory management
  - search_history.py — past session conversation search

Call ``register_all()`` once at startup to import and register every tool.
Importing this package alone does NOT trigger registration — the caller
controls when tools are wired.
"""

from __future__ import annotations

from loguru import logger

from tools.registry import BaseTool, REGISTRY, ToolRegistry  # noqa: F401

# ---------------------------------------------------------------------------
# Explicit registration — call register_all() once at startup.
# No more import-time side effects. Missing dependencies disable individual
# tools; they don't crash the entire package.
# ---------------------------------------------------------------------------

# (module, friendly_name) pairs. Each module registers its tools via
# @REGISTRY.register decorators that fire at class-definition time.
_TOOL_MODULES: list[tuple[str, str]] = [
    ("tools.files", "file tools"),
    ("tools.capture", "capture tools"),
    ("tools.desktop", "desktop tools"),
    ("tools.computer_use", "computer-use tools"),
    ("tools.grounding_vision", "grounding-vision tools"),
    ("tools.web", "web tools"),
    ("tools.memory", "memory tools"),
    ("tools.search_history", "search-history tools"),
]


def register_all() -> None:
    """Import every tool module, logging which succeed and which are skipped.

    Call once at startup (e.g. from ``voice_bot.build_components()``).
    After this returns, ``REGISTRY`` is populated and ``register_handlers()``
    can wire tools onto the LLM.
    """
    for module_name, label in _TOOL_MODULES:
        try:
            __import__(module_name)
        except ImportError as exc:
            logger.warning(f"tools: skipping {label} ({module_name}) — import failed: {exc}")
        except Exception as exc:
            logger.warning(f"tools: skipping {label} ({module_name}) — {type(exc).__name__}: {exc}")
    logger.info(f"tools: {len(REGISTRY.all())} registered")


# ---------------------------------------------------------------------------
# Module-level __getattr__ — auto-resolves tool names from the registry.
# For backward compat: `tools.read_file(...)` → `REGISTRY.get("read_file").execute(...)`
# ---------------------------------------------------------------------------

# Attribute name → registry tool name. Only needed for aliases.
_COMPAT_ALIASES: dict[str, str] = {
    "patch_file": "patch",
}


def __getattr__(name: str):
    """Resolve `tools.<name>(...)` → `REGISTRY.get(<name>).execute(...)`."""
    if name.startswith("_"):
        raise AttributeError(name)
    tool_name = _COMPAT_ALIASES.get(name, name)
    try:
        return REGISTRY.get(tool_name).execute
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(no tool registered as {tool_name!r})"
        ) from None
