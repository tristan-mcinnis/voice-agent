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
    ("tools.external_agents", "external-agent tools"),
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

    # Dynamic external-agent tools: one ``ask_<key>`` per ``agents.consult.<key>``,
    # one ``spawn_<key>`` per ``agents.coding.<key>``. YAML is the single source
    # of truth — adding a new provider is a config edit, not a Python edit.
    try:
        from tools.external_agents import register_dynamic_tools
        register_dynamic_tools()
    except Exception as exc:
        logger.warning(f"tools: dynamic external-agent registration failed — {type(exc).__name__}: {exc}")

    logger.info(f"tools: {len(REGISTRY.all())} registered")


# Callers should use ``REGISTRY.get(name).execute(...)`` directly — there is
# no module-level attribute fallback. One tool-call API across the codebase.
