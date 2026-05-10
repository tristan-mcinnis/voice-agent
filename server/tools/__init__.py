"""Voice-bot tools — multimodal function-call capabilities.

Importing this package registers all tools on the global REGISTRY.
The registry, base class, and handler wiring live in `registry.py`.
Domain-specific tool implementations live in:
  - files.py     — file system operations, shell, search
  - desktop.py   — macOS desktop automation (clipboard, browser, capture, terminal, system info)
  - web.py       — web search and external APIs
  - vision.py    — image description chain (no tool classes, called by desktop.py)

Backward-compat callables (e.g. ``tools.read_file(...)``) delegate to the
canonical ``BaseTool.execute()`` implementations via the REGISTRY.
"""

from tools.registry import BaseTool, REGISTRY, ToolRegistry  # noqa: F401

# Side-effect imports: each module registers its tools on REGISTRY at import time.
import tools.files     # noqa: F401
import tools.capture   # noqa: F401
import tools.desktop   # noqa: F401
import tools.computer_use  # noqa: F401
import tools.grounding_vision  # noqa: F401
import tools.web       # noqa: F401
import tools.memory    # noqa: F401
import tools.search_history  # noqa: F401

# Reassign tools whose names collide with submodule names (memory → tools/memory.py,
# search_history → tools/search_history.py). Submodule imports bind the module object
# to the package namespace, which shadows __getattr__. We override those here.
# All other tools resolve via __getattr__ below.
memory = REGISTRY.get("memory").execute
search_history = REGISTRY.get("search_history").execute
patch_memory = REGISTRY.get("memory").execute  # legacy alias


# ---------------------------------------------------------------------------
# Module-level __getattr__ — auto-resolves tool names from the registry.
# No more per-tool boilerplate. Adding a tool = one file; __init__.py never changes.
# ---------------------------------------------------------------------------

# Attribute name → registry tool name. Only needed for aliases (e.g. the
# compat name differs from the tool's `name` field). Most names are 1:1.
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
