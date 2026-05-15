"""Single source of truth for the agent's home directory and subdirectories.

``voice_bot.py`` sets the ``VOICE_AGENT_HOME`` env var at import time
(before any other module reads it). Every other module calls the functions
here instead of reading the env var directly.

Why a module rather than scattered ``os.getenv`` calls: change the env var
name once, the error message once, and the subdirectory layout once —
nothing else drifts.
"""

from __future__ import annotations

import os
from pathlib import Path


def agent_home() -> Path:
    """Return the agent's data directory (``.voice-agent/``).

    ``voice_bot.py`` must be imported before the first call so the env var
    is set.
    """
    home = os.environ.get("VOICE_AGENT_HOME")
    if not home:
        raise RuntimeError(
            "VOICE_AGENT_HOME is not set. voice_bot.py must be imported first "
            "(it sets the env var at module level)."
        )
    return Path(home).expanduser().resolve()


def memories_dir() -> Path:
    """Return the memories subdirectory (``.voice-agent/memories/``)."""
    d = agent_home() / "memories"
    d.mkdir(parents=True, exist_ok=True)
    return d


def skills_dir() -> Path:
    """Return the skills subdirectory (``.voice-agent/skills/``)."""
    return agent_home() / "skills"


def memory_tree_dir() -> Path:
    """Return the memory-tree leaf store (``.voice-agent/memories/tree/``).

    Holds individual leaf files — longer recollections that don't belong
    in the always-loaded USER.md / MEMORY.md curated index. The agent
    drills in on demand via the ``memory_tree`` tool.
    """
    d = memories_dir() / "tree"
    d.mkdir(parents=True, exist_ok=True)
    return d
