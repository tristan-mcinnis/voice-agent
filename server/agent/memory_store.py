"""Memory snapshot loader — read-only views of persistent memory files.

USER.md and MEMORY.md are loaded once per session as frozen snapshots.
Even if a sub-agent or tool writes to the files mid-session, the current
session's prompt does not update — only the file on disk does. This
prevents context rot and keeps provider-side prompt caches warm.

Character limits are defined here as the single source of truth. Both
MemoryStore (read side) and MemoryTool (write side) import from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from agent.paths import memories_dir

# Character limits — voice context is small, so keep these tight.
# Single source of truth: MemoryTool imports these for write-side enforcement;
# MemoryStore uses them on read to guard against bloated prompt injection.
USER_CHAR_LIMIT = 1375
MEMORY_CHAR_LIMIT = 2200


class MemoryStore:
    """Loads bounded memory snapshots from disk.

    The base path defaults to ``./.voice-agent/memories/`` (relative to the
    current working directory). Each load method returns the file content
    wrapped in XML-like tags so the LLM can distinguish the cognitive-stack
    layers.

    If a memory file exceeds its character limit (e.g. from a manual edit
    or bug), the loader truncates with a warning — preventing a bloated
    file from silently inflating the system prompt.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path = (base_path or memories_dir()).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_user_md(self) -> str:
        """Return the user-profile snapshot wrapped in ``<user_profile>``."""
        path = self.base_path / "USER.md"
        content = _read_or_default(path, "No user profile data recorded.")
        return f"<user_profile>\n{_truncate_if_needed(content, USER_CHAR_LIMIT, "USER.md")}\n</user_profile>"

    def load_memory_md(self) -> str:
        """Return the project-memory snapshot wrapped in ``<persistent_memory>``."""
        path = self.base_path / "MEMORY.md"
        content = _read_or_default(path, "No persistent project memory.")
        return f"<persistent_memory>\n{_truncate_if_needed(content, MEMORY_CHAR_LIMIT, "MEMORY.md")}\n</persistent_memory>"

    def read_raw(self, filename: str) -> str:
        """Read a raw memory file without wrapping tags."""
        path = self.base_path / filename
        return _read_or_default(path, "")

    def write_raw(self, filename: str, content: str) -> None:
        """Write a memory file directly (used by ``patch_memory`` tool)."""
        (self.base_path / filename).write_text(content, encoding="utf-8")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _read_or_default(path: Path, default: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def _truncate_if_needed(content: str, limit: int, label: str) -> str:
    """Truncate *content* to *limit* chars with a warning log.

    Guard-rail: if a memory file grows past its limit (manual edit, bug),
    we truncate rather than silently injecting the full text into the LLM
    system prompt.
    """
    if len(content) <= limit:
        return content
    logger.warning(
        f"memory_store: {label} is {len(content)} chars, "
        f"truncating to {limit} (limit is {USER_CHAR_LIMIT if 'USER' in label else MEMORY_CHAR_LIMIT})"
    )
    return content[:limit]
