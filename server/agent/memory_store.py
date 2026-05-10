"""Memory snapshot loader — read-only views of persistent memory files.

USER.md and MEMORY.md are loaded once per session as frozen snapshots.
Even if a sub-agent or tool writes to the files mid-session, the current
session's prompt does not update — only the file on disk does. This
prevents context rot and keeps provider-side prompt caches warm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.paths import memories_dir


class MemoryStore:
    """Loads bounded memory snapshots from disk.

    The base path defaults to ``./.voice-agent/memories/`` (relative to the
    current working directory). Each load method returns the file content
    wrapped in XML-like tags so the LLM can distinguish the cognitive-stack
    layers.
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
        return f"<user_profile>\n{content}\n</user_profile>"

    def load_memory_md(self) -> str:
        """Return the project-memory snapshot wrapped in ``<persistent_memory>``."""
        path = self.base_path / "MEMORY.md"
        content = _read_or_default(path, "No persistent project memory.")
        return f"<persistent_memory>\n{content}\n</persistent_memory>"

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
