"""Memory layer — single source of truth for persistent memory file format and operations.

Owns:

- The § delimiter format
- Character limits (USER_CHAR_LIMIT, MEMORY_CHAR_LIMIT)
- Read: loading entries into prompt-ready wrapped text
- Write: add / replace / list with near-duplicate detection and pressure checks

Lives in ``agent/`` because it is the agent's persistent-memory primitive.
``tools/memory.py`` is the thin tool-registry adapter on top of it.

Usage::

    from agent.memory_layer import MemoryLayer

    layer = MemoryLayer(base_path=Path(".voice-agent/memories"))

    # Read side (what PromptBuilder uses):
    user_snapshot = layer.load_user_prompt()
    memory_snapshot = layer.load_memory_prompt()

    # Write side (what MemoryTool delegates to):
    result = layer.add("user", "User prefers TypeScript.")
    result = layer.replace("user", old_text="TypeScript", content="User prefers Rust.")
    result = layer.list_entries("user")
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

from agent.paths import memories_dir as _agent_memories_dir


def _default_memories_dir() -> Path:
    """Canonical memories directory via ``agent.paths``.

    Falls back to ``./.voice-agent/memories`` when ``VOICE_AGENT_HOME`` is
    unset — tests construct a ``MemoryLayer()`` before
    ``voice_bot.set_agent_home()`` runs.
    """
    try:
        return _agent_memories_dir()
    except RuntimeError:
        d = Path(".voice-agent") / "memories"
        d.mkdir(parents=True, exist_ok=True)
        return d.expanduser().resolve()

# Character limits — voice context is small, so keep these tight.
USER_CHAR_LIMIT = 1375
MEMORY_CHAR_LIMIT = 2200


# ---------------------------------------------------------------------------
# Shared accessor — one MemoryLayer per process.
#
# Without this, ``MemoryTool`` and ``PromptBuilder`` could construct separate
# instances against different base paths and writes from the tool would never
# appear in the prompt snapshot.
# ---------------------------------------------------------------------------

_shared_layer: "MemoryLayer | None" = None


def get_memory_layer() -> "MemoryLayer":
    """Return the process-wide ``MemoryLayer``, constructed lazily."""
    global _shared_layer
    if _shared_layer is None:
        _shared_layer = MemoryLayer()
    return _shared_layer


def reset_memory_layer() -> None:
    """Clear the shared layer. Tests that re-point ``VOICE_AGENT_HOME`` call this."""
    global _shared_layer
    _shared_layer = None


def _limit_for(target: str) -> int:
    return USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT


# ---------------------------------------------------------------------------
# File-format helpers (§-delimited entries)
# ---------------------------------------------------------------------------

def _load_entries(path: Path) -> list[str]:
    """Load entries from a §-delimited memory file. Returns empty list if missing."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split("§") if e.strip()]


def _save_entries(path: Path, entries: list[str]) -> None:
    """Write entries back atomically — temp file, fsync, rename.

    A non-atomic ``write_text`` could leave the file empty or truncated if
    the process is killed mid-write, and the next session would load an
    empty MEMORY.md / USER.md into the system prompt. The temp-then-rename
    pattern means the on-disk file is either fully the old version or
    fully the new one — never partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n§\n".join(entries).strip()
    if content:
        content += "\n"

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        # Clean up the temp file so we don't litter on errors.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _total_chars(entries: list[str]) -> int:
    """Sum of entry lengths (excluding delimiters)."""
    return sum(len(e) for e in entries)


# ---------------------------------------------------------------------------
# MemoryLayer — unified read + write
# ---------------------------------------------------------------------------

class MemoryLayer:
    """Owns the file format, character limits, and operations for persistent memory.

    The PromptBuilder calls the read methods to get prompt-ready snapshots.
    The MemoryTool delegates to the write methods.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path = (base_path or _default_memories_dir()).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        # The LLM can dispatch multiple tool calls in a single turn; each
        # tool runs in its own ``asyncio.to_thread`` worker. Without a lock,
        # two concurrent memory.add calls would each read the same starting
        # state and one update would be silently lost.
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read API (for PromptBuilder)
    # ------------------------------------------------------------------

    def load_user_prompt(self) -> str:
        """Return USER.md wrapped in <user_profile> tags for the system prompt."""
        path = self.base_path / "USER.md"
        content = _read_or_default(path, "No user profile data recorded.")
        return f"<user_profile>\n{_truncate_if_needed(content, USER_CHAR_LIMIT, "USER.md")}\n</user_profile>"

    def load_memory_prompt(self) -> str:
        """Return MEMORY.md wrapped in <persistent_memory> tags for the system prompt."""
        path = self.base_path / "MEMORY.md"
        content = _read_or_default(path, "No persistent project memory.")
        return f"<persistent_memory>\n{_truncate_if_needed(content, MEMORY_CHAR_LIMIT, "MEMORY.md")}\n</persistent_memory>"

    # ------------------------------------------------------------------
    # Write API (for MemoryTool)
    # ------------------------------------------------------------------

    def list_entries(self, target: str) -> dict:
        """Return indexed entries and char-usage info for ``target`` (user|memory)."""
        path = self._path_for(target)
        entries = _load_entries(path)
        limit = _limit_for(target)
        label = self._label_for(target)

        if not entries:
            return {
                "result": f"{label} is empty ({_total_chars(entries)}/{limit} chars used).",
                "entries": [],
                "chars_used": 0,
                "char_limit": limit,
            }

        indexed = {str(i): entry for i, entry in enumerate(entries)}
        lines = [f"{label} ({_total_chars(entries)}/{limit} chars used):"]
        for i, entry in enumerate(entries):
            preview = entry if len(entry) <= 80 else entry[:77] + "..."
            lines.append(f"  [{i}] {preview}")
        return {
            "result": "\n".join(lines),
            "entries": indexed,
            "chars_used": _total_chars(entries),
            "char_limit": limit,
        }

    def add(self, target: str, content: str) -> dict:
        """Add an entry with near-duplicate detection and pressure check.

        The read-then-write is serialised by ``_write_lock`` so two
        concurrent tool calls can't read the same starting state and
        clobber each other's updates.
        """
        with self._write_lock:
            return self._add_locked(target, content)

    def _add_locked(self, target: str, content: str) -> dict:
        path = self._path_for(target)
        entries = _load_entries(path)
        limit = _limit_for(target)
        label = self._label_for(target)

        content = content.strip()
        if not content:
            return {"result": "Cannot add empty content."}

        # Near-duplicate check.
        for entry in entries:
            if content in entry or entry in content:
                return {
                    "result": (
                        f"Similar entry already exists in {label}: "
                        f"\"{entry if len(entry) <= 100 else entry[:97] + '...'}\". "
                        "Use action='replace' to update it, or phrase differently."
                    )
                }

        new_total = _total_chars(entries) + len(content)
        if new_total > limit:
            over_by = new_total - limit
            current_list = self.list_entries(target)["result"]
            return {
                "result": (
                    f"Memory pressure: adding this entry ({len(content)} chars) would exceed "
                    f"{label}'s {limit}-char limit by {over_by} chars "
                    f"({_total_chars(entries)}/{limit} used). "
                    f"Use action='list' to review entries and action='replace' to swap "
                    f"a less important one, or shorten your content.\n\n{current_list}"
                )
            }

        entries.append(content)
        _save_entries(path, entries)
        logger.info(
            f"memory: added {len(content)}-char entry to {label} "
            f"({_total_chars(entries)}/{limit})"
        )
        return {
            "result": (
                f"Added to {label} ({_total_chars(entries)}/{limit} chars used). "
                f"Entry: \"{content if len(content) <= 100 else content[:97] + '...'}\""
            ),
            "chars_used": _total_chars(entries),
            "char_limit": limit,
        }

    def replace(self, target: str, content: str, old_text: str = "") -> dict:
        """Replace an entry matched by ``old_text`` substring.

        Serialised by ``_write_lock`` against concurrent add/replace.
        """
        with self._write_lock:
            return self._replace_locked(target, content, old_text)

    def _replace_locked(
        self, target: str, content: str, old_text: str = "",
    ) -> dict:
        path = self._path_for(target)
        entries = _load_entries(path)
        label = self._label_for(target)
        limit = _limit_for(target)

        content = content.strip()
        old_text = old_text.strip()
        if not content:
            return {"result": "Cannot replace with empty content."}

        if old_text:
            for i, entry in enumerate(entries):
                if old_text in entry:
                    entries[i] = content
                    _save_entries(path, entries)
                    logger.info(f"memory: replaced entry {i} in {label}")
                    return {
                        "result": (
                            f"Replaced entry [{i}] in {label}: "
                            f"\"{entry if len(entry) <= 80 else entry[:77] + '...'}\" "
                            f"→ \"{content if len(content) <= 80 else content[:77] + '...'}\""
                        ),
                        "replaced_index": i,
                        "chars_used": _total_chars(entries),
                        "char_limit": limit,
                    }
            return {
                "result": (
                    f"No entry in {label} contains {old_text!r}. "
                    "Use action='list' to see current entries and their indices."
                )
            }

        # No old_text — fall back to list.
        current_list = self.list_entries(target)["result"]
        return {
            "result": (
                "For 'replace', provide 'old_text' — a substring of the entry to replace. "
                f"Here are the current entries:\n\n{current_list}"
            )
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, target: str) -> Path:
        return self.base_path / f"{target.upper()}.md"

    @staticmethod
    def _label_for(target: str) -> str:
        return "USER.md" if target == "user" else "MEMORY.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_or_default(path: Path, default: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def _truncate_if_needed(content: str, limit: int, label: str) -> str:
    """Truncate *content* to *limit* chars with a warning log."""
    if len(content) <= limit:
        return content
    logger.warning(
        f"memory_layer: {label} is {len(content)} chars, "
        f"truncating to {limit}"
    )
    return content[:limit]
