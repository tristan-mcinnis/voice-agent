"""Memory tool — the agent's self-editing brain.

Implements the Hermes-inspired multi-layered memory architecture:
  - USER.md: personal facts, preferences, user context (≤ 1,375 chars)
  - MEMORY.md: project facts, environment, learned lessons (≤ 2,200 chars)

Entries are separated by `§` delimiters for high-density recall. The agent can
add new entries (with pressure checks), replace existing ones via substring
matching, or list current entries with indices.

At session start, PromptBuilder loads these files as frozen snapshots injected
into the system prompt. The stable prefix stays warm in provider-side caches.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from tools.registry import BaseTool, REGISTRY

# Character limits — voice context is small, so keep these tight.
USER_CHAR_LIMIT = 1375
MEMORY_CHAR_LIMIT = 2200


def _agent_memories_dir() -> Path:
    home = os.environ.get("VOICE_AGENT_HOME")
    if not home:
        raise RuntimeError(
            "VOICE_AGENT_HOME is not set. voice_bot.py must be imported first "
            "(it sets the env var at module level)."
        )
    return Path(home) / "memories"


def _limit_for(target: str) -> int:
    return USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT


def _load_entries(path: Path) -> list[str]:
    """Load entries from a §-delimited memory file. Returns empty list if missing."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split("§") if e.strip()]


def _save_entries(path: Path, entries: list[str]) -> None:
    """Write entries back, joined by §. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n§\n".join(entries).strip()
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _total_chars(entries: list[str]) -> int:
    """Sum of entry lengths (excluding delimiters)."""
    return sum(len(e) for e in entries)


class MemoryTool(BaseTool):
    """Edit the agent's persistent memory files — the curated long-term brain.

    The agent should use this to record facts that should survive across
    sessions: user preferences, project context, environment details,
    lessons learned, and important decisions.

    Entry format: each entry is a block of text separated by `§` delimiters.
    The file on disk looks like::

        User prefers TypeScript over Python.
        §
        Project uses PostgreSQL 16 with PostGIS.
        §
        Deployed on Fly.io, us-east-1 region.
    """

    name = "memory"
    description = (
        "Manage the agent's persistent memory files. "
        "Use action='add' to record a new fact (auto-cap at char limits). "
        "Use action='replace' to update an existing fact by matching old_text. "
        "Use action='list' to see current entries with indices. "
        "Target is 'user' (USER.md, personal facts) or 'memory' (MEMORY.md, project facts)."
    )
    parameters = {
        "action": {
            "type": "string",
            "enum": ["add", "replace", "list"],
            "description": "What to do: 'add' a new entry, 'replace' an existing one, or 'list' all entries.",
        },
        "target": {
            "type": "string",
            "enum": ["user", "memory"],
            "description": "Which file: 'user' for USER.md (personal facts, max 1375 chars), 'memory' for MEMORY.md (project facts, max 2200 chars).",
        },
        "content": {
            "type": "string",
            "description": "The new entry text. Required for 'add' and 'replace'.",
        },
        "old_text": {
            "type": "string",
            "description": "Substring to match for 'replace'. The entry containing this text will be replaced with 'content'.",
        },
    }
    required = ["action", "target"]
    category = "system"
    speak_text: Optional[str] = None

    def execute(
        self,
        action: str,
        target: str,
        content: str = "",
        old_text: str = "",
    ) -> dict:
        dir_ = _agent_memories_dir()
        dir_.mkdir(parents=True, exist_ok=True)
        path = dir_ / f"{target.upper()}.md"
        entries = _load_entries(path)
        limit = _limit_for(target)
        label = "USER.md" if target == "user" else "MEMORY.md"

        if action == "list":
            return self._do_list(label, entries, limit)

        if action == "add":
            return self._do_add(label, path, entries, content, limit)

        if action == "replace":
            return self._do_replace(label, path, entries, content, old_text)

        return {"result": f"Unknown action {action!r}. Use 'add', 'replace', or 'list'."}

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _do_list(self, label: str, entries: list[str], limit: int) -> dict:
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

    def _do_add(
        self, label: str, path: Path, entries: list[str], content: str, limit: int
    ) -> dict:
        content = content.strip()
        if not content:
            return {"result": "Cannot add empty content."}

        # Check for near-duplicate.
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
            current_list = self._do_list(label, entries, limit)["result"]
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
        logger.info(f"memory: added {len(content)}-char entry to {label} ({_total_chars(entries)}/{limit})")
        return {
            "result": (
                f"Added to {label} ({_total_chars(entries)}/{limit} chars used). "
                f"Entry: \"{content if len(content) <= 100 else content[:97] + '...'}\""
            ),
            "chars_used": _total_chars(entries),
            "char_limit": limit,
        }

    def _do_replace(
        self,
        label: str,
        path: Path,
        entries: list[str],
        content: str,
        old_text: str,
    ) -> dict:
        content = content.strip()
        old_text = old_text.strip()
        if not content:
            return {"result": "Cannot replace with empty content."}

        if old_text:
            # Substring match — find the entry containing old_text.
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
                        "char_limit": _limit_for(label.split(".")[0].lower()),
                    }
            return {
                "result": (
                    f"No entry in {label} contains {old_text!r}. "
                    "Use action='list' to see current entries and their indices."
                )
            }

        # No old_text — replace by index? Not supported yet. Fall back to list.
        current_list = self._do_list(label, entries, _limit_for(label.split(".")[0].lower()))["result"]
        return {
            "result": (
                "For 'replace', provide 'old_text' — a substring of the entry to replace. "
                f"Here are the current entries:\n\n{current_list}"
            )
        }


# Register on import — canonical name is 'memory'.
REGISTRY.register(MemoryTool)
