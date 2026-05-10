"""Memory patch tool — closes the learning loop for the cognitive stack.

Allows the agent to update USER.md and MEMORY.md so that subsequent
sessions (which load fresh read-only snapshots) inherit the new facts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from tools.registry import BaseTool, REGISTRY


def _agent_memories_dir() -> Path:
    """Return the memories directory from VOICE_AGENT_HOME env var.

    voice_bot.py sets VOICE_AGENT_HOME at import time; all other modules
    read it as the single source of truth.
    """
    home = os.environ.get("VOICE_AGENT_HOME")
    if not home:
        raise RuntimeError(
            "VOICE_AGENT_HOME is not set. voice_bot.py must be imported first "
            "(it sets the env var at module level)."
        )
    return Path(home) / "memories"


class PatchMemoryTool(BaseTool):
    """Update persistent memory files on disk."""

    name = "patch_memory"
    description = (
        "Update the agent's persistent memory files (USER.md or MEMORY.md). "
        "Use this to record long-term facts, user preferences, lessons learned, "
        "or project state that should survive across sessions. "
        "Operations: 'update' replaces the whole file; 'append' adds to the end."
    )
    parameters = {
        "file": {
            "type": "string",
            "description": "Which memory file to patch: 'USER.md' or 'MEMORY.md'",
            "enum": ["USER.md", "MEMORY.md"],
        },
        "insight": {
            "type": "string",
            "description": "The content to write or append",
        },
        "operation": {
            "type": "string",
            "description": "Whether to replace ('update') or add to ('append') the file",
            "enum": ["update", "append"],
            "default": "append",
        },
    }
    required = ["file", "insight"]
    category = "system"
    speak_text: Optional[str] = None

    def execute(self, file: str, insight: str, operation: str = "append") -> dict:
        memories_dir = _agent_memories_dir()
        memories_dir.mkdir(parents=True, exist_ok=True)
        target = memories_dir / file

        if operation == "append":
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            delimiter = "\n§\n" if existing.strip() else ""
            new_content = existing + delimiter + insight
        else:
            new_content = insight

        target.write_text(new_content, encoding="utf-8")
        return {"result": f"Updated {file} ({operation})."}


# Register on import
REGISTRY.register(PatchMemoryTool)
