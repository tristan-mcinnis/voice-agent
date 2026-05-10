"""Memory tool — thin adapter over MemoryLayer for the tool registry.

The MemoryLayer module owns the file format, character limits, and
operations. This tool class translates the LLM-facing action/target/content
parameter schema into MemoryLayer method calls.

Implements the Hermes-inspired multi-layered memory architecture:
  - USER.md: personal facts, preferences, user context (≤ 1,375 chars)
  - MEMORY.md: project facts, environment, learned lessons (≤ 2,200 chars)

PromptBuilder reads these files as frozen snapshots via MemoryLayer.
"""

from __future__ import annotations

from typing import Optional

from tools.registry import BaseTool, REGISTRY
from tools.memory_layer import MemoryLayer, USER_CHAR_LIMIT, MEMORY_CHAR_LIMIT  # noqa: F401  # re-exported

# Module-level instance — shared by the tool class and PromptBuilder.
_layer = MemoryLayer()


class MemoryTool(BaseTool):
    """Edit the agent's persistent memory files — the curated long-term brain.

    The agent should use this to record facts that should survive across
    sessions: user preferences, project context, environment details,
    lessons learned, and important decisions.
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
    guidance = """
## Memory Tool Usage

You have a `memory` tool that manages two persistent files:

- **USER.md** — facts about the user (preferences, background, style). Max 1,375 characters.
- **MEMORY.md** — project/environment facts (tech stack, decisions, state). Max 2,200 characters.

Entries are separated by `§` delimiters. The actions are:

- **list** — See current entries and their indices. Use this FIRST before adding or replacing, so you know what's already stored.
- **add** — Append a new entry. The tool checks for near-duplicates and enforces the character limit. If memory pressure is reported, use `replace` to swap a less-important entry instead of adding.
- **replace** — Update an existing entry by providing the `old_text` substring to match. The entry containing that text is replaced with your `content`.

### When to use memory

Record facts that should survive across sessions:
- User preferences: "User prefers short answers without follow-up questions."
- Project context: "Working on voice-agent repo — Pipecat bot with Soniox STT/TTS."
- Decisions made: "Chose SQLite FTS5 for session search over Chroma."
- Lessons learned: "DeepSeek's thinking mode adds 2s latency — keep disabled for voice."

### When NOT to use memory

- Transient facts about the current conversation only.
- Information already present in the current session context.
- Questions the user asked (those are in session logs via `search_history`)."""

    def execute(
        self,
        action: str,
        target: str,
        content: str = "",
        old_text: str = "",
    ) -> dict:
        if action == "list":
            return _layer.list_entries(target)
        if action == "add":
            return _layer.add(target, content)
        if action == "replace":
            return _layer.replace(target, content, old_text)
        return {"result": f"Unknown action {action!r}. Use 'add', 'replace', or 'list'."}


# Register on import — canonical name is 'memory'.
REGISTRY.register(MemoryTool)
