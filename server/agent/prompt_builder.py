"""Cognitive-stack prompt builder — assembles the frozen system prompt once per session.

Inspired by the Hermes agent architecture:
https://github.com/nousresearch/hermes-agent

The Cognitive Stack is injected in a strict hierarchy of truth:

    Soul → Memory → User → Project Rules → Tools → Skills

By keeping these layers stable and at the very top of the prompt, the
LLM provider can cache the bulk of the system message across turns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.memory_store import MemoryStore

# Fallback identity when ~/.hermes/SOUL.md is missing or empty.
DEFAULT_AGENT_IDENTITY = """You are a friendly multimodal voice assistant.
Your output will be spoken aloud — keep answers brief, conversational,
and free of markdown or special characters."""

# Project-context discovery priority ladder. First match wins.
_CONTEXT_PRIORITY = [
    ".hermes.md",
    "HERMES.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
]


def _get_agent_home() -> Path:
    return Path.home() / ".hermes"


class PromptBuilder:
    """Orchestrates the multi-layer system-prompt assembly."""

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        registry=None,
        default_identity: Optional[str] = None,
    ) -> None:
        self.memory_store = memory_store or MemoryStore()
        self.registry = registry
        self.default_identity = default_identity or DEFAULT_AGENT_IDENTITY

    # ------------------------------------------------------------------
    # Layer loaders
    # ------------------------------------------------------------------

    def load_soul_md(self) -> str:
        """Load the agent's identity from ``~/.hermes/SOUL.md``.

        Falls back to ``default_identity`` (usually the config's
        ``system_prompt``) when the file is missing or contains only
        template comments.
        """
        soul_path = _get_agent_home() / "SOUL.md"
        if soul_path.exists():
            content = soul_path.read_text(encoding="utf-8").strip()
            # Heuristic: ignore files that are only HTML comments, markdown
            # headings, or whitespace. A file with just a title is still empty.
            substantive: list[str] = []
            in_comment = False
            for ln in content.splitlines():
                stripped = ln.strip()
                if not stripped:
                    continue
                if stripped.startswith("<!--"):
                    in_comment = True
                if in_comment:
                    if "-->" in stripped:
                        in_comment = False
                    continue
                if stripped.startswith("#"):
                    continue
                substantive.append(ln)
            if substantive:
                return content
        return self.default_identity

    def build_context_files_prompt(self, cwd: Path) -> str:
        """Scan *cwd* for project-specific rule files.

        Uses a "first match wins" priority ladder to prevent rule
        conflicts. Returns an empty string when none are found.
        """
        for filename in _CONTEXT_PRIORITY:
            target = cwd / filename
            if target.exists():
                return f"\n# Project Rules ({filename})\n{target.read_text(encoding='utf-8')}"
        return ""

    def get_tool_descriptions(self) -> str:
        """Return the category-grouped tool inventory from the registry."""
        if self.registry is None:
            return ""
        return self.registry.capabilities_summary()

    def get_relevant_skills_index(self, _user_input: str = "") -> str:
        """List available skills from ``~/.hermes/skills/``.

        Each skill is a directory containing a ``SKILL.md`` file. The
        index extracts the first substantive line as a one-line summary.
        """
        skills_dir = _get_agent_home() / "skills"
        if not skills_dir.exists():
            return ""

        entries: list[str] = []
        for skill_path in sorted(skills_dir.iterdir()):
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue
            desc = _extract_first_line(skill_md.read_text(encoding="utf-8"))
            if desc:
                entries.append(f"- {skill_path.name}: {desc}")

        if not entries:
            return ""
        return "Available skills:\n" + "\n".join(entries)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble_system_prompt(
        self,
        soul: str,
        user: str,
        memory: str,
        context: str,
        tools: str,
        skills: str,
    ) -> str:
        """Concatenate layers into the final frozen system prompt."""
        parts: list[str] = []

        # Soul / Identity
        if soul:
            # Backward-compat: if the soul (or fallback default_identity)
            # still contains the legacy {tool_capabilities} placeholder,
            # substitute it inline so existing configs don't break.
            if "{tool_capabilities}" in soul and tools:
                soul = soul.replace("{tool_capabilities}", tools)
                tools = ""
            parts.append(soul)

        # Cognitive stack — Memory and User profile
        if memory:
            parts.append(memory)
        if user:
            parts.append(user)

        # Project-specific rules
        if context:
            parts.append(context)

        # Tool inventory
        if tools:
            parts.append(
                "\n# Available Tools\n"
                "You have access to the following tools, by category:\n"
                f"{tools}"
            )

        # Skills index
        if skills:
            parts.append(f"\n# Skills\n{skills}")

        return "\n\n".join(parts)

    def build(self, user_input: str = "", cwd: Optional[Path] = None) -> str:
        """Run the full assembly sequence once per session.

        Args:
            user_input: The user's first message (used for skill relevance
                filtering in future iterations).
            cwd: Directory to scan for project context files. Defaults to
                the current working directory.
        """
        soul = self.load_soul_md()
        user_profile = self.memory_store.load_user_md()
        project_memory = self.memory_store.load_memory_md()
        project_context = self.build_context_files_prompt(cwd or Path.cwd())
        tool_docs = self.get_tool_descriptions()
        skills_index = self.get_relevant_skills_index(user_input)

        return self.assemble_system_prompt(
            soul=soul,
            user=user_profile,
            memory=project_memory,
            context=project_context,
            tools=tool_docs,
            skills=skills_index,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_first_line(text: str) -> str:
    """Return the first non-empty, non-comment line, stripped of markdown heading markers."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<!--"):
            return stripped.lstrip("# ").strip()
    return ""
