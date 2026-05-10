"""Session history search — raw conversation recall from past session logs.

This is the "Raw Recall" layer of the memory architecture. While USER.md and
MEMORY.md hold curated long-term facts, the session logs hold every turn. This
tool lets the agent search across past conversations.

Implementation: uses ripgrep on the JSONL session log directory, then parses
matching lines to extract user-spoke and bot-spoke events. Fast enough for
thousands of log lines; no SQLite dependency needed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from tools.registry import REGISTRY, BaseTool

_DEFAULT_LOG_DIR = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))


def _search_logs(query: str, max_results: int = 5) -> str:
    """Search session logs for `query` and return relevant conversation blocks."""
    log_dir = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))
    if not log_dir.exists():
        return f"No session logs found at {log_dir}. Start a voice session first."

    # Collect all JSONL files, newest first.
    jsonl_files = sorted(
        log_dir.glob("session-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not jsonl_files:
        return f"No session log files found in {log_dir}. Start a voice session first."

    # Use ripgrep when available; fall back to Python otherwise.
    rg = subprocess.run(
        [
            "rg", "--no-heading", "--with-filename",
            "--max-count", str(max_results),
            "-e", query, str(log_dir),
        ],
        capture_output=True, text=True, timeout=10.0,
    )

    matches: list[tuple[str, dict]] = []
    if rg.returncode in (0, 1) and rg.stdout.strip():
        for line in rg.stdout.strip().split("\n"):
            if ":" not in line:
                continue
            try:
                file_path, _, json_str = line.partition(":")
                parsed = json.loads(json_str)
                matches.append((file_path, parsed))
            except (json.JSONDecodeError, ValueError):
                continue
    else:
        # Fallback: Python search through recent files.
        for f in jsonl_files[:5]:  # Only search 5 most recent files
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if query.lower() in line.lower():
                        parsed = json.loads(line)
                        matches.append((f.name, parsed))
                        if len(matches) >= max_results * 2:
                            break
            except Exception:
                continue
            if len(matches) >= max_results * 2:
                break

    if not matches:
        return f"No past conversations matching {query!r}."

    # Collect conversation blocks: group by session, show user-speak + bot-speak pairs.
    # For voice output, keep it brief — one block per match.
    blocks: list[str] = []
    sessions_seen: set[str] = set()
    for file_name, event in matches:
        sid = event.get("session_id", "?")
        kind = event.get("event", "")
        text = event.get("text", "") or ""
        iso = event.get("iso", "")[:19] or ""

        if kind not in ("user-spoke", "bot-spoke"):
            continue
        if not text.strip():
            continue

        speaker = "You" if kind == "user-spoke" else "Bot"
        if len(text) > 200:
            text = text[:197] + "..."

        key = f"{sid}:{iso[:10]}"
        if key not in sessions_seen:
            sessions_seen.add(key)
            if len(blocks) < max_results:
                blocks.append(f"[{iso[:10]}] {speaker}: {text}")

    if not blocks:
        return f"No conversation turns matching {query!r}."

    header = f"Found {len(blocks)} conversation block{'s' if len(blocks) != 1 else ''} matching {query!r}:"
    return header + "\n" + "\n".join(blocks)


class SearchHistoryTool(BaseTool):
    """Search past voice conversations for specific topics, facts, or discussions.

    This searches the raw session logs (JSONL) — not the curated memory files.
    Use this when the user asks "what did we talk about last time?" or
    "have we discussed X before?"
    """

    name = "search_history"
    category = "system"
    speak_text = "Searching through past conversations."
    description = (
        "Search past voice conversation logs for specific topics. "
        "Use when the user asks about previous discussions, things they "
        "mentioned before, or any conversation from a past session. "
        "This searches raw session transcripts, not the curated memory files."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "What to search for — a keyword, phrase, or topic. Case-insensitive.",
        },
        "max_results": {
            "type": "integer",
            "description": "Max conversation blocks to return. Default 5.",
        },
    }
    required = ["query"]

    def execute(self, query: str, max_results: int = 5) -> str:
        return _search_logs(query, max_results)


# Register on import
REGISTRY.register(SearchHistoryTool)
