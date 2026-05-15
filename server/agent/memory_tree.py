"""Memory Tree — flat leaf store for long recollections.

The always-loaded ``MEMORY.md`` / ``USER.md`` snapshot stays small and
frozen so the cache prefix doesn't shift turn to turn. Anything longer
than a one-liner — a meeting summary, a debugging post-mortem, a
multi-paragraph context dump — lives here as a leaf file. The agent
calls the ``memory_tree`` tool to write a leaf, search across leaves,
or read one back in full.

Inspired by OpenHuman's Memory Tree, kept deliberately flat: every leaf
sits at the same depth, search is a single substring/token scan, and
there is no background summarizer. If flat leaves prove insufficient,
this module is small enough that hierarchical rollups can be added
later without touching callers.

Latency profile (the constraint that picked this design): writing a
leaf is one ``write()`` call; searching ``N`` leaves is ``O(N · L)``
token scans where ``L`` is leaf size. With 1 KB leaves we comfortably
hit thousands without nearing voice-pipeline budgets.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from agent.paths import memory_tree_dir


# Leaf size cap — voice context is small; long enough for a focused
# topic, short enough that one search hit fits in a single tool reply.
LEAF_CHAR_LIMIT = 4000

_TITLE_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Leaf:
    """One leaf file's metadata + content. Returned by read/search."""
    leaf_id: str        # filename without .md
    title: str
    content: str
    created_at: float   # unix ts from mtime


# ---------------------------------------------------------------------------
# Slug + id construction
# ---------------------------------------------------------------------------

def _slugify(title: str, max_len: int = 60) -> str:
    """Return a kebab-case slug safe for filenames."""
    base = title.lower().strip()
    base = _TITLE_SLUG_RE.sub("-", base).strip("-")
    return (base or "untitled")[:max_len]


def _new_leaf_id(title: str) -> str:
    """Date-prefixed slug — sorts chronologically and stays human-readable."""
    return f"{time.strftime('%Y-%m-%d')}-{_slugify(title)}"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryTree:
    """Flat leaf store rooted at ``.voice-agent/memories/tree/``.

    All public methods return data; nothing raises on the happy path so
    they can be wired straight to a tool registry.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path = base_path or memory_tree_dir()
        self.base_path.mkdir(parents=True, exist_ok=True)

    # -- read API ------------------------------------------------------

    def list_leaves(self, limit: int = 50) -> list[dict]:
        """Return [{leaf_id, title, created_at, preview}, …] newest first."""
        files = sorted(
            self.base_path.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:max(1, limit)]
        out: list[dict] = []
        for path in files:
            leaf = self._read_leaf(path)
            preview = leaf.content[:120].replace("\n", " ")
            if len(leaf.content) > 120:
                preview += "…"
            out.append({
                "leaf_id": leaf.leaf_id,
                "title": leaf.title,
                "created_at": leaf.created_at,
                "preview": preview,
            })
        return out

    def read(self, leaf_id: str) -> Optional[Leaf]:
        path = self.base_path / f"{leaf_id}.md"
        if not path.exists():
            return None
        return self._read_leaf(path)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Token-overlap ranking. Returns [{leaf_id, title, score, snippet}, …]."""
        terms = _tokenize(query)
        if not terms:
            return []
        scored: list[tuple[int, Leaf]] = []
        for path in self.base_path.glob("*.md"):
            leaf = self._read_leaf(path)
            haystack = (leaf.title + "\n" + leaf.content).lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, leaf))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        out: list[dict] = []
        for score, leaf in scored[:max(1, k)]:
            out.append({
                "leaf_id": leaf.leaf_id,
                "title": leaf.title,
                "score": score,
                "snippet": _snippet_around(leaf.content, terms),
            })
        return out

    # -- write API -----------------------------------------------------

    def write(self, title: str, content: str) -> dict:
        title = title.strip() or "untitled"
        content = content.strip()
        if not content:
            return {"result": "Cannot write an empty leaf."}
        if len(content) > LEAF_CHAR_LIMIT:
            return {
                "result": (
                    f"Leaf exceeds {LEAF_CHAR_LIMIT}-char limit "
                    f"({len(content)} chars). Split it or summarise."
                ),
            }
        leaf_id = _new_leaf_id(title)
        path = self._unique_path(leaf_id)
        body = f"# {title}\n\n{content}\n"
        path.write_text(body, encoding="utf-8")
        logger.info(f"memory_tree: wrote leaf {path.name} ({len(content)} chars)")
        return {
            "result": f"Wrote leaf {path.stem} ({len(content)} chars).",
            "leaf_id": path.stem,
        }

    def delete(self, leaf_id: str) -> dict:
        path = self.base_path / f"{leaf_id}.md"
        if not path.exists():
            return {"result": f"No leaf {leaf_id!r}."}
        path.unlink()
        logger.info(f"memory_tree: deleted leaf {leaf_id}")
        return {"result": f"Deleted leaf {leaf_id}."}

    # -- internals -----------------------------------------------------

    def _unique_path(self, leaf_id: str) -> Path:
        path = self.base_path / f"{leaf_id}.md"
        if not path.exists():
            return path
        # Same title/day collision — append a counter.
        for i in range(2, 100):
            candidate = self.base_path / f"{leaf_id}-{i}.md"
            if not candidate.exists():
                return candidate
        # Pathological: 100 collisions on one day. Overwrite #99.
        return self.base_path / f"{leaf_id}-99.md"

    def _read_leaf(self, path: Path) -> Leaf:
        text = path.read_text(encoding="utf-8")
        title, body = _split_title(text)
        return Leaf(
            leaf_id=path.stem,
            title=title or path.stem,
            content=body,
            created_at=path.stat().st_mtime,
        )


# ---------------------------------------------------------------------------
# Helpers (pure — unit-testable in isolation)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _split_title(text: str) -> tuple[str, str]:
    """Extract ``# Title`` header and the body. Tolerates missing header."""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).strip()
        return title, body
    return "", text.strip()


def _snippet_around(content: str, terms: list[str], radius: int = 80) -> str:
    """Return a window around the first term hit, padded with ellipses."""
    lower = content.lower()
    idx = -1
    for term in terms:
        pos = lower.find(term)
        if pos != -1 and (idx == -1 or pos < idx):
            idx = pos
    if idx == -1:
        return content[: 2 * radius]
    start = max(0, idx - radius)
    end = min(len(content), idx + radius)
    snippet = content[start:end].strip().replace("\n", " ")
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"


# ---------------------------------------------------------------------------
# Process-wide accessor (mirrors get_memory_layer)
# ---------------------------------------------------------------------------

_shared: Optional[MemoryTree] = None


def get_memory_tree() -> MemoryTree:
    global _shared
    if _shared is None:
        _shared = MemoryTree()
    return _shared


def reset_memory_tree() -> None:
    global _shared
    _shared = None
