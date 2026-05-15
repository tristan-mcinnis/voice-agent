"""memory_tree tool — write/search/read/list leaf files.

A thin adapter over ``agent.memory_tree.MemoryTree``. Same pattern as
``tools/memory.py`` over ``agent.memory_layer.MemoryLayer``: this file
owns the LLM-facing parameter schema, the module under ``agent/`` owns
the format and operations.

The four actions are deliberately separate so the LLM can drill in
progressively: ``search`` to find candidates, ``read`` to open one,
``list`` for recent memory, ``write`` to persist a new long-form
recollection. Latency is sub-millisecond per call until leaf counts
get large — at which point the search loop becomes O(N · L) string
scans, still well below voice budgets.
"""

from __future__ import annotations

from typing import Optional

from tools.registry import BaseTool, REGISTRY
from agent.memory_tree import get_memory_tree, LEAF_CHAR_LIMIT


@REGISTRY.register
class MemoryTreeTool(BaseTool):
    name = "memory_tree"
    category = "system"
    speak_text: Optional[str] = None
    description = (
        "Long-form persistent memory — write, search, read, or list 'leaf' files. "
        "Use for recollections that don't fit the short curated MEMORY.md / USER.md "
        "index: meeting summaries, debugging post-mortems, multi-paragraph context. "
        "Actions: 'write' (new leaf), 'search' (token search), 'read' (full text by id), "
        "'list' (recent leaves), 'delete' (remove a leaf)."
    )
    parameters = {
        "action": {
            "type": "string",
            "enum": ["write", "search", "read", "list", "delete"],
            "description": "Operation to perform.",
        },
        "title": {
            "type": "string",
            "description": "Title for 'write'. Becomes part of the leaf id.",
        },
        "content": {
            "type": "string",
            "description": f"Body for 'write'. Max {LEAF_CHAR_LIMIT} chars.",
        },
        "query": {
            "type": "string",
            "description": "Search terms for 'search'. Space-separated tokens.",
        },
        "leaf_id": {
            "type": "string",
            "description": "Leaf identifier for 'read' or 'delete' (returned by search/list/write).",
        },
        "limit": {
            "type": "integer",
            "description": "Max items for 'list' or 'search'. Default 10.",
        },
    }
    required = ["action"]
    guidance = """
## Memory Tree Usage

The `memory_tree` tool stores long recollections that are too big for
USER.md/MEMORY.md but worth keeping. Each leaf is a markdown file in
`.voice-agent/memories/tree/`.

Use cases:
- A multi-paragraph meeting summary the user wants to recall later
- A detailed debugging post-mortem with code snippets
- Context dumps from a specific project's onboarding

Workflow:
1. `search` with a few topic tokens → get candidate leaves with snippets
2. `read` the most relevant leaf id → get full content
3. `write` to persist new long recollections

Keep MEMORY.md/USER.md for short, frequently-needed facts (the agent
always sees those). Use the tree for content the agent only needs when
the user brings up the topic. A leaf is capped at 4000 chars — split
or summarise anything longer."""

    def execute(
        self,
        action: str,
        title: str = "",
        content: str = "",
        query: str = "",
        leaf_id: str = "",
        limit: int = 10,
    ) -> dict:
        tree = get_memory_tree()

        if action == "write":
            if not content:
                return {"result": "Provide 'content' for write."}
            return tree.write(title=title or "untitled", content=content)

        if action == "search":
            if not query:
                return {"result": "Provide 'query' for search."}
            hits = tree.search(query, k=max(1, limit))
            if not hits:
                return {"result": f"No leaves match {query!r}.", "hits": []}
            lines = [f"{len(hits)} match{'es' if len(hits) != 1 else ''} for {query!r}:"]
            for h in hits:
                lines.append(f"  [{h['leaf_id']}] ({h['score']}) {h['title']} — {h['snippet']}")
            return {"result": "\n".join(lines), "hits": hits}

        if action == "read":
            if not leaf_id:
                return {"result": "Provide 'leaf_id' for read."}
            leaf = tree.read(leaf_id)
            if leaf is None:
                return {"result": f"No leaf {leaf_id!r}."}
            return {
                "result": f"# {leaf.title}\n\n{leaf.content}",
                "leaf_id": leaf.leaf_id,
                "title": leaf.title,
            }

        if action == "list":
            items = tree.list_leaves(limit=max(1, limit))
            if not items:
                return {"result": "No leaves stored yet.", "leaves": []}
            lines = [f"{len(items)} leaf{'s' if len(items) != 1 else ''} (newest first):"]
            for it in items:
                lines.append(f"  [{it['leaf_id']}] {it['title']} — {it['preview']}")
            return {"result": "\n".join(lines), "leaves": items}

        if action == "delete":
            if not leaf_id:
                return {"result": "Provide 'leaf_id' for delete."}
            return tree.delete(leaf_id)

        return {"result": f"Unknown action {action!r}."}
