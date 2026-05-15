"""agentmemory (rohitg00/agentmemory) backend — durable, shared memory tier.

This is the alternative to the file-based ``MemoryLayer``. When a user
already self-hosts agentmemory (e.g. to share durable memory across
Claude Code, Cursor, Codex, and OpenCode), pointing the voice bot at the
same service means every agent reads and writes the same memory.

Wire it up by setting in ``config.yaml``::

    memory:
      backend: agentmemory
      agentmemory:
        ws_url: ws://localhost:49134
        project_user: voice-agent-user
        project_memory: voice-agent-memory

agentmemory is reached via the iii SDK over WebSocket. The SDK is an
optional dependency — ``import iii`` happens lazily inside the methods
so simply *configuring* this backend doesn't crash startup. If the SDK
is missing or the service is unreachable, each call returns a useful
error string and the bot stays usable.

Mapping the iii memory primitives to the voice-agent's two-file model:

  ``add(user, …)``     → ``mem::remember`` on ``project_user``
  ``add(memory, …)``   → ``mem::remember`` on ``project_memory``
  ``list_entries(…)``  → ``mem::smart-search`` with an empty query
  ``replace(…)``       → ``mem::forget`` followed by ``mem::remember``
  ``load_*_prompt()``  → ``mem::context`` (or smart-search fallback)

The exact iii function-id strings and payload shapes are documented at
the agentmemory repo; if rohit changes them, only this module needs to
move. Everything else in the codebase talks to the
``MemoryBackend`` protocol.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from agent.memory_layer import USER_CHAR_LIMIT, MEMORY_CHAR_LIMIT


_FN_REMEMBER = "mem::remember"
_FN_OBSERVE = "mem::observe"
_FN_FORGET = "mem::forget"
_FN_SEARCH = "mem::smart-search"
_FN_CONTEXT = "mem::context"


class AgentMemoryBackend:
    """Talk to a self-hosted agentmemory instance via the iii SDK.

    Each call lazily acquires a connected worker. The connection is
    cached on the instance, but every method tolerates a missing SDK
    or a dead service by returning a structured error — the bot never
    crashes because the memory tier is down.
    """

    def __init__(
        self,
        *,
        ws_url: str,
        project_user: str = "voice-agent-user",
        project_memory: str = "voice-agent-memory",
        timeout: float = 10.0,
    ) -> None:
        self.ws_url = ws_url
        self.project_user = project_user
        self.project_memory = project_memory
        self.timeout = timeout
        self._iii: Any = None
        self._iii_unavailable_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _worker(self) -> Any:
        """Return a connected iii worker, or ``None`` if unavailable.

        Caches both success and failure: once we know the SDK is
        missing, subsequent calls short-circuit without retrying the
        import on every memory write.
        """
        if self._iii is not None:
            return self._iii
        if self._iii_unavailable_reason is not None:
            return None
        try:
            from iii import register_worker  # type: ignore[import-not-found]
        except ImportError as exc:
            self._iii_unavailable_reason = (
                f"iii SDK not installed ({exc}). Run `pip install iii-sdk` to enable."
            )
            logger.warning(f"agentmemory: {self._iii_unavailable_reason}")
            return None
        try:
            worker = register_worker(self.ws_url)
            worker.connect()
        except Exception as exc:
            self._iii_unavailable_reason = (
                f"iii worker failed to connect to {self.ws_url}: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning(f"agentmemory: {self._iii_unavailable_reason}")
            return None
        self._iii = worker
        return worker

    def _project_for(self, target: str) -> str:
        return self.project_user if target == "user" else self.project_memory

    def _trigger(self, function_id: str, payload: dict) -> Any:
        """Fire one iii function call. Surfaces errors as a string result."""
        worker = self._worker()
        if worker is None:
            return {"_error": self._iii_unavailable_reason}
        try:
            return worker.trigger({"function_id": function_id, "payload": payload})
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning(f"agentmemory: {function_id} failed: {err}")
            return {"_error": err}

    # ------------------------------------------------------------------
    # Read API (used by PromptBuilder)
    # ------------------------------------------------------------------

    def load_user_prompt(self) -> str:
        body = self._load_context(self.project_user)
        return f"<user_profile>\n{body}\n</user_profile>"

    def load_memory_prompt(self) -> str:
        body = self._load_context(self.project_memory)
        return f"<persistent_memory>\n{body}\n</persistent_memory>"

    def _load_context(self, project: str) -> str:
        result = self._trigger(_FN_CONTEXT, {"project": project})
        if isinstance(result, dict) and result.get("_error"):
            return f"(agentmemory unavailable: {result['_error']})"
        if isinstance(result, dict):
            text = result.get("context") or result.get("result") or ""
            return str(text).strip() or "(no entries)"
        return str(result).strip() or "(no entries)"

    # ------------------------------------------------------------------
    # Write API (used by MemoryTool)
    # ------------------------------------------------------------------

    def list_entries(self, target: str) -> dict:
        project = self._project_for(target)
        limit = USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT
        result = self._trigger(_FN_SEARCH, {"project": project, "query": ""})
        if isinstance(result, dict) and result.get("_error"):
            return {"result": f"agentmemory unavailable: {result['_error']}"}
        entries = _extract_entries(result)
        chars = sum(len(e) for e in entries)
        if not entries:
            return {
                "result": f"{project} is empty (0/{limit} chars used).",
                "entries": {},
                "chars_used": 0,
                "char_limit": limit,
            }
        indexed = {str(i): e for i, e in enumerate(entries)}
        lines = [f"{project} ({chars}/{limit} chars used):"]
        for i, e in enumerate(entries):
            preview = e if len(e) <= 80 else e[:77] + "..."
            lines.append(f"  [{i}] {preview}")
        return {
            "result": "\n".join(lines),
            "entries": indexed,
            "chars_used": chars,
            "char_limit": limit,
        }

    def add(self, target: str, content: str) -> dict:
        content = content.strip()
        if not content:
            return {"result": "Cannot add empty content."}
        project = self._project_for(target)
        result = self._trigger(_FN_REMEMBER, {"project": project, "insight": content})
        if isinstance(result, dict) and result.get("_error"):
            return {"result": f"agentmemory unavailable: {result['_error']}"}
        return {"result": f"Added to {project}: \"{content[:97] + '...' if len(content) > 100 else content}\""}

    def replace(self, target: str, content: str, old_text: str = "") -> dict:
        content = content.strip()
        old_text = old_text.strip()
        if not content:
            return {"result": "Cannot replace with empty content."}
        if not old_text:
            return {"result": "Provide `old_text` — a substring of the entry to replace."}
        project = self._project_for(target)
        forget = self._trigger(_FN_FORGET, {"project": project, "match": old_text})
        if isinstance(forget, dict) and forget.get("_error"):
            return {"result": f"agentmemory unavailable: {forget['_error']}"}
        remember = self._trigger(_FN_REMEMBER, {"project": project, "insight": content})
        if isinstance(remember, dict) and remember.get("_error"):
            return {"result": f"agentmemory unavailable: {remember['_error']}"}
        return {"result": f"Replaced entry matching {old_text!r} in {project}."}


def _extract_entries(result: Any) -> list[str]:
    """Best-effort flattening of iii's smart-search result shape.

    The exact JSON shape isn't pinned in the public docs, so we cover the
    obvious cases: top-level list, ``results``/``entries``/``observations``
    key, or a single string. Anything unexpected returns an empty list
    rather than crashing the LLM-facing call.
    """
    if result is None:
        return []
    if isinstance(result, list):
        return [str(e).strip() for e in result if str(e).strip()]
    if isinstance(result, dict):
        for key in ("results", "entries", "observations", "items"):
            val = result.get(key)
            if isinstance(val, list):
                return [
                    (item if isinstance(item, str) else item.get("insight") or item.get("text") or "")
                    for item in val
                ]
        text = result.get("context") or result.get("result")
        if isinstance(text, str):
            return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(result, str):
        return [line.strip() for line in result.splitlines() if line.strip()]
    return []
