"""Memory backend protocol — the seam between the agent and *where* memory lives.

``MemoryLayer`` (the file-based default) implements this protocol. An
external backend like rohitg00/agentmemory implements the same protocol
behind a different storage tier; the agent never has to know which is
active.

The five methods are the entire contract:

  - ``load_user_prompt()``  → wrapped USER.md snapshot for PromptBuilder
  - ``load_memory_prompt()`` → wrapped MEMORY.md snapshot for PromptBuilder
  - ``list_entries(target)`` → indexed entries dict for the LLM
  - ``add(target, content)`` → append-or-pressure result dict
  - ``replace(target, content, old_text)`` → swap-or-not-found dict

Why a Protocol and not an ABC: file backend predates this seam by a year
and lives in ``agent/memory_layer.py``. Forcing it to inherit from a base
class is churn for no benefit — Protocol is structural, so it conforms
without any code change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryBackend(Protocol):
    """The five methods every backend must expose. See module docstring."""

    def load_user_prompt(self) -> str: ...

    def load_memory_prompt(self) -> str: ...

    def list_entries(self, target: str) -> dict: ...

    def add(self, target: str, content: str) -> dict: ...

    def replace(self, target: str, content: str, old_text: str = "") -> dict: ...
