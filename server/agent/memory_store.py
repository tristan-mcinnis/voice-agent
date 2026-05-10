"""Memory snapshot loader — thin backward-compat wrapper over MemoryLayer.

Prefer ``tools.memory_layer.MemoryLayer`` directly. This module exists so
existing callers don't break while the migration completes.

Character limits are re-exported from ``tools.memory_layer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.memory_layer import MemoryLayer, USER_CHAR_LIMIT, MEMORY_CHAR_LIMIT  # noqa: F401


class MemoryStore:
    """Thin wrapper over MemoryLayer for backward compatibility.

    Delegates all reads to ``MemoryLayer``. Direct use of ``MemoryLayer``
    is preferred — this class exists for the transition.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self._layer = MemoryLayer(base_path=base_path)

    def load_user_md(self) -> str:
        return self._layer.load_user_prompt()

    def load_memory_md(self) -> str:
        return self._layer.load_memory_prompt()

    def read_raw(self, filename: str) -> str:
        path = self._layer.base_path / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def write_raw(self, filename: str, content: str) -> None:
        (self._layer.base_path / filename).write_text(content, encoding="utf-8")
