"""Tests for the memory backend seam.

Two ends to check:

  1. ``MemoryBackend`` protocol is satisfied by the existing
     ``MemoryLayer`` (file backend) without any code change — that's
     the point of using a structural Protocol.
  2. ``AgentMemoryBackend`` degrades cleanly when the ``iii`` SDK
     isn't installed: every method returns a structured error string
     rather than raising.

We don't test the iii network round-trip here — that would require a
running service. The integration is a thin trigger/translation layer,
and the unavailability path is the part that matters for "bot stays
usable when memory tier is down."
"""

from __future__ import annotations

from agent.memory_backend import MemoryBackend
from agent.memory_layer import MemoryLayer
from agent.agentmemory_backend import AgentMemoryBackend, _extract_entries


def test_file_backend_satisfies_protocol():
    layer = MemoryLayer()
    assert isinstance(layer, MemoryBackend)


def test_agentmemory_backend_satisfies_protocol():
    backend = AgentMemoryBackend(ws_url="ws://localhost:49134")
    assert isinstance(backend, MemoryBackend)


def test_agentmemory_add_returns_error_without_sdk():
    # The iii SDK is not installed in the test environment — the
    # backend must surface a non-crashing error string.
    backend = AgentMemoryBackend(ws_url="ws://invalid:9")
    result = backend.add("user", "anything")
    assert isinstance(result, dict)
    assert "result" in result
    assert "unavailable" in result["result"].lower() or "iii" in result["result"].lower()


def test_agentmemory_add_rejects_empty_content():
    backend = AgentMemoryBackend(ws_url="ws://invalid:9")
    assert backend.add("user", "   ")["result"].startswith("Cannot add")


def test_agentmemory_replace_requires_old_text():
    backend = AgentMemoryBackend(ws_url="ws://invalid:9")
    result = backend.replace("user", "new content", old_text="")
    assert "old_text" in result["result"]


def test_agentmemory_list_returns_unavailable_when_no_sdk():
    backend = AgentMemoryBackend(ws_url="ws://invalid:9")
    result = backend.list_entries("user")
    assert "unavailable" in result["result"].lower()


def test_agentmemory_load_prompts_return_wrapped_tags():
    backend = AgentMemoryBackend(ws_url="ws://invalid:9")
    assert backend.load_user_prompt().startswith("<user_profile>")
    assert backend.load_user_prompt().endswith("</user_profile>")
    assert backend.load_memory_prompt().startswith("<persistent_memory>")
    assert backend.load_memory_prompt().endswith("</persistent_memory>")


# --- _extract_entries shape robustness --------------------------------------

def test_extract_entries_from_list_of_strings():
    assert _extract_entries(["a", "b", "  ", "c"]) == ["a", "b", "c"]


def test_extract_entries_from_results_key():
    payload = {"results": ["one", "two"]}
    assert _extract_entries(payload) == ["one", "two"]


def test_extract_entries_from_dict_items_with_insight_key():
    payload = {"entries": [{"insight": "x"}, {"insight": "y"}]}
    assert _extract_entries(payload) == ["x", "y"]


def test_extract_entries_from_context_string():
    assert _extract_entries({"context": "line1\nline2\n\nline3"}) == ["line1", "line2", "line3"]


def test_extract_entries_from_raw_string():
    assert _extract_entries("a\nb") == ["a", "b"]


def test_extract_entries_handles_none_and_unknown():
    assert _extract_entries(None) == []
    assert _extract_entries(42) == []
    assert _extract_entries({"weird": "shape"}) == []
