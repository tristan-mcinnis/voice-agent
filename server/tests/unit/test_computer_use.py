"""Unit tests for computer-use pure logic — no real screen, no real input.

Covers:
  - safety-gate sensitive-context detection (`_refuse_if_sensitive`)
  - element-index cache resolution (`_resolve_element`)
  - `_AXElement.center` and `.short` formatting

Coord parsing for the grounding-vision tool lives in test_grounding_vision.py.
"""

from __future__ import annotations

import pytest

import tools.computer_use as cu


# ---------------------------------------------------------------------------
# _AXElement helpers
# ---------------------------------------------------------------------------

class TestAXElement:
    def test_center_is_position_plus_half_size(self):
        e = cu._AXElement(role="AXButton", label="Play", x=100, y=200, w=40, h=20)
        assert e.center() == (120, 210)

    def test_short_strips_AX_prefix_and_includes_index(self):
        e = cu._AXElement(role="AXButton", label="Play", x=10, y=20, w=4, h=6)
        line = e.short(7)
        assert line.startswith("[7] Button")
        assert "Play" in line
        # Center: (10 + 4//2, 20 + 6//2) = (12, 23)
        assert "(12,23)" in line

    def test_short_truncates_long_label(self):
        e = cu._AXElement(role="AXButton", label="x" * 200, x=0, y=0, w=10, h=10)
        line = e.short(0)
        assert "…" in line
        assert len(line) < 200

    def test_short_unlabeled_fallback(self):
        e = cu._AXElement(role="AXButton", label="", x=0, y=0, w=10, h=10)
        assert "(unlabeled)" in e.short(0)


# ---------------------------------------------------------------------------
# Safety gate — _refuse_if_sensitive
# ---------------------------------------------------------------------------

class TestSafetyGate:
    @pytest.fixture(autouse=True)
    def _patch_context(self, monkeypatch):
        """Default: pretend the front context is *not* sensitive."""
        monkeypatch.setattr(cu, "_sensitive_context", lambda: None)

    def test_passes_when_context_not_sensitive(self):
        assert cu._refuse_if_sensitive("click", confirm=False) is None
        assert cu._refuse_if_sensitive("click", confirm=True) is None

    def test_blocks_when_sensitive_and_unconfirmed(self, monkeypatch):
        monkeypatch.setattr(cu, "_sensitive_context", lambda: "1Password — Vault")
        msg = cu._refuse_if_sensitive("type 'x'", confirm=False)
        assert msg is not None
        assert "BLOCKED" in msg
        assert "1Password" in msg
        assert "type 'x'" in msg
        assert "confirm=true" in msg

    def test_bypasses_when_sensitive_but_confirmed(self, monkeypatch):
        monkeypatch.setattr(cu, "_sensitive_context", lambda: "1Password")
        assert cu._refuse_if_sensitive("click", confirm=True) is None

    def test_swallows_context_lookup_errors(self, monkeypatch):
        # If _sensitive_context itself raises, we don't want to crash the tool.
        # In practice _front_window_title catches its own exceptions, so the
        # sensitive_context returns None on failure — we model that here.
        monkeypatch.setattr(cu, "_sensitive_context", lambda: None)
        assert cu._refuse_if_sensitive("anything", confirm=False) is None


# ---------------------------------------------------------------------------
# Element-index cache resolution
# ---------------------------------------------------------------------------

class TestResolveElement:
    @pytest.fixture(autouse=True)
    def _isolate_cache(self):
        """Save and restore the module-level cache around each test."""
        saved_proc = list(cu._LAST_PROCESS)
        saved_elems = dict(cu._LAST_ELEMENTS)
        cu._LAST_PROCESS.clear()
        cu._LAST_ELEMENTS.clear()
        yield
        cu._LAST_PROCESS.clear()
        cu._LAST_PROCESS.extend(saved_proc)
        cu._LAST_ELEMENTS.clear()
        cu._LAST_ELEMENTS.update(saved_elems)

    def test_returns_none_when_cache_empty(self):
        assert cu._resolve_element(0) is None

    def test_resolves_valid_index(self):
        e0 = cu._AXElement(role="AXButton", label="A", x=0, y=0, w=1, h=1)
        e1 = cu._AXElement(role="AXButton", label="B", x=0, y=0, w=1, h=1)
        cu._LAST_ELEMENTS["TestApp"] = [e0, e1]
        cu._LAST_PROCESS.append("TestApp")
        assert cu._resolve_element(0) is e0
        assert cu._resolve_element(1) is e1

    def test_returns_none_for_out_of_range(self):
        cu._LAST_ELEMENTS["TestApp"] = [
            cu._AXElement(role="AXButton", label="A", x=0, y=0, w=1, h=1)
        ]
        cu._LAST_PROCESS.append("TestApp")
        assert cu._resolve_element(5) is None
        assert cu._resolve_element(-1) is None

    def test_uses_most_recent_process(self):
        ea = cu._AXElement(role="AXButton", label="from-A", x=0, y=0, w=1, h=1)
        eb = cu._AXElement(role="AXButton", label="from-B", x=0, y=0, w=1, h=1)
        cu._LAST_ELEMENTS["AppA"] = [ea]
        cu._LAST_ELEMENTS["AppB"] = [eb]
        cu._LAST_PROCESS.extend(["AppA", "AppB"])
        # Most recent is AppB
        assert cu._resolve_element(0) is eb


