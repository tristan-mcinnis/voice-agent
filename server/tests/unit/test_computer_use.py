"""Unit tests for computer-use pure logic — no real screen, no real input.

Covers:
  - SafetyGate sensitive-context detection
  - AXEngine element-index cache resolution
  - AXElement.center and .short formatting

Coord parsing for the grounding-vision tool lives in test_grounding_vision.py.
"""

from __future__ import annotations

import pytest

from tools.ax_engine import AXElement, AXEngine
from tools.safety_gate import SafetyGate


# ---------------------------------------------------------------------------
# AXElement helpers
# ---------------------------------------------------------------------------

class TestAXElement:
    def test_center_is_position_plus_half_size(self):
        e = AXElement(role="AXButton", label="Play", x=100, y=200, w=40, h=20)
        assert e.center() == (120, 210)

    def test_short_strips_AX_prefix_and_includes_index(self):
        e = AXElement(role="AXButton", label="Play", x=10, y=20, w=4, h=6)
        line = e.short(7)
        assert line.startswith("[7] Button")
        assert "Play" in line
        # Center: (10 + 4//2, 20 + 6//2) = (12, 23)
        assert "(12,23)" in line

    def test_short_truncates_long_label(self):
        e = AXElement(role="AXButton", label="x" * 200, x=0, y=0, w=10, h=10)
        line = e.short(0)
        assert "…" in line
        assert len(line) < 200

    def test_short_unlabeled_fallback(self):
        e = AXElement(role="AXButton", label="", x=0, y=0, w=10, h=10)
        assert "(unlabeled)" in e.short(0)


# ---------------------------------------------------------------------------
# Safety gate — SafetyGate.check
# ---------------------------------------------------------------------------

class TestSafetyGate:
    def test_passes_when_context_not_sensitive(self, monkeypatch):
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: None)
        assert gate.check("click", confirm=False) is None
        assert gate.check("click", confirm=True) is None

    def test_blocks_when_sensitive_and_unconfirmed(self, monkeypatch):
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: "1Password — Vault")
        msg = gate.check("type 'x'", confirm=False)
        assert msg is not None
        assert "BLOCKED" in msg
        assert "1Password" in msg
        assert "type 'x'" in msg
        assert "confirm=true" in msg

    def test_bypasses_when_sensitive_but_confirmed(self, monkeypatch):
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: "1Password")
        assert gate.check("click", confirm=True) is None


# ---------------------------------------------------------------------------
# Element-index cache resolution — AXEngine
# ---------------------------------------------------------------------------

class TestAXEngineResolve:
    @pytest.fixture
    def engine(self):
        return AXEngine()

    def test_returns_none_when_cache_empty(self, engine):
        assert engine.resolve(0) is None

    def test_resolves_valid_index(self, engine):
        e0 = AXElement(role="AXButton", label="A", x=0, y=0, w=1, h=1)
        e1 = AXElement(role="AXButton", label="B", x=0, y=0, w=1, h=1)
        engine.last_elements["TestApp"] = [e0, e1]
        engine.last_process.append("TestApp")
        assert engine.resolve(0) is e0
        assert engine.resolve(1) is e1

    def test_returns_none_for_out_of_range(self, engine):
        engine.last_elements["TestApp"] = [
            AXElement(role="AXButton", label="A", x=0, y=0, w=1, h=1)
        ]
        engine.last_process.append("TestApp")
        assert engine.resolve(5) is None
        assert engine.resolve(-1) is None

    def test_uses_most_recent_process(self, engine):
        ea = AXElement(role="AXButton", label="from-A", x=0, y=0, w=1, h=1)
        eb = AXElement(role="AXButton", label="from-B", x=0, y=0, w=1, h=1)
        engine.last_elements["AppA"] = [ea]
        engine.last_elements["AppB"] = [eb]
        engine.last_process.extend(["AppA", "AppB"])
        # Most recent is AppB
        assert engine.resolve(0) is eb
