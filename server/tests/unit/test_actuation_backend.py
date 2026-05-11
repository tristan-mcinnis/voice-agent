"""Unit tests for the actuation backend seam.

No real screen, no real input — everything is mocked at the backend interface.
"""

from __future__ import annotations

import pytest

from tools.actuation_backend import (
    ActuationBackend,
    PyAutoGUIBackend,
    SafetyGateBackend,
    get_actuation_backend,
    reset_actuation_backend,
)
from tools.safety_gate import SafetyGate


# ---------------------------------------------------------------------------
# Fake backend for injection
# ---------------------------------------------------------------------------

class FakeBackend(ActuationBackend):
    """Records every call so tests can assert on behaviour without real IO."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left") -> str:
        self.calls.append(("click", (x, y), {"double": double, "button": button}))
        return f"clicked ({x},{y})"

    def type_text(self, text: str, interval: float = 0.02) -> str:
        self.calls.append(("type_text", (text,), {"interval": interval}))
        return f"typed {len(text)}"

    def press_key(self, keys: str) -> str:
        self.calls.append(("press_key", (keys,), {}))
        return f"pressed {keys}"

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> str:
        self.calls.append(("scroll", (amount,), {"x": x, "y": y}))
        return f"scrolled {amount}"

    def mouse_move(self, x: int, y: int, duration: float = 0.1) -> str:
        self.calls.append(("mouse_move", (x, y), {"duration": duration}))
        return f"moved ({x},{y})"


# ---------------------------------------------------------------------------
# SafetyGateBackend — the real deepening value
# ---------------------------------------------------------------------------

class TestSafetyGateBackend:
    def test_passes_through_when_gate_clear(self, monkeypatch):
        inner = FakeBackend()
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: None)
        backend = SafetyGateBackend(inner, gate=gate)

        result = backend.click(10, 20, confirm=False)
        assert result == "clicked (10,20)"
        assert inner.calls == [("click", (10, 20), {"double": False, "button": "left"})]

    def test_blocks_when_sensitive_and_unconfirmed(self, monkeypatch):
        inner = FakeBackend()
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: "1Password")
        backend = SafetyGateBackend(inner, gate=gate)

        result = backend.click(10, 20, confirm=False)
        assert "BLOCKED" in result
        assert inner.calls == []  # inner backend never touched

    def test_bypasses_when_confirmed(self, monkeypatch):
        inner = FakeBackend()
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: "1Password")
        backend = SafetyGateBackend(inner, gate=gate)

        result = backend.click(10, 20, confirm=True)
        assert result == "clicked (10,20)"
        assert len(inner.calls) == 1

    def test_all_methods_pass_confirm_kwarg(self, monkeypatch):
        inner = FakeBackend()
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: None)
        backend = SafetyGateBackend(inner, gate=gate)

        backend.type_text("hello", confirm=True)
        backend.press_key("enter", confirm=False)
        backend.scroll(5, confirm=True)
        backend.mouse_move(100, 200, confirm=False)

        assert len(inner.calls) == 4
        assert inner.calls[0][0] == "type_text"
        assert inner.calls[1][0] == "press_key"
        assert inner.calls[2][0] == "scroll"
        assert inner.calls[3][0] == "mouse_move"

    def test_type_text_preview_truncation_in_gate_message(self, monkeypatch):
        """The gate check message truncates long text to 30 chars."""
        inner = FakeBackend()
        gate = SafetyGate()
        monkeypatch.setattr(gate, "_sensitive_context", lambda: "Secrets")
        backend = SafetyGateBackend(inner, gate=gate)

        long_text = "x" * 100
        result = backend.type_text(long_text, confirm=False)
        assert "BLOCKED" in result
        assert "..." in result  # preview was truncated


# ---------------------------------------------------------------------------
# PyAutoGUIBackend — we can't click a real screen, but we can assert on
# import/configuration behaviour and error paths.
# ---------------------------------------------------------------------------

class TestPyAutoGUIBackend:
    def test_lazy_import_fails_cleanly_when_pyautogui_missing(self, monkeypatch):
        backend = PyAutoGUIBackend()
        monkeypatch.setattr(
            backend, "_pg", None  # reset any prior import
        )
        # Make __import__ raise for pyautogui
        real_import = __builtins__["__import__"]

        def fake_import(name, *args, **kwargs):
            if name == "pyautogui":
                raise ImportError("No module named pyautogui")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(__builtins__, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyautogui not installed"):
            backend._get_pg()

    def test_click_returns_descriptive_string(self, monkeypatch):
        backend = PyAutoGUIBackend()
        calls = []

        class _FakePG:
            def click(self, *a, **k):
                calls.append(("click", a, k))

            def doubleClick(self, *a, **k):
                calls.append(("doubleClick", a, k))

        monkeypatch.setattr(backend, "_pg", _FakePG())
        result = backend.click(5, 10, double=True, button="right")
        assert "Double-clicked" in result
        assert "(5,10)" in result
        assert "right" in result

    def test_press_key_splits_plus(self, monkeypatch):
        backend = PyAutoGUIBackend()
        calls = []

        class _FakePG:
            def press(self, k):
                calls.append(("press", k))

            def hotkey(self, *keys):
                calls.append(("hotkey", keys))

        monkeypatch.setattr(backend, "_pg", _FakePG())
        backend.press_key("cmd+f")
        assert calls == [("hotkey", ("cmd", "f"))]

    def test_press_key_single_key(self, monkeypatch):
        backend = PyAutoGUIBackend()
        calls = []

        class _FakePG:
            def press(self, k):
                calls.append(("press", k))

            def hotkey(self, *keys):
                calls.append(("hotkey", keys))

        monkeypatch.setattr(backend, "_pg", _FakePG())
        backend.press_key("enter")
        assert calls == [("press", "enter")]

    def test_press_key_empty_raises(self, monkeypatch):
        backend = PyAutoGUIBackend()

        class _FakePG:
            def press(self, k):
                pass

            def hotkey(self, *keys):
                pass

        monkeypatch.setattr(backend, "_pg", _FakePG())
        with pytest.raises(ValueError, match="empty key string"):
            backend.press_key("")


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

class TestSingletonLifecycle:
    def test_reset_clears_singleton(self):
        reset_actuation_backend()
        # After reset, a fresh call builds a new backend.
        # We can't easily assert on *which* backend without config,
        # but we can at least confirm the function doesn't crash.
        b1 = get_actuation_backend()
        reset_actuation_backend()
        b2 = get_actuation_backend()
        # They should be different objects after reset.
        assert b1 is not b2
