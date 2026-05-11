"""Actuation backend — unified seam for GUI automation.

Consolidates the cua / pyautogui / safety-gate triad that was duplicated
across every computer-use tool. One factory reads config and returns the
correct stacked backend; each tool calls a single method.

Seam: ``ActuationBackend`` interface.
Adapters:
  - ``PyAutoGUIBackend`` — drives macOS via pyautogui.
  - ``CuaBackend`` — re-uses ``tools.cua_backend.CuaBackend`` (already
    string-compatible).
Decorator:
  - ``SafetyGateBackend`` — wraps any backend with HITL gating.

Usage in a tool::

    from tools.actuation_backend import get_actuation_backend

    backend = get_actuation_backend()
    return backend.click(x, y, double=False, button="left", confirm=False)
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from tools.safety_gate import SafetyGate


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class ActuationBackend:
    """Low-level GUI actuation. Every method returns a result string on success
    or raises on failure.
    """

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left") -> str:
        raise NotImplementedError

    def type_text(self, text: str, interval: float = 0.02) -> str:
        raise NotImplementedError

    def press_key(self, keys: str) -> str:
        raise NotImplementedError

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
        raise NotImplementedError

    def mouse_move(self, x: int, y: int, duration: float = 0.1) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class PyAutoGUIBackend(ActuationBackend):
    """Actuation via pyautogui. Lazy-imports and configures pyautogui once."""

    def __init__(self) -> None:
        self._pg = None

    def _get_pg(self):
        if self._pg is None:
            try:
                import pyautogui
            except ImportError as exc:
                raise ImportError(
                    "pyautogui not installed. Run: pip install pyautogui"
                ) from exc
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.05
            self._pg = pyautogui
        return self._pg

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left") -> str:
        pg = self._get_pg()
        if double:
            pg.doubleClick(x, y, button=button)
        else:
            pg.click(x, y, button=button)
        return f"{'Double-clicked' if double else 'Clicked'} ({x},{y}) [{button}]"

    def type_text(self, text: str, interval: float = 0.02) -> str:
        pg = self._get_pg()
        try:
            pg.typewrite(text, interval=interval)
        except Exception as exc:
            # pyautogui.typewrite can't handle non-ASCII — fall back to AppleScript.
            import subprocess
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to keystroke "{escaped}"'],
                capture_output=True, text=True, check=True, timeout=10.0,
            )
        return f"Typed {len(text)} character{'s' if len(text) != 1 else ''}."

    def press_key(self, keys: str) -> str:
        pg = self._get_pg()
        parts = [k.strip().lower() for k in keys.split("+") if k.strip()]
        if not parts:
            raise ValueError("press_key: empty key string.")
        if len(parts) == 1:
            pg.press(parts[0])
        else:
            pg.hotkey(*parts)
        return f"Pressed {keys}."

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
        pg = self._get_pg()
        if x is not None and y is not None:
            pg.scroll(amount, x=x, y=y)
        else:
            pg.scroll(amount)
        time.sleep(0.15)
        return f"Scrolled {amount:+d}{f' at ({x},{y})' if x is not None else ''}."

    def mouse_move(self, x: int, y: int, duration: float = 0.1) -> str:
        pg = self._get_pg()
        pg.moveTo(x, y, duration=duration)
        return f"Moved to ({x},{y})."


# ---------------------------------------------------------------------------
# Decorator — HITL safety gate
# ---------------------------------------------------------------------------

class SafetyGateBackend(ActuationBackend):
    """Wraps any ``ActuationBackend`` with the sensitive-app safety gate.

    Each method accepts an optional ``confirm`` kwarg. When ``confirm=False``
    (the default) and the frontmost app/window matches a sensitive pattern,
    the call returns a BLOCKED message instead of delegating to the inner
    backend.
    """

    def __init__(self, backend: ActuationBackend, gate: Optional[SafetyGate] = None) -> None:
        self._backend = backend
        self._gate = gate or SafetyGate()

    def click(self, x: int, y: int, *, double: bool = False, button: str = "left", confirm: bool = False) -> str:
        refusal = self._gate.check(f"click at ({x},{y})", confirm)
        if refusal:
            return refusal
        return self._backend.click(x, y, double=double, button=button)

    def type_text(self, text: str, interval: float = 0.02, confirm: bool = False) -> str:
        preview = text if len(text) <= 30 else text[:27] + "..."
        refusal = self._gate.check(f"type {preview!r}", confirm)
        if refusal:
            return refusal
        return self._backend.type_text(text, interval=interval)

    def press_key(self, keys: str, confirm: bool = False) -> str:
        refusal = self._gate.check(f"press {keys}", confirm)
        if refusal:
            return refusal
        return self._backend.press_key(keys)

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None, confirm: bool = False) -> str:
        refusal = self._gate.check(f"scroll {amount:+d}", confirm)
        if refusal:
            return refusal
        return self._backend.scroll(amount, x=x, y=y)

    def mouse_move(self, x: int, y: int, duration: float = 0.1, confirm: bool = False) -> str:
        refusal = self._gate.check(f"move mouse to ({x},{y})", confirm)
        if refusal:
            return refusal
        return self._backend.mouse_move(x, y, duration=duration)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_actuation_backend: ActuationBackend | None = None


def _try_cua() -> Optional[ActuationBackend]:
    """Return a CuaBackend if config says ``backend: cua`` and the binary is
    reachable, otherwise ``None``.
    """
    try:
        from config import get_config
        backend_name = get_config().computer_use.backend
    except Exception:
        return None
    if backend_name != "cua":
        return None
    try:
        from tools.cua_backend import get_cua_backend, CuaBackendError
        return get_cua_backend()
    except Exception as exc:
        logger.warning(
            f"computer_use.backend=cua but cua-driver unavailable "
            f"({type(exc).__name__}: {exc}); falling back to native."
        )
        return None


def _build_backend() -> ActuationBackend:
    """Construct the stacked backend: SafetyGate → (Cua | PyAutoGUI)."""
    inner = _try_cua() or PyAutoGUIBackend()
    return SafetyGateBackend(inner)


def get_actuation_backend() -> ActuationBackend:
    """Return the lazily-constructed actuation backend singleton."""
    global _actuation_backend
    if _actuation_backend is None:
        _actuation_backend = _build_backend()
    return _actuation_backend


def reset_actuation_backend() -> None:
    """Clear the singleton. Used by tests that need a fresh backend stack."""
    global _actuation_backend
    _actuation_backend = None
