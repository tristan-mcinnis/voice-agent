"""Computer-use tools — drive the macOS GUI like a Hermes-style vision-action agent.

Two complementary paths:

  1. **AX path (primary, Shortcat-style).** ``list_ui_elements`` walks the
     macOS Accessibility tree of the frontmost app and returns every clickable
     element with a label + screen coordinates. ``click_element(index)`` then
     clicks the chosen one. Reliable for native apps (Spotify, Finder, Mail,
     System Settings…) because the labels come from the OS, not the pixels.

  2. **Coordinate path (fallback).** ``click_at(x, y)``, ``type_text``,
     ``press_key``, ``scroll`` — for apps with poor AX (Electron, canvas,
     games). The agent screenshots, reasons about coordinates, then acts.

Actuation (clicks, typing, keys, scroll, move) is delegated to the unified
``ActuationBackend`` seam in ``tools.actuation_backend``.  That module handles
pyautogui vs cua-driver selection, safety-gate checks, and macOS validation —
one place, not six.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

from loguru import logger

from tools._macos import is_macos, macos_only
from tools.actuation_backend import get_actuation_backend
from tools.ax_engine import AXEngine, AXElement
from tools.registry import REGISTRY, BaseTool

# Module-level instances — constructed once, shared by all computer-use tools.
_ax_engine = AXEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_element(index: int) -> Optional[AXElement]:
    """Look up an element by index in the most recently listed process."""
    return _ax_engine.resolve(index)


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

@REGISTRY.register
class ListUIElementsTool(BaseTool):
    name = "list_ui_elements"
    category = "input"
    speak_text = "Looking at the screen."
    description = (
        "List every clickable/interactive UI element in the frontmost macOS "
        "app (buttons, menu items, text fields, links, rows). Returns an "
        "indexed list. Call this BEFORE click_element to find the target. "
        "Uses the macOS Accessibility API — the labels come from the OS, not "
        "from screenshot OCR, so it's far more reliable than guessing pixels."
    )
    parameters = {
        "filter": {
            "type": "string",
            "description": (
                "Optional case-insensitive substring to filter labels. "
                "E.g. 'play', 'jazz', 'search'."
            ),
        },
        "max_items": {
            "type": "integer",
            "description": "Cap to N rows (default 80).",
        },
    }
    guidance = (
        "## Computer-use workflow\n"
        "When the user asks you to interact with a Mac app's UI:\n"
        "1. Call list_ui_elements (optionally with filter='word') to discover targets.\n"
        "2. Call click_element(index=N) — N is the [N] from list_ui_elements.\n"
        "3. Verify with take_screenshot if the action's effect isn't obvious.\n"
        "Use click_at(x, y) only when AX listing is empty (Electron/canvas apps).\n"
        "Use type_text after focusing a text field; use press_key for shortcuts.\n"
        "\n"
        "## Safety gate\n"
        "If the frontmost app/window matches a sensitive pattern (passwords, "
        "banking, system settings, etc.), click_*/type_text/press_key will "
        "return a 'BLOCKED' message. When that happens: speak the intended "
        "action aloud, get a verbal yes from the user, then call again with "
        "confirm=true. Never set confirm=true without that verbal yes."
    )

    def execute(self, filter: str = "", max_items: int = 80) -> str:
        if not is_macos():
            return macos_only("list_ui_elements")
        try:
            proc, elements, display = _ax_engine.list_frontmost(
                filter=filter, max_items=max_items
            )
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip() or str(exc)
            return (
                f"Could not read UI elements: {err}\n"
                "Grant Accessibility permission to your terminal/Python in "
                "System Settings → Privacy & Security → Accessibility."
            )
        except Exception as exc:
            return f"Could not read UI elements: {type(exc).__name__}: {exc}"

        if not elements:
            return (
                f"No interactive UI elements found in {proc}. The app may use "
                "a non-native (Electron/canvas) UI — try take_screenshot + click_at."
            )

        return display


@REGISTRY.register
class ClickElementTool(BaseTool):
    name = "click_element"
    category = "input"
    speak_text = "Clicking."
    description = (
        "Click a UI element by its index from the most recent list_ui_elements "
        "call. The most reliable way to interact with native macOS apps."
    )
    parameters = {
        "index": {
            "type": "integer",
            "description": "Element index from list_ui_elements (the [N] prefix).",
        },
        "double": {
            "type": "boolean",
            "description": "Double-click instead of single. Default false.",
        },
        "confirm": {
            "type": "boolean",
            "description": (
                "Required true to override the sensitive-app safety gate. "
                "Only set after the user has verbally confirmed."
            ),
        },
    }
    required = ["index"]

    def execute(self, index: int, double: bool = False, confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("click_element")
        elem = _resolve_element(index)
        if elem is None:
            return (
                f"No element with index {index}. Call list_ui_elements first, "
                "or the index is out of range."
            )
        x, y = elem.center()
        label = elem.label or f"{elem.role.replace('AX', '')} at ({x},{y})"
        result = get_actuation_backend().click(x, y, double=double, confirm=confirm)
        return f"{result} [{index}] {label}"


@REGISTRY.register
class ClickAtTool(BaseTool):
    name = "click_at"
    category = "input"
    speak_text = "Clicking."
    description = (
        "Click at absolute screen coordinates (in pixels). Fallback when "
        "list_ui_elements doesn't expose the target — typically Electron, "
        "canvas, or game UIs. Take a screenshot first to find coordinates."
    )
    parameters = {
        "x": {"type": "integer", "description": "X pixel (0 = left)."},
        "y": {"type": "integer", "description": "Y pixel (0 = top)."},
        "double": {
            "type": "boolean",
            "description": "Double-click. Default false.",
        },
        "button": {
            "type": "string",
            "description": "'left', 'right', or 'middle'. Default 'left'.",
        },
        "confirm": {
            "type": "boolean",
            "description": "Override the sensitive-app safety gate after verbal confirmation.",
        },
    }
    required = ["x", "y"]

    def execute(self, x: int, y: int, double: bool = False, button: str = "left", confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("click_at")
        return get_actuation_backend().click(x, y, double=double, button=button, confirm=confirm)


@REGISTRY.register
class TypeTextTool(BaseTool):
    name = "type_text"
    category = "input"
    speak_text = "Typing."
    description = (
        "Type a string into the focused text field. Click the field first "
        "(via click_element or click_at) to focus it. Mimics human typing "
        "speed to avoid tripping anti-bot keystroke detection."
    )
    parameters = {
        "text": {"type": "string", "description": "The text to type."},
        "interval": {
            "type": "number",
            "description": "Per-character delay in seconds. Default 0.02.",
        },
        "confirm": {
            "type": "boolean",
            "description": "Override the sensitive-app safety gate after verbal confirmation.",
        },
    }
    required = ["text"]

    def execute(self, text: str, interval: float = 0.02, confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("type_text")
        return get_actuation_backend().type_text(text, interval=interval, confirm=confirm)


@REGISTRY.register
class PressKeyTool(BaseTool):
    name = "press_key"
    category = "input"
    speak_text = "Pressing."
    description = (
        "Press a key or key combo. Examples: 'enter', 'escape', 'tab', "
        "'cmd+f', 'cmd+shift+t', 'space', 'down'. Use for shortcuts and "
        "navigation that text input can't express."
    )
    parameters = {
        "keys": {
            "type": "string",
            "description": "Key or '+' joined combo, e.g. 'cmd+f' or 'enter'.",
        },
        "confirm": {
            "type": "boolean",
            "description": "Override the sensitive-app safety gate after verbal confirmation.",
        },
    }
    required = ["keys"]

    def execute(self, keys: str, confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("press_key")
        return get_actuation_backend().press_key(keys, confirm=confirm)


@REGISTRY.register
class ScrollTool(BaseTool):
    name = "scroll"
    category = "input"
    description = (
        "Scroll the mouse wheel at the cursor or at a given coordinate. "
        "Positive = up, negative = down. Use to reveal off-screen elements "
        "before listing or clicking them."
    )
    parameters = {
        "amount": {
            "type": "integer",
            "description": "Scroll units. + up, - down. Typical: 5 to 20.",
        },
        "x": {"type": "integer", "description": "Optional X to scroll at."},
        "y": {"type": "integer", "description": "Optional Y to scroll at."},
        "confirm": {
            "type": "boolean",
            "description": "Override the sensitive-app safety gate after verbal confirmation.",
        },
    }
    required = ["amount"]

    def execute(self, amount: int, x: Optional[int] = None, y: Optional[int] = None, confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("scroll")
        return get_actuation_backend().scroll(amount, x=x, y=y, confirm=confirm)


@REGISTRY.register
class ShortcatClickTool(BaseTool):
    name = "shortcat_click"
    category = "input"
    speak_text = "Clicking."
    description = (
        "Click a UI element by its visible label using ShortCat — a macOS "
        "command palette that indexes the Accessibility tree of the FRONTMOST "
        "app. Press the activation hotkey, type the label, hit Enter; ShortCat "
        "fuzzy-matches and clicks. Often more reliable than list_ui_elements + "
        "click_element for menu items, toolbar buttons, and sidebar entries — "
        "especially in apps where AX enumeration is shallow. Requires "
        "Shortcat.app installed (https://shortcat.app/) and `shortcat.enabled: "
        "true` in config.yaml."
    )
    parameters = {
        "label": {
            "type": "string",
            "description": (
                "Visible label or substring of the target element — "
                "e.g. 'Reload', 'New Document', 'Downloads', 'Wi-Fi'. "
                "Pick the shortest unambiguous prefix."
            ),
        },
        "confirm": {
            "type": "boolean",
            "description": (
                "Required true to override the sensitive-app safety gate. "
                "Only set after the user has verbally confirmed."
            ),
        },
    }
    required = ["label"]

    def execute(self, label: str, confirm: bool = False) -> str:
        if not is_macos():
            return macos_only("shortcat_click")

        # Lazy config import — keeps this tool's dependencies isolated and
        # leaves the rest of the registry usable when config.yaml is absent.
        try:
            from config import get_config
            cfg = get_config().shortcat
        except Exception as exc:
            return f"shortcat config unavailable: {type(exc).__name__}: {exc}"
        if not cfg.enabled:
            return (
                "shortcat is disabled in config.yaml. Set "
                "`shortcat.enabled: true` to use this tool."
            )

        # Make sure Shortcat.app is running — its hotkey is a no-op otherwise.
        try:
            subprocess.run(
                ["open", "-a", "Shortcat"], check=False,
                capture_output=True, timeout=5.0,
            )
        except Exception:
            pass

        try:
            import pyautogui as pg
        except ImportError as exc:
            return f"pyautogui not installed: {exc}"

        parts = [k.strip().lower() for k in cfg.hotkey.split("+") if k.strip()]
        if not parts:
            return f"shortcat hotkey {cfg.hotkey!r} is empty in config."

        try:
            if len(parts) == 1:
                pg.press(parts[0])
            else:
                pg.hotkey(*parts)
            time.sleep(cfg.palette_delay_ms / 1000.0)
            pg.typewrite(label, interval=0.025)
            time.sleep(0.15)
            pg.press("enter")
        except Exception as exc:
            return f"shortcat dispatch failed: {type(exc).__name__}: {exc}"

        time.sleep(cfg.settle_ms / 1000.0)
        return f"Dispatched ShortCat click for {label!r}."


@REGISTRY.register
class MouseMoveTool(BaseTool):
    name = "mouse_move"
    category = "input"
    description = (
        "Move the mouse cursor without clicking. Useful to dismiss tooltips "
        "or hover-reveal a menu before listing elements."
    )
    parameters = {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "duration": {
            "type": "number",
            "description": "Seconds to take. Default 0.1 (snappy).",
        },
    }
    required = ["x", "y"]

    def execute(self, x: int, y: int, duration: float = 0.1) -> str:
        if not is_macos():
            return macos_only("mouse_move")
        return get_actuation_backend().mouse_move(x, y, duration=duration)


@REGISTRY.register
class CuaStatusTool(BaseTool):
    name = "cua_status"
    category = "system"
    description = (
        "Report whether the cua-driver actuation backend is reachable. Prints "
        "the configured backend, the cua-driver binary path, and the live MCP "
        "tool inventory. Use to verify the cua integration before flipping "
        "computer_use.backend in config.yaml."
    )

    def execute(self) -> str:
        try:
            from config import get_config
            cfg = get_config().computer_use
        except Exception as exc:
            return f"computer_use config unavailable: {type(exc).__name__}: {exc}"

        lines = [
            f"configured backend: {cfg.backend}",
            f"cua binary: {cfg.cua_binary}",
            f"rpc timeout: {cfg.cua_timeout}s",
        ]
        try:
            from tools.cua_backend import get_cua_backend
            backend = get_cua_backend()
        except Exception as exc:
            lines.append(f"cua-driver: UNREACHABLE ({type(exc).__name__}: {exc})")
            return "\n".join(lines)

        names = backend.tool_names()
        lines.append(f"cua-driver: OK, {len(names)} tool(s) exposed")
        if names:
            lines.append("tools: " + ", ".join(names))
        return "\n".join(lines)
