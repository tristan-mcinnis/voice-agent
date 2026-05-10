"""Desktop-automation tools — clipboard, browser, capture, terminal, system info.

All macOS-specific tools live here. Non-Darwin platforms get clear error
messages so the tool schemas still register (the LLM sees the error and adapts).
Vision capture tools (screenshot, webcam, window) call through to `tools.vision`
for the description chain.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

from tools.registry import REGISTRY, BaseTool
from tools.vision import capture_path, describe_image, no_vision_message


# ---------------------------------------------------------------------------
# macOS helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return sys.platform == "darwin"


def _macos_only_msg(label: str) -> str:
    return f"{label} only works on macOS — current platform is {sys.platform}."


def _osascript(script: str, *, timeout: float = 5.0) -> str:
    """Run an AppleScript and return stripped stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout, check=True,
    )
    return result.stdout.strip()


def _frontmost_app_name() -> str:
    return _osascript(
        'tell application "System Events" to '
        'return name of first application process whose frontmost is true'
    )


# Apps we know how to query for URL / page-text / tabs.
_BROWSERS = {
    "Google Chrome", "Safari", "Arc", "Brave Browser",
    "Microsoft Edge", "Firefox",
}


def _frontmost_browser() -> Optional[str]:
    """Return the frontmost browser app name, or None if the front app isn't a known browser."""
    try:
        app = _frontmost_app_name()
    except Exception:
        return None
    return app if app in _BROWSERS else None


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def read_clipboard() -> str:
    """Return the current system clipboard text, or an explanatory string."""
    try:
        import pyperclip
    except ImportError:
        return "Clipboard tool unavailable: pyperclip is not installed."
    try:
        text = pyperclip.paste()
    except Exception as exc:
        return f"Clipboard read failed: {exc}"
    text = (text or "").strip()
    if not text:
        return "The clipboard is empty."
    return text


# ---------------------------------------------------------------------------
# Selection / focused input
# ---------------------------------------------------------------------------

def read_selected_text() -> str:
    """Copy whatever the user has selected in the frontmost app (Cmd+C) and return it.

    Restores the previous clipboard contents so the user's clipboard isn't
    silently clobbered.
    """
    if not _is_macos():
        return _macos_only_msg("Selected-text reading")
    try:
        import pyperclip
    except ImportError:
        return "Selected-text tool unavailable: pyperclip is not installed."

    try:
        original = pyperclip.paste()
    except Exception:
        original = None

    try:
        _osascript(
            'tell application "System Events" to keystroke "c" using command down'
        )
    except Exception as exc:
        return f"Could not trigger copy in the frontmost app: {exc}"

    # Give the OS a beat to update the clipboard. ~150ms is reliable across apps.
    time.sleep(0.18)

    try:
        text = (pyperclip.paste() or "").strip()
    except Exception as exc:
        return f"Could not read selected text: {exc}"

    if original is not None and original != text:
        try:
            pyperclip.copy(original)
        except Exception:
            pass

    if not text:
        return "Nothing was selected, or the frontmost app didn't respond to copy."
    return text


def read_focused_input() -> str:
    """Read the value of the focused text input/field in the frontmost app.

    Uses the AX (Accessibility) API via System Events; requires the host app
    (Terminal/iTerm/Claude/etc.) to have Accessibility permission granted in
    System Settings → Privacy & Security → Accessibility.
    """
    if not _is_macos():
        return _macos_only_msg("Focused-input reading")

    script = '''
    tell application "System Events"
        try
            set frontApp to first process whose frontmost is true
            set focused to value of attribute "AXFocusedUIElement" of frontApp
            return value of focused
        on error errMsg
            return "AX_ERROR:" & errMsg
        end try
    end tell
    '''
    try:
        result = _osascript(script)
    except Exception as exc:
        return f"Could not read focused input: {exc}"

    if result.startswith("AX_ERROR:"):
        return (
            f"Could not read focused input ({result[len('AX_ERROR:'):].strip()}). "
            "Make sure Accessibility permission is granted in System Settings."
        )
    if not result:
        return "No focused text input found."
    return result


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------

def _browser_url_script(app: str) -> str:
    if app == "Safari":
        return f'tell application "{app}" to return URL of current tab of front window'
    return f'tell application "{app}" to return URL of active tab of front window'


def read_browser_url() -> str:
    """Return the URL of the active tab in the frontmost browser."""
    if not _is_macos():
        return _macos_only_msg("Browser URL reading")
    app = _frontmost_browser()
    if app is None:
        return (
            "No supported browser is in the foreground. "
            "Bring Chrome, Safari, Arc, Brave, Edge, or Firefox to the front first."
        )
    try:
        url = _osascript(_browser_url_script(app))
    except Exception as exc:
        return f"Could not read URL from {app}: {exc}"
    return f"{app}: {url}" if url else f"{app}: (empty URL)"


def read_browser_page_text(max_chars: int = 8000) -> str:
    """Return innerText of the current browser page (truncated to max_chars).

    Requires the browser to allow JavaScript from Apple Events:
    Safari → Develop → Allow JavaScript from Apple Events.
    Chrome/Arc → View → Developer → Allow JavaScript from Apple Events.
    """
    if not _is_macos():
        return _macos_only_msg("Browser page reading")
    app = _frontmost_browser()
    if app is None:
        return "No supported browser is in the foreground."

    if app == "Safari":
        script = (
            'tell application "Safari" to '
            'do JavaScript "document.body.innerText" in current tab of front window'
        )
    elif app in {"Google Chrome", "Arc", "Brave Browser", "Microsoft Edge"}:
        script = (
            f'tell application "{app}" to tell active tab of front window to '
            'execute javascript "document.body.innerText"'
        )
    else:
        return f"Page-text reading not implemented for {app}."

    try:
        text = _osascript(script, timeout=8.0)
    except Exception as exc:
        return (
            f"Could not read page text from {app}: {exc}. "
            "Enable 'Allow JavaScript from Apple Events' in the browser's Develop menu."
        )

    text = text.strip()
    if not text:
        return f"The current {app} page has no readable text."
    if len(text) > max_chars:
        return f"{app} page text (truncated to {max_chars} chars):\n{text[:max_chars]}"
    return f"{app} page text:\n{text}"


def list_browser_tabs() -> str:
    """List title + URL of every tab in every window of the frontmost browser."""
    if not _is_macos():
        return _macos_only_msg("Browser tabs listing")
    app = _frontmost_browser()
    if app is None:
        return "No supported browser is in the foreground."

    if app == "Safari":
        script = '''
        tell application "Safari"
            set output to ""
            repeat with w in windows
                repeat with t in tabs of w
                    set output to output & (name of t) & "||" & (URL of t) & linefeed
                end repeat
            end repeat
            return output
        end tell
        '''
    else:
        script = f'''
        tell application "{app}"
            set output to ""
            repeat with w in windows
                repeat with t in tabs of w
                    set output to output & (title of t) & "||" & (URL of t) & linefeed
                end repeat
            end repeat
            return output
        end tell
        '''

    try:
        raw = _osascript(script, timeout=8.0)
    except Exception as exc:
        return f"Could not list tabs in {app}: {exc}"

    tabs = []
    for line in raw.splitlines():
        if "||" in line:
            title, url = line.split("||", 1)
            tabs.append(f"{title.strip()} — {url.strip()}")

    if not tabs:
        return f"No open tabs in {app}."
    header = f"{app} has {len(tabs)} tab{'s' if len(tabs) != 1 else ''}:"
    if len(tabs) > 25:
        return (
            header + "\n"
            + "\n".join(tabs[:25])
            + f"\n...and {len(tabs) - 25} more."
        )
    return header + "\n" + "\n".join(tabs)


# ---------------------------------------------------------------------------
# Capture (screenshot, webcam, window, region, display)
# ---------------------------------------------------------------------------

def take_screenshot(question: str = "") -> dict:
    """Capture the primary display, optionally describe it."""
    path = capture_path("screenshot")
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab()
        img.convert("RGB").save(path, "JPEG", quality=85)
    except Exception as exc:
        return {"result": f"Screenshot capture failed: {exc}"}

    prompt = question.strip() or "Briefly describe what's on this screen."
    description = describe_image(path, prompt)
    if description:
        return {"result": f"Screenshot taken. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


def capture_webcam(question: str = "") -> dict:
    """Capture one frame from the default webcam, optionally describe it."""
    try:
        import cv2
    except ImportError:
        return {"result": "Webcam tool unavailable: opencv-python is not installed."}

    path = capture_path("webcam")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return {"result": "Webcam capture failed: could not open default camera."}
    try:
        for _ in range(3):
            ok, frame = cap.read()
        if not ok or frame is None:
            return {"result": "Webcam capture failed: no frame received."}
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cap.release()

    prompt = question.strip() or "Briefly describe what you see in this webcam image."
    description = describe_image(path, prompt)
    if description:
        return {"result": f"Webcam image captured. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


def capture_frontmost_window(question: str = "") -> dict:
    """Capture the bounds of the frontmost window via screencapture and describe it."""
    if not _is_macos():
        return {"result": _macos_only_msg("Window capture")}

    script = '''
    tell application "System Events"
        set frontApp to first process whose frontmost is true
        set frontWindow to front window of frontApp
        set {x, y} to position of frontWindow
        set {w, h} to size of frontWindow
        return (x as string) & " " & (y as string) & " " & (w as string) & " " & (h as string)
    end tell
    '''
    try:
        bounds = _osascript(script).split()
        x, y, w, h = bounds
    except Exception as exc:
        return {"result": f"Could not get frontmost window bounds: {exc}"}

    path = capture_path("window")
    try:
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", "-R", f"{x},{y},{w},{h}", path],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return {"result": f"Window capture failed: {exc}"}

    prompt = question.strip() or "Briefly describe what's in this window."
    description = describe_image(path, prompt)
    if description:
        return {"result": f"Window captured. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


def capture_screen_region(
    x: int, y: int, width: int, height: int, question: str = ""
) -> dict:
    """Capture a rectangular region of the screen and describe it."""
    if not _is_macos():
        return {"result": _macos_only_msg("Region capture")}
    path = capture_path("region")
    try:
        subprocess.run(
            [
                "screencapture", "-x", "-t", "jpg", "-R",
                f"{x},{y},{width},{height}", path,
            ],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return {"result": f"Region capture failed: {exc}"}

    prompt = question.strip() or "Briefly describe what's in this screen region."
    description = describe_image(path, prompt)
    if description:
        return {"result": f"Region captured. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


def capture_display(display: int = 1, question: str = "") -> dict:
    """Capture an entire display by index (1 = primary) and describe it."""
    if not _is_macos():
        return {"result": _macos_only_msg("Display capture")}
    path = capture_path(f"display{display}")
    try:
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", f"-D{display}", path],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return {"result": f"Display capture failed: {exc}"}

    prompt = question.strip() or f"Briefly describe what's on display {display}."
    description = describe_image(path, prompt)
    if description:
        return {"result": f"Display {display} captured. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def get_frontmost_app() -> str:
    """Return the name of the frontmost (focused) application."""
    if not _is_macos():
        return _macos_only_msg("Frontmost-app reading")
    try:
        return f"Frontmost app: {_frontmost_app_name()}"
    except Exception as exc:
        return f"Could not determine frontmost app: {exc}"


def list_running_apps() -> str:
    """List foreground (visible) running applications."""
    if not _is_macos():
        return _macos_only_msg("Running apps listing")
    script = (
        'tell application "System Events" to '
        'return name of every application process whose background only is false'
    )
    try:
        output = _osascript(script)
    except Exception as exc:
        return f"Could not list running apps: {exc}"

    apps = sorted(
        {a.strip() for a in output.split(",") if a.strip()}, key=str.lower
    )
    return f"{len(apps)} running app{'s' if len(apps) != 1 else ''}: " + ", ".join(apps)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def read_terminal_output(max_lines: int = 80) -> str:
    """Return visible output from the frontmost Terminal.app or iTerm2 window."""
    if not _is_macos():
        return _macos_only_msg("Terminal output reading")

    try:
        app = _frontmost_app_name()
    except Exception as exc:
        return f"Could not determine frontmost app: {exc}"

    if app == "Terminal":
        script = (
            'tell application "Terminal" to '
            'return contents of selected tab of front window'
        )
    elif app in {"iTerm2", "iTerm"}:
        script = '''
        tell application "iTerm2"
            tell current session of current window
                return contents
            end tell
        end tell
        '''
    else:
        return (
            f"Frontmost app is {app}, not a known terminal. "
            "Bring Terminal or iTerm2 to the foreground."
        )

    try:
        text = _osascript(script, timeout=5.0)
    except Exception as exc:
        return f"Could not read {app} output: {exc}"

    if not text.strip():
        return f"{app} has no visible output."

    lines = text.splitlines()
    if len(lines) > max_lines:
        return (
            f"{app} output (last {max_lines} of {len(lines)} lines):\n"
            + "\n".join(lines[-max_lines:])
        )
    return f"{app} output:\n{text}"


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

@REGISTRY.register
class ReadClipboardTool(BaseTool):
    name = "read_clipboard"
    category = "input"
    description = (
        "Read the current system clipboard text. Use when the user asks "
        "about what they copied or what's on their clipboard."
    )

    def execute(self) -> str:
        return read_clipboard()


@REGISTRY.register
class ReadSelectedTextTool(BaseTool):
    name = "read_selected_text"
    category = "input"
    speak_text = "Reading your selection."
    description = (
        "Read whatever text the user currently has selected/highlighted in the "
        "frontmost app (macOS only). Triggers Cmd+C and reads the result. "
        "Use when the user says 'look at what I have selected' or 'what does this say'."
    )

    def execute(self) -> str:
        return read_selected_text()


@REGISTRY.register
class ReadFocusedInputTool(BaseTool):
    name = "read_focused_input"
    category = "input"
    description = (
        "Read the value of the focused text field in the frontmost app (macOS only). "
        "Use when the user says 'what am I typing' or 'look at this input'. "
        "Often fails on Electron-based apps (VS Code, Slack, Gemini, etc.) — "
        "prefer `read_selected_text` there."
    )

    def execute(self) -> str:
        return read_focused_input()


@REGISTRY.register
class ReadBrowserUrlTool(BaseTool):
    name = "read_browser_url"
    category = "browser"
    speak_text = "Checking your browser."
    description = (
        "Return the URL of the current tab of the frontmost browser "
        "(Chrome, Safari, Arc, Brave, Edge, Firefox; macOS only)."
    )

    def execute(self) -> str:
        return read_browser_url()


@REGISTRY.register
class ReadBrowserPageTextTool(BaseTool):
    name = "read_browser_page_text"
    category = "browser"
    speak_text = "Reading that page."
    description = (
        "Return the visible text of the current browser page via JavaScript "
        "(macOS only). Requires 'Allow JavaScript from Apple Events' in the browser. "
        "Use when the user asks about the page they're looking at."
    )
    parameters = {
        "max_chars": {
            "type": "integer",
            "description": "Truncate after this many chars. Default 8000.",
        },
    }

    def execute(self, max_chars: int = 8000) -> str:
        return read_browser_page_text(max_chars)


@REGISTRY.register
class ListBrowserTabsTool(BaseTool):
    name = "list_browser_tabs"
    category = "browser"
    speak_text = "Checking your tabs."
    description = (
        "List the title + URL of every open tab in the frontmost browser (macOS only)."
    )

    def execute(self) -> str:
        return list_browser_tabs()


@REGISTRY.register
class TakeScreenshotTool(BaseTool):
    name = "take_screenshot"
    category = "vision"
    speak_text = "Looking at your screen."
    description = (
        "Capture the user's screen and (if a vision provider is configured) "
        "describe what's on it. Use when the user asks about what's on their screen."
    )
    parameters = {
        "question": {
            "type": "string",
            "description": "What the user wants to know about the screen, used as the vision prompt. Optional.",
        },
    }

    def execute(self, question: str = "") -> dict:
        return take_screenshot(question)


@REGISTRY.register
class CaptureFrontmostWindowTool(BaseTool):
    name = "capture_frontmost_window"
    category = "vision"
    speak_text = "Looking at your window."
    description = (
        "Capture the frontmost application window (macOS only) and describe it via "
        "the configured vision provider. Use when the user says 'look at this window'."
    )
    parameters = {
        "question": {"type": "string", "description": "What to look for. Optional."},
    }

    def execute(self, question: str = "") -> dict:
        return capture_frontmost_window(question)


@REGISTRY.register
class CaptureScreenRegionTool(BaseTool):
    name = "capture_screen_region"
    category = "vision"
    speak_text = "Looking at that area."
    description = (
        "Capture a rectangular region of the screen (macOS only) and describe it. "
        "Coordinates are in screen points: (0,0) is top-left."
    )
    parameters = {
        "x": {"type": "integer", "description": "Left edge in screen points."},
        "y": {"type": "integer", "description": "Top edge in screen points."},
        "width": {"type": "integer", "description": "Region width in points."},
        "height": {"type": "integer", "description": "Region height in points."},
        "question": {"type": "string", "description": "What to look for. Optional."},
    }
    required = ["x", "y", "width", "height"]

    def execute(
        self, x: int, y: int, width: int, height: int, question: str = ""
    ) -> dict:
        return capture_screen_region(x, y, width, height, question)


@REGISTRY.register
class CaptureDisplayTool(BaseTool):
    name = "capture_display"
    category = "vision"
    speak_text = "Looking at that display."
    description = (
        "Capture a specific display/monitor by index (1 = primary; macOS only). "
        "Note: monitors, not virtual desktops/Spaces — those aren't capturable."
    )
    parameters = {
        "display": {"type": "integer", "description": "Display index (1 = primary)."},
        "question": {"type": "string", "description": "What to look for. Optional."},
    }

    def execute(self, display: int = 1, question: str = "") -> dict:
        return capture_display(display, question)


@REGISTRY.register
class CaptureWebcamTool(BaseTool):
    name = "capture_webcam"
    category = "vision"
    speak_text = "Let me see."
    description = (
        "Capture one frame from the user's webcam and (if vision is configured) "
        "describe it. Use when the user asks 'what do you see' or wants you to "
        "look at them or what they're holding."
    )
    parameters = {
        "question": {
            "type": "string",
            "description": "What the user wants you to look for, used as the vision prompt. Optional.",
        },
    }

    def execute(self, question: str = "") -> dict:
        return capture_webcam(question)


@REGISTRY.register
class GetFrontmostAppTool(BaseTool):
    name = "get_frontmost_app"
    category = "system"
    description = (
        "Return the name of the application currently in the foreground "
        "(macOS only). Use when the user asks 'what app am I in?'."
    )

    def execute(self) -> str:
        return get_frontmost_app()


@REGISTRY.register
class ListRunningAppsTool(BaseTool):
    name = "list_running_apps"
    category = "system"
    description = "List the foreground (visible) running applications (macOS only)."

    def execute(self) -> str:
        return list_running_apps()


@REGISTRY.register
class ReadTerminalOutputTool(BaseTool):
    name = "read_terminal_output"
    category = "system"
    speak_text = "Checking your terminal."
    description = (
        "Return the visible output of the frontmost Terminal.app or iTerm2 window "
        "(macOS only). Use when the user says 'look at my terminal'."
    )
    parameters = {
        "max_lines": {
            "type": "integer",
            "description": "Cap to last N lines. Default 80.",
        },
    }

    def execute(self, max_lines: int = 80) -> str:
        return read_terminal_output(max_lines)
