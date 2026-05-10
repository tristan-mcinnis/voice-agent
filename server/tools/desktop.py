"""Desktop-automation tools — clipboard, browser, terminal, system info.

Vision capture tools (screenshot, webcam, window, region, display) live in
``tools/capture.py`` — this module only handles automation that reads state
from macOS.

All implementations live in BaseTool.execute() methods — the canonical interface.
Thin backward-compat callables are re-exported from __init__.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Optional

from tools._macos import (
    is_macos,
    macos_only,
    osascript,
    frontmost_app_name,
    frontmost_browser,
    browser_url_script,
)
from tools.registry import REGISTRY, BaseTool


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
        if not is_macos():
            return macos_only("Selected-text reading")
        try:
            import pyperclip
        except ImportError:
            return "Selected-text tool unavailable: pyperclip is not installed."

        try:
            original = pyperclip.paste()
        except Exception:
            original = None

        try:
            osascript(
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
        if not is_macos():
            return macos_only("Focused-input reading")

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
            result = osascript(script)
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
        if not is_macos():
            return macos_only("Browser URL reading")
        app = frontmost_browser()
        if app is None:
            return (
                "No supported browser is in the foreground. "
                "Bring Chrome, Safari, Arc, Brave, Edge, or Firefox to the front first."
            )
        try:
            url = osascript(browser_url_script(app))
        except Exception as exc:
            return f"Could not read URL from {app}: {exc}"
        return f"{app}: {url}" if url else f"{app}: (empty URL)"


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
        if not is_macos():
            return macos_only("Browser page reading")
        app = frontmost_browser()
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
            text = osascript(script, timeout=8.0)
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


@REGISTRY.register
class ListBrowserTabsTool(BaseTool):
    name = "list_browser_tabs"
    category = "browser"
    speak_text = "Checking your tabs."
    description = (
        "List the title + URL of every open tab in the frontmost browser (macOS only)."
    )

    def execute(self) -> str:
        if not is_macos():
            return macos_only("Browser tabs listing")
        app = frontmost_browser()
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
            raw = osascript(script, timeout=8.0)
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


@REGISTRY.register
class GetFrontmostAppTool(BaseTool):
    name = "get_frontmost_app"
    category = "system"
    description = (
        "Return the name of the application currently in the foreground "
        "(macOS only). Use when the user asks 'what app am I in?'."
    )

    def execute(self) -> str:
        if not is_macos():
            return macos_only("Frontmost-app reading")
        try:
            return f"Frontmost app: {frontmost_app_name()}"
        except Exception as exc:
            return f"Could not determine frontmost app: {exc}"


@REGISTRY.register
class ListRunningAppsTool(BaseTool):
    name = "list_running_apps"
    category = "system"
    description = "List the foreground (visible) running applications (macOS only)."

    def execute(self) -> str:
        if not is_macos():
            return macos_only("Running apps listing")
        script = (
            'tell application "System Events" to '
            'return name of every application process whose background only is false'
        )
        try:
            output = osascript(script)
        except Exception as exc:
            return f"Could not list running apps: {exc}"

        apps = sorted(
            {a.strip() for a in output.split(",") if a.strip()}, key=str.lower
        )
        return f"{len(apps)} running app{'s' if len(apps) != 1 else ''}: " + ", ".join(apps)


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
        if not is_macos():
            return macos_only("Terminal output reading")

        try:
            app = frontmost_app_name()
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
            text = osascript(script, timeout=5.0)
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
