"""Shared macOS scripting adapter — single source of truth for AppleScript calls.

Every desktop-automation tool that needs osascript, frontmost-app detection,
or browser enumeration imports from here. No more copy-pasted helpers.

This is a private module (`_` prefix) — tool modules use it internally; it's
not re-exported from `tools/__init__.py`.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


def is_macos() -> bool:
    """True on macOS, False otherwise. Check this before calling osascript."""
    return sys.platform == "darwin"


def macos_only(label: str) -> str:
    """Standard "only works on macOS" error message."""
    return f"{label} only works on macOS — current platform is {sys.platform}."


def osascript(script: str, *, timeout: float = 5.0) -> str:
    """Run an AppleScript and return stripped stdout. Raises on non-zero exit.

    Every AppleScript call in the codebase goes through this function — single
    choke point for timeout defaults, string escaping, and error formatting.
    """
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout, check=True,
    )
    return result.stdout.strip()


def frontmost_app_name() -> str:
    """Return the name of the frontmost macOS application."""
    return osascript(
        'tell application "System Events" to '
        'return name of first application process whose frontmost is true'
    )


# Apps we know how to query for URL / page-text / tabs.
_BROWSERS = {
    "Google Chrome", "Safari", "Arc", "Brave Browser",
    "Microsoft Edge", "Firefox",
}


def frontmost_browser() -> Optional[str]:
    """Return the frontmost browser app name, or None if the front app isn't a known browser."""
    try:
        app = frontmost_app_name()
    except Exception:
        return None
    return app if app in _BROWSERS else None


def browser_url_script(app: str) -> str:
    """AppleScript to get the URL of a browser's active tab."""
    if app == "Safari":
        return f'tell application "{app}" to return URL of current tab of front window'
    return f'tell application "{app}" to return URL of active tab of front window'
