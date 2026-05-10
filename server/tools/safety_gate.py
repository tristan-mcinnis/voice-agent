"""Safety gate ("Tirith"-style HITL) — blocks mutating computer-use actions in sensitive contexts.

Extracted from ``computer_use.py`` so any tool category can apply the same
HITL pattern without coupling to pyautogui, AX enumeration, or Shortcat.

The gate checks the frontmost macOS app name + focused window title against
a substring blocklist. A match returns a BLOCKED message; the agent must
verbalise the action, get verbal confirmation, and re-call with ``confirm=True``.

Usage::

    from tools.safety_gate import SafetyGate

    gate = SafetyGate()

    def execute(self, x: int, y: int, confirm: bool = False) -> str:
        refusal = gate.check(f"click at ({x},{y})", confirm)
        if refusal:
            return refusal
        # ... proceed with action
"""

from __future__ import annotations

from typing import Optional


# Substring patterns checked against frontmost app name + window title.
# Lowercase. A match forces the agent to verbalize intent and call again
# with confirm=True. Tune to taste — anything financial / credential / system.
_SENSITIVE_PATTERNS = (
    "1password", "lastpass", "bitwarden", "keychain", "keychain access",
    "password", "secrets", "wallet",
    "banking", "bank of america", "chase", "wells fargo", "citi",
    "venmo", "paypal", "stripe", "square",
    "system settings", "system preferences", "privacy & security",
    "filevault", "firewall",
    "delete", "erase", "format",
)


def _frontmost_app_name() -> str:
    """Return the name of the frontmost macOS app, or '' on failure."""
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to '
             'return name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=3.0, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _front_window_title() -> str:
    """Return the title of the frontmost app's focused window, or ''."""
    try:
        from AppKit import NSWorkspace  # type: ignore
        from ApplicationServices import (  # type: ignore
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
        )
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ""
        ref = AXUIElementCreateApplication(int(app.processIdentifier()))
        err, win = AXUIElementCopyAttributeValue(ref, "AXFocusedWindow", None)
        if err or win is None:
            return ""
        err, title = AXUIElementCopyAttributeValue(win, "AXTitle", None)
        return "" if err else str(title or "")
    except Exception:
        return ""


class SafetyGate:
    """Check whether a mutating action should be blocked by HITL.

    The gate is stateless — each ``check()`` re-reads the frontmost app
    and window title, so it always reflects the current desktop context.

    Tests can override ``_patterns`` to inject custom sensitive contexts
    without touching the module-level tuple.
    """

    def __init__(self, patterns: tuple[str, ...] | None = None) -> None:
        self._patterns = patterns or _SENSITIVE_PATTERNS

    def check(self, action: str, confirm: bool) -> Optional[str]:
        """Return a BLOCKED message if the front context is sensitive and unconfirmed.

        Args:
            action: Human-readable description of the action (e.g.
                "click element [3]", "type 'password123'", "press cmd+w").
            confirm: Whether the user has verbally confirmed. When True,
                the gate always passes.

        Returns:
            A refusal string if blocked, or None if the action may proceed.
        """
        if confirm:
            return None
        ctx = self._sensitive_context()
        if ctx is None:
            return None
        return (
            f"BLOCKED: about to {action} in a sensitive context ({ctx}). "
            "Speak this aloud to the user, ask them to confirm, then call again "
            "with confirm=true. Do NOT proceed silently."
        )

    def _sensitive_context(self) -> Optional[str]:
        """Return a description of the sensitive app/window if any, else None."""
        proc = _frontmost_app_name()
        title = _front_window_title()
        haystack = f"{proc} {title}".lower()
        for pat in self._patterns:
            if pat in haystack:
                return f"{proc} — {title}" if title else proc
        return None
