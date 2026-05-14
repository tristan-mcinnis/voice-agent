"""Vision capture tools — screenshot, webcam, window, region, display.

Two layers:

  - ``ScreenCapture`` — the single seam for getting an image off the screen
    or webcam. Five methods (``screenshot``, ``window``, ``region``,
    ``display``, ``webcam``), each returning a saved image path. Every tool
    that needs pixels goes through this — including ``grounding_vision`` —
    so there is exactly one screencapture invocation in the codebase.

  - Tool classes — thin ``BaseTool`` subclasses that pick a capture method
    and route the result through ``_capture_and_describe``.

All implementations live in ``BaseTool.execute()`` methods — the canonical
interface.
"""

from __future__ import annotations

import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from tools._macos import is_macos, macos_only, osascript
from tools.registry import REGISTRY, BaseTool
from tools.vision import describe_image, no_vision_message


# ---------------------------------------------------------------------------
# Capture path helpers
# ---------------------------------------------------------------------------

def captures_dir() -> Path:
    """Where capture tools write their images. Mirrors session_log's VOICE_BOT_LOG_DIR."""
    base = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))
    d = base / "captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def capture_path(kind: str, ext: str = "jpg") -> str:
    """Build a timestamped path under logs/captures/."""
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return str(captures_dir() / f"{kind}-{ts}.{ext}")


# ---------------------------------------------------------------------------
# ScreenCapture — single seam for all pixel acquisition
# ---------------------------------------------------------------------------

class ScreenCapture:
    """Acquire pixels from the screen or webcam. One path per modality."""

    def screenshot(self, kind: str = "screenshot") -> str:
        """Full-screen capture. Uses macOS screencapture when available,
        falls back to PIL.ImageGrab so non-macOS hosts still work for
        screenshots."""
        path = capture_path(kind)
        if is_macos():
            subprocess.run(
                ["screencapture", "-x", "-t", "jpg", path],
                check=True, capture_output=True, timeout=8.0,
            )
            return path
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.convert("RGB").save(path, "JPEG", quality=85)
        return path

    def window(self, kind: str = "window") -> str:
        """Frontmost-window capture (macOS only)."""
        if not is_macos():
            raise RuntimeError(macos_only("Window capture"))
        script = '''
        tell application "System Events"
            set frontApp to first process whose frontmost is true
            set frontWindow to front window of frontApp
            set {x, y} to position of frontWindow
            set {w, h} to size of frontWindow
            return (x as string) & " " & (y as string) & " " & (w as string) & " " & (h as string)
        end tell
        '''
        bounds = osascript(script, timeout=5.0).split()
        x, y, w, h = bounds
        path = capture_path(kind)
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", "-R", f"{x},{y},{w},{h}", path],
            check=True, capture_output=True, timeout=10.0,
        )
        return path

    def region(self, x: int, y: int, width: int, height: int, kind: str = "region") -> str:
        """Rectangular region capture (macOS only)."""
        if not is_macos():
            raise RuntimeError(macos_only("Region capture"))
        path = capture_path(kind)
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", "-R",
             f"{x},{y},{width},{height}", path],
            check=True, capture_output=True, timeout=10.0,
        )
        return path

    def display(self, n: int = 1, kind: str | None = None) -> str:
        """Specific display/monitor capture by index (macOS only, 1 = primary)."""
        if not is_macos():
            raise RuntimeError(macos_only("Display capture"))
        label = kind or f"display{n}"
        path = capture_path(label)
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", f"-D{n}", path],
            check=True, capture_output=True, timeout=10.0,
        )
        return path

    def webcam(self, kind: str = "webcam") -> str:
        """One frame from the default camera."""
        import cv2
        path = capture_path(kind)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open default camera.")
        try:
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("No frame received from webcam.")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        finally:
            cap.release()
        return path


# Module-level instance — the seam. Tests can substitute a fake at
# ``tools.capture.SCREEN_CAPTURE`` before calling the tools.
SCREEN_CAPTURE = ScreenCapture()


# ---------------------------------------------------------------------------
# VisionCapture adapter — capture + describe in one shot
# ---------------------------------------------------------------------------

def _capture_and_describe(
    kind: str,
    capture_fn: Callable[[], str],
    prompt: str,
) -> dict:
    """Run ``capture_fn`` to get an image path, then describe it.

    ``capture_fn()`` is a zero-arg callable returning the saved path. All
    callers route through ``SCREEN_CAPTURE`` so there is exactly one screen-
    capture implementation in the codebase.
    """
    try:
        path = capture_fn()
    except Exception as exc:
        return {"result": f"{kind} capture failed: {exc}"}

    description = describe_image(path, prompt)
    if description:
        return {"result": f"{kind} captured. {description}", "image_path": path}
    return {"result": no_vision_message(), "image_path": path}


# ---------------------------------------------------------------------------
# Tool classes — one per capture type, each ~10-20 lines
# ---------------------------------------------------------------------------

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
        prompt = question.strip() or "Briefly describe what's on this screen."
        return _capture_and_describe("screenshot", SCREEN_CAPTURE.screenshot, prompt)


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
        prompt = question.strip() or "Briefly describe what you see in this webcam image."
        return _capture_and_describe("webcam", SCREEN_CAPTURE.webcam, prompt)


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
        if not is_macos():
            return {"result": macos_only("Window capture")}
        prompt = question.strip() or "Briefly describe what's in this window."
        return _capture_and_describe("window", SCREEN_CAPTURE.window, prompt)


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
        if not is_macos():
            return {"result": macos_only("Region capture")}
        prompt = question.strip() or "Briefly describe what's in this screen region."
        return _capture_and_describe(
            "region",
            lambda: SCREEN_CAPTURE.region(x, y, width, height),
            prompt,
        )


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
        if not is_macos():
            return {"result": macos_only("Display capture")}
        prompt = question.strip() or f"Briefly describe what's on display {display}."
        return _capture_and_describe(
            f"display{display}",
            lambda: SCREEN_CAPTURE.display(display),
            prompt,
        )
