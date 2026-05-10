"""Multimodal function-call tools shared by bot.py and local_bot.py.

Every tool is a `Tool` record (schema + sync function + optional spoken filler).
`register_all` walks the registry once: registers each handler on the LLM and
returns the list of `FunctionSchema` to wrap in a `ToolsSchema`. Adding a tool
means adding one entry to `TOOLS`.

Heavy imports are lazy so missing optional deps don't crash the bot — they just
disable the affected tool with an explanatory error string.

Vision describe uses OpenAI gpt-4o-mini if OPENAI_API_KEY is set; otherwise the
image is saved and the tool returns the path only.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema

SCREENSHOT_PATH = os.path.join(tempfile.gettempdir(), "voice_bot_screenshot.png")
WEBCAM_PATH = os.path.join(tempfile.gettempdir(), "voice_bot_webcam.jpg")


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
# Image capture (screenshot, webcam) + optional vision describe
# ---------------------------------------------------------------------------

def _prepare_image(image_path: str, max_width: int) -> tuple[str, str]:
    """Read the image, optionally downscale, return (base64, ext)."""
    if max_width and max_width > 0:
        try:
            from PIL import Image
            from io import BytesIO

            with Image.open(image_path) as img:
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.convert("RGB").resize(new_size, Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, "JPEG", quality=85)
                    return base64.b64encode(buf.getvalue()).decode(), "jpeg"
        except Exception as exc:
            logger.warning(f"Image downscale failed, sending original: {exc}")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "png" if image_path.endswith(".png") else "jpeg"
    return b64, ext


def _downscaled_image_path(image_path: str, max_width: int) -> str:
    """Write a downscaled JPEG next to the original; return its path.

    Used by the in-process MLX path which takes a file path, not bytes.
    """
    if not max_width or max_width <= 0:
        return image_path
    try:
        from PIL import Image as PILImage

        with PILImage.open(image_path) as img:
            if img.width <= max_width:
                return image_path
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            resized = img.convert("RGB").resize(new_size, PILImage.LANCZOS)
            small_path = image_path.rsplit(".", 1)[0] + ".small.jpg"
            resized.save(small_path, "JPEG", quality=85)
            return small_path
    except Exception as exc:
        logger.warning(f"Image downscale failed, sending original: {exc}")
        return image_path


_REASONING_MARKERS = (
    "wait,", "actually,", "let me think", "let's think", "let's keep",
    "let's go", "possible answer", "or simpler", "or even shorter",
    "the user", "the question", "i need to", "key elements",
)


def _strip_reasoning(text: str) -> str:
    """Drop chain-of-thought preamble from reasoning-model output.

    Some reasoning models (kimi-k2.x at high load, etc.) dump their CoT into
    `content` instead of `reasoning_content`. Heuristic: if reasoning markers
    appear, take the last non-empty paragraph that doesn't itself contain
    markers, and unwrap surrounding quotes.
    """
    if not text:
        return text
    lower = text.lower()
    if not any(m in lower for m in _REASONING_MARKERS):
        return text.strip()

    # Walk lines in reverse and collect the longest contiguous tail of
    # non-marker lines — that's typically the final answer.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    tail: list[str] = []
    for line in reversed(lines):
        if any(m in line.lower() for m in _REASONING_MARKERS):
            break
        tail.append(line)
    if tail:
        result = " ".join(reversed(tail)).strip()
    else:
        result = lines[-1] if lines else text
    return result.strip().strip('"').strip("'").strip()


def _try_describe_with(provider, image_path: str, prompt: str) -> str | None:
    """Attempt one provider. Returns text on success, None on any failure."""
    full_prompt = (
        f"{prompt}\n\n{provider.brevity_suffix}".strip()
        if provider.brevity_suffix else prompt
    )

    if provider.kind == "mlx":
        try:
            import mlx_vision
        except ImportError:
            logger.info(f"vision: skipping {provider.name} — mlx_vlm not installed")
            return None
        small_path = _downscaled_image_path(image_path, provider.max_image_width)
        try:
            t0 = time.monotonic()
            text = _strip_reasoning(mlx_vision.describe(
                provider.model, small_path, full_prompt,
                max_tokens=provider.max_tokens,
            ))
            elapsed = time.monotonic() - t0
            kb = os.path.getsize(small_path) // 1024
            logger.info(
                f"vision: {provider.name}/{provider.model} "
                f"img_kb={kb} out_chars={len(text)} elapsed={elapsed:.1f}s"
            )
            return text or None
        except Exception as exc:
            logger.warning(f"vision: {provider.name} failed: {exc}")
            return None

    # kind == "openai" (default): OpenAI-compatible HTTP endpoint.
    from openai import OpenAI

    if provider.api_key_env:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            logger.info(
                f"vision: skipping {provider.name} — {provider.api_key_env} unset"
            )
            return None
    else:
        # LM Studio and other unauthenticated endpoints. SDK requires a string.
        api_key = "no-auth"

    b64, ext = _prepare_image(image_path, provider.max_image_width)

    client = OpenAI(
        api_key=api_key, base_url=provider.base_url,
        max_retries=0, timeout=provider.timeout,
    )

    last_exc = None
    for attempt in range(2):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=provider.model,
                messages=[{
                    "role": "user",
                    # Image FIRST, text SECOND — Kimi's docs canonicalise this
                    # order. OpenAI is order-agnostic so this is safe everywhere.
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{ext};base64,{b64}",
                        }},
                        {"type": "text", "text": full_prompt},
                    ],
                }],
                max_tokens=provider.max_tokens,
            )
            elapsed = time.monotonic() - t0
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            # Reasoning models (kimi-k2.x, deepseek-reasoner, etc.) can leave
            # `content` empty and put the answer in `reasoning_content`. Fall
            # back to that if content is empty.
            reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
            text = _strip_reasoning(content or reasoning)
            logger.info(
                f"vision: {provider.name}/{provider.model} "
                f"img_kb={len(b64)*3//4//1024} out_chars={len(text)} "
                f"src={'content' if content else 'reasoning'} "
                f"elapsed={elapsed:.1f}s"
            )
            return text or None
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and "overload" in str(exc).lower():
                time.sleep(2.0)
                continue
            break
    logger.warning(f"vision: {provider.name} failed: {last_exc}")
    return None


def _describe_image(image_path: str, prompt: str) -> str | None:
    """Walk the vision provider chain in config.yaml; first success wins.

    Skipped if `api_key_env` is set in config but the env var is missing.
    Any other failure (timeout, 503, 429, model crash, overload) falls
    through to the next provider.
    """
    from config import load_config

    providers = load_config().vision
    if not providers:
        return None

    for provider in providers:
        result = _try_describe_with(provider, image_path, prompt)
        if result:
            return result
    return None


def _no_vision_message() -> str:
    """Build a spoken-friendly message explaining why vision didn't work."""
    from config import load_config

    providers = load_config().vision
    if not providers:
        return (
            "I captured the image but vision is disabled in config.yaml. "
            "Configure at least one vision provider there to enable describe."
        )
    names = ", ".join(p.name for p in providers)
    missing = [
        p.api_key_env for p in providers
        if p.api_key_env and not os.getenv(p.api_key_env)
    ]
    if missing:
        return (
            f"I captured the image but every vision provider failed. "
            f"Missing env vars: {', '.join(missing)}. "
            f"Tried providers: {names}."
        )
    return (
        f"I captured the image but every vision provider failed. "
        f"Tried: {names}. Check the server logs for the exact errors."
    )


def take_screenshot(question: str = "") -> str:
    """Capture the primary display, optionally describe it, return a sentence for the LLM."""
    try:
        from PIL import ImageGrab

        img = ImageGrab.grab()
        img.convert("RGB").save(SCREENSHOT_PATH, "PNG")
    except Exception as exc:
        return f"Screenshot capture failed: {exc}"

    prompt = question.strip() or "Briefly describe what's on this screen."
    description = _describe_image(SCREENSHOT_PATH, prompt)
    if description:
        return f"Screenshot taken. {description}"
    return _no_vision_message()


def capture_webcam(question: str = "") -> str:
    """Capture one frame from the default webcam, optionally describe it."""
    try:
        import cv2
    except ImportError:
        return "Webcam tool unavailable: opencv-python is not installed."

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return "Webcam capture failed: could not open default camera."
    try:
        # First frame is often blank/dark on macOS — read a couple to let exposure settle.
        for _ in range(3):
            ok, frame = cap.read()
        if not ok or frame is None:
            return "Webcam capture failed: no frame received."
        cv2.imwrite(WEBCAM_PATH, frame)
    finally:
        cap.release()

    prompt = question.strip() or "Briefly describe what you see in this webcam image."
    description = _describe_image(WEBCAM_PATH, prompt)
    if description:
        return f"Webcam image captured. {description}"
    return _no_vision_message()


# ---------------------------------------------------------------------------
# Web search via Serper
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Google search via Serper. Returns formatted top results."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Web search unavailable: SERPER_API_KEY is not set."
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Web search failed: {exc}"

    results = data.get("organic", [])[:max_results]
    if not results:
        return f"No results found for: {query}"
    lines = [f"Top results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')} — {r.get('snippet', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: weather (dummy data) — kept here so adding any tool is one entry below.
# ---------------------------------------------------------------------------

def get_current_weather(location: str = "", format: str = "fahrenheit") -> dict:
    """Dummy weather to demonstrate function calling end-to-end."""
    return {"conditions": "nice", "temperature": "75"}


# ---------------------------------------------------------------------------
# Platform helpers (macOS-only tools below)
#
# Most "look at my screen / window / file / browser" tools shell out to
# `osascript` or `screencapture`. They short-circuit with a clear message on
# non-Darwin platforms so the cloud bot (Linux container) can still register
# them — the LLM just sees an explanatory error and adapts.
# ---------------------------------------------------------------------------

WINDOW_PATH = os.path.join(tempfile.gettempdir(), "voice_bot_window.png")
REGION_PATH = os.path.join(tempfile.gettempdir(), "voice_bot_region.png")
DISPLAY_PATH = os.path.join(tempfile.gettempdir(), "voice_bot_display.png")

# Apps we know how to query for URL / page-text / tabs.
_BROWSERS = {
    "Google Chrome", "Safari", "Arc", "Brave Browser",
    "Microsoft Edge", "Firefox",
}


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


def _frontmost_browser() -> Optional[str]:
    """Return the frontmost browser app name, or None if the front app isn't a known browser."""
    try:
        app = _frontmost_app_name()
    except Exception:
        return None
    return app if app in _BROWSERS else None


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
        _osascript('tell application "System Events" to keystroke "c" using command down')
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
# File system
# ---------------------------------------------------------------------------

def read_file(path: str, max_bytes: int = 100_000) -> str:
    """Read a file and return its content (truncated to max_bytes)."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {p}"
    if p.is_dir():
        return f"{p} is a directory, not a file. Use list_folder instead."
    try:
        size = p.stat().st_size
        with open(p, "r", errors="replace") as fh:
            content = fh.read(max_bytes)
        if size > max_bytes:
            return (
                f"File {p.name} ({size} bytes — truncated to {max_bytes}):\n{content}"
            )
        return f"File {p.name} ({size} bytes):\n{content}"
    except Exception as exc:
        return f"Could not read {p}: {exc}"


def list_folder(path: str, recursive: bool = False, max_items: int = 50) -> str:
    """List the contents of a folder. `recursive` walks subfolders."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Folder not found: {p}"
    if not p.is_dir():
        return f"{p} is not a folder."

    try:
        items = list(p.rglob("*")) if recursive else list(p.iterdir())
    except Exception as exc:
        return f"Could not list {p}: {exc}"

    # Folders first, then files, alphabetically.
    items.sort(key=lambda i: (not i.is_dir(), i.name.lower()))
    total = len(items)
    shown = items[:max_items]

    lines = [f"Folder {p} contains {total} item{'s' if total != 1 else ''}:"]
    for item in shown:
        kind = "folder" if item.is_dir() else "file"
        rel = item.relative_to(p) if recursive else Path(item.name)
        lines.append(f"  {kind}: {rel}")
    if total > max_items:
        lines.append(f"  ...and {total - max_items} more.")
    return "\n".join(lines)


def read_finder_selection() -> str:
    """Return the POSIX paths currently selected in Finder."""
    if not _is_macos():
        return _macos_only_msg("Finder selection")

    script = '''
    tell application "Finder"
        set selectedItems to selection
        if (count of selectedItems) is 0 then
            return ""
        end if
        set output to ""
        repeat with itemRef in selectedItems
            set output to output & POSIX path of (itemRef as alias) & linefeed
        end repeat
        return output
    end tell
    '''
    try:
        output = _osascript(script, timeout=10.0)
    except subprocess.TimeoutExpired:
        return "Finder didn't respond in time. Click on Finder once and try again."
    except Exception as exc:
        return f"Could not read Finder selection: {exc}"

    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths:
        return "Nothing is selected in Finder."
    if len(paths) == 1:
        return f"Finder selection: {paths[0]}"
    return f"Finder selection ({len(paths)} items):\n" + "\n".join(paths)


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
        # Safari uses `name of t` instead of `title of t`.
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
        return header + "\n" + "\n".join(tabs[:25]) + f"\n...and {len(tabs) - 25} more."
    return header + "\n" + "\n".join(tabs)


# ---------------------------------------------------------------------------
# Window / region / display capture (vision-described)
# ---------------------------------------------------------------------------

def capture_frontmost_window(question: str = "") -> str:
    """Capture the bounds of the frontmost window via screencapture and describe it."""
    if not _is_macos():
        return _macos_only_msg("Window capture")

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
        return f"Could not get frontmost window bounds: {exc}"

    try:
        # `-x` silences the camera-shutter sound; `-R` is non-interactive region.
        subprocess.run(
            ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", WINDOW_PATH],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return f"Window capture failed: {exc}"

    prompt = question.strip() or "Briefly describe what's in this window."
    description = _describe_image(WINDOW_PATH, prompt)
    if description:
        return f"Window captured. {description}"
    return _no_vision_message()


def capture_screen_region(
    x: int, y: int, width: int, height: int, question: str = ""
) -> str:
    """Capture a rectangular region of the screen and describe it."""
    if not _is_macos():
        return _macos_only_msg("Region capture")
    try:
        subprocess.run(
            ["screencapture", "-x", "-R", f"{x},{y},{width},{height}", REGION_PATH],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return f"Region capture failed: {exc}"

    prompt = question.strip() or "Briefly describe what's in this screen region."
    description = _describe_image(REGION_PATH, prompt)
    if description:
        return f"Region captured. {description}"
    return _no_vision_message()


def capture_display(display: int = 1, question: str = "") -> str:
    """Capture an entire display by index (1 = primary) and describe it.

    Note: this is *monitors*, not Spaces. macOS doesn't expose virtual desktops
    to screencapture; only physical displays.
    """
    if not _is_macos():
        return _macos_only_msg("Display capture")
    try:
        subprocess.run(
            ["screencapture", "-x", f"-D{display}", DISPLAY_PATH],
            check=True, capture_output=True, timeout=10.0,
        )
    except Exception as exc:
        return f"Display capture failed: {exc}"

    prompt = question.strip() or f"Briefly describe what's on display {display}."
    description = _describe_image(DISPLAY_PATH, prompt)
    if description:
        return f"Display {display} captured. {description}"
    return _no_vision_message()


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

    apps = sorted({a.strip() for a in output.split(",") if a.strip()}, key=str.lower)
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
        return f"Frontmost app is {app}, not a known terminal. Bring Terminal or iTerm2 to the foreground."

    try:
        text = _osascript(script, timeout=5.0)
    except Exception as exc:
        return f"Could not read {app} output: {exc}"

    if not text.strip():
        return f"{app} has no visible output."

    lines = text.splitlines()
    if len(lines) > max_lines:
        return f"{app} output (last {max_lines} of {len(lines)} lines):\n" + "\n".join(lines[-max_lines:])
    return f"{app} output:\n{text}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """Declarative binding of name+schema+function+spoken filler."""

    schema: FunctionSchema
    fn: Callable[..., Any]
    speak_text: Optional[str] = None


TOOLS: list[Tool] = [
    Tool(
        schema=FunctionSchema(
            name="read_clipboard",
            description=(
                "Read the current system clipboard text. Use when the user asks "
                "about what they copied or what's on their clipboard."
            ),
            properties={},
            required=[],
        ),
        fn=read_clipboard,
    ),
    Tool(
        schema=FunctionSchema(
            name="take_screenshot",
            description=(
                "Capture the user's screen and (if a vision provider is configured) "
                "describe what's on it. Use when the user asks about what's on their screen."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": "What the user wants to know about the screen, used as the vision prompt. Optional.",
                },
            },
            required=[],
        ),
        fn=take_screenshot,
        speak_text="Looking at your screen.",
    ),
    Tool(
        schema=FunctionSchema(
            name="capture_webcam",
            description=(
                "Capture one frame from the user's webcam and (if vision is configured) "
                "describe it. Use when the user asks 'what do you see' or wants you to "
                "look at them or what they're holding."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": "What the user wants you to look for, used as the vision prompt. Optional.",
                },
            },
            required=[],
        ),
        fn=capture_webcam,
        speak_text="Let me see.",
    ),
    Tool(
        schema=FunctionSchema(
            name="web_search",
            description=(
                "Search the web with Google (via Serper) for current information. "
                "Use for facts, news, or anything you don't already know."
            ),
            properties={
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "How many results (1-10). Default 5."},
            },
            required=["query"],
        ),
        fn=web_search,
        speak_text="Searching the web.",
    ),
    Tool(
        schema=FunctionSchema(
            name="read_selected_text",
            description=(
                "Read whatever text the user currently has selected/highlighted in the "
                "frontmost app (macOS only). Triggers Cmd+C and reads the result. "
                "Use when the user says 'look at what I have selected' or 'what does this say'."
            ),
            properties={},
            required=[],
        ),
        fn=read_selected_text,
        speak_text="Reading your selection.",
    ),
    Tool(
        schema=FunctionSchema(
            name="read_focused_input",
            description=(
                "Read the value of the focused text field in the frontmost app (macOS only). "
                "Use when the user says 'what am I typing' or 'look at this input'."
            ),
            properties={},
            required=[],
        ),
        fn=read_focused_input,
    ),
    Tool(
        schema=FunctionSchema(
            name="read_file",
            description=(
                "Read the contents of a file at a given absolute or ~-prefixed path. "
                "Truncates to the first 100KB. Use when the user asks about a specific file."
            ),
            properties={
                "path": {"type": "string", "description": "Absolute or ~-prefixed path to the file."},
                "max_bytes": {"type": "integer", "description": "Truncate after this many bytes. Default 100000."},
            },
            required=["path"],
        ),
        fn=read_file,
        speak_text="Opening that file.",
    ),
    Tool(
        schema=FunctionSchema(
            name="list_folder",
            description=(
                "List the contents of a folder (sorted, folders first). "
                "Set recursive=true to walk subfolders."
            ),
            properties={
                "path": {"type": "string", "description": "Absolute or ~-prefixed folder path."},
                "recursive": {"type": "boolean", "description": "Walk subfolders. Default false."},
                "max_items": {"type": "integer", "description": "Cap items shown. Default 50."},
            },
            required=["path"],
        ),
        fn=list_folder,
        speak_text="Listing that folder.",
    ),
    Tool(
        schema=FunctionSchema(
            name="read_finder_selection",
            description=(
                "Return the file or folder paths currently selected in Finder (macOS only). "
                "Use when the user says 'look at what I picked in Finder'."
            ),
            properties={},
            required=[],
        ),
        fn=read_finder_selection,
        speak_text="Checking Finder.",
    ),
    Tool(
        schema=FunctionSchema(
            name="read_browser_url",
            description=(
                "Return the URL of the current tab of the frontmost browser "
                "(Chrome, Safari, Arc, Brave, Edge, Firefox; macOS only)."
            ),
            properties={},
            required=[],
        ),
        fn=read_browser_url,
        speak_text="Checking your browser.",
    ),
    Tool(
        schema=FunctionSchema(
            name="read_browser_page_text",
            description=(
                "Return the visible text of the current browser page via JavaScript "
                "(macOS only). Requires 'Allow JavaScript from Apple Events' in the browser. "
                "Use when the user asks about the page they're looking at."
            ),
            properties={
                "max_chars": {"type": "integer", "description": "Truncate after this many chars. Default 8000."},
            },
            required=[],
        ),
        fn=read_browser_page_text,
        speak_text="Reading that page.",
    ),
    Tool(
        schema=FunctionSchema(
            name="list_browser_tabs",
            description=(
                "List the title + URL of every open tab in the frontmost browser (macOS only)."
            ),
            properties={},
            required=[],
        ),
        fn=list_browser_tabs,
        speak_text="Checking your tabs.",
    ),
    Tool(
        schema=FunctionSchema(
            name="capture_frontmost_window",
            description=(
                "Capture the frontmost application window (macOS only) and describe it via "
                "the configured vision provider. Use when the user says 'look at this window'."
            ),
            properties={
                "question": {"type": "string", "description": "What to look for. Optional."},
            },
            required=[],
        ),
        fn=capture_frontmost_window,
        speak_text="Looking at your window.",
    ),
    Tool(
        schema=FunctionSchema(
            name="capture_screen_region",
            description=(
                "Capture a rectangular region of the screen (macOS only) and describe it. "
                "Coordinates are in screen points: (0,0) is top-left."
            ),
            properties={
                "x": {"type": "integer", "description": "Left edge in screen points."},
                "y": {"type": "integer", "description": "Top edge in screen points."},
                "width": {"type": "integer", "description": "Region width in points."},
                "height": {"type": "integer", "description": "Region height in points."},
                "question": {"type": "string", "description": "What to look for. Optional."},
            },
            required=["x", "y", "width", "height"],
        ),
        fn=capture_screen_region,
        speak_text="Looking at that area.",
    ),
    Tool(
        schema=FunctionSchema(
            name="capture_display",
            description=(
                "Capture a specific display/monitor by index (1 = primary; macOS only). "
                "Note: monitors, not virtual desktops/Spaces — those aren't capturable."
            ),
            properties={
                "display": {"type": "integer", "description": "Display index (1 = primary)."},
                "question": {"type": "string", "description": "What to look for. Optional."},
            },
            required=[],
        ),
        fn=capture_display,
        speak_text="Looking at that display.",
    ),
    Tool(
        schema=FunctionSchema(
            name="get_frontmost_app",
            description=(
                "Return the name of the application currently in the foreground (macOS only). "
                "Use when the user asks 'what app am I in?'."
            ),
            properties={},
            required=[],
        ),
        fn=get_frontmost_app,
    ),
    Tool(
        schema=FunctionSchema(
            name="list_running_apps",
            description=(
                "List the foreground (visible) running applications (macOS only)."
            ),
            properties={},
            required=[],
        ),
        fn=list_running_apps,
    ),
    Tool(
        schema=FunctionSchema(
            name="read_terminal_output",
            description=(
                "Return the visible output of the frontmost Terminal.app or iTerm2 window "
                "(macOS only). Use when the user says 'look at my terminal'."
            ),
            properties={
                "max_lines": {"type": "integer", "description": "Cap to last N lines. Default 80."},
            },
            required=[],
        ),
        fn=read_terminal_output,
        speak_text="Checking your terminal.",
    ),
    Tool(
        schema=FunctionSchema(
            name="get_current_weather",
            description="Get the current weather (demo — returns dummy data).",
            properties={
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the user's location.",
                },
            },
            required=["location", "format"],
        ),
        fn=get_current_weather,
        speak_text="Let me check on that.",
    ),
]


def _make_handler(tool: Tool, session_log=None):
    """Wrap a sync tool fn as a Pipecat function-call handler with an optional spoken filler.

    The tool's sync `fn` is run inside `asyncio.to_thread`, which is critical:
    every tool here does blocking I/O (screen capture, webcam read, network
    calls). Running them on the asyncio event loop directly starves the
    Soniox STT/TTS WebSockets of heartbeats and the connections drop after
    ~30s with `Error: 408 _receive_messages - Request timeout`.

    Result-shaping rule: dict results pass through unchanged (the LLM sees them as
    structured data); any other result is wrapped in `{"result": value}`.

    If `session_log` is provided, each call writes `tool-called` (with args) and
    either `tool-result` (with the result string/dict) or `tool-error` (with the
    exception text) to the session log.
    """
    import asyncio

    async def handler(params):
        from pipecat.frames.frames import TTSSpeakFrame

        args = params.arguments or {}
        if session_log is not None:
            session_log.event("tool-called", name=tool.schema.name, args=args)

        if tool.speak_text:
            await params.llm.push_frame(TTSSpeakFrame(tool.speak_text))
        try:
            try:
                result = await asyncio.to_thread(tool.fn, **args)
            except TypeError:
                # Tolerate the LLM passing extra/wrong-named args — call with no args.
                result = await asyncio.to_thread(tool.fn)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if session_log is not None:
                session_log.event("tool-error", name=tool.schema.name, error=err)
            await params.result_callback({"result": f"Tool {tool.schema.name} failed: {err}"})
            return

        if session_log is not None:
            session_log.event("tool-result", name=tool.schema.name, result=result)
        if isinstance(result, dict):
            await params.result_callback(result)
        else:
            await params.result_callback({"result": result})

    return handler


def register_all(llm, session_log=None) -> list[FunctionSchema]:
    """Register every tool on the given LLM and return the FunctionSchemas to put in a ToolsSchema.

    Pass `session_log` (a `session_log.SessionLog`) to record tool calls, results,
    and errors as kebab-case events alongside user/bot speech.
    """
    for tool in TOOLS:
        llm.register_function(tool.schema.name, _make_handler(tool, session_log=session_log))
    return [tool.schema for tool in TOOLS]
