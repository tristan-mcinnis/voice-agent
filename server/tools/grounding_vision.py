"""Grounding-vision tool — find UI element coordinates via the vision chain.

This is the bridge that lets the agent click in apps with no Accessibility
tree (Spotify, games, web canvas). Workflow:

    1. ``list_ui_elements`` returns nothing useful (Electron/canvas UI).
    2. Agent calls ``find_on_screen(instruction="the green play button")``.
    3. Tool screenshots, asks the vision chain for {x, y} pixels, returns coords.
    4. Agent passes coords to ``click_at``.

**Quality caveat.** This is a stopgap that reuses the existing descriptive
vision chain (Qwen3-VL-2B / Kimi) with a coordinate-extraction prompt. Those
models were not grounding-trained — accuracy is rough (often ±50 px). For
production-grade accuracy, configure a UI-grounding model (UI-TARS-7B,
OS-Atlas-Base-7B, ShowUI-2B) as a vision provider. See ADR-0004.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Optional

from tools._macos import is_macos, macos_only
from tools.registry import REGISTRY, BaseTool
from tools.capture import capture_path  # type: ignore
from tools import vision


def _parse_coords(text: str) -> Optional[dict]:
    """Extract {"x": int, "y": int} from a model response.

    Tolerates surrounding prose, code fences, and various coord formats.
    Returns None on failure or when the model signals "not found" via
    negative coords.
    """
    if not text:
        return None
    s = text.strip().strip("`").strip()
    # Strip ```json ... ``` fences
    if s.startswith("json"):
        s = s[4:].strip()

    # 1) Try direct JSON parse.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
            x, y = int(parsed["x"]), int(parsed["y"])
            return {"x": x, "y": y} if x >= 0 and y >= 0 else None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2) Find {"x": N, "y": N} substring (model wrapped JSON in prose).
    m = re.search(r'"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)', text)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return {"x": x, "y": y} if x >= 0 and y >= 0 else None

    # 3) Click-tag form: <click>x, y</click> or <box>x1,y1,x2,y2</box>.
    m = re.search(r"<click>\s*(\d+)\s*,\s*(\d+)\s*</click>", text)
    if m:
        return {"x": int(m.group(1)), "y": int(m.group(2))}
    m = re.search(r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>", text)
    if m:
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        return {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2}

    # 4) Last resort: first "N, N" pair anywhere in the text.
    m = re.search(r"\b(\d{2,4})\s*[,\s]+\s*(\d{2,4})\b", text)
    if m:
        return {"x": int(m.group(1)), "y": int(m.group(2))}

    return None


_GROUNDING_PROMPT_TEMPLATE = (
    "You are a UI grounding assistant. The image is the user's full screen.\n"
    "Find this element: {instruction}\n\n"
    "Respond with ONLY raw JSON, no prose, no code fences:\n"
    '{{"x": <pixel_x_from_left>, "y": <pixel_y_from_top>}}\n\n'
    "If you cannot find the element with confidence, respond exactly:\n"
    '{{"x": -1, "y": -1}}'
)


@REGISTRY.register
class FindOnScreenTool(BaseTool):
    name = "find_on_screen"
    category = "vision"
    speak_text = "Looking for that."
    description = (
        "Locate a UI element on screen by description, returning pixel "
        "coordinates {x, y} you can pass to click_at. Use AFTER "
        "list_ui_elements returns nothing useful (Electron/canvas/game UIs "
        "like Spotify). Accuracy depends on the configured vision model — "
        "rough with descriptive models, precise with a grounding model "
        "(UI-TARS / OS-Atlas). See ADR-0004."
    )
    parameters = {
        "instruction": {
            "type": "string",
            "description": (
                "Natural-language description of the element. Be specific: "
                "'the green play button at the bottom', 'the search box at "
                "the top of the sidebar'. Avoid 'the button' alone."
            ),
        },
    }
    required = ["instruction"]
    guidance = None  # the description carries enough context

    def execute(self, instruction: str) -> dict:
        if not is_macos():
            return {"result": macos_only("find_on_screen")}
        if not instruction or not instruction.strip():
            return {"result": "find_on_screen needs a non-empty instruction."}

        path = capture_path("find", "jpg")
        try:
            subprocess.run(
                ["screencapture", "-x", "-t", "jpg", path],
                check=True, capture_output=True, timeout=8.0,
            )
        except Exception as exc:
            return {"result": f"Screenshot failed: {type(exc).__name__}: {exc}"}

        prompt = _GROUNDING_PROMPT_TEMPLATE.format(instruction=instruction.strip())
        text = vision.describe_image(path, prompt)
        if text is None:
            return {"result": vision.no_vision_message(), "image_path": path}

        coords = _parse_coords(text)
        if coords is None:
            return {
                "result": (
                    f"Vision returned no usable coordinates for {instruction!r}. "
                    f"Raw output: {text[:200]}"
                ),
                "image_path": path,
            }

        return {
            "x": coords["x"],
            "y": coords["y"],
            "instruction": instruction,
            "image_path": path,
            "result": f"Found {instruction!r} at ({coords['x']}, {coords['y']}).",
        }
