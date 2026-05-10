"""ShortCat-as-grounder feasibility test.

Hypothesis: instead of a small VLM guessing pixel coordinates, drive ShortCat's
command palette to navigate by accessibility label (type-and-Enter), and use
the VLM only as a verifier of the resulting screen state.

Run from repo root:
    cd server && source venv/bin/activate
    python tests/test_shortcat.py

Env knobs:
    SHORTCAT_HOTKEY   pyautogui hotkey, default "cmd+alt+space"
    SHORTCAT_SETTLE   ms to wait after Enter before verify, default 600
    SHORTCAT_PALETTE_DELAY  ms to wait after hotkey before typing, default 250

Prereqs (manual):
  - Shortcat.app installed and granted Accessibility permission
  - Activation hotkey set to ⌘⌥Space (or override via env)
  - Vision provider configured (config.yaml `vision:`)
  - Terminal/Python granted Accessibility for keystroke synthesis
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the server package importable when run from server/ or repo root.
HERE = Path(__file__).resolve().parent
SERVER = HERE.parent
sys.path.insert(0, str(SERVER))

from tools.vision import describe_image, set_vision_config  # noqa: E402
from config import load_config  # noqa: E402


HOTKEY = os.getenv("SHORTCAT_HOTKEY", "cmd+alt+space")
PALETTE_DELAY = float(os.getenv("SHORTCAT_PALETTE_DELAY", "250")) / 1000.0
SETTLE_MS = float(os.getenv("SHORTCAT_SETTLE", "600")) / 1000.0
APP_LAUNCH_DELAY = 1.2  # seconds after `open -a` before driving


LOG_DIR = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RUN_DIR = LOG_DIR / f"shortcat-test-{TS}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"shortcat-test-{TS}.jsonl"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    app: str                  # `open -a` target
    setup_keys: list[str] = field(default_factory=list)  # extra prep keystrokes
    query: str = ""           # what to type in the palette
    confirm_key: str = "enter"  # press to commit
    expected_prompt: str = ""  # VLM yes/no question on `after.png`


SCENARIOS: list[Scenario] = [
    Scenario(
        name="textedit_new_doc",
        app="TextEdit",
        setup_keys=["escape"],  # dismiss any open dialog
        query="New Document",
        expected_prompt=(
            "Is there a NEW empty TextEdit document window visible "
            "(blank white page, no text)? Answer strictly 'yes' or 'no' on "
            "the first line, then a brief reason."
        ),
    ),
    Scenario(
        name="finder_downloads",
        app="Finder",
        query="Downloads",
        expected_prompt=(
            "Is the Finder window now showing the Downloads folder "
            "(title bar or sidebar selection says 'Downloads')? "
            "Answer strictly 'yes' or 'no' on the first line, then a brief reason."
        ),
    ),
    Scenario(
        name="safari_address_bar",
        app="Safari",
        query="Address",
        expected_prompt=(
            "Is Safari's address/search bar focused (cursor in URL field, "
            "field highlighted)? Answer strictly 'yes' or 'no' on the first "
            "line, then a brief reason."
        ),
    ),
    Scenario(
        name="textedit_format_menu",
        app="TextEdit",
        query="Format",
        expected_prompt=(
            "Is the TextEdit Format menu currently OPEN, showing menu items "
            "like Font, Text, etc.? Answer strictly 'yes' or 'no' on the "
            "first line, then a brief reason."
        ),
    ),
    Scenario(
        name="system_settings_wifi",
        app="System Settings",
        query="Wi-Fi",
        expected_prompt=(
            "Is the System Settings Wi-Fi pane visible (shows networks, "
            "Wi-Fi toggle, or 'Wi-Fi' as the pane title)? Answer strictly "
            "'yes' or 'no' on the first line, then a brief reason."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Driving primitives
# ---------------------------------------------------------------------------

def _import_pyautogui():
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.04
    return pyautogui


def screenshot(path: Path) -> None:
    """Full-screen JPEG via macOS screencapture (no focus side effects)."""
    subprocess.run(
        ["screencapture", "-x", "-t", "jpg", str(path)],
        check=True, capture_output=True, timeout=8.0,
    )


# Frontmost-process name doesn't always match the user-facing app name —
# e.g. opening "System Settings" can leave frontmost as "Settings" on newer
# macOS. Match against any alias for the test target.
_APP_ALIASES: dict[str, tuple[str, ...]] = {
    "System Settings": ("system settings", "settings", "system preferences"),
    "TextEdit": ("textedit",),
    "Finder": ("finder",),
    "Safari": ("safari",),
}


def _matches_target(target: str, observed: str) -> bool:
    aliases = _APP_ALIASES.get(target, (target.lower(),))
    obs = observed.lower()
    return any(a in obs for a in aliases)


def activate_app(app: str) -> None:
    """Launch + raise + wait until the target app is genuinely frontmost.

    Two failure modes the simple `open -a` form misses:
      - First-launch cold start (System Settings) takes >2 s to raise its window.
      - A residual app (e.g. Safari from the previous scenario) keeps focus
        while the new app is still launching; keystrokes then synthesize into
        the wrong window.
    Belt-and-braces: `open -a`, AppleScript activate, then poll frontmost for
    up to 5 s, retrying the activate every second.
    """
    subprocess.run(
        ["open", "-a", app], check=True, capture_output=True, timeout=10.0,
    )
    deadline = time.time() + 5.0
    last_activate = 0.0
    while time.time() < deadline:
        now = time.time()
        if now - last_activate > 0.9:
            subprocess.run(
                ["osascript", "-e", f'tell application "{app}" to activate'],
                check=False, capture_output=True, timeout=3.0,
            )
            last_activate = now
        if _matches_target(app, frontmost()):
            break
        time.sleep(0.2)
    time.sleep(0.4)


def frontmost() -> str:
    try:
        out = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to '
             'return name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=4.0, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "(unknown)"


def press(combo: str, pg) -> None:
    parts = [k.strip().lower() for k in combo.split("+") if k.strip()]
    if len(parts) == 1:
        pg.press(parts[0])
    else:
        pg.hotkey(*parts)


def shortcat_dispatch(query: str, confirm_key: str, pg) -> None:
    press(HOTKEY, pg)
    time.sleep(PALETTE_DELAY)
    pg.typewrite(query, interval=0.025)
    time.sleep(0.15)
    press(confirm_key, pg)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def parse_yes_no(text: str) -> Optional[bool]:
    if not text:
        return None
    first = text.strip().splitlines()[0].lower()
    if first.startswith("yes"):
        return True
    if first.startswith("no"):
        return False
    if "yes" in first and "no" not in first:
        return True
    if "no" in first and "yes" not in first:
        return False
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(s: Scenario, pg) -> dict:
    rec: dict = {"name": s.name, "app": s.app, "query": s.query}
    t0 = time.perf_counter()

    try:
        activate_app(s.app)
    except Exception as exc:
        rec["error"] = f"activate failed: {exc}"
        return rec
    rec["frontmost_before"] = frontmost()

    for k in s.setup_keys:
        try:
            press(k, pg)
            time.sleep(0.15)
        except Exception:
            pass

    before = RUN_DIR / f"{s.name}-before.jpg"
    after = RUN_DIR / f"{s.name}-after.jpg"
    try:
        screenshot(before)
    except Exception as exc:
        rec["error"] = f"before-shot failed: {exc}"
        return rec

    t_dispatch = time.perf_counter()
    try:
        shortcat_dispatch(s.query, s.confirm_key, pg)
    except Exception as exc:
        rec["error"] = f"dispatch failed: {exc}"
        return rec
    time.sleep(SETTLE_MS)
    rec["dispatch_ms"] = round((time.perf_counter() - t_dispatch) * 1000)
    rec["frontmost_after"] = frontmost()

    try:
        screenshot(after)
    except Exception as exc:
        rec["error"] = f"after-shot failed: {exc}"
        return rec

    rec["before_path"] = str(before)
    rec["after_path"] = str(after)

    t_vlm = time.perf_counter()
    try:
        desc = describe_image(str(after), s.expected_prompt) or ""
    except Exception as exc:
        desc = f"(vlm error: {exc})"
    rec["vlm_ms"] = round((time.perf_counter() - t_vlm) * 1000)
    rec["vlm_text"] = desc.strip()
    rec["vlm_verdict"] = parse_yes_no(desc)
    rec["total_ms"] = round((time.perf_counter() - t0) * 1000)

    # Always Esc to dismiss any stuck palette / open menu before the next run.
    try:
        press("escape", pg)
    except Exception:
        pass
    time.sleep(0.25)
    return rec


def main() -> int:
    print(f"=== ShortCat feasibility test ===", flush=True)
    print(f"Hotkey: {HOTKEY}", flush=True)
    print(f"Palette delay: {PALETTE_DELAY*1000:.0f} ms", flush=True)
    print(f"Settle: {SETTLE_MS*1000:.0f} ms", flush=True)
    print(f"Run dir: {RUN_DIR}", flush=True)
    print(f"Log:     {LOG_PATH}", flush=True)
    print()

    # Initialize the vision provider chain — describe_image is a no-op without this.
    try:
        cfg = load_config()
        set_vision_config(cfg.vision)
        print(f"Vision providers: {[getattr(p, 'kind', '?') for p in cfg.vision]}", flush=True)
    except Exception as exc:
        print(f"WARN: vision config load failed: {exc}", flush=True)

    # Make sure ShortCat is running before we start firing its hotkey.
    subprocess.run(["open", "-a", "Shortcat"], check=False,
                   capture_output=True, timeout=10.0)
    time.sleep(1.0)

    pg = _import_pyautogui()
    results: list[dict] = []
    with LOG_PATH.open("w") as f:
        for s in SCENARIOS:
            print(f"--- {s.name} (app={s.app}, query={s.query!r}) ---", flush=True)
            rec = run_scenario(s, pg)
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            verdict = rec.get("vlm_verdict")
            mark = "✓" if verdict is True else ("✗" if verdict is False else "?")
            err = rec.get("error", "")
            line = f"  {mark} verdict={verdict}"
            if "dispatch_ms" in rec:
                line += f"  dispatch={rec['dispatch_ms']}ms"
            if "total_ms" in rec:
                line += f"  total={rec['total_ms']}ms"
            if err:
                line += f"  ERROR: {err}"
            print(line, flush=True)
            if rec.get("vlm_text"):
                print(f"     vlm: {rec['vlm_text'][:200]}", flush=True)
            time.sleep(0.4)

    # Summary
    n = len(results)
    yes = sum(1 for r in results if r.get("vlm_verdict") is True)
    no = sum(1 for r in results if r.get("vlm_verdict") is False)
    unk = n - yes - no
    print()
    print(f"=== Summary ===", flush=True)
    print(f"  yes:     {yes}/{n}", flush=True)
    print(f"  no:      {no}/{n}", flush=True)
    print(f"  unknown: {unk}/{n}", flush=True)
    dispatches = [r["dispatch_ms"] for r in results if "dispatch_ms" in r]
    if dispatches:
        dispatches.sort()
        p50 = dispatches[len(dispatches)//2]
        print(f"  dispatch p50: {p50} ms  (n={len(dispatches)})", flush=True)
    print(f"  log: {LOG_PATH}", flush=True)
    return 0 if yes >= max(1, n // 2) else 2


if __name__ == "__main__":
    sys.exit(main())
