"""Standalone smoke tests for tools.py.

Calls each tool function directly (no Pipecat / no LLM) and prints a
per-tool result with elapsed time. macOS-only tools short-circuit on
other platforms.

Usage:
    source venv/bin/activate
    python test_tools.py
    python test_tools.py --webcam        # also exercise webcam (slow)
    python test_tools.py --vision        # also exercise vision-described capture tools
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add server/ to sys.path so `import tools` works from the tests/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, float, str]] = []


def run(name: str, fn, *args, **kwargs) -> bool:
    """Call `fn(*args, **kwargs)` and record outcome. Return True on PASS."""
    print(f"\n--- {name} ".ljust(72, "-"))
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  RAISED ({elapsed:.2f}s): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results.append((name, FAIL, elapsed, str(exc)))
        return False
    elapsed = time.monotonic() - t0
    if isinstance(result, dict):
        preview = repr(result)
    else:
        text = str(result)
        preview = text if len(text) <= 600 else text[:600] + f"... [+{len(text) - 600} chars]"
    print(f"  ({elapsed:.2f}s) {preview}")
    results.append((name, PASS, elapsed, preview[:120]))
    return True


def skip(name: str, reason: str) -> None:
    print(f"\n--- {name} ".ljust(72, "-"))
    print(f"  SKIP: {reason}")
    results.append((name, SKIP, 0.0, reason))


def main() -> int:
    print("=" * 72)
    print("VOICE-BOT TOOL SMOKE TESTS")
    print("=" * 72)
    print(f"Platform: {sys.platform}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"CWD:      {Path.cwd()}")

    want_webcam = "--webcam" in sys.argv
    want_vision = "--vision" in sys.argv

    # --- Text ---------------------------------------------------------------
    run("read_clipboard", tools.read_clipboard)
    run("read_selected_text", tools.read_selected_text)
    run("read_focused_input", tools.read_focused_input)

    # --- File system --------------------------------------------------------
    here = Path(__file__).resolve().parent
    run("read_file (this script)", tools.read_file, str(__file__))
    run("read_file (offset/limit)", tools.read_file, str(__file__), 1, 5)
    run("read_file (nonexistent)", tools.read_file, "/no/such/path/__voice_bot_missing__")
    run("read_file (directory)", tools.read_file, str(here))
    run("list_folder (server dir)", tools.list_folder, str(here))
    if (here / "assets").exists():
        run("list_folder (recursive assets)", tools.list_folder, str(here / "assets"), True)
    run("list_folder (nonexistent)", tools.list_folder, "/no/such/folder/__missing__")
    run("read_finder_selection", tools.read_finder_selection)

    # --- File mutation (sandbox in /tmp) -------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory(prefix="voice-bot-test-") as sandbox:
        sb = Path(sandbox)
        f1 = sb / "sub" / "hello.py"
        run("write_file (creates parents)", tools.write_file, str(f1), "def hi():\n    return 1\n")
        run("append_to_file", tools.append_to_file, str(f1), "# trailing\n")
        run("file_info (existing)", tools.file_info, str(f1))
        run("file_info (missing)", tools.file_info, "/no/such/file")
        run("patch (fuzzy + diff + syntax)", tools.patch_file, str(f1), "return 1", "return 42")
        run("patch (non-unique fail)", tools.patch_file, str(f1), "def", "DEF")
        run("make_directory", tools.make_directory, str(sb / "newdir/a/b"))
        run("copy_path (file)", tools.copy_path, str(f1), str(sb / "copy.py"))
        run("copy_path (dir recursive)", tools.copy_path, str(sb / "newdir"), str(sb / "newdir-copy"))
        run("move_path", tools.move_path, str(f1), str(sb / "moved.py"))
        run("delete_path (file)", tools.delete_path, str(sb / "moved.py"))
        run("delete_path (dir refuses without recursive)", tools.delete_path, str(sb / "newdir"))
        run("delete_path (dir recursive)", tools.delete_path, str(sb / "newdir"), True)
        run("delete_path (refuses $HOME)", tools.delete_path, str(Path.home()))
        run("search_files (content)", tools.search_files, "content", "return 42", str(sb))
        run("search_files (filename)", tools.search_files, "filename", "*.py", str(sb))

        # Hard cap on list_folder regardless of LLM-supplied max_items
        big = sb / "many"; big.mkdir()
        for i in range(60):
            (big / f"f{i:02d}.txt").touch()
        run("list_folder (hard cap 30 vs requested 50)", tools.list_folder, str(big), False, 50)

    # --- run_terminal_command -----------------------------------------------
    run("run_terminal_command (basic)", tools.run_terminal_command, "echo hello && pwd", 5, "/tmp")
    run("run_terminal_command (timeout)", tools.run_terminal_command, "sleep 5", 1)
    run("run_terminal_command (stdout cap)",
        tools.run_terminal_command,
        "for i in $(seq 1 500); do echo line-$i-padding; done", 10)
    run("run_terminal_command (stderr captured)",
        tools.run_terminal_command, "ls /no/such/path 2>&1 1>/dev/null; echo done >&2", 5)

    # --- Browser ------------------------------------------------------------
    run("read_browser_url", tools.read_browser_url)
    run("read_browser_page_text", tools.read_browser_page_text)
    run("list_browser_tabs", tools.list_browser_tabs)

    # --- System -------------------------------------------------------------
    run("get_frontmost_app", tools.get_frontmost_app)
    run("list_running_apps", tools.list_running_apps)

    # --- Terminal -----------------------------------------------------------
    run("read_terminal_output", tools.read_terminal_output)

    # --- Capture (vision) ---------------------------------------------------
    if want_vision:
        run("take_screenshot", tools.take_screenshot, "")
        run("capture_frontmost_window", tools.capture_frontmost_window, "What window is this?")
        run("capture_screen_region (top-left 400x400)", tools.capture_screen_region, 0, 0, 400, 400)
        run("capture_display (1)", tools.capture_display, 1)
    else:
        skip("take_screenshot", "pass --vision to run (calls vision provider)")
        skip("capture_frontmost_window", "pass --vision to run (calls vision provider)")
        skip("capture_screen_region", "pass --vision to run (calls vision provider)")
        skip("capture_display", "pass --vision to run (calls vision provider)")

    if want_webcam:
        run("capture_webcam", tools.capture_webcam, "")
    else:
        skip("capture_webcam", "pass --webcam to run")

    # --- Web search ---------------------------------------------------------
    if os.getenv("SERPER_API_KEY"):
        run("web_search", tools.web_search, "Pipecat AI voice bot", 3)
    else:
        skip("web_search", "SERPER_API_KEY not set")

    # --- Demo ---------------------------------------------------------------
    run("get_current_weather", tools.get_current_weather, "San Francisco, CA", "fahrenheit")

    # --- Registry sanity ----------------------------------------------------
    print("\n" + "=" * 72)
    registry = tools.REGISTRY
    all_tools = registry.all()
    print(f"REGISTRY has {len(all_tools)} entries:")
    for t in all_tools:
        print(f"  - [{t.category}] {t.name}")

    print("\nSchema round-trip (from BaseTool.to_schema):")
    for schema in registry.schemas():
        req = ",".join(schema.required) if schema.required else "-"
        props = ",".join(schema.properties.keys()) if schema.properties else "-"
        print(f"  - {schema.name}  required=[{req}]  props=[{props}]")

    print("\nCapability summary (injected into {tool_capabilities}):")
    print(registry.capabilities_summary())

    # Exercise BaseTool.execute via the registry (proves the class layer works,
    # not just the underlying module helpers we already called above).
    print("\nRegistry execute() smoke calls:")
    smoke = [
        ("read_clipboard", {}),
        ("read_file", {"path": __file__, "offset": 1, "limit": 5}),
        ("get_current_weather", {"location": "San Francisco, CA", "format": "fahrenheit"}),
    ]
    for name, kwargs in smoke:
        try:
            out = registry.get(name).execute(**kwargs)
            preview = repr(out)[:120]
            print(f"  PASS  {name}({kwargs}) -> {preview}")
        except Exception as exc:
            print(f"  FAIL  {name}({kwargs}) -> {type(exc).__name__}: {exc}")
            results.append((f"registry.{name}", FAIL, 0.0, str(exc)))

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    p = sum(1 for _, status, *_ in results if status == PASS)
    f = sum(1 for _, status, *_ in results if status == FAIL)
    s = sum(1 for _, status, *_ in results if status == SKIP)
    print(f"  {p} pass, {f} fail, {s} skip   (out of {len(results)})")
    if f:
        print("\nFailures:")
        for name, status, _, msg in results:
            if status == FAIL:
                print(f"  - {name}: {msg}")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
