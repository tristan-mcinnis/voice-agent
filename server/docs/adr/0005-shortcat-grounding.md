# ADR-0005: ShortCat as a label-based grounding alternative

**Status:** accepted

**Date:** 2026-05-10

**Context:** ADR-0004 shipped two grounding paths for computer-use:

1. `list_ui_elements` + `click_element` — walks the AX tree of the frontmost
   app and clicks by index. Reliable when AX exposes the target.
2. `find_on_screen` — asks the vision chain for `(x, y)` pixel coordinates.
   Honest stopgap, but the descriptive VLMs we use (Qwen3-VL-2B, Kimi) are
   not grounding-trained and miss by tens of pixels. UI-TARS-7B-DPO would
   close the gap but is a 7B model — too heavy for an idle voice loop.

A third option fits in between: ShortCat (https://shortcat.app/), a macOS
command palette that already indexes the system AX tree and dispatches
clicks by label. Users drive it with: hotkey → type label → Enter.

**Hypothesis:** if we drive ShortCat by synthesizing the same keystrokes a
human would, we get label-based grounding for free, and the small VLM
becomes a *verifier* (did the right thing happen?) rather than a *grounder*
(where do I click?). Verification is far easier than grounding for small
models.

**Test:** `server/tests/test_shortcat.py` runs five scripted scenarios
(TextEdit New Document, Finder Downloads, Safari Address bar, TextEdit
Format menu, System Settings Wi-Fi). Each scenario activates the target
app, screenshots, fires the hotkey + label + Enter, screenshots again, and
asks the VLM a yes/no question about the after-state.

Result over three runs (2026-05-10):
- 3/5 PASS — TextEdit New Document, Finder Downloads, Safari Address bar.
- 2/5 FAIL with the same root cause: the harness's `activate_app("X")` did
  not reliably steal focus from a residual Safari frontmost between
  scenarios, even with a 5 s poll + alias-table + repeated AppleScript
  activate. Keystrokes synthesized into Safari's address bar instead of
  the target app. ShortCat's hotkey was never received.
- ShortCat dispatch p50: 1607 ms (includes app launch + activate); the
  type-and-Enter portion alone is ~300 ms.
- VLM verifier (MLX `Qwen3-VL-2B-Instruct-4bit`) parsed yes/no cleanly on
  every scenario.

The 2 failures are a test-runner artifact, not a ShortCat limitation: in
the production voice loop the user is *already* using the target app, so
the cross-app focus race does not apply.

**Decision:** Add a single tool, `shortcat_click(label)`, alongside the
existing AX path and grounding fallback. Order of preference for the
agent:

1. **Native AX:** `list_ui_elements` + `click_element` — best when AX
   labels are stable and unique. Index-based, no fuzzy matching.
2. **ShortCat:** `shortcat_click("Reload")` — best when the target's label
   is obvious but enumeration is noisy (e.g. menu items behind submenus,
   toolbar buttons in apps with shallow AX).
3. **Vision grounding:** `find_on_screen` — last resort for
   Electron/canvas/games, until UI-TARS or similar is wired into the
   vision chain.

**Implementation:**

- `tools/computer_use.py::ShortcatClickTool` — synthesizes `hotkey →
  typewrite(label) → Enter` via pyautogui. Reads its hotkey/timing from
  `config.yaml::shortcat`. Lazy-imports `config` so a missing config still
  leaves the tool registered with a clear failure message.
- `config.py::ShortcatConfig` — `enabled`, `hotkey`, `palette_delay_ms`,
  `settle_ms`. Disabled by default; flipping `enabled: true` in
  `config.yaml` is the only step needed once Shortcat.app is installed.
- The existing sensitive-app safety gate (`_refuse_if_sensitive`) wraps
  this tool too — typing into 1Password's palette match is just as
  destructive as typing into 1Password directly.
- `tools/__init__.py` re-exports `shortcat_click` for compat.

**Why driving ShortCat with synthetic keystrokes is OK:**

- ShortCat has no public CLI / URL scheme / AppleScript / scripting
  bridge. We checked the docs and the bundle. Synthetic keystrokes are
  the supported integration surface; that's how every human user drives
  it.
- Required for production: Shortcat.app installed, granted Accessibility
  permission, activation hotkey set to `⌘⌥Space` (the documented default
  `⌘⇧Space` collides with Spotlight, so the test config and the example
  config use `⌘⌥Space`).

**Why not just extend ShortCat into our process (e.g. read its config,
script it directly):**

- ShortCat has no API. Reverse-engineering its private mach interface or
  reaching into its plist/preferences would be brittle and would break
  on every ShortCat update.
- The hotkey-driven path is the same path real users use. If it breaks,
  the user can verify by trying it themselves — clean failure mode.

**Consequences:**

- The agent gains a third grounding path. Decision tree it should follow:
  *known native app + label visible in `list_ui_elements`* → AX click;
  *label obvious but element buried/Electron-ish* → `shortcat_click`;
  *no labelled target, only pixels* → `find_on_screen`.
- Hard requirement: the **target app must already be frontmost** when
  `shortcat_click` is called. ShortCat indexes the focused app, not the
  whole desktop. If the agent suspects the wrong app is frontmost, it
  should call `get_frontmost_app` first.
- ShortCat is a third-party dep the user must install and configure once.
  We do not bundle, install, or configure it for them. The tool returns
  a clean error if it's missing.
- Dispatch cost: ~300 ms type-and-Enter, plus ShortCat's match latency
  (~50 ms typically). Comparable to the AX path; far faster than a
  vision-grounded click that round-trips a screenshot through MLX.
- Open question: ambiguity. If the typed query has multiple top matches,
  ShortCat displays them with 1-3 char hint labels. The current tool
  always presses Enter, picking the top match. Future iteration: surface
  the open palette to the VLM and let it pick a hint label when the top
  match is wrong. The test harness already exercises this loosely; the
  tool can be extended without an API change (add a `disambiguate=true`
  parameter that screenshots the open palette and routes through
  `find_on_screen`).
- Tools added to the registry (1 new): `shortcat_click`.
