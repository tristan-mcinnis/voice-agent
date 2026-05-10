# ADR-0004: Computer-use tools — AX-primary, vision fallback, gated

**Status:** accepted

**Date:** 2026-05-10

**Context:** The voice agent could *describe* the screen (vision chain) and
*read* its state (clipboard, selected text, browser URL/page text), but it
could not *act* on the GUI. Sessions like 2026-05-10 18:34 show the failure
mode: when asked to play an Evening Jazz playlist on Spotify, the agent
hallucinated a Spotify playlist URI, opened the wrong playlist, then spiraled
through web searches and blind AppleScript keystrokes — never closing the
loop because it had no way to *see-then-click*.

A "computer use" capability needs three things working together:

1. **A way to discover targets** — what's on screen, with labels.
2. **A way to act** — click, type, press keys, scroll.
3. **A way to stay safe** — not click in 1Password, not type into a banking
   tab without the user's consent.

The Hermes v0.13 reference design uses a single `computer_use` dispatcher
backed by screenshots and a Vision-Action loop, normalising coordinates to
0–1000. That design assumes a grounding-trained VL model is available and
treats every app as opaque pixels. On macOS, native apps publish a structured
Accessibility (AX) tree — Shortcat-style enumeration is more reliable for
the 80% case. Falling back to vision-grounded coordinates is the right
escape hatch for the 20% (Electron, canvas, games).

**Decision:** Ship a hybrid layer in `tools/computer_use.py` plus
`tools/grounding_vision.py`:

1. **AX path (primary).** `list_ui_elements` walks the macOS Accessibility
   tree of the frontmost app via PyObjC (`ApplicationServices.AXUIElementRef`),
   bounded by `max_elements=600` and `max_depth=25`. Returns an indexed list
   with role, label, and screen coordinates. ~0.2 s for VS Code's 119
   elements; AppleScript `entire contents` was tried first and timed out at
   15 s on the same target.

2. **Click by index (primary action).** `click_element(index=N)` resolves
   `N` against a module-level cache of the most recent listing, then clicks
   the element's center via `pyautogui`.

3. **Coordinate path (fallback).** `click_at(x, y)`, `type_text`,
   `press_key`, `scroll`, `mouse_move` for cases where AX exposes nothing
   useful (Spotify shows only 3 elements — its three traffic-light buttons).

4. **Grounding bridge.** `find_on_screen(instruction)` screenshots and asks
   the existing vision chain for `{x, y}` pixel coordinates. **Stopgap
   quality** — descriptive models (Qwen3-VL-2B, Kimi) are not grounding-
   trained and miss by tens of pixels. Production-grade accuracy requires
   adding a UI-grounding model as a vision provider:
   - **UI-TARS-7B-DPO** (`mlx-community/UI-TARS-7B-DPO-mlx`) — outputs
     `<click>x,y</click>` tokens. Recommended.
   - **OS-Atlas-Base-7B** — outputs `<box>x1,y1,x2,y2</box>`.
   - **ShowUI-2B** — smaller, less accurate.

   The parser in `grounding_vision._parse_coords` handles all three formats
   plus raw JSON, so swapping models is config-only once one is wired into
   `vision:` (or a future `grounding_vision:` chain).

5. **Safety gate ("Tirith"-style HITL).** Every mutating tool
   (`click_element`, `click_at`, `type_text`, `press_key`, `scroll`) checks
   `_sensitive_context()` — substring match of the frontmost app name +
   focused-window title against a list including `1password`, `keychain`,
   `banking`, `paypal`, `system settings`, `delete`, `erase`, etc. A match
   returns `BLOCKED:` instructing the agent to verbalise the action and
   re-call with `confirm=true`. `mouse_move` is ungated (movement alone
   cannot commit destructive actions).

**Why pyautogui over cliclick or pure-AppleScript:**
- AppleScript `click at {x, y}` only works against an app's scripting
  dictionary, not arbitrary screen coords. Doesn't help for non-scriptable
  UIs.
- `cliclick` works but adds an external CLI dependency.
- `pyautogui` is pure-Python on top of PyObjC (already a transitive dep),
  handles mouse, keyboard, and scroll uniformly, and is widely understood.
  Soft-dependency: lazy-imported so a missing wheel disables only this
  layer.

**Why a dedicated grounding tool instead of extending the vision chain:**
- The descriptive vision chain returns prose. Grounding needs structured
  coordinates. Different prompt, different output schema, different success
  criteria — separate tool keeps each callsite honest.
- The vision chain's prompts are tuned for natural-language description;
  forcing them to also emit JSON coordinates would degrade the descriptive
  case.

**Consequences:**
- The agent can now drive native macOS apps reliably (Finder, Mail, Safari,
  System Settings) via AX. The Spotify failure from 2026-05-10 would now
  resolve via `find_on_screen("the evening jazz playlist row") → click_at`.
- First run requires Accessibility permission for the terminal/Python
  binary. macOS prompts on first AX call; after that, persistent.
- The safety gate is a **substring blocklist**, not a context-aware
  classifier. False positives are possible (a tab whose title contains
  "delete" gets blocked). Tune `_SENSITIVE_PATTERNS` to taste.
- `confirm=true` is a single-bit override — the gate trusts the agent to
  only set it after a verbal yes. The voice loop has no harder enforcement
  available without a separate confirmation tool.
- `find_on_screen` is honest about its limits: it returns the vision
  model's best guess and surfaces raw output when parsing fails, so the
  agent can choose to retry, scroll, or escalate to the user.
- Adding UI-TARS later is a config-only change once the grounding chain is
  parameterised. The `_parse_coords` parser already handles UI-TARS' click
  tokens.
- Tools added to the registry (8 new): `list_ui_elements`, `click_element`,
  `click_at`, `type_text`, `press_key`, `scroll`, `mouse_move`,
  `find_on_screen`.
