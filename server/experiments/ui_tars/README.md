# UI-TARS grounding-vision experiment (planned)

**Status:** placeholder — not yet implemented.

## Idea

Drop a UI-grounding model into the existing vision chain so `find_on_screen`
returns precise pixel coordinates instead of the rough estimates the
descriptive vision chain (Qwen3-VL-2B / Kimi) currently gives.

## Why this isn't urgent

The current chain works well enough for large targets (list rows, big
buttons). The agent can also iterate: screenshot → coords → click → verify
with another screenshot → re-prompt if wrong. That loop tolerates ±50 px
slop on a 5K display for most real tasks.

The upgrade only matters when:

- The user asks the agent to hit a small icon on a dense Electron/canvas UI
  (Spotify mini player, in-app close buttons, game HUDs).
- Verify-and-retry is too slow for the voice loop (each VL call is ~2–5 s).

## Candidate models

| Model | Size | Output format | Notes |
|---|---|---|---|
| `mlx-community/UI-TARS-7B-DPO-mlx` | ~7 GB | `<click>x,y</click>` | Best accuracy. Apple-Silicon-only. |
| `mlx-community/OS-Atlas-Base-7B-mlx` | ~7 GB | `<box>x1,y1,x2,y2</box>` | Comparable; box output we'd center. |
| `showlab/ShowUI-2B` | ~2 GB | varies | Smaller, less accurate; good for laptops. |

`tools/grounding_vision._parse_coords` already handles all three output formats.

## What "wiring it up" looks like

1. Add the chosen model to `requirements.txt` (or document a manual
   `mlx-vlm`-driven download).
2. Add a third entry to the `vision:` chain in `config.example.yaml` with
   `kind: mlx`, `model: <hf-id>`, plus a `purpose: grounding` flag if we
   want to keep it separate from the descriptive chain.
3. Optional: introduce a `grounding_vision:` chain in `config.py` so
   `find_on_screen` consults a different list than `take_screenshot`. Keeps
   each call honest about what it's asking for.
4. Run `find_on_screen` against a set of fixture screenshots
   (Spotify, Finder, System Settings, a game) and measure pixel error
   vs. ground truth.

## When to revisit

- After the safety gate has been used in real sessions (real false-positive
  rate informs whether to invest more in computer-use overall).
- If `find_on_screen` becomes a frequent tool call in session logs and the
  agent's miss rate is observably hurting UX.

## See also

- ADR-0004 — Computer-use tools (AX + coords + safety gate)
- `server/tools/grounding_vision.py` — parser is upgrade-ready
