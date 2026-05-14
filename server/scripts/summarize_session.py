"""Aggregate one session's latency + cache stats into a single report.

The pipeline's ``LatencyTracer`` emits one ``turn-latency`` event per turn and
``SessionLogProcessor`` emits one ``llm-usage`` event per LLM call. Per-turn
lines are useful in the moment; aggregating across a session is what you want
when comparing two settings (e.g. ``smart_turn_enabled`` on vs off).

Usage::

    python -m scripts.summarize_session logs/session-2026-05-14-21-00-00.jsonl
    python -m scripts.summarize_session logs/session-*.jsonl    # multi-file

Output:

    session-2026-05-14-21-00-00.jsonl
      turns: 12   llm calls: 12

      latency (ms)            p50    p95    max
      stt_endpoint           120    340    520
      llm_first_token        180    410    680
      tts_first_audio         60    120    180
      text_to_speaker         90    160    220
      total_response         420    780   1180

      llm cache
      cache_read tokens (sum): 14 320  / 15 800 prompt tokens (90.6%)
      cache_creation tokens:    1 480
      completion tokens:        2 110

Pure functions ``load_events`` / ``percentile`` / ``summarize`` are
unit-testable; ``main`` is the only side-effectful piece.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

_LATENCY_FIELDS = (
    ("stt_endpoint_ms", "stt_endpoint"),
    ("llm_first_token_ms", "llm_first_token"),
    ("tts_first_audio_ms", "tts_first_audio"),
    ("text_to_speaker_ms", "text_to_speaker"),
    ("total_response_ms", "total_response"),
)


@dataclass
class SessionSummary:
    path: Path
    turns: int
    llm_calls: int
    latency: dict[str, dict[str, float]]   # label -> {p50, p95, max}
    cache_read: int
    cache_creation: int
    prompt_total: int
    completion_total: int
    config: dict          # session-config event payload (without ts/iso wrapper)


def load_events(path: Path) -> list[dict]:
    """Parse a JSONL session log. Malformed lines are skipped silently."""
    events: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def percentile(values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile. ``pct`` is in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


_CONFIG_NOISE_KEYS = {"ts", "iso", "session_id", "event"}


def summarize(events: Iterable[dict], path: Path) -> SessionSummary:
    """Reduce a session's events into one SessionSummary."""
    events = list(events)
    turns = [e for e in events if e.get("event") == "turn-latency"]
    usage = [e for e in events if e.get("event") == "llm-usage"]
    cfg_events = [e for e in events if e.get("event") == "session-config"]
    config: dict = {}
    if cfg_events:
        config = {k: v for k, v in cfg_events[0].items() if k not in _CONFIG_NOISE_KEYS}

    latency: dict[str, dict[str, float]] = {}
    for field, label in _LATENCY_FIELDS:
        vals = [e[field] for e in turns if e.get(field) is not None]
        if vals:
            latency[label] = {
                "p50": percentile(vals, 50) or 0.0,
                "p95": percentile(vals, 95) or 0.0,
                "max": max(vals),
            }

    def total(key: str) -> int:
        return sum((e.get(key) or 0) for e in usage)

    return SessionSummary(
        path=path,
        turns=len(turns),
        llm_calls=len(usage),
        latency=latency,
        cache_read=total("cache_read_input_tokens"),
        cache_creation=total("cache_creation_input_tokens"),
        prompt_total=total("prompt_tokens"),
        completion_total=total("completion_tokens"),
        config=config,
    )


def format_report(summary: SessionSummary) -> str:
    """Human-readable single-session report."""
    lines: list[str] = [summary.path.name]
    lines.append(f"  turns: {summary.turns}   llm calls: {summary.llm_calls}")
    cfg = summary.config
    if cfg:
        flags = [
            f"model={cfg.get('llm_model', '?')}",
            f"voice={cfg.get('tts_voice', '?')}",
            f"stream_clauses={cfg.get('stream_clauses')}",
            f"smart_turn={cfg.get('smart_turn_enabled')}",
            f"user_speech_timeout={cfg.get('user_speech_timeout')}",
        ]
        lines.append(f"  config: {'  '.join(flags)}")
    lines.append("")

    if summary.latency:
        lines.append("  latency (ms)            p50    p95    max")
        for _, label in _LATENCY_FIELDS:
            stats = summary.latency.get(label)
            if not stats:
                continue
            lines.append(
                f"  {label:<18}  {stats['p50']:6.0f} {stats['p95']:6.0f} {stats['max']:6.0f}"
            )
        lines.append("")

    if summary.llm_calls:
        prompt = summary.prompt_total
        cache_read = summary.cache_read
        hit_rate = (cache_read / prompt * 100.0) if prompt else 0.0
        lines.append("  llm cache")
        lines.append(
            f"  cache_read tokens (sum): {cache_read:>7} / "
            f"{prompt:>6} prompt tokens ({hit_rate:.1f}%)"
        )
        lines.append(f"  cache_creation tokens:   {summary.cache_creation:>7}")
        lines.append(f"  completion tokens:       {summary.completion_total:>7}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.split("\n\nUsage::")[1].strip(), file=sys.stderr)
        return 2

    paths: list[Path] = []
    for arg in argv:
        paths.extend(sorted(Path().glob(arg)) if any(c in arg for c in "*?[") else [Path(arg)])

    if not paths:
        print("no files matched", file=sys.stderr)
        return 1

    for i, path in enumerate(paths):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        summary = summarize(load_events(path), path)
        if i:
            print()
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
