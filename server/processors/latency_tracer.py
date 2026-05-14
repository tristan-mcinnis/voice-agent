"""Per-turn latency tracer — measures the perceived-response-latency budget.

Voice UX lives or dies by the gap between *user stops speaking* and *bot starts
speaking*. That total is the sum of four phases:

    user_stopped → llm_started      (STT endpoint + turn-stop strategy + queue)
    llm_started  → first_text       (LLM first-token latency)
    first_text   → first_audio      (TTS first-byte latency)
    first_audio  → bot_started      (audio-out pipeline + speaker)

Without timings you can't tell which phase to optimise. This processor stamps
each phase from frames the pipeline already emits and writes one
``turn-latency`` event per turn to the session log, plus a one-line ⏱  summary
to stderr.

Pure logic — ``compute_turn_latency`` — is split out for testability.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from processors.session_log import SessionLog


# Phases recorded per turn. Order matches the expected event sequence.
PHASES = (
    "user_stopped",
    "llm_started",
    "first_text",
    "tts_started",
    "first_audio",
    "bot_started",
)


# ---------------------------------------------------------------------------
# Pure logic — testable without a pipeline
# ---------------------------------------------------------------------------

def compute_turn_latency(stamps: dict[str, float]) -> dict[str, Optional[float]]:
    """Compute per-phase and total latency in milliseconds from raw timestamps.

    Args:
        stamps: monotonic timestamps keyed by phase name (see ``PHASES``).
            Missing keys are tolerated — their dependent deltas come back
            ``None``.

    Returns:
        Dict of millisecond deltas. ``None`` for any delta whose endpoints
        weren't both observed.
    """
    def delta(a: str, b: str) -> Optional[float]:
        if a in stamps and b in stamps:
            return round((stamps[b] - stamps[a]) * 1000.0, 1)
        return None

    return {
        "stt_endpoint_ms": delta("user_stopped", "llm_started"),
        "llm_first_token_ms": delta("llm_started", "first_text"),
        "tts_first_audio_ms": delta("first_text", "first_audio"),
        "text_to_speaker_ms": delta("first_text", "bot_started"),
        "total_response_ms": delta("user_stopped", "bot_started"),
    }


def format_summary(metrics: dict[str, Optional[float]]) -> str:
    """One-line stderr summary. Skips fields that are ``None``."""
    parts = []
    labels = [
        ("stt_endpoint_ms", "stt→llm"),
        ("llm_first_token_ms", "llm→text"),
        ("tts_first_audio_ms", "text→audio"),
        ("text_to_speaker_ms", "text→speaker"),
        ("total_response_ms", "total"),
    ]
    for key, label in labels:
        value = metrics.get(key)
        if value is not None:
            parts.append(f"{label} {value:.0f}ms")
    return "⏱  turn latency  " + "  ".join(parts) if parts else "⏱  turn latency  (no data)"


# ---------------------------------------------------------------------------
# FrameProcessor adapter
# ---------------------------------------------------------------------------

class LatencyTracer(FrameProcessor):
    """Stamps each phase of a turn and logs the budget on bot-start.

    Place alongside ``SessionLogProcessor`` (after wake-gate). It sees user
    frames flowing downstream and bot-speaking frames flowing upstream — same
    vantage point as ``EchoSuppressor``.

    State is per-turn: ``UserStoppedSpeakingFrame`` resets the clock,
    ``BotStartedSpeakingFrame`` emits the metrics. Phases observed only
    once-per-turn (the *first* TextFrame, the *first* TTSAudioRawFrame); later
    frames in the same turn don't overwrite the stamp.
    """

    def __init__(self, session_log: SessionLog):
        super().__init__()
        self._log = session_log
        self._stamps: dict[str, float] = {}

    def _stamp(self, phase: str, *, once: bool = True) -> None:
        if once and phase in self._stamps:
            return
        self._stamps[phase] = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            # Start of a new turn budget — clear any prior state.
            self._stamps = {"user_stopped": time.monotonic()}
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._stamp("llm_started")
        elif isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if text:
                self._stamp("first_text")
        elif isinstance(frame, TTSStartedFrame):
            self._stamp("tts_started")
        elif isinstance(frame, TTSAudioRawFrame):
            self._stamp("first_audio")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._stamp("bot_started")
            self._emit()

        await self.push_frame(frame, direction)

    def _emit(self) -> None:
        if "user_stopped" not in self._stamps:
            # Bot spoke without a preceding user turn (intro, wake ack, etc.).
            # Skip — these aren't response-latency turns.
            return
        metrics = compute_turn_latency(self._stamps)
        self._log.event("turn-latency", **metrics)
        logger.info(format_summary(metrics))
