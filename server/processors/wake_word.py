"""Wake-word gate for the local voice bot.

Sits in the pipeline immediately after the STT service. While **asleep**, it
drops `TranscriptionFrame` / `InterimTranscriptionFrame` so the LLM never sees
them. When a transcript contains the wake phrase (e.g. "hey ava"), it flips to
**awake**, speaks an acknowledgement ("Ready."), and lets subsequent
transcripts pass through normally.

It returns to **asleep** when:
  * a transcript contains the configured sleep phrase ("go to sleep"), or
  * no user speech has been seen for `idle_timeout_seconds`.

Phrase matching is case- and punctuation-insensitive substring matching on
finalized transcripts. Interim transcripts are dropped while asleep but never
trigger a state change (they thrash too much).
"""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from config import WakeWordConfig

_NORMALIZE = re.compile(r"[^a-z0-9 ]+")


# ---------------------------------------------------------------------------
# Pure logic — testable without a pipeline
# ---------------------------------------------------------------------------

def normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation — the canonical form for phrase matching."""
    return _NORMALIZE.sub(" ", text.lower()).strip()


def check_transition(
    *,
    awake: bool,
    wake_phrase: str,
    sleep_phrase: str,
    text: str,
    is_final: bool,
) -> tuple[bool, str | None]:
    """Given current state and a transcript, return (new_awake, ack_text).

    Args:
        awake: Current awake state.
        wake_phrase: The wake phrase (already normalized).
        sleep_phrase: The sleep phrase (already normalized).
        text: The transcript text.
        is_final: True for TranscriptionFrame, False for interim.

    Returns:
        (new_awake, ack_text_or_None). ack_text is set when a state
        transition produces a spoken acknowledgement.
    """
    normalized = normalize_phrase(text or "")

    if not awake:
        if is_final and wake_phrase and wake_phrase in normalized:
            return True, None  # ack handled by caller via config
        return False, None

    # Awake: optionally sleep on phrase.
    if is_final and sleep_phrase and sleep_phrase in normalized:
        return False, None  # sleep ack handled by caller via config

    return True, None


# ---------------------------------------------------------------------------
# FrameProcessor adapter
# ---------------------------------------------------------------------------

class WakeWordGate(FrameProcessor):
    def __init__(self, config: WakeWordConfig):
        super().__init__()
        self._cfg = config
        self._wake = normalize_phrase(config.phrase)
        self._sleep = normalize_phrase(config.sleep_phrase)
        self._awake = config.start_awake
        self._last_activity = time.monotonic()
        self._idle_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not self._cfg.enabled or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcript(frame)
            return

        await self.push_frame(frame, direction)

    async def _handle_transcript(self, frame):
        text = frame.text or ""
        is_final = isinstance(frame, TranscriptionFrame)

        new_awake, _ = check_transition(
            awake=self._awake,
            wake_phrase=self._wake,
            sleep_phrase=self._sleep,
            text=text,
            is_final=is_final,
        )

        if not self._awake and new_awake:
            # Transitioning: asleep → awake.
            await self._wake_up()
            return

        if self._awake and not new_awake:
            # Transitioning: awake → asleep.
            await self._go_to_sleep(speak_ack=True)
            return

        # No transition — either stay asleep (drop frame) or stay awake (pass through).
        if not self._awake:
            return

        self._last_activity = time.monotonic()
        self._ensure_idle_task()
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _wake_up(self):
        self._awake = True
        self._last_activity = time.monotonic()
        logger.info(f"👂 Wake word detected ({self._cfg.phrase!r}) — agent awake")
        if self._cfg.ack_text:
            await self.push_frame(
                TTSSpeakFrame(text=self._cfg.ack_text), FrameDirection.DOWNSTREAM
            )
        self._ensure_idle_task()

    async def _go_to_sleep(self, *, speak_ack: bool = False):
        if not self._awake:
            return
        self._awake = False
        logger.info(f"💤 Going to sleep — say {self._cfg.phrase!r} to wake me")
        if speak_ack and self._cfg.sleep_ack_text:
            await self.push_frame(
                TTSSpeakFrame(text=self._cfg.sleep_ack_text), FrameDirection.DOWNSTREAM
            )

    def _ensure_idle_task(self):
        if self._cfg.idle_timeout_seconds <= 0:
            return
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watchdog())

    async def _idle_watchdog(self):
        timeout = self._cfg.idle_timeout_seconds
        while self._awake:
            elapsed = time.monotonic() - self._last_activity
            remaining = timeout - elapsed
            if remaining <= 0:
                await self._go_to_sleep()
                return
            await asyncio.sleep(remaining)
