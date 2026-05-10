"""Drop STT frames while the bot is speaking (and shortly after).

`LocalAudioTransport` has no echo cancellation, so the bot's own TTS audio
bleeds into the mic and Soniox transcribes it. Even with the mute strategies, a transcript finalized just after the bot starts/stops
speaking can poison the LLM context (and the session log).

This processor sits right after STT and drops `TranscriptionFrame` and
`InterimTranscriptionFrame` from the moment a `BotStartedSpeakingFrame`
arrives until `holdoff_seconds` after the matching `BotStoppedSpeakingFrame`.
"""

from __future__ import annotations

import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# ---------------------------------------------------------------------------
# Pure logic — testable without a pipeline
# ---------------------------------------------------------------------------

def should_suppress(
    now: float,
    bot_speaking: bool,
    suppress_until: float,
) -> bool:
    """Return True when a transcript should be dropped.

    Args:
        now: Current monotonic timestamp.
        bot_speaking: Whether the bot is currently producing audio.
        suppress_until: The monotonic timestamp until which suppression
            extends after the bot stops speaking.
    """
    return bot_speaking or now < suppress_until


# ---------------------------------------------------------------------------
# FrameProcessor adapter
# ---------------------------------------------------------------------------

class EchoSuppressor(FrameProcessor):
    def __init__(self, holdoff_seconds: float = 1.0):
        super().__init__()
        self._holdoff = holdoff_seconds
        self._bot_speaking = False
        self._suppress_until = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._suppress_until = time.monotonic() + self._holdoff

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if should_suppress(time.monotonic(), self._bot_speaking, self._suppress_until):
                logger.debug(f"echo-suppressed: {frame.text!r}")
                return

        await self.push_frame(frame, direction)
