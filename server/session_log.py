"""Per-session structured logs for voice-bot turns and tool calls.

Writes JSON Lines to `logs/session-YYYY-MM-DD-HH-MM-SS.jsonl`. Each line is one
event record. The intent: these files become a structured memory the bot can
later surface, search, or feed back into future contexts.

Event names use kebab-case so they're stable as future identifiers:

    session-started
    user-spoke              {text}
    bot-spoke               {text}
    tool-called             {name, args}
    tool-result             {name, result}
    tool-error              {name, error}
    session-ended

Usage:

    log = SessionLog.for_now()                     # opens logs/session-…jsonl
    log.event("session-started", session_id=…)
    log.close()                                    # also writes session-ended

    proc = SessionLogProcessor(log)                # FrameProcessor; pipe in
    pipeline = Pipeline([..., proc, ...])          # captures user/bot text
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_LOG_DIR = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))


class SessionLog:
    """Append-only JSONL session log.

    Each `event()` writes one line: a JSON object with `ts` (epoch seconds),
    `iso` (UTC ISO 8601), `session_id`, `event` (kebab-case name), plus the
    caller's fields. The file is line-buffered so consumers can `tail -f`.
    """

    def __init__(self, path: Path, *, session_id: str | None = None):
        self.path = path
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)
        self.event("session-started", path=str(path))

    @classmethod
    def for_now(cls, log_dir: Path = DEFAULT_LOG_DIR) -> "SessionLog":
        """Open a new log file named `session-{utc-timestamp}.jsonl`."""
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return cls(log_dir / f"session-{ts}.jsonl")

    def event(self, event_name: str, /, **fields: Any) -> None:
        # `event_name` is positional-only so callers can pass arbitrary kwargs
        # in `fields` (including `name=...`) without colliding.
        record = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event_name,
            **fields,
        }
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"Session log write failed for {event_name!r}: {exc}")
            return

        # Mirror a one-line summary to the console so a tail of stdout reads
        # naturally alongside the JSONL file.
        if "text" in fields:
            logger.info(f"[{event_name}] {fields['text']!r}")
        else:
            payload = {k: v for k, v in fields.items() if k not in ("path",)}
            logger.info(f"[{event_name}] {payload}" if payload else f"[{event_name}]")

    def close(self) -> None:
        try:
            self.event("session-ended")
        finally:
            try:
                self._fh.close()
            except Exception:
                pass


class SessionLogProcessor(FrameProcessor):
    """FrameProcessor that captures user transcription and bot text.

    Sits between the transport and STT (or anywhere user transcription frames
    flow) and between the LLM and TTS (or anywhere bot TextFrames flow). It
    aggregates per-turn text and emits `user-spoke` / `bot-spoke` events when
    the corresponding stop-speaking frame arrives.

    Bot text dedup: Pipecat emits both incremental deltas *and* cumulative
    snapshots as `TextFrame`s — sometimes per-token, sometimes per-sentence
    snapshots that include all text-so-far. Naive concatenation gives
    `"He I' I'm Your Your fri Your friend Your friendly..."`. We collapse
    cumulative snapshots: if the new chunk starts with the last buffered
    chunk, replace (it's a longer cumulative); otherwise append (new
    sentence/segment).
    """

    def __init__(self, session_log: SessionLog):
        super().__init__()
        self._log = session_log
        self._user_buf: list[str] = []
        self._bot_buf: list[str] = []

    @staticmethod
    def _add_chunk(buf: list[str], chunk: str) -> None:
        if buf and chunk.startswith(buf[-1]):
            buf[-1] = chunk
        else:
            buf.append(chunk)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                self._add_chunk(self._user_buf, text)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            text = " ".join(self._user_buf).strip()
            self._user_buf.clear()
            if text:
                self._log.event("user-spoke", text=text)
        elif isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if text:
                self._add_chunk(self._bot_buf, text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            text = " ".join(self._bot_buf).strip()
            self._bot_buf.clear()
            if text:
                self._log.event("bot-spoke", text=text)

        await self.push_frame(frame, direction)
