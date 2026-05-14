"""Per-session structured logs for voice-bot turns and tool calls.

Writes JSON Lines to `logs/session-YYYY-MM-DD-HH-MM-SS.jsonl`. Each line is one
event record. The intent: these files become a structured memory the bot can
later surface, search, or feed back into future contexts.

Event names use kebab-case so they're stable as future identifiers:

    session-started
    user-spoke              {text}
    llm-response-started    (LLM began generating a reply)
    llm-response-ended      (LLM finished generating; no reply implies a hang)
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
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_LOG_DIR = Path(os.getenv("VOICE_BOT_LOG_DIR", "logs"))


# ---------------------------------------------------------------------------
# Pure logic — testable without a pipeline
# ---------------------------------------------------------------------------

def _leaves(chunks: list[str]) -> list[str]:
    """Return chunks not strictly extended by any other chunk in the list.

    Pipecat emits a mix of per-token deltas AND per-sentence cumulative
    snapshots as ``TextFrame``s. Collect every chunk, then keep only the
    "leaves" — chunks that no other chunk extends as a longer prefix. This
    collapses each cumulative snapshot stream down to its final form while
    preserving distinct sentences.
    """
    out: list[str] = []
    for i, c in enumerate(chunks):
        extended = any(
            j != i and other != c and other.startswith(c)
            for j, other in enumerate(chunks)
        )
        if not extended:
            out.append(c)
    # De-dupe identical chunks while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


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

    def record_config(self, config) -> None:
        """Snapshot the active runtime config so the summariser can A/B sessions.

        Only the dials that affect latency, prompt caching, or turn-taking go
        in — these are the variables a user would actually compare across
        sessions. Provider names are included so a future Soniox-vs-Deepgram
        diff also lands cleanly.
        """
        self.event(
            "session-config",
            llm_provider=config.llm.provider,
            llm_model=config.llm.model,
            stt_provider=config.stt.provider,
            tts_provider=config.tts.provider,
            tts_voice=config.tts.voice,
            stream_clauses=config.tts.stream_clauses,
            clause_min_words=config.tts.clause_min_words,
            user_speech_timeout=config.turn.user_speech_timeout,
            echo_holdoff_seconds=config.turn.echo_holdoff_seconds,
            smart_turn_enabled=config.turn.smart_turn_enabled,
            connection_timeout_seconds=config.turn.connection_timeout_seconds,
            wake_word_enabled=config.wake_word.enabled,
        )

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

    Bot text dedup: Pipecat emits a mix of per-token deltas AND per-sentence
    cumulative snapshots as `TextFrame`s. Naive concatenation gives gibberish
    like `"I' I'm Your Your fri Your friend Your friendly..."`. We collect
    every chunk between BotStartedSpeakingFrame and BotStoppedSpeakingFrame,
    then keep only the "leaves" — chunks that no other chunk extends as a
    longer prefix. That collapses each cumulative snapshot stream down to its
    final form while preserving distinct sentences.
    """

    def __init__(self, session_log: SessionLog, *, track_usage: bool = False):
        super().__init__()
        self._log = session_log
        self._user_chunks: list[str] = []
        self._bot_chunks: list[str] = []
        # Only one instance per pipeline should track LLM token usage —
        # otherwise every MetricsFrame (which propagates to all processors)
        # gets logged twice. Set ``track_usage=True`` on the post-LLM
        # instance only.
        self._track_usage = track_usage

    @staticmethod
    def _leaves(chunks: list[str]) -> list[str]:
        """Backward-compat shim — delegates to the module-level function."""
        return _leaves(chunks)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._log.event("llm-response-started")
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._log.event("llm-response-ended")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_chunks.clear()
        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                self._user_chunks.append(text)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            text = " ".join(self._leaves(self._user_chunks)).strip()
            self._user_chunks.clear()
            if text:
                self._log.event("user-spoke", text=text)
        elif isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if text:
                self._bot_chunks.append(text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            text = " ".join(self._leaves(self._bot_chunks)).strip()
            self._bot_chunks.clear()
            if text:
                self._log.event("bot-spoke", text=text)
        elif self._track_usage and isinstance(frame, MetricsFrame):
            self._log_usage(frame)

        await self.push_frame(frame, direction)

    def _log_usage(self, frame: MetricsFrame) -> None:
        """Emit an ``llm-usage`` event for each LLM usage entry in the frame.

        DeepSeek (and other OpenAI-compatible providers) report cache stats in
        ``cache_read_input_tokens``. A non-zero value means the cognitive-stack
        prefix is being reused — exactly what ``PromptBuilder`` is designed
        to enable. ``cache_creation_input_tokens`` shows the first-turn cost
        of seeding the cache.
        """
        for entry in frame.data:
            if not isinstance(entry, LLMUsageMetricsData):
                continue
            usage = entry.value
            self._log.event(
                "llm-usage",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
