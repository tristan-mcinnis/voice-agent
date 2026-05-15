"""Integration tests for the per-turn observation processors.

Pumps synthetic Pipecat frames through SessionLogProcessor + LatencyTracer
in the order a real run produces them, asserting that the events the
session JSONL is supposed to contain actually land. Catches the class of
bug we hit on 2026-05-14:

  - Bot text was being cleared before the BotStoppedSpeakingFrame fired,
    so ``bot-spoke`` events never had any text.
  - LLM ``MetricsFrame``s were silently dropped because of a separate
    Pipecat flag, so ``llm-usage`` events never appeared.

Both bugs would have shown up the second a fake-frame test asserted
"these events end up in the log."
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from processors.latency_tracer import LatencyTracer
from processors.session_log import SessionLog, SessionLogProcessor


async def _setup(processor):
    """Wire the bare minimum that Pipecat 1.1's FrameProcessor expects.

    Without this, ``process_frame`` raises ``TaskManager is still not
    initialized`` because the harness usually does this from Pipeline.setup.
    """
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=tm)
    await processor.setup(setup)


async def _push(processor, frame: Frame) -> None:
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)


def _read_events(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture
def session_log():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="test-session-",
    )
    tmp.close()
    log = SessionLog(Path(tmp.name))
    yield log
    log.close()
    Path(tmp.name).unlink(missing_ok=True)


class TestSessionLogProcessor:
    def test_bot_spoke_lands_with_per_token_chunks(self, session_log):
        """Reproduces the 2026-05-14 bug: per-token text → bot-spoke event."""
        proc = SessionLogProcessor(session_log)

        async def run():
            await _setup(proc)
            await _push(proc, LLMFullResponseStartFrame())
            # Per-token deltas, exactly like DeepSeek streaming.
            for tok in ["Hi", ",", " Tristan", "."]:
                await _push(proc, LLMTextFrame(text=tok))
            await _push(proc, LLMFullResponseEndFrame())
            await _push(proc, BotStartedSpeakingFrame())
            await _push(proc, BotStoppedSpeakingFrame())

        asyncio.run(run())

        events = _read_events(session_log.path)
        bot_spoke = [e for e in events if e["event"] == "bot-spoke"]
        assert len(bot_spoke) == 1
        # Joined cleanly — no phantom spaces.
        assert bot_spoke[0]["text"] == "Hi, Tristan."

    def test_user_spoke_lands(self, session_log):
        proc = SessionLogProcessor(session_log)

        async def run():
            await _setup(proc)
            await _push(proc, TranscriptionFrame(
                text="hello world", user_id="u", timestamp="2026-05-15",
            ))
            await _push(proc, UserStoppedSpeakingFrame())

        asyncio.run(run())

        events = _read_events(session_log.path)
        user_spoke = [e for e in events if e["event"] == "user-spoke"]
        assert len(user_spoke) == 1
        assert user_spoke[0]["text"] == "hello world"

    def test_session_log_processor_ignores_metrics_frames(self, session_log):
        """SessionLogProcessor must never emit llm-usage — that lives in
        LLMUsageLogProcessor now, single-owner across the pipeline."""
        proc = SessionLogProcessor(session_log)
        usage = LLMTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        metrics = MetricsFrame(data=[
            LLMUsageMetricsData(processor="llm", model="x", value=usage)
        ])

        async def run():
            await _setup(proc)
            await _push(proc, metrics)

        asyncio.run(run())

        events = _read_events(session_log.path)
        assert not any(e["event"] == "llm-usage" for e in events)


class TestLLMUsageLogProcessor:
    def test_metrics_frame_emits_llm_usage_event(self, session_log):
        """Pin the cache-hit logging path: MetricsFrame → llm-usage event."""
        from processors import LLMUsageLogProcessor
        proc = LLMUsageLogProcessor(session_log)

        usage = LLMTokenUsage(
            prompt_tokens=1500, completion_tokens=80, total_tokens=1580,
            cache_read_input_tokens=1300,
        )
        metrics = MetricsFrame(data=[
            LLMUsageMetricsData(processor="llm", model="deepseek-v4-flash", value=usage)
        ])

        async def run():
            await _setup(proc)
            await _push(proc, metrics)

        asyncio.run(run())

        events = _read_events(session_log.path)
        usage_events = [e for e in events if e["event"] == "llm-usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["prompt_tokens"] == 1500
        assert usage_events[0]["cache_read_input_tokens"] == 1300


class TestLatencyTracer:
    def test_full_turn_emits_turn_latency_event(self, session_log):
        tracer = LatencyTracer(session_log)

        async def run():
            await _setup(tracer)
            await _push(tracer, UserStoppedSpeakingFrame())
            await asyncio.sleep(0.01)  # measurable phase deltas
            await _push(tracer, LLMFullResponseStartFrame())
            await asyncio.sleep(0.01)
            await _push(tracer, LLMTextFrame(text="Hi"))
            await asyncio.sleep(0.01)
            await _push(tracer, TTSStartedFrame())
            await _push(tracer, BotStartedSpeakingFrame())

        asyncio.run(run())

        events = _read_events(session_log.path)
        latency = [e for e in events if e["event"] == "turn-latency"]
        assert len(latency) == 1
        # Every phase should have produced a positive number.
        assert latency[0]["stt_endpoint_ms"] is not None
        assert latency[0]["llm_first_token_ms"] is not None
        assert latency[0]["text_to_speaker_ms"] is not None
        assert latency[0]["total_response_ms"] > 0

    def test_intro_turn_does_not_emit(self, session_log):
        """Bot speaks without a prior user turn (e.g. wake-word ack) → no event."""
        tracer = LatencyTracer(session_log)

        async def run():
            await _setup(tracer)
            await _push(tracer, LLMFullResponseStartFrame())
            await _push(tracer, LLMTextFrame(text="Ready."))
            await _push(tracer, BotStartedSpeakingFrame())

        asyncio.run(run())

        events = _read_events(session_log.path)
        assert not any(e["event"] == "turn-latency" for e in events)
