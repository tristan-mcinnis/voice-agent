"""Unit tests for scripts.summarize_session — pure aggregation logic."""

from pathlib import Path

from scripts.summarize_session import (
    SessionSummary,
    format_report,
    percentile,
    summarize,
)


class TestPercentile:
    def test_empty(self):
        assert percentile([], 50) is None

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_median_odd(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_p95(self):
        # 20 evenly-spaced values: p95 lands at the 19.05th element (interp).
        values = list(range(1, 21))
        p95 = percentile(values, 95)
        assert p95 is not None
        assert 19.0 < p95 < 20.1


class TestSummarize:
    def test_counts_turns_and_calls(self):
        events = [
            {"event": "user-spoke"},
            {"event": "turn-latency", "total_response_ms": 500.0},
            {"event": "turn-latency", "total_response_ms": 600.0},
            {"event": "llm-usage", "prompt_tokens": 100, "completion_tokens": 20,
             "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0,
             "total_tokens": 120, "reasoning_tokens": None},
        ]
        s = summarize(events, Path("test.jsonl"))
        assert s.turns == 2
        assert s.llm_calls == 1

    def test_latency_percentiles(self):
        events = [
            {"event": "turn-latency", "total_response_ms": 100.0,
             "stt_endpoint_ms": 50.0},
            {"event": "turn-latency", "total_response_ms": 200.0,
             "stt_endpoint_ms": 60.0},
        ]
        s = summarize(events, Path("test.jsonl"))
        assert s.latency["total_response"]["max"] == 200.0
        assert s.latency["stt_endpoint"]["max"] == 60.0

    def test_skips_none_latency_fields(self):
        # A bot-only "intro" turn has user_stopped == None, so all fields None.
        events = [
            {"event": "turn-latency", "total_response_ms": None,
             "stt_endpoint_ms": None},
            {"event": "turn-latency", "total_response_ms": 500.0},
        ]
        s = summarize(events, Path("test.jsonl"))
        assert s.latency["total_response"]["max"] == 500.0
        assert "stt_endpoint" not in s.latency  # no non-None values

    def test_cache_aggregation(self):
        events = [
            {"event": "llm-usage", "prompt_tokens": 1000, "completion_tokens": 50,
             "cache_read_input_tokens": 800, "cache_creation_input_tokens": 200,
             "total_tokens": 1050, "reasoning_tokens": None},
            {"event": "llm-usage", "prompt_tokens": 1100, "completion_tokens": 60,
             "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0,
             "total_tokens": 1160, "reasoning_tokens": None},
        ]
        s = summarize(events, Path("test.jsonl"))
        assert s.prompt_total == 2100
        assert s.cache_read == 1700
        assert s.cache_creation == 200
        assert s.completion_total == 110

    def test_empty_session(self):
        s = summarize([], Path("test.jsonl"))
        assert s.turns == 0
        assert s.llm_calls == 0
        assert s.latency == {}


class TestFormatReport:
    def test_renders_when_no_data(self):
        s = SessionSummary(
            path=Path("empty.jsonl"), turns=0, llm_calls=0, latency={},
            cache_read=0, cache_creation=0, prompt_total=0, completion_total=0,
        )
        out = format_report(s)
        assert "empty.jsonl" in out
        assert "turns: 0" in out
        # No "latency" or "llm cache" sections when there's nothing to show.
        assert "p50" not in out
        assert "llm cache" not in out

    def test_renders_cache_hit_rate(self):
        s = SessionSummary(
            path=Path("ok.jsonl"), turns=1, llm_calls=1, latency={},
            cache_read=900, cache_creation=100, prompt_total=1000,
            completion_total=50,
        )
        out = format_report(s)
        assert "90.0%" in out
