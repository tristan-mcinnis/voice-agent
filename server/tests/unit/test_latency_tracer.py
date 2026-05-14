"""Unit tests for compute_turn_latency() — pure logic, no pipeline."""

from processors.latency_tracer import compute_turn_latency, format_summary


class TestComputeTurnLatency:
    def test_full_turn(self):
        # Stamps in monotonic seconds — 50ms STT, 300ms LLM, 80ms TTS, 40ms speaker.
        stamps = {
            "user_stopped": 100.000,
            "llm_started":  100.050,
            "first_text":   100.350,
            "tts_started":  100.380,
            "first_audio":  100.430,
            "bot_started":  100.470,
        }
        m = compute_turn_latency(stamps)
        assert m["stt_endpoint_ms"] == 50.0
        assert m["llm_first_token_ms"] == 300.0
        assert m["tts_first_audio_ms"] == 80.0
        assert m["text_to_speaker_ms"] == 120.0
        assert m["total_response_ms"] == 470.0

    def test_missing_stamps_yield_none(self):
        stamps = {"user_stopped": 0.0, "llm_started": 0.1}
        m = compute_turn_latency(stamps)
        assert m["stt_endpoint_ms"] == 100.0
        assert m["llm_first_token_ms"] is None
        assert m["tts_first_audio_ms"] is None
        assert m["total_response_ms"] is None

    def test_empty_stamps(self):
        m = compute_turn_latency({})
        assert all(v is None for v in m.values())

    def test_rounded_to_one_decimal(self):
        stamps = {"user_stopped": 0.0, "llm_started": 0.0123456}
        m = compute_turn_latency(stamps)
        assert m["stt_endpoint_ms"] == 12.3


class TestFormatSummary:
    def test_full_summary_present(self):
        m = compute_turn_latency({
            "user_stopped": 0.0,
            "llm_started":  0.05,
            "first_text":   0.35,
            "first_audio":  0.43,
            "bot_started":  0.47,
        })
        s = format_summary(m)
        assert "stt→llm 50ms" in s
        assert "llm→text 300ms" in s
        assert "total 470ms" in s

    def test_skips_none_fields(self):
        m = {"stt_endpoint_ms": 50.0, "llm_first_token_ms": None,
             "tts_first_audio_ms": None, "text_to_speaker_ms": None,
             "total_response_ms": None}
        s = format_summary(m)
        assert "stt→llm 50ms" in s
        assert "llm→text" not in s
        assert "total" not in s

    def test_all_none_no_data(self):
        m = {k: None for k in [
            "stt_endpoint_ms", "llm_first_token_ms", "tts_first_audio_ms",
            "text_to_speaker_ms", "total_response_ms",
        ]}
        assert format_summary(m) == "⏱  turn latency  (no data)"
