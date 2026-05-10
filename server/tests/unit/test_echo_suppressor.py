"""Unit tests for echo_suppressor.should_suppress() — pure logic, no pipeline."""

import time
import pytest
from processors.echo_suppressor import should_suppress


class TestShouldSuppress:
    def test_bot_speaking_always_suppresses(self):
        """While the bot is speaking, transcripts are always suppressed."""
        now = time.monotonic()
        assert should_suppress(now, bot_speaking=True, suppress_until=0.0) is True

    def test_within_holdoff_suppresses(self):
        """Within the holdoff window after bot stops, transcripts are suppressed."""
        now = time.monotonic()
        suppress_until = now + 0.5  # holdoff ends 0.5s in the future
        assert should_suppress(now, bot_speaking=False, suppress_until=suppress_until) is True

    def test_after_holdoff_passes(self):
        """After the holdoff window, transcripts pass through."""
        now = time.monotonic()
        suppress_until = now - 0.1  # holdoff expired 0.1s ago
        assert should_suppress(now, bot_speaking=False, suppress_until=suppress_until) is False

    def test_zero_holdoff_passes_immediately(self):
        """With zero holdoff, transcripts pass as soon as bot stops."""
        now = time.monotonic()
        assert should_suppress(now, bot_speaking=False, suppress_until=now) is False

    def test_bot_speaking_overrides_holdoff(self):
        """Bot speaking takes priority over holdoff being expired."""
        now = time.monotonic()
        # holdoff expired, but bot is still speaking
        assert should_suppress(now, bot_speaking=True, suppress_until=now - 10.0) is True
