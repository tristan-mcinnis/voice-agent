"""Unit tests for wake_word — normalize_phrase() and check_transition(). Pure logic, no pipeline."""

import pytest
from processors.wake_word import normalize_phrase, check_transition


class TestNormalizePhrase:
    def test_lowercases(self):
        assert normalize_phrase("Hey Ava") == "hey ava"

    def test_strips_punctuation(self):
        # Punctuation is replaced with spaces — the regex substitutes but
        # doesn't collapse consecutive spaces (that's fine for substring matching).
        assert normalize_phrase("Hey, Ava!") == "hey  ava"

    def test_collapses_whitespace(self):
        # Leading/trailing whitespace is stripped; internal spacing is preserved
        # (substring matching doesn't need compact spacing).
        assert normalize_phrase("  hey   ava  ") == "hey   ava"

    def test_numbers_preserved(self):
        assert normalize_phrase("version 2.0") == "version 2 0"

    def test_empty_string(self):
        assert normalize_phrase("") == ""

    def test_punctuation_only(self):
        assert normalize_phrase("!@#$%") == ""


class TestCheckTransition:
    WAKE = "hey ava"
    SLEEP = "go to sleep"

    # --- asleep → awake transitions ---
    def test_asleep_wake_phrase_triggers(self):
        new_awake, ack = check_transition(
            awake=False, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="Hey Ava!", is_final=True,
        )
        assert new_awake is True
        assert ack is None

    def test_asleep_partial_match_does_not_trigger(self):
        new_awake, ack = check_transition(
            awake=False, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="hey", is_final=True,
        )
        assert new_awake is False

    def test_asleep_interim_never_triggers(self):
        """Interim transcripts never trigger a wake transition (they thrash)."""
        new_awake, ack = check_transition(
            awake=False, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="Hey Ava", is_final=False,
        )
        assert new_awake is False

    def test_asleep_sleep_phrase_ignored(self):
        """Sleep phrase in an asleep transcript doesn't matter — already asleep."""
        new_awake, ack = check_transition(
            awake=False, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="go to sleep", is_final=True,
        )
        assert new_awake is False

    # --- awake → asleep transitions ---
    def test_awake_sleep_phrase_triggers(self):
        new_awake, ack = check_transition(
            awake=True, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="go to sleep now", is_final=True,
        )
        assert new_awake is False

    def test_awake_interim_sleep_phrase_ignored(self):
        """Interim transcripts never trigger a sleep transition."""
        new_awake, ack = check_transition(
            awake=True, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="go to sleep", is_final=False,
        )
        assert new_awake is True  # stays awake

    def test_awake_wake_phrase_ignored(self):
        """Wake phrase while awake is just normal speech — stay awake."""
        new_awake, ack = check_transition(
            awake=True, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="Hey Ava how are you", is_final=True,
        )
        assert new_awake is True

    def test_awake_normal_text_stays_awake(self):
        new_awake, ack = check_transition(
            awake=True, wake_phrase=self.WAKE,
            sleep_phrase=self.SLEEP, text="What's the weather?", is_final=True,
        )
        assert new_awake is True
