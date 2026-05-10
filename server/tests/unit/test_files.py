"""Unit tests for tools.files._fuzzy_find() — fuzzy matching logic."""

import pytest
from tools.files import _fuzzy_find


class TestFuzzyFind:
    def test_exact_match(self):
        start, end = _fuzzy_find("hello world", "hello")
        assert (start, end) == (0, 5)

    def test_exact_match_middle(self):
        start, end = _fuzzy_find("abc hello world xyz", "hello world")
        assert (start, end) == (4, 15)

    def test_non_unique_returns_none(self):
        """If needle appears more than once exactly, return None."""
        assert _fuzzy_find("hello hello", "hello") is None

    def test_not_found_returns_none(self):
        assert _fuzzy_find("hello world", "goodbye") is None

    def test_whitespace_tolerant_match(self):
        """Whitespace differences are tolerated in fuzzy mode."""
        start, end = _fuzzy_find("hello    world", "hello world")
        assert start >= 0  # found somewhere

    def test_whitespace_tolerant_newlines(self):
        """Newlines are treated as whitespace."""
        start, end = _fuzzy_find("hello\nworld", "hello world")
        assert start >= 0

    def test_whitespace_tolerant_non_unique_returns_none(self):
        """If fuzzy match produces multiple hits, return None."""
        # "hello world" appears exactly at position 0, so exact match wins
        # even before fuzzy matching starts. Need a case where exact fails.
        # "hello   world" with needle "hello  world" — both fuzzy-match
        # to the same pattern, producing ambiguity.
        assert _fuzzy_find("hello   world\nhello    world", "hello  world") is None

    def test_empty_haystack(self):
        assert _fuzzy_find("", "needle") is None

    def test_empty_needle(self):
        # Empty needle matches everywhere — non-unique
        assert _fuzzy_find("hello", "") is None
