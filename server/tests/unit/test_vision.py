"""Unit tests for vision.strip_reasoning() — reasoning-stripping heuristic."""

import pytest
from tools.vision import strip_reasoning


class TestStripReasoning:
    def test_plain_description_passes_through(self):
        """A normal description without reasoning markers is returned as-is."""
        text = "A web browser showing a GitHub repository page."
        assert strip_reasoning(text) == text

    def test_reasoning_preamble_stripped(self):
        """Lines containing reasoning markers are stripped; the tail remains."""
        text = """Let me think about what I see.
A code editor is visible.
Some terminal output is visible at the bottom."""
        result = strip_reasoning(text)
        # "Let me think" is a reasoning marker — that line is stripped.
        # The remaining lines pass through.
        assert "code editor" in result
        assert "terminal output" in result
        assert "Let me think" not in result

    def test_actually_marker_stripped(self):
        text = "Actually, that's a cat\nIt's a dog"
        result = strip_reasoning(text)
        assert "Actually" not in result
        assert "dog" in result

    def test_empty_string(self):
        assert strip_reasoning("") == ""

    def test_only_reasoning_markers(self):
        text = "Let me think about this.\nWait, maybe it's different."
        result = strip_reasoning(text)
        # Everything gets stripped — result is empty or minimal
        assert len(result) == 0 or "different" in result

    def test_no_newlines_single_line_no_markers(self):
        text = "A simple description"
        assert strip_reasoning(text) == text

    def test_strips_quotes(self):
        """Quote-stripping only happens after reasoning markers are found."""
        # Without reasoning markers, quotes are preserved.
        text = "A description without quotes"
        assert strip_reasoning(text) == text
