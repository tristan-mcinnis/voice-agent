"""Unit tests for ClauseTextAggregator — pure logic, no TTS service."""

import asyncio

from processors.clause_aggregator import ClauseTextAggregator, next_release_index


async def _collect(agg: ClauseTextAggregator, text: str) -> list[str]:
    out: list[str] = []
    async for piece in agg.aggregate(text):
        out.append(piece.text)
    return out


class TestNextReleaseIndex:
    def test_no_punct_returns_none(self):
        assert next_release_index("hello world", min_words=1) is None

    def test_clause_with_lookahead_releases(self):
        text = "first clause, next"
        idx = next_release_index(text, min_words=2)
        assert idx is not None
        assert "first clause," in text[:idx]

    def test_clause_without_lookahead_returns_none(self):
        assert next_release_index("hello, ", min_words=1) is None
        assert next_release_index("hello,", min_words=1) is None

    def test_min_words_blocks_short_clauses(self):
        # "Hi," has 1 word — blocked at min_words=3.
        assert next_release_index("Hi, next more text", min_words=3) is None

    def test_sentence_punct_also_releases(self):
        idx = next_release_index("Hello world. Next sentence", min_words=2)
        assert idx is not None

    def test_multiple_punct_returns_first(self):
        text = "one two, three four. five"
        idx = next_release_index(text, min_words=2)
        assert text[:idx].rstrip() == "one two,"


class TestClauseAggregator:
    def test_releases_at_comma(self):
        agg = ClauseTextAggregator(min_words=2)
        out = asyncio.run(_collect(agg, "hello world, and now more"))
        assert out == ["hello world,"]
        assert agg.text.text == "and now more"

    def test_no_release_below_min_words(self):
        agg = ClauseTextAggregator(min_words=3)
        # "Hi," = 1 word (blocked). Second comma fires on "Hi, next thing here,"
        # (4 words) — but lookahead is still pending after the comma.
        out = asyncio.run(_collect(agg, "Hi, next thing here, "))
        assert out == []

    def test_flush_returns_remaining(self):
        agg = ClauseTextAggregator(min_words=2)
        asyncio.run(_collect(agg, "partial clause"))
        flushed = asyncio.run(agg.flush())
        assert flushed is not None
        assert flushed.text == "partial clause"
        assert asyncio.run(agg.flush()) is None

    def test_interruption_clears_buffer(self):
        agg = ClauseTextAggregator(min_words=1)
        asyncio.run(_collect(agg, "buffered text"))
        asyncio.run(agg.handle_interruption())
        assert agg.text.text == ""

    def test_multiple_clauses_in_one_chunk(self):
        agg = ClauseTextAggregator(min_words=2)
        out = asyncio.run(_collect(agg, "first clause, second clause. third clause more"))
        assert out == ["first clause,", "second clause."]
