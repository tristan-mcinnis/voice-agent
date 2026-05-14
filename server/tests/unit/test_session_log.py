"""Unit tests for SessionLogProcessor._leaves() — dedup logic, no pipeline."""

import pytest
from processors.session_log import SessionLogProcessor, _smart_join


class TestSmartJoin:
    def test_per_token_chunks_no_phantom_spaces(self):
        # The bug exposed by the live run on 2026-05-14: ``" ".join`` produced
        # ``"Hi , Tristan ."`` because each token was a distinct chunk.
        chunks = ["Hi", ",", " Tristan", "."]
        assert _smart_join(chunks) == "Hi, Tristan."

    def test_distinct_sentences_get_spaces(self):
        chunks = ["Hello.", "How are you?", "I'm doing well."]
        assert _smart_join(chunks) == "Hello. How are you? I'm doing well."

    def test_single_chunk_passthrough(self):
        assert _smart_join(["I'm Your friendly assistant."]) == "I'm Your friendly assistant."

    def test_empty_input(self):
        assert _smart_join([]) == ""

    def test_skips_empty_chunks(self):
        assert _smart_join(["Hi", "", "there"]) == "Hi there"

    def test_closing_punct_no_leading_space(self):
        # Closing brackets / quotes shouldn't get pushed away from the word.
        assert _smart_join(["He said", " \"hi\""]) == "He said \"hi\""

    def test_chunk_already_ends_with_space(self):
        assert _smart_join(["Hi ", "there"]) == "Hi there"


class TestLeaves:
    def test_single_chunk(self):
        assert SessionLogProcessor._leaves(["hello"]) == ["hello"]

    def test_cumulative_snapshots_collapse_to_final(self):
        """Cumulative snapshots: each extends the prior — only the last survives."""
        chunks = ["I'", "I'm", "I'm Your", "I'm Your friend", "I'm Your friendly"]
        assert SessionLogProcessor._leaves(chunks) == ["I'm Your friendly"]

    def test_distinct_sentences_all_survive(self):
        """When chunks are distinct sentences, none extends another — all survive."""
        chunks = ["Hello.", "How are you?", "I'm doing well."]
        assert SessionLogProcessor._leaves(chunks) == ["Hello.", "How are you?", "I'm doing well."]

    def test_identical_duplicates_removed(self):
        """Identical chunks are deduplicated."""
        chunks = ["hello", "hello", "world", "world"]
        assert SessionLogProcessor._leaves(chunks) == ["hello", "world"]

    def test_partial_overlap(self):
        """A chunk that's a prefix of another is removed; the longer one stays."""
        chunks = ["The quick", "The quick brown fox", "separate"]
        assert SessionLogProcessor._leaves(chunks) == ["The quick brown fox", "separate"]

    def test_reverse_order(self):
        """Leaves works regardless of chunk order — the longer variant always wins."""
        chunks = ["The quick brown fox", "The quick"]
        assert SessionLogProcessor._leaves(chunks) == ["The quick brown fox"]

    def test_empty_list(self):
        assert SessionLogProcessor._leaves([]) == []

    def test_all_prefixes_of_one(self):
        chunks = ["a", "ab", "abc", "abcd"]
        assert SessionLogProcessor._leaves(chunks) == ["abcd"]
