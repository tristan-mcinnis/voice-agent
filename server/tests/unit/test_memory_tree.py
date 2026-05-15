"""Tests for the Memory Tree leaf store.

Each test scopes the store to a tempdir so they're hermetic and the
real `.voice-agent/memories/tree/` is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.memory_tree import (
    LEAF_CHAR_LIMIT,
    MemoryTree,
    _slugify,
    _snippet_around,
    _split_title,
    _tokenize,
)


@pytest.fixture
def tree(tmp_path: Path) -> MemoryTree:
    return MemoryTree(base_path=tmp_path)


# --- pure helpers -----------------------------------------------------------

def test_slugify_lowercases_and_dashes():
    assert _slugify("Hello, World!") == "hello-world"


def test_slugify_collapses_runs():
    assert _slugify("a / b // c") == "a-b-c"


def test_slugify_empty_falls_back():
    assert _slugify("") == "untitled"


def test_slugify_truncates():
    assert len(_slugify("x" * 200, max_len=30)) == 30


def test_tokenize_drops_punctuation():
    assert _tokenize("Hello, World! 123-abc") == ["hello", "world", "123", "abc"]


def test_split_title_extracts_header():
    title, body = _split_title("# Topic\n\nbody line one\nbody line two")
    assert title == "Topic"
    assert body == "body line one\nbody line two"


def test_split_title_handles_missing_header():
    title, body = _split_title("just body")
    assert title == ""
    assert body == "just body"


def test_snippet_around_centres_on_first_hit():
    content = "intro " * 50 + "FINDME here " + "outro " * 50
    snippet = _snippet_around(content, ["findme"], radius=40)
    assert "FINDME" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")


def test_snippet_around_no_match_returns_prefix():
    content = "lorem ipsum dolor sit amet"
    assert _snippet_around(content, ["zzz"], radius=10) == content[:20]


# --- write/read -------------------------------------------------------------

def test_write_creates_leaf(tree: MemoryTree):
    result = tree.write(title="My Topic", content="some content")
    assert "leaf_id" in result
    assert result["leaf_id"].endswith("my-topic")


def test_write_then_read_round_trip(tree: MemoryTree):
    leaf_id = tree.write(title="Topic", content="hello world")["leaf_id"]
    leaf = tree.read(leaf_id)
    assert leaf is not None
    assert leaf.title == "Topic"
    assert leaf.content == "hello world"


def test_write_rejects_empty_content(tree: MemoryTree):
    assert "empty" in tree.write(title="t", content="   ")["result"].lower()


def test_write_rejects_oversized_content(tree: MemoryTree):
    big = "x" * (LEAF_CHAR_LIMIT + 1)
    out = tree.write(title="t", content=big)
    assert "exceeds" in out["result"].lower()
    assert "leaf_id" not in out


def test_read_missing_returns_none(tree: MemoryTree):
    assert tree.read("nope") is None


def test_write_collision_creates_suffix(tree: MemoryTree):
    a = tree.write(title="Same", content="one")["leaf_id"]
    b = tree.write(title="Same", content="two")["leaf_id"]
    assert a != b
    assert b.startswith(a) and b.endswith("-2")


# --- search -----------------------------------------------------------------

def test_search_finds_by_content(tree: MemoryTree):
    tree.write(title="A", content="the quick brown fox")
    tree.write(title="B", content="completely unrelated text")
    hits = tree.search("brown", k=5)
    assert len(hits) == 1
    assert hits[0]["title"] == "A"


def test_search_ranks_by_token_count(tree: MemoryTree):
    tree.write(title="Few", content="brown")
    tree.write(title="Many", content="brown brown brown")
    hits = tree.search("brown", k=5)
    assert [h["title"] for h in hits] == ["Many", "Few"]


def test_search_empty_query_returns_empty(tree: MemoryTree):
    tree.write(title="A", content="anything")
    assert tree.search("", k=5) == []


def test_search_includes_snippet(tree: MemoryTree):
    tree.write(title="A", content=("padding " * 20) + "needle " + ("trail " * 20))
    hits = tree.search("needle", k=1)
    assert "needle" in hits[0]["snippet"].lower()


# --- list -------------------------------------------------------------------

def test_list_returns_newest_first(tree: MemoryTree):
    first = tree.write(title="First", content="x")["leaf_id"]
    second = tree.write(title="Second", content="y")["leaf_id"]
    listed = [it["leaf_id"] for it in tree.list_leaves()]
    # Both present; second written more recently so should appear first.
    assert second in listed and first in listed
    assert listed.index(second) <= listed.index(first)


def test_list_respects_limit(tree: MemoryTree):
    for i in range(5):
        tree.write(title=f"T{i}", content=f"body {i}")
    assert len(tree.list_leaves(limit=2)) == 2


def test_list_empty_store(tree: MemoryTree):
    assert tree.list_leaves() == []


# --- delete -----------------------------------------------------------------

def test_delete_removes_leaf(tree: MemoryTree):
    leaf_id = tree.write(title="Tmp", content="x")["leaf_id"]
    out = tree.delete(leaf_id)
    assert "Deleted" in out["result"]
    assert tree.read(leaf_id) is None


def test_delete_missing_is_safe(tree: MemoryTree):
    assert "No leaf" in tree.delete("nope")["result"]
