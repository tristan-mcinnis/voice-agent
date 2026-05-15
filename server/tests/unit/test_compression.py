"""Unit tests for tools.compression — the TokenJuice layer.

Each test exercises one reduction in isolation so a regression points
straight at the offending switch in CompressionOptions.
"""

from __future__ import annotations

from tools.compression import (
    CompressionOptions,
    compress_result,
    compress_text,
    measure,
)


# ---------------------------------------------------------------------------
# compress_text
# ---------------------------------------------------------------------------

def test_disabled_passes_through_unchanged():
    opts = CompressionOptions(enabled=False)
    raw = "<p>  hello   \n\n\nworld  </p>"
    assert compress_text(raw, opts) == raw


def test_html_tags_stripped():
    out = compress_text("<p>hello <b>world</b></p>")
    assert out == "hello world"


def test_html_script_and_style_dropped():
    raw = "<style>body{color:red}</style>before<script>alert(1)</script>after"
    assert compress_text(raw) == "before after"


def test_html_entities_decoded():
    assert compress_text("a&nbsp;b &amp; c &lt;d&gt;") == "a b & c <d>"


def test_long_url_shortened():
    url = "https://example.com/" + "x" * 200
    out = compress_text("see " + url, CompressionOptions(url_max_chars=40))
    assert "…" in out
    assert len(out) < 60


def test_short_url_left_alone():
    url = "https://example.com/foo"
    assert url in compress_text("see " + url)


def test_url_shortening_keeps_host():
    url = "https://example.com/" + "x" * 200
    out = compress_text(url, CompressionOptions(url_max_chars=40))
    assert out.startswith("https://example.com/")


def test_control_chars_stripped():
    raw = "hello\x00\x07world\x1b[31m"
    assert compress_text(raw) == "helloworld[31m"


def test_newlines_and_tabs_preserved():
    raw = "line1\n\tline2"
    assert compress_text(raw) == "line1\n\tline2"


def test_blank_runs_collapsed():
    raw = "a\n\n\n\n\nb\n\n\n\nc"
    # >=3 consecutive newlines collapse to exactly 2.
    assert compress_text(raw) == "a\n\nb\n\nc"


def test_trailing_line_whitespace_stripped():
    raw = "foo   \nbar\t \nbaz"
    assert compress_text(raw) == "foo\nbar\nbaz"


def test_max_chars_truncates_with_marker():
    raw = "x" * 5000
    out = compress_text(raw, CompressionOptions(max_chars=100))
    assert out.startswith("x" * 100)
    assert "truncated" in out
    assert "4900" in out


def test_max_chars_zero_means_unlimited():
    raw = "x" * 5000
    assert compress_text(raw, CompressionOptions(max_chars=0)) == raw


def test_empty_input():
    assert compress_text("") == ""


# ---------------------------------------------------------------------------
# compress_result — structure walker
# ---------------------------------------------------------------------------

def test_compress_result_walks_dicts():
    out = compress_result({"a": "<p>hi</p>", "b": 1})
    assert out == {"a": "hi", "b": 1}


def test_compress_result_walks_lists():
    out = compress_result(["<p>x</p>", 1, None, True])
    assert out == ["x", 1, None, True]


def test_compress_result_skips_identifier_keys():
    raw_path = "/Users/me/logs/captures/img-2026-05-15.jpg"
    out = compress_result({
        "image_path": raw_path,
        "exit_code": 0,
        "task_id": "abc-123-very-long-id-that-must-survive",
        "result": "<p>ok</p>",
    })
    assert out["image_path"] == raw_path
    assert out["exit_code"] == 0
    assert out["task_id"] == "abc-123-very-long-id-that-must-survive"
    assert out["result"] == "ok"


def test_compress_result_disabled_returns_same_object():
    val = {"a": "<p>x</p>"}
    assert compress_result(val, CompressionOptions(enabled=False)) is val


def test_compress_result_passes_through_primitives():
    assert compress_result(42) == 42
    assert compress_result(None) is None
    assert compress_result(True) is True


# ---------------------------------------------------------------------------
# measure — bytes-in / bytes-out accounting
# ---------------------------------------------------------------------------

def test_measure_string():
    assert measure("hello") == 5


def test_measure_dict_recursive():
    assert measure({"a": "hi", "b": ["xx", "yyy"]}) == 2 + 2 + 3


def test_measure_ignores_non_strings():
    assert measure({"a": "x", "n": 12345, "ok": True}) == 1


def test_measure_empty():
    assert measure("") == 0
    assert measure({}) == 0
    assert measure([]) == 0
