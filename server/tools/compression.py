"""Tool-result compression — the TokenJuice layer.

Tool outputs are often web scrapes, file dumps, or search payloads dense
with boilerplate the LLM never reads: HTML tags, repeated whitespace,
absurdly long URLs, control characters. Every one of those tokens pushes
real signal further down the context, costs money, and risks blowing the
frozen-prefix cache.

`compress_text(text, options)` and `compress_result(value, options)` are
pure, dependency-free reductions:

  - HTML tags stripped (best-effort regex; the input is already noisy)
  - Script/style blocks dropped entirely
  - URLs longer than `url_max_chars` shortened to `<scheme>://<host>/…`
  - Non-printable control chars removed (newlines/tabs kept)
  - Runs of blank lines collapsed to one
  - Trailing whitespace stripped per line
  - Optional hard cap via `max_chars` with a `…(truncated, N chars)` suffix

The registry handler runs this on every tool result before handing it to
the LLM (see `tools/registry.py:_make_handler`). It runs strictly in
proportion to what came back — bytes-in / bytes-out is logged at debug
level so you can spot which tools benefit most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressionOptions:
    enabled: bool = True
    strip_html: bool = True
    shorten_urls: bool = True
    url_max_chars: int = 80
    collapse_whitespace: bool = True
    strip_control: bool = True
    max_chars: int = 0  # 0 = unlimited


DEFAULT_OPTIONS = CompressionOptions()


# ---------------------------------------------------------------------------
# Regexes (compiled once)
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^<>]{1,2000}>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_ENTITY_MAP = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'",
}
# A liberal URL matcher — captures scheme://host/path-and-query. The
# trailing class excludes whitespace and a few delimiters that would
# obviously end a URL in prose.
_URL_RE = re.compile(r"(https?://[^\s<>\"')\]]+)", re.IGNORECASE)
# Control chars except \t \n \r.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_REPEAT_SPACES_RE = re.compile(r"[ \t]{2,}")


# ---------------------------------------------------------------------------
# Core text reduction
# ---------------------------------------------------------------------------

def _shorten_url(url: str, max_chars: int) -> str:
    if len(url) <= max_chars:
        return url
    # Keep scheme://host plus the first path segment if it fits.
    m = re.match(r"(https?://[^/]+)(/.*)?", url, re.IGNORECASE)
    if not m:
        return url[: max(8, max_chars - 1)] + "…"
    host, path = m.group(1), m.group(2) or ""
    if len(host) >= max_chars:
        return host[: max_chars - 1] + "…"
    budget = max_chars - len(host) - 1
    if budget <= 1 or not path:
        return host + "/…"
    return host + path[:budget] + "…"


def _strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        text = _SCRIPT_STYLE_RE.sub(" ", text)
        text = _HTML_COMMENT_RE.sub(" ", text)
        text = _HTML_TAG_RE.sub(" ", text)
    if "&" in text:
        for entity, replacement in _HTML_ENTITY_MAP.items():
            if entity in text:
                text = text.replace(entity, replacement)
    return text


def compress_text(text: str, options: CompressionOptions = DEFAULT_OPTIONS) -> str:
    """Apply the configured reductions to *text* and return the result."""
    if not options.enabled or not text:
        return text

    out = text

    if options.strip_html and (("<" in out and ">" in out) or "&" in out):
        out = _strip_html(out)

    if options.strip_control:
        out = _CONTROL_RE.sub("", out)

    if options.shorten_urls:
        out = _URL_RE.sub(lambda m: _shorten_url(m.group(1), options.url_max_chars), out)

    if options.collapse_whitespace:
        out = _TRAILING_WS_RE.sub("\n", out)
        out = _REPEAT_SPACES_RE.sub(" ", out)
        out = _BLANK_LINES_RE.sub("\n\n", out)
        out = out.strip()

    if options.max_chars > 0 and len(out) > options.max_chars:
        kept = out[: options.max_chars]
        dropped = len(out) - options.max_chars
        out = f"{kept}\n…(truncated, {dropped} more chars)"

    return out


# ---------------------------------------------------------------------------
# Result walker — tools return strings, dicts, or lists. Compress in place
# without changing structure so the LLM still sees the same JSON shape.
# ---------------------------------------------------------------------------

# Keys whose values are typically identifiers/paths/numbers that compression
# would either no-op on or harm (truncating a path is worse than leaving it).
_SKIP_KEYS = frozenset({
    "exit_code", "image_path", "task_id", "pid", "status",
    "chars_used", "char_limit", "replaced_index",
})


def compress_result(value: Any, options: CompressionOptions = DEFAULT_OPTIONS) -> Any:
    """Recursively compress text inside a tool result.

    Strings are reduced; dicts/lists are walked. Numbers, bools, None pass
    through. Keys in ``_SKIP_KEYS`` are left untouched — they tend to be
    identifiers where stripping would lose information.
    """
    if not options.enabled:
        return value
    if isinstance(value, str):
        return compress_text(value, options)
    if isinstance(value, dict):
        return {
            k: (v if k in _SKIP_KEYS else compress_result(v, options))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [compress_result(v, options) for v in value]
    return value


def measure(value: Any) -> int:
    """Cheap proxy for token cost — total length of string content in *value*."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(measure(v) for v in value.values())
    if isinstance(value, list):
        return sum(measure(v) for v in value)
    return 0
