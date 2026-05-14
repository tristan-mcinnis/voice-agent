"""Clause-streaming text aggregator for TTS.

Pipecat's default ``SimpleTextAggregator`` releases text to TTS at sentence
boundaries (``.!?``). That's safe but slow — the first audible byte of a
long answer waits for the LLM to finish a whole sentence.

``ClauseTextAggregator`` adds intra-sentence release points (``,;:—``) so TTS
can start speaking the first clause while the LLM is still streaming the
rest. Tradeoff: prosody is slightly choppier at clause boundaries, and edge
cases like ``"$1,000"`` split mid-number. Worth it on long answers (saves
200–400ms of first-audio latency); not worth it on short ones.

A minimum-word threshold suppresses absurdly short releases like ``"Hi,"``.

Pure logic — ``next_release_index`` — is split out for testability.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Optional

from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

# Characters that end a releasable chunk. Sentence-end (.!?) and clause-end
# (,;:—). All require non-whitespace lookahead before they fire, so trailing
# punctuation at end-of-stream falls through to ``flush()``.
_RELEASE_CHARS = frozenset(".!?,;:—")

# Word count is approximated by whitespace runs — good enough to gate
# "Hi, …" but not so strict that it blocks short legitimate clauses.
_WORD_RE = re.compile(r"\S+")


def next_release_index(
    text: str,
    *,
    min_words: int,
) -> Optional[int]:
    """Return the index AFTER the first releasable clause/sentence ending.

    A release point is any of ``.!?,;:—`` followed by at least one whitespace
    char and then a non-whitespace char (the lookahead). The release index is
    the position right after the trailing whitespace, so the next aggregation
    starts cleanly on the next clause.

    Returns ``None`` when the buffer hasn't yet seen a confirmed release.
    """
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in _RELEASE_CHARS:
            continue
        # Need whitespace after the punct.
        j = i + 1
        if j >= n or not text[j].isspace():
            continue
        # Skip whitespace; require a non-whitespace lookahead char.
        while j < n and text[j].isspace():
            j += 1
        if j >= n:
            return None  # Still waiting for lookahead.
        # Confirmed: clause ends at i+1. Check the word-count gate on the
        # clause itself (text[: i+1]).
        clause = text[: i + 1]
        if len(_WORD_RE.findall(clause)) < min_words:
            continue
        return j  # Next aggregation resumes at the lookahead char.
    return None


class ClauseTextAggregator(BaseTextAggregator):
    """Aggregator that releases at clause AND sentence boundaries.

    Args:
        min_words: Don't release a clause shorter than this many words. Set
            to 1 to release every punctuation hit; default 3 avoids "Hi," →
            release alone.
    """

    def __init__(self, *, min_words: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._buf = ""
        self._min_words = min_words

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._buf.strip(" "), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        if self._aggregation_type == AggregationType.TOKEN:
            if text:
                yield Aggregation(text=text, type=AggregationType.TOKEN)
            return

        self._buf += text
        # Multiple releases may have arrived in a single chunk.
        while True:
            cut = next_release_index(self._buf, min_words=self._min_words)
            if cut is None:
                return
            head, self._buf = self._buf[:cut], self._buf[cut:]
            stripped = head.strip(" ")
            if stripped:
                yield Aggregation(text=stripped, type=AggregationType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        if not self._buf:
            return None
        out = self._buf
        self._buf = ""
        stripped = out.strip(" ")
        if not stripped:
            return None
        return Aggregation(text=stripped, type=AggregationType.SENTENCE)

    async def handle_interruption(self):
        self._buf = ""

    async def reset(self):
        self._buf = ""
