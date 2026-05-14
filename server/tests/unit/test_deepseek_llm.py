"""Unit tests for DeepSeek cache-field patching — pure logic."""

import asyncio

from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from services.deepseek_llm import _patch_chunk_with_deepseek_cache, _wrap_with_cache_patch


def _make_chunk(usage_dict: dict | None) -> ChatCompletionChunk:
    """Build a minimal chunk. Usage may be None or a dict of usage fields."""
    payload = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [],
    }
    if usage_dict is not None:
        payload["usage"] = usage_dict
    return ChatCompletionChunk.model_validate(payload)


class TestPatch:
    def test_deepseek_flat_field_promoted(self):
        chunk = _make_chunk({
            "prompt_tokens": 1500, "completion_tokens": 80, "total_tokens": 1580,
            "prompt_cache_hit_tokens": 1300, "prompt_cache_miss_tokens": 200,
        })
        _patch_chunk_with_deepseek_cache(chunk)
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens_details is not None
        assert chunk.usage.prompt_tokens_details.cached_tokens == 1300

    def test_openai_standard_field_left_alone(self):
        # If the response already has the nested OpenAI field, don't overwrite.
        chunk = _make_chunk({
            "prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 900},
            "prompt_cache_hit_tokens": 999,  # Should be ignored.
        })
        _patch_chunk_with_deepseek_cache(chunk)
        assert chunk.usage.prompt_tokens_details.cached_tokens == 900

    def test_missing_usage_is_noop(self):
        chunk = _make_chunk(None)
        _patch_chunk_with_deepseek_cache(chunk)
        assert chunk.usage is None

    def test_no_cache_fields_at_all(self):
        # Plain non-cached response — patch leaves it untouched.
        chunk = _make_chunk({
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        })
        _patch_chunk_with_deepseek_cache(chunk)
        # Either None or unset details — both are fine; key is cached_tokens stays absent.
        details = chunk.usage.prompt_tokens_details
        assert details is None or details.cached_tokens is None

    def test_explicit_zero_is_preserved(self):
        # First-turn DeepSeek response: cache miss everywhere.
        chunk = _make_chunk({
            "prompt_tokens": 1500, "completion_tokens": 80, "total_tokens": 1580,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1500,
        })
        _patch_chunk_with_deepseek_cache(chunk)
        assert chunk.usage.prompt_tokens_details.cached_tokens == 0


class TestWrapStream:
    def test_yields_patched_chunks(self):
        async def fake_stream():
            yield _make_chunk({
                "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                "prompt_cache_hit_tokens": 70,
            })
            yield _make_chunk(None)  # mid-stream content chunk with no usage

        async def collect():
            return [c async for c in _wrap_with_cache_patch(fake_stream())]

        out = asyncio.run(collect())
        assert len(out) == 2
        assert out[0].usage.prompt_tokens_details.cached_tokens == 70
        assert out[1].usage is None
