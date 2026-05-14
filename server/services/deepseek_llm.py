"""DeepSeek-aware LLM service that surfaces its proprietary cache fields.

Pipecat's ``OpenAILLMService`` extracts cache stats from
``chunk.usage.prompt_tokens_details.cached_tokens`` — the OpenAI-standard
location. DeepSeek's API doesn't populate that nested field; instead it
returns flat ``usage.prompt_cache_hit_tokens`` and
``usage.prompt_cache_miss_tokens``. Without this shim, every
``llm-usage`` event in the session log would report
``cache_read_input_tokens=null`` even when caching is actively saving
80%+ of prompt tokens.

This subclass wraps the streaming response and copies DeepSeek's flat
field into the standard nested location *before* the parent class
builds its ``LLMTokenUsage``. The parent code path is untouched —
cache stats just start arriving with real numbers.

Activate by setting ``llm.provider: deepseek`` in ``config.yaml``;
``voice_bot.build_llm`` picks this class up from the provider name.
"""

from __future__ import annotations

from typing import AsyncIterator

from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import PromptTokensDetails
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


def _patch_chunk_with_deepseek_cache(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    """Mutate a chunk in place so its OpenAI-standard cache field is populated.

    Returns the same chunk for chaining. No-op when usage is absent or the
    standard field is already set (any non-None value wins over DeepSeek's).
    """
    usage = chunk.usage
    if usage is None:
        return chunk

    # Already populated by an OpenAI-style response — leave alone.
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens is not None:
        return chunk

    # DeepSeek's flat field. Lives on the Pydantic extras when the SDK
    # model has ``extra="allow"``; also accessible via getattr fallback.
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is None and getattr(usage, "model_extra", None):
        hit = usage.model_extra.get("prompt_cache_hit_tokens")
    if hit is None:
        return chunk

    if usage.prompt_tokens_details is None:
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=hit)
    else:
        usage.prompt_tokens_details.cached_tokens = hit
    return chunk


class DeepSeekLLMService(OpenAILLMService):
    """``OpenAILLMService`` with DeepSeek-specific cache-stat surfacing.

    Behaviourally identical to the parent for every other code path. The
    only override is ``get_chat_completions``: it wraps the returned async
    iterator so each chunk is patched as it flows through, *before* the
    parent's ``_process_context`` builds the ``LLMTokenUsage`` that
    becomes the ``llm-usage`` session-log event.
    """

    async def get_chat_completions(
        self, context: LLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        stream = await super().get_chat_completions(context)
        return _wrap_with_cache_patch(stream)


async def _wrap_with_cache_patch(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield each chunk after applying the DeepSeek cache patch."""
    async for chunk in stream:
        yield _patch_chunk_with_deepseek_cache(chunk)
