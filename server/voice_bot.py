"""Shared voice-bot construction.

`build_components` reads `config.yaml` (via `config.py`) and produces the
STT / TTS / LLM / context / aggregator stack that both `bot.py` (Daily WebRTC
+ Pipecat Cloud) and `local_bot.py` (mic+speakers) build their pipelines on
top of. The transport, animation, and any transport-specific lifecycle (RTVI
handlers, mute strategies) stay outside.

To swap providers/models (DeepSeek ↔ Kimi/Moonshot ↔ OpenAI ↔ …), edit
`config.yaml`. Anything that's an OpenAI-compatible chat-completions endpoint
plugs in by changing `llm.provider`, `llm.base_url`, `llm.model`,
`llm.api_key_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.transcriptions.language import Language

import tools as tools_module
from agent.prompt_builder import PromptBuilder
from config import Config, load_config, require_api_key


# Project-local agent home — keeps voice-agent data separate from any other
# agent system on this machine. Exported via env var so tools (e.g.
# patch_memory) write to the same location the PromptBuilder reads from.
_PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("VOICE_AGENT_HOME", str(_PROJECT_ROOT / ".voice-agent"))


@dataclass
class BotComponents:
    """The shared pieces every transport plugs into a Pipeline."""

    stt: SonioxSTTService
    tts: SonioxTTSService
    llm: OpenAILLMService
    context: LLMContext
    context_aggregator: LLMContextAggregatorPair
    config: Config


def _language_from_hint(hint: str) -> Language:
    return getattr(Language, hint.upper())


def _build_stt(config: Config) -> SonioxSTTService:
    cfg = config.stt
    if cfg.provider != "soniox":
        raise NotImplementedError(
            f"STT provider {cfg.provider!r} not wired up. Add a builder in voice_bot._build_stt."
        )
    return SonioxSTTService(
        api_key=require_api_key(cfg.api_key_env, for_what="Soniox STT"),
        params=SonioxInputParams(
            language_hints=[_language_from_hint(h) for h in cfg.language_hints],
        ),
    )


def _build_tts(config: Config) -> SonioxTTSService:
    cfg = config.tts
    if cfg.provider != "soniox":
        raise NotImplementedError(
            f"TTS provider {cfg.provider!r} not wired up. Add a builder in voice_bot._build_tts."
        )
    return SonioxTTSService(
        api_key=require_api_key(cfg.api_key_env, for_what="Soniox TTS"),
        settings=SonioxTTSService.Settings(voice=cfg.voice),
    )


def _build_llm(config: Config) -> OpenAILLMService:
    cfg = config.llm
    return OpenAILLMService(
        api_key=require_api_key(cfg.api_key_env, for_what=f"{cfg.provider} LLM"),
        base_url=cfg.base_url,
        settings=OpenAILLMService.Settings(model=cfg.model, extra=cfg.extra),
    )


def build_components(
    *,
    initial_user_message: Optional[str] = None,
    user_aggregator_params: Optional[LLMUserAggregatorParams] = None,
    session_log=None,
    config: Optional[Config] = None,
) -> BotComponents:
    """Construct the shared STT + TTS + LLM + context stack from config.

    Args:
        initial_user_message: Seed the context with a user message before the
            bot's first turn. The local-audio transport needs this — without a
            preceding user turn the LLMContextAggregatorPair's user-side state
            machine doesn't transition cleanly after the intro, and the next
            spoken user turn is never picked up. Daily/WebRTC doesn't need it
            because the client UI sends a real user-turn signal.
        user_aggregator_params: Pass in to install custom mute / turn strategies
            (see `local_bot._local_user_aggregator_params`). Daily/WebRTC
            transports leave this `None` because the browser handles AEC.
        session_log: Optional `SessionLog` instance. When provided, tool calls
            get logged as kebab-case events alongside user/bot speech.
        config: Optional pre-loaded `Config`. Defaults to `load_config()`.
    """
    cfg = config or load_config()

    stt = _build_stt(cfg)
    tts = _build_tts(cfg)
    llm = _build_llm(cfg)

    schemas = tools_module.REGISTRY.register_handlers(llm, session_log=session_log)
    tools = ToolsSchema(standard_tools=schemas)

    # Assemble the frozen system prompt once per session.
    # The PromptBuilder layers Soul → Memory → User → Rules → Tools → Skills
    # so that the stable prefix stays warm in provider-side caches.
    project_root = Path(__file__).parent.parent
    prompt_builder = PromptBuilder(
        registry=tools_module.REGISTRY,
        default_identity=cfg.system_prompt,
        agent_home=project_root / ".voice-agent",
    )
    system_prompt = prompt_builder.build(
        user_input=initial_user_message or "",
        cwd=project_root,
    )
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
    ]
    if initial_user_message is not None:
        messages.append({"role": "user", "content": initial_user_message})
    context = LLMContext(messages, tools)

    if user_aggregator_params is not None:
        context_aggregator = LLMContextAggregatorPair(
            context, user_params=user_aggregator_params
        )
    else:
        context_aggregator = LLMContextAggregatorPair(context)

    return BotComponents(
        stt=stt, tts=tts, llm=llm,
        context=context, context_aggregator=context_aggregator,
        config=cfg,
    )
