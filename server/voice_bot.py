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
from config import Config, get_config, load_config, require_api_key
from tools.vision import init_vision


_PROJECT_ROOT = Path(__file__).parent.parent


def set_agent_home() -> Path:
    """Set the VOICE_AGENT_HOME env var and return the path.

    Call once at startup before any memory or prompt code runs.
    Uses setdefault so it's safe to call from tests that pre-configure
    the env var.
    """
    home = _PROJECT_ROOT / ".voice-agent"
    os.environ.setdefault("VOICE_AGENT_HOME", str(home))
    return home


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


def build_stt(config: Config) -> SonioxSTTService:
    """Construct the STT service from config. Public — usable in tests."""
    cfg = config.stt
    return SonioxSTTService(
        api_key=require_api_key(cfg.api_key_env, for_what="Soniox STT"),
        params=SonioxInputParams(
            language_hints=[_language_from_hint(h) for h in cfg.language_hints],
        ),
    )


def build_tts(config: Config) -> SonioxTTSService:
    """Construct the TTS service from config. Public — usable in tests.

    When ``tts.stream_clauses`` is true, the service's internal text
    aggregator is swapped for a ``ClauseTextAggregator`` so first-audio
    fires at clause boundaries instead of full sentences.
    """
    cfg = config.tts
    tts = SonioxTTSService(
        api_key=require_api_key(cfg.api_key_env, for_what="Soniox TTS"),
        settings=SonioxTTSService.Settings(voice=cfg.voice),
    )
    if cfg.stream_clauses:
        from processors.clause_aggregator import install_on_tts
        install_on_tts(tts, min_words=cfg.clause_min_words)
    return tts


def build_llm(config: Config) -> OpenAILLMService:
    """Construct the LLM service from config. Public — usable in tests.

    When ``llm.provider == "deepseek"``, returns a ``DeepSeekLLMService``
    so that DeepSeek's flat ``prompt_cache_hit_tokens`` is surfaced into
    the OpenAI-standard ``cache_read_input_tokens`` metric — otherwise
    the cache stats in session logs would always be ``null``.
    """
    cfg = config.llm
    service_cls: type[OpenAILLMService]
    if cfg.provider.lower() == "deepseek":
        from services.deepseek_llm import DeepSeekLLMService
        service_cls = DeepSeekLLMService
    else:
        service_cls = OpenAILLMService
    return service_cls(
        api_key=require_api_key(cfg.api_key_env, for_what=f"{cfg.provider} LLM"),
        base_url=cfg.base_url,
        settings=service_cls.Settings(model=cfg.model, extra=cfg.extra),
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
        config: Optional pre-loaded `Config`. Defaults to `get_config()`.
    """
    cfg = config or get_config()

    # One-time setup: agent home directory, vision chain, tool registration.
    set_agent_home()
    init_vision(cfg.vision)
    tools_module.register_all()

    stt = build_stt(cfg)
    tts = build_tts(cfg)
    llm = build_llm(cfg)

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
