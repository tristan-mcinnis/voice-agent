"""Local voice bot — mic → Soniox STT → DeepSeek LLM → Soniox TTS → speakers.

Reuses the shared STT/TTS/LLM stack from ``voice_bot.py``. The no-AEC turn
lifecycle — mute strategies, VAD-only turn-start, echo suppressor, wake-word
gate, connection rendezvous, diagnostics — is owned by ``turn_policy``.
This module just builds the transport, asks the policy for its pieces, and
assembles the pipeline.

Per-session structured logs are written to ``logs/session-<ts>.jsonl`` as
the session runs (see ``session_log.py``).
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from config import init_config
from processors import SessionLog
from hotkey_interrupt import install_interrupt_hotkey
from turn_policy import LocalAudioTurnPolicy
from voice_bot import build_components, set_agent_home

# Set the agent home before any agent.* imports resolve VOICE_AGENT_HOME.
set_agent_home()

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


async def main():
    cfg = init_config()
    session_log = SessionLog.for_now()
    logger.info(f"Session log: {session_log.path}")

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    policy = LocalAudioTurnPolicy(config=cfg, session_log=session_log)

    components = build_components(
        initial_user_message="Hello!",
        user_aggregator_params=policy.aggregator_params,
        session_log=session_log,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            *policy.processors_pre_aggregator(),
            components.stt,
            *policy.processors_after_stt(),
            components.context_aggregator.user(),
            components.llm,
            *policy.processors_post_llm(),
            components.tts,
            transport.output(),
            components.context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    policy.install_lifecycle(
        stt=components.stt,
        tts=components.tts,
        context_aggregator=components.context_aggregator,
    )
    install_interrupt_hotkey(task, hotkey=cfg.hotkey.interrupt)

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        session_log.close()


if __name__ == "__main__":
    print("\n🎤 Speaking — say something into your mic\n")
    asyncio.run(main())
