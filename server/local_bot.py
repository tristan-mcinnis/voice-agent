"""Local voice bot — mic → Soniox STT → DeepSeek LLM → Soniox TTS → speakers.

Reuses the shared STT/TTS/LLM stack from `voice_bot.py`. All mic/speaker-only
concerns (mute strategies, turn strategies, connection rendezvous, diagnostics)
live in `local_audio.py` so this file is just transport + pipeline assembly.

Per-session structured logs are written to `logs/session-<ts>.jsonl` as the
session runs (see `session_log.py`).

Note: in Pipecat 1.1.0 the `vad_analyzer` parameter on `LocalAudioTransportParams`
is silently ignored (not a Pydantic field, dropped on construction). To get
`VADUserStartedSpeakingFrame` events — which `VADUserTurnStartStrategy` listens
for — the VAD analyzer must live in a `VADProcessor` placed in the pipeline.
Without this, the bot speaks its intro and then never picks up the user's reply.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from echo_suppressor import EchoSuppressor
from hotkey_interrupt import install_interrupt_hotkey
from local_audio import install_local_audio_lifecycle, local_user_aggregator_params
from session_log import SessionLog, SessionLogProcessor
from voice_bot import build_components
from wake_word import WakeWordGate

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


async def main():
    session_log = SessionLog.for_now()
    logger.info(f"Session log: {session_log.path}")

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    components = build_components(
        # Seed a user "Hello!" before the bot's first turn. Without this, the
        # user-side aggregator's state machine doesn't transition cleanly after
        # the intro and the next spoken user turn is silently dropped.
        initial_user_message="Hello!",
        user_aggregator_params=local_user_aggregator_params(),
        session_log=session_log,
    )

    # VADProcessor emits VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame.
    # The DailyTransport handles VAD internally; LocalAudioTransport doesn't, so
    # we must place a VADProcessor in the pipeline to get those frames.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    log_proc = SessionLogProcessor(session_log)
    wake_gate = WakeWordGate(components.config.wake_word)
    echo_suppressor = EchoSuppressor(holdoff_seconds=1.0)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            components.stt,
            echo_suppressor,
            wake_gate,
            log_proc,
            components.context_aggregator.user(),
            components.llm,
            components.tts,
            transport.output(),
            components.context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    install_local_audio_lifecycle(
        stt=components.stt,
        tts=components.tts,
        context_aggregator=components.context_aggregator,
    )
    install_interrupt_hotkey(task)

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        session_log.close()


if __name__ == "__main__":
    print("\n🎤 Speaking — say something into your mic\n")
    asyncio.run(main())
