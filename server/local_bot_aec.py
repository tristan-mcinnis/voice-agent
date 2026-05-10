"""Local voice bot WITH barge-in via Speex acoustic echo cancellation.

Same components as `local_bot.py`, but the pipeline is rewired so the user can
interrupt the bot mid-sentence:

  * `AlwaysUserMuteStrategy` is dropped — the mic stays open during bot turns.
  * `EchoSuppressor` is removed — STT transcripts are no longer dropped while
    the bot speaks.
  * `AECMicProcessor` cleans bot echo out of the mic input before it reaches
    STT, and `AECReferenceTap` feeds the bot's TTS output to the canceller as
    the far-end reference.

When the user starts speaking mid-bot-turn, Pipecat's intrinsic interruption
handling (triggered by `UserStartedSpeakingFrame`) cancels the in-flight LLM
and TTS — which only works because AEC keeps the bot from interrupting itself.

`local_bot.py` is left untouched. Pick the entry point that matches your
hardware:

  * Open speakers, want barge-in → `python local_bot_aec.py`
  * Headphones, or you don't need barge-in → `python local_bot.py`

Requires `speexdsp` (with the Apple-Silicon install fix in `aec_processor.py`'s
docstring) and `scipy`.
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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from aec_processor import AECMicProcessor, AECReferenceTap, AECState
from local_audio import install_local_audio_lifecycle
from session_log import SessionLog, SessionLogProcessor
from voice_bot import build_components
from wake_word import WakeWordGate

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


def _aec_user_aggregator_params() -> LLMUserAggregatorParams:
    """Like `local_audio.local_user_aggregator_params` but WITHOUT
    `AlwaysUserMuteStrategy` — barge-in needs an open mic during bot turns.

    `MuteUntilFirstBotComplete…` is kept so the very first intro doesn't get
    drowned by mic noise before AEC has any reference to cancel against.
    """
    return LLMUserAggregatorParams(
        user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)],
        ),
    )


async def main():
    session_log = SessionLog.for_now()
    logger.info(f"Session log: {session_log.path}")
    logger.info("AEC barge-in mode — mic stays open during bot turns")

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    components = build_components(
        initial_user_message="Hello!",
        user_aggregator_params=_aec_user_aggregator_params(),
        session_log=session_log,
    )

    aec_state = AECState(mic_sample_rate=16000)
    aec_mic = AECMicProcessor(aec_state)
    aec_ref = AECReferenceTap(aec_state)

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    log_proc = SessionLogProcessor(session_log)
    wake_gate = WakeWordGate(components.config.wake_word)

    pipeline = Pipeline(
        [
            transport.input(),
            aec_mic,                       # echo-cancel mic before VAD/STT
            vad,
            components.stt,
            wake_gate,
            log_proc,
            components.context_aggregator.user(),
            components.llm,
            components.tts,
            aec_ref,                       # tap TTS as far-end reference
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

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        session_log.close()


if __name__ == "__main__":
    print("\n🎤 Speaking — say something into your mic (interrupt anytime)\n")
    asyncio.run(main())
