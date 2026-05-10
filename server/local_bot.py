"""Local voice bot — mic → Soniox STT → DeepSeek LLM → Soniox TTS → speakers.

Reuses the shared STT/TTS/LLM stack from `voice_bot.py`.

Per-session structured logs are written to `logs/session-<ts>.jsonl` as the
session runs (see `session_log.py`).

## Local-audio pipeline concerns

When the transport is `LocalAudioTransport` (mic + speakers), there is no
browser/Daily AEC to suppress speaker bleed-back into the mic. That forces
three concerns that don't exist in a Daily/WebRTC transport:

1. **Mute strategies.** Mute the mic during every bot turn so the bot doesn't
   self-interrupt by hearing its own output. Plus mute from startup until the
   first bot turn completes — otherwise ambient mic noise gets transcribed in
   the ~1s gap between the trigger firing and the first TTS chunk landing,
   which broadcasts an "interruption" and kills the LLM call before the bot
   ever speaks.

2. **Turn-start strategy.** Use VAD only. The default also includes
   `TranscriptionUserTurnStartStrategy`, which opens a phantom turn the
   instant Soniox emits a transcript — including transcripts of speaker-echo
   of the bot's own intro. That phantom turn never cleanly closes, so the
   user's real reply is silently swallowed.

3. **Connection rendezvous.** Both STT and TTS must be connected before
   pushing the seed context frame, otherwise the bot's intro is lost.

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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute.always_user_mute_strategy import (
    AlwaysUserMuteStrategy,
)
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from processors import EchoSuppressor, SessionLog, SessionLogProcessor, WakeWordGate
from hotkey_interrupt import install_interrupt_hotkey
from voice_bot import build_components

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Local-audio mute + turn strategies
# ---------------------------------------------------------------------------

def _local_user_aggregator_params(
    *, user_speech_timeout: float = 0.6
) -> LLMUserAggregatorParams:
    """Mute + turn strategies for mic-and-speakers transports.

    Speech-timeout stop is simpler than the smart-turn ONNX model and fires
    reliably on short replies.
    """
    return LLMUserAggregatorParams(
        user_mute_strategies=[
            MuteUntilFirstBotCompleteUserMuteStrategy(),
            AlwaysUserMuteStrategy(),
        ],
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=user_speech_timeout
                )
            ],
        ),
    )


def _install_local_audio_lifecycle(
    *,
    stt: SonioxSTTService,
    tts: SonioxTTSService,
    context_aggregator: LLMContextAggregatorPair,
) -> None:
    """Wire the dual-connection rendezvous and diagnostic event handlers.

    On both `stt.on_connected` and `tts.on_connected`, push the seed context
    frame exactly once — that triggers the bot's introduction.

    Diagnostics expose every link in the mic → STT → user-turn → LLM → TTS
    chain so a failed run is immediately attributable to a specific stage.
    """
    state = {"stt_ready": False, "tts_ready": False, "triggered": False}

    async def try_trigger():
        if state["stt_ready"] and state["tts_ready"] and not state["triggered"]:
            state["triggered"] = True
            logger.info("Both services connected — triggering introduction")
            await context_aggregator.user().push_context_frame()

    @stt.event_handler("on_connected")
    async def _on_stt_connected(stt):
        state["stt_ready"] = True
        logger.info("Soniox STT connected")
        await try_trigger()

    @tts.event_handler("on_connected")
    async def _on_tts_connected(tts):
        state["tts_ready"] = True
        logger.info("Soniox TTS connected")
        await try_trigger()

    @context_aggregator.user().event_handler("on_user_mute_started")
    async def _on_user_mute_started(*_args):
        logger.info("🔇 USER MUTED (bot is speaking)")

    @context_aggregator.user().event_handler("on_user_mute_stopped")
    async def _on_user_mute_stopped(*_args):
        logger.info("🎤 USER UNMUTED (mic is open)")

    @context_aggregator.user().event_handler("on_user_turn_started")
    async def _on_user_turn_started(*_args):
        logger.info("🗣️  USER TURN STARTED — bot detected you started speaking")

    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(*_args):
        logger.info("✅ USER TURN STOPPED — sending to LLM now")

    @context_aggregator.user().event_handler("on_user_turn_idle")
    async def _on_user_turn_idle(*_args):
        logger.info("💤 user idle (no speech for a while)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        initial_user_message="Hello!",
        user_aggregator_params=_local_user_aggregator_params(),
        session_log=session_log,
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    log_proc_pre = SessionLogProcessor(session_log)
    log_proc_post = SessionLogProcessor(session_log)
    wake_gate = WakeWordGate(components.config.wake_word)
    echo_suppressor = EchoSuppressor(holdoff_seconds=1.0)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            components.stt,
            echo_suppressor,
            wake_gate,
            log_proc_pre,
            components.context_aggregator.user(),
            components.llm,
            log_proc_post,
            components.tts,
            transport.output(),
            components.context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    _install_local_audio_lifecycle(
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
