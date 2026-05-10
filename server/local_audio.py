"""Local-audio-only pipeline concerns.

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

Lifting all of this out of `local_bot.py` keeps the *why* in one module a
future explorer can read top-to-bottom.
"""

from __future__ import annotations

from loguru import logger

from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies


def local_user_aggregator_params(
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
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)],
        ),
    )


def install_local_audio_lifecycle(
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
