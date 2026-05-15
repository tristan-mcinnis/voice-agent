"""Local-audio TurnPolicy — one module for the no-AEC turn lifecycle.

When the transport is ``LocalAudioTransport`` (mic + speakers), there is no
browser/Daily AEC to suppress speaker bleed-back into the mic. That forces
five concerns that don't exist on a Daily/WebRTC transport, and they're
load-bearing in concert:

1. **Mute strategies.** Mute the mic during every bot turn so the bot
   doesn't self-interrupt by hearing its own output. Plus mute from startup
   until the first bot turn completes — otherwise ambient mic noise gets
   transcribed in the ~1s gap between the trigger firing and the first TTS
   chunk landing, broadcasting an "interruption" that kills the LLM call.

2. **Turn-start strategy.** Use VAD only. The default also includes
   ``TranscriptionUserTurnStartStrategy``, which opens a phantom turn the
   instant Soniox emits a transcript — including echo-of-bot transcripts.
   That phantom turn never cleanly closes; the user's real reply is
   silently swallowed.

3. **VAD analyzer placement.** In Pipecat 1.1.0 the ``vad_analyzer``
   parameter on ``LocalAudioTransportParams`` is silently dropped on
   construction. To get ``VADUserStartedSpeakingFrame`` events — which
   ``VADUserTurnStartStrategy`` listens for — the analyzer must live in a
   ``VADProcessor`` placed in the pipeline.

4. **Echo suppression.** ``EchoSuppressor`` drops transcription frames
   while the bot is speaking and for a brief holdoff afterward. Without
   AEC, Soniox transcribes the bot's own TTS bleeding back into the mic.

5. **Connection rendezvous.** Push the seed context frame only after both
   STT and TTS WebSockets are connected; otherwise the bot's intro is lost.

This module owns all five and exposes them as one cohesive policy. Adding
a transport with hardware AEC (Daily/WebRTC) means writing a sibling
policy module, not editing the local-bot wiring.

Glossary (see ``CONTEXT.md``): **TurnPolicy** — the bundle of mute
strategies, turn-start/stop strategies, echo suppressor, wake-word gate,
VAD processor placement, and connection rendezvous that together define
how a turn starts and ends on a given transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.turns.user_mute.always_user_mute_strategy import (
    AlwaysUserMuteStrategy,
)
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import (
    BaseUserTurnStopStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from config import Config
from connection_rendezvous import ConnectionRendezvous
from processors import (
    EchoSuppressor,
    LatencyTracer,
    LLMUsageLogProcessor,
    SessionLog,
    SessionLogProcessor,
    WakeWordGate,
)


_DIAG_EVENTS = (
    ("on_user_mute_started", "🔇 USER MUTED (bot is speaking)"),
    ("on_user_mute_stopped", "🎤 USER UNMUTED (mic is open)"),
    ("on_user_turn_started", "🗣️  USER TURN STARTED — bot detected you started speaking"),
    ("on_user_turn_stopped", "✅ USER TURN STOPPED — sending to LLM now"),
    ("on_user_turn_idle",    "💤 user idle (no speech for a while)"),
)


@dataclass
class LocalAudioTurnPolicy:
    """Cohesive bundle of pipeline processors + lifecycle wiring for local audio.

    Construct with config; consume via:

      - ``aggregator_params`` — pass to ``build_components(...)``
      - ``processors_pre_aggregator(stt)`` — pipeline stages before the user aggregator
      - ``processors_post_aggregator(llm)`` — pipeline stages between LLM and TTS
      - ``install_lifecycle(stt, tts, context_aggregator)`` — wires connection
        rendezvous + diagnostics
    """

    config: Config
    session_log: SessionLog

    def __post_init__(self) -> None:
        self.vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
        self.echo_suppressor = EchoSuppressor(
            holdoff_seconds=self.config.turn.echo_holdoff_seconds
        )
        self.wake_gate = WakeWordGate(self.config.wake_word)
        self.log_proc_pre = SessionLogProcessor(self.session_log)
        self.log_proc_post = SessionLogProcessor(self.session_log)
        # LLM token-usage logging lives in its own processor — single owner
        # so we don't double-count cache hits when MetricsFrames propagate.
        self.usage_log = LLMUsageLogProcessor(self.session_log)
        self.latency_tracer = LatencyTracer(self.session_log)

    # ------------------------------------------------------------------
    # Aggregator params — mute + turn strategies for no-AEC transports
    # ------------------------------------------------------------------

    @property
    def aggregator_params(self) -> LLMUserAggregatorParams:
        """Mute + turn strategies for mic-and-speakers transports.

        Stop strategy is selected by ``config.turn.smart_turn_enabled``:
          - false (default): ``SpeechTimeoutUserTurnStopStrategy`` — fixed
            silence timeout. Simple, reliable, but adds ~500ms of dead air.
          - true: smart-turn-v3 ONNX model wrapped in
            ``TurnAnalyzerUserTurnStopStrategy``. Cuts dead air on
            definite-end utterances. Falls back to speech-timeout if
            ``pipecat-ai[local-smart-turn]`` isn't installed.
        """
        stop_strategy = self._build_stop_strategy()
        return LLMUserAggregatorParams(
            user_mute_strategies=[
                MuteUntilFirstBotCompleteUserMuteStrategy(),
                AlwaysUserMuteStrategy(),
            ],
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[stop_strategy],
            ),
        )

    def _build_stop_strategy(self) -> BaseUserTurnStopStrategy:
        """Return the turn-stop strategy implied by config.

        Lazy-imports smart-turn deps so the default config doesn't require
        ``pipecat-ai[local-smart-turn]``. On any import or load failure,
        logs a warning and falls back to speech-timeout — the bot stays
        usable.
        """
        speech_timeout = SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=self.config.turn.user_speech_timeout
        )
        if not self.config.turn.smart_turn_enabled:
            return speech_timeout

        try:
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
                LocalSmartTurnAnalyzerV3,
            )
            from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
                TurnAnalyzerUserTurnStopStrategy,
            )
        except Exception as exc:
            logger.warning(
                f"smart_turn_enabled=true but deps missing ({exc!s}); "
                f"falling back to speech-timeout. Install with: "
                f"pip install 'pipecat-ai[local-smart-turn]'"
            )
            return speech_timeout

        try:
            analyzer = LocalSmartTurnAnalyzerV3()
        except Exception as exc:
            logger.warning(
                f"smart-turn-v3 failed to load ({exc!s}); "
                f"falling back to speech-timeout."
            )
            return speech_timeout

        logger.info("🧠 smart-turn-v3 enabled — using ONNX endpointer")
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def processors_pre_aggregator(self) -> list:
        """Stages between transport.input() and the user aggregator.

        Order matters: VAD → STT → echo-suppress → wake-gate → log.
        STT is constructed by ``build_components``, so the caller splices
        it in. This method returns the stages adjacent to STT.
        """
        return [self.vad]

    def processors_after_stt(self) -> list:
        """Stages between STT and the user aggregator."""
        return [
            self.echo_suppressor,
            self.wake_gate,
            self.log_proc_pre,
            self.latency_tracer,
        ]

    def processors_post_llm(self) -> list:
        """Stages between the LLM and TTS."""
        return [self.log_proc_post, self.usage_log]

    # ------------------------------------------------------------------
    # Lifecycle — connection rendezvous + diagnostics
    # ------------------------------------------------------------------

    def install_lifecycle(
        self,
        *,
        stt: SonioxSTTService,
        tts: SonioxTTSService,
        context_aggregator: LLMContextAggregatorPair,
    ) -> None:
        """Wire the dual-connection rendezvous and observability handlers.

        Pushes the seed context frame exactly once when both STT and TTS
        are connected — that triggers the bot's introduction.
        """
        timeout = self.config.turn.connection_timeout_seconds

        async def _on_rendezvous_timeout(pending: list[str]) -> None:
            self.session_log.event("connection-timeout", pending=pending,
                                   timeout_seconds=timeout)

        rendezvous = ConnectionRendezvous(
            callback=context_aggregator.user().push_context_frame,
            timeout_seconds=timeout if timeout > 0 else None,
            on_timeout=_on_rendezvous_timeout,
        )

        @stt.event_handler("on_connected")
        async def _on_stt_connected(stt):
            logger.info("Soniox STT connected")
            await rendezvous.stt_ready()

        @tts.event_handler("on_connected")
        async def _on_tts_connected(tts):
            logger.info("Soniox TTS connected")
            await rendezvous.tts_ready()

        self._install_diagnostics(context_aggregator)

    @staticmethod
    def _install_diagnostics(context_aggregator: LLMContextAggregatorPair) -> None:
        """Pure-logging event handlers — every link in the chain logs."""
        user = context_aggregator.user()
        for event, msg in _DIAG_EVENTS:
            async def _handler(*_args, _msg: str = msg) -> None:
                logger.info(_msg)
            user.event_handler(event)(_handler)
