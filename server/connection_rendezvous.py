"""Connection rendezvous — fire a callback once when two async events both arrive.

Used by local_bot.py to wait for both STT and TTS WebSocket connections before
pushing the seed context frame (triggering the bot's introduction).

The concept is documented in CONTEXT.md under "Connection Rendezvous".
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


class ConnectionRendezvous:
    """Wait for two events, then fire a callback exactly once.

    Thread-safe only for the two designated callers — ``stt_ready()`` and
    ``tts_ready()`` may each be called from any thread, but must each be
    called at most once.

    Usage::

        rendezvous = ConnectionRendezvous(callback=context_aggregator.user().push_context_frame)

        @stt.event_handler("on_connected")
        async def _on_stt_connected(stt):
            logger.info("Soniox STT connected")
            await rendezvous.stt_ready()

        @tts.event_handler("on_connected")
        async def _on_tts_connected(tts):
            logger.info("Soniox TTS connected")
            await rendezvous.tts_ready()
    """

    def __init__(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._callback = callback
        self._stt_ready = False
        self._tts_ready = False
        self._fired = False
        self._lock = asyncio.Lock()

    async def stt_ready(self) -> None:
        """Signal that the STT connection is established."""
        self._stt_ready = True
        await self._try_fire()

    async def tts_ready(self) -> None:
        """Signal that the TTS connection is established."""
        self._tts_ready = True
        await self._try_fire()

    async def _try_fire(self) -> None:
        async with self._lock:
            if self._stt_ready and self._tts_ready and not self._fired:
                self._fired = True
                await self._callback()
