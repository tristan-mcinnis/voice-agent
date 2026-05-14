"""Connection rendezvous — fire a callback once when two async events both arrive.

Used by local_bot.py to wait for both STT and TTS WebSocket connections before
pushing the seed context frame (triggering the bot's introduction).

When a ``timeout_seconds`` is supplied, the rendezvous will log which side(s)
failed to connect and call an optional ``on_timeout`` after that long. Without
this, a stalled WebSocket (bad API key, network blip, provider outage) would
silently hang the bot forever with no startup audio.

The concept is documented in CONTEXT.md under "Connection Rendezvous".
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger


class ConnectionRendezvous:
    """Wait for two events, then fire a callback exactly once.

    Args:
        callback: Coroutine fired when both ``stt_ready()`` and
            ``tts_ready()`` have been called.
        timeout_seconds: Optional deadline (seconds) measured from
            construction. When the deadline passes without both sides
            ready, ``on_timeout`` is invoked (or a warning is logged if
            none provided) with the list of sides that never reported.
            ``None`` disables the deadline.
        on_timeout: Optional coroutine called with a list of pending
            sides (e.g. ``["stt"]``) when the deadline expires. Ignored
            if ``timeout_seconds`` is ``None``.

    Both ``stt_ready()`` and ``tts_ready()`` may be called from any task,
    but each at most once.
    """

    def __init__(
        self,
        callback: Callable[[], Awaitable[None]],
        *,
        timeout_seconds: Optional[float] = None,
        on_timeout: Optional[Callable[[list[str]], Awaitable[None]]] = None,
    ) -> None:
        self._callback = callback
        self._stt_ready = False
        self._tts_ready = False
        self._fired = False
        self._lock = asyncio.Lock()
        self._timeout_seconds = timeout_seconds
        self._on_timeout = on_timeout
        self._timeout_task: asyncio.Task | None = None
        if timeout_seconds is not None:
            self._timeout_task = asyncio.create_task(
                self._wait_for_timeout(), name="rendezvous-timeout"
            )

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
                if self._timeout_task is not None and not self._timeout_task.done():
                    self._timeout_task.cancel()
                await self._callback()

    async def _wait_for_timeout(self) -> None:
        try:
            await asyncio.sleep(self._timeout_seconds or 0.0)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._fired:
                return
            pending = pending_sides(self._stt_ready, self._tts_ready)
        logger.error(
            f"ConnectionRendezvous timed out after {self._timeout_seconds:.1f}s "
            f"waiting for: {', '.join(pending)}. Check API keys and network."
        )
        if self._on_timeout is not None:
            try:
                await self._on_timeout(pending)
            except Exception as exc:
                logger.warning(f"on_timeout callback failed: {exc!s}")


# ---------------------------------------------------------------------------
# Pure logic — testable without async
# ---------------------------------------------------------------------------

def pending_sides(stt_ready: bool, tts_ready: bool) -> list[str]:
    """Return the names of sides that haven't reported ready yet."""
    out: list[str] = []
    if not stt_ready:
        out.append("stt")
    if not tts_ready:
        out.append("tts")
    return out
