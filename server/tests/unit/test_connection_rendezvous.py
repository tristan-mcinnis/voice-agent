"""Unit tests for ConnectionRendezvous — async coordination logic."""

import asyncio
import pytest
from connection_rendezvous import ConnectionRendezvous, pending_sides


class TestPendingSides:
    def test_both_pending(self):
        assert pending_sides(False, False) == ["stt", "tts"]

    def test_only_stt_ready(self):
        assert pending_sides(True, False) == ["tts"]

    def test_only_tts_ready(self):
        assert pending_sides(False, True) == ["stt"]

    def test_both_ready(self):
        assert pending_sides(True, True) == []


class TestTimeout:
    def test_on_timeout_fires_with_pending(self):
        timed_out_with: list[str] | None = None

        async def fired():
            raise AssertionError("callback should not fire")

        async def on_timeout(pending):
            nonlocal timed_out_with
            timed_out_with = pending

        async def run():
            rv = ConnectionRendezvous(
                callback=fired, timeout_seconds=0.05, on_timeout=on_timeout,
            )
            await rv.tts_ready()  # only one side
            await asyncio.sleep(0.1)
            assert timed_out_with == ["stt"]

        asyncio.run(run())

    def test_no_timeout_if_both_ready_first(self):
        timed_out = False

        async def fired():
            pass

        async def on_timeout(pending):
            nonlocal timed_out
            timed_out = True

        async def run():
            rv = ConnectionRendezvous(
                callback=fired, timeout_seconds=0.05, on_timeout=on_timeout,
            )
            await rv.stt_ready()
            await rv.tts_ready()
            await asyncio.sleep(0.1)
            assert timed_out is False

        asyncio.run(run())


class TestConnectionRendezvous:
    def test_fires_once_when_both_ready(self):
        fired = False

        async def callback():
            nonlocal fired
            fired = True

        async def run():
            rv = ConnectionRendezvous(callback=callback)
            await rv.stt_ready()
            assert not fired  # only one ready, not yet
            await rv.tts_ready()
            assert fired  # both ready, callback fired

        asyncio.run(run())

    def test_reverse_order(self):
        """TTs first, then STT — still fires exactly once."""
        fired_count = 0

        async def callback():
            nonlocal fired_count
            fired_count += 1

        async def run():
            rv = ConnectionRendezvous(callback=callback)
            await rv.tts_ready()
            assert fired_count == 0
            await rv.stt_ready()
            assert fired_count == 1

        asyncio.run(run())

    def test_does_not_fire_twice(self):
        """Calling stt_ready() or tts_ready() multiple times doesn't refire."""
        fired_count = 0

        async def callback():
            nonlocal fired_count
            fired_count += 1

        async def run():
            rv = ConnectionRendezvous(callback=callback)
            await rv.stt_ready()
            await rv.tts_ready()
            assert fired_count == 1
            # Call again — should not fire again
            await rv.stt_ready()
            await rv.tts_ready()
            assert fired_count == 1

        asyncio.run(run())

    def test_concurrent_arrival(self):
        """When both signals arrive concurrently, callback fires exactly once."""
        fired_count = 0

        async def callback():
            nonlocal fired_count
            fired_count += 1

        async def run():
            rv = ConnectionRendezvous(callback=callback)
            await asyncio.gather(rv.stt_ready(), rv.tts_ready())
            assert fired_count == 1

        asyncio.run(run())

    def test_only_one_ready_never_fires(self):
        """If only one side connects, callback never fires."""
        fired = False

        async def callback():
            nonlocal fired
            fired = True

        async def run():
            rv = ConnectionRendezvous(callback=callback)
            await rv.stt_ready()
            assert not fired

        asyncio.run(run())
