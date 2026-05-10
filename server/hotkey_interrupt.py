"""Global hotkey to interrupt the bot mid-sentence.

`LocalAudioTransport` has no AEC, so the mic is muted while the bot is speaking
(see `local_bot.py`). That makes voice barge-in impossible — but a global
hotkey works fine. Pressing the hotkey from anywhere on the desktop pushes an
`InterruptionTaskFrame` into the running pipeline, which Pipecat converts into
a downstream `InterruptionFrame` that cancels the in-flight LLM and TTS.

The first press in a fresh terminal will prompt for macOS Accessibility
permission (System Settings → Privacy & Security → Accessibility). pynput
silently does nothing until that's granted.

Default hotkey: ⌘⇧I. Override with `HOTKEY_INTERRUPT` env var using pynput
syntax, e.g. `HOTKEY_INTERRUPT='<ctrl>+<alt>+i'`.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger
from pynput import keyboard

from pipecat.frames.frames import InterruptionTaskFrame
from pipecat.pipeline.task import PipelineTask


def install_interrupt_hotkey(
    task: PipelineTask, *, hotkey: str | None = None
) -> keyboard.GlobalHotKeys:
    """Start a background pynput listener that interrupts `task` on hotkey.

    Returns the listener so the caller can stop it on shutdown if desired.
    """
    combo = hotkey or os.getenv("HOTKEY_INTERRUPT", "<cmd>+<shift>+i")
    loop = asyncio.get_running_loop()

    def on_pressed():
        # Listener fires on a pynput thread; bounce into the asyncio loop
        # because PipelineTask.queue_frame is not thread-safe.
        logger.info(f"⌨️  Hotkey {combo} — interrupting bot")
        asyncio.run_coroutine_threadsafe(task.queue_frame(InterruptionTaskFrame()), loop)

    # Workaround for pynput on Python 3.14 / macOS: the Darwin backend calls
    # on_press/on_release with a single `key` arg, but GlobalHotKeys expects
    # `(key, injected)`. Make `injected` optional so the listener doesn't
    # crash on every keystroke.
    class _CompatGlobalHotKeys(keyboard.GlobalHotKeys):
        def _on_press(self, key, injected=False):
            return super()._on_press(key, injected)

        def _on_release(self, key, injected=False):
            return super()._on_release(key, injected)

    listener = _CompatGlobalHotKeys({combo: on_pressed})
    listener.start()
    logger.info(
        f"⌨️  Interrupt hotkey active: {combo} "
        f"(grant Accessibility permission if first run)"
    )
    return listener
