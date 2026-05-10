"""Acoustic echo cancellation for `LocalAudioTransport` (mic + speakers).

Without AEC, the bot's TTS output bleeds into the mic and Soniox transcribes
it — which is why `local_bot.py` mutes the mic for every bot turn and runs
`EchoSuppressor`. That makes barge-in (interrupting the bot mid-sentence)
impossible: the user's voice never reaches STT while the bot is speaking.

This module solves it the way browsers/Daily/WebRTC do: subtract the bot's own
output from the mic input using Speex AEC. With AEC running, the mute strategy
and `EchoSuppressor` can be dropped, and Pipecat's built-in interruption
(triggered by `UserStartedSpeakingFrame` mid-bot-turn) takes over.

Two cooperating processors share state:

  * `AECReferenceTap`  — placed AFTER the TTS service. Pass-through that
    captures every `TTSAudioRawFrame` and pushes it (resampled to mic rate)
    into a shared ring buffer. This is the "far-end" reference signal.
  * `AECMicProcessor`  — placed RIGHT AFTER `transport.input()`. For each
    `InputAudioRawFrame`, pulls a same-length chunk of reference and runs the
    Speex echo canceller, replacing the audio bytes with the cleaned signal.

Sample rates: Pipecat defaults to 16 kHz mic in, 24 kHz TTS out. Speex EC
requires near and far at the same rate, so the reference tap resamples 24k → 16k
via `scipy.signal.resample_poly` (factor 2/3, kaiser window). Cheap and
adequate for AEC reference quality.

Caveats / known limitations:
  * Pure software AEC with no device-latency calibration. There will be a small
    fixed offset between when the speaker plays a TTS chunk and when the mic
    records its echo; Speex's adaptive filter is forgiving but not magic. If
    cancellation is poor, increase `filter_length_ms` (covers more echo tail)
    or add a small reference-signal delay.
  * Stereo TTS is downmixed to mono by averaging channels.
  * If a `TTSAudioRawFrame` arrives at a sample rate other than 24k, the
    resampler is rebuilt for that rate.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from loguru import logger
from scipy.signal import resample_poly
from speexdsp import EchoCanceller

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AECState:
    """Shared echo canceller + far-end reference ring buffer.

    One instance is shared between an `AECReferenceTap` and an
    `AECMicProcessor`. All access is guarded by a single lock — the pipeline
    runs in asyncio but Speex EC is not reentrant.
    """

    def __init__(
        self,
        *,
        mic_sample_rate: int = 16000,
        frame_ms: int = 20,
        filter_length_ms: int = 200,
    ):
        self.mic_sample_rate = mic_sample_rate
        self.frame_size = mic_sample_rate * frame_ms // 1000  # samples
        self.frame_bytes = self.frame_size * 2  # int16
        filter_length = mic_sample_rate * filter_length_ms // 1000
        self._ec = EchoCanceller.create(self.frame_size, filter_length, mic_sample_rate)
        self._lock = threading.Lock()
        # Ring buffer of far-end PCM bytes at mic_sample_rate, mono int16.
        # Cap at ~2 seconds — anything older is past the AEC tail.
        self._ref = bytearray()
        self._ref_cap = mic_sample_rate * 2 * 2
        self._tts_resample_from: int | None = None  # cached source rate

    def push_reference(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        """Append a TTS chunk to the far-end reference, resampling as needed."""
        # Downmix stereo → mono first.
        samples = np.frombuffer(pcm, dtype=np.int16)
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels).mean(axis=1).astype(np.int16)

        if sample_rate != self.mic_sample_rate:
            # resample_poly with a small lowpass filter; up=mic_rate, down=src_rate.
            # gcd-reduce to keep filter cheap.
            from math import gcd
            g = gcd(self.mic_sample_rate, sample_rate)
            up = self.mic_sample_rate // g
            down = sample_rate // g
            samples = resample_poly(samples.astype(np.float32), up, down)
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            self._tts_resample_from = sample_rate

        with self._lock:
            self._ref.extend(samples.tobytes())
            if len(self._ref) > self._ref_cap:
                # Drop the oldest, keep the most recent _ref_cap bytes.
                del self._ref[: len(self._ref) - self._ref_cap]

    def process(self, near_pcm: bytes) -> bytes:
        """Run the echo canceller on a mic chunk; pad reference with silence
        when the speaker isn't playing anything."""
        out = bytearray()
        with self._lock:
            for i in range(0, len(near_pcm), self.frame_bytes):
                near = near_pcm[i : i + self.frame_bytes]
                if len(near) < self.frame_bytes:
                    # Tail of an odd-sized frame — pass through unprocessed.
                    out.extend(near)
                    break
                if len(self._ref) >= self.frame_bytes:
                    far = bytes(self._ref[: self.frame_bytes])
                    del self._ref[: self.frame_bytes]
                else:
                    far = b"\x00" * self.frame_bytes
                out.extend(self._ec.process(near, far))
        return bytes(out)


class AECReferenceTap(FrameProcessor):
    """Capture TTS audio as the far-end reference. Pass-through on all frames."""

    def __init__(self, state: AECState):
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            try:
                self._state.push_reference(
                    frame.audio, frame.sample_rate, frame.num_channels
                )
            except Exception as e:  # never block the pipeline on AEC bookkeeping
                logger.warning(f"AEC reference push failed: {e}")
        await self.push_frame(frame, direction)


class AECMicProcessor(FrameProcessor):
    """Run echo cancellation on incoming mic frames before STT sees them."""

    def __init__(self, state: AECState):
        super().__init__()
        self._state = state
        self._warned_rate = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, InputAudioRawFrame)
            and frame.audio
            and direction == FrameDirection.DOWNSTREAM
        ):
            if frame.sample_rate != self._state.mic_sample_rate:
                if not self._warned_rate:
                    logger.warning(
                        f"AEC: mic sample_rate={frame.sample_rate} but state "
                        f"configured for {self._state.mic_sample_rate}; "
                        f"skipping AEC on this stream."
                    )
                    self._warned_rate = True
                await self.push_frame(frame, direction)
                return
            cleaned = self._state.process(frame.audio)
            new_frame = InputAudioRawFrame(
                audio=cleaned,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(new_frame, direction)
            return
        await self.push_frame(frame, direction)
