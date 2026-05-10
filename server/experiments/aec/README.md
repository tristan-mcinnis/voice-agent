# AEC (Acoustic Echo Cancellation) Experiment

**Status:** Experimental — not yet reliable enough for the default pipeline.

## What it does

Replaces the mute-strategy approach (`AlwaysUserMuteStrategy` + `EchoSuppressor`)
with Speex software AEC. When AEC is running, the mic stays open during bot turns
and `UserStartedSpeakingFrame` triggers Pipecat's intrinsic interruption — true
voice barge-in.

## How it works

Two cooperating processors share an `AECState`:

- `AECReferenceTap` — captures TTS output as the far-end reference signal
- `AECMicProcessor` — subtracts reference from mic input before STT sees it

The reference is resampled from TTS rate (24 kHz) to mic rate (16 kHz) via
`scipy.signal.resample_poly`.

## Known issues

- No device-latency calibration — there's a fixed offset between speaker
  playback and mic recording that varies by hardware.
- Speex adaptive filter is forgiving but not magic. Cancellation quality
  varies across devices.
- Requires `speexdsp` (with Apple-Silicon build workaround) and `scipy`.

## Entry point

```bash
python experiments/aec/local_bot_aec.py
```

## Original location

Moved from `server/aec_processor.py` and `server/local_bot_aec.py` during the
2025-05-10 architecture deepening.
