"""Soniox STT smoke test — streams test_tts.wav and prints transcripts.

Soniox expects raw binary audio frames after the JSON config (NOT base64-in-JSON),
and an empty string to signal end-of-stream. See:
https://soniox.com/docs/stt/rt/real-time-transcription
"""
import asyncio
import json
import os
import wave

from dotenv import load_dotenv
from websockets.asyncio.client import connect

load_dotenv()
API_KEY = os.getenv("SONIOX_API_KEY")
URL = "wss://stt-rt.soniox.com/transcribe-websocket"
AUDIO_FILE = "test_tts.wav"


async def test_stt():
    if not os.path.exists(AUDIO_FILE):
        print(f"File not found: {AUDIO_FILE}. Run test_tts.py first.")
        return

    with wave.open(AUDIO_FILE, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    print(f"Connecting to {URL}...")
    async with connect(URL) as ws:
        print("Connected!")

        await ws.send(json.dumps({
            "api_key": API_KEY,
            "model": "stt-rt-preview",
            "audio_format": "pcm_s16le",
            "sample_rate": rate,
            "num_channels": 1,
            "language_hints": ["en"],
        }))
        print("Config sent, streaming audio...")

        # Stream audio as raw binary at ~real-time pacing.
        # 3840 bytes = 80ms at 24kHz mono 16-bit; matches Soniox example.
        chunk_size = 3840
        for i in range(0, len(audio_data), chunk_size):
            await ws.send(audio_data[i:i + chunk_size])
            await asyncio.sleep(0.08)

        # Empty string = end of audio
        await ws.send("")
        print("End-of-stream sent, waiting for final transcripts...")

        async for message in ws:
            data = json.loads(message)
            if "tokens" in data:
                text = "".join(t.get("text", "") for t in data["tokens"])
                final = "✓" if data.get("finished") else "…"
                if text.strip():
                    print(f"  {final} {text}")
                if data.get("finished"):
                    break
            elif "error_code" in data:
                print(f"ERROR: {data}")
                break


asyncio.run(test_stt())
