"""
Simple Soniox TTS test — correct API format from docs.
Saves audio to test_tts.wav
"""
import asyncio
import os
import json
import base64
import wave
from pathlib import Path
from dotenv import load_dotenv
from websockets.asyncio.client import connect

load_dotenv()
API_KEY = os.getenv("SONIOX_API_KEY")
URL = "wss://tts-rt.soniox.com/tts-websocket"

async def test_tts():
    print(f"Connecting to {URL}...")
    async with connect(URL) as ws:
        print("Connected!")

        # Config message
        config = {
            "api_key": API_KEY,
            "stream_id": "test-stream",
            "model": "tts-rt-v1",
            "voice": "Emma",
            "language": "en",
            "audio_format": "pcm_s16le",
            "sample_rate": 24000,
        }
        print(f"Sending config: {json.dumps(config)}")
        await ws.send(json.dumps(config))

        # Text message (correct format: text + text_end + stream_id, no "type")
        text = "Hello world! This is a test of Soniox text to speech."
        text_msg = {
            "text": text,
            "text_end": True,
            "stream_id": "test-stream",
        }
        print(f"Sending text: {json.dumps(text_msg)}")
        await ws.send(json.dumps(text_msg))

        print("Receiving audio...")
        audio_chunks = []
        async for message in ws:
            data = json.loads(message)
            if "audio" in data:
                chunk = base64.b64decode(data["audio"])
                audio_chunks.append(chunk)
                done = data.get("audio_end", False)
                print(f"  Audio chunk: {len(chunk)} bytes {'(final)' if done else ''}")
                if done:
                    break
            elif "terminated" in data:
                print(f"Stream terminated")
                break
            elif "error_code" in data:
                print(f"ERROR: {data}")
                break
            else:
                print(f"  RAW: {data}")

        if audio_chunks:
            audio = b"".join(audio_chunks)
            output_path = Path(__file__).resolve().parent / "test_tts.wav"
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
            # 24kHz mono 16-bit = 48000 bytes/sec
            print(f"Saved {output_path} ({len(audio)} bytes, {len(audio)/48000:.1f}s)")

asyncio.run(test_tts())
