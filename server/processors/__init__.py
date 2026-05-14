"""Pipeline processors — FrameProcessor subclasses that sit in the pipeline.

Each module is a single FrameProcessor (or a tight cluster around one):
  - echo_suppressor.py — drops STT frames while the bot is speaking
  - wake_word.py       — wake-word gate (asleep/awake state machine)
  - session_log.py     — per-session JSONL logger + SessionLogProcessor
  - latency_tracer.py  — per-turn response-latency budget
"""

from processors.echo_suppressor import EchoSuppressor  # noqa: F401
from processors.latency_tracer import LatencyTracer  # noqa: F401
from processors.session_log import SessionLog, SessionLogProcessor  # noqa: F401
from processors.wake_word import WakeWordGate  # noqa: F401
