"""Smoke tests for build_llm / build_tts dispatching logic.

Validates the conditional branches added when wiring DeepSeek's cache shim
and the clause-streaming aggregator. The underlying clients don't hit the
network on construction, so fake API keys are safe.
"""

import os

import pytest

from config import Config, LLMConfig, STTConfig, TTSConfig
from voice_bot import build_llm, build_tts


def _make_config(**overrides) -> Config:
    """Minimal Config for builder smoke-tests."""
    defaults = dict(
        llm=LLMConfig(
            provider="deepseek",
            base_url="https://api.example.com/v1",
            model="fake-model",
            api_key_env="FAKE_LLM_KEY",
            extra={},
        ),
        stt=STTConfig(
            provider="soniox",
            api_key_env="FAKE_STT_KEY",
            language_hints=["en"],
        ),
        tts=TTSConfig(
            provider="soniox",
            api_key_env="FAKE_TTS_KEY",
            voice="Mina",
        ),
        vision=[],
        system_prompt="test",
        wake_word=__import__("config").WakeWordConfig(),
        shortcat=__import__("config").ShortcatConfig(),
        turn=__import__("config").TurnConfig(),
        hotkey=__import__("config").HotkeyConfig(),
        computer_use=__import__("config").ComputerUseConfig(),
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    monkeypatch.setenv("FAKE_LLM_KEY", "x")
    monkeypatch.setenv("FAKE_TTS_KEY", "x")
    monkeypatch.setenv("FAKE_STT_KEY", "x")


class TestBuildLLM:
    def test_deepseek_provider_yields_deepseek_subclass(self):
        from services.deepseek_llm import DeepSeekLLMService
        cfg = _make_config()
        llm = build_llm(cfg)
        assert isinstance(llm, DeepSeekLLMService)

    def test_deepseek_provider_is_case_insensitive(self):
        from services.deepseek_llm import DeepSeekLLMService
        cfg = _make_config(llm=LLMConfig(
            provider="DeepSeek", base_url="x", model="x",
            api_key_env="FAKE_LLM_KEY", extra={},
        ))
        assert isinstance(build_llm(cfg), DeepSeekLLMService)

    def test_other_provider_yields_base_openai(self):
        from pipecat.services.openai.llm import OpenAILLMService
        from services.deepseek_llm import DeepSeekLLMService
        cfg = _make_config(llm=LLMConfig(
            provider="openai", base_url="x", model="x",
            api_key_env="FAKE_LLM_KEY", extra={},
        ))
        llm = build_llm(cfg)
        assert isinstance(llm, OpenAILLMService)
        assert not isinstance(llm, DeepSeekLLMService)


class TestBuildTTS:
    def test_default_keeps_simple_aggregator(self):
        from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
        tts = build_tts(_make_config())
        assert isinstance(tts._text_aggregator, SimpleTextAggregator)

    def test_stream_clauses_swaps_aggregator(self):
        from processors.clause_aggregator import ClauseTextAggregator
        cfg = _make_config(tts=TTSConfig(
            provider="soniox", api_key_env="FAKE_TTS_KEY",
            voice="Mina", stream_clauses=True, clause_min_words=2,
        ))
        tts = build_tts(cfg)
        assert isinstance(tts._text_aggregator, ClauseTextAggregator)
        assert tts._text_aggregator._min_words == 2
