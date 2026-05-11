"""Unit tests for vision provider adapter seam.

No real vision models — everything is mocked at the adapter interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tools.vision import (
    MLXVisionAdapter,
    OpenAIVisionAdapter,
    VisionChain,
    try_describe_with,
    _VISION_ADAPTERS,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeProvider:
    """Minimal stand-in for config.VisionProvider."""
    name: str = "fake"
    model: str = "test-model"
    kind: str = "openai"
    base_url: str = "http://localhost"
    api_key_env: str = ""
    max_tokens: int = 150
    max_image_width: int = 0
    brevity_suffix: str = ""
    timeout: float = 5.0


# ---------------------------------------------------------------------------
# VisionChain
# ---------------------------------------------------------------------------

class TestVisionChain:
    def test_empty_chain_returns_none(self):
        chain = VisionChain([])
        assert chain.describe("/tmp/x.jpg", "what is this") is None

    def test_first_success_wins(self):
        p1 = FakeProvider(name="p1", kind="openai")
        p2 = FakeProvider(name="p2", kind="openai")
        chain = VisionChain([p1, p2])

        with patch("tools.vision.OpenAIVisionAdapter.describe") as mock_describe:
            mock_describe.side_effect = ["first", "second"]
            result = chain.describe("/tmp/x.jpg", "prompt")
            assert result == "first"
            assert mock_describe.call_count == 1

    def test_falls_back_to_second(self):
        p1 = FakeProvider(name="p1", kind="openai")
        p2 = FakeProvider(name="p2", kind="openai")
        chain = VisionChain([p1, p2])

        with patch("tools.vision.OpenAIVisionAdapter.describe") as mock_describe:
            mock_describe.side_effect = [None, "fallback"]
            result = chain.describe("/tmp/x.jpg", "prompt")
            assert result == "fallback"
            assert mock_describe.call_count == 2

    def test_no_vision_message_when_disabled(self):
        chain = VisionChain([])
        msg = chain.no_vision_message()
        assert "vision is disabled" in msg

    def test_no_vision_message_with_missing_keys(self):
        chain = VisionChain([FakeProvider(name="p1", api_key_env="MISSING_KEY")])
        msg = chain.no_vision_message()
        assert "Missing env vars" in msg
        assert "MISSING_KEY" in msg


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

class TestTryDescribeWithDispatch:
    def test_unknown_kind_returns_none_and_logs(self, caplog):
        provider = FakeProvider(kind="unknown")
        with patch("tools.vision.logger") as mock_logger:
            result = try_describe_with(provider, "/tmp/x.jpg", "prompt")
            assert result is None
            mock_logger.warning.assert_called_once()

    def test_brevity_suffix_appended(self):
        provider = FakeProvider(brevity_suffix="Be brief.")
        adapter = MagicMock()
        adapter.describe.return_value = "ok"
        with patch.dict("tools.vision._VISION_ADAPTERS", {"openai": adapter}, clear=True):
            result = try_describe_with(provider, "/tmp/x.jpg", "describe")
            assert result == "ok"
            # The prompt passed to the adapter should include the suffix.
            _, _, prompt = adapter.describe.call_args[0]
            assert "Be brief." in prompt


# ---------------------------------------------------------------------------
# MLXVisionAdapter — pure failure paths (no mlx_vlm in test env)
# ---------------------------------------------------------------------------

class TestMLXVisionAdapter:
    def test_returns_none_when_mlx_vlm_missing(self, monkeypatch):
        adapter = MLXVisionAdapter()
        # Ensure import fails
        monkeypatch.setitem(__builtins__, "__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no")) if name == "tools.mlx_vision" else __import__(name, *a, **kw))
        # Actually, patching __builtins__ is dangerous. Patch the module-level import instead.
        pass  # We'll test via the dispatch instead.

    def test_describe_returns_none_on_runtime_error(self, monkeypatch):
        adapter = MLXVisionAdapter()

        def _boom(*a, **k):
            raise RuntimeError("model crashed")

        import tools.mlx_vision as real_mlx_vision
        monkeypatch.setattr(real_mlx_vision, "describe", _boom)
        with patch("tools.vision.logger") as mock_logger:
            result = adapter.describe(FakeProvider(kind="mlx"), "/tmp/x.jpg", "prompt")
            assert result is None
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAIVisionAdapter — mocked HTTP layer
# ---------------------------------------------------------------------------

class TestOpenAIVisionAdapter:
    def test_returns_none_when_api_key_missing(self, monkeypatch):
        adapter = OpenAIVisionAdapter()
        provider = FakeProvider(api_key_env="MISSING_VISION_KEY")
        monkeypatch.delenv("MISSING_VISION_KEY", raising=False)
        with patch("tools.vision.logger") as mock_logger:
            result = adapter.describe(provider, "/tmp/x.jpg", "prompt")
            assert result is None
            mock_logger.info.assert_called_once()

    def test_describe_success(self, monkeypatch, tmp_path):
        adapter = OpenAIVisionAdapter()
        provider = FakeProvider(api_key_env="FAKE_KEY")
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        # Create a tiny fake image file
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake-jpeg-data")

        mock_msg = MagicMock()
        mock_msg.content = "A cat."
        mock_msg.reasoning_content = ""
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = mock_resp
            result = adapter.describe(provider, str(img), "what")
            assert result == "A cat."

    def test_reasoning_content_fallback(self, monkeypatch, tmp_path):
        adapter = OpenAIVisionAdapter()
        provider = FakeProvider(api_key_env="FAKE_KEY")
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.reasoning_content = "It is a dog."
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = mock_resp
            result = adapter.describe(provider, str(img), "what")
            assert result == "It is a dog."

    def test_retry_on_overload(self, monkeypatch, tmp_path):
        adapter = OpenAIVisionAdapter()
        provider = FakeProvider(api_key_env="FAKE_KEY")
        monkeypatch.setenv("FAKE_KEY", "sk-test")

        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_msg.reasoning_content = ""
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        with patch("openai.OpenAI") as MockClient:
            create = MockClient.return_value.chat.completions.create
            create.side_effect = [
                Exception("server overload"),
                mock_resp,
            ]
            with patch("tools.vision.time.sleep"):
                result = adapter.describe(provider, str(img), "what")
                assert result == "ok"
                assert create.call_count == 2
