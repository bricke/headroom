"""Unit tests for Gemini handler routing helpers and routing gate (Phase 2+).

Tests:
- _gemini_body_to_openai_body(): generationConfig mapping
- _openai_response_to_gemini_response(): format conversion + finish reason mapping
- handle_gemini_generate_content routing gate (mocked selfhosted)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.proxy.handlers.gemini import GeminiHandlerMixin


# ---------------------------------------------------------------------------
# Minimal concrete subclass so we can instantiate GeminiHandlerMixin
# ---------------------------------------------------------------------------


class _FakeGeminiHandler(GeminiHandlerMixin):
    def __init__(self, config: Any = None, backend: Any = None) -> None:
        self.config = config or _make_config()
        self.anthropic_backend = backend


def _make_config(
    routing_enabled: bool = True,
    routing_prefer: str = "selfhosted",
    selfhosted_model: str | None = "qwen2.5-72b",
) -> MagicMock:
    cfg = MagicMock()
    cfg.routing_enabled = routing_enabled
    cfg.routing_prefer = routing_prefer
    cfg.routing_selfhosted_model = selfhosted_model
    return cfg


def _make_backend() -> MagicMock:
    return MagicMock(name="anyllm-openai")


# ---------------------------------------------------------------------------
# _gemini_body_to_openai_body
# ---------------------------------------------------------------------------


class TestGeminiBodyToOpenAI:
    def setup_method(self) -> None:
        self.handler = _FakeGeminiHandler()

    def test_model_rewritten_when_selfhosted_model_configured(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = self.handler._gemini_body_to_openai_body({}, messages)
        assert result["model"] == "qwen2.5-72b"

    def test_model_absent_when_selfhosted_model_not_configured(self) -> None:
        handler = _FakeGeminiHandler(config=_make_config(selfhosted_model=None))
        messages = [{"role": "user", "content": "hello"}]
        result = handler._gemini_body_to_openai_body({}, messages)
        assert "model" not in result

    def test_messages_passed_through(self) -> None:
        messages = [{"role": "user", "content": "ping"}]
        result = self.handler._gemini_body_to_openai_body({}, messages)
        assert result["messages"] is messages

    def test_generation_config_max_output_tokens(self) -> None:
        body = {"generationConfig": {"maxOutputTokens": 256}}
        result = self.handler._gemini_body_to_openai_body(body, [])
        assert result["max_tokens"] == 256

    def test_generation_config_temperature(self) -> None:
        body = {"generationConfig": {"temperature": 0.7}}
        result = self.handler._gemini_body_to_openai_body(body, [])
        assert result["temperature"] == 0.7

    def test_generation_config_top_p(self) -> None:
        body = {"generationConfig": {"topP": 0.9}}
        result = self.handler._gemini_body_to_openai_body(body, [])
        assert result["top_p"] == 0.9

    def test_generation_config_stop_sequences(self) -> None:
        body = {"generationConfig": {"stopSequences": ["<|end|>"]}}
        result = self.handler._gemini_body_to_openai_body(body, [])
        assert result["stop"] == ["<|end|>"]

    def test_missing_generation_config_fields_not_injected(self) -> None:
        result = self.handler._gemini_body_to_openai_body({}, [])
        for key in ("max_tokens", "temperature", "top_p", "stop"):
            assert key not in result


# ---------------------------------------------------------------------------
# _openai_response_to_gemini_response
# ---------------------------------------------------------------------------


class TestOpenAIResponseToGemini:
    def setup_method(self) -> None:
        self.handler = _FakeGeminiHandler()

    def _openai_resp(
        self,
        content: str = "Hello!",
        finish_reason: str = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> dict:
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def test_candidate_text_extracted(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp("Hello!"), "gemini-pro"
        )
        assert resp["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"

    def test_candidate_role_is_model(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(), "gemini-pro"
        )
        assert resp["candidates"][0]["content"]["role"] == "model"

    def test_finish_reason_stop_mapped(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(finish_reason="stop"), "gemini-pro"
        )
        assert resp["candidates"][0]["finishReason"] == "STOP"

    def test_finish_reason_length_mapped(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(finish_reason="length"), "gemini-pro"
        )
        assert resp["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_finish_reason_content_filter_mapped(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(finish_reason="content_filter"), "gemini-pro"
        )
        assert resp["candidates"][0]["finishReason"] == "SAFETY"

    def test_unknown_finish_reason_defaults_to_stop(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(finish_reason="whatever"), "gemini-pro"
        )
        assert resp["candidates"][0]["finishReason"] == "STOP"

    def test_usage_metadata_mapped(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(prompt_tokens=10, completion_tokens=5), "gemini-pro"
        )
        meta = resp["usageMetadata"]
        assert meta["promptTokenCount"] == 10
        assert meta["candidatesTokenCount"] == 5
        assert meta["totalTokenCount"] == 15

    def test_model_version_set(self) -> None:
        resp = self.handler._openai_response_to_gemini_response(
            self._openai_resp(), "gemini-1.5-pro"
        )
        assert resp["modelVersion"] == "gemini-1.5-pro"

    def test_empty_content_handled(self) -> None:
        openai_body = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
            ]
        }
        resp = self.handler._openai_response_to_gemini_response(openai_body, "gemini-pro")
        assert resp["candidates"][0]["content"]["parts"][0]["text"] == ""

    def test_no_usage_field(self) -> None:
        openai_body = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ]
        }
        resp = self.handler._openai_response_to_gemini_response(openai_body, "gemini-pro")
        assert "usageMetadata" not in resp
