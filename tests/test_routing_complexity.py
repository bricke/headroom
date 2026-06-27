"""Unit tests for the complexity scorer and Phase 3 decide() logic.

Covers:
- score_complexity(): Anthropic, OpenAI, and Gemini format bodies
- Individual signal contributions (baseline + content)
- _score_last_message(): content-signal unit tests
- Regression: same message scores identically regardless of turn count
- decide() with routing_prefer="auto"
- Circuit open overrides complexity routing
- Threshold edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.backend_decision import BackendDecision, decide
from headroom.proxy.routing_complexity import _score_last_message, score_complexity
from headroom.proxy.routing_health import RoutingHealthProber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    routing_prefer: str = "auto",
    threshold: float = 0.5,
    routing_enabled: bool = True,
) -> MagicMock:
    cfg = MagicMock()
    cfg.routing_enabled = routing_enabled
    cfg.routing_prefer = routing_prefer
    cfg.routing_complexity_threshold = threshold
    return cfg


def make_backend() -> MagicMock:
    return MagicMock(name="anyllm-openai")


def make_prober(available: bool = True) -> RoutingHealthProber:
    prober = RoutingHealthProber(api_base="http://localhost:8080", interval_seconds=0)
    if not available:
        for _ in range(prober.state._threshold):
            prober.state.record_failure()
    return prober


def _anthropic_body(
    content: str = "hi",
    system: str = "",
    tools: int = 0,
    turns: int = 1,
    structured: bool = False,
) -> dict:
    messages = [{"role": "user", "content": content}]
    for _ in range(turns - 1):
        messages.insert(0, {"role": "assistant", "content": "ok"})
        messages.insert(0, {"role": "user", "content": "more"})
    body: dict = {"messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": f"tool_{i}"} for i in range(tools)]
    if structured:
        body["response_format"] = {"type": "json_schema"}
    return body


def _gemini_body(
    content: str = "hi",
    system: str = "",
    tools: int = 0,
    turns: int = 1,
    structured: bool = False,
) -> dict:
    contents = [{"role": "user", "parts": [{"text": content}]}]
    for _ in range(turns - 1):
        contents.insert(0, {"role": "model", "parts": [{"text": "ok"}]})
        contents.insert(0, {"role": "user", "parts": [{"text": "more"}]})
    body: dict = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [{"name": f"fn_{i}"}]} for i in range(tools)]
    if structured:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    return body


# ---------------------------------------------------------------------------
# _score_last_message — content signal unit tests
# ---------------------------------------------------------------------------


class TestScoreLastMessage:
    def test_empty_text_scores_zero(self) -> None:
        assert _score_last_message("") == 0.0
        assert _score_last_message("   ") == 0.0

    def test_greeting_scores_near_zero(self) -> None:
        for greeting in ("hi", "thanks", "thank you", "great", "ok", "sounds good"):
            score = _score_last_message(greeting)
            assert score < 0.05, f"expected low score for {greeting!r}, got {score}"

    def test_implement_keyword_scores_above_zero(self) -> None:
        score = _score_last_message("implement a login page")
        assert score > 0.0

    def test_code_block_raises_score(self) -> None:
        without = _score_last_message("fix this")
        with_code = _score_last_message("fix this\n```python\npass\n```")
        assert with_code > without

    def test_multiple_bullets_raises_score(self) -> None:
        plain = _score_last_message("do something")
        bulleted = _score_last_message(
            "do something\n- task one\n- task two\n- task three"
        )
        assert bulleted > plain

    def test_numbered_list_raises_score(self) -> None:
        plain = _score_last_message("do something")
        numbered = _score_last_message(
            "do something\n1. first step\n2. second step\n3. third step"
        )
        assert numbered > plain

    def test_multistep_markers_raise_score(self) -> None:
        plain = _score_last_message("implement a feature")
        multistep = _score_last_message(
            "implement a feature step-by-step, then write tests, after that add docs"
        )
        assert multistep > plain

    def test_long_message_raises_length_score(self) -> None:
        short = _score_last_message("hi")
        long = _score_last_message("x" * 1600)  # above _CAP_MSG_LENGTH of 1500
        assert long > short

    def test_low_kw_with_high_kw_not_zeroed(self) -> None:
        # "can you implement" has both a low-kw match and a high-kw match.
        # High-kw presence should prevent the zero override.
        score = _score_last_message("can you implement a caching layer?")
        assert score > 0.0

    def test_combined_complex_request_scores_high(self) -> None:
        text = (
            "Implement a full OAuth2 authentication system with JWT refresh tokens.\n"
            "- Add login and logout endpoints\n"
            "- Integrate with the existing user database\n"
            "- Write unit tests for each endpoint\n"
            "```python\n# example skeleton\npass\n```\n"
            "Do this step-by-step, then update the docs."
        )
        score = _score_last_message(text)
        assert score > 0.25

    def test_score_bounded_0_1(self) -> None:
        score = _score_last_message("x" * 100_000 + " implement " * 100)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Regression: content vs context length
# ---------------------------------------------------------------------------


class TestContentVsContext:
    def test_same_message_same_score_regardless_of_turn_count(self) -> None:
        """Core regression: 'implement X' should score the same at turn 1 and turn 20."""
        msg = "implement a caching layer"
        score_turn_1 = score_complexity(_anthropic_body(content=msg, turns=1))
        score_turn_20 = score_complexity(_anthropic_body(content=msg, turns=20))
        assert score_turn_1 == score_turn_20

    def test_greeting_at_high_tool_count_scores_below_threshold(self) -> None:
        """'thanks!' with 10 tools + structured output should still be below 0.5.

        Baseline max: tools=0.20 + structured=0.10 = 0.30. Content for 'thanks!' = 0.0.
        Total = 0.30 < 0.5 threshold → routes selfhosted.
        """
        body = _anthropic_body(content="thanks!", tools=10, structured=True)
        score = score_complexity(body)
        assert score < 0.5


# ---------------------------------------------------------------------------
# score_complexity — Anthropic/OpenAI format
# ---------------------------------------------------------------------------


class TestScoreComplexityAnthropicOpenAI:
    def test_trivial_request_scores_low(self) -> None:
        score = score_complexity(_anthropic_body("hi"))
        assert score < 0.15

    def test_score_increases_with_message_length(self) -> None:
        short = score_complexity(_anthropic_body("hi"))
        long = score_complexity(_anthropic_body("x" * 1600))  # above 1500 cap
        assert long > short

    def test_score_increases_with_tool_count(self) -> None:
        no_tools = score_complexity(_anthropic_body(tools=0))
        many_tools = score_complexity(_anthropic_body(tools=10))
        assert many_tools > no_tools

    def test_score_increases_with_system_prompt(self) -> None:
        no_sys = score_complexity(_anthropic_body(system=""))
        long_sys = score_complexity(_anthropic_body(system="x" * 3000))
        assert long_sys > no_sys

    def test_structured_output_adds_to_score(self) -> None:
        plain = score_complexity(_anthropic_body())
        structured = score_complexity(_anthropic_body(structured=True))
        assert structured > plain

    def test_openai_system_message_counted(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "x" * 3000},
                {"role": "user", "content": "hi"},
            ]
        }
        score = score_complexity(body)
        assert score > 0.1

    def test_openai_json_object_counts_as_structured(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        }
        score = score_complexity(body)
        plain = score_complexity({"messages": [{"role": "user", "content": "hi"}]})
        assert score > plain

    def test_score_bounded_0_1(self) -> None:
        saturated = _anthropic_body(
            content="x" * 50_000,
            system="y" * 5000,
            tools=20,
            turns=30,
            structured=True,
        )
        score = score_complexity(saturated)
        assert 0.0 <= score <= 1.0

    def test_empty_body_scores_zero(self) -> None:
        assert score_complexity({}) == 0.0


# ---------------------------------------------------------------------------
# score_complexity — Gemini format
# ---------------------------------------------------------------------------


class TestScoreComplexityGemini:
    def test_trivial_gemini_request_scores_low(self) -> None:
        score = score_complexity(_gemini_body("hi"))
        assert score < 0.15

    def test_gemini_long_content_scores_higher(self) -> None:
        short = score_complexity(_gemini_body("hi"))
        long = score_complexity(_gemini_body("x" * 2000))  # above 1500 cap
        assert long > short

    def test_gemini_system_instruction_counted(self) -> None:
        no_sys = score_complexity(_gemini_body(system=""))
        long_sys = score_complexity(_gemini_body(system="y" * 3000))
        assert long_sys > no_sys

    def test_gemini_structured_output_counted(self) -> None:
        plain = score_complexity(_gemini_body())
        structured = score_complexity(_gemini_body(structured=True))
        assert structured > plain

    def test_gemini_score_bounded_0_1(self) -> None:
        saturated = _gemini_body(
            content="x" * 50_000,
            system="y" * 5000,
            tools=20,
            turns=30,
            structured=True,
        )
        score = score_complexity(saturated)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# decide() — routing_prefer="auto"
# ---------------------------------------------------------------------------


class TestDecideAuto:
    def test_simple_request_routes_to_selfhosted(self) -> None:
        cfg = make_config(threshold=0.5)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body)
        assert decision.target == "selfhosted"
        assert decision.reason == "complexity_below_threshold"
        assert decision.complexity_score is not None
        assert decision.complexity_score < 0.5

    def test_complex_request_routes_to_frontier(self) -> None:
        cfg = make_config(threshold=0.1)
        body = _anthropic_body(content="x" * 20_000, tools=10, structured=True)
        decision = decide(cfg, make_backend(), body)
        assert decision.target == "frontier"
        assert decision.reason == "complexity_above_threshold"

    def test_complexity_score_included_in_decision(self) -> None:
        cfg = make_config(threshold=0.5)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body)
        assert isinstance(decision.complexity_score, float)

    def test_circuit_open_routes_to_frontier_regardless_of_complexity(self) -> None:
        cfg = make_config(threshold=0.9)
        prober = make_prober(available=False)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body, prober)
        assert decision.target == "frontier"
        assert decision.reason == "selfhosted_circuit_open"
        assert decision.selfhosted_available is False

    def test_circuit_closed_uses_complexity(self) -> None:
        cfg = make_config(threshold=0.5)
        prober = make_prober(available=True)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body, prober)
        assert decision.target == "selfhosted"

    def test_no_backend_always_frontier(self) -> None:
        cfg = make_config()
        decision = decide(cfg, None, _anthropic_body("hi"))
        assert decision.target == "frontier"
        assert decision.reason == "no_backend_configured"

    def test_routing_disabled_always_selfhosted(self) -> None:
        cfg = make_config(routing_enabled=False)
        decision = decide(cfg, make_backend(), _anthropic_body("hi"))
        assert decision.target == "selfhosted"

    def test_threshold_boundary_at_score(self) -> None:
        cfg = make_config(threshold=0.0)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body)
        assert decision.target == "frontier"

    def test_prefer_selfhosted_ignores_complexity(self) -> None:
        cfg = make_config(routing_prefer="selfhosted", threshold=0.1)
        body = _anthropic_body(content="x" * 50_000, tools=10, structured=True)
        decision = decide(cfg, make_backend(), body)
        assert decision.target == "selfhosted"
        assert decision.complexity_score is None

    def test_prefer_frontier_ignores_complexity(self) -> None:
        cfg = make_config(routing_prefer="frontier", threshold=0.9)
        body = _anthropic_body("hi")
        decision = decide(cfg, make_backend(), body)
        assert decision.target == "frontier"
        assert decision.complexity_score is None
