"""Unit tests for the complexity scorer and Phase 3 decide() logic.

Covers:
- score_complexity(): Anthropic, OpenAI, and Gemini format bodies
- Individual signal contributions
- decide() with routing_prefer="auto"
- Circuit open overrides complexity routing
- Threshold edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.backend_decision import BackendDecision, decide
from headroom.proxy.routing_complexity import score_complexity
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
# score_complexity — Anthropic/OpenAI format
# ---------------------------------------------------------------------------


class TestScoreComplexityAnthropicOpenAI:
    def test_trivial_request_scores_low(self) -> None:
        score = score_complexity(_anthropic_body("hi"))
        assert score < 0.15

    def test_score_increases_with_message_length(self) -> None:
        short = score_complexity(_anthropic_body("hi"))
        long = score_complexity(_anthropic_body("x" * 10_000))
        assert long > short

    def test_score_increases_with_tool_count(self) -> None:
        no_tools = score_complexity(_anthropic_body(tools=0))
        many_tools = score_complexity(_anthropic_body(tools=10))
        assert many_tools > no_tools

    def test_score_increases_with_system_prompt(self) -> None:
        no_sys = score_complexity(_anthropic_body(system=""))
        long_sys = score_complexity(_anthropic_body(system="x" * 3000))
        assert long_sys > no_sys

    def test_score_increases_with_turn_count(self) -> None:
        one_turn = score_complexity(_anthropic_body(turns=1))
        many_turns = score_complexity(_anthropic_body(turns=20))
        assert many_turns > one_turn

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
        long = score_complexity(_gemini_body("x" * 10_000))
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
        cfg = make_config(threshold=0.9)  # high threshold → would selfhost simple request
        prober = make_prober(available=False)
        body = _anthropic_body("hi")  # trivially simple
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
        # A request with score exactly at threshold should route to frontier (>=).
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
