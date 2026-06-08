"""Per-request complexity scorer for capability-aware routing.

Returns a score in [0.0, 1.0] representing how complex a request is.
When used with ``routing_prefer=auto``, requests whose score meets or
exceeds ``routing_complexity_threshold`` are routed to the frontier;
requests below the threshold go to the selfhosted endpoint.

Tuning the threshold to the local model's capability:
  - More capable model (e.g. Qwen-72B): raise the threshold (e.g. 0.7)
    → more requests stay local, fewer reach the frontier.
  - Less capable model (e.g. Qwen-9B): lower the threshold (e.g. 0.3)
    → more requests are routed to the frontier.

Signal weights are intentionally fixed. The threshold is the single
operator-facing knob (``HEADROOM_ROUTING_COMPLEXITY_THRESHOLD``).

Supports Anthropic Messages API format, OpenAI Chat Completions format,
and Gemini generateContent format — detected from body structure.
"""

from __future__ import annotations

from typing import Any

# Fixed weights; must sum to 1.0.
_W_TOKENS = 0.40
_W_TOOLS = 0.25
_W_SYSTEM = 0.15
_W_TURNS = 0.10
_W_STRUCTURED = 0.10

# Normalisation caps — score saturates at 1.0 above these.
_CAP_TOKENS = 8_000   # estimated tokens
_CAP_TOOLS = 10       # tool definitions
_CAP_SYSTEM = 2_000   # system prompt characters
_CAP_TURNS = 20       # non-system messages


def _estimate_tokens(text: str) -> int:
    """Rough tokens estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def _extract_text(value: Any) -> str:
    """Recursively extract all text from a content field (str, list of blocks, etc.)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return value["text"]
        return " ".join(_extract_text(v) for v in value.values() if isinstance(v, (str, list, dict)))
    return ""


def _score_anthropic_openai(body: dict) -> float:
    """Score an Anthropic Messages API or OpenAI Chat Completions body."""
    messages: list[dict] = body.get("messages", [])

    # System prompt: Anthropic top-level string, or OpenAI system role message.
    system_text = ""
    if isinstance(body.get("system"), str):
        system_text = body["system"]
    elif isinstance(body.get("system"), list):
        system_text = _extract_text(body["system"])
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text += " " + _extract_text(msg.get("content", ""))
        else:
            non_system.append(msg)

    messages_text = " ".join(_extract_text(m.get("content", "")) for m in non_system)

    structured = 0.0
    resp_format = body.get("response_format", {})
    if isinstance(resp_format, dict) and resp_format.get("type") in ("json_schema", "json_object"):
        structured = 1.0

    return _combine(
        messages_text=messages_text,
        system_text=system_text,
        tools=body.get("tools", []),
        turn_count=len(non_system),
        structured=structured,
    )


def _score_gemini(body: dict) -> float:
    """Score a Gemini generateContent body."""
    contents: list[dict] = body.get("contents", [])
    messages_text = " ".join(
        _extract_text(c.get("parts", [])) for c in contents
    )

    system_instruction = body.get("systemInstruction", {})
    system_text = _extract_text(system_instruction.get("parts", []))

    gen_config: dict = body.get("generationConfig", {})
    structured = 1.0 if (
        gen_config.get("responseSchema")
        or gen_config.get("responseMimeType") == "application/json"
    ) else 0.0

    tools_list: list = body.get("tools", [])

    return _combine(
        messages_text=messages_text,
        system_text=system_text,
        tools=tools_list,
        turn_count=len(contents),
        structured=structured,
    )


def _combine(
    *,
    messages_text: str,
    system_text: str,
    tools: list,
    turn_count: int,
    structured: float,
) -> float:
    token_score = min(_estimate_tokens(messages_text) / _CAP_TOKENS, 1.0)
    tool_score = min(len(tools) / _CAP_TOOLS, 1.0)
    system_score = min(len(system_text) / _CAP_SYSTEM, 1.0)
    turn_score = min(turn_count / _CAP_TURNS, 1.0)

    return (
        _W_TOKENS * token_score
        + _W_TOOLS * tool_score
        + _W_SYSTEM * system_score
        + _W_TURNS * turn_score
        + _W_STRUCTURED * structured
    )


def score_complexity(body: dict) -> float:
    """Return a complexity score in [0.0, 1.0] for the given request body.

    Handles Anthropic, OpenAI, and Gemini wire formats automatically.
    A score of 0.0 is a trivial single-turn request with no tools.
    A score of 1.0 is a saturated long multi-turn request with many tools
    and structured output demanded.
    """
    if "contents" in body:
        return _score_gemini(body)
    return _score_anthropic_openai(body)
