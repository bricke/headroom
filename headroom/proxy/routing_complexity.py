"""Per-request complexity scorer for capability-aware routing.

Returns a score in [0.0, 1.0] representing how complex a request is.
When used with ``routing_prefer=auto``, requests whose score meets or
exceeds ``routing_complexity_threshold`` are routed to the frontier;
requests below the threshold go to the selfhosted endpoint.

The score has two layers:

**Baseline** (fixed per-request context signals, total weight 0.40):
  - Tool definition count       weight 0.20, cap 10
  - System prompt char length   weight 0.10, cap 2000
  - Structured output requested weight 0.10, boolean

**Content** (last user message only, total weight 0.60):
  - Message character length    weight 0.15, cap 1500
  - Code block presence         weight 0.10, cap 3 backtick-triples
  - Task count                  weight 0.15, cap 5
  - Complexity keywords         weight 0.10, cap 4 matches
  - Multi-step markers          weight 0.10, cap 3 matches

Scoring the *last* user message instead of cumulative context means that
a simple "thanks!" at turn 20 still routes to the selfhosted model, while
"implement OAuth2 with JWT refresh tokens" routes to frontier on turn 1.

Threshold tuning guidance:
  - More capable local model (e.g. 70B): raise the threshold (e.g. 0.7).
  - Less capable local model (e.g. 9B): lower the threshold (e.g. 0.3).

Supports Anthropic Messages API format, OpenAI Chat Completions format,
and Gemini generateContent format — detected from body structure.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Baseline weights and caps
# ---------------------------------------------------------------------------
_W_TOOLS = 0.20
_W_SYSTEM = 0.10
_W_STRUCTURED = 0.10

_CAP_TOOLS = 10
_CAP_SYSTEM = 2_000

# ---------------------------------------------------------------------------
# Content weights and caps (last user message only)
# ---------------------------------------------------------------------------
_W_MSG_LENGTH = 0.15
_W_CODE_BLOCKS = 0.10
_W_TASK_COUNT = 0.15
_W_COMPLEXITY_KW = 0.10
_W_MULTISTEP = 0.10

_CAP_MSG_LENGTH = 1_500
_CAP_CODE_BLOCKS = 3      # backtick-triple occurrences (2 per fenced block)
_CAP_TASKS = 5
_CAP_HIGH_KW = 4
_CAP_MULTISTEP = 3

assert (
    abs(
        _W_TOOLS + _W_SYSTEM + _W_STRUCTURED
        + _W_MSG_LENGTH + _W_CODE_BLOCKS + _W_TASK_COUNT
        + _W_COMPLEXITY_KW + _W_MULTISTEP
        - 1.0
    )
    < 1e-9
), "Complexity scorer weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Compiled regex patterns (stdlib re only)
# ---------------------------------------------------------------------------
_RE_CODE_BLOCK = re.compile(r"```")

_RE_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_RE_ADDITIVE = re.compile(
    r"\b(?:also|additionally|furthermore|and\s+(?:then|also|make\s+sure|ensure|please))\b",
    re.IGNORECASE,
)

_RE_HIGH_KW = re.compile(
    r"\b(?:implement|refactor|architect|design|migrate|integrate|scaffold|"
    r"optimize|debug|diagnose|build|create|develop|rewrite|extend|"
    r"add|generate|write|fix|update|modify|convert|extract|parse)\b",
    re.IGNORECASE,
)
_RE_LOW_KW = re.compile(
    r"\b(?:hi|hello|thanks|thank\s+you|thx|ok|okay|got\s+it|sounds\s+good|"
    r"great|perfect|sure|yes|no|what\s+is|what's|explain|describe|"
    r"tell\s+me\s+about|how\s+does|can\s+you\s+clarify)\b",
    re.IGNORECASE,
)

_RE_MULTISTEP = re.compile(
    r"\b(?:step[\s-]by[\s-]step|after\s+that|finally\b|followed\s+by|"
    r"once\s+you've|when\s+(?:that's|it's)\s+done|"
    r"first\b[^\n]{0,200}\bthen\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(value: Any) -> str:
    """Recursively extract all text from a content field (str, list of blocks, etc.)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return value["text"]
        return " ".join(
            _extract_text(v)
            for v in value.values()
            if isinstance(v, (str, list, dict))
        )
    return ""


def _last_user_text_anthropic_openai(messages: list[dict]) -> str:
    """Return the text of the last user-role message in an Anthropic/OpenAI body."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text(msg.get("content", ""))
    return ""


def _last_user_text_gemini(contents: list[dict]) -> str:
    """Return the text of the last user-role entry in a Gemini contents list."""
    for item in reversed(contents):
        if item.get("role") == "user":
            return _extract_text(item.get("parts", []))
    return ""


# ---------------------------------------------------------------------------
# Content scorer (last user message only)
# ---------------------------------------------------------------------------

def _score_last_message(text: str) -> float:
    """Score the complexity of a single message. Returns a weighted [0.0, 1.0] sum."""
    if not text.strip():
        return 0.0

    length_score = min(len(text) / _CAP_MSG_LENGTH, 1.0)

    code_score = min(len(_RE_CODE_BLOCK.findall(text)) / _CAP_CODE_BLOCKS, 1.0)

    task_count = (
        len(_RE_BULLET.findall(text))
        + len(_RE_NUMBERED.findall(text))
        + len(_RE_ADDITIVE.findall(text))
    )
    task_score = min(task_count / _CAP_TASKS, 1.0)

    high_kw_count = len(_RE_HIGH_KW.findall(text))
    low_kw_match = bool(_RE_LOW_KW.search(text))
    if low_kw_match and high_kw_count == 0:
        kw_score = 0.0
    else:
        kw_score = min(high_kw_count / _CAP_HIGH_KW, 1.0)

    multistep_score = min(len(_RE_MULTISTEP.findall(text)) / _CAP_MULTISTEP, 1.0)

    return (
        _W_MSG_LENGTH * length_score
        + _W_CODE_BLOCKS * code_score
        + _W_TASK_COUNT * task_score
        + _W_COMPLEXITY_KW * kw_score
        + _W_MULTISTEP * multistep_score
    )


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------

def _combine(
    *,
    last_user_text: str,
    system_text: str,
    tools: list,
    structured: float,
) -> float:
    tool_score = min(len(tools) / _CAP_TOOLS, 1.0)
    system_score = min(len(system_text) / _CAP_SYSTEM, 1.0)
    content_score = _score_last_message(last_user_text)

    return (
        _W_TOOLS * tool_score
        + _W_SYSTEM * system_score
        + _W_STRUCTURED * structured
        + content_score
    )


def _score_anthropic_openai(body: dict) -> float:
    """Score an Anthropic Messages API or OpenAI Chat Completions body."""
    messages: list[dict] = body.get("messages", [])

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

    structured = 0.0
    resp_format = body.get("response_format", {})
    if isinstance(resp_format, dict) and resp_format.get("type") in (
        "json_schema",
        "json_object",
    ):
        structured = 1.0

    return _combine(
        last_user_text=_last_user_text_anthropic_openai(non_system),
        system_text=system_text,
        tools=body.get("tools", []),
        structured=structured,
    )


def _score_gemini(body: dict) -> float:
    """Score a Gemini generateContent body."""
    contents: list[dict] = body.get("contents", [])

    system_instruction = body.get("systemInstruction", {})
    system_text = _extract_text(system_instruction.get("parts", []))

    gen_config: dict = body.get("generationConfig", {})
    structured = 1.0 if (
        gen_config.get("responseSchema")
        or gen_config.get("responseMimeType") == "application/json"
    ) else 0.0

    tools_list: list = body.get("tools", [])

    return _combine(
        last_user_text=_last_user_text_gemini(contents),
        system_text=system_text,
        tools=tools_list,
        structured=structured,
    )


def score_complexity(body: dict) -> float:
    """Return a complexity score in [0.0, 1.0] for the given request body.

    Handles Anthropic, OpenAI, and Gemini wire formats automatically.
    A score of 0.0 is a trivial single-turn request with no tools.
    A score of 1.0 is a saturated request with many tools, a long system
    prompt, structured output, and a complex multi-step user message.
    """
    if "contents" in body:
        return _score_gemini(body)
    return _score_anthropic_openai(body)
