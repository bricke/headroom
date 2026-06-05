# Design: Capability-Aware Backend Routing (Local/Remote ↔ Frontier)

**Status:** Phase 2 complete and committed
**Target:** `bricke/headroom` fork — branch `feat/backend-routing`
**Author:** Matteo Brichese
**Date:** 2026-06-04 (updated 2026-06-05)

---

## 1. Summary

Add a per-request routing layer to the Headroom proxy that decides, for each
incoming request, whether to serve it from a **capable self-hosted model**
(company-hosted or LAN-hosted, served via llama.cpp / Ollama) or from a
**frontier API** (paid subscription / API key).

The goal is **token / cost optimization**: anything a capable self-hosted model
can handle well is served there at zero marginal token cost; only requests that
genuinely need frontier capability — or that arrive while the self-hosted model
is unreachable — escalate to the paid frontier provider.

This rides on infrastructure that **already exists** in Headroom. The only
genuinely new pieces are (a) a routing *decision*, (b) the ability to hold
*more than one* backend at once, and (c) the configuration to describe them.

---

## 2. Goals and non-goals

### Goals
- Route each request to self-hosted vs frontier based on a cheap, local decision.
- Preserve Headroom's existing compression / memory / telemetry pipeline on
  whichever backend is chosen.
- Make the self-hosted endpoint a plain configurable URL so transport
  (SSH tunnel, VPN, LAN) is an ops concern, not an app concern.
- Degrade gracefully: if the self-hosted model is unreachable, fall back to
  frontier (or vice versa) without failing the request.

### Non-goals
- **Not building an agent.** This is a routing proxy feature, nothing more.
- **Not a privacy/redaction feature.** Privacy is explicitly *not* the driver.
  (A data-governance constraint is noted in §9 as a possible future overlay,
  not part of this work.)
- Not replacing Headroom's compression logic or its existing backend libraries.

---

## 3. Scenarios

The feature is designed around one **main scenario** with two sub-variants. All
three are the *same mechanism* — only the endpoint address and the routing
threshold differ.

### 3.1 Main scenario — Company-provided authenticated remote endpoint
A company exposes a large, capable model (e.g. **Qwen3.5-70B**, served behind the
scenes via llama.cpp / Ollama or similar) as an **authenticated, OpenAI-compatible
API endpoint**. The developer is given three things: a **URL** (`base_url`), an
**API key**, and a **model name**. The developer has free access to it.

This is the same connection shape as the frontier tier — `{url, key, model}` —
just pointed at a free/company-provided model instead of a paid one. There is no
manual SSH tunnel; reaching the URL may still require being on the corporate VPN,
but that is a network precondition, not app-managed transport.

Implications that shape the design:
- The company tier is both **high-capability** (close to frontier on most
  tasks) and **zero marginal cost** → route the large majority of traffic there;
  the frontier subscription becomes the *exception*, not the default.
- **Availability is still a key routing signal**, not complexity: the endpoint can
  be unreachable when off-VPN / off-network, and may be rate-limited. A
  health/reachability check matters more than the complexity classifier here.
- Realistic policy: *company endpoint reachable → use it for almost everything;
  unreachable → fall back to frontier.*

> **Terminology:** the doc says "self-hosted" for the non-frontier tier, but in
> this main scenario it is really a *company-provided authenticated endpoint*. The
> mechanism is identical — both tiers are `{url, key, model}` — so "self-hosted"
> below should be read as "the company/local capable-model tier."

### 3.2 Sub-variant A — LAN-hosted capable model
Same as the main scenario, but the 35B/70B model lives on a box on the local
network instead of behind a tunnel/VPN. Differences: near-always-up, lower
latency, no tunnel to manage. The config differs only in the `api_base` URL and a
more permissive availability assumption.

### 3.3 Sub-variant B — Small local model (original idea)
A small local model (e.g. 7–9B) on the same machine or LAN. Here the **complexity
classifier becomes the primary signal again** (the small model genuinely can't
handle hard queries), and the frontier is escalated to more often. Same
mechanism, threshold tuned lower (escalate more aggressively).

---

## 4. How Headroom works today (relevant findings)

From reading the fork (`headroom/`):

- **Backends are already abstracted.** `backends/base.py` defines the `Backend`
  ABC; all backends normalize to the canonical Anthropic Messages format. Methods:
  `name`, `send_message`, `stream_message`, `send_openai_message`,
  `stream_openai_message`, `map_model_id`, `supports_model`, `close`.
- **The self-hosted destination already exists.** `backends/anyllm.py`
  (`AnyLLMBackend`) speaks to "38+ providers (OpenAI, Mistral, Groq, **Ollama**,
  Bedrock, …)"; `backends/litellm.py` covers 100+; `providers/openai_compatible.py`
  exists. A llama.cpp/Ollama endpoint is reachable today with zero new backend code.
- **The proxy is single-backend, chosen once at startup.** In
  `proxy/server.py:524`, `self.anthropic_backend` is built from the scalar
  `config.backend` (default `"anthropic"`, via `--backend` / `HEADROOM_BACKEND`)
  through `create_proxy_backend()` (`providers/registry.py:135`). **This single
  assumption is the core thing this feature must widen.**
- **The backend is consumed at ~12 call sites** across `proxy/handlers/streaming.py`,
  `proxy/handlers/anthropic.py`, `proxy/handlers/openai.py` (e.g.
  `self.anthropic_backend.stream_message(...)`, `.send_message(...)`,
  `.send_openai_message(...)`). → editing every site is brittle; prefer a wrapper.
- **There is an established per-request "decision" pattern.** `proxy/`
  contains `compression_decision.py`, `image_compression_decision.py`,
  `memory_decision.py`. `CompressionDecision` is a frozen dataclass built via a
  `.decide()` factory that also surfaces the constituent booleans for dashboard
  observability. Our routing decision should mirror this exactly.

---

## 5. Proposed design

### 5.1 Core approach — decision-aware handler branch (NOT a wrapper)

**Revised after reading the code (supersedes the earlier wrapper idea).** The
frontier (Anthropic) tier is **not** a `Backend` object: `create_proxy_backend`
returns `None` for `backend == "anthropic"`, and the handlers treat
`self.anthropic_backend is None` as "use the **native Anthropic passthrough**,"
which carries Headroom's prompt-caching / prefix-freeze optimizations. Routing
frontier traffic through a `Backend` wrapper would lose those — defeating the
point of building on Headroom. **Decision: frontier stays on the native path.**

So instead of a wrapper, we make the *existing* dispatch branch decision-aware.
Today each handler does:

```python
if self.anthropic_backend is not None:   # global: backend handles everything
    ... use backend ...
else:
    ... native Anthropic passthrough ...
```

We gate that condition per request:

```python
if self._use_selfhosted_backend(body, headers, model):
    ... use backend (= company/local capable model) ...
else:
    ... native Anthropic passthrough (= frontier) ...   # caching intact
```

- `self.anthropic_backend` becomes the **self-hosted/company tier** (an
  `AnyLLMBackend`/OpenAI-compatible backend pointed at the company URL+key+model).
- The **frontier tier is the unchanged native else-branch.**
- `_use_selfhosted_backend(...)` returns `True` unconditionally when routing is
  **disabled** (legacy behavior: configured backend handles everything), and the
  routing decision (`selfhosted`/`frontier`) when **enabled**.
- When routing to self-hosted, the incoming `body["model"]` (a `claude-*` id) is
  rewritten to the configured company model name before dispatch.

Dispatch branch points to gate (3 files): `proxy/handlers/anthropic.py:1663`,
`proxy/handlers/openai.py:1927`, and the `_stream_response_bedrock` caller in
`proxy/handlers/streaming.py`. The decision logic itself lives in one place
(`_use_selfhosted_backend` + `proxy/backend_decision.py`).

### 5.2 The routing decision — `proxy/backend_decision.py`

A new module mirroring `compression_decision.py`:

```python
@dataclass(frozen=True)
class BackendDecision:
    target: Literal["selfhosted", "frontier"]
    reason: str                      # "selfhosted_unreachable" | "complexity_high" | ...
    complexity_score: float | None   # observability
    selfhosted_healthy: bool
    forced_by: str | None            # header/model-alias override, if any

    @classmethod
    def decide(cls, *, request, config, health) -> "BackendDecision": ...
```

Decision order (main/remote scenario):
1. **Explicit override** — a request header or model alias forces a target
   (e.g. `x-headroom-route: frontier`). Highest priority.
2. **Availability gate** — if the chosen-preferred backend is unhealthy, route to
   the other (the dominant signal in the remote scenario).
3. **Complexity classifier** — score the request; above `threshold` → frontier,
   else self-hosted. (Primary signal in sub-variant B; secondary in 3.1/3.2.)

Streaming note: because responses stream, an answer-grading *cascade* (run local,
judge output, then escalate) is impractical here. **This feature uses upfront
routing** — decide before dispatch. A cascade is a possible later extension for
non-streaming paths only.

### 5.3 Complexity classifier

Start with a **heuristic scorer** (OpenJarvis's `complexity.py` is a good model):
a 0.0–1.0 score from cheap signals — length, code/math regex, multi-step/reasoning
markers, question/sub-task count. Deterministic, local, far cheaper than the call
it gates. Leave room to swap in a learned classifier later (trained on Headroom's
own request/outcome traces, which the telemetry already records).

### 5.4 Health / availability check

A lightweight async reachability probe for the self-hosted backend (cheap HTTP
ping / `/health` or a cached last-success timestamp with a short TTL), consulted
by `BackendDecision.decide`. This is what makes the tunnel/VPN scenario robust.
Must never block the request path for long — fail toward the configured fallback.

---

## 6. Configuration

The central change behind §1: **`config.backend` is scalar today; we need to
describe two backends plus a routing policy.** Proposed additive config (new
`RoutingConfig`, backward-compatible — absent/disabled means today's behavior):

```toml
[routing]
enabled = true
prefer  = "selfhosted"          # default target when both healthy & query is easy
threshold = 0.5                 # complexity score above which -> frontier
fallback_when_unreachable = true

[routing.selfhosted]
# Main scenario: company-provided authenticated endpoint = {url, key, model}.
backend  = "anyllm"             # or openai_compatible / litellm — all take base_url+key+model
provider = "openai"             # OpenAI-compatible; use "ollama" for raw LAN Ollama (sub-variant)
api_base = "https://llm.company.example/v1"   # the company-provided URL
api_key  = "${COMPANY_LLM_API_KEY}"           # the company-provided key (env-expanded)
model    = "qwen3.5-70b"                       # the company-provided model name
health_path = "/v1/models"      # reachability probe (or /api/tags for raw Ollama)
timeout_ms = 2000

[routing.frontier]
backend  = "anthropic"          # existing default path
model    = "claude-..."         # subscription / API
# credentials via existing env (ANTHROPIC_API_KEY, etc.)
```

Both tiers are connection-symmetric (`{url, key, model}`); only the model's
capability/cost and the routing threshold differ.

Notes:
- **Main scenario:** point `api_base`/`api_key`/`model` at the company-provided
  values. Reaching the URL may require corporate VPN — a network precondition the
  app does not manage.
- **Raw-Ollama sub-variants** (LAN box, or SSH-tunnelled host): set
  `provider = "ollama"`, `api_base` to the LAN IP (or `localhost` when tunnelled),
  and leave `api_key` empty. Transport (tunnel/VPN) is an OS-level concern; the app
  only sees `api_base`.
- Per-scenario tuning is *just config*: company/LAN capable model → high `threshold`,
  `prefer = selfhosted`; small local model → low `threshold`.
- Keep the existing single-`--backend` path working when `[routing].enabled` is
  false — this feature is opt-in.

---

## 7. Where it integrates (file-level)

| Concern | File | Change |
|---|---|---|
| Hold two backends, build `RoutingBackend` | `proxy/server.py` (~524) | call `create_proxy_backend` twice; wrap in `RoutingBackend` |
| Backend wrapper | `backends/routing.py` *(new)* | implements `Backend` ABC, delegates per decision |
| Decision value + factory | `proxy/backend_decision.py` *(new)* | mirror `CompressionDecision` |
| Complexity scorer | `proxy/` or `relevance/` *(new helper)* | heuristic score 0–1 |
| Health probe | `backends/routing.py` or small helper | async reachability w/ TTL cache |
| Config schema | `config.py` | add `RoutingConfig` + sub-configs |
| CLI/env | `cli.py`, `proxy/server.py` (~2971, ~3229) | flags/env to enable + point endpoints |
| Observability | existing dashboard/telemetry | surface `BackendDecision` fields (target, reason, score) |

Handlers (`proxy/handlers/*.py`) ideally need **no change** — they keep calling
`self.anthropic_backend`, which is now a `RoutingBackend`.

---

## 8. Plan of action (phased)

**Phase 0 — Confirm seams (½ day)**
- Read `create_proxy_backend` (`providers/registry.py:135`) signature & return.
- Confirm the 4 dispatch methods are the only backend entry points used by handlers.
- Confirm `AnyLLMBackend` reaches an Ollama/llama.cpp endpoint with a manual smoke test.

**Phase 1 — Two backends, no routing yet (1 day)**
- Add `RoutingConfig` to `config.py` (disabled by default).
- In `server.py`, when enabled, build both `selfhosted` and `frontier` backends.
- Add `RoutingBackend` that delegates **always to `prefer`** (no decision yet).
- Verify the existing pipeline still works end-to-end through the wrapper.

**Phase 2 — Availability routing (1 day)**
- Implement the health probe + TTL cache (authenticated GET against `health_path`,
  e.g. `/v1/models`, sending the company key).
- `RoutingBackend` routes to `prefer` when healthy, else fallback.
- **Treat "unavailable" as more than unreachable.** The company endpoint is an
  authenticated, possibly rate-limited API, so trip the fallback on:
  unreachable / timeout, **HTTP 429 (rate-limited)**, and 5xx. On a 429, mark the
  tier temporarily unavailable (short cooldown / honor `Retry-After` if present)
  so subsequent requests route to frontier until it recovers — don't hammer it.
- This alone delivers the main scenario's core value (free company model when
  available, frontier when it is down, throttled, or off-VPN).

**Phase 3 — Complexity routing (1–2 days)**
- Implement `proxy/backend_decision.py` + heuristic complexity scorer.
- Wire decision order: override → availability → complexity threshold.
- Make `threshold`, `prefer` config-driven.

**Phase 4 — Observability & overrides (1 day)**
- Surface `BackendDecision` (target, reason, score) in telemetry/dashboard and logs.
- Support per-request override header / model alias.
- Track token-savings attribution (requests served self-hosted = frontier tokens avoided).

**Phase 5 — Hardening & docs (1 day)**
- Tests: decision matrix, fallback when unreachable, streaming paths, opt-out path.
- README/docs for the three scenarios + config examples.
- Decide upstream-PR vs fork-only (open an issue upstream first to gauge scope fit).

**Later / optional**
- Learned router trained on recorded traces (replace heuristic).
- Non-streaming answer-grading cascade.
- 3-tier routing (true-local + company-remote + frontier).

---

## 9. Implementation notes (Phase 1)

### What was built
Phase 1 implements static capability-aware routing with no availability probing or
complexity scoring — a direct `routing_prefer` toggle between two pre-configured
tiers. All changes are fork-local and do not affect the upstream passthrough path.

### Files changed
| File | Change |
|------|--------|
| `headroom/proxy/models.py` | Added 5 routing fields to `ProxyConfig` |
| `headroom/proxy/backend_decision.py` | New module: `BackendDecision`, `decide()`, `apply_selfhosted_model()` |
| `headroom/proxy/server.py` | Routing-aware backend construction in `HeadroomProxy.__init__`; `_proxy_config_from_env()` wiring |
| `headroom/cli/proxy.py` | Env-var wiring into `ProxyConfig` construction |
| `headroom/providers/registry.py` | `create_proxy_backend` extended with `api_base`/`api_key` params |
| `headroom/backends/anyllm.py` | `AnyLLM.create()` now receives `api_key`/`api_base` |
| `headroom/proxy/handlers/anthropic.py` | Handler gate changed to `decide_backend()` branch |
| `headroom/proxy/handlers/openai.py` | Same gate pattern |
| `docker-compose.override.yml` | Local smoke-test override (not for production) |

### Configuration (env vars)
```
HEADROOM_ROUTING_ENABLED=true
HEADROOM_ROUTING_PREFER=selfhosted        # or "frontier"
HEADROOM_ROUTING_SELFHOSTED_API_BASE=http://dante:8080/v1
HEADROOM_ROUTING_SELFHOSTED_API_KEY=<any-non-empty-string>
HEADROOM_ROUTING_SELFHOSTED_MODEL=ministral-3-8b-creative-q5_k_m.gguf
HEADROOM_ANYLLM_PROVIDER=openai           # any OpenAI-compatible endpoint
```
Build arg: `HEADROOM_EXTRAS=proxy,code,anyllm` — the `anyllm` extra must be present
or `create_proxy_backend` silently returns `None` and all traffic falls back to frontier.

### Bugs found during implementation

**1. `RequestLogger` shadows the module-level Python logger in `server.py`**

Inside `HeadroomProxy.__init__`, a local variable `logger` is assigned to a
`RequestLogger` instance. `RequestLogger` only has a `.log()` method — no
`.info()`, `.warning()`, or `.error()`. Any call to `create_proxy_backend(logger=logger)`
in the routing branch triggered an `AttributeError` inside the registry, which was
caught by the registry's own `except Exception` handler and silently returned `None`.
The routing branch was effectively dead.

Fix: the routing branch in `__init__` now uses `_proxy_logger =
logging.getLogger("headroom.proxy")` and passes that to `create_proxy_backend`.

**2. `AnyLLM.create()` ignores the `api_key` stored on the backend instance**

`AnyLLMBackend.__init__` stored `api_key` as `self.api_key` but passed only the
provider string to `AnyLLM.create(self.provider)`. The `any-llm` library then
looked for the `OPENAI_API_KEY` environment variable, which isn't set for a
self-hosted endpoint, and raised `MissingApiKeyError`. The registry's except handler
caught this and returned `None`.

Fix: `AnyLLMBackend.__init__` now builds a `create_kwargs` dict with `api_key`
and `api_base` and passes them to `AnyLLM.create(self.provider, **create_kwargs)`.
This allows any non-empty placeholder key (e.g. `"llama-cpp-no-key"`) to satisfy
the library's validation without affecting the actual self-hosted endpoint.

### Smoke test results (2026-06-05)
- **Selfhosted path** (`routing_prefer=selfhosted`): response model
  `ministral-3-8b-creative-q5_k_m.gguf`, served by llama.cpp on `dante:8080`. ✓
- **Frontier path** (`routing_prefer=frontier`): request forwarded to
  `api.anthropic.com`, confirmed by Anthropic `req_011...` request-ID in the
  rejection (no API key set in test shell). ✓
- **Headroom optimizations preserved**: prefix-freeze and prompt-caching remain
  active on the frontier path; the selfhosted path bypasses them (expected — the
  `decide()` gate is upstream of the compression pipeline). Future work should
  evaluate whether compression is worthwhile before the selfhosted backend.

---

## 10. Implementation notes (Phase 2)

### What was built
Phase 2 adds availability awareness to the router: a circuit breaker driven by
live-request feedback, an optional background health prober, and automatic
fallback from selfhosted to frontier on 429/5xx responses or backend exceptions.
Frontier failures are **not** retried — the error is returned to the agent as-is.

### Files changed
| File | Change |
|------|--------|
| `headroom/proxy/routing_health.py` | **New module**: `RoutingHealthState` (circuit breaker) + `RoutingHealthProber` (TCP or HTTP GET probe) |
| `headroom/proxy/backend_decision.py` | `decide()` now accepts optional `health` prober; routes to frontier when circuit is open (`reason="selfhosted_circuit_open"`) |
| `headroom/proxy/models.py` | 4 new `ProxyConfig` fields: `routing_health_check_path`, `routing_health_check_interval`, `routing_circuit_failure_threshold`, `routing_circuit_cooldown_seconds` |
| `headroom/proxy/server.py` | Creates `RoutingHealthProber` in `__init__`; starts/stops it in `startup`/`shutdown`; env-var wiring |
| `headroom/cli/proxy.py` | Env-var wiring |
| `headroom/proxy/handlers/anthropic.py` | Saves `_original_model`; passes health prober to `decide_backend`; fallback on 429/5xx or exception |
| `headroom/proxy/handlers/openai.py` | Same pattern |

### Configuration (new env vars)
```
# Probe mode: unset = TCP connect-only (provider-agnostic, default).
# Set to a path (e.g. /v1/models) to do an HTTP GET instead.
HEADROOM_ROUTING_HEALTH_CHECK_PATH=        # e.g. /v1/models (optional)

# Probe interval in seconds. 0 = lazy mode (no background probe, default).
# Any positive value starts a background probe at that interval.
HEADROOM_ROUTING_HEALTH_CHECK_INTERVAL=0   # e.g. 300 for 5-minute probing

# Circuit breaker: open after N consecutive failures.
HEADROOM_ROUTING_CIRCUIT_FAILURE_THRESHOLD=5

# Cooldown: circuit stays open for N seconds before allowing through again.
HEADROOM_ROUTING_CIRCUIT_COOLDOWN_SECONDS=60
```

### Health probe modes
Two probe modes are available, selected by `HEADROOM_ROUTING_HEALTH_CHECK_PATH`:

- **TCP (default, path unset):** opens a TCP connection to the `api_base`
  host:port and immediately closes it. Provider-agnostic — works on any HTTP
  server without assuming a specific API surface. Zero load on the LLM process.
- **HTTP GET (path set):** sends a GET to the configured path. Useful to verify
  the API surface is responding, not just that the port is open. The path must
  be chosen carefully — `/v1/models` is standard for OpenAI-compatible servers
  but is not universal. Use the TCP default when in doubt.

### Background probe vs lazy mode
- **Lazy mode (`interval=0`, default):** no background task. The circuit breaker
  is driven entirely by live-request feedback. Recovery happens on the next live
  request after the cooldown expires (half-open: one request allowed through).
- **Background probe (`interval>0`):** a background asyncio task probes the
  endpoint on the configured interval and feeds results into the circuit breaker.
  Use when you want proactive recovery without waiting for a live request — at the
  cost of background network activity even when idle.

### Fallback behaviour
For every selfhosted request:
1. If the circuit is **open** at decision time → `decide()` returns
   `target="frontier"` with `reason="selfhosted_circuit_open"` — no request is
   sent to the selfhosted endpoint.
2. If the circuit is **closed** but the live request returns 429 or 5xx →
   the handler records a failure (may open the circuit), restores the original
   `claude-*` model name in the body, and falls through to the frontier path.
3. If the live request **raises an exception** (connection refused, timeout, etc.)
   → same as (2), plus `_finalize_pre_upstream()` is called to release the
   pre-upstream semaphore before falling through.
4. If the **frontier path fails** → the error is returned to the agent. No further
   fallback.

### Design decision: no `/v1/models` default
An earlier draft used `/v1/models` as the default health-check path. This was
changed to a TCP probe because `/v1/models` is not guaranteed on all
OpenAI-compatible servers (some minimal implementations only expose
`/v1/chat/completions`). The TCP probe is provider-agnostic and imposes zero
load on the LLM. Users who want API-surface validation can set
`HEADROOM_ROUTING_HEALTH_CHECK_PATH=/v1/models` explicitly.

### Handler integration pattern (implementation detail)
Both `anthropic.py` and `openai.py` use an identical routing gate structure:

```python
_original_model = body.get("model")
_health = getattr(self, "health_prober", None)   # hoisted once; None-safe
_route = decide_backend(self.config, self.anthropic_backend, body, _health)
if _route.use_selfhosted and _route.routing_enabled:
    apply_selfhosted_model(self.config, body)
if _route.use_selfhosted:
    try:
        # ... dispatch to selfhosted ...
        if status in (429, 500, 502, 503, 504):
            if _health: _health.record_request_failure(status)
            body["model"] = _original_model   # restore before falling through
            logger.warning(...)
            # no return → falls through to frontier block below
        else:
            if _health: _health.record_request_success()
            # ... success path (returns) ...
    except Exception as e:
        if _health: _health.record_request_failure(503)
        body["model"] = _original_model
        await _finalize_pre_upstream()   # release semaphore before falling through
        # no return → falls through to frontier block below

# Frontier block runs naturally after the if-block when selfhosted fails
```

Key invariants:
- `_health` is looked up exactly once per request (hoisted before the if-block).
  `getattr(self, "health_prober", None)` is used instead of `self.health_prober`
  so the handler is safe in unit-test contexts where the attribute is absent.
- Fallback is achieved by **fall-through** (no `return` in failure paths), not by
  re-executing logic. The frontier block is always reached when selfhosted fails.
- `_original_model` is restored in the body before falling through so the frontier
  receives the original `claude-*` model name, not the rewritten selfhosted name.
- `_finalize_pre_upstream()` (Anthropic handler only) is idempotent — safe to call
  in the exception handler before the frontier path calls it again. It is
  flag-guarded internally (`_stage_timings_emitted`).
- Streaming selfhosted paths **return** immediately (no fallback supported for
  streaming — the SSE stream is already open). Fallback only applies to
  non-streaming requests.

### Code quality cleanup applied before commit
Before committing, two issues were cleaned up:
1. **DRY fix**: `_health = getattr(self, "health_prober", None)` was previously
   assigned twice in each handler (once in the 429/5xx branch, again in the
   `except` block). Hoisted to a single assignment before the routing gate.
2. **KISS fix**: `_selfhosted_failed` was a dead variable — it was set to `True`
   in failure paths but never read. Fallback is structural (fall-through), not
   conditional on a flag. The variable was removed.

### Test coverage
**Unit tests — 40 tests, all pass (`tests/test_routing_health.py` + `tests/test_routing_decision.py`)**
- `RoutingHealthState`: threshold, cooldown, half-open recovery, re-open after
  half-open failure, single-failure threshold, success reset.
- `RoutingHealthProber` lifecycle: lazy start (no task created), active start
  (task created + initial probe), stop, idempotent stop.
- TCP probe: success records success, failure records failure.
- HTTP probe: 2xx records success, 5xx records failure, exception records failure.
- Probe mode selection: TCP when `health_path=None`, HTTP when path set.
- Host:port parsing from `api_base` (including default ports for http/https).
- `decide()` matrix: no backend, routing disabled, prefer-selfhosted
  (healthy / circuit-open / no prober), circuit opens progressively, circuit
  resets after success, prefer-frontier ignores health.
- `BackendDecision` properties: `use_selfhosted`, frozen dataclass enforcement.

**Docker integration tests (manual, against running compose stack)**

| Test | Selfhosted config | Result |
|------|-------------------|--------|
| Happy path | dante:8080 (real llama.cpp) | 200, model=`ministral-3-8b-...gguf` ✓ |
| Exception → frontier fallback | dante:19999 (nothing listening) | `any-llm` retried twice, raised, handler fell through; Anthropic `req_011...` ID confirms frontier reached ✓ |
| Frontier failure = no further fallback | bad Anthropic key | Auth error returned to client as-is ✓ |
| OpenAI handler path | dante:8080 | 200, model=`ministral-3-8b-...gguf` ✓ |

### Commits on `feat/backend-routing`
| Commit | Description |
|--------|-------------|
| `c542c81` | Phase 1: static two-tier routing (capability-aware preference) |
| `af5b55c` | Phase 2: availability routing with circuit breaker (this work) |

---

## 11. Open questions / decisions to make
1. **Override interface** — header name and/or model-alias convention for forcing a tier.
2. **Routing granularity** — global, or only when a specific model alias is requested?
3. **Health-probe cost** — ping-per-request vs cached TTL; what TTL is acceptable.
4. **Fallback symmetry** — only self-hosted→frontier, or also frontier→self-hosted
   (e.g. subscription rate-limited)?
5. **Data governance (future overlay, not this work)** — in a company context,
   *some* content may be required to stay on the internal model and be barred from
   the external frontier. This would become a hard routing *constraint* layered on
   top of cost optimization. Out of scope now; flagged so the decision module's
   shape can accommodate it.

---

## 12. Risks
- **Single-backend assumption is load-bearing.** Widening it touches startup and
  the handler dispatch contract; the wrapper approach contains the blast radius but
  must faithfully implement the whole `Backend` ABC (incl. `name`, `close`, model mapping).
- **Streaming forecloses answer-grading cascades** — accepted; upfront routing only.
- **Router/health overhead must stay << the call it gates**, or the optimization is self-defeating.
- **Upstream scope** — backend-routing may be seen as out-of-scope for a
  compression project; confirm before investing in a PR vs keeping it fork-local.

---

## 13. PR notes (for when the PR is opened)

### What this PR adds
Two-phase capability-aware routing that lets a single Headroom instance
intelligently split traffic between a **local/self-hosted LLM** and the
**Anthropic frontier**, with automatic failover.

**Phase 1 — static preference routing** (`c542c81`)
- New `ProxyConfig` fields: `routing_enabled`, `routing_prefer`
  (`selfhosted` | `frontier`), `routing_selfhosted_api_base`,
  `routing_selfhosted_api_key`, `routing_selfhosted_model`.
- `proxy/backend_decision.py` — immutable `BackendDecision` dataclass +
  `decide()` function. Mirrors the `CompressionDecision` pattern.
- `apply_selfhosted_model()` rewrites `body["model"]` in-place before dispatch.
- Both `anthropic.py` and `openai.py` handlers updated with the routing gate.
- Env vars: `HEADROOM_ROUTING_ENABLED`, `HEADROOM_ROUTING_PREFER`,
  `HEADROOM_ROUTING_SELFHOSTED_API_BASE`, `HEADROOM_ROUTING_SELFHOSTED_API_KEY`,
  `HEADROOM_ROUTING_SELFHOSTED_MODEL`.

**Phase 2 — availability routing with circuit breaker** (`af5b55c`)
- New `proxy/routing_health.py`: `RoutingHealthState` (circuit breaker state
  machine) + `RoutingHealthProber` (lifecycle + TCP/HTTP probe dispatch).
- Circuit opens after N consecutive failures; goes half-open after cooldown;
  resets on first success.
- Default is **lazy mode** (`HEADROOM_ROUTING_HEALTH_CHECK_INTERVAL=0`): no
  background task — circuit driven entirely by live request outcomes.
- Optional active mode: asyncio background task at user-set interval. TCP probe
  by default; HTTP GET opt-in via `HEADROOM_ROUTING_HEALTH_CHECK_PATH`.
- Handlers: on selfhosted 429/5xx or exception, model is restored, failure is
  recorded, and the request falls through to the frontier block. **No return
  statement in failure paths** — fallback is structural.
- Frontier failures are **not caught** — they propagate directly to the client.
- Additional env vars: `HEADROOM_ROUTING_HEALTH_CHECK_PATH`,
  `HEADROOM_ROUTING_HEALTH_CHECK_INTERVAL`,
  `HEADROOM_ROUTING_CIRCUIT_FAILURE_THRESHOLD`,
  `HEADROOM_ROUTING_CIRCUIT_COOLDOWN_SECONDS`.

### Key design decisions reviewers should know
1. **Frontier stays on native Anthropic passthrough.** We never wrap frontier
   traffic in a `Backend` object. This preserves Headroom's prompt-caching and
   prefix-freeze optimisations, which depend on the native `_retry_request` path.
2. **No fallback for streaming selfhosted.** Once an SSE stream is open there is
   nowhere to fall back to. Streaming requests to selfhosted return immediately;
   only non-streaming requests participate in fallback.
3. **`decide()` is pure and dependency-light.** It takes config + optional prober
   and returns a frozen dataclass. No I/O, no side effects — easy to unit test
   and slots into Phase 3 (complexity) without structural change.
4. **`RoutingHealthProber` is created in `server.py`** (not in `backend_decision.py`)
   to keep the decision module stateless. The handlers access it via
   `getattr(self, "health_prober", None)` so they are safe in unit-test contexts.
5. **Health prober lifecycle**: `start()` in `server.startup()`;
   `stop()` in `server.shutdown()` **before** `http_client.aclose()` to avoid
   aiohttp teardown races.

### Files changed summary
| File | Change |
|------|--------|
| `headroom/proxy/routing_health.py` | New — circuit breaker + prober |
| `headroom/proxy/backend_decision.py` | Extended `decide()` with health param |
| `headroom/proxy/models.py` | 9 new `ProxyConfig` fields (5 Phase 1, 4 Phase 2) |
| `headroom/proxy/server.py` | Creates/starts/stops `RoutingHealthProber` |
| `headroom/cli/proxy.py` | Wires all env vars into `ProxyConfig` |
| `headroom/proxy/handlers/anthropic.py` | Routing gate + fallback pattern |
| `headroom/proxy/handlers/openai.py` | Same routing gate + fallback pattern |
| `docs/design/local-frontier-routing.md` | This document |
| `tests/test_routing_health.py` | 26 unit tests |
| `tests/test_routing_decision.py` | 14 unit tests |

### Test evidence
- 40 unit tests, all pass.
- Docker integration: happy path, bad-endpoint fallback, frontier-failure
  propagation, and OpenAI handler path — all verified manually.

### What is NOT in this PR (future phases)
- **Phase 3**: Complexity routing — heuristic scorer sends simple requests to
  selfhosted, complex ones to frontier.
- **Phase 4**: Observability — Prometheus metrics for routing decisions,
  per-request override headers.
- **Phase 5**: Hardening + docs — README section, docker-compose example,
  final integration tests.
