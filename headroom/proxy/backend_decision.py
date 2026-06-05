"""Capability-aware backend routing decision.

Decides, per request, whether to serve from the configured **self-hosted** tier
(a company-provided or LAN-hosted capable model, reached via the translated
``Backend``) or from the native **frontier** (Anthropic) passthrough — which
preserves Headroom's prompt-caching / prefix-freeze optimizations.

Mirrors the per-request "decision" pattern used by ``compression_decision.py``.

This module is intentionally dependency-light: it takes the proxy config, the
(optional) configured backend, and the request body, and returns a small frozen
value type. Phase 1 implements only the preference/legacy logic; availability
(Phase 2) and complexity (Phase 3) signals slot into :func:`decide` later.

See ``docs/design/local-frontier-routing.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # avoid runtime import cycle with proxy.models / backends.base
    from headroom.backends.base import Backend
    from headroom.proxy.models import ProxyConfig

Target = Literal["selfhosted", "frontier"]


@dataclass(frozen=True)
class BackendDecision:
    """Outcome of routing a single request.

    Attributes mirror ``CompressionDecision`` in spirit: the chosen ``target``
    plus the constituent signals, so the decision is observable in telemetry.
    """

    target: Target
    reason: str
    # Observability fields (populated as later phases add signals).
    routing_enabled: bool = False
    selfhosted_available: bool = True
    complexity_score: float | None = None
    forced_by: str | None = None

    @property
    def use_selfhosted(self) -> bool:
        return self.target == "selfhosted"


def decide(
    config: ProxyConfig,
    backend: Backend | None,
    body: dict[str, Any],
) -> BackendDecision:
    """Return the routing decision for one request.

    Behavior:
    - No configured backend  -> always ``frontier`` (native path); nothing to route to.
    - Routing disabled       -> ``selfhosted`` whenever a backend exists (legacy:
      the configured backend handles every request).
    - Routing enabled        -> Phase 1: honor ``routing_prefer``. (Availability
      and complexity signals are layered in by Phases 2-3.)
    """
    if backend is None:
        return BackendDecision(target="frontier", reason="no_backend_configured")

    if not config.routing_enabled:
        return BackendDecision(
            target="selfhosted",
            reason="routing_disabled_legacy_backend",
        )

    # Phase 1: constant preference. Phases 2-3 insert override -> availability ->
    # complexity-threshold ahead of this fallthrough.
    target: Target = config.routing_prefer
    return BackendDecision(
        target=target,
        reason=f"prefer_{target}",
        routing_enabled=True,
    )


def apply_selfhosted_model(config: ProxyConfig, body: dict[str, Any]) -> None:
    """Rewrite ``body['model']`` to the self-hosted tier's model name in place.

    The incoming request names a frontier model (e.g. a ``claude-*`` id) that the
    company/local endpoint won't recognize. When a self-hosted model name is
    configured, swap it in before dispatching to the self-hosted backend. No-op
    if unset (the backend then sees the original model id).
    """
    if config.routing_selfhosted_model:
        body["model"] = config.routing_selfhosted_model
