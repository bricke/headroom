"""Unit tests for backend routing decision logic (Phase 2).

Tests decide() with the health prober wired in — verifies circuit-open
routing, prefer-frontier, lazy mode, and no-backend cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.proxy.backend_decision import BackendDecision, decide
from headroom.proxy.routing_health import RoutingHealthProber


def make_config(
    routing_enabled: bool = True,
    routing_prefer: str = "selfhosted",
) -> MagicMock:
    cfg = MagicMock()
    cfg.routing_enabled = routing_enabled
    cfg.routing_prefer = routing_prefer
    return cfg


def make_backend() -> MagicMock:
    return MagicMock(name="anyllm-openai")


def make_prober(available: bool = True) -> RoutingHealthProber:
    prober = RoutingHealthProber(api_base="http://localhost:8080", interval_seconds=0)
    if not available:
        # Force circuit open by exceeding the threshold
        for _ in range(prober.state._threshold):
            prober.state.record_failure()
    return prober


# ---------------------------------------------------------------------------
# No backend configured → always frontier
# ---------------------------------------------------------------------------


class TestNoBackend:
    def test_no_backend_routes_to_frontier(self) -> None:
        cfg = make_config()
        decision = decide(cfg, backend=None, body={})
        assert decision.target == "frontier"
        assert decision.reason == "no_backend_configured"

    def test_no_backend_ignores_health(self) -> None:
        cfg = make_config()
        prober = make_prober(available=False)
        decision = decide(cfg, backend=None, body={}, health=prober)
        assert decision.target == "frontier"
        assert decision.reason == "no_backend_configured"


# ---------------------------------------------------------------------------
# Routing disabled → legacy selfhosted
# ---------------------------------------------------------------------------


class TestRoutingDisabled:
    def test_routing_disabled_routes_to_selfhosted(self) -> None:
        cfg = make_config(routing_enabled=False)
        backend = make_backend()
        decision = decide(cfg, backend=backend, body={})
        assert decision.target == "selfhosted"
        assert decision.reason == "routing_disabled_legacy_backend"

    def test_routing_disabled_ignores_health(self) -> None:
        cfg = make_config(routing_enabled=False)
        backend = make_backend()
        prober = make_prober(available=False)
        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "selfhosted"
        assert decision.reason == "routing_disabled_legacy_backend"


# ---------------------------------------------------------------------------
# Routing enabled, prefer selfhosted
# ---------------------------------------------------------------------------


class TestPreferSelfhosted:
    def test_healthy_endpoint_routes_to_selfhosted(self) -> None:
        cfg = make_config(routing_prefer="selfhosted")
        backend = make_backend()
        prober = make_prober(available=True)
        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "selfhosted"
        assert decision.reason == "prefer_selfhosted"
        assert decision.routing_enabled is True
        assert decision.selfhosted_available is True

    def test_no_health_prober_routes_to_selfhosted(self) -> None:
        cfg = make_config(routing_prefer="selfhosted")
        backend = make_backend()
        decision = decide(cfg, backend=backend, body={}, health=None)
        assert decision.target == "selfhosted"
        assert decision.reason == "prefer_selfhosted"

    def test_open_circuit_routes_to_frontier(self) -> None:
        cfg = make_config(routing_prefer="selfhosted")
        backend = make_backend()
        prober = make_prober(available=False)
        assert prober.is_available is False

        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "frontier"
        assert decision.reason == "selfhosted_circuit_open"
        assert decision.routing_enabled is True
        assert decision.selfhosted_available is False

    def test_circuit_opens_progressively(self) -> None:
        cfg = make_config(routing_prefer="selfhosted")
        backend = make_backend()
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            failure_threshold=3,
            interval_seconds=0,
        )

        # Before threshold: still routes selfhosted
        prober.record_request_failure(503)
        prober.record_request_failure(503)
        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "selfhosted"

        # At threshold: circuit opens → routes frontier
        prober.record_request_failure(503)
        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "frontier"
        assert decision.reason == "selfhosted_circuit_open"

    def test_circuit_resets_after_success(self) -> None:
        cfg = make_config(routing_prefer="selfhosted")
        backend = make_backend()
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            failure_threshold=2,
            interval_seconds=0,
        )
        prober.record_request_failure(500)
        prober.record_request_failure(500)
        assert prober.is_available is False

        prober.record_request_success()
        assert prober.is_available is True

        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "selfhosted"


# ---------------------------------------------------------------------------
# Routing enabled, prefer frontier
# ---------------------------------------------------------------------------


class TestPreferFrontier:
    def test_prefer_frontier_routes_to_frontier(self) -> None:
        cfg = make_config(routing_prefer="frontier")
        backend = make_backend()
        decision = decide(cfg, backend=backend, body={})
        assert decision.target == "frontier"
        assert decision.reason == "prefer_frontier"

    def test_prefer_frontier_ignores_health_state(self) -> None:
        """Health prober should not affect prefer=frontier routing."""
        cfg = make_config(routing_prefer="frontier")
        backend = make_backend()
        prober = make_prober(available=False)
        decision = decide(cfg, backend=backend, body={}, health=prober)
        assert decision.target == "frontier"


# ---------------------------------------------------------------------------
# BackendDecision properties
# ---------------------------------------------------------------------------


class TestBackendDecision:
    def test_use_selfhosted_property(self) -> None:
        d = BackendDecision(target="selfhosted", reason="test")
        assert d.use_selfhosted is True

    def test_use_frontier_property(self) -> None:
        d = BackendDecision(target="frontier", reason="test")
        assert d.use_selfhosted is False

    def test_frozen_dataclass(self) -> None:
        d = BackendDecision(target="selfhosted", reason="test")
        with pytest.raises(Exception):  # frozen=True raises FrozenInstanceError
            d.target = "frontier"  # type: ignore[misc]
