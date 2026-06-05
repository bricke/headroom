"""Unit tests for routing health prober and circuit breaker (Phase 2).

These tests cover RoutingHealthState and RoutingHealthProber in isolation —
no network, no Docker, no Rust extension needed.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.proxy.routing_health import RoutingHealthProber, RoutingHealthState


# ---------------------------------------------------------------------------
# RoutingHealthState — circuit breaker logic
# ---------------------------------------------------------------------------


class TestRoutingHealthState:
    def make_state(self, threshold: int = 3, cooldown: float = 60.0) -> RoutingHealthState:
        return RoutingHealthState(failure_threshold=threshold, cooldown_seconds=cooldown)

    def test_initially_available(self) -> None:
        state = self.make_state()
        assert state.is_available() is True
        assert state.circuit_open is False
        assert state.consecutive_failures == 0

    def test_single_failure_does_not_open_circuit(self) -> None:
        state = self.make_state(threshold=3)
        state.record_failure()
        assert state.circuit_open is False
        assert state.is_available() is True
        assert state.consecutive_failures == 1

    def test_circuit_opens_at_threshold(self) -> None:
        state = self.make_state(threshold=3)
        state.record_failure()
        state.record_failure()
        assert state.circuit_open is False
        state.record_failure()  # hits threshold
        assert state.circuit_open is True
        assert state.is_available() is False

    def test_circuit_opens_beyond_threshold(self) -> None:
        state = self.make_state(threshold=2)
        for _ in range(5):
            state.record_failure()
        assert state.circuit_open is True
        assert state.is_available() is False

    def test_success_resets_failures_and_closes_circuit(self) -> None:
        state = self.make_state(threshold=2)
        state.record_failure()
        state.record_failure()
        assert state.circuit_open is True
        state.record_success()
        assert state.circuit_open is False
        assert state.consecutive_failures == 0
        assert state.is_available() is True

    def test_circuit_reopens_after_cooldown_failure(self) -> None:
        """Half-open: after cooldown, one request is allowed through.
        If it fails again, circuit re-opens."""
        state = self.make_state(threshold=2, cooldown=0.05)
        state.record_failure()
        state.record_failure()
        assert state.circuit_open is True

        time.sleep(0.1)  # let cooldown expire
        # half-open: is_available returns True once
        assert state.is_available() is True
        # now fail again — circuit re-opens
        state.record_failure()
        assert state.circuit_open is True
        assert state.is_available() is False

    def test_circuit_recovers_via_success_after_cooldown(self) -> None:
        state = self.make_state(threshold=1, cooldown=0.05)
        state.record_failure()
        assert state.circuit_open is True

        time.sleep(0.1)
        assert state.is_available() is True
        state.record_success()
        assert state.circuit_open is False
        assert state.is_available() is True

    def test_threshold_one(self) -> None:
        state = self.make_state(threshold=1)
        state.record_failure()
        assert state.circuit_open is True

    def test_is_available_false_during_cooldown(self) -> None:
        state = self.make_state(threshold=1, cooldown=60.0)
        state.record_failure()
        # circuit is open, cooldown has not expired
        assert state.is_available() is False
        assert state.circuit_open is True


# ---------------------------------------------------------------------------
# RoutingHealthProber — lifecycle and probe dispatch
# ---------------------------------------------------------------------------


class TestRoutingHealthProberLifecycle:
    def test_lazy_mode_does_not_start_task(self) -> None:
        """interval=0 → start() should not create a background task."""
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            interval_seconds=0,
        )
        # No event loop needed — start() won't launch the task.
        assert prober._task is None

    def test_is_available_initially_true(self) -> None:
        prober = RoutingHealthProber(api_base="http://localhost:8080")
        assert prober.is_available is True

    def test_record_request_failure_feeds_circuit(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            failure_threshold=2,
        )
        prober.record_request_failure(429)
        assert prober.state.consecutive_failures == 1
        assert prober.is_available is True

        prober.record_request_failure(503)
        assert prober.state.consecutive_failures == 2
        assert prober.is_available is False  # circuit now open

    def test_record_request_success_resets_circuit(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            failure_threshold=1,
        )
        prober.record_request_failure(500)
        assert prober.is_available is False
        prober.record_request_success()
        assert prober.is_available is True

    @pytest.mark.asyncio
    async def test_lazy_start_does_not_probe(self) -> None:
        """interval=0 → start() does NOT run an initial probe."""
        prober = RoutingHealthProber(
            api_base="http://localhost:9999",  # nothing listening
            interval_seconds=0,
        )
        with patch.object(prober, "_probe", new_callable=AsyncMock) as mock_probe:
            await prober.start()
            mock_probe.assert_not_called()
        assert prober._task is None

    @pytest.mark.asyncio
    async def test_active_start_probes_immediately(self) -> None:
        """interval>0 → start() runs an initial probe before scheduling."""
        prober = RoutingHealthProber(
            api_base="http://localhost:9999",
            interval_seconds=999,
        )
        with patch.object(prober, "_probe", new_callable=AsyncMock) as mock_probe:
            await prober.start()
            mock_probe.assert_called_once()
        assert prober._task is not None
        await prober.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:9999",
            interval_seconds=999,
        )
        with patch.object(prober, "_probe", new_callable=AsyncMock):
            await prober.start()
            assert prober._task is not None
            await prober.stop()
            assert prober._task is None

    @pytest.mark.asyncio
    async def test_stop_when_no_task_is_safe(self) -> None:
        prober = RoutingHealthProber(api_base="http://localhost:8080", interval_seconds=0)
        await prober.stop()  # should not raise


class TestRoutingHealthProberProbes:
    @pytest.mark.asyncio
    async def test_tcp_probe_success_records_success(self) -> None:
        prober = RoutingHealthProber(api_base="http://localhost:8080", interval_seconds=0)
        # Inject one failure so we can verify recovery is logged.
        prober.state.record_failure()

        with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_conn:
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()
            mock_conn.return_value = (MagicMock(), writer)
            await prober._probe_tcp()

        assert prober.state.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_tcp_probe_failure_records_failure(self) -> None:
        prober = RoutingHealthProber(api_base="http://localhost:8080", interval_seconds=0)

        with patch("asyncio.open_connection", side_effect=ConnectionRefusedError("refused")):
            await prober._probe_tcp()

        assert prober.state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_http_probe_2xx_records_success(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            health_path="/v1/models",
            interval_seconds=0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await prober._probe_http()

        assert prober.state.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_http_probe_5xx_records_failure(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            health_path="/v1/models",
            interval_seconds=0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await prober._probe_http()

        assert prober.state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_http_probe_exception_records_failure(self) -> None:
        prober = RoutingHealthProber(
            api_base="http://localhost:8080",
            health_path="/v1/models",
            interval_seconds=0,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("timeout"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await prober._probe_http()

        assert prober.state.consecutive_failures == 1

    def test_probe_mode_selected_by_health_path(self) -> None:
        tcp_prober = RoutingHealthProber(api_base="http://localhost:8080")
        assert tcp_prober._health_path is None

        http_prober = RoutingHealthProber(
            api_base="http://localhost:8080", health_path="/v1/models"
        )
        assert http_prober._health_path == "/v1/models"

    def test_host_port_parsed_from_api_base(self) -> None:
        prober = RoutingHealthProber(api_base="http://192.168.0.175:8080/v1")
        assert prober._host == "192.168.0.175"
        assert prober._port == 8080

    def test_default_port_http(self) -> None:
        prober = RoutingHealthProber(api_base="http://example.com/v1")
        assert prober._port == 80

    def test_default_port_https(self) -> None:
        prober = RoutingHealthProber(api_base="https://example.com/v1")
        assert prober._port == 443
