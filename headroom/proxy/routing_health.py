"""Health prober and circuit breaker for capability-aware backend routing.

Tracks self-hosted endpoint availability via periodic background probes and
live-request failure feedback. When the circuit opens (too many consecutive
failures), ``decide()`` routes to frontier until the cooldown expires.

Two probe modes (selected at construction time):
- **TCP** (default, ``health_path=None``): opens a TCP connection to the
  host:port derived from ``api_base``.  Provider-agnostic — works on any
  HTTP server without assuming a specific API surface.
- **HTTP GET** (``health_path="/v1/models"`` or any path): sends a lightweight
  GET to the given path.  Useful when you want to verify the API surface, not
  just connectivity.  Opt in via ``HEADROOM_ROUTING_HEALTH_CHECK_PATH``.

See ``docs/design/local-frontier-routing.md``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class RoutingHealthState:
    """Mutable circuit-breaker state for one self-hosted endpoint."""

    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self.consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0  # monotonic timestamp

    @property
    def circuit_open(self) -> bool:
        if self._circuit_open_until == 0.0:
            return False
        return time.monotonic() < self._circuit_open_until

    def is_available(self) -> bool:
        """Return True if the endpoint should be tried for the next request."""
        if not self.circuit_open:
            return True
        # Half-open: allow one probe through once cooldown expires.
        # The next call to record_failure will re-open if needed.
        if time.monotonic() >= self._circuit_open_until:
            self._circuit_open_until = 0.0
            return True
        return False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self._circuit_open_until = 0.0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self._threshold:
            open_until = time.monotonic() + self._cooldown
            self._circuit_open_until = open_until
            logger.warning(
                "routing: selfhosted circuit opened after %d consecutive failures "
                "(cooldown=%.0fs, resets at monotonic=%.1f)",
                self.consecutive_failures,
                self._cooldown,
                open_until,
            )


class RoutingHealthProber:
    """Periodic health prober for the self-hosted routing endpoint.

    Runs a background asyncio task that checks the endpoint every
    ``interval_seconds``.  Live-request 429/5xx outcomes are also fed in via
    :meth:`record_request_failure` so the circuit can open without waiting for
    the next scheduled probe.

    Probe mode is determined by ``health_path``:
    - ``None`` (default): TCP connection check — provider-agnostic.
    - A path string (e.g. ``"/v1/models"``): HTTP GET to that path.
      Set ``HEADROOM_ROUTING_HEALTH_CHECK_PATH`` to opt in.
    """

    def __init__(
        self,
        api_base: str,
        health_path: str | None = None,
        interval_seconds: int = 30,
        failure_threshold: int = 5,
        cooldown_seconds: int = 60,
        api_key: str | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._health_path = health_path  # None → TCP probe
        self._interval = interval_seconds
        self._api_key = api_key
        parsed = urlparse(self._api_base)
        self._host = parsed.hostname or "localhost"
        default_port = 443 if parsed.scheme == "https" else 80
        self._port = parsed.port or default_port
        self.state = RoutingHealthState(
            failure_threshold=failure_threshold,
            cooldown_seconds=float(cooldown_seconds),
        )
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the health prober.

        When ``interval_seconds > 0``: probe immediately, then schedule
        recurring background checks at that interval.

        When ``interval_seconds == 0`` (lazy mode): no background probe is
        started.  The circuit breaker still operates via live-request feedback
        (:meth:`record_request_failure` / :meth:`record_request_success`), and
        recovery is detected on the next successful live request after the
        cooldown expires.
        """
        if self._interval > 0:
            await self._probe()
            self._task = asyncio.create_task(self._run(), name="routing-health-prober")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Public interface for live-request feedback
    # ------------------------------------------------------------------

    def record_request_failure(self, status_code: int) -> None:
        """Called when a live selfhosted request returns a failure status."""
        logger.debug("routing: selfhosted live-request failure (status=%d)", status_code)
        self.state.record_failure()

    def record_request_success(self) -> None:
        """Called when a live selfhosted request succeeds."""
        self.state.record_success()

    @property
    def is_available(self) -> bool:
        return self.state.is_available()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self._probe()

    async def _probe(self) -> None:
        if self._health_path is not None:
            await self._probe_http()
        else:
            await self._probe_tcp()

    async def _probe_tcp(self) -> None:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            self._on_probe_success()
        except Exception as exc:
            logger.debug("routing: TCP probe failed (%s:%d): %s", self._host, self._port, exc)
            self.state.record_failure()

    async def _probe_http(self) -> None:
        url = f"{self._api_base}{self._health_path}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code < 500:
                self._on_probe_success()
            else:
                logger.debug("routing: HTTP probe returned %d", resp.status_code)
                self.state.record_failure()
        except Exception as exc:
            logger.debug("routing: HTTP probe failed: %s", exc)
            self.state.record_failure()

    def _on_probe_success(self) -> None:
        was_unavailable = not self.state.is_available()
        self.state.record_success()
        if was_unavailable:
            mode = f"HTTP GET {self._health_path}" if self._health_path else f"TCP {self._host}:{self._port}"
            logger.info("routing: selfhosted endpoint recovered (%s)", mode)
