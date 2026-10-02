"""Circuit breaker for the fetch layer.

If Yahoo starts answering 403/429 en masse (rate ban, fingerprint flagged),
retrying forever only deepens the ban. The breaker:

  CLOSED     — normal operation
  OPEN       — requests are refused immediately for `cooldown_seconds`
  HALF_OPEN  — after cooldown, a single probe request is allowed through;
               success closes the breaker, failure re-opens it

Pure in-process (per container). Simple time math on `time.monotonic`.
"""
from __future__ import annotations

import time
from enum import Enum

from loguru import logger


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a fetch is attempted while the breaker is OPEN."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 900) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1, cooldown_seconds)
        self.state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self.total_opens = 0

    def _now(self) -> float:
        return time.monotonic()

    def allow_request(self) -> bool:
        if self.state is BreakerState.CLOSED:
            return True
        if self.state is BreakerState.OPEN:
            assert self._opened_at is not None
            if self._now() - self._opened_at >= self.cooldown_seconds:
                self.state = BreakerState.HALF_OPEN
                logger.info("Circuit breaker HALF_OPEN — probing with one request")
                return True
            return False
        return True  # HALF_OPEN: one probe in flight

    def record_success(self) -> None:
        if self.state is not BreakerState.CLOSED:
            logger.info("Circuit breaker CLOSED — upstream healthy again")
        self.state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN:
            self._open()
            return
        if (
            self.state is BreakerState.CLOSED
            and self._consecutive_failures >= self.failure_threshold
        ):
            self._open()

    def _open(self) -> None:
        self.state = BreakerState.OPEN
        self._opened_at = self._now()
        self.total_opens += 1
        logger.error(
            "Circuit breaker OPEN after {} consecutive hard failures — "
            "pausing upstream requests for {}s to let any ban cool off",
            self._consecutive_failures,
            self.cooldown_seconds,
        )

    @property
    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_opens": self.total_opens,
        }
