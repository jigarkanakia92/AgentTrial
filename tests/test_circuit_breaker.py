"""Circuit breaker state machine."""
from __future__ import annotations

import pytest

import scraper.circuit_breaker as cb_module
from scraper.circuit_breaker import BreakerState, CircuitBreaker


@pytest.fixture
def fake_clock(monkeypatch):
    state = {"now": 1000.0}

    def _now():
        return state["now"]

    monkeypatch.setattr(cb_module.time, "monotonic", _now)
    return state


def test_opens_after_threshold(fake_clock):
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow_request()


def test_half_open_probe_after_cooldown(fake_clock):
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.allow_request()

    fake_clock["now"] += 61  # cooldown elapsed
    assert breaker.state is BreakerState.OPEN
    assert breaker.allow_request()          # probe allowed
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.record_failure()                # probe failed -> reopen
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow_request()


def test_success_closes_breaker(fake_clock):
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    fake_clock["now"] += 61
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow_request()


def test_single_failure_never_opens(fake_clock):
    breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=60)
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow_request()
