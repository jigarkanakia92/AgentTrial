"""Ticker-based option pulling, durable caching, JSON safety, and degradation."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from uuid import uuid4

import pytest

import analyzer.options_data as options
from analyzer.config import AnalyzerSettings
from db import repository
from db.models import utcnow


async def test_pull_persists_full_chain_and_metrics(option_db, option_provider):
    snapshot = await options.fetch_option_data(" aapl ")
    assert snapshot is not None and snapshot.id is not None
    assert snapshot.ticker == "AAPL"
    assert snapshot.expiration_date == option_provider.expiry
    assert option_provider.chain_calls == [option_provider.expiry.isoformat()]
    assert len(snapshot.available_expirations) == 2
    assert snapshot.spot_price == 100
    assert snapshot.atm_call_iv == pytest.approx(0.2)
    assert snapshot.atm_put_iv == pytest.approx(0.3)
    assert snapshot.total_call_open_interest == 150
    assert snapshot.total_put_open_interest == 300
    assert snapshot.put_call_oi_ratio == 2
    assert snapshot.high_iv_skew_bearish is True
    assert "high_iv_skew_bearish" in snapshot.summary

    saved = await repository.fetch_latest_option_data("aapl")
    assert saved.id == snapshot.id
    assert saved.calls[0]["contractSymbol"] == "AAPL_C100"
    assert saved.calls[0]["newProviderField"] == "preserved"
    assert saved.calls[0]["lastTradeDate"].endswith("Z")
    assert saved.calls[1]["volume"] is None
    assert saved.calls[1]["bid"] is None
    assert saved.calls[1]["lastTradeDate"] is None
    assert len(saved.puts) == 2
    json.dumps(saved.calls, allow_nan=False)
    json.dumps(saved.puts, allow_nan=False)


async def test_db_cache_survives_memory_reset_and_missing_provider(
    option_db, option_provider, monkeypatch
):
    original = await options.fetch_option_data("AAPL")
    options._CACHE.clear()
    monkeypatch.setattr(options, "YFINANCE_AVAILABLE", False)
    cached = await options.fetch_option_data("aapl")
    assert cached.id == original.id
    assert cached.fetched_at.tzinfo is not None
    assert option_provider.ticker_calls == ["AAPL"]
    assert await options.fetch_option_summary("AAPL") == original.summary
    assert len(await repository.fetch_option_data_history("AAPL")) == 1


async def test_forced_pull_retains_history(option_db, option_provider):
    first = await options.fetch_option_data("AAPL")
    second = await options.fetch_option_data("AAPL", force_refresh=True)
    assert first.id != second.id
    assert len(option_provider.chain_calls) == 2
    history = await repository.fetch_option_data_history("AAPL")
    assert [row.id for row in history] == [second.id, first.id]


async def test_stale_data_is_refreshed(option_db, option_provider):
    stale = options.OptionSnapshot(
        ticker="AAPL",
        expiration_date=option_provider.expiry,
        fetched_at=utcnow() - timedelta(hours=2),
        summary="old quote",
    )
    old_id = await repository.insert_option_data(stale.to_row())
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.id != old_id
    assert "old quote" not in snapshot.summary
    assert len(await repository.fetch_option_data_history("AAPL")) == 2


async def test_expired_chain_is_not_reused_even_with_fresh_timestamp(
    option_db, option_provider
):
    expired = options.OptionSnapshot(
        ticker="AAPL",
        expiration_date=utcnow().date() - timedelta(days=1),
        summary="expired",
    )
    old_id = await repository.insert_option_data(expired.to_row())
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.id != old_id
    assert snapshot.expiration_date == option_provider.expiry


async def test_zero_ttl_always_pulls(option_db, option_provider, monkeypatch):
    cache_reads = []

    async def forbidden_cache(*args, **kwargs):
        cache_reads.append(True)
        raise AssertionError("TTL=0 must not read cache")

    monkeypatch.setattr(repository, "fetch_latest_option_data", forbidden_cache)
    settings = AnalyzerSettings(options_cache_ttl_seconds=0)
    first = await options.fetch_option_data("AAPL", settings=settings)
    second = await options.fetch_option_data("AAPL", settings=settings)
    assert first.id != second.id
    assert len(option_provider.chain_calls) == 2
    assert cache_reads == []


async def test_no_listed_options_is_saved_and_reused(option_db, option_provider):
    option_provider.listed = []
    first = await options.fetch_option_data("AAPL")
    assert first.status == "no_options"
    assert first.expiration_date is None
    assert first.calls == first.puts == []
    assert first.id is not None
    assert "No listed options" in first.summary
    second = await options.fetch_option_data("AAPL")
    assert second.id == first.id
    assert option_provider.ticker_calls == ["AAPL"]
    assert option_provider.chain_calls == []


@pytest.mark.parametrize("failure", ["list", "chain"])
async def test_provider_error_never_replaces_old_data_or_reuses_stale_data(
    option_db, option_provider, failure
):
    stale = options.OptionSnapshot(
        ticker="AAPL",
        expiration_date=option_provider.expiry,
        fetched_at=utcnow() - timedelta(hours=2),
        summary="stale",
    )
    old_id = await repository.insert_option_data(stale.to_row())
    option_provider.fail = failure
    assert await options.fetch_option_data("AAPL") is None
    history = await repository.fetch_option_data_history("AAPL")
    assert len(history) == 1 and history[0].id == old_id


async def test_timeout_does_not_save_a_snapshot(option_db, monkeypatch):
    monkeypatch.setattr(options, "YFINANCE_AVAILABLE", True)

    async def slow_thread(*args, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError("cancelled before this point")

    monkeypatch.setattr(options.asyncio, "to_thread", slow_thread)
    result = await options.fetch_option_data(
        "AAPL", settings=AnalyzerSettings(options_fetch_timeout_seconds=0.01)
    )
    assert result is None
    assert await repository.fetch_option_data_history("AAPL") == []


async def test_database_failure_returns_live_data_without_an_invalid_id(
    option_provider, monkeypatch
):
    async def unavailable(*args, **kwargs):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(repository, "fetch_latest_option_data", unavailable)
    monkeypatch.setattr(repository, "insert_option_data", unavailable)
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot is not None
    assert snapshot.id is None
    assert snapshot.calls and snapshot.puts
    assert "ATM call IV" in snapshot.summary


async def test_missing_price_does_not_fabricate_atm_metrics(option_db, option_provider):
    option_provider.price = None
    option_provider.underlying = {}
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.id is not None
    assert snapshot.spot_price is None
    assert snapshot.atm_call_iv is None and snapshot.atm_put_iv is None
    assert snapshot.high_iv_skew_bearish is False
    assert snapshot.put_call_oi_ratio == 2  # OI does not depend on price
    assert "Spot: N/A" in snapshot.summary


async def test_underlying_quote_is_a_price_fallback(option_db, option_provider):
    option_provider.price = None
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.spot_price == 100
    assert snapshot.atm_call_iv == pytest.approx(0.2)


async def test_missing_open_interest_is_unknown_not_a_partial_total(
    option_db, option_provider
):
    option_provider.calls.loc[1, "openInterest"] = float("nan")
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.total_call_open_interest is None
    assert snapshot.total_put_open_interest == 300
    assert snapshot.put_call_oi_ratio is None
    assert snapshot.calls[1]["openInterest"] is None


async def test_zero_call_open_interest_does_not_divide_by_zero(
    option_db, option_provider
):
    option_provider.calls["openInterest"] = 0
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.total_call_open_interest == 0
    assert snapshot.put_call_oi_ratio is None


async def test_missing_iv_does_not_become_a_zero_estimate(option_db, option_provider):
    option_provider.calls.loc[0, "impliedVolatility"] = float("nan")
    snapshot = await options.fetch_option_data("AAPL")
    assert snapshot.atm_call_iv is None
    assert snapshot.atm_put_iv == pytest.approx(0.3)
    assert snapshot.high_iv_skew_bearish is False
    assert "ATM call IV: N/A" in snapshot.summary


async def test_empty_chain_for_a_listed_expiry_is_not_cached(
    option_db, option_provider
):
    option_provider.calls = option_provider.calls.iloc[:0]
    option_provider.puts = option_provider.puts.iloc[:0]
    assert await options.fetch_option_data("AAPL") is None
    assert await repository.fetch_option_data_history("AAPL") == []


async def test_invalid_expirations_are_not_misreported_as_no_options(
    option_db, option_provider
):
    option_provider.listed = ["layout_changed", "2020-01-01"]
    assert await options.fetch_option_data("AAPL") is None
    assert await repository.fetch_option_data_history("AAPL") == []


async def test_missing_optional_provider_is_nonfatal(option_db, monkeypatch):
    monkeypatch.setattr(options, "YFINANCE_AVAILABLE", False)
    assert await options.fetch_option_data("UNKNOWN") is None


@pytest.mark.parametrize("ticker", ["", "  ", "A" * 21])
async def test_invalid_ticker_is_nonfatal_and_never_calls_yahoo(
    ticker, option_provider
):
    assert await options.fetch_option_data(ticker) is None
    assert option_provider.ticker_calls == []


async def test_cli_pulls_unique_tickers_and_reports_unsaved_data(monkeypatch, capsys):
    calls = []

    async def fake_pull(ticker, **kwargs):
        calls.append((ticker, kwargs["force_refresh"]))
        return options.OptionSnapshot(
            id=uuid4() if ticker == "AAPL" else None,
            ticker=ticker,
            status="no_options",
            summary="No listed options.",
        )

    monkeypatch.setattr(options, "fetch_option_data", fake_pull)
    assert await options._pull_tickers(["aapl", " AAPL ", "msft"], True) == 1
    assert calls == [("AAPL", True), ("MSFT", True)]
    output = capsys.readouterr().out
    assert "AAPL: snapshot=" in output
    assert "MSFT: snapshot=NOT SAVED" in output


def test_options_settings_can_be_tuned_from_environment(monkeypatch):
    monkeypatch.setenv("OPTIONS_CACHE_TTL_SECONDS", "900")
    monkeypatch.setenv("OPTIONS_FETCH_TIMEOUT_SECONDS", "12.5")
    settings = AnalyzerSettings(_env_file=None)
    assert settings.options_cache_ttl_seconds == 900
    assert settings.options_fetch_timeout_seconds == 12.5
