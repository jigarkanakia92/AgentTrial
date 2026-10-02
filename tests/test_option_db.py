"""Durable per-ticker snapshots, history, idempotency, and analysis lineage."""

from __future__ import annotations

from datetime import timedelta, timezone

import pytest
from sqlalchemy import delete, select

from analyzer.options_data import OptionSnapshot
from db import repository
from db.models import OptionData, StockAnalysis, utcnow


def snapshot_row(ticker="AAPL", *, hours_ago=0, source="Yahoo Finance"):
    return OptionSnapshot(
        ticker=ticker,
        source=source,
        fetched_at=utcnow() - timedelta(hours=hours_ago),
        expiration_date=utcnow().date() + timedelta(days=7),
        spot_price=100,
        atm_call_iv=0.2,
        atm_put_iv=0.3,
        calls=[{"contractSymbol": "CALL", "strike": 100, "openInterest": 10}],
        puts=[{"contractSymbol": "PUT", "strike": 100, "openInterest": 20}],
        summary="option summary",
    ).to_row()


async def test_snapshot_insert_is_idempotent_and_does_not_overwrite(option_db):
    row = snapshot_row(" aapl ")
    first_id = await repository.insert_option_data(row)
    second_id = await repository.insert_option_data(
        {**row, "summary": "should not replace original"}
    )
    assert first_id == second_id
    history = await repository.fetch_option_data_history(" aapl ")
    assert len(history) == 1
    assert history[0].ticker == "AAPL"
    assert history[0].summary == "option summary"
    assert history[0].calls[0]["contractSymbol"] == "CALL"


async def test_latest_and_history_are_ticker_scoped_and_newest_first(option_db):
    old_id = await repository.insert_option_data(snapshot_row(hours_ago=2))
    new_id = await repository.insert_option_data(snapshot_row(hours_ago=1))
    await repository.insert_option_data(snapshot_row("MSFT"))
    assert (await repository.fetch_latest_option_data("aapl")).id == new_id
    assert await repository.fetch_latest_option_data("MISSING") is None
    assert [row.id for row in await repository.fetch_option_data_history("AAPL")] == [
        new_id,
        old_id,
    ]
    assert [
        row.id for row in await repository.fetch_option_data_history("AAPL", limit=1)
    ] == [new_id]
    assert len(await repository.fetch_option_data_history("msft")) == 1


async def test_latest_snapshot_respects_freshness_and_source_filters(option_db):
    yahoo_id = await repository.insert_option_data(snapshot_row(hours_ago=1))
    other_id = await repository.insert_option_data(snapshot_row(source="Other feed"))
    assert (await repository.fetch_latest_option_data("AAPL")).id == other_id
    assert (
        await repository.fetch_latest_option_data("AAPL", source="Yahoo Finance")
    ).id == yahoo_id
    assert (
        await repository.fetch_latest_option_data(
            "AAPL", source="Yahoo Finance", since=utcnow() - timedelta(minutes=30)
        )
        is None
    )


async def test_utc_conversion_dedupes_equivalent_fetch_times(option_db):
    row = snapshot_row()
    eastern_offset = timezone(timedelta(hours=-4))
    same_time = row["fetched_at"].astimezone(eastern_offset)
    first_id = await repository.insert_option_data(row)
    second_id = await repository.insert_option_data({**row, "fetched_at": same_time})
    assert first_id == second_id
    assert len(await repository.fetch_option_data_history("AAPL")) == 1


async def test_no_options_snapshot_allows_null_expiration(option_db):
    snapshot_id = await repository.insert_option_data(
        OptionSnapshot(
            ticker="AAPL", status="no_options", summary="No listed options."
        ).to_row()
    )
    saved = await repository.fetch_latest_option_data("AAPL")
    assert saved.id == snapshot_id
    assert saved.status == "no_options" and saved.expiration_date is None
    assert saved.calls == saved.puts == []


@pytest.mark.parametrize("ticker", ["", " ", "X" * 21])
async def test_invalid_tickers_are_rejected(option_db, ticker):
    with pytest.raises(ValueError):
        await repository.fetch_latest_option_data(ticker)


async def test_history_requires_positive_limit(option_db):
    with pytest.raises(ValueError):
        await repository.fetch_option_data_history("AAPL", limit=0)


async def test_analysis_links_snapshot_and_survives_its_deletion(option_db):
    option_id = await repository.insert_option_data(snapshot_row())
    payload = {
        "ticker": "AAPL",
        "confidence_score": 70,
        "sentiment": "Positive",
        "swing_trading_candidate": True,
        "news_pointers": ["earnings"],
        "confidence_after_news_and_option": 59.5,
        "option_data_id": option_id,
    }
    await repository.upsert_stock_analysis(payload)
    async with option_db.begin() as connection:
        linked = await connection.scalar(select(StockAnalysis.option_data_id))
        assert linked == option_id
        await connection.execute(delete(OptionData).where(OptionData.id == option_id))
    async with option_db.connect() as connection:
        assert await connection.scalar(select(StockAnalysis.option_data_id)) is None
        assert await connection.scalar(select(StockAnalysis.ticker)) == "AAPL"


async def test_same_day_analysis_updates_snapshot_reference(option_db):
    first = await repository.insert_option_data(snapshot_row(hours_ago=1))
    second = await repository.insert_option_data(snapshot_row())
    payload = {
        "ticker": "AAPL",
        "confidence_score": 70,
        "sentiment": "Positive",
        "news_pointers": "news",
        "confidence_after_news_and_option": 70,
    }
    await repository.upsert_stock_analysis({**payload, "option_data_id": first})
    await repository.upsert_stock_analysis({**payload, "option_data_id": second})
    async with option_db.connect() as connection:
        assert (
            await connection.execute(select(StockAnalysis.option_data_id))
        ).scalars().all() == [second]


async def test_freshness_cutoff_is_normalized_to_utc(option_db):
    row = snapshot_row(hours_ago=1)
    await repository.insert_option_data(row)
    eastern_offset = timezone(timedelta(hours=-4))
    cutoff = (utcnow() - timedelta(minutes=30)).astimezone(eastern_offset)
    assert await repository.fetch_latest_option_data("AAPL", since=cutoff) is None
