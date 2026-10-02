"""Repository tests on in-memory SQLite (aiosqlite).

The repository's ON CONFLICT upserts are dialect-aware; these tests prove
the portable path (sqlite) while Postgres uses the same statements natively.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

import db.repository as repository
from db.models import Base, StockAnalysis


def article_row(headline: str, tickers: str | None = None) -> dict:
    url = f"https://finance.yahoo.com/news/{headline.lower().replace(' ', '-')}-1.html"
    import hashlib

    return {
        "headline": headline,
        "description": "desc",
        "url": url,
        "url_hash": hashlib.sha256(url.encode()).hexdigest(),
        "published_at": datetime.now(timezone.utc) - timedelta(hours=2),
        "tickers": tickers,
        "source": "Yahoo Finance",
        "parse_strategy": "css:test",
    }


@pytest.fixture
async def db():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    repository.override_engine(engine)
    yield engine
    await engine.dispose()
    # leave module-level state clean for other tests
    from db.repository import dispose_engine

    await dispose_engine()


async def test_insert_skips_duplicates(db):
    rows = [article_row("First headline"), article_row("Second headline")]
    inserted = await repository.insert_articles_skipping_existing(rows)
    assert inserted == 2

    # same URLs again + one new
    rows_again = [
        article_row("First headline"),
        article_row("Second headline"),
        article_row("Third headline"),
    ]
    inserted = await repository.insert_articles_skipping_existing(rows_again)
    assert inserted == 1
    assert await repository.count_articles() == 3


async def test_filter_existing_url_hashes(db):
    rows = [article_row("Alpha one"), article_row("Beta two")]
    await repository.insert_articles_skipping_existing(rows)
    existing = await repository.filter_existing_url_hashes([r["url_hash"] for r in rows])
    assert len(existing) == 2
    assert await repository.filter_existing_url_hashes(["nonexistent"]) == set()
    assert await repository.filter_existing_url_hashes([]) == set()


async def test_unprocessed_lifecycle(db):
    rows = [article_row("Process me"), article_row("Process me too")]
    await repository.insert_articles_skipping_existing(rows)

    since = datetime.now(timezone.utc) - timedelta(hours=16)
    pending = await repository.fetch_unprocessed_articles(since)
    assert len(pending) == 2

    await repository.mark_articles_processed([a.id for a in pending])
    assert await repository.fetch_unprocessed_articles(since) == []


async def test_unprocessed_window_filters_old_articles(db):
    row = article_row("Ancient news")
    row["published_at"] = datetime.now(timezone.utc) - timedelta(days=3)
    await repository.insert_articles_skipping_existing([row])

    since = datetime.now(timezone.utc) - timedelta(hours=16)
    assert await repository.fetch_unprocessed_articles(since) == []


async def test_upsert_stock_analysis_dedupes_per_day(db):
    payload = {
        "ticker": "AAPL",
        "stock_name": "Apple Inc.",
        "confidence_score": 72.5,
        "sentiment": "Positive",
        "swing_trading_candidate": True,
        "news_pointers": "[news] beat earnings",
        "option_data_analysis": None,
        "confidence_after_news_and_option": 65.0,
        "source_article_ids": ["id-1", "id-2"],
        "llm_persona_votes": {"news_fundamentalist": {"sentiment": "Positive"}},
    }
    await repository.upsert_stock_analysis(payload)
    await repository.upsert_stock_analysis({**payload, "confidence_score": 88.0})

    async with repository.get_engine().connect() as conn:
        result = await conn.execute(
            select(
                StockAnalysis.ticker,
                StockAnalysis.confidence_score,
                StockAnalysis.source_article_ids,
            )
        )
        rows = result.all()
    assert len(rows) == 1
    assert rows[0].ticker == "AAPL"
    assert float(rows[0].confidence_score) == 88.0
    assert rows[0].source_article_ids == ["id-1", "id-2"]


async def test_count_articles(db):
    assert await repository.count_articles() == 0
    await repository.insert_articles_skipping_existing([article_row("One"), article_row("Two")])
    assert await repository.count_articles() == 2
