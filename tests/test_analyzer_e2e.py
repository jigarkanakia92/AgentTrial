"""Analyzer cycle integration test: seeded DB + faked LLM -> stock_analysis.

Runs the REAL pipeline (grouping, options degradation, aggregation, upsert,
processed-marking) with only the LLM network call faked.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import analyzer.pipeline as pipeline_module
import db.repository as repository
from analyzer.config import AnalyzerSettings
from analyzer.pipeline import run_analysis_job
from analyzer.schemas import PersonaVerdict
from db.models import Base, StockAnalysis

VERDICTS = {
    "news_fundamentalist": {
        "sentiment": "Positive", "confidence_score": 80,
        "swing_trading_candidate": False, "news_pointers": ["earnings beat"],
        "option_commentary": None,
    },
    "risk_manager": {
        "sentiment": "Positive", "confidence_score": 60,
        "swing_trading_candidate": False, "news_pointers": ["valuation rich"],
        "option_commentary": None,
    },
    "swing_trader": {
        "sentiment": "Positive", "confidence_score": 90,
        "swing_trading_candidate": True, "news_pointers": ["momentum intact"],
        "option_commentary": "Call OI heavy at ATM.",
    },
}


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    repository.override_engine(engine)
    yield engine
    from db.repository import dispose_engine

    await dispose_engine()


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace LLMClient.ask_persona with canned verdicts."""
    calls: list[tuple[str, str]] = []

    async def fake_ask(self, persona_key, ticker, news_bundle, option_data=None):
        calls.append((persona_key, ticker))
        return PersonaVerdict.model_validate(VERDICTS[persona_key])

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", fake_ask)
    # options data "unavailable" — pipeline must continue without it
    async def no_opts(ticker):
        return None

    async def no_name(ticker):
        return "Fake Corp"

    monkeypatch.setattr(pipeline_module, "fetch_option_summary", no_opts)
    monkeypatch.setattr(pipeline_module, "fetch_stock_name", no_name)
    return calls


async def _seed(db, headline, tickers, hours_ago=2):
    import hashlib

    url = f"https://finance.yahoo.com/news/{headline.lower().replace(' ', '-')}.html"
    await repository.insert_articles_skipping_existing(
        [
            {
                "headline": headline,
                "url": url,
                "url_hash": hashlib.sha256(url.encode()).hexdigest(),
                "published_at": datetime.now(timezone.utc) - timedelta(hours=hours_ago),
                "tickers": tickers,
            }
        ]
    )


async def test_full_analysis_cycle(db, fake_llm):
    await _seed(db, "Apple beats earnings", "AAPL,MSFT")
    await _seed(db, "Tesla cuts prices", "TSLA")
    await _seed(db, "No ticker in this one", None)

    settings = AnalyzerSettings(llm_api_key="k", max_concurrent_llm=2)
    summary = await run_analysis_job(settings)

    # 3 tickers analyzed (AAPL & MSFT share an article), nothing failed
    assert summary.tickers_analyzed == 3
    assert summary.tickers_failed == 0
    assert summary.articles_fetched == 3
    assert summary.unattributed_articles == 1
    assert summary.articles_marked_processed == 3

    # all three personas called for each ticker
    assert len(fake_llm) == 9

    # analysis rows exist and aggregate correctly
    async with repository.get_engine().connect() as conn:
        result = await conn.execute(
            select(
                StockAnalysis.ticker,
                StockAnalysis.stock_name,
                StockAnalysis.sentiment,
                StockAnalysis.confidence_score,
                StockAnalysis.swing_trading_candidate,
                StockAnalysis.llm_persona_votes,
                StockAnalysis.news_pointers,
            )
        )
        by_ticker = {r.ticker: r for r in result.all()}

    assert set(by_ticker) == {"AAPL", "MSFT", "TSLA"}
    aapl = by_ticker["AAPL"]
    assert aapl.sentiment == "Positive"
    assert float(aapl.confidence_score) == round((80 + 60 + 90) / 3, 2)
    assert aapl.swing_trading_candidate is False          # 1/3 votes only
    assert set(aapl.llm_persona_votes) == {
        "news_fundamentalist", "risk_manager", "swing_trader",
    }
    assert "[swing_trader] momentum intact" in aapl.news_pointers
    assert aapl.stock_name == "Fake Corp"

    # re-run: nothing unprocessed left, no new rows
    summary2 = await run_analysis_job(settings)
    assert summary2.articles_fetched == 0
    assert summary2.tickers_analyzed == 0


async def test_all_personas_failing_leaves_articles_unprocessed(db, monkeypatch):
    from analyzer.llm_client import PersonaError

    async def failing_ask(self, persona_key, ticker, news_bundle, option_data=None):
        raise PersonaError("provider down")

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", failing_ask)
    monkeypatch.setattr(
        pipeline_module, "fetch_option_summary", _async_none
    )
    monkeypatch.setattr(
        pipeline_module, "fetch_stock_name", _async_none
    )

    await _seed(db, "Intel guides lower", "INTC")
    summary = await run_analysis_job(AnalyzerSettings(llm_api_key="k"))

    assert summary.tickers_analyzed == 0
    assert summary.tickers_failed == 1
    assert summary.articles_marked_processed == 0  # retried next cycle

    since = datetime.now(timezone.utc) - timedelta(hours=16)
    assert len(await repository.fetch_unprocessed_articles(since)) == 1


async def _async_none(*a, **k):
    return None
