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
    async def no_opts(ticker, **kwargs):
        return None

    async def no_name(ticker):
        return "Fake Corp"

    monkeypatch.setattr(pipeline_module, "fetch_option_data", no_opts)
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
        pipeline_module, "fetch_option_data", _async_none
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


async def test_option_chain_is_saved_linked_and_reused_by_analysis(
    option_db, option_provider, monkeypatch
):
    seen_options = []

    async def fake_ask(self, persona_key, ticker, news_bundle, option_data=None):
        seen_options.append(option_data)
        return PersonaVerdict.model_validate(VERDICTS[persona_key])

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", fake_ask)
    await _seed(option_db, "Apple earnings with option chain", "AAPL")
    settings = AnalyzerSettings(llm_api_key="k")
    summary = await run_analysis_job(settings)
    assert summary.tickers_analyzed == 1 and summary.tickers_failed == 0
    snapshot = await repository.fetch_latest_option_data("AAPL")
    assert snapshot is not None and snapshot.calls and snapshot.puts
    assert len(seen_options) == 3
    assert all(value == snapshot.summary for value in seen_options)
    async with option_db.connect() as connection:
        assert await connection.scalar(select(StockAnalysis.option_data_id)) == snapshot.id
        confidence = await connection.scalar(select(StockAnalysis.confidence_after_news_and_option))
        assert float(confidence) == round(round((80 + 60 + 90) / 3, 2) * 0.85, 2)

    # New news inside the TTL reuses the SAME saved chain, not another pull.
    await _seed(option_db, "Apple followup news", "AAPL")
    assert (await run_analysis_job(settings)).tickers_analyzed == 1
    assert len(option_provider.chain_calls) == 1
    assert len(await repository.fetch_option_data_history("AAPL")) == 1


async def test_option_write_failure_does_not_break_news_analysis(
    option_db, option_provider, monkeypatch
):
    async def fake_ask(self, persona_key, ticker, news_bundle, option_data=None):
        assert option_data is not None
        return PersonaVerdict.model_validate(VERDICTS[persona_key])

    async def failing_write(data):
        raise RuntimeError("option data write temporarily unavailable")

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", fake_ask)
    monkeypatch.setattr(repository, "insert_option_data", failing_write)
    await _seed(option_db, "Apple news despite option DB issue", "AAPL")
    summary = await run_analysis_job(AnalyzerSettings(llm_api_key="k"))
    assert summary.tickers_analyzed == 1 and summary.error is None
    assert await repository.fetch_option_data_history("AAPL") == []
    async with option_db.connect() as connection:
        assert await connection.scalar(select(StockAnalysis.option_data_id)) is None
        assert await connection.scalar(select(StockAnalysis.ticker)) == "AAPL"


async def test_yahoo_options_failure_keeps_news_only_analysis_working(
    option_db, option_provider, monkeypatch
):
    option_provider.fail = "chain"

    async def fake_ask(self, persona_key, ticker, news_bundle, option_data=None):
        assert option_data is None
        return PersonaVerdict.model_validate(VERDICTS[persona_key])

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", fake_ask)
    await _seed(option_db, "Apple news despite Yahoo failure", "AAPL")
    summary = await run_analysis_job(AnalyzerSettings(llm_api_key="k"))
    assert summary.tickers_analyzed == 1 and summary.articles_marked_processed == 1
    assert await repository.fetch_option_data_history("AAPL") == []


async def test_options_are_retained_even_if_all_llm_personas_fail(
    option_db, option_provider, monkeypatch
):
    from analyzer.llm_client import PersonaError

    async def failing_ask(self, *args, **kwargs):
        raise PersonaError("LLM provider unavailable")

    monkeypatch.setattr(pipeline_module.LLMClient, "ask_persona", failing_ask)
    await _seed(option_db, "Apple chain independent of LLM result", "AAPL")
    summary = await run_analysis_job(AnalyzerSettings())
    assert summary.tickers_failed == 1 and summary.articles_marked_processed == 0
    snapshot = await repository.fetch_latest_option_data("AAPL")
    assert snapshot is not None and snapshot.calls and snapshot.puts
    since = datetime.now(timezone.utc) - timedelta(hours=16)
    assert len(await repository.fetch_unprocessed_articles(since)) == 1
