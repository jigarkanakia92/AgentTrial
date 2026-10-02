"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def classic_html() -> str:
    return (FIXTURES / "yahoo_topic_classic.html").read_text(encoding="utf-8")


@pytest.fixture
def redesign_html() -> str:
    return (FIXTURES / "yahoo_topic_redesign.html").read_text(encoding="utf-8")


@pytest.fixture
def rss_xml() -> str:
    return (FIXTURES / "sample_feed.xml").read_text(encoding="utf-8")


@pytest.fixture
async def option_db():
    """Real portable schema, with foreign keys enforced for snapshot links."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    from db import repository
    from db.models import Base

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    repository.override_engine(engine)
    try:
        yield engine
    finally:
        await repository.dispose_engine()


@pytest.fixture
def option_provider(monkeypatch):
    """Fake Yahoo I/O but use REAL pandas frames, including NaN/NaT values."""
    import pandas as pd

    import analyzer.options_data as module
    from db.models import utcnow

    expiry = utcnow().date() + timedelta(days=7)
    later = expiry + timedelta(days=7)
    state = SimpleNamespace(
        ticker_calls=[],
        chain_calls=[],
        fail=None,
        price=100.0,
        listed=[later.isoformat(), expiry.isoformat()],
        expiry=expiry,
        underlying={"regularMarketPrice": 100.0},
        calls=pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL_C100",
                    "strike": 100.0,
                    "lastPrice": 2.0,
                    "bid": 1.9,
                    "ask": 2.1,
                    "volume": 10,
                    "openInterest": 100,
                    "impliedVolatility": 0.20,
                    "inTheMoney": False,
                    "lastTradeDate": pd.Timestamp("2026-10-01T14:30:00Z"),
                    "newProviderField": "preserved",
                },
                {
                    "contractSymbol": "AAPL_C110",
                    "strike": 110.0,
                    "lastPrice": 1.0,
                    "bid": float("inf"),
                    "ask": 1.1,
                    "volume": float("nan"),
                    "openInterest": 50,
                    "impliedVolatility": 0.25,
                    "inTheMoney": False,
                    "lastTradeDate": pd.NaT,
                    "newProviderField": None,
                },
            ]
        ),
        puts=pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL_P100",
                    "strike": 100.0,
                    "lastPrice": 2.1,
                    "bid": 2.0,
                    "ask": 2.2,
                    "volume": 20,
                    "openInterest": 200,
                    "impliedVolatility": 0.30,
                    "inTheMoney": False,
                },
                {
                    "contractSymbol": "AAPL_P90",
                    "strike": 90.0,
                    "lastPrice": 1.1,
                    "bid": 1.0,
                    "ask": 1.2,
                    "volume": 5,
                    "openInterest": 100,
                    "impliedVolatility": 0.35,
                    "inTheMoney": False,
                },
            ]
        ),
    )

    class FakeTicker:
        def __init__(self, ticker):
            state.ticker_calls.append(ticker)

        @property
        def fast_info(self):
            return {"last_price": state.price}

        @property
        def options(self):
            if state.fail == "list":
                raise RuntimeError("Yahoo options endpoint unavailable")
            return state.listed

        def option_chain(self, expiry):
            state.chain_calls.append(expiry)
            if state.fail == "chain":
                raise RuntimeError("Yahoo options endpoint unavailable")
            return SimpleNamespace(
                calls=state.calls, puts=state.puts, underlying=state.underlying
            )

        @property
        def info(self):
            return {"shortName": "Fake Corp"}

    monkeypatch.setattr(module, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(module, "yf", SimpleNamespace(Ticker=FakeTicker))
    module._CACHE.clear()
    yield state
    module._CACHE.clear()
