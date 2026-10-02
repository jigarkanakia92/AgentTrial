"""End-to-end scraper cycle tests with fully faked HTTP + DB.

Proves the resilience contract: fetch failures, challenge pages, DOM drift
and DB outages all end in a clean RunReport — never an exception out of
``run()``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import db.repository as repository
import scraper.yahoo_news_scraper as yns
from db.models import Base
from scraper.config import ScraperSettings
from scraper.fetchers import (
    FetchOutcome,
    PageFetchError,
)
from scraper.yahoo_news_scraper import StopReason, YahooNewsScraper


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


def make_scraper(settings: ScraperSettings | None = None) -> YahooNewsScraper:
    scraper = YahooNewsScraper(settings or ScraperSettings())
    # Never construct real HTTP backends in tests.
    scraper.fetcher.start = _noop
    scraper.fetcher.close = _noop
    return scraper


async def _noop(*args, **kwargs):
    return None


def fake_get(pages: dict[int, object]):
    """pages: page number -> html | Exception"""

    async def _get(url: str) -> FetchOutcome:
        # extract page number from .../{page}/
        page = int(url.rstrip("/").split("/")[-1])
        item = pages.get(page, "")
        if isinstance(item, Exception):
            raise item
        return FetchOutcome(url, item or "", "fake", 200)

    return _get


CLASSIC = (
    open("tests/fixtures/yahoo_topic_classic.html", encoding="utf-8").read()
)


async def test_happy_backfill_run(db, monkeypatch):
    """Empty DB: pages 1..max are fetched, each with distinct content."""
    settings = ScraperSettings(max_pages_safety=5, backfill_days=2, selectors_file="missing.yaml")
    scraper = make_scraper(settings)

    def page_html(page: int) -> str:
        # give every page unique article URLs
        return CLASSIC.replace(".html", f"-p{page}.html")

    async def get_fn(url: str) -> FetchOutcome:
        page = int(url.rstrip("/").split("/")[-1])
        return FetchOutcome(url, page_html(page), "fake", 200)

    monkeypatch.setattr(scraper.fetcher, "get", get_fn)

    report = await scraper.run()
    assert report.stop_reason == StopReason.COMPLETED
    assert report.pages_fetched == 5           # empty DB: backfills all pages
    assert report.articles_new == 25
    assert await repository.count_articles() == 25


async def test_incremental_stop_on_known_article(db, monkeypatch):
    # Pre-seed DB with an article that appears on page 1.
    import hashlib

    known_url = "https://finance.yahoo.com/news/apple-unveils-m4-chip-123045678.html"
    await repository.insert_articles_skipping_existing(
        [
            {
                "headline": "Apple unveils next-gen M5 chip as its AI push accelerates (AAPL)",
                "url": known_url,
                "url_hash": hashlib.sha256(known_url.encode()).hexdigest(),
                "published_at": datetime.now(timezone.utc),
            }
        ]
    )

    settings = ScraperSettings(max_pages_safety=10, selectors_file="missing.yaml")
    scraper = make_scraper(settings)
    fetched: list[int] = []

    async def tracking_get(url: str) -> FetchOutcome:
        page = int(url.rstrip("/").split("/")[-1])
        fetched.append(page)
        return FetchOutcome(url, CLASSIC, "fake", 200)

    monkeypatch.setattr(scraper.fetcher, "get", tracking_get)
    report = await scraper.run()

    assert report.stop_reason == StopReason.INCREMENTAL
    assert fetched == [1]                       # stopped after first page
    assert report.articles_new == 4             # 4 new, 1 known duplicate
    assert report.duplicates_skipped == 1


async def test_fetch_failures_end_cycle_cleanly(db, monkeypatch):
    settings = ScraperSettings(
        max_pages_safety=10,
        max_consecutive_fetch_failures=3,
        selectors_file="missing.yaml",
    )
    scraper = make_scraper(settings)
    monkeypatch.setattr(
        scraper.fetcher,
        "get",
        fake_get({i: PageFetchError("down") for i in range(1, 10)}),
    )
    report = await scraper.run()

    assert report.stop_reason == StopReason.FETCH_FAILURES
    assert report.pages_failed == 3
    assert report.pages_fetched == 0


async def test_dom_drift_suspected_when_pages_go_empty(db, monkeypatch):
    """Yahoo redesigns AND json-ld/rss/heuristics all fail -> drift alert,
    cycle ends politely, nothing is written."""
    settings = ScraperSettings(
        max_pages_safety=10,
        max_consecutive_empty_pages=3,
        selectors_file="missing.yaml",
    )
    scraper = make_scraper(settings)

    async def no_rss(fetch):
        return []

    async def get_fn(url: str) -> FetchOutcome:
        return FetchOutcome(url, "<html><body><p>challenge-ish junk</p></body></html>", "fake", 200)

    monkeypatch.setattr(scraper.fetcher, "get", get_fn)
    monkeypatch.setattr(
        yns.NewsParser, "parse_rss_feeds", staticmethod(no_rss), raising=False
    )

    report = await scraper.run()
    assert report.stop_reason == StopReason.NO_NEW_ARTICLES
    assert report.selector_drift_suspected is True
    assert report.articles_new == 0
    assert await repository.count_articles() == 0


async def test_circuit_open_stops_cycle(db, monkeypatch):
    from scraper.circuit_breaker import CircuitOpenError

    settings = ScraperSettings(selectors_file="missing.yaml")
    scraper = make_scraper(settings)
    calls = {"n": 0}

    async def get_fn(url: str) -> FetchOutcome:
        calls["n"] += 1
        raise CircuitOpenError("open")

    monkeypatch.setattr(scraper.fetcher, "get", get_fn)
    report = await scraper.run()
    assert report.stop_reason == StopReason.CIRCUIT_OPEN
    assert report.circuit_opened is True


async def test_db_outage_ends_cycle_cleanly(monkeypatch):
    """No DB at all: run() returns a report, raises nothing."""
    async def dead_count():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(repository, "count_articles", dead_count)
    scraper = make_scraper(ScraperSettings(selectors_file="missing.yaml"))

    async def get_fn(url: str) -> FetchOutcome:
        return FetchOutcome(url, CLASSIC, "fake", 200)

    monkeypatch.setattr(scraper.fetcher, "get", get_fn)
    report = await scraper.run()
    assert report.stop_reason == StopReason.DB_UNAVAILABLE
    assert "connection refused" in (report.error or "")
