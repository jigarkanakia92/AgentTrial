"""Async repository layer — the ONLY code that touches the database.

Design rules (both services depend on this, so it must be boring):

* Engine/session factory are created **lazily** — importing this module never
  connects to anything (keeps unit tests and container startup happy).
* All writes are idempotent (``ON CONFLICT DO NOTHING/UPDATE``) so a scraper
  or analyzer crash-and-retry can never create duplicates.
* Transient DB failures (connection resets, failovers) are retried with
  exponential backoff before surfacing.
* Dialect-aware upserts: native ON CONFLICT for PostgreSQL and SQLite,
  row-by-row fallback anywhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Iterable, Sequence

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.config import DatabaseSettings
from db.models import NewsArticle, StockAnalysis, utc_midnight, utcnow

# ---------------------------------------------------------------------------
# Lazy engine management
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        url = DatabaseSettings().database_url
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            kwargs.update(pool_size=10, max_overflow=5)
        _engine = create_async_engine(url, **kwargs)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        logger.debug("Database engine initialised for {}", url.split("@")[-1])
    return _engine


def override_engine(engine: AsyncEngine) -> None:
    """Testing hook: swap in a test engine (SQLite in-memory, etc.)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def _sf() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _session_factory is not None
    return _session_factory


async def wait_for_db(max_attempts: int = 30, delay: float = 2.0) -> bool:
    """Block until the DB accepts connections (containers race at startup).

    Returns True when ready; logs and returns False after exhausting
    attempts instead of raising — schedulers simply try again next cycle.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            async with _sf()() as session:
                await session.execute(select(1))
            return True
        except SQLAlchemyError as exc:
            logger.warning(
                "DB not ready (attempt {}/{}): {}", attempt, max_attempts, exc
            )
            if attempt < max_attempts:
                import asyncio

                await asyncio.sleep(delay)
    logger.error("Database unreachable after {} attempts", max_attempts)
    return False


_RETRYABLE = (DBAPIError, OperationalError)
_db_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)


# ---------------------------------------------------------------------------
# news_articles
# ---------------------------------------------------------------------------


@_db_retry
async def count_articles() -> int:
    """Exact row count of news_articles (cheap: dedup keeps it modest)."""
    from sqlalchemy import func

    async with _sf()() as session:
        result = await session.execute(select(func.count()).select_from(NewsArticle))
        return int(result.scalar_one())


@_db_retry
async def filter_existing_url_hashes(hashes: Sequence[str]) -> set[str]:
    """Return the subset of ``hashes`` already present in news_articles."""
    if not hashes:
        return set()
    existing: set[str] = set()
    async with _sf()() as session:
        # chunk to stay well under parameter limits
        for chunk_start in range(0, len(hashes), 500):
            chunk = list(hashes)[chunk_start : chunk_start + 500]
            result = await session.execute(
                select(NewsArticle.url_hash).where(NewsArticle.url_hash.in_(chunk))
            )
            existing.update(result.scalars().all())
    return existing


def _on_conflict_insert(table, rows: list[dict]):
    """Dialect-aware INSERT ... ON CONFLICT DO NOTHING."""
    dialect = get_engine().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(table).values(rows)
        return stmt.on_conflict_do_nothing(index_elements=["url_hash"])
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        stmt = insert(table).values(rows)
        return stmt.on_conflict_do_nothing(index_elements=["url_hash"])
    return None  # generic fallback handled by caller


@_db_retry
async def insert_articles_skipping_existing(articles: list[dict]) -> int:
    """Insert scraped articles; silently skip URL-hash collisions.

    Every dict must already carry ``url_hash``. Returns the number of rows
    actually inserted. One malformed row never aborts the batch: the whole
    batch is attempted as a conflict-safe statement first, and if even that
    fails (schema drift, constraint surprises) we fall back to row-by-row
    inserts so one bad row only sacrifices itself.
    """
    if not articles:
        return 0
    inserted = 0
    async with _sf()() as session:
        stmt = _on_conflict_insert(NewsArticle, articles)
        if stmt is not None:
            try:
                result = await session.execute(stmt)
                await session.commit()
                return int(result.rowcount or 0)
            except SQLAlchemyError as exc:
                logger.warning("Batch insert failed ({}); falling back to row-by-row", exc)
                await session.rollback()

        for row in articles:
            try:
                session.add(NewsArticle(**row))
                await session.commit()
                inserted += 1
            except SQLAlchemyError as exc:
                logger.debug("Skipped article {} — {}", row.get("url_hash", "?"), exc)
                await session.rollback()
    return inserted


@_db_retry
async def fetch_unprocessed_articles(
    since: datetime, limit: int | None = None
) -> list[NewsArticle]:
    async with _sf()() as session:
        stmt = select(NewsArticle).where(
            NewsArticle.published_at >= since,
            NewsArticle.processed.is_(False),
        )
        stmt = stmt.order_by(NewsArticle.published_at.desc())
        if limit:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


@_db_retry
async def mark_articles_processed(ids: Iterable[uuid.UUID]) -> int:
    ids = list(ids)
    if not ids:
        return 0
    async with _sf()() as session:
        result = await session.execute(
            update(NewsArticle)
            .where(NewsArticle.id.in_(ids))
            .values(processed=True)
        )
        await session.commit()
        return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# stock_analysis
# ---------------------------------------------------------------------------


def _analysis_row(data: dict) -> dict:
    """Map an analyzer result dict onto StockAnalysis columns safely."""
    allowed = {
        "ticker",
        "stock_name",
        "confidence_score",
        "sentiment",
        "swing_trading_candidate",
        "news_pointers",
        "option_data_analysis",
        "confidence_after_news_and_option",
        "source_article_ids",
        "llm_persona_votes",
    }
    row = {k: v for k, v in data.items() if k in allowed}
    row.setdefault("analysis_date", utc_midnight())
    row.setdefault("source_article_ids", [])
    row.setdefault("llm_persona_votes", {})
    # news_pointers may arrive as a list from the aggregator; the column is
    # Text — coerce at the boundary (detail remains in llm_persona_votes).
    if isinstance(row.get("news_pointers"), (list, tuple)):
        row["news_pointers"] = "\n".join(str(p) for p in row["news_pointers"])
    return row


_ANALYSIS_MUTABLE_COLUMNS = (
    "stock_name",
    "confidence_score",
    "sentiment",
    "swing_trading_candidate",
    "news_pointers",
    "option_data_analysis",
    "confidence_after_news_and_option",
    "source_article_ids",
    "llm_persona_votes",
    "updated_at",
)


@_db_retry
async def upsert_stock_analysis(data: dict) -> None:
    """Insert or refresh today's analysis row for one ticker.

    ``analysis_date`` is normalized to UTC midnight so re-running the
    analyzer on the same day updates the existing row (per the unique
    constraint) instead of duplicating it.
    """
    row = _analysis_row(data)
    row["analysis_date"] = utc_midnight()
    row["updated_at"] = utcnow()

    async with _sf()() as session:
        dialect = get_engine().dialect.name
        stmt = None
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert

            stmt = insert(StockAnalysis).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=[StockAnalysis.analysis_date, StockAnalysis.ticker],
                set_={
                    col: getattr(stmt.excluded, col)
                    for col in _ANALYSIS_MUTABLE_COLUMNS
                },
            )
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert

            stmt = insert(StockAnalysis).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=[StockAnalysis.analysis_date, StockAnalysis.ticker],
                set_={
                    col: getattr(stmt.excluded, col)
                    for col in _ANALYSIS_MUTABLE_COLUMNS
                },
            )

        if stmt is not None:
            await session.execute(stmt)
        else:  # generic dialects: merge-by-select fallback
            existing = await session.execute(
                select(StockAnalysis).where(
                    StockAnalysis.analysis_date == row["analysis_date"],
                    StockAnalysis.ticker == row["ticker"],
                )
            )
            obj = existing.scalars().first()
            if obj is None:
                session.add(StockAnalysis(**row))
            else:
                for col in _ANALYSIS_MUTABLE_COLUMNS:
                    setattr(obj, col, row.get(col, getattr(obj, col)))
        await session.commit()
