"""SQLAlchemy 2.0 ORM models.

Two tables, exactly as specified in yahooscraper.md:

* ``news_articles``  — raw scrape output (deduped by ``url_hash``)
* ``stock_analysis`` — one aggregated LLM verdict per (day, ticker)

Portability note: JSON columns use :data:`FLEXIBLE_JSON` — native ``JSONB``
on PostgreSQL, plain ``JSON`` elsewhere (e.g. SQLite in local dev/tests).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

FLEXIBLE_JSON = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """Timezone-aware UTC now (naive-UTC `datetime.utcnow` is deprecated)."""
    return datetime.now(timezone.utc)


def utc_midnight(now: datetime | None = None) -> datetime:
    """Analysis dates are normalized to UTC midnight so the unique
    constraint (analysis_date, ticker) deduplicates re-runs within a day."""
    now = now or utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


class Base(DeclarativeBase):
    pass


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(100), default="Yahoo Finance")
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=utcnow
    )
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    headline: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    # Dedup key: sha256 of the normalized (query/fragment-stripped) URL.
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tickers: Mapped[str | None] = mapped_column(Text, nullable=True)  # "AAPL,MSFT"
    raw_html_cache: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    # Observability: which parse strategy captured this article.
    parse_strategy: Mapped[str | None] = mapped_column(String(40), nullable=True)

    __table_args__ = (
        Index("ix_news_published_processed", "published_at", "processed"),
    )


class StockAnalysis(Base):
    __tablename__ = "stock_analysis"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=utc_midnight
    )
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    stock_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 2))  # 0-100
    sentiment: Mapped[str] = mapped_column(String(20))  # Positive/Negative/Neutral
    swing_trading_candidate: Mapped[bool] = mapped_column(Boolean, default=False)
    news_pointers: Mapped[str] = mapped_column(Text, default="")
    option_data_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_after_news_and_option: Mapped[float] = mapped_column(Numeric(5, 2))
    source_article_ids: Mapped[list] = mapped_column(FLEXIBLE_JSON, default=list)
    llm_persona_votes: Mapped[dict] = mapped_column(FLEXIBLE_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("analysis_date", "ticker", name="uq_analysis_date_ticker"),
    )
