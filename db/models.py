"""SQLAlchemy 2.0 ORM models.

News, analysis, and persisted options-chain snapshots:

* ``news_articles``  — raw scrape output (deduped by ``url_hash``)
* ``stock_analysis`` — one aggregated LLM verdict per (day, ticker)
* ``option_data``    — historical nearest-expiry option chains per ticker

Portability note: JSON columns use :data:`FLEXIBLE_JSON` — native ``JSONB``
on PostgreSQL, plain ``JSON`` elsewhere (e.g. SQLite in local dev/tests).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
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
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
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


class OptionData(Base):
    """One immutable option-chain snapshot for a ticker at a fetch time.

    ``calls`` and ``puts`` preserve every provider column as JSON records.
    IV values are fractions (0.25 = 25%); timestamps are UTC. A provider-reported
    ``no_options`` result is stored too, but caught provider failures are not.
    """

    __tablename__ = "option_data"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(100), default="Yahoo Finance")
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    expiration_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    spot_price: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    atm_call_iv: Mapped[float | None] = mapped_column(Numeric(12, 8), nullable=True)
    atm_put_iv: Mapped[float | None] = mapped_column(Numeric(12, 8), nullable=True)
    total_call_open_interest: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    total_put_open_interest: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    put_call_oi_ratio: Mapped[float | None] = mapped_column(
        Numeric(18, 8), nullable=True
    )
    high_iv_skew_bearish: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    available_expirations: Mapped[list] = mapped_column(FLEXIBLE_JSON, default=list)
    calls: Mapped[list] = mapped_column(FLEXIBLE_JSON, default=list)
    puts: Mapped[list] = mapped_column(FLEXIBLE_JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        # This unique index also serves latest/history lookups by ticker.
        UniqueConstraint(
            "ticker", "fetched_at", name="uq_option_data_ticker_fetched_at"
        ),
        Index("ix_option_data_ticker_expiration", "ticker", "expiration_date"),
        CheckConstraint("status IN ('ok', 'no_options')", name="ck_option_data_status"),
        CheckConstraint(
            "(status = 'ok' AND expiration_date IS NOT NULL) OR "
            "(status = 'no_options' AND expiration_date IS NULL)",
            name="ck_option_data_expiration_status",
        ),
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
    option_data_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey(
            "option_data.id", name="fk_analysis_option_data", ondelete="SET NULL"
        ),
        nullable=True,
        index=True,
    )
    option_data_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_after_news_and_option: Mapped[float] = mapped_column(Numeric(5, 2))
    source_article_ids: Mapped[list] = mapped_column(FLEXIBLE_JSON, default=list)
    llm_persona_votes: Mapped[dict] = mapped_column(FLEXIBLE_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("analysis_date", "ticker", name="uq_analysis_date_ticker"),
    )
