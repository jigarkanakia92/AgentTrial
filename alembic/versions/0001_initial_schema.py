"""initial schema: news_articles + stock_analysis

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-02
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

FLEXIBLE_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "news_articles",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("url_hash", sa.String(64), nullable=False),
        sa.Column("tickers", sa.Text(), nullable=True),
        sa.Column("raw_html_cache", sa.Text(), nullable=True),
        sa.Column("processed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("parse_strategy", sa.String(40), nullable=True),
        sa.UniqueConstraint("url_hash", name="uq_news_url_hash"),
    )
    op.create_index("ix_news_published_at", "news_articles", ["published_at"])
    op.create_index("ix_news_url_hash", "news_articles", ["url_hash"])
    op.create_index(
        "ix_news_published_processed", "news_articles", ["published_at", "processed"]
    )

    op.create_table(
        "stock_analysis",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("analysis_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("stock_name", sa.String(200), nullable=True),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("sentiment", sa.String(20), nullable=False),
        sa.Column("swing_trading_candidate", sa.Boolean(), nullable=False),
        sa.Column("news_pointers", sa.Text(), nullable=False),
        sa.Column("option_data_analysis", sa.Text(), nullable=True),
        sa.Column("confidence_after_news_and_option", sa.Numeric(5, 2), nullable=False),
        sa.Column("source_article_ids", FLEXIBLE_JSON, nullable=False),
        sa.Column("llm_persona_votes", FLEXIBLE_JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("analysis_date", "ticker", name="uq_analysis_date_ticker"),
    )
    op.create_index("ix_analysis_date", "stock_analysis", ["analysis_date"])
    op.create_index("ix_analysis_ticker", "stock_analysis", ["ticker"])


def downgrade() -> None:
    op.drop_table("stock_analysis")
    op.drop_table("news_articles")
