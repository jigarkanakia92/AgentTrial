"""persist per-ticker option-chain snapshots and link analyses

Revision ID: 0002_option_data
Revises: 0001_initial
Create Date: 2026-10-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_option_data"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

FLEXIBLE_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "option_data",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expiration_date", sa.Date(), nullable=True),
        sa.Column("spot_price", sa.Numeric(18, 6), nullable=True),
        sa.Column("atm_call_iv", sa.Numeric(12, 8), nullable=True),
        sa.Column("atm_put_iv", sa.Numeric(12, 8), nullable=True),
        sa.Column("total_call_open_interest", sa.BigInteger(), nullable=True),
        sa.Column("total_put_open_interest", sa.BigInteger(), nullable=True),
        sa.Column("put_call_oi_ratio", sa.Numeric(18, 8), nullable=True),
        sa.Column("high_iv_skew_bearish", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("available_expirations", FLEXIBLE_JSON, nullable=False),
        sa.Column("calls", FLEXIBLE_JSON, nullable=False),
        sa.Column("puts", FLEXIBLE_JSON, nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "ticker", "fetched_at", name="uq_option_data_ticker_fetched_at"
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'no_options')", name="ck_option_data_status"
        ),
        sa.CheckConstraint(
            "(status = 'ok' AND expiration_date IS NOT NULL) OR "
            "(status = 'no_options' AND expiration_date IS NULL)",
            name="ck_option_data_expiration_status",
        ),
    )
    op.create_index(
        "ix_option_data_ticker_expiration", "option_data", ["ticker", "expiration_date"]
    )
    # Batch mode also supports adding a foreign key to an existing SQLite
    # table, while PostgreSQL uses normal ALTER TABLE statements.
    with op.batch_alter_table("stock_analysis") as batch_op:
        batch_op.add_column(sa.Column("option_data_id", sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            "fk_analysis_option_data",
            "option_data",
            ["option_data_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("ix_stock_analysis_option_data_id", ["option_data_id"])


def downgrade() -> None:
    with op.batch_alter_table("stock_analysis") as batch_op:
        batch_op.drop_index("ix_stock_analysis_option_data_id")
        batch_op.drop_constraint("fk_analysis_option_data", type_="foreignkey")
        batch_op.drop_column("option_data_id")
    op.drop_table("option_data")
