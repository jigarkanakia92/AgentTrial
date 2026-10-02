"""Exercise real Alembic upgrades/downgrades without touching a user's DB."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from db.models import OptionData

ROOT = Path(__file__).resolve().parents[1]


def migrate(database_url, *args):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT,
        env={**os.environ, "DATABASE_URL": database_url},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_fresh_database_migrates_option_table(tmp_path):
    path = tmp_path / "fresh.db"
    migrate(f"sqlite+aiosqlite:///{path}", "upgrade", "head")
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(option_data)")
        }
        assert columns == {column.name for column in OptionData.__table__.columns}
        assert (
            connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
            == "0002_option_data"
        )
        json_columns = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(option_data)")
        }
        assert json_columns["calls"] == json_columns["puts"] == "JSON"


def test_upgrade_and_downgrade_preserve_existing_analysis(tmp_path):
    path = tmp_path / "existing.db"
    url = f"sqlite+aiosqlite:///{path}"
    migrate(url, "upgrade", "0001_initial")
    row_id = uuid4().hex
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO stock_analysis
            (id, analysis_date, ticker, confidence_score, sentiment, swing_trading_candidate,
             news_pointers, confidence_after_news_and_option, source_article_ids,
             llm_persona_votes, created_at, updated_at)
            VALUES (?, '2026-10-01 00:00:00', 'AAPL', 80, 'Positive', 1,
                    'existing news', 80, '[]', '{}', '2026-10-01', '2026-10-01')""",
            (row_id,),
        )
    migrate(url, "upgrade", "head")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT id, ticker, option_data_id FROM stock_analysis"
        ).fetchone() == (row_id, "AAPL", None)
        links = connection.execute("PRAGMA foreign_key_list(stock_analysis)").fetchall()
        assert any(
            row[2] == "option_data"
            and row[3] == "option_data_id"
            and row[6] == "SET NULL"
            for row in links
        )
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(stock_analysis)")
        }
        assert "ix_stock_analysis_option_data_id" in indexes

    migrate(url, "downgrade", "0001_initial")
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stock_analysis)")
        }
        assert "option_data_id" not in columns
        assert connection.execute(
            "SELECT id, ticker FROM stock_analysis"
        ).fetchone() == (row_id, "AAPL")
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='option_data'"
            ).fetchone()
            is None
        )
    migrate(url, "upgrade", "head")


def test_postgresql_migration_uses_jsonb_and_nullable_foreign_key():
    sql = migrate(
        "postgresql+asyncpg://unused:unused@localhost/unused",
        "upgrade",
        "head",
        "--sql",
    )
    assert "CREATE TABLE option_data" in sql
    assert "calls JSONB NOT NULL" in sql and "puts JSONB NOT NULL" in sql
    assert "ADD COLUMN option_data_id UUID" in sql
    assert (
        "FOREIGN KEY(option_data_id) REFERENCES option_data (id) ON DELETE SET NULL"
        in sql
    )
