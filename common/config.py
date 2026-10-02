"""Shared settings used by both services."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Async database URL, 12-factor style.

    Defaults to a local SQLite file so `pytest` and quick local runs work
    with zero infrastructure; Docker Compose injects the Postgres URL.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./news_intel.db"


def is_postgres(url: str) -> bool:
    return url.startswith("postgresql")
