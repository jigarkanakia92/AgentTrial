"""Scraper service configuration (12-factor, env-driven).

Every knob has a safe default; nothing is mandatory. Env prefix is
``SCRAPER_`` except ``DATABASE_URL`` which is shared across services.
"""
from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScraperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Shared with the analyzer — unprefixed alias.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./news_intel.db",
        validation_alias=AliasChoices("DATABASE_URL", "SCRAPER_DATABASE_URL"),
    )

    # --- crawling ------------------------------------------------------------
    list_url_template: str = "https://finance.yahoo.com/topic/stock-market-news/{page}/"
    page_timeout_seconds: int = 20
    request_delay_min: float = 1.5     # human-like jitter window, seconds
    request_delay_max: float = 4.0
    inter_page_delay_min: float = 2.0
    inter_page_delay_max: float = 5.0

    # Token bucket: 30 req / 10 min — well under Yahoo's observed ~200/hr ceiling.
    rate_limit_max_requests: int = 30
    rate_limit_period_seconds: float = 600

    # --- pagination / stop conditions ----------------------------------------
    max_pages_safety: int = 50          # hard cap, whatever happens
    backfill_days: int = 2              # first-run lookback on an empty DB
    max_consecutive_fetch_failures: int = 3
    max_consecutive_empty_pages: int = 3  # likely DOM drift → stop politely

    # --- fallbacks / protection ----------------------------------------------
    playwright_fallback_enabled: bool = True
    fetch_retry_attempts: int = 3
    circuit_failure_threshold: int = 5   # consecutive hard failures → open
    circuit_cooldown_seconds: int = 900  # pause 15 min when the breaker opens

    # --- parsing -------------------------------------------------------------
    selectors_file: str = "scraper/selectors.yaml"
    min_articles_per_strategy: int = 3   # a strategy must prove itself
    store_raw_html: bool = False         # keep raw page HTML for re-parsing

    # --- service -------------------------------------------------------------
    interval_minutes: int = 15
    log_level: str = "INFO"

    @property
    def user_agent_jitter_seconds(self) -> tuple[float, float]:
        return (self.request_delay_min, self.request_delay_max)
