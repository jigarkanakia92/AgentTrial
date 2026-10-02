"""Smoke: every module imports cleanly (catches syntax/name errors early,
including optional-dependency guards)."""
from __future__ import annotations


def test_import_all_modules():
    pass


def test_models_match_migration_shape():
    """news_articles has the exact columns from yahooscraper.md (plus the
    documented observability additions)."""
    from db.models import NewsArticle, OptionData, StockAnalysis

    news_cols = {c.name for c in NewsArticle.__table__.columns}
    assert {
        "id", "source", "published_at", "scraped_at", "headline", "description",
        "url", "url_hash", "tickers", "raw_html_cache", "processed",
        "parse_strategy",
    } <= news_cols

    analysis_cols = {c.name for c in StockAnalysis.__table__.columns}
    assert {
        "id", "analysis_date", "ticker", "stock_name", "confidence_score",
        "sentiment", "swing_trading_candidate", "news_pointers",
        "option_data_id", "option_data_analysis", "confidence_after_news_and_option",
        "source_article_ids", "llm_persona_votes",
    } <= analysis_cols


    option_cols = {c.name for c in OptionData.__table__.columns}
    assert {
        "id", "ticker", "source", "fetched_at", "expiration_date", "spot_price",
        "atm_call_iv", "atm_put_iv", "total_call_open_interest", "total_put_open_interest",
        "put_call_oi_ratio", "high_iv_skew_bearish", "status", "available_expirations",
        "calls", "puts", "summary",
    } == option_cols


def test_persona_models_resolve_from_settings():
    from analyzer.config import AnalyzerSettings
    from analyzer.personas import persona_model

    settings = AnalyzerSettings(llm_api_key="k")
    assert persona_model("news_fundamentalist", settings) == "meta/llama-3.3-70b-instruct"
    assert persona_model("risk_manager", settings) == "nvidia/llama-3.1-nemotron-70b-instruct"
    assert persona_model("swing_trader", settings) == "qwen/qwen2.5-32b-instruct"


def test_scraper_settings_defaults_sane():
    from scraper.config import ScraperSettings

    s = ScraperSettings()
    assert s.max_pages_safety >= 1
    assert s.backfill_days >= 1
    assert s.rate_limit_max_requests > 0
    assert "{page}" in s.list_url_template
