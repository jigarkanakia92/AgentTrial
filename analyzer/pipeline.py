"""Analyzer pipeline: unprocessed news (16h) → per-ticker LLM personas →
aggregated verdict → stock_analysis upsert.

Resilience contract:
* Persona failures are isolated: an aggregation needs >=1 valid persona;
  failing models are logged and dropped, never crash the cycle.
* Articles from successfully-analyzed tickers are marked processed;
  articles from failed tickers stay unprocessed and are retried next cycle.
* Articles with no ticker attribution are marked processed immediately
  (nothing to analyze — otherwise they'd be re-fetched forever).
* ``run_analysis_job()`` never raises; it returns a summary report dict.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from loguru import logger
from pydantic import BaseModel

from analyzer.config import AnalyzerSettings
from analyzer.llm_client import LLMClient, PersonaError
from analyzer.options_data import fetch_option_summary, fetch_stock_name
from analyzer.schemas import PersonaVerdict
from db import repository
from db.models import NewsArticle

DAMPEN_KEYWORDS = ("high_iv_skew_bearish", "bearish")


# ---------------------------------------------------------------------------
# Pure aggregation logic (unit-testable without any I/O)
# ---------------------------------------------------------------------------


def aggregate_persona_results(
    persona_results: dict[str, PersonaVerdict], option_data: str | None
) -> dict:
    """Merge persona verdicts into a single analysis row payload."""
    verdicts = list(persona_results.values())

    scores = [v.confidence_score for v in verdicts]
    avg_confidence = round(sum(scores) / len(scores), 2)

    # Majority sentiment: strict winner, otherwise a deterministic,
    # conservative tie-break — conflicting personas resolve to Neutral
    # rather than flapping between Positive/Negative run to run.
    counts = Counter(v.sentiment for v in verdicts)
    top_count = max(counts.values())
    tied = [s for s, c in counts.items() if c == top_count]
    sentiment = tied[0] if len(tied) == 1 else "Neutral"

    swing_votes = [v.swing_trading_candidate for v in verdicts]
    swing_trading = sum(swing_votes) > len(swing_votes) / 2

    all_pointers: list[str] = []
    for key in persona_results:  # deterministic order
        for pointer in persona_results[key].news_pointers:
            bullet = f"[{key}] {pointer}"
            if bullet not in all_pointers:
                all_pointers.append(bullet)

    option_commentaries = [
        f"[{key}] {persona_results[key].option_commentary}"
        for key in persona_results
        if persona_results[key].option_commentary
    ]

    # Post-adjustment: strongly bearish options skew against a Positive call
    # dampens confidence by 15%.
    final_confidence = avg_confidence
    if (
        option_data
        and sentiment == "Positive"
        and any(kw in option_data.lower() for kw in DAMPEN_KEYWORDS)
    ):
        final_confidence = round(avg_confidence * 0.85, 2)

    return {
        "confidence_score": avg_confidence,
        "sentiment": sentiment,
        "swing_trading_candidate": swing_trading,
        "news_pointers": all_pointers,
        "option_data_analysis": " | ".join(option_commentaries) or None,
        "confidence_after_news_and_option": max(0.0, min(100.0, final_confidence)),
        "llm_persona_votes": {
            key: verdict.model_dump(mode="json")
            for key, verdict in persona_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Per-ticker analysis
# ---------------------------------------------------------------------------


def group_articles_by_ticker(articles: list[NewsArticle]) -> dict[str, list[NewsArticle]]:
    grouped: dict[str, list[NewsArticle]] = defaultdict(list)
    for article in articles:
        for ticker in (article.tickers or "").split(","):
            ticker = ticker.strip().upper()
            if ticker:
                grouped[ticker].append(article)
    return dict(grouped)


def build_news_bundle(articles: list[NewsArticle], max_articles: int) -> str:
    selected = articles[:max_articles]
    return "\n\n".join(
        f"- [{a.published_at:%Y-%m-%d %H:%M UTC}] {a.headline}\n"
        f"  {(a.description or '').strip()}\n"
        f"  ({a.url})"
        for a in selected
    )


async def analyze_ticker(
    ticker: str,
    articles: list[NewsArticle],
    llm: LLMClient,
    settings: AnalyzerSettings,
) -> dict | None:
    """All-persona analysis for one ticker; None if every persona failed."""
    news_bundle = build_news_bundle(articles, settings.max_articles_per_ticker)
    option_data = await fetch_option_summary(ticker)

    persona_results: dict[str, PersonaVerdict] = {}
    for persona_key in (
        "news_fundamentalist",
        "risk_manager",
        "swing_trader",
    ):
        try:
            persona_results[persona_key] = await llm.ask_persona(
                persona_key, ticker, news_bundle, option_data
            )
            logger.info("Persona {} done for {}", persona_key, ticker)
        except PersonaError as exc:
            logger.warning("Persona {} failed for {}: {}", persona_key, ticker, exc)
        except Exception as exc:  # noqa: BLE001 — absolute persona isolation
            logger.exception("Persona {} crashed for {}: {}", persona_key, ticker, exc)

    if len(persona_results) < settings.min_personas_for_analysis:
        return None

    aggregated = aggregate_persona_results(persona_results, option_data)
    stock_name = await fetch_stock_name(ticker)

    return {
        "ticker": ticker,
        "stock_name": stock_name,
        "source_article_ids": [str(a.id) for a in articles],
        **aggregated,
    }


# ---------------------------------------------------------------------------
# Cycle orchestration
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    articles_fetched: int = 0
    unattributed_articles: int = 0
    tickers_analyzed: int = 0
    tickers_failed: int = 0
    articles_marked_processed: int = 0
    error: str | None = None


async def run_analysis_job(settings: AnalyzerSettings | None = None) -> RunSummary:
    settings = settings or AnalyzerSettings()
    summary = RunSummary(started_at=datetime.now(timezone.utc))
    llm = LLMClient(settings)
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=settings.lookback_hours)
        try:
            articles = await repository.fetch_unprocessed_articles(since=since)
        except Exception as exc:  # noqa: BLE001
            summary.error = f"db unavailable: {exc}"
            logger.error("Analyzer cycle aborted — DB unavailable: {}", exc)
            return _finish(summary)

        if not articles:
            logger.info("No unprocessed articles in the last {}h window", settings.lookback_hours)
            return _finish(summary)

        summary.articles_fetched = len(articles)
        grouped = group_articles_by_ticker(articles)
        unattributed = [a for a in articles if a.id not in {
            art.id for group in grouped.values() for art in group
        }]
        summary.unattributed_articles = len(unattributed)
        logger.info(
            "Analyzing {} tickers from {} articles ({} unattributed)",
            len(grouped), len(articles), len(unattributed),
        )

        semaphore = asyncio.Semaphore(settings.max_concurrent_llm)
        succeeded_articles: list = []
        failed_articles: list = []

        async def bound(ticker: str, arts: list[NewsArticle]) -> tuple[str, dict | None]:
            async with semaphore:
                return ticker, await analyze_ticker(ticker, arts, llm, settings)

        results = await asyncio.gather(
            *(bound(t, arts) for t, arts in grouped.items()),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.error("Ticker analysis task crashed: {}", result)
                continue
            ticker, payload = result
            if payload is None:
                summary.tickers_failed += 1
                failed_articles.extend(grouped[ticker])
                continue
            try:
                await repository.upsert_stock_analysis(payload)
                summary.tickers_analyzed += 1
                succeeded_articles.extend(grouped[ticker])
                logger.info(
                    "Stored analysis for {} ({}, confidence {})",
                    ticker,
                    payload["sentiment"],
                    payload["confidence_score"],
                )
            except Exception as exc:  # noqa: BLE001 — DB hiccup ≠ cycle failure
                summary.tickers_failed += 1
                failed_articles.extend(grouped[ticker])
                logger.error("Failed to store analysis for {}: {}", ticker, exc)

        # Only mark processed what was actually consumed; failures retry next cycle.
        processed_count = 0
        if unattributed:
            try:
                await repository.mark_articles_processed([a.id for a in unattributed])
                processed_count += len(unattributed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not mark unattributed articles processed: {}", exc)
        if succeeded_articles:
            try:
                processed_count += await repository.mark_articles_processed(
                    [a.id for a in succeeded_articles]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not mark articles processed: {}", exc)
        summary.articles_marked_processed = processed_count

        if summary.tickers_failed:
            summary.error = f"{summary.tickers_failed} ticker(s) failed; will retry next cycle"
        return _finish(summary)
    except Exception as exc:  # noqa: BLE001 — the scheduler must survive anything
        summary.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Analysis job crashed (captured; scheduler survives)")
        return _finish(summary)
    finally:
        llm.close()


def _finish(summary: RunSummary) -> RunSummary:
    summary.finished_at = datetime.now(timezone.utc)
    logger.info(
        "Analysis cycle complete: articles={} analyzed_tickers={} failed={} "
        "processed={} error={} took={:.1f}s",
        summary.articles_fetched,
        summary.tickers_analyzed,
        summary.tickers_failed,
        summary.articles_marked_processed,
        summary.error,
        (summary.finished_at - summary.started_at).total_seconds(),
    )
    return summary
