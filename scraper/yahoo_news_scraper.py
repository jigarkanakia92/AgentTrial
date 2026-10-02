"""Yahoo Finance news scraper — resilient orchestration layer.

Guarantees (this is what "structural changes must not break the bot" means
in code):

* ``run()`` NEVER raises. Every failure mode is captured in a RunReport.
* One bad page never aborts the cycle (consecutive-failure cap instead).
* A Yahoo DOM redesign cannot abort the cycle: the parser falls through
  CSS → JSON-LD → RSS → heuristics, and only a *sustained* zero-extraction
  streak (configurable) ends the cycle with a SELECTOR_DRIFT alert.
* A rate ban (403/429 storm) trips the circuit breaker and pauses requests
  for a cooldown instead of hammering deeper into the ban.
* DB unavailability ends the cycle cleanly; the scheduler retries later.
* Dedup via normalized-URL sha256 — idempotent inserts, crash-safe re-runs.

Stop conditions (in priority order):
  1. safety cap ``max_pages_safety``
  2. circuit breaker opened
  3. too many consecutive fetch failures
  4. previously-scraped article reached (incremental mode)
  5. backfill cutoff reached on an empty DB (``backfill_days``)
  6. empty/no-new pages streak (likely caught up, or DOM drift)
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from loguru import logger

from db import repository
from scraper.circuit_breaker import BreakerState, CircuitBreaker, CircuitOpenError
from scraper.config import ScraperSettings
from scraper.fetchers import ChallengeDetectedError, PageFetchError, ResilientFetcher
from scraper.parsing import NewsParser
from scraper.selectors import SelectorRegistryLoader


class StopReason:
    COMPLETED = "completed_max_pages"
    INCREMENTAL = "reached_previously_scraped_news"
    BACKFILL_CUTOFF = "reached_backfill_cutoff"
    NO_NEW_ARTICLES = "no_new_articles_streak"
    FETCH_FAILURES = "too_many_fetch_failures"
    CIRCUIT_OPEN = "circuit_breaker_open"
    DB_UNAVAILABLE = "database_unavailable"
    FATAL = "unexpected_error"


@dataclass
class RunReport:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    stop_reason: str | None = None
    pages_fetched: int = 0
    pages_failed: int = 0
    pages_no_articles: int = 0
    articles_found: int = 0
    articles_new: int = 0
    duplicates_skipped: int = 0
    strategies_used: dict = field(default_factory=dict)
    rss_used: bool = False
    circuit_opened: bool = False
    selector_drift_suspected: bool = False
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            "run complete: stop={stop} pages={pages}(fail={fails}) "
            "found={found} new={new} dupes={dupes} strategies={strats} "
            "circuit_open={cb} drift={drift} took={dur:.1f}s"
        ).format(
            stop=self.stop_reason,
            pages=self.pages_fetched,
            fails=self.pages_failed,
            found=self.articles_found,
            new=self.articles_new,
            dupes=self.duplicates_skipped,
            strats=self.strategies_used,
            cb=self.circuit_opened,
            drift=self.selector_drift_suspected,
            dur=self.duration_seconds,
        )


class YahooNewsScraper:
    def __init__(self, settings: ScraperSettings | None = None) -> None:
        self.settings = settings or ScraperSettings()
        self.registry_loader = SelectorRegistryLoader(self.settings.selectors_file)
        self.breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_failure_threshold,
            cooldown_seconds=self.settings.circuit_cooldown_seconds,
        )
        self.fetcher = ResilientFetcher(self.settings, self.breaker)

    # ------------------------------------------------------------------
    async def run(self) -> RunReport:
        """One full scrape cycle. Never raises."""
        report = RunReport()
        try:
            await self._run_inner(report)
        except CircuitOpenError as exc:
            report.stop_reason = StopReason.CIRCUIT_OPEN
            report.circuit_opened = self.breaker.state is BreakerState.OPEN
            report.error = str(exc)
        except Exception as exc:  # noqa: BLE001 — absolute last resort
            report.stop_reason = StopReason.FATAL
            report.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Scraper run crashed (captured, scheduler survives)")
        finally:
            report.finished_at = datetime.now(timezone.utc)
            report.strategies_used = dict(self._parser.strategy_usage) if hasattr(self, "_parser") else {}
            logger.info(report.summary())
            try:
                await self.fetcher.close()
            except Exception:  # noqa: BLE001
                pass
        return report

    # ------------------------------------------------------------------
    async def _run_inner(self, report: RunReport) -> None:
        settings = self.settings
        registry = self.registry_loader.load()   # hot-reload every run
        self._parser = NewsParser(registry)
        await self.fetcher.start()

        # --- DB readiness: without it we cannot dedupe or persist -------
        try:
            db_empty = await repository.count_articles() == 0
        except Exception as exc:  # noqa: BLE001
            report.stop_reason = StopReason.DB_UNAVAILABLE
            report.error = str(exc)
            logger.error("Database unavailable — aborting this cycle: {}", exc)
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.backfill_days)
        mode = "BACKFILL (empty DB)" if db_empty else "INCREMENTAL"
        logger.info("Scraper cycle starting in {} mode", mode)

        consecutive_fetch_failures = 0
        consecutive_empty_pages = 0

        for page_num in range(1, settings.max_pages_safety + 1):
            if page_num > 1:
                await asyncio.sleep(
                    random.uniform(
                        settings.inter_page_delay_min, settings.inter_page_delay_max
                    )
                )

            # --- fetch ---------------------------------------------------
            page_url = registry.list_url_template.format(page=page_num)
            try:
                outcome = await self.fetcher.get(page_url)
            except CircuitOpenError:
                report.circuit_opened = True
                report.stop_reason = StopReason.CIRCUIT_OPEN
                logger.error("Circuit breaker open — ending cycle politely")
                return
            except (PageFetchError, ChallengeDetectedError) as exc:
                report.pages_failed += 1
                consecutive_fetch_failures += 1
                logger.warning(
                    "Page {} fetch failed ({}/{}): {}",
                    page_num,
                    consecutive_fetch_failures,
                    settings.max_consecutive_fetch_failures,
                    exc,
                )
                if consecutive_fetch_failures >= settings.max_consecutive_fetch_failures:
                    report.stop_reason = StopReason.FETCH_FAILURES
                    logger.error("Too many consecutive fetch failures — ending cycle")
                    return
                continue  # skip this page, keep going

            report.pages_fetched += 1
            consecutive_fetch_failures = 0

            # --- parse (multi-strategy, never raises) --------------------
            parsed = self._parser.parse_page(outcome.html, page_url)

            # RSS rescue: if the DOM yields nothing on the first page, try feeds
            if not parsed.articles and page_num == 1:

                async def _fetch_text(url: str) -> str:
                    outcome = await self.fetcher.get(url)
                    return outcome.html

                rss_articles = await self._parser.parse_rss_feeds(_fetch_text)
                if rss_articles:
                    report.rss_used = True
                    parsed = parsed.model_copy(
                        update={
                            "articles": rss_articles,
                            "strategy": "rss",
                            "below_threshold": False,
                        }
                    )
                    logger.info(
                        "DOM strategies empty — RSS rescue produced {} articles",
                        len(rss_articles),
                    )

            if not parsed.articles:
                report.pages_no_articles += 1
                consecutive_empty_pages += 1
                logger.warning(
                    "Page {} produced 0 articles ({}/{}) — Yahoo markup may "
                    "have changed; edit {} if this persists",
                    page_num,
                    consecutive_empty_pages,
                    settings.max_consecutive_empty_pages,
                    settings.selectors_file,
                )
                if consecutive_empty_pages >= settings.max_consecutive_empty_pages:
                    report.selector_drift_suspected = parsed.below_threshold
                    report.stop_reason = StopReason.NO_NEW_ARTICLES
                    logger.error(
                        "SELECTOR DRIFT SUSPECTED — {} consecutive pages with "
                        "zero articles. Aborting cycle without writes; the "
                        "scheduler will retry and an alert is warranted.",
                        consecutive_empty_pages,
                    )
                    return
                continue

            consecutive_empty_pages = 0
            report.articles_found += len(parsed.articles)

            # --- dedupe against DB ---------------------------------------
            candidates = parsed.articles
            try:
                existing = await repository.filter_existing_url_hashes(
                    [a.url_hash for a in candidates]
                )
            except Exception as exc:  # noqa: BLE001
                report.stop_reason = StopReason.DB_UNAVAILABLE
                report.error = str(exc)
                logger.error("DB read failed mid-cycle — stopping: {}", exc)
                return

            fresh = [a for a in candidates if a.url_hash not in existing]
            report.duplicates_skipped += len(candidates) - len(fresh)

            if fresh:
                rows = [a.to_row() for a in fresh]
                if not settings.store_raw_html:
                    for row in rows:
                        row.pop("raw_html_cache", None)
                try:
                    inserted = await repository.insert_articles_skipping_existing(rows)
                except Exception as exc:  # noqa: BLE001
                    report.stop_reason = StopReason.DB_UNAVAILABLE
                    report.error = str(exc)
                    logger.error("DB write failed mid-cycle — stopping: {}", exc)
                    return
                report.articles_new += inserted
                logger.info(
                    "Page {}: {} new / {} duplicate articles (strategy={}, "
                    "inserted={})",
                    page_num,
                    len(fresh),
                    len(candidates) - len(fresh),
                    parsed.strategy,
                    inserted,
                )
            else:
                logger.info("Page {}: no new articles", page_num)

            # --- stop conditions ------------------------------------------
            if not db_empty and existing:
                report.stop_reason = StopReason.INCREMENTAL
                logger.info(
                    "Hit previously scraped news on page {} — incremental stop",
                    page_num,
                )
                return
            if db_empty and fresh and all(
                a.published_at < cutoff for a in fresh
            ):
                report.stop_reason = StopReason.BACKFILL_CUTOFF
                logger.info("Reached {}-day backfill cutoff", settings.backfill_days)
                return

        report.stop_reason = StopReason.COMPLETED
        logger.warning("Hit max_pages_safety cap ({} pages)", settings.max_pages_safety)


async def main() -> RunReport:
    """One-shot scrape cycle (CLI: ``python -m scraper.yahoo_news_scraper``)."""
    from common.logging import setup_logging

    settings = ScraperSettings()
    setup_logging("scraper", settings.log_level)
    scraper = YahooNewsScraper(settings)
    return await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
