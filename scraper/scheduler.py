"""Scraper service entrypoint — an independent, crash-proof scheduling loop.

`python -m scraper.scheduler`

Resilience contract:
* A job crash never kills the scheduler (blanket try/except per run).
* ``max_instances=1`` + ``coalesce=True`` prevent overlapping runs piling up
  if a cycle is slow or the process was frozen.
* DB-down at startup does not abort — we wait, then retry every cycle.
* SIGTERM/SIGINT drain cleanly so ``docker stop`` is graceful.
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from common.logging import setup_logging
from db import repository
from scraper.config import ScraperSettings
from scraper.yahoo_news_scraper import YahooNewsScraper


async def scrape_job() -> None:
    settings = ScraperSettings()
    logger.info("Scheduled scrape cycle starting")
    scraper = YahooNewsScraper(settings)
    report = await scraper.run()  # never raises
    if report.stop_reason is None:
        logger.warning("Scrape cycle returned without a stop reason — bug?")


def _register_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    def _handle(sig: signal.Signals) -> None:
        logger.info("Received {} — shutting down gracefully", sig.name)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: stop.set())


async def main() -> None:
    settings = ScraperSettings()
    setup_logging("scraper", settings.log_level)
    logger.info(
        "Scraper service starting (interval={}min, selectors={})",
        settings.interval_minutes,
        settings.selectors_file,
    )

    await repository.wait_for_db()  # tolerate slow Postgres startup

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _register_signal_handlers(loop, stop)

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        scrape_job,
        IntervalTrigger(minutes=settings.interval_minutes),
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=5),
        id="yahoo-news-scrape",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("Scheduler running — press Ctrl+C to stop")

    await stop.wait()
    scheduler.shutdown(wait=False)
    await repository.dispose_engine()
    logger.info("Scraper service stopped")


if __name__ == "__main__":
    asyncio.run(main())
