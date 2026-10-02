"""Analyzer service entrypoint — independent crash-proof scheduling loop.

`python -m analyzer.scheduler`
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from analyzer.config import AnalyzerSettings
from analyzer.pipeline import run_analysis_job
from common.logging import setup_logging
from db import repository


async def analysis_job() -> None:
    settings = AnalyzerSettings()
    logger.info("Scheduled analysis cycle starting")
    await run_analysis_job(settings)  # never raises


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
    settings = AnalyzerSettings()
    setup_logging("analyzer", settings.log_level)
    logger.info(
        "Analyzer service starting (interval={}min, endpoint={}, models={}/{}/{})",
        settings.analysis_interval_minutes,
        settings.llm_base_url,
        settings.model_news_analyst,
        settings.model_risk_manager,
        settings.model_swing_trader,
    )

    await repository.wait_for_db()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    _register_signal_handlers(loop, stop)

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        analysis_job,
        IntervalTrigger(minutes=settings.analysis_interval_minutes),
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=10),
        id="stock-analysis",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.start()
    logger.info("Scheduler running — press Ctrl+C to stop")

    await stop.wait()
    scheduler.shutdown(wait=False)
    await repository.dispose_engine()
    logger.info("Analyzer service stopped")


if __name__ == "__main__":
    asyncio.run(main())
