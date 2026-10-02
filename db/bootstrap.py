"""Convenience: create schema directly (local dev / quick starts).

`python -m db.bootstrap`

Production/Docker should use Alembic (`alembic upgrade head`); this script
is for `DATABASE_URL=sqlite+aiosqlite:///./news_intel.db` style local runs
where migrations are overkill.
"""
from __future__ import annotations

import asyncio

from common.logging import setup_logging
from db import repository
from db.models import Base
from loguru import logger


async def create_all() -> None:
    engine = repository.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Schema ensured on {}", engine.url)
    await repository.dispose_engine()


if __name__ == "__main__":
    setup_logging("bootstrap")
    asyncio.run(create_all())
