"""Convenience: create schema directly (local dev / quick starts).

`python -m db.bootstrap`

Use Alembic (`alembic upgrade head`) for existing databases, including
local ones. This script only creates missing tables; it does NOT add new
columns such as stock_analysis.option_data_id to an existing table.
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
