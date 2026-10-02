"""Structured logging bootstrap (loguru) shared by both services."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[service]}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[service]} | "
    "{name}:{line} - {message}"
)


def setup_logging(service: str, level: str = "INFO") -> None:
    """Configure console + rotating file logs for one service.

    Idempotent: safe to call multiple times (e.g. scheduler + job code).
    Rotation/retention keep the logs directory bounded in long-running
    containers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logger.configure(extra={"service": service})
        return

    log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_sink = str(log_dir / f"{service}.log")
        logger.add(
            file_sink,
            level=level,
            format=_FILE_FORMAT,
            rotation="20 MB",
            retention=10,
            compression="zip",
            enqueue=True,          # multiprocess/process-safe writes
            backtrace=False,
            diagnose=False,        # never leak variable values in prod logs
        )
    except OSError:  # read-only filesystem etc. — console logging still works
        pass

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, backtrace=False)
    logger.configure(extra={"service": service})
    _CONFIGURED = True
