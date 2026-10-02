"""Options-chain / market-data helper (yfinance, best-effort).

Contract: functions here NEVER raise. Market data is a nice-to-have input
for the swing-trader persona; yfinance itself is famously brittle (Yahoo
endpoints change), so every failure degrades to ``None`` and the pipeline
continues without options context. Results are TTL-cached per ticker to
stay polite to Yahoo's data endpoints.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - optional install
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 30 * 60


def _cached(key: str):
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    return None


def _store(key: str, value: Any) -> Any:
    _CACHE[key] = (time.monotonic(), value)
    return value


def _sync_fetch_option_summary(ticker: str) -> str | None:
    stock = yf.Ticker(ticker)

    spot: float | None = None
    try:
        info = stock.fast_info
        spot = float(info["last_price"]) if info and "last_price" in info else None
    except Exception:  # noqa: BLE001
        spot = None

    expirations = list(stock.options or [])
    if not expirations:
        return f"No listed options found for {ticker}."

    expiry = expirations[0]  # nearest expiry — swing trading horizon
    chain = stock.option_chain(expiry)
    calls, puts = chain.calls, chain.puts

    if spot is None:
        try:
            spot = float(calls.iloc[(calls["strike"] - calls["strike"].median()).abs().argmin()]["strike"])
        except Exception:  # noqa: BLE001
            spot = None

    lines = [f"Spot: {spot if spot is not None else 'N/A'}", f"Nearest expiry: {expiry}"]

    try:
        if spot is not None and not calls.empty and not puts.empty:
            call_row = calls.iloc[(calls["strike"] - spot).abs().argmin()]
            put_row = puts.iloc[(puts["strike"] - spot).abs().argmin()]
            call_iv = float(call_row.get("impliedVolatility") or 0) * 100
            put_iv = float(put_row.get("impliedVolatility") or 0) * 100
            lines.append(f"ATM call IV: {call_iv:.1f}% | ATM put IV: {put_iv:.1f}%")
            if put_iv and call_iv and put_iv > call_iv * 1.15:
                lines.append("Note: put IV skew vs calls suggests bearish hedging demand (high_iv_skew_bearish).")
            total_call_oi = float(calls["openInterest"].fillna(0).sum())
            total_put_oi = float(puts["openInterest"].fillna(0).sum())
            if total_call_oi > 0:
                lines.append(
                    f"Put/Call OI ratio: {total_put_oi / total_call_oi:.2f} "
                    f"(>1 leans bearish, <0.7 leans bullish)"
                )
    except Exception as exc:  # noqa: BLE001 — chain layout changed? degrade
        logger.debug("Option chain summarization degraded for {}: {}", ticker, exc)

    return "\n".join(lines)


async def fetch_option_summary(ticker: str) -> str | None:
    """Compact options summary for the LLM prompt, or None on any failure."""
    if not YFINANCE_AVAILABLE:
        logger.debug("yfinance not installed — options context disabled")
        return None
    cached = _cached(f"opts:{ticker}")
    if cached is not None:
        return cached
    try:
        summary = await asyncio.wait_for(
            asyncio.to_thread(_sync_fetch_option_summary, ticker), timeout=45.0
        )
        return _store(f"opts:{ticker}", summary)
    except Exception as exc:  # noqa: BLE001 — data is optional, never fatal
        logger.warning("Options data unavailable for {}: {}", ticker, exc)
        return None


async def fetch_stock_name(ticker: str) -> str | None:
    """Best-effort company name for the report row."""
    if not YFINANCE_AVAILABLE:
        return None
    cached = _cached(f"name:{ticker}")
    if cached is not None:
        return cached
    try:
        def _lookup() -> str | None:
            info: dict = yf.Ticker(ticker).info or {}
            name = info.get("shortName") or info.get("longName")
            return str(name)[:200] if name else None

        name = await asyncio.wait_for(asyncio.to_thread(_lookup), timeout=30.0)
        return _store(f"name:{ticker}", name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stock name lookup failed for {}: {}", ticker, exc)
        return None
