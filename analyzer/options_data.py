"""Fetch and persist ticker-keyed Yahoo option chains (best-effort).

Fresh snapshots are reused from ``option_data`` across analyzer restarts;
new pulls append history instead of overwriting it. Only the nearest expiry
is fetched, matching the swing-trading horizon of the original helper.
Caught provider errors/timeouts never become misleading ``no_options`` snapshots.
The public fetch functions degrade gracefully if Yahoo or the DB is down.

One-shot pull, independent of news and LLM credentials::

    python -m analyzer.options_data AAPL MSFT --refresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from analyzer.config import AnalyzerSettings
from db import repository
from db.models import utcnow

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - optional install
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False

OPTION_SOURCE = "Yahoo Finance"
# Company names remain a small process-local cache; options use the database.
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 30 * 60


class OptionSnapshot(BaseModel):
    """Transport object matching the option_data table; IV is a fraction.

    ``id`` is set only after a successful DB write (or when loading from
    the DB), so news analysis never references an unsaved snapshot.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID | None = None
    ticker: str = Field(min_length=1, max_length=20)
    source: str = OPTION_SOURCE
    fetched_at: datetime = Field(default_factory=utcnow)
    expiration_date: date | None = None
    spot_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    atm_call_iv: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    atm_put_iv: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_call_open_interest: int | None = Field(default=None, ge=0)
    total_put_open_interest: int | None = Field(default=None, ge=0)
    put_call_oi_ratio: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    high_iv_skew_bearish: bool = False
    status: Literal["ok", "no_options"] = "ok"
    available_expirations: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)
    puts: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("fetched_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        # SQLite does not round-trip tzinfo; its stored times are UTC.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def to_row(self) -> dict:
        return self.model_dump(exclude={"id"})


def _cached(key: str):
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    return None


def _store(key: str, value: Any) -> Any:
    _CACHE[key] = (time.monotonic(), value)
    return value


def _number(value: Any) -> float | None:
    """Missing/non-finite/negative provider values are unknown, not zero."""
    try:
        if value is None or isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _chain_records(frame: Any) -> list[dict[str, Any]]:
    # pandas handles numpy scalars, NaN/NaT, and timezone-aware timestamps.
    # Keep every column, including new ones Yahoo/yfinance may introduce.
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    if not isinstance(records, list) or any(
        not isinstance(row, dict) for row in records
    ):
        raise ValueError("invalid option-chain record layout")
    json.dumps(records, allow_nan=False)  # JSONB cannot accept NaN/Infinity
    return records


def _atm_iv(records: list[dict], spot: float | None) -> float | None:
    if spot is None or spot <= 0:
        return None
    candidates = [row for row in records if (_number(row.get("strike")) or 0) > 0]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(float(row["strike"]) - spot))
    return _number(nearest.get("impliedVolatility"))


def _total_open_interest(records: list[dict]) -> int | None:
    values = [_number(row.get("openInterest")) for row in records]
    known = [value for value in values if value is not None]
    # Report an unknown total when any contract's OI is missing, rather
    # than present a partial total as the whole chain's open interest.
    if not known or len(known) != len(values):
        return None
    return int(sum(known))


def _summary(snapshot: OptionSnapshot) -> str:
    if snapshot.status == "no_options":
        return f"No listed options found for {snapshot.ticker}."
    lines = [
        f"Ticker: {snapshot.ticker}",
        f"Fetched at: {snapshot.fetched_at:%Y-%m-%d %H:%M:%S UTC}",
        f"Spot: {snapshot.spot_price if snapshot.spot_price is not None else 'N/A'}",
        f"Nearest expiry: {snapshot.expiration_date}",
    ]
    call_iv = (
        f"{snapshot.atm_call_iv * 100:.1f}%"
        if snapshot.atm_call_iv is not None
        else "N/A"
    )
    put_iv = (
        f"{snapshot.atm_put_iv * 100:.1f}%"
        if snapshot.atm_put_iv is not None
        else "N/A"
    )
    lines.append(f"ATM call IV: {call_iv} | ATM put IV: {put_iv}")
    if snapshot.total_call_open_interest is not None:
        lines.append(f"Total call open interest: {snapshot.total_call_open_interest}")
    if snapshot.total_put_open_interest is not None:
        lines.append(f"Total put open interest: {snapshot.total_put_open_interest}")
    if snapshot.put_call_oi_ratio is not None:
        lines.append(f"Put/Call OI ratio: {snapshot.put_call_oi_ratio:.2f}")
    if snapshot.high_iv_skew_bearish:
        lines.append(
            "Note: put IV skew vs calls suggests bearish hedging demand "
            "(high_iv_skew_bearish)."
        )
    return "\n".join(lines)


def _sync_fetch_option_data(ticker: str) -> OptionSnapshot:
    """Blocking provider work; exceptions are isolated by the async wrapper."""
    stock = yf.Ticker(ticker)
    spot: float | None = None
    try:
        spot = _number(stock.fast_info["last_price"])
        if spot == 0:
            spot = None
    except Exception:  # noqa: BLE001 — price is optional
        pass

    listed = list(stock.options or [])
    if not listed:
        snapshot = OptionSnapshot(ticker=ticker, spot_price=spot, status="no_options")
        snapshot.summary = _summary(snapshot)
        return snapshot

    # Do not trust provider ordering or a stale/invalid expiration string.
    expirations: list[date] = []
    for value in listed:
        try:
            parsed = date.fromisoformat(str(value))
            if parsed >= utcnow().date():
                expirations.append(parsed)
        except ValueError:
            continue
    expirations = sorted(set(expirations))
    if not expirations:
        raise ValueError("provider returned no usable option expirations")

    expiry = expirations[0]
    chain = stock.option_chain(expiry.isoformat())
    calls, puts = _chain_records(chain.calls), _chain_records(chain.puts)
    if not calls and not puts:
        raise ValueError("provider returned an empty chain for a listed expiration")
    if spot is None:
        underlying = getattr(chain, "underlying", None) or {}
        spot = _number(underlying.get("regularMarketPrice")) or None
    # Do NOT invent a spot price from the median strike when quotes fail.
    call_iv, put_iv = _atm_iv(calls, spot), _atm_iv(puts, spot)
    call_oi, put_oi = _total_open_interest(calls), _total_open_interest(puts)
    ratio = put_oi / call_oi if call_oi and put_oi is not None else None
    snapshot = OptionSnapshot(
        ticker=ticker,
        expiration_date=expiry,
        spot_price=spot,
        atm_call_iv=call_iv,
        atm_put_iv=put_iv,
        total_call_open_interest=call_oi,
        total_put_open_interest=put_oi,
        put_call_oi_ratio=ratio,
        high_iv_skew_bearish=bool(call_iv and put_iv and put_iv > call_iv * 1.15),
        available_expirations=[value.isoformat() for value in expirations],
        calls=calls,
        puts=puts,
    )
    snapshot.summary = _summary(snapshot)
    return snapshot


async def fetch_option_data(
    ticker: str,
    *,
    settings: AnalyzerSettings | None = None,
    force_refresh: bool = False,
) -> OptionSnapshot | None:
    """Reuse fresh DB data or fetch and save a new nearest-expiry snapshot.

    Yahoo failure: None; no bogus snapshot is saved and old history remains.
    DB failure: log and return live data with id=None, so news-only analysis
    can continue without referencing an unsaved row. Stale data is never
    silently reused. TTL=0 or force_refresh=True always attempts Yahoo.
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker or len(ticker) > 20:
            raise ValueError("ticker must contain 1–20 characters")
        settings = settings or AnalyzerSettings()
        now = utcnow()
        if not force_refresh and settings.options_cache_ttl_seconds > 0:
            try:
                saved = await repository.fetch_latest_option_data(
                    ticker,
                    since=now - timedelta(seconds=settings.options_cache_ttl_seconds),
                    source=OPTION_SOURCE,
                )
                if saved is not None:
                    snapshot = OptionSnapshot.model_validate(saved)
                    age = (now - snapshot.fetched_at).total_seconds()
                    unexpired = (
                        snapshot.expiration_date is None
                        or snapshot.expiration_date >= now.date()
                    )
                    if 0 <= age <= settings.options_cache_ttl_seconds and unexpired:
                        return snapshot
            except Exception as exc:  # noqa: BLE001 — DB cache is best-effort
                logger.warning("Options cache read failed for {}: {}", ticker, exc)

        if not YFINANCE_AVAILABLE:
            logger.debug("yfinance not installed — no live options data for {}", ticker)
            return None
        # The deadline bounds the caller; a running yfinance thread may
        # finish later, but cannot write a snapshot after this await times out.
        snapshot = await asyncio.wait_for(
            asyncio.to_thread(_sync_fetch_option_data, ticker),
            timeout=settings.options_fetch_timeout_seconds,
        )
        try:
            snapshot.id = await repository.insert_option_data(snapshot.to_row())
        except Exception as exc:  # noqa: BLE001 — data still useful for prompt
            logger.warning("Could not persist options for {}: {}", ticker, exc)
        return snapshot
    except Exception as exc:  # noqa: BLE001 — optional data must never crash analysis
        logger.warning("Options data unavailable for {}: {}", ticker, exc)
        return None


async def fetch_option_summary(ticker: str) -> str | None:
    """Backward-compatible compact prompt summary, now backed by option_data."""
    snapshot = await fetch_option_data(ticker)
    return snapshot.summary if snapshot else None


async def fetch_stock_name(ticker: str) -> str | None:
    """Best-effort company name; unchanged by options persistence."""
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


async def _pull_tickers(tickers: list[str], force_refresh: bool) -> int:
    settings = AnalyzerSettings()
    failures = 0
    try:
        for ticker in dict.fromkeys(value.strip().upper() for value in tickers):
            snapshot = await fetch_option_data(
                ticker, settings=settings, force_refresh=force_refresh
            )
            if snapshot is None:
                failures += 1
                print(f"{ticker}: options unavailable")
                continue
            saved = (
                str(snapshot.id) if snapshot.id else "NOT SAVED (database unavailable)"
            )
            print(
                f"{snapshot.ticker}: snapshot={saved} calls={len(snapshot.calls)} puts={len(snapshot.puts)}"
            )
            print(snapshot.summary)
            if snapshot.id is None:
                failures += 1
    finally:
        await repository.dispose_engine()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull and store Yahoo options by ticker."
    )
    parser.add_argument("tickers", nargs="+", help="e.g. AAPL MSFT TSLA")
    parser.add_argument(
        "--refresh", action="store_true", help="bypass the persistent TTL cache"
    )
    args = parser.parse_args()
    from common.logging import setup_logging

    setup_logging("options")
    return asyncio.run(_pull_tickers(args.tickers, args.refresh))


if __name__ == "__main__":
    raise SystemExit(main())
