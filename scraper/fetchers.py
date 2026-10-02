"""Resilient page fetching with layered backends.

Request flow for every page::

    rate limiter (token bucket) + human jitter
      └─ fast backend (curl_cffi TLS-spoof  →  httpx plain)
           ├─ 200 & no challenge markers  →  success
           ├─ 403/429 or bot-challenge    →  escalate to Playwright+stealth
           ├─ 5xx / timeout / network     →  tenacity retry (exp. backoff)
           └─ other 4xx                   →  hard fail this page

Every hard failure feeds the shared CircuitBreaker. Playwright is imported
lazily and entirely optional — a missing install degrades to fast-path-only
with a warning, never a crash.
"""
from __future__ import annotations

import asyncio
import random
from typing import NamedTuple

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scraper.circuit_breaker import CircuitBreaker, CircuitOpenError
from scraper.config import ScraperSettings

# ---------------------------------------------------------------------------
# Optional dependencies — everything here is guarded on purpose.
# ---------------------------------------------------------------------------

try:  # primary fast path: TLS/JA3 fingerprint spoofing
    from curl_cffi.requests import AsyncSession
    from curl_cffi.requests.exceptions import CurlError

    CURL_CFFI_AVAILABLE = True
except Exception:  # pragma: no cover - depends on install
    AsyncSession = None  # type: ignore[assignment]
    CurlError = Exception  # type: ignore[misc,assignment]
    CURL_CFFI_AVAILABLE = False

try:  # plain HTTP fallback
    import httpx

    HTTPX_AVAILABLE = True
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

PLAYWRIGHT_AVAILABLE = True  # confirmed lazily on first use

# Strings that scream "bot challenge page, not real content".
CHALLENGE_MARKERS = (
    "just a moment",
    "cf-challenge",
    "challenge-platform",
    "cf_chl_opt",
    "enable javascript and cookies",
    "attention required",
    "access denied",
    "unusual traffic",
    "captcha",
)


class PageFetchError(RuntimeError):
    """A page could not be fetched after all retries/backends."""


class TransientFetchError(PageFetchError):
    """Retryable: timeouts, network hiccups, 5xx."""


class ChallengeDetectedError(PageFetchError):
    """The response is a bot-challenge page, not content."""


class FetchOutcome(NamedTuple):
    url: str
    html: str
    backend: str
    status: int


def looks_like_challenge(html: str) -> bool:
    if not html or len(html) < 500:
        return True  # empty / stumped responses behave like challenges
    lowered = html[:4000].lower()
    return any(marker in lowered for marker in CHALLENGE_MARKERS)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class CurlCffiBackend:
    name = "curl_cffi"

    def __init__(self) -> None:
        self._session: "AsyncSession | None" = None
        self.available = CURL_CFFI_AVAILABLE

    async def start(self, headers: dict) -> None:
        if self.available:
            self._session = AsyncSession(impersonate="chrome124")

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001 — closing must never raise
                pass
            self._session = None

    async def fetch(self, url: str, headers: dict, timeout: float) -> tuple[int, str]:
        assert self._session is not None
        resp = await self._session.get(url, headers=headers, timeout=int(timeout))
        return resp.status_code, resp.text


class HttpxBackend:
    name = "httpx"

    def __init__(self) -> None:
        self._client: "httpx.AsyncClient | None" = None
        self.available = HTTPX_AVAILABLE

    async def start(self, headers: dict) -> None:
        if self.available:
            self._client = httpx.AsyncClient(
                follow_redirects=True, headers=headers, timeout=30.0
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str, headers: dict, timeout: float) -> tuple[int, str]:
        assert self._client is not None
        resp = await self._client.get(url, headers=headers)
        return resp.status_code, resp.text


class PlaywrightBackend:
    """Full browser render — expensive, used only on 403/429/challenge."""

    name = "playwright"

    def __init__(self) -> None:
        self.available = PLAYWRIGHT_AVAILABLE
        self._import_error: str | None = None

    async def close(self) -> None:  # browser lifecycle is per-request
        return None

    async def fetch(self, url: str, headers: dict, timeout: float) -> tuple[int, str]:
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover - optional install
            self.available = False
            self._import_error = str(exc)
            raise PageFetchError(
                "Playwright not installed — cannot render challenge pages"
            ) from exc

        from scraper.user_agents import random_user_agent

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=random_user_agent(),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                )
                page = await context.new_page()
                try:  # stealth is best-effort hardening, never a hard dep
                    from playwright_stealth import stealth_async

                    await stealth_async(page)
                except Exception:  # noqa: BLE001
                    logger.debug("playwright_stealth unavailable — continuing unpatched")

                await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                # Mimic a human scanning the page instead of instant extraction.
                for _ in range(random.randint(2, 4)):
                    await page.mouse.wheel(0, random.randint(300, 800))
                    await asyncio.sleep(random.uniform(0.4, 1.2))
                # Challenge pages usually redirect after solving; give it a beat.
                await asyncio.sleep(random.uniform(1.0, 2.0))
                html = await page.content()
                return 200, html
            finally:
                await browser.close()


# ---------------------------------------------------------------------------
# Rate limiter (aiolimiter if present, minimal fallback otherwise)
# ---------------------------------------------------------------------------


class _FallbackLimiter:
    """Very small sliding-window limiter — stand-in when aiolimiter is absent."""

    def __init__(self, max_rate: int, time_period: float) -> None:
        self._max_rate = max_rate
        self._period = time_period
        self._events: list[float] = []
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "_FallbackLimiter":
        async with self._lock:
            import time as _time

            now = _time.monotonic()
            self._events = [t for t in self._events if now - t < self._period]
            if len(self._events) >= self._max_rate:
                wait = self._period - (now - self._events[0])
                if wait > 0:
                    await asyncio.sleep(wait)
            self._events.append(_time.monotonic())
        return self

    async def __aexit__(self, *exc) -> None:
        return None


def _build_limiter(settings: ScraperSettings):
    try:
        from aiolimiter import AsyncLimiter

        return AsyncLimiter(
            max_rate=settings.rate_limit_max_requests,
            time_period=settings.rate_limit_period_seconds,
        )
    except Exception:  # pragma: no cover - optional install
        return _FallbackLimiter(
            settings.rate_limit_max_requests, settings.rate_limit_period_seconds
        )


# ---------------------------------------------------------------------------
# The public fetcher
# ---------------------------------------------------------------------------

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    TransientFetchError,
)
if HTTPX_AVAILABLE:
    _RETRYABLE_EXCEPTIONS += (httpx.TransportError, httpx.HTTPStatusError)
if CURL_CFFI_AVAILABLE:
    _RETRYABLE_EXCEPTIONS += (CurlError,)


class ResilientFetcher:
    """One instance per scraper run; owns backends, limiter and breaker."""

    def __init__(self, settings: ScraperSettings, breaker: CircuitBreaker) -> None:
        self.settings = settings
        self.breaker = breaker
        self._limiter = _build_limiter(settings)
        backends: list = []
        if not CURL_CFFI_AVAILABLE:
            logger.warning("curl_cffi not installed — TLS-spoof fast path disabled")
        else:
            backends.append(CurlCffiBackend())
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed — plain-HTTP fallback disabled")
        else:
            backends.append(HttpxBackend())
        self.fast_backends = backends
        self.playwright = PlaywrightBackend()
        if not settings.playwright_fallback_enabled:
            logger.info("Playwright fallback disabled by configuration")

    @property
    def _headers(self) -> dict:
        from scraper.user_agents import browser_headers, random_user_agent

        return browser_headers(random_user_agent())

    async def start(self) -> None:
        for backend in [*self.fast_backends, self.playwright]:
            try:
                await backend.start(self._headers)
            except Exception as exc:  # noqa: BLE001
                backend.available = False
                logger.warning("Backend {} failed to start: {}", backend.name, exc)

    async def close(self) -> None:
        for backend in [*self.fast_backends, self.playwright]:
            await backend.close()

    async def get(self, url: str) -> FetchOutcome:
        """Fetch one page through the full resilience stack. Never returns
        junk HTML: either real content or an exception."""
        if not self.breaker.allow_request():
            raise CircuitOpenError(
                f"circuit open — refusing request to {url} (cooldown in progress)"
            )
        try:
            outcome = await self._get_with_retries(url)
            self.breaker.record_success()
            return outcome
        except Exception:
            self.breaker.record_failure()
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    async def _get_with_retries(self, url: str) -> FetchOutcome:
        async with self._limiter:
            await asyncio.sleep(random.uniform(*self.settings.user_agent_jitter_seconds))
            return await self._fetch_once(url)

    async def _fetch_once(self, url: str) -> FetchOutcome:
        last_error: Exception | None = None
        for backend in [b for b in self.fast_backends if b.available]:
            try:
                status, html = await backend.fetch(
                    url, self._headers, float(self.settings.page_timeout_seconds)
                )
            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning("Backend {} transient error on {}: {}", backend.name, url, exc)
                continue
            except Exception as exc:  # noqa: BLE001 — unknown backend blow-ups
                last_error = TransientFetchError(str(exc))
                logger.warning("Backend {} unexpected error on {}: {}", backend.name, url, exc)
                continue

            if status == 200 and not looks_like_challenge(html):
                return FetchOutcome(url, html, backend.name, status)
            if status in (403, 429):
                logger.warning("{} from {} — escalating to browser render", status, url)
                return await self._fetch_with_playwright(url, reason=status)
            if looks_like_challenge(html):
                logger.warning("Challenge page detected on {} — escalating", url)
                return await self._fetch_with_playwright(url, reason="challenge")
            if status >= 500:
                last_error = TransientFetchError(f"HTTP {status} from {backend.name}")
                continue
            raise PageFetchError(f"HTTP {status} (non-retryable) from {backend.name}")

        if last_error:
            raise TransientFetchError(str(last_error))
        raise PageFetchError("No fetch backend available (install curl_cffi or httpx)")

    async def _fetch_with_playwright(self, url: str, reason) -> FetchOutcome:
        if not self.settings.playwright_fallback_enabled or not self.playwright.available:
            raise ChallengeDetectedError(
                f"{url} requires browser render ({reason}) but Playwright "
                "fallback is disabled/unavailable"
            )
        try:
            status, html = await self.playwright.fetch(
                url, self._headers, float(self.settings.page_timeout_seconds)
            )
        except PageFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ChallengeDetectedError(
                f"browser render failed for {url}: {exc}"
            ) from exc

        if looks_like_challenge(html):
            raise ChallengeDetectedError(f"browser render still challenged for {url}")
        logger.info("Browser render succeeded for {} (reason: {})", url, reason)
        return FetchOutcome(url, html, "playwright", status)
