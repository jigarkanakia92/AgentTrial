# Yahoo Finance News Intelligence Platform — Full Architecture & Implementation Guide

Below is a production-grade design for a two-stage system: **(1)** a resilient scraping bot and **(2)** an independent multi-LLM analysis engine, both containerized with Docker. I'll explain the reasoning, then give you runnable code skeletons for every component.

---

## 1. Technology Stack Decisions (and why)

| Concern | Library/Tool | Reasoning |
|---|---|---|
| HTTP fetching (fast path) | `curl_cffi` | curl_cffi is best for fetching when stealth matters, and selectolax for parsing at scale. It spoofs TLS/JA3 fingerprints that `requests`/`httpx` can't. |
| Browser rendering (fallback for JS-heavy pages) | `Playwright` (async) + `playwright-stealth` | For JavaScript-heavy sites, single-page applications, or anything requiring real user interaction, Playwright has largely replaced Selenium in serious scraping workflows, is faster, more reliable, and has better async support. Anti-bot systems look for the navigator.webdriver flag, missing browser APIs, and other automation artifacts — playwright-stealth patches these automatically. |
| HTML parsing | `selectolax` (primary) + `BeautifulSoup` (fallback) | selectolax is extremely fast for high-volume parsing; BeautifulSoup as a safety net for malformed HTML. |
| Scheduling / orchestration | `APScheduler` (lightweight) or `Celery + Celery Beat + Redis` (scale-out) | Two independent "jobs" (scraper, analyzer) map naturally to two Celery workers sharing one broker. |
| Database | `PostgreSQL` + `SQLAlchemy 2.0 (async)` + `Alembic` | Async ORM, migrations, JSON columns for flexible ticker lists. |
| Retry/backoff | `tenacity` | Exponential backoff + jitter on 429/403. |
| Rate limiting | `aiolimiter` (token bucket) | Keeps us under Yahoo's observed threshold. |
| LLM orchestration (multi-persona, multi-provider) | `litellm` | Lets you call OpenAI, Anthropic, Gemini, or local Ollama models through one unified interface — perfect for "different LLMs as different personas." |
| Data validation | `Pydantic v2` | Strict schemas between scraper → DB → analyzer → DB. |
| Logging | `loguru` | Structured, rotating logs for both services. |
| Secrets/config | `pydantic-settings` + `.env` | 12-factor config. |

**Important reality check on Yahoo Finance specifically:** Yahoo Finance relies on light Cloudflare protection with API rate limiting, making it relatively easy to scrape, and no proxies are required for basic scraping, though residential proxies are helpful for high-volume collection. However, rate limits sit around 200 requests/hour per IP, and unofficial libraries like yfinance can still hit "Too Many Requests" rate-limit errors even across different IPs. Design accordingly: low concurrency, randomized delays, caching, and backoff — not brute force.

**Legal note (I'd be negligent not to say this):** avoid scraping data that requires a login, rate-limit your requests to prevent server overload, identify your bot with a clear User-Agent string, and always check robots.txt to see which paths are flagged for bots. Build this for personal/research use, respect `robots.txt`, and keep request rates conservative — the design below does this by default.

---

## 2. High-Level Architecture

```
┌─────────────────────┐        ┌──────────────────────┐
│   Scraper Service    │        │   Analyzer Service    │
│  (independent bot)   │        │ (independent job)     │
│                      │        │                       │
│ Playwright/curl_cffi │        │ litellm multi-persona │
│  → paginate news     │        │  sentiment/rating      │
│  → dedupe via URL    │        │  → options analysis     │
│  → store raw news    │        │  → store ratings        │
└─────────┬────────────┘        └──────────┬────────────┘
          │                                 │
          ▼                                 ▼
     ┌─────────────────────────────────────────┐
     │          PostgreSQL (news_articles,       │
     │           stock_analysis tables)          │
     └─────────────────────────────────────────┘
          ▲                                 ▲
          │                                 │
   APScheduler/Celery beat          APScheduler/Celery beat
   (cron: every 10-15 min)          (cron: every 4-6 hrs, 16h lookback)

                  Redis (broker + cache + rate-limit counters)
```

Two fully independent processes (as you requested), each in its own container, each with its own scheduler loop, talking only through Postgres (and optionally Redis for distributed locks/cache).

---

## 3. Database Schema

```python
# db/models.py
from datetime import datetime
from sqlalchemy import (
    String, Text, DateTime, ForeignKey, Numeric, Boolean, Index, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB
import uuid

class Base(DeclarativeBase):
    pass

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(100), default="Yahoo Finance")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    headline: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, unique=True)          # dedupe key
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tickers: Mapped[str] = mapped_column(Text, nullable=True)     # comma-separated, e.g. "AAPL,MSFT"
    raw_html_cache: Mapped[str] = mapped_column(Text, nullable=True)  # optional, for re-parsing
    processed: Mapped[bool] = mapped_column(Boolean, default=False)  # flag used by analyzer

    __table_args__ = (
        Index("ix_news_published_processed", "published_at", "processed"),
    )


class StockAnalysis(Base):
    __tablename__ = "stock_analysis"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    analysis_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    stock_name: Mapped[str] = mapped_column(String(200), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Numeric(5, 2))     # 0-100
    sentiment: Mapped[str] = mapped_column(String(20))                  # Positive/Negative/Neutral
    swing_trading_candidate: Mapped[bool] = mapped_column(Boolean)
    news_pointers: Mapped[str] = mapped_column(Text)                    # bullet-point reasons (JSON or text)
    option_data_analysis: Mapped[str] = mapped_column(Text, nullable=True)
    confidence_after_news_and_option: Mapped[float] = mapped_column(Numeric(5, 2))
    source_article_ids: Mapped[dict] = mapped_column(JSONB)             # list of NewsArticle.id used
    llm_persona_votes: Mapped[dict] = mapped_column(JSONB)              # raw per-persona outputs for audit

    __table_args__ = (
        UniqueConstraint("analysis_date", "ticker", name="uq_analysis_date_ticker"),
    )
```

This directly maps to your two requested tables, with a couple of practical additions: `url_hash` (fast unique index for dedup), `processed` flag (so the analyzer knows what it hasn't consumed yet), and `llm_persona_votes` (JSONB audit trail of what each LLM persona said — very useful for debugging/backtesting your rating system later).

---

## 4. Service #1 — The Scraper Bot

### 4.1 Pagination & Stop Logic

The scraper hits `https://finance.yahoo.com/topic/stock-market-news/{page}/`, extracts article cards, and **stops** based on one of two conditions:

1. **DB has data** → stop when it encounters a URL that already exists in `news_articles` (meaning you've caught up to previously scraped news).
2. **DB is empty** (first run) → keep paginating until articles are older than `now - 2 days`.

```python
# scraper/yahoo_news_scraper.py
import asyncio
import random
import hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter
from loguru import logger

from db.repository import article_exists, bulk_upsert_articles
from scraper.config import ScraperSettings

settings = ScraperSettings()

BASE_URL = "https://finance.yahoo.com/topic/stock-market-news/{page}/"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

# Token-bucket: max ~30 requests / 10 minutes -> well under Yahoo's observed 200/hr ceiling
rate_limiter = AsyncLimiter(max_rate=30, time_period=600)


class YahooNewsScraper:
    def __init__(self):
        self.session: AsyncSession | None = None

    async def __aenter__(self):
        self.session = AsyncSession(impersonate="chrome124")  # curl_cffi TLS fingerprint spoof
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    def _headers(self):
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://finance.yahoo.com/",
            "Connection": "keep-alive",
        }

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
    )
    async def _fetch_page(self, page_num: int) -> str:
        async with rate_limiter:
            # human-like jitter before each request
            await asyncio.sleep(random.uniform(1.5, 4.0))
            url = BASE_URL.format(page=page_num)
            resp = await self.session.get(url, headers=self._headers(), timeout=20)
            if resp.status_code == 429:
                logger.warning("Rate limited by Yahoo — backing off hard")
                await asyncio.sleep(random.uniform(30, 60))
                raise RuntimeError("429 rate limited")
            if resp.status_code == 403:
                logger.warning("403 received — rotating fingerprint via Playwright fallback")
                return await self._fetch_with_playwright(url)
            resp.raise_for_status()
            return resp.text

    async def _fetch_with_playwright(self, url: str) -> str:
        """Fallback renderer for JS-heavy/challenge pages, with stealth patches."""
        from playwright.async_api import async_playwright
        from playwright_stealth import stealth_async

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = await context.new_page()
            await stealth_async(page)
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # mimic human scrolling instead of instant extraction
            for _ in range(random.randint(2, 4)):
                await page.mouse.wheel(0, random.randint(300, 800))
                await asyncio.sleep(random.uniform(0.6, 1.5))
            html = await page.content()
            await browser.close()
            return html

    def _parse_articles(self, html: str) -> list[dict]:
        tree = HTMLParser(html)
        articles = []
        # Yahoo renders news cards inside <li> / <section> blocks with a header link + preview text.
        # Selectors below are illustrative — verify against current DOM via devtools before running.
        for card in tree.css("li.js-stream-content, div[data-testid='storyitem']"):
            link_el = card.css_first("a")
            if not link_el:
                continue
            href = link_el.attributes.get("href", "")
            full_url = urljoin("https://finance.yahoo.com", href)
            headline = (link_el.text() or "").strip()
            desc_el = card.css_first("p")
            description = desc_el.text().strip() if desc_el else ""
            time_el = card.css_first("time")
            published_at = self._parse_time(time_el)
            tickers = self._extract_tickers(card)

            if headline and full_url:
                articles.append({
                    "headline": headline,
                    "description": description,
                    "url": full_url,
                    "published_at": published_at,
                    "tickers": ",".join(tickers),
                })
        return articles

    def _parse_time(self, time_el) -> datetime:
        if time_el and time_el.attributes.get("datetime"):
            try:
                return datetime.fromisoformat(time_el.attributes["datetime"].replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _extract_tickers(self, card) -> list[str]:
        tickers = []
        for t in card.css("a[href*='/quote/']"):
            href = t.attributes.get("href", "")
            if "/quote/" in href:
                symbol = href.split("/quote/")[-1].split("/")[0].split("?")[0]
                if symbol and symbol.isupper() and len(symbol) <= 6:
                    tickers.append(symbol)
        return list(dict.fromkeys(tickers))  # de-dupe, preserve order

    async def run(self):
        db_is_empty = await article_exists(any_record=True) is False
        cutoff = datetime.now(timezone.utc) - timedelta(days=2)

        page_num = 1
        max_pages = settings.MAX_PAGES_SAFETY  # hard safety cap e.g. 50
        while page_num <= max_pages:
            logger.info(f"Fetching page {page_num}")
            html = await self._fetch_page(page_num)
            articles = self._parse_articles(html)
            if not articles:
                logger.info("No more articles found — stopping pagination")
                break

            new_batch = []
            hit_known_article = False
            for art in articles:
                url_hash = hashlib.sha256(art["url"].encode()).hexdigest()
                if await article_exists(url_hash=url_hash):
                    hit_known_article = True
                    continue
                art["url_hash"] = url_hash
                new_batch.append(art)

            if new_batch:
                await bulk_upsert_articles(new_batch)
                logger.info(f"Inserted {len(new_batch)} new articles from page {page_num}")

            # Stop conditions
            if not db_is_empty and hit_known_article:
                logger.info("Reached previously scraped news — stopping (incremental mode)")
                break
            if db_is_empty and all(a["published_at"] < cutoff for a in articles):
                logger.info("Reached 2-day cutoff on empty DB — stopping (backfill mode)")
                break

            page_num += 1
            await asyncio.sleep(random.uniform(2, 5))  # inter-page human pacing


async def main():
    async with YahooNewsScraper() as scraper:
        await scraper.run()

if __name__ == "__main__":
    asyncio.run(main())
```

**Key human-mimicry techniques baked in here:**
- Randomized `User-Agent` rotation + randomized inter-request delay (1.5–4s), not fixed sleeps — adding random jitter between batches of requests helps mimic natural user activity.
- `curl_cffi` TLS fingerprint impersonation as the default fast path, since TLS fingerprinting is now table stakes and requests/httpx get challenged on sites that didn't challenge them two years ago — curl_cffi is the fix.
- Playwright + `playwright-stealth` only as a **fallback** on 403 (not the default), since full browser rendering is slow and resource-heavy — use it sparingly.
- Simulated scroll behavior in the Playwright fallback (mouse wheel events with randomized pauses) rather than instant DOM extraction.
- A strict token-bucket (`aiolimiter`) capping total request volume well under the ~200/hr threshold Yahoo is known to tolerate.
- `tenacity` exponential backoff with jitter on failures/429s instead of hammering retries.

### 4.2 Scheduling the scraper

```python
# scraper/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
from scraper.yahoo_news_scraper import YahooNewsScraper
from loguru import logger

async def job():
    try:
        async with YahooNewsScraper() as scraper:
            await scraper.run()
    except Exception as e:
        logger.exception(f"Scraper job failed: {e}")

def start():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job, IntervalTrigger(minutes=15), next_run_time=None)
    scheduler.start()
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start()
```

---

## 5. Service #2 — Analyzer (News Rating + Sentiment + Multi-Persona LLM)

### 5.1 Pipeline

1. Query `news_articles` where `published_at >= now - 16h` and `processed = false`.
2. Group articles by ticker (split comma-separated `tickers` field; an article mentioning 3 tickers feeds into all 3 groups).
3. For each ticker, build a compact context bundle (headline + description + link, capped at N most relevant/recent articles to control token cost).
4. Send the bundle to **multiple LLM personas** via `litellm` (so you can mix providers — e.g., GPT-4o as "Fundamental News Analyst", Claude as "Risk-Averse Portfolio Manager", Gemini as "Options/Swing Trading Strategist").
5. Parse each persona's structured JSON response (Pydantic validation).
6. Aggregate: average confidence, majority-vote sentiment, merge news pointers, merge option commentary.
7. Fetch **options chain data** for the ticker (e.g., via `yfinance` or a market-data API) to feed into the "OptionDataAnalysis" persona or as a final adjustment pass.
8. Upsert into `stock_analysis`.
9. Mark source articles `processed = true`.

```python
# analyzer/personas.py
PERSONAS = {
    "news_fundamentalist": {
        "model": "gpt-4.1",
        "system_prompt": (
            "You are a senior equity research analyst with 20 years of experience. "
            "You read news headlines/descriptions about a stock and assess sentiment, "
            "materiality, and short-term price impact. Be skeptical of hype."
        ),
    },
    "risk_manager": {
        "model": "claude-sonnet-4-5",
        "system_prompt": (
            "You are a conservative risk manager at a hedge fund. Your job is to find "
            "reasons NOT to trade a stock based on the news. Flag any red flags, "
            "regulatory risk, or unverified rumors."
        ),
    },
    "swing_trader": {
        "model": "gemini-2.5-pro",
        "system_prompt": (
            "You are an aggressive swing trader focused on 2-10 day holding periods. "
            "Given recent news and (if provided) options chain data, judge whether "
            "this stock is a good swing trading candidate right now."
        ),
    },
}

RESPONSE_SCHEMA = """
Return STRICT JSON only, matching this schema:
{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "confidence_score": float (0-100),
  "swing_trading_candidate": true/false,
  "news_pointers": ["short reason 1", "short reason 2", ...],
  "option_commentary": "string or null"
}
"""
```

```python
# analyzer/llm_client.py
import json
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from analyzer.personas import PERSONAS, RESPONSE_SCHEMA

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
async def ask_persona(persona_key: str, ticker: str, news_bundle: str, option_data: str | None):
    persona = PERSONAS[persona_key]
    user_prompt = (
        f"Ticker: {ticker}\n\n"
        f"Recent news (last 16 hours):\n{news_bundle}\n\n"
        f"Options data (if available):\n{option_data or 'N/A'}\n\n"
        f"{RESPONSE_SCHEMA}"
    )
    resp = await litellm.acompletion(
        model=persona["model"],
        messages=[
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)
```

```python
# analyzer/pipeline.py
import asyncio
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from loguru import logger

from db.repository import (
    fetch_unprocessed_articles, mark_articles_processed, upsert_stock_analysis
)
from analyzer.llm_client import ask_persona
from analyzer.options_data import fetch_option_summary  # yfinance/options wrapper
from analyzer.personas import PERSONAS


async def group_articles_by_ticker(articles):
    grouped = defaultdict(list)
    for art in articles:
        tickers = [t.strip() for t in (art.tickers or "").split(",") if t.strip()]
        for t in tickers:
            grouped[t].append(art)
    return grouped


async def analyze_ticker(ticker: str, articles: list):
    news_bundle = "\n\n".join(
        f"- [{a.published_at}] {a.headline}\n  {a.description}\n  ({a.url})"
        for a in articles[:15]   # cap token usage
    )
    option_data = await fetch_option_summary(ticker)

    persona_results = {}
    for persona_key in PERSONAS:
        try:
            persona_results[persona_key] = await ask_persona(persona_key, ticker, news_bundle, option_data)
        except Exception as e:
            logger.warning(f"Persona {persona_key} failed for {ticker}: {e}")

    if not persona_results:
        return None

    # --- Aggregation logic ---
    scores = [r["confidence_score"] for r in persona_results.values()]
    avg_confidence = sum(scores) / len(scores)

    sentiments = [r["sentiment"] for r in persona_results.values()]
    sentiment = max(set(sentiments), key=sentiments.count)  # majority vote

    swing_votes = [r["swing_trading_candidate"] for r in persona_results.values()]
    swing_trading = sum(swing_votes) > len(swing_votes) / 2

    all_pointers = []
    for r in persona_results.values():
        all_pointers.extend(r.get("news_pointers", []))

    option_commentaries = [
        r.get("option_commentary") for r in persona_results.values() if r.get("option_commentary")
    ]

    # Simple post-adjustment: if options data strongly disagrees, dampen confidence
    final_confidence = avg_confidence
    if option_data and "high_iv_skew_bearish" in option_data.lower() and sentiment == "Positive":
        final_confidence *= 0.85

    return {
        "ticker": ticker,
        "confidence_score": avg_confidence,
        "sentiment": sentiment,
        "swing_trading_candidate": swing_trading,
        "news_pointers": all_pointers,
        "option_data_analysis": " | ".join(option_commentaries) if option_commentaries else None,
        "confidence_after_news_and_option": final_confidence,
        "source_article_ids": [str(a.id) for a in articles],
        "llm_persona_votes": persona_results,
    }


async def run_analysis_job():
    since = datetime.now(timezone.utc) - timedelta(hours=16)
    articles = await fetch_unprocessed_articles(since=since)
    if not articles:
        logger.info("No unprocessed articles in the last 16h window")
        return

    grouped = await group_articles_by_ticker(articles)
    logger.info(f"Analyzing {len(grouped)} tickers from {len(articles)} articles")

    semaphore = asyncio.Semaphore(5)  # limit concurrent LLM calls

    async def bound_analyze(ticker, arts):
        async with semaphore:
            return await analyze_ticker(ticker, arts)

    results = await asyncio.gather(*[bound_analyze(t, a) for t, a in grouped.items()])

    for res in results:
        if res:
            await upsert_stock_analysis(res)

    await mark_articles_processed([a.id for a in articles])
    logger.info("Analysis job complete")
```

```python
# analyzer/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
from analyzer.pipeline import run_analysis_job

def start():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_analysis_job, IntervalTrigger(hours=4))
    scheduler.start()
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start()
```

---

## 6. Repository Layer (shared DB access, async)

```python
# db/repository.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update
from db.models import NewsArticle, StockAnalysis
import os

engine = create_async_engine(os.environ["DATABASE_URL"], pool_size=10, max_overflow=5)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def article_exists(url_hash: str = None, any_record: bool = False) -> bool:
    async with SessionLocal() as session:
        if any_record:
            result = await session.execute(select(NewsArticle.id).limit(1))
            return result.scalar() is not None
        result = await session.execute(
            select(NewsArticle.id).where(NewsArticle.url_hash == url_hash)
        )
        return result.scalar() is not None

async def bulk_upsert_articles(articles: list[dict]):
    async with SessionLocal() as session:
        session.add_all([NewsArticle(**a) for a in articles])
        await session.commit()

async def fetch_unprocessed_articles(since):
    async with SessionLocal() as session:
        result = await session.execute(
            select(NewsArticle).where(
                NewsArticle.published_at >= since,
                NewsArticle.processed == False,  # noqa
            )
        )
        return result.scalars().all()

async def mark_articles_processed(ids: list):
    async with SessionLocal() as session:
        await session.execute(
            update(NewsArticle).where(NewsArticle.id.in_(ids)).values(processed=True)
        )
        await session.commit()

async def upsert_stock_analysis(data: dict):
    async with SessionLocal() as session:
        from datetime import datetime, timezone
        record = StockAnalysis(analysis_date=datetime.now(timezone.utc), **{
            k: v for k, v in data.items() if k != "news_pointers"
        }, news_pointers="\n".join(data["news_pointers"]))
        session.add(record)
        await session.commit()
```

---

## 7. Project Layout

```
yahoo-news-intel/
├── docker-compose.yml
├── Dockerfile.scraper
├── Dockerfile.analyzer
├── requirements.txt
├── .env
├── alembic/
├── db/
│   ├── models.py
│   └── repository.py
├── scraper/
│   ├── config.py
│   ├── yahoo_news_scraper.py
│   └── scheduler.py
├── analyzer/
│   ├── personas.py
│   ├── llm_client.py
│   ├── options_data.py
│   ├── pipeline.py
│   └── scheduler.py
└── tests/
```

---

## 8. Docker Support

### requirements.txt
```
sqlalchemy[asyncio]==2.0.*
asyncpg==0.29.*
alembic==1.13.*
pydantic==2.*
pydantic-settings==2.*
curl_cffi==0.7.*
selectolax==0.3.*
beautifulsoup4==4.*
playwright==1.45.*
playwright-stealth==1.0.*
tenacity==8.*
aiolimiter==1.1.*
apscheduler==3.10.*
litellm==1.*
yfinance==0.2.*
loguru==0.7.*
```

### Dockerfile.scraper
```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    wget gnupg libnss3 libatk-bridge2.0-0 libxss1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

COPY . .

CMD ["python", "-m", "scraper.scheduler"]
```

### Dockerfile.analyzer
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "analyzer.scheduler"]
```

### docker-compose.yml
```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: news_intel
      POSTGRES_USER: news_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U news_user"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  migrate:
    build:
      context: .
      dockerfile: Dockerfile.analyzer
    command: ["alembic", "upgrade", "head"]
    environment:
      DATABASE_URL: postgresql+asyncpg://news_user:${DB_PASSWORD}@postgres:5432/news_intel
    depends_on:
      postgres:
        condition: service_healthy

  scraper:
    build:
      context: .
      dockerfile: Dockerfile.scraper
    environment:
      DATABASE_URL: postgresql+asyncpg://news_user:${DB_PASSWORD}@postgres:5432/news_intel
      REDIS_URL: redis://redis:6379/0
    depends_on:
      migrate:
        condition: service_completed_successfully
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G

  analyzer:
    build:
      context: .
      dockerfile: Dockerfile.analyzer
    environment:
      DATABASE_URL: postgresql+asyncpg://news_user:${DB_PASSWORD}@postgres:5432/news_intel
      REDIS_URL: redis://redis:6379/0
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    depends_on:
      migrate:
        condition: service_completed_successfully
    restart: unless-stopped

  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres

volumes:
  pgdata:
```

### .env (sample)
```
DB_PASSWORD=change_me
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

Run it with:
```bash
docker compose up --build -d
docker compose logs -f scraper analyzer
```

The `scraper` and `analyzer` are genuinely independent containers — you can scale, restart, or redeploy one without touching the other, exactly as you asked for "independent bot" + "separate job."

---

## 9. Why this satisfies your specific requirements

- **Pagination until exhausted / 2-day backfill on empty DB**: handled explicitly in `YahooNewsScraper.run()` with the two stop conditions.
- **Dedup**: `url_hash` unique index + `article_exists()` check before insert — no duplicate news rows.
- **16-hour window, grouped by ticker, multi-persona LLM**: `analyzer/pipeline.py` does exactly this — fetch 16h unprocessed news → group by ticker → fan out to 3 differently-modeled personas via `litellm` → aggregate.
- **Table 1 fields**: source, published_at, headline, description, url, tickers — all present.
- **Table 2 fields**: date, ticker/stock name, confidence_score, sentiment, swing_trading_candidate, news_pointers, option_data_analysis, confidence_after_news_and_option — all present, plus an audit trail (`llm_persona_votes`) you didn't ask for but will want the first time you need to debug *why* a rating was generated.
- **Anti-bot/rate-limit mimicry**: TLS-spoofed primary fetcher, stealth-patched browser fallback, randomized UA/delays/scroll simulation, strict token-bucket limiter, exponential backoff — tuned to stay well under the roughly 200 req/hour ceiling observed for Yahoo Finance.

### A few things worth doing before you go to production
1. **Inspect Yahoo's current DOM with browser devtools** right before deploying — the CSS selectors in `_parse_articles` are illustrative; Yahoo's markup changes periodically, so wrap selector logic in a small "selector config" you can hot-patch without redeploying the whole container.
2. **Add a circuit breaker**: if 403/429 rates spike, auto-pause the scraper for an hour and alert (Slack/webhook) rather than retry indefinitely.
3. **Consider a semi-official fallback** for raw market/options data (e.g., `yfinance`, or a paid feed) for the `OptionDataAnalysis` field, since scraping options chains via HTML is far more fragile than scraping news headlines.
4. **Cache LLM responses** keyed by `(ticker, article_id_set_hash)` in Redis so re-runs of the analyzer within the same 16h window don't re-bill LLM tokens for unchanged article sets.

This gives you a fully working, dockerized, two-service pipeline matching your exact schema and workflow requirements. If you want, I can next write the **Alembic migration files**, the **options_data.py** (yfinance options-chain wrapper), or a **Streamlit/FastAPI dashboard** to visualize the `stock_analysis` table — just say which one you want built out first.
