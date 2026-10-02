# Yahoo Finance News Intelligence Platform

A production-grade, containerized two-service system:

1. **Scraper service** — a resilient Yahoo Finance news bot that paginates
   topic listings, dedupes by URL, and stores raw news in Postgres.
2. **Analyzer service** — an independent job that reads the last 16 hours of
   news, groups it by ticker, fans it out to **multiple LLM personas**, and
   writes aggregated sentiment/rating rows.

The two services share **only the database** — restart, scale, or redeploy
either without touching the other.

> Full design rationale: see [`yahooscraper.md`](./yahooscraper.md) (the
> architecture spec this repo implements).

---

## Key differences from the spec (as requested)

### 1. LLM providers → OpenAI-compatible (NVIDIA NIM)

The spec used `litellm` with one provider per persona (OpenAI / Anthropic /
Gemini). This implementation speaks the **OpenAI wire protocol** to a single
configurable endpoint — defaulting to **NVIDIA's NIM endpoint**:

| Env var | Default |
|---|---|
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `LLM_API_KEY` | — (your `nvapi-...` key) |
| `ANALYST_MODEL` | `meta/llama-3.3-70b-instruct` |
| `RISK_MODEL` | `nvidia/llama-3.1-nemotron-70b-instruct` |
| `SWING_MODEL` | `qwen/qwen2.5-32b-instruct` |

* Uses the official `openai` SDK (`AsyncOpenAI(base_url=..., api_key=...)`).
* Each persona can run a **different NVIDIA-hosted model** — pick any id
  from <https://build.nvidia.com/explore>. All three can be the same model.
* Works with **any** OpenAI-compatible provider by changing `LLM_BASE_URL`
  (OpenAI, vLLM, Ollama `http://localhost:11434/v1`, Together, LM Studio...).
* `response_format={"type":"json_object"}` is attempted first and
  **auto-degrades** to plain prompting when a model rejects it; malformed
  LLM JSON gets one self-repair round-trip before the persona is dropped.
* `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` are gone. One key, one endpoint.

### 2. Scraper is fault-tolerant to structural changes

Yahoo redesigns its news listing periodically; a hardcoded CSS selector is a
ticking time bomb. The bot assumes **every layer can fail** and degrades
instead of dying:

```
fetch    : curl_cffi (TLS-spoof) → httpx (plain) → Playwright+stealth (403/challenge)
parse    : CSS selector sets → embedded JSON-LD → RSS feeds → heuristic anchors
protect  : circuit breaker (403/429 storm → 15-min pause) + token bucket + jitter
schedule : crash-proof loop, max_instances=1, graceful SIGTERM drain
persist  : idempotent ON CONFLICT inserts, DB-down ends the cycle cleanly
```

| Failure | Bot behavior |
|---|---|
| Yahoo renames CSS classes | JSON-LD strategy takes over (schema.org data survives redesigns) |
| JSON-LD also gone | RSS feeds (`/news/rssindex`) rescue the cycle |
| Everything DOM broken | Heuristic `/news/` anchor scan; drift alert logged |
| 0 articles for N consecutive pages | `SELECTOR DRIFT SUSPECTED` alert, cycle aborts **without writes**, retries next cycle |
| 403/429 rate ban | Circuit breaker opens → 15 min pause → single probe → re-open or close |
| Playwright not installed | Warning only; fast path continues |
| Postgres down | Cycle ends cleanly (`DB_UNAVAILABLE`), scheduler retries later |
| Malformed article card | Item skipped; rest of the page processes |
| One model down / returns junk | Other personas still aggregate; failing ticker's articles stay unprocessed for retry |
| Selector changes needed | **Hot-patch `scraper/selectors.yaml`** (volume-mounted, no rebuild) |

**Hot-patching selectors without a rebuild:**

```bash
# 1. inspect the new DOM in devtools
# 2. edit the first selector set in scraper/selectors.yaml
# 3. restart only the scraper
docker compose restart scraper
```

A malformed edit to the YAML can never take the bot down — the loader
validates it and falls back to built-in defaults with an error log.

---

## Architecture

```
┌─────────────────────┐          ┌──────────────────────┐
│   Scraper service   │          │   Analyzer service   │
│ python -m           │          │ python -m            │
│   scraper.scheduler │          │   analyzer.scheduler │
│                     │          │                      │
│ multi-backend fetch │          │ OpenAI-compatible    │
│ multi-strategy parse│          │ LLM personas (NVIDIA)│
│ circuit breaker     │          │ aggregate + upsert   │
└─────────┬───────────┘          └──────────┬───────────┘
          │        (only shared state)      │
          ▼                                 ▼
   ┌─────────────────────────────────────────────┐
   │  PostgreSQL: news_articles, stock_analysis  │
   └─────────────────────────────────────────────┘
   scraper: every 15 min   ·   analyzer: every 4 h (16 h lookback)
```

## Project layout

```
├── scraper/                  # Service 1
│   ├── config.py             #   env-driven settings (SCRAPER_* prefix)
│   ├── selectors.yaml        #   ★ hot-patchable selector registry
│   ├── selectors.py          #   validated loader (falls back to defaults)
│   ├── circuit_breaker.py    #   CLOSED/OPEN/HALF_OPEN state machine
│   ├── fetchers.py           #   curl_cffi → httpx → Playwright, retries
│   ├── parsing.py            #   CSS → JSON-LD → RSS → heuristics
│   ├── yahoo_news_scraper.py #   run loop + RunReport (never raises)
│   ├── user_agents.py        #   UA rotation + headers
│   └── scheduler.py          #   crash-proof APScheduler loop
├── analyzer/                 # Service 2
│   ├── config.py             #   env-driven settings (ANALYSIS_* / LLM_*)
│   ├── personas.py           #   3 personas, one NVIDIA model each
│   ├── llm_client.py         #   AsyncOpenAI → NIM; retries; JSON repair
│   ├── schemas.py            #   PersonaVerdict + defensive JSON extraction
│   ├── options_data.py       #   yfinance options summary (best-effort)
│   ├── pipeline.py           #   group → personas → aggregate → upsert
│   └── scheduler.py
├── db/
│   ├── models.py             # news_articles, stock_analysis (SQLAlchemy 2.0)
│   ├── repository.py         # idempotent ON CONFLICT writes, lazy engine
│   └── bootstrap.py          # create-all for local SQLite dev
├── alembic/                  # async migrations (0001_initial)
├── tests/                    # 64 tests incl. simulated Yahoo DOM redesign
├── docker-compose.yml · Dockerfile.scraper · Dockerfile.analyzer
└── requirements*.txt · .env.example · Makefile
```

## Schema (per spec, plus observability fields)

* **`news_articles`** — source, published_at, scraped_at, headline,
  description, url, `url_hash` (sha256 of normalized URL — query/fragment
  stripped, so tracker-bearing duplicates collapse), tickers, `processed`
  flag, and `parse_strategy` (which extraction path captured it — invaluable
  when debugging DOM drift).
* **`stock_analysis`** — analysis_date (UTC-midnight normalized so the
  unique `(analysis_date, ticker)` constraint dedupes re-runs), ticker,
  stock_name, confidence_score, sentiment, swing_trading_candidate,
  news_pointers, option_data_analysis, confidence_after_news_and_option,
  plus `llm_persona_votes` (full per-persona JSON audit trail).

## Quick start

### Docker (production path)

```bash
cp .env.example .env          # set DB_PASSWORD and LLM_API_KEY (nvapi-...)
docker compose up --build -d
docker compose logs -f scraper analyzer

# optional DB UI
docker compose --profile tools up -d pgadmin   # http://localhost:5050
```

`migrate` runs `alembic upgrade head` first; scraper/analyzer start only
after it succeeds.

### Local development (no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

export DATABASE_URL=sqlite+aiosqlite:///./news_intel.db
export LLM_API_KEY=nvapi-...                       # or OPENAI_API_KEY etc.
python -m db.bootstrap                             # create tables

python -m scraper.yahoo_news_scraper               # one scrape cycle
python -m analyzer.pipeline                        # one analysis cycle

python -m scraper.scheduler                        # long-running services
python -m analyzer.scheduler
```

> Note: `curl_cffi`/`playwright` are optional at runtime — if not installed
> the fetcher logs a warning and uses the remaining backends. For the full
> anti-bot stack, `pip install curl_cffi` and
> `playwright install --with-deps chromium`.

### Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

Highlights worth reading:

* `tests/test_parsing.py::test_jsonld_strategy_survives_redesign` —
  simulates a **complete Yahoo DOM redesign** (every class renamed, div
  cards); JSON-LD extraction recovers all articles.
* `tests/test_scraper_run.py` — fetch failures, circuit trips, DB outages
  and DOM drift all end in clean `RunReport`s; `run()` **never raises**.
* `tests/test_llm_client.py` — NVIDIA model routing per persona, JSON-mode
  degradation, repair round-trips, transient-error retries.

## Scraping etiquette (kept from the spec, enforced in code)

* Token bucket: 30 requests / 10 min — well under Yahoo's observed
  ~200 req/hour ceiling — plus 1.5–4 s random jitter per request and
  2–5 s between pages.
* Exponential backoff on 429/5xx; hard pause via circuit breaker on
  sustained 403/429.
* Playwright (full browser) only as a 403/challenge fallback, never default.
* Respect `robots.txt`, scrape only public listing pages, no login-walled
  content. Build for personal/research use.

## Runbook: Yahoo changed their markup (drift alert)

1. Log shows `SELECTOR DRIFT SUSPECTED` and/or `Page N produced 0 articles`.
2. Check which strategies were used in the run summary (`strategies=` map).
3. If `json-ld`/`rss` carried the cycle — no action needed; patch selectors
   at leisure.
4. If everything is empty: open devtools, update the first
   `selector_sets` entry in `scraper/selectors.yaml`, restart the scraper.
5. `parse_strategy` on stored rows tells you exactly which strategy each
   article came from — useful for verifying the fix.
