"""Multi-strategy article extraction — the core DOM-drift insurance.

Why four strategies? Yahoo redesigns its news listing markup every so often.
A single CSS selector set is a ticking time bomb. This parser degrades
gracefully through strategies ordered by precision:

1. ``css:<set>``   — CSS selectors from the hot-patchable registry (precise,
                     breaks when markup changes)
2. ``json-ld``     — <script type="application/ld+json"> NewsArticle objects
                     embedded by the site; survives almost any CSS redesign
3. ``rss``         — Yahoo's RSS feeds; survives any HTML change at all
4. ``heuristics``  — any <a href=".../news/..."> with substantial text
                     (last resort, no timestamps)

Each strategy is individually exception-isolated: a strategy that explodes
is skipped with a warning, never taking the run down. The first strategy
producing >= ``min_articles_per_strategy`` valid articles wins; otherwise
the best structured (css/json-ld) partial result is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urldefrag, urljoin, urlparse

from loguru import logger
from pydantic import BaseModel, Field, field_validator

try:
    from selectolax.parser import HTMLParser

    SELECTOLAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional install
    HTMLParser = None  # type: ignore[assignment]
    SELECTOLAX_AVAILABLE = False

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except Exception:  # pragma: no cover - optional install
    feedparser = None  # type: ignore[assignment]
    FEEDPARSER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Article model
# ---------------------------------------------------------------------------

DEFAULT_TICKER_RE = r"\(([A-Z]{1,6}(?:[-.][A-Z]{1,3})?)\)"


class ScrapedArticle(BaseModel):
    """One validated article regardless of which strategy produced it."""

    headline: str = Field(min_length=8)
    url: str
    description: str = ""
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tickers: list[str] = Field(default_factory=list)
    source: str = "Yahoo Finance"
    strategy: str = "unknown"

    @field_validator("headline", "description")
    @classmethod
    def _clean_text(cls, v: str) -> str:
        return re.sub(r"\s+", " ", (v or "")).strip()

    @field_validator("published_at")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def url_hash(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()

    def to_row(self) -> dict[str, Any]:
        """Flatten to a news_articles DB row dict."""
        return {
            "headline": self.headline[:2000],
            "description": (self.description or None),
            "url": self.url,
            "url_hash": self.url_hash,
            "published_at": self.published_at,
            "tickers": ",".join(self.tickers) or None,
            "source": self.source,
            "parse_strategy": self.strategy,
        }


class ParsedPage(BaseModel):
    articles: list[ScrapedArticle]
    strategy: str
    rejected: int = 0
    below_threshold: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_url(raw_url: str, base_url: str, allowed_hosts: set[str]) -> str | None:
    """Absolute-ize, strip trackers (query/fragment), validate host."""
    if not raw_url:
        return None
    raw_url = raw_url.strip()
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    absolute = urljoin(base_url, raw_url)
    absolute, _ = urldefrag(absolute)
    absolute = absolute.split("?", 1)[0]
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    host = parsed.netloc.lower()
    if allowed_hosts and host not in allowed_hosts:
        return None
    return absolute


def parse_published(value: Any) -> datetime | None:
    """Best-effort datetime from ISO strings, epoch seconds/millis, or None."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:  # milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract_tickers_from_text(text: str, regex: str, stoplist: set[str]) -> list[str]:
    """`(AAPL)`-style parenthetical ticker extraction from headline/description."""
    found: list[str] = []
    for match in re.findall(regex or DEFAULT_TICKER_RE, text or ""):
        symbol = match.upper().strip()
        if not symbol or symbol in stoplist:
            continue
        if symbol not in found:
            found.append(symbol)
    return found


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class CssStrategy:
    name = "css"

    def parse(self, html: str, page_url: str, registry) -> list[ScrapedArticle]:
        if not SELECTOLAX_AVAILABLE:
            raise RuntimeError("selectolax is required for CSS parsing")
        articles: list[ScrapedArticle] = []
        seen: set[str] = set()
        for sset in registry.selector_sets:
            count_before = len(articles)
            try:
                articles.extend(
                    self._parse_with_set(html, page_url, registry, sset, seen)
                )
            except Exception as exc:  # noqa: BLE001 — one bad set never kills all
                logger.warning(
                    "CSS selector set '{}' failed: {}", sset.get("name", "?"), exc
                )
            if len(articles) > count_before and articles:
                break  # first productive set wins; do not double-count others
        return articles

    def _parse_with_set(self, html, page_url, registry, sset, seen):
        tree = HTMLParser(html)
        out: list[ScrapedArticle] = []
        for card in tree.css(sset["card"]):
            try:
                article = self._parse_card(card, page_url, registry, sset)
            except Exception:  # noqa: BLE001 — a malformed card is skipped
                continue
            if article is None or article.url in seen:
                continue
            seen.add(article.url)
            out.append(article)
        return out

    def _parse_card(self, card, page_url, registry, sset) -> ScrapedArticle | None:
        link = card.css_first("a[href]")
        if link is None:
            return None
        href = (link.attributes.get("href") or "").strip()
        url = normalize_url(href, page_url, registry.allowed_hosts)
        if not url:
            return None

        headline = ""
        for sel in filter(None, [sset.get("headline"), "h3", "h2", "a"]):
            node = card.css_first(sel)
            if node is not None:
                headline = _node_text(node)
                if headline:
                    break
        if not headline:
            headline = _node_text(link)
        if len(headline) < 8:
            return None

        description = ""
        if sset.get("description"):
            desc_node = card.css_first(sset["description"])
            if desc_node is not None:
                description = _node_text(desc_node)

        published = None
        if sset.get("time"):
            time_node = card.css_first(sset["time"])
            if time_node is not None:
                attrs = time_node.attributes
                published = parse_published(
                    attrs.get("datetime") or attrs.get("data-timestamp")
                )

        tickers = self._tickers_from_card(card, registry)
        if not tickers:
            tickers = extract_tickers_from_text(
                f"{headline} {description}",
                registry.headline_ticker_regex,
                registry.ticker_stoplist,
            )

        return ScrapedArticle(
            headline=headline,
            url=url,
            description=description,
            published_at=published or datetime.now(timezone.utc),
            tickers=tickers,
            strategy=f"css:{sset.get('name', '?')}",
        )

    def _tickers_from_card(self, card, registry) -> list[str]:
        pattern = registry.quote_link_pattern
        tickers: list[str] = []
        for a in card.css("a[href]"):
            href = a.attributes.get("href") or ""
            if pattern in href:
                symbol = href.split(pattern)[-1].split("/")[0].split("?")[0]
                symbol = re.sub(r"[^A-Z0-9.\-]", "", symbol.upper())
                if symbol and len(symbol) <= 8 and symbol not in registry.ticker_stoplist:
                    if symbol not in tickers:
                        tickers.append(symbol)
        return tickers


class JsonLdStrategy:
    """Embedded schema.org NewsArticle JSON — the most redesign-proof signal."""

    name = "json-ld"
    _TYPES = {"newsarticle", "article", "newsposting", "reportagenewsarticle"}

    def parse(self, html: str, page_url: str, registry) -> list[ScrapedArticle]:
        if not SELECTOLAX_AVAILABLE:
            raise RuntimeError("selectolax is required for JSON-LD extraction")
        tree = HTMLParser(html)
        articles: list[ScrapedArticle] = []
        seen: set[str] = set()
        for node in tree.css('script[type="application/ld+json"]'):
            raw = node.text() if node.text() else ""
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue  # malformed block — try the next one
            for item in self._walk(payload):
                try:
                    article = self._to_article(item, page_url, registry)
                except Exception:  # noqa: BLE001
                    continue
                if article and article.url not in seen:
                    seen.add(article.url)
                    articles.append(article)
        return articles

    def _walk(self, payload: Any):
        """Yield dicts that look like NewsArticle-ish objects from any layout:
        a bare object, a list, or an @graph container. Preserves order."""
        stack = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                stack.extend(reversed(current))  # keep document order (LIFO)
            elif isinstance(current, dict):
                types = current.get("@type", "")
                types = {t.lower() for t in types} if isinstance(types, list) else {str(types).lower()}
                if types & self._TYPES:
                    yield current
                graph = current.get("@graph")
                if graph:
                    stack.append(graph)

    def _to_article(self, item: dict, page_url: str, registry) -> ScrapedArticle | None:
        headline = str(item.get("headline") or item.get("alternativeHeadline") or "").strip()
        if len(headline) < 8:
            return None
        raw_url = item.get("url") or item.get("mainEntityOfPage")
        if isinstance(raw_url, dict):
            raw_url = raw_url.get("@id")
        url = normalize_url(str(raw_url or ""), page_url, registry.allowed_hosts)
        if not url:
            return None
        published = parse_published(
            item.get("datePublished") or item.get("dateCreated") or item.get("uploadDate")
        )
        description = str(item.get("description") or "").strip()
        tickers = extract_tickers_from_text(
            f"{headline} {description}",
            registry.headline_ticker_regex,
            registry.ticker_stoplist,
        )
        return ScrapedArticle(
            headline=headline,
            url=url,
            description=description,
            published_at=published or datetime.now(timezone.utc),
            tickers=tickers,
            strategy=self.name,
        )


class HeuristicAnchorStrategy:
    """Last-resort scan: every substantial /news/ link on the page."""

    name = "heuristics"
    _NOISE_PATTERNS = (
        "/news/author/", "mailto:", "javascript:", "/videos/", "/live-tv/",
    )

    def parse(self, html: str, page_url: str, registry) -> list[ScrapedArticle]:
        if not SELECTOLAX_AVAILABLE:
            raise RuntimeError("selectolax is required for heuristic parsing")
        tree = HTMLParser(html)
        articles: list[ScrapedArticle] = []
        seen: set[str] = set()
        for a in tree.css("a[href]"):
            href = a.attributes.get("href") or ""
            if "/news/" not in href:
                continue
            if any(p in href for p in self._NOISE_PATTERNS):
                continue
            headline = _node_text(a)
            if len(headline) < 25:  # substantial text only — nav links are short
                continue
            url = normalize_url(href, page_url, registry.allowed_hosts)
            if not url or url in seen:
                continue
            seen.add(url)
            articles.append(
                ScrapedArticle(
                    headline=headline[:500],
                    url=url,
                    description="",
                    published_at=datetime.now(timezone.utc),
                    tickers=extract_tickers_from_text(
                        headline, registry.headline_ticker_regex, registry.ticker_stoplist
                    ),
                    strategy=self.name,
                )
            )
        return articles


def _node_text(node) -> str:
    try:
        return re.sub(r"\s+", " ", node.text() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# RSS strategy (network — orchestrated by the scraper, not parse_page)
# ---------------------------------------------------------------------------


def parse_rss(xml_text: str, registry) -> list[ScrapedArticle]:
    """Parse an RSS/Atom feed body into articles (sync; feedparser optional)."""
    articles: list[ScrapedArticle] = []
    if FEEDPARSER_AVAILABLE:
        feed = feedparser.parse(xml_text)
        for entry in feed.entries:
            try:
                headline = re.sub(r"\s+", " ", entry.get("title", "")).strip()
                if len(headline) < 8:
                    continue
                url = normalize_url(
                    entry.get("link", ""), "https://finance.yahoo.com", registry.allowed_hosts
                )
                if not url:
                    continue
                published = None
                if entry.get("published_parsed"):
                    import calendar

                    published = datetime.fromtimestamp(
                        calendar.timegm(entry.published_parsed), tz=timezone.utc
                    )
                description = re.sub(
                    r"<[^>]+>", " ", entry.get("summary", "") or ""
                )
                description = re.sub(r"\s+", " ", description).strip()
                articles.append(
                    ScrapedArticle(
                        headline=headline,
                        url=url,
                        description=description,
                        published_at=published or datetime.now(timezone.utc),
                        tickers=extract_tickers_from_text(
                            f"{headline} {description}",
                            registry.headline_ticker_regex,
                            registry.ticker_stoplist,
                        ),
                        strategy="rss",
                    )
                )
            except Exception:  # noqa: BLE001 — skip malformed entries
                continue
        return articles

    # Minimal regex fallback when feedparser is not installed.
    for title, link in re.findall(
        r"<item>.*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>.*?"
        r"<link>(.*?)</link>.*?</item>",
        xml_text or "",
        re.DOTALL,
    ):
        try:
            headline = re.sub(r"\s+", " ", title).strip()
            url = normalize_url(
                link.strip(), "https://finance.yahoo.com", registry.allowed_hosts
            )
            if headline and url:
                articles.append(
                    ScrapedArticle(
                        headline=headline, url=url, strategy="rss-fallback"
                    )
                )
        except Exception:  # noqa: BLE001
            continue
    return articles


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class NewsParser:
    """Runs strategies in order; first one meeting the article threshold wins."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self._css = CssStrategy()
        self._jsonld = JsonLdStrategy()
        self._heuristics = HeuristicAnchorStrategy()
        self.strategy_usage: Counter = Counter()

    def parse_page(self, html: str, page_url: str) -> ParsedPage:
        """Never raises. Returns whatever the best strategy extracted."""
        minimum = self.registry.min_articles_per_strategy
        rejected_total = 0
        best_structured: list[ScrapedArticle] = []

        # 1) CSS selector sets (ordered, precise)
        try:
            css_articles = self._isolate(lambda: self._css.parse(html, page_url, self.registry))
        except Exception as exc:  # noqa: BLE001 — paranoia: strategy isolation
            logger.warning("CSS strategy exploded and was skipped: {}", exc)
            css_articles = []
        structured = [a for a in css_articles]
        if len(structured) >= minimum:
            return self._record(structured, structured[0].strategy, rejected_total, False)
        if len(structured) > len(best_structured):
            best_structured = structured

        # 2) JSON-LD
        try:
            jsonld = self._isolate(lambda: self._jsonld.parse(html, page_url, self.registry))
        except Exception as exc:  # noqa: BLE001
            logger.warning("JSON-LD strategy exploded and was skipped: {}", exc)
            jsonld = []
        if len(jsonld) >= minimum:
            return self._record(jsonld, self._jsonld.name, rejected_total, False)
        if len(jsonld) > len(best_structured):
            best_structured = jsonld

        # 3) Heuristic anchor scan (must clear the bar on its own)
        try:
            heur = self._isolate(lambda: self._heuristics.parse(html, page_url, self.registry))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Heuristic strategy exploded and was skipped: {}", exc)
            heur = []
        if len(heur) >= minimum:
            return self._record(heur, self._heuristics.name, rejected_total, False)

        # Everything under threshold: keep the best structured partial (if any),
        # otherwise signal a likely DOM drift by returning an empty page.
        if best_structured:
            return self._record(best_structured, best_structured[0].strategy, rejected_total, True)
        return self._record([], "none", rejected_total, True)

    async def parse_rss_feeds(
        self, fetch: Callable[[str], Awaitable[str]]
    ) -> list[ScrapedArticle]:
        """Fetch + parse every configured RSS feed; merge and dedupe."""
        merged: list[ScrapedArticle] = []
        seen: set[str] = set()
        for feed_url in self.registry.rss_feeds:
            try:
                body = await fetch(feed_url)
            except Exception as exc:  # noqa: BLE001 — a dead feed is skipped
                logger.warning("RSS feed {} failed: {}", feed_url, exc)
                continue
            for article in parse_rss(body, self.registry):
                if article.url not in seen:
                    seen.add(article.url)
                    merged.append(article)
        if merged:
            self.strategy_usage["rss"] += len(merged)
        return merged

    def _isolate(self, fn):
        return fn()

    def _record(self, articles, strategy, rejected, below_threshold) -> ParsedPage:
        self.strategy_usage[strategy] += len(articles)
        return ParsedPage(
            articles=articles,
            strategy=strategy,
            rejected=rejected,
            below_threshold=below_threshold,
        )
