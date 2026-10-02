"""Multi-strategy parsing tests — the core of 'DOM changes must not break
the bot'. Each test simulates one real-world failure mode of Yahoo markup."""
from __future__ import annotations


from scraper.parsing import (
    NewsParser,
    normalize_url,
    parse_published,
)
from scraper.selectors import SelectorRegistryLoader

PAGE_URL = "https://finance.yahoo.com/topic/stock-market-news/1/"


def make_parser() -> NewsParser:
    return NewsParser(SelectorRegistryLoader("nonexistent.yaml").load())


# ---------------------------------------------------------------------------
# Strategy 1: CSS selectors on the classic DOM
# ---------------------------------------------------------------------------


def test_css_strategy_extracts_classic_dom(classic_html):
    parser = make_parser()
    page = parser.parse_page(classic_html, PAGE_URL)

    assert len(page.articles) == 5
    assert page.strategy.startswith("css:")
    assert not page.below_threshold

    first = page.articles[0]
    assert first.headline.startswith("Apple unveils")
    assert first.url == "https://finance.yahoo.com/news/apple-unveils-m4-chip-123045678.html"
    assert first.tickers == ["AAPL", "MSFT"]
    assert first.published_at.year == 2026
    assert first.description.startswith("Apple showed off")


def test_relative_and_protocol_relative_urls_resolved(classic_html):
    parser = make_parser()
    page = parser.parse_page(classic_html, PAGE_URL)
    urls = {a.url for a in page.articles}
    assert "https://finance.yahoo.com/news/tesla-deliveries-beat-q3-093011223.html" in urls
    assert "https://finance.yahoo.com/news/fed-officials-signal-patient-rate-path-114522334.html" in urls


def test_ticker_regex_fallback_on_text(classic_html):
    parser = make_parser()
    page = parser.parse_page(classic_html, PAGE_URL)
    by_headline = {a.headline: a for a in page.articles}
    # card with no /quote/ links still gets the ticker from "(AAPL)" in text
    assert by_headline[
        "Apple unveils next-gen M5 chip as its AI push accelerates (AAPL)"
    ].tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Strategy 2: JSON-LD rescues a complete redesign
# ---------------------------------------------------------------------------


def test_jsonld_strategy_survives_redesign(redesign_html):
    """The big one: Yahoo renamed every class and switched to div cards.
    CSS strategies find nothing; JSON-LD saves the run."""
    parser = make_parser()
    page = parser.parse_page(redesign_html, PAGE_URL)

    assert len(page.articles) == 4
    assert page.strategy == "json-ld"

    amzn = page.articles[0]
    assert amzn.headline.startswith("Amazon expands")
    assert amzn.url.endswith("amazon-same-day-logistics-expansion-180011001.html")
    assert amzn.tickers == ["AMZN"]  # from headline regex
    assert amzn.published_at.year == 2026

    # mainEntityOfPage.@id form resolves too
    netflix = page.articles[-1]
    assert netflix.url.endswith("netflix-ad-tier-refresh-210044004.html")


# ---------------------------------------------------------------------------
# Strategy 4: heuristic anchor scan (no JSON-LD either)
# ---------------------------------------------------------------------------


def test_heuristic_strategy_last_resort():
    html = """
    <html><body><div class="totally-new-layout">
      <a href="/news/stock-market-rallies-on-fed-hopes-today-late-edition.html">Stock market rallies on Fed hopes as tech leads a broad advance</a>
      <a href="/news/retail-earnings-season-kicks-off-with-mixed-results-report.html">Retail earnings season kicks off with mixed results and cautious guidance</a>
      <a href="/news/energy-stocks-slide-as-crude-inventories-build-again-update.html">Energy stocks slide as crude inventories build for a fifth straight week</a>
      <a href="/markets/">Markets section</a>
      <a href="https://other-site.com/news/external-article-that-is-very-long.html">External article that should be filtered by host allowlist</a>
    </div></body></html>
    """
    parser = make_parser()
    page = parser.parse_page(html, PAGE_URL)

    assert page.strategy == "heuristics"
    assert len(page.articles) == 3
    urls = {a.url for a in page.articles}
    assert all(u.startswith("https://finance.yahoo.com/news/") for u in urls)


# ---------------------------------------------------------------------------
# Total garbage never crashes; drift is reported
# ---------------------------------------------------------------------------


def test_garbage_html_returns_empty_without_raising():
    for junk in ("", "<html><body><p>hello</p></body></html>", None, "<<<broken>>>"):
        parser = make_parser()
        page = parser.parse_page(junk or "", PAGE_URL)
        assert page.articles == []
        assert page.below_threshold is True


def test_malformed_jsonld_block_is_skipped_not_fatal():
    html = """
    <html><head>
      <script type="application/ld+json">{"@type":"NewsArticle", broken json</script>
      <script type="application/ld+json">{"@type":"NewsArticle","headline":"Valid article survives malformed siblings","url":"https://finance.yahoo.com/news/valid-article-123.html","datePublished":"2026-10-01T10:00:00Z"}</script>
    </head><body></body></html>
    """
    parser = make_parser()
    page = parser.parse_page(html, PAGE_URL)
    # 1 valid article < min_articles(3) → below-threshold structured partial
    assert len(page.articles) == 1
    assert page.articles[0].headline == "Valid article survives malformed siblings"


# ---------------------------------------------------------------------------
# Unit: URL normalization + published parsing
# ---------------------------------------------------------------------------


def test_normalize_url_strips_trackers_and_validates_host():
    base = "https://finance.yahoo.com/topic/x/"
    assert normalize_url("/news/a.html?guccounter=1#top", base, {"finance.yahoo.com"}) == \
        "https://finance.yahoo.com/news/a.html"
    assert normalize_url("//finance.yahoo.com/news/b.html", base, {"finance.yahoo.com"}) == \
        "https://finance.yahoo.com/news/b.html"
    assert normalize_url("https://evil.com/news/c.html", base, {"finance.yahoo.com"}) is None
    assert normalize_url("javascript:void(0)", base, {"finance.yahoo.com"}) is None
    assert normalize_url("", base, {"finance.yahoo.com"}) is None


def test_parse_published_variants():
    from datetime import datetime, timezone

    assert parse_published("2026-10-01T14:30:00Z") == datetime(
        2026, 10, 1, 14, 30, tzinfo=timezone.utc
    )
    assert parse_published("2026-10-01T14:30:00.000Z").hour == 14
    assert parse_published(1790816400) == datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc)
    assert parse_published(1790816400000) == datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc)  # ms
    assert parse_published("not a date") is None
    assert parse_published(None) is None


# ---------------------------------------------------------------------------
# RSS strategy
# ---------------------------------------------------------------------------


def test_rss_strategy_parses_feed(rss_xml):
    from scraper.parsing import parse_rss
    from scraper.selectors import SelectorRegistryLoader

    registry = SelectorRegistryLoader("nonexistent.yaml").load()
    articles = parse_rss(rss_xml, registry)

    assert len(articles) == 3
    assert all(a.strategy == "rss" for a in articles)
    first = articles[0]
    assert first.url.endswith("oil-prices-slide-opec-080011999.html")
    assert first.tickers == ["XOM"]
    assert first.published_at.hour == 8
