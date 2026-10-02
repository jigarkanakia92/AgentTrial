"""Selector registry: hot-patchability + graceful failure."""
from __future__ import annotations

import textwrap

from scraper.parsing import NewsParser
from scraper.selectors import SelectorRegistryLoader

PAGE_URL = "https://finance.yahoo.com/topic/stock-market-news/1/"


def test_missing_file_falls_back_to_defaults():
    registry = SelectorRegistryLoader("definitely/not/here.yaml").load()
    assert registry.selector_sets  # built-in sets present
    assert registry.allowed_hosts == {"finance.yahoo.com"}


def test_malformed_yaml_falls_back_to_defaults(tmp_path):
    bad = tmp_path / "selectors.yaml"
    bad.write_text("selector_sets: [ {broken", encoding="utf-8")
    registry = SelectorRegistryLoader(bad).load()
    assert registry.source == "<defaults>"
    assert len(registry.selector_sets) == 2


def test_invalid_set_is_dropped_valid_set_kept(tmp_path):
    cfg = textwrap.dedent(
        """
        selector_sets:
          - name: broken-set
            card: 123           # not a string -> dropped
            headline: "a"
          - name: good-set
            card: "div.my-card"
            headline: "a.headline"
            description: "p.summary"
            time: "time[datetime]"
            ticker_link: "a[href*='/quote/']"
        """
    )
    path = tmp_path / "selectors.yaml"
    path.write_text(cfg, encoding="utf-8")

    registry = SelectorRegistryLoader(path).load()
    assert [s["name"] for s in registry.selector_sets] == ["good-set"]

    html = """
    <html><body>
      <div class="my-card">
        <a class="headline" href="/news/custom-markup-article-is-handled-gracefully.html">Custom markup article is handled gracefully by hot-patched selectors</a>
        <p class="summary">A description from the hypothetical new layout.</p>
        <time datetime="2026-10-01T09:00:00Z"></time>
        <a href="/quote/IBM">IBM</a>
      </div>
    </body></html>
    """
    page = NewsParser(registry).parse_page(html, PAGE_URL)
    assert page.strategy == "css:good-set"
    assert len(page.articles) == 1
    article = page.articles[0]
    assert article.url.endswith("custom-markup-article-is-handled-gracefully.html")
    assert article.tickers == ["IBM"]


def test_deep_merge_preserves_defaults_for_missing_keys(tmp_path):
    cfg = "page:\n  rss_feeds:\n    - https://example.com/feed.xml\n"
    path = tmp_path / "selectors.yaml"
    path.write_text(cfg, encoding="utf-8")
    registry = SelectorRegistryLoader(path).load()
    # override applied
    assert registry.rss_feeds == ["https://example.com/feed.xml"]
    # defaults preserved
    assert registry.selector_sets
    assert registry.headline_ticker_regex
