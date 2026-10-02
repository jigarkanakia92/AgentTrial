"""Hot-patchable selector registry.

The registry is loaded from a YAML file (default ``scraper/selectors.yaml``)
so Yahoo DOM changes can be absorbed by editing data — not code. Failure
modes are handled defensively:

* file missing                 -> built-in defaults
* file malformed / wrong types -> built-in defaults + ERROR log
* individual set invalid       -> that set dropped, remaining sets kept

The scraper re-validates and reloads the file on every run, so a mounted
volume edit + container restart (or just the next scheduler cycle if the
loader detects an mtime change) applies the patch without a rebuild.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

DEFAULT_REGISTRY: dict[str, Any] = {
    "version": 0,
    "updated": "built-in",
    "page": {
        "list_url_template": "https://finance.yahoo.com/topic/stock-market-news/{page}/",
        "rss_feeds": [
            "https://finance.yahoo.com/news/rssindex",
            "https://finance.yahoo.com/topic/stock-market-news/rssindex",
        ],
    },
    "min_articles_per_strategy": 3,
    "allowed_url_hosts": ["finance.yahoo.com"],
    "ticker_extraction": {
        "quote_link_pattern": "/quote/",
        "headline_ticker_regex": r"\(([A-Z]{1,6}(?:[-.][A-Z]{1,3})?)\)",
        "stoplist": [
            "AI", "CEO", "CFO", "CTO", "COO", "IPO", "SEC", "ETF", "NYSE",
            "NASDAQ", "DJIA", "EPS", "GDP", "FED", "IRA", "LLC", "Inc", "Corp",
        ],
    },
    "selector_sets": [
        {
            "name": "yahoo-stream-2024",
            "card": "li.js-stream-content, div[data-testid='storyitem'], div.story-item",
            "headline": "h3 a, a",
            "description": "p",
            "time": "time[datetime], time[data-timestamp]",
            "ticker_link": "a[href*='/quote/']",
        },
        {
            "name": "yahoo-web-components",
            "card": "li.stream-item, yf-fin-stream-item, section article, article",
            "headline": "h3 a, h2 a, a[aria-label], a",
            "description": "p",
            "time": "time[datetime]",
            "ticker_link": "a[href*='/quote/']",
        },
    ],
}

_REQUIRED_SET_KEYS = ("card", "headline")


class SelectorRegistry:
    """Validated snapshot of selectors + tuning knobs for one run."""

    def __init__(self, data: dict[str, Any], source: str) -> None:
        self.data = data
        self.source = source

    # -- convenience accessors -------------------------------------------------
    @property
    def list_url_template(self) -> str:
        return self.data["page"]["list_url_template"]

    @property
    def rss_feeds(self) -> list[str]:
        return list(self.data["page"].get("rss_feeds", []))

    @property
    def selector_sets(self) -> list[dict[str, str]]:
        return self.data["selector_sets"]

    @property
    def min_articles_per_strategy(self) -> int:
        return int(self.data.get("min_articles_per_strategy", 3))

    @property
    def allowed_hosts(self) -> set[str]:
        return set(self.data.get("allowed_url_hosts", ["finance.yahoo.com"]))

    @property
    def ticker_stoplist(self) -> set[str]:
        return {str(t).upper() for t in self.data["ticker_extraction"].get("stoplist", [])}

    @property
    def headline_ticker_regex(self) -> str:
        return self.data["ticker_extraction"].get(
            "headline_ticker_regex", r"\(([A-Z]{1,6}(?:[-.][A-Z]{1,3})?)\)"
        )

    @property
    def quote_link_pattern(self) -> str:
        return self.data["ticker_extraction"].get("quote_link_pattern", "/quote/")


class SelectorRegistryLoader:
    """Loads + hot-reloads the registry; never raises."""

    def __init__(self, path: str | Path = "scraper/selectors.yaml") -> None:
        self.path = Path(path)
        self._registry: SelectorRegistry | None = None
        self._mtime: float | None = None

    def load(self, force: bool = False) -> SelectorRegistry:
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
        except OSError:
            mtime = None

        if (
            not force
            and self._registry is not None
            and mtime == self._mtime
        ):
            return self._registry

        self._mtime = mtime
        self._registry = self._load_validated()
        return self._registry

    def _load_validated(self) -> SelectorRegistry:
        if not self.path.exists():
            if self._registry is None or self._registry.source != "<defaults>":
                logger.debug(
                    "Selector file {} not found — using built-in defaults", self.path
                )
            return SelectorRegistry(copy.deepcopy(DEFAULT_REGISTRY), "<defaults>")

        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                raise ValueError("YAML root must be a mapping")
            merged = self._deep_merge(copy.deepcopy(DEFAULT_REGISTRY), raw)
            merged["selector_sets"] = self._validate_sets(merged.get("selector_sets", []))
            if not merged["selector_sets"]:
                logger.error(
                    "No valid selector sets in {} — DOM sets disabled; "
                    "JSON-LD/RSS/heuristic strategies still active",
                    self.path,
                )
            registry = SelectorRegistry(merged, str(self.path))
            logger.info(
                "Selector registry loaded from {} (v{}, {} sets)",
                self.path,
                merged.get("version", "?"),
                len(merged["selector_sets"]),
            )
            return registry
        except Exception as exc:  # malformed YAML, wrong types, anything
            logger.error(
                "Failed to load selector file {} ({}). Falling back to built-in "
                "defaults — fix the file to apply your changes.",
                self.path,
                exc,
            )
            return SelectorRegistry(copy.deepcopy(DEFAULT_REGISTRY), "<defaults>")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                SelectorRegistryLoader._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _validate_sets(sets: Any) -> list[dict[str, str]]:
        valid: list[dict[str, str]] = []
        if not isinstance(sets, list):
            return valid
        for i, entry in enumerate(sets):
            if not isinstance(entry, dict):
                logger.warning("selector_sets[{}] is not a mapping — dropped", i)
                continue
            bad = [
                k for k in _REQUIRED_SET_KEYS
                if not isinstance(entry.get(k), str) or not entry.get(k, "").strip()
            ]
            if bad:
                logger.warning(
                    "selector_sets[{}] ('{}') has invalid/missing string keys {} "
                    "— dropped",
                    i,
                    entry.get("name", "?"),
                    bad,
                )
                continue
            valid.append({k: str(v) for k, v in entry.items()})
        return valid


# Module-level default loader used by the scraper; tests can instantiate their own.
_default_loader = SelectorRegistryLoader()


def get_registry() -> SelectorRegistry:
    return _default_loader.load()


def reload_registry(force: bool = True) -> SelectorRegistry:
    return _default_loader.load(force=force)
