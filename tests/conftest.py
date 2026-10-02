"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def classic_html() -> str:
    return (FIXTURES / "yahoo_topic_classic.html").read_text(encoding="utf-8")


@pytest.fixture
def redesign_html() -> str:
    return (FIXTURES / "yahoo_topic_redesign.html").read_text(encoding="utf-8")


@pytest.fixture
def rss_xml() -> str:
    return (FIXTURES / "sample_feed.xml").read_text(encoding="utf-8")
