"""Aggregation logic + article grouping (pure functions, no I/O)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from analyzer.pipeline import (
    aggregate_persona_results,
    group_articles_by_ticker,
)
from analyzer.schemas import PersonaVerdict
from db.models import NewsArticle


def verdict(sentiment="Positive", score=70.0, swing=True, pointers=None, option=None):
    return PersonaVerdict(
        sentiment=sentiment,
        confidence_score=score,
        swing_trading_candidate=swing,
        news_pointers=pointers or ["pointer"],
        option_commentary=option,
    )


def make_article(tickers: str | None) -> NewsArticle:
    return NewsArticle(
        headline=f"Headline {uuid4().hex[:8]}",
        url=f"https://finance.yahoo.com/news/{uuid4().hex}.html",
        url_hash=uuid4().hex,
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
        tickers=tickers,
    )


# ---------------------------------------------------------------------------


def test_majority_sentiment_and_average_confidence():
    result = aggregate_persona_results(
        {
            "news_fundamentalist": verdict("Positive", 80, swing=False),
            "risk_manager": verdict("Positive", 70, swing=False),
            "swing_trader": verdict("Negative", 60, swing=True),
        },
        option_data=None,
    )
    assert result["sentiment"] == "Positive"
    assert result["confidence_score"] == 70.0
    assert result["swing_trading_candidate"] is False  # 1/3 votes
    assert result["confidence_after_news_and_option"] == 70.0


def test_tie_breaks_to_neutral_then_positive():
    tie = aggregate_persona_results(
        {
            "a": verdict("Positive", 50),
            "b": verdict("Negative", 50),
        },
        None,
    )
    assert tie["sentiment"] == "Neutral"

    tie2 = aggregate_persona_results(
        {
            "a": verdict("Positive", 50),
            "b": verdict("Neutral", 50),
            "c": verdict("Negative", 50),
        },
        None,
    )
    assert tie2["sentiment"] == "Neutral"


def test_swing_strict_majority():
    result = aggregate_persona_results(
        {
            "a": verdict(swing=True),
            "b": verdict(swing=True),
            "c": verdict(swing=False),
        },
        None,
    )
    assert result["swing_trading_candidate"] is True


def test_pointers_merged_with_persona_attribution():
    result = aggregate_persona_results(
        {
            "news_fundamentalist": verdict(pointers=["beat on revenue", "guidance raised"]),
            "risk_manager": verdict(sentiment="Negative", pointers=["valuation rich"]),
        },
        None,
    )
    assert result["news_pointers"] == [
        "[news_fundamentalist] beat on revenue",
        "[news_fundamentalist] guidance raised",
        "[risk_manager] valuation rich",
    ]


def test_bearish_option_skew_dampens_positive_confidence():
    result = aggregate_persona_results(
        {
            "a": verdict("Positive", 80),
            "b": verdict("Positive", 60),
        },
        option_data="ATM put IV skew suggests high_iv_skew_bearish hedging",
    )
    assert result["confidence_score"] == 70.0
    assert result["confidence_after_news_and_option"] == 59.5  # 70 * 0.85


def test_bearish_option_skew_does_not_touch_negative_calls():
    result = aggregate_persona_results(
        {"a": verdict("Negative", 60), "b": verdict("Negative", 40)},
        option_data="high_iv_skew_bearish",
    )
    assert result["confidence_after_news_and_option"] == 50.0  # unchanged


def test_persona_votes_audited():
    result = aggregate_persona_results({"a": verdict("Positive", 55)}, None)
    assert result["llm_persona_votes"]["a"]["sentiment"] == "Positive"


# ---------------------------------------------------------------------------


def test_grouping_splits_multi_ticker_articles():
    aapl_msft = make_article("AAPL,MSFT")
    tsla = make_article("TSLA")
    none = make_article(None)

    grouped = group_articles_by_ticker([aapl_msft, tsla, none])
    assert set(grouped.keys()) == {"AAPL", "MSFT", "TSLA"}
    assert grouped["AAPL"] == [aapl_msft]
    assert grouped["MSFT"] == [aapl_msft]
    assert grouped["TSLA"] == [tsla]


def test_grouping_normalizes_case_and_ignores_blanks():
    grouped = group_articles_by_ticker([make_article(" aapl , , MSFT ")])
    assert set(grouped.keys()) == {"AAPL", "MSFT"}
