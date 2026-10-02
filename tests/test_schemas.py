"""LLM output parsing: fences, prose wrappers, type coercion, garbage."""
from __future__ import annotations

import json

import pytest

from analyzer.schemas import (
    PersonaResponseError,
    extract_json_object,
    parse_persona_verdict,
)

GOOD = {
    "sentiment": "Positive",
    "confidence_score": 72.5,
    "swing_trading_candidate": True,
    "news_pointers": ["strong deliveries beat", "EV margins improving"],
    "option_commentary": None,
}


def test_extract_raw_json():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    assert extract_json_object(text)["sentiment"] == "Positive"


def test_extract_prose_wrapped_json():
    text = 'Here is my analysis:\n{"sentiment": "Negative", "confidence_score": 40}\nHope that helps!'
    assert extract_json_object(text)["confidence_score"] == 40


def test_extract_json_with_braces_inside_strings():
    text = '{"note": "use {curly} braces", "sentiment": "Neutral", "confidence_score": 5}'
    parsed = extract_json_object(text)
    assert parsed["note"] == "use {curly} braces"


def test_extract_garbage_raises():
    with pytest.raises(PersonaResponseError):
        extract_json_object("")
    with pytest.raises(PersonaResponseError):
        extract_json_object("no json here at all")
    with pytest.raises(PersonaResponseError):
        extract_json_object('{"unbalanced": true')


def test_verdict_coercion_aliases_and_types():
    verdict = parse_persona_verdict(
        '{"sentiment": "BULLISH", "confidence_score": "87%", '
        '"swing_trading_candidate": "yes", "news_pointers": "one pointer only"}'
    )
    assert verdict.sentiment == "Positive"
    assert verdict.confidence_score == 87.0
    assert verdict.swing_trading_candidate is True
    assert verdict.news_pointers == ["one pointer only"]


def test_verdict_clamps_and_defaults():
    verdict = parse_persona_verdict(
        '{"sentiment": "idk", "confidence_score": 150, "swing_trading_candidate": false}'
    )
    assert verdict.sentiment == "Neutral"       # unknown alias -> Neutral
    assert verdict.confidence_score == 100.0    # clamped
    assert verdict.option_commentary is None


def test_verdict_coerces_structural_garbage_defensively():
    """Design: wrong TYPES are coerced (list sentiment -> Neutral, bad
    confidence -> midpoint) instead of rejecting the whole verdict; only
    unparseable payloads raise PersonaResponseError."""
    verdict = parse_persona_verdict(
        '{"sentiment": ["not", "a", "string"], "confidence_score": "abc", "extra": 1}'
    )
    assert verdict.sentiment == "Neutral"
    assert verdict.confidence_score == 50.0

    verdict2 = parse_persona_verdict(
        '{"sentiment": "bearish", "confidence_score": "abc"}'
    )
    assert verdict2.confidence_score == 50.0
    assert verdict2.sentiment == "Negative"


def test_extra_keys_ignored():
    verdict = parse_persona_verdict(
        '{"sentiment": "Positive", "confidence_score": 60, "irrelevant": "x", '
        '"reasoning": "blah blah"}'
    )
    assert verdict.confidence_score == 60
