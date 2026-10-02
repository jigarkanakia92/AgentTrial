"""Strict response schemas + bulletproof JSON extraction for LLM output.

LLMs are unreliable formatters. This module assumes the model may:
* wrap JSON in ```json fences```,
* prefix/suffix it with prose ("Here is the analysis: ..."),
* use wrong casing or types ("sentiment": "bullish", confidence as "87"),
* return nothing useful at all.

Everything is coerced defensively; anything unfixable raises
:class:`PersonaResponseError` and the pipeline simply drops that persona.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Sentiment = Literal["Positive", "Negative", "Neutral"]

_SENTIMENT_ALIASES = {
    "positive": "Positive",
    "bullish": "Positive",
    "p": "Positive",
    "negative": "Negative",
    "bearish": "Negative",
    "n": "Negative",
    "neutral": "Neutral",
    "mixed": "Neutral",
    "unknown": "Neutral",
    "none": "Neutral",
}

_TRUTHY = {"true", "yes", "y", "1", "buy", "long"}
_FALSY = {"false", "no", "n", "0", "sell", "short", "avoid", "pass", ""}


class PersonaResponseError(ValueError):
    """The persona output could not be parsed into a valid verdict."""


class PersonaVerdict(BaseModel):
    """Validated single-persona verdict."""

    model_config = ConfigDict(extra="ignore")

    sentiment: Sentiment = "Neutral"
    confidence_score: float = Field(ge=0.0, le=100.0)
    swing_trading_candidate: bool = False
    news_pointers: list[str] = Field(default_factory=list, max_length=25)
    option_commentary: str | None = None

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalize_sentiment(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "Neutral"
        return _SENTIMENT_ALIASES.get(v.strip().lower(), "Neutral")

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            value = float(str(v).replace("%", "").strip())
        except (TypeError, ValueError):
            return 50.0  # midpoint rather than rejecting the whole verdict
        return max(0.0, min(100.0, value))

    @field_validator("swing_trading_candidate", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        text = str(v).strip().lower()
        if text in _TRUTHY:
            return True
        if text in _FALSY:
            return False
        return False

    @field_validator("news_pointers", mode="before")
    @classmethod
    def _clean_pointers(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        cleaned = [str(p).strip() for p in v if str(p).strip()]
        return cleaned[:25]

    @field_validator("option_commentary", mode="before")
    @classmethod
    def _clean_commentary(cls, v: Any) -> str | None:
        if v is None:
            return None
        text = str(v).strip()
        return text or None


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of arbitrary LLM prose.

    Handles: raw JSON, ```fenced``` JSON, prose-wrapped JSON. Raises
    PersonaResponseError when nothing parseable exists.
    """
    if not text or not text.strip():
        raise PersonaResponseError("empty response")

    candidate = text.strip()

    # 1) strip markdown fences if present
    if "```" in candidate:
        fence = candidate.split("```")
        # prefer a fenced block explicitly tagged json
        blocks = [
            b.strip().removeprefix("json").strip()
            for b in fence
            if b.strip()
        ]
        for block in blocks:
            if block.startswith("{"):
                candidate = block
                break

    # 2) direct parse
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3) brace matching: first '{' to its balanced close
    start = candidate.find("{")
    if start == -1:
        raise PersonaResponseError(f"no JSON object found in: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = candidate[start : index + 1]
                try:
                    parsed = json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise PersonaResponseError(
                        f"unparseable JSON object: {snippet[:200]!r}"
                    ) from exc
                if isinstance(parsed, dict):
                    return parsed
    raise PersonaResponseError(f"unbalanced JSON object in: {text[:200]!r}")


def parse_persona_verdict(raw_content: str) -> PersonaVerdict:
    """String -> validated PersonaVerdict. Raises PersonaResponseError."""
    data = extract_json_object(raw_content)
    try:
        return PersonaVerdict.model_validate(data)
    except ValidationError as exc:
        logger.debug("Verdict validation failed for {}: {}", str(data)[:200], exc)
        raise PersonaResponseError(f"schema mismatch: {exc.error_count()} errors") from exc
