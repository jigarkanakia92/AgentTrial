"""LLM persona definitions.

Each persona gets its OWN model id (resolved from AnalyzerSettings at call
time so env vars can swap models without code changes) and a system prompt
that pushes it into a distinct analytical role. All personas speak to the
same OpenAI-compatible endpoint.
"""
from __future__ import annotations

from typing import Callable

from analyzer.config import AnalyzerSettings

RESPONSE_SCHEMA = """Return STRICT JSON only, with exactly these keys:
{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "confidence_score": <float 0-100>,
  "swing_trading_candidate": <true|false>,
  "news_pointers": ["short reason 1", "short reason 2"],
  "option_commentary": "<string or null>"
}
No prose, no markdown fences, JSON only."""

PERSONAS: dict[str, dict] = {
    "news_fundamentalist": {
        "model_setting": "model_news_analyst",
        "system_prompt": (
            "You are a senior equity research analyst with 20 years of experience. "
            "You read news headlines and descriptions about a stock and assess "
            "sentiment, materiality, and short-term price impact. Be skeptical of "
            "hype and distinguish company-specific news from market-wide noise."
        ),
    },
    "risk_manager": {
        "model_setting": "model_risk_manager",
        "system_prompt": (
            "You are a conservative risk manager at a hedge fund. Your job is to "
            "find reasons NOT to trade a stock based on the news. Flag red flags, "
            "regulatory risk, litigation, dilution, or unverified rumors. Assign a "
            "confidence_score reflecting the strength of evidence in the news."
        ),
    },
    "swing_trader": {
        "model_setting": "model_swing_trader",
        "system_prompt": (
            "You are an aggressive swing trader focused on 2-10 day holding "
            "periods. Given recent news and (if provided) options chain data, "
            "judge whether this stock is a good swing trading candidate right "
            "now and how confident you are in that call."
        ),
    },
}


def persona_model(persona_key: str, settings: AnalyzerSettings) -> str:
    """Resolve the configured model id for a persona."""
    return getattr(settings, PERSONAS[persona_key]["model_setting"])


def persona_system_prompt(persona_key: str) -> str:
    return PERSONAS[persona_key]["system_prompt"]


def build_user_prompt(
    ticker: str,
    news_bundle: str,
    option_data: str | None,
    schema: str = RESPONSE_SCHEMA,
) -> str:
    return (
        f"Ticker: {ticker}\n\n"
        f"Recent news (last 16 hours):\n{news_bundle}\n\n"
        f"Options data (may be unavailable):\n{option_data or 'N/A'}\n\n"
        f"{schema}"
    )


def persona_keys() -> list[str]:
    return list(PERSONAS.keys())


def model_resolver(settings: AnalyzerSettings) -> Callable[[str], str]:
    return lambda key: persona_model(key, settings)
