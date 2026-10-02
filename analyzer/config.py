"""Analyzer service configuration.

The LLM layer speaks the OpenAI wire protocol to whatever endpoint
``LLM_BASE_URL`` points at. Default is NVIDIA's hosted NIM endpoint
(``https://integrate.api.nvidia.com/v1``) with an ``nvapi-...`` key —
any other OpenAI-compatible provider (OpenAI, vLLM, Ollama, Together,
LM Studio, ...) works by changing the two env vars.
"""
from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalyzerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./news_intel.db",
        validation_alias=AliasChoices("DATABASE_URL", "ANALYSIS_DATABASE_URL"),
    )

    # --- OpenAI-compatible endpoint (NVIDIA NIM by default) ------------------
    llm_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "NVIDIA_BASE_URL"),
    )
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"
        ),
    )
    llm_timeout_seconds: float = 90.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    llm_retry_attempts: int = 3
    llm_retry_min_wait: float = 2.0
    llm_retry_max_wait: float = 20.0
    use_json_mode: bool = True   # auto-degrades if the model rejects it

    # --- one model per persona (any NIM model id) ----------------------------
    model_news_analyst: str = Field(
        default="meta/llama-3.3-70b-instruct",
        validation_alias=AliasChoices("ANALYST_MODEL", "MODEL_NEWS_ANALYST"),
    )
    model_risk_manager: str = Field(
        default="nvidia/llama-3.1-nemotron-70b-instruct",
        validation_alias=AliasChoices("RISK_MODEL", "MODEL_RISK_MANAGER"),
    )
    model_swing_trader: str = Field(
        default="qwen/qwen2.5-32b-instruct",
        validation_alias=AliasChoices("SWING_MODEL", "MODEL_SWING_TRADER"),
    )

    # --- Yahoo options (persistent, ticker-keyed cache) ---------------------
    options_cache_ttl_seconds: int = Field(default=1800, ge=0)
    options_fetch_timeout_seconds: float = Field(default=45.0, gt=0)

    # --- pipeline --------------------------------------------------------------
    lookback_hours: int = 16
    max_articles_per_ticker: int = 15
    max_concurrent_llm: int = 5
    min_personas_for_analysis: int = 1
    analysis_interval_minutes: int = 240
    log_level: str = "INFO"
