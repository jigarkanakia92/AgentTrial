"""LLM client — OpenAI SDK pointed at an OpenAI-compatible endpoint.

Default target: **NVIDIA NIM** (``https://integrate.api.nvidia.com/v1``).
The exact same code talks to OpenAI, vLLM, Ollama, or any other
OpenAI-compatible provider by changing LLM_BASE_URL/LLM_API_KEY.

Resilience features:
* Lazy client construction; missing key fails fast per-persona (the rest of
  the pipeline keeps running) instead of crashing the service.
* Hand-rolled retry loop with exponential backoff on transient provider
  errors (429, 5xx, timeouts, connection resets).
* ``response_format=json_object`` is attempted first and **automatically
  dropped** if the endpoint/model rejects it (not all NIM models support it).
* One self-repair round-trip: if the model's JSON is broken, we send it back
  with "fix this JSON" before giving up on the persona.
* Every failure is a raised PersonaError — the pipeline isolates personas,
  so one bad model never loses the whole analysis.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from analyzer.config import AnalyzerSettings
from analyzer.personas import build_user_prompt, persona_model, persona_system_prompt
from analyzer.schemas import PersonaResponseError, PersonaVerdict, parse_persona_verdict

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class PersonaError(RuntimeError):
    """A persona call failed after all retries — pipeline drops this persona."""


class LLMClient:
    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self.settings = settings or AnalyzerSettings()
        self._client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self.settings.llm_api_key:
                raise PersonaError(
                    "LLM not configured: set LLM_API_KEY (or NVIDIA_API_KEY). "
                    "Skipping persona analysis this cycle."
                )
            self._client = AsyncOpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,  # we own retries (tenacity-style, with backoff)
            )
        return self._client

    def close(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    async def ask_persona(
        self,
        persona_key: str,
        ticker: str,
        news_bundle: str,
        option_data: str | None = None,
    ) -> PersonaVerdict:
        """Ask one persona; returns a validated verdict or raises PersonaError."""
        model = persona_model(persona_key, self.settings)
        messages = [
            {"role": "system", "content": persona_system_prompt(persona_key)},
            {
                "role": "user",
                "content": build_user_prompt(ticker, news_bundle, option_data),
            },
        ]

        content, used_json_mode = await self._chat(model, messages)
        try:
            return parse_persona_verdict(content)
        except PersonaResponseError as first_error:
            # --- one self-repair round-trip -------------------------------
            logger.debug(
                "Persona {} returned malformed JSON for {}; attempting repair",
                persona_key,
                ticker,
            )
            repair_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON for the required "
                        f"schema ({first_error}). Reply again with ONLY the JSON "
                        "object, no prose, no markdown fences."
                    ),
                },
            ]
            content2, _ = await self._chat(model, repair_messages, force_plain=not used_json_mode)
            try:
                return parse_persona_verdict(content2)
            except PersonaResponseError:
                raise PersonaError(
                    f"persona {persona_key} produced unparseable output twice for {ticker}"
                ) from first_error

    # ------------------------------------------------------------------
    async def _chat(
        self, model: str, messages: list[dict], force_plain: bool = False
    ) -> tuple[str, bool]:
        """Send one chat request with retries. Returns (content, json_mode_used).

        ``json_mode_used`` tells the caller whether response_format survived —
        some OpenAI-compatible endpoints reject it outright.
        """
        client = self._get_client()
        use_json_mode = self.settings.use_json_mode and not force_plain
        last_error: Exception | None = None

        for attempt in range(1, self.settings.llm_retry_attempts + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = await client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise PersonaError(f"model {model} returned empty content")
                return content, use_json_mode
            except BadRequestError as exc:
                if use_json_mode and "response_format" in str(exc).lower():
                    logger.warning(
                        "Model {} rejected response_format=json_object — retrying "
                        "without JSON mode", model,
                    )
                    use_json_mode = False
                    continue  # immediate retry, does not consume an attempt fairly
                raise PersonaError(f"model {model} rejected request: {exc}") from exc
            except RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning(
                    "Transient LLM error on {} (attempt {}/{}): {}",
                    model, attempt, self.settings.llm_retry_attempts, exc,
                )
            except APIError as exc:  # non-retryable provider error
                raise PersonaError(f"model {model} failed: {exc}") from exc

            if attempt < self.settings.llm_retry_attempts:
                await asyncio.sleep(self._backoff(attempt))

        raise PersonaError(
            f"model {model} unreachable after {self.settings.llm_retry_attempts} "
            f"attempts: {last_error}"
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter: min_wait * 2^(n-1), capped."""
        base = self.settings.llm_retry_min_wait * (2 ** (attempt - 1))
        return min(base, self.settings.llm_retry_max_wait) * random.uniform(0.5, 1.5)
