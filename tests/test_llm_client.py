"""LLMClient tests — all provider I/O is faked; no network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

import analyzer.llm_client as llm_module
from analyzer.config import AnalyzerSettings
from analyzer.llm_client import LLMClient, PersonaError

VERDICT = {
    "sentiment": "Positive",
    "confidence_score": 70,
    "swing_trading_candidate": True,
    "news_pointers": ["beat expectations"],
    "option_commentary": None,
}


def fast_settings(**overrides) -> AnalyzerSettings:
    """Settings with instant retries for tests."""
    defaults = dict(
        llm_api_key="nvapi-test-key",
        llm_retry_attempts=3,
        llm_retry_min_wait=0.0,
        llm_retry_max_wait=0.01,
    )
    defaults.update(overrides)
    return AnalyzerSettings(**defaults)


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(kwargs)
        content = step
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def attach(client: LLMClient, script) -> FakeCompletions:
    fake = FakeCompletions(script)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    client._get_client = lambda: fake_client  # bypass real OpenAI construction
    return fake


def bad_request(message: str) -> BadRequestError:  # noqa: F821
    from openai import BadRequestError

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return BadRequestError(message, response=response, body=None)


# ---------------------------------------------------------------------------


async def test_valid_json_returns_verdict():
    client = LLMClient(fast_settings())
    fake = attach(client, [json.dumps(VERDICT)])

    verdict = await client.ask_persona("news_fundamentalist", "AAPL", "- news line")
    assert verdict.sentiment == "Positive"
    assert verdict.confidence_score == 70

    call = fake.calls[0]
    assert call["model"] == "meta/llama-3.3-70b-instruct"  # NVIDIA model id
    assert call["response_format"] == {"type": "json_object"}
    assert "AAPL" in call["messages"][1]["content"]
    assert "senior equity research analyst" in call["messages"][0]["content"]


async def test_fenced_json_parsed_without_repair():
    client = LLMClient(fast_settings())
    fake = attach(client, ["```json\n" + json.dumps(VERDICT) + "\n```"])
    verdict = await client.ask_persona("risk_manager", "TSLA", "- news")
    assert verdict.swing_trading_candidate is True
    assert len(fake.calls) == 1


async def test_each_persona_gets_its_own_model():
    client = LLMClient(fast_settings())
    fake = attach(client, [json.dumps(VERDICT)] * 3)
    for key in ("news_fundamentalist", "risk_manager", "swing_trader"):
        await client.ask_persona(key, "MSFT", "- news")
    models = [c["model"] for c in fake.calls]
    assert models == [
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "qwen/qwen2.5-32b-instruct",
    ]


async def test_broken_json_triggers_one_repair_roundtrip():
    client = LLMClient(fast_settings())
    broken = 'Sure! Here is my take:\n{"sentiment": "Positive", "confidence_score": 65, oops'
    fake = attach(client, [broken, json.dumps(VERDICT)])

    verdict = await client.ask_persona("swing_trader", "NVDA", "- news")
    assert verdict.sentiment == "Positive"
    assert len(fake.calls) == 2
    repair_prompt = fake.calls[1]["messages"][-1]["content"]
    assert "ONLY the JSON" in repair_prompt


async def test_unrepairable_output_raises_persona_error():
    client = LLMClient(fast_settings())
    attach(client, ["garbage", "still garbage"])
    with pytest.raises(PersonaError, match="unparseable"):
        await client.ask_persona("risk_manager", "AMD", "- news")


async def test_transient_errors_are_retried():
    class FakeTimeout(Exception):
        pass

    monkeypatched = (FakeTimeout,)
    original = llm_module.RETRYABLE_EXCEPTIONS
    llm_module.RETRYABLE_EXCEPTIONS = monkeypatched
    try:
        client = LLMClient(fast_settings(llm_retry_attempts=3))
        fake = attach(client, [FakeTimeout("boom"), json.dumps(VERDICT)])
        verdict = await client.ask_persona("news_fundamentalist", "GOOG", "- news")
        assert verdict.confidence_score == 70
        assert len(fake.calls) == 2
    finally:
        llm_module.RETRYABLE_EXCEPTIONS = original


async def test_response_format_rejection_degrades_to_plain_mode():
    client = LLMClient(fast_settings())
    fake = attach(
        client,
        [
            bad_request("response_format is not supported by this model"),
            json.dumps(VERDICT),
        ],
    )
    verdict = await client.ask_persona("news_fundamentalist", "META", "- news")
    assert verdict.sentiment == "Positive"
    assert "response_format" not in fake.calls[-1]          # plain mode used
    assert "response_format" in fake.calls[0]               # attempted first


async def test_hard_bad_request_raises_immediately():
    client = LLMClient(fast_settings())
    attach(client, [bad_request("context length exceeded")])
    with pytest.raises(PersonaError, match="rejected request"):
        await client.ask_persona("swing_trader", "NFLX", "- news")


async def test_missing_api_key_fails_fast_per_persona():
    client = LLMClient(fast_settings(llm_api_key=""))
    with pytest.raises(PersonaError, match="LLM not configured"):
        await client.ask_persona("risk_manager", "XOM", "- news")


async def test_empty_content_retried_then_persona_error():
    client = LLMClient(fast_settings(llm_retry_attempts=2))
    attach(client, ["", "   "])
    with pytest.raises(PersonaError, match="empty content"):
        await client.ask_persona("news_fundamentalist", "BA", "- news")
