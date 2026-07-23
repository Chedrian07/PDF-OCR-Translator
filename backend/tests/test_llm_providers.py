"""app/llm/providers.py 요청/응답 계약 고정 (Localight tests/test_llm.py 이식).

httpx.MockTransport 주입으로 실제 네트워크 없이 페이로드 형태를 검증한다:
store:false, developer 롤, 중첩/최상위 reasoning 구분, Ollama think 매핑,
reasoning summary 추출, 원시 chain-of-thought 미노출.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.llm.providers import OllamaClient, OpenAIClient


def openai_client(handler) -> OpenAIClient:
    return OpenAIClient(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        responses_models=("gpt-test",),
        chat_models=("chat-test",),
        default_responses_model="gpt-test",
        default_chat_model="chat-test",
        transport=httpx.MockTransport(handler),
    )


def test_responses_api_payload_and_reasoning_summary() -> None:
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        return httpx.Response(
            200,
            json={
                "model": "gpt-test-2026",
                "output": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Checked terminology."}]},
                    {"type": "message", "content": [{"type": "output_text", "text": "번역 결과"}]},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )

    result = asyncio.run(
        openai_client(handler).generate(
            provider="openai-responses",
            model=None,
            system="translate",
            prompt="paper text",
            reasoning_effort="high",
            reasoning_summary="concise",
            thinking=True,
        )
    )

    assert observed["store"] is False
    assert observed["reasoning"] == {"effort": "high", "summary": "concise"}
    assert result.content == "번역 결과"
    assert result.reasoning_summary == "Checked terminology."
    assert result.remote is True


def test_chat_completions_payload_uses_reasoning_effort() -> None:
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "model": "chat-test",
                "choices": [{"message": {"content": "채팅 번역"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            },
        )

    result = asyncio.run(
        openai_client(handler).generate(
            provider="openai-chat",
            model="chat-test",
            system="translate",
            prompt="paper text",
            reasoning_effort="medium",
            reasoning_summary="detailed",
            thinking=True,
        )
    )

    assert observed["reasoning_effort"] == "medium"
    assert observed["messages"][0]["role"] == "developer"
    assert "reasoning" not in observed
    assert result.content == "채팅 번역"
    assert result.reasoning_summary is None


def test_ollama_thinking_is_requested_but_raw_trace_is_not_exposed() -> None:
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 1}]})
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": "로컬 번역", "thinking": "raw private trace"},
                "prompt_eval_count": 8,
                "eval_count": 3,
            },
        )

    client = OllamaClient(
        "http://localhost:11434",
        "qwen3:8b",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        client.generate(
            model=None,
            system="translate",
            prompt="paper text",
            reasoning_effort="high",
            reasoning_summary="none",
            thinking=True,
        )
    )

    assert observed["think"] == "high"
    assert result.content == "로컬 번역"
    assert result.reasoning_summary is None
