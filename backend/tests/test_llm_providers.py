"""app/llm/providers.py 요청/응답 계약 고정 (Localight tests/test_llm.py 이식).

httpx.MockTransport 주입으로 실제 네트워크 없이 페이로드 형태를 검증한다:
store:false, developer 롤, 중첩/최상위 reasoning 구분, Ollama think 매핑,
reasoning summary 추출, 원시 chain-of-thought 미노출.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.llm.providers import LlmError, OllamaClient, OpenAIClient


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


def test_허용목록_밖_모델은_업스트림_요청_전에_거절된다() -> None:
    """LLM_OPENAI_*_MODELS가 실효를 갖게 — 임의 model 문자열이 그대로 전달되면 안 된다."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover — 호출되면 실패
        raise AssertionError("허용목록 밖 모델로 업스트림 요청이 나갔다")

    with pytest.raises(LlmError, match="not an allowed"):
        asyncio.run(
            openai_client(handler).generate(
                provider="openai-responses",
                model="gpt-4o-secret",
                system="s",
                prompt="p",
                reasoning_effort="low",
                reasoning_summary="none",
                thinking=True,
            )
        )


def test_전용키_미설정이면_LLM_OPENAI_API_KEY를_안내한다() -> None:
    """안내 문구가 번역 키(OPENAI_API_KEY)를 다시 넣도록 유도하면 안 된다."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("키 없이 업스트림 요청이 나갔다")

    client = OpenAIClient(
        api_key="",
        base_url="https://api.openai.test/v1",
        responses_models=("gpt-test",),
        chat_models=("chat-test",),
        default_responses_model="gpt-test",
        default_chat_model="chat-test",
        transport=httpx.MockTransport(handler),
    )
    assert client.configured is False
    with pytest.raises(LlmError, match="LLM_OPENAI_API_KEY"):
        asyncio.run(
            client.generate(
                provider="openai-chat",
                model=None,
                system="s",
                prompt="p",
                reasoning_effort="low",
                reasoning_summary="none",
                thinking=False,
            )
        )


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


# ── local-openai (루프백 OpenAI 호환 서버) — mlx-integration-11, infra-docs-15 ──

from app.llm.providers import (  # noqa: E402
    QA_PROVIDER_IDS,
    GenerationResult,
    LlmRouter,
    LocalOpenAIClient,
)


def local_client(handler, **kw) -> LocalOpenAIClient:
    return LocalOpenAIClient(
        "http://127.0.0.1:1235/v1",
        default_model=kw.pop("default_model", "qwen-local"),
        models=kw.pop("models", ("qwen-small",)),
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _ask(client: LocalOpenAIClient, **kw):
    args = dict(model=None, system="answer", prompt="page", reasoning_effort="low",
                reasoning_summary="concise", thinking=False)
    args.update(kw)
    return asyncio.run(client.generate(**args))


def test_local_openai_payload는_thinking을_템플릿_인자로_보낸다() -> None:
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        observed["auth"] = request.headers.get("authorization")
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"model": "qwen-local", "choices": [
            {"message": {"content": "답변", "reasoning_content": "raw private trace",
                         "reasoning": "more trace"}}]})

    result = _ask(local_client(handler))
    assert observed["chat_template_kwargs"] == {"enable_thinking": False}
    assert observed["messages"][0]["role"] == "system"       # 로컬 템플릿은 developer를 모름
    assert "store" not in observed and "reasoning_effort" not in observed
    assert observed["max_tokens"] >= 4096                    # mlx_lm 기본 512로 잘리지 않게
    assert observed["auth"] is None                          # 전용 키가 없으면 헤더도 없다
    assert result.content == "답변" and result.reasoning_summary is None
    assert result.provider == "local-openai" and result.remote is False

    _ask(local_client(handler, api_key="local-key"), thinking=True, reasoning_effort="high")
    assert observed["chat_template_kwargs"] == {"enable_thinking": True}
    assert observed["reasoning_effort"] == "high" and observed["auth"] == "Bearer local-key"


def test_local_openai_사고_흔적은_답변에서_걷어낸다() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "Let me think...\n</think>\n\n최종 답변"}}]})

    assert _ask(local_client(handler)).content == "최종 답변"

    def unclosed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "<think>still thinking"}}]})

    with pytest.raises(LlmError, match="empty response"):
        _ask(local_client(unclosed))


@pytest.mark.parametrize("content", [
    # reasoning을 분리하지 않는 서버(llama.cpp --reasoning-format none 등)의 태그 없는 원시 사고
    "Okay, the user asks about page 3. Let me think step by step. First, the figure shows",
    "3쪽의 그림은 모델 구조를 보여 주며, 인코더는 여섯 개의 층으로",   # 중간에 잘린 답
], ids=["raw-thinking", "cut-answer"])
def test_local_openai_max_tokens에서_잘린_답은_돌려주지_않는다(content) -> None:
    """finish_reason=length를 무시해 잘린 답이나 원시 사고를 완전한 답변으로 반환했다 —
    '원시 chain-of-thought 비노출' 계약 위반이고 사용자는 잘린 줄 모른다(translate-11)."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{
            "index": 0, "finish_reason": "length",
            "message": {"role": "assistant", "content": content},
        }], "usage": {"completion_tokens": body["max_tokens"]}})

    with pytest.raises(LlmError, match="max_tokens") as exc:
        _ask(local_client(handler), thinking=True)
    assert content not in str(exc.value)


def test_local_openai_허용목록_밖_모델은_요청_전에_거절() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("허용목록 밖 모델로 요청이 나갔다")

    with pytest.raises(LlmError, match="not an allowed local-openai model"):
        _ask(local_client(handler), model="mlx-community/other-27b")


def test_local_openai_리다이렉트는_따라가지_않는다() -> None:
    """루프백 서버가 외부로 리다이렉트해도 원문이 따라 나가지 않는다(SSRF)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://evil.example/v1/chat"})

    with pytest.raises(LlmError, match="rejected"):
        _ask(local_client(handler))
    assert calls == ["http://127.0.0.1:1235/v1/chat/completions"]


def _router(local: LocalOpenAIClient | None) -> LlmRouter:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    return LlmRouter(
        openai=openai_client(dead),
        ollama=OllamaClient("http://127.0.0.1:11434", "qwen3:8b",
                            transport=httpx.MockTransport(dead)),
        default_provider="openai-responses",
        default_reasoning_effort="low",
        local_openai=local,
    )


def test_local_openai는_구성됐을_때만_공급자_목록에_광고한다() -> None:
    def models_ok(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen-local"}]})

    without = asyncio.run(_router(None).providers())
    assert "local-openai" not in [p["id"] for p in without["providers"]]

    catalog = asyncio.run(_router(local_client(models_ok)).providers())
    entry = next(p for p in catalog["providers"] if p["id"] == "local-openai")
    assert entry["available"] is True and entry["remote"] is False
    assert entry["models"] == ["qwen-small", "qwen-local"]
    assert entry["default_model"] == "qwen-local"

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    offline = asyncio.run(_router(local_client(down)).providers())
    assert next(p for p in offline["providers"] if p["id"] == "local-openai")["available"] is False


def test_router가_local_openai로_라우팅하고_구성여부를_알려준다() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "로컬 답"}}]})

    router = _router(local_client(handler))
    result = asyncio.run(router.ask(
        question="q", context="c", provider="local-openai", model=None,
        reasoning_effort="default", reasoning_summary="none", thinking=False,
    ))
    assert isinstance(result, GenerationResult) and result.content == "로컬 답"
    assert router.configured("local-openai") and not _router(None).configured("local-openai")
    assert router.default_model("local-openai") == "qwen-local"
    assert "local-openai" in QA_PROVIDER_IDS
    with pytest.raises(LlmError, match="not configured"):
        asyncio.run(_router(None).ask(
            question="q", context="c", provider="local-openai", model=None,
            reasoning_effort="default", reasoning_summary="none", thinking=False,
        ))
