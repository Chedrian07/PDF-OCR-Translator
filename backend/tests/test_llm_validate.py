"""app/llm/validate.py — 엔드포인트 allowlist(Localight tests/test_config.py 이식) + 라우터 팩토리.

계약: OLLAMA_BASE_URL은 온디바이스 호스트만, LLM_OPENAI_BASE_URL은 공식
https://api.openai.com 호스트만 허용 — 위반 시 기동 시점 ValueError.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import build_router
from app.llm.providers import LlmRouter
from app.llm.validate import local_url, openai_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://host.docker.internal:11434",
        "http://ollama:11434",
    ],
)
def test_local_ollama_endpoints_are_allowed(url: str) -> None:
    assert local_url(url) == url


def test_external_ollama_endpoint_is_blocked() -> None:
    with pytest.raises(ValueError, match="on-device"):
        local_url("https://ollama.example.com")


def test_official_openai_endpoint_is_allowed() -> None:
    assert openai_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


def test_non_official_openai_endpoint_is_blocked() -> None:
    with pytest.raises(ValueError, match="official"):
        openai_url("https://openai-proxy.example.com/v1")


def test_build_router_smoke_uses_settings_defaults() -> None:
    """기본 Settings(검증 없이 직접 생성)로 라우터가 조립되고 기본값이 배선되는지 확인."""
    settings = Settings(engine="fake", device="cpu")
    router = build_router(settings)

    assert isinstance(router, LlmRouter)
    assert router.default_provider == settings.llm_provider == "openai-responses"
    assert router.default_reasoning_effort == settings.llm_reasoning_effort
    assert router.openai.default_model("openai-responses") == settings.llm_openai_responses_model
    assert router.openai.default_model("openai-chat") == settings.llm_openai_chat_model
    assert router.openai.responses_models == settings.llm_openai_responses_models
    assert router.ollama.base_url == settings.ollama_base_url
    assert router.ollama.default_model == settings.ollama_model
