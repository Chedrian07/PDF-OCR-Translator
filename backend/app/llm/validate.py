"""LLM 엔드포인트 보안 검증 + Settings 기반 라우터 팩토리.

local_url : Ollama 엔드포인트를 온디바이스 호스트 allowlist로 제한 (루프백/도커 내부만).
openai_url: OpenAI 엔드포인트를 공식 https://api.openai.com 호스트로 고정.

둘 다 잘못된 값이면 ValueError — Settings.from_env()가 호출하므로 잘못된
OLLAMA_BASE_URL/LLM_OPENAI_BASE_URL은 기동 시점에 즉시 실패한다.
(번역 서브시스템의 OPENAI_BASE_URL은 별개 키 — 여기서 검증하지 않는다.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .providers import LlmRouter, OllamaClient, OpenAIClient

if TYPE_CHECKING:
    from ..config import Settings


def local_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("OLLAMA_BASE_URL must use http or https")
    local_hosts = {
        "127.0.0.1",
        "localhost",
        "::1",
        # Explicit Docker-local endpoints. Arbitrary hostnames remain blocked.
        "host.docker.internal",
        "ollama",
    }
    if parsed.hostname not in local_hosts:
        raise ValueError("Localight only permits an on-device Ollama endpoint")
    if parsed.username or parsed.password:
        raise ValueError("OLLAMA_BASE_URL must not contain credentials")
    return value.rstrip("/")


def openai_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "api.openai.com":
        raise ValueError("LLM_OPENAI_BASE_URL must use the official https://api.openai.com host")
    if parsed.username or parsed.password:
        raise ValueError("LLM_OPENAI_BASE_URL must not contain credentials")
    return value.rstrip("/")


def build_router(settings: "Settings") -> LlmRouter:
    """Settings의 llm_*/ollama_* 필드로 OpenAI/Ollama 클라이언트와 라우터를 조립한다."""
    openai = OpenAIClient(
        api_key=settings.openai_api_key,
        base_url=settings.llm_openai_base_url,
        responses_models=settings.llm_openai_responses_models,
        chat_models=settings.llm_openai_chat_models,
        default_responses_model=settings.llm_openai_responses_model,
        default_chat_model=settings.llm_openai_chat_model,
    )
    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    return LlmRouter(
        openai=openai,
        ollama=ollama,
        default_provider=settings.llm_provider,
        default_reasoning_effort=settings.llm_reasoning_effort,
    )
