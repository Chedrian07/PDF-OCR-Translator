"""LLM 엔드포인트 보안 검증 + Settings 기반 라우터 팩토리.

local_url : Ollama 엔드포인트를 온디바이스 호스트 allowlist로 제한 (루프백/도커 내부만).
openai_url: OpenAI 엔드포인트를 공식 https://api.openai.com 호스트로 고정.
local_openai_url: 로컬 OpenAI 호환 서버(local-openai)를 루프백·host.docker.internal로 제한.

둘 다 잘못된 값이면 ValueError — Settings.from_env()가 호출하므로 잘못된
OLLAMA_BASE_URL/LLM_OPENAI_BASE_URL은 기동 시점에 즉시 실패한다.
(번역 서브시스템의 OPENAI_BASE_URL은 별개 키 — 여기서 검증하지 않는다.
 인증 키도 LLM_OPENAI_API_KEY로 분리한다 — 번역용 OPENAI_API_KEY는 임의 게이트웨이를
 가리킬 수 있어, 여기에 재사용하면 그 키가 api.openai.com으로 전송된다.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .providers import LlmRouter, LocalOpenAIClient, OllamaClient, OpenAIClient

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


# local-openai 허용 호스트 — 정확히 이 문자열들만. hostname 비교라 십진·8진·16진 IP
# 표기(2130706433, 0177.0.0.1, 0x7f000001), 축약형(127.1), IPv4-mapped IPv6, 끝점
# 붙은 이름(localhost.), 임의 DNS 이름은 모두 거절된다(SSRF 방어 — 문서 원문이 나간다).
_LOCAL_OPENAI_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})


def local_openai_url(value: str) -> str:
    """LLM_LOCAL_OPENAI_BASE_URL 검증 — 루프백·host.docker.internal의 http(s)만.

    Q&A 질문과 페이지 원문이 그대로 전송되므로 Ollama와 같은 온디바이스 원칙을 지킨다.
    자격증명(userinfo)·query·fragment도 거부한다(키는 LLM_LOCAL_OPENAI_API_KEY로만).
    """
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("LLM_LOCAL_OPENAI_BASE_URL must use http or https")
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise ValueError("LLM_LOCAL_OPENAI_BASE_URL must not contain credentials")
    if parsed.hostname not in _LOCAL_OPENAI_HOSTS:
        raise ValueError(
            "LLM_LOCAL_OPENAI_BASE_URL only permits a loopback or host.docker.internal server"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("LLM_LOCAL_OPENAI_BASE_URL must not contain a query or fragment")
    try:
        parsed.port  # noqa: B018 — 포트 범위·형식 검증(잘못되면 ValueError)
    except ValueError as exc:
        raise ValueError("LLM_LOCAL_OPENAI_BASE_URL has an invalid port") from exc
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
        api_key=settings.llm_openai_api_key,
        base_url=settings.llm_openai_base_url,
        responses_models=settings.llm_openai_responses_models,
        chat_models=settings.llm_openai_chat_models,
        default_responses_model=settings.llm_openai_responses_model,
        default_chat_model=settings.llm_openai_chat_model,
    )
    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    local_openai = None
    if settings.llm_local_openai_base_url:
        # 전용 키만 쓴다 — 번역용 OPENAI_API_KEY·Q&A용 LLM_OPENAI_API_KEY 폴백 금지
        local_openai = LocalOpenAIClient(
            settings.llm_local_openai_base_url,
            default_model=settings.llm_local_openai_model,
            models=settings.llm_local_openai_models,
            api_key=settings.llm_local_openai_api_key,
        )
    return LlmRouter(
        openai=openai,
        ollama=ollama,
        default_provider=settings.llm_provider,
        default_reasoning_effort=settings.llm_reasoning_effort,
        local_openai=local_openai,
    )
