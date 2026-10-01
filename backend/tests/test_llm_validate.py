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


def test_build_router는_번역키를_Q_A에_재사용하지_않는다() -> None:
    """OPENAI_API_KEY는 임의 게이트웨이(OpenRouter 등)를 가리킬 수 있고
    llm_openai_base_url은 api.openai.com 고정이라, 폴백은 곧 키 유출이다."""
    settings = Settings(
        engine="fake", device="cpu", openai_api_key="sk-or-v1-translation", llm_openai_api_key=""
    )
    router = build_router(settings)

    assert router.openai.api_key == ""
    assert router.openai.configured is False


def test_build_router는_전용_Q_A키를_사용한다() -> None:
    settings = Settings(
        engine="fake", device="cpu", openai_api_key="sk-or-v1-translation",
        llm_openai_api_key="sk-qa-only",
    )
    router = build_router(settings)

    assert router.openai.api_key == "sk-qa-only"
    assert router.openai.configured is True


# ── local-openai (루프백 OpenAI 호환 서버) — mlx-integration-11, infra-docs-15 ──

from app.llm.validate import local_openai_url  # noqa: E402


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:1235/v1",
    "http://localhost:1234/v1",
    "http://LOCALHOST:1234/v1",
    "http://[::1]:8080/v1",
    "http://host.docker.internal:1235/v1",
    "https://127.0.0.1/v1",
])
def test_local_openai_루프백_주소는_허용(url: str) -> None:
    assert local_openai_url(url + "/") == url


@pytest.mark.parametrize("url", [
    "http://user:pw@127.0.0.1:1235/v1",        # userinfo
    "http://127.0.0.1@evil.example/v1",        # userinfo로 위장한 외부 호스트
    "http://@127.0.0.1/v1",
    "http://[::ffff:127.0.0.1]/v1",            # IPv4-mapped IPv6
    "http://[fe80::1]/v1",                     # 링크로컬 IPv6
    "http://[::2]/v1",
    "http://2130706433/v1",                    # 십진 IP
    "http://0177.0.0.1/v1",                    # 8진 IP
    "http://0x7f000001/v1",                    # 16진 IP
    "http://127.1/v1",                         # 축약형
    "http://127.0.0.2/v1",                     # 루프백 대역이지만 허용목록 밖
    "http://localhost./v1",                    # 끝점 붙은 이름
    "http://localhost.evil.example/v1",        # DNS 이름
    "http://omlx.local/v1",
    "http://ollama:11434/v1",                  # Ollama 전용 compose 서비스명은 여기선 불가
    "http://10.0.0.5:1234/v1",                 # 사설망
    "ftp://127.0.0.1/v1",
    "http://127.0.0.1:1235/v1?tenant=x",
    "http://127.0.0.1:1235/v1#frag",
    "http://127.0.0.1:99999/v1",
])
def test_local_openai_SSRF_우회_주소는_거부(url: str) -> None:
    with pytest.raises(ValueError, match="LLM_LOCAL_OPENAI_BASE_URL"):
        local_openai_url(url)


def test_local_openai_설정은_전용키만_쓰고_구성됐을_때만_라우터에_붙는다(monkeypatch) -> None:
    from app import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv_file", lambda *a, **k: None)
    for key in ("LLM_LOCAL_OPENAI_BASE_URL", "LLM_LOCAL_OPENAI_MODEL",
                "LLM_LOCAL_OPENAI_MODELS", "LLM_LOCAL_OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-translation")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "sk-qa-openai")

    plain = Settings.from_env()
    assert plain.llm_local_openai_base_url == "" and build_router(plain).local_openai is None

    monkeypatch.setenv("LLM_LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:1235/v1/")
    monkeypatch.setenv("LLM_LOCAL_OPENAI_MODEL", "qwen3.6-27b")
    monkeypatch.setenv("LLM_LOCAL_OPENAI_MODELS", "qwen3.6-27b, gemma-4")
    monkeypatch.setenv("LLM_PROVIDER", "local-openai")
    s = Settings.from_env()
    assert s.llm_local_openai_base_url == "http://127.0.0.1:1235/v1"
    assert s.llm_local_openai_models == ("qwen3.6-27b", "gemma-4")
    assert s.llm_local_openai_api_key == ""            # 번역·OpenAI 키로 폴백하지 않는다
    router = build_router(s)
    assert router.local_openai.api_key == "" and router.default_provider == "local-openai"
    assert router.default_model("local-openai") == "qwen3.6-27b"

    monkeypatch.setenv("LLM_LOCAL_OPENAI_API_KEY", "  local-key  ")
    assert build_router(Settings.from_env()).local_openai.api_key == "local-key"


@pytest.mark.parametrize(("env", "message"), [
    ({"LLM_LOCAL_OPENAI_BASE_URL": "http://192.168.0.9:1234/v1",
      "LLM_LOCAL_OPENAI_MODEL": "m"}, "loopback"),
    ({"LLM_LOCAL_OPENAI_BASE_URL": "http://127.0.0.1:1234/v1"}, "LLM_LOCAL_OPENAI_MODEL"),
    ({"LLM_PROVIDER": "local-openai"}, "LLM_LOCAL_OPENAI_BASE_URL"),
])
def test_local_openai_잘못된_설정은_기동시_거부(monkeypatch, env, message) -> None:
    from app import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv_file", lambda *a, **k: None)
    for key in ("LLM_LOCAL_OPENAI_BASE_URL", "LLM_LOCAL_OPENAI_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()
