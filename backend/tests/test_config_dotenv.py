"""load_dotenv_file — 로컬 실행(macOS Metal 등)에서 .env 자동 로드.

계약: 이미 설정된 os.environ 키는 절대 덮지 않는다 (compose 주입값 우선).
Q&A 전용 키 분리·ALLOWED_HOSTS 와일드카드 경고 등 기동 설정 계약도 여기서 고정한다.
"""

import logging
import os
from pathlib import Path

import app.config as config_module
from app.config import Settings, load_dotenv_file


def test_dotenv_로드_및_기존값_보존(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# 주석\n"
        "\n"
        'OPENAI_BASE_URL="https://example.com/v1"\n'
        "OPENAI_MODEL=test-model\n"
        "TRANSLATE_REASONING=off\n"
        "잘못된줄없음\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATE_REASONING", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://already-set/v1")  # 기존값

    load_dotenv_file(env)

    assert os.environ["OPENAI_BASE_URL"] == "https://already-set/v1"  # 안 덮음
    assert os.environ["OPENAI_MODEL"] == "test-model"                 # 따옴표 벗김·주입
    assert os.environ["TRANSLATE_REASONING"] == "off"


def test_dotenv_파일없음_무해(tmp_path):
    load_dotenv_file(tmp_path / "없는파일.env")  # 예외 없이 no-op


def test_dotenv_자동탐색은_정본보다_워크스페이스_루트까지_확장(tmp_path, monkeypatch):
    """cwd/final에 없으면 프로젝트 루트 .env도 찾는다 (Metal 로컬 실행 계약)."""
    project = tmp_path / "project"
    final = project / "final"
    fake_config = final / "backend" / "app" / "config.py"
    fake_config.parent.mkdir(parents=True)
    (project / ".env").write_text("OPENAI_MODEL=workspace-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "project" / "final" / "backend")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(config_module, "__file__", str(fake_config))

    load_dotenv_file()

    assert os.environ["OPENAI_MODEL"] == "workspace-model"
    assert Path(config_module.__file__) == fake_config


def test_QA키는_번역키를_폴백하지_않는다(monkeypatch):
    """LLM_OPENAI_API_KEY 미설정 시 llm_openai_api_key는 빈 문자열이어야 한다 —
    번역용 OPENAI_API_KEY는 임의 게이트웨이 키일 수 있고, Q&A는 항상
    api.openai.com으로 전송되므로 폴백이 곧 키 유출이다."""
    monkeypatch.setattr(config_module, "load_dotenv_file", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-translation")
    monkeypatch.delenv("LLM_OPENAI_API_KEY", raising=False)

    s = Settings.from_env()

    assert s.openai_api_key == "sk-or-v1-translation"
    assert s.llm_openai_api_key == ""

    monkeypatch.setenv("LLM_OPENAI_API_KEY", "  sk-qa-only  ")
    assert Settings.from_env().llm_openai_api_key == "sk-qa-only"


def _app_settings(tmp_path, hosts: list[str]) -> Settings:
    return Settings(
        engine="fake",
        device="cpu",
        data_dir=tmp_path / "data",
        preload_model=False,
        frontend_dir=tmp_path / "no-frontend",  # 정적 마운트 비활성화
        allowed_hosts=hosts,
    )


def test_와일드카드_ALLOWED_HOSTS는_기동시_경고한다(tmp_path, caplog):
    """무인증 서비스라 Host 검증이 사실상 꺼진 사실을 운영자가 알아야 한다."""
    from app.main import create_app

    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(_app_settings(tmp_path, ["*"]))

    assert any("ALLOWED_HOSTS" in r.getMessage() for r in caplog.records)


def test_명시적_ALLOWED_HOSTS는_경고하지_않는다(tmp_path, caplog):
    from app.main import create_app

    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(_app_settings(tmp_path, ["localhost", "127.0.0.1"]))

    assert not any("ALLOWED_HOSTS" in r.getMessage() for r in caplog.records)
