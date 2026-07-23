"""load_dotenv_file — 로컬 실행(macOS Metal 등)에서 .env 자동 로드.

계약: 이미 설정된 os.environ 키는 절대 덮지 않는다 (compose 주입값 우선).
"""

import os
from pathlib import Path

import app.config as config_module
from app.config import load_dotenv_file


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
