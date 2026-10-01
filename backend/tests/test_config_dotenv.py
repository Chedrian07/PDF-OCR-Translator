"""load_dotenv_file — 로컬 실행(macOS Metal 등)에서 .env 자동 로드.

계약: 이미 설정된 os.environ 키는 절대 덮지 않는다 (compose 주입값 우선).
파싱 규칙은 docker compose와 같다(python-dotenv) — 인라인 주석·따옴표·export·CRLF.
Q&A 전용 키 분리·ALLOWED_HOSTS 와일드카드 경고 등 기동 설정 계약도 여기서 고정한다.
"""

import logging
import os
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

import app.config as config_module
from app.config import Settings, load_dotenv_file
from app.translate.types import TranslateConfig

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_environ():
    """load_dotenv_file은 os.environ을 직접 바꾼다 — monkeypatch가 모르는 키(미설정
    상태에서 delenv한 키 포함)가 다음 테스트로 새지 않게 테스트마다 되돌린다."""
    saved = dict(os.environ)
    yield
    for key in set(os.environ) - set(saved):
        del os.environ[key]
    for key, value in saved.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


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


def test_dotenv_자동탐색은_cwd와_저장소_루트만_본다(tmp_path, monkeypatch):
    """cwd에 없으면 저장소 루트 .env를 찾는다(Metal 로컬 실행 계약). 저장소 밖 상위
    디렉터리의 .env는 무관한 프로젝트의 키·엔드포인트일 수 있어 읽지 않는다 —
    예전 final/ 워크스페이스 배치의 잔재(parents[3])가 그 경로였다."""
    outside = tmp_path / "projects"
    repo = outside / "PDF-OCR-Translator"
    fake_config = repo / "backend" / "app" / "config.py"
    fake_config.parent.mkdir(parents=True)
    (outside / ".env").write_text(
        "OPENAI_BASE_URL=https://other-project/v1\nOPENAI_MODEL=outside-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo / "backend")
    for key in ("OPENAI_BASE_URL", "OPENAI_MODEL", "TRANSLATE_MODEL", "DISABLE_DOTENV"):
        monkeypatch.delenv(key, raising=False)  # DISABLE_DOTENV: conftest 격리 스위치 해제
    monkeypatch.setattr(config_module, "__file__", str(fake_config))

    load_dotenv_file()
    assert "OPENAI_BASE_URL" not in os.environ  # 저장소 밖 .env는 무시한다
    assert "OPENAI_MODEL" not in os.environ

    (repo / ".env").write_text(
        "OPENAI_MODEL=repo-model\nTRANSLATE_MODEL=repo-only\n", encoding="utf-8",
    )
    load_dotenv_file()
    assert os.environ["OPENAI_MODEL"] == "repo-model"

    # cwd가 먼저이고, 처음 찾은 파일 하나만 읽는다(서로 다른 비밀 파일을 합치지 않음)
    (repo / "backend" / ".env").write_text("OPENAI_MODEL=cwd-model\n", encoding="utf-8")
    del os.environ["OPENAI_MODEL"], os.environ["TRANSLATE_MODEL"]
    load_dotenv_file()
    assert os.environ["OPENAI_MODEL"] == "cwd-model"
    assert "TRANSLATE_MODEL" not in os.environ
    assert Path(config_module.__file__) == fake_config


def test_DISABLE_DOTENV는_자동탐색만_끄고_명시_경로는_읽는다(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=cwd-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISABLE_DOTENV", "1")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    load_dotenv_file()
    assert "OPENAI_MODEL" not in os.environ

    load_dotenv_file(env)
    assert os.environ["OPENAI_MODEL"] == "cwd-model"


def test_dotenv는_compose와_같은_규칙으로_주석_따옴표_export_CRLF를_읽는다(tmp_path, monkeypatch):
    """.env.example 줄의 '#'만 지우면 설명 주석까지 값이 돼 OCR_FAST_DECODE=1이
    False로 뒤집히고 OCR_DEVICE=metal이 기동에 실패했다(compose에서는 정상)."""
    lines = [
        "\ufeffOCR_FAST_DECODE=1         # 커스텀 그리디 디코드 루프",  # BOM + 인라인 주석
        "OCR_DEVICE=metal            # cpu | cuda | metal",
        "ALLOWED_HOSTS=localhost,127.0.0.1   # Host 헤더 화이트리스트(콤마 구분, DNS 방어)",
        "LLM_OPENAI_API_KEY='sk-#not-a-comment'   # 따옴표 안의 #은 값이다",
        'OLLAMA_MODEL="qwen3 # tagged"',
        "export MODEL_REVISION=abc123",
        "OPENAI_BASE_URL=http://127.0.0.1:1234/v1?a=b=c",
        "OCR_LANGUAGES=eng+kor#붙은샵은값",
        "PRELOAD_MODEL",  # '='가 없는 줄은 건너뛴다
    ]
    env = tmp_path / ".env"
    env.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))  # CRLF
    for key in (
        "OCR_FAST_DECODE", "OCR_DEVICE", "ALLOWED_HOSTS", "LLM_OPENAI_API_KEY",
        "OLLAMA_MODEL", "MODEL_REVISION", "OPENAI_BASE_URL", "OCR_LANGUAGES", "PRELOAD_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    load_dotenv_file(env)

    assert os.environ["OCR_FAST_DECODE"] == "1"
    assert os.environ["OCR_DEVICE"] == "metal"
    assert os.environ["ALLOWED_HOSTS"] == "localhost,127.0.0.1"
    assert os.environ["LLM_OPENAI_API_KEY"] == "sk-#not-a-comment"
    assert os.environ["OLLAMA_MODEL"] == "qwen3 # tagged"
    assert os.environ["MODEL_REVISION"] == "abc123"
    assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1?a=b=c"
    assert os.environ["OCR_LANGUAGES"] == "eng+kor#붙은샵은값"
    assert "PRELOAD_MODEL" not in os.environ

    s = Settings.from_env()  # 주석이 값에 섞였다면 여기서 ValueError나 설정 반전
    assert s.fast_decode is True
    assert s.device == "metal"
    assert s.allowed_hosts == ["localhost", "127.0.0.1"]
    assert s.llm_openai_api_key == "sk-#not-a-comment"
    assert s.model_revision == "abc123"


def test_dotenv_로드_경로는_INFO로_남기고_값은_남기지_않는다(tmp_path, monkeypatch, caplog):
    env = tmp_path / ".env"
    env.write_text("PROBE_SECRET_KEY=sk-should-never-be-logged\n", encoding="utf-8")
    monkeypatch.delenv("PROBE_SECRET_KEY", raising=False)

    with caplog.at_level(logging.INFO, logger="app.config"):
        load_dotenv_file(env)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert str(env) in text
    assert "sk-should-never-be-logged" not in text


# ── PAGE_SEPARATOR 이스케이프 해석 (감사 api-jobs-12) ──

def test_PAGE_SEPARATOR는_한글과_이스케이프를_함께_보존한다(monkeypatch):
    """unicode_escape가 UTF-8 바이트를 Latin-1로 읽어 한글·전각 대시 구분자가
    result.md의 모든 페이지 경계에서 'â\\x80\\x94 í\\x8e\\x98…'로 깨졌다."""
    monkeypatch.setenv("PAGE_SEPARATOR", r"\n\n— 페이지 —\n\n")

    assert Settings.from_env().page_separator == "\n\n— 페이지 —\n\n"


def test_PAGE_SEPARATOR_기본값과_이미_풀린_개행은_그대로다(monkeypatch):
    monkeypatch.delenv("PAGE_SEPARATOR", raising=False)
    assert Settings.from_env().page_separator == "\n\n---\n\n"

    monkeypatch.setenv("PAGE_SEPARATOR", "\n\n===\n\n")  # compose·셸이 넘긴 실제 개행
    assert Settings.from_env().page_separator == "\n\n===\n\n"

    monkeypatch.setenv("PAGE_SEPARATOR", r"\t|é|\\")  # 기존 ASCII 이스케이프·Latin-1 문자
    assert Settings.from_env().page_separator == "\t|é|\\"


def test_PAGE_SEPARATOR_큰따옴표_dotenv_값도_한글이_보존된다(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('PAGE_SEPARATOR="\\n\\n— 페이지 —\\n\\n"\n', encoding="utf-8")
    monkeypatch.delenv("PAGE_SEPARATOR", raising=False)

    load_dotenv_file(env)  # 큰따옴표 안 \n은 dotenv가 먼저 개행으로 푼다(compose와 같음)

    assert Settings.from_env().page_separator == "\n\n— 페이지 —\n\n"


def test_PAGE_SEPARATOR_깨진_이스케이프는_변수명과_함께_실패한다(monkeypatch):
    monkeypatch.setenv("PAGE_SEPARATOR", "끝이 백슬래시\\")

    with pytest.raises(ValueError, match="PAGE_SEPARATOR"):
        Settings.from_env()


# ── 숫자 env 검증 (감사 api-jobs-11, sidecar-12) ──

@pytest.mark.parametrize(("name", "raw", "rule"), [
    ("RENDER_DPI", "abc", "정수여야"),
    ("JOB_TTL_DAYS", "7d", "정수여야"),
    ("MAX_PAGE_OUTPUT_CHARS", "lots", "정수여야"),
    ("TRANSLATE_GLOBAL_CONCURRENCY", "x", "정수여야"),
    ("OCR_FIDELITY_THRESHOLD", "high", "숫자여야"),
])
def test_숫자_env_파싱_오류는_변수명과_값을_담는다(monkeypatch, name, raw, rule):
    """예전에는 `invalid literal for int()`만 남아 어느 키인지 traceback을 읽어야 했다."""
    monkeypatch.setenv(name, raw)

    with pytest.raises(ValueError) as excinfo:
        Settings.from_env()

    message = str(excinfo.value)
    assert name in message and repr(raw) in message and rule in message


@pytest.mark.parametrize(("name", "raw"), [
    ("RENDER_DPI", "600"),           # 업로드 dpi 검증(72–400) 밖 → dpi 없는 업로드가 전부 400
    ("RENDER_DPI", "71"),
    ("PAGES_PER_CHUNK", "0"),
    ("MAX_PAGES", "0"),              # 모든 업로드 거부
    ("MAX_UPLOAD_MB", "0"),          # 모든 업로드 413
    ("MAX_LENGTH", "0"),
    ("OCR_DECODE_BLOCK", "0"),
    ("JOB_TTL_DAYS", "-1"),
    ("FAKE_DELAY", "-0.5"),          # time.sleep 음수 → 모든 페이지 실패
    ("OCR_SIDECAR_CONNECT_TIMEOUT_S", "0"),  # requests ValueError → 모든 페이지 실패
    ("OCR_SIDECAR_READ_TIMEOUT_S", "-1"),
    ("OCR_SIDECAR_READ_TIMEOUT_S", "nan"),
    ("OCR_SIDECAR_HEALTH_TIMEOUT_S", "inf"),
    ("OCR_FIDELITY_MAX_RETRY_RATIO", "inf"),  # int(total*inf) OverflowError
])
def test_범위_밖_숫자_env는_기동_시_변수명과_함께_실패한다(monkeypatch, name, raw):
    monkeypatch.setenv(name, raw)

    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_숫자_env_경계값과_빈_값은_통과한다(monkeypatch):
    for name, raw in (
        ("RENDER_DPI", "72"), ("PAGES_PER_CHUNK", "1"), ("MAX_UPLOAD_MB", "1"),
        ("JOB_TTL_DAYS", "0"), ("OCR_SIDECAR_READ_TIMEOUT_S", "0.5"), ("FAKE_DELAY", "0"),
    ):
        monkeypatch.setenv(name, raw)
    monkeypatch.setenv("MAX_PAGES", "")  # compose가 넘기는 빈 값 = 미설정

    s = Settings.from_env()

    assert (s.render_dpi, s.pages_per_chunk, s.max_upload_mb) == (72, 1, 1)
    assert (s.job_ttl_days, s.sidecar_read_timeout_s, s.fake_delay) == (0, 0.5, 0.0)
    assert s.max_pages == 200
    monkeypatch.setenv("RENDER_DPI", "400")
    assert Settings.from_env().render_dpi == 400


def test_sidecar_모델_대기_상한은_음수를_0으로_보정한다(monkeypatch):
    monkeypatch.setenv("OCR_SIDECAR_MODEL_WAIT_S", "-5")

    assert Settings.from_env().sidecar_model_wait_s == 0.0


def test_남용_방어_상한의_오타는_기본값으로_강등된다(monkeypatch, caplog):
    """기존 의도 유지 — 운영 중 남용 방어 설정 실수가 기동 실패나 500이 되면 안 된다."""
    monkeypatch.setenv("QA_RATE_LIMIT_PER_MIN", "abc")

    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = Settings.from_env()

    assert s.qa_rate_limit_per_min == 30
    assert any("QA_RATE_LIMIT_PER_MIN" in r.getMessage() for r in caplog.records)


# ── .env.example 주석 해제 계약 (감사 infra-docs-1, mlx-integration-7, api-jobs-2) ──

def test_env_example은_어느_줄을_주석_해제해도_유효한_값이다(tmp_path, monkeypatch):
    """README는 '.env.example에서 #을 지우고 값을 넣으라'고 안내한다. 설명이 같은 줄에
    있으면 설명까지 값이 돼 OCR_FAST_DECODE=1이 False로 뒤집히고 OCR_DEVICE=metal이
    기동에 실패했다 — 설명은 별도 줄에 두고, 전부 주석 해제해도 기동돼야 한다."""
    lines = []
    for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^# ?([A-Z][A-Z0-9_]*=.*)$", line)
        lines.append(m.group(1) if m else line)
    env = tmp_path / ".env"
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")

    values = dotenv_values(env)
    assert len(values) > 50  # 주석 해제가 조용히 0줄이면 아래 단언이 무력화된다
    assert {k: v for k, v in values.items() if v and "#" in v} == {}

    for key in values:
        monkeypatch.delenv(key, raising=False)
    load_dotenv_file(env)
    s = Settings.from_env()  # 설명이 값에 섞였다면 여기서 ValueError나 설정 반전
    assert s.fast_decode is True and s.preload_model is True
    assert s.device == "cpu"
    assert s.allowed_hosts == ["localhost", "127.0.0.1"]
    TranslateConfig.from_env({
        **os.environ, "OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "OPENAI_MODEL": "m",
    })


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
