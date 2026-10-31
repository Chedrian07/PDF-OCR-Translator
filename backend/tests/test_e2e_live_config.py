"""Live harness configuration stays real, isolated, and silent about credentials."""

import pytest

import e2e_live_app as harness
from app.config import _APP_ENV_KEYS
from app.translate.types import TranslateAPIError, TranslateConfig


def test_only_translation_settings_are_loaded_from_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENAI_BASE_URL=https://provider.example/v1\nOPENAI_API_KEY=test-provider-key\n"
        "OPENAI_MODEL=real-model\nTRANSLATE_REASONING=off\nTRANSLATE_STREAM=0\n"
        "OCR_ENGINE=retired\nOCR_DEVICE=invalid-retired-device\nMODEL_ID=old-model\n"
        "MODEL_REVISION=old-revision\nOCR_SIDECAR_URL=http://retired:9\n"
        "LLM_OPENAI_API_KEY=unrelated-qa-key\nOPENAI_UNKNOWN_KNOB=ignored\n"
    )
    env = harness.live_environment({}, dotenv_path=dotenv, data_dir=tmp_path / "data")
    assert env["OPENAI_API_KEY"] == "test-provider-key"
    assert env["TRANSLATE_REASONING"] == "off" and env["TRANSLATE_STREAM"] == "0"
    assert env["OCR_ENGINE"] == "unlimited2" and env["OCR_DEVICE"] == "cuda"
    assert "MODEL_ID" not in env and "MODEL_REVISION" not in env
    assert env["OCR_SIDECAR_URL"] == ""
    assert env["LLM_OPENAI_API_KEY"] == "" and "OPENAI_UNKNOWN_KNOB" not in env
    assert env["DISABLE_DOTENV"] == "1" and env["OCR_TEST_MODE"] == "0"


def test_shell_translation_overrides_dotenv_and_only_explicit_ocr_route_is_kept(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=dotenv-key\nOCR_DEVICE=mps\n")
    base = {
        "OPENAI_API_KEY": "shell-key", "E2E_OCR_DEVICE": "cpu",
        "OCR_SIDECAR_URL": "http://127.0.0.1:10000", "OCR_DEVICE": "fake",
        "DATA_DIR": "/unrelated-real-jobs", "OCR_RETIRED_SETTING": "wrong",
        "LLM_LOCAL_OPENAI_API_KEY": "unrelated-key", "GITHUB_TOKEN": "unrelated-token",
        "PATH": "/bin", "HTTPS_PROXY": "http://configured-proxy:8080",
    }
    env = harness.live_environment(base, dotenv_path=dotenv, data_dir=tmp_path / "isolated")
    assert env["OPENAI_API_KEY"] == "shell-key"
    assert env["OCR_DEVICE"] == "cpu"
    assert env["OCR_SIDECAR_URL"] == base["OCR_SIDECAR_URL"]
    assert env["DATA_DIR"] == str(tmp_path / "isolated")
    assert "OCR_RETIRED_SETTING" not in env and "GITHUB_TOKEN" not in env
    assert env["LLM_LOCAL_OPENAI_API_KEY"] == ""
    assert env["PATH"] == "/bin" and env["HTTPS_PROXY"] == base["HTTPS_PROXY"]


def test_all_other_app_knobs_are_isolated_without_changing_parent(tmp_path):
    base = {key: "parent-sentinel" for key in _APP_ENV_KEYS}
    base.update(E2E_OCR_DEVICE="cuda", OCR_SIDECAR_URL="http://127.0.0.1:10000")
    before = base.copy()
    env = harness.live_environment(base, dotenv_path=tmp_path / "missing", data_dir=tmp_path)
    assert base == before
    inherited = {key for key, value in env.items() if value == "parent-sentinel"}
    assert inherited == harness._TRANSLATION_KEYS


def test_provider_probe_takes_precedence_over_frontend_catch_all(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.mount("/", StaticFiles(directory=tmp_path, html=True))
    cfg = TranslateConfig(base_url="https://provider.example/v1", api_key="test-key", model="real")
    monkeypatch.setattr(harness.TranslateConfig, "from_env", lambda: cfg)
    calls = []

    def probe(config):
        calls.append(config)
        return {"status": "blocked-auth", "reason": "authentication"}, 409

    monkeypatch.setattr(harness, "probe_provider", probe)
    harness.add_provider_probe(app)
    response = TestClient(app).post("/e2e/live-provider-check")
    assert response.status_code == 409
    assert response.json() == {"status": "blocked-auth", "reason": "authentication"}
    assert calls == [cfg]


@pytest.mark.parametrize(
    "answer, failure, expected, status",
    [
        ("공개 합성 번역 시험입니다.", None, "ready", 200),
        ("English echo", None, "blocked-provider", 409),
        (None, "번역 API 인증 실패 — secret-value-must-not-leak", "blocked-auth", 409),
        (None, "provider error with secret-value-must-not-leak", "blocked-provider", 409),
        (None, "번역 API 오류 (HTTP 402): secret-value-must-not-leak", "blocked-provider", 409),
    ],
)
def test_live_probe_is_bounded_and_reports_only_sanitized_status(
    monkeypatch, answer, failure, expected, status
):
    calls = []

    class Client:
        def __init__(self, cfg):
            assert cfg.max_retries == 0 and cfg.timeout_s == 30
            self.session = self

        def complete(self, system, user, *, max_tokens):
            calls.append((user, max_tokens))
            if failure:
                raise TranslateAPIError(failure)
            return answer

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(harness, "OpenAICompatClient", Client)
    cfg = TranslateConfig(base_url="https://provider.example/v1", api_key="test-key", model="real")
    body, code = harness.probe_provider(cfg)
    assert body["status"] == expected and code == status
    assert "secret-value" not in str(body)
    assert calls == [("This is a public synthetic translation test.", cfg.max_output_tokens), "closed"]
    if failure and "HTTP 402" in failure:
        assert body["http_status"] == 402
