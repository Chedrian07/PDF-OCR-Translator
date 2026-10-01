"""verify-e2e 하네스 자식 프로세스 환경 격리 (감사 infra-docs-3·tests-baseline-3).

DISABLE_DOTENV는 .env 자동 로딩만 끈다. 개발자 셸에 export된 실키·공급자 설정은
os.environ 복사로 그대로 넘어가 [7] C-1 단계가 실키로 api.openai.com을 부르고(과금·
문서 본문 반출), PAGE_SEPARATOR 같은 노브가 섞여 CI와 로컬 결과가 갈렸다.
판정 로직 테스트는 tests/test_verify_e2e_harness.py, 작업 디렉터리 락은
tests/test_ci_ops_contracts.py에 있다.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "_envtest_verify_e2e", REPO / "scripts" / "verify_e2e.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness()

# 개발자 셸에 흔히 있는 값들 — 실키 모양 값은 결과 어디에도 남으면 안 된다.
_REAL = "sk-real-do-not-leak"
_PARENT = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/dev",
    "LANG": "ko_KR.UTF-8",
    "TMPDIR": "/tmp/dev",
    "TESSDATA_PREFIX": "/opt/tessdata",
    "LLM_OPENAI_API_KEY": _REAL,
    "OPENAI_API_KEY": _REAL,
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_ORG_ID": "org-real",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "OLLAMA_MODEL": "qwen3:32b",
    "PAGE_SEPARATOR": "\\n=====\\n",
    "TRANSLATE_REASONING": "high",
    "QA_RATE_LIMIT_PER_MIN": "1",
    "HF_TOKEN": _REAL,
    "ANTHROPIC_API_KEY": _REAL,
    "GITHUB_TOKEN": _REAL,
    "AWS_SECRET_ACCESS_KEY": _REAL,
    "HTTPS_PROXY": "http://proxy.corp:3128",
    "no_proxy": "corp",
    "MOCK_TRANSLATE_RATIO": "0",
}


def _backend(parent=None, extra=None):
    return harness.backend_env(
        _PARENT if parent is None else parent, Path("/tmp/harness-data"),
        "http://127.0.0.1:18899", extra or {},
    )


def test_inherited_secrets_never_reach_the_backend():
    env = _backend()
    leaked = sorted(k for k, v in env.items() if _REAL in v)
    assert not leaked, f"상속된 실키가 하네스 백엔드로 샜다: {leaked}"
    assert "LLM_OPENAI_API_KEY" not in env, "C-1 단계가 '키 없음' 상태를 못 본다"
    # 번역 키·주소는 목 값으로만 채워진다
    assert env["OPENAI_API_KEY"] == "sk-mock-translation-key"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:18899/v1"


def test_provider_settings_are_pinned_to_loopback_mocks():
    env = _backend()
    assert env["LLM_PROVIDER"] == "openai-responses"
    # 개발자의 로컬 Ollama에도 닿지 않게 버림 포트로 묶는다
    assert env["OLLAMA_BASE_URL"] == "http://127.0.0.1:9"
    assert "OLLAMA_MODEL" not in env and "OPENAI_ORG_ID" not in env
    proxies = [k for k in env if re.fullmatch(r"(?i)(https?|all|no|ftp)_proxy", k)]
    assert not proxies, f"루프백 트래픽에 프록시가 낄 수 있다: {proxies}"


def test_every_documented_app_knob_comes_from_the_harness():
    """env 키 계약(.env.example 전수 문서화) 위에 서 있다 — 새 노브도 자동으로 격리된다."""
    keys = harness.app_env_keys()
    assert {"PAGE_SEPARATOR", "LLM_OPENAI_API_KEY", "QA_RATE_LIMIT_PER_MIN",
            "TRANSLATE_REASONING", "OLLAMA_BASE_URL", "DATA_DIR"} <= keys
    sentinel = "inherited-from-shell"
    env = _backend({**{k: sentinel for k in keys}, "PATH": "/bin"})
    inherited = sorted(k for k, v in env.items() if v == sentinel)
    assert not inherited, f"셸 값이 그대로 들어간 앱 노브: {inherited}"


def test_execution_environment_is_kept():
    env = _backend()
    for key in ("PATH", "HOME", "LANG", "TMPDIR", "TESSDATA_PREFIX"):
        assert env[key] == _PARENT[key], f"{key}는 실행 환경 자체라 남아야 한다"
    assert env["DISABLE_DOTENV"] == "1" and env["DATA_DIR"] == "/tmp/harness-data"


def test_fault_extras_still_override_harness_values():
    """[8] 결함 주입은 extra로 OPENAI_BASE_URL(?fault=)·FAULT를 덮어쓴다."""
    url = "http://127.0.0.1:18899/v1?fault=drop_placeholder"
    env = _backend(extra={"OPENAI_BASE_URL": url, "FAULT": "echo"})
    assert env["OPENAI_BASE_URL"] == url and env["FAULT"] == "echo"


def test_mock_env_is_scrubbed_and_keeps_its_own_knobs():
    env = harness.mock_env(_PARENT, "refusal")
    assert env["FAULT"] == "refusal"
    assert env["MOCK_TRANSLATE_RATIO"] == "0", "개발자가 준 목 노브는 존중한다"
    assert not any(_REAL in v for v in env.values())
    assert harness.mock_env({"PATH": "/bin"}, "")["MOCK_TRANSLATE_RATIO"] == "0.4"


@pytest.mark.parametrize("name", ["GPG_KEY", "CLIENT_SECRET_ID", "DB_PASSWORD", "ACTIONS_RUNTIME_TOKEN"])
def test_generic_credential_names_are_dropped(name):
    assert name not in harness.scrubbed_env({name: "x", "PATH": "/bin"})


def test_missing_env_example_still_scrubs_credentials(tmp_path):
    keys = harness.app_env_keys(tmp_path / "missing.env.example")
    env = harness.scrubbed_env(_PARENT, app_keys=keys)
    assert not any(_REAL in v for v in env.values())


def test_harness_http_calls_bypass_shell_proxies(monkeypatch):
    """자식 환경에서 프록시를 지운 것과 같은 이유 — 하네스 자신의 루프백 호출(목·백엔드)도
    셸의 HTTP(S)_PROXY를 타지 않는다. 타면 127.0.0.1 호출이 막혀 '목 LLM 기동 실패'가 난다."""
    import urllib.request

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    fresh = _load_harness()  # 오프너는 임포트 시점의 환경으로 만들어진다
    leaky = [h.proxies for h in fresh._LOOPBACK.handlers
             if isinstance(h, urllib.request.ProxyHandler) and h.proxies]
    assert not leaky, f"하네스 HTTP 호출이 셸 프록시를 탄다: {leaky}"


# ── 자식 프로세스 수명 — 기동 실패 시 고아를 남기지 않는다 ──

class _FakeProc:
    started: list["_FakeProc"] = []

    def __init__(self, args, **kwargs):
        self.args = args
        self.signals: list[int] = []
        self._rc = None
        _FakeProc.started.append(self)

    def poll(self):
        return self._rc

    def send_signal(self, sig):
        self.signals.append(sig)
        self._rc = -sig

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        self._rc = -9


@pytest.mark.parametrize("ready", [[False], [True, False]], ids=["mock-down", "backend-down"])
def test_failed_startup_reaps_already_started_children(tmp_path, monkeypatch, ready):
    """__enter__가 실패하면 with 문은 __exit__을 부르지 않는다 — 이미 띄운 목이 고아로 남아
    포트를 쥐면 다음 실행이 그 고아(FAULT 없음)와 대화해 결함 주입이 무력화됐다(실측)."""
    _FakeProc.started = []
    answers = iter(ready)
    monkeypatch.setattr(harness.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(harness, "wait_http", lambda *a, **k: next(answers))
    monkeypatch.setattr(harness, "WORK", tmp_path)

    servers = harness.Servers(tmp_path / "data", {"FAULT": "echo"})
    with pytest.raises(SystemExit):
        with servers:
            pytest.fail("기동 실패인데 본문이 실행됐다")

    assert len(_FakeProc.started) == len(ready)
    orphans = [p.args for p in _FakeProc.started if p.poll() is None]
    assert not orphans, f"기동 실패 후 남은 자식 프로세스: {orphans}"
    if servers.api_log is not None:
        assert servers.api_log.closed


def test_non_operator_app_keys_are_scrubbed_too():
    """.env.example에 없는 앱 키(env 계약의 NOT_OPERATOR_KNOBS)도 셸 값이 새지 않는다 — 특히
    PDF_WORKER_MODE=inline(테스트 세션 값)이 새면 하네스가 PyMuPDF 격리를 끈 채 검증한다."""
    from test_ci_ops_contracts import NOT_OPERATOR_KNOBS

    assert NOT_OPERATOR_KNOBS <= harness._UNDOCUMENTED_APP_KEYS
    env = _backend({"PATH": "/bin", "PDF_WORKER_MODE": "inline", "FAKE_DELAY": "9"})
    assert "PDF_WORKER_MODE" not in env and "FAKE_DELAY" not in env
