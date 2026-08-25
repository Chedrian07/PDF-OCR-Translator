"""운영 계약(CI·compose·네이티브 빌드·하네스 격리) 회귀 테스트.

여기 있는 것들은 전부 "고쳐도 아무도 안 보면 다시 조용히 풀리는" 종류다.
compose 스레딩 누락은 컨테이너 배포 후에야, 빌드 핀 해제는 몇 달 뒤 이미지 재빌드
실패로, 하네스 작업 디렉터리 충돌은 원인 불명의 하네스 실패로 드러난다.
다른 하네스 판정 로직 테스트는 tests/test_verify_e2e_harness.py 에 있다.
"""

import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_ciops_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


verify_e2e = _load("verify_e2e")
mock_llm = _load("mock_llm")


# ─────────────────── docker-compose 환경변수 스레딩 ───────────────────
# .env는 .dockerignore로 이미지 안에 없다 → compose environment에 없는 키는
# 컨테이너에서 조용히 무시된다. "문서에는 있는데 전달되지 않는" 상태의 재발 방지.

BACKEND_SERVICES = ("ocr-cpu", "ocr-cuda", "ocr-ovis", "ocr-paddle")


@pytest.fixture(scope="module")
def compose() -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — compose 계약 검사 생략")
    return yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("svc", BACKEND_SERVICES)
def test_engine_independent_knobs_reach_every_backend_service(compose, svc):
    """엔진과 무관한 노브는 backend 4개 서비스 전부에 있어야 한다.

    PAGE_SEPARATOR는 merge.py(result.md 조립)·render.py(doc-page 분할)·
    qa.py(페이지 컨텍스트)가 읽는다 — OCR 엔진 선택과 아무 관계가 없다.
    ocr-ovis/ocr-paddle에서 빠져 있어 .env 값이 sidecar 스택에서만 무시됐다.
    """
    env = compose["services"][svc]["environment"]
    for key in ("PAGE_SEPARATOR", "LLM_OPENAI_API_KEY"):
        assert key in env, f"{svc}에 {key} 스레딩 누락"


def test_max_length_only_where_a_consumer_exists(compose):
    """반대 방향 — 소비처 없는 서비스에 노브를 복붙하지 않는다.

    MAX_LENGTH의 유일한 소비처는 engine/unlimited.py다. sidecar 스택(ocr-ovis/
    ocr-paddle)에 넣으면 "설정했는데 안 먹는" 노브가 문서와 함께 굳는다.
    """
    src = (REPO / "backend" / "app" / "engine" / "unlimited.py").read_text(encoding="utf-8")
    assert "max_length=s.max_length" in src, "MAX_LENGTH 소비처가 바뀌었다 — 이 계약 재검토"
    for svc in ("ocr-cpu", "ocr-cuda"):
        assert "MAX_LENGTH" in compose["services"][svc]["environment"]
    for svc in ("ocr-ovis", "ocr-paddle"):
        assert "MAX_LENGTH" not in compose["services"][svc]["environment"]


# ─────────────────── 네이티브 빌드 의존성 고정 ───────────────────

def test_native_build_requirements_are_exactly_pinned():
    """backend/Dockerfile이 이미지 빌드 시점에 PEP 517 격리 빌드를 돌린다.

    상한 없는 `>=`면 그날 PyPI 최신이 툴체인이 되어 같은 커밋이 재현되지 않는다.
    실측(2026-08): `>=0.10`/`>=2.12`가 scikit-build-core 1.0.3 / pybind11 3.1.0으로
    해석됐다(둘 다 메이저 2번 건너뜀).
    """
    data = tomllib.loads((REPO / "native" / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = data["build-system"]["requires"]
    assert reqs, "build-system.requires가 비었다"
    for spec in reqs:
        assert "==" in spec, f"고정되지 않은 빌드 의존성: {spec}"
    names = {re.split(r"[=<>!~ ]", s, maxsplit=1)[0] for s in reqs}
    assert {"scikit-build-core", "pybind11", "ninja"} <= names


def test_dockerfile_still_builds_native_at_image_build_time():
    """위 계약의 근거 — Dockerfile이 네이티브를 빌드하지 않게 되면 재검토한다."""
    df = (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "/src/native" in df


# ─────────────────── CI가 실제로 무엇을 돌리는가 ───────────────────

@pytest.fixture(scope="module")
def ci() -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — CI 계약 검사 생략")
    return yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _job_script(job: dict) -> str:
    return "\n".join(str(s.get("run", "")) for s in job["steps"])


def test_verify_e2e_harness_is_wired_into_ci(ci):
    """주요 안전망이 '사람이 기억해야만 도는' 상태로 되돌아가지 않게."""
    job = ci["jobs"]["verify-e2e"]
    # PR에서도 돌아야 의미가 있다 — e2e-mock처럼 nightly 전용 if:가 붙으면 실패.
    assert "if" not in job, "verify-e2e에 트리거 제한이 붙었다 (PR에서 안 돌면 안전망이 아니다)"
    assert "scripts/verify_e2e.py" in _job_script(job)


def test_backend_native_job_exercises_the_installed_native_path(ci):
    """uocr_native가 설치된 backend 경로가 CI에서 실제로 도는가.

    uocr-native는 backend 의존성 그래프 밖이라, 설치 스텝이 사라지면 이 잡은
    backend 잡의 복제가 되고 test_native_ops.py의 패리티 검사는 통째로 skip된다.
    """
    script = _job_script(ci["jobs"]["backend-native"])
    assert "../native" in script, "네이티브 설치 스텝이 사라졌다"
    assert "HAVE_NATIVE" in script, "조용한 skip을 막는 단언 스텝이 사라졌다"
    # uv run은 환경을 uv.lock에 맞춰 재동기화한다. uocr-native는 lock 밖이라 정리
    # 대상이 될지가 uv 버전·설정에 달려 있어, 설치 뒤에는 인터프리터를 직접 부른다.
    after_install = script.split("../native", 1)[1]
    assert "uv run" not in after_install, "설치 후 uv run 재동기화에 운을 맡기고 있다"


# ─────────────────── 하네스 작업 디렉터리 격리 ───────────────────

def test_work_lock_blocks_a_concurrent_run(tmp_path):
    """포트는 갈리지만 작업 디렉터리는 안 갈린다 — 겹치면 서로의 잡을 rmtree 한다."""
    lock = verify_e2e.acquire_work_lock(tmp_path / "w")
    assert lock.is_file()
    # 살아 있는 다른 프로세스가 잡고 있는 것처럼 위조 (init pid 1은 항상 살아 있다).
    lock.write_text("1 0\n")
    with pytest.raises(SystemExit) as e:
        verify_e2e.acquire_work_lock(tmp_path / "w")
    # 종료코드 2 = 단언 실패(1)가 아니라 실행 자체를 못 함 — 감싸는 쪽이 구분해야 한다.
    assert e.value.code == 2


def test_work_lock_takes_over_a_stale_lock(tmp_path):
    """kill -9/정전으로 남은 락 때문에 다음 실행이 영영 막히면 안 된다."""
    work = tmp_path / "w"
    work.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (work / verify_e2e.LOCK_NAME).write_text(f"{dead.pid} 0\n")
    lock = verify_e2e.acquire_work_lock(work)
    assert lock.read_text().split()[0] == str(os.getpid())


def test_release_work_lock_does_not_delete_someone_elses(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    lock = work / verify_e2e.LOCK_NAME
    lock.write_text("1 0\n")
    verify_e2e.release_work_lock(lock)
    assert lock.is_file(), "남의 락을 지웠다"


# ─────────────────── 목 결함 모드가 실제로 배선됐는가 ───────────────────

def test_all_mock_fault_modes_are_exercised_by_the_harness():
    """목이 제공하는 결함 모드 중 하네스가 안 도는 것이 없어야 한다.

    문서(§11.1)는 5+2종 커버리지를 주장했는데 하네스는 4종만 돌았다 —
    drop_placeholder·http429는 코드만 있고 한 번도 실행되지 않았다.
    """
    declared = set(re.findall(r"[?&]fault=([a-z0-9|_]+)", mock_llm.__doc__))
    modes = {m for group in declared for m in group.split("|")}
    assert modes, "목 docstring에서 결함 모드 목록을 못 읽었다"
    src = (SCRIPTS / "verify_e2e.py").read_text(encoding="utf-8")
    wired = set(verify_e2e._FAULT_EXPECT) | set(re.findall(r'"(http\d{3})"', src))
    assert modes <= wired, f"하네스가 안 도는 결함 모드: {sorted(modes - wired)}"


def test_mock_placeholder_vocabulary_matches_masking_module():
    """목이 `<m…>`만 알면 수식 없는 문서에서 drop_placeholder가 no-op이 된다.

    실측: 논문 6페이지의 보호 토큰은 c(인용) 23 · f(참조) 5 · u(URL) 4 · m(수식) 0.
    옛 정규식으로는 지울 것이 하나도 없어 결함 주입이 통과했다.
    """
    from app.translate.masking import _PLACEHOLDER_RE

    def kinds(pattern: str) -> set[str]:
        return set(re.search(r"\[([a-z]+)\]", pattern).group(1))

    assert kinds(mock_llm.PLACEHOLDER_RE.pattern) == kinds(_PLACEHOLDER_RE.pattern)


def test_mock_drop_placeholder_removes_non_math_placeholders():
    """접두 집합이 맞는지 동작으로도 확인한다 (정규식 리팩터에 견디게)."""
    from app.translate.masking import mask

    masked, mapping = mask("See Figure 3 and [12] at https://example.com/x for details.")
    assert mapping and not any(k[0] == "m" for k in mapping)
    dropped = mock_llm._apply_fault("drop_placeholder", masked)
    assert not mock_llm.PLACEHOLDER_RE.search(dropped), "비수식 플레이스홀더가 안 지워졌다"


def test_fault_query_route_survives_base_url_join():
    """문서가 주장하는 `?fault=` 경로가 실제로 목에 도달하는가.

    client._endpoint_url()이 base의 query를 버리도록 바뀌면 하네스의
    drop_placeholder 주입이 조용히 무력화된다(= 정상 번역이 돼 결함이 안 잡힌다).
    """
    from app.translate.client import _endpoint_url, _normalize_base_url

    base = _normalize_base_url("http://127.0.0.1:8899/v1?fault=drop_placeholder")
    url = _endpoint_url(base, "chat/completions")
    assert url == "http://127.0.0.1:8899/v1/chat/completions?fault=drop_placeholder"
    # 목 쪽 판정도 같은 문자열에 걸린다.
    path = url.split("127.0.0.1:8899", 1)[1]
    assert "fault=" in path and path.split("fault=", 1)[1].split("&")[0] == "drop_placeholder"


def test_protected_token_count_uses_the_real_masker():
    """하네스가 보호 토큰을 세는 자 — 마스킹 규칙이 바뀌어도 같은 자를 쓴다."""
    text = "See Figure 3 and [12] at https://example.com/x for details."
    assert verify_e2e.protected_token_count(text) == 3
    assert verify_e2e.protected_token_count("plain sentence") == 0
