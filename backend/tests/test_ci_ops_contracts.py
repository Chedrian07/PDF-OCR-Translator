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


# ── 코드가 읽는 env 키 ↔ compose·.env.example 전수 대조 ──
# 위 두 테스트는 "이미 터진 누락"만 고정한다. 새 노브가 생길 때마다 사람이 여기에
# 줄을 추가해야 한다면 다음 사이클에 또 빠진다 — 실제로 TRUSTED_PROXY_HOPS·
# PDF_EXPORT_MAX_CONCURRENT·PDF_EXPORT_QUEUE_TIMEOUT_S와 남용 방어 4종이 그렇게
# 누락됐다(문서에는 있는데 컨테이너에는 전달되지 않아 "조였다고 믿는" 상태).
# 그래서 키 목록을 **코드에서 뽑아** 대조한다: 새 키는 기본적으로 "전 서비스 + 문서"
# 를 요구하고, 예외는 아래 표에 이유와 함께 명시해야만 통과한다.

# env 키를 읽는 모듈 — 여기 없는 파일에 새 키가 생기면 스캔에 잡히지 않는다.
ENV_SOURCE_FILES = (
    "backend/app/config.py",            # Settings.from_env (대부분의 노브)
    "backend/app/api.py",               # TRUSTED_PROXY_HOPS
    "backend/app/pipeline/derived.py",  # PDF_EXPORT_MAX_CONCURRENT/_QUEUE_TIMEOUT_S
    "backend/app/translate/types.py",   # TranslateConfig.from_env
)
_ENV_KEY_RE = re.compile(r'"([A-Z][A-Z0-9_]{3,})"')

# .env.example에 적지 않는 키 — 운영자가 조정할 대상이 아니다.
NOT_OPERATOR_KNOBS = {
    "DATA_DIR",     # Dockerfile ENV(/data) + compose 볼륨이 정한다
    "FRONTEND_DIR", # 이미지 안 경로 — 리포 상대 탐색이 기본
    "FAKE_DELAY",   # FakeEngine 전용(테스트/데모)
    "DISABLE_DOTENV",  # 테스트·하네스 격리 스위치 — 컨테이너에는 .env 자체가 없다
}

# 소비처가 일부 스택에만 있는 키 → 그 서비스에만 둔다 (§8: 전부에 복붙하면
# 소비되지 않는 값이 문서와 함께 굳는다). 그 외 키는 backend 4개 전부에 있어야 한다.
_UNLIMITED_ONLY = frozenset({"ocr-cpu", "ocr-cuda"})   # 로컬 모델 생성 경로만 읽는다
_SIDECAR_ONLY = frozenset({"ocr-ovis", "ocr-paddle"})  # sidecar HTTP 클라이언트만 읽는다
SERVICE_SCOPED: dict[str, frozenset[str]] = {
    "MAX_LENGTH": _UNLIMITED_ONLY,
    "MAX_PAGE_OUTPUT_CHARS": _UNLIMITED_ONLY,
    "MAX_PAGE_OUTPUT_TOKENS": _UNLIMITED_ONLY,
    "MODEL_ID": _UNLIMITED_ONLY,
    "MODEL_REVISION": _UNLIMITED_ONLY,
    "OCR_DTYPE": _UNLIMITED_ONLY,
    "OCR_DECODE_BLOCK": _UNLIMITED_ONLY,
    "OCR_FAST_DECODE": _UNLIMITED_ONLY,
    "PAGES_PER_CHUNK": _UNLIMITED_ONLY,
    "OCR_CPU_THREADS": frozenset({"ocr-cpu"}),  # CUDA 스택은 torch CPU 스레드가 무의미
    # MLX 엔진은 macOS 호스트 네이티브 전용(컨테이너에는 Metal이 없다) — 어느 서비스도 소비하지 않는다
    "OCR_MLX_QUANT_BITS": frozenset(),
    "OCR_SIDECAR_URL": _SIDECAR_ONLY,
    "OCR_SIDECAR_CONNECT_TIMEOUT_S": _SIDECAR_ONLY,
    "OCR_SIDECAR_READ_TIMEOUT_S": _SIDECAR_ONLY,
    "OCR_SIDECAR_HEALTH_TIMEOUT_S": _SIDECAR_ONLY,
    "OCR_SIDECAR_MAX_RESPONSE_MB": _SIDECAR_ONLY,
    "OCR_SIDECAR_RETRIES": _SIDECAR_ONLY,
    "OCR_SIDECAR_MODEL_WAIT_S": _SIDECAR_ONLY,
    "OCR_REMOTE_PAGE_CONCURRENCY": _SIDECAR_ONLY,
    **dict.fromkeys(NOT_OPERATOR_KNOBS, frozenset()),
}


def _env_keys_read_by_code() -> set[str]:
    keys: set[str] = set()
    for rel in ENV_SOURCE_FILES:
        keys |= set(_ENV_KEY_RE.findall((REPO / rel).read_text(encoding="utf-8")))
    return keys


def test_env_key_scanner_still_sees_the_known_knobs():
    """스캐너가 조용히 0개를 반환하면 아래 두 계약이 통과하며 무력화된다."""
    keys = _env_keys_read_by_code()
    sentinel = {
        "TRUSTED_PROXY_HOPS", "PDF_EXPORT_MAX_CONCURRENT", "PDF_EXPORT_QUEUE_TIMEOUT_S",
        "QA_RATE_LIMIT_PER_MIN", "TRANSLATE_MAX_ACTIVE", "OPENAI_BASE_URL",
    }
    assert sentinel <= keys, f"env 키 스캐너가 놓친 키: {sorted(sentinel - keys)}"


def test_every_env_key_the_code_reads_is_documented_in_env_example():
    """코드가 읽는 키는 .env.example에 있어야 한다 — 없으면 존재 자체를 모른다."""
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    missing = sorted(
        k for k in _env_keys_read_by_code() - NOT_OPERATOR_KNOBS
        if not re.search(rf"\b{k}\b", text)
    )
    assert not missing, f".env.example에 없는 노브: {missing}"


@pytest.mark.parametrize("svc", BACKEND_SERVICES)
def test_every_env_key_the_code_reads_is_threaded_into_compose(compose, svc):
    """엔진 무관 노브는 backend 4개 전부에, 스택 전용 노브는 그 스택에만.

    양방향 계약이다 — 누락(컨테이너에서 조용히 무시)과 과잉 복붙(소비처 없는 값이
    문서와 함께 굳음)을 같은 자로 잡는다. 새 키는 예외 표에 없으면 '전부' 취급이라
    기본값이 안전한 쪽이다.
    """
    env = compose["services"][svc]["environment"]
    for key in sorted(_env_keys_read_by_code()):
        expected = svc in SERVICE_SCOPED.get(key, frozenset(BACKEND_SERVICES))
        assert (key in env) is expected, (
            f"{svc}: {key} 스레딩 {'누락' if expected else '과잉'} "
            "(SERVICE_SCOPED에 이유를 적거나 compose를 고칠 것)"
        )


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


# ─────────────────── 컨테이너 하드닝 (감사 security-6·infra-docs-13) ───────────────────
# CPU 스택 실측으로 검증한 하드닝이 새 서비스 추가·리팩터에서 조용히 빠지지 않게 한다.

ALL_COMPOSE_SERVICES = (*BACKEND_SERVICES, "ovisocr2", "paddleocr-vl")
# 같은 CPU 이미지를 쓰는 백엔드 — 쓰기 경로가 /data·/tmp뿐임을 실측했다(read_only 대상).
# ocr-cuda·sidecar는 GPU 호스트에서 검증하기 전까지 read_only를 걸지 않는다.
READ_ONLY_SERVICES = ("ocr-cpu", "ocr-ovis", "ocr-paddle")


@pytest.mark.parametrize("svc", ALL_COMPOSE_SERVICES)
def test_every_compose_service_drops_all_capabilities(compose, svc):
    spec = compose["services"][svc]
    assert spec.get("cap_drop") == ["ALL"], f"{svc}: cap_drop: [ALL] 누락"
    assert "no-new-privileges:true" in spec.get("security_opt", []), f"{svc}: no-new-privileges 누락"


@pytest.mark.parametrize("svc", READ_ONLY_SERVICES)
def test_cpu_image_backends_run_with_a_read_only_root(compose, svc):
    spec = compose["services"][svc]
    assert spec.get("read_only") is True, f"{svc}: read_only 누락"
    tmpfs = spec.get("tmpfs", [])
    assert any(str(t).startswith("/tmp:") for t in tmpfs), f"{svc}: /tmp tmpfs가 없으면 업로드 스풀이 실패한다"
    assert any("/data" in str(v) for v in spec.get("volumes", [])), f"{svc}: /data는 볼륨이어야 쓴다"


@pytest.mark.parametrize("svc", BACKEND_SERVICES)
def test_backends_cap_process_count(compose, svc):
    limits = compose["services"][svc]["deploy"]["resources"]["limits"]
    assert int(limits.get("pids", 0)) > 0, f"{svc}: pids 상한 누락(fork 폭주 차단)"


@pytest.mark.parametrize("svc", BACKEND_SERVICES)
def test_hardening_keeps_the_owner_network_defaults(compose, svc):
    """하드닝은 노출 기본값을 바꾸지 않는다 — 0.0.0.0 게시·ALLOWED_HOSTS='*'는 소유자 결정이다."""
    spec = compose["services"][svc]
    assert any(str(p).startswith("${BIND_HOST:-0.0.0.0}:") for p in spec["ports"]), spec["ports"]
    assert spec["environment"]["ALLOWED_HOSTS"] == "${ALLOWED_HOSTS:-*}"


_FROM_RE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)
DOCKERFILES = ("backend/Dockerfile", "services/ovisocr2/Dockerfile", "services/paddleocr_vl/Dockerfile")


@pytest.mark.parametrize("path", DOCKERFILES)
def test_base_images_are_pinned_by_digest(path):
    """가변 태그는 재빌드마다 OS·툴체인이 바뀌어 같은 버전 라벨에 다른 이미지가 나간다."""
    text = (REPO / path).read_text(encoding="utf-8")
    images = _FROM_RE.findall(text)
    assert images, f"{path}: FROM을 못 찾았다"
    unpinned = [i for i in images if not re.search(r"@sha256:[0-9a-f]{64}$", i)]
    assert not unpinned, f"{path}: digest 미고정 베이스 {unpinned}"


@pytest.mark.parametrize("path", ("backend/Dockerfile", "services/paddleocr_vl/Dockerfile"))
def test_digest_pinned_debian_bases_still_get_security_updates(path):
    """digest 고정만 하면 OS 보안 패치가 영구 동결된다 — apt-get upgrade와 반드시 함께 간다."""
    assert "apt-get upgrade -y" in (REPO / path).read_text(encoding="utf-8")


def test_runtime_image_code_is_read_only_and_shutdown_is_bounded():
    df = (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8")
    copies = [line for line in df.splitlines() if line.startswith("COPY") and "/srv/" in line]
    assert copies and not any("--chown" in c for c in copies), f"앱 코드가 실행 사용자 소유다: {copies}"
    assert "UV_COMPILE_BYTECODE=1" in df and "compileall" in df, "바이트코드 사전 컴파일이 빠졌다"
    assert re.search(r"^USER app$", df, re.MULTILINE)
    # 열린 SSE가 graceful shutdown을 막아 SIGKILL(137)로 죽던 문제(감사 concurrency-11)
    assert "--timeout-graceful-shutdown" in df


@pytest.mark.parametrize("svc", ("ovisocr2", "paddleocr_vl"))
def test_sidecar_locks_are_hashed_and_installed_with_hash_checking(svc):
    base = REPO / "services" / svc
    lock = (base / "requirements.lock").read_text(encoding="utf-8")
    reqs = re.findall(r"^([A-Za-z0-9._-]+)(?:==| @ )", lock, re.MULTILINE)
    assert reqs, "requirements.lock에 고정된 요구사항이 없다"
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock.split("\n", 1)[1] if lock.startswith("#") else lock)
    unhashed = [b.split()[0] for b in blocks if re.match(r"[A-Za-z0-9]", b) and "--hash=sha256:" not in b]
    assert not unhashed, f"해시 없는 요구사항: {unhashed}"
    df = (base / "Dockerfile").read_text(encoding="utf-8")
    assert "--require-hashes" in df and "--no-deps" in df and "requirements.lock" in df


# ─────────────────── 공급망 — CI·릴리스 게이트 (감사 gap2-8·infra-docs-16~19) ───────────────────

WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("wf", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_full_commit_sha(wf):
    """태그는 다른 커밋으로 다시 밀릴 수 있다 — 버전은 주석으로만 남긴다."""
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", wf.read_text(encoding="utf-8"), re.MULTILINE)
    assert uses, f"{wf.name}: uses를 못 찾았다"
    loose = [u for u in uses if not u.startswith("./") and not re.search(r"@[0-9a-f]{40}$", u)]
    assert not loose, f"{wf.name}: SHA 미고정 액션 {loose}"


def test_ci_audits_dependencies_without_silent_skips(ci):
    script = _job_script(ci["jobs"]["dependency-audit"])
    assert "pip-audit" in script and "--extra cpu" in script
    # torch '+cpu' 같은 로컬 버전은 PyPI에 없어 '감사 불가'로 조용히 빠진다 — --strict가 막는다
    assert "--strict" in script
    for svc in ("ovisocr2", "paddleocr_vl"):
        assert svc in script, f"sidecar 잠금 {svc}를 감사하지 않는다"


_ADVISORY_ID = re.compile(r"\b(?:PYSEC-\d{4}-\d+|CVE-\d{4}-\d+|GHSA(?:-[0-9a-z]{4}){3})\b")


def _audit_policy(text: str) -> tuple[set[str], set[str], set[str]]:
    """감사 스크립트 → (수용 권고 ID, 재검토 기한, pip-audit 버전 고정)."""
    block = re.search(r"\b(?:ignores|IGNORES)=\((.*?)^\s*\)", text, re.MULTILINE | re.DOTALL)
    assert block, "수용 권고 목록(ignores=(…))을 못 찾았다"
    return (
        set(_ADVISORY_ID.findall(block.group(1))),
        set(re.findall(r"\b20\d\d-\d\d-\d\d\b", text)),
        set(re.findall(r"pip-audit==[\w.]+", text)),
    )


def test_local_dependency_audit_matches_the_ci_job(ci):
    """make audit(scripts/dependency_audit.sh)는 CI 잡과 같은 수용 목록·기한·도구 버전이어야
    한다 — 한쪽만 고치면 로컬은 통과하는데 CI가 실패하거나, 만료된 수용이 로컬에 남는다."""
    ci_ids, ci_dates, ci_tool = _audit_policy(_job_script(ci["jobs"]["dependency-audit"]))
    local = (SCRIPTS / "dependency_audit.sh").read_text(encoding="utf-8")
    ids, dates, tool = _audit_policy(local)
    assert ci_ids and ids == ci_ids
    assert ci_dates == {"2027-04-01"} and dates == ci_dates
    assert ci_tool and tool == ci_tool
    for flag in ("--extra cpu", "--strict", "--no-deps", "-s osv"):
        assert flag in local, f"CI와 다른 감사 방식: {flag} 없음"


def test_ci_builds_and_smokes_the_image_without_pushing(ci):
    job = ci["jobs"]["docker-image"]
    build = next(s for s in job["steps"] if "build-push-action" in str(s.get("uses", "")))
    assert build["with"]["push"] is False and build["with"]["file"] == "backend/Dockerfile"
    assert "scripts/smoke_image.sh" in _job_script(job)


@pytest.fixture(scope="module")
def release() -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — 릴리스 계약 검사 생략")
    return yaml.safe_load((REPO / ".github/workflows/release.yml").read_text(encoding="utf-8"))


def test_release_scans_the_image_before_pushing(release):
    steps = release["jobs"]["publish-images"]["steps"]
    runs = [str(s.get("run", "")) for s in steps]
    scan = next(i for i, r in enumerate(runs) if "trivy" in r)
    push = next(i for i, r in enumerate(runs) if "docker push" in r)
    assert scan < push, "취약점 스캔이 push 뒤에 있다"
    assert "--exit-code 1" in runs[scan] and "--ignorefile" in runs[scan]
    assert re.search(r"aquasec/trivy:[\w.]+@sha256:[0-9a-f]{64}", runs[scan]), "trivy 이미지 digest 미고정"


def test_release_waits_for_ci_and_ships_checksummed_assets(release):
    verify = "\n".join(str(s.get("run", "")) for s in release["jobs"]["verify-release"]["steps"])
    assert "sleep" in verify and "status != \"completed\"" in verify, "CI 진행 중이면 기다려야 한다"
    build = next(s for s in release["jobs"]["publish-images"]["steps"]
                 if "build-push-action" in str(s.get("uses", "")))
    assert "cache-to" not in build["with"], "태그 ref 캐시는 다음 태그에서 복원되지 않는다"
    assets = release["jobs"]["release-assets"]
    assert assets["permissions"] == {"contents": "write"}
    assert "SHA256SUMS" in _job_script(assets) and "gh release upload" in _job_script(assets)


def test_accepted_image_advisories_carry_a_reason_and_expiry():
    yaml = pytest.importorskip("yaml")
    entries = yaml.safe_load((REPO / ".github/trivyignore.yaml").read_text(encoding="utf-8"))
    for e in entries["vulnerabilities"]:
        assert e.get("statement") and e.get("expired_at"), f"{e['id']}: 이유·재검토 기한 누락"
        assert e.get("purls"), f"{e['id']}: purl로 현재 고정 버전에만 걸어야 한다"


# ─────────────────── 문서·스크립트 드리프트 ───────────────────

def test_security_policy_does_not_contradict_the_compose_wiring(compose):
    """SECURITY.md만 'compose가 남용 방어 변수를 전달하지 않는다'고 적어 운영자가 .env 조정을
    포기하게 만들었다(감사 security-5·infra-docs-10). 문서의 변수는 실제로 전달돼야 한다."""
    text = (REPO / "SECURITY.md").read_text(encoding="utf-8")
    assert "do not thread" not in text
    for key in ("QA_RATE_LIMIT_PER_MIN", "QA_MAX_CONCURRENT",
                "TRANSLATE_RATE_LIMIT_PER_MIN", "TRANSLATE_MAX_ACTIVE"):
        assert key in text
        for svc in BACKEND_SERVICES:
            assert key in compose["services"][svc]["environment"], f"{svc}: {key}"


def test_scripts_do_not_import_the_legacy_fitz_module():
    """app 코드와 같은 규칙 — PyMuPDF 1.28+는 `import fitz` 때 폐지 경고를 찍는다."""
    offenders = [
        f"{p.name}:{n}" for p in sorted(SCRIPTS.glob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*(import fitz\b|from fitz\b)", line)
    ]
    assert not offenders, f"레거시 fitz 임포트: {offenders}"
