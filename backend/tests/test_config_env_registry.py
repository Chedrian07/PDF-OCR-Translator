"""알려진 env 키 레지스트리(config.KNOWN_ENV_KEYS)의 드리프트 계약.

load_dotenv_file은 레지스트리 밖의 .env 키를 '이 앱이 읽지 않는 키'로 알린다(감사
translate-llm-13·infra-docs-7·mlx-integration-8). 레지스트리가 코드보다 늦으면 진짜
노브에 거짓 경고가, 앞서면 사라진 키가 조용히 통과한다 — 그래서 양방향으로 대조한다.
- 코드가 읽는 키(test_ci_ops_contracts의 스캐너 + 앱 전체의 os.environ 직접 조회) ⊆ 앱 키
- .env.example 키 줄·docker-compose.yml의 ${…} ⊆ 레지스트리 (.env를 복사한 사용자에게 거짓 경고 금지)
- 반대로 레지스트리의 키는 저마다 실제 출처(코드·배포 파일·하네스)가 있어야 한다.
"""

import re
from pathlib import Path

from test_ci_ops_contracts import NOT_OPERATOR_KNOBS, _env_keys_read_by_code

from app import config

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "backend" / "app"
# os.environ.get("X") · os.environ["X"] · os.environ.setdefault("X", …) · os.getenv("X")
_DIRECT_ENV_READ = re.compile(
    r"""(?:os\.environ(?:\.get|\.setdefault|\.pop)?\(|os\.environ\[|os\.getenv\()\s*["']([A-Z][A-Z0-9_]+)["']"""
)
_ENV_EXAMPLE_KEY = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)
_COMPOSE_VAR = re.compile(r"\$\{([A-Z][A-Z0-9_]*)")


def _direct_env_reads() -> set[str]:
    keys: set[str] = set()
    for path in APP.rglob("*.py"):
        keys |= set(_DIRECT_ENV_READ.findall(path.read_text(encoding="utf-8")))
    return keys


def _keys_read_by_app() -> set[str]:
    return _env_keys_read_by_code() | _direct_env_reads()


def _deploy_file_keys() -> set[str]:
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    return set(_ENV_EXAMPLE_KEY.findall(example)) | set(_COMPOSE_VAR.findall(compose))


def test_scanners_still_find_the_known_readers():
    """스캐너가 조용히 0개를 반환하면 아래 계약이 통과하며 무력화된다."""
    direct = _direct_env_reads()
    assert {"OCR_NGRAM_HOST", "OCR_SDPA", "OCR_CUDA_GRAPHS", "PYTORCH_ENABLE_MPS_FALLBACK"} <= direct
    assert {"OCR_DEVICE", "OPENAI_BASE_URL", "BIND_HOST", "OVIS_MODEL_ID"} <= (
        _env_keys_read_by_code() | _deploy_file_keys()
    )


def test_every_key_the_code_reads_is_registered_as_an_app_key():
    missing = sorted(_keys_read_by_app() - config._APP_ENV_KEYS)
    assert not missing, f"config._APP_ENV_KEYS에 없는 env 키(코드가 읽음): {missing}"


def test_every_documented_or_compose_key_is_known():
    """.env.example을 복사하거나 compose 변수를 .env에 둔 사용자가 거짓 경고를 받지 않는다."""
    missing = sorted(_deploy_file_keys() - config.KNOWN_ENV_KEYS)
    assert not missing, f"config.KNOWN_ENV_KEYS에 없는 문서·compose 키: {missing}"


def test_registered_keys_still_have_a_source():
    """반대 방향 — 더는 아무도 읽지 않는 키가 레지스트리에 남으면 그 오타는 경고 없이 통과한다."""
    stale_app = sorted(config._APP_ENV_KEYS - _keys_read_by_app())
    assert not stale_app, f"코드가 더는 읽지 않는 앱 키: {stale_app}"
    stale_deploy = sorted(config._DEPLOY_ENV_KEYS - _deploy_file_keys())
    assert not stale_deploy, f".env.example·compose에 없는 배포 키: {stale_deploy}"

    harness_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (REPO / "scripts", REPO / "backend" / "tests", REPO / "frontend" / "tests")
        for path in root.rglob("*")
        if path.suffix in (".py", ".mjs", ".sh") and path.name != Path(__file__).name
    )
    stale_harness = sorted(
        key for key in config._HARNESS_ENV_KEYS if not re.search(rf"\b{key}\b", harness_text)
    )
    assert not stale_harness, f"하네스·테스트가 더는 읽지 않는 키: {stale_harness}"


def test_categories_do_not_overlap_and_the_registry_is_not_scanned_as_reads():
    """레지스트리는 공백 구분 블록이라 CI 스캐너가 '코드가 읽는 키'로 세지 않는다 — 셌다면
    하네스 키까지 .env.example·compose 스레딩을 요구받는다."""
    assert not config._APP_ENV_KEYS & config._DEPLOY_ENV_KEYS
    assert not config._APP_ENV_KEYS & config._HARNESS_ENV_KEYS
    assert not config._DEPLOY_ENV_KEYS & config._HARNESS_ENV_KEYS
    assert not config._HARNESS_ENV_KEYS & _env_keys_read_by_code()
    assert "REASONING_EFFORT" not in _env_keys_read_by_code()
    assert "REASONING_EFFORT" not in config.KNOWN_ENV_KEYS


# 앱이 다른 도구(torch)에 넘기려고 setdefault하는 키 — 운영자 노브가 아니다(ARCHITECTURE에 문서화)
_TOOL_OWNED_KEYS = frozenset({"PYTORCH_ENABLE_MPS_FALLBACK"})


def test_keys_read_directly_anywhere_in_the_app_are_documented_in_env_example():
    """CI 스캐너(ENV_SOURCE_FILES)는 다섯 파일만 본다 — 그 밖(vendor·native_ops·engine)에서
    os.environ으로 읽는 노브도 .env.example에 있어야 운영자가 존재를 안다. OCR_SDPA·
    OCR_NGRAM_HOST가 코드·compose·ARCHITECTURE에는 있는데 .env.example에만 빠져 있었다."""
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    direct = _direct_env_reads()
    assert {"OCR_SDPA", "OCR_NGRAM_HOST"} <= direct
    missing = sorted(
        key for key in direct - NOT_OPERATOR_KNOBS - _TOOL_OWNED_KEYS
        if not re.search(rf"\b{key}\b", text)
    )
    assert not missing, f".env.example에 없는 노브(앱이 직접 읽음): {missing}"
