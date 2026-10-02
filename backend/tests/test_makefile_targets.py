"""루트 Makefile 타깃 계약 — `make -n`이 실제로 실행할 명령으로 검증한다.

개발 서버의 종료 대기 상한, 기기 의존 opt-in 테스트 타깃(MPS·MLX 실가중치), CI와 같은
의존성 감사 타깃을 고정한다. 타깃이 가리키는 테스트 파일·스위치·스크립트가 바뀌면
타깃이 조용히 아무것도 돌리지 않게 되므로 그 연결도 함께 본다.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _recipe(target: str) -> str:
    if shutil.which("make") is None:
        pytest.skip("make 없음")
    out = subprocess.run(
        ["make", "-s", "-n", "-C", str(REPO), target],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout


@pytest.mark.parametrize("target", ["dev", "dev-metal", "dev-textlayer"])
def test_dev_servers_bound_their_graceful_shutdown(target):
    """열린 SSE·keep-alive 연결이 Ctrl+C·--reload 재시작을 무기한 붙잡지 않게 한다
    (Dockerfile의 uvicorn과 같은 5초)."""
    recipe = _recipe(target)
    assert "uvicorn app.main:app" in recipe
    assert re.search(r"--timeout-graceful-shutdown\s+5\b", recipe), recipe


@pytest.mark.parametrize("target,switch,files", [
    ("test-mps", "OCR_MPS_TESTS", ["tests/test_mps_contract.py", "tests/test_objc_pool.py"]),
    ("test-mlx-real", "OCR_MLX_REAL_TESTS", ["tests/test_mlx_model_parity.py"]),
])
def test_opt_in_device_targets_enable_the_switch_their_tests_read(target, switch, files):
    recipe = _recipe(target)
    assert f"{switch}=1" in recipe
    assert "-rs" in recipe.split()  # 건너뛴 테스트의 사유를 보인다 — 조용한 skip ≠ 통과
    for rel in files:
        assert rel in recipe
        test_file = REPO / "backend" / rel
        assert test_file.is_file(), rel
        # 스위치 이름이 바뀌면 타깃은 전부 skip인 채 초록으로 끝난다
        assert switch in test_file.read_text(encoding="utf-8"), rel


def test_audit_target_runs_the_ci_equivalent_script():
    assert _recipe("audit").strip() == "./scripts/dependency_audit.sh"
    script = REPO / "scripts" / "dependency_audit.sh"
    assert script.is_file() and os.access(script, os.X_OK)


def test_every_target_is_phony():
    """같은 이름의 파일·디렉터리(예: audit/)가 생기면 make가 그 타깃을 '최신'으로 보고 건너뛴다."""
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    phony_block = re.search(r"^\.PHONY:((?:.*\\\n)*.*)$", text, re.MULTILINE).group(1)
    phony = set(phony_block.replace("\\\n", " ").split())
    targets = set(re.findall(r"^([a-z][a-z0-9-]*):", text, re.MULTILINE))
    assert targets and targets <= phony, sorted(targets - phony)


# ─────────────── test-mlx-real의 HF 캐시·스냅샷 가드 (감사 mlx-3·infra-docs-4) ───────────────
# 테스트는 .env를 읽지 않는다(conftest가 DISABLE_DOTENV=1) — 캐시 위치를 .env에만 둔 개발자는
# make dev가 받은 스냅샷을 두고 실가중치 2건이 LocalEntryNotFoundError로 실패했다(주석은
# '건너뛴다'였다). 타깃은 scripts/require_hf_snapshot.py를 거쳐 make dev와 같은 캐시를 보고,
# 가중치가 없으면 pytest를 돌리지 않고 종료코드 2로 멈춘다.

GUARD = REPO / "scripts" / "require_hf_snapshot.py"
_HF_CACHE_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")


def _load_guard():
    spec = importlib.util.spec_from_file_location("_make_require_hf_snapshot", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mlx_real_target_runs_pytest_behind_the_snapshot_guard():
    recipe = _recipe("test-mlx-real")
    head, sep, tail = recipe.partition("../scripts/require_hf_snapshot.py --")
    assert sep, recipe
    assert "OCR_MLX_REAL_TESTS=1" in head  # 가드가 exec하는 pytest까지 스위치가 이어진다
    assert re.match(r"\s*(?:\\\s*)?\.venv/bin/python -m pytest\b", tail), recipe


def test_snapshot_guard_forwards_only_hf_cache_location_keys(tmp_path):
    guard = _load_guard()
    assert set(guard.CACHE_KEYS) == set(_HF_CACHE_KEYS)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "HF_HOME=/from/dotenv\n"
        "export HF_HUB_CACHE='/cache dir/hub'  # 주석\n"
        "HF_TOKEN=hf_secret\nOPENAI_API_KEY=sk-secret\n",
        encoding="utf-8",
    )
    # 자격증명·앱 설정은 넘기지 않는다 — 파서는 make dev와 같은 python-dotenv(export·따옴표·주석)
    assert guard.dotenv_cache_env(dotenv, {}) == {
        "HF_HOME": "/from/dotenv", "HF_HUB_CACHE": "/cache dir/hub",
    }
    # 셸에 이미 있는 값이 이긴다 — load_dotenv_file과 같은 우선순위
    assert guard.dotenv_cache_env(dotenv, {"HF_HOME": "/shell"}) == {"HF_HUB_CACHE": "/cache dir/hub"}
    assert guard.dotenv_cache_env(tmp_path / "missing.env", {}) == {}
    assert guard.dotenv_cache_env(None, {}) == {}
    # 자동 탐색은 make dev와 같은 함수(app.config._find_dotenv)다 — 이름이 바뀌면 여기서 깨진다
    assert guard._auto_dotenv({"DISABLE_DOTENV": "1"}) is None
    found = guard._auto_dotenv({})
    assert found is None or found.name == ".env"


def test_snapshot_guard_stops_before_the_command_without_cached_weights(tmp_path):
    from app.config import Settings

    s = Settings()  # 실가중치 테스트가 쓰는 코드 기본 스냅샷
    hub = tmp_path / "hub"
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"HF_HUB_CACHE={hub}\n", encoding="utf-8")
    marker = tmp_path / "ran"
    probe = f"import os, pathlib; pathlib.Path({str(marker)!r}).write_text(os.environ['HF_HUB_CACHE'])"
    cmd = [sys.executable, str(GUARD), "--dotenv", str(dotenv), "--", sys.executable, "-c", probe]
    env = {k: v for k, v in os.environ.items() if k not in _HF_CACHE_KEYS}

    def run():
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

    out = run()
    assert out.returncode == 2, out.stderr
    assert not marker.exists(), "스냅샷 없이 명령(pytest)이 돌았다"
    assert "make dev" in out.stderr and str(hub) in out.stderr

    snap = hub / f"models--{s.model_id.replace('/', '--')}" / "snapshots" / s.model_revision
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    out = run()
    assert out.returncode == 2 and not marker.exists(), "가중치 없는 부분 스냅샷을 통과시켰다"

    (snap / "model-00001-of-00001.safetensors").write_bytes(b"")
    out = run()
    assert out.returncode == 0, out.stderr
    assert marker.read_text(encoding="utf-8") == str(hub)  # .env의 캐시 위치가 명령까지 갔다
