"""루트 Makefile 타깃 계약 — `make -n`이 실제로 실행할 명령으로 검증한다.

개발 서버의 종료 대기 상한, 기기 의존 opt-in 테스트 타깃(MPS·MLX 실가중치), CI와 같은
의존성 감사 타깃을 고정한다. 타깃이 가리키는 테스트 파일·스위치·스크립트가 바뀌면
타깃이 조용히 아무것도 돌리지 않게 되므로 그 연결도 함께 본다.
"""

import os
import re
import shutil
import subprocess
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
