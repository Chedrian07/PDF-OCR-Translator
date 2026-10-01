"""보안 하한 — 알려진 취약 버전으로 lock이 내려가면 CI에서 바로 실패시킨다.

pip-audit 같은 패키지 단위 스캐너는 PyMuPDF 휠에 동봉된 MuPDF C 라이브러리의
버전을 보지 못한다(pymupdf 1.27.2.2도 '취약점 0건'으로 보였지만 동봉 MuPDF
1.27.2에는 CVE-2026-3308이 남아 있었다). 그래서 실제로 임포트되는 라이브러리의
버전을 직접 읽어 수정판 미만이면 실패한다.

하한은 각 권고가 고쳐진 첫 버전이다. pyproject의 정확 고정(==)과는 별개라 고정을
올리는 건 자유지만 이 선 아래로는 내려가면 안 된다. transformers 4.57.1·torch
2.10.0에 남은 권고는 수용한 잔여 위험이다 — 근거는 backend/pyproject.toml의 고정
옆 주석에 있다.
"""

from __future__ import annotations

import ast
from importlib.metadata import version as dist_version
from pathlib import Path

import pytest
from packaging.version import Version

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# CVE-2026-3308 수정 커밋(a26f0142)이 처음 들어간 MuPDF 릴리스
MUPDF_FIXED = "1.28.0"

# (배포 이름, 하한, 근거 권고) — 하한 = 권고가 고쳐진 첫 버전
SECURITY_FLOORS = [
    ("pymupdf", MUPDF_FIXED, "CVE-2026-3308 — MuPDF 1.28.0을 동봉한 첫 PyMuPDF"),
    ("pillow", "12.3.0", "crop/paste 좌표 오버플로 OOB 쓰기 등 12.2.0·12.3.0 HIGH 권고"),
    ("urllib3", "2.8.0", "CVE-2026-97689(chunked 무제한 버퍼링)·CVE-2026-97688·CVE-2026-97687"),
    ("anyio", "4.14.2", "CVE-2026-63374·CVE-2026-64847·CVE-2026-63349"),
]


@pytest.mark.parametrize(
    ("dist", "floor", "why"), SECURITY_FLOORS, ids=[row[0] for row in SECURITY_FLOORS]
)
def test_installed_version_meets_security_floor(dist, floor, why):
    installed = Version(dist_version(dist))
    assert installed >= Version(floor), (
        f"{dist} {installed} < 보안 하한 {floor} ({why}) — uv.lock이 취약 버전으로 내려갔다"
    )


def test_bundled_mupdf_has_the_cve_2026_3308_fix():
    """PyMuPDF가 실제로 링크한 MuPDF 버전을 본다 — 시스템 MuPDF로 빌드한 PyMuPDF처럼
    패키지 버전과 동봉 버전이 어긋나는 경우까지 잡는다."""
    import pymupdf

    bundled = Version(pymupdf.mupdf_version)
    assert bundled >= Version(MUPDF_FIXED), (
        f"MuPDF {bundled}에는 CVE-2026-3308 수정(a26f0142, {MUPDF_FIXED}부터)이 없다 — "
        "업로드 PDF의 이미지 XObject 하나로 렌더 중 힙 OOB 쓰기가 난다"
    )


def test_app_code_does_not_import_the_legacy_fitz_module():
    """PyMuPDF 1.28+는 `import fitz` 때 stdout에 폐지 경고를 찍고(서버 콘솔 오염)
    향후 이 모듈을 없앤다. app 코드는 같은 객체인 `import pymupdf as fitz`를 쓴다."""
    offenders = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(name == "fitz" or name.startswith("fitz.") for name in names):
                offenders.append(f"{path.relative_to(APP_DIR.parent)}:{node.lineno}")
    assert not offenders, f"레거시 fitz 임포트: {offenders} — `import pymupdf as fitz`로 바꾼다"
