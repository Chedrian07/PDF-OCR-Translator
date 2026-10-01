"""MLX 포팅 패키지 표면 — 어느 플랫폼에서나 도는 검사(Linux CI 포함).

엔진(후속 단계)은 ``mlx_status()``로 사용 가능 여부를 먼저 묻고, mlx가 필요한 심볼은
지연 임포트로만 끌어온다. 패키지 임포트 자체는 mlx·torch를 올리지 않아야 한다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import app.vendor.unlimited_ocr_mlx as pkg

BACKEND = Path(__file__).resolve().parents[1]


def test_mlx_status_rejects_non_apple_silicon(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    ok, why = pkg.mlx_status()
    assert ok is False and "Apple Silicon" in why
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(pkg.platform, "machine", lambda: "x86_64")
    ok, why = pkg.mlx_status()
    assert ok is False and "x86_64" in why


def test_mlx_status_reports_import_failure(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(pkg.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    ok, why = pkg.mlx_status()
    assert ok is False and "uv sync --extra mlx" in why


def test_package_import_is_lazy_and_unknown_names_fail():
    code = (
        "import sys\n"
        "import app.vendor.unlimited_ocr_mlx as p\n"
        "assert callable(p.prepare_multi) and callable(p.save_results_multi)\n"
        "print(int('mlx.core' in sys.modules), int('torch' in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["0", "0"]
    with pytest.raises(AttributeError):
        pkg.no_such_symbol  # noqa: B018
    assert {"load", "generate", "infer", "infer_multi", "warmup", "mlx_status"} <= set(pkg.__all__)
