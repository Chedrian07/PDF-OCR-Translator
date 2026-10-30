"""디바이스/엔진 선택의 단일 진입점. 계약: docs/ARCHITECTURE.md §6"""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import sys

from ..config import Settings
from .base import OCREngine

logger = logging.getLogger(__name__)

# auto = unlimited 엔진이 쓸 수 있는 가장 빠른 디바이스(resolve_auto_device). mps는 config가
# metal로 바꾼다. mlx = Apple Silicon in-process MLX 엔진(unlimited_mlx.py) — metal(torch MPS)은 폴백.
VALID_DEVICES = ("auto", "cpu", "cuda", "metal", "mlx")
VALID_ENGINES = ("unlimited2", "fake")


def _torch_accelerators() -> tuple[bool, bool, str]:
    """(CUDA 가용, MPS 가용, 메모) — torch가 없거나 조회가 실패하면 없는 것으로 본다."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001 — 휠 로드 실패(OSError 등)도 '없음'으로
        return False, False, f"torch 임포트 실패({type(e).__name__})"
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — 드라이버 조회 실패 = 없음
        cuda = False
    try:
        mps = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        mps = False
    return cuda, mps, ""


def resolve_auto_device() -> str:
    """OCR_DEVICE=auto(unlimited 엔진) → mlx → cuda → metal → cpu 중 처음 쓸 수 있는 것.

    Apple Silicon의 기본은 MLX다(감사 MLX-01: 8쪽 청크 MPS 34 s/쪽 → MLX 3.9 s/쪽, 재현율
    동일). mlx 판정은 플랫폼을 먼저 봐서 Linux·Docker에서는 mlx를 임포트하지 않고, 그다음
    torch로 CUDA·MPS를 본다 — CPU 이미지(torch CPU 휠)는 둘 다 없어 cpu가 된다. compose는
    서비스마다 OCR_DEVICE를 명시하므로 auto를 거치지 않는다. 결정과 사유는 INFO로 남긴다.
    """
    from .unlimited_mlx import mlx_unavailable_reason

    why = mlx_unavailable_reason()
    if why is None:
        logger.info("OCR_DEVICE=auto → mlx (Apple Silicon in-process MLX 엔진)")
        return "mlx"
    cuda, mps, note = _torch_accelerators()
    device = "cuda" if cuda else "metal" if mps else "cpu"
    logger.info(
        "OCR_DEVICE=auto → %s (mlx 불가: %s; CUDA %s; MPS %s%s)",
        device, why, "있음" if cuda else "없음", "있음" if mps else "없음",
        f"; {note}" if note else "",
    )
    return device


def build_engine(settings: Settings) -> OCREngine:
    device = settings.device
    if device not in VALID_DEVICES:
        raise ValueError(f"알 수 없는 OCR_DEVICE: {device!r} (사용 가능: {', '.join(VALID_DEVICES)})")

    if settings.engine == "fake":
        if os.environ.get("OCR_TEST_MODE") != "1":
            raise ValueError("OCR_ENGINE=fake is test-only; set OCR_TEST_MODE=1 in the test harness")
        from .fake import FakeEngine

        # 가짜 엔진은 하드웨어를 쓰지 않는다 — auto는 탐지 없이 cpu로 표기한다
        return FakeEngine(device="cpu" if device == "auto" else device, delay=settings.fake_delay)

    if settings.engine != "unlimited2":
        raise ValueError(
            f"OCR_ENGINE={settings.engine!r} is retired. Use OCR_ENGINE=unlimited2 "
            "with baidu/Unlimited-OCR-2. No legacy OCR fallback is available."
        )
    from ..config import _DEFAULT_REVISION
    if settings.model_id != "baidu/Unlimited-OCR-2" or settings.model_revision != _DEFAULT_REVISION:
        raise ValueError("Use MODEL_ID=baidu/Unlimited-OCR-2 and MODEL_REVISION=" + _DEFAULT_REVISION)

    if device == "cpu" and sys.platform == "darwin" and platform.machine() == "arm64":
        # 명시 선택이어도 알린다 (감사 api-jobs-5) — .env.example 줄을 그대로 풀면 cpu가 된다
        logger.warning(
            "Apple Silicon에서 OCR_DEVICE=cpu — Unlimited-OCR를 CPU(fp32)로 돌려 MLX보다 수십 배 "
            "느립니다. OCR_DEVICE를 비우면(auto) MLX를, 없으면 torch MPS를 씁니다"
        )
    if device == "auto":
        device = resolve_auto_device()
        # 엔진(과 health의 device)은 실제 디바이스를 본다 — 호출자의 settings는 그대로 둔다
        settings = dataclasses.replace(settings, device=device)

    if device == "cuda":
        from .sglang import SGLangEngine
        return SGLangEngine(settings)

    if device == "mlx":
        # mlx 자체는 load()에서야 임포트한다 — 생성만으로는 mlx가 필요 없다(torch 엔진과 같은 규칙)
        from .unlimited_mlx import UnlimitedMLXEngine

        return UnlimitedMLXEngine(settings)

    # torch는 무거우므로 여기서야 임포트
    from .unlimited import UnlimitedEngine

    return UnlimitedEngine(settings)
