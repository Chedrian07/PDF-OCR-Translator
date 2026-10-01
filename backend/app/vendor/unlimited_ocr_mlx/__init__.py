"""baidu/Unlimited-OCR의 MLX 포팅 (Apple Silicon in-process 엔진용). 출처·패치: PROVENANCE.md

mlx-vlm 0.7.4(MIT) 모델 코드를 벤더링했다 — mlx-vlm·torch에 의존하지 않는다
(허용 의존: mlx, numpy, PIL, huggingface_hub, transformers 토크나이저).

mlx가 없는 환경(Linux CI·Docker)에서도 패키지와 순수 모듈(config·processing·
postprocess·resample)은 임포트된다. mlx가 필요한 심볼은 처음 접근할 때 지연
임포트한다(PEP 562) — ``mlx_status()``로 사용 가능 여부를 먼저 확인할 것.
"""

from __future__ import annotations

import importlib
import platform
import sys

from .config import IMAGE_TOKEN, IMAGE_TOKEN_ID, STOP_STR, ModelConfig  # noqa: F401
from .postprocess import (  # noqa: F401
    decode_outputs,
    draw_bounding_boxes,
    extract_coordinates_and_label,
    process_image_with_refs,
    re_match,
    save_results_multi,
    save_results_single,
)
from .processing import OCRInputs, prepare_multi, prepare_single  # noqa: F401

# mlx 의존 심볼 → 모듈. 하위 모듈 이름은 내보내는 이름과 겹치면 안 된다 — 하위 모듈이
# 임포트되는 순간 패키지 속성이 그 모듈로 덮여 지연 조회(__getattr__)를 건너뛴다
# (그래서 generate → generation.py, infer → inference.py).
_LAZY = {
    "load": ".loader",
    "build_model": ".loader",
    "load_tokenizer": ".loader",
    "resolve_snapshot": ".loader",
    "quantize_decoder": ".loader",
    "LoadInfo": ".loader",
    "DTYPES": ".loader",
    "SUPPORTED_QUANT_BITS": ".loader",
    "Model": ".model",
    "generate": ".generation",
    "GenerationResult": ".generation",
    "NoRepeatNgramProcessor": ".ngram",
    "infer": ".inference",
    "infer_multi": ".inference",
    "InferResult": ".inference",
    "warmup": ".inference",
}

__all__ = [
    "IMAGE_TOKEN",
    "IMAGE_TOKEN_ID",
    "STOP_STR",
    "ModelConfig",
    "OCRInputs",
    "decode_outputs",
    "draw_bounding_boxes",
    "extract_coordinates_and_label",
    "mlx_status",
    "prepare_multi",
    "prepare_single",
    "process_image_with_refs",
    "re_match",
    "save_results_multi",
    "save_results_single",
    *_LAZY,
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def mlx_status() -> tuple[bool, str]:
    """(사용 가능, 사유). macOS arm64 + mlx 임포트 + Metal 장치가 모두 있어야 True."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False, f"MLX는 Apple Silicon macOS 전용입니다 (현재 {sys.platform}/{platform.machine()})"
    try:
        import mlx.core as mx
    except Exception as e:  # noqa: BLE001 - ImportError 외 휠 로드 실패도 사유로
        return False, f"mlx를 임포트할 수 없습니다 ({type(e).__name__}: {e}) — `uv sync --extra mlx`"
    try:
        if not mx.metal.is_available():
            return False, "Metal 장치를 사용할 수 없습니다"
    except Exception as e:  # noqa: BLE001
        return False, f"Metal 상태 확인 실패: {e}"
    return True, f"mlx {mx.__version__}"
