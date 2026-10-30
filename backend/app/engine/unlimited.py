"""baidu/Unlimited-OCR 실엔진 (벤더링 코드 사용, CPU/CUDA/Metal 공용).

공식 사용 파라미터 (모델 README):
- 멀티페이지: prompt='<image>Multi page parsing.', image_size=1024,
  no_repeat_ngram_size=35, ngram_window=1024
- 단일(gundam): prompt='<image>document parsing.', base_size=1024,
  image_size=640, crop_mode=True, ngram_window=128
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
import threading

from ..config import Settings
from .base import (
    EngineCapabilities,
    EngineError,
    OCREngine,
    OutputLimitError,
    RepetitiveOutputError,
)
from .hf_snapshot import pretrained_local_first
from .objc_pool import autorelease_pool
from .repetition import SemanticRepetitionDetector

logger = logging.getLogger(__name__)


def _default_mps_fallback_env() -> bool:
    """macOS에서 PYTORCH_ENABLE_MPS_FALLBACK=1을 기본값으로 둔다 — MPS 미구현 op의 CPU 폴백(안전망).

    torch는 이 env를 라이브러리 로드(첫 ``import torch``) 때 한 번만 읽는다. 엔진 생성 시점은
    늦을 수 있다 — OCR_DEVICE=auto는 registry가 엔진을 만들기 **전에** torch로 CUDA·MPS를
    조회한다(audit torch-3). 그래서 이 모듈 임포트 때 둔다: registry의 auto 판정도
    unlimited_mlx(→ 이 모듈)를 torch 조회보다 먼저 임포트한다. 운영자가 정한 값(0 포함)은
    그대로 둔다. 반환: torch가 이미 이 env 없이 로드돼 기본값이 적용되지 못했는가."""
    if sys.platform != "darwin":
        return False
    missed = "torch" in sys.modules and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return missed


_TORCH_LOADED_WITHOUT_MPS_FALLBACK = _default_mps_fallback_env()

MULTI_PROMPT = "<image>Multi page parsing."
SINGLE_PROMPT = "<image>document parsing."
NGRAM_SIZE = 35
MULTI_NGRAM_WINDOW = 1024
SINGLE_NGRAM_WINDOW = 128

# 사용자 노출 디바이스명(OCR_DEVICE) → torch 디바이스 문자열
_TORCH_DEVICES = {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}

# from_pretrained의 meta-디바이스 초기화(init_empty_weights)는 nn.Module.register_parameter를
# 프로세스 전역으로 몽키패치한다 — 동시 호출 시 다른 스레드가 로드한 가중치가 meta로 되돌아가
# .to(device)가 "Cannot copy out of meta tensor"로 실패. 프로세스 전체에서 직렬화해야 안전하다.
_LOAD_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


def torch_device_name(device: str) -> str:
    return _TORCH_DEVICES.get(device, device)


# 고정된 torch(2.10)의 MPS 백엔드 최소 macOS — 미만이면 is_available()이 False다
# (libtorch: "The MPS backend is supported on MacOS 14.0+").
_MIN_MACOS_FOR_MPS = "14.0"


def _macos_version() -> str:
    import platform

    return platform.mac_ver()[0] or "알 수 없음"


def _mps_bf16_supported() -> bool:
    # bf16은 macOS 14+ 에서만 지원 — 실제 할당으로 프로브 (torch 버전별 API 차이 회피)
    import torch

    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except Exception:
        return False


def _cuda_bf16_supported() -> bool:
    # bf16 텐서코어는 Ampere(sm_80) 이상 — pre-Ampere GPU에서 bf16을 쓰면 커널이
    # 없어 실패하거나 에뮬레이션으로 극단적으로 느려진다. MPS와 동일하게 프로브한다.
    import torch

    try:
        try:
            # 에뮬레이션은 '지원'으로 세지 않는다 (구버전 torch엔 이 인자가 없다)
            return bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            return bool(torch.cuda.is_bf16_supported())
    except Exception:  # 드라이버 부재·조회 실패 — 보수적으로 미지원 취급
        return False


@functools.lru_cache(maxsize=1)
def _apple_chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return out or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


class _GenerationOutcome:
    """한 번의 infer/infer_multi 생성 결과 플래그 — generate_fn 래퍼가 채운다."""

    __slots__ = ("length_capped",)

    def __init__(self) -> None:
        # EOS 없이 MAX_LENGTH(총 길이 상한)에 닿아 끝났는가 = 출력이 잘림
        self.length_capped = False


def _length_limit_message(max_length: int) -> str:
    return (
        f"생성 길이 상한(MAX_LENGTH={max_length:,}토큰)에 도달해 EOS 없이 출력이 잘렸습니다"
    )


def _resolve_dtype(device: str, dtype_name: str):
    import torch

    if dtype_name == "auto":
        if device == "cuda":
            if _cuda_bf16_supported():
                return torch.bfloat16
            logger.warning("이 GPU는 bfloat16을 지원하지 않아 float16으로 동작합니다 "
                           "(pre-Ampere — OCR_DTYPE으로 명시 지정 가능)")
            return torch.float16
        if device == "metal":
            if _mps_bf16_supported():
                return torch.bfloat16
            # 고정된 torch 2.10의 MPS는 macOS 14+에서만 켜지고 거기서는 bf16이 항상 되므로
            # 사실상 도달하지 않는다 — 프로브 자체가 실패한 비정상 상태(드라이버·설치)다.
            logger.warning("MPS bfloat16 프로브가 실패해 float32로 동작합니다 (macOS %s — "
                           "보통 일어나지 않는 상태이니 torch 설치를 확인하세요; 느림, "
                           "OCR_DTYPE=float16으로 바꿀 수 있음)", _macos_version())
            return torch.float32
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"알 수 없는 OCR_DTYPE: {dtype_name!r} (auto|bfloat16|float16|float32)")


class UnlimitedEngine(OCREngine):
    name = "unlimited2"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.device = settings.device
        self.torch_device = torch_device_name(settings.device)
        self._model = None
        self._tokenizer = None
        self.dtype_name = ""
        if self.device == "metal" and _TORCH_LOADED_WITHOUT_MPS_FALLBACK:
            # 기본값은 모듈 임포트 때 둔다(_default_mps_fallback_env) — 그보다 먼저 torch를
            # 올린 호출자(스크립트·도구)에서는 적용되지 않으니 조용히 넘어가지 않는다
            logger.warning(
                "torch가 PYTORCH_ENABLE_MPS_FALLBACK 없이 이 엔진 모듈보다 먼저 임포트돼 MPS 미구현 "
                "op의 CPU 폴백(안전망)이 꺼져 있습니다 — 그런 op를 만나면 청크가 NotImplementedError로 "
                "실패합니다. 프로세스 환경에 PYTORCH_ENABLE_MPS_FALLBACK=1을 두면 임포트 순서와 "
                "무관하게 켜집니다"
            )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def _uses_mps(self) -> bool:
        """MPS 작업 구간을 ObjC 오토릴리스 풀로 감쌀지 — 잡 워커는 끝나지 않는 스레드라
        풀이 없으면 autorelease 객체가 영구히 쌓인다(gap1-metal-real-e2e-1)."""
        return self.torch_device == "mps"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            model_id=self._settings.model_id,
            model_revision=self._settings.model_revision,
            provider="in-process",
            runtime="torch",
            supports_multi_page=True,
            preferred_chunk_size=None,  # settings.pages_per_chunk 사용 (기존 동작)
            stream_granularity="token",
            layout_capability="full",
            figure_capability=True,
        )

    def load(self) -> None:
        # 프리로드 스레드(main)와 워커 스레드(jobs)가 동시에 진입할 수 있다 —
        # 늦게 온 쪽은 _LOAD_LOCK에서 완료를 기다렸다가 로드된 모델을 그대로 재사용.
        if self._model is not None:
            return
        with _LOAD_LOCK:
            if self._model is None:
                # 프리로드가 없으면 워커(끝나지 않는 스레드)가 로드한다 — .to('mps')의
                # ObjC 임시 객체가 그 스레드에 남지 않게 풀로 감싼다 (objc_pool 참조)
                with autorelease_pool(self._uses_mps):
                    self._load_locked()

    def _load_locked(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from ..vendor.unlimited_ocr2 import UnlimitedOCRForCausalLM

        s = self._settings
        if self.device == "cuda" and not torch.cuda.is_available():
            raise EngineError(
                "OCR_DEVICE=cuda 이지만 CUDA를 사용할 수 없습니다. "
                "GPU/드라이버와 컨테이너의 gpus 설정을 확인하세요."
            )
        if self.device == "metal" and not torch.backends.mps.is_available():
            hint = (
                f"Apple Silicon Mac + macOS {_MIN_MACOS_FOR_MPS} 이상이 필요합니다 "
                f"(설치된 torch {torch.__version__.split('+')[0]}의 MPS 요구사항 — "
                f"현재 macOS {_macos_version()})."
                if torch.backends.mps.is_built()
                else "설치된 torch가 MPS 없이 빌드되었습니다 — Apple Silicon Mac의 저장소 루트에서 "
                     "`make setup-mlx`로 다시 설치하세요(`uv sync --extra metal` 단독은 mlx와 C++ "
                     "모듈을 지웁니다). Docker/Linux에서는 Metal을 쓸 수 없습니다."
            )
            raise EngineError(f"OCR_DEVICE=metal 이지만 MPS를 사용할 수 없습니다. {hint}")
        if self.device == "cuda":
            # SAM/CLIP 프리필은 고정 크기 conv — cudnn 오토튠 이득. tf32는 잔여 fp32 matmul용
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.device == "cpu" and s.cpu_threads > 0:
            torch.set_num_threads(s.cpu_threads)
            logger.info("torch CPU 스레드: %d", s.cpu_threads)
        dtype = _resolve_dtype(self.device, s.dtype)
        self.dtype_name = str(dtype).replace("torch.", "")
        logger.info("모델 로딩 시작: %s@%s (device=%s/%s dtype=%s)",
                    s.model_id, s.model_revision[:8], self.device, self.torch_device, self.dtype_name)
        # 고정 커밋 스냅샷은 캐시에서 먼저 읽는다 — 로드마다 Hub에 묻지 않는다(hf_snapshot)
        tokenizer = pretrained_local_first(
            AutoTokenizer.from_pretrained, s.model_id, s.model_revision,
        )
        # 이 아키텍처는 mha_eager(SlidingWindowLlamaAttention)만 구현 → eager 필수
        model = pretrained_local_first(
            UnlimitedOCRForCausalLM.from_pretrained,
            s.model_id,
            s.model_revision,
            dtype=dtype,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model = self._place_model(model)
        # loaded/load()의 락 없는 빠른 경로가 _model 을 기준으로 판단하므로 마지막에 대입
        self._tokenizer = tokenizer
        self._model = model
        logger.info("모델 로딩 완료")

    def _place_model(self, model):
        """eval → (cuda/mps) P17 융합 MoE 스택 프리빌드 → 디바이스 이동 → MPS 캐시 반환.

        스택은 이동 **전**(가중치가 CPU에 있을 때) 만든다 — expert별 디바이스 버퍼가 생기지
        않고, 이후 .to()는 이미 디바이스에 있는 스택 뷰를 그대로 둔다(메모리 중복 0).
        첫 디코드 때 디바이스에서 지연 재스택하던 구 동작은 MPS 힙 단편화로 드라이버
        메모리 +1.7~3.3GB가 남고 첫 실행 피크가 15~16GB였다(audit MPS-2).
        OCR_MOE_FUSED=0이면 프리빌드하지 않는다(legacy 개별 가중치 그대로)."""
        model = model.eval().to(self.torch_device)
        self._release_device_cache()
        return model

    def gpu_name(self) -> str | None:
        if self.device == "metal":
            return _apple_chip_name()
        if self.device != "cuda":
            return None
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - 방어적
            return None

    # ── 내부 ───────────────────────────────────────────────────

    def _release_device_cache(self) -> None:
        """청크 사이 MPS 캐시 반환 — 유니파이드 메모리라 장문서 잡의 시스템 메모리 압박을 줄인다."""
        if self.torch_device != "mps":
            return
        try:
            import torch

            with autorelease_pool(True):
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover - 방어적
            pass

    def _run(self, image_paths, out_dir, sink, cancel, *, single):
        import torch
        from PIL import Image
        from transformers import TextStreamer
        from .base import JobCanceled
        from .postprocess import BlockStream, save_results_multi, save_results_single, _split_multi_pages

        self.load()
        sink = BlockStream(sink)
        s = self._settings
        repetition = SemanticRepetitionDetector(
            max_page_chars=s.max_page_output_chars,
            max_page_tokens=s.max_page_output_tokens,
            expected_pages=len(image_paths),
        )

        class SinkStreamer(TextStreamer):
            def put(self, value):
                super().put(value)
                repetition.feed_tokens(int(value.numel()))

            def on_finalized_text(self, text, stream_end=False):
                if not repetition.feed(text, stream_end=stream_end) and text:
                    sink.on_text(text)

        streamer = SinkStreamer(self._tokenizer, skip_prompt=False, skip_special_tokens=False)
        try:
            # Upstream switches DASH routing on the model. Serialize requests, and
            # always let its finally restore those switches before releasing the lock.
            with _RUN_LOCK, autorelease_pool(self._uses_mps), torch.inference_mode():
                if cancel.is_set():
                    raise JobCanceled()
                res = self._model.infer(
                    self._tokenizer,
                    image_file=[str(p) for p in image_paths],
                    prompt="Multi page parsing.",
                    crop_mode=single,
                    base_size=1024, image_size=640,
                    max_length=s.max_length,
                    no_repeat_ngram_size=NGRAM_SIZE,
                    ngram_window=SINGLE_NGRAM_WINDOW if single else MULTI_NGRAM_WINDOW,
                    max_page_tokens=s.max_page_output_tokens or 20000,
                    return_dict=True, eval_mode=True,
                    streamer=streamer,
                    should_stop=lambda: cancel.is_set() or repetition.detected,
                )
            sink.end()
            if cancel.is_set():
                raise JobCanceled()
            if repetition.detected:
                raise RepetitiveOutputError(repetition.message)
            images = []
            try:
                for path in image_paths:
                    with Image.open(path) as img:
                        images.append(img.convert("RGB"))
                if single:
                    pages = _split_multi_pages(res["text"])
                    output = save_results_single("\n".join(pages), images[0], str(out_dir))
                else:
                    output = save_results_multi(res["text"], images, str(out_dir))
            finally:
                for img in images:
                    img.close()
            if res["finish_reason"] == "length":
                raise OutputLimitError(_length_limit_message(s.max_length),
                                       partial_output=None if single else output)
            return output
        except InterruptedError as exc:
            if cancel.is_set():
                raise JobCanceled() from exc
            if repetition.detected:
                raise RepetitiveOutputError(repetition.message) from exc
            raise
        finally:
            self._release_device_cache()

    def run_multi(self, image_paths, out_dir, sink, cancel):
        return self._run(image_paths, out_dir, sink, cancel, single=False)

    def run_single(self, image_path, out_dir, sink, cancel):
        return self._run([image_path], out_dir, sink, cancel, single=True)
