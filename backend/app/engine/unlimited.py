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
import threading
from pathlib import Path

from ..config import Settings
from ..native_ops import make_ngram_logits_processor
from .base import (
    EngineCapabilities,
    EngineError,
    OCREngine,
    OutputLimitError,
    RepetitiveOutputError,
    StreamSink,
)
from .objc_pool import autorelease_pool
from .repetition import SemanticRepetitionDetector

logger = logging.getLogger(__name__)

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


def torch_device_name(device: str) -> str:
    return _TORCH_DEVICES.get(device, device)


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
            logger.warning("이 macOS/torch 조합은 MPS bfloat16을 지원하지 않아 float32로 동작합니다 "
                           "(느림 — OCR_DTYPE=float16 시도 가능)")
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
    name = "unlimited"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.device = settings.device
        self.torch_device = torch_device_name(settings.device)
        self._model = None
        self._tokenizer = None
        self.dtype_name = ""
        if self.device == "metal":
            # MPS 미구현 op를 CPU로 폴백 — torch 첫 임포트 전에 설정돼야 적용된다
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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

        from ..vendor.unlimited_ocr import UnlimitedOCRForCausalLM

        s = self._settings
        if self.device == "cuda" and not torch.cuda.is_available():
            raise EngineError(
                "OCR_DEVICE=cuda 이지만 CUDA를 사용할 수 없습니다. "
                "GPU/드라이버와 컨테이너의 gpus 설정을 확인하세요."
            )
        if self.device == "metal" and not torch.backends.mps.is_available():
            hint = (
                "Apple Silicon Mac + macOS 12.3 이상이 필요합니다."
                if torch.backends.mps.is_built()
                else "설치된 torch가 MPS 없이 빌드되었습니다 — macOS arm64용 휠로 재설치하세요 "
                     "(backend에서 `uv sync --extra metal`). Docker/Linux에서는 Metal을 쓸 수 없습니다."
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
        tokenizer = AutoTokenizer.from_pretrained(s.model_id, revision=s.model_revision)
        # 이 아키텍처는 mha_eager(SlidingWindowLlamaAttention)만 구현 → eager 필수
        model = UnlimitedOCRForCausalLM.from_pretrained(
            s.model_id,
            revision=s.model_revision,
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
        from ..vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

        model = model.eval()
        if self.torch_device in ("cuda", "mps"):
            stacked = prebuild_fused_moe(model, self.torch_device)
            if stacked:
                logger.info("융합 MoE(P17) expert 스택 프리빌드: %d개 레이어", stacked)
        model = model.to(self.torch_device)
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

    def _gen_extras(
        self,
        sink: StreamSink,
        cancel: threading.Event,
        ngram_window: int,
        repetition: SemanticRepetitionDetector,
        outcome: _GenerationOutcome | None = None,
    ) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        tokenizer = self._tokenizer
        eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)

        class _SinkStreamer(TextStreamer):
            def put(self, value) -> None:
                # 첫 put은 프롬프트이며 TextStreamer(skip_prompt=True)가 버린다.
                # 이후 put의 실제 생성 토큰 수는 디코딩/공백 유무와 무관한 hard
                # limit에 사용한다. fast_decode에서는 블록 크기만큼의 overshoot만
                # 가능하고 이 산출물은 전부 폐기된다.
                is_prompt = self.next_tokens_are_prompt
                super().put(value)
                # 먼저 디코딩해야 이 블록 안의 <PAGE>가 문자 감지기의 페이지
                # 상태를 초기화한다. 그 뒤 블록 전체를 새 페이지에 보수적으로
                # 계수하면 경계에서 이전 페이지를 잘못 초과시키거나 토큰을 잃지 않는다.
                if not is_prompt:
                    repetition.feed_tokens(int(value.numel()))

            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                text = text.replace(eos_text, "\n")
                repeated = repetition.feed(text, stream_end=stream_end)
                if text and not repeated:
                    sink.on_text(text)

        class _CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return cancel.is_set() or repetition.detected

        extras: dict = {
            "streamer": _SinkStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
            "stopping_criteria": StoppingCriteriaList([_CancelCriteria()]),
        }
        extras["logits_processor"] = make_ngram_logits_processor(
            NGRAM_SIZE, ngram_window, self.torch_device
        )
        # 디코드 경로는 항상 generate_fn(벤더 P15)으로 주입한다 — fast_decode든 HF generate
        # 폴백(OCR_FAST_DECODE=0)이든 반환 텐서로 "EOS 없이 MAX_LENGTH 도달"을 판정해야
        # 잘린 출력을 정상 결과로 채택하지 않는다(audit decode-correctness-1).
        # 폴백 래퍼는 벤더의 기본 분기와 똑같이 model.generate(**gen_kwargs)를 부른다.
        from .fast_decode import hit_length_limit

        if outcome is None:
            outcome = _GenerationOutcome()
        if self._settings.fast_decode:
            from .fast_decode import fast_greedy_decode

            block = self._settings.decode_block

            def _decode(model, gen_kwargs):
                return fast_greedy_decode(model, gen_kwargs, block=block)
        else:
            def _decode(model, gen_kwargs):
                return model.generate(**gen_kwargs)

        def _generate_fn(model, gen_kwargs):
            output_ids = _decode(model, gen_kwargs)
            outcome.length_capped = hit_length_limit(output_ids, gen_kwargs)
            return output_ids

        extras["generate_fn"] = _generate_fn
        return extras

    # ── OCREngine 구현 ─────────────────────────────────────────

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        s = self._settings
        repetition = SemanticRepetitionDetector(
            max_page_chars=s.max_page_output_chars,
            max_page_tokens=s.max_page_output_tokens,
            expected_pages=len(image_paths),
        )
        outcome = _GenerationOutcome()
        try:
            try:
                # MPS: 이미지 H2D 복사·생성 루프·후처리 전체를 풀로 감싼다 — fast_decode는
                # 스텝마다 따로 비우지만 HF generate 폴백(OCR_FAST_DECODE=0)은 이 바깥
                # 풀이 유일한 회수 지점이다.
                with autorelease_pool(self._uses_mps):
                    outputs, _tokens = self._model.infer_multi(
                        self._tokenizer,
                        prompt=MULTI_PROMPT,
                        image_files=[str(p) for p in image_paths],
                        output_path=str(out_dir),
                        image_size=1024,
                        max_length=s.max_length,
                        no_repeat_ngram_size=NGRAM_SIZE,
                        ngram_window=MULTI_NGRAM_WINDOW,
                        save_results=True,
                        **self._gen_extras(
                            sink, cancel, MULTI_NGRAM_WINDOW, repetition, outcome
                        ),
                    )
            except Exception as exc:
                if repetition.detected and not cancel.is_set():
                    raise RepetitiveOutputError(repetition.message) from exc
                raise
            if repetition.detected and not cancel.is_set():
                raise RepetitiveOutputError(repetition.message)
            if outcome.length_capped and not cancel.is_set():
                # 스트리머는 디코드 끝에서 이미 flush됐다. 잘린 청크(꼬리 페이지 손실)를
                # 정상 결과로 넘기지 않고 runner의 페이지별 복구 경로를 태운다.
                raise OutputLimitError(_length_limit_message(s.max_length))
        finally:
            self._release_device_cache()
        # 취소 시에도 부분 출력을 반환한다 — 병합 후 취소 처리는 runner 몫
        return outputs

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        s = self._settings
        repetition = SemanticRepetitionDetector(
            max_page_chars=s.max_page_output_chars,
            max_page_tokens=s.max_page_output_tokens,
            expected_pages=1,
        )
        outcome = _GenerationOutcome()
        try:
            try:
                with autorelease_pool(self._uses_mps):  # run_multi와 같은 이유
                    outputs = self._model.infer(
                        self._tokenizer,
                        prompt=SINGLE_PROMPT,
                        image_file=str(image_path),
                        output_path=str(out_dir),
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        max_length=s.max_length,
                        no_repeat_ngram_size=NGRAM_SIZE,
                        ngram_window=SINGLE_NGRAM_WINDOW,
                        save_results=True,
                        **self._gen_extras(
                            sink, cancel, SINGLE_NGRAM_WINDOW, repetition, outcome
                        ),
                    )
            except Exception as exc:
                if repetition.detected and not cancel.is_set():
                    raise RepetitiveOutputError(repetition.message) from exc
                raise
            if repetition.detected and not cancel.is_set():
                raise RepetitiveOutputError(repetition.message)
            if outcome.length_capped and not cancel.is_set():
                raise OutputLimitError(_length_limit_message(s.max_length))
        finally:
            self._release_device_cache()
        return outputs or ""
