"""vLLM 기반 OvisOCR2 로더/추론 — 모델 카드 공식 예제(vLLM 0.22.1)를 따른다.

OpenAI 호환 서버를 별도로 띄우지 않고 sidecar 프로세스가 vLLM Python API로
모델을 직접 로드한다 (불필요한 이중 구조 회피 — docs/OVISOCR2_CUDA_5070TI.md).
"""

from __future__ import annotations

import logging
import threading
from typing import NamedTuple

from .config import OvisConfig
from .lifecycle import PermanentLoadError, is_engine_dead

logger = logging.getLogger(__name__)

# 모델 카드 공식 OCR 프롬프트 (선행 개행 포함, 원문 그대로)
OFFICIAL_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_'
    '{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as "
    "HTML: <table>...</table>. Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)


# 엔진 사망을 나타내는 예외 시그니처 (클래스명/메시지 소문자 부분 일치).
# vLLM 버전에 따라 EngineDeadError가 RuntimeError 등으로 표면화될 수 있어
# 타입 하나에 의존하지 않고, 연속 실패 카운터(_WEDGE_THRESHOLD)를 병행한다.
_ENGINE_DEAD_MARKERS = ("enginedead", "enginecore", "engine core")
_WEDGE_THRESHOLD = 3
_WEDGE_PREFIX = "엔진 비정상: "


class InferOutput(NamedTuple):
    """추론 1회 결과 — finish_reason='length'면 출력 토큰 상한에서 잘린 것이다."""

    text: str
    finish_reason: str | None


class OvisModel:
    """단일 프로세스·단일 모델. 추론은 락으로 직렬화한다 (max_num_seqs=1)."""

    def __init__(self, cfg: OvisConfig) -> None:
        self.cfg = cfg
        self._llm = None
        self._prompt: str | None = None
        self._sampling_cls = None
        self._infer_lock = threading.Lock()
        self._infer_failures = 0  # 연속 추론 실패 수 (성공 시 리셋)
        # load_error: 확정된 실패만(재시도 소진·결정적 실패·웨지 신고) — health status=error.
        # load_retry: 일시적 로드 실패 뒤 재시도 대기 중인 상태(health에 그대로 노출).
        # restart_required: 엔진이 프로세스째 죽어 컨테이너 재시작만 남은 상태.
        self.load_error: str | None = None
        self.load_retry: dict | None = None
        self.restart_required = False

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    @staticmethod
    def _require_cuda() -> None:
        """CUDA를 못 쓰면 명확한 메시지로 실패한다 (CPU 무음 강등 없음 — 5070 Ti 전용)."""
        try:
            import torch
        except ImportError:
            return  # torch 미설치 환경(테스트 등) — vLLM 로드 시점에 걸린다
        if not torch.cuda.is_available():
            # 기다려도 생기지 않는다 — 로드 재시도 대상이 아니다(lifecycle.supervise_load)
            raise PermanentLoadError(
                "CUDA를 사용할 수 없습니다 — OvisOCR2 sidecar는 GPU 전용입니다. "
                "compose의 `gpus: all`과 nvidia-container-toolkit, 호스트 드라이버, "
                "cu129 베이스 이미지를 확인하세요."
            )

    def load(self) -> None:
        """vLLM 엔진 로드 1회 시도 — 실패는 예외로 전파한다.

        재시도·load_error 확정은 호출자(lifecycle.supervise_load) 몫이다 — 여기서
        load_error를 세우면 재시도 대기 중에도 health가 status=error가 되어 backend가
        잡을 즉시 하드 실패시킨다."""
        try:
            # vLLM은 CUDA 없이는 로드가 하드 실패하지만(무음 CPU 강등 없음),
            # RTX 5070 Ti 전용 배포임을 명시하고 진단을 앞당기기 위해 먼저 확인한다.
            self._require_cuda()
            from vllm import LLM, SamplingParams

            cfg = self.cfg
            logger.info(
                "OvisOCR2 로딩: %s@%s (util=%.2f, max_len=%d, gdn=%s)",
                cfg.model_id, cfg.model_revision[:8] or "latest",
                cfg.gpu_memory_utilization, cfg.max_model_len, cfg.gdn_prefill_backend,
            )
            llm = LLM(
                model=cfg.model_id,
                revision=cfg.model_revision or None,
                tokenizer_revision=cfg.model_revision or None,
                tensor_parallel_size=1,
                dtype=cfg.dtype,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.max_model_len,
                max_num_seqs=cfg.max_num_seqs,
                # 컨슈머 Blackwell(sm_120)에서는 triton(FLA) 경로가 동작 경로 —
                # 모델 카드 공식 예제와 동일 인자
                gdn_prefill_backend=cfg.gdn_prefill_backend,
            )
            self._prompt = llm.get_tokenizer().apply_chat_template(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": OFFICIAL_PROMPT},
                    ],
                }],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self._sampling_cls = SamplingParams
            self._llm = llm
            self.load_error = None
            logger.info("OvisOCR2 로딩 완료")
        except Exception:
            logger.warning("OvisOCR2 로딩 시도 실패", exc_info=True)
            raise

    def infer(
        self, image, max_pixels: int | None = None, max_output_tokens: int | None = None
    ) -> InferOutput:
        """PIL 이미지 1장 → raw 마크다운 텍스트 + finish_reason.

        OOM 처리(1회 해상도 강등)·엔진 사망 시 재시작 예약은 호출자 몫이다."""
        if self._llm is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        cfg = self.cfg
        params = self._sampling_cls(
            max_tokens=max_output_tokens or cfg.max_output_tokens, temperature=0.0
        )
        request = {
            "prompt": self._prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "min_pixels": cfg.min_pixels,
                    "max_pixels": max_pixels or cfg.max_pixels,
                }
            },
        }
        with self._infer_lock:
            try:
                outputs = self._llm.generate([request], params, use_tqdm=False)
            except Exception as e:
                self._note_infer_failure(e)
                raise
            self._note_infer_success()
        completion = outputs[0].outputs[0]
        # vLLM CompletionOutput.finish_reason: "stop"(EOS) | "length"(max_tokens 도달)
        return InferOutput(completion.text, getattr(completion, "finish_reason", None))

    def _note_infer_failure(self, e: Exception) -> None:
        """엔진 사망·웨지 감지 — health를 영원히 ok로 두지 않는다.

        vLLM V1의 EngineCore는 별도 프로세스라 하드 CUDA OOM 등으로 죽으면 이후
        모든 generate가 실패한다(자동 재기동 없음). 그런데 uvicorn은 살아 있어
        restart 정책이 발동하지 않았다.
        - 사망 시그니처: 같은 프로세스에서는 회복할 수 없다 — 엔진을 버리고
          restart_required를 세운다. 호출자가 503을 돌려준 뒤 프로세스를 끝내고
          (lifecycle.schedule_restart) backend는 재기동을 기다려 그 페이지를 다시
          보낸다. load_error는 비운다 — status=error+미로드는 backend가 '기다려도
          안 풀리는 로드 실패'로 보고 잡을 즉시 실패시킨다.
        - 연속 실패 임계: 오탐일 수 있어 _llm은 유지하고 load_error(웨지 신고)만
          세운다. 다음 성공이 자동으로 복구한다."""
        self._infer_failures += 1
        if is_engine_dead(e, _ENGINE_DEAD_MARKERS):
            self._llm = None
            self.restart_required = True
            self.load_error = None
            logger.error("vLLM 엔진 사망 판정 (%s: %s) — 컨테이너 재시작이 필요합니다",
                         e.__class__.__name__, str(e)[:200])
            return
        if self._infer_failures >= _WEDGE_THRESHOLD:
            reason = f"연속 {self._infer_failures}회 추론 실패"
            self.load_error = (
                f"{_WEDGE_PREFIX}{reason} — 마지막 오류 {e.__class__.__name__}: {e}"[:500]
            )
            logger.error("vLLM 엔진 비정상 판정 (%s) — health를 error로 전환", reason)

    def _note_infer_success(self) -> None:
        self._infer_failures = 0
        if self.load_error and self.load_error.startswith(_WEDGE_PREFIX):
            self.load_error = None  # 웨지 오탐 자동 복구 (로드 시점 오류는 건드리지 않음)

    @staticmethod
    def release_cache() -> None:
        """OOM 후 CUDA 캐시 반환 (best-effort).

        주의: vLLM V1은 EngineCore를 별도 프로세스로 띄우므로 이 호출은 API
        프로세스의 캐시만 비운다 — 엔진 VRAM에는 닿지 않는 보수적 완화책이다
        (하드 엔진 OOM은 위 _note_infer_failure의 웨지 감지가 표면화한다)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - 방어적
            pass
