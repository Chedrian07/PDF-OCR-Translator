"""baidu/Unlimited-OCR의 Apple Silicon in-process MLX 엔진 (OCR_DEVICE=mlx · auto의 1순위).

계약은 torch 엔진(unlimited.py)과 같다 — 같은 프롬프트·해상도·no-repeat-ngram(35, 창
1024/128)·MAX_LENGTH, 같은 산출물(result.md·raw_pages.json·boxes.json·images/·
result_with_boxes*.jpg 이름 규약), 같은 capability(멀티페이지 문맥·토큰 스트리밍·full
layout·figure). 모델 코드는 app/vendor/unlimited_ocr_mlx이고, torch 벤더와의 수치·파일
패리티는 그 PROVENANCE.md와 tests/test_mlx_*.py가 고정한다.

- 스트리밍: HF TextStreamer(skip_special_tokens=False)와 같은 텍스트를 같은 단위로
  sink.on_text에 낸다(EOS 문자열은 '\\n' — torch와 동일). TextStreamer가 줄 단위 캐시를
  토큰마다 통째로 다시 디코드하는 O(L²)만 접두 메모(_PrefixMemoDecoder)로 없앤다.
- 반복 감지: SemanticRepetitionDetector에 torch와 같은 순서(텍스트 먼저, 토큰 수 다음)로
  공급하고, 감지되면 생성을 멈춰 RepetitiveOutputError를 던진다.
- 취소·반복 감지는 토큰마다 확인한다 — 멈출 때 버리는 GPU 스텝은 최대 1개(약 3 ms).
- torch와 다른 점: MAX_LENGTH에서 EOS 없이 끊기면 OutputLimitError를 던진다. 잘린 출력은
  마지막 페이지(들)가 조용히 빠진 상태라 채택하지 않고 runner가 페이지 단위로 복구한다.

mlx는 load() 이후에만 임포트한다 — Linux·Docker에서 이 모듈을 임포트해도(registry의
auto 판정 등) mlx·torch·transformers가 올라오지 않는다.
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import Settings
from .base import (
    EngineCapabilities,
    EngineError,
    OCREngine,
    OutputLimitError,
    RepetitiveOutputError,
    StreamSink,
)
from .repetition import SemanticRepetitionDetector
from .unlimited import (
    MULTI_NGRAM_WINDOW,
    MULTI_PROMPT,
    NGRAM_SIZE,
    SINGLE_NGRAM_WINDOW,
    SINGLE_PROMPT,
    _apple_chip_name,
)

logger = logging.getLogger(__name__)

# metal extra를 함께 골라야 한다 — uv sync는 고른 extra만 남기므로 mlx만 고르면 torch(MPS
# 폴백)가, metal만 고르면 mlx가 지워진다.
MLX_INSTALL_HINT = "cd backend && uv sync --extra metal --extra mlx (= make setup-mlx)"

# OCR_DTYPE → MLX 가중치 dtype. auto = bfloat16: Apple GPU가 bf16을 네이티브로 돌리고, 감사
# 스파이크·패리티 측정(8쪽 3.9 s/쪽, 재현율 0.9748)이 bf16 기준이다.
# vendor loader.DTYPES와 같은 이름이어야 한다(tests/test_mlx_engine.py가 대조).
MLX_DTYPES = ("bfloat16", "float16", "float32")
# 0 = 양자화 없음. 8 = vendor loader.SUPPORTED_QUANT_BITS — 4비트는 숫자 오인식(2504→2304) 실측.
MLX_QUANT_BITS = (0, 8)


def mlx_unavailable_reason() -> str | None:
    """MLX 엔진을 쓸 수 없는 이유(사용자 노출 문구) — 쓸 수 있으면 None.

    플랫폼을 먼저 본다 — Apple Silicon이 아니면 mlx를 임포트조차 하지 않는다.
    registry의 auto 판정 로그와 OCR_DEVICE=mlx의 로드 오류가 같은 문구를 쓴다."""
    machine = platform.machine()
    if sys.platform != "darwin" or machine != "arm64":
        return (
            f"MLX는 Apple Silicon(macOS arm64) 로컬 실행 전용입니다 (현재 {sys.platform}/{machine}"
            " — Docker 컨테이너에는 Metal이 없다)"
        )
    try:
        import mlx.core as mx
    except Exception as e:  # noqa: BLE001 — ImportError 외 휠 로드 실패(OSError 등)도 사유로
        return f"mlx를 임포트할 수 없습니다 ({type(e).__name__}: {e}) — `{MLX_INSTALL_HINT}`로 설치하세요"
    try:
        available = bool(mx.metal.is_available())
    except Exception as e:  # noqa: BLE001
        return f"Metal 상태를 확인할 수 없습니다 ({type(e).__name__}: {e})"
    if not available:
        return "Metal GPU를 사용할 수 없습니다 (가상 머신·원격 세션 등)"
    return None


def resolve_mlx_dtype(dtype: str) -> str:
    """OCR_DTYPE → MLX dtype 이름 (잘못된 값은 torch _resolve_dtype과 같은 문구로 거부)."""
    if dtype == "auto":
        return "bfloat16"
    if dtype in MLX_DTYPES:
        return dtype
    raise ValueError(f"알 수 없는 OCR_DTYPE: {dtype!r} (auto|bfloat16|float16|float32)")


# ── 스트리밍 ───────────────────────────────────────────────────


class _PrefixMemoDecoder:
    """TextStreamer의 줄 단위 캐시 디코드를 접두 메모로 증분화한다 — 결과 문자열은 같다.

    TextStreamer는 토큰마다 개행 이후 캐시 전체를 다시 디코드한다. 개행 없는 긴 행(HTML
    표 한 줄, 반복 루프)에서는 O(L²)다 — 실측 6,000토큰 한 줄에 1.56 s(→ 메모 0.02 s).
    바이트 수준 BPE(ByteLevel 디코더, 공백 정리 없음)의 디코드는 바이트를 이어 붙여 한
    번에 UTF-8로 푸는 것이라, 앞부분 바이트가 완결된 문자로 끝나면
    decode(a + b) == decode(a) + decode(b)다. 디코드 결과가 U+FFFD가 아닌 문자로 끝나면
    그 지점이 문자 경계이므로 접두를 고정하고 이후에는 꼬리만 디코드한다(U+FFFD로 끝나면
    다음 바이트가 문자를 완성할 수 있어 고정하지 않는다).

    TextStreamer는 캐시 리스트에 extend만 하고 플러시 때 새 리스트로 바꾼다 — 다른 리스트
    객체가 오거나 짧아지면 메모를 버리고 처음부터 디코드한다(정확성은 항상 유지).
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] | None = None
        self._n = 0
        self._text = ""

    def decode(self, ids, **kwargs) -> str:
        if ids is not self._ids or len(ids) < self._n:
            self._ids, self._n, self._text = ids, 0, ""
        tail = ids[self._n:]
        text = self._text + (self._tokenizer.decode(tail, **kwargs) if tail else "")
        if text and text[-1] != "\ufffd":
            self._n, self._text = len(ids), text
        return text


def _decode_is_char_additive(tokenizer) -> bool:
    """문자 경계에서 디코드가 가법적인 토크나이저인가 — 아니면 메모 없이 전체 디코드한다.

    Unlimited-OCR 토크나이저는 ByteLevel 디코더 + clean_up_tokenization_spaces=False다.
    SentencePiece(Metaspace)처럼 첫 토큰 앞 공백을 지우거나 공백 정리 규칙이 있는
    토크나이저는 조각 디코드가 전체 디코드와 달라지므로 쓰지 않는다."""
    if getattr(tokenizer, "clean_up_tokenization_spaces", True):
        return False
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    return type(decoder).__name__ == "ByteLevel"


def make_sink_streamer(
    tokenizer, sink: StreamSink, repetition: SemanticRepetitionDetector, eos_text: str
):
    """생성 토큰 → sink 텍스트 델타 스트리머 (torch 엔진 _SinkStreamer와 같은 의미).

    on_token 콜백은 생성 토큰만 넘기므로 skip_prompt=False다(torch는 generate가 프롬프트를
    먼저 put해 skip_prompt=True). 토큰마다 put(np.array([id]))로 호출한다."""
    from transformers import TextStreamer  # torch 없이도 임포트된다 (토크나이저와 같은 패키지)

    class _SinkStreamer(TextStreamer):
        def put(self, value) -> None:
            # 먼저 디코딩해야 이 토큰 안의 <PAGE>가 감지기의 페이지 상태를 초기화한다 —
            # 그 뒤 새 페이지에 토큰 수를 센다 (torch _SinkStreamer와 같은 순서).
            super().put(value)
            repetition.feed_tokens(len(value))

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            text = text.replace(eos_text, "\n")
            repeated = repetition.feed(text, stream_end=stream_end)
            if text and not repeated:
                sink.on_text(text)

    decoder = _PrefixMemoDecoder(tokenizer) if _decode_is_char_additive(tokenizer) else tokenizer
    return _SinkStreamer(decoder, skip_prompt=False, skip_special_tokens=False)


# ── 엔진 ───────────────────────────────────────────────────────


class UnlimitedMLXEngine(OCREngine):
    name = "unlimited"

    def __init__(self, settings: Settings) -> None:
        # 설정 오류는 기동 시점(create_app)에 바로 드러낸다 — 모델 로드는 프리로드 몫
        bits = int(settings.mlx_quant_bits or 0)
        if bits not in MLX_QUANT_BITS:
            raise ValueError(
                f"OCR_MLX_QUANT_BITS={bits!r}: 0 또는 8만 지원합니다 "
                "(4비트는 숫자 오인식으로 미지원)"
            )
        self._settings = settings
        self.device = "mlx"
        self._mlx_dtype = resolve_mlx_dtype(settings.dtype)
        self._quant_bits = bits
        # health의 dtype — 양자화는 디코더만이라 기준 dtype에 +q8을 붙인다
        self.dtype_name = f"{self._mlx_dtype}+q8" if bits == 8 else self._mlx_dtype
        self._model = None
        self._tokenizer = None
        self._eos_text = ""
        # 프리로드 스레드(main)와 워커 스레드(jobs)가 load()에 동시에 들어올 수 있다
        self._load_lock = threading.Lock()
        # GPU는 하나다 — 같은 모델로 동시 생성은 처리량 이득 없이 메모리만 늘린다
        self._run_lock = threading.Lock()
        self.last_generation = None  # 직전 실행의 GenerationResult (관측·테스트용)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def capabilities(self) -> EngineCapabilities:
        # torch UnlimitedEngine과 같은 capability — runner의 청크·충실도 게이트 판단이 같다
        return EngineCapabilities(
            model_id=self._settings.model_id,
            model_revision=self._settings.model_revision,
            provider="in-process",
            supports_multi_page=True,
            preferred_chunk_size=None,  # settings.pages_per_chunk 사용
            stream_granularity="token",
            layout_capability="full",
            figure_capability=True,
        )

    def load(self) -> None:
        # 늦게 온 쪽은 락에서 완료를 기다렸다가 로드된 모델을 그대로 쓴다 (멱등)
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                self._load_locked()

    def _load_locked(self) -> None:
        why = mlx_unavailable_reason()
        if why is not None:
            raise EngineError(
                f"OCR_DEVICE=mlx 이지만 MLX를 사용할 수 없습니다: {why}. "
                "OCR_DEVICE=auto(기본)는 쓸 수 있는 디바이스(cuda→metal→cpu)를 고릅니다"
            )
        model, tokenizer = self._load_weights()
        eos_text = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
        self._warmup(model, tokenizer)
        self._tokenizer = tokenizer
        self._eos_text = eos_text
        # loaded/load()의 락 없는 빠른 경로가 _model을 기준으로 판단하므로 마지막에 대입
        self._model = model

    def _load_weights(self):
        """고정 스냅샷 → (MLX 모델, 토크나이저). HF_HOME/HF_HUB_OFFLINE 등은 vendor 로더가 따른다."""
        from ..vendor import unlimited_ocr_mlx as mlx_ocr

        s = self._settings
        logger.info(
            "MLX 모델 로딩 시작: %s@%s (device=mlx dtype=%s)",
            s.model_id, s.model_revision[:8], self.dtype_name,
        )
        model, tokenizer, info = mlx_ocr.load(
            s.model_id,
            s.model_revision,
            dtype=self._mlx_dtype,
            quantize_bits=self._quant_bits or None,
        )
        logger.info(
            "MLX 모델 로딩 완료: 파라미터 %.2f GB, %.1f s (%s)",
            info.parameter_bytes / 1e9, info.load_seconds, info.model_path,
        )
        return model, tokenizer

    def _warmup(self, model, tokenizer) -> None:
        """Metal 커널 JIT를 프리로드에서 치른다 — 없으면 첫 잡의 첫 토큰이 수배 늦다(실측 4배)."""
        from ..vendor.unlimited_ocr_mlx import warmup

        try:
            seconds = warmup(model, tokenizer)
        except Exception:  # noqa: BLE001 — 최적화일 뿐: 실패해도 로드된 모델은 유효하다
            logger.warning("MLX 워밍업 실패 — 첫 잡의 첫 토큰이 느릴 수 있습니다", exc_info=True)
            return
        logger.info("MLX 워밍업 완료: %.2f s", seconds)

    def gpu_name(self) -> str | None:
        return _apple_chip_name() if sys.platform == "darwin" else None

    # ── 내부 ───────────────────────────────────────────────────

    def _release_device_cache(self) -> None:
        """실행 후 MLX 버퍼 캐시 반환 (torch MPS empty_cache 자리) — 유니파이드 메모리라
        장문서 잡의 시스템 메모리 압박을 줄인다. 활성 메모리는 가중치 크기로 돌아간다."""
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - 방어적
            pass

    def _generate(
        self,
        call: Callable,
        sink: StreamSink,
        cancel: threading.Event,
        repetition: SemanticRepetitionDetector,
        label: str,
    ):
        """vendor infer 호출 1회 + 스트리밍·취소·반복 감지·잘림 판정 (run_multi/run_single 공통)."""
        streamer = make_sink_streamer(self._tokenizer, sink, repetition, self._eos_text)

        def on_token(token_id: int) -> None:
            streamer.put(np.array([token_id]))

        def should_stop() -> bool:
            return cancel.is_set() or repetition.detected

        with self._run_lock:
            try:
                try:
                    result = call(on_token, should_stop)
                    # 남은 꼬리 플러시 — 마지막 조각도 감지기를 거친다 (HF generate의 end())
                    streamer.end()
                except Exception as exc:
                    if repetition.detected and not cancel.is_set():
                        raise RepetitiveOutputError(repetition.message) from exc
                    raise
            finally:
                self._release_device_cache()

        gen = result.generation
        self.last_generation = gen
        logger.info(
            "MLX %s: 프롬프트 %d + 생성 %d 토큰, TTFT %.2f s, 디코드 %.1f tok/s, 종료=%s",
            label, gen.prompt_length, len(gen.token_ids), gen.prefill_seconds,
            gen.decode_tokens_per_s, gen.finish_reason,
        )
        if cancel.is_set():
            # 취소 시에도 부분 출력을 반환한다 — 병합 후 취소 처리는 runner 몫 (torch와 동일)
            return result
        if repetition.detected:
            raise RepetitiveOutputError(repetition.message)
        if gen.hit_max_length:
            raise OutputLimitError(
                f"생성이 MAX_LENGTH={self._settings.max_length} 토큰(프롬프트 "
                f"{gen.prompt_length} + 생성 {len(gen.token_ids)})에 닿아 출력이 잘렸습니다"
            )
        return result

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
        from ..vendor.unlimited_ocr_mlx import infer_multi

        def call(on_token, should_stop):
            return infer_multi(
                self._model,
                self._tokenizer,
                prompt=MULTI_PROMPT,
                image_files=[str(p) for p in image_paths],
                output_path=str(out_dir),
                image_size=1024,
                save_results=True,
                max_length=s.max_length,
                no_repeat_ngram_size=NGRAM_SIZE,
                ngram_window=MULTI_NGRAM_WINDOW,
                on_token=on_token,
                should_stop=should_stop,
            )

        result = self._generate(call, sink, cancel, repetition, f"multi {len(image_paths)}쪽")
        return result.text or ""

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
        from ..vendor.unlimited_ocr_mlx import infer

        def call(on_token, should_stop):
            return infer(
                self._model,
                self._tokenizer,
                prompt=SINGLE_PROMPT,
                image_file=str(image_path),
                output_path=str(out_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                max_length=s.max_length,
                no_repeat_ngram_size=NGRAM_SIZE,
                ngram_window=SINGLE_NGRAM_WINDOW,
                on_token=on_token,
                should_stop=should_stop,
            )

        result = self._generate(call, sink, cancel, repetition, "single")
        return result.text or ""
