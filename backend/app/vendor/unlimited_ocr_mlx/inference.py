# Ported from baidu/Unlimited-OCR modeling_unlimitedocr.py (MIT, Copyright (c) 2026 Baidu;
# 라이선스 전문: ../unlimited_ocr/LICENSE) — UnlimitedOCRForCausalLM.infer()/infer_multi()
# 흐름(전처리 → 생성 → 디코드 → save_results)을 모듈 함수로. 출처·패치 내역: PROVENANCE.md
# 모듈 이름은 inference — 패키지가 내보내는 infer 함수와 이름이 겹치지 않게(generation.py 참조).
"""torch 메서드와 같은 인자·같은 파일 산출물의 MLX 진입점.

torch와 다른 점(의도):
- 반환이 ``InferResult``다 — 엔진이 잘림(``generation.hit_max_length``)·중단 사유를 알아야
  한다(torch는 마크다운 문자열만 돌려줘 MAX_LENGTH 잘림을 알 수 없었다).
- 스트리머/StoppingCriteria/logits_processor 대신 ``on_token``·``should_stop`` 콜백과
  ``no_repeat_ngram_size``/``ngram_window``(GPU 프로세서)를 받는다. 그리디 전용
  (torch temperature 인자 없음 — 앱은 그리디만 쓴다).
- 기본 TPS 출력 스트리머(stdout 출력)가 없다.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from PIL import Image

from .generation import GenerationResult, generate
from .model import Model
from .postprocess import count_output_tokens, decode_outputs, save_results_multi, save_results_single
from .processing import prepare_multi, prepare_single


@dataclass
class InferResult:
    # infer_multi: save_results면 처리된 마크다운, 아니면 디코드 원문(torch 반환 1번째)
    # infer: save_results면 처리된 마크다운, 아니면 None (torch와 동일)
    text: str | None
    # infer_multi의 torch 반환 2번째(len(text_encode(outputs))) — infer는 None
    output_tokens: int | None
    raw_text: str  # 치환 전 디코드 원문 (EOS 문자열 제거·strip)
    generation: GenerationResult


def infer_multi(
    model: Model,
    tokenizer,
    prompt: str = "",
    image_files=None,
    output_path: str = "",
    image_size: int = 640,
    save_results: bool = False,
    max_length: int = 32768,
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
    on_token: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> InferResult:
    """멀티페이지(무크롭) 추론 — torch infer_multi와 같은 인자 기본값."""
    inputs = prepare_multi(tokenizer, prompt, image_files, image_size=image_size)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/images", exist_ok=True)
    gen = generate(
        model,
        inputs,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
        ngram_window=ngram_window,
        on_token=on_token,
        should_stop=should_stop,
    )
    outputs = decode_outputs(tokenizer, gen.token_ids)
    raw = outputs
    output_tokens = count_output_tokens(tokenizer, outputs)
    if save_results:
        outputs = save_results_multi(outputs, inputs.images, output_path)
    return InferResult(text=outputs, output_tokens=output_tokens, raw_text=raw, generation=gen)


def infer(
    model: Model,
    tokenizer,
    prompt: str = "",
    image_file: str = "",
    output_path: str = "",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = False,
    max_length: int = 32768,
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
    on_token: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> InferResult:
    """단일 이미지 추론(gundam: crop_mode=True) — torch infer와 같은 인자 기본값."""
    if not (prompt and image_file):
        raise ValueError("prompt와 image_file이 모두 필요합니다 (텍스트 전용 추론 미지원)")
    inputs = prepare_single(
        tokenizer, prompt, image_file, base_size=base_size, image_size=image_size, crop_mode=crop_mode
    )
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/images", exist_ok=True)
    gen = generate(
        model,
        inputs,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
        ngram_window=ngram_window,
        on_token=on_token,
        should_stop=should_stop,
    )
    outputs = decode_outputs(tokenizer, gen.token_ids)
    text = save_results_single(outputs, inputs.images[0], output_path) if save_results else None
    return InferResult(text=text, output_tokens=None, raw_text=outputs, generation=gen)


def warmup(model: Model, tokenizer, *, max_new_tokens: int = 4) -> float:
    """Metal 커널 JIT를 미리 치른다 — 프리로드 직후 1회 (반환: 걸린 초).

    첫 생성은 커널 컴파일로 TTFT가 수배 길다(실측 1쪽 0.61 s → 이후 0.14 s; 스파이크의
    첫 프로세스 TTFT 1.2 s). 흰 합성 이미지로 두 모드(1024 전역 + 640 타일)를 짧게 돌린다.
    파일은 임시 디렉터리에만 쓰고 지운다. 생성 결과는 버린다."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="uocr-mlx-warmup-") as d:
        page = os.path.join(d, "page.png")
        wide = os.path.join(d, "wide.png")
        Image.new("RGB", (850, 1100), (255, 255, 255)).save(page)
        Image.new("RGB", (1280, 640), (255, 255, 255)).save(wide)  # 2x1 타일 경로
        for inputs in (
            prepare_multi(tokenizer, "<image>Multi page parsing.", [page], image_size=1024),
            prepare_single(tokenizer, "<image>document parsing.", wide),
        ):
            generate(
                model, inputs, max_length=inputs.prompt_length + max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=35, ngram_window=128,
            )
    mx.clear_cache()
    return time.perf_counter() - t0
