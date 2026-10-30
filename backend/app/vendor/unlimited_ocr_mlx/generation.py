# Adapted from mlx-vlm 0.7.4 (MIT): mlx_vlm/generate/ar.py generate_step — mlx-lm식
# async_eval 파이프라이닝만 남긴 그리디 전용 루프. [local patch M2] 원샷 프리필,
# [local patch M3] ngram·콜백·총 길이 상한. 출처·패치 내역: PROVENANCE.md
# 모듈 이름은 generation — 패키지가 내보내는 generate 함수와 같은 이름의 하위 모듈이
# 임포트되면 패키지 속성이 모듈로 덮여 ``from ... import generate``가 모듈을 돌려준다.
"""그리디 생성 루프 — torch infer/infer_multi의 generate(...)·fast_decode 자리.

계약 (torch 경로와 같은 의미):
- 그리디 argmax, EOS(tokenizer eos=1)를 만나면 그 토큰까지 포함해 끝낸다(HF와 동일).
- ``max_length``는 프롬프트 + 생성 **총 길이** 상한(HF max_length). 상한에서 EOS 없이
  끊기면 ``finish_reason == "length"`` / ``hit_max_length`` — 출력이 잘린 것이다
  (엔진은 OutputLimitError로 올려 runner의 페이지 단위 복구를 태운다).
- no-repeat-ngram: n, 창 w는 프롬프트+생성 시퀀스 기준(ngram.py).
- ``on_token(id)``: 생성 토큰마다 순서대로(EOS 포함) 호출 — 스트리밍용.
- ``should_stop()``: 토큰마다 호출, True면 그 토큰까지 내고 멈춘다(취소·반복 감지).

파이프라이닝: 토큰 k의 값을 호스트로 읽기(``.item()``) 전에 토큰 k+1 그래프를 만들어
``mx.async_eval``로 띄운다 — GPU가 쉬지 않는다. 멈출 때(EOS·stop) 이미 띄운 다음
스텝 1개는 버린다(방출하지 않음, 계산 낭비 최대 1토큰). 길이 상한에 닿는 스텝은 아예
띄우지 않는다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx

from .model import Model
from .processing import OCRInputs

FINISH_EOS = "eos"
FINISH_LENGTH = "length"
FINISH_STOP = "stop"


@dataclass
class GenerationResult:
    # 생성 토큰 — EOS로 끝났으면 EOS 포함 (torch output_ids[0, P:]와 같은 내용)
    token_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0
    finish_reason: str = FINISH_LENGTH  # "eos" | "length" | "stop"
    prefill_seconds: float = 0.0  # 비전 인코딩 + 프리필 + 첫 토큰까지 (TTFT)
    decode_seconds: float = 0.0  # 첫 토큰 이후 마지막 토큰까지

    @property
    def hit_max_length(self) -> bool:
        """MAX_LENGTH(총 길이)에 닿아 EOS 없이 끊김 — 출력이 잘렸다."""
        return self.finish_reason == FINISH_LENGTH

    @property
    def stopped(self) -> bool:
        return self.finish_reason == FINISH_STOP

    @property
    def decode_tokens_per_s(self) -> float:
        n = len(self.token_ids)
        return (n - 1) / self.decode_seconds if n > 1 and self.decode_seconds > 0 else 0.0


def generate(
    model: Model,
    inputs: OCRInputs,
    *,
    max_length: int,
    eos_token_id: int = 1,
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
    on_token: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> GenerationResult:
    """프롬프트 1개(멀티페이지 청크 또는 단일 이미지)를 끝까지 생성한다.

    스레드: 호출 스레드의 기본 스트림에서 돈다(전역 스트림을 만들지 않음). 모델 가중치는
    읽기 전용이지만 같은 모델로 동시 생성은 GPU를 나눠 쓸 뿐이라 호출자가 직렬화한다."""
    prompt_len = inputs.prompt_length
    result = GenerationResult(prompt_length=prompt_len)
    if prompt_len >= max_length:
        # torch fast_decode와 같이 생성 없이 반환 — 프롬프트만으로 상한 소진(잘림)
        result.finish_reason = FINISH_LENGTH
        return result

    lm = model.language_model
    cache = model.make_cache()
    for c in cache:
        c.prepare(inputs.input_ids)
    expected_pages = int((inputs.input_ids == 128827).sum())
    tokens = result.token_ids
    t0 = time.perf_counter()
    feats = model.encode_images(inputs.global_views, inputs.crops, inputs.spatial_crop)
    embeds = model.get_input_embeddings(inputs.input_ids, inputs.images_seq_mask, feats)
    logits = lm(None, inputs_embeds=embeds, cache=cache, last_only=True)[:, -1, :]
    del feats, embeds
    t_first = None
    finish = FINISH_LENGTH
    marker_count = 0
    try:
        while prompt_len + len(tokens) < max_length:
            if should_stop is not None and should_stop():
                finish = FINISH_STOP
                break
            n = no_repeat_ngram_size
            if n > 0 and len(tokens) >= n:
                prefix = tuple(tokens[-(n-1):]) if n > 1 else ()
                start = max(0,len(tokens)-ngram_window) if ngram_window > 0 else 0
                banned = {tokens[i+n-1] for i in range(start,len(tokens)-n+1)
                          if tuple(tokens[i:i+n-1]) == prefix}
                for token in banned:
                    logits[:,token] = -mx.inf
            if marker_count > expected_pages:
                token = eos_token_id
            elif marker_count < expected_pages:
                logits[:,eos_token_id] = -mx.inf
                token = int(mx.argmax(logits,axis=-1).item())
            else:
                token = int(mx.argmax(logits,axis=-1).item())
            if t_first is None:
                t_first = time.perf_counter()
            tokens.append(token)
            marker_count += int(token == 128827)
            if on_token is not None:
                on_token(token)
            if token == eos_token_id:
                finish = FINISH_EOS
                break
            if prompt_len+len(tokens) >= max_length:
                break
            for c in cache:
                c.prepare([token])
            logits = lm(mx.array([[token]],dtype=mx.int32),cache=cache)[:, -1, :]
            mx.eval(logits)
            if len(tokens)%256 == 0:
                mx.clear_cache()
    finally:
        cache.clear()
        mx.clear_cache()
    t_end = time.perf_counter()
    result.finish_reason = finish
    result.prefill_seconds = (t_first or t_end) - t0
    result.decode_seconds = t_end - (t_first or t_end)
    return result
