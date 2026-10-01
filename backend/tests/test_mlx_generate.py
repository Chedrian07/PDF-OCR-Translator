"""MLX 생성 루프(generate.py) 계약 — 스크립트된 가짜 LM으로 검증.

torch 경로와 같은 의미: EOS는 출력에 포함하고 멈춘다, max_length는 프롬프트+생성 총
길이 상한이며 거기서 끊기면 hit_max_length, should_stop은 토큰마다 확인한다, ngram
금지는 프롬프트+생성 시퀀스 기준이다. async_eval 파이프라이닝 때문에 멈출 때 계산이
버려지는 스텝은 최대 1개여야 하고, 길이 상한을 넘는 스텝은 아예 띄우지 않아야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.native_ops import banned_ngram_tokens_py
from app.vendor.unlimited_ocr_mlx import mlx_status

_OK, _WHY = mlx_status()
if not _OK:
    pytest.skip(_WHY, allow_module_level=True)

import mlx.core as mx  # noqa: E402

from app.vendor.unlimited_ocr_mlx.generate import generate  # noqa: E402
from app.vendor.unlimited_ocr_mlx.processing import OCRInputs  # noqa: E402

V = 16


class ScriptedLM:
    """호출 k번째(0 = 프리필)에 script[k]가 최댓값인 로짓을 낸다 (지연 mx 배열)."""

    def __init__(self, script=None, preference=None):
        self.script = list(script or [])
        self.preference = preference  # 고정 선호 순서 (ngram 테스트용)
        self.calls = 0
        self.decode_inputs: list[mx.array] = []

    def __call__(self, inputs, inputs_embeds=None, cache=None, last_only=False):
        k = self.calls
        self.calls += 1
        if inputs is not None:
            self.decode_inputs.append(inputs)
        if self.preference is not None:
            ranks = np.empty(V, dtype=np.float32)
            ranks[self.preference] = np.arange(len(self.preference), 0, -1, dtype=np.float32)
            return mx.array(ranks)[None, None, :]
        tok = self.script[k] if k < len(self.script) else 15
        return (mx.arange(V) == tok).astype(mx.float32)[None, None, :] * 10.0


class FakeModel:
    def __init__(self, lm):
        self.language_model = lm
        self.encoded = 0

    def make_cache(self):
        return []

    def encode_images(self, global_views, crops=None, spatial_crop=(1, 1)):
        self.encoded += 1
        return mx.zeros((1, 4))

    def get_input_embeddings(self, input_ids, images_seq_mask, image_features):
        return mx.zeros((1, len(input_ids), 4))


def _inputs(prompt) -> OCRInputs:
    ids = np.asarray(prompt, dtype=np.int64)
    return OCRInputs(
        input_ids=ids,
        images_seq_mask=np.zeros(len(ids), bool),
        global_views=np.zeros((1, 4, 4, 3), np.float32),
        crops=None,
        images_spatial_crop=[[1, 1]],
        images=[],
    )


def _decode_calls(lm: ScriptedLM) -> int:
    return lm.calls - 1  # 프리필 1회 제외


def test_eos_is_emitted_included_and_stops_with_at_most_one_extra_step():
    lm = ScriptedLM([5, 6, 7, 1, 8, 9])
    seen = []
    res = generate(FakeModel(lm), _inputs([0, 2, 3]), max_length=100, on_token=seen.append)
    assert res.token_ids == [5, 6, 7, 1]
    assert seen == [5, 6, 7, 1]
    assert res.finish_reason == "eos" and not res.hit_max_length
    assert res.prompt_length == 3
    # 파이프라이닝: 토큰 4개를 내는 동안 디코드 스텝은 최대 4번(EOS 뒤 1개는 버림)
    assert _decode_calls(lm) <= len(res.token_ids)
    # 디코드 입력은 직전 토큰 (1,1)
    assert [int(x.item()) for x in lm.decode_inputs] == [5, 6, 7, 1][: len(lm.decode_inputs)]


def test_max_length_is_total_length_and_sets_the_flag():
    lm = ScriptedLM([5, 6, 7, 8, 9])
    res = generate(FakeModel(lm), _inputs([0, 2, 3, 4, 2]), max_length=8)
    assert res.token_ids == [5, 6, 7]  # 5 + 3 == 8
    assert res.finish_reason == "length" and res.hit_max_length
    # 상한에 닿는 스텝은 띄우지 않는다 — 버리는 계산 없음
    assert _decode_calls(lm) == len(res.token_ids) - 1


def test_eos_on_the_last_allowed_position_counts_as_eos():
    lm = ScriptedLM([5, 6, 1])
    res = generate(FakeModel(lm), _inputs([0, 2]), max_length=5)
    assert res.token_ids == [5, 6, 1]
    assert res.finish_reason == "eos" and not res.hit_max_length


def test_should_stop_is_checked_after_every_token():
    lm = ScriptedLM([5, 6, 7, 8, 9, 10])
    seen: list[int] = []
    res = generate(
        FakeModel(lm), _inputs([0, 2]), max_length=100,
        on_token=seen.append, should_stop=lambda: len(seen) >= 3,
    )
    assert res.token_ids == [5, 6, 7] == seen
    assert res.finish_reason == "stop" and res.stopped and not res.hit_max_length
    assert _decode_calls(lm) <= len(res.token_ids)


def test_prompt_at_or_over_max_length_generates_nothing():
    lm = ScriptedLM([5])
    model = FakeModel(lm)
    res = generate(model, _inputs([0, 2, 3]), max_length=3)
    assert res.token_ids == [] and res.finish_reason == "length" and res.hit_max_length
    assert lm.calls == 0 and model.encoded == 0


def test_callback_errors_propagate():
    def boom(_tok):
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        generate(FakeModel(ScriptedLM([5, 6])), _inputs([0, 2]), max_length=10, on_token=boom)


@pytest.mark.parametrize("n,w", [(2, 8), (3, 6), (2, 0)])
def test_ngram_bans_follow_reference_over_prompt_and_generated(n, w):
    preference = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]
    prompt = [0, 2, 3, 2, 4, 9, 2, 5]
    new = 24
    res = generate(
        FakeModel(ScriptedLM(preference=preference)), _inputs(prompt),
        max_length=len(prompt) + new, eos_token_id=99, no_repeat_ngram_size=n, ngram_window=w,
    )
    seq = list(prompt)
    expected = []
    for _ in range(new):
        banned = set(banned_ngram_tokens_py(seq, n, w if w > 0 else 10**9))
        tok = next(t for t in preference if t not in banned)
        expected.append(tok)
        seq.append(tok)
    assert res.token_ids == expected
    assert len(set(expected)) > 1  # 금지가 실제로 선택을 바꿨다


def test_timings_and_rate_are_reported():
    res = generate(FakeModel(ScriptedLM([5, 6, 7, 1])), _inputs([0, 2]), max_length=50)
    assert res.prefill_seconds >= 0 and res.decode_seconds >= 0
    assert res.decode_tokens_per_s >= 0
