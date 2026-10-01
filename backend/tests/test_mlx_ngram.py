"""MLX no-repeat-ngram 프로세서 — 앱 레퍼런스(native_ops.banned_ngram_tokens_py) 패리티.

레퍼런스 의미론: seq = 프롬프트 + 생성 토큰 전체, 마지막 w개 안에서 현재 (n-1)-그램
다음에 왔던 토큰 금지. MLX 구현은 마지막 w개만 디바이스에 두고(윈도 동치) 지연 평가로
마스크를 만든다. mlx(Apple Silicon)가 없으면 모듈 전체를 건너뛴다.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.native_ops import banned_ngram_tokens_py as reference
from app.vendor.unlimited_ocr_mlx import mlx_status

_OK, _WHY = mlx_status()
if not _OK:
    pytest.skip(_WHY, allow_module_level=True)

import mlx.core as mx  # noqa: E402

from app.vendor.unlimited_ocr_mlx.ngram import (  # noqa: E402
    NoRepeatNgramProcessor,
    banned_mask,
    banned_tokens,
)

UNBOUNDED = 10**9  # window<=0 ↔ 무제한 창 (HF no_repeat_ngram_size)


def _ref(seq, n, w):
    return reference(seq, n, w if w and w > 0 else UNBOUNDED)


EDGE_CASES = [
    ([], 3, 10),
    ([1, 2], 3, 10),  # L < n
    ([1, 2, 3], 3, 10),  # L == n, 반복 없음
    ([1, 2, 1, 2], 3, 10),  # 접두어 (1,2) 다음 1 금지
    ([1, 2, 3, 1, 2, 4, 1, 2], 3, 100),  # 3·4 둘 다 금지
    ([5] * 40, 1, 8),  # n=1: 창 안 전 토큰 금지
    (list(range(10)) * 3, 2, 7),
    ([1, 2, 3, 9, 9, 1, 2], 3, 100),
    ([1, 2, 3] + [9] * 50 + [1, 2], 3, 4),  # 창 밖이면 금지 안 함
    ([7] * 35, 35, 1024),  # L == n
    ([7] * 36, 35, 1024),  # 첫 금지 발생
    ([7] * 100, 35, 34),  # w < n → 없음
    ([7] * 100, 35, 35),  # w == n → 1개 후보
    ([7] * 100, 35, 36),
    ([3, 4] * 600, 35, 1024),  # L > w (창 슬라이드)
    ([3, 4] * 600, 35, 128),  # 단일 모드 창
    ([3, 4] * 20, 3, 0),  # 무제한 창
    (list(range(64)) * 2, 35, None),
]


@pytest.mark.parametrize("seq,n,w", EDGE_CASES)
def test_edge_cases_match_reference(seq, n, w):
    assert banned_tokens(seq, n, w) == _ref(seq, n, w)


def _low_entropy_seq(rng: random.Random, length: int, vocab: int) -> list[int]:
    base = [rng.randrange(vocab) for _ in range(rng.randint(1, 40))]
    return [base[i % len(base)] if rng.random() < 0.9 else rng.randrange(vocab) for i in range(length)]


def test_randomized_static_parity_600_cases():
    rng = random.Random(20261001)
    for _ in range(600):
        n = rng.choice([1, 2, 3, 5, 34, 35, 36])
        w = rng.choice([0, 1, 2, 8, 34, 35, 36, 64, 128, 1024])
        seq = _low_entropy_seq(rng, rng.randint(0, 1500), rng.choice([5, 50, 129280]))
        assert banned_tokens(seq, n, w) == _ref(seq, n, w), (n, w, len(seq))


def test_incremental_state_matches_reference_every_step():
    """generate()가 쓰는 경로: reset(프롬프트) 후 토큰마다 __call__ → append."""
    rng = random.Random(7)
    vocab = 24
    steps = 0
    for _ in range(40):
        n = rng.choice([2, 3, 35])
        w = rng.choice([8, 64, 128, 1024])
        seq = _low_entropy_seq(rng, 700, vocab)
        p = rng.randint(1, 400)
        proc = NoRepeatNgramProcessor(n, w)
        proc.reset(seq[:p])
        for k in range(p, len(seq)):
            out = np.array(proc(mx.zeros((1, vocab)))[0])
            got = np.flatnonzero(np.isneginf(out)).tolist()
            assert got == _ref(seq[:k], n, w), (n, w, p, k)
            proc.append(mx.array([seq[k]], dtype=mx.int32))
            steps += 1
    assert steps >= 400


def test_mask_only_touches_banned_logits_and_keeps_dtype():
    seq = [1, 2, 3, 1, 2, 4, 1, 2]
    proc = NoRepeatNgramProcessor(3, 100)
    proc.reset(seq)
    logits = mx.arange(8, dtype=mx.float32).astype(mx.bfloat16)[None]
    out = proc(logits)
    assert out.dtype == mx.bfloat16
    arr = np.array(out.astype(mx.float32))[0]
    assert np.isneginf(arr[[3, 4]]).all()
    keep = [i for i in range(8) if i not in (3, 4)]
    np.testing.assert_array_equal(arr[keep], np.arange(8, dtype=np.float32)[keep])


def test_history_is_capped_to_window():
    proc = NoRepeatNgramProcessor(3, 16)
    proc.reset(list(range(100)))
    assert proc.history.shape[0] == 16
    for t in range(5):
        proc.append(mx.array([t], dtype=mx.int32))
    assert proc.history.shape[0] == 16
    assert np.array(proc.history).tolist()[-5:] == [0, 1, 2, 3, 4]


def test_banned_mask_none_when_too_short():
    assert banned_mask(mx.array([1, 2], dtype=mx.int32), 3, 10) is None


def test_invalid_ngram_size_rejected():
    with pytest.raises(ValueError):
        NoRepeatNgramProcessor(0, 10)
    with pytest.raises(RuntimeError):
        NoRepeatNgramProcessor(3, 10)(mx.zeros((1, 4)))
