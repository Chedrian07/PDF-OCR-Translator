# Local module (not in mlx-vlm, which has no no-repeat-ngram) — [local patch M3] 일부.
# 감사 스파이크의 NoRepeatNgramGPU(mlx_bench.py)를 정리한 것. 출처·패치 내역: PROVENANCE.md
"""슬라이딩 윈도 no-repeat-ngram 로짓 프로세서 (MLX 지연 평가, 호스트 동기화 없음).

의미론 = ``app.native_ops.banned_ngram_tokens_py(seq, n, w)`` = 벤더 torch
``SlidingWindowNoRepeatNgramProcessor``: ``seq``는 **프롬프트 + 지금까지 생성한
토큰** 전체이고, 마지막 ``w``개 안에서 현재 (n-1)-그램 접두어 다음에 왔던 토큰을
금지한다. 앱 기본값은 n=35, w=1024(멀티페이지)/128(단일).

윈도 동치: 레퍼런스가 보는 ngram 시작 위치는 [max(0, L-w), L-n]뿐이고 그 ngram과 현재
접두어는 모두 마지막 w개 토큰 안에 있다 — 그래서 마지막 w개만 GPU에 유지해도 결과가
같다(native_ops.py 독스트링의 증명과 동일). ``window<=0``(또는 None)은 무제한 창 —
torch infer의 ``no_repeat_ngram_size``만 준 경우(HF NoRepeatNGram)와 같은 의미다.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def banned_mask(hist: mx.array, ngram_size: int, vocab_size: int) -> mx.array | None:
    """hist(마지막 w개 토큰, int32 [H]) 기준 금지 토큰 마스크 bool [vocab] (없으면 None).

    H는 파이썬 int(모양)라 분기해도 디바이스 동기화가 없다."""
    n = int(ngram_size)
    H = int(hist.shape[0])
    if H < n:
        return None
    m = H - n + 1  # 창 안의 완전한 n-그램 수 (>= 1)
    if n == 1:
        match = mx.ones((m,), dtype=mx.bool_)
    else:
        idx = mx.arange(m)[:, None] + mx.arange(n - 1)[None, :]
        windows = hist[idx]  # [m, n-1] 각 n-그램의 접두어
        suffix = hist[H - (n - 1) :]  # 현재 접두어(마지막 n-1개)
        match = mx.all(windows == suffix[None, :], axis=1)
    nxt = hist[n - 1 :]  # [m] 각 접두어 다음 토큰 = 금지 후보
    # 비매치 후보는 vocab 번째(버림 슬롯)로 보내 분기 없이 스캐터
    slots = mx.where(match, nxt, mx.array(vocab_size, dtype=nxt.dtype))
    counts = mx.zeros((vocab_size + 1,), dtype=mx.int32).at[slots].add(1)
    return (counts > 0)[:vocab_size]


class NoRepeatNgramProcessor:
    """상태 = 마지막 window개 토큰(디바이스 상주). reset → (append → __call__)*."""

    def __init__(self, ngram_size: int, window: int | None):
        if int(ngram_size) < 1:
            raise ValueError("ngram_size는 1 이상이어야 합니다")
        self.ngram_size = int(ngram_size)
        self.window = int(window) if window and int(window) > 0 else None
        self._hist: mx.array | None = None

    def reset(self, prompt_ids) -> None:
        """프롬프트 토큰으로 초기화 (레퍼런스 seq는 프롬프트부터 시작한다)."""
        arr = np.asarray(prompt_ids, dtype=np.int32).reshape(-1)
        if self.window is not None:
            arr = arr[-self.window :]
        self._hist = mx.array(arr)

    def append(self, token: mx.array) -> None:
        """직전 생성 토큰(지연 평가 mx.array, 원소 1개)을 이력에 붙인다 — 동기화 없음."""
        if self._hist is None:
            raise RuntimeError("reset()을 먼저 호출해야 합니다")
        hist = mx.concatenate([self._hist, token.reshape(1).astype(mx.int32)])
        if self.window is not None and hist.shape[0] > self.window:
            hist = hist[-self.window :]
        self._hist = hist

    @property
    def history(self) -> mx.array | None:
        return self._hist

    def __call__(self, logits: mx.array) -> mx.array:
        """logits [..., vocab]에 금지(-inf)를 적용해 돌려준다."""
        if self._hist is None:
            raise RuntimeError("reset()을 먼저 호출해야 합니다")
        ban = banned_mask(self._hist, self.ngram_size, logits.shape[-1])
        if ban is None:
            return logits
        return mx.where(ban, mx.array(-mx.inf, dtype=logits.dtype), logits)


def banned_tokens(sequence, ngram_size: int, window: int | None) -> list[int]:
    """테스트·진단용: 전체 시퀀스에 대한 금지 토큰(정렬) — banned_ngram_tokens_py와 같은 형태."""
    seq = np.asarray(list(sequence), dtype=np.int32)
    if seq.size == 0:
        return []
    proc = NoRepeatNgramProcessor(ngram_size, window)
    proc.reset(seq)
    vocab = int(seq.max()) + 1
    ban = banned_mask(proc.history, proc.ngram_size, vocab)
    if ban is None:
        return []
    return np.flatnonzero(np.array(ban)).tolist()
