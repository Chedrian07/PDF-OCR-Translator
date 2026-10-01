# Local module (not in mlx-vlm) — [local patch M7] PROVENANCE.md 참조.
"""torch ``F.interpolate``와 같은 가중치를 내는 1-D 리샘플 행렬 (numpy 전용).

torch 레퍼런스(deepencoder.py)는 위치 임베딩을 입력 크기에 맞춰 리샘플한다:

- SAM ``get_abs_pos_sam``·CLIP ``get_abs_pos``: ``mode='bicubic', antialias=True,
  align_corners=False`` (gundam 크롭 640px → SAM 64→40, CLIP 16→10 축소).
- SAM ``get_rel_pos``: ``mode='linear', align_corners=False`` (전역 어텐션 블록의
  상대 위치표 127→79 축소).

mlx-vlm 0.7.4는 여기서 torch와 다르다(CLIP은 채널-우선 텐서에 nn.Upsample을 걸어
축이 뒤섞이고, rel_pos 선형 보간은 ``i*scale`` 좌표라 반-픽셀 중심이 어긋난다) —
1024 멀티페이지 모드는 크기가 같아 리샘플이 없지만 640 크롭(gundam)에서 출력이
갈린다. 두 보간 모두 축별로 분리 가능한 선형 연산이므로, torch CPU 커널
(UpSampleKernel.cpp의 ``_compute_indices_min_size_weights_aa``·``area_pixel_
compute_source_index``)과 같은 float32 산술로 [out, in] 가중치 행렬을 만들고
호출 측이 행렬곱으로 적용한다. 테스트가 torch 결과와 1e-6 수준 일치를 고정한다.
"""

from __future__ import annotations

import functools
import math

import numpy as np

_F = np.float32


def _cubic_aa(x: np.float32) -> np.float32:
    # torch aa_filter (bicubic): Keys 3차 컨볼루션, a = -0.5 (PIL과 동일)
    a = _F(-0.5)
    x = _F(abs(x))
    if x < _F(1.0):
        return _F(((a + _F(2.0)) * x - (a + _F(3.0))) * x * x + _F(1.0))
    if x < _F(2.0):
        return _F((((x - _F(5.0)) * x + _F(8.0)) * x - _F(4.0)) * a)
    return _F(0.0)


@functools.lru_cache(maxsize=32)
def bicubic_aa_matrix(in_size: int, out_size: int) -> np.ndarray:
    """``F.interpolate(mode='bicubic', antialias=True, align_corners=False)`` 1-D 가중치.

    반환: float32 [out_size, in_size] — ``out = W @ in`` (축 하나에 적용).
    """
    if in_size <= 0 or out_size <= 0:
        raise ValueError("in_size/out_size는 1 이상이어야 합니다")
    scale = _F(in_size) / _F(out_size)
    interp_size = 4  # bicubic
    support = _F(interp_size * 0.5) * scale if scale >= _F(1.0) else _F(interp_size * 0.5)
    invscale = _F(1.0) / scale if scale >= _F(1.0) else _F(1.0)
    max_interp = int(math.ceil(float(support))) * 2 + 1
    w = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        center = _F(scale * _F(i + 0.5))
        xmin = max(int(center - support + _F(0.5)), 0)
        xsize = min(int(center + support + _F(0.5)), in_size) - xmin
        xsize = min(max(xsize, 0), max_interp)
        ws = [_cubic_aa(_F((_F(j + xmin) - center + _F(0.5)) * invscale)) for j in range(xsize)]
        total = _F(0.0)
        for v in ws:
            total = _F(total + v)
        for j, v in enumerate(ws):
            w[i, xmin + j] = _F(v / total) if total != _F(0.0) else v
    w.setflags(write=False)
    return w


@functools.lru_cache(maxsize=32)
def linear_matrix(in_size: int, out_size: int) -> np.ndarray:
    """``F.interpolate(mode='linear', align_corners=False)``(antialias 없음) 1-D 가중치.

    반환: float32 [out_size, in_size].
    """
    if in_size <= 0 or out_size <= 0:
        raise ValueError("in_size/out_size는 1 이상이어야 합니다")
    scale = _F(in_size) / _F(out_size)
    w = np.zeros((out_size, in_size), dtype=np.float32)
    for i in range(out_size):
        src = _F(scale * _F(i + 0.5) - _F(0.5))
        if src < _F(0.0):
            src = _F(0.0)
        i0 = int(math.floor(float(src)))
        i0 = min(i0, in_size - 1)
        i1 = min(i0 + 1, in_size - 1)
        lam = _F(src - _F(i0))
        w[i, i0] += _F(_F(1.0) - lam)
        w[i, i1] += lam
    w.setflags(write=False)
    return w
