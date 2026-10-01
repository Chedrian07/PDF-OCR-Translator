"""[local patch M7] 위치 임베딩 리샘플이 torch ``F.interpolate``와 같은지.

gundam 640 크롭에서 SAM 위치 임베딩(64→40)·CLIP 위치 임베딩(16→10)은 bicubic
antialias, SAM 전역 블록의 상대 위치표(127→79)는 linear로 축소된다. 가중치 행렬은
numpy만 쓰므로 torch만 있으면(Linux CI 포함) 비교한다. MLX 적용 함수는 mlx가 있을 때.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.vendor.unlimited_ocr_mlx import mlx_status
from app.vendor.unlimited_ocr_mlx.resample import bicubic_aa_matrix, linear_matrix

torch = pytest.importorskip("torch")
F = torch.nn.functional


@pytest.mark.parametrize("src,dst", [(16, 10), (64, 40), (16, 16), (64, 64), (64, 30), (16, 24)])
def test_bicubic_antialias_matches_torch(src, dst):
    x = np.random.default_rng(src * 100 + dst).standard_normal((1, 5, src, src)).astype(np.float32)
    ref = F.interpolate(torch.from_numpy(x), size=(dst, dst), mode="bicubic", antialias=True,
                        align_corners=False).numpy()
    w = bicubic_aa_matrix(src, dst)
    got = np.einsum("oi,bcij,pj->bcop", w, x, w)
    np.testing.assert_allclose(got, ref, rtol=0, atol=5e-6)


@pytest.mark.parametrize("src,dst", [(127, 79), (27, 27), (127, 127), (127, 50)])
def test_linear_matches_torch(src, dst):
    x = np.random.default_rng(src + dst).standard_normal((1, 4, src)).astype(np.float32)
    ref = F.interpolate(torch.from_numpy(x), size=dst, mode="linear").numpy()
    got = np.einsum("oi,bci->bco", linear_matrix(src, dst), x)
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6)


def test_matrices_are_cached_and_read_only():
    a = bicubic_aa_matrix(16, 10)
    assert a is bicubic_aa_matrix(16, 10)
    with pytest.raises(ValueError):
        a[0, 0] = 1.0
    with pytest.raises(ValueError):
        linear_matrix(0, 3)


_OK, _WHY = mlx_status()


@pytest.mark.skipif(not _OK, reason=_WHY)
def test_mlx_position_embeddings_match_torch_functions():
    import mlx.core as mx

    from app.vendor.unlimited_ocr import deepencoder as de
    from app.vendor.unlimited_ocr_mlx.sam import get_abs_pos_sam, get_rel_pos
    from app.vendor.unlimited_ocr_mlx.vision import get_abs_pos

    rng = np.random.default_rng(0)
    sam_pos = rng.standard_normal((1, 64, 64, 32)).astype(np.float32)
    ref = de.get_abs_pos_sam(torch.from_numpy(sam_pos), 40).numpy()
    np.testing.assert_allclose(np.array(get_abs_pos_sam(mx.array(sam_pos), 40)), ref, atol=5e-6)

    clip_pos = rng.standard_normal((1, 257, 48)).astype(np.float32)
    ref = de.get_abs_pos(torch.from_numpy(clip_pos), 101).numpy()
    np.testing.assert_allclose(np.array(get_abs_pos(mx.array(clip_pos), 101)), ref, atol=5e-6)
    assert get_abs_pos(mx.array(clip_pos), 257).shape == (1, 257, 48)

    rel = rng.standard_normal((127, 16)).astype(np.float32)
    ref = de.get_rel_pos(40, 40, torch.from_numpy(rel)).numpy()
    np.testing.assert_allclose(np.array(get_rel_pos(40, 40, mx.array(rel))), ref, atol=2e-6)
