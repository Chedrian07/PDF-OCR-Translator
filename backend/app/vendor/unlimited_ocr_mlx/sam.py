# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/deepseekocr/sam.py
# 레퍼런스: app/vendor/unlimited_ocr/deepencoder.py (ImageEncoderViT, torch).
# 로컬 패치: [local patch M7] 위치 임베딩·상대 위치표 리샘플을 torch와 같은 가중치로
# (resample.py). 출처·패치 내역: PROVENANCE.md
"""SAM ViT-B 이미지 인코더 (NHWC 레이아웃 — MLX conv 규약)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import SAMViTConfig
from .resample import bicubic_aa_matrix, linear_matrix


def get_abs_pos_sam(abs_pos: mx.array, tgt_size: int) -> mx.array:
    """절대 위치 임베딩 [1, S, S, C]를 [1, T, T, C]로 — torch ``get_abs_pos_sam``.

    torch는 float32로 올려 ``F.interpolate(bicubic, antialias=True)`` 후 원 dtype으로
    내린다. [local patch M7] 같은 가중치 행렬을 가로(W)→세로(H) 순으로 적용한다
    (torch의 분리형 AA 커널 순서)."""
    dtype = abs_pos.dtype
    src_size = abs_pos.shape[1]
    if src_size == tgt_size:
        return abs_pos
    w = mx.array(bicubic_aa_matrix(src_size, tgt_size))  # [T, S]
    p = abs_pos[0].astype(mx.float32)  # [S(h), S(w), C]
    p = w @ p  # 가로 축: (T,S) @ (S_h, S_w, C) → [S_h, T, C]
    s_h, t, c = p.shape
    p = (w @ p.reshape(s_h, t * c)).reshape(tgt_size, t, c)  # 세로 축 → [T, T, C]
    return p[None].astype(dtype)


class MLPBlock(nn.Module):
    """MLP block with GELU activation (torch nn.GELU 기본 = erf 정확식)."""

    def __init__(self, embedding_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    """상대 위치표를 q/k 크기에 맞춰 추출 — torch ``get_rel_pos``.

    [local patch M7] 길이가 다르면 torch ``F.interpolate(mode='linear')``와 같은
    가중치(반-픽셀 중심 좌표)로 float32 리샘플 후 원 dtype으로 내린다.
    (mlx-vlm 0.7.4는 ``i*scale`` 좌표라 640 크롭의 전역 블록에서 어긋났다.)"""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        w = mx.array(linear_matrix(rel_pos.shape[0], max_rel_dist))  # [M, L]
        rel_pos_resized = (w @ rel_pos.astype(mx.float32)).astype(dtype)  # [M, C]
    else:
        rel_pos_resized = rel_pos

    # q/k 크기가 다르면 짧은 쪽 좌표를 늘린다 (torch와 같은 식, .long() 절단)
    q_coords = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[mx.array(relative_coords.astype(np.int64))]


def add_decomposed_rel_pos(
    q: mx.array,
    rel_pos_h: mx.array,
    rel_pos_w: mx.array,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> mx.array:
    """분해형 상대 위치 바이어스 — 반환 [B, q_h*q_w, k_h*k_w] (torch attn_bias와 동일)."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = mx.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = mx.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)
    return (rel_h + rel_w).reshape(B, q_h * q_w, k_h * k_w)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "use_rel_pos에는 input_size가 필요합니다"
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        # qkv: (3, B, nHead, H*W, C) — torch와 같은 [3, heads, head_dim] 출력 배치
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, nH, HW, C]

        if self.use_rel_pos:
            attn_bias = add_decomposed_rel_pos(
                q.reshape(B * self.num_heads, H * W, -1),
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
            ).reshape(B, self.num_heads, H * W, H * W)
            x = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=attn_bias
            )
        else:
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.reshape(B, self.num_heads, H, W, -1).transpose(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


def window_partition(x: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    """겹치지 않는 창으로 분할 (필요하면 0 패딩) — [B*nW, ws, ws, C]."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: mx.array, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> mx.array:
    """창 분할 역변환 + 패딩 제거 — [B, H, W, C]."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class Block(nn.Module):
    """Transformer block with window attention and residual propagation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (Conv2d, bias 있음 — torch 기본)."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class SAMEncoder(nn.Module):
    """SAM ViT-B + neck + net_2/net_3 다운샘플 — 출력 [B, H/64, W/64, 1024] (NHWC)."""

    def __init__(self, cfg: SAMViTConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SAMViTConfig()
        self.img_size = cfg.image_size
        grid = cfg.image_size // cfg.patch_size
        self.patch_embed = PatchEmbed(cfg.patch_size, 3, cfg.width)
        # 사전학습 해상도(1024/16=64) 기준 절대 위치 임베딩
        self.pos_embed = mx.zeros((1, grid, grid, cfg.width))
        self.blocks = [
            Block(
                dim=cfg.width,
                num_heads=cfg.heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=True,
                use_rel_pos=True,
                window_size=cfg.window_size if i not in cfg.global_attn_indexes else 0,
                input_size=(grid, grid),
                eps=cfg.layer_norm_eps,
            )
            for i in range(cfg.layers)
        ]
        out = cfg.prompt_embed_dim
        # torch neck: Conv1x1 → LayerNorm2d → Conv3x3 → LayerNorm2d (채널 정규화 =
        # NHWC 마지막 축 LayerNorm, eps 1e-6)
        self.neck = [
            nn.Conv2d(cfg.width, out, kernel_size=1, bias=False),
            nn.LayerNorm(out, eps=1e-6),
            nn.Conv2d(out, out, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(out, eps=1e-6),
        ]
        c2, c3 = cfg.downsample_channels
        self.net_2 = nn.Conv2d(out, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        x = x + get_abs_pos_sam(self.pos_embed, x.shape[1])
        for blk in self.blocks:
            x = blk(x)
        for layer in self.neck:
            x = layer(x)
        x = self.net_2(x)
        return self.net_3(x)
