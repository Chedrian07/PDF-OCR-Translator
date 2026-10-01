# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/deepseekocr/vision.py
# 레퍼런스: app/vendor/unlimited_ocr/deepencoder.py (VitModel, torch).
# 로컬 패치: [local patch M1] quick_gelu + LayerNorm eps 1e-5,
#            [local patch M7] 위치 임베딩 리샘플을 torch 가중치로.
# 출처·패치 내역: PROVENANCE.md
"""CLIP-L/14 비전 타워 — SAM 출력을 patch_embeds로 받아 [B, 1+h*w, 1024]를 낸다."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig
from .resample import bicubic_aa_matrix


class QuickGELU(nn.Module):
    """[local patch M1] torch deepencoder.quick_gelu: ``x * sigmoid(1.702 * x)``.

    mlx-vlm 0.7.4는 nn.GELU(정확식)를 써서 CLIP 출력이 torch 대비 상대오차 0.309였다
    (감사 스파이크, fp32 teacher-forcing). 1쪽 det 블록 19 → 11로 레이아웃이 달라졌다."""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(f"dims({dims})가 num_heads({num_heads})로 나누어떨어지지 않습니다")
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5
        self.qkv_proj = nn.Linear(dims, dims * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # torch NoTPAttention: qkv.view(B, L, 3, heads, hd) — 출력 축이 [3, heads, hd]
        qkv = self.qkv_proj(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        B, L, _ = queries.shape
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = QuickGELU()  # [local patch M1] (upstream: nn.GELU())
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, qkv_bias=True)
        # [local patch M1] eps = torch layernorm_epsilon 1e-5 (upstream config: 1e-6)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


def get_abs_pos(abs_pos: mx.array, tgt_size: int) -> mx.array:
    """CLIP 위치 임베딩 [1, 1+S*S, C] → [1, 1+T*T, C] — torch ``get_abs_pos``.

    [local patch M7] torch는 float32로 ``F.interpolate(bicubic, antialias=True)`` 후
    원 dtype으로 내린다(640 크롭: 16→10). mlx-vlm 0.7.4는 채널-우선 텐서에
    nn.Upsample(채널-마지막 규약)을 걸어 채널 축을 리샘플했다 — 결과가 뒤섞인다."""
    dim = abs_pos.shape[-1]
    cls_token, old_pos_embed = abs_pos[0, :1], abs_pos[0, 1:]
    src_size = int(math.sqrt(abs_pos.shape[1] - 1))
    tgt = int(math.sqrt(tgt_size))
    if src_size == tgt:
        return abs_pos
    dtype = abs_pos.dtype
    w = mx.array(bicubic_aa_matrix(src_size, tgt))  # [T, S]
    p = old_pos_embed.reshape(src_size, src_size, dim).astype(mx.float32)
    p = w @ p  # 가로 축 → [S, T, C]
    p = (w @ p.reshape(src_size, tgt * dim)).reshape(tgt, tgt, dim)  # 세로 축
    new_pos_embed = p.astype(dtype).reshape(tgt * tgt, dim)
    return mx.concatenate([cls_token, new_pos_embed], axis=0)[None]


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = mx.zeros((self.embed_dim,))
        # 체크포인트에 있는 가중치(strict 로드용) — 실사용 경로는 SAM 출력을 patch_embeds로 받는다
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array, patch_embeds: mx.array | None = None) -> mx.array:
        batch_size = x.shape[0]
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(x)  # NHWC
        # torch: patch_embeds.flatten(2).transpose(1, 2) — (h, w) 행 우선 = NHWC reshape
        patch_embeds = patch_embeds.reshape(batch_size, -1, self.embed_dim)
        class_embeds = mx.broadcast_to(
            self.class_embedding.astype(patch_embeds.dtype), (batch_size, 1, self.embed_dim)
        )
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)
        pos = get_abs_pos(self.position_embedding.weight[None], embeddings.shape[1])
        return embeddings + pos


class NoTPTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        config = config or VisionConfig()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.pre_layernorm_eps)
        self.transformer = NoTPTransformer(config)

    def __call__(self, x: mx.array, patch_embeds: mx.array | None = None) -> mx.array:
        x = self.embeddings(x, patch_embeds)
        x = self.pre_layrnorm(x)
        return self.transformer(x)
