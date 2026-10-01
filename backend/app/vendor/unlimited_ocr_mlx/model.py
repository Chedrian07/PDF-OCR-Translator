# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/deepseekocr/deepseekocr.py (Model,
# MlpProjector, sanitize) + mlx_vlm/models/unlimited_ocr/unlimitedocr.py (멀티페이지
# 단일 <image> 경로) + vision.py/language.py의 sanitize.
# 레퍼런스: app/vendor/unlimited_ocr/modeling_unlimitedocr.py (UnlimitedOCRModel.forward).
# 출처·패치 내역: PROVENANCE.md
"""SAM + CLIP + projector + DeepseekV2 디코더 조립과 가중치 키 매핑."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig, ProjectorConfig
from .language import LanguageModel, stack_experts
from .sam import SAMEncoder
from .vision import VisionModel

# HF(torch) 체크포인트 키 접두어 → 이 모듈 트리 접두어. 표에 없는 키는 거부(strict).
_KEY_PREFIXES = (
    ("lm_head.", "language_model.lm_head."),
    ("model.embed_tokens.", "language_model.model.embed_tokens."),
    ("model.layers.", "language_model.model.layers."),
    ("model.norm.", "language_model.model.norm."),
    ("model.sam_model.", "sam_model."),
    ("model.vision_model.", "vision_model."),
    ("model.projector.", "projector."),
)
_KEY_EXACT = {
    "model.image_newline": "image_newline",
    # 업스트림 오타(view_seperator) 키를 mlx-vlm 이름(view_separator)으로
    "model.view_seperator": "view_separator",
}
# torch Conv2d [O, I, kh, kw] → MLX Conv2d [O, kh, kw, I]
_CONV_KEYS = frozenset({
    "sam_model.patch_embed.proj.weight",
    "sam_model.neck.0.weight",
    "sam_model.neck.2.weight",
    "sam_model.net_2.weight",
    "sam_model.net_3.weight",
    "vision_model.embeddings.patch_embedding.weight",
})


def _map_key(key: str) -> str:
    if key in _KEY_EXACT:
        return _KEY_EXACT[key]
    for src, dst in _KEY_PREFIXES:
        if key.startswith(src):
            return dst + key[len(src):]
    raise ValueError(f"알 수 없는 가중치 키: {key!r}")


def sanitize(weights: dict[str, mx.array], config: ModelConfig) -> dict[str, mx.array]:
    """HF safetensors(torch 레이아웃) 가중치를 이 모델 트리에 맞춘다.

    mlx-vlm의 Model/VisionModel/LanguageModel.sanitize 3단계를 한 곳에서 같은 순서로:
    키 매핑 → conv 전치 → 전문가 스택. torch state_dict에만 있는 버퍼
    (vision_model.embeddings.position_ids)는 버린다 — 실체크포인트에는 없다."""
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.endswith("position_ids"):
            continue
        new_key = _map_key(key)
        if new_key in _CONV_KEYS:
            if value.ndim != 4:
                raise ValueError(f"{key}: conv 가중치는 4차원이어야 합니다 (shape={value.shape})")
            value = value.transpose(0, 2, 3, 1)
        out[new_key] = value
    return stack_experts(out, config.text_config)


class MlpProjector(nn.Module):
    """projector_type='linear' — Linear(2048 → 1280), bias 있음 (torch MlpProjector)."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layers = nn.Linear(config.input_dim, config.n_embed)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sam_model = SAMEncoder(config.sam_config)
        self.vision_model = VisionModel(config.vision_config)
        self.projector = MlpProjector(config.projector_config)
        self.language_model = LanguageModel(config.text_config)
        n_embed = config.projector_config.n_embed
        self.image_newline = mx.zeros((n_embed,))
        self.view_separator = mx.zeros((n_embed,))

    @property
    def dtype(self) -> mx.Dtype:
        """연산 dtype — 양자화되지 않는 파라미터(image_newline) 기준."""
        return self.image_newline.dtype

    # ── 이미지 특징 (torch UnlimitedOCRModel.forward와 같은 조립 순서) ──

    def _encode(self, pixels: mx.array) -> mx.array:
        """[B, S, S, 3] → [B, (S/64)^2, n_embed]: concat(CLIP[:,1:], SAM) → projector."""
        sam_feats = self.sam_model(pixels)  # [B, h, w, 1024]
        clip_feats = self.vision_model(pixels, sam_feats)  # [B, 1+h*w, 1024]
        b = sam_feats.shape[0]
        feats = mx.concatenate(
            [clip_feats[:, 1:], sam_feats.reshape(b, -1, sam_feats.shape[-1])], axis=-1
        )
        return self.projector(feats)

    def _add_newline(self, feats: mx.array, rows: int, cols: int) -> mx.array:
        """[rows*cols, n] 격자 각 행 끝에 image_newline을 붙여 [rows*(cols+1), n]으로."""
        n = feats.shape[-1]
        grid = feats.reshape(rows, cols, n)
        newline = mx.broadcast_to(self.image_newline.astype(grid.dtype)[None, None, :], (rows, 1, n))
        return mx.concatenate([grid, newline], axis=1).reshape(-1, n)

    def encode_images(
        self,
        global_views: np.ndarray,
        crops: np.ndarray | None = None,
        spatial_crop: tuple[int, int] = (1, 1),
    ) -> mx.array:
        """프롬프트 이미지 토큰 자리에 들어갈 특징 [n_image_tokens, n_embed].

        - crops 없음(멀티페이지·작은 단일 이미지): 각 전역 뷰마다
          ``[global(+newline), view_separator]`` 를 페이지 순서대로 잇는다.
        - crops 있음(gundam): ``[local(+newline), global(+newline), view_separator]``
          — torch crop 분기와 같은 순서·배치(로컬은 crop 전체를 한 배치로).

        torch는 ``sum(pixels) != 0`` 으로 분기하지만 여기서는 crops 유무를 명시적으로
        넘겨받는다(합이 우연히 0인 입력에서 torch는 shape 불일치로 실패한다).
        페이지마다 평가해 다중 페이지의 비전 중간값이 한꺼번에 상주하지 않게 한다."""
        dtype = self.dtype
        n = self.config.projector_config.n_embed
        sep = self.view_separator.astype(dtype)[None, :]
        if crops is not None and len(crops) > 0:
            local = self._encode(mx.array(crops).astype(dtype))  # [M, hw2, n]
            glob = self._encode(mx.array(global_views[:1]).astype(dtype))[0]  # [hw, n]
            h = int(math.isqrt(glob.shape[0]))
            glob = self._add_newline(glob, h, h)
            width_crop_num, height_crop_num = int(spatial_crop[0]), int(spatial_crop[1])
            hw2 = local.shape[1]
            h2 = int(math.isqrt(hw2))
            local = (
                local.reshape(height_crop_num, width_crop_num, h2, h2, n)
                .transpose(0, 2, 1, 3, 4)
                .reshape(height_crop_num * h2, width_crop_num * h2, n)
            )
            local = self._add_newline(
                local.reshape(-1, n), height_crop_num * h2, width_crop_num * h2
            )
            out = mx.concatenate([local, glob, sep], axis=0)
            mx.eval(out)
            return out
        parts = []
        for i in range(len(global_views)):
            g = self._encode(mx.array(global_views[i : i + 1]).astype(dtype))[0]
            h = int(math.isqrt(g.shape[0]))
            g = mx.concatenate([self._add_newline(g, h, h), sep], axis=0)
            mx.eval(g)
            parts.append(g)
        return mx.concatenate(parts, axis=0)

    def get_input_embeddings(
        self,
        input_ids: np.ndarray,
        images_seq_mask: np.ndarray,
        image_features: mx.array | None,
    ) -> mx.array:
        """토큰 임베딩 [1, L, D]의 이미지 위치를 특징으로 교체 (torch masked 대입과 동일)."""
        ids = mx.array(np.asarray(input_ids, dtype=np.int32))[None]
        embeds = self.language_model.model.embed_tokens(ids)
        positions = np.flatnonzero(np.asarray(images_seq_mask, dtype=bool))
        if image_features is None:
            if len(positions):
                raise ValueError("이미지 토큰 자리가 있는데 이미지 특징이 없습니다")
            return embeds
        if image_features.shape[0] != len(positions):
            raise ValueError(
                f"이미지 특징 {image_features.shape[0]}개 != 이미지 토큰 자리 {len(positions)}개"
            )
        embeds[0, mx.array(positions.astype(np.int32))] = image_features.astype(embeds.dtype)
        return embeds

    def __call__(self, inputs, inputs_embeds=None, cache=None, last_only: bool = False):
        return self.language_model(inputs, inputs_embeds=inputs_embeds, cache=cache, last_only=last_only)

    def make_cache(self):
        return self.language_model.make_cache()
