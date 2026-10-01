# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/deepseekocr/config.py +
# mlx_vlm/models/unlimited_ocr/config.py — 이 모델(model_type "unlimited-ocr")에 필요한
# 필드만 남긴 축약판. 출처·패치 내역: PROVENANCE.md
"""Unlimited-OCR MLX 설정.

값의 진리원은 torch 벤더 코드(app/vendor/unlimited_ocr)다 — torch는 HF config.json의
**최상위** 키를 DeepseekV2Config 기본값 위에 읽고(language_config 하위 dict는 쓰지
않는다), SAM ViT-B·CLIP-L 크기는 deepencoder.py에 하드코딩한다. 여기서는 같은 값을
같은 위치에서 읽고, 비전 크기는 config.json의 vision_config.width(정보용 사본)가 있으면
그 값을, 없으면 torch 하드코딩 값을 쓴다(축소 설정 테스트용 — 실가중치는 strict 로드가
모양을 검증한다). 이 포팅이 구현하지 않는 변형(MLA·YaRN·group-limited 라우팅 등)은
조용히 다르게 돌지 않도록 로드 단계에서 거부한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# torch infer/infer_multi 하드코딩 값 (modeling_unlimitedocr.py)
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_ID = 0
STOP_STR = "<｜end▁of▁sentence｜>"


@dataclass
class TextConfig:
    """DeepseekV2 디코더 (torch DeepseekV2Config 기본값 + config.json 최상위 키)."""

    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    n_shared_experts: int | None = 2
    n_routed_experts: int | None = 64
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    topk_method: str = "greedy"
    scoring_func: str = "softmax"
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    hidden_act: str = "silu"
    # R-SWA 링 창 — torch: sliding_window_size or sliding_window (infer/infer_multi)
    sliding_window: int | None = 128
    bos_token_id: int = BOS_ID
    eos_token_id: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, cfg: dict[str, Any]) -> "TextConfig":
        # 미구현 변형은 거부 — 다른 모델이 '그럴듯하게' 로드되는 것을 막는다.
        # torch DeepseekV2Config 기본값이 use_mla=True라 키가 없으면 MLA로 본다.
        # (use_mla=False면 torch는 SlidingWindowLlamaAttention을 쓰고 qk_*/lora 키는
        #  쓰지 않으므로 검사하지 않는다.)
        if cfg.get("use_mla", True):
            raise ValueError("use_mla=True(MLA 어텐션)는 지원하지 않습니다")
        heads = int(cfg.get("num_attention_heads", cls.num_attention_heads))
        hidden = int(cfg.get("hidden_size", cls.hidden_size))
        if cfg.get("head_dim") not in (None, hidden // heads):
            raise ValueError("head_dim != hidden_size/num_attention_heads는 지원하지 않습니다")
        if cfg.get("rope_scaling") is not None:
            raise ValueError("rope_scaling은 지원하지 않습니다 (torch 기본 RoPE만)")
        if cfg.get("topk_method", "greedy") != "greedy":
            raise ValueError(f"topk_method={cfg.get('topk_method')!r}는 지원하지 않습니다")
        if cfg.get("scoring_func", "softmax") != "softmax":
            raise ValueError(f"scoring_func={cfg.get('scoring_func')!r}는 지원하지 않습니다")
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act={cfg.get('hidden_act')!r}는 지원하지 않습니다")
        if int(cfg.get("ep_size", 1) or 1) != 1:
            raise ValueError("ep_size>1(전문가 병렬)은 지원하지 않습니다")
        d = cls()
        out = {}
        for f in (
            "vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "n_shared_experts", "n_routed_experts", "num_experts_per_tok",
            "routed_scaling_factor", "norm_topk_prob", "moe_layer_freq",
            "first_k_dense_replace", "max_position_embeddings", "rms_norm_eps",
            "rope_theta", "attention_bias", "bos_token_id", "eos_token_id",
        ):
            out[f] = cfg.get(f, getattr(d, f))
        if out["num_key_value_heads"] is None:
            out["num_key_value_heads"] = out["num_attention_heads"]
        out["rms_norm_eps"] = float(out["rms_norm_eps"])
        out["rope_theta"] = float(out["rope_theta"])
        out["sliding_window"] = cfg.get("sliding_window_size") or cfg.get("sliding_window")
        return cls(**out)


@dataclass
class SAMViTConfig:
    """SAM ViT-B (torch deepencoder.build_sam_vit_b 하드코딩 값)."""

    image_size: int = 1024
    width: int = 768
    layers: int = 12
    heads: int = 12
    patch_size: int = 16
    window_size: int = 14
    mlp_ratio: float = 4.0
    prompt_embed_dim: int = 256
    global_attn_indexes: tuple[int, ...] = (2, 5, 8, 11)
    downsample_channels: tuple[int, ...] = (512, 1024)
    layer_norm_eps: float = 1e-6


@dataclass
class VisionConfig:
    """CLIP-L/14 (torch deepencoder.vit_model_cfg 하드코딩 값).

    layer_norm_eps=1e-5: torch layernorm_epsilon. mlx-vlm은 1e-6을 써서 출력이
    어긋났다 — [local patch M1] (PROVENANCE.md)."""

    layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    pre_layernorm_eps: float = 1e-5


@dataclass
class ProjectorConfig:
    projector_type: str = "linear"
    input_dim: int = 2048
    n_embed: int = 1280


@dataclass
class ModelConfig:
    text_config: TextConfig = field(default_factory=TextConfig)
    sam_config: SAMViTConfig = field(default_factory=SAMViTConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)
    model_type: str = "unlimited-ocr"
    image_token_id: int = IMAGE_TOKEN_ID
    tile_tag: str = "2D"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ModelConfig":
        model_type = cfg.get("model_type")
        if model_type != "unlimited-ocr":
            raise ValueError(
                f"model_type={model_type!r} — 이 MLX 포팅은 'unlimited-ocr'만 지원합니다"
            )
        if cfg.get("tile_tag", "2D") != "2D":
            raise ValueError(f"tile_tag={cfg.get('tile_tag')!r}는 지원하지 않습니다 (2D만)")
        text = TextConfig.from_hf(cfg)

        pc = cfg.get("projector_config") or {}
        projector = ProjectorConfig(
            projector_type=pc.get("projector_type", "linear"),
            input_dim=int(pc.get("input_dim", 2048)),
            n_embed=int(pc.get("n_embed", 1280)),
        )
        if projector.projector_type != "linear":
            raise ValueError(
                f"projector_type={projector.projector_type!r}는 지원하지 않습니다 (linear만)"
            )
        if projector.n_embed != text.hidden_size:
            raise ValueError("projector n_embed와 디코더 hidden_size가 다릅니다")

        width = ((cfg.get("vision_config") or {}).get("width")) or {}
        sam_d = width.get("sam_vit_b") or {}
        clip_d = width.get("clip-l-14-224") or {}
        sd = SAMViTConfig()
        sam = SAMViTConfig(
            width=int(sam_d.get("width", sd.width)),
            layers=int(sam_d.get("layers", sd.layers)),
            heads=int(sam_d.get("heads", sd.heads)),
            global_attn_indexes=tuple(sam_d.get("global_attn_indexes", sd.global_attn_indexes)),
            downsample_channels=tuple(sam_d.get("downsample_channels", sd.downsample_channels)),
        )
        cd = VisionConfig()
        clip = VisionConfig(
            layers=int(clip_d.get("layers", cd.layers)),
            hidden_size=int(clip_d.get("width", cd.hidden_size)),
            num_attention_heads=int(clip_d.get("heads", cd.num_attention_heads)),
            image_size=int(clip_d.get("image_size", cd.image_size)),
            patch_size=int(clip_d.get("patch_size", cd.patch_size)),
        )
        # 특징 결합 차원 검증: concat(CLIP hidden, SAM net_3 채널) == projector 입력
        if clip.hidden_size + sam.downsample_channels[-1] != projector.input_dim:
            raise ValueError("CLIP hidden + SAM 출력 채널이 projector input_dim과 다릅니다")
        if sam.downsample_channels[-1] != clip.hidden_size:
            raise ValueError("SAM 출력 채널이 CLIP hidden_size와 달라 patch_embeds로 쓸 수 없습니다")
        return cls(
            text_config=text,
            sam_config=sam,
            vision_config=clip,
            projector_config=projector,
            model_type=model_type,
        )
