# Adapted from mlx-vlm 0.7.4 (MIT): mlx_vlm/utils.py (load_model의 safetensors 수집·
# sanitize·strict load 흐름, quantize의 class_predicate). 로컬 패치: [local patch M5]
# 8비트 전용 인메모리 양자화. 출처·패치 내역: PROVENANCE.md
"""고정 HF 스냅샷(baidu/Unlimited-OCR)을 변환 없이 MLX로 올린다.

- 스냅샷 해석: ``huggingface_hub.snapshot_download`` (HF_HOME/HF_HUB_CACHE/
  HF_HUB_OFFLINE 등 표준 env를 그대로 따른다 — torch 경로와 같은 캐시·blob 공유).
- 가중치: ``mx.load``(safetensors) → ``model.sanitize`` → ``load_weights(strict=True)``
  (키·모양이 하나라도 다르면 실패) → 요청 dtype으로 캐스트.
- 토크나이저: transformers ``AutoTokenizer``(torch 경로와 같은 파일·같은 디코드 규칙).
  transformers는 torch가 설치돼 있으면 임포트 시 torch를 함께 올린다(이 패키지는
  torch를 직접 임포트하지 않는다).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .config import ModelConfig
from .model import Model, sanitize

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
# [local patch M5] 감사 스파이크(MLX-06): 8비트는 처리량 1.44배·품질 동등, 4비트는
# arXiv 번호 2504 → 2304 같은 숫자 오인식 — 8비트만 허용한다.
SUPPORTED_QUANT_BITS = (8,)
DEFAULT_GROUP_SIZE = 64
# 스냅샷에서 필요한 파일만 (config·인덱스·토크나이저 json + 가중치)
ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]


@dataclass(frozen=True)
class LoadInfo:
    model_path: Path
    model_id: str | None
    revision: str | None
    dtype: str
    quantization: dict | None  # {"bits": 8, "group_size": 64, "mode": "affine"} | None
    load_seconds: float
    parameter_bytes: int


def resolve_snapshot(
    model_id: str,
    revision: str | None,
    *,
    local_files_only: bool | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """설정(model_id/revision)이 가리키는 로컬 스냅샷 경로 — 필요하면 내려받는다.

    오프라인(HF_HUB_OFFLINE=1 또는 local_files_only=True)이면 캐시에 있는 스냅샷만 쓴다."""
    from huggingface_hub import snapshot_download

    kwargs: dict = {"repo_id": model_id, "revision": revision, "allow_patterns": ALLOW_PATTERNS}
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(**kwargs))


def load_config(model_path: str | Path) -> ModelConfig:
    with open(Path(model_path) / "config.json", encoding="utf-8") as f:
        return ModelConfig.from_dict(json.load(f))


def load_raw_weights(model_path: str | Path) -> dict[str, mx.array]:
    """스냅샷의 safetensors 전부를 (지연) 로드 — 인덱스가 있으면 인덱스의 샤드만."""
    model_path = Path(model_path)
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        with open(index, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = sorted(p.name for p in model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"safetensors 가중치가 없습니다: {model_path}")
    weights: dict[str, mx.array] = {}
    for name in files:
        weights.update(mx.load(str(model_path / name)))
    return weights


def quantize_decoder(
    model: Model, bits: int = 8, group_size: int = DEFAULT_GROUP_SIZE, mode: str = "affine"
) -> dict:
    """[local patch M5] 디코더(language_model)만 인메모리 양자화.

    대상: embed_tokens·q/k/v/o·dense MLP·shared experts·switch_mlp(전문가 스택)·lm_head.
    제외: SAM·CLIP·projector(비전 경로 — 스파이크 변환본은 projector도 양자화했지만
    여기서는 bf16 유지), MoE 게이트(fp32 라우팅, M6)·RMSNorm(Linear가 아님)."""
    if bits not in SUPPORTED_QUANT_BITS:
        raise ValueError(
            f"OCR MLX 양자화는 {SUPPORTED_QUANT_BITS}비트만 지원합니다 (요청 {bits}) — "
            "4비트는 숫자 오인식(2504→2304)이 측정됐다"
        )

    def predicate(_path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % group_size == 0

    nn.quantize(
        model.language_model, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate
    )
    return {"bits": bits, "group_size": group_size, "mode": mode}


def build_model(
    model_path: str | Path,
    *,
    dtype: str = "bfloat16",
    quantize_bits: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[Model, dict | None]:
    """스냅샷 디렉터리 → 가중치가 평가된 Model (strict 로드)."""
    if dtype not in DTYPES:
        raise ValueError(f"알 수 없는 dtype: {dtype!r} ({'|'.join(DTYPES)})")
    config = load_config(model_path)
    model = Model(config)
    weights = sanitize(load_raw_weights(model_path), config)
    model.load_weights(list(weights.items()), strict=True)
    del weights
    target = DTYPES[dtype]
    model.update(
        tree_map(
            lambda p: p.astype(target) if mx.issubdtype(p.dtype, mx.floating) else p,
            model.parameters(),
        )
    )
    quant = None
    if quantize_bits:
        quant = quantize_decoder(model, bits=int(quantize_bits), group_size=group_size)
    model.eval()
    mx.eval(model.parameters())
    return model, quant


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


def load(
    model_id: str,
    revision: str | None = None,
    *,
    dtype: str = "bfloat16",
    quantize_bits: int | None = None,
    local_files_only: bool | None = None,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
):
    """(model, tokenizer, LoadInfo). model_path를 주면 HF 해석 없이 그 디렉터리를 쓴다."""
    t0 = time.perf_counter()
    path = (
        Path(model_path)
        if model_path is not None
        else resolve_snapshot(
            model_id, revision, local_files_only=local_files_only, cache_dir=cache_dir
        )
    )
    model, quant = build_model(path, dtype=dtype, quantize_bits=quantize_bits)
    tokenizer = load_tokenizer(path)
    nbytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    info = LoadInfo(
        model_path=path,
        model_id=model_id,
        revision=revision,
        dtype=dtype,
        quantization=quant,
        load_seconds=time.perf_counter() - t0,
        parameter_bytes=int(nbytes),
    )
    return model, tokenizer, info
