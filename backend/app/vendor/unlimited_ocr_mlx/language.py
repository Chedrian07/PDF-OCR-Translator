# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/deepseekocr/language.py
# (LlamaAttention·MoEGate·DeepseekV2MoE·디코더) + mlx_vlm/models/unlimited_ocr/language.py
# (링 캐시 모델) + mlx_vlm/models/mlp.py (DeepseekMLP). MLA·YaRN 분기는 이 모델이 쓰지
# 않아 뺐다(config가 거부). 로컬 패치: [local patch M6] MoE 라우팅 수치를 torch와 같은
# fp32로. 출처·패치 내역: PROVENANCE.md
"""DeepseekV2 MoE 디코더 (Unlimited-OCR: 12층, MHA 10헤드, 64 routed top-6 + shared 2)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .cache import RingSlidingKVCache
from .config import TextConfig
from .switch_layers import SwitchGLU, swiglu


class DeepseekV2MLP(nn.Module):
    def __init__(self, config: TextConfig, hidden_size: int | None = None, intermediate_size: int | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LlamaAttention(nn.Module):
    """torch SlidingWindowLlamaAttention의 연산 경로 (RoPE = rotate_half, base 1e4)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim
        self.scale = head_dim**-0.5
        bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)
        # transformers LlamaRotaryEmbedding(default) == 비-traditional(rotate_half) RoPE
        self.rope = nn.RoPE(head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self, x: mx.array, mask=None, cache: RingSlidingKVCache | None = None
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            # 캐시에 넣기 전에 절대 위치(offset)로 RoPE — 링 덮어쓰기와 무관하게 위치 증가
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # [local patch M6] torch MoEGate: F.linear(x.float(), weight.float()) →
        # softmax(dtype=float32) → topk. mlx-vlm 0.7.4는 bf16 matmul 결과(bf16로 반올림된
        # 게이트 로짓)로 top-k를 골라 근접 전문가 선택이 torch와 갈릴 수 있었다.
        gates = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if k > 1 and self.norm_topk_prob:
            denominator = weights.sum(axis=-1, keepdims=True) + 1e-20
            weights = weights / denominator * self.routed_scaling_factor
        else:
            weights = weights * self.routed_scaling_factor
        return inds, weights  # weights: float32 (torch topk_weight와 같은 dtype)


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )
        self.gate = MoEGate(config)
        self.has_shared = config.n_shared_experts is not None
        if self.has_shared:
            self.shared_experts = DeepseekV2MLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)  # [B, L, K, D] (x.dtype)
        # [local patch M6] torch moe_infer: expert_out.type(fp32) * topk_weight(fp32)를
        # fp32로 합산한 뒤 원 dtype으로 내린다 (mlx-vlm 0.7.4: bf16 곱·합).
        y = (y.astype(mx.float32) * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.has_shared:
            y = y + self.shared_experts(x)
        return y


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache: RingSlidingKVCache | None = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DeepseekV2Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array | None,
        inputs_embeds: mx.array | None = None,
        cache: list[RingSlidingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        # 프리필은 인과 마스크, 디코드(q=1)는 마스크 없음 — torch DeepseekV2Model과 동일
        mask = "causal" if h.shape[1] > 1 else None
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array | None,
        inputs_embeds: mx.array | None = None,
        cache: list[RingSlidingKVCache] | None = None,
        last_only: bool = False,
    ) -> mx.array:
        h = self.model(inputs, inputs_embeds=inputs_embeds, cache=cache)
        if last_only:
            # torch vendor patch P21과 같은 이유: 생성은 마지막 위치 로짓만 쓴다 —
            # 프리필에서 (길이 × vocab) 로짓을 실체화하지 않는다.
            h = h[:, -1:, :]
        return self.lm_head(h)

    def make_cache(self) -> list[RingSlidingKVCache]:
        window = self.config.sliding_window
        if not window:
            raise ValueError("sliding_window_size가 없는 설정은 지원하지 않습니다 (R-SWA 전제)")
        return [RingSlidingKVCache(window) for _ in self.layers]

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return stack_experts(weights, self.config)


def stack_experts(weights: dict[str, mx.array], config: TextConfig) -> dict[str, mx.array]:
    """전문가별 gate/up/down 가중치(E개)를 switch_mlp 스택 [E, out, in]으로 묶는다.

    키는 매핑 후 이름(language_model.model.layers.*) 기준 — model.sanitize가 부른다."""
    prefix_root = "language_model.model.layers"
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"{prefix_root}.{layer_idx}"
        for name in ("gate_proj", "down_proj", "up_proj"):
            first = f"{prefix}.mlp.experts.0.{name}.weight"
            if first in weights:
                to_join = [
                    weights.pop(f"{prefix}.mlp.experts.{e}.{name}.weight")
                    for e in range(config.n_routed_experts)
                ]
                weights[f"{prefix}.mlp.switch_mlp.{name}.weight"] = mx.stack(to_join)
    return weights
