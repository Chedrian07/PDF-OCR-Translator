"""MPS(Metal) 전용 벤더 패치·게이트 계약 — CPU CI에서 도는 계약 + 옵트인 실기기 검증.

MPS 우회 패치는 Metal에서만 의미가 있어 CPU CI가 회귀를 못 잡았다(audit
tests-baseline-5, infra-docs-14, apple-mps-6). 여기서는
- 하드웨어 없이 검증 가능한 게이트·정책을 한 곳에 고정하고(P11·P12·P16·P17·P18·P20·
  MPS 정적 ngram 창),
- ``OCR_MPS_TESTS=1``이고 MPS가 있는 Mac에서만 같은 경로를 실제 mps 텐서로 돌린다
  (torch·macOS 업그레이드 전후에 ``OCR_MPS_TESTS=1 pytest tests/test_mps_contract.py``).
"""

import contextlib
import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")

from app.native_ops import (  # noqa: E402
    TorchSlidingWindowNoRepeatNgram,
    banned_ngram_tokens_py,
    make_ngram_logits_processor,
)
from app.vendor.unlimited_ocr import modeling_deepseekv2 as md  # noqa: E402
from app.vendor.unlimited_ocr import modeling_unlimitedocr as mu  # noqa: E402


def _mps_opt_in() -> bool:
    return (
        os.environ.get("OCR_MPS_TESTS") == "1"
        and sys.platform == "darwin"
        and torch.backends.mps.is_available()
    )


mps_only = pytest.mark.skipif(
    not _mps_opt_in(), reason="OCR_MPS_TESTS=1 + Apple Silicon MPS 전용 실기기 검증"
)


# ───────────────────────── 하드웨어 무관 계약 (CI) ─────────────────────────


def test_p12_autocast_is_disabled_on_mps():
    """P12: MPS는 dtype과 무관하게 autocast를 쓰지 않는다 — torch 2.10 MPS autocast(bf16)가
    로짓을 오염시켜 16번째 토큰부터 출력이 붕괴함을 M4 Max에서도 재확인(2026-10)."""
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        ctx = mu._autocast_ctx(torch.device("mps"), dtype)
        assert isinstance(ctx, contextlib.nullcontext), dtype


def test_p1_autocast_still_applies_off_mps():
    """P12가 CPU/CUDA 경로를 건드리지 않았는지 — 반정밀도면 autocast, fp32면 no-op."""
    assert isinstance(mu._autocast_ctx(torch.device("cpu"), torch.float32), contextlib.nullcontext)
    assert isinstance(mu._autocast_ctx(torch.device("cpu"), torch.bfloat16), torch.autocast)


@pytest.mark.parametrize(
    "env,expected",
    [
        (None, {"cuda": True, "mps": False, "cpu": False}),
        ("", {"cuda": True, "mps": False, "cpu": False}),
        ("1", {"cuda": True, "mps": True, "cpu": True}),
        (" ON ", {"cuda": True, "mps": True, "cpu": True}),
        ("0", {"cuda": False, "mps": False, "cpu": False}),
        ("off", {"cuda": False, "mps": False, "cpu": False}),
        ("bogus", {"cuda": True, "mps": False, "cpu": False}),
    ],
)
def test_p16_sdpa_policy(monkeypatch, env, expected):
    """P16: 기본은 CUDA만 SDPA, MPS/CPU는 eager. OCR_SDPA=1/0이 전 디바이스를 덮어쓴다."""
    if env is None:
        monkeypatch.delenv("OCR_SDPA", raising=False)
    else:
        monkeypatch.setenv("OCR_SDPA", env)
    for device_type, want in expected.items():
        assert md._sdpa_enabled(device_type) is want, (env, device_type)


def _tiny_attention_model():
    from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config

    torch.manual_seed(0)
    cfg = DeepseekV2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        first_k_dense_replace=99, moe_layer_freq=1, topk_method="greedy",
        scoring_func="softmax", n_group=1, topk_group=1, hidden_act="silu",
        aux_loss_alpha=0.0, use_mla=False, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return md.DeepseekV2Model(cfg).eval()


def test_p16_attention_consults_the_policy(monkeypatch):
    """어텐션이 실제로 _sdpa_enabled를 따른다 — 정책 함수만 맞고 forward가 다른 숨은
    조건을 쓰면 위 테스트는 공허하다. 정책을 켜면 SDPA를, 끄면 eager를 탄다."""
    calls = {"n": 0}
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    ids = torch.randint(0, 64, (1, 5), generator=torch.Generator().manual_seed(1))
    model = _tiny_attention_model()
    monkeypatch.setattr(md, "_sdpa_enabled", lambda device_type, env=None: False)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    assert calls["n"] == 0
    monkeypatch.setattr(md, "_sdpa_enabled", lambda device_type, env=None: True)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    assert calls["n"] == model.config.num_hidden_layers


def _inject_bool_index(embeds, mask, features):
    out = embeds.clone()
    out[0][mask[0]] = features  # 벤더 P11과 같은 형태: inputs_embeds[idx][_img_mask] = …
    return out


def _inject_masked_scatter(embeds, mask, features):
    out = embeds.clone()
    out[0].masked_scatter_(mask[0].unsqueeze(-1), features)  # 업스트림 원본(브로드캐스트 마스크)
    return out


def test_p11_bool_index_injection_equals_masked_scatter_on_cpu():
    """P11: 이미지 임베딩 주입을 bool 인덱싱 대입으로 바꿔도 CPU 결과는 masked_scatter_와 동일
    (MPS에서는 브로드캐스트 masked_scatter_가 첫 원소만 기록해 즉시 EOS가 났다)."""
    gen = torch.Generator().manual_seed(3)
    embeds = torch.randn(1, 12, 8, generator=gen)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, 2:9] = True
    features = torch.randn(7, 8, generator=gen)
    assert torch.equal(
        _inject_bool_index(embeds, mask, features), _inject_masked_scatter(embeds, mask, features)
    )


def test_p11_vendor_forward_does_not_use_masked_scatter():
    """벤더 이미지 주입부(UnlimitedOCRModel.forward)에 masked_scatter_ 호출이 돌아오면 안 된다
    (주석이 아니라 AST의 실제 호출로 검사)."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(mu.UnlimitedOCRModel.forward)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "masked_scatter_" not in called and "masked_scatter" not in called
    assert "inputs_embeds[idx][_img_mask] =" in inspect.getsource(mu.UnlimitedOCRModel.forward)


def test_mps_gate_matrix(monkeypatch):
    """MPS가 받는 디코드 경로 한눈에 — 기본 env에서 디바이스별 패치 게이트."""
    for key in ("OCR_MOE_FUSED", "OCR_MOE_FAST", "OCR_SDPA", "OCR_NGRAM_HOST"):
        monkeypatch.delenv(key, raising=False)
    # P17 융합 MoE 디코드: CUDA·MPS on, CPU off (N==1)
    from types import SimpleNamespace

    from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config

    moe = md.DeepseekV2MoE(DeepseekV2Config(
        hidden_size=16, n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        n_shared_experts=1, topk_method="greedy", scoring_func="softmax",
    ))
    for device_type, want in (("cuda", True), ("mps", True), ("cpu", False)):
        x = SimpleNamespace(device=SimpleNamespace(type=device_type), shape=(1, 16))
        assert moe._should_use_fused(x, None) is want, device_type
    # P18 패스트패스: MPS 전용 기본 (P17 킬스위치를 끈 MPS 폴백)
    assert [md._moe_fast_enabled(d) for d in ("cuda", "mps", "cpu")] == [False, True, False]
    # P16: MPS eager
    assert md._sdpa_enabled("mps") is False
    # 정적 ngram 창: MPS만
    assert make_ngram_logits_processor(35, 1024, "mps")[0].static_shape is True
    assert make_ngram_logits_processor(35, 1024, "cuda")[0].static_shape is False
    # P20: 캐시 기본은 int 슬롯 모드 (텐서 모드는 CUDA Graph 진입 시에만)
    from transformers.cache_utils import DynamicCache

    assert md._ring_tensor_mode(DynamicCache()) is False


# ───────────────────────── 옵트인 실기기(MPS) 검증 ─────────────────────────


@mps_only
def test_mps_p11_bool_index_injection_matches_cpu():
    gen = torch.Generator().manual_seed(5)
    embeds = torch.randn(1, 40, 16, generator=gen)
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, 3:31] = True
    features = torch.randn(28, 16, generator=gen)
    want = _inject_bool_index(embeds, mask, features)
    got = _inject_bool_index(embeds.to("mps"), mask.to("mps"), features.to("mps")).cpu()
    assert torch.equal(got, want)


@mps_only
def test_mps_static_ngram_matches_reference():
    """MPS 정적 창 프로세서 ↔ 레퍼런스 무작위 패리티(n=35·w=1024/128 포함)."""
    rng = random.Random(0)
    cases = [(35, 1024, 900), (35, 128, 400), (35, 1024, 1500)] + [
        (rng.randrange(1, 6), rng.randrange(1, 40), rng.randrange(0, 80)) for _ in range(200)
    ]
    for n, w, length in cases:
        vocab = 6 if n < 6 else 24
        seq = [rng.randrange(vocab) for _ in range(length)]
        if length > 40:
            seq[length // 3: length // 3 + n - 1] = seq[-(n - 1):] if n > 1 else seq[length // 3: length // 3]
        proc = TorchSlidingWindowNoRepeatNgram(n, w, static_shape=True)
        scores = proc(torch.tensor([seq], device="mps"), torch.zeros(1, vocab, device="mps")).cpu()
        got = sorted(i for i in range(vocab) if scores[0, i] == float("-inf"))
        assert got == banned_ngram_tokens_py(seq, n, w), (n, w, length)


@mps_only
def test_mps_static_ngram_does_not_grow_rss_per_length():
    """정적 창이면 새 시퀀스 길이마다 MPSGraph를 컴파일하지 않는다(동적이면 길이당 ~2.9MB)."""
    from test_objc_pool import _resident_mb

    vocab = 129_280
    ids = torch.randint(0, 1000, (1, 1500), device="mps")
    scores = torch.randn(1, vocab, device="mps")
    proc = TorchSlidingWindowNoRepeatNgram(35, 1024, static_shape=True)
    for length in range(100, 110):
        proc(ids[:, :length], scores.clone()).argmax(-1).tolist()
    torch.mps.synchronize()
    before = _resident_mb()
    for length in range(300, 500):
        out = proc(ids[:, :length], scores.clone()).argmax(-1)
        if length % 8 == 0:
            out.tolist()
    torch.mps.synchronize()
    assert _resident_mb() - before < 60.0


@mps_only
def test_mps_p18_fastpath_bit_identical_bf16(monkeypatch):
    from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config

    torch.manual_seed(0)
    cfg = DeepseekV2Config(
        hidden_size=64, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        n_shared_experts=1, topk_method="greedy", scoring_func="softmax", n_group=1,
        topk_group=1, hidden_act="silu", aux_loss_alpha=0.0,
    )
    moe = md.DeepseekV2MoE(cfg).eval().to(device="mps", dtype=torch.bfloat16)
    moe._fused_env = False  # P17 끔 → P18/legacy 비교
    x = torch.randn(1, 1, 64, dtype=torch.bfloat16, device="mps")
    with torch.no_grad():
        monkeypatch.setenv("OCR_MOE_FAST", "1")
        fast = moe(x)
        monkeypatch.setenv("OCR_MOE_FAST", "0")
        slow = moe(x)
    assert torch.equal(fast.cpu(), slow.cpu())


@mps_only
def test_mps_p17_prebuilt_stacks_share_storage_and_match_p18():
    """MPS: 이동 전 프리빌드 → 스택 뷰 공유 유지, 융합 디코드 ≈ P18(bf16 허용오차),
    반복 호출에도 PyTorch 할당 메모리가 늘지 않는다."""
    from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config

    torch.manual_seed(0)
    cfg = DeepseekV2Config(
        hidden_size=128, n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
        n_shared_experts=1, topk_method="greedy", scoring_func="softmax", n_group=1,
        topk_group=1, hidden_act="silu", aux_loss_alpha=0.0,
    )
    moe = md.DeepseekV2MoE(cfg).eval().to(torch.bfloat16)
    assert md.prebuild_fused_moe(moe, "mps") == 1
    moe = moe.to("mps")
    g, u, d = moe._fused_w
    for i, e in enumerate(moe.experts):
        assert e.gate_proj.weight.data_ptr() == g[i].data_ptr()
        assert e.down_proj.weight.data_ptr() == d[i].data_ptr()
    x = torch.randn(1, 128, dtype=torch.bfloat16, device="mps")
    with torch.no_grad():
        topk_idx, topk_weight, _ = moe.gate(x.view(1, 1, 128))
        legacy = moe.moe_infer(x, topk_idx, topk_weight)
        fused = moe._moe_infer_fused(x, topk_idx, topk_weight)
        assert torch.allclose(fused.float().cpu(), legacy.float().cpu(), rtol=2e-2, atol=2e-2)
        torch.mps.synchronize()
        before = torch.mps.current_allocated_memory()
        for _ in range(50):
            moe._moe_infer_fused(x, topk_idx, topk_weight)
        torch.mps.synchronize()
        assert torch.mps.current_allocated_memory() <= before + 1 * 2**20


@mps_only
def test_mps_tiny_vendor_decode_matches_cpu():
    """tiny 벤더 모델(fp32)의 MPS eager 디코드(P19 rotary 캐시·P20 int 링·정적 ngram 창·
    오토릴리스 풀)가 CPU 디코드와 토큰 동일."""
    from test_decode_tiny_vendor import PROMPT, _tiny_vendor_model

    from app.engine.fast_decode import fast_greedy_decode

    def run(device):
        model = _tiny_vendor_model().to(device)
        kwargs = {
            "input_ids": torch.tensor([PROMPT], device=device),
            "images": None, "images_seq_mask": None, "images_spatial_crop": None,
            "do_sample": False, "eos_token_id": None, "max_length": 64, "use_cache": True,
            "logits_processor": make_ngram_logits_processor(3, 16, "mps" if device == "mps" else "cpu"),
        }
        return fast_greedy_decode(model, kwargs, block=4)[0].tolist()

    assert run("mps") == run("cpu")
