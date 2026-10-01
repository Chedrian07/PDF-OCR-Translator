"""[vendor patch P17] DeepseekV2MoE 융합 디코드 경로 수치/게이트/프리빌드 검증.

CUDA·MPS 없이 CPU에서 검증한다. 융합 경로는 발동 조건에 디바이스(cuda/mps)를 포함하지만,
수치 검증은 내부 함수(_moe_infer_fused)를 직접 호출해 그 게이트를 우회한다.

수치 동일성 수준 (요구 1):
- N=1(디코드)은 **CPU에서** bitwise 동일(torch.equal). 토큰 1개당 expert별 단일행
  matmul이라 eager의 mm과 fused의 bmm이 같은 값을 낸다(전 시드 실측 확인). CUDA는
  재스택만으로 cuBLAS 커널 선택이 바뀌어 비트가 다를 수 있고, MPS는 실모델 토큰
  동일만 실측됐다 — 이 bitwise 단언은 CPU 계약이다.
- N>1은 eager가 같은 expert에 배정된 여러 토큰을 하나의 mm으로 묶는 반면 fused는
  (token,k) 쌍마다 [1,h] bmm을 돌려 BLAS 누적 순서가 달라진다. dtype 캐스트/최종
  가중합 순서는 moe_infer와 bitwise 동일하게 맞췄으므로 이 차이는 순수 matmul 누적
  round-off(fp32, 실측 max_rel ~2e-7)뿐 → rtol=1e-3으로 완화(대여유 마진).
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config  # noqa: E402
from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2MoE  # noqa: E402

HIDDEN = 32


def _build_moe(seed: int = 0, **overrides) -> DeepseekV2MoE:
    # 미니 config: n_routed_experts=8, num_experts_per_tok=2, hidden 32,
    # moe_intermediate 64, n_shared_experts=1. topk_method는 config 기본값이
    # 오타('gready')라 greedy 분기를 타도록 명시한다.
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        hidden_size=HIDDEN,
        moe_intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        topk_method="greedy",
        norm_topk_prob=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        ep_size=1,
        **overrides,
    )
    return DeepseekV2MoE(cfg).eval()


@torch.no_grad()
def _route(moe: DeepseekV2MoE, n_tokens: int, seed: int = 0):
    # 게이트로 실제 라우팅을 뽑아 eager/fused에 동일 입력·동일 라우팅을 먹인다.
    torch.manual_seed(seed)
    h = torch.randn(1, n_tokens, moe.config.hidden_size)
    topk_idx, topk_weight, _ = moe.gate(h)
    x = h.view(-1, moe.config.hidden_size)
    return x, topk_idx, topk_weight


class _FakeTensor:
    """_should_use_fused는 x.device.type / x.shape / topk_ids.shape만 본다.

    CPU에는 CUDA·MPS 텐서가 없으므로, 가속기 상황의 크기/ep 게이트를 검증하려면
    이 덕타이핑 스텁으로 대신한다(실제 연산은 하지 않음).
    """

    def __init__(self, *shape, device_type: str = "cuda"):
        self.shape = shape
        self.device = SimpleNamespace(type=device_type)


# ── 요구 1·2: 수치 동일성 (N=1 bitwise, N>1 fp round-off) ──


@pytest.mark.parametrize("n_tokens", [1, 4, 16])
def test_fused_matches_eager(n_tokens):
    moe = _build_moe()
    x, topk_idx, topk_weight = _route(moe, n_tokens)
    ref = moe.moe_infer(x, topk_idx, topk_weight)
    got = moe._moe_infer_fused(x, topk_idx, topk_weight)
    assert ref.shape == got.shape == (n_tokens, HIDDEN)
    if n_tokens == 1:
        # 디코드 소배치: bitwise 동일 (모듈 독스트링 참조)
        assert torch.equal(ref, got)
    else:
        # N>1: eager mm vs fused per-pair bmm의 fp32 누적차(실측 ~2e-7)만 허용
        assert torch.allclose(ref, got, rtol=1e-3, atol=1e-5)


def test_fused_bitwise_decode_multiple_seeds():
    # N=1 bitwise 동일성이 시드에 무관함(라우팅이 바뀌어도)을 확인.
    for seed in range(8):
        moe = _build_moe(seed=seed)
        x, topk_idx, topk_weight = _route(moe, 1, seed=seed)
        ref = moe.moe_infer(x, topk_idx, topk_weight)
        got = moe._moe_infer_fused(x, topk_idx, topk_weight)
        assert torch.equal(ref, got), f"seed={seed}"


# ── 요구 3: 뷰 재지정 후 기존 경로 불변 + 스택 스토리지 공유 ──


def test_view_rebind_preserves_eager_and_shares_storage():
    moe = _build_moe()
    x, topk_idx, topk_weight = _route(moe, 4)

    ref_before = moe.moe_infer(x, topk_idx, topk_weight)  # 개별 weight 텐서
    moe._build_fused_experts()  # gate/up/down weight를 스택 뷰로 재지정
    ref_after = moe.moe_infer(x, topk_idx, topk_weight)  # 뷰(동일 데이터)로 재실행

    # 데이터가 그대로 복사된 뷰이므로 기존 루프 결과는 bitwise 불변
    assert torch.equal(ref_before, ref_after)

    g, u, d = moe._fused_w
    for i, e in enumerate(moe.experts):
        # 각 expert weight가 스택 텐서 i번째 슬라이스와 같은 원소 포인터
        assert e.gate_proj.weight.data_ptr() == g[i].data_ptr()
        assert e.up_proj.weight.data_ptr() == u[i].data_ptr()
        assert e.down_proj.weight.data_ptr() == d[i].data_ptr()
        # 그리고 스택 텐서 내부(동일 스토리지)를 가리킴 = VRAM 중복 없음
        assert (
            e.gate_proj.weight.untyped_storage().data_ptr()
            == g.untyped_storage().data_ptr()
        )


def test_build_fused_idempotent():
    # 두 번째 호출은 재스택하지 않고 같은 스택 텐서를 유지(디바이스/dtype 일치 시).
    moe = _build_moe()
    moe._build_fused_experts()
    first = moe._fused_w
    moe._build_fused_experts()
    assert moe._fused_w is first


# ── 요구 4: 발동 조건(킬스위치·CUDA·크기·ep) ──


def test_env_default_on(monkeypatch):
    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    assert _build_moe()._fused_env_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "False", " no "])
def test_env_killswitch_off(monkeypatch, val):
    monkeypatch.setenv("OCR_MOE_FUSED", val)
    moe = _build_moe()
    assert moe._fused_env_enabled() is False
    # 킬스위치가 우선 — 다른 조건과 무관하게 발동 안 함
    assert moe._should_use_fused(_FakeTensor(1, HIDDEN), _FakeTensor(1, 2)) is False


def test_should_use_fused_skips_cpu(monkeypatch):
    # CPU 텐서는 legacy moe_infer 그대로 (가속기 전용 정책).
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()
    x, topk_idx, _ = _route(moe, 1)
    assert x.device.type == "cpu"
    assert moe._should_use_fused(x, topk_idx) is False


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_should_use_fused_on_accelerators(monkeypatch, device_type):
    # MPS도 허용 — P18의 레이어당 .tolist() 동기화(토큰당 11회)를 없애 M4 Max 디코드
    # 1.77x(토큰 동일 실측, audit MPS-2). 그 외 디바이스 타입은 발동 안 함.
    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    moe = _build_moe()
    x = _FakeTensor(1, HIDDEN, device_type=device_type)
    assert moe._should_use_fused(x, _FakeTensor(1, 2)) is True
    xla = _FakeTensor(1, HIDDEN, device_type="xla")
    assert moe._should_use_fused(xla, _FakeTensor(1, 2)) is False


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_should_use_fused_size_gate(monkeypatch, device_type):
    # 가속기(스텁)에서 **N==1(디코드)만** 발동 검증 — N>1은 mm↔bmm 누적
    # 순서차로 근접 argmax가 뒤집혀 E2E가 갈라진 실측(표 colspan 손상) 때문에 제외.
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()

    def x(n):
        return _FakeTensor(n, HIDDEN, device_type=device_type)

    assert moe._should_use_fused(x(1), _FakeTensor(1, 2)) is True
    assert moe._should_use_fused(x(2), _FakeTensor(2, 2)) is False
    assert moe._should_use_fused(x(32), _FakeTensor(32, 2)) is False


def test_should_use_fused_ep_gate(monkeypatch):
    # ep_size>1(전문가 병렬)에서는 융합 경로 미발동 → 기존 all-to-all moe_infer.
    monkeypatch.setenv("OCR_MOE_FUSED", "1")
    moe = _build_moe()
    moe.ep_size = 2
    assert moe._should_use_fused(_FakeTensor(1, HIDDEN), _FakeTensor(1, 2)) is False


# ── forward 통합: CPU eval은 기존 eager 경로로 정상 동작(엣지 회귀 가드) ──


def test_forward_eval_cpu_runs():
    moe = _build_moe()
    h = torch.randn(1, 4, moe.config.hidden_size)
    with torch.no_grad():
        y = moe(h)
    assert y.shape == h.shape
    assert torch.isfinite(y).all()


# ── 로드 시점 프리빌드(prebuild_fused_moe) — 메모리 중복 없음 + 수치 불변 ──


def _tiny_moe_model(seed: int = 0):
    """MoE 2층(첫 층 dense) tiny DeepseekV2Model — 프리빌드가 모든 MoE 레이어를 찾는지 본다."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2Model

    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        first_k_dense_replace=1, moe_layer_freq=1, topk_method="greedy",
        scoring_func="softmax", n_group=1, topk_group=1, hidden_act="silu",
        aux_loss_alpha=0.0, use_mla=False, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return DeepseekV2Model(cfg).eval()


def _moe_layers(model):
    return [m for m in model.modules() if isinstance(m, DeepseekV2MoE)]


def _expert_storages(moe):
    return {
        p.untyped_storage().data_ptr()
        for e in moe.experts
        for p in (e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight)
    }


def test_prebuild_rebinds_every_expert_as_view_without_duplication(monkeypatch):
    """프리빌드 후 expert 가중치는 레이어당 스택 3개(gate/up/down)의 뷰뿐 — 개별 버퍼 0개.
    스택 바이트 = 원래 expert 가중치 바이트(복사본이 남지 않음)."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    model = _tiny_moe_model()
    layers = _moe_layers(model)
    assert len(layers) == 2
    before_bytes = sum(
        p.numel() * p.element_size()
        for moe in layers for e in moe.experts
        for p in (e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight)
    )
    assert len(_expert_storages(layers[0])) == 8 * 3  # 프리빌드 전: expert별 개별 버퍼

    assert prebuild_fused_moe(model, "cpu") == 2
    stacked_bytes = 0
    for moe in layers:
        g, u, d = moe._fused_w
        assert _expert_storages(moe) == {
            g.untyped_storage().data_ptr(),
            u.untyped_storage().data_ptr(),
            d.untyped_storage().data_ptr(),
        }
        for i, e in enumerate(moe.experts):
            assert e.gate_proj.weight.data_ptr() == g[i].data_ptr()
            assert e.up_proj.weight.data_ptr() == u[i].data_ptr()
            assert e.down_proj.weight.data_ptr() == d[i].data_ptr()
        stacked_bytes += sum(t.untyped_storage().nbytes() for t in (g, u, d))
    assert stacked_bytes == before_bytes


def test_prebuild_keeps_model_forward_bitwise_identical_on_cpu(monkeypatch):
    """뷰 재지정 후 프리필(N>1)·디코드(N==1) 모두 legacy와 비트 동일(CPU 계약)."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    seq = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        ref = _tiny_moe_model()(input_ids=seq, use_cache=False).last_hidden_state
        model = _tiny_moe_model()
        prebuild_fused_moe(model, "cpu")
        got = model(input_ids=seq, use_cache=False).last_hidden_state
    assert torch.equal(ref, got)


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_fused_matches_legacy_with_prebuilt_views(monkeypatch, n_tokens):
    """프리빌드(뷰 공유) 상태에서도 fused ↔ legacy 등가 — N=1 bitwise, N>1 round-off만."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    model = _tiny_moe_model(seed=3)
    prebuild_fused_moe(model, "cpu")
    for moe in _moe_layers(model):
        x, topk_idx, topk_weight = _route(moe, n_tokens, seed=n_tokens)
        ref = moe.moe_infer(x, topk_idx, topk_weight)
        got = moe._moe_infer_fused(x, topk_idx, topk_weight)
        if n_tokens == 1:
            assert torch.equal(ref, got)
        else:
            assert torch.allclose(ref, got, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("val", ["0", "off", "False"])
def test_prebuild_respects_kill_switch(monkeypatch, val):
    """OCR_MOE_FUSED=0 → 프리빌드 없음(legacy 개별 가중치 그대로 = 완전 legacy 복원)."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

    monkeypatch.setenv("OCR_MOE_FUSED", val)
    model = _tiny_moe_model()
    assert prebuild_fused_moe(model, "cpu") == 0
    for moe in _moe_layers(model):
        assert getattr(moe, "_fused_w", None) is None
        assert len(_expert_storages(moe)) == 8 * 3


def test_prebuild_then_lazy_build_is_a_noop(monkeypatch):
    """프리빌드한 스택은 첫 디코드의 지연 빌드(device=None)가 다시 만들지 않는다."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import prebuild_fused_moe

    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    model = _tiny_moe_model()
    prebuild_fused_moe(model, "cpu")
    for moe in _moe_layers(model):
        first = moe._fused_w
        moe._build_fused_experts()
        assert moe._fused_w is first


def test_env_parse_shared_with_module_helper(monkeypatch):
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import fused_moe_env_enabled

    for value, expected in (("", True), ("1", True), ("bogus", True), ("0", False), (" OFF ", False)):
        monkeypatch.setenv("OCR_MOE_FUSED", value)
        assert fused_moe_env_enabled() is expected, value
        assert _build_moe()._fused_env_enabled() is expected, value
