from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers.cache_utils import DynamicCache  # noqa: E402

from app.engine.fast_decode import fast_greedy_decode  # noqa: E402

VOCAB = 16
EOS = 9


class StubModel:
    """(마지막 토큰+1) % 8 을 다음 토큰으로 내는 결정적 스텁.
    eos_after 스텝을 넘기면 EOS(9)를 낸다. 캐시 소비량을 DynamicCache 속성으로 추적."""

    def __init__(self, eos_after=None):
        self.eos_after = eos_after
        self.steps = 0
        self.generate_called = False

    def generate(self, **kw):
        self.generate_called = True
        return kw["input_ids"]

    def prepare_inputs_for_generation(self, input_ids, **kw):
        past = kw.get("past_key_values")
        seen = getattr(past, "seen", 0)
        return {"input_ids": input_ids[:, seen:], "past_key_values": past}

    def __call__(self, input_ids=None, past_key_values=None, return_dict=True, **kw):
        past_key_values.seen = getattr(past_key_values, "seen", 0) + input_ids.shape[1]
        self.steps += 1
        last = int(input_ids[0, -1])
        if self.eos_after is not None and self.steps > self.eos_after:
            target = EOS
        else:
            target = (last + 1) % 8
        logits = torch.full((1, input_ids.shape[1], VOCAB), -10.0)
        logits[0, -1, target] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class StubStreamer:
    def __init__(self):
        self.puts: list[list[int]] = []
        self.ended = False

    def put(self, value):
        v = value[0] if value.dim() > 1 else value
        self.puts.append(v.tolist())

    def end(self):
        self.ended = True


def _run(model, prompt=(0, 1), block=4, **overrides):
    gen_kwargs = {
        "input_ids": torch.tensor([list(prompt)], dtype=torch.long),
        "eos_token_id": EOS,
        "max_length": 64,
        "do_sample": False,
        **overrides,
    }
    return fast_greedy_decode(model, gen_kwargs, block=block)


def test_generates_until_eos_and_trims_overrun():
    model = StubModel(eos_after=10)
    out = _run(model, block=4)
    # 10 스텝 체인 + 11번째 EOS. 블록 경계(12)까지 1개 초과 생성됐다가 절단됨
    assert out.shape == (1, 13)
    assert int(out[0, -1]) == EOS
    assert EOS not in out[0, :-1].tolist()
    expected_chain = [2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
    assert out[0, 2:-1].tolist() == expected_chain


def test_streamer_receives_prompt_then_tokens_in_order():
    model = StubModel(eos_after=10)
    streamer = StubStreamer()
    out = _run(model, block=4, streamer=streamer)
    assert streamer.ended
    assert streamer.puts[0] == [0, 1]  # 프롬프트 (skip_prompt 스트리머 소비용)
    streamed = [t for chunk in streamer.puts[1:] for t in chunk]
    assert streamed == out[0, 2:].tolist()  # EOS 포함, 초과분/중복/누락 없음


def test_logits_processors_applied_every_step():
    calls = []

    class CountingProc:
        def __call__(self, ids, scores):
            calls.append(ids.shape[1])
            return scores

    model = StubModel(eos_after=6)
    out = _run(model, block=4, logits_processor=[CountingProc()])
    generated_steps = model.steps
    assert len(calls) == generated_steps
    assert int(out[0, -1]) == EOS


def test_stopping_criteria_checked_at_block_boundary():
    model = StubModel(eos_after=None)  # EOS 없음

    class StopAt8:
        def __call__(self, ids, scores, **kw):
            return ids.shape[1] >= 2 + 8

    out = _run(model, block=4, stopping_criteria=[StopAt8()])
    assert out.shape == (1, 2 + 8)  # 블록 경계(8토큰)에서 정지


def test_max_length_cap_flushes_pending():
    model = StubModel(eos_after=None)
    streamer = StubStreamer()
    out = _run(model, block=4, streamer=streamer, max_length=9)
    assert out.shape == (1, 9)
    streamed = [t for chunk in streamer.puts[1:] for t in chunk]
    assert streamed == out[0, 2:].tolist()  # 7토큰(블록 4+3) 전부 스트림됨


def test_do_sample_falls_back_to_hf_generate():
    model = StubModel()
    out = _run(model, do_sample=True)
    assert model.generate_called
    assert out.tolist() == [[0, 1]]


def test_block_one_equals_larger_block_output():
    a = _run(StubModel(eos_after=9), block=1)
    b = _run(StubModel(eos_after=9), block=7)
    assert a.tolist() == b.tolist()  # 블록 크기는 결과에 영향 없음 (동기화 빈도만)


class RecordingModel(StubModel):
    """prepare_inputs_for_generation에 전달된 kwargs를 스텝별로 기록하는 스텁.
    attention_mask 전달 여부, position_ids 값 스냅샷, position_ids 텐서 객체를 남긴다."""

    def __init__(self, eos_after=None):
        super().__init__(eos_after=eos_after)
        self.saw_attention_mask = False
        self.position_id_values: list[list[int]] = []
        self.position_id_tensors: list = []  # 참조 유지 → id() 재사용(GC) 방지

    def prepare_inputs_for_generation(self, input_ids, **kw):
        if "attention_mask" in kw:
            self.saw_attention_mask = True
        pos = kw.get("position_ids")
        self.position_id_values.append(pos[0].tolist())  # 호출 시점 값 스냅샷
        self.position_id_tensors.append(pos)             # 객체 자체 보관 (아래 id 검사용)
        return super().prepare_inputs_for_generation(input_ids, **kw)


def test_passes_position_ids_and_no_attention_mask():
    """attention_mask를 넘기지 않고, position_ids가 프리필 [0..L-1] → 이후 [L],[L+1],…로 증가."""
    model = RecordingModel(eos_after=6)
    prompt = (0, 1, 2)  # L=3
    _run(model, prompt=prompt, block=4)

    assert model.saw_attention_mask is False  # (a) attention_mask 미전달
    assert model.position_id_values[0] == [0, 1, 2]  # (b) 프리필: 전체 위치
    # 이후 디코드 스텝: [3], [4], [5], … 단조 증가
    for step, value in enumerate(model.position_id_values[1:], start=len(prompt)):
        assert value == [step]


def test_position_ids_is_a_fresh_tensor_each_step():
    """(c) 매 스텝 position_ids는 직전과 다른 새 텐서 객체 — P19 rotary 캐시가
    id(position_ids)로 스텝을 식별하므로 in-place 증가(동일 id 재사용)는 회귀다."""
    model = RecordingModel(eos_after=6)
    _run(model, prompt=(0, 1, 2), block=4)

    tensors = model.position_id_tensors
    assert len(tensors) >= 3  # 프리필 + 최소 몇 스텝은 돌아야 의미 있는 검사
    ids = [id(t) for t in tensors]  # 전부 살아있어 id 재사용 불가
    assert len(set(ids)) == len(ids)  # 모든 스텝이 서로 다른 객체


# ── CUDA Graph 경로(U2) 발동 조건 — env·is_cuda 스텁으로 분기만 검증 (CUDA 불필요) ──

def _graph_ready_stubs():
    ids = SimpleNamespace(is_cuda=True)
    proc = SimpleNamespace(ngram_size=35, window=128)
    model = SimpleNamespace(config=SimpleNamespace(_ring_window=128))
    return ids, [proc], model


def test_graph_precheck_env_and_inputs(monkeypatch):
    from app.engine.fast_decode import _should_try_cuda_graph

    ids, procs, model = _graph_ready_stubs()
    monkeypatch.delenv("OCR_CUDA_GRAPHS", raising=False)
    assert _should_try_cuda_graph(ids, procs, model, 8) is True  # 기본 on

    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("OCR_CUDA_GRAPHS", off)
        assert _should_try_cuda_graph(ids, procs, model, 8) is False
    monkeypatch.setenv("OCR_CUDA_GRAPHS", "1")
    assert _should_try_cuda_graph(ids, procs, model, 8) is True
    monkeypatch.delenv("OCR_CUDA_GRAPHS", raising=False)

    # CPU 입력 → eager
    assert _should_try_cuda_graph(SimpleNamespace(is_cuda=False), procs, model, 8) is False
    # 프로세서 0개/2개 → 우리 ngram 하나라는 가정 위배 → eager
    assert _should_try_cuda_graph(ids, [], model, 8) is False
    assert _should_try_cuda_graph(ids, procs * 2, model, 8) is False
    # window/ngram_size 없는 프로세서 → eager
    assert _should_try_cuda_graph(ids, [object()], model, 8) is False
    # 링 윈도우 미설정 모델 → eager
    assert _should_try_cuda_graph(ids, procs, SimpleNamespace(config=SimpleNamespace()), 8) is False
    # block > ngram window → 블록 회수(recent)가 링 용량을 넘음 → eager
    assert _should_try_cuda_graph(ids, procs, model, 129) is False
    assert _should_try_cuda_graph(ids, procs, model, 128) is True


def test_graph_precheck_respects_graph_capable_marker(monkeypatch):
    """graph_capable=False 마커(OCR_NGRAM_HOST 절연 레버) 프로세서는 그래프 금지 —
    호스트 강제 절연이 GPU 상주 GraphNgram으로 조용히 대체되지 않게 한다."""
    from app.engine.fast_decode import _should_try_cuda_graph
    from app.native_ops import make_ngram_logits_processor

    ids, procs, model = _graph_ready_stubs()
    monkeypatch.delenv("OCR_CUDA_GRAPHS", raising=False)

    banned = SimpleNamespace(ngram_size=35, window=128, graph_capable=False)
    assert _should_try_cuda_graph(ids, [banned], model, 8) is False
    # 마커가 True거나 없으면 기존 동작(그래프 허용) 유지
    allowed = SimpleNamespace(ngram_size=35, window=128, graph_capable=True)
    assert _should_try_cuda_graph(ids, [allowed], model, 8) is True
    assert _should_try_cuda_graph(ids, procs, model, 8) is True

    # 실제 강제 호스트 프로세서(OCR_NGRAM_HOST=1)도 동일하게 게이트된다
    monkeypatch.setenv("OCR_NGRAM_HOST", "1")
    forced = make_ngram_logits_processor(35, 128, "cuda")
    assert _should_try_cuda_graph(ids, forced, model, 8) is False


def test_cpu_decode_never_enters_graph_path(monkeypatch):
    """CPU 스텁 플로우는 그래프 함수를 절대 타지 않는다 (호출되면 즉시 실패)."""
    import app.engine.fast_decode as fd

    def boom(*a, **k):
        raise AssertionError("graph path must not trigger on cpu")

    monkeypatch.setattr(fd, "_cuda_graph_greedy_decode", boom)
    model = StubModel(eos_after=6)
    out = _run(model, block=4)
    assert int(out[0, -1]) == EOS


# ── 그래프 캡처 전제(P19 rotary 캐시 × 정적 pos 버퍼, P20 링 텐서 상태) ──
# tiny DeepseekV2 모델(CPU)로 벤더 어텐션의 실코드를 구동해 검증한다.

def _tiny_ring_model():
    from app.vendor.unlimited_ocr.configuration_deepseek_v2 import DeepseekV2Config
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2Model

    torch.manual_seed(0)
    cfg = DeepseekV2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        first_k_dense_replace=99,  # 전 레이어 dense MLP → MoE 경로와 무관
        moe_layer_freq=1, topk_method="greedy", scoring_func="softmax",
        n_group=1, topk_group=1, hidden_act="silu", aux_loss_alpha=0.0,
        use_mla=False,  # mha_eager → SlidingWindowLlamaAttention (P19·P20 대상)
        max_position_embeddings=64, rope_theta=10000.0,
    )
    cfg._attn_implementation = "eager"
    model = DeepseekV2Model(cfg).eval()
    assert type(model.layers[0].self_attn).__name__ == "SlidingWindowLlamaAttention"
    model.config._ring_window = 2  # 링 경로 활성 (워밍업 2스텝 후 정상상태)
    return model


def _drive_decode(model, seq, pos_mode, rope_calls=None):
    """prefill 4토큰 후 나머지를 티처포싱 디코드. pos_mode:
    'fresh'  = 매 스텝 새 position_ids 텐서 (eager 계약)
    'static' = 단일 [1,1] 버퍼를 in-place add_ (그래프 캡처 계약 — id 불변)
    반환: 스텝별 last_hidden_state 목록, 최종 cache."""
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=seq[:, :4], use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    hs = []
    static_pos = torch.tensor([[4]], dtype=torch.long)
    per_step_calls = []
    for step in range(4, seq.shape[1]):
        pos = torch.tensor([[step]], dtype=torch.long) if pos_mode == "fresh" else static_pos
        if rope_calls is not None:
            rope_calls["n"] = 0
        with torch.no_grad():
            out = model(input_ids=seq[:, step:step + 1], use_cache=True,
                        past_key_values=cache, position_ids=pos)
        cache = out.past_key_values
        hs.append(out.last_hidden_state.clone())
        if rope_calls is not None:
            per_step_calls.append(rope_calls["n"])
        if pos_mode == "static":
            static_pos.add_(1)  # 그래프의 pos_ids.add_(1)과 동일 — 같은 객체 재사용
    return hs, cache, per_step_calls


def test_p19_rope_recomputed_with_static_position_buffer(monkeypatch):
    """그래프 캡처 전제 검증: 동일 position_ids 객체(id 불변)를 in-place 증분해도
    ① layer 0가 스텝마다 rotary를 재계산하고(→ 캡처에 rotary 커널이 기록됨)
    ② 출력이 fresh-텐서 스텝과 비트 동일하다(스테일 cos/sin 없음)."""
    from transformers.models.llama import modeling_llama

    calls = {"n": 0}
    real = modeling_llama.LlamaRotaryEmbedding.forward

    def counting(self, x, position_ids):
        calls["n"] += 1
        return real(self, x, position_ids)

    monkeypatch.setattr(modeling_llama.LlamaRotaryEmbedding, "forward", counting)

    seq = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(42))
    hs_fresh, _, _ = _drive_decode(_tiny_ring_model(), seq, "fresh")
    hs_static, _, static_calls = _drive_decode(_tiny_ring_model(), seq, "static", rope_calls=calls)

    # ① 정적 버퍼(같은 id)여도 매 스텝 rotary 1회 재계산 — 0회(스테일 재사용)면 캡처가 위치를 못 따라간다
    assert static_calls == [1] * len(static_calls)
    # ② 값 동일 — P19 캐시가 위치 갱신을 놓치면 여기서 갈라진다 (워밍업 2 + 링 정상상태 4 스텝 포함)
    for a, b in zip(hs_fresh, hs_static):
        assert torch.equal(a, b)


def test_p20_ring_state_defaults_to_host_int_and_wraps():
    """P20 기본(eager): 링 슬롯은 호스트 int(_ring_pos) 하나뿐이고 텐서 슬롯은 없다 —
    MPS에서 index_copy_가 KV 길이에 비례하는 비용(8쪽 청크 토큰당 15.9ms)이라 텐서
    슬롯은 CUDA Graph 모드에서만 쓴다. 정상상태 스텝마다 +1 mod W로 랩어라운드하고
    캐시 길이는 prefill+W로 고정된다."""
    model = _tiny_ring_model()  # W=2
    seq = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(7))
    _, cache, _ = _drive_decode(model, seq, "fresh")

    assert not hasattr(cache, "_ring_pos_t")  # 이중 상태 금지 — 텐서 표현 없음
    assert getattr(cache, "_ring_tensor_mode", False) is False
    assert set(cache._ring_pos) == {0, 1}
    assert all(type(v) is int for v in cache._ring_pos.values())
    # 디코드 6스텝 = 워밍업 2 + 정상상태 4 → ring_pos = 4 % 2 = 0
    assert cache._ring_pos[0] == 0
    assert cache.get_seq_length() == 4 + 2  # prefill(4)+W(2)로 고정 — 정상상태에서 불변


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_p20_ring_multi_token_chunk_advances_slot(tensor_mode):
    """정상상태에서 q_len=2 조각 투입 — t-루프가 두 모드 모두 슬롯을 2회 전진(랩 포함)."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import ring_slots_to_tensor

    model = _tiny_ring_model()  # W=2
    seq = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(3))
    _, cache, _ = _drive_decode(model, seq, "fresh")  # 디코드 5스텝 → ring_pos = 3 % 2 = 1
    assert cache._ring_pos[0] == 1
    if tensor_mode:
        ring_slots_to_tensor(cache)

    chunk = torch.randint(0, 64, (1, 2), generator=torch.Generator().manual_seed(4))
    pos = torch.tensor([[9, 10]], dtype=torch.long)
    with torch.no_grad():
        model(input_ids=chunk, use_cache=True, past_key_values=cache, position_ids=pos)
    slots = cache._ring_pos_t if tensor_mode else cache._ring_pos
    assert int(slots[0]) == (1 + 2) % 2  # 두 슬롯 전진 + 랩
    assert cache.get_seq_length() == 4 + 2  # 여전히 고정 길이(링 덮어쓰기)


def test_p20_mode_switch_keeps_single_state():
    """int→텐서→int 전환은 값을 보존하고 항상 한 표현만 남긴다(그래프 진입·폴백 계약)."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import ring_slots_to_int, ring_slots_to_tensor

    model = _tiny_ring_model()  # W=2
    seq = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(5))
    _, cache, _ = _drive_decode(model, seq, "fresh")
    before = dict(cache._ring_pos)

    ring_slots_to_tensor(cache)
    assert not hasattr(cache, "_ring_pos") and cache._ring_tensor_mode is True
    for li, t in cache._ring_pos_t.items():
        assert torch.is_tensor(t) and t.dtype == torch.long and t.dim() == 0
        assert int(t) == before[li]

    ring_slots_to_int(cache)
    assert not hasattr(cache, "_ring_pos_t") and cache._ring_tensor_mode is False
    assert cache._ring_pos == before


def _drive_with_switches(model, seq, switches):
    """prefill 4토큰 후 티처포싱 디코드 — switches {step: fn(cache)}로 스텝 직전 모드 전환."""
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=seq[:, :4], use_cache=True, past_key_values=cache)
    cache = out.past_key_values
    hs = []
    for step in range(4, seq.shape[1]):
        if step in switches:
            switches[step](cache)
        pos = torch.tensor([[step]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=seq[:, step:step + 1], use_cache=True,
                        past_key_values=cache, position_ids=pos)
        cache = out.past_key_values
        hs.append(out.last_hidden_state.clone())
    return hs


def test_p20_tensor_and_int_slots_are_bitwise_identical_across_switches():
    """링 여러 바퀴 동안 int→텐서(그래프 진입)→int(폴백) 전환이 섞여도 순수 int 실행과
    스텝별 출력이 비트 동일 — 두 경로가 같은 슬롯에 같은 K/V를 쓴다."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import ring_slots_to_int, ring_slots_to_tensor

    seq = torch.randint(0, 64, (1, 20), generator=torch.Generator().manual_seed(11))
    ref = _drive_with_switches(_tiny_ring_model(), seq, {})
    mixed = _drive_with_switches(
        _tiny_ring_model(), seq, {8: ring_slots_to_tensor, 13: ring_slots_to_int}
    )
    assert len(ref) == len(mixed) == 16
    for a, b in zip(ref, mixed):
        assert torch.equal(a, b)


def test_p20_int_mode_never_uses_index_copy(monkeypatch):
    """기본 eager 경로는 index_copy_를 호출하지 않는다(MPS O(KV) 비용 회귀 방지).
    텐서 모드(그래프)에서만 index_copy_가 쓰인다."""
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import ring_slots_to_tensor

    calls = {"n": 0}
    real = torch.Tensor.index_copy_

    def spy(self, *args, **kwargs):
        calls["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_copy_", spy)
    seq = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(2))
    _drive_with_switches(_tiny_ring_model(), seq, {})
    assert calls["n"] == 0
    _drive_with_switches(_tiny_ring_model(), seq, {7: ring_slots_to_tensor})
    assert calls["n"] > 0


def test_graph_precheck_skips_when_moe_kill_switch_off(monkeypatch):
    """OCR_MOE_FUSED=0(벤더 P17 킬스위치)이면 그래프 경로를 아예 타지 않는다.

    legacy `moe_infer`는 레이어마다 `tokens_per_expert.cpu().numpy()`로 호스트
    동기화를 하므로 CUDA Graph 캡처가 **매 청크 확정 실패**한다 — 워밍업을 버리고
    traceback을 남긴 뒤 eager로 폴백할 뿐 아니라, 0/1 파리티 비교가 MoE 경로가
    아니라 graph↔eager 차이를 재게 된다."""
    from app.engine.fast_decode import _graph_skip_reason, _should_try_cuda_graph

    ids, procs, model = _graph_ready_stubs()
    monkeypatch.delenv("OCR_CUDA_GRAPHS", raising=False)

    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("OCR_MOE_FUSED", off)
        assert _should_try_cuda_graph(ids, procs, model, 8) is False
        assert "OCR_MOE_FUSED" in _graph_skip_reason(ids, procs, model, 8)
    # 기본(미설정)·명시 on은 기존대로 그래프 허용
    for on in ("1", "true", ""):
        monkeypatch.setenv("OCR_MOE_FUSED", on)
        assert _should_try_cuda_graph(ids, procs, model, 8) is True
    monkeypatch.delenv("OCR_MOE_FUSED", raising=False)
    assert _should_try_cuda_graph(ids, procs, model, 8) is True


def test_moe_kill_switch_parse_matches_vendor(monkeypatch):
    """킬스위치 해석이 벤더 `DeepseekV2MoE._fused_env_enabled`와 어긋나면
    한쪽만 legacy로 도는 조합(=캡처 확정 실패)이 다시 생긴다."""
    from app.engine.fast_decode import _moe_fused_enabled
    from app.vendor.unlimited_ocr.modeling_deepseekv2 import DeepseekV2MoE

    class _Probe:
        _fused_env = None

    for value in ("0", "false", "no", "off", "OFF", " 0 ", "1", "true", "yes", "", "bogus"):
        monkeypatch.setenv("OCR_MOE_FUSED", value)
        probe = _Probe()
        assert _moe_fused_enabled() is DeepseekV2MoE._fused_env_enabled(probe), value


# ── MPS 오토릴리스 풀 배선 (gap1-metal-real-e2e-1) — MPS 없이 스파이로 검증 ──

def _spy_pool(events):
    import contextlib

    @contextlib.contextmanager
    def pool(enabled=True):
        events.append(("enter", enabled))
        try:
            yield
        finally:
            events.append(("exit", enabled))

    return pool


def test_needs_autorelease_pool_only_for_mps_tensors():
    from app.engine.fast_decode import _needs_autorelease_pool

    assert _needs_autorelease_pool(SimpleNamespace(device=SimpleNamespace(type="mps"))) is True
    assert _needs_autorelease_pool(torch.zeros(1, dtype=torch.long)) is False
    assert _needs_autorelease_pool(SimpleNamespace(device=SimpleNamespace(type="cuda"))) is False
    assert _needs_autorelease_pool(object()) is False


def test_mps_decode_drains_autorelease_pool_every_step(monkeypatch):
    """MPS 입력이면 프롬프트 put + 디코드 스텝(프리필 포함)마다 풀을 1회씩 push/pop —
    끝나지 않는 워커 스레드에 ObjC 객체가 토큰당 ~25KB씩 쌓이던 누수의 회귀 방지."""
    import app.engine.fast_decode as fd

    events: list = []
    monkeypatch.setattr(fd, "autorelease_pool", _spy_pool(events))
    monkeypatch.setattr(fd, "_needs_autorelease_pool", lambda ids: True)
    model = StubModel(eos_after=10)
    out = _run(model, block=4, streamer=StubStreamer())

    assert int(out[0, -1]) == EOS  # 출력 계약 불변
    enters = [e for e in events if e[0] == "enter"]
    assert enters == [("enter", True)] * (model.steps + 1)
    assert events.count(("exit", True)) == len(enters)  # 짝이 맞음(LIFO, 누락 없음)


def test_cpu_decode_keeps_pool_disabled(monkeypatch):
    import app.engine.fast_decode as fd

    events: list = []
    monkeypatch.setattr(fd, "autorelease_pool", _spy_pool(events))
    _run(StubModel(eos_after=6), block=4, streamer=StubStreamer())
    assert events and all(enabled is False for _, enabled in events)


# ── 길이 상한 판정(hit_length_limit) — 잘린 출력을 정상 결과로 채택하지 않기 위한 계약 ──

def test_hit_length_limit_flags_only_eosless_max_length_outputs():
    from app.engine.fast_decode import hit_length_limit

    capped = _run(StubModel(eos_after=None), block=4, max_length=9)
    assert capped.shape == (1, 9)
    assert hit_length_limit(capped, {"max_length": 9, "eos_token_id": EOS}) is True

    finished = _run(StubModel(eos_after=3), block=4, max_length=64)
    assert int(finished[0, -1]) == EOS
    assert hit_length_limit(finished, {"max_length": 64, "eos_token_id": EOS}) is False

    # 상한 바로 그 자리에서 EOS로 끝났으면 잘린 것이 아니다
    exact = torch.tensor([[0, 1, 2, EOS]])
    assert hit_length_limit(exact, {"max_length": 4, "eos_token_id": EOS}) is False
    # EOS id 목록·EOS 미지정
    assert hit_length_limit(exact, {"max_length": 4, "eos_token_id": [3, EOS]}) is False
    assert hit_length_limit(exact, {"max_length": 4, "eos_token_id": None}) is True
    # 중단 기준(취소·반복)으로 상한 전에 멈춘 출력은 상한 도달이 아니다
    stopped = torch.tensor([[0, 1, 2]])
    assert hit_length_limit(stopped, {"max_length": 9, "eos_token_id": EOS}) is False
