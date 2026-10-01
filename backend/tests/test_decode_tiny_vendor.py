"""실제 벤더 모델 코드(tiny UnlimitedOCRForCausalLM)로 디코드 배선을 CPU에서 검증한다.

StubModel 기반 test_fast_decode.py는 벤더 prepare_inputs_for_generation의 링 분기·
P19 rotary 캐시·P20 링 슬롯·P21 logits 슬라이싱을 거치지 않는다. 여기서는 SAM/CLIP을
빼고(빌더를 None으로 대체 — 이미지 없는 텍스트 프롬프트만 쓴다) 2층 tiny 모델을 만들어
fast_greedy_decode·CUDA Graph 오케스트레이션(가짜 torch.cuda)을 실코드로 구동한다.
"""

import contextlib

import pytest

torch = pytest.importorskip("torch")

import app.engine.fast_decode as fd  # noqa: E402
from app.native_ops import TorchSlidingWindowNoRepeatNgram  # noqa: E402

PROMPT = [0, 5, 7, 9, 11, 13, 17]


def _tiny_vendor_model(ring_window: int = 3):
    import app.vendor.unlimited_ocr.modeling_unlimitedocr as mu

    builders = (mu.build_sam_vit_b, mu.build_clip_l)
    mu.build_sam_vit_b = lambda: None  # 비전 타워 없이 생성 (텍스트 프롬프트 전용)
    mu.build_clip_l = lambda: None
    try:
        torch.manual_seed(0)
        cfg = mu.UnlimitedOCRConfig(
            vocab_size=48, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
            first_k_dense_replace=1, moe_layer_freq=1,  # layer0 dense + layer1 MoE
            topk_method="greedy", scoring_func="softmax", n_group=1, topk_group=1,
            hidden_act="silu", aux_loss_alpha=0.0, use_mla=False,
            max_position_embeddings=256, rope_theta=10000.0,
            bos_token_id=0, eos_token_id=1, pad_token_id=1,
        )
        cfg._attn_implementation = "eager"
        model = mu.UnlimitedOCRForCausalLM(cfg).eval()
    finally:
        mu.build_sam_vit_b, mu.build_clip_l = builders
    # infer/infer_multi가 generate 직전에 하는 설정과 동일 (링 윈도우 활성)
    model.config._ring_window = ring_window
    model.config.sliding_window = None
    return model


def _gen_kwargs(max_length: int, eos=None, ngram=(3, 16)):
    return {
        "input_ids": torch.tensor([PROMPT], dtype=torch.long),
        "images": None,
        "images_seq_mask": None,
        "images_spatial_crop": None,
        "do_sample": False,
        "eos_token_id": eos,
        "max_length": max_length,
        "use_cache": True,
        "logits_processor": [TorchSlidingWindowNoRepeatNgram(*ngram)],
    }


class _Streamer:
    def __init__(self) -> None:
        self.puts: list[list[int]] = []
        self.ended = False

    def put(self, value) -> None:
        self.puts.append((value[0] if value.dim() > 1 else value).tolist())

    def end(self) -> None:
        self.ended = True

    def generated(self) -> list[int]:
        return [t for chunk in self.puts[1:] for t in chunk]


def _install_fake_cuda(monkeypatch, capture_error: Exception | None = None):
    """CUDA 없이 그래프 오케스트레이션을 돌리는 최소 스텁.

    스트림은 no-op, 캡처(torch.cuda.graph)는 capture_error를 던진다 — 캡처 실패 →
    eager 폴백 경로를 실코드로 구동하기 위함(캡처 성공·리플레이는 흉내 내지 않는다)."""

    class _Stream:
        def wait_stream(self, other) -> None:
            return None

    class _Graph:
        def replay(self) -> None:  # pragma: no cover - 캡처가 항상 실패하므로 도달 불가
            raise AssertionError("replay는 이 스텁에서 도달하면 안 된다")

    @contextlib.contextmanager
    def _graph(graph, *args, **kwargs):
        raise capture_error or RuntimeError("capture unsupported (test stub)")
        yield  # pragma: no cover

    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: _Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    monkeypatch.setattr(torch.cuda, "graph", _graph)
    monkeypatch.setattr(fd, "_should_try_cuda_graph", lambda *a, **k: True)


def _eager_reference(max_length: int, eos=None, block: int = 4):
    streamer = _Streamer()
    kwargs = _gen_kwargs(max_length, eos)
    kwargs["streamer"] = streamer
    out = fd.fast_greedy_decode(_tiny_vendor_model(), kwargs, block=block)
    return out[0].tolist(), streamer


def test_graph_capture_failure_falls_back_with_int_ring_slots(monkeypatch):
    """캡처 실패 → eager 폴백: 출력·스트림이 순수 eager와 동일하고, 그래프 단계에서
    텐서로 전환했던 링 슬롯이 int로 복귀한다(P20 단일 상태)."""
    ref, ref_streamer = _eager_reference(max_length=40)

    _install_fake_cuda(monkeypatch)
    import app.vendor.unlimited_ocr.modeling_deepseekv2 as md

    seen_modes: list[bool] = []
    real_to_int = md.ring_slots_to_int

    def spy_to_int(cache):
        seen_modes.append(bool(getattr(cache, "_ring_tensor_mode", False)))
        real_to_int(cache)
        seen_modes.append(bool(getattr(cache, "_ring_tensor_mode", False)))

    monkeypatch.setattr(md, "ring_slots_to_int", spy_to_int)
    streamer = _Streamer()
    kwargs = _gen_kwargs(40)
    kwargs["streamer"] = streamer
    out = fd.fast_greedy_decode(_tiny_vendor_model(), kwargs, block=4)

    assert out[0].tolist() == ref
    assert streamer.generated() == ref_streamer.generated() == ref[len(PROMPT):]
    assert streamer.ended
    assert seen_modes == [True, False]  # 그래프 단계는 텐서 모드, 폴백 후 int 모드


def test_graph_capture_failure_respects_eos_cut(monkeypatch):
    """폴백 eager 구간에서 나온 EOS도 eager와 같은 위치에서 절단된다."""
    ring_window, block = 3, 4
    ref, _ = _eager_reference(max_length=60)
    generated = ref[len(PROMPT):]
    # 폴백 구간(워밍업 W+block + 사이드스트림 block 이후)에서 처음 등장하는 토큰을 EOS로
    first_seen: dict[int, int] = {}
    for index, token in enumerate(generated):
        first_seen.setdefault(token, index)
    eos_index = next(
        i for i, tok in enumerate(generated)
        if i > ring_window + 2 * block and first_seen[tok] == i
    )
    eos = generated[eos_index]
    ref_eos, _ = _eager_reference(max_length=60, eos=eos)
    assert ref_eos == ref[: len(PROMPT) + eos_index + 1]

    _install_fake_cuda(monkeypatch)
    out = fd.fast_greedy_decode(_tiny_vendor_model(ring_window), _gen_kwargs(60, eos=eos), block=block)
    assert out[0].tolist() == ref_eos


def test_graph_side_stream_warmup_respects_max_length(monkeypatch):
    """max_length가 (P+W+block, P+W+2·block) 구간이면 사이드스트림 예열이 남은 길이만큼만
    돌아야 한다 — 예전엔 block개를 무조건 돌려 최대 block-1개를 초과 생성·스트리밍했다
    (audit decode-correctness-6). 상한에 닿으면 캡처 없이 끝난다."""
    ring_window, block = 3, 4
    max_length = len(PROMPT) + ring_window + block + 3
    ref, ref_streamer = _eager_reference(max_length=max_length, block=block)
    assert len(ref) == max_length

    _install_fake_cuda(monkeypatch)
    captures: list = []
    monkeypatch.setattr(torch.cuda, "graph", lambda g, *a, **k: captures.append(g))
    streamer = _Streamer()
    kwargs = _gen_kwargs(max_length)
    kwargs["streamer"] = streamer
    out = fd.fast_greedy_decode(_tiny_vendor_model(ring_window), kwargs, block=block)

    assert out[0].tolist() == ref  # 길이 == max_length, eager와 동일
    assert streamer.generated() == ref_streamer.generated()  # 초과 토큰 스트리밍 없음
    assert captures == []  # 상한 도달 → 캡처 단계에 들어가지 않음
    assert fd.hit_length_limit(out, kwargs) is True


@pytest.mark.parametrize("block", [1, 3, 8])
def test_fast_decode_matches_hf_generate_on_vendor_model(block):
    """실제 벤더 prepare_inputs(링 분기)·P19/P20·P21 위에서 fast_greedy_decode가
    HF generate와 토큰 동일 — 블록 크기는 동기화 빈도만 바꾼다."""
    from transformers import LogitsProcessorList

    model = _tiny_vendor_model()
    hf_kwargs = _gen_kwargs(48)
    hf_kwargs["logits_processor"] = LogitsProcessorList(hf_kwargs["logits_processor"])
    hf_kwargs["attention_mask"] = torch.ones_like(hf_kwargs["input_ids"])
    with torch.no_grad():
        expected = model.generate(**hf_kwargs)
    got = fd.fast_greedy_decode(_tiny_vendor_model(), _gen_kwargs(48), block=block)
    assert got[0].tolist() == expected[0].tolist()
