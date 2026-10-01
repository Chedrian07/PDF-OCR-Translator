"""MLX 포팅 ↔ torch 벤더 수치 패리티.

A. 전처리(mlx 불필요, torch+torchvision 필요): 두 모드의 프롬프트 ids·이미지 마스크·
   픽셀·타일이 torch infer/infer_multi가 만드는 gen_kwargs와 비트 단위로 같다. torch
   메서드는 모델 없이(스텁 self + P15 generate_fn 캡처) 돌린다.
B. 작은 무작위 모델(mlx + torch 필요): torch 벤더 모델(SAM/CLIP 층수·폭만 줄임)과 MLX
   모델에 같은 가중치를 넣는다 — torch state_dict를 safetensors 스냅샷으로 써서 실가중치와
   **같은 로더·sanitize 경로**(loader.build_model)로 올린다. 프리필 + 링(W=4)을 여러 번
   도는 teacher-forced 디코드에서 fp32 로짓을 비교하고, torch fast_greedy_decode(+앱 ngram
   프로세서)와 그리디 생성 결과를 비교한다.
C. 실가중치(OCR_MLX_REAL_TESTS=1, Apple Silicon): 고정 스냅샷으로 bf16 그리디 접두와
   torch CPU fp32 로짓을 비교한다(수 GB·수십 초 — 기본 스위트에서는 건너뜀).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import types
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.vendor.unlimited_ocr_mlx import mlx_status
from app.vendor.unlimited_ocr_mlx import processing as P
from app.vendor.unlimited_ocr_mlx.config import IMAGE_TOKEN_ID

_MLX_OK, _MLX_WHY = mlx_status()
needs_mlx = pytest.mark.skipif(not _MLX_OK, reason=_MLX_WHY)
REAL = os.environ.get("OCR_MLX_REAL_TESTS", "").strip() == "1"
real_weights = pytest.mark.skipif(
    not (REAL and _MLX_OK), reason="OCR_MLX_REAL_TESTS=1 + Apple Silicon mlx 필요 (실가중치)"
)

MULTI_PROMPT = "<image>Multi page parsing."
SINGLE_PROMPT = "<image>document parsing."
REPO = Path(__file__).resolve().parents[2]
SAMPLE_PDF = REPO / "sample" / "2504.19874v1.pdf"


class FakeTokenizer:
    """프롬프트 텍스트를 작은 id로 — 전처리·작은 모델 테스트용 (어휘 512 미만)."""

    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 7) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + int(i) % 26) for i in ids)


def _write_images(tmp_path: Path, sizes, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    paths = []
    for i, (w, h) in enumerate(sizes):
        # 무작위 잡음 + 부드러운 그라디언트(크롭·리샘플이 실제로 값을 바꾸게)
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), (xx + yy) % 256], -1)
        noise = rng.integers(0, 64, (h, w, 3))
        a = ((base + noise) % 256).astype(np.uint8)
        p = tmp_path / f"img{i}_{w}x{h}.png"
        Image.fromarray(a).save(p)
        paths.append(str(p))
    return paths


@pytest.fixture(scope="module")
def torch_vendor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from app.vendor.unlimited_ocr import deepencoder as de
    from app.vendor.unlimited_ocr import modeling_unlimitedocr as mu

    return types.SimpleNamespace(torch=torch, mu=mu, de=de)


def _stub_self(torch):
    class Stub:
        config = types.SimpleNamespace(sliding_window_size=128, sliding_window=128)

        def parameters(self):
            yield torch.zeros(1)

        def disable_torch_init(self):
            pass

    return Stub()


def _capture(tv, call) -> dict:
    """torch infer/infer_multi를 실행하고 generate_fn에 넘어간 gen_kwargs를 돌려준다."""
    cap: dict = {}

    def gen_fn(model, kw):
        cap.update(kw)
        return tv.torch.cat([kw["input_ids"], tv.torch.tensor([[1]])], dim=1)

    call(gen_fn)
    return cap


def _assert_inputs_equal(cap: dict, mine: P.OCRInputs, crop_expected: bool):
    assert cap["input_ids"][0].tolist() == mine.input_ids.tolist()
    assert cap["images_seq_mask"][0].tolist() == mine.images_seq_mask.tolist()
    assert cap["images_spatial_crop"].tolist() == mine.images_spatial_crop
    crop_t, ori_t = cap["images"][0]
    np.testing.assert_array_equal(ori_t.numpy().transpose(0, 2, 3, 1), mine.global_views)
    if crop_expected:
        np.testing.assert_array_equal(crop_t.numpy().transpose(0, 2, 3, 1), mine.crops)
    else:
        assert mine.crops is None and float(crop_t.abs().sum()) == 0.0


# ── A. 전처리 패리티 ──


@pytest.mark.parametrize("image_size", [1024, 640])
def test_multi_prompt_and_pixels_match_torch(torch_vendor, tmp_path, image_size):
    tok = FakeTokenizer()
    paths = _write_images(tmp_path, [(340, 440), (1700, 2200), (500, 260)])
    cap = _capture(
        torch_vendor,
        lambda fn: torch_vendor.mu.UnlimitedOCRForCausalLM.infer_multi(
            _stub_self(torch_vendor.torch), tok, prompt=MULTI_PROMPT, image_files=paths,
            output_path=str(tmp_path / "t"), image_size=image_size, generate_fn=fn,
        ),
    )
    mine = P.prepare_multi(tok, MULTI_PROMPT, paths, image_size=image_size)
    _assert_inputs_equal(cap, mine, crop_expected=False)
    q = {1024: 16, 640: 10}[image_size]
    assert int(mine.images_seq_mask.sum()) == 3 * ((q + 1) * q + 1)


@pytest.mark.parametrize(
    "size,ratio",
    [((1280, 620), [2, 1]), ((1000, 700), [3, 2]), ((620, 1240), [1, 2]), ((620, 1500), [2, 5]), ((600, 600), [1, 1]), ((300, 420), [1, 1])],
)
def test_single_gundam_prompt_and_pixels_match_torch(torch_vendor, tmp_path, size, ratio):
    tok = FakeTokenizer()
    (path,) = _write_images(tmp_path, [size])
    cap = _capture(
        torch_vendor,
        lambda fn: torch_vendor.mu.UnlimitedOCRForCausalLM.infer(
            _stub_self(torch_vendor.torch), tok, prompt=SINGLE_PROMPT, image_file=path,
            output_path=str(tmp_path / "t"), base_size=1024, image_size=640, crop_mode=True,
            save_results=False, generate_fn=fn,
        ),
    )
    mine = P.prepare_single(tok, SINGLE_PROMPT, path, base_size=1024, image_size=640, crop_mode=True)
    assert mine.images_spatial_crop == [ratio]
    _assert_inputs_equal(cap, mine, crop_expected=ratio != [1, 1])


def test_prompt_must_have_exactly_one_image_token(tmp_path):
    (path,) = _write_images(tmp_path, [(64, 64)])
    with pytest.raises(ValueError):
        P.prepare_multi(FakeTokenizer(), "no image token", [path])
    with pytest.raises(ValueError):
        P.prepare_single(FakeTokenizer(), "<image><image>x", path)
    with pytest.raises(ValueError):
        P.prepare_multi(FakeTokenizer(), MULTI_PROMPT, [])


def _real_snapshot_dir() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        from app.config import Settings

        s = Settings()
        return snapshot_download(
            s.model_id, revision=s.model_revision, local_files_only=True, allow_patterns=["*.json"]
        )
    except Exception:  # noqa: BLE001 - 캐시 없음(CI) → 건너뜀
        return None


def test_prompt_ids_with_real_tokenizer_match_torch(torch_vendor, tmp_path):
    snap = _real_snapshot_dir()
    if snap is None:
        pytest.skip("고정 리비전 토크나이저가 로컬 HF 캐시에 없음")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    paths = _write_images(tmp_path, [(1700, 2200), (1280, 620)])
    mine = P.prepare_multi(tok, MULTI_PROMPT, paths[:1], image_size=1024)
    expected = [0] + [IMAGE_TOKEN_ID] * 273 + tok.encode("Multi page parsing.", add_special_tokens=False)
    assert mine.input_ids.tolist() == expected
    assert expected[-4:] == [37460, 4366, 76466, 16]  # 감사 스파이크가 기록한 앱 프롬프트
    cap = _capture(
        torch_vendor,
        lambda fn: torch_vendor.mu.UnlimitedOCRForCausalLM.infer_multi(
            _stub_self(torch_vendor.torch), tok, prompt=MULTI_PROMPT, image_files=paths,
            output_path=str(tmp_path / "t"), image_size=1024, generate_fn=fn,
        ),
    )
    _assert_inputs_equal(cap, P.prepare_multi(tok, MULTI_PROMPT, paths, image_size=1024), False)
    cap = _capture(
        torch_vendor,
        lambda fn: torch_vendor.mu.UnlimitedOCRForCausalLM.infer(
            _stub_self(torch_vendor.torch), tok, prompt=SINGLE_PROMPT, image_file=paths[1],
            output_path=str(tmp_path / "t"), base_size=1024, image_size=640, crop_mode=True,
            save_results=False, generate_fn=fn,
        ),
    )
    _assert_inputs_equal(cap, P.prepare_single(tok, SINGLE_PROMPT, paths[1]), True)


# ── B. 작은 무작위 모델 ──

RING_W = 4
IMG_SUB = 3  # 작은 어휘에서 이미지 자리 토큰 id (그 자리 임베딩은 이미지 특징으로 덮인다)
TINY_TEXT = dict(
    vocab_size=512,
    hidden_size=1280,  # torch projector n_embed가 1280으로 하드코딩돼 있어 고정
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=3,  # 0: dense, 1-2: MoE
    num_attention_heads=10,
    num_key_value_heads=10,
    n_shared_experts=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    first_k_dense_replace=1,
    topk_method="greedy",
    n_group=1,
    topk_group=1,
    use_mla=False,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=0,
    qk_rope_head_dim=0,
    v_head_dim=128,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    sliding_window_size=RING_W,
    sliding_window=RING_W,
    bos_token_id=0,
    eos_token_id=1,
)
TINY_SAM = dict(width=64, layers=2, heads=2, global_attn_indexes=[1], downsample_channels=[512, 1024])
TINY_CLIP_LAYERS = 2


def _tiny_hf_config() -> dict:
    return {
        "model_type": "unlimited-ocr",
        "tile_tag": "2D",
        **TINY_TEXT,
        "projector_config": {"projector_type": "linear", "input_dim": 2048, "n_embed": 1280},
        "vision_config": {
            "width": {
                "sam_vit_b": TINY_SAM,
                "clip-l-14-224": {
                    "width": 1024, "layers": TINY_CLIP_LAYERS, "heads": 16,
                    "image_size": 224, "patch_size": 14,
                },
            }
        },
    }


def _randomize_(torch, model) -> None:
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g)
            if p.ndim == 1 and name.endswith(".weight"):  # RMSNorm·LayerNorm(2d) 스케일
                p.copy_(1 + 0.1 * noise)
            elif name.endswith(".bias"):
                p.copy_(0.02 * noise)
            elif "pos_embed" in name or "rel_pos" in name or "position_embedding" in name:
                p.copy_(0.1 * noise)
            elif p.ndim == 1:  # class_embedding·image_newline·view_seperator
                p.copy_(noise / p.numel() ** 0.5)
            else:
                p.copy_(noise / p[0].numel() ** 0.5)
        model.lm_head.weight.mul_(8.0)  # 뾰족한 분포 → 그리디 근접 동률 회피


@pytest.fixture(scope="module")
def tiny_pair(torch_vendor, tmp_path_factory):
    if not _MLX_OK:
        pytest.skip(_MLX_WHY)
    import mlx.core as mx

    from app.vendor.unlimited_ocr_mlx.loader import build_model

    torch, mu, de = torch_vendor.torch, torch_vendor.mu, torch_vendor.de

    def tiny_sam():
        return de.ImageEncoderViT(
            depth=TINY_SAM["layers"], embed_dim=TINY_SAM["width"], img_size=1024, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=TINY_SAM["heads"],
            patch_size=16, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=TINY_SAM["global_attn_indexes"], window_size=14, out_chans=256,
        )

    def tiny_clip():
        cfg = copy.deepcopy(de.vit_model_cfg)
        cfg.num_layers = TINY_CLIP_LAYERS
        return de.VitModel(cfg=cfg, freeze_embed=False, freeze_pre_norm=False)

    cfg = mu.UnlimitedOCRConfig(**TINY_TEXT)
    cfg._attn_implementation = "eager"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mu, "build_sam_vit_b", tiny_sam)
        mp.setattr(mu, "build_clip_l", tiny_clip)
        torch.manual_seed(0)
        tmodel = mu.UnlimitedOCRForCausalLM(cfg).eval()
    _randomize_(torch, tmodel)

    snap = tmp_path_factory.mktemp("tiny-snapshot")
    (snap / "config.json").write_text(json.dumps(_tiny_hf_config()), encoding="utf-8")
    state = {k: mx.array(v.detach().cpu().numpy()) for k, v in tmodel.state_dict().items()}
    mx.save_safetensors(str(snap / "model.safetensors"), state)
    mmodel, quant = build_model(snap, dtype="float32")
    assert quant is None
    return types.SimpleNamespace(torch=torch, tmodel=tmodel, mmodel=mmodel, snap=snap)


def _sub_ids(inp: P.OCRInputs) -> np.ndarray:
    ids = inp.input_ids.copy()
    ids[inp.images_seq_mask] = IMG_SUB
    return ids


def _torch_images(torch, inp: P.OCRInputs, dummy_size: int):
    ori = torch.from_numpy(np.ascontiguousarray(inp.global_views.transpose(0, 3, 1, 2)))
    if inp.crops is not None:
        crop = torch.from_numpy(np.ascontiguousarray(inp.crops.transpose(0, 3, 1, 2)))
    else:  # torch infer/infer_multi의 0 더미 (합 0 → 무크롭 분기)
        crop = torch.zeros((1, 3, dummy_size, dummy_size))
    return [(crop, ori)]


def _torch_teacher_forced(pair, inp: P.OCRInputs, forced, dummy_size: int, window: int = RING_W):
    from transformers.cache_utils import DynamicCache

    torch, model = pair.torch, pair.tmodel
    ids = _sub_ids(inp)
    p_len = len(ids)
    model.config._ring_window = window  # infer/infer_multi가 생성 전에 하는 설정
    model.config.sliding_window = None
    out_logits = []
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids.tolist()]),
            images=_torch_images(torch, inp, dummy_size),
            images_seq_mask=torch.tensor([inp.images_seq_mask.tolist()]),
            images_spatial_crop=torch.tensor(inp.images_spatial_crop),
            past_key_values=DynamicCache(),
            use_cache=True,
            position_ids=torch.arange(p_len)[None],
        )
        out_logits.append(out.logits[0, -1].numpy())
        pkv = out.past_key_values
        for i, tok in enumerate(forced):
            out = model(
                input_ids=torch.tensor([[tok]]), past_key_values=pkv, use_cache=True,
                position_ids=torch.tensor([[p_len + i]]),
            )
            pkv = out.past_key_values
            out_logits.append(out.logits[0, -1].numpy())
    assert pkv._prefill_length[0] == p_len
    return out_logits


def _mlx_teacher_forced(model, inp: P.OCRInputs, forced, window: int | None = None):
    import mlx.core as mx

    from app.vendor.unlimited_ocr_mlx.cache import RingSlidingKVCache

    feats = model.encode_images(inp.global_views, inp.crops, inp.spatial_crop)
    emb = model.get_input_embeddings(_sub_ids(inp), inp.images_seq_mask, feats)
    lm = model.language_model
    cache = (
        model.make_cache()
        if window is None
        else [RingSlidingKVCache(window) for _ in lm.layers]
    )
    out = [np.array(lm(None, inputs_embeds=emb, cache=cache, last_only=True)[0, -1])]
    for tok in forced:
        out.append(np.array(lm(mx.array([[tok]]), cache=cache)[0, -1]))
    assert cache[0].prefill_length == len(inp.input_ids)
    return out


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _tiny_inputs(mode: str, tmp_path: Path):
    tok = FakeTokenizer()
    if mode == "multi":
        paths = _write_images(tmp_path, [(340, 440), (500, 260)], seed=3)
        return P.prepare_multi(tok, MULTI_PROMPT, paths, image_size=1024), 1024
    if mode == "single_crops":
        (path,) = _write_images(tmp_path, [(1280, 620)], seed=4)  # 2x1 타일 + 전역
        return P.prepare_single(tok, SINGLE_PROMPT, path), 1024
    (path,) = _write_images(tmp_path, [(500, 600)], seed=5)  # 640 이하 → 전역만
    return P.prepare_single(tok, SINGLE_PROMPT, path), 1024


FORCED = [7, 9, 11, 7, 9, 11, 200, 300, 7, 9, 11, 42]  # W=4 링을 세 바퀴 돈다


@needs_mlx
@pytest.mark.parametrize("mode", ["multi", "single_crops", "single_small"])
def test_tiny_fp32_logits_match_torch(tiny_pair, tmp_path, mode):
    inp, dummy = _tiny_inputs(mode, tmp_path)
    if mode == "single_crops":
        assert inp.crops is not None and inp.spatial_crop == (2, 1)
    t_logits = _torch_teacher_forced(tiny_pair, inp, FORCED, dummy)
    m_logits = _mlx_teacher_forced(tiny_pair.mmodel, inp, FORCED)
    for step, (m, t) in enumerate(zip(m_logits, t_logits)):
        assert _rel(m, t) < 1e-4, (mode, step, _rel(m, t))
        assert int(np.argmax(m)) == int(np.argmax(t)), (mode, step)


@needs_mlx
def test_tiny_ring_window_is_exercised(tiny_pair, tmp_path):
    """음성 대조: 링 창을 바꾸면(W=5) 워밍업이 끝난 뒤부터 torch(W=4)와 갈라져야 한다 —
    위 패리티가 링 의미론을 실제로 검증한다는 증거."""
    inp, dummy = _tiny_inputs("multi", tmp_path)
    t_logits = _torch_teacher_forced(tiny_pair, inp, FORCED, dummy)
    m_logits = _mlx_teacher_forced(tiny_pair.mmodel, inp, FORCED, window=RING_W + 1)
    rels = [_rel(m, t) for m, t in zip(m_logits, t_logits)]
    assert max(rels[: RING_W + 1]) < 1e-4  # 프리필 + 워밍업 구간은 같다
    assert max(rels[RING_W + 1 :]) > 1e-3


@needs_mlx
@pytest.mark.parametrize("mode", ["multi", "single_crops"])
def test_tiny_greedy_generation_matches_torch_fast_decode(tiny_pair, tmp_path, mode):
    from app.engine.fast_decode import fast_greedy_decode
    from app.native_ops import make_ngram_logits_processor
    from app.vendor.unlimited_ocr_mlx.generate import generate

    torch = tiny_pair.torch
    inp, dummy = _tiny_inputs(mode, tmp_path)
    ids = _sub_ids(inp)
    p_len = len(ids)
    n, w, new = 3, 16, 40
    model = tiny_pair.tmodel
    model.config._ring_window = RING_W
    model.config.sliding_window = None
    gen_kwargs = dict(
        input_ids=torch.tensor([ids.tolist()]),
        images=_torch_images(torch, inp, dummy),
        images_seq_mask=torch.tensor([inp.images_seq_mask.tolist()]),
        images_spatial_crop=torch.tensor(inp.images_spatial_crop),
        do_sample=False,
        eos_token_id=1,
        max_length=p_len + new,
        logits_processor=make_ngram_logits_processor(n, w, "cpu"),
        use_cache=True,
    )
    t_out = fast_greedy_decode(model, gen_kwargs, block=1)[0, p_len:].tolist()

    seen: list[int] = []
    res = generate(
        tiny_pair.mmodel, dataclasses.replace(inp, input_ids=ids), max_length=p_len + new,
        eos_token_id=1, no_repeat_ngram_size=n, ngram_window=w, on_token=seen.append,
    )
    assert res.token_ids == t_out
    assert seen == res.token_ids
    if res.finish_reason == "length":
        assert len(res.token_ids) == new and res.hit_max_length
    else:
        assert res.token_ids[-1] == 1


@needs_mlx
def test_strict_load_rejects_missing_or_unknown_keys(tiny_pair, tmp_path):
    import mlx.core as mx

    from app.vendor.unlimited_ocr_mlx.loader import build_model

    full = mx.load(str(tiny_pair.snap / "model.safetensors"))
    cfg = (tiny_pair.snap / "config.json").read_text(encoding="utf-8")
    cases = {
        "missing": {k: v for k, v in full.items() if k != "model.layers.1.mlp.gate.weight"},
        "extra": {**full, "model.layers.0.self_attn.extra_proj.weight": mx.zeros((2, 2))},
        "unknown_prefix": {**full, "vision_tower.weight": mx.zeros((2,))},
        "bad_shape": {**full, "model.sam_model.net_2.weight": mx.zeros((512, 256, 3, 1))},
    }
    for name, weights in cases.items():
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(cfg, encoding="utf-8")
        mx.save_safetensors(str(d / "model.safetensors"), weights)
        with pytest.raises(ValueError):
            build_model(d, dtype="float32")


@needs_mlx
def test_unsupported_config_is_rejected(tmp_path):
    from app.vendor.unlimited_ocr_mlx.config import ModelConfig

    base = _tiny_hf_config()
    for patch in (
        {"model_type": "deepseek_vl_v2"},
        {"use_mla": True},
        {"rope_scaling": {"type": "yarn", "factor": 2.0}},
        {"topk_method": "noaux_tc"},
        {"projector_config": {"projector_type": "downsample_mlp_gelu"}},
    ):
        with pytest.raises(ValueError):
            ModelConfig.from_dict({**base, **patch})
    cfg = dict(base)
    cfg.pop("use_mla")  # torch DeepseekV2Config 기본값 use_mla=True → MLA로 본다
    with pytest.raises(ValueError):
        ModelConfig.from_dict(cfg)


@needs_mlx
def test_8bit_quantizes_only_the_decoder(tiny_pair, tmp_path):
    import mlx.nn as nn

    from app.vendor.unlimited_ocr_mlx.loader import build_model
    from app.vendor.unlimited_ocr_mlx.switch_layers import QuantizedSwitchLinear

    qmodel, quant = build_model(tiny_pair.snap, dtype="float32", quantize_bits=8)
    assert quant == {"bits": 8, "group_size": 64, "mode": "affine"}
    lm = qmodel.language_model
    assert isinstance(lm.lm_head, nn.QuantizedLinear)
    assert isinstance(lm.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(lm.model.layers[0].mlp.down_proj, nn.QuantizedLinear)
    moe = lm.model.layers[1].mlp
    assert isinstance(moe.switch_mlp.gate_proj, QuantizedSwitchLinear)
    assert isinstance(moe.shared_experts.up_proj, nn.QuantizedLinear)
    assert not hasattr(moe.gate, "scales")  # 라우팅 게이트는 양자화하지 않는다
    # 비전 경로(SAM·CLIP·projector)는 그대로
    assert type(qmodel.projector.layers) is nn.Linear
    assert type(qmodel.vision_model.transformer.layers[0].mlp.fc1) is nn.Linear
    assert type(qmodel.sam_model.blocks[0].attn.qkv) is nn.Linear

    inp, _ = _tiny_inputs("multi", tmp_path)
    ref = _mlx_teacher_forced(tiny_pair.mmodel, inp, FORCED[:3])
    got = _mlx_teacher_forced(qmodel, inp, FORCED[:3])
    # 무작위 가중치(lm_head ×8)라 8비트 오차가 학습된 가중치보다 크게 쌓인다 — 정성 확인만
    for m, r in zip(got, ref):
        assert _rel(m, r) < 0.15

    for bits in (4, 3, 6):
        with pytest.raises(ValueError):
            build_model(tiny_pair.snap, dtype="float32", quantize_bits=bits)


# ── C. 실가중치 (opt-in) ──

# 감사 스파이크(mlx-vlm 0.7.4 + CLIP 수정, bf16) 1쪽 그리디의 첫 64토큰 — torch MPS bf16
# 레퍼런스(page1_default_ids)와도 같은 구간이다(첫 분기는 122번째).
EXPECTED_PAGE1_IDS = [
    100855, 16412, 32, 128818, 121695, 19053, 764, 1349, 14, 223, 16326, 14, 223, 4739, 14,
    223, 25624, 63, 128819, 64751, 28, 6793, 22, 16, 1809, 6048, 88, 19, 764, 13794, 6547, 41,
    63, 223, 1449, 4648, 223, 939, 23, 201, 128818, 10212, 764, 10081, 14, 223, 9250, 14, 223,
    29178, 14, 223, 7593, 63, 128819, 56861, 3541, 35420, 28, 12551, 26081, 17639, 1878, 418,
]


@pytest.fixture(scope="module")
def real_page1(tmp_path_factory) -> str:
    from app.pipeline.pdf import render_pdf_pages

    pages = render_pdf_pages(SAMPLE_PDF, tmp_path_factory.mktemp("pages"), dpi=200, max_pages=1000)
    return str(pages[0])


def _settings():
    from app.config import Settings

    return Settings()


@real_weights
def test_real_bf16_greedy_prefix_matches_reference(real_page1):
    from app.vendor.unlimited_ocr_mlx import generate, load

    s = _settings()
    model, tok, info = load(s.model_id, s.model_revision, dtype="bfloat16", local_files_only=True)
    assert info.parameter_bytes > 6e9
    inp = P.prepare_multi(tok, MULTI_PROMPT, [real_page1], image_size=1024)
    assert inp.prompt_length == 278
    res = generate(model, inp, max_length=278 + 64, no_repeat_ngram_size=35, ngram_window=1024)
    assert res.token_ids == EXPECTED_PAGE1_IDS
    assert res.hit_max_length


@real_weights
def test_real_fp32_logits_match_torch_cpu(real_page1, tmp_path):
    torch = pytest.importorskip("torch")
    from app.config import Settings
    from app.engine.unlimited import UnlimitedEngine
    from app.vendor.unlimited_ocr_mlx import load

    s = _settings()
    model, tok, _ = load(s.model_id, s.model_revision, dtype="float32", local_files_only=True)
    eng = UnlimitedEngine(
        Settings(engine="unlimited", device="cpu", dtype="float32", data_dir=tmp_path / "d",
                 preload_model=False)
    )
    eng.load()
    tmodel = eng._model
    forced = EXPECTED_PAGE1_IDS[:16]
    # 멀티: 프롬프트 + 강제 16토큰을 캐시 없이 한 번에 (W=128 미만이라 링 의미론과 같다)
    inp = P.prepare_multi(tok, MULTI_PROMPT, [real_page1], image_size=1024)
    # 단일 gundam: 1쪽 위쪽(2:1 비율) → 2x1 타일 + 전역
    top = Path(tmp_path) / "top.png"
    Image.open(real_page1).crop((0, 0, 1700, 800)).save(top)
    inp_single = P.prepare_single(tok, SINGLE_PROMPT, str(top))
    assert inp_single.spatial_crop == (2, 1)
    for x, extra in ((inp, forced), (inp_single, [])):
        ids = np.concatenate([x.input_ids, np.asarray(extra, dtype=np.int64)])
        mask = np.concatenate([x.images_seq_mask, np.zeros(len(extra), bool)])
        feats = model.encode_images(x.global_views, x.crops, x.spatial_crop)
        emb = model.get_input_embeddings(ids, mask, feats)
        m = np.array(model.language_model(None, inputs_embeds=emb, last_only=True)[0, -1])
        with torch.no_grad():
            out = tmodel(
                input_ids=torch.tensor([ids.tolist()]),
                images=_torch_images(torch, x, 1024),
                images_seq_mask=torch.tensor([mask.tolist()]),
                images_spatial_crop=torch.tensor(x.images_spatial_crop),
                use_cache=False,
            )
        t = out.logits[0, -1].float().numpy()
        assert _rel(m, t) < 1e-4
        assert int(np.argmax(m)) == int(np.argmax(t))
