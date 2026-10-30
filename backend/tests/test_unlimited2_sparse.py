"""Real arithmetic parity against pinned torch SR-SWA, including page changes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_sparse_prefill_decode_and_document_isolation():
    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache
    from app.vendor.unlimited_ocr2.configuration_deepseek_v2 import DeepseekV2Config
    from app.vendor.unlimited_ocr2 import modeling_deepseekv2 as ref
    from app.vendor.unlimited_ocr2.modeling_unlimitedocr import _uocr_dash_decode_allow
    from app.vendor.unlimited_ocr_mlx.config import TextConfig
    from app.vendor.unlimited_ocr_mlx.sparse import SparseAttention, SparseCache

    cfg = json.loads(
        (Path(__file__).parent / "fixtures/unlimited2-config.json").read_text()
    )
    cfg.update(
        hidden_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=1,
        n_routed_experts=None,
    )
    cfg["sparse_attention_config"]["dash_use_triton"] = False
    config = DeepseekV2Config(**copy.deepcopy(cfg))
    torch.manual_seed(12)
    attention = ref._BaseAttn(config, 0).eval()
    mlx_attention = SparseAttention(TextConfig.from_hf(cfg), 0)
    mlx_attention.load_weights(
        [
            (k, mx.array(v.detach().numpy()))
            for k, v in attention.state_dict().items()
            if not k.startswith("indexer.")
        ],
        strict=True,
    )
    decoder = SimpleNamespace(
        config=config, layers=[SimpleNamespace(self_attn=attention)]
    )
    state = ref.configure_dash_inference(decoder, True)
    prompt = [0, 128827] + [128815] * 273 + [128827] + [128815] * 273 + [4, 5, 6]
    try:
        for _document in range(2):
            tcache = DynamicCache()
            mcache = SparseCache(cfg["sparse_attention_config"])
            with torch.no_grad():
                for ids in (prompt, [128827], [42], [128827], [48]):
                    tokens = torch.tensor([ids])
                    x = torch.randn(1, len(ids), 128)
                    attention._sparse_source_input_ids = tokens
                    attention._dash_dec_allow = _uocr_dash_decode_allow(
                        attention.indexer, tokens, tcache
                    )
                    pos = torch.arange(
                        tcache.get_seq_length(), tcache.get_seq_length() + len(ids)
                    )[None]
                    want = attention(x, position_ids=pos, past_key_value=tcache)[
                        0
                    ].numpy()
                    mcache.prepare(ids)
                    actual = np.array(mlx_attention(mx.array(x.numpy()), cache=mcache))
                    np.testing.assert_allclose(actual, want, atol=1e-5, rtol=2e-4)
    finally:
        ref.configure_dash_inference(decoder, False, state=state)
