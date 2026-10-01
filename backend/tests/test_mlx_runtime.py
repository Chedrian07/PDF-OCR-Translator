"""MLX 포팅 런타임 계약 — torch 없이 전 구간이 돌고, 스레드를 바꿔도 결과가 같다.

torch·mlx-vlm 없이(전처리 → 비전 → 프리필 → 생성 → 후처리) 동작해야 mlx extra만 깐
macOS 호스트에서 엔진이 선다. 엔진은 로드(프리로드 스레드)와 생성(워커 스레드)을 다른
스레드에서 하므로, 다른 스레드·동시 실행에서도 같은 토큰이 나와야 한다(감사 스파이크가
mlx 0.32.3에서 확인한 동작을 고정). 가중치는 작은 무작위 MLX 모델(어휘만 실제 크기)이다.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.vendor.unlimited_ocr_mlx import mlx_status

_OK, _WHY = mlx_status()
if not _OK:
    pytest.skip(_WHY, allow_module_level=True)

BACKEND = Path(__file__).resolve().parents[1]

# 작은 MLX 전용 설정: 어휘는 실제 크기(이미지 토큰 128815가 임베딩 범위 안), 나머지는 축소
TINY_CONFIG = {
    "model_type": "unlimited-ocr",
    "vocab_size": 129280,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "topk_method": "greedy",
    "use_mla": False,
    "sliding_window_size": 8,
    "projector_config": {"projector_type": "linear", "input_dim": 256, "n_embed": 64},
    "vision_config": {
        "width": {
            "sam_vit_b": {
                "width": 32, "layers": 2, "heads": 2, "global_attn_indexes": [1],
                "downsample_channels": [64, 128],
            },
            "clip-l-14-224": {"width": 128, "layers": 1, "heads": 2, "image_size": 224, "patch_size": 14},
        }
    },
}

SCRIPT = textwrap.dedent(
    """
    import sys
    sys.modules["torch"] = None          # import torch → ImportError
    sys.modules["mlx_vlm"] = None
    import json, tempfile
    from pathlib import Path
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from app.vendor.unlimited_ocr_mlx import Model, ModelConfig, generate, prepare_multi
    from app.vendor.unlimited_ocr_mlx import decode_outputs, save_results_multi

    class Tok:
        eos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [2 + ord(c) % 50 for c in text]
        def decode(self, ids, skip_special_tokens=False):
            return "<PAGE>" + "".join(chr(97 + i % 26) for i in ids)

    cfg = json.loads(sys.argv[1])
    mx.random.seed(0)
    model = Model(ModelConfig.from_dict(cfg))
    mx.eval(model.parameters())
    d = Path(tempfile.mkdtemp())
    Image.fromarray(np.full((96, 80, 3), 200, np.uint8)).save(d / "p.png")
    tok = Tok()
    inp = prepare_multi(tok, "<image>Multi page parsing.", [str(d / "p.png")], image_size=1024)
    res = generate(model, inp, max_length=inp.prompt_length + 6, no_repeat_ngram_size=35,
                   ngram_window=1024)
    md = save_results_multi(decode_outputs(tok, res.token_ids), inp.images, str(d / "out"))
    assert (d / "out" / "result.md").read_text(encoding="utf-8") == md
    print(len(res.token_ids), res.finish_reason, int("torch" in sys.modules and sys.modules["torch"] is not None))
    """
)


def test_full_pipeline_runs_without_torch_or_mlx_vlm():
    import json

    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(TINY_CONFIG)],
        cwd=BACKEND, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["6", "length", "0"]


@pytest.fixture(scope="module")
def tiny_model_and_inputs(tmp_path_factory):
    import mlx.core as mx

    from app.vendor.unlimited_ocr_mlx import Model, ModelConfig, prepare_single

    class Tok:
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [2 + ord(c) % 50 for c in text]

    mx.random.seed(1)
    model = Model(ModelConfig.from_dict(TINY_CONFIG))
    mx.eval(model.parameters())
    d = tmp_path_factory.mktemp("img")
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (620, 1280, 3), dtype=np.uint8)).save(d / "wide.png")
    inp = prepare_single(Tok(), "<image>document parsing.", str(d / "wide.png"))
    assert inp.crops is not None  # gundam 타일 경로까지 포함
    return model, inp


def _run(model, inp) -> list[int]:
    from app.vendor.unlimited_ocr_mlx import generate

    res = generate(model, inp, max_length=inp.prompt_length + 24, no_repeat_ngram_size=35, ngram_window=128)
    return res.token_ids


def test_generation_is_identical_on_worker_and_concurrent_threads(tiny_model_and_inputs):
    model, inp = tiny_model_and_inputs
    expected = _run(model, inp)
    assert len(expected) == 24

    results: dict[str, list[int]] = {}
    errors: list[BaseException] = []

    def work(name: str) -> None:
        try:
            results[name] = _run(model, inp)
        except BaseException as e:  # noqa: BLE001 - 스레드 예외를 본 스레드로
            errors.append(e)

    t = threading.Thread(target=work, args=("worker",))
    t.start()
    t.join()
    threads = [threading.Thread(target=work, args=(f"c{i}",)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    assert results == {"worker": expected, "c0": expected, "c1": expected}
