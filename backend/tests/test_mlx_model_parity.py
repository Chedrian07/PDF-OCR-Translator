"""V2 prompt/image parity; full SR-SWA arithmetic is test_unlimited2_sparse.py."""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app.vendor.unlimited_ocr_mlx.processing import prepare_multi, prepare_single


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        # Real special-token boundaries, with deterministic ordinary test tokens.
        result = []
        while text:
            if text.startswith("<PAGE>"):
                result.append(128827)
                text = text[6:]
            else:
                result.append(10 + ord(text[0]))
                text = text[1:]
        return result


@pytest.mark.parametrize("single", [False, True])
def test_v2_prompt_tokens_and_image_masks(tmp_path, single):
    torch = pytest.importorskip("torch")
    from app.vendor.unlimited_ocr2.modeling_unlimitedocr import _uocr_build_inputs

    paths = []
    for i in range(1 if single else 2):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (800, 1000), (127, 150, 210)).save(p)
        paths.append(p)
    tok = Tokenizer()
    got = (
        prepare_single(tok, "<image>Multi page parsing.", paths[0])
        if single
        else prepare_multi(tok, "<image>Multi page parsing.", paths, image_size=1024)
    )
    ids, imgs, mask, spatial = _uocr_build_inputs(
        tok,
        "<PAGE><image>" * len(paths) + "Multi page parsing.",
        got.images,
        crop_mode=single,
    )
    np.testing.assert_array_equal(got.input_ids, ids.numpy())
    np.testing.assert_array_equal(got.images_seq_mask, mask.numpy())
    np.testing.assert_array_equal(got.images_spatial_crop, spatial.numpy())
    # Upstream quantizes preprocessing to bf16 before device placement.
    expected = imgs[1].float().permute(0, 2, 3, 1).numpy()
    quantized = torch.from_numpy(got.global_views).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(quantized, expected)


def test_config_rejects_old_model():
    from app.vendor.unlimited_ocr_mlx.config import TextConfig

    with pytest.raises(ValueError, match="DASH"):
        TextConfig.from_hf({"use_mla": False})
    cfg = json.loads(
        (Path(__file__).parent / "fixtures/unlimited2-config.json").read_text()
    )
    assert TextConfig.from_hf(cfg).sparse_attention_config["page_token_id"] == 128827


def test_real_mlx_checkpoint_when_requested(tmp_path):
    import os
    import threading

    if os.environ.get("OCR_MLX_REAL_TESTS") != "1":
        pytest.skip("OCR_MLX_REAL_TESTS=1 enables real Apple Silicon inference")
    from app.config import Settings
    from app.engine.base import NullSink
    from app.engine.unlimited_mlx import UnlimitedMLXEngine
    from PIL import ImageDraw

    path = tmp_path / "real.png"
    image = Image.new("RGB", (640, 640), "white")
    ImageDraw.Draw(image).text((40, 40), "Unlimited OCR 2", fill="black", font_size=36)
    image.save(path)
    engine = UnlimitedMLXEngine(Settings(device="mlx", max_length=4096))
    text = engine.run_multi([path], tmp_path / "result", NullSink(), threading.Event())
    assert engine.loaded and "OCR" in text
    assert engine.capabilities().model_id == "baidu/Unlimited-OCR-2"
