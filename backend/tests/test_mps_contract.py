"""New checkpoint image injection and optional real Metal smoke."""

import os
import threading

import pytest
from PIL import Image

from app.config import Settings
from app.engine.base import NullSink
from app.engine.unlimited import UnlimitedEngine


def test_image_injection_does_not_use_cuda_or_broadcast_mask():
    import inspect
    from app.vendor.unlimited_ocr2.modeling_unlimitedocr import UnlimitedOCRModel

    src = inspect.getsource(UnlimitedOCRModel.forward)
    assert ".cuda()" not in src
    assert "masked_scatter_" not in src
    assert ".to(inputs_embeds.device).bool()" in src


@pytest.mark.skipif(
    os.environ.get("OCR_MPS_TESTS") != "1", reason="Opt-in real Apple Silicon test"
)
def test_real_metal_model(tmp_path):
    import torch

    assert torch.backends.mps.is_available(), "Metal hardware required"
    p = tmp_path / "page.png"
    Image.new("RGB", (640, 640), "white").save(p)
    e = UnlimitedEngine(Settings(device="metal", max_length=4096))
    result = e.run_multi([p], tmp_path / "out", NullSink(), threading.Event())
    assert e.loaded
    assert "<PAGE>" in result
