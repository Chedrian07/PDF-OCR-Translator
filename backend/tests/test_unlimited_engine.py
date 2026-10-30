"""Torch v2 adapter: streaming, artifact contract, cancellation and output limits."""

import contextlib
import threading
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.engine.base import JobCanceled, NullSink, OutputLimitError
from app.engine.unlimited import UnlimitedEngine


class Tokenizer:
    eos_token_id = 1

    def decode(self, ids, **kwargs):
        return "Hello"


class Model:
    def __init__(self, finish="stop", cancel=None):
        self.finish = finish
        self.cancel = cancel
        self.kwargs = None

    def infer(self, tokenizer, **kwargs):
        self.kwargs = kwargs
        if self.cancel:
            self.cancel.set()
        if kwargs["should_stop"]():
            raise InterruptedError()
        kwargs["streamer"].on_finalized_text(
            "<PAGE><BLOCK>text[1,2,500,400]<FORK>Hello"
        )
        return {
            "text": "<PAGE><BLOCK>text[1,2,500,400]<FORK>Hello",
            "finish_reason": self.finish,
        }


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_artifacts_and_device_pool(tmp_path, monkeypatch, single, device):
    import app.engine.unlimited as mod

    pools = []

    @contextlib.contextmanager
    def pool(enabled):
        pools.append(enabled)
        yield

    monkeypatch.setattr(mod, "autorelease_pool", pool)
    e = UnlimitedEngine(Settings(device=device))
    e._model = Model()
    e._tokenizer = Tokenizer()
    monkeypatch.setattr(e, "_release_device_cache", lambda: None)
    p = tmp_path / "p.png"
    Image.new("RGB", (100, 100)).save(p)
    out = tmp_path / "out"
    fn = e.run_single if single else e.run_multi
    text = fn(p if single else [p], out, NullSink(), threading.Event())
    assert "Hello" in text and "<BLOCK>" not in text
    assert (out / "raw_pages.json").is_file()
    assert "<|det|>" in (out / "raw_pages.json").read_text()
    assert pools == [device == "metal"]
    assert e._model.kwargs["prompt"] == "Multi page parsing."
    assert e._model.kwargs["crop_mode"] is single
    assert e.capabilities().runtime == "torch"


def test_length_cap_keeps_recoverable_multi_prefix(tmp_path, monkeypatch):
    e = UnlimitedEngine(Settings())
    e._model = Model("length")
    e._tokenizer = Tokenizer()
    monkeypatch.setattr(e, "_release_device_cache", lambda: None)
    p = tmp_path / "p.png"
    Image.new("RGB", (100, 100)).save(p)
    with pytest.raises(OutputLimitError) as err:
        e.run_multi([p], tmp_path / "out", NullSink(), threading.Event())
    assert "Hello" in err.value.partial_output


def test_cancel_is_never_returned_as_success(tmp_path, monkeypatch):
    cancel = threading.Event()
    e = UnlimitedEngine(Settings())
    e._model = Model(cancel=cancel)
    e._tokenizer = Tokenizer()
    released = []
    monkeypatch.setattr(e, "_release_device_cache", lambda: released.append(True))
    with pytest.raises(JobCanceled):
        e.run_multi([Path("unused.png")], tmp_path, NullSink(), cancel)
    assert released
