import asyncio
import json
import threading

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.engine.base import EngineError, JobCanceled, NullSink, OutputLimitError
from app.engine.postprocess import BlockStream, normalize_blocks
from app.engine.sglang import SGLangEngine


class Sink:
    def __init__(self):
        self.text = ""

    def on_text(self, text):
        self.text += text


def test_v2_block_headers_across_every_stream_boundary():
    raw = "<PAGE><BLOCK>title<LEVEL1>[1,2,300,40]<FORK>Hello\n<BLOCK>chart[1,40,300,200]<FORK>"
    for split in range(len(raw) + 1):
        sink = Sink()
        stream = BlockStream(sink)
        stream.on_text(raw[:split])
        stream.on_text(raw[split:])
        stream.end()
        assert sink.text == normalize_blocks(raw)
        assert "<BLOCK>" not in sink.text
        assert "<|ref|>image<|/ref|>" in sink.text


def test_official_prompt_and_sampling(tmp_path):
    p = tmp_path / "p.png"
    Image.new("RGB", (2, 2)).save(p)
    engine = SGLangEngine(Settings())
    payload = engine.payload([p, p], single=False, rid="request-1")
    assert (
        payload["messages"][0]["content"][-1]["text"]
        == "<PAGE><image><PAGE><image>Multi page parsing."
    )
    assert payload["skip_special_tokens"] is False
    assert payload["custom_params"]["page_token_id"] == 128827
    assert payload["custom_params"]["window_size"] == 1024
    assert payload["images_config"] == {"crop_mode": False}
    assert payload["rid"] == "request-1"


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_artifacts_and_truncation(tmp_path, monkeypatch, finish):
    engine = SGLangEngine(Settings())
    monkeypatch.setattr(engine, "load", lambda: None)

    async def stream(payload, sink, cancel, repetition):
        return "<PAGE><BLOCK>image[0,0,500,500]<FORK>\ncaption", finish

    monkeypatch.setattr(engine, "_stream", stream)
    p = tmp_path / "p.png"
    Image.new("RGB", (100, 100)).save(p)
    if finish == "length":
        with pytest.raises(OutputLimitError) as error:
            engine.run_multi([p], tmp_path / "out", NullSink(), threading.Event())
        assert "images/page_0_0.jpg" in error.value.partial_output
    else:
        assert "caption" in engine.run_multi(
            [p], tmp_path / "out", NullSink(), threading.Event()
        )
    assert (tmp_path / "out/images/page_0_0.jpg").is_file()
    assert (tmp_path / "out/raw_pages.json").is_file()


@pytest.mark.parametrize("kind", ["cancel", "truncated", "limit"])
def test_stream_cleanup_and_failures(monkeypatch, kind):
    requests = []

    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            if kind == "cancel":
                await asyncio.sleep(10)
            elif kind == "limit":
                yield b"x" * (1024 * 1024 + 1)
            else:
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    def handle(request):
        requests.append(request)
        if request.url.path == "/abort_request":
            return httpx.Response(200)
        return httpx.Response(200, stream=Bytes())

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    e = SGLangEngine(Settings(sidecar_max_response_mb=1))
    cancel = threading.Event()
    if kind == "cancel":
        cancel.set()
    from app.engine.repetition import SemanticRepetitionDetector

    with pytest.raises(JobCanceled if kind == "cancel" else EngineError):
        asyncio.run(
            e._stream({"rid": "mine"}, NullSink(), cancel, SemanticRepetitionDetector())
        )
    assert requests[-1].url.path == "/abort_request"
    assert json.loads(requests[-1].content) == {"rid": "mine"}


def test_server_identity_mismatch_is_fatal(monkeypatch):
    def handle(request):
        return httpx.Response(
            200,
            json={
                "model_path": "/models/old/snapshots/wrong",
                "version": "0.5.20.dev13+g75b9b06b02",
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(EngineError, match="identity mismatch"):
        SGLangEngine(Settings()).load()


def test_server_page_guard_cannot_hide_truncated_pages():
    from app.engine.sglang import page_budget_reached

    assert page_budget_reached([128827, 5, 6, 7, 128827, 8, 1], 3)
    assert not page_budget_reached([128827, 5, 6, 128827, 8, 1], 3)
