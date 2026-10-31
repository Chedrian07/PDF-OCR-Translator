import asyncio
import json
import threading

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.engine.base import EngineError, JobCanceled, NullSink, OutputLimitError
from app.engine.postprocess import BlockStream, normalize_blocks
from app.engine.sglang import SGLangEngine, SGLangUnavailable


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


@pytest.fixture
def sglang_service(monkeypatch):
    settings = Settings()
    routes = {
        "/ready": (200, None),
        "/model_info": (
            200,
            {
                "model_path": "/models/models--baidu--Unlimited-OCR-2/snapshots/"
                + settings.model_revision,
            },
        ),
        "/server_info": (200, {"version": "0.5.20.dev13+g75b9b06b02"}),
    }
    calls = []

    def handle(request):
        calls.append(request.url.path)
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    return SGLangEngine(settings), routes, calls


def test_repeated_readiness_checks_do_not_generate_tokens(sglang_service):
    engine, _, calls = sglang_service
    engine.load()
    assert engine.provider_health()["status"] == "ready"
    assert engine.loaded
    assert calls == ["/ready", "/model_info", "/server_info"] * 2


@pytest.mark.parametrize(
    "endpoint, response, expected_error",
    [
        ("/ready", (503, None), SGLangUnavailable),
        ("/server_info", (503, None), SGLangUnavailable),
        ("/model_info", (200, {"model_path": "/models/other"}), EngineError),
        ("/server_info", (200, {"version": "wrong"}), EngineError),
        ("/model_info", (200, []), SGLangUnavailable),
        ("/model_info", (200, {"model_path": None}), SGLangUnavailable),
        ("/server_info", (200, []), SGLangUnavailable),
    ],
)
def test_readiness_failure_invalidates_previous_success(
    sglang_service, endpoint, response, expected_error
):
    engine, routes, _ = sglang_service
    engine.load()
    routes[endpoint] = response
    with pytest.raises(expected_error):
        engine.load()
    assert not engine.loaded
    assert engine._health["status"] == "unavailable"
    assert engine.provider_health()["status"] == "unavailable"


@pytest.mark.parametrize(
    "endpoint, response, message",
    [
        ("/model_info", {"model_path": "/models/other"}, "model identity mismatch"),
        ("/server_info", {"version": "wrong"}, "runtime version mismatch"),
    ],
)
def test_each_ocr_request_rechecks_identity(
    sglang_service, monkeypatch, tmp_path, endpoint, response, message
):
    engine, routes, calls = sglang_service
    streams = []

    async def stream(payload, sink, cancel, repetition):
        streams.append(payload)
        return "<PAGE><BLOCK>text[0,0,500,500]<FORK>readable", "stop"

    monkeypatch.setattr(engine, "_stream", stream)
    image = tmp_path / "page.png"
    Image.new("RGB", (20, 20)).save(image)
    engine.run_single(image, tmp_path / "first", NullSink(), threading.Event())
    routes[endpoint] = (200, response)
    with pytest.raises(EngineError, match=message):
        engine.run_single(image, tmp_path / "second", NullSink(), threading.Event())
    assert calls == ["/ready", "/model_info", "/server_info"] * 2
    assert len(streams) == 1
    assert not engine.loaded


@pytest.mark.parametrize("failure", [httpx.ConnectError, SGLangUnavailable, ValueError])
def test_stream_unavailability_invalidates_readiness(
    sglang_service, monkeypatch, tmp_path, failure
):
    engine, _, _ = sglang_service

    async def stream(payload, sink, cancel, repetition):
        raise failure("service disconnected")

    monkeypatch.setattr(engine, "_stream", stream)
    image = tmp_path / "page.png"
    Image.new("RGB", (20, 20)).save(image)
    with pytest.raises(SGLangUnavailable):
        engine.run_single(image, tmp_path / "out", NullSink(), threading.Event())
    assert not engine.loaded
    assert engine._health["status"] == "unavailable"


def test_server_page_guard_cannot_hide_truncated_pages():
    from app.engine.sglang import page_budget_reached

    assert page_budget_reached([128827, 5, 6, 7, 128827, 8, 1], 3)
    assert not page_budget_reached([128827, 5, 6, 128827, 8, 1], 3)


def test_payload_normalizes_only_model_pixels_and_keeps_source(tmp_path):
    import base64
    from io import BytesIO
    from PIL import ImageDraw

    path = tmp_path / "scan.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 9, 19), fill=(80, 80, 80))
    image.save(path)
    original = path.read_bytes()
    payload = SGLangEngine(Settings()).payload([path], single=False, rid="contrast")
    url = payload["messages"][0]["content"][0]["image_url"]["url"]
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as prepared:
        assert prepared.size == image.size
        assert prepared.getpixel((1, 1)) == (0, 0, 0)
        assert prepared.getpixel((50, 50)) == (255, 255, 255)
    assert path.read_bytes() == original
