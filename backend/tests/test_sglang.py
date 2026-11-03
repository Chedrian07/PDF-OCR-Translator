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
        "/server_info": (
            200,
            {
                "version": "0.5.20.dev13+g75b9b06b02",
                "grammar_backend": "none",
                "disable_outlines_disk_cache": True,
            },
        ),
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
    "changes",
    [
        {"grammar_backend": "outlines"},
        {"grammar_backend": "xgrammar"},
        {"disable_outlines_disk_cache": False},
        {"disable_outlines_disk_cache": "true"},
    ],
)
def test_runtime_optional_grammar_cache_configuration_is_verified(
    sglang_service, changes
):
    engine, routes, _ = sglang_service
    engine.load()
    runtime = {**routes["/server_info"][1], **changes}
    routes["/server_info"] = (200, runtime)
    with pytest.raises(EngineError, match="optional grammar/cache must be disabled"):
        engine.load()
    assert not engine.loaded


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


def test_provider_health_separates_startup_from_a_lost_sidecar(sglang_service):
    """준비된 적 없는 미준비는 starting(정상 기동 — 웹은 '로딩 중'), 준비된 뒤의 미준비는
    unavailable(웹의 '엔진 서버 연결 안 됨')이다 — 예전에는 둘 다 unavailable이었다."""
    engine, routes, _ = sglang_service
    routes["/ready"] = (503, None)
    health = engine.provider_health()
    assert health["status"] == "starting"
    assert health["error"] == "Unlimited-OCR-2 SGLang is not ready"
    routes["/ready"] = (200, None)
    assert engine.provider_health()["status"] == "ready"
    routes["/ready"] = (503, None)
    assert engine.provider_health()["status"] == "unavailable"
    assert not engine.loaded


def _stuck_load(monkeypatch, engine):
    """정지된 sidecar 컨테이너의 대역 — 이름 조회(getaddrinfo)가 httpx 타임아웃 밖에서 멈춘다."""
    release, entered = threading.Event(), []

    def stuck():
        entered.append(threading.current_thread().name)
        assert release.wait(10)
        engine._mark_unavailable()
        raise SGLangUnavailable("Unlimited-OCR-2 SGLang is not ready")

    monkeypatch.setattr(engine, "load", stuck)
    return release, entered


def test_health_does_not_wait_for_a_stuck_sidecar_probe(sglang_service, monkeypatch):
    """health는 확인을 상한까지만 기다리고 직전 결과를 낸다 — 예전에는 정지된 sidecar에서
    매번 10초 걸렸다. 멈춘 확인 위에 확인을 쌓지 않고, 끝나면 다음 health가 그 결과를 본다."""
    import time

    import app.engine.sglang as sglang_mod

    monkeypatch.setattr(sglang_mod, "HEALTH_PROBE_WAIT_S", 0.2)
    engine, _, _ = sglang_service
    assert engine.provider_health()["status"] == "ready"
    release, entered = _stuck_load(monkeypatch, engine)
    started = time.monotonic()
    for _ in range(3):
        assert engine.provider_health()["status"] == "ready"   # 직전 결과 — 오경보 없음
    assert time.monotonic() - started < 2
    assert entered == ["sglang-health-probe"]                 # 확인은 하나만 돈다
    release.set()
    deadline = time.monotonic() + 5
    while engine.provider_health()["status"] != "unavailable":
        assert time.monotonic() < deadline, "끝난 확인의 결과가 보이지 않는다"
    assert not engine.loaded


@pytest.mark.parametrize("preloaded, expected", [(False, "starting"), (True, "ready")])
def test_health_before_any_probe_finished_uses_the_last_load(
    sglang_service, monkeypatch, preloaded, expected,
):
    """끝난 확인이 아직 없으면 마지막 load()(프리로드·워커) 결과로 답한다 — 첫 확인이 늦다고
    준비된 sidecar를 '연결 안 됨'으로, 기동 중인 sidecar를 장애로 보이지 않는다."""
    import app.engine.sglang as sglang_mod

    monkeypatch.setattr(sglang_mod, "HEALTH_PROBE_WAIT_S", 0.2)
    engine, _, _ = sglang_service
    if preloaded:
        engine.load()
    release, _ = _stuck_load(monkeypatch, engine)
    try:
        assert engine.provider_health()["status"] == expected
    finally:
        release.set()


def test_health_probe_errors_still_reach_the_health_route(sglang_service, monkeypatch):
    """확인 스레드의 예기치 못한 예외도 health 호출자에게 전달된다(api.health가 error로 낸다) —
    삼키면 health가 직전 결과를 계속 내고, 다음 확인도 다시 시작된다."""
    engine, _, _ = sglang_service

    def broken():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(engine, "load", broken)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="probe exploded"):
            engine.provider_health()


def test_provider_health_reports_a_misconfigured_sidecar_even_before_first_ready(sglang_service):
    """대기해도 풀리지 않는 불일치는 기동 중으로 숨기지 않는다."""
    engine, routes, _ = sglang_service
    routes["/server_info"] = (200, {**routes["/server_info"][1], "version": "wrong"})
    health = engine.provider_health()
    assert health["status"] == "unavailable"
    assert "version mismatch" in health["error"]


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
