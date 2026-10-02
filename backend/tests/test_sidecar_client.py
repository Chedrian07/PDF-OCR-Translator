"""SidecarClient 테스트 — 실제 sidecar 없이 stub HTTP 서버로 검증."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
from PIL import Image

from app.engine.base import JobCanceled
from app.sidecar.client import (
    SidecarClient,
    SidecarError,
    SidecarProtocolError,
    SidecarUnavailableError,
)

ENGINE = "ovisocr2"


def _health_body(**over) -> dict:
    body = {
        "status": "ok", "protocol_version": 1, "engine": ENGINE,
        "model_id": "ATH-MaaS/OvisOCR2", "model_revision": "rev",
        "runtime": "vllm", "runtime_version": "0.22.1",
        "device": "cuda", "dtype": "bfloat16",
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "gpu_total_mb": 16384, "gpu_free_mb": 12000, "model_loaded": True,
    }
    body.update(over)
    return body


def _parse_body(**over) -> dict:
    body = {
        "protocol_version": 1, "engine": ENGINE,
        "model_id": "ATH-MaaS/OvisOCR2", "model_revision": "rev",
        "page": {
            "page_index": 0,
            "markdown": "# 제목\n\n[[FIGURE:0]]\n\n본문",
            "blocks": [{"type": "image", "bbox": [100, 200, 800, 700],
                        "content": "", "order": 0, "figure_index": 0}],
            "provider_raw": None, "warnings": [],
        },
        "timings": {"inference_ms": 5.0},
    }
    body.update(over)
    return body


class _StubServer(ThreadingHTTPServer):
    """요청 처리 스레드를 기억해 close()가 기다릴 수 있게 하고, 클라이언트가 먼저 끊은 연결의
    쓰기 실패는 조용히 넘긴다. 예전에는 시간 초과·취소 테스트의 느린 처리 스레드가 테스트가 끝난
    뒤에도 잠들어 있다가, 이미 닫힌 소켓에 쓰며 socketserver가 'Exception occurred during
    processing of request' 배너와 traceback을 pytest 캡처 밖(다른 테스트 출력 사이)에 찍었다
    (P4 macOS 스위트 6회 중 2회)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handler_threads: list[threading.Thread] = []

    def process_request(self, request, client_address):
        thread = threading.Thread(
            target=self.process_request_thread, args=(request, client_address), daemon=True,
        )
        self.handler_threads.append(thread)
        thread.start()

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return  # 클라이언트가 먼저 끊었다(시간 초과·취소 시나리오) — 정상
        super().handle_error(request, client_address)


class StubSidecar:
    """시나리오 주입형 stub — behavior 콜러블이 (status, bytes) 또는 지연을 결정.

    느린 시나리오는 time.sleep 대신 closing.wait(초)로 기다린다 — close()가 깨워 처리 스레드가
    테스트 안에서 끝난다."""

    def __init__(self):
        self.closing = threading.Event()
        self.health_response = _health_body()
        self.parse_behavior = lambda: (200, json.dumps(_parse_body()).encode())
        self.requests_seen = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: N802 — 테스트 소음 제거
                pass

            def do_GET(self):  # noqa: N802
                if self.path != "/health":
                    self.send_error(404)
                    return
                body = json.dumps(outer.health_response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)  # 본문 소비 (multipart 내용은 시나리오와 무관)
                outer.requests_seen.append(self.path)
                status, body = outer.parse_behavior()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:  # 취소 테스트에서 클라이언트가 먼저 끊음
                    pass

        self.server = _StubServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.closing.set()  # 느린 시나리오를 깨운다
        self.server.shutdown()
        self.server.server_close()
        for thread in self.server.handler_threads:
            thread.join(timeout=5)


@pytest.fixture
def stub():
    s = StubSidecar()
    yield s
    s.close()


@pytest.fixture
def image_path(tmp_path) -> Path:
    p = tmp_path / "page.png"
    Image.new("RGB", (100, 140), "white").save(p)
    return p


def _client(url, **over) -> SidecarClient:
    kw = dict(engine_name=ENGINE, connect_timeout_s=2.0, read_timeout_s=5.0,
              health_timeout_s=2.0, max_response_mb=1, retries=1)
    kw.update(over)
    return SidecarClient(url, **kw)


def test_health_ok(stub):
    h = _client(stub.url).health()
    assert h.status == "ok"
    assert h.gpu_name == "NVIDIA GeForce RTX 5070 Ti"
    assert h.model_loaded is True


def test_health_engine_mismatch(stub):
    stub.health_response = _health_body(engine="paddleocr_vl")
    with pytest.raises(SidecarProtocolError, match="엔진 불일치"):
        _client(stub.url).health()


def test_health_unreachable():
    with pytest.raises(SidecarUnavailableError, match="연결할 수 없습니다"):
        _client("http://127.0.0.1:1").health()


def test_parse_success(stub, image_path):
    resp = _client(stub.url).parse_page(image_path, 0, "req-1", {}, None)
    assert resp.engine == ENGINE
    assert resp.page.markdown.startswith("# 제목")
    assert resp.page.blocks[0].bbox == (100, 200, 800, 700)


def test_parse_500_is_sidecar_error(stub, image_path):
    stub.parse_behavior = lambda: (500, json.dumps({"detail": "추론 실패: OOM"}).encode())
    with pytest.raises(SidecarError, match="추론 실패"):
        _client(stub.url).parse_page(image_path, 0, "r", {}, None)


def test_parse_422_is_protocol_error(stub, image_path):
    stub.parse_behavior = lambda: (422, json.dumps({"detail": "options 스키마 위반"}).encode())
    with pytest.raises(SidecarProtocolError, match="요청 거부"):
        _client(stub.url).parse_page(image_path, 0, "r", {}, None)


def test_parse_invalid_json(stub, image_path):
    stub.parse_behavior = lambda: (200, b'{"protocol_version": 1, "eng')  # 잘린 응답
    with pytest.raises(SidecarProtocolError, match="유효한 JSON이 아닙니다"):
        _client(stub.url).parse_page(image_path, 0, "r", {}, None)


def test_parse_schema_violation(stub, image_path):
    stub.parse_behavior = lambda: (200, json.dumps({"protocol_version": 1}).encode())
    with pytest.raises(SidecarProtocolError, match="스키마 위반"):
        _client(stub.url).parse_page(image_path, 0, "r", {}, None)


def test_parse_protocol_version_mismatch(stub, image_path):
    stub.parse_behavior = lambda: (200, json.dumps(_parse_body(protocol_version=2)).encode())
    with pytest.raises(SidecarProtocolError, match="버전 불일치"):
        _client(stub.url).parse_page(image_path, 0, "r", {}, None)


def test_parse_oversized_response(stub, image_path):
    huge = _parse_body()
    huge["page"]["markdown"] = "가" * (1024 * 1024)  # 1MB 상한 초과 (UTF-8 3바이트)
    stub.parse_behavior = lambda: (200, json.dumps(huge).encode())
    with pytest.raises(SidecarProtocolError, match="상한.*초과"):
        _client(stub.url, max_response_mb=1).parse_page(image_path, 0, "r", {}, None)


def test_parse_read_timeout(stub, image_path):
    def slow():
        stub.closing.wait(3.0)
        return (200, json.dumps(_parse_body()).encode())

    stub.parse_behavior = slow
    t0 = time.monotonic()
    with pytest.raises(SidecarUnavailableError, match="응답 시간 초과"):
        _client(stub.url, read_timeout_s=0.5).parse_page(image_path, 0, "r", {}, None)
    assert time.monotonic() - t0 < 2.5


def test_parse_connection_refused_retries_then_unavailable(image_path):
    with pytest.raises(SidecarUnavailableError, match="연결할 수 없습니다"):
        _client("http://127.0.0.1:1", retries=1).parse_page(image_path, 0, "r", {}, None)


def test_cancel_mid_request_closes_connection(stub, image_path):
    def very_slow():
        stub.closing.wait(10.0)
        return (200, json.dumps(_parse_body()).encode())

    stub.parse_behavior = very_slow
    cancel = threading.Event()
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(JobCanceled):
            _client(stub.url, read_timeout_s=30.0).parse_page(image_path, 0, "r", {}, cancel)
    finally:
        timer.cancel()
    # 취소는 read_timeout(30s)을 기다리지 않고 즉시 연결을 끊는다
    assert time.monotonic() - t0 < 3.0


def test_pre_send_classification():
    """재시도 대상은 '요청 전달 전 실패'뿐 — 전달 후 실패를 재시도하면 중복 추론."""
    import urllib3

    from app.sidecar.client import is_pre_send_failure  # noqa: PLC0415 — 국소 임포트

    refused = requests.exceptions.ConnectionError(
        urllib3.exceptions.MaxRetryError(
            pool=None, url="/v1/parse",
            reason=urllib3.exceptions.NewConnectionError(None, "Connection refused"),
        )
    )
    assert is_pre_send_failure(refused) is True
    # ConnectTimeout은 ConnectionError의 서브클래스이기도 하다 — 여전히 재시도 대상
    assert is_pre_send_failure(requests.exceptions.ConnectTimeout()) is True
    # 요청 전달 후 서버가 끊음 → 재시도 금지
    mid_flight = requests.exceptions.ConnectionError(
        urllib3.exceptions.ProtocolError("Connection aborted", ConnectionResetError())
    )
    assert is_pre_send_failure(mid_flight) is False
    assert is_pre_send_failure(requests.exceptions.ReadTimeout()) is False
    assert is_pre_send_failure(requests.exceptions.ChunkedEncodingError()) is False


def test_mid_flight_disconnect_not_retried(stub, image_path):
    """서버가 응답 도중 끊으면 재시도하지 않고 즉시 명확한 오류 (중복 추론 방지)."""
    calls = {"n": 0}

    def truncated():
        calls["n"] += 1
        # Content-Length보다 짧게 보내고 닫는다 → 클라이언트는 전송 후 절단을 본다
        return (200, b'{"protocol_version": 1, "engine": "ovisocr2"')

    stub.parse_behavior = truncated
    client = _client(stub.url, retries=1)
    with pytest.raises(SidecarError):
        client.parse_page(image_path, 0, "r", {}, None)
    assert calls["n"] == 1, "전달 후 실패는 재시도하지 않는다"


def test_cancel_before_request(stub, image_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(JobCanceled):
        _client(stub.url).parse_page(image_path, 0, "r", {}, cancel)
    assert stub.requests_seen == []  # 요청 자체를 보내지 않는다


def test_health_body_cut_midway_maps_to_unavailable():
    """응답 본문 도중 연결이 끊기면(재기동 중인 sidecar) requests의 ChunkedEncodingError가
    매핑 없이 새어 나가 준비 대기가 '모델 로드 실패'로 끝났다 — 일시적 오류로 매핑한다."""
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve_truncated() -> None:
        conn, _ = server.accept()
        try:
            conn.recv(4096)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: 1000\r\n\r\n{\"status\": \"ok\""  # 1000바이트 선언, 15바이트 전송
            )
        finally:
            conn.close()

    thread = threading.Thread(target=serve_truncated, daemon=True)
    thread.start()
    try:
        with pytest.raises(SidecarUnavailableError) as excinfo:
            _client(f"http://127.0.0.1:{port}").health()
        assert excinfo.value.transient  # 준비 대기가 계속 기다린다
    finally:
        thread.join(timeout=5)
        server.close()


def test_read_timeout_is_marked_no_same_page_retry():
    """읽기 타임아웃은 같은 페이지 즉시 재요청 금지 — runner가 재시도 없이 페이지 격리로 간다."""
    from app.sidecar.client import SidecarTimeoutError

    assert SidecarTimeoutError("x").retry_same_page is False
    assert getattr(SidecarUnavailableError("x"), "retry_same_page", True) is True


def test_stub_close_leaves_no_handler_thread_or_stray_traceback(image_path, capfd):
    """시간 초과로 클라이언트가 떠난 느린 처리 스레드도 close() 안에서 끝나고, 닫힌 소켓에 쓰는
    실패가 socketserver 배너로 새지 않는다 — 다음 테스트 출력에 섞이던 소음의 회귀 방지."""
    stub = StubSidecar()
    try:
        def slow():
            stub.closing.wait(3.0)
            return (200, json.dumps(_parse_body()).encode())

        stub.parse_behavior = slow
        with pytest.raises(SidecarUnavailableError, match="응답 시간 초과"):
            _client(stub.url, read_timeout_s=0.3).parse_page(image_path, 0, "r", {}, None)
    finally:
        started = time.monotonic()
        stub.close()
    assert time.monotonic() - started < 2.0                 # 3초를 다 기다리지 않는다
    assert stub.server.handler_threads
    assert not any(thread.is_alive() for thread in stub.server.handler_threads)
    assert "Exception occurred during processing of request" not in capfd.readouterr().err
