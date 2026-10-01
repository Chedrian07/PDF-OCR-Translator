"""HTTP 계층 테스트 — fastapi·httpx·pillow가 있을 때만 (CI sidecar 잡이 설치한다).

paddle·GPU 없이 가짜 파이프라인을 꽂아 /health 확장 필드, 끈적한 CUDA 오류 시 503 +
재시작 예약, 폼 필드 상한을 확인한다(감사 sidecar-6·7, gap2-6).
"""

import io
import json
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
Image = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

from app import lifecycle  # noqa: E402
from app import main as sidecar  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "official_page.json")
                     .read_text(encoding="utf-8"))


class _FakePipeline:
    def __init__(self, result):
        self.result = result

    def predict(self, image_path, **kwargs):
        if isinstance(self.result, BaseException):
            raise self.result
        return [types.SimpleNamespace(json=self.result)]


@pytest.fixture
def client(monkeypatch):
    m = sidecar.model
    monkeypatch.setattr(m, "_pipeline", None)
    monkeypatch.setattr(m, "_infer_failures", 0)
    monkeypatch.setattr(m, "load_error", None)
    monkeypatch.setattr(m, "load_retry", None)
    monkeypatch.setattr(m, "restart_required", False)
    monkeypatch.setattr(m, "release_cache", lambda: None)
    restarts: list[str] = []
    monkeypatch.setattr(lifecycle, "schedule_restart",
                        lambda log, reason, **kw: restarts.append(reason) or True)
    c = TestClient(sidecar.app)  # with 블록 없이 — lifespan(실제 파이프라인 로드)을 돌리지 않는다
    c.restarts = restarts
    return c


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1240, 1754), "white").save(buf, "PNG")
    return buf.getvalue()


def _parse(client):
    return client.post("/v1/parse", files={"file": ("p.png", _png(), "image/png")},
                       data={"page_index": "0", "request_id": "t", "options": "{}"})


def test_health_exposes_retry_and_restart_state(client):
    sidecar.model.load_retry = {"attempt": 2, "max_attempts": 5, "next_retry_s": 30.0,
                                "last_error": "ConnectionError: blip"}
    h = client.get("/health").json()
    # 재시도 대기 중에는 status=ok·model_loaded=false — backend가 '로딩 중'으로 기다린다
    assert h["status"] == "ok" and h["model_loaded"] is False
    assert h["load_retry"]["attempt"] == 2 and h["restarting"] is False


def test_successful_parse_still_works(client):
    sidecar.model._pipeline = _FakePipeline(FIXTURE)
    r = _parse(client)
    assert r.status_code == 200, r.text
    assert "2026년 상반기 연구 보고서" in r.json()["page"]["markdown"]


def test_sticky_cuda_error_answers_503_and_schedules_a_restart(client):
    """502는 backend에서 페이지 실패로 확정되지만 503은 재기동을 기다려 그 페이지를 다시 보낸다."""
    sidecar.model._pipeline = _FakePipeline(
        OSError("(External) CUDA error(700), an illegal memory access was encountered."))
    r = _parse(client)
    assert r.status_code == 503 and "재시작" in r.json()["detail"]
    assert len(client.restarts) == 1
    assert _parse(client).status_code == 503
    h = client.get("/health").json()
    assert h["restarting"] is True and h["model_loaded"] is False and h["status"] == "ok"


def test_ordinary_inference_failure_is_still_502(client):
    sidecar.model._pipeline = _FakePipeline(ValueError("bad page"))
    assert _parse(client).status_code == 502
    assert client.restarts == []


def test_urlencoded_flood_is_rejected_by_form_limits(client):
    """starlette 1.3.1(GHSA-82w8-qh3p-5jfq 수정판)은 urlencoded 폼에도 필드 수 상한을 건다.

    예전 lock(fastapi 0.115.6 → starlette 0.41)은 필드를 전부 파싱한 뒤 422로 답했다."""
    sidecar.model._pipeline = _FakePipeline(FIXTURE)
    body = "&".join(f"f{i}=x" for i in range(5000))
    r = client.post("/v1/parse", content=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 400, r.text
    assert "Too many fields" in json.dumps(r.json())
