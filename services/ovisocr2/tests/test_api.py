"""HTTP 계층 테스트 — fastapi·httpx·pillow가 있을 때만 (CI sidecar 잡이 설치한다).

모델·vLLM·GPU 없이 가짜 엔진을 꽂아 /health 확장 필드, 엔진 사망 시 503 + 재시작 예약,
출력 상한 절단 경고를 확인한다(감사 sidecar-6·7·11).
"""

import io
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
Image = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

from app import lifecycle  # noqa: E402
from app import main as sidecar  # noqa: E402


class _Completion:
    def __init__(self, text, finish_reason):
        self.text = text
        self.finish_reason = finish_reason


class _FakeLLM:
    def __init__(self, result):
        self.result = result

    def generate(self, requests, params, use_tqdm=False):
        if isinstance(self.result, BaseException):
            raise self.result
        return [type("Out", (), {"outputs": [_Completion(*self.result)]})()]


class EngineDeadError(RuntimeError):
    pass


@pytest.fixture
def client(monkeypatch):
    m = sidecar.model
    monkeypatch.setattr(m, "_llm", None)
    monkeypatch.setattr(m, "_prompt", "<prompt>")
    monkeypatch.setattr(m, "_sampling_cls", lambda **kw: kw)
    monkeypatch.setattr(m, "_infer_failures", 0)
    monkeypatch.setattr(m, "load_error", None)
    monkeypatch.setattr(m, "load_retry", None)
    monkeypatch.setattr(m, "restart_required", False)
    restarts: list[str] = []
    monkeypatch.setattr(lifecycle, "schedule_restart",
                        lambda log, reason, **kw: restarts.append(reason) or True)
    c = TestClient(sidecar.app)  # with 블록 없이 — lifespan(실제 모델 로드)을 돌리지 않는다
    c.restarts = restarts
    return c


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, "PNG")
    return buf.getvalue()


def _parse(client):
    return client.post("/v1/parse", files={"file": ("p.png", _png(), "image/png")},
                       data={"page_index": "0", "request_id": "t", "options": "{}"})


def test_health_exposes_retry_and_restart_state(client):
    sidecar.model.load_retry = {"attempt": 1, "max_attempts": 5, "next_retry_s": 15.0,
                                "last_error": "ConnectionError: blip"}
    h = client.get("/health").json()
    # 재시도 대기 중에는 status=ok·model_loaded=false — backend가 '로딩 중'으로 기다린다
    assert h["status"] == "ok" and h["model_loaded"] is False
    assert h["load_retry"]["attempt"] == 1 and h["restarting"] is False


def test_truncated_page_is_flagged_and_warned(client):
    sidecar.model._llm = _FakeLLM(("<table><tr><td>잘린 표", "length"))
    r = _parse(client)
    assert r.status_code == 200
    page = r.json()["page"]
    assert page["truncated"] is True
    assert any("출력 토큰 상한" in w for w in page["warnings"])


def test_complete_page_is_not_flagged(client):
    sidecar.model._llm = _FakeLLM(("# 제목\n본문", "stop"))
    page = _parse(client).json()["page"]
    assert page["truncated"] is False
    assert not any("출력 토큰 상한" in w for w in page["warnings"])


def test_engine_death_answers_503_and_schedules_a_restart(client):
    """502는 backend에서 페이지 실패로 확정되지만 503은 재기동을 기다려 그 페이지를 다시 보낸다."""
    sidecar.model._llm = _FakeLLM(EngineDeadError("EngineCore encountered an issue"))
    r = _parse(client)
    assert r.status_code == 503 and "재시작" in r.json()["detail"]
    assert len(client.restarts) == 1
    # 재시작 전까지 들어오는 요청도 503(기다리면 풀림)이고, health는 하드 실패 조합이 아니다
    assert _parse(client).status_code == 503
    h = client.get("/health").json()
    assert h["restarting"] is True and h["model_loaded"] is False and h["status"] == "ok"


def test_ordinary_inference_failure_is_still_502(client):
    sidecar.model._llm = _FakeLLM(ValueError("bad page"))
    assert _parse(client).status_code == 502
    assert client.restarts == []


def test_urlencoded_flood_is_rejected_by_form_limits(client):
    """starlette 1.3.1(GHSA-82w8-qh3p-5jfq 수정판)은 urlencoded 폼에도 필드 수 상한을 건다.

    예전 스택(starlette 0.41·python-multipart 0.0.20)은 필드를 전부 파싱한 뒤 422(file
    누락)로 답했다 — 수십만 필드면 그 파싱 동안 이벤트 루프가 멈춘다."""
    sidecar.model._llm = _FakeLLM(("ok", "stop"))
    body = "&".join(f"f{i}=x" for i in range(5000))
    r = client.post("/v1/parse", content=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 400, r.text
    assert "Too many fields" in json.dumps(r.json())
