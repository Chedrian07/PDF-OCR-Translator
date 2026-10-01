"""잡 수명주기 회귀 테스트 — 대기·실행 중 취소/삭제, 재시작 복원, 모델 로드 실패, 종료.

전체 스위트가 한 번도 실행하지 않던 분기(대기·실행 중 잡 삭제, 워커의 로드 실패 마감
등)를 실제 Worker + FakeEngine으로 태운다(감사 tests-baseline-7).
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from conftest import wait_done


def _upload(client, pdf_bytes: bytes):
    response = client.post(
        "/api/jobs", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _wait_status(client, job_id: str, status: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body.get("status") == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"{job_id}가 {status}가 되지 않음: {client.get(f'/api/jobs/{job_id}').json()}")


@pytest.fixture
def slow_client(settings):
    """FakeEngine 페이지당 0.4s — 앞 잡이 도는 동안 뒤 잡이 대기열에 머문다."""
    from app.main import create_app

    settings.fake_delay = 0.4
    with TestClient(create_app(settings)) as client:
        yield client


# ── F1-5: 대기 잡 취소·삭제는 즉시 반영된다 ─────────────────────────────────
def test_canceling_a_queued_job_behind_a_running_one_is_immediate(slow_client, sample_pdf):
    """예전에는 앞 잡이 끝날 때까지 queued·대기열 위치 1·'취소 중…'에 머물렀고, 뒤 잡들의
    위치도 그만큼 부풀었다. 아직 워커가 맡지 않은 대기 잡은 즉시 canceled로 마감한다."""
    client = slow_client
    broker = client.app.state.broker
    running = _upload(client, sample_pdf)
    _wait_status(client, running, "running")
    queued = _upload(client, sample_pdf)
    behind = _upload(client, sample_pdf)
    assert client.get(f"/api/jobs/{behind}").json()["queue_position"] == 2
    subscriber = broker.subscribe(queued)
    try:
        r = client.post(f"/api/jobs/{queued}/cancel")
        assert r.status_code == 202
        assert r.json()["status"] == "canceled"
        body = client.get(f"/api/jobs/{queued}").json()
        assert body["status"] == "canceled" and "queue_position" not in body
        assert client.get(f"/api/jobs/{behind}").json()["queue_position"] == 1
        # 열려 있던 SSE 구독자는 바로 종료 이벤트를 받는다
        assert subscriber.get(timeout=2) == (
            "error", {"message": "사용자에 의해 취소되었습니다", "canceled": True},
        )
    finally:
        broker.unsubscribe(queued, subscriber)

    # 재접속 스냅샷도 즉시 종료다
    with client.stream("GET", f"/api/jobs/{queued}/events") as stream:
        events = [line for line in stream.iter_lines() if line.startswith("event: ")]
    assert events == ["event: error"]

    # 워커는 취소된 잡을 건너뛰고(다시 돌리지 않는다) 뒤 잡을 처리한다
    assert wait_done(client, behind, timeout=30)["status"] == "done"
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "canceled"
    assert wait_done(client, running, timeout=30)["status"] == "done"


def test_deleting_a_queued_job_closes_its_subscribers(slow_client, sample_pdf):
    """다른 탭의 SSE는 삭제된 대기 잡에서 ping만 받으며 '대기 중'에 영원히 머물렀다."""
    client = slow_client
    broker = client.app.state.broker
    running = _upload(client, sample_pdf)
    _wait_status(client, running, "running")
    queued = _upload(client, sample_pdf)
    job_dir = client.app.state.store.get(queued).dir
    subscriber = broker.subscribe(queued)
    try:
        assert client.delete(f"/api/jobs/{queued}").status_code == 204
        event, data = subscriber.get(timeout=2)
        assert event == "error" and data["deleted"] is True and data["canceled"] is True
    finally:
        broker.unsubscribe(queued, subscriber)
    assert client.get(f"/api/jobs/{queued}").status_code == 404
    assert not job_dir.exists()
    assert wait_done(client, running, timeout=30)["status"] == "done"
    assert client.app.state.worker.is_alive()


def test_deleting_a_running_job_cleans_up_and_keeps_the_worker(slow_client, sample_pdf):
    """실행 중 잡 DELETE는 러너가 취소를 마감한 뒤 디렉터리를 지운다 — 워커는 살아남아
    다음 잡을 처리한다(이 분기는 스위트에서 한 번도 실행되지 않았다)."""
    client = slow_client
    running = _upload(client, sample_pdf)
    _wait_status(client, running, "running")
    job_dir = client.app.state.store.get(running).dir
    after = _upload(client, sample_pdf)

    assert client.delete(f"/api/jobs/{running}").status_code == 204
    assert wait_done(client, after, timeout=30)["status"] == "done"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (
        job_dir.exists() or client.get(f"/api/jobs/{running}").status_code != 404
    ):
        time.sleep(0.02)
    assert not job_dir.exists()
    assert client.get(f"/api/jobs/{running}").status_code == 404
    assert running not in {j["job_id"] for j in client.get("/api/jobs").json()["jobs"]}
    assert client.app.state.worker.is_alive()


def test_sse_stream_closes_when_its_job_vanishes_without_an_event(client):
    """방어: 종료 이벤트 없이 잡이 사라져도(삭제 경합) 스트림이 ping만 받으며 남지 않는다."""
    from starlette.requests import Request

    from app.api import job_events

    store = client.app.state.store
    job = store.create("vanish.pdf", "multi", 200)        # 제출하지 않은 대기 잡

    async def drive() -> list[str]:
        request = Request({"type": "http", "app": client.app, "headers": []})

        async def receive():
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        request._receive = receive
        response = await job_events(request, job.id)
        chunks: list[str] = []
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if len(chunks) == 2:                       # retry + progress 스냅샷
                    store.remove(job.id)                   # 이벤트 없이 사라진다
        finally:
            await response.body_iterator.aclose()
        return chunks

    started = time.monotonic()
    chunks = asyncio.run(asyncio.wait_for(drive(), timeout=10))
    assert time.monotonic() - started < 5
    assert chunks[0] == "retry: 3000\n\n" and chunks[1].startswith("event: progress")
    store.delete_dir(job)


# ── F1-6: 재시작 시 시작하지 못한 대기 잡을 다시 제출한다 ─────────────────────────
def test_restart_resubmits_queued_jobs_that_never_started(settings, sample_pdf):
    """첫 앱은 워커를 띄우지 않아(lifespan 없음) 업로드가 대기열에만 들어간다 — 그 상태로
    '재시작'하면 같은 잡이 오류가 아니라 대기열로 돌아와 처리된다."""
    from app.main import create_app

    first = create_app(settings)
    jid = _upload(TestClient(first), sample_pdf)
    assert first.state.store.get(jid).status == "queued"
    first.state.owner_lock.release()                   # 프로세스 종료 = 소유 락 회수

    with TestClient(create_app(settings)) as client:
        body = wait_done(client, jid, timeout=30)
        assert body["status"] == "done", body
        assert body["error"] is None
