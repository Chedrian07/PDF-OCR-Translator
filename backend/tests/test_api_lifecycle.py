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


# ── F1-7: 정상 종료가 열린 SSE 스트림에 막히지 않는다 ───────────────────────────
def test_shutdown_hooks_chain_the_server_handlers_and_flag_the_app():
    """uvicorn은 연결이 다 닫힌 뒤에야 lifespan 종료를 보낸다 — 앱은 서버의 신호
    처리기를 감싸 종료 요청을 알아야 한다. 원래 처리기(uvicorn handle_exit)는 그대로
    불리고, 수명이 끝나면 원래대로 돌려 놓는다."""
    import signal
    import threading

    from app.main import ShutdownSignal, _install_shutdown_hooks, _remove_shutdown_hooks

    assert threading.current_thread() is threading.main_thread()
    original = signal.getsignal(signal.SIGTERM)
    received: list[int] = []

    def server_handler(signum, frame):              # uvicorn Server.handle_exit 자리
        received.append(signum)

    signal.signal(signal.SIGTERM, server_handler)
    flag = ShutdownSignal()
    try:
        installed = _install_shutdown_hooks(flag)
        assert signal.SIGTERM in installed
        signal.raise_signal(signal.SIGTERM)
        assert flag.requested is True
        assert received == [signal.SIGTERM]           # 서버의 정상 종료 절차도 그대로
        _remove_shutdown_hooks(installed)
        assert signal.getsignal(signal.SIGTERM) is server_handler
    finally:
        signal.signal(signal.SIGTERM, original)


def test_shutdown_hooks_do_nothing_off_the_main_thread():
    import threading

    from app.main import ShutdownSignal, _install_shutdown_hooks

    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(installed=_install_shutdown_hooks(ShutdownSignal())),
    )
    thread.start()
    thread.join(5)
    assert result["installed"] == {}


def _drive_until_end(app, make_response, *, after: int, action) -> tuple[list[str], float]:
    """SSE body_iterator를 직접 돌린다(TestClient는 스트림 전체를 버퍼링한다).
    after개 청크를 받은 뒤 action()을 부르고, 스트림이 스스로 끝날 때까지 잰다."""
    from starlette.requests import Request

    async def drive():
        request = Request({"type": "http", "app": app, "headers": []})

        async def receive():
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        request._receive = receive
        response = await make_response(request)
        chunks: list[str] = []
        started = None
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if len(chunks) == after:
                    action()
                    started = time.monotonic()
        finally:
            await response.body_iterator.aclose()
        return chunks, time.monotonic() - started

    return asyncio.run(asyncio.wait_for(drive(), timeout=10))


def test_job_sse_stream_ends_promptly_once_shutdown_starts(client):
    from app.api import job_events

    store = client.app.state.store
    job = store.create("stream.pdf", "multi", 200)            # 제출하지 않은 대기 잡 — 이벤트 없음
    shutdown = client.app.state.shutdown
    try:
        chunks, elapsed = _drive_until_end(
            client.app, lambda request: job_events(request, job.id),
            after=2, action=lambda: setattr(shutdown, "requested", True),
        )
        assert chunks[1].startswith("event: progress")
        assert elapsed < 2.5                                   # 다음 폴(≤1s)에서 끝난다
    finally:
        shutdown.requested = False
        store.delete_dir(job)


def test_translation_sse_stream_ends_promptly_once_shutdown_starts(client):
    import json
    import threading

    from app.api import translate_events
    from app.pipeline import artifacts

    st = client.app.state
    job = st.store.create("translating.pdf", "multi", 200)
    job.status = "done"
    state_path = artifacts.translate_state(job.dir, "ko")
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"lang": "ko", "status": "running", "current": 1, "total": 9}))
    with st.translate_lock:
        st.translate_tasks[(job.id, "ko")] = {"thread": None, "cancel": threading.Event()}
    try:
        chunks, elapsed = _drive_until_end(
            client.app, lambda request: translate_events(request, job.id, "ko"),
            after=2, action=lambda: setattr(st.shutdown, "requested", True),
        )
        assert chunks[1].startswith("event: progress")
        assert elapsed < 2.5
    finally:
        st.shutdown.requested = False
        with st.translate_lock:
            st.translate_tasks.pop((job.id, "ko"), None)
        st.store.delete_dir(job)


def test_closed_app_flags_shutdown_for_leftover_streams(settings):
    from app.main import create_app

    app = create_app(settings)
    with TestClient(app):
        assert app.state.shutdown.requested is False
    assert app.state.shutdown.requested is True


# ── 모델 로드 실패·로딩 대기 중 취소 (api-jobs-16 · tests-baseline-7) ────────────
def test_worker_model_load_failure_fails_the_job_and_shows_in_health(
    client, sample_pdf, monkeypatch,
):
    """PRELOAD_MODEL=0에서 첫 잡의 로드가 실패하면 잡은 '모델 로드 실패'로 끝나는데,
    health는 model_load_error:null(=아직 로딩 전)로 보였다 — 워커의 실패도 드러낸다."""
    engine = client.app.state.engine
    assert client.app.state.settings.preload_model is False

    def _unavailable():
        raise RuntimeError("MPS backend not available (simulated)")

    monkeypatch.setattr(engine, "load", _unavailable)
    failed = _upload(client, sample_pdf)
    body = wait_done(client, failed)
    assert body["status"] == "error"
    assert body["error"].startswith("모델 로드 실패: MPS backend not available")
    health = client.get("/api/health").json()
    assert health["model_loaded"] is False
    assert "MPS backend not available" in health["model_load_error"]
    assert health["worker_alive"] is True

    monkeypatch.undo()                                  # 다음 잡에서 로드 재시도 → 성공
    assert wait_done(client, _upload(client, sample_pdf))["status"] == "done"
    health = client.get("/api/health").json()
    assert health["model_loaded"] is True and health["model_load_error"] is None
    assert client.app.state.load_state["error"] is None


def test_cancel_while_waiting_for_the_model_finishes_as_canceled(client, sample_pdf, monkeypatch):
    """워커가 맡은 뒤(모델 로딩 대기 중)의 취소는 canceling → 워커가 canceled로 마감한다."""
    import threading

    from app.engine.base import JobCanceled

    engine = client.app.state.engine
    entered = threading.Event()

    def _wait_for_model(cancel, on_wait=None):
        entered.set()
        assert cancel.wait(10), "취소가 오지 않았다"
        raise JobCanceled()

    monkeypatch.setattr(engine, "wait_until_ready", _wait_for_model)
    jid = _upload(client, sample_pdf)
    assert entered.wait(5)
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 202 and r.json()["status"] == "canceling"   # 이미 워커가 맡았다
    body = wait_done(client, jid)
    assert body["status"] == "canceled" and body["error"] == "사용자에 의해 취소되었습니다"
    monkeypatch.undo()
    assert wait_done(client, _upload(client, sample_pdf))["status"] == "done"


# ── 번역 라우트 수명주기 (tests-baseline-9 · tests-baseline-7) ───────────────────
@pytest.fixture
def translate_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    return monkeypatch


def _wait_translate_idle(client, jid: str, lang: str = "ko", timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with client.app.state.translate_lock:
            if (jid, lang) not in client.app.state.translate_tasks:
                return
        time.sleep(0.02)
    raise AssertionError("번역 스레드가 끝나지 않았다")


def test_translate_failure_before_the_engine_starts_records_the_real_reason(
    client, sample_pdf, translate_env,
):
    """엔진 밖(클라이언트 생성)에서 실패하면 state.json이 접수 때의 running으로 남아,
    다음 조회가 '서버가 재시작되어 번역이 중단되었습니다'로 오보했다(재현)."""
    jid = _upload(client, sample_pdf)
    assert wait_done(client, jid)["status"] == "done"

    def _bad_client(*args, **kwargs):
        raise ValueError("Invalid IPv6 URL")            # OPENAI_BASE_URL='http://[::1/v1'

    def _never(*args, **kwargs):
        raise AssertionError("엔진까지 가면 안 된다")

    translate_env.setattr("app.api.OpenAICompatClient", _bad_client)
    translate_env.setattr("app.api.run_translation", _never)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_translate_idle(client, jid)
    state = client.get(f"/api/jobs/{jid}/translate/state?lang=ko").json()
    assert state["status"] == "error"
    assert "Invalid IPv6 URL" in state["error"]
    assert "재시작" not in state["error"]
    assert state["finished_at"]


def test_duplicate_translate_post_while_running_is_200_running(client, sample_pdf, translate_env):
    import json
    import threading
    from types import SimpleNamespace

    jid = _upload(client, sample_pdf)
    assert wait_done(client, jid)["status"] == "done"
    gate = threading.Event()
    calls: list[int] = []

    def _slow_translation(job_dir, lang, cfg, **kwargs):
        calls.append(1)
        state = job_dir / "translations" / lang / "state.json"
        state.write_text(json.dumps({"lang": lang, "status": "running", "current": 0, "total": 1}))
        assert gate.wait(10)
        state.write_text(json.dumps({"lang": lang, "status": "done", "current": 1, "total": 1}))
        return SimpleNamespace(status="done", total=1, translated=1, cached=0, skipped=0,
                               kept_original=[])

    translate_env.setattr("app.api.run_translation", _slow_translation)
    try:
        assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
        again = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
        assert again.status_code == 200
        assert again.json() == {"job_id": jid, "lang": "ko", "status": "running"}
    finally:
        gate.set()
        _wait_translate_idle(client, jid)
    assert calls == [1]                                    # 중복 실행 없음


# ── 좌표 레이아웃이 없는 잡(figure-only)의 폴백 (tests-baseline-7) ──────────────────
def test_done_job_without_layout_falls_back_to_semantic_exports(client, sample_pdf):
    jid = _upload(client, sample_pdf)
    assert wait_done(client, jid)["status"] == "done"
    job_dir = client.app.state.store.get(jid).dir
    (job_dir / "layout.json").unlink()                     # figure_only 엔진의 결과 형태

    doc = client.get(f"/api/jobs/{jid}/document.html")
    assert doc.status_code == 200
    assert "layout-page-image" not in doc.text             # facsimile이 아니라 읽기용 HTML
    assert 'class="doc-page"' in doc.text
    page = client.get(f"/api/jobs/{jid}/page/1")
    assert page.status_code == 200 and page.headers["content-type"].startswith("image/png")
    (job_dir / "result.ko.md").write_text("# 번역", encoding="utf-8")
    pdf = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert pdf.status_code == 409 and "document.html" in pdf.json()["detail"]
