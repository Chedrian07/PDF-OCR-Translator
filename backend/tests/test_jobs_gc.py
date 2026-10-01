"""리소스 상한 라운드 검증 — 잡 TTL GC(JobStore.gc_expired)·work/ 터미널 정리
및 워커 루프 내구성(JobStore.save 실패·잡 예외에도 워커가 죽지 않는다).

디스크 시계 조작: meta.json mtime을 os.utime으로 과거로 밀어 TTL 경과를 흉내낸다.
"""

import logging
import os
import threading
import time

from app.config import Settings
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore, Worker
from app.main import create_app
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes


def _make_job(store: JobStore, status: str = "done", age_days: float = 0.0):
    job = store.create("doc.pdf", "multi", dpi=72)
    job.status = status
    store.save(job)
    if age_days:
        past = time.time() - age_days * 86400
        os.utime(job.dir / "meta.json", (past, past))
    return job


# ── JobStore.gc_expired ─────────────────────────────────────────


def test_gc_removes_expired_terminal_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs")
    old_done = _make_job(store, "done", age_days=10)
    old_error = _make_job(store, "error", age_days=10)
    assert store.gc_expired(7) == 2
    for job in (old_done, old_error):
        assert store.get(job.id) is None
        assert not job.dir.exists()


def test_gc_keeps_fresh_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs")
    fresh = _make_job(store, "done", age_days=1)
    assert store.gc_expired(7) == 0
    assert store.get(fresh.id) is not None
    assert fresh.dir.exists()


def test_gc_never_deletes_active_jobs(tmp_path):
    """queued/running·보호(번역 스레드 활성) 잡은 아무리 오래돼도 삭제 금지.

    보호 검사는 삭제 직전 잡별 콜백 — GC 패스 도중 시작된 번역도 잡히도록."""
    store = JobStore(tmp_path / "jobs")
    running = _make_job(store, "running", age_days=100)
    queued = _make_job(store, "queued", age_days=100)
    translating = _make_job(store, "done", age_days=100)
    checked: list[str] = []

    def _is_protected(job_id: str) -> bool:
        checked.append(job_id)
        return job_id == translating.id

    assert store.gc_expired(7, is_protected=_is_protected) == 0
    assert translating.id in checked  # 콜백이 실제로 잡별 호출됨
    for job in (running, queued, translating):
        assert store.get(job.id) is not None
        assert job.dir.exists()


def test_gc_translation_activity_counts_as_activity(tmp_path):
    """OCR meta가 TTL을 넘겨도 최근 번역(state.json)이 있으면 보존한다."""
    store = JobStore(tmp_path / "jobs")
    job = _make_job(store, "done", age_days=100)
    tdir = job.dir / "translations" / "ko"
    tdir.mkdir(parents=True)
    (tdir / "state.json").write_text("{}", encoding="utf-8")  # 지금 = 신선한 번역 활동
    assert store.gc_expired(7) == 0
    assert job.dir.exists()

    # 번역 활동까지 오래되면 삭제된다
    past = time.time() - 100 * 86400
    os.utime(tdir / "state.json", (past, past))
    assert store.gc_expired(7) == 1
    assert not job.dir.exists()


def test_gc_disabled_when_ttl_zero(tmp_path):
    store = JobStore(tmp_path / "jobs")
    old = _make_job(store, "done", age_days=1000)
    assert store.gc_expired(0) == 0
    assert store.gc_expired(-1) == 0
    assert store.get(old.id) is not None
    assert old.dir.exists()


# ── work/ 터미널 정리 (runner.execute_job finally) ───────────────


def _run_fake_job(tmp_path, engine=None, pdf_bytes=None):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    if pdf_bytes is None:
        pdf_bytes = make_pdf_bytes(pages=2, with_image=False)
    (job.dir / "source.pdf").write_bytes(pdf_bytes)
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=1,
    )
    engine = engine or FakeEngine(delay=0.0)
    engine.load()
    execute_job(job, store, EventBroker(), engine, settings, threading.Event())
    return job


def test_work_dir_removed_on_done(tmp_path):
    job = _run_fake_job(tmp_path)
    assert job.status == "done"
    assert not (job.dir / "work").exists()
    # 필요 산출물은 병합 시 이미 잡 루트로 이동돼 보존된다
    assert (job.dir / "result.md").is_file()
    assert list((job.dir / "images").glob("*.jpg"))
    assert list((job.dir / "layout").glob("*.jpg"))


def _textless_pdf_bytes(pages: int) -> bytes:
    """텍스트 레이어가 없는 PDF — 1페이지 청크는 텍스트 레이어부터 시도해 복구되므로
    (C-1), 전 청크 실패를 재현하려면 복구할 텍스트가 없어야 한다."""
    import pymupdf

    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    data = doc.tobytes()
    doc.close()
    return data


def test_work_dir_removed_on_error(tmp_path):
    """전 청크 실패(status=error)여도 실패 청크의 work/ 잔여물이 남지 않는다."""

    class FailingEngine(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            out_dir.mkdir(parents=True, exist_ok=True)  # 실패 전 부분 산출물 흉내
            raise RuntimeError("모의 실패")

    job = _run_fake_job(
        tmp_path, engine=FailingEngine(delay=0.0), pdf_bytes=_textless_pdf_bytes(2)
    )
    assert job.status == "error"
    assert not (job.dir / "work").exists()


def test_재시작_중단_잡의_work_잔여물_정리(tmp_path):
    """재시작으로 error 강등된 잡은 다시 실행되지 않아 runner의 finally가 못 돈다 —
    복원 시점에 work/를 치운다. 상태가 안 바뀐 터미널 잡은 그대로 둔다."""
    store = JobStore(tmp_path / "jobs")
    interrupted = _make_job(store, "running")
    (interrupted.dir / "work" / "chunk_00").mkdir(parents=True)
    (interrupted.dir / "work" / "chunk_00" / "boxes.json").write_text("[]", encoding="utf-8")
    finished = _make_job(store, "done")
    (finished.dir / "work").mkdir()

    revived = JobStore(store.jobs_dir)
    revived.load_existing()
    assert revived.get(interrupted.id).status == "error"
    assert not (interrupted.dir / "work").exists()
    assert (finished.dir / "work").exists()


# ── 워커 루프 내구성 (Worker.run 예외 방벽) ──────────────────────


class _SaveFailingStore(JobStore):
    """지정한 잡의 save를 OSError로 실패시켜 마감 경로 붕괴(디스크 만원)를 흉내낸다."""

    def __init__(self, jobs_dir):
        super().__init__(jobs_dir)
        self.fail_ids: set[str] = set()

    def save(self, job):
        if job.id in self.fail_ids:
            raise OSError(28, "No space left on device")
        super().save(job)


def test_잡_예외에도_워커가_살아남아_다음_잡을_처리한다(tmp_path):
    """execute_job이 예외를 관통시켜도 워커 스레드가 죽으면 안 된다.

    죽으면 이후 제출된 잡이 전부 영구 queued로 남고 재시작 외에 복구 수단이 없다."""
    store = _SaveFailingStore(tmp_path / "jobs")
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=1,
    )
    engine = FakeEngine(delay=0.0)
    engine.load()
    cancel_events: dict[str, threading.Event] = {}
    worker = Worker(store, EventBroker(), engine, settings, cancel_events)

    bad = store.create("bad.pdf", "multi", dpi=72)
    (bad.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=1, with_image=False))
    store.fail_ids.add(bad.id)  # 이 잡의 모든 save가 실패 → 오류 마감 경로까지 붕괴
    good = store.create("good.pdf", "multi", dpi=72)
    (good.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=1, with_image=False))

    worker.start()
    try:
        worker.submit(bad)
        worker.submit(good)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and good.status not in ("done", "error"):
            time.sleep(0.02)
        assert good.status == "done"  # 워커가 살아남아 다음 잡을 처리했다
        assert worker.is_alive()
        assert bad.status == "error"  # 실패한 잡은 터미널로 마감(running 고착 없음)
        assert cancel_events == {}  # finally에서 모든 경로의 Event가 정리된다
    finally:
        worker.stop()
        worker.join(timeout=5.0)


def test_queued_cancellation_publishes_terminal_sse_event(tmp_path):
    """대기열에서 취소된 잡도 연결된 SSE를 종료하고 재연결 토큰을 정리한다."""
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    engine = FakeEngine(delay=0.0)
    settings = Settings(engine="fake", device="cpu", data_dir=tmp_path / "data")
    cancel_events: dict[str, threading.Event] = {}
    worker = Worker(store, broker, engine, settings, cancel_events)
    job = store.create("canceled.pdf", "multi", dpi=72)
    broker.publish(job.id, "token", {"text": "previous output"})
    subscriber = broker.subscribe(job.id)

    # 시작 전에 큐와 취소를 준비해 타이밍에 의존하지 않고 dequeue 전 취소를 재현한다.
    worker.submit(job)
    cancel_events[job.id].set()
    worker.stop()
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert job.status == "canceled"
    assert not engine.loaded
    assert cancel_events == {}
    event, data = subscriber.get_nowait()
    assert event == "error"
    assert data == {"message": job.error, "canceled": True}
    broker.unsubscribe(job.id, subscriber)
    reconnected, replay, _truncated = broker.subscribe_with_replay(job.id)
    assert replay == ""
    broker.unsubscribe(job.id, reconnected)


def test_메타_기록_실패는_잡_흐름을_깨지_않는다(tmp_path, monkeypatch, caplog):
    """save는 best-effort — FileNotFoundError(삭제 경합)는 조용히, 그 외 OSError는
    경고만 남기고 삼킨다(호출자·워커로 전파 금지)."""
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)

    def _no_space(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("app.jobs.os.replace", _no_space)
    with caplog.at_level(logging.WARNING, logger="app.jobs"):
        store.save(job)  # 예외가 새어 나오지 않는다
    assert "잡 메타 기록 실패" in caplog.text

    monkeypatch.undo()
    caplog.clear()
    store.delete_dir(job)  # 디렉터리가 사라진 뒤의 save = 정상 경합 경로
    with caplog.at_level(logging.WARNING, logger="app.jobs"):
        store.save(job)
    assert caplog.text == ""


# ── queue_position (제출 순서) ──────────────────────────────────


def test_queue_position은_생성_순서가_아니라_제출_순서를_따른다(tmp_path):
    """create()는 업로드 시작 시점, submit()은 업로드 완료 시점이라 순서가 어긋난다.
    실제 처리 순서는 워커 큐 제출 순서이므로 위치도 그 기준이어야 한다."""
    store = JobStore(tmp_path / "jobs")
    slow = store.create("slow.pdf", "multi", dpi=72)  # 먼저 생성(대용량 업로드 중)
    fast = store.create("fast.pdf", "multi", dpi=72)
    # 아직 아무도 제출 전 — 생성 순서로 안정 정렬
    assert (store.queue_position(slow), store.queue_position(fast)) == (1, 2)

    store.mark_submitted(fast)  # 작은 파일이 먼저 업로드를 끝내 먼저 큐에 들어간다
    assert (store.queue_position(fast), store.queue_position(slow)) == (1, 2)
    store.mark_submitted(slow)
    assert (store.queue_position(fast), store.queue_position(slow)) == (1, 2)

    fast.status = "running"
    assert store.queue_position(fast) is None
    assert store.queue_position(slow) == 1


# ── 목록 응답 경량화 (Job.to_dict include_files) ─────────────────


def test_목록용_result_블록은_파일_URL을_생략한다(tmp_path):
    """include_files=False면 pages/layouts/images 디렉터리 스캔을 건너뛴다.
    키 자체는 유지 → 기존 클라이언트 계약 불변."""
    job = _run_fake_job(tmp_path)
    full = job.to_dict()["result"]
    listed = job.to_dict(include_files=False)["result"]

    assert full["pages"] and full["images"] and full["layouts"]
    assert listed["pages"] == [] and listed["images"] == [] and listed["layouts"] == []
    assert listed.keys() == full.keys()
    assert listed["markdown_url"] == full["markdown_url"]
    assert listed["has_layout"] == full["has_layout"]


# ── lifespan 배선 (main.create_app) ─────────────────────────────


def test_startup_gc_task_wired(settings):
    """JOB_TTL_DAYS>0면 시작 시 1회 GC가 돌아 만료 잡이 사라진다."""
    from fastapi.testclient import TestClient

    settings.job_ttl_days = 7
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    old = _make_job(JobStore(settings.jobs_dir), "done", age_days=10)

    app = create_app(settings)
    with TestClient(app) as client:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and old.dir.exists():
            time.sleep(0.02)
        assert not old.dir.exists()
        assert client.get(f"/api/jobs/{old.id}").status_code == 404


def test_startup_gc_disabled_by_default(settings):
    """기본값(JOB_TTL_DAYS=0)이면 GC 태스크가 아예 뜨지 않아 오래된 잡도 보존."""
    from fastapi.testclient import TestClient

    assert settings.job_ttl_days == 0
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    old = _make_job(JobStore(settings.jobs_dir), "done", age_days=1000)

    app = create_app(settings)
    with TestClient(app) as client:
        time.sleep(0.3)
        assert old.dir.exists()
        assert client.get(f"/api/jobs/{old.id}").status_code == 200


# ── 삭제·GC와 겹친 빌더가 되살린 고아 (concurrency-8) ────────────────────────


def test_delete_dir_marks_the_job_for_inflight_builders(tmp_path):
    """DELETE뿐 아니라 TTL GC도 delete_dir를 지난다 — 진행 중 내보내기 빌더가 끝난 뒤
    되살린 디렉터리를 스스로 치우는 근거(delete_requested)가 여기서 생긴다."""
    store = JobStore(tmp_path / "jobs")
    job = _make_job(store, "done", age_days=10)
    assert job.delete_requested is False
    assert store.gc_expired(7) == 1
    assert job.delete_requested is True


def test_restart_sweeps_orphan_job_dirs_without_meta(tmp_path):
    """meta.json 없는 잡 디렉터리(삭제 직후 끝난 빌더가 되살린 것)는 목록·GC 어디에도
    없어 영구히 남았다 — 재시작 때 잡 ID 형식인 것만 정리한다."""
    store = JobStore(tmp_path / "jobs")
    kept = _make_job(store, "done")
    orphan = store.jobs_dir / "j_0123456789ab"
    orphan.mkdir()
    (orphan / "export.ko.dual.pdf").write_bytes(b"%PDF-1.4 dual")
    foreign = store.jobs_dir / "lost+found"          # 잡이 아닌 디렉터리는 건드리지 않는다
    foreign.mkdir()

    revived = JobStore(store.jobs_dir)
    revived.load_existing()
    assert not orphan.exists()
    assert foreign.is_dir()
    assert revived.get(kept.id) is not None and kept.dir.is_dir()


# ── 재시작 시 대기 잡 복원 (pipeline-ocr-11·concurrency-12) ─────────────────────


def test_restart_keeps_submitted_queued_jobs_and_errors_the_rest(tmp_path):
    """시작도 안 한 대기 잡까지 '서버 재시작으로 중단' 오류가 돼 다시 올려야 했다.
    업로드·검증을 마치고 제출된 잡만 대기 상태로 돌려주고(생성 순서), 업로드 도중 죽은
    잡(제출 표식 없음)·원본이 사라진 잡·실행 중이던 잡은 예전처럼 오류로 마감한다."""
    store = JobStore(tmp_path / "jobs")
    pdf = make_pdf_bytes(pages=1, with_image=False)

    def _queued(name: str, created_at: str, *, submit: bool, source: bytes | None):
        job = store.create(name, "multi", dpi=72)
        job.created_at = created_at
        if source is not None:
            (job.dir / "source.pdf").write_bytes(source)
        if submit:
            store.mark_submitted(job)
        store.save(job)
        return job

    second = _queued("b.pdf", "2026-10-01T00:00:02+00:00", submit=True, source=pdf)
    first = _queued("a.pdf", "2026-10-01T00:00:01+00:00", submit=True, source=pdf)
    uploading = _queued("c.pdf", "2026-10-01T00:00:03+00:00", submit=False, source=b"%PDF-1.4")
    vanished = _queued("d.pdf", "2026-10-01T00:00:04+00:00", submit=True, source=None)
    running = _make_job(store, "running")

    revived = JobStore(store.jobs_dir)
    restored = revived.load_existing()
    assert [job.id for job in restored] == [first.id, second.id]   # 생성 순서
    for job in (first, second):
        restored_job = revived.get(job.id)
        assert restored_job.status == "queued" and restored_job.submitted
    for job in (uploading, vanished, running):
        assert revived.get(job.id).status == "error"
        assert revived.get(job.id).error == "서버 재시작으로 중단되었습니다"


def test_submit_survives_a_failed_submitted_marker_write(tmp_path):
    """제출 표식 기록 실패가 제출 자체를 막으면 업로드가 유령 queued 잡이 된다."""
    store = _SaveFailingStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    store.fail_ids.add(job.id)
    store.mark_submitted(job)                          # 예외가 새지 않는다
    assert job.submitted and job.submit_seq == 1
