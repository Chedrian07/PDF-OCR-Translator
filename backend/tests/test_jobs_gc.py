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


def _run_fake_job(tmp_path, engine=None):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
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


def test_work_dir_removed_on_error(tmp_path):
    """전 청크 실패(status=error)여도 실패 청크의 work/ 잔여물이 남지 않는다."""

    class FailingEngine(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            out_dir.mkdir(parents=True, exist_ok=True)  # 실패 전 부분 산출물 흉내
            raise RuntimeError("모의 실패")

    job = _run_fake_job(tmp_path, engine=FailingEngine(delay=0.0))
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
