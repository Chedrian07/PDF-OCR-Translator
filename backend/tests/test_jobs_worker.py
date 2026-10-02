"""잡 워커(jobs.Worker)와 잡 메타 계약.

- Metal(torch MPS) 잡은 잡 단위 ObjC 오토릴리스 풀 안에서 돈다 — 워커는 끝나지 않는
  스레드라 엔진·디코드 루프 바깥에서 autorelease된 객체의 마지막 회수 지점이다.
  MLX는 실측(8쪽 실가중치 잡 5회 연속: RSS 2627→2629→2645→2645→2647MB, MLX 활성 메모리
  6363MB 고정)에서 잡별 증가가 없어 풀을 켜지 않는다. CPU·CUDA에는 비울 객체가 없다.
- 잡은 실행 시작·종료 시각(started_at/finished_at)을 meta와 API 응답에 남긴다. 모르는
  시각(대기 중 취소의 시작, 재시작으로 중단된 잡의 종료, 구버전 meta)은 null이다.
"""

import contextlib
import time

import pytest

from app.config import Settings
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore, Worker

from conftest import make_pdf_bytes


class _PoolSpy:
    """autorelease_pool 대체 — 켜진(enabled=True) 풀의 현재 깊이를 기록한다."""

    def __init__(self) -> None:
        self.depth = 0
        self.entries: list[bool] = []

    def __call__(self, enabled: bool = True):
        spy = self

        @contextlib.contextmanager
        def _pool():
            spy.entries.append(enabled)
            spy.depth += 1 if enabled else 0
            try:
                yield
            finally:
                spy.depth -= 1 if enabled else 0

        return _pool()


class _DepthRecordingEngine(FakeEngine):
    """run_multi 시점의 풀 깊이를 기록한다(FakeEngine 자체는 풀을 쓰지 않는다)."""

    def __init__(self, device: str, spy: _PoolSpy) -> None:
        super().__init__(device=device, delay=0.0)
        self.spy = spy
        self.depths: list[int] = []

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.depths.append(self.spy.depth)
        return super().run_multi(image_paths, out_dir, sink, cancel)


def _blank_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _run_one_job(tmp_path, engine) -> str:
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0,
    )
    worker = Worker(store, broker, engine, settings, {})
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
    engine.load()
    worker.start()
    try:
        worker.submit(job)
        deadline = time.monotonic() + 30
        while job.status not in ("done", "error", "canceled"):
            assert time.monotonic() < deadline, job.status
            time.sleep(0.02)
    finally:
        worker.stop()
        worker.join(timeout=10)
    return job.status


@pytest.mark.parametrize(
    ("device", "expected_depth"),
    [("metal", 1), ("cpu", 0), ("cuda", 0), ("mlx", 0)],
)
def test_worker_wraps_metal_jobs_in_an_objc_pool(tmp_path, monkeypatch, device, expected_depth):
    import app.engine.objc_pool as objc_pool

    spy = _PoolSpy()
    monkeypatch.setattr(objc_pool, "autorelease_pool", spy)
    engine = _DepthRecordingEngine(device, spy)

    assert _run_one_job(tmp_path, engine) == "done"
    assert engine.depths == [expected_depth]
    assert spy.depth == 0  # 잡이 끝나면 풀을 비운다(예외 경로 포함 with 블록)


def test_worker_pool_flag_follows_the_resolved_engine_device(tmp_path):
    """settings.device는 'auto'일 수 있다 — 실제 디바이스는 registry가 푼 engine.device다."""
    settings = Settings(engine="fake", device="auto", data_dir=tmp_path / "data")
    store = JobStore(tmp_path / "jobs")
    metal = Worker(store, EventBroker(), FakeEngine(device="metal"), settings, {})
    mlx = Worker(store, EventBroker(), FakeEngine(device="mlx"), settings, {})
    assert metal._drains_objc_pool_per_job() is True
    assert mlx._drains_objc_pool_per_job() is False


def test_worker_keeps_running_after_a_job_fails_inside_the_pool(tmp_path, monkeypatch):
    """풀 안에서 잡이 예외로 끝나도 풀은 닫히고 워커 스레드는 다음 잡을 받는다."""
    import app.engine.objc_pool as objc_pool

    spy = _PoolSpy()
    monkeypatch.setattr(objc_pool, "autorelease_pool", spy)

    class _Boom(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            raise RuntimeError("모의 실패")

        def run_single(self, image_path, out_dir, sink, cancel):
            raise RuntimeError("모의 실패")

    engine = _Boom(device="metal", delay=0.0)
    store = JobStore(tmp_path / "jobs")
    settings = Settings(engine="fake", device="cpu", data_dir=tmp_path / "data")
    worker = Worker(store, EventBroker(), engine, settings, {})
    jobs = []
    for _ in range(2):
        job = store.create("doc.pdf", "multi", dpi=72)
        (job.dir / "source.pdf").write_bytes(_blank_pdf())  # 텍스트 레이어 없음 → 복구 불가
        jobs.append(job)
    engine.load()
    worker.start()
    try:
        for job in jobs:
            worker.submit(job)
        deadline = time.monotonic() + 30
        while any(j.status not in ("done", "error", "canceled") for j in jobs):
            assert time.monotonic() < deadline, [j.status for j in jobs]
            time.sleep(0.02)
    finally:
        worker.stop()
        worker.join(timeout=10)
    assert [j.status for j in jobs] == ["error", "error"]  # 첫 잡의 실패 뒤에도 다음 잡을 받았다
    assert spy.entries == [True, True] and spy.depth == 0


# ── 실행 시작·종료 시각(started_at/finished_at) ─────────────────────────────


def test_finished_jobs_record_start_and_finish_times(tmp_path):
    import json
    from datetime import datetime

    engine = FakeEngine(delay=0.0)
    store = JobStore(tmp_path / "jobs")
    settings = Settings(engine="fake", device="cpu", data_dir=tmp_path / "data")
    worker = Worker(store, EventBroker(), engine, settings, {})
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
    assert (job.started_at, job.finished_at) == (None, None)
    engine.load()
    worker.start()
    try:
        worker.submit(job)
        deadline = time.monotonic() + 30
        while job.status != "done":
            assert time.monotonic() < deadline, job.status
            time.sleep(0.02)
    finally:
        worker.stop()
        worker.join(timeout=10)

    started = datetime.fromisoformat(job.started_at)
    finished = datetime.fromisoformat(job.finished_at)
    assert datetime.fromisoformat(job.created_at) <= started <= finished
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert (meta["started_at"], meta["finished_at"]) == (job.started_at, job.finished_at)
    body = job.to_dict()
    assert (body["started_at"], body["finished_at"]) == (job.started_at, job.finished_at)

    restored = JobStore(tmp_path / "jobs")
    restored.load_existing()
    again = restored.get(job.id)
    assert (again.started_at, again.finished_at) == (job.started_at, job.finished_at)


def test_jobs_canceled_before_starting_have_no_start_time(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    assert store.try_cancel_queued(job, "사용자에 의해 취소되었습니다")
    assert job.status == "canceled"
    assert job.started_at is None and job.finished_at is not None


def test_interrupted_and_legacy_jobs_leave_unknown_times_empty(tmp_path):
    """재시작으로 중단된 잡은 실제로 멈춘 시각을 모른다 — 재시작 시각을 적으면 처리 시간이
    서버가 내려가 있던 시간만큼 부풀려 보인다. 구버전 meta에는 두 필드가 없다."""
    import json

    store = JobStore(tmp_path / "jobs")
    running = store.create("doc.pdf", "multi", dpi=72)
    running.mark_running()
    store.save(running)
    legacy_dir = tmp_path / "jobs" / "j_0123456789ab"
    legacy_dir.mkdir()
    (legacy_dir / "meta.json").write_text(json.dumps({
        "id": "j_0123456789ab", "filename": "old.pdf", "mode": "multi", "dpi": 72,
        "status": "done", "created_at": "2026-09-01T00:00:00+00:00",
    }), encoding="utf-8")

    restored = JobStore(tmp_path / "jobs")
    restored.load_existing()
    interrupted = restored.get(running.id)
    assert interrupted.status == "error"
    assert interrupted.started_at == running.started_at and interrupted.finished_at is None
    legacy = restored.get("j_0123456789ab")
    assert (legacy.started_at, legacy.finished_at) == (None, None)
    assert legacy.to_dict()["started_at"] is None


# ── 종료 요청 뒤 워커는 새 잡을 맡지 않는다 (P4 docker stop) ─────────────────────


def test_stopped_worker_leaves_queued_jobs_for_the_next_start(tmp_path):
    """stop() 뒤에는 큐에 남은 대기 잡을 꺼내지 않는다 — queued·제출 표식 그대로 남아 다음
    기동(load_existing)이 다시 제출한다. 예전에는 sentinel 앞의 잡을 이어서 맡아, 프로세스가
    끝나는 순간 running으로 남아 재시작 때 '서버 재시작으로 중단' 오류가 됐다."""
    import json
    import threading

    started = threading.Event()
    release = threading.Event()

    class _Gated(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            started.set()
            assert release.wait(10), "게이트가 열리지 않았다"
            return super().run_multi(image_paths, out_dir, sink, cancel)

    engine = _Gated(delay=0.0)
    store = JobStore(tmp_path / "jobs")
    settings = Settings(engine="fake", device="cpu", data_dir=tmp_path / "data")
    worker = Worker(store, EventBroker(), engine, settings, {})
    first, second = (store.create("doc.pdf", "multi", dpi=72) for _ in range(2))
    for job in (first, second):
        (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=1, with_image=False))
    engine.load()
    worker.start()
    try:
        worker.submit(first)
        worker.submit(second)
        assert started.wait(10)
        assert worker.current_job_id == first.id
        worker.stop()                    # 앞 잡이 도는 중에 종료 요청
    finally:
        release.set()
        worker.join(timeout=15)
    assert not worker.is_alive()
    assert first.status == "done"                       # 진행 중이던 잡은 끝까지
    assert worker.current_job_id is None
    assert second.status == "queued" and not second.claimed
    meta = json.loads((second.dir / "meta.json").read_text(encoding="utf-8"))
    assert (meta["status"], meta["submitted"]) == ("queued", True)
    restored = JobStore(tmp_path / "jobs").load_existing()
    assert [job.id for job in restored] == [second.id]   # 다음 기동이 다시 제출한다
