"""잡 워커(jobs.Worker)와 잡 메타 계약.

- Metal(torch MPS) 잡은 잡 단위 ObjC 오토릴리스 풀 안에서 돈다 — 워커는 끝나지 않는
  스레드라 엔진·디코드 루프 바깥에서 autorelease된 객체의 마지막 회수 지점이다.
  MLX는 실측(8쪽 실가중치 잡 5회 연속: RSS 2627→2629→2645→2645→2647MB, MLX 활성 메모리
  6363MB 고정)에서 잡별 증가가 없어 풀을 켜지 않는다. CPU·CUDA에는 비울 객체가 없다.
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
