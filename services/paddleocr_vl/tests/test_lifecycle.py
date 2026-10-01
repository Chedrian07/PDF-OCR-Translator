"""로드 재시도·CUDA 고착 시 자가 재시작 테스트 — 표준 라이브러리만 (감사 sidecar-6·sidecar-7)."""

import logging

import pytest

from app import lifecycle
from app.lifecycle import (
    PermanentLoadError,
    is_engine_dead,
    is_transient_load_error,
    supervise_load,
)

log = logging.getLogger("test-lifecycle")


class _HTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


class _FakeModel:
    def __init__(self, failures: list[BaseException]):
        self._failures = list(failures)
        self.calls = 0
        self.load_error = None
        self.load_retry = None
        self.seen_retry: list[dict | None] = []

    def load(self):
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)


# ── 분류 ──

@pytest.mark.parametrize("exc", [
    ConnectionError("reset by peer"),
    TimeoutError("read timed out"),
    OSError("We couldn't connect to 'https://huggingface.co'"),
    _HTTPError(503),
    _HTTPError(429),
    RuntimeError("PaddleX 모델 다운로드 실패: 502 Bad Gateway"),
])
def test_network_and_capacity_failures_are_retried(exc):
    assert is_transient_load_error(exc)


@pytest.mark.parametrize("exc", [
    PermanentLoadError("CUDA 디바이스가 보이지 않습니다"),
    ModuleNotFoundError("No module named 'paddleocr'"),
    _HTTPError(401),
    _HTTPError(404),
])
def test_deterministic_failures_are_not_retried(exc):
    assert not is_transient_load_error(exc)


def test_classification_follows_the_exception_chain():
    """HF hub는 원인 예외를 감싸 다시 던진다 — 체인 안쪽의 404도 결정적이다."""
    try:
        try:
            raise _HTTPError(404)
        except _HTTPError as inner:
            raise OSError("revision not found") from inner
    except OSError as outer:
        assert not is_transient_load_error(outer)


def test_sticky_cuda_markers_match_wrapped_paddle_errors():
    """paddle은 CUDA 오류를 OSError 등으로 감싸 던진다 — 체인 안쪽까지 본다."""
    from app.model import _CUDA_FATAL_MARKERS

    try:
        try:
            raise OSError("(External) CUDA error(700), an illegal memory access was encountered.")
        except OSError as inner:
            raise RuntimeError("predict failed") from inner
    except RuntimeError as outer:
        assert is_engine_dead(outer, _CUDA_FATAL_MARKERS)
    oom = MemoryError("ResourceExhaustedError: Out of memory error on GPU 0")
    assert not is_engine_dead(oom, _CUDA_FATAL_MARKERS), "OOM은 강등 재시도로 풀린다"


# ── supervise_load ──

def test_transient_failure_then_success_keeps_health_loading_not_error():
    """재시도 대기 중 load_error가 서면 backend가 잡을 즉시 하드 실패시킨다."""
    model = _FakeModel([ConnectionError("blip"), TimeoutError("slow hub")])
    sleeps: list[float] = []

    def _sleep(s):
        sleeps.append(s)
        model.seen_retry.append(dict(model.load_retry))
        assert model.load_error is None

    assert supervise_load(model, log, backoff=(1.0, 2.0, 3.0), sleep=_sleep)
    assert model.calls == 3 and sleeps == [1.0, 2.0]
    assert model.load_error is None and model.load_retry is None
    assert model.seen_retry[0]["attempt"] == 1 and model.seen_retry[0]["max_attempts"] == 4
    assert "blip" in model.seen_retry[0]["last_error"]


def test_permanent_failure_is_fixed_immediately():
    model = _FakeModel([PermanentLoadError("CUDA 없음")])
    assert not supervise_load(model, log, backoff=(1.0, 2.0), sleep=pytest.fail)
    assert model.calls == 1
    assert model.load_error.startswith("PermanentLoadError: CUDA 없음")
    assert model.load_retry is None


def test_retries_are_bounded_and_the_last_error_is_reported():
    model = _FakeModel([ConnectionError(f"try {i}") for i in range(10)])
    assert not supervise_load(model, log, backoff=(0.0, 0.0), sleep=lambda s: None)
    assert model.calls == 3
    assert model.load_error == "ConnectionError: try 2"
    assert model.load_retry is None


def test_default_backoff_fits_inside_the_backend_model_wait():
    """backend는 OCR_SIDECAR_MODEL_WAIT_S(기본 900초)까지 기다린다 — 그 안에 재시도가 끝나야
    대기 중인 잡이 복구된 sidecar를 만난다."""
    assert sum(lifecycle.LOAD_BACKOFF_S) < 900
    assert len(lifecycle.LOAD_BACKOFF_S) >= 3


# ── schedule_restart ──

def test_restart_is_scheduled_once_with_a_nonzero_exit_code(monkeypatch):
    monkeypatch.setattr(lifecycle, "_restart_timer", None)
    codes: list[int] = []
    assert lifecycle.schedule_restart(log, "illegal memory access", delay_s=0.0,
                                      exit_fn=codes.append)
    assert not lifecycle.schedule_restart(log, "again", delay_s=0.0, exit_fn=codes.append)
    lifecycle._restart_timer.join(timeout=2)
    assert codes == [lifecycle.RESTART_EXIT_CODE] and lifecycle.RESTART_EXIT_CODE != 0
