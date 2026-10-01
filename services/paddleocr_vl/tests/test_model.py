"""PaddleModel 상태 전이 테스트 — paddle·GPU 없이 가짜 파이프라인으로 (감사 sidecar-6·7).

예전에는 predict_page에 실패 집계가 전혀 없어, CUDA 오류가 고착돼 모든 추론이 502가
되어도 health는 계속 status=ok였다.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from app.config import PaddleConfig
from app.lifecycle import PermanentLoadError, is_transient_load_error
from app.model import PaddleModel

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "official_page.json")
                     .read_text(encoding="utf-8"))


class _FakePipeline:
    def __init__(self, results):
        self._results = list(results)

    def predict(self, image_path, **kwargs):
        r = self._results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return [types.SimpleNamespace(json=r)]


def _model(results) -> PaddleModel:
    m = PaddleModel(PaddleConfig.from_env())
    m._pipeline = _FakePipeline(results)
    return m


_STICKY = OSError("(External) CUDA error(700), an illegal memory access was encountered.")


def test_successful_prediction_returns_the_official_json():
    m = _model([FIXTURE])
    assert m.predict_page("/tmp/page.png") == FIXTURE
    assert m.load_error is None and not m.restart_required


def test_sticky_cuda_error_drops_the_pipeline_and_requests_restart():
    m = _model([_STICKY])
    with pytest.raises(OSError):
        m.predict_page("/tmp/page.png")
    assert m.restart_required and not m.loaded
    # status=error+미로드는 backend가 '기다려도 안 풀리는 로드 실패'로 보고 잡을 실패시킨다
    assert m.load_error is None


def test_consecutive_failures_raise_a_self_healing_wedge_report():
    m = _model([RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), FIXTURE])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            m.predict_page("/tmp/page.png")
    assert m.load_error is None, "임계 전에는 신고하지 않는다"
    with pytest.raises(RuntimeError):
        m.predict_page("/tmp/page.png")
    assert m.load_error.startswith("엔진 비정상: ") and m.loaded
    assert m.predict_page("/tmp/page.png") == FIXTURE
    assert m.load_error is None, "성공 1회로 웨지 신고가 풀린다"


def test_oom_alone_does_not_request_a_restart():
    m = _model([MemoryError("ResourceExhaustedError: Out of memory error on GPU 0")])
    with pytest.raises(MemoryError):
        m.predict_page("/tmp/page.png")
    assert not m.restart_required and m.loaded


def test_missing_cuda_device_is_a_permanent_load_failure(monkeypatch):
    fake_paddle = types.SimpleNamespace(device=types.SimpleNamespace(
        is_compiled_with_cuda=lambda: True,
        cuda=types.SimpleNamespace(device_count=lambda: 0),
    ))
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    m = PaddleModel(PaddleConfig.from_env())
    with pytest.raises(PermanentLoadError) as e:
        m._require_cuda()
    assert not is_transient_load_error(e.value)


def test_load_failure_does_not_set_load_error_itself(monkeypatch):
    """load_error 확정은 supervise_load 몫 — 재시도 대기 중 health가 error가 되면 안 된다."""
    m = PaddleModel(PaddleConfig.from_env())

    def _boom():
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(m, "_pin_revision", _boom)
    with pytest.raises(ConnectionError):
        m.load()
    assert m.load_error is None and not m.loaded


def test_failures_after_a_sticky_cuda_error_do_not_raise_a_wedge_report():
    """재시작 대기 중 뒤따른 실패가 웨지 신고(status=error)를 세우면 backend는 재시작을
    기다리지 않고 잡을 실패시킨다."""
    m = _model([_STICKY])
    with pytest.raises(OSError):
        m.predict_page("/tmp/page.png")
    for _ in range(5):
        m._note_infer_failure(RuntimeError("late failure"))
    assert m.restart_required and m.load_error is None
    with pytest.raises(RuntimeError, match="로드되지 않았습니다"):
        m.predict_page("/tmp/page.png")
