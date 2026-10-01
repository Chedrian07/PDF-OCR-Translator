"""OvisModel 상태 전이 테스트 — vLLM·GPU 없이 가짜 엔진으로 (감사 sidecar-6·7·11)."""

import sys
import types

import pytest

from app.config import OvisConfig
from app.lifecycle import PermanentLoadError, is_transient_load_error
from app.model import InferOutput, OvisModel


class _Completion:
    def __init__(self, text, finish_reason):
        self.text = text
        self.finish_reason = finish_reason


class _FakeLLM:
    def __init__(self, results):
        self._results = list(results)

    def generate(self, requests, params, use_tqdm=False):
        r = self._results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return [type("Out", (), {"outputs": [_Completion(*r)]})()]


def _model(results) -> OvisModel:
    m = OvisModel(OvisConfig.from_env())
    m._llm = _FakeLLM(results)
    m._prompt = "<prompt>"
    m._sampling_cls = lambda **kw: kw
    return m


class EngineDeadError(RuntimeError):
    pass


def test_infer_reports_finish_reason_for_truncation():
    m = _model([("# 제목\n본문", "stop"), ("<table><tr><td>…", "length")])
    assert m.infer(object()) == InferOutput("# 제목\n본문", "stop")
    out = m.infer(object())
    assert out.finish_reason == "length" and out.text.startswith("<table>")


def test_engine_death_drops_the_engine_and_requests_restart():
    """같은 프로세스에서는 회복 불가 — 미로드+status ok(기다릴 수 있는 상태)로 바꾼다."""
    m = _model([EngineDeadError("EngineCore encountered an issue")])
    with pytest.raises(EngineDeadError):
        m.infer(object())
    assert m.restart_required and not m.loaded
    # status=error+미로드는 backend가 '기다려도 안 풀리는 로드 실패'로 보고 잡을 실패시킨다
    assert m.load_error is None


def test_engine_death_clears_an_earlier_wedge_report():
    m = _model([RuntimeError("x"), RuntimeError("y"), RuntimeError("z"),
                EngineDeadError("engine core died")])
    for _ in range(3):
        with pytest.raises(RuntimeError):
            m.infer(object())
    assert m.load_error and m.loaded  # 웨지 신고 — 모델은 유지
    with pytest.raises(EngineDeadError):
        m.infer(object())
    assert m.restart_required and m.load_error is None


def test_wedge_report_still_recovers_on_success():
    m = _model([RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), ("ok", "stop")])
    for _ in range(3):
        with pytest.raises(RuntimeError):
            m.infer(object())
    assert m.load_error.startswith("엔진 비정상: ") and not m.restart_required
    assert m.infer(object()).text == "ok"
    assert m.load_error is None


def test_missing_cuda_is_a_permanent_load_failure(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(PermanentLoadError) as e:
        OvisModel._require_cuda()
    assert not is_transient_load_error(e.value)


def test_load_failure_does_not_set_load_error_itself(monkeypatch):
    """load_error 확정은 supervise_load 몫 — 재시도 대기 중 health가 error가 되면 안 된다."""
    def _boom(**kwargs):
        raise ConnectionError("hub unreachable")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_boom, SamplingParams=dict))
    m = OvisModel(OvisConfig.from_env())
    with pytest.raises(ConnectionError):
        m.load()
    assert m.load_error is None and not m.loaded
