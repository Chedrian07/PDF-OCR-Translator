"""UnlimitedEngine 배선 계약 — 모델 없이 가짜 벤더 모델로 검증한다.

- MPS 작업 구간(로드·infer·infer_multi·empty_cache)은 ObjC 오토릴리스 풀 안에서
  돈다 — 잡 워커는 끝나지 않는 스레드라 풀이 없으면 객체가 영구히 쌓인다
  (audit gap1-metal-real-e2e-1). CPU/CUDA는 풀을 켜지 않는다.
- cuda/mps 로드는 P17 융합 MoE 스택을 디바이스 이동 **전에** 만든다 — expert별
  디바이스 버퍼·지연 재스택 단편화를 없앤다(audit MPS-2).
"""

import contextlib
import threading
from pathlib import Path

import pytest

import app.engine.unlimited as unlimited_mod
from app.config import Settings
from app.engine.unlimited import UnlimitedEngine


class _Sink:
    def __init__(self) -> None:
        self.text = ""

    def on_text(self, text: str) -> None:
        self.text += text


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids, **kwargs):
        return "<eos>"


class _PoolSpy:
    """autorelease_pool 대체 — 활성(enabled=True) 풀 깊이를 기록한다."""

    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0
        self.entries: list[bool] = []

    def __call__(self, enabled: bool = True):
        spy = self

        @contextlib.contextmanager
        def _pool():
            spy.entries.append(enabled)
            if enabled:
                spy.depth += 1
                spy.max_depth = max(spy.max_depth, spy.depth)
            try:
                yield
            finally:
                if enabled:
                    spy.depth -= 1

        return _pool()


class _RecordingModel:
    """infer/infer_multi 호출 시점의 풀 깊이를 기록하는 가짜 벤더 모델."""

    def __init__(self, spy: _PoolSpy) -> None:
        self.spy = spy
        self.depth_seen: list[int] = []

    def infer_multi(self, tokenizer, **kwargs):
        self.depth_seen.append(self.spy.depth)
        return "<PAGE>\nok", 1

    def infer(self, tokenizer, **kwargs):
        self.depth_seen.append(self.spy.depth)
        return "ok"


def _engine(monkeypatch, device: str, **overrides) -> UnlimitedEngine:
    # metal 생성자는 PYTORCH_ENABLE_MPS_FALLBACK을 setdefault한다 — 테스트 밖으로 새지 않게
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    settings = Settings(engine="unlimited", device=device, preload_model=False, **overrides)
    engine = UnlimitedEngine(settings)
    engine._tokenizer = _Tokenizer()
    monkeypatch.setattr(unlimited_mod, "make_ngram_logits_processor", lambda *args: [])
    return engine


@pytest.mark.parametrize("fast_decode", [True, False])
def test_metal_runs_inside_autorelease_pool(tmp_path, monkeypatch, fast_decode):
    spy = _PoolSpy()
    monkeypatch.setattr(unlimited_mod, "autorelease_pool", spy)
    engine = _engine(monkeypatch, "metal", fast_decode=fast_decode)
    model = _RecordingModel(spy)
    engine._model = model
    monkeypatch.setattr(engine, "_release_device_cache", lambda: None)

    engine.run_multi([Path("p.png")], tmp_path / "m", _Sink(), threading.Event())
    engine.run_single(Path("p.png"), tmp_path / "s", _Sink(), threading.Event())

    # HF generate 폴백(fast_decode=False)도 이 바깥 풀이 유일한 회수 지점이다
    assert model.depth_seen == [1, 1]
    assert spy.depth == 0  # 모두 pop됨


def test_cpu_never_enables_autorelease_pool(tmp_path, monkeypatch):
    spy = _PoolSpy()
    monkeypatch.setattr(unlimited_mod, "autorelease_pool", spy)
    engine = _engine(monkeypatch, "cpu")
    model = _RecordingModel(spy)
    engine._model = model

    engine.run_multi([Path("p.png")], tmp_path / "m", _Sink(), threading.Event())
    engine.run_single(Path("p.png"), tmp_path / "s", _Sink(), threading.Event())

    assert model.depth_seen == [0, 0]
    assert spy.entries and not any(spy.entries)


def test_metal_model_load_runs_inside_autorelease_pool(monkeypatch):
    """프리로드가 없으면 워커 스레드가 로드한다 — .to('mps')의 임시 객체도 회수."""
    spy = _PoolSpy()
    monkeypatch.setattr(unlimited_mod, "autorelease_pool", spy)
    engine = _engine(monkeypatch, "metal")
    engine._model = None
    seen: list[int] = []

    def fake_load_locked():
        seen.append(spy.depth)
        engine._model = object()

    monkeypatch.setattr(engine, "_load_locked", fake_load_locked)
    engine.load()
    engine.load()  # 이미 로드됨 — 재진입 없음
    assert seen == [1]
    assert spy.depth == 0


class _PlacementModel:
    def __init__(self, log: list) -> None:
        self.log = log

    def eval(self):
        self.log.append("eval")
        return self

    def to(self, device):
        self.log.append(("to", device))
        return self


@pytest.mark.parametrize("device,torch_device", [("metal", "mps"), ("cuda", "cuda")])
def test_place_model_prebuilds_fused_stacks_before_device_move(monkeypatch, device, torch_device):
    import app.vendor.unlimited_ocr.modeling_deepseekv2 as md

    log: list = []
    monkeypatch.setattr(md, "prebuild_fused_moe", lambda model, dev: log.append(("prebuild", dev)) or 11)
    engine = _engine(monkeypatch, device)
    monkeypatch.setattr(engine, "_release_device_cache", lambda: log.append("release"))
    model = _PlacementModel(log)

    assert engine._place_model(model) is model
    assert log == ["eval", ("prebuild", torch_device), ("to", torch_device), "release"]


def test_place_model_on_cpu_keeps_legacy_expert_weights(monkeypatch):
    import app.vendor.unlimited_ocr.modeling_deepseekv2 as md

    log: list = []

    def boom(model, dev):
        raise AssertionError("CPU는 융합 경로를 쓰지 않으므로 프리빌드 금지")

    monkeypatch.setattr(md, "prebuild_fused_moe", boom)
    engine = _engine(monkeypatch, "cpu")
    monkeypatch.setattr(engine, "_release_device_cache", lambda: log.append("release"))
    engine._place_model(_PlacementModel(log))
    assert log == ["eval", ("to", "cpu"), "release"]
