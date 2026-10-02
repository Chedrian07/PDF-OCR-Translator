"""UnlimitedEngine 배선 계약 — 모델 없이 가짜 벤더 모델로 검증한다.

- MPS 작업 구간(로드·infer·infer_multi·empty_cache)은 ObjC 오토릴리스 풀 안에서
  돈다 — 잡 워커는 끝나지 않는 스레드라 풀이 없으면 객체가 영구히 쌓인다
  (audit gap1-metal-real-e2e-1). CPU/CUDA는 풀을 켜지 않는다.
- cuda/mps 로드는 P17 융합 MoE 스택을 디바이스 이동 **전에** 만든다 — expert별
  디바이스 버퍼·지연 재스택 단편화를 없앤다(audit MPS-2).
- 프로덕션 배선: 엔진 → generate_fn(벤더 P15) → fast_greedy_decode(OCR_FAST_DECODE=1)
  또는 HF generate(=0). 어느 쪽이든 EOS 없이 MAX_LENGTH에 닿으면 OutputLimitError로
  올려 runner의 페이지별 복구를 태운다(audit decode-correctness-1, tests-baseline-4).
"""

import contextlib
import json
import os
import subprocess
import sys
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


# ── 프로덕션 디코드 배선 + 길이 상한(OutputLimitError) ──

EOS_ID = 9


class _PieceTokenizer:
    """토큰 id → 고정 조각 텍스트 (EOS는 '<eos>')."""

    eos_token_id = EOS_ID

    def decode(self, token_ids, **kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join("<eos>" if int(t) == EOS_ID else f"t{int(t)} " for t in token_ids)


class _ChainDecoder:
    """(마지막 토큰+1) % 8을 내는 결정적 디코더. eos_after 스텝을 넘기면 EOS.

    fast_greedy_decode용 prepare_inputs/forward와, HF generate 폴백을 흉내 내는
    generate()(스트리머·중단 기준·max_length 계약 동일)를 함께 제공한다."""

    def __init__(self, eos_after=None, on_step=None):
        import torch

        self.torch = torch
        self.eos_after = eos_after
        self.on_step = on_step
        self.steps = 0

    def _next(self, last: int) -> int:
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self.steps)
        if self.eos_after is not None and self.steps > self.eos_after:
            return EOS_ID
        return (last + 1) % 8

    def prepare_inputs_for_generation(self, input_ids, **kw):
        return {"input_ids": input_ids[:, -1:]}

    def __call__(self, input_ids=None, return_dict=True, **kw):
        from types import SimpleNamespace

        logits = self.torch.full((1, input_ids.shape[1], 16), -10.0)
        logits[0, -1, self._next(int(input_ids[0, -1]))] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=None)

    def generate(self, **kw):
        ids = kw["input_ids"]
        streamer = kw.get("streamer")
        if streamer is not None:
            streamer.put(ids)
        while ids.shape[1] < kw["max_length"]:
            tok = self._next(int(ids[0, -1]))
            ids = self.torch.cat([ids, self.torch.tensor([[tok]])], dim=-1)
            if streamer is not None:
                streamer.put(self.torch.tensor([tok]))
            if tok == kw.get("eos_token_id"):
                break
            if any(bool(c(ids, None)) for c in kw.get("stopping_criteria") or []):
                break
        if streamer is not None:
            streamer.end()
        return ids


class _VendorLikeModel:
    """벤더 infer/infer_multi의 P15 계약만 흉내 낸다: gen_kwargs를 만들고
    generate_fn(self, gen_kwargs)을 부른다(generate_fn이 없으면 실패)."""

    def __init__(self, decoder: _ChainDecoder) -> None:
        self.decoder = decoder
        self.gen_kwargs: list[dict] = []

    def _run(self, kwargs) -> int:
        import torch

        gen_kwargs = {
            "input_ids": torch.tensor([[0, 1]]),
            "eos_token_id": EOS_ID,
            "max_length": kwargs["max_length"],
            "do_sample": False,
            "streamer": kwargs["streamer"],
            "stopping_criteria": kwargs["stopping_criteria"],
            "logits_processor": kwargs["logits_processor"],
        }
        self.gen_kwargs.append(gen_kwargs)
        out = kwargs["generate_fn"](self.decoder, gen_kwargs)
        return int(out.shape[1])

    def infer_multi(self, tokenizer, **kwargs):
        return "<PAGE>\nchunk", self._run(kwargs)

    def infer(self, tokenizer, **kwargs):
        self._run(kwargs)
        return "page"


def _decode_engine(monkeypatch, decoder, **overrides):
    overrides.setdefault("max_length", 12)
    engine = _engine(monkeypatch, "cpu", **overrides)
    engine._tokenizer = _PieceTokenizer()
    model = _VendorLikeModel(decoder)
    engine._model = model
    return engine, model


def test_fast_decode_wiring_passes_decode_block_and_streams(tmp_path, monkeypatch):
    """OCR_FAST_DECODE=1: 엔진이 generate_fn으로 fast_greedy_decode를 주입하고
    OCR_DECODE_BLOCK을 그대로 넘기며, 생성 텍스트가 sink로 흐른다."""
    import app.engine.fast_decode as fd

    blocks: list[int] = []
    real = fd.fast_greedy_decode

    def spy(model, gen_kwargs, block=8):
        blocks.append(block)
        return real(model, gen_kwargs, block=block)

    monkeypatch.setattr(fd, "fast_greedy_decode", spy)
    engine, _ = _decode_engine(monkeypatch, _ChainDecoder(eos_after=4), decode_block=3)
    sink = _Sink()
    out = engine.run_multi([Path("p.png")], tmp_path / "m", sink, threading.Event())

    assert out == "<PAGE>\nchunk"
    assert blocks == [3]
    assert sink.text.startswith("t2 t3 t4 t5 ")  # 프롬프트(0,1) 뒤 체인, EOS 포함 flush


@pytest.mark.parametrize("fast_decode", [True, False])
def test_length_cap_raises_output_limit_error_after_flush(tmp_path, monkeypatch, fast_decode):
    """EOS 없이 MAX_LENGTH에 닿으면 잘린 출력 — OutputLimitError(RepetitiveOutputError
    하위)로 runner의 페이지별 복구 경로를 태운다. 스트림은 이미 flush돼 있다."""
    from app.engine.base import OutputLimitError, RepetitiveOutputError

    engine, model = _decode_engine(monkeypatch, _ChainDecoder(eos_after=None), fast_decode=fast_decode)
    sink = _Sink()
    with pytest.raises(OutputLimitError, match="MAX_LENGTH=12") as info:
        engine.run_multi([Path("p.png")], tmp_path / "m", sink, threading.Event())
    assert isinstance(info.value, RepetitiveOutputError)
    assert sink.text.count("t") == 12 - 2  # 상한까지 생성된 10토큰이 전부 스트림됨
    # 잘린 multi 출력(run_multi 형식)을 실어 runner가 끝까지 생성된 앞 페이지를 살린다
    assert info.value.partial_output == "<PAGE>\nchunk"
    assert "chunk" not in str(info.value)  # 문서 내용은 메시지(로그·잡 경고)에 넣지 않는다

    engine2, _ = _decode_engine(monkeypatch, _ChainDecoder(eos_after=None), fast_decode=fast_decode)
    with pytest.raises(OutputLimitError) as single_info:
        engine2.run_single(Path("p.png"), tmp_path / "s", _Sink(), threading.Event())
    assert single_info.value.partial_output is None  # 한 쪽짜리라 살릴 앞 페이지가 없다


@pytest.mark.parametrize("fast_decode", [True, False])
def test_eos_before_cap_is_a_normal_result(tmp_path, monkeypatch, fast_decode):
    engine, _ = _decode_engine(monkeypatch, _ChainDecoder(eos_after=3), fast_decode=fast_decode)
    assert engine.run_single(Path("p.png"), tmp_path / "s", _Sink(), threading.Event()) == "page"


def test_cancel_takes_precedence_over_length_cap(tmp_path, monkeypatch):
    """상한에 닿은 같은 블록에서 사용자가 취소했다면 취소가 우선 — 부분 출력 반환."""
    cancel = threading.Event()

    def on_step(step: int) -> None:
        if step >= 10:
            cancel.set()

    engine, _ = _decode_engine(monkeypatch, _ChainDecoder(eos_after=None, on_step=on_step))
    out = engine.run_multi([Path("p.png")], tmp_path / "m", _Sink(), cancel)
    assert out == "<PAGE>\nchunk"
    assert cancel.is_set()


def test_page_token_budget_still_wins_over_length_cap(tmp_path, monkeypatch):
    """페이지 토큰 예산이 먼저 차면 반복/상한 감지(RepetitiveOutputError)로 멈춘다 —
    실제 디코드 경로(스트리머 feed_tokens)로 예산 연동을 검증한다."""
    from app.engine.base import OutputLimitError, RepetitiveOutputError

    engine, _ = _decode_engine(
        monkeypatch, _ChainDecoder(eos_after=None), max_length=40, max_page_output_tokens=5
    )
    with pytest.raises(RepetitiveOutputError, match="토큰 상한") as info:
        engine.run_multi([Path("p.png")], tmp_path / "m", _Sink(), threading.Event())
    assert not isinstance(info.value, OutputLimitError)


def test_vendor_entry_points_keep_generate_fn_hook():
    """벤더 P15 계약: infer/infer_multi가 generate_fn(기본 None)을 받는다 — 재동기화로
    훅이 사라지면 엔진 주입이 조용히 무시되고 HF generate로 강등된다."""
    import inspect

    from app.vendor.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM

    for name in ("infer", "infer_multi"):
        param = inspect.signature(getattr(UnlimitedOCRForCausalLM, name)).parameters.get("generate_fn")
        assert param is not None and param.default is None, name


# ── Metal 안내 문구 (audit apple-mps-8) ──


def test_metal_unavailable_hint_names_macos14_and_current_version(monkeypatch):
    """torch 2.10 MPS는 macOS 14+ 필요 — 예전 '12.3 이상' 안내는 13.x 사용자를 오도했다."""
    import platform

    import torch

    from app.engine.base import EngineError

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("13.6.1", ("", "", ""), "arm64"))
    engine = _engine(monkeypatch, "metal")
    with pytest.raises(EngineError) as info:
        engine._load_locked()
    message = str(info.value)
    assert "macOS 14.0 이상" in message and "현재 macOS 13.6.1" in message
    assert "12.3" not in message


def test_metal_bf16_probe_failure_warns_with_macos_version(monkeypatch, caplog):
    import platform

    import torch

    monkeypatch.setattr(unlimited_mod, "_mps_bf16_supported", lambda: False)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.6.2", ("", "", ""), "arm64"))
    with caplog.at_level("WARNING", logger="app.engine.unlimited"):
        assert unlimited_mod._resolve_dtype("metal", "auto") is torch.float32
    assert "macOS 26.6.2" in caplog.text and "프로브가 실패" in caplog.text


# ── MPS CPU 폴백 env는 torch 첫 임포트 전에 (audit torch-3) ──

# 하위 프로세스 공통 준비 — darwin으로 간주해 Linux CI에서도 같은 경로를 탄다. mlx는 미설치처럼
# 막고, torch는 가짜 모듈(CUDA 없음·MPS 있음)로 바꿔 **첫 임포트 시점**의 env를 기록한다.
# torch는 PYTORCH_ENABLE_MPS_FALLBACK을 라이브러리 로드 때 한 번만 읽으므로 그 시점 값이
# 실제 폴백 여부다(실측: 임포트 뒤에 설정하면 linalg.eig(mps)가 NotImplementedError).
# urllib.request는 플랫폼을 바꾸기 **전에** 올린다 — 임포트 시점에 sys.platform이 darwin이면
# macOS 전용 _scproxy를 가져오는데(CPython urllib/request.py), app.config → httpx 경로가 그
# 모듈을 처음 올리므로 Linux CI에서는 하위 프로세스가 ModuleNotFoundError로 죽었다.
_ISOLATED_PRELUDE = '''
import importlib.abc, importlib.machinery, json, logging, os, sys, types
import urllib.request  # noqa: F401 — 실제 플랫폼으로 먼저 임포트(_scproxy는 macOS 전용)
sys.platform = "darwin"
seen, warned = [], []


class _Capture(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING and "PYTORCH_ENABLE_MPS_FALLBACK" in record.getMessage():
            warned.append(record.getMessage())


logging.getLogger("app.engine.unlimited").addHandler(_Capture())


class _StubTorch(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        seen.append(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        module.cuda = types.SimpleNamespace(is_available=lambda: False)
        module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("test: mlx 미설치")
        if name == "torch":
            return importlib.machinery.ModuleSpec("torch", _StubTorch())
        return None


sys.meta_path.insert(0, _Finder())
'''


def _run_isolated(code: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    env["DISABLE_DOTENV"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATED_PRELUDE + code],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_auto_metal_path_sets_mps_fallback_before_torch_is_first_imported():
    """mlx가 없는 Mac의 OCR_DEVICE=auto: registry가 엔진을 만들기 **전에** torch로 CUDA·MPS를
    조회한다 — 엔진 생성 시 setdefault는 늦어 문서화된 안전망이 조용히 꺼졌다."""
    out = _run_isolated(
        "from app.config import Settings\n"
        "from app.engine.registry import build_engine\n"
        "engine = build_engine(Settings(engine='unlimited', device='auto', preload_model=False))\n"
        "print(json.dumps({'engine': type(engine).__name__, 'device': engine.device,"
        " 'seen': seen, 'warned': warned}))\n"
    )
    assert out == {"engine": "UnlimitedEngine", "device": "metal", "seen": ["1"], "warned": []}


@pytest.mark.parametrize(
    "preset,device,warns,final",
    [
        (None, "metal", True, "1"),  # torch가 env 없이 먼저 로드됨 — 폴백 꺼짐을 알린다
        (None, "cpu", False, "1"),  # MPS를 쓰지 않는 엔진은 조용히
        ("0", "metal", False, "0"),  # 운영자가 끈 값은 존중(덮어쓰지 않고 경고도 없음)
        ("1", "metal", False, "1"),  # 프로세스 env로 켠 경우는 임포트 순서와 무관하게 적용됨
    ],
)
def test_metal_engine_warns_when_torch_was_loaded_without_the_fallback(preset, device, warns, final):
    out = _run_isolated(
        f"preset = {preset!r}\n"
        "if preset is not None:\n"
        "    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = preset\n"
        "import torch  # 엔진 모듈보다 먼저 — 스크립트·도구가 torch를 먼저 올린 경우\n"
        "from app.config import Settings\n"
        "from app.engine.unlimited import UnlimitedEngine\n"
        f"UnlimitedEngine(Settings(engine='unlimited', device={device!r}, preload_model=False))\n"
        "print(json.dumps({'seen': seen, 'warned': warned,"
        " 'env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}))\n"
    )
    assert out["seen"] == [preset]
    assert bool(out["warned"]) is warns
    assert out["env"] == final


# ── 고정 스냅샷은 캐시에서 먼저 (P4 Docker: 로드마다 huggingface.co 조회) ──


def test_load_reads_the_pinned_snapshot_from_the_cache_without_the_hub(monkeypatch):
    """캐시가 완전해도 로드마다 Hub에 묻던(선택 파일 HEAD 6회·API 1회) 경로를 막는다 — 고정 커밋
    리비전이면 토크나이저·모델 모두 local_files_only로 먼저 읽는다(app.engine.hf_snapshot)."""
    import transformers

    from app.vendor.unlimited_ocr import UnlimitedOCRForCausalLM

    calls: list[tuple[str, tuple, dict]] = []

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    def _model(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return _FakeModel()

    def _tokenizer(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return _Tokenizer()

    monkeypatch.setattr(UnlimitedOCRForCausalLM, "from_pretrained", _model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _tokenizer)
    settings = Settings(engine="unlimited", device="cpu", preload_model=False)
    engine = UnlimitedEngine(settings)
    engine._load_locked()

    assert [name for name, _args, _kwargs in calls] == ["tokenizer", "model"]
    for _name, args, kwargs in calls:
        assert args == (settings.model_id,)
        assert kwargs["revision"] == settings.model_revision
        assert kwargs["local_files_only"] is True
    assert calls[1][2]["attn_implementation"] == "eager"
    assert engine.loaded
