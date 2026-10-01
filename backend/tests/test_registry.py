"""디바이스/엔진 registry 테스트 — Metal(MPS) 포함. 계약: docs/ARCHITECTURE.md §6"""

import pytest

from app.config import Settings
from app.engine.fake import FakeEngine
from app.engine.registry import build_engine


def _fake_settings(**kw) -> Settings:
    return Settings(engine="fake", preload_model=False, fake_delay=0.0, **kw)


def test_metal_device_builds_fake_engine():
    eng = build_engine(_fake_settings(device="metal"))
    assert isinstance(eng, FakeEngine)
    assert eng.device == "metal"


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="OCR_DEVICE"):
        build_engine(_fake_settings(device="tpu"))


def test_metal_unlimited_engine_constructs_without_torch():
    # 생성 시점엔 torch가 필요 없다 — MPS 가용성 검증은 load()에서 수행
    eng = build_engine(Settings(engine="unlimited", device="metal"))
    assert eng.device == "metal"
    assert eng.torch_device == "mps"
    assert not eng.loaded


def test_torch_device_mapping():
    from app.engine.unlimited import torch_device_name

    assert torch_device_name("cpu") == "cpu"
    assert torch_device_name("cuda") == "cuda"
    assert torch_device_name("metal") == "mps"


def test_env_mps_alias(monkeypatch):
    monkeypatch.setenv("OCR_DEVICE", "mps")
    assert Settings.from_env().device == "metal"


def test_mlx_quant_bits_from_env(monkeypatch):
    monkeypatch.delenv("OCR_MLX_QUANT_BITS", raising=False)
    assert Settings.from_env().mlx_quant_bits == 0
    # 빈 값은 미설정과 같다 (compose·.env의 빈 선택 키 관례)
    for raw, want in (("8", 8), ("0", 0), (" 8 ", 8), ("", 0)):
        monkeypatch.setenv("OCR_MLX_QUANT_BITS", raw)
        assert Settings.from_env().mlx_quant_bits == want


@pytest.mark.parametrize("raw", ["4", "16", "-8", "8bit", "1.5"])
def test_mlx_quant_bits_rejects_other_values_naming_the_variable(monkeypatch, raw):
    """4비트(숫자 오인식 실측)·오타는 기동 시 변수명과 함께 실패 — 조용히 bf16으로 돌면
    운영자는 양자화가 켜졌다고 믿는다."""
    monkeypatch.setenv("OCR_MLX_QUANT_BITS", raw)
    with pytest.raises(ValueError, match="OCR_MLX_QUANT_BITS"):
        Settings.from_env()


def test_page_output_limits_from_env(monkeypatch):
    monkeypatch.setenv("MAX_PAGE_OUTPUT_CHARS", "12345")
    monkeypatch.setenv("MAX_PAGE_OUTPUT_TOKENS", "4321")

    settings = Settings.from_env()

    assert settings.max_page_output_chars == 12_345
    assert settings.max_page_output_tokens == 4_321


def test_page_output_limits_zero_disables(monkeypatch):
    """0 이하 상한은 비활성(None) — 감지기 ValueError로 잡이 죽는 설정 실수 방지."""
    monkeypatch.setenv("MAX_PAGE_OUTPUT_CHARS", "0")
    monkeypatch.setenv("MAX_PAGE_OUTPUT_TOKENS", "-1")

    settings = Settings.from_env()

    assert settings.max_page_output_chars is None
    assert settings.max_page_output_tokens is None


def test_resolve_dtype():
    torch = pytest.importorskip("torch")
    from app.engine.unlimited import _resolve_dtype

    assert _resolve_dtype("metal", "auto") in (torch.bfloat16, torch.float32)
    assert _resolve_dtype("metal", "float16") is torch.float16
    assert _resolve_dtype("cpu", "auto") is torch.float32
    # cuda/auto도 MPS와 동일하게 bf16 지원을 프로브한다 — 실행 환경(GPU 유무·세대)에
    # 따라 bf16 또는 float16이며, 무조건 bf16이던 과거 동작은 pre-Ampere에서 실패했다
    assert _resolve_dtype("cuda", "auto") in (torch.bfloat16, torch.float16)
    with pytest.raises(ValueError, match="OCR_DTYPE"):
        _resolve_dtype("metal", "int8")


def test_cuda_auto_dtype_falls_back_when_bf16_unsupported(monkeypatch):
    """pre-Ampere(sm<80) GPU에서 bf16을 강행하면 커널 부재로 로드가 실패한다 —
    auto는 float16으로 강등돼야 한다 (MPS 경로와 동일한 방어)."""
    torch = pytest.importorskip("torch")
    from app.engine import unlimited as unlimited_mod

    monkeypatch.setattr(unlimited_mod, "_cuda_bf16_supported", lambda: False)
    assert unlimited_mod._resolve_dtype("cuda", "auto") is torch.float16

    monkeypatch.setattr(unlimited_mod, "_cuda_bf16_supported", lambda: True)
    assert unlimited_mod._resolve_dtype("cuda", "auto") is torch.bfloat16
    # 명시 지정은 프로브와 무관하게 그대로 존중한다
    monkeypatch.setattr(unlimited_mod, "_cuda_bf16_supported", lambda: False)
    assert unlimited_mod._resolve_dtype("cuda", "bfloat16") is torch.bfloat16


def test_fake_engine_capabilities_do_not_claim_real_model():
    """FakeEngine이 model_id를 비워 두면 health/잡 메타가 settings.model_id
    (baidu/Unlimited-OCR)로 폴백해 데모 출력이 실모델 결과처럼 기록된다."""
    eng = build_engine(_fake_settings(device="cpu"))
    caps = eng.capabilities()
    assert caps.model_id == "fake-engine"
    assert caps.provider == "in-process"
    # 청크·스트리밍 계약은 기존과 동일해야 한다 (하위 호환)
    assert caps.supports_multi_page is True
    assert caps.stream_granularity == "token"
    assert caps.preferred_chunk_size is None


def test_load_concurrent_calls_load_once(monkeypatch):
    # 프리로드 스레드와 워커 스레드가 load()에 동시 진입해도 from_pretrained는
    # 정확히 1회만 실행돼야 한다 (동시 로드 → meta 텐서 잔류 → .to() 실패 회귀 방지)
    pytest.importorskip("torch")
    import threading
    import time

    import transformers

    from app.engine.unlimited import UnlimitedEngine
    from app.vendor.unlimited_ocr import UnlimitedOCRForCausalLM

    calls: list[int] = []

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    def _fake_model_fp(*args, **kwargs):
        calls.append(1)
        time.sleep(0.2)  # 두 번째 스레드가 로딩 도중에 진입하도록 지연
        return _FakeModel()

    monkeypatch.setattr(UnlimitedOCRForCausalLM, "from_pretrained", _fake_model_fp)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: object())

    eng = UnlimitedEngine(Settings(engine="unlimited", device="cpu", preload_model=False))
    errors: list[Exception] = []

    def _load():
        try:
            eng.load()
        except Exception as e:  # noqa: BLE001 — 스레드 안 예외를 본문으로 전달
            errors.append(e)

    threads = [threading.Thread(target=_load) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert len(calls) == 1
    assert eng.loaded


def test_health_reports_metal(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = _fake_settings(
        device="metal",
        data_dir=tmp_path / "data",
        frontend_dir=tmp_path / "no-frontend",
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/api/health").json()
    assert body["device"] == "metal"


# ── OCR_DEVICE=mlx · auto (Apple Silicon 기본 = MLX, 감사 api-jobs-5·mlx-integration-12) ──


def test_valid_devices_include_mlx_and_auto():
    from app.engine.registry import VALID_DEVICES

    assert {"auto", "cpu", "cuda", "metal", "mlx"} == set(VALID_DEVICES)


def test_mlx_device_builds_the_mlx_engine_without_probing(monkeypatch):
    from app.engine import unlimited_mlx
    from app.engine.unlimited_mlx import UnlimitedMLXEngine

    def boom():
        raise AssertionError("명시 mlx는 생성 시점에 가용성을 탐지하지 않는다 (load()에서 검증)")

    monkeypatch.setattr(unlimited_mlx, "mlx_unavailable_reason", boom)
    eng = build_engine(Settings(engine="unlimited", device="mlx", mlx_quant_bits=8))
    assert isinstance(eng, UnlimitedMLXEngine)
    assert (eng.name, eng.device, eng.dtype_name, eng.loaded) == ("unlimited", "mlx", "bfloat16+q8", False)


def _stub_probes(monkeypatch, *, mlx: bool, cuda: bool = False, mps: bool = False) -> list[str]:
    """auto 판정의 하드웨어 탐지를 고정한다 — 호출 기록을 돌려준다."""
    from app.engine import registry, unlimited_mlx

    calls: list[str] = []

    def mlx_reason():
        calls.append("mlx")
        return None if mlx else "mlx를 임포트할 수 없습니다 (테스트)"

    def torch_probe():
        calls.append("torch")
        return cuda, mps, ""

    monkeypatch.setattr(unlimited_mlx, "mlx_unavailable_reason", mlx_reason)
    monkeypatch.setattr(registry, "_torch_accelerators", torch_probe)
    return calls


@pytest.mark.parametrize(
    "mlx,cuda,mps,expected,torch_device",
    [
        (True, True, True, "mlx", None),       # Apple Silicon + mlx: MLX가 1순위 (torch 탐지 생략)
        (False, True, True, "cuda", "cuda"),
        (False, False, True, "metal", "mps"),  # mlx 미설치 Mac → torch MPS 폴백
        (False, False, False, "cpu", "cpu"),   # Linux CPU 이미지(torch CPU 휠)
    ],
)
def test_auto_resolves_mlx_then_cuda_then_metal_then_cpu(
    monkeypatch, caplog, mlx, cuda, mps, expected, torch_device
):
    import logging

    from app.engine.unlimited import UnlimitedEngine
    from app.engine.unlimited_mlx import UnlimitedMLXEngine

    calls = _stub_probes(monkeypatch, mlx=mlx, cuda=cuda, mps=mps)
    settings = Settings(engine="unlimited", device="auto", preload_model=False)
    with caplog.at_level(logging.INFO, logger="app.engine.registry"):
        eng = build_engine(settings)
    assert eng.device == expected
    if expected == "mlx":
        assert isinstance(eng, UnlimitedMLXEngine) and calls == ["mlx"]
    else:
        assert isinstance(eng, UnlimitedEngine) and eng.torch_device == torch_device
        assert calls == ["mlx", "torch"]
    assert f"OCR_DEVICE=auto → {expected}" in caplog.text  # 결정을 남긴다
    assert settings.device == "auto"  # 호출자의 설정은 바꾸지 않는다


@pytest.mark.parametrize("engine", ["fake", "textlayer", "ovisocr2"])
def test_auto_does_not_probe_hardware_for_engines_without_a_local_model(monkeypatch, engine):
    calls = _stub_probes(monkeypatch, mlx=True, cuda=True, mps=True)
    eng = build_engine(
        Settings(engine=engine, device="auto", preload_model=False, sidecar_url="http://ovisocr2:8080")
    )
    assert calls == []
    assert eng.device == {"fake": "cpu", "textlayer": "cpu", "ovisocr2": "cuda"}[engine]


def test_auto_on_linux_never_imports_mlx():
    """Linux(CPU/CUDA 이미지)에서는 플랫폼만 보고 mlx 임포트를 시도조차 하지 않는다."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, platform\n"
        "sys.platform = 'linux'\n"
        "platform.machine = lambda: 'x86_64'\n"
        "from app.config import Settings\n"
        "from app.engine import registry\n"
        "registry._torch_accelerators = lambda: (False, False, '')\n"
        "eng = registry.build_engine(Settings(engine='unlimited', device='auto'))\n"
        "print(eng.device, 'mlx' in sys.modules, 'mlx.core' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=120, check=True,
    )
    assert out.stdout.split() == ["cpu", "False", "False"]


@pytest.mark.parametrize("mlx,expected,dtype", [(True, "mlx", "bfloat16"), (False, "metal", "auto")])
def test_health_reports_the_resolved_device(monkeypatch, tmp_path, mlx, expected, dtype):
    from fastapi.testclient import TestClient

    from app.main import create_app

    _stub_probes(monkeypatch, mlx=mlx, cuda=False, mps=True)
    settings = Settings(
        engine="unlimited", device="auto", preload_model=False,
        data_dir=tmp_path / "data", frontend_dir=tmp_path / "no-frontend",
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/api/health").json()
    assert (body["engine"], body["device"], body["dtype"]) == ("unlimited", expected, dtype)
    assert body["model_loaded"] is False


def test_env_default_device_is_auto(monkeypatch):
    """OCR_DEVICE 미설정·빈 값 = auto — 로컬 실행 기본(Apple Silicon의 make dev가 CPU fp32로
    조용히 돌던 함정 제거). 직접 생성한 Settings는 탐지 없는 cpu 그대로라 결정적이다."""
    monkeypatch.delenv("OCR_DEVICE", raising=False)
    assert Settings.from_env().device == "auto"
    monkeypatch.setenv("OCR_DEVICE", "  ")
    assert Settings.from_env().device == "auto"
    monkeypatch.setenv("OCR_DEVICE", " MLX ")
    assert Settings.from_env().device == "mlx"
    assert Settings().device == "cpu"


def test_containers_pin_their_device_so_auto_stays_a_local_default():
    """auto 기본이 컨테이너 의미를 바꾸지 않는 전제 — backend 서비스는 OCR_DEVICE를 cpu/cuda로
    고정하고(.env로 auto가 새어 들어가도 기본값 기준), Dockerfile은 기본값을 두지 않는다."""
    import re
    from pathlib import Path

    yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — compose 검사 생략")
    repo = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load((repo / "docker-compose.yml").read_text(encoding="utf-8"))
    effective = {}
    for name, svc in compose["services"].items():
        build = svc.get("build")
        if not isinstance(build, dict) or build.get("dockerfile") != "backend/Dockerfile":
            continue
        value = str(svc["environment"]["OCR_DEVICE"])
        m = re.fullmatch(r"\$\{OCR_DEVICE:-(\w+)\}", value)
        effective[name] = m.group(1) if m else value
    assert effective == {"ocr-cpu": "cpu", "ocr-cuda": "cuda", "ocr-ovis": "cpu", "ocr-paddle": "cpu"}
    assert "OCR_DEVICE" not in (repo / "backend" / "Dockerfile").read_text(encoding="utf-8")


def _make_recipe(target: str, *args: str) -> str:
    """`make -n <target>`가 실제로 실행할 명령 (make가 없으면 건너뛴다)."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("make") is None:
        pytest.skip("make 없음")
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["make", "-s", "-n", "-C", str(repo), target, *args],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout


@pytest.mark.parametrize("target", ["setup-mlx", "setup-metal"])
def test_apple_setup_targets_keep_mlx_torch_and_native_together(target):
    """uv sync는 고른 extra만 남긴다 — metal 단독 sync가 mlx·native를 지웠다(Phase 0)."""
    recipe = _make_recipe(target)
    assert "uv sync --extra metal --extra mlx" in recipe
    assert "uv pip install ../native" in recipe


def test_dev_leaves_the_device_to_auto_and_dotenv_while_dev_metal_pins_mps():
    """make dev가 OCR_DEVICE를 박으면 .env의 OCR_DEVICE가 무시된다(.env는 기존 env를 안 덮는다)."""
    assert "OCR_DEVICE" not in _make_recipe("dev")
    assert "OCR_DEVICE=metal uv run uvicorn app.main:app" in _make_recipe("dev-metal")
