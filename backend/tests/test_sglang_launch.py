"""The OCR-only launcher cannot enable optional pickle-backed grammar caches."""

import importlib.util
import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "services/unlimited_ocr2/launch.py"
    spec = importlib.util.spec_from_file_location("_test_sglang_launch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "snapshot_download", lambda *a, **kw: "/pinned/snapshot"
    )
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("OUTLINES_CACHE_DIR", "/untrusted/reused/cache")
    monkeypatch.setenv("SGLANG_DISABLE_OUTLINES_DISK_CACHE", "false")
    return module


def test_launcher_enforces_disabled_grammar_and_private_cache(launcher):
    args = launcher.command([])
    assert args[args.index("--grammar-backend") + 1] == "none"
    assert args.count("--grammar-backend") == 1
    assert "--disable-outlines-disk-cache" in args
    assert os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] == "true"
    cache = Path(os.environ["OUTLINES_CACHE_DIR"])
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert cache.stat().st_uid == os.getuid()
    assert list(cache.iterdir()) == []
    previous = cache
    launcher.command([])
    assert Path(os.environ["OUTLINES_CACHE_DIR"]) != previous


@pytest.mark.parametrize(
    "extra",
    [
        ["--grammar-backend", "outlines"],
        ["--grammar-backend=outlines"],
        ["--grammar-back", "outlines"],
        ["--grammar", "outlines"],
        ["--config", "untrusted.yaml"],
        ["--config=untrusted.yaml"],
        ["--disable-outlines-disk-cache=false"],
        ["--enable-disk-cache"],
        ["--model-path", "/different/model"],
    ],
)
def test_launcher_rejects_arbitrary_config_grammar_cache_and_abbreviated_overrides(
    launcher, extra
):
    with pytest.raises(SystemExit) as error:
        launcher.command(extra)
    assert error.value.code == 2


def test_launcher_allows_explicit_network_binding_without_arbitrary_forwarding(
    launcher,
):
    args = launcher.command(["--host", "0.0.0.0", "--port", "11000"])
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--port") + 1] == "11000"
