"""ObjC 오토릴리스 풀 헬퍼(app.engine.objc_pool) — MPS 워커 스레드 누수 차단.

잡 워커는 끝나지 않는 스레드라 MPS 작업이 autorelease한 ObjC 객체를 비울 풀이
없으면 토큰당 ~25KB씩 영구히 쌓인다(audit gap1-metal-real-e2e-1). 헬퍼 계약:
darwin이 아니면 no-op, push/pop은 LIFO로 짝이 맞고 예외에도 pop한다.
실제 MPS 누수 회수는 OCR_MPS_TESTS=1(Apple Silicon)에서만 돈다.
"""

import ctypes
import os
import sys
import threading

import pytest

from app.engine import objc_pool


@pytest.fixture
def fresh_loader():
    objc_pool._pool_functions.cache_clear()
    yield
    objc_pool._pool_functions.cache_clear()


def _spy_functions(log):
    tokens = iter(range(1, 100))

    def push():
        token = next(tokens)
        log.append(("push", token))
        return token

    def pop(token):
        log.append(("pop", token))

    return push, pop


def test_noop_off_darwin(monkeypatch, fresh_loader):
    monkeypatch.setattr(objc_pool.sys, "platform", "linux")
    assert objc_pool.available() is False
    entered = False
    with objc_pool.autorelease_pool():
        entered = True
    assert entered


def test_disabled_pool_never_loads_libobjc(monkeypatch):
    def boom():
        raise AssertionError("enabled=False인데 libobjc 로드를 시도함")

    monkeypatch.setattr(objc_pool, "_pool_functions", boom)
    with objc_pool.autorelease_pool(False):
        pass


def test_nested_pools_pop_in_lifo_order(monkeypatch):
    log: list = []
    fns = _spy_functions(log)
    monkeypatch.setattr(objc_pool, "_pool_functions", lambda: fns)
    with objc_pool.autorelease_pool():
        with objc_pool.autorelease_pool():
            pass
    assert log == [("push", 1), ("push", 2), ("pop", 2), ("pop", 1)]


def test_pool_pops_even_when_block_raises(monkeypatch):
    log: list = []
    fns = _spy_functions(log)
    monkeypatch.setattr(objc_pool, "_pool_functions", lambda: fns)
    with pytest.raises(RuntimeError, match="boom"):
        with objc_pool.autorelease_pool():
            raise RuntimeError("boom")
    assert log == [("push", 1), ("pop", 1)]


@pytest.mark.skipif(sys.platform != "darwin", reason="libobjc는 macOS 전용")
def test_real_libobjc_push_pop_on_worker_thread(fresh_loader):
    """실제 libobjc: 풀 토큰이 유효하고, 워커 스레드에서 중첩 사용해도 안전하다."""
    assert objc_pool.available() is True
    errors: list = []

    def work():
        try:
            for _ in range(100):
                with objc_pool.autorelease_pool():
                    with objc_pool.autorelease_pool():
                        pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert errors == []


def _resident_mb() -> float:
    """mach task_info(TASK_VM_INFO)의 resident_size (MB)."""

    class _Info(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("region_count", ctypes.c_int32),
            ("page_size", ctypes.c_int32),
            ("resident_size", ctypes.c_uint64),
            ("rest", ctypes.c_uint64 * 32),
        ]

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
    info = _Info()
    count = ctypes.c_uint(ctypes.sizeof(info) // 4)
    assert libc.task_info(task, 22, ctypes.byref(info), ctypes.byref(count)) == 0
    return info.resident_size / 2**20


def _mps_opt_in() -> bool:
    if os.environ.get("OCR_MPS_TESTS") != "1" or sys.platform != "darwin":
        return False
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(torch.backends.mps.is_available())


@pytest.mark.skipif(not _mps_opt_in(), reason="OCR_MPS_TESTS=1 + Apple Silicon MPS 전용")
def test_pool_drains_mps_sync_objects_on_worker_thread():
    """MPS D2H 동기화(.item())가 워커 스레드에 남기는 autorelease 객체를 풀이 회수한다.

    실측(M4 Max, torch 2.10): 풀 없이 동기화 1회당 ~0.45KB 누적, 풀 안에서는 0."""
    import torch

    x = torch.randn(64, 64, device="mps")
    growth: dict = {}

    def run(use_pool: bool) -> None:
        for _ in range(50):  # 이 스레드에서 커널·그래프 예열
            with objc_pool.autorelease_pool(use_pool):
                (x @ x).sum().item()
        torch.mps.synchronize()
        before = _resident_mb()
        for _ in range(4000):
            with objc_pool.autorelease_pool(use_pool):
                (x @ x).sum().item()
        torch.mps.synchronize()
        growth[use_pool] = _resident_mb() - before

    for use_pool in (False, True):
        th = threading.Thread(target=run, args=(use_pool,))
        th.start()
        th.join()
    if growth[False] < 0.5:
        pytest.skip(f"이 torch에서는 풀 없이도 누적이 재현되지 않음 (+{growth[False]:.2f}MB)")
    assert growth[True] < growth[False] * 0.3, growth
