"""Objective-C 오토릴리스 풀 — MPS 디코드 워커 스레드의 ObjC 임시 객체 회수.

PyTorch MPS 백엔드는 커맨드 버퍼·MPSGraph 텐서 래퍼 같은 Objective-C 임시 객체를
autorelease로 넘긴다. 메인 런루프가 없는 파이썬 스레드에는 이를 주기적으로 비우는
풀이 없어서, 그 스레드가 끝날 때까지 객체가 쌓인다. 잡 워커는 끝나지 않는
스레드라 사실상 무한 누적이다 — 실측(M4 Max, torch 2.10): 생성 토큰당 ~25KB,
잡 7개 동안 RSS 0.55→6.03GB. ``torch.mps.empty_cache()``는 GPU 버퍼만 돌려주므로
이 힙에는 효과가 없다 (audit gap1-metal-real-e2e-1).

``autorelease_pool()``은 libobjc의 ``objc_autoreleasePoolPush/Pop``을 ctypes로 감싼
컨텍스트 매니저다. darwin이 아니거나 libobjc를 열 수 없으면 아무것도 하지 않는다.
push/pop은 같은 스레드에서 LIFO로 짝이 맞아야 하므로 반드시 ``with`` 블록으로만
쓴다(중첩 가능, 예외가 나도 finally에서 pop).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import sys

logger = logging.getLogger(__name__)

_LIBOBJC_PATH = "/usr/lib/libobjc.A.dylib"


@functools.lru_cache(maxsize=1)
def _pool_functions():
    """(push, pop) ctypes 함수 쌍 — 쓸 수 없는 환경이면 None (지연 로드·1회 캐시)."""
    if sys.platform != "darwin":
        return None
    import ctypes

    try:
        lib = ctypes.CDLL(_LIBOBJC_PATH)
        push = lib.objc_autoreleasePoolPush
        pop = lib.objc_autoreleasePoolPop
    except (OSError, AttributeError) as exc:  # pragma: no cover - 비정상 macOS 설치
        logger.warning("libobjc 오토릴리스 풀을 쓸 수 없어 생략합니다: %s", exc)
        return None
    push.restype = ctypes.c_void_p
    push.argtypes = []
    pop.restype = None
    pop.argtypes = [ctypes.c_void_p]
    return push, pop


def available() -> bool:
    """이 프로세스에서 실제 풀을 쓸 수 있는지 (darwin + libobjc)."""
    return _pool_functions() is not None


@contextlib.contextmanager
def autorelease_pool(enabled: bool = True):
    """블록 안에서 autorelease된 ObjC 객체를 블록이 끝날 때 해제한다.

    enabled=False이거나 풀을 쓸 수 없는 환경이면 no-op. 호출자는 MPS 작업을 할 때만
    enabled=True로 둔다 — CPU/CUDA 경로에는 비울 ObjC 객체가 없다."""
    fns = _pool_functions() if enabled else None
    if fns is None:
        yield
        return
    push, pop = fns
    token = push()
    try:
        yield
    finally:
        pop(token)
