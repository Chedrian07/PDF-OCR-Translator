"""모델 로드 재시도 · 엔진 사망 시 자가 재시작 — 표준 라이브러리만 쓴다(테스트에서 직접 임포트).

로드 재시도 (감사 sidecar-6):
  첫 기동의 HF 다운로드가 네트워크 순단·5xx·rate limit으로 한 번만 실패해도, 예전에는
  load_error가 굳어 운영자가 컨테이너를 수동 재시작할 때까지 모든 잡이 즉시 실패했다
  (프로세스는 살아 있어 restart 정책도 안 걸린다). 일시적 실패는 지수 백오프로 다시
  시도하고, 그동안 health는 status=ok·model_loaded=false — backend가 '로딩 중'으로 보고
  기다리는 조합 — 를 유지한다. 재시도해도 풀리지 않는 실패(CUDA 가드·설치 누락·
  401/403/404)와 마지막 시도의 실패만 load_error로 고정한다(backend는 즉시 하드 실패).

엔진 사망 시 자가 재시작 (감사 sidecar-7):
  추론 엔진이 프로세스째 죽으면(vLLM EngineCore 사망 등) 같은 프로세스 안에서는 회복할
  수 없다. 그 요청에는 503을 돌려주고 잠시 뒤 프로세스를 끝내 Docker restart 정책
  (unless-stopped)이 컨테이너를 다시 띄우게 한다. backend는 503·연결 끊김을 '재시작/
  재로드 중'으로 보고 준비될 때까지 기다렸다가 그 페이지만 다시 보낸다(engine/sidecar.py).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator

# 재시도 사이 대기(초). 대기 합 225초 — backend의 모델 대기 상한 기본값(900초,
# OCR_SIDECAR_MODEL_WAIT_S) 안에서 끝나 잡이 기다리는 동안 복구될 수 있다.
LOAD_BACKOFF_S: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0)
# 503 응답이 소켓으로 나갈 시간을 준 뒤 종료한다
RESTART_DELAY_S = 1.5
# 컨테이너 재시작을 요청하는 종료 코드 — 0이 아니어야 restart 정책이 확실히 동작한다
RESTART_EXIT_CODE = 3


class PermanentLoadError(RuntimeError):
    """재시도해도 풀리지 않는 로드 실패 — 예: CUDA를 볼 수 없는 컨테이너."""


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _http_status(exc: BaseException) -> int | None:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def is_transient_load_error(exc: BaseException) -> bool:
    """다시 시도하면 풀릴 수 있는 실패인가.

    결정적인 것만 골라내고 나머지는 일시적으로 본다 — 재시도 횟수가 상한돼 있어
    잘못 분류해도 비용은 '하드 실패가 몇 분 늦어지는 것'뿐이고, 반대로 일시적 실패를
    결정적으로 잘못 보면 수동 재시작 전까지 스택 전체를 못 쓴다."""
    for e in _chain(exc):
        if isinstance(e, (PermanentLoadError, ImportError)):
            return False
        status = _http_status(e)
        # 401/403(토큰·게이트)·404(저장소·revision 오타)는 기다려도 그대로다
        if status is not None and 400 <= status < 500 and status not in (408, 429):
            return False
    return True


def is_engine_dead(exc: BaseException, markers: tuple[str, ...]) -> bool:
    """예외 체인의 클래스명·메시지에 엔진 사망 시그니처가 있는가 (소문자 부분 일치)."""
    return any(
        m in f"{e.__class__.__name__}: {e}".lower() for e in _chain(exc) for m in markers
    )


def supervise_load(
    model,
    log: logging.Logger,
    *,
    backoff: tuple[float, ...] = LOAD_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """model.load()를 일시적 실패에 한해 backoff대로 다시 시도한다. 성공하면 True.

    model은 load()·load_error·load_retry를 가진 객체다. 재시도 중에는 load_error를
    비워 두고(health status=ok) 진행 상황을 load_retry로만 알린다."""
    attempts = len(backoff) + 1
    for attempt in range(1, attempts + 1):
        try:
            model.load()
        except Exception as e:  # noqa: BLE001 — 분류해서 재시도하거나 load_error로 확정
            reason = f"{e.__class__.__name__}: {e}"[:500]
            if attempt == attempts or not is_transient_load_error(e):
                model.load_retry = None
                model.load_error = reason
                log.exception("모델 로드 실패 (시도 %d/%d) — 더 재시도하지 않습니다",
                              attempt, attempts)
                return False
            delay = backoff[attempt - 1]
            model.load_retry = {
                "attempt": attempt,
                "max_attempts": attempts,
                "next_retry_s": delay,
                "last_error": reason,
            }
            log.warning("모델 로드 실패 (시도 %d/%d) — %.0f초 뒤 재시도: %s",
                        attempt, attempts, delay, reason)
            sleep(delay)
        else:
            model.load_retry = None
            return True
    return False  # pragma: no cover — 루프는 위에서 항상 반환한다


_restart_lock = threading.Lock()
_restart_timer: threading.Timer | None = None


def schedule_restart(
    log: logging.Logger,
    reason: str,
    *,
    delay_s: float = RESTART_DELAY_S,
    exit_fn: Callable[[int], None] = os._exit,
) -> bool:
    """잠시 뒤 프로세스를 끝낸다(멱등). 이번 호출이 예약했으면 True.

    os._exit을 쓰는 이유: SIGTERM 경로는 uvicorn이 열린 연결을 기다리며 멈출 수 있고,
    이미 죽은 엔진의 정리 코드는 또 실패할 수 있다. 종료 코드는 0이 아니어야
    restart 정책이 컨테이너를 다시 띄운다."""
    global _restart_timer
    with _restart_lock:
        if _restart_timer is not None:
            return False
        log.error("추론 엔진 복구 불가 (%s) — %.1f초 뒤 프로세스를 종료해 컨테이너 재시작을 "
                  "요청합니다", reason, delay_s)
        timer = threading.Timer(delay_s, exit_fn, args=(RESTART_EXIT_CODE,))
        timer.daemon = True
        timer.start()
        _restart_timer = timer
        return True
