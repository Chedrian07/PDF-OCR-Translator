"""PyMuPDF 작업 격리 — spawn 방식 상주 워커 프로세스 풀 (감사 A11).

## 왜 별도 프로세스인가

MuPDF C 호출은 GIL을 쥔 채 돌고(`nm -u _mupdf.so`에 PyEval_SaveThread가 없다) 중간에
멈출 지점이 없다. 서버 프로세스 안에서 돌리면:

- 수 KB짜리 중첩 Form XObject PDF 하나가 단일 OCR 워커를 영구히 점거하고, GIL 때문에
  /api/health·취소까지 멈춘다 — 재시작만이 복구 수단이었다(security-2).
- 평면 10^7 path 문서의 분석(get_drawings ~1.6KB/path)이 컨테이너 메모리 상한을 넘겨
  서버째 OOM-kill되고, 대기 중이던 다른 사용자의 잡까지 error가 된다(gap3-…-2).
- 번역 PDF 예열 빌드 37초 동안 같은 프로세스의 OCR 디코드가 31.7→1.0 tok/s로 굶는다
  (gap1-metal-real-e2e-2·concurrency-2) — MLX·torch 디코드 루프도 토큰마다 GIL이 필요하다.
- 빌드 스레드 N개는 코어 하나를 나눠 쓸 뿐이고(가속비 1.00, concurrency-3·pdf-export-9),
  한 프로세스에서 MuPDF를 여러 스레드로 쓰는 것은 업스트림 비지원이다(손상 PDF 2스레드에서
  SIGSEGV 재현).

그래서 MuPDF 작업을 `'모듈:함수'` 이름으로 지정해 별도 프로세스에서 실행한다. 인자·결과는
피클로 오가고(경로·문자열·dict·dataclass), 작업마다 벽시계 상한을 둔다. 상한을 넘거나
워커가 죽으면(SIGSEGV·OOM-kill) 그 워커만 종료하고 다음 호출이 새 워커를 띄운다 — 서버
프로세스와 다른 잡은 영향을 받지 않는다.

## 풀

- ``ocr``(1개): OCR 워커 스레드의 입력 페이지 렌더·충실도 분석·텍스트 레이어 복구·페이지
  정합 텍스트·폰트 실측 주입·textlayer 엔진 추출. 잡은 직렬이라 1개면 된다.
- ``export``(PDF_EXPORT_MAX_CONCURRENT개, 0 이하면 min(8, CPU)): 번역·대조 PDF 빌드,
  facsimile 래스터, 레이아웃 폰트 백필. OCR이 내보내기 뒤에 줄서지 않게 풀을 나눈다.
- ``probe``(2개): 업로드 검증(probe_pdf + 복잡도 게이트). 업로드가 수 분짜리 빌드나 적대적
  페이지를 처리 중인 OCR 워커 뒤에 줄서지 않게 따로 둔다.

워커는 처음 쓸 때 띄우고(lazy), 앱 lifespan 종료와 atexit에서 멈춘다. 작업 수
(_MAX_TASKS_PER_WORKER)나 최대 RSS(_RECYCLE_RSS_BYTES)를 넘긴 워커는 반납 시 교체한다.

## 실행 모드 (PDF_WORKER_MODE)

``process``(기본)와 ``inline``. inline은 예전처럼 호출 스레드에서 바로 실행한다(상한·격리
없음). 테스트 세션(conftest)이 inline으로 두는 이유: 많은 테스트가 pymupdf·pdf_export
내부를 프로세스 안에서 monkeypatch하기 때문이다. 격리 자체는 전용 테스트가 process 모드로
검증한다. 워커 **안에서** 다시 이 모듈을 거치는 호출(예: 빌드 안의 폰트 주입)은 항상 inline이다.

## 워커 프로세스

- 이 모듈과 작업 모듈만 임포트한다(app.pipeline.pdf·fidelity·pdf_fonts·pdf_export·
  engine.textlayer — torch·mlx·app.main·설정/LLM 계층은 끌어오지 않는다). 부모의
  `__main__`도 다시 실행하지 않는다(_spawn_without_main 참조).
- 로그는 stderr로(서버 콘솔·docker logs에 그대로 섞인다), 프로세스 이름(pdf-ocr-1 등)을 붙인다.
- SIGINT는 무시한다(개발 서버 Ctrl+C는 부모가 정리한다).
- 임시 파일(tempfile — 폰트 서브셋 등)은 워커 전용 디렉터리(시스템 임시 경로의
  pdfocr-worker-<풀>-<pid>)에 만든다. 상한 초과로 종료된 워커는 정리 코드를 못 돌리므로
  부모가 그 디렉터리를 지우고, 이전 서버가 SIGKILL로 남긴 것은 첫 풀 생성 때 쓸어 낸다.
- 작업마다 `signal.alarm(상한 + 여유)`를 건다. SIGALRM 기본 동작은 커널이 프로세스를 끝내는
  것이라 GIL이 필요 없다 — 부모가 SIGKILL로 사라져 아무도 죽여 주지 않아도 적대적 작업이 영원히
  CPU를 태우지 않는다.
- 자격 증명처럼 보이는 환경 변수(…KEY·…TOKEN·…SECRET·…PASSWORD — 번역·Q&A API 키, HF 토큰)를
  기동 즉시 지운다. 워커는 비밀이 필요 없고, MuPDF 메모리 결함(예: CVE-2026-3308)으로 워커가
  장악돼도 키를 읽지 못하게 하는 심층 방어다. 워커는 자원·장애 격리 경계이지 권한 샌드박스가
  아니다(같은 사용자·같은 파일 시스템).
- Linux에서는 /proc/self/oom_score_adj를 1000으로 올려 메모리 압박 시 커널이 서버가 아니라
  워커를 먼저 고르게 한다. PDF_WORKER_MEM_LIMIT_MB(>0)면 RLIMIT_AS도 건다(macOS는 커널이
  RLIMIT_AS를 강제하지 않아 건너뛴다).

## 페이지 격리(quarantine)

페이지 단위 작업(run_page)이 상한을 넘거나 워커를 죽인 (파일, 페이지)는 프로세스 메모에
남겨, 같은 잡의 다음 분석(충실도·정합·폰트 주입·텍스트 복구)이 같은 페이지에서 상한을 또
기다리지 않고 곧바로 건너뛰게 한다.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import importlib
import itertools
import logging
import math
import multiprocessing
import os
import pickle
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from multiprocessing import connection as mp_connection
from multiprocessing import spawn as mp_spawn
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── 설정 키 (호출 시점에 os.environ에서 읽는다 — derived의 PDF_EXPORT_*와 같은 방식) ──
MODE_ENV = "PDF_WORKER_MODE"
PAGE_TIMEOUT_ENV = "PDF_PAGE_TIMEOUT_S"
BUILD_TIMEOUT_ENV = "PDF_EXPORT_BUILD_TIMEOUT_S"
MEM_LIMIT_ENV = "PDF_WORKER_MEM_LIMIT_MB"
MAX_PAGE_CONTENT_ENV = "PDF_MAX_PAGE_CONTENT_MB"
MAX_PAGE_CALLS_ENV = "PDF_MAX_PAGE_XOBJECT_CALLS"
# export 풀 크기는 derived의 빌드 슬롯 상한과 같은 키를 쓴다(그 모듈을 임포트하지 않으려고
# 이름만 같이 둔다) — 슬롯을 쥔 빌드가 워커를 기다리지 않게 한다.
EXPORT_SLOTS_ENV = "PDF_EXPORT_MAX_CONCURRENT"

MODE_PROCESS = "process"
MODE_INLINE = "inline"

# 페이지 한 장의 MuPDF 작업(렌더·분석·추출) 상한. 정상 페이지는 0.01~수 초이고, 업로드
# 게이트 상한(아래)에 걸리기 직전의 페이지도 M4 Max 실측 렌더 ~8s·벡터 추출 ~13s라 느린
# 서버에서도 넉넉하다. 이것을 넘는 페이지는 흰 페이지로 대체된다.
_PAGE_TIMEOUT_DEFAULT = 60.0
# 번역·대조 PDF 빌드 한 건의 상한 — 실측 ~0.5s/쪽(200쪽 ~100s)의 몇 배.
_BUILD_TIMEOUT_DEFAULT = 900.0
# 업로드 게이트(probe_pdf): 페이지가 그리게 하는 콘텐츠(콘텐츠 스트림 + 도달하는 XObject·
# 패턴·Type3 글리프 스트림, 서로 다른 스트림은 한 번씩)의 압축 해제 바이트, 그리고 Form
# XObject 중첩을 펼친 그리기 호출 수. 실측(M4 Max): 평면 path 10^6개 = 33MB → 렌더 1.5s·
# 벡터 추출 3.4s/1.66GB, 펼친 호출 10^6회 → 렌더 3.6~4.4s·추출 6~7s/1.3GB. 정상 논문
# 표본은 페이지당 최대 0.3MB·129회. 상한은 그 수백 배로 넉넉히 잡되 서버가 감당할 수 있는
# 선(64MB ≈ 평면 path 1.9×10^6개, 2×10^6회)에 둔다. 0 이하 = 그 검사 끄기.
_MAX_PAGE_CONTENT_MB_DEFAULT = 64
_MAX_PAGE_CALLS_DEFAULT = 2_000_000
_EXPORT_POOL_FALLBACK_MAX = 8

POOL_OCR = "ocr"
POOL_EXPORT = "export"
POOL_PROBE = "probe"
_PROBE_POOL_SIZE = 2

# 워커 교체 기준 — MuPDF 저장소·파이썬 힙 단편화가 프로세스 수명 내내 쌓이지 않게.
_MAX_TASKS_PER_WORKER = 200
_RECYCLE_RSS_BYTES = 1536 * 1024 * 1024
# 취소 콜백·풀 대기를 확인하는 간격
_POLL_S = 0.1
# terminate(SIGTERM) 뒤 kill(SIGKILL)까지 기다리는 시간 — 파이썬은 SIGTERM 처리기를 두지
# 않으므로 C 코드 안이어도 커널이 즉시 끝낸다. 그래도 남아 있으면 SIGKILL.
_KILL_GRACE_S = 1.0
# 워커 안 SIGALRM 자가 종료는 부모 상한보다 이만큼 늦게 — 부모가 먼저 정리하는 것이 정상 경로다.
_ALARM_GRACE_S = 10
# 이 시간 동안 작업이 없으면 워커가 캐시한 문서를 닫는다(지운 잡 파일의 inode를 붙잡지 않게).
_IDLE_DOC_CLOSE_S = 30.0
# 페이지 격리 메모 상한(문서 수)
_QUARANTINE_MAX_DOCS = 256


# ── 예외 ─────────────────────────────────────────────────────────────────


class PdfWorkerError(RuntimeError):
    """격리 실행 자체의 실패(작업이 낸 예외가 아니다) — 상한 초과·워커 비정상 종료 등."""


class PdfWorkerTimeout(PdfWorkerError):
    """작업이 벽시계 상한을 넘어 워커를 종료했다."""

    def __init__(self, seconds: float | None, what: str = "PDF 처리") -> None:
        self.seconds = seconds
        self.what = what
        limit = f"{seconds:g}초" if seconds else "상한"
        super().__init__(f"{what}가 시간 상한({limit})을 넘어 중단했습니다")

    def __reduce__(self):
        return (type(self), (self.seconds, self.what))


class PdfPageQuarantined(PdfWorkerTimeout):
    """앞서 시간 상한을 넘었거나 워커를 죽인 페이지 — 다시 시도하지 않고 건너뛴다."""

    def __init__(self, page_index: int) -> None:
        self.page_index = page_index
        self.seconds = None
        self.what = "PDF 처리"
        PdfWorkerError.__init__(
            self, f"{page_index + 1}페이지는 앞서 처리 상한을 넘은 페이지라 건너뜁니다",
        )

    def __reduce__(self):
        return (type(self), (self.page_index,))


class PdfWorkerCrashed(PdfWorkerError):
    """워커 프로세스가 결과 없이 끝났다(SIGSEGV·OOM-kill·os._exit 등)."""

    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode
        super().__init__(
            f"PDF 처리 프로세스가 비정상 종료했습니다 (exit {exitcode}) — "
            "손상됐거나 처리할 수 없는 PDF일 수 있습니다"
        )

    def __reduce__(self):
        return (type(self), (self.exitcode,))


class PdfWorkerCanceled(PdfWorkerError):
    """취소 콜백이 참이 되어 진행 중이던 작업의 워커를 종료했다."""

    def __init__(self) -> None:
        super().__init__("취소되어 PDF 처리를 중단했습니다")

    def __reduce__(self):
        return (type(self), ())


class PdfWorkerBusy(PdfWorkerError):
    """제한 시간 안에 빈 워커를 얻지 못했다 — 일시적 과부하(재시도하면 성공할 수 있다)."""

    def __init__(self, pool: str, waited: float) -> None:
        self.pool = pool
        self.waited = waited
        super().__init__("PDF 처리 대기열이 가득 찼습니다 — 잠시 후 다시 시도하세요")

    def __reduce__(self):
        return (type(self), (self.pool, self.waited))


class PdfWorkerRemoteError(PdfWorkerError):
    """작업이 낸 예외를 그대로 옮길 수 없을 때(피클 불가)의 대체 — 원래 타입 이름과 문구."""

    def __init__(self, type_name: str, message: str) -> None:
        self.type_name = type_name
        self.message = message
        super().__init__(f"{type_name}: {message}" if message else type_name)

    def __reduce__(self):
        return (type(self), (self.type_name, self.message))


class _RemoteTraceback(Exception):
    """워커 쪽 traceback 문자열 — 다시 던진 예외의 __cause__로 붙여 로그에서 원인을 보인다."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return f"\n(PDF 워커 traceback)\n{self.text}"


# ── 설정 읽기 ─────────────────────────────────────────────────────────────

_WARNED_ENV: set[tuple[str, str]] = set()
_WARNED_LOCK = threading.Lock()


def _warn_env_once(name: str, raw: str, default: object) -> None:
    with _WARNED_LOCK:
        if (name, raw) in _WARNED_ENV:
            return
        _WARNED_ENV.add((name, raw))
    logger.warning("%s 값이 올바르지 않습니다 (%r) — 기본값 %s 사용", name, raw, default)


def _env_number(name: str, default: float) -> float:
    """숫자 env — 오타는 서비스를 멈추지 않고 기본값으로 강등한다(한 번만 경고)."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _warn_env_once(name, raw, default)
        return default
    if not math.isfinite(value):
        _warn_env_once(name, raw, default)
        return default
    return value


def _env_timeout(name: str, default: float) -> float | None:
    """초 단위 상한 — 0 이하는 '상한 없음'(None)."""
    value = _env_number(name, default)
    return value if value > 0 else None


def page_timeout() -> float | None:
    """페이지 한 장의 MuPDF 작업 상한(초, PDF_PAGE_TIMEOUT_S). None = 상한 없음."""
    return _env_timeout(PAGE_TIMEOUT_ENV, _PAGE_TIMEOUT_DEFAULT)


def export_build_timeout() -> float | None:
    """번역·대조 PDF 빌드 한 건의 상한(초, PDF_EXPORT_BUILD_TIMEOUT_S). None = 상한 없음."""
    return _env_timeout(BUILD_TIMEOUT_ENV, _BUILD_TIMEOUT_DEFAULT)


def worker_mem_limit_mb() -> int:
    """워커당 가상 메모리 상한(MB, PDF_WORKER_MEM_LIMIT_MB) — 0 = 끄기, Linux에서만 적용."""
    return max(0, int(_env_number(MEM_LIMIT_ENV, 0)))


def max_page_content_bytes() -> int:
    """업로드 게이트의 페이지당 압축 해제 콘텐츠 상한(바이트). 0 = 검사 끄기."""
    mb = _env_number(MAX_PAGE_CONTENT_ENV, _MAX_PAGE_CONTENT_MB_DEFAULT)
    return int(mb * 1024 * 1024) if mb > 0 else 0


def max_page_xobject_calls() -> int:
    """업로드 게이트의 페이지당 펼친 XObject 호출 상한. 0 = 검사 끄기."""
    calls = _env_number(MAX_PAGE_CALLS_ENV, _MAX_PAGE_CALLS_DEFAULT)
    return int(calls) if calls > 0 else 0


# 워커 프로세스 안인가 — 거기서는 이 모듈을 거치는 호출이 전부 inline이다.
_IN_WORKER = False


def in_worker() -> bool:
    return _IN_WORKER


def mode() -> str:
    """현재 실행 모드 — 워커 안이면 inline, 아니면 PDF_WORKER_MODE(기본 process)."""
    if _IN_WORKER:
        return MODE_INLINE
    raw = (os.environ.get(MODE_ENV) or "").strip().lower()
    if not raw or raw == MODE_PROCESS:
        return MODE_PROCESS
    if raw == MODE_INLINE:
        return MODE_INLINE
    _warn_env_once(MODE_ENV, raw, MODE_PROCESS)
    return MODE_PROCESS


def _export_pool_size() -> int:
    slots = int(_env_number(EXPORT_SLOTS_ENV, 2))
    if slots > 0:
        return slots
    return max(1, min(_EXPORT_POOL_FALLBACK_MAX, os.cpu_count() or 2))


_POOL_SIZES: dict[str, Callable[[], int]] = {
    POOL_OCR: lambda: 1,
    POOL_EXPORT: _export_pool_size,
    POOL_PROBE: lambda: _PROBE_POOL_SIZE,
}


# ── 호출 맥락: 어느 풀을 쓸 것인가 ──────────────────────────────────────────
# 같은 함수(render_pdf_pages·enrich_layout_fonts)를 OCR 경로와 내보내기 경로가 함께 쓴다.
# 시그니처를 바꾸면 주입된 렌더러·빌더(테스트 대역 포함)가 깨지므로, 내보내기 쪽 호출부가
# `with pool_scope(POOL_EXPORT):`로 감싸고 함수 안에서는 current_pool()로 읽는다.
_POOL_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pdf_worker_pool", default=None,
)


@contextlib.contextmanager
def pool_scope(name: str):
    token = _POOL_SCOPE.set(name)
    try:
        yield
    finally:
        _POOL_SCOPE.reset(token)


def current_pool(default: str = POOL_OCR) -> str:
    return _POOL_SCOPE.get() or default


# ── 작업 이름 해석 ─────────────────────────────────────────────────────────


def resolve(target: str) -> Callable[..., Any]:
    """'패키지.모듈:함수' → 호출 대상. 호출 시점에 찾으므로 inline 모드에서는 모듈 전역을
    바꾼 monkeypatch가 그대로 보인다. 대상 이름은 코드가 정한다(사용자 입력이 아니다)."""
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"작업 이름 형식은 'module:function'입니다: {target!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# ── 워커 프로세스 쪽 ───────────────────────────────────────────────────────

_DOC_CACHE: list[tuple[tuple, Any]] = []


def _doc_key(pdf_path) -> tuple | None:
    """파일 정체성 — 같은 경로라도 교체(새 inode)·수정되면 다른 문서다."""
    try:
        resolved = os.path.realpath(os.fspath(pdf_path))
        stat = os.stat(resolved)
    except (OSError, TypeError, ValueError):
        return None
    return (resolved, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _close_cached_document() -> None:
    while _DOC_CACHE:
        _key, doc = _DOC_CACHE.pop()
        with contextlib.suppress(Exception):
            doc.close()


@contextlib.contextmanager
def open_document(pdf_path):
    """페이지 작업용 PyMuPDF 문서 — 워커 안에서는 같은 파일을 작업 사이에 재사용한다.

    페이지마다 작업을 나누면 (페이지 단위 상한·격리) 매번 여는 비용이 생긴다. 정상 파일은
    수 ms지만 xref가 깨져 repair가 필요한 큰 파일은 열 때마다 수 초다. 워커는 단일 스레드라
    마지막 문서 하나를 캐시해도 안전하다. 작업이 예외로 끝나면 워커 루프가 캐시를 버린다
    (MuPDF 오류 뒤의 문서 상태를 믿지 않는다). 워커 밖(inline)에서는 열고 닫는다.
    열기 실패는 MuPDF 예외 그대로 올린다 — 호출부마다 처리 방식이 다르다.
    """
    from .pdf import quiet_fitz  # 지연 임포트 — pdf가 이 모듈을 임포트한다

    fitz = quiet_fitz()
    if not _IN_WORKER:
        doc = fitz.open(os.fspath(pdf_path))
        try:
            yield doc
        finally:
            doc.close()
        return
    key = _doc_key(pdf_path)
    if key is not None and _DOC_CACHE and _DOC_CACHE[0][0] == key:
        yield _DOC_CACHE[0][1]
        return
    _close_cached_document()
    doc = fitz.open(os.fspath(pdf_path))
    if key is not None:
        _DOC_CACHE.append((key, doc))
        yield doc
        return
    try:
        yield doc
    finally:
        doc.close()


# 워커 자신의 최대 RSS를 읽는 곳(Linux procfs) — 없으면 ru_maxrss로 폴백한다
_PROC_STATUS = "/proc/self/status"
# (워커) 기동 시점의 ru_maxrss(바이트) — 물려받은 값과 이 워커가 키운 값을 가르는 기준
_START_MAXRSS = 0


def _ru_maxrss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover — POSIX 전용 배포
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _peak_rss_bytes() -> int:
    """(워커) 이 워커 자신의 최대 RSS(바이트) — 반납 시 교체 판단(_RECYCLE_RSS_BYTES)에 쓴다.

    Linux의 ru_maxrss는 fork·exec를 넘어 부모의 최댓값을 물려받는다(커널 signal->maxrss).
    모델을 올린 서버(수 GB)가 띄운 워커는 첫 작업부터 상한을 넘은 것으로 보여 작업마다 폐기·
    재생성됐다(상주 워커·문서 캐시 무력화, 감사 infra-docs-1). /proc/self/status의 VmHWM은
    exec마다 새로 시작하는 이 프로세스 메모리 맵의 최댓값이라 그것을 먼저 읽는다. 없으면(macOS
    등) ru_maxrss를 쓰되 기동 시점 값(물려받은 몫)을 넘지 않았으면 이 워커의 값이 아니므로 0이다.
    """
    try:
        with open(_PROC_STATUS, "rb") as status:
            for line in status:
                if line.startswith(b"VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    peak = _ru_maxrss_bytes()
    return peak if peak > _START_MAXRSS else 0


def _apply_memory_limit(limit_mb: int) -> None:
    """Linux에서만 RLIMIT_AS를 건다 — 넘으면 malloc이 실패해 작업이 예외로 끝난다(OOM-kill 대신).

    macOS 커널은 RLIMIT_AS를 강제하지 않으므로(설정은 되지만 무시) 건너뛴다."""
    if limit_mb <= 0 or not sys.platform.startswith("linux"):
        return
    try:
        import resource

        limit = limit_mb * 1024 * 1024
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ImportError, OSError, ValueError) as error:
        logger.warning("PDF 워커 메모리 상한 적용 실패: %s", error)


def _prefer_oom_kill() -> None:
    """Linux: 메모리 압박 시 커널 OOM killer가 서버 대신 이 워커를 먼저 고르게 한다.

    자기 oom_score_adj를 **올리는** 것은 권한 없이 된다(cap_drop ALL 컨테이너 포함)."""
    if not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text("1000", encoding="ascii")


# 워커에서 지울 환경 변수 이름 — 대소문자 무시, 이름 어디에든 들어 있으면
_SECRET_ENV_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSW(OR)?D|CREDENTIAL)", re.IGNORECASE)


def _drop_secret_env() -> None:
    """(워커) 자격 증명처럼 보이는 환경 변수를 지운다 — PDF 작업에는 필요 없다."""
    for name in [key for key in os.environ if _SECRET_ENV_NAME.search(key)]:
        os.environ.pop(name, None)


_SCRATCH_PREFIX = "pdfocr-worker-"


def _scratch_dir(pool_name: str, pid: int) -> Path:
    """워커 전용 임시 디렉터리 — 부모·워커가 같은 규칙으로 찾는다(TMPDIR 상속)."""
    return Path(tempfile.gettempdir()) / f"{_SCRATCH_PREFIX}{pool_name}-{pid}"


def _use_scratch_dir(pool_name: str) -> Path | None:
    """(워커) tempfile 기본 경로를 워커 전용 디렉터리로 돌린다."""
    scratch = _scratch_dir(pool_name, os.getpid())
    try:
        scratch.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        return None
    tempfile.tempdir = str(scratch)
    return scratch


_SWEPT_SCRATCH = False
_SWEPT_GUARD = threading.Lock()


def _sweep_orphan_scratch() -> None:
    """이전 서버가 SIGKILL로 남긴 워커 임시 디렉터리를 지운다(그 pid가 살아 있으면 둔다)."""
    global _SWEPT_SCRATCH
    with _SWEPT_GUARD:
        if _SWEPT_SCRATCH:
            return
        _SWEPT_SCRATCH = True
    try:
        entries = list(Path(tempfile.gettempdir()).glob(f"{_SCRATCH_PREFIX}*-*"))
    except OSError:
        return
    for entry in entries:
        try:
            pid = int(entry.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        try:
            os.kill(pid, 0)
            continue  # 살아 있는 프로세스(다른 서버의 워커일 수 있다)
        except ProcessLookupError:
            pass
        except OSError:
            continue  # 권한 없음 = 다른 사용자의 살아 있는 프로세스
        shutil.rmtree(entry, ignore_errors=True)


def _configure_child_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s [%(processName)s]: %(message)s",
        force=True,
    )


def _portable_exception(exc: BaseException) -> tuple:
    """작업 예외를 부모로 옮길 형태로 — 피클 왕복이 되면 원래 예외, 아니면 타입 이름·문구."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:  # noqa: BLE001 — 피클 불가 예외(SWIG MuPDF 예외 등)
        name = f"{type(exc).__module__}.{type(exc).__qualname__}"
        return ("remote", name, str(exc)[:2000], text)
    return ("exc", exc, text)


# SIGALRM은 POSIX 전용이다(Windows에는 없다 — 그때는 부모의 상한만 남는다). 신호 이름을
# 문자열로 쓰지 않는다 — env 키 계약 스캐너(test_ci_ops_contracts)가 대문자 문자열을 env 키로 센다.
try:
    _ALARM_SIGNAL = signal.SIGALRM
except AttributeError:  # pragma: no cover — POSIX 밖
    _ALARM_SIGNAL = None
_HAS_ALARM = _ALARM_SIGNAL is not None and hasattr(signal, "alarm")


def _child_main(conn, pool_name: str, mem_limit_mb: int, log_level: int) -> None:
    """워커 프로세스 본체 — 작업을 하나씩 받아 실행하고 결과를 돌려준다."""
    global _IN_WORKER, _START_MAXRSS
    _IN_WORKER = True
    _START_MAXRSS = _ru_maxrss_bytes()
    _drop_secret_env()
    with contextlib.suppress(ValueError, OSError, AttributeError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    _configure_child_logging(log_level)
    _apply_memory_limit(mem_limit_mb)
    _prefer_oom_kill()
    scratch = _use_scratch_dir(pool_name)
    try:
        while True:
            try:
                if not conn.poll(_IDLE_DOC_CLOSE_S):
                    _close_cached_document()
                    continue
                message = conn.recv()
            except (EOFError, OSError):
                return  # 부모가 사라졌다
            if message is None:
                return  # 정상 종료 요청
            task_id, target, args, kwargs, alarm_s = message
            if alarm_s and _HAS_ALARM:
                signal.alarm(int(alarm_s))
            try:
                reply = (task_id, True, resolve(target)(*args, **(kwargs or {})))
            except BaseException as exc:  # noqa: BLE001 — 작업 단위 격리
                reply = (task_id, False, _portable_exception(exc))
                _close_cached_document()
            finally:
                if _HAS_ALARM:
                    signal.alarm(0)
            try:
                conn.send((*reply, _peak_rss_bytes()))
            except (BrokenPipeError, EOFError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001 — 결과를 피클할 수 없다
                failure = (
                    "remote", f"{type(exc).__module__}.{type(exc).__qualname__}",
                    f"작업 결과를 부모 프로세스로 보낼 수 없습니다: {str(exc)[:500]}",
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
                )
                try:
                    conn.send((task_id, False, failure, _peak_rss_bytes()))
                except (OSError, EOFError):
                    return
    finally:
        _close_cached_document()
        with contextlib.suppress(Exception):
            conn.close()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


# ── 부모 쪽: 워커 하나 ─────────────────────────────────────────────────────

_TASK_IDS = itertools.count(1)
# 프로세스 생성은 직렬화한다 — spawn 준비(메인 모듈·sys.path 스냅샷)를 여러 스레드가
# 동시에 하지 않게.
_SPAWN_LOCK = threading.Lock()


@contextlib.contextmanager
def _spawn_without_main():
    """spawn 자식이 부모의 `__main__`을 다시 실행하지 않게 한다(_SPAWN_LOCK 안에서만).

    multiprocessing spawn은 메인 모듈에 정의된 객체를 풀 수 있도록 자식에서 부모의 메인
    스크립트를 `__mp_main__`으로 다시 실행한다. 이 워커는 그것이 필요 없다 — 대상
    (_child_main)은 이 모듈에 있고 작업은 '모듈:함수' 이름으로 임포트한다. 다시 실행하면
    `uvicorn` 콘솔 스크립트(Docker CMD·make dev)는 워커마다 uvicorn·click·watchfiles·anyio
    (약 200개 모듈, -X importtime 실측)를 싣고, 가드 없는 스크립트는 자식에서 앱을 또 만든다.
    준비 데이터에서 메인 모듈 항목만 빼고 나머지(sys.path·cwd·authkey)는 그대로 쓴다."""
    original = getattr(mp_spawn, "get_preparation_data", None)
    if original is None:  # pragma: no cover — CPython 내부 변화 시 기본 동작
        yield
        return

    def _prepare(name):
        data = original(name)
        data.pop("init_main_from_path", None)
        data.pop("init_main_from_name", None)
        return data

    mp_spawn.get_preparation_data = _prepare
    try:
        yield
    finally:
        mp_spawn.get_preparation_data = original


def _child_log_level() -> int:
    return logging.getLogger().getEffectiveLevel()


class _WorkerProcess:
    """spawn 워커 하나 — 한 번에 한 호출 스레드만 쓴다(풀이 보장)."""

    def __init__(self, pool_name: str, index: int) -> None:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self.pool_name = pool_name
        self.name = f"pdf-{pool_name}-{index}"
        self.process = ctx.Process(
            target=_child_main,
            args=(child_conn, pool_name, worker_mem_limit_mb(), _child_log_level()),
            name=self.name,
            daemon=True,
        )
        with _SPAWN_LOCK, _spawn_without_main():
            self.process.start()
        child_conn.close()
        self.conn = parent_conn
        self.tasks = 0
        self.completed = 0
        self.peak_rss = 0
        self.last_exitcode: int | None = None
        # 종료·정리는 호출 스레드와 풀 종료(shutdown) 스레드가 동시에 할 수 있다 —
        # Process.close()를 두 번 부르면 AttributeError(_sentinel)가 난다.
        self._lifecycle = threading.Lock()
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self.process.pid

    def alive(self) -> bool:
        try:
            return self.process.is_alive()
        except ValueError:  # 이미 close()된 Process
            return False

    def call(
        self, target: str, args: tuple, kwargs: dict, *,
        timeout: float | None, cancel: Callable[[], bool] | None,
    ) -> Any:
        task_id = next(_TASK_IDS)
        alarm = int(math.ceil(timeout)) + _ALARM_GRACE_S if timeout else 0
        try:
            self.conn.send((task_id, target, args, kwargs, alarm))
        except (OSError, EOFError, ValueError) as error:
            self.kill()
            raise PdfWorkerCrashed(self.exitcode()) from error
        self.tasks += 1
        deadline = None if timeout is None else time.monotonic() + timeout
        sentinel = self.process.sentinel
        while True:
            wait_s: float | None = _POLL_S if cancel is not None else None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()
                    raise PdfWorkerTimeout(timeout)
                wait_s = remaining if wait_s is None else min(wait_s, remaining)
            try:
                ready = mp_connection.wait([self.conn, sentinel], timeout=wait_s)
            except (OSError, ValueError) as error:
                # 풀 종료(shutdown) 스레드가 이 워커를 닫았다 — 결과는 오지 않는다
                self.kill()
                raise PdfWorkerCrashed(self.exitcode()) from error
            if self.conn in ready:
                return self._receive(task_id)
            if sentinel in ready:
                self.kill()
                raise PdfWorkerCrashed(self.exitcode())
            if cancel is not None and cancel():
                self.kill()
                raise PdfWorkerCanceled()

    def _receive(self, task_id: int) -> Any:
        try:
            reply = self.conn.recv()
        except (EOFError, OSError) as error:
            self.kill()
            raise PdfWorkerCrashed(self.exitcode()) from error
        except Exception as error:  # noqa: BLE001 — 결과를 부모에서 풀 수 없다
            self.kill()  # 프로토콜 상태를 믿을 수 없다
            raise PdfWorkerRemoteError(type(error).__name__, str(error)[:500]) from error
        try:
            rtask, ok, payload, rss = reply
        except (TypeError, ValueError):
            self.kill()
            raise PdfWorkerCrashed(self.exitcode()) from None
        if rtask != task_id:
            self.kill()
            raise PdfWorkerCrashed(self.exitcode())
        self.completed += 1
        self.peak_rss = max(self.peak_rss, int(rss or 0))
        if ok:
            return payload
        raise _rebuild_exception(payload)

    def exitcode(self) -> int | None:
        try:
            return self.process.exitcode
        except ValueError:  # close()된 Process — 닫기 직전에 남긴 값
            return self.last_exitcode

    def kill(self) -> None:
        """terminate → 유예 → kill. C 코드 안에서도 커널이 끝낸다(파이썬 처리기 없음).
        여러 스레드가 동시에 불러도 한 번만 정리한다."""
        with self._lifecycle:
            if self._closed:
                return
            try:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(_KILL_GRACE_S)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(_KILL_GRACE_S)
            except ValueError:
                pass  # 이미 close()된 Process
            self._close_locked()

    def stop(self, timeout: float) -> None:
        """정상 종료 요청(None) → 대기 → 남으면 kill."""
        with contextlib.suppress(Exception):
            self.conn.send(None)
        try:
            self.process.join(timeout)
        except ValueError:
            pass
        if self.alive():
            self.kill()
        else:
            self.close()

    def close(self) -> None:
        with self._lifecycle:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            self.conn.close()
        try:
            if self.process.is_alive():
                return  # 살아 있는 프로세스는 닫지 않는다(kill이 먼저다)
            pid = self.process.pid
            self.last_exitcode = self.process.exitcode
            self.process.close()
        except (ValueError, AttributeError):
            pid = None
        self._closed = True
        if pid is not None and self.last_exitcode not in (None, 0):
            # 종료당한 워커는 자기 임시 디렉터리를 못 지웠다(폰트 서브셋 등)
            shutil.rmtree(_scratch_dir(self.pool_name, pid), ignore_errors=True)


def _rebuild_exception(payload: tuple) -> BaseException:
    kind = payload[0] if payload else None
    if kind == "exc":
        _kind, exc, text = payload
        if isinstance(exc, BaseException):
            exc.__cause__ = _RemoteTraceback(text)
            return exc
    if kind == "remote":
        _kind, name, message, text = payload
        error = PdfWorkerRemoteError(name, message)
        error.__cause__ = _RemoteTraceback(text)
        return error
    return PdfWorkerRemoteError("unknown", repr(payload)[:200])


# ── 부모 쪽: 풀 ────────────────────────────────────────────────────────────


class WorkerPool:
    """이름 있는 워커 풀 — 호출 스레드마다 워커 하나를 빌려 주고, 상한·사망 시 교체한다."""

    def __init__(self, name: str, size: Callable[[], int]) -> None:
        self.name = name
        self._size = size
        self._cond = threading.Condition()
        self._idle: list[_WorkerProcess] = []
        self._all: set[_WorkerProcess] = set()
        self._checked_out = 0
        self._closed = False
        self._index = itertools.count(1)
        self.counters: dict[str, int] = dict.fromkeys(
            ("tasks", "timeouts", "crashes", "canceled", "spawned", "recycled", "rejected_busy"),
            0,
        )

    def size(self) -> int:
        try:
            return max(1, int(self._size()))
        except Exception:  # noqa: BLE001 — 설정 오류가 풀을 멈추지 않게
            return 1

    def run(
        self, target: str, args: tuple = (), kwargs: dict | None = None, *,
        timeout: float | None, cancel: Callable[[], bool] | None = None,
        wait: float | None = None,
    ) -> Any:
        worker = self._checkout(wait=wait, cancel=cancel)
        try:
            return worker.call(
                target, tuple(args), dict(kwargs or {}), timeout=timeout, cancel=cancel,
            )
        except PdfWorkerTimeout:
            self._count("timeouts")
            logger.warning("PDF 워커 %s: %s 작업이 시간 상한(%ss)을 넘어 종료", worker.name,
                           target, timeout)
            raise
        except PdfWorkerCrashed as error:
            self._count("crashes")
            logger.warning(
                "PDF 워커 %s가 %s 작업 중 비정상 종료 (exit %s%s)", worker.name, target,
                error.exitcode, "" if worker.completed else " — 첫 작업 전: 임포트 실패일 수 있음",
            )
            raise
        except PdfWorkerCanceled:
            self._count("canceled")
            raise
        finally:
            self._count("tasks")
            self._checkin(worker)

    def _count(self, key: str) -> None:
        with self._cond:
            self.counters[key] = self.counters.get(key, 0) + 1

    def _checkout(
        self, *, wait: float | None, cancel: Callable[[], bool] | None,
    ) -> _WorkerProcess:
        started = time.monotonic()
        deadline = None if wait is None else started + wait
        with self._cond:
            while True:
                if self._closed:
                    raise PdfWorkerError(f"PDF 워커 풀({self.name})이 종료됐습니다")
                while self._idle:
                    worker = self._idle.pop()
                    if worker.alive():
                        self._checked_out += 1
                        return worker
                    self._all.discard(worker)  # 쉬는 동안 죽었다(SIGALRM·OOM 등)
                    worker.close()
                if self._checked_out < self.size():
                    self._checked_out += 1
                    break
                if cancel is not None and cancel():
                    raise PdfWorkerCanceled()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self.counters["rejected_busy"] += 1
                    raise PdfWorkerBusy(self.name, time.monotonic() - started)
                self._cond.wait(_POLL_S if remaining is None else min(_POLL_S, remaining))
        try:
            worker = _WorkerProcess(self.name, next(self._index))
        except BaseException:
            with self._cond:
                self._checked_out -= 1
                self._cond.notify()
            raise
        with self._cond:
            self.counters["spawned"] += 1
            if self._closed:
                self._checked_out -= 1
                self._cond.notify()
                closed = True
            else:
                self._all.add(worker)
                closed = False
        if closed:
            worker.kill()
            raise PdfWorkerError(f"PDF 워커 풀({self.name})이 종료됐습니다")
        return worker

    def _checkin(self, worker: _WorkerProcess) -> None:
        retire = False
        with self._cond:
            self._checked_out -= 1
            alive = worker.alive()
            keep = (
                not self._closed
                and alive
                and worker in self._all
                and worker.tasks < _MAX_TASKS_PER_WORKER
                and worker.peak_rss < _RECYCLE_RSS_BYTES
                and len(self._idle) + self._checked_out < self.size()
            )
            if keep:
                self._idle.append(worker)
            else:
                self._all.discard(worker)
                retire = True
                if alive:
                    self.counters["recycled"] += 1
            self._cond.notify()
        if retire:
            if alive:
                worker.stop(_KILL_GRACE_S)
            else:
                worker.close()

    def shutdown(self, timeout: float = 2.0) -> None:
        """풀을 닫는다 — 쉬는 워커는 정상 종료, 작업 중인 워커는 종료(그 호출은 Crashed)."""
        with self._cond:
            self._closed = True
            idle = list(self._idle)
            busy = [w for w in self._all if w not in idle]
            self._idle.clear()
            self._all.clear()
            self._cond.notify_all()
        for worker in idle:
            worker.stop(timeout)
        for worker in busy:
            worker.kill()

    def stats(self) -> dict:
        with self._cond:
            return {
                "size": self.size(),
                "workers": len(self._all),
                "in_use": self._checked_out,
                **self.counters,
            }


_POOLS: dict[str, WorkerPool] = {}
_POOLS_GUARD = threading.Lock()


def get_pool(name: str) -> WorkerPool:
    size = _POOL_SIZES.get(name)
    if size is None:
        raise ValueError(f"알 수 없는 PDF 워커 풀: {name!r}")
    with _POOLS_GUARD:
        pool = _POOLS.get(name)
        if pool is None:
            pool = _POOLS[name] = WorkerPool(name, size)
            created = True
        else:
            created = False
    if created:
        _sweep_orphan_scratch()
    return pool


def shutdown_pools(timeout: float = 2.0) -> None:
    """모든 풀을 닫는다(lifespan 종료·atexit). 다음 호출은 새 풀을 lazily 만든다."""
    with _POOLS_GUARD:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        try:
            pool.shutdown(timeout)
        except Exception:  # noqa: BLE001 — 종료 경로는 끝까지 간다
            logger.warning("PDF 워커 풀 종료 실패: %s", pool.name, exc_info=True)


atexit.register(shutdown_pools)


def pool_stats() -> dict:
    """/api/health용 요약 — 모드와 풀별 워커 수·작업·상한 초과·비정상 종료 횟수."""
    with _POOLS_GUARD:
        pools = dict(_POOLS)
    return {"mode": mode(), "pools": {name: pool.stats() for name, pool in pools.items()}}


# ── 공개 실행 API ──────────────────────────────────────────────────────────


def run(
    target: str,
    args: tuple = (),
    kwargs: dict | None = None,
    *,
    pool: str | None = None,
    timeout: float | None,
    cancel: Callable[[], bool] | None = None,
    cancel_kwarg: str | None = None,
    wait: float | None = None,
) -> Any:
    """target('모듈:함수')을 격리 실행하고 결과를 돌려준다.

    process 모드: 풀(pool, 없으면 current_pool())의 워커에서 실행한다. timeout을 넘으면
    PdfWorkerTimeout, 워커가 죽으면 PdfWorkerCrashed, cancel()이 참이 되면 PdfWorkerCanceled
    (어느 경우든 그 워커는 종료되고 다음 호출이 새로 띄운다). wait 안에 빈 워커가 없으면
    PdfWorkerBusy. 작업이 낸 예외는 그대로(피클 불가면 PdfWorkerRemoteError로) 다시 던진다.

    inline 모드: 호출 스레드에서 바로 실행한다(상한·격리 없음). cancel_kwarg를 주면 cancel
    콜백을 그 이름의 키워드 인자로 넘긴다 — 작업이 자기 루프 안에서 취소를 확인하는 경우
    (예: 충실도 분석의 벡터 순회) 프로세스 모드에서는 부모가 워커를 끝내 같은 효과를 낸다.
    """
    kwargs = dict(kwargs or {})
    if mode() == MODE_INLINE:
        if cancel_kwarg and cancel is not None:
            kwargs[cancel_kwarg] = cancel
        return resolve(target)(*args, **kwargs)
    return get_pool(pool or current_pool()).run(
        target, args, kwargs, timeout=timeout, cancel=cancel, wait=wait,
    )


# ── 페이지 격리 메모 ───────────────────────────────────────────────────────

_QUARANTINE: OrderedDict[tuple, set[int]] = OrderedDict()
_QUARANTINE_GUARD = threading.Lock()


def is_quarantined(pdf_path, page_index: int) -> bool:
    key = _doc_key(pdf_path)
    if key is None:
        return False
    with _QUARANTINE_GUARD:
        return page_index in _QUARANTINE.get(key, ())


def quarantine(pdf_path, page_index: int) -> None:
    key = _doc_key(pdf_path)
    if key is None:
        return
    with _QUARANTINE_GUARD:
        pages = _QUARANTINE.pop(key, set())
        pages.add(page_index)
        _QUARANTINE[key] = pages
        while len(_QUARANTINE) > _QUARANTINE_MAX_DOCS:
            _QUARANTINE.popitem(last=False)


def run_page(
    target: str,
    pdf_path,
    page_index: int,
    args: tuple = (),
    kwargs: dict | None = None,
    *,
    pool: str | None = None,
    timeout: float | None = None,
    cancel: Callable[[], bool] | None = None,
    cancel_kwarg: str | None = None,
) -> Any:
    """페이지 한 장 작업 — target(pdf_path, page_index, *args, **kwargs).

    timeout을 주지 않으면 PDF_PAGE_TIMEOUT_S. 앞서 상한을 넘거나 워커를 죽인 (파일, 페이지)는
    다시 돌리지 않고 PdfPageQuarantined(PdfWorkerTimeout의 하위)를 바로 던진다.
    """
    if is_quarantined(pdf_path, page_index):
        raise PdfPageQuarantined(page_index)
    try:
        return run(
            target, (pdf_path, page_index, *args), kwargs, pool=pool,
            timeout=page_timeout() if timeout is None else timeout,
            cancel=cancel, cancel_kwarg=cancel_kwarg,
        )
    except (PdfWorkerTimeout, PdfWorkerCrashed):
        quarantine(pdf_path, page_index)
        raise
