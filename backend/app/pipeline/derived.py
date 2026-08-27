"""파생 산출물 보장(ensure) 계층.

라우트는 "이 잡·이 언어의 번역 PDF/대조 PDF/facsimile PNG가 최신 상태로 존재함"만
요구한다. 그 요구를 만족시키는 캐시 세대 판정·잠금·원자적 교체·정리는 전부 여기가
소유한다. HTTP를 모르며(HTTPException을 던지지 않는다) 실패는 파이프라인 예외
(PdfExportError) 또는 PdfExportBusyError로만 표현한다.

빌더(build_translated_pdf·build_dual_pdf·render_pdf_pages)는 호출 시 주입할 수 있다 —
기본값은 파이프라인 구현이고, 라우트 계층은 자기 모듈 전역을 넘겨 테스트가 그
지점을 교체할 수 있게 한다.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from . import artifacts
from .pdf import render_pdf_pages
from .pdf_export import (
    PDF_EXPORT_FORMAT_VERSION,
    PdfExportError,
    build_dual_pdf,
    build_translated_pdf,
)

logger = logging.getLogger(__name__)

# facsimile 래스터는 잡 단위로 직렬화한다. 전역 락이면 200페이지 잡 하나의 첫
# 렌더가 **다른 잡의** /layout·/page까지 수 분간 막았다. 락 범위에는 export PDF
# 빌드도 포함해, 같은 잡의 동시 첫 진입이 같은 PDF를 두 번 만들지 않게 한다.
_FACSIMILE_LOCKS: dict[str, threading.RLock] = {}
_FACSIMILE_LOCKS_GUARD = threading.Lock()
# 사용 중인 락은 절대 버리지 않는다 — 쓰는 도중 항목이 사라지면 새 락이 생겨
# 상호배제가 깨진다(같은 PDF 중복 빌드). job_id → 현재 진입 수.
_JOB_LOCK_REFS: dict[str, int] = {}
_FACSIMILE_LOCKS_MAX = 512
# (job_id, lang) → 마지막으로 검증에 성공한 (signature, marker 지문, 검증 시각).
# 페이지 이미지 요청마다 전 페이지를 stat하지 않기 위한 프로세스 내 메모 — marker가
# 바뀌거나 사라지면 곧바로 전체 검증으로 되돌아간다(디스크가 진실의 원천).
# TTL을 두어 marker는 그대로인데 PNG만 사라진 경우(외부 정리·부분 유실)에도
# 프로세스 수명 내내 굳지 않고 스스로 재검증·재생성한다.
_FACSIMILE_VERIFIED: dict[tuple[str, str], tuple] = {}
_FACSIMILE_VERIFIED_GUARD = threading.Lock()
_FACSIMILE_VERIFIED_MAX = 512
_FACSIMILE_MEMO_TTL_S = 60.0
# 강제 종료로 남은 staging 임시 디렉터리를 다음 렌더 때 청소한다 — 어느 경로에서도
# 지워지지 않아 영구 잔존하던 것. 진행 중인 렌더와 겹치지 않게 충분히 오래된 것만.
_STAGING_STALE_S = 3600.0


# ── 내보내기 빌드 전역 동시성 상한 ────────────────────────────────────────
# 잡 단위 락은 **같은 잡**의 중복 빌드만 막는다. 서로 다른 잡 N개가 동시에 요청되면
# N개 빌드가 함께 돈다(실측 ~11s/17p, CPU 포화 — 폰트 서브셋·리댁션 청킹 전에는
# 같은 문서가 ~43s였다). PDF_EXPORT_FORMAT_VERSION이 오르면 기존 배포의 전 캐시가
# 한꺼번에 무효화되므로 업그레이드 직후 이 폭주가 실제로 일어난다 — 전역 상한을
# 두고 대기가 길어지면 매달리는 대신 거절한다.
PDF_EXPORT_MAX_CONCURRENT_ENV = "PDF_EXPORT_MAX_CONCURRENT"
PDF_EXPORT_QUEUE_TIMEOUT_ENV = "PDF_EXPORT_QUEUE_TIMEOUT_S"
# 기본값: 빌드는 단일 스레드 CPU 작업이라 2면 코어를 놀리지 않으면서도 폭주는 막는다.
# 0 이하로 두면 상한 비활성(예전 동작).
_PDF_EXPORT_MAX_CONCURRENT_DEFAULT = 2
# 한 건이 ~11s(17페이지 실측)이므로 30s면 앞선 두어 건은 기다려 주고, 그보다
# 길면 재시도가 낫다. 캐시 적중은 이제 락을 기다리지 않으므로(_ensure_translated_pdf)
# 이 예산에 걸리는 것은 진짜로 빌드를 기다리는 요청뿐이다.
_PDF_EXPORT_QUEUE_TIMEOUT_DEFAULT = 30.0
# 같은 (job, lang) 예열이 도는 동안 들어온 **사용자 클릭**의 대기 상한. 일반
# 대기열 상한(30s)과 분리한다: 예열은 사용자가 원하는 바로 그 PDF를 만드는
# 중이므로 기다림이 곧 진행이고, 끝나면 캐시 적중이다. 30s로 묶으면 46쪽
# 문서(실측 75s)에서 번역 직후 다운로드가 항상 503이 된다 — 실제로 그랬다.
PDF_EXPORT_WARM_WAIT_ENV = "PDF_EXPORT_WARM_WAIT_S"
_PDF_EXPORT_WARM_WAIT_DEFAULT = 180.0

_PDF_EXPORT_SLOTS: threading.BoundedSemaphore | None = None
_PDF_EXPORT_SLOTS_SIZE = 0
_PDF_EXPORT_SLOTS_GUARD = threading.Lock()


class PdfExportBusyError(Exception):
    """내보내기 빌드 슬롯을 제한 시간 안에 얻지 못했다.

    PdfExportError(입력 누락·손상 = 잡 상태 문제)와 **구분되는** 일시적 과부하다.
    라우트는 이것을 503 + Retry-After로 옮긴다 — 재시도하면 성공할 수 있다.
    """

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            "PDF 내보내기 대기열이 가득 찼습니다 — 잠시 후 다시 시도하세요"
        )
        self.retry_after = retry_after


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s 값이 정수가 아닙니다 (%r) — 기본값 %d 사용", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s 값이 숫자가 아닙니다 (%r) — 기본값 %s 사용", name, raw, default)
        return default


# ── 요청 단위 대기 예산 ───────────────────────────────────────────────────
# 전역 슬롯에만 타임아웃이 있고 **잡 락에는 없어서**, 같은 잡에 동시 K건이 들어오면
# 각 요청이 앞 요청의 타임아웃을 차례로 기다렸다(실측: 상한 0.5s에 6건 →
# 0.51/1.02/…/3.06s로 정확히 N배 누적, 전부 503). 그 동안 sync 라우트용 스레드풀
# 토큰을 물고 있어 다른 라우트까지 굶는다. 잡 락과 슬롯을 **하나의 예산**으로 묶어
# 총 대기가 요청당 상한을 넘지 않게 한다. 예산은 실제 '대기'에서만 깎이고 빌드·렌더
# 시간은 깎지 않는다 — 느린 빌드가 뒤따르는 획득을 굶기지 않게.
_EXPORT_WAIT = threading.local()


def _export_queue_timeout() -> float:
    return _env_float(PDF_EXPORT_QUEUE_TIMEOUT_ENV, _PDF_EXPORT_QUEUE_TIMEOUT_DEFAULT)


def _warm_wait_timeout() -> float:
    return _env_float(PDF_EXPORT_WARM_WAIT_ENV, _PDF_EXPORT_WARM_WAIT_DEFAULT)


def _warm_inflight(job_id: str, lang: str) -> bool:
    with _WARM_GUARD:
        return (job_id, lang) in _WARM_INFLIGHT


def _busy_error() -> PdfExportBusyError:
    return PdfExportBusyError(retry_after=max(1, int(_export_queue_timeout())))


@contextlib.contextmanager
def export_wait_budget(seconds: float | None = None):
    """이 스레드의 대기 예산을 연다 — 이미 열려 있으면 그대로 공유한다(중첩 안전).

    라우트 하나가 여러 ensure를 연달아 호출할 때(예: /pdf?view=dual은 단일 PDF +
    대조 PDF) 바깥에서 한 번 열어 두면 그 요청 전체의 대기 상한이 하나로 묶인다.
    """
    owner = getattr(_EXPORT_WAIT, "remaining", None) is None
    if owner:
        _EXPORT_WAIT.remaining = max(
            0.0, _export_queue_timeout() if seconds is None else seconds,
        )
    try:
        yield
    finally:
        if owner:
            _EXPORT_WAIT.remaining = None


def _acquire_within_budget(acquire) -> bool:
    """남은 예산만큼만 기다리고, 실제로 기다린 시간을 예산에서 뺀다.

    예산 밖에서 호출되면(다른 모듈이 직접 쓰는 경우) 예전처럼 무한 대기한다.
    """
    remaining = getattr(_EXPORT_WAIT, "remaining", None)
    if remaining is None:
        return bool(acquire())
    started = time.monotonic()
    try:
        return bool(acquire(timeout=remaining))
    finally:
        _EXPORT_WAIT.remaining = max(0.0, remaining - (time.monotonic() - started))


def _pdf_export_slots() -> threading.BoundedSemaphore | None:
    """전역 빌드 슬롯. 상한 설정이 바뀌면 새 세마포어로 갈아끼운다(운영 중 조정·테스트).

    상한이 0 이하면 None — 상한 없이 예전처럼 동작한다.
    """
    global _PDF_EXPORT_SLOTS, _PDF_EXPORT_SLOTS_SIZE
    size = _env_int(PDF_EXPORT_MAX_CONCURRENT_ENV, _PDF_EXPORT_MAX_CONCURRENT_DEFAULT)
    with _PDF_EXPORT_SLOTS_GUARD:
        if size <= 0:
            _PDF_EXPORT_SLOTS = None
            _PDF_EXPORT_SLOTS_SIZE = 0
            return None
        if _PDF_EXPORT_SLOTS is None or _PDF_EXPORT_SLOTS_SIZE != size:
            _PDF_EXPORT_SLOTS = threading.BoundedSemaphore(size)
            _PDF_EXPORT_SLOTS_SIZE = size
        return _PDF_EXPORT_SLOTS


@contextlib.contextmanager
def export_build_slot():
    """빌드 한 건 분량의 전역 슬롯을 잡는다 (캐시 적중 경로에서는 잡지 않는다).

    잡 단위 락을 **쥔 채로만** 슬롯을 기다린다. 슬롯 보유자는 자기 잡 락만 쥐고
    있고 다른 잡 락을 기다리지 않으므로 순환 대기가 생기지 않는다.
    """
    slots = _pdf_export_slots()
    if slots is None:
        yield
        return
    # 예산을 body 전체에 걸쳐 열어 둔다 — 중첩된 획득(래스터 슬롯 등)이 같은 상한을 쓴다.
    with export_wait_budget():
        if not _acquire_within_budget(slots.acquire):
            raise _busy_error()
        try:
            yield
        finally:
            with contextlib.suppress(ValueError):
                # 상한 설정이 도중에 바뀌어 세마포어가 교체되면 release가 초과일 수 있다.
                slots.release()


# ── 잡 단위 렌더 락 ───────────────────────────────────────────────────────
def _job_render_lock(job_id: str) -> threading.RLock:
    """잡 단위 렌더 락 — 다른 잡의 레이아웃/페이지 요청을 막지 않는다."""
    with _FACSIMILE_LOCKS_GUARD:
        return _FACSIMILE_LOCKS.setdefault(job_id, threading.RLock())


@contextlib.contextmanager
def _job_render_guard(job_id: str):
    """잡 단위 렌더 락 진입 + 사용 수 추적.

    락 dict은 잡이 늘어날수록 무한히 커진다(TTL GC로 사라진 잡은 DELETE 훅도
    타지 않는다). 상한을 넘으면 **아무도 쓰고 있지 않은** 항목만 버린다 —
    사용 중 항목을 버리면 같은 잡에 락이 둘 생겨 중복 빌드가 되살아난다.

    획득 순서는 잡 락 → 전역 슬롯을 유지하고(순환 대기 없음), 잡 락 대기도 슬롯과
    같은 예산 아래 둔다 — 예전에는 무한 대기라 동시 K건의 대기가 K배로 누적됐다."""
    with _FACSIMILE_LOCKS_GUARD:
        lock = _FACSIMILE_LOCKS.setdefault(job_id, threading.RLock())
        _JOB_LOCK_REFS[job_id] = _JOB_LOCK_REFS.get(job_id, 0) + 1
    try:
        with export_wait_budget():
            if not _acquire_within_budget(lock.acquire):
                raise _busy_error()
            try:
                yield
            finally:
                lock.release()
    finally:
        with _FACSIMILE_LOCKS_GUARD:
            remaining = _JOB_LOCK_REFS.get(job_id, 1) - 1
            if remaining > 0:
                _JOB_LOCK_REFS[job_id] = remaining
            else:
                _JOB_LOCK_REFS.pop(job_id, None)
            _evict_idle_job_locks()


def _evict_idle_job_locks() -> None:
    """_FACSIMILE_LOCKS_GUARD를 쥔 채로만 호출한다."""
    if len(_FACSIMILE_LOCKS) <= _FACSIMILE_LOCKS_MAX:
        return
    for old in list(_FACSIMILE_LOCKS):
        if len(_FACSIMILE_LOCKS) <= _FACSIMILE_LOCKS_MAX:
            return
        if not _JOB_LOCK_REFS.get(old):
            _FACSIMILE_LOCKS.pop(old, None)


# ── facsimile 검증 메모 ───────────────────────────────────────────────────
def _forget_job_caches(job_id: str) -> None:
    """잡 삭제 시 프로세스 내 파생 캐시(락·검증 메모)를 함께 버린다."""
    with _FACSIMILE_LOCKS_GUARD:
        if not _JOB_LOCK_REFS.get(job_id):
            _FACSIMILE_LOCKS.pop(job_id, None)
    # 락 없이 순회하면 다른 잡의 facsimile 준비가 같은 dict에 키를 넣는 순간
    # "dictionary changed size during iteration"으로 DELETE가 500이 된다.
    with _FACSIMILE_VERIFIED_GUARD:
        for key in [k for k in _FACSIMILE_VERIFIED if k[0] == job_id]:
            _FACSIMILE_VERIFIED.pop(key, None)


def _forget_facsimile_memo(job_id: str, lang: str | None) -> None:
    """한 (잡, 언어)의 검증 메모만 버린다 — 디스크가 메모와 어긋난 것을 확인했을 때."""
    if lang is None:
        return
    with _FACSIMILE_VERIFIED_GUARD:
        _FACSIMILE_VERIFIED.pop((job_id, lang), None)


def _facsimile_memo_get(key: tuple[str, str]) -> tuple | None:
    """TTL 안의 메모만 돌려준다(만료면 None → 전체 재검증)."""
    with _FACSIMILE_VERIFIED_GUARD:
        entry = _FACSIMILE_VERIFIED.get(key)
    if entry is None or len(entry) != 3:
        return None
    signature, marker_id, verified_at = entry
    if time.monotonic() - verified_at >= _FACSIMILE_MEMO_TTL_S:
        return None
    return signature, marker_id


def _facsimile_memo_set(key: tuple[str, str], signature: dict, marker_id: tuple) -> None:
    """메모 갱신 + 상한 유지 — 잡 삭제 훅만으로는 GC로 사라진 잡의 항목이 남는다."""
    with _FACSIMILE_VERIFIED_GUARD:
        if key not in _FACSIMILE_VERIFIED and len(_FACSIMILE_VERIFIED) >= _FACSIMILE_VERIFIED_MAX:
            # 삽입 순서 = dict 순서 — 가장 오래된 항목부터 버린다.
            for old in list(_FACSIMILE_VERIFIED)[
                : len(_FACSIMILE_VERIFIED) - _FACSIMILE_VERIFIED_MAX + 1
            ]:
                _FACSIMILE_VERIFIED.pop(old, None)
        _FACSIMILE_VERIFIED[key] = (signature, marker_id, time.monotonic())


# ── 폰트 설정 정체성 ──────────────────────────────────────────────────────
def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def _load_pdf_export_report(job, lang: str) -> dict:
    try:
        loaded = json.loads(
            artifacts.export_report(job.dir, lang).read_text(encoding="utf-8")
        )
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _pdf_export_font_id(settings) -> str:
    """PDF_EXPORT_FONT 설정의 정체성. 경로뿐 아니라 크기·mtime까지 넣어, 같은 경로에
    다른 폰트를 덮어써도 export.{lang}.pdf 캐시가 무효화되게 한다."""
    raw = (settings.pdf_export_font or "").strip()
    if not raw:
        return "auto"
    try:
        stat = Path(raw).stat()
    except OSError:
        return f"{raw}:missing"
    return f"{raw}:{stat.st_size}:{stat.st_mtime_ns}"


def _font_marker_path(job, lang: str) -> Path:
    return artifacts.export_font_marker(job.dir, lang)


def _write_pdf_export_font_id(job, lang: str, font_id: str) -> None:
    """어떤 폰트 설정으로 만든 PDF인지 원자적으로 남긴다(리포트는 파이프라인 소유라
    여기서 건드리지 않는다). 기록 실패는 다음 요청의 재빌드로만 이어진다."""
    marker = _font_marker_path(job, lang)
    tmp = artifacts.export_font_marker_tmp(job.dir, lang)
    try:
        tmp.write_text(font_id, encoding="utf-8")
        os.replace(tmp, marker)
    except OSError:
        logger.warning("PDF 폰트 표식 저장 실패: %s", marker.name)
    finally:
        tmp.unlink(missing_ok=True)


# ── 파생 산출물 보장 ──────────────────────────────────────────────────────
def _translated_pdf_cache(job, lang: str, font_id: str) -> tuple[bool, Path, dict]:
    """(캐시가 최신인가, 산출물 경로, 리포트).

    모든 입력은 원자적 교체(os.replace)로만 갱신되므로 락 없이 읽어도 반쪽짜리
    파일을 보지 않는다. 최악의 경우 빌드 직후 리포트·폰트 표식이 아직 안 써진
    찰나를 봐서 '낡음'으로 판정하는데, 그러면 락을 잡고 다시 확인하게 되므로
    안전한 방향의 오판이다.
    """
    out = artifacts.export_pdf(job.dir, lang)
    report = _load_pdf_export_report(job, lang)
    try:
        latest_input = max(
            artifacts.source_pdf(job.dir).stat().st_mtime_ns,
            artifacts.layout(job.dir).stat().st_mtime_ns,
            artifacts.layout(job.dir, lang).stat().st_mtime_ns,
        )
        current = (
            out.is_file()
            and out.stat().st_mtime_ns >= latest_input
            and report.get("format_version") == PDF_EXPORT_FORMAT_VERSION
            # 폰트는 입력 파일이 아니라 설정이라 mtime 비교로는 잡히지 않는다 —
            # PDF_EXPORT_FONT를 바꾸면 예전 폰트로 조판된 캐시가 계속 나갔다.
            and _read_text_or_none(_font_marker_path(job, lang)) == font_id
        )
    except OSError:
        # build_translated_pdf가 누락 입력을 사용자용 PdfExportError로 변환한다.
        current = False
    return current, out, report


def _ensure_translated_pdf(job, lang: str, settings, *, build=build_translated_pdf):
    """번역 레이아웃과 같은 세대의 PDF를 만들거나 캐시에서 돌려준다.

    빌드는 잡 단위 락 안에서 한다 — /pdf·/layout·/page가 동시에 첫 진입하면 같은
    export.{lang}.pdf를 여러 번 만들게 된다(수십 초짜리 작업).

    **캐시 적중 판정은 락 밖에서 먼저** 한다. 예전에는 락을 잡은 뒤에 판정해서,
    이미 만들어져 있는 PDF를 받으러 온 요청이 진행 중인 빌드 뒤에 줄을 섰다 —
    대기 예산(기본 30s)을 다 쓰면 **웜 캐시인데도 503**이 나갔다(재현: 리더가
    연 빌드가 락을 40초 쥔 사이 들어온 다운로드 클릭이 30.1초 뒤 503). 락을
    잡은 뒤 한 번 더 확인해 중복 빌드는 그대로 막는다(double-checked).
    """
    font_id = _pdf_export_font_id(settings)
    current, out, report = _translated_pdf_cache(job, lang, font_id)
    if current:
        return out, report
    # 캐시가 없고 같은 잡의 예열이 돌고 있으면, 그 예열이 곧 이 요청의 답이다.
    # 일반 대기열 상한 대신 예열 대기 상한을 연다(중첩 안전 — 예열 스레드 자신은
    # 이미 예산 0을 열어 둔 상태라 여기서 덮어쓰지 않는다).
    budget = _warm_wait_timeout() if _warm_inflight(job.id, lang) else None
    with export_wait_budget(budget), _job_render_guard(job.id):
        # 락을 기다리는 사이 다른 스레드가 같은 PDF를 완성했을 수 있다.
        current, out, report = _translated_pdf_cache(job, lang, font_id)
        if current:
            return out, report
        with export_build_slot():
            built = build(job.dir, lang, fontfile=settings.pdf_export_font)
        _write_pdf_export_font_id(job, lang, font_id)
        return built.path, built.report()


def _dual_pdf_cache_current(source_pdf: Path, translated_pdf: Path, out: Path) -> bool:
    """대조 PDF가 두 입력보다 새것인가. 판정 불가(입력 누락)면 '낡음'."""
    try:
        latest_input = max(
            source_pdf.stat().st_mtime_ns,
            translated_pdf.stat().st_mtime_ns,
        )
    except OSError:
        # build_dual_pdf가 누락 파일을 사용자에게 읽을 수 있는 PdfExportError로 바꾼다.
        return False
    return out.is_file() and out.stat().st_mtime_ns >= latest_input


def _ensure_dual_pdf(job, lang: str, translated_pdf: Path, *, build=build_dual_pdf) -> Path:
    """원본·번역 대조 PDF 캐시를 번역 단일 PDF와 같은 세대로 유지한다.

    빌드는 단일 PDF와 같은 잡 단위 락 안에서 한다 — 락 밖이면 프런트 기본
    다운로드 경로(view=dual)의 동시 첫 요청이 같은 대조 PDF를 중복으로 만든다.
    캐시 적중 판정은 `_ensure_translated_pdf`와 같은 이유로 락 밖에서 먼저 한다.
    """
    source_pdf = artifacts.source_pdf(job.dir)
    out = artifacts.export_dual_pdf(job.dir, lang)
    if _dual_pdf_cache_current(source_pdf, translated_pdf, out):
        return out
    with _job_render_guard(job.id):
        if _dual_pdf_cache_current(source_pdf, translated_pdf, out):
            return out
        with export_build_slot():
            return build(source_pdf, translated_pdf, out)


def _page_numbers(pages: list) -> list[int]:
    """layout 페이지 목록 → 렌더 파일명에 쓰는 페이지 번호(누락 시 순서 폴백)."""
    return [
        int(page.get("page", index)) if isinstance(page, dict) else index
        for index, page in enumerate(pages, start=1)
    ]


def _sweep_stale_staging(rendered_dir: Path, lang: str) -> int:
    """강제 종료(SIGKILL·정전)로 남은 staging 임시 디렉터리를 청소한다.

    정상 경로는 finally에서 지우므로 여기 걸리는 것은 프로세스가 죽어 주인이 없는
    잔해뿐이다. 잡 단위 락 안에서 호출하지만, 데이터 디렉터리를 공유하는 다른
    프로세스가 렌더 중일 가능성까지 감안해 충분히 오래된 것만 지운다. 삭제 수 반환."""
    now = time.time()
    removed = 0
    try:
        entries = list(rendered_dir.glob(artifacts.facsimile_staging_glob(lang)))
    except OSError:
        return 0
    for entry in entries:
        try:
            if not entry.is_dir() or now - entry.stat().st_mtime < _STAGING_STALE_S:
                continue
        except OSError:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    if removed:
        logger.info("고아 facsimile staging %d개 정리: %s", removed, rendered_dir)
    return removed


def _ensure_facsimile_pages(
    job, page_numbers: list[int], lang: str | None, settings,
    *, render=render_pdf_pages, build=build_translated_pdf,
) -> Path:
    """HTML의 시각 기준면이 될 페이지 PNG 디렉터리를 보장한다.

    원문은 OCR 입력에 쓰인 pages/를 그대로 재사용한다. 번역본은 동일한
    export.{lang}.pdf를 job.dpi로 렌더해, HTML과 다운로드 PDF가 픽셀 수준에서
    같은 결과를 보게 한다. marker는 PDF 크기·mtime·DPI·페이지 수를 포함한다.
    """
    if lang is None:
        pages_dir = artifacts.pages_dir(job.dir)
        if not pages_dir.is_dir():
            raise PdfExportError("원본 페이지 이미지가 없습니다")
        return pages_dir

    target = artifacts.rendered_dir(job.dir, lang)
    marker = artifacts.facsimile_marker(job.dir, lang)
    # PDF 빌드까지 락 안에서 수행한다 — 락 밖이면 같은 잡의 동시 첫 진입이 같은
    # export.{lang}.pdf를 중복으로 만든다.
    with _job_render_guard(job.id):
        pdf_path, _report = _ensure_translated_pdf(job, lang, settings, build=build)
        pdf_stat = pdf_path.stat()
        signature = {
            "pdf_size": pdf_stat.st_size,
            "pdf_mtime_ns": pdf_stat.st_mtime_ns,
            "dpi": int(job.dpi),
            "pages": len(page_numbers),
        }

        def _cache_valid() -> bool:
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if saved != signature:
                return False
            expected = [artifacts.page_image(target, number) for number in page_numbers]
            return bool(expected) and all(path.is_file() for path in expected)

        def _marker_id() -> tuple:
            try:
                stat = marker.stat()
            except OSError:
                return ()
            return (stat.st_size, stat.st_mtime_ns)

        memo_key = (job.id, lang)
        # 같은 세대를 이 프로세스에서 이미 검증했으면 전 페이지 stat을 건너뛴다.
        # marker 지문까지 함께 비교하므로, 캐시 무효화(marker 삭제/재기록)는 메모리
        # 메모를 우회하지 못한다 — 디스크가 여전히 진실의 원천이다. 다만 marker는
        # 그대로인데 PNG만 사라지는 경우가 있어 메모에는 TTL이 있다(자가 복구).
        marker_id = _marker_id()
        if marker_id and _facsimile_memo_get(memo_key) == (signature, marker_id):
            return target
        if _cache_valid():
            _facsimile_memo_set(memo_key, signature, _marker_id())
            return target
        target.mkdir(parents=True, exist_ok=True)
        _sweep_stale_staging(artifacts.rendered_root(job.dir), lang)
        # 재생성은 파일 단위로 원자적이어야 한다. 예전에는 기존 PNG를 먼저 지우고
        # 같은 자리에 다시 렌더해, 그 사이 /files 요청이 404나 반쯤 쓰인 이미지를
        # 받았다. 임시 디렉터리에 렌더한 뒤 os.replace로 갈아끼운다.
        staging = artifacts.facsimile_staging(job.dir, lang)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            # 래스터도 빌드와 같은 전역 상한 아래에 둔다 — 리더 기본 경로는
            # 빌드 1회 + 전 페이지 래스터 1회인데 예전에는 앞의 절반만 상한을
            # 받아, 상한 1인데도 서로 다른 잡 4개의 래스터가 함께 돌았다(실측).
            with export_build_slot():
                render(
                    pdf_path,
                    staging,
                    dpi=int(job.dpi),
                    max_pages=settings.max_pages,
                )
            fresh = sorted(staging.glob("page_*.png"))
            for path in fresh:
                os.replace(path, target / path.name)
            keep = {path.name for path in fresh}
            for old in target.glob("page_*.png"):
                if old.name not in keep:
                    old.unlink(missing_ok=True)
            tmp = artifacts.facsimile_marker_tmp(job.dir, lang)
            try:
                tmp.write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
                os.replace(tmp, marker)
            finally:
                tmp.unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        _facsimile_memo_set(memo_key, signature, _marker_id())
    return target


def _try_facsimile_pages(
    job, pages: list, lang: str | None, settings,
    *, render=render_pdf_pages, build=build_translated_pdf,
) -> Path | None:
    """레이아웃 HTML은 PDF export 결함 때문에 완전히 사라지지 않도록 폴백한다.

    과부하(PdfExportBusyError)는 여기서 삼키지 않는다 — 조용히 저품질 렌더로
    떨어지는 대신 라우트가 503으로 알리고 재시도하게 한다.
    """
    try:
        return _ensure_facsimile_pages(
            job, _page_numbers(pages), lang, settings, render=render, build=build,
        )
    except (PdfExportError, OSError, ValueError):
        logger.exception(
            "facsimile 페이지 준비 실패 — 좌표 텍스트 렌더로 폴백: %s (%s)",
            job.id,
            lang or "orig",
        )
        return None


# ── 백그라운드 캐시 예열 ──────────────────────────────────────────────────
# 번역이 끝나면 export.{lang}.pdf 캐시가 무효화된다(api._run_translate_thread).
# 그런데 사용자가 다운로드 버튼을 누르는 것은 바로 그 직후다 — 버튼이 보이는
# 순간이 캐시가 가장 확실히 비어 있는 순간이라, 첫 클릭이 빌드 전체를 요청
# 안에서 기다린다(실측 ~11–16s, 패치 전 ~43s). 번역 완료 시점에 미리 만들어
# 두면 그 대기가 클릭에서 사라진다.
#
# 예열은 **사용자 요청을 밀어내면 안 된다**. 그래서
#  - 슬롯·락을 즉시 얻지 못하면 조용히 포기한다(대기 예산 0). 클릭이 직접
#    만들면 되고, 예열이 큐를 잡고 있다가 진짜 클릭을 503으로 만들지 않는다.
#  - (job, lang)마다 하나만 돈다. 번역 완료와 폰트 백필이 연달아 예열을
#    부탁해도 빌드는 한 번이다.
_WARM_INFLIGHT: set[tuple[str, str]] = set()
_WARM_GUARD = threading.Lock()


def warm_translated_pdf(job, lang: str, settings, *, build=build_translated_pdf) -> bool:
    """export.{lang}.pdf를 지금 만들어 둔다 — 이미 최신이면 아무 일도 하지 않는다.

    호출 스레드에서 동기로 돈다. 백그라운드 실행은 `warm_translated_pdf_async`.
    실제로 빌드를 돌렸는지 여부와 무관하게, 끝났을 때 캐시가 최신이면 True.
    """
    with export_wait_budget():
        # 예산을 0으로 만들어 어떤 대기도 하지 않게 한다 — 경합하면 포기한다.
        _EXPORT_WAIT.remaining = 0.0
        try:
            _ensure_translated_pdf(job, lang, settings, build=build)
        except PdfExportBusyError:
            logger.debug("PDF 예열 건너뜀(경합): %s/%s", job.id, lang)
            return False
        except (PdfExportError, OSError, ValueError):
            # 예열 실패는 사용자에게 알리지 않는다 — 클릭 시 같은 경로가 다시
            # 시도하고, 그때는 진짜 오류로 보고된다.
            logger.warning("PDF 예열 실패: %s/%s", job.id, lang, exc_info=True)
            return False
    return True


def warm_translated_pdf_async(job, lang: str, settings, *, build=build_translated_pdf) -> bool:
    """`warm_translated_pdf`를 데몬 스레드에서 돌린다. 시작했으면 True.

    같은 (job, lang) 예열이 이미 돌고 있으면 새로 띄우지 않는다.
    """
    key = (job.id, lang)
    with _WARM_GUARD:
        if key in _WARM_INFLIGHT:
            return False
        _WARM_INFLIGHT.add(key)

    def _run() -> None:
        try:
            warm_translated_pdf(job, lang, settings, build=build)
        finally:
            with _WARM_GUARD:
                _WARM_INFLIGHT.discard(key)

    threading.Thread(
        target=_run, name=f"pdf-warm-{job.id}-{lang}", daemon=True,
    ).start()
    return True
