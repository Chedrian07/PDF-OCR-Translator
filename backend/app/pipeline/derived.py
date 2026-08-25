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
# N개 빌드가 함께 돈다(실측 9.4s/16p, CPU 포화). PDF_EXPORT_FORMAT_VERSION이 오르면
# 기존 배포의 전 캐시가 한꺼번에 무효화되므로 업그레이드 직후 이 폭주가 실제로
# 일어난다 — 전역 상한을 두고 대기가 길어지면 매달리는 대신 거절한다.
PDF_EXPORT_MAX_CONCURRENT_ENV = "PDF_EXPORT_MAX_CONCURRENT"
PDF_EXPORT_QUEUE_TIMEOUT_ENV = "PDF_EXPORT_QUEUE_TIMEOUT_S"
# 기본값: 빌드는 단일 스레드 CPU 작업이라 2면 코어를 놀리지 않으면서도 폭주는 막는다.
# 0 이하로 두면 상한 비활성(예전 동작).
_PDF_EXPORT_MAX_CONCURRENT_DEFAULT = 2
# 한 건이 ~10s이므로 30s면 앞선 몇 건은 기다려 주고, 그보다 길면 재시도가 낫다.
_PDF_EXPORT_QUEUE_TIMEOUT_DEFAULT = 30.0

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
    timeout = _env_float(PDF_EXPORT_QUEUE_TIMEOUT_ENV, _PDF_EXPORT_QUEUE_TIMEOUT_DEFAULT)
    if not slots.acquire(timeout=timeout):
        raise PdfExportBusyError(retry_after=max(1, int(timeout)))
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
    사용 중 항목을 버리면 같은 잡에 락이 둘 생겨 중복 빌드가 되살아난다."""
    with _FACSIMILE_LOCKS_GUARD:
        lock = _FACSIMILE_LOCKS.setdefault(job_id, threading.RLock())
        _JOB_LOCK_REFS[job_id] = _JOB_LOCK_REFS.get(job_id, 0) + 1
    try:
        with lock:
            yield
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
def _ensure_translated_pdf(job, lang: str, settings, *, build=build_translated_pdf):
    """번역 레이아웃과 같은 세대의 PDF를 만들거나 캐시에서 돌려준다.

    잡 단위 락 안에서 수행한다 — /pdf·/layout·/page가 동시에 첫 진입하면 같은
    export.{lang}.pdf를 여러 번 만들게 된다(수십 초짜리 작업).
    """
    source_pdf = artifacts.source_pdf(job.dir)
    orig_layout = artifacts.layout(job.dir)
    trans_layout = artifacts.layout(job.dir, lang)
    out = artifacts.export_pdf(job.dir, lang)
    font_id = _pdf_export_font_id(settings)
    with _job_render_guard(job.id):
        report = _load_pdf_export_report(job, lang)
        try:
            latest_input = max(
                source_pdf.stat().st_mtime_ns,
                orig_layout.stat().st_mtime_ns,
                trans_layout.stat().st_mtime_ns,
            )
            cache_current = (
                out.is_file()
                and out.stat().st_mtime_ns >= latest_input
                and report.get("format_version") == PDF_EXPORT_FORMAT_VERSION
                # 폰트는 입력 파일이 아니라 설정이라 mtime 비교로는 잡히지 않는다 —
                # PDF_EXPORT_FONT를 바꾸면 예전 폰트로 조판된 캐시가 계속 나갔다.
                and _read_text_or_none(_font_marker_path(job, lang)) == font_id
            )
        except OSError:
            # build_translated_pdf가 누락 입력을 사용자용 PdfExportError로 변환한다.
            cache_current = False
        if not cache_current:
            with export_build_slot():
                built = build(job.dir, lang, fontfile=settings.pdf_export_font)
            _write_pdf_export_font_id(job, lang, font_id)
            return built.path, built.report()
    return out, report


def _ensure_dual_pdf(job, lang: str, translated_pdf: Path, *, build=build_dual_pdf) -> Path:
    """원본·번역 대조 PDF 캐시를 번역 단일 PDF와 같은 세대로 유지한다.

    단일 PDF와 같은 잡 단위 락 안에서 수행한다 — 락 밖이면 프런트 기본 다운로드
    경로(view=dual)의 동시 첫 요청이 같은 대조 PDF를 중복으로 만든다."""
    source_pdf = artifacts.source_pdf(job.dir)
    out = artifacts.export_dual_pdf(job.dir, lang)
    with _job_render_guard(job.id):
        try:
            latest_input = max(
                source_pdf.stat().st_mtime_ns,
                translated_pdf.stat().st_mtime_ns,
            )
        except OSError:
            # build_dual_pdf가 누락 파일을 사용자에게 읽을 수 있는 PdfExportError로 바꾼다.
            with export_build_slot():
                return build(source_pdf, translated_pdf, out)
        if not out.is_file() or out.stat().st_mtime_ns < latest_input:
            with export_build_slot():
                return build(source_pdf, translated_pdf, out)
    return out


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
