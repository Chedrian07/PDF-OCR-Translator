"""REST + SSE 라우트. 계약: docs/ARCHITECTURE.md §5"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

import anyio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from . import native_ops
from .pipeline.pdf import probe_pdf, render_pdf_pages
from .pipeline.pdf_export import (
    PDF_EXPORT_FORMAT_VERSION,
    PdfExportError,
    build_dual_pdf,
    build_translated_pdf,
)
from .pipeline.layout import (
    render_document_standalone,
    render_layout_html,
    render_layout_standalone,
)
from .pipeline.render import render_document_html, render_markdown_html
# 번역 API 스레드는 완성된 코어를 호출한다. API 레이어 테스트는 상태·SSE 계약을
# 고립시키기 위해 이 심(run_translation)을 페이크로 교체한다.
from .translate import SUPPORTED_LANGS, TranslateConfig, TranslateError, run_translation
from .translate.client import OpenAICompatClient
from .llm import LlmError
from .qa import AskRequest, get_page_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_ALLOWED_FILE_DIRS = ("pages", "images", "layout", "rendered")
_UPLOAD_CHUNK = 1024 * 1024
_PREVIEW_MAX_BYTES = 2_000_000

# facsimile 래스터는 잡 단위로 직렬화한다. 전역 락이면 200페이지 잡 하나의 첫
# 렌더가 **다른 잡의** /layout·/page까지 수 분간 막았다. 락 범위에는 export PDF
# 빌드도 포함해, 같은 잡의 동시 첫 진입이 같은 PDF를 두 번 만들지 않게 한다.
_FACSIMILE_LOCKS: dict[str, threading.RLock] = {}
_FACSIMILE_LOCKS_GUARD = threading.Lock()
# (job_id, lang) → 마지막으로 검증에 성공한 (signature, marker 지문). 페이지 이미지
# 요청마다 전 페이지를 stat하지 않기 위한 프로세스 내 메모 — marker가 바뀌거나
# 사라지면 곧바로 전체 검증으로 되돌아간다(디스크가 진실의 원천).
_FACSIMILE_VERIFIED: dict[tuple[str, str], tuple] = {}
# layout.json 페이지 번호 캐시 — /page/{n}은 유효성 검사만 필요한데 매 요청 전체
# JSON을 재파싱했다. 키에 크기·mtime을 넣어 갱신 시 자동 무효화된다.
_LAYOUT_PAGE_NUMBERS: dict[tuple[str, int, int], tuple[int, ...]] = {}
_LAYOUT_PAGE_NUMBERS_GUARD = threading.Lock()
_LAYOUT_PAGE_NUMBERS_MAX = 256

# SSE 구독 루프 전용 스레드 예산 — 각 연결이 blocking 폴(_sse_poll, 최대 1초)로
# 스레드 하나를 상시 점유하므로, 공용 AnyIO 풀(기본 40 토큰 — sync 라우트·업로드
# probe·프리뷰 렌더가 공유)과 분리해 SSE 다중 접속이 나머지 요청을 굶기지 않게
# 한다. 64 = 로컬 단일 사용자 도구 기준 동시 스트림(잡+번역 탭) 상한으로 충분히
# 크고, 초과 연결은 끊기지 않고 다음 폴 차례만 대기한다.
_SSE_LIMITER = anyio.CapacityLimiter(64)

# ── 남용 방어 (인증 없는 서비스의 비용 상한) ──────────────────────────────
# 이 서비스는 인증·토큰이 없고 기본 바인딩이 0.0.0.0이다. 같은 네트워크의 누구나
# POST /qa로 운영자의 유료 LLM 키를 무제한 소비하거나 200페이지 번역을 반복
# 트리거할 수 있다(전역 세마포어는 동시 HTTP 슬롯만 제한한다). 인증을 새로 만드는
# 대신 잡·IP 단위 슬라이딩 윈도우와 동시 실행 상한을 두고, 초과분은 429 +
# Retry-After로 거절한다. 기본값은 1인 로컬 사용을 방해하지 않게 넉넉히 잡는다.
_RATE_WINDOW_S = 60.0
_RATE_KEYS_MAX = 4096  # 키 dict 상한 — 초과 시 만료 키를 정리한다
_GUARD_INIT_LOCK = threading.Lock()


class _SlidingWindowLimiter:
    """프로세스 내 슬라이딩 윈도우 레이트리밋 (limit<=0이면 비활성)."""

    def __init__(self, limit: int, window: float = _RATE_WINDOW_S) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """허용되면 None, 상한 초과면 Retry-After(초)를 돌려준다."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > _RATE_KEYS_MAX:
                self._prune(now)
            hits = [t for t in self._hits.get(key, ()) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return max(1.0, self.window - (now - hits[0]))
            hits.append(now)
            self._hits[key] = hits
            return None

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
            self._hits.pop(key, None)


class _AbuseGuard:
    """잡·IP 단위 레이트리밋 + 동시 실행 상한 묶음."""

    def __init__(self, limit_per_min: int, max_concurrent: int) -> None:
        self.limiter = _SlidingWindowLimiter(limit_per_min)
        self.max_concurrent = max_concurrent
        self._slots = (
            threading.BoundedSemaphore(max_concurrent) if max_concurrent > 0 else None
        )

    def check_rate(self, keys) -> None:
        for key in keys:
            retry = self.limiter.hit(key)
            if retry is not None:
                raise HTTPException(
                    429,
                    "요청이 너무 잦습니다 — 잠시 후 다시 시도하세요",
                    headers={"Retry-After": str(max(1, int(retry)))},
                )

    def acquire(self) -> bool:
        return True if self._slots is None else self._slots.acquire(blocking=False)

    def release(self) -> None:
        if self._slots is not None:
            self._slots.release()


def _client_key(request: Request) -> str:
    client = request.client
    return client.host if client is not None else "unknown"


def _abuse_guard(st, name: str) -> _AbuseGuard:
    """앱 상태에 가드를 지연 생성한다 — 라우트 계층이 소유하므로 앱 팩토리는 불변.
    상한은 Settings(QA_RATE_LIMIT_PER_MIN·QA_MAX_CONCURRENT·
    TRANSLATE_RATE_LIMIT_PER_MIN·TRANSLATE_MAX_ACTIVE, 0 이하면 해당 상한 비활성)."""
    guards = getattr(st, "abuse_guards", None)
    if guards is None:
        with _GUARD_INIT_LOCK:
            guards = getattr(st, "abuse_guards", None)
            if guards is None:
                cfg = st.settings
                guards = {
                    "qa": _AbuseGuard(cfg.qa_rate_limit_per_min, cfg.qa_max_concurrent),
                    "translate": _AbuseGuard(
                        cfg.translate_rate_limit_per_min, cfg.translate_max_active
                    ),
                }
                st.abuse_guards = guards
    return guards[name]


def _state(request: Request):
    return request.app.state


def _get_job(request: Request, job_id: str):
    job = _state(request).store.get(job_id)
    if job is None:
        raise HTTPException(404, "잡을 찾을 수 없습니다")
    return job


# ── 번역(한국어) 공용 헬퍼 ────────────────────────────────────────────────
def _check_lang(lang: str) -> None:
    """구체 lang 값 검증 (쿼리에서는 None(원본)을 먼저 걸러낸 뒤 호출)."""
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(400, "지원하지 않는 언어")


def _translate_dir(job, lang: str) -> Path:
    return job.dir / "translations" / lang


def _translate_channel(job_id: str, lang: str) -> str:
    """번역 SSE 브로커 채널 키 — OCR 잡 이벤트(job_id)와 네임스페이스를 분리한다."""
    return f"{job_id}:translate:{lang}"


def _translated_markdown_or_404(job, lang: str) -> str:
    p = job.dir / f"result.{lang}.md"
    if not p.is_file():
        raise HTTPException(404, "한국어 번역본이 없습니다 — 먼저 번역을 실행하세요")
    return p.read_text(encoding="utf-8")


def _read_translate_state(job, lang: str) -> dict | None:
    """translations/{lang}/state.json 로드 (없거나 손상되면 None)."""
    p = _translate_dir(job, lang) / "state.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_translate_state(job, lang: str, state: dict) -> None:
    d = _translate_dir(job, lang)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / ".state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "state.json")


def _stale_adjusted_state(request: Request, job, lang: str) -> dict | None:
    """state.json을 읽되 stale-running을 조정한다: status=="running"인데 레지스트리에
    태스크가 없으면(서버 재시작 등) error로 원자적 재기록 후 반환. 파일이 없으면 None.

    순서 불변식: 레지스트리 확인 → state 읽기 (락 안에서 함께). 스레드는 최종 state를
    기록한 뒤에야 레지스트리에서 빠지므로(_run_translate_thread finally), "없음"을
    관찰한 뒤 읽은 state는 최종본이 보장된다 — read→check 순서면 그 사이 완료된
    번역을 stale로 오판해 done을 error로 덮어쓴다. 락은 새 실행 등록(translate_start)
    과도 직렬화해 "없음 관찰 직후 등록된 새 running"을 error로 덮는 창도 막는다."""
    st = _state(request)
    with st.translate_lock:
        alive = (job.id, lang) in st.translate_tasks
        state = _read_translate_state(job, lang)
        if state is None:
            return None
        if not alive and state.get("status") == "running":
            state["status"] = "error"
            state["error"] = "서버가 재시작되어 번역이 중단되었습니다 — 다시 실행하세요"
            _write_translate_state(job, lang, state)
    return state


def _run_translate_thread(
    st, job, lang: str, cfg: TranslateConfig,
    cancel: threading.Event, force: bool, page_separator: str,
) -> None:
    """번역 워커 스레드 본문. 진행/완료/오류를 브로커 채널로 중계하고 레지스트리를 정리한다.
    state.json은 run_translation(엔진)이 직접 기록하므로 여기서는 이벤트만 발행한다."""
    broker = st.broker
    channel = _translate_channel(job.id, lang)

    def _progress(current: int, total: int) -> None:
        broker.publish(channel, "progress", {
            "phase": "translate", "lang": lang,
            "current": current, "total": total, "status": "running",
        })

    try:
        client = OpenAICompatClient(cfg, request_semaphore=st.translate_api_slots)
        result = run_translation(
            job.dir, lang, cfg,
            page_separator=page_separator,
            progress=_progress,
            cancel=cancel,
            force=force,
            client=client,
        )
        if getattr(result, "status", None) == "canceled":
            broker.publish(channel, "error", {
                "message": "번역이 취소되었습니다", "canceled": True,
            })
        else:
            broker.publish(channel, "done", {
                "phase": "translate", "lang": lang,
                "markdown_url": f"/api/jobs/{job.id}/markdown?lang={lang}",
                "html_url": f"/api/jobs/{job.id}/html?lang={lang}",
                "layout_url": f"/api/jobs/{job.id}/layout?lang={lang}",
                "counts": {
                    "total": getattr(result, "total", 0),
                    "translated": getattr(result, "translated", 0),
                    "cached": getattr(result, "cached", 0),
                    "skipped": getattr(result, "skipped", 0),
                    "kept_original": len(getattr(result, "kept_original", []) or []),
                },
            })
            # ko 번역본을 포함해 다시 만들도록 archive.zip 캐시를 무효화한다.
            (job.dir / "archive.zip").unlink(missing_ok=True)
            # 번역이 갱신됐으므로 내보내기 PDF 캐시도 무효화한다.
            (job.dir / f"export.{lang}.pdf").unlink(missing_ok=True)
            (job.dir / f"export.{lang}.report.json").unlink(missing_ok=True)
            (job.dir / f"export.{lang}.dual.pdf").unlink(missing_ok=True)
            # 번역 PDF에서 만든 HTML 기준면 캐시도 다음 요청에서 다시 렌더한다.
            (job.dir / "rendered" / lang / ".source.json").unlink(missing_ok=True)
    except TranslateError as e:
        # SSE는 구독자가 없으면 이벤트를 버린다 — 서버 로그에도 반드시 남긴다
        logger.exception("번역 실패: %s (lang=%s)", job.id, lang)
        broker.publish(channel, "error", {"message": str(e), "canceled": False})
    except Exception as e:  # noqa: BLE001 — 스레드가 조용히 죽지 않도록 SSE로 중계
        logger.exception("번역 실패: %s (lang=%s)", job.id, lang)
        broker.publish(channel, "error", {"message": str(e), "canceled": False})
    finally:
        with st.translate_lock:
            st.translate_tasks.pop((job.id, lang), None)


def _translate_available() -> bool:
    """POST /translate의 503 판정과 동일 경로(TranslateConfig.from_env) 재사용.
    env 딕셔너리 조회 + 문자열 파싱뿐이라 주기 폴링(health)에도 충분히 가볍다."""
    try:
        TranslateConfig.from_env()
        return True
    except (TranslateError, ValueError):
        # ValueError: 숫자형 env 오타(예: TRANSLATE_CONCURRENCY=abc)로 int()/float() 파싱 실패.
        # health가 500으로 죽지 않도록 '번역 불가'로만 강등한다.
        return False


def _qa_available(st) -> bool:
    """Q&A가 실제로 가능한지 — 기본 프로바이더의 구성 여부로 판정한다.

    openai-*는 LLM_OPENAI_API_KEY 유무(OpenAIClient.configured)로 네트워크 호출 없이
    알 수 있다. ollama는 로컬 데몬 기동 여부를 동기 조회할 수 없어(health는 폴링 대상)
    True로 두고 실시간 가용성은 /api/providers가 알려준다."""
    provider = st.settings.llm_provider
    if provider.startswith("openai-"):
        return bool(getattr(getattr(st.llm_router, "openai", None), "configured", False))
    return True


@router.get("/health")
def health(request: Request) -> dict:
    st = _state(request)
    engine = st.engine
    caps = engine.capabilities()
    # provider_health는 sidecar 엔진만 non-None. sidecar가 죽어 있어도 메인 앱
    # health는 200이어야 한다 — provider 상태는 provider_health.status로 구분.
    try:
        provider = engine.provider_health()
    except Exception as e:  # noqa: BLE001 — health가 500으로 죽지 않게
        provider = {"status": "error", "error": str(e)[:300]}
    return {
        "status": "ok",
        "engine": engine.name,
        "device": engine.device,
        "dtype": engine.dtype_name or st.settings.dtype,
        "model_id": caps.model_id or st.settings.model_id,
        "model_loaded": engine.loaded,
        # 프리로드 실패 후 워커 재시도가 성공하면 과거 오류는 더 이상 유효하지 않다
        "model_load_error": None if engine.loaded else st.load_state.get("error"),
        "gpu_name": engine.gpu_name(),
        "native_ops": native_ops.HAVE_NATIVE,
        # 워커 스레드 생존 여부 — 죽으면 잡이 영원히 queued로 남으므로 운영 가시성 필수
        "worker_alive": st.worker.is_alive(),
        "max_upload_mb": st.settings.max_upload_mb,
        "translate_available": _translate_available(),
        # ── 신규 필드 (추가만 — 기존 필드 의미 불변) ──
        "model_revision": caps.model_revision or None,
        "provider": caps.provider,
        "capabilities": {
            "multi_page_context": caps.supports_multi_page,
            "stream_granularity": caps.stream_granularity,
            "layout": caps.layout_capability,
            "figures": caps.figure_capability,
        },
        "provider_health": provider,
        # ── Localight Q&A 통합 필드 (추가만) ──
        "qa_available": _qa_available(st),
        "llm_default_provider": st.settings.llm_provider,
    }


@router.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("multi"),
    dpi: int | None = Form(None),
) -> dict:
    st = _state(request)
    settings = st.settings

    if mode not in ("multi", "per_page"):
        raise HTTPException(400, "mode는 multi 또는 per_page 여야 합니다")
    dpi = dpi if dpi is not None else settings.render_dpi
    if not (72 <= dpi <= 400):
        raise HTTPException(400, "dpi는 72–400 범위여야 합니다")
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다")

    caps = st.engine.capabilities()
    job = st.store.create(
        filename=filename, mode=mode, dpi=dpi,
        engine_info={
            "engine": st.engine.name,
            "model_id": caps.model_id or settings.model_id,
            "model_revision": caps.model_revision or None,
            "provider": caps.provider,
        },
    )
    dest = job.dir / "source.pdf"
    size = 0
    try:
        # 디스크 쓰기·probe는 async 핸들러 안의 동기 blocking — 워커 스레드로 오프로드
        # (특히 손상 PDF는 MuPDF repair 스캔으로 수 초 걸려 이벤트 루프가 멎는다).
        async with await anyio.open_file(dest, "wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"업로드 상한({settings.max_upload_mb}MB)을 초과했습니다")
                await out.write(chunk)
        if size == 0:
            raise HTTPException(400, "빈 파일입니다")
        with dest.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise HTTPException(400, "PDF 형식이 아닙니다")
        try:
            await anyio.to_thread.run_sync(probe_pdf, dest, settings.max_pages)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    except HTTPException:
        await anyio.to_thread.run_sync(st.store.delete_dir, job)
        raise
    except BaseException:
        # 검증 실패(HTTPException) 외의 어떤 중단에서도 유령 queued 잡이 남으면 안 된다:
        # 디스크 오류(ENOSPC)나 업로드 중 연결 끊김(CancelledError)으로 여기 오면 잡은
        # 영원히 submit되지 않은 채 목록·대기열 위치만 왜곡한다. 취소된 스코프에서는
        # await가 즉시 다시 취소되므로 정리는 동기로 수행한다.
        st.store.delete_dir(job)
        raise

    st.worker.submit(job)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs")
def list_jobs(request: Request) -> dict:
    # 목록은 잡별 페이지/이미지 URL 전수 스캔이 필요 없다(프런트는 단건 조회에서 쓴다).
    # 폴링마다 잡×디렉터리 전체를 훑던 비용을 없앤다 — 응답 키는 그대로 유지된다.
    store = _state(request).store
    return {
        "jobs": [
            j.to_dict(queue_position=store.queue_position(j), include_files=False)
            for j in store.list()
        ]
    }


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    job = _get_job(request, job_id)
    return job.to_dict(queue_position=_state(request).store.queue_position(job))


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_poll(q: queue.Queue):
    try:
        return q.get(timeout=1.0)
    except queue.Empty:
        return None


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    job = _get_job(request, job_id)
    broker = _state(request).broker

    async def gen():
        # 구독 등록과 이전 토큰 스냅샷을 원자적으로 수행 — 업로드 응답과
        # EventSource 연결 사이, 또는 자동 재연결 동안 생성된 출력도 복구한다.
        q, replay, replay_truncated = broker.subscribe_with_replay(job_id)
        try:
            yield "retry: 3000\n\n"
            # 접속 시 스냅샷
            if job.status == "done":
                yield _sse_format("done", {
                    "markdown_url": f"/api/jobs/{job_id}/markdown",
                    "archive_url": f"/api/jobs/{job_id}/archive",
                })
                return
            if job.status in ("error", "canceled"):
                yield _sse_format("error", {
                    "message": job.error or "오류",
                    "canceled": job.status == "canceled",
                })
                return
            yield _sse_format("progress", {**job.progress, "status": job.status})
            if replay:
                yield _sse_format("replay", {
                    "text": replay,
                    "truncated": replay_truncated,
                    "current_page": job.progress.get("current_page", 0),
                    "total_pages": job.progress.get("total_pages", 0),
                })

            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                item = await anyio.to_thread.run_sync(
                    functools.partial(_sse_poll, q), limiter=_SSE_LIMITER,
                )
                if item is None:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": ping\n\n"
                    continue
                idle = 0
                event, data = item
                yield _sse_format(event, data)
                if event in ("done", "error"):
                    return
        finally:
            broker.unsubscribe(job_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _read_markdown(job) -> tuple[str, bool]:
    md_path = job.dir / "result.md"
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    return text, job.status != "done"


def _load_figure_boxes(job) -> dict | None:
    """벤더 P13 → merge가 통합한 images/boxes.json (없으면 풀폭 폴백)."""
    p = job.dir / "images" / "boxes.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@router.get("/jobs/{job_id}/markdown")
def job_markdown(request: Request, job_id: str, lang: str | None = None) -> PlainTextResponse:
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
        text = _translated_markdown_or_404(job, lang)
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
    text, partial = _read_markdown(job)
    headers = {"X-Partial": "true"} if partial else {}
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/jobs/{job_id}/html")
def job_html(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    job = _get_job(request, job_id)
    base = f"/api/jobs/{job_id}/files"
    sep = _state(request).settings.page_separator
    if lang is not None:
        _check_lang(lang)
        text = _translated_markdown_or_404(job, lang)
        html = render_document_html(
            text, base, figure_boxes=_load_figure_boxes(job), page_separator=sep,
        )
        return HTMLResponse(html)
    text, partial = _read_markdown(job)
    html = render_document_html(
        text, base, figure_boxes=_load_figure_boxes(job), page_separator=sep,
    )
    headers = {"X-Partial": "true"} if partial else {}
    return HTMLResponse(html, headers=headers)


def _backfill_layout_fonts(job, pages: list) -> None:
    """기존 잡 지연 백필: layout.json에 실측 폰트 크기(fs)가 빠진 비이미지 블록이
    있고 source.pdf가 있으면, 텍스트 레이어에서 뽑아 in-place 주입 후 원자적 저장.
    재변환 없이 이미 변환된 잡도 개선된다. enrichment 실패는 절대 500을 내지 않음."""
    src = job.dir / "source.pdf"
    if not src.exists():
        return
    try:
        from .pipeline.pdf_fonts import ENRICH_VERSION, enrich_layout_fonts
    except Exception:
        return
    # 버전 스탬프 기반: enrichment 스키마가 갱신되면(예: 세로쓰기 감지 추가)
    # 기존 잡도 1회 재백필된다. 스탬프가 최신이면 매 요청 재스캔하지 않는다.
    needs = any(
        isinstance(pg, dict) and int(pg.get("fonts_v") or 0) < ENRICH_VERSION
        for pg in pages
    )
    if not needs:
        return
    try:
        if enrich_layout_fonts(src, pages):
            # 요청별 고유 tmp — 동시 백필 요청이 같은 tmp에 겹쳐 쓰는 레이스 차단.
            # (병합 워커의 .layout.json.tmp와도 이름이 겹치지 않는다.)
            tmp = job.dir / f".layout.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, job.dir / "layout.json")
            finally:
                tmp.unlink(missing_ok=True)
    except Exception:
        pass  # 백필 실패는 렌더를 막지 않는다 (폴백 휴리스틱으로 표시)


def _load_layout_pages(job, lang: str | None = None) -> list:
    """lang=None이면 원본 layout.json, lang이면 번역본 layout.{lang}.json을 로드."""
    if lang is not None:
        p = job.dir / f"layout.{lang}.json"
        missing = "한국어 번역본이 없습니다 — 먼저 번역을 실행하세요"
    else:
        p = job.dir / "layout.json"
        missing = "레이아웃 데이터가 없습니다"
    if not p.is_file():
        raise HTTPException(404, missing)
    try:
        pages = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(pages, list)
    except Exception as e:
        raise HTTPException(500, "레이아웃 데이터를 읽을 수 없습니다") from e
    # 폰트 백필은 원본 layout.json만 대상: 번역본 페이지는 엔진이 fonts_v 스탬프를
    # 복사해 두므로 no-op이고, _backfill_layout_fonts는 결과를 원본 경로에 쓰기 때문에
    # 번역본에 실행하면 안 된다.
    if lang is None:
        _backfill_layout_fonts(job, pages)
    return pages


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def _load_pdf_export_report(job, lang: str) -> dict:
    try:
        loaded = json.loads(
            (job.dir / f"export.{lang}.report.json").read_text(encoding="utf-8")
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
    return job.dir / f"export.{lang}.font.txt"


def _write_pdf_export_font_id(job, lang: str, font_id: str) -> None:
    """어떤 폰트 설정으로 만든 PDF인지 원자적으로 남긴다(리포트는 파이프라인 소유라
    여기서 건드리지 않는다). 기록 실패는 다음 요청의 재빌드로만 이어진다."""
    marker = _font_marker_path(job, lang)
    tmp = job.dir / f".export.{lang}.font.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(font_id, encoding="utf-8")
        os.replace(tmp, marker)
    except OSError:
        logger.warning("PDF 폰트 표식 저장 실패: %s", marker.name)
    finally:
        tmp.unlink(missing_ok=True)


def _ensure_translated_pdf(job, lang: str, settings) -> tuple[Path, dict]:
    """번역 레이아웃과 같은 세대의 PDF를 만들거나 캐시에서 돌려준다.

    잡 단위 락 안에서 수행한다 — /pdf·/layout·/page가 동시에 첫 진입하면 같은
    export.{lang}.pdf를 여러 번 만들게 된다(수십 초짜리 작업).
    """
    source_pdf = job.dir / "source.pdf"
    orig_layout = job.dir / "layout.json"
    trans_layout = job.dir / f"layout.{lang}.json"
    out = job.dir / f"export.{lang}.pdf"
    font_id = _pdf_export_font_id(settings)
    with _job_render_lock(job.id):
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
            built = build_translated_pdf(
                job.dir, lang, fontfile=settings.pdf_export_font,
            )
            _write_pdf_export_font_id(job, lang, font_id)
            return built.path, built.report()
    return out, report


def _ensure_dual_pdf(job, lang: str, translated_pdf: Path) -> Path:
    """원본·번역 대조 PDF 캐시를 번역 단일 PDF와 같은 세대로 유지한다."""
    source_pdf = job.dir / "source.pdf"
    out = job.dir / f"export.{lang}.dual.pdf"
    try:
        latest_input = max(
            source_pdf.stat().st_mtime_ns,
            translated_pdf.stat().st_mtime_ns,
        )
    except OSError:
        # build_dual_pdf가 누락 파일을 사용자에게 읽을 수 있는 PdfExportError로 바꾼다.
        return build_dual_pdf(source_pdf, translated_pdf, out)
    if not out.is_file() or out.stat().st_mtime_ns < latest_input:
        return build_dual_pdf(source_pdf, translated_pdf, out)
    return out


def _forget_job_caches(job_id: str) -> None:
    """잡 삭제 시 프로세스 내 파생 캐시(락·검증 메모)를 함께 버린다."""
    with _FACSIMILE_LOCKS_GUARD:
        _FACSIMILE_LOCKS.pop(job_id, None)
    for key in [k for k in _FACSIMILE_VERIFIED if k[0] == job_id]:
        _FACSIMILE_VERIFIED.pop(key, None)


def _job_render_lock(job_id: str) -> threading.RLock:
    """잡 단위 렌더 락 — 다른 잡의 레이아웃/페이지 요청을 막지 않는다."""
    with _FACSIMILE_LOCKS_GUARD:
        return _FACSIMILE_LOCKS.setdefault(job_id, threading.RLock())


def _page_numbers(pages: list) -> list[int]:
    """layout 페이지 목록 → 렌더 파일명에 쓰는 페이지 번호(누락 시 순서 폴백)."""
    return [
        int(page.get("page", index)) if isinstance(page, dict) else index
        for index, page in enumerate(pages, start=1)
    ]


def _cached_page_numbers(job, lang: str | None) -> list[int]:
    """페이지 번호만 필요한 호출부(/page/{n})용 캐시.

    예전에는 페이지 이미지 요청마다 layout.json 전체를 재파싱했다. 파일 크기·mtime을
    키에 넣어 산출물이 갱신되면 자동으로 다시 읽는다."""
    path = job.dir / (f"layout.{lang}.json" if lang else "layout.json")
    try:
        stat = path.stat()
    except OSError:
        raise HTTPException(404, "레이아웃 데이터가 없습니다") from None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _LAYOUT_PAGE_NUMBERS_GUARD:
        cached = _LAYOUT_PAGE_NUMBERS.get(key)
    if cached is not None:
        return list(cached)
    numbers = tuple(_page_numbers(_load_layout_pages(job, lang)))
    with _LAYOUT_PAGE_NUMBERS_GUARD:
        if len(_LAYOUT_PAGE_NUMBERS) >= _LAYOUT_PAGE_NUMBERS_MAX:
            _LAYOUT_PAGE_NUMBERS.clear()
        _LAYOUT_PAGE_NUMBERS[key] = numbers
    return list(numbers)


def _ensure_facsimile_pages(job, page_numbers: list[int], lang: str | None, settings) -> Path:
    """HTML의 시각 기준면이 될 페이지 PNG 디렉터리를 보장한다.

    원문은 OCR 입력에 쓰인 pages/를 그대로 재사용한다. 번역본은 동일한
    export.{lang}.pdf를 job.dpi로 렌더해, HTML과 다운로드 PDF가 픽셀 수준에서
    같은 결과를 보게 한다. marker는 PDF 크기·mtime·DPI·페이지 수를 포함한다.
    """
    if lang is None:
        pages_dir = job.dir / "pages"
        if not pages_dir.is_dir():
            raise PdfExportError("원본 페이지 이미지가 없습니다")
        return pages_dir

    target = job.dir / "rendered" / lang
    marker = target / ".source.json"
    # PDF 빌드까지 락 안에서 수행한다 — 락 밖이면 같은 잡의 동시 첫 진입이 같은
    # export.{lang}.pdf를 중복으로 만든다.
    with _job_render_lock(job.id):
        pdf_path, _report = _ensure_translated_pdf(job, lang, settings)
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
            expected = [target / f"page_{number:04d}.png" for number in page_numbers]
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
        # 메모를 우회하지 못한다 — 디스크가 여전히 진실의 원천이다.
        marker_id = _marker_id()
        if marker_id and _FACSIMILE_VERIFIED.get(memo_key) == (signature, marker_id):
            return target
        if _cache_valid():
            _FACSIMILE_VERIFIED[memo_key] = (signature, _marker_id())
            return target
        target.mkdir(parents=True, exist_ok=True)
        # 재생성은 파일 단위로 원자적이어야 한다. 예전에는 기존 PNG를 먼저 지우고
        # 같은 자리에 다시 렌더해, 그 사이 /files 요청이 404나 반쯤 쓰인 이미지를
        # 받았다. 임시 디렉터리에 렌더한 뒤 os.replace로 갈아끼운다.
        staging = job.dir / "rendered" / f".{lang}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            render_pdf_pages(
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
            tmp = target / f".source.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
                os.replace(tmp, marker)
            finally:
                tmp.unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        _FACSIMILE_VERIFIED[memo_key] = (signature, _marker_id())
    return target


def _try_facsimile_pages(job, pages: list, lang: str | None, settings) -> Path | None:
    """레이아웃 HTML은 PDF export 결함 때문에 완전히 사라지지 않도록 폴백한다."""
    try:
        return _ensure_facsimile_pages(job, _page_numbers(pages), lang, settings)
    except (PdfExportError, OSError, ValueError):
        logger.exception(
            "facsimile 페이지 준비 실패 — 좌표 텍스트 렌더로 폴백: %s (%s)",
            job.id,
            lang or "orig",
        )
        return None


@router.get("/jobs/{job_id}/layout.html")
def job_layout_download(
    request: Request,
    job_id: str,
    lang: str | None = None,
) -> RedirectResponse:
    """구버전 레이아웃 HTML URL을 정식 document.html 내보내기로 통합한다."""
    _get_job(request, job_id)
    suffix = ""
    if lang is not None:
        _check_lang(lang)
        suffix = f"?lang={lang}"
    return RedirectResponse(
        url=f"/api/jobs/{job_id}/document.html{suffix}",
        status_code=307,
    )


@router.get("/jobs/{job_id}/document.html")
def job_document_download(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    """주 HTML 내보내기.

    완료된 좌표 레이아웃 잡은 PDF와 동일한 facsimile HTML을 내보낸다. 레이아웃이
    없는 figure_only 잡이나 부분 결과만 기존 읽기용 semantic HTML로 폴백한다.
    """
    from urllib.parse import quote

    job = _get_job(request, job_id)
    st = _state(request)
    if lang is not None:
        _check_lang(lang)
        text = _translated_markdown_or_404(job, lang)
    else:
        text, _partial = _read_markdown(job)  # 미완료 잡도 부분 결과 내보내기 허용(/markdown과 동일)
    layout_path = job.dir / (f"layout.{lang}.json" if lang else "layout.json")
    if job.status == "done" and layout_path.is_file():
        pages = _load_layout_pages(job, lang)
        pages_dir = _try_facsimile_pages(job, pages, lang, st.settings)
        if pages_dir is not None:
            stem = Path(job.filename).stem or "document"
            html = render_layout_standalone(
                pages,
                job.dir,
                stem,
                st.settings.resolve_frontend_dir(),
                lang=lang,
                pages_dir=pages_dir,
                facsimile=True,
            )
            suffix = f".{lang}" if lang else ""
            fname = f"{stem}{suffix}.html"
            return HTMLResponse(html, headers={
                "Content-Disposition":
                    f"attachment; filename=\"document{suffix}.html\"; "
                    f"filename*=UTF-8''{quote(fname)}",
            })
    inner = render_document_html(
        text, f"/api/jobs/{job_id}/files",
        figure_boxes=_load_figure_boxes(job),
        page_separator=st.settings.page_separator,
    )
    stem = Path(job.filename).stem or "document"
    html = render_document_standalone(
        inner, job.dir, stem, st.settings.resolve_frontend_dir(), lang=lang,
    )
    suffix = f".{lang}" if lang else ""
    fname = f"{stem}{suffix}.html"
    return HTMLResponse(html, headers={
        "Content-Disposition":
            f"attachment; filename=\"document{suffix}.html\"; filename*=UTF-8''{quote(fname)}",
    })


@router.get("/jobs/{job_id}/layout")
def job_layout(request: Request, job_id: str, lang: str | None = None) -> HTMLResponse:
    """PDF facsimile 레이아웃 뷰 — 페이지 기준면 + 검색 가능한 OCR 텍스트 레이어."""
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
    pages = _load_layout_pages(job, lang)
    st = _state(request)
    pages_dir = _try_facsimile_pages(job, pages, lang, st.settings)
    page_src = None
    if pages_dir is not None:
        rel = pages_dir.relative_to(job.dir).as_posix()

        def page_src(page_number):
            return f"/api/jobs/{job_id}/files/{rel}/page_{page_number:04d}.png"
    return HTMLResponse(render_layout_html(
        pages,
        f"/api/jobs/{job_id}/files",
        lang=lang,
        page_src=page_src,
        facsimile=page_src is not None,
    ))


@router.get("/jobs/{job_id}/page/{page_number}")
def job_page_image(
    request: Request,
    job_id: str,
    page_number: int,
    lang: str | None = None,
) -> FileResponse:
    """리더용 최종 페이지 이미지.

    원문은 source 렌더, lang=ko는 번역 PDF 렌더를 반환한다. 프런트가 언어를
    바꿔도 왼쪽 페이지가 계속 영문으로 남던 기존 경로를 대체한다.
    """
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
        _translated_markdown_or_404(job, lang)
    layout_path = job.dir / (f"layout.{lang}.json" if lang else "layout.json")
    if not layout_path.is_file():
        # figure_only 엔진은 번역 텍스트 좌표가 없어 최종 페이지 합성이 불가능하다.
        # 기존 원본 페이지를 유지하고 오른쪽 semantic 번역문을 계속 제공한다.
        path = job.dir / "pages" / f"page_{page_number:04d}.png"
        if page_number < 1 or not path.is_file():
            raise HTTPException(404, "페이지 이미지를 찾을 수 없습니다")
        return FileResponse(path, media_type="image/png")
    page_numbers = _cached_page_numbers(job, lang)
    if page_number not in set(page_numbers):
        raise HTTPException(404, "페이지를 찾을 수 없습니다")
    try:
        pages_dir = _ensure_facsimile_pages(
            job, page_numbers, lang, _state(request).settings,
        )
    except PdfExportError as e:
        # 내보내기 불가는 서버 결함이 아니라 잡 상태(입력 누락·손상) 문제다 —
        # 500 대신 사용자에게 그대로 보여줄 수 있는 409로 알린다.
        raise HTTPException(409, str(e)) from e
    path = pages_dir / f"page_{page_number:04d}.png"
    if not path.is_file():
        raise HTTPException(404, "페이지 이미지를 찾을 수 없습니다")
    return FileResponse(path, media_type="image/png")


def _alignment_bbox(value) -> list[float] | None:
    """Unlimited-OCR의 0..999 bbox를 안전한 클라이언트 좌표로 정규화한다."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(v == v and abs(v) != float("inf") for v in coords):
        return None
    x1, y1, x2, y2 = (min(1000.0, max(0.0, v)) for v in coords)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _layout_page(pages: list, page_number: int) -> dict | None:
    for page in pages:
        if isinstance(page, dict) and page.get("page") == page_number:
            return page
    return None


def _alignment_payload(
    source_page: dict,
    target_page: dict,
    page: int,
    lang: str | None,
) -> dict:
    """이미 한 번 로드한 페이지 쌍을 viewer alignment 계약으로 변환한다."""
    source_blocks = source_page.get("blocks")
    target_blocks = target_page.get("blocks")
    if not isinstance(source_blocks, list) or not isinstance(target_blocks, list):
        raise HTTPException(500, "레이아웃 블록 데이터가 올바르지 않습니다")
    if len(source_blocks) != len(target_blocks):
        raise HTTPException(409, "원문과 번역 레이아웃의 블록 대응이 올바르지 않습니다")

    aligned = []
    for index, (source, target) in enumerate(zip(source_blocks, target_blocks)):
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        bbox = _alignment_bbox(source.get("bbox"))
        if bbox is None:
            continue
        source_type = str(source.get("type") or "text")
        target_type = str(target.get("type") or "text")
        if source_type != target_type or target.get("bbox") != source.get("bbox"):
            if lang is not None:
                raise HTTPException(
                    409,
                    "원문과 번역 레이아웃의 좌표 대응이 올바르지 않습니다",
                )
            continue
        source_text = str(source.get("content") or "").strip()
        target_text = str(target.get("content") or "").strip()
        if not source_text and not target_text:
            continue
        visibly_translated = (
            lang is not None
            and " ".join(target_text.split()) != " ".join(source_text.split())
        )
        aligned.append({
            "id": f"p{page}-b{index}",
            "index": index,
            "type": source_type,
            "bbox": bbox,
            "source": source_text,
            "target": target_text or source_text,
            "translated": visibly_translated,
        })

    return {
        "page": page,
        "width": source_page.get("width"),
        "height": source_page.get("height"),
        "bbox_space": 1000,
        "lang": lang or "orig",
        "blocks": aligned,
    }


@router.get("/jobs/{job_id}/alignment")
def job_alignment(
    request: Request,
    job_id: str,
    page: int = 1,
    lang: str | None = None,
) -> dict:
    """원문 PDF bbox와 같은 인덱스의 번역 블록을 리더용으로 연결한다.

    번역 파이프라인은 layout.json을 깊은 복사한 뒤 content만 교체한다. 이 계약을
    요청 시 다시 검증해, 순서가 어긋난 번역을 잘못된 원문 위치에 표시하지 않는다.
    """
    if page < 1:
        raise HTTPException(422, "페이지 번호는 1 이상이어야 합니다")
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)

    source_pages = _load_layout_pages(job)
    source_page = _layout_page(source_pages, page)
    if source_page is None:
        raise HTTPException(404, "페이지를 찾을 수 없습니다")

    target_page = source_page
    if lang is not None:
        target_pages = _load_layout_pages(job, lang)
        target_page = _layout_page(target_pages, page)
        if target_page is None:
            raise HTTPException(409, "번역 레이아웃의 페이지 대응이 올바르지 않습니다")

    return _alignment_payload(source_page, target_page, page, lang)


def _viewer_artifact_revision(job, lang: str | None) -> str:
    """뷰어 계약 캐시 키. 내용 전체를 재해시하지 않고 원자적으로 교체되는 산출물의
    파일명·크기·mtime을 묶는다. 강제 재번역/재병합 시 URL 계약이 즉시 바뀐다."""
    names = ["meta.json", "layout.json", "result.md"]
    if lang is not None:
        names.extend([
            f"layout.{lang}.json",
            f"result.{lang}.md",
            f"translations/{lang}/state.json",
        ])
    parts = []
    for name in names:
        path = job.dir / name
        try:
            stat = path.stat()
        except OSError:
            parts.append(f"{name}:missing")
        else:
            parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _viewer_cache_headers(revision: str) -> dict[str, str]:
    return {
        "Cache-Control": "private, no-cache",
        "ETag": f'"viewer-{revision}"',
        "Vary": "Authorization",
    }


@router.get("/jobs/{job_id}/viewer-manifest")
def job_viewer_manifest(
    request: Request,
    job_id: str,
    lang: str | None = None,
) -> Response:
    """전체 화면 reader의 작고 명시적인 부트스트랩 계약.

    페이지 GET에서 번역 PDF 전체를 지연 생성하지 않도록 source 이미지 URL과
    번역/좌표 capability를 분리한다. 변경 가능한 로컬 산출물이므로 조건부 재검증을
    강제하고, ETag가 일치하면 본문 없이 304를 반환한다.
    """
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
    revision = _viewer_artifact_revision(job, lang)
    headers = _viewer_cache_headers(revision)
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)

    source_layout = job.dir / "layout.json"
    target_layout = job.dir / f"layout.{lang}.json" if lang else source_layout
    source_images = sorted((job.dir / "pages").glob("page_*.png"))
    page_count = 0
    try:
        page_count = int(job.progress.get("total_pages") or 0)
    except (TypeError, ValueError):
        page_count = 0
    page_count = max(page_count, len(source_images))

    translate_state = _read_translate_state(job, lang) if lang else None
    language_status = (
        str((translate_state or {}).get("status") or "missing")
        if lang else "ready"
    )
    warnings = list(job.warnings)
    body = {
        "schema_version": 1,
        "artifact_revision": revision,
        "job_id": job.id,
        "status": job.status,
        "partial": job.status != "done",
        "document": {
            "filename": job.filename,
            "page_count": page_count,
            "ready_page_count": len(source_images),
        },
        "language": {
            "requested": lang or "orig",
            "status": language_status,
        },
        "capabilities": {
            "source_page_image": bool(source_images),
            # 뷰어 좌측 기준면은 항상 source다. 번역 PDF raster 준비 여부는 별도 표기해
            # 원문 이미지를 한국어 이미지로 잘못 라벨링하는 200 fallback을 막는다.
            "translated_page_image": bool(
                lang
                and target_layout.is_file()
                and (job.dir / "rendered" / lang / ".source.json").is_file()
            ),
            "alignment": source_layout.is_file() and target_layout.is_file(),
            "outline": target_layout.is_file(),
        },
        "quality": {
            "state": "degraded" if warnings else "ok",
            "warning_count": len(warnings),
            "warnings": warnings[:20],
        },
        "links": {
            "source_page_template": f"/api/jobs/{job.id}/page/{{page}}?revision={revision}",
            "pages": f"/api/jobs/{job.id}/viewer/pages",
            "alignment_template": f"/api/jobs/{job.id}/alignment?page={{page}}",
            "outline": f"/api/jobs/{job.id}/outline",
            "html": f"/api/jobs/{job.id}/html",
            "events": f"/api/jobs/{job.id}/events",
        },
        "limits": {"max_page_batch": 16},
    }
    return JSONResponse(body, headers=headers)


@router.get("/jobs/{job_id}/viewer/pages")
def job_viewer_pages(
    request: Request,
    job_id: str,
    start: int = 1,
    limit: int = 4,
    lang: str | None = None,
    include: str = "alignment",
) -> JSONResponse:
    """인접 페이지 메타/좌표를 한 번의 layout 파싱으로 돌려주는 제한된 배치."""
    if start < 1:
        raise HTTPException(422, "시작 페이지는 1 이상이어야 합니다")
    if limit < 1 or limit > 16:
        raise HTTPException(422, "페이지 배치 크기는 1에서 16 사이여야 합니다")
    if include not in ("none", "alignment"):
        raise HTTPException(422, "include는 none 또는 alignment여야 합니다")
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)

    source_pages = _load_layout_pages(job)
    target_pages = _load_layout_pages(job, lang) if lang else source_pages
    source_by_number = {
        page.get("page"): page
        for page in source_pages
        if isinstance(page, dict) and isinstance(page.get("page"), int)
    }
    target_by_number = {
        page.get("page"): page
        for page in target_pages
        if isinstance(page, dict) and isinstance(page.get("page"), int)
    }
    page_numbers = sorted(number for number in source_by_number if number >= start)
    selected = page_numbers[:limit]
    items = []
    revision = _viewer_artifact_revision(job, lang)
    for page_number in selected:
        source_page = source_by_number[page_number]
        target_page = target_by_number.get(page_number)
        if target_page is None:
            raise HTTPException(409, "번역 레이아웃의 페이지 대응이 올바르지 않습니다")
        item = {
            "page": page_number,
            "width": source_page.get("width"),
            "height": source_page.get("height"),
            "source_image_url":
                f"/api/jobs/{job.id}/page/{page_number}?revision={revision}",
        }
        if include == "alignment":
            item["alignment"] = _alignment_payload(
                source_page, target_page, page_number, lang,
            )
        items.append(item)
    next_start = selected[-1] + 1 if selected and len(page_numbers) > len(selected) else None
    return JSONResponse({
        "schema_version": 1,
        "artifact_revision": revision,
        "total": len(source_by_number),
        "next_start": next_start,
        "items": items,
    }, headers=_viewer_cache_headers(revision))


@router.get("/jobs/{job_id}/outline")
def job_outline(request: Request, job_id: str, lang: str | None = None) -> dict:
    """OCR title 블록으로 문서 목차를 만든다 — reader 연구 도구의 탐색 기준."""
    job = _get_job(request, job_id)
    if lang is not None:
        _check_lang(lang)
    pages = _load_layout_pages(job, lang)
    items = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page")
        if not isinstance(page_number, int):
            continue
        for block in page.get("blocks", ()):
            if not isinstance(block, dict) or block.get("type") != "title":
                continue
            text = str(block.get("content") or "").strip()
            if not text:
                continue
            fs = block.get("fs")
            level = 1 if not items else (2 if isinstance(fs, (int, float)) and fs >= 2.2 else 3)
            items.append({
                "page": page_number,
                "level": level,
                "text": text[:500],
            })
    return {"items": items}


@router.get("/jobs/{job_id}/files/{file_path:path}")
def job_file(request: Request, job_id: str, file_path: str) -> FileResponse:
    job = _get_job(request, job_id)
    # 정규화(상위참조 해석 + 심볼릭 링크 추적)를 **먼저** 하고, 그 결과가 허용
    # 디렉터리 하위인지 검사한다. 첫 경로 세그먼트만 allowlist와 비교하면
    # `pages/../source.pdf`처럼 잡 디렉터리 안의 임의 파일(원본 업로드 PDF,
    # meta.json, translations/**)이 그대로 서빙된다.
    try:
        full = (job.dir / file_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(404, "파일을 찾을 수 없습니다") from None
    roots = [(job.dir / name).resolve() for name in _ALLOWED_FILE_DIRS]
    if not any(full.is_relative_to(root) for root in roots) or not full.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    return FileResponse(full)


@router.get("/jobs/{job_id}/archive")
def job_archive(request: Request, job_id: str) -> FileResponse:
    job = _get_job(request, job_id)
    if job.status != "done":
        raise HTTPException(409, "아직 변환이 완료되지 않았습니다")
    zip_path = job.dir / "archive.zip"
    if not zip_path.is_file():
        # 요청별 고유 tmp — 동시 요청 둘이 같은 tmp에 겹쳐 써 손상 zip이 캐시되는
        # 레이스 차단(sync 핸들러는 스레드풀 병렬). 둘 다 완주하면 마지막 replace가 승자.
        tmp = job.dir / f".archive.{uuid.uuid4().hex}.tmp"
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                md = job.dir / "result.md"
                if md.is_file():
                    zf.write(md, "result.md")
                # 변환 메타(엔진/모델/경고)도 동봉 — 어떤 모델로 변환했는지 아카이브만으로 확인 가능
                meta = job.dir / "meta.json"
                if meta.is_file():
                    zf.write(meta, "meta.json")
                # 번역본(result.ko.md 등)도 포함 — 번역 완료 시 이 zip 캐시가 삭제돼
                # 다음 요청에서 번역본까지 담아 재생성된다. (glob은 result.md 자신은 제외)
                for extra in sorted(job.dir.glob("result.*.md")):
                    zf.write(extra, extra.name)
                images = job.dir / "images"
                if images.is_dir():
                    for f in sorted(images.iterdir()):
                        if f.is_file():
                            zf.write(f, f"images/{f.name}")
            tmp.replace(zip_path)
        finally:
            tmp.unlink(missing_ok=True)
    stem = Path(job.filename).stem or "result"
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{stem}.markdown.zip",
    )


@router.get("/jobs/{job_id}/pdf")
def job_pdf(
    request: Request,
    job_id: str,
    lang: str = "ko",
    view: str = "single",
) -> FileResponse:
    """레이아웃 보존 번역 PDF 내보내기 — layout.json/layout.{lang}.json 기반으로
    번역된 텍스트 블록만 원본 PDF에서 교체한다(수식·그림·표는 원본 유지).
    ``view=dual``이면 원본과 번역 페이지를 좌우로 붙인 대조 PDF를 반환한다.
    캐시는 번역 완료 시(_run_translate_thread) 무효화된다."""
    job = _get_job(request, job_id)
    _check_lang(lang)
    if view not in ("single", "dual"):
        raise HTTPException(400, "지원하지 않는 PDF 보기 형식")
    if job.status != "done":
        raise HTTPException(409, "아직 변환이 완료되지 않았습니다")
    _translated_markdown_or_404(job, lang)
    trans_layout = job.dir / f"layout.{lang}.json"
    if not (job.dir / "layout.json").is_file() or not trans_layout.is_file():
        raise HTTPException(
            409,
            "이 잡에는 좌표 레이아웃이 없어 PDF 내보내기를 지원하지 않습니다"
            " — HTML 내보내기(document.html)를 사용하세요",
        )
    try:
        translated_pdf, report = _ensure_translated_pdf(
            job, lang, _state(request).settings,
        )
        out = _ensure_dual_pdf(job, lang, translated_pdf) if view == "dual" else translated_pdf
    except PdfExportError as e:
        raise HTTPException(500, str(e))
    # 헤더는 숫자만 사용해 비ASCII 경고문·원문이 HTTP 메타데이터로 새지 않게 한다.
    specialist = report.get("specialist_kept")
    if not isinstance(specialist, dict):
        specialist = {}
    headers = {
        "X-UOCR-PDF-Replaced": str(int(report.get("replaced") or 0)),
        "X-UOCR-PDF-Preserved": str(int(report.get("kept") or 0)),
        "X-UOCR-PDF-Relocated": str(int(report.get("relocated") or 0)),
        "X-UOCR-PDF-Table-Cells": str(int(report.get("table_cells_replaced") or 0)),
        "X-UOCR-PDF-Warnings": str(int(report.get("warning_count") or 0)),
        "X-UOCR-PDF-Specialist-Preserved": str(sum(
            int(value or 0)
            for value in specialist.values()
            if isinstance(value, (int, float))
        )),
    }
    stem = Path(job.filename).stem or "document"
    return FileResponse(
        out, media_type="application/pdf", filename=f"{stem}.{lang}.pdf", headers=headers)


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(request: Request, job_id: str) -> dict:
    """삭제 없이 중단 — 부분 결과(result.md, 완료된 청크의 이미지)는 보존된다."""
    st = _state(request)
    job = _get_job(request, job_id)
    if job.status in ("done", "error", "canceled"):
        return {"job_id": job_id, "status": job.status}
    st.cancel_events.setdefault(job_id, threading.Event()).set()
    return {"job_id": job_id, "status": "canceling"}


# ── 번역(한국어) 라우트 ───────────────────────────────────────────────────
@router.post("/jobs/{job_id}/translate")
async def translate_start(request: Request, job_id: str) -> JSONResponse:
    """번역 시작. body JSON {"lang":"ko","force":false} (기본 lang="ko").

    반환: 이미 실행 중이면 200 {"status":"running"}, state가 done이고 force가
    아니면 200 {"status":"done"}, 그 외에는 데몬 스레드를 띄우고 202 {"status":"running"}.
    """
    st = _state(request)
    job = _get_job(request, job_id)
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        body = {}
    lang = body.get("lang") or "ko"
    force = bool(body.get("force", False))

    _check_lang(lang)
    if job.status != "done":
        raise HTTPException(409, "변환이 완료된 잡만 번역할 수 있습니다")
    # 무인증 서비스의 비용 상한: 잡·IP 단위 레이트리밋(초과 시 429 + Retry-After).
    guard = _abuse_guard(st, "translate")
    guard.check_rate((f"translate:job:{job_id}", f"translate:ip:{_client_key(request)}"))
    try:
        cfg = TranslateConfig.from_env()
    except TranslateError as e:
        raise HTTPException(503, str(e)) from e

    with st.translate_lock:
        if (job_id, lang) in st.translate_tasks:
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "running"}, status_code=200
            )
        # 번역 스레드는 잡마다 병렬로 뜬다 — 동시에 도는 번역 수 자체에도 상한을
        # 둬야 유료 API 호출과 스레드가 무한히 늘지 않는다.
        if 0 < guard.max_concurrent <= len(st.translate_tasks):
            raise HTTPException(
                429,
                "동시에 실행 중인 번역이 너무 많습니다 — 잠시 후 다시 시도하세요",
                headers={"Retry-After": "30"},
            )
        state = _read_translate_state(job, lang)
        if not force and state is not None and state.get("status") == "done":
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "done"}, status_code=200
            )
        cancel = threading.Event()
        thread = threading.Thread(
            target=_run_translate_thread,
            args=(st, job, lang, cfg, cancel, force, st.settings.page_separator),
            name=f"translate-{job_id}-{lang}", daemon=True,
        )
        st.translate_tasks[(job_id, lang)] = {"thread": thread, "cancel": cancel}
        thread.start()
    return JSONResponse({"job_id": job_id, "lang": lang, "status": "running"}, status_code=202)


@router.get("/jobs/{job_id}/translate/state")
def translate_state(request: Request, job_id: str, lang: str = "ko") -> dict:
    """번역 상태. 없으면 {"status":"none","lang"}. stale-running은 error로 조정해 반환."""
    job = _get_job(request, job_id)
    _check_lang(lang)
    state = _stale_adjusted_state(request, job, lang)
    if state is None:
        return {"status": "none", "lang": lang}
    return state


@router.post("/jobs/{job_id}/translate/cancel")
def translate_cancel(request: Request, job_id: str, lang: str = "ko") -> JSONResponse:
    """실행 중이면 cancel 이벤트를 set하고 202 canceling, 아니면 현재 상태를 반환."""
    st = _state(request)
    job = _get_job(request, job_id)
    _check_lang(lang)
    with st.translate_lock:
        task = st.translate_tasks.get((job_id, lang))
        if task is not None:
            task["cancel"].set()
            return JSONResponse(
                {"job_id": job_id, "lang": lang, "status": "canceling"}, status_code=202
            )
    state = _stale_adjusted_state(request, job, lang)
    if state is None:
        return JSONResponse(
            {"job_id": job_id, "lang": lang, "status": "none"}, status_code=200
        )
    return JSONResponse(state, status_code=200)


@router.get("/jobs/{job_id}/translate/events")
async def translate_events(request: Request, job_id: str, lang: str = "ko") -> StreamingResponse:
    """번역 진행 SSE (job_events와 동일 패턴). 스냅샷: done→done 후 종료, error/canceled→
    error 후 종료, running→progress 스냅샷 후 구독 루프. state가 없으면 404."""
    st = _state(request)
    job = _get_job(request, job_id)
    _check_lang(lang)
    # POST /translate가 202를 준 직후에는 워커 스레드가 아직 state.json을 쓰기 전일
    # 수 있다. 그 창에서 404를 주면 프런트가 SSE를 포기하고 폴백도 못 한다 —
    # 레지스트리에 태스크가 있으면 '실행 중'으로 보고 스트림을 연다.
    with st.translate_lock:
        registered = (job_id, lang) in st.translate_tasks
    if not registered and _stale_adjusted_state(request, job, lang) is None:
        raise HTTPException(404, "번역 상태가 없습니다")
    broker = st.broker
    channel = _translate_channel(job_id, lang)

    def _done_data() -> dict:
        return {
            "phase": "translate", "lang": lang,
            "markdown_url": f"/api/jobs/{job_id}/markdown?lang={lang}",
            "html_url": f"/api/jobs/{job_id}/html?lang={lang}",
            "layout_url": f"/api/jobs/{job_id}/layout?lang={lang}",
        }

    async def gen():
        # 구독 먼저, 그 다음 스냅샷 재조회 — job_events와 같은 순서라 구독~완료 사이 이벤트를
        # 놓치지 않는다 (엔진이 state를 먼저 쓰고 스레드가 이후 done/error를 발행하므로,
        # 스냅샷이 running이면 종료 이벤트는 아직 큐로 들어온다).
        q = broker.subscribe(channel)
        try:
            yield "retry: 3000\n\n"
            state = _stale_adjusted_state(request, job, lang) or {"status": "none"}
            status = state.get("status")
            if status == "done":
                yield _sse_format("done", _done_data())
                return
            if status in ("error", "canceled"):
                yield _sse_format("error", {
                    "message": state.get("error") or "오류",
                    "canceled": status == "canceled",
                })
                return
            yield _sse_format("progress", {
                "phase": "translate", "lang": lang,
                "current": state.get("current") or 0,
                "total": state.get("total") or 0,
                "status": "running",
            })

            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                item = await anyio.to_thread.run_sync(
                    functools.partial(_sse_poll, q), limiter=_SSE_LIMITER,
                )
                if item is None:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": ping\n\n"
                    continue
                idle = 0
                event, data = item
                yield _sse_format(event, data)
                if event in ("done", "error"):
                    return
        finally:
            broker.unsubscribe(channel, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── 페이지 Q&A + LLM 프로바이더 카탈로그 (Localight 이식) ──────────────────
@router.get("/providers")
async def llm_providers(request: Request) -> dict:
    """프로바이더 카탈로그 — LlmRouter.providers() 그대로 반환
    (default_provider/default_reasoning_effort/providers[]).
    Ollama 모델 목록은 /api/tags 라이브 조회(:cloud·remote_host 필터링 포함)."""
    return await _state(request).llm_router.providers()


_QA_PROVIDERS = ("openai-responses", "openai-chat", "ollama")
# app/llm/providers.py는 클라이언트 입력 거절(모델 허용목록·프로바이더 위반)과 실제
# 업스트림 장애를 같은 LlmError로 던진다. 앞의 것은 재시도해도 소용없으므로 400,
# 뒤의 것만 503으로 매핑한다. 문구의 단일 출처는 providers.py이며
# tests/test_api_qa.py가 실제 문구로 이 매핑을 고정한다.
_LLM_INPUT_ERROR_MARKERS = (
    "is not an allowed",
    "is not an on-device",
    "Unsupported OpenAI provider",
    "Unknown LLM provider",
)


def _llm_error_status(message: str) -> int:
    return 400 if any(marker in message for marker in _LLM_INPUT_ERROR_MARKERS) else 503


@router.post("/jobs/{job_id}/qa")
async def job_qa(request: Request, job_id: str, body: AskRequest) -> dict:
    """완료된 잡의 단일 페이지 텍스트를 컨텍스트로 질문에 답한다 (Localight /ask 계약).

    404 잡 없음 / 409 미완료 잡(번역과 동일) / 422 페이지 범위 밖·빈 페이지 /
    400 잘못된 프로바이더·허용목록 밖 모델 / 429 상한 초과 / 503 업스트림 LLM 장애.
    컨텍스트는 해당 페이지 하나뿐 — 캐시 없음.
    """
    st = _state(request)
    job = _get_job(request, job_id)
    if job.status != "done":
        raise HTTPException(409, "변환이 완료된 잡만 질문할 수 있습니다")
    if body.provider is not None and body.provider not in _QA_PROVIDERS:
        raise HTTPException(400, f"지원하지 않는 LLM 프로바이더: {body.provider}")
    # 무인증 서비스의 비용 상한 — 운영자의 유료 키를 임의 사용자가 소진하지 못하게 한다.
    guard = _abuse_guard(st, "qa")
    guard.check_rate((f"qa:job:{job_id}", f"qa:ip:{_client_key(request)}"))

    llm_router = st.llm_router
    page = body.page or 1
    # result.md 읽기+분할은 blocking 파일 IO — async 핸들러의 루프를 막지 않게 오프로드
    page_count, text = await anyio.to_thread.run_sync(
        get_page_context, job.dir, page, st.settings.page_separator
    )
    if not (1 <= page <= page_count):
        raise HTTPException(422, f"페이지 {page}은 1-{page_count} 범위를 벗어났습니다")
    if not text:
        raise HTTPException(422, "이 페이지에서 텍스트를 찾지 못했습니다")

    provider = body.provider or st.settings.llm_provider
    model = body.model or llm_router.default_model(provider)
    # 요청이 effort를 생략하면 서버 기본(LLM_REASONING_EFFORT) — 'default'는 그대로
    # 통과시켜 프로바이더 페이로드에서 생략되게 한다 (Localight 매핑 유지).
    effort = body.reasoning_effort or st.settings.llm_reasoning_effort
    context = f"[Page {page}]\n{text}"
    if not guard.acquire():
        raise HTTPException(
            429,
            "동시에 처리 중인 질문이 너무 많습니다 — 잠시 후 다시 시도하세요",
            headers={"Retry-After": "5"},
        )
    try:
        result = await llm_router.ask(
            question=body.question,
            context=context,
            provider=provider,
            model=model,
            reasoning_effort=effort,
            reasoning_summary=body.reasoning_summary,
            thinking=body.thinking,
        )
    except LlmError as e:
        raise HTTPException(_llm_error_status(str(e)), str(e)) from e
    finally:
        guard.release()
    # Localight /ask 응답 형태 그대로
    return {
        "answer": result.content,
        "provider": result.provider,
        "model": result.model,
        "page": page,
        "reasoning_effort": result.reasoning_effort,
        "reasoning_summary": result.reasoning_summary,
        "thinking_requested": result.thinking_requested,
        "usage": result.usage,
        "remote": result.remote,
        "local_only": not result.remote,
    }


@router.post("/jobs/{job_id}/render-preview")
async def render_preview(request: Request, job_id: str) -> HTMLResponse:
    """클라이언트가 보낸 (정리된) 마크다운을 안전 렌더 — 라이브 미리보기용.
    /html과 동일한 렌더러라 XSS 이스케이프·표 복원·이미지 URL 재작성이 적용된다."""
    job = _get_job(request, job_id)
    # 스트리밍 수신하며 상한을 먼저 검사 — 전체를 메모리에 적재한 뒤 검사하면
    # 상한 초과 본문도 일단 다 받게 되어 상한의 의미가 없다.
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _PREVIEW_MAX_BYTES:
            raise HTTPException(413, "미리보기 본문이 너무 큽니다 (2MB 초과)")
        chunks.append(chunk)
    text = b"".join(chunks).decode("utf-8", "replace")

    # 렌더(2MB에 ~0.2초)는 async 핸들러의 이벤트 루프를 막지 않게 오프로드.
    # 공유 _md(markdown-it) 인스턴스는 렌더 시 상태 변이가 없어(파스 상태는
    # 호출별 StateCore) 스레드 안전 — sync 핸들러(/html 등)가 이미 스레드풀에서
    # 동시 사용 중인 기존 불변식이다.
    def _render() -> str:
        return render_markdown_html(
            text, f"/api/jobs/{job_id}/files", figure_boxes=_load_figure_boxes(job)
        )

    return HTMLResponse(await anyio.to_thread.run_sync(_render))


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(request: Request, job_id: str) -> Response:
    st = _state(request)
    job = _get_job(request, job_id)
    job.delete_requested = True
    ev = st.cancel_events.get(job_id)
    if ev is not None:
        ev.set()
    # 이 잡의 실행 중 번역 스레드도 함께 취소 — 삭제된 디렉터리에 유료 API 호출과
    # 파일 기록을 계속하지 않게 한다 (레지스트리 정리는 _run_translate_thread finally 몫).
    with st.translate_lock:
        for (jid, _lang), task in st.translate_tasks.items():
            if jid == job_id:
                task["cancel"].set()
    if job.status != "running":
        # queued 잡은 워커가 dequeue 시 delete_requested를 보고 정리하지만,
        # 디렉터리와 목록은 지금 바로 제거해 UI에서 사라지게 한다.
        st.store.delete_dir(job)
    _forget_job_caches(job_id)
    return Response(status_code=204)
