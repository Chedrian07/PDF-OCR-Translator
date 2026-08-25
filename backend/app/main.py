"""FastAPI 앱 팩토리. 실행: uvicorn app.main:app"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings
from .engine import build_engine
from .jobs import EventBroker, JobStore, Worker
from .llm import build_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_GC_INTERVAL_S = 6 * 60 * 60  # 잡 TTL GC 주기 — 시작 시 1회 + 6시간마다

# ── 업로드 본문 상한 (멀티파트 파싱 전에 차단) ────────────────────────────
# POST /api/jobs의 MAX_UPLOAD_MB 검사는 Starlette가 폼을 파싱한 **뒤**에 돈다 —
# 그 시점엔 초과 본문이 이미 임시 스풀 파일에 전량 기록돼 있어, 인증이 없는 이
# 서비스에서 업로드 한 번으로 디스크를 소진할 수 있다. 그래서 파싱 이전 단계인
# ASGI 계층에서 먼저 끊는다.
_UPLOAD_PATH = "/api/jobs"
# 멀티파트 봉투(경계 문자열·파트 헤더·mode/dpi 필드) 여유분. 실제 봉투는 수백
# 바이트지만, 정확히 MAX_UPLOAD_MB인 PDF가 봉투 몇 바이트 때문에 거절되면 회귀이므로
# 넉넉히 잡는다 — 64KiB는 상한(기본 100MB) 대비 무시할 수 있고, 이 여유분 안으로
# 새어 들어온 본문은 라우트의 기존 스트리밍 검사가 413으로 잡는다.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _route_path(scope: dict) -> str:
    """라우터가 보는 경로 — 리버스 프록시 뒤(root_path)에서도 경로 스코프가 맞게."""
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    return path[len(root):] if root and path.startswith(root) else path


def _declared_length(scope: dict) -> int | None:
    """Content-Length 헤더 값 — 없거나 정수가 아니면 None(길이 미상)."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class UploadBodyLimitMiddleware:
    """POST /api/jobs 전용 ASGI 본문 상한.

    경로를 업로드 라우트로 한정한다 — /render-preview(자체 2MB 상한)·SSE 스트림·
    다운로드 응답은 이 미들웨어를 그대로 통과한다.
    """

    def __init__(self, app, max_bytes: int, max_mb: int) -> None:
        self.app = app
        self.limit = max_bytes + _MULTIPART_OVERHEAD_BYTES
        self.detail = f"업로드 상한({max_mb}MB)을 초과했습니다"

    async def __call__(self, scope, receive, send) -> None:
        if not (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and _route_path(scope).rstrip("/") == _UPLOAD_PATH
        ):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.limit:
            await self._reject(send)  # 본문을 한 바이트도 읽지 않고 거절
            return

        state = {"received": 0, "exceeded": False, "started": False}

        async def guarded_receive():
            # 길이 미상(chunked)이면 누적 바이트를 세다가 상한에서 끊는다
            message = await receive()
            if message["type"] == "http.request" and not state["exceeded"]:
                state["received"] += len(message.get("body", b""))
                if state["received"] > self.limit:
                    state["exceeded"] = True
                    if not state["started"]:
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message) -> None:
            if state["exceeded"]:
                return  # 413을 이미 보냈다 — 앱의 후속 응답은 버린다
            if message["type"] == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except Exception:
            # 끊긴 본문을 만난 폼 파서가 던진 오류 — 이미 413으로 응답했다
            if not state["exceeded"]:
                raise

    async def _reject(self, send) -> None:
        body = json.dumps({"detail": self.detail}, ensure_ascii=False).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                # 남은 본문을 계속 받아 버리지 않도록 연결을 닫는다
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    store = JobStore(settings.jobs_dir)
    store.load_existing()
    broker = EventBroker()
    engine = build_engine(settings)  # 잘못된 OCR_DEVICE/OCR_ENGINE은 여기서 즉시 실패
    cancel_events: dict[str, threading.Event] = {}
    worker = Worker(store, broker, engine, settings, cancel_events)
    load_state: dict = {"error": None}

    def _preload() -> None:
        try:
            engine.load()
        except Exception as e:  # noqa: BLE001 — 헬스에 노출하고 잡 제출 시 재시도
            # 일시적 조건(sidecar가 아직 준비 중)은 정상적인 기동 과정이다 —
            # 무서운 traceback 대신 info로 남기고, 잡 제출 시 워커가 대기한다.
            if getattr(e, "transient", False):
                logger.info("모델 프리로드 대기: %s", str(e)[:200])
            else:
                logger.exception("모델 프리로드 실패")
            load_state["error"] = str(e)[:500]

    async def _gc_loop(app_: FastAPI) -> None:
        """잡 TTL GC — 시작 직후 1회 + _GC_INTERVAL_S 주기. 번역 스레드가 살아 있는
        잡은 삭제 직전 잡별 레지스트리 확인으로 보호(스냅샷 방식이면 GC 패스 도중
        시작된 번역이 빠진다), 파일 IO(rmtree)는 스레드로 오프로드."""

        def _is_protected(job_id: str) -> bool:
            with app_.state.translate_lock:
                return any(jid == job_id for jid, _lang in app_.state.translate_tasks)

        while True:
            try:
                await asyncio.to_thread(store.gc_expired, settings.job_ttl_days, _is_protected)
            except Exception:  # noqa: BLE001 — GC 실패가 다음 주기를 막지 않게
                logger.exception("잡 GC 실패")
            await asyncio.sleep(_GC_INTERVAL_S)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        if settings.preload_model and not engine.loaded:
            threading.Thread(target=_preload, name="model-preload", daemon=True).start()
        # JOB_TTL_DAYS>0일 때만 기동 — 기본 0 = 사용자 데이터 자동 삭제 비활성(opt-in)
        gc_task = asyncio.create_task(_gc_loop(_app)) if settings.job_ttl_days > 0 else None
        yield
        if gc_task is not None:
            gc_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gc_task
        worker.stop()

    app = FastAPI(title="Unlimited-OCR — PDF → Markdown", lifespan=lifespan)
    # 업로드 본문 상한 — 폼 파싱(=임시 스풀 파일 기록) 이전에 끊는다.
    # 먼저 등록하므로 TrustedHost 검증이 바깥에 남는다(Host 위조는 그대로 400).
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_bytes=settings.max_upload_bytes,
        max_mb=settings.max_upload_mb,
    )
    # Host 헤더 화이트리스트 — DNS rebinding 방어 (무인증 서비스, README §보안).
    # Starlette가 포트를 떼고 비교하므로 localhost:8000도 localhost로 통과한다.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    # 와일드카드는 Host 검증을 사실상 끈다 — 무인증 서비스이므로 운영자가 신뢰 경계를
    # 인지하도록 기동 시 1회 경고한다 (compose 기본값이 '*'라 조용히 켜지기 쉽다).
    if any("*" in host for host in settings.allowed_hosts):
        logger.warning(
            "ALLOWED_HOSTS=%s — 와일드카드가 있어 모든 Host 헤더를 허용합니다. "
            "이 서비스는 인증이 없으니 서버 IP·호스트명만 나열하세요 (README §보안)",
            ",".join(settings.allowed_hosts),
        )
    app.state.settings = settings
    app.state.store = store
    app.state.broker = broker
    app.state.engine = engine
    app.state.worker = worker
    app.state.cancel_events = cancel_events
    app.state.load_state = load_state
    # 번역 태스크 레지스트리: 키 (job_id, lang) → {"thread","cancel"}.
    # OCR 워커(단일 스레드 직렬)와 달리 번역은 잡별 데몬 스레드로 병렬 실행된다.
    app.state.translate_tasks: dict[tuple[str, str], dict] = {}
    app.state.translate_lock = threading.Lock()
    # 잡별 worker와 별도로, 여러 잡이 동시에 번역돼도 한 프로세스가 upstream에
    # 보내는 실제 HTTP 합계는 설정 상한을 넘지 않는다.
    app.state.translate_api_slots = threading.BoundedSemaphore(
        settings.translate_global_concurrency
    )
    # Localight LLM 라우터 (페이지 Q&A + 프로바이더 카탈로그). 잘못된 LLM env는
    # Settings.from_env(local_url/openai_url/_env_choice)가 이미 기동 시점에 ValueError로
    # 걸러냈으므로 여기서는 조립만 한다 — 네트워크 호출 없음.
    app.state.llm_router = build_router(settings)

    app.include_router(router)

    frontend = settings.resolve_frontend_dir()
    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
        logger.info("프론트엔드 서빙: %s", frontend)
    else:
        logger.warning("프론트엔드 디렉터리를 찾지 못했습니다 (FRONTEND_DIR 설정 가능)")

    return app


app = create_app()
