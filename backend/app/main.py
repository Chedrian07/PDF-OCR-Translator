"""FastAPI 앱 팩토리. 실행: uvicorn app.main:app

`app`은 PEP 562 지연 속성이다 — 이 모듈을 import만 해서는 앱이 만들어지지 않는다
(아래 `__getattr__` 참조).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import signal
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

from . import __version__
from .api import router
from .config import Settings
from .engine import build_engine
from .jobs import EventBroker, JobStore, Worker
from .llm import build_router
from .owner_lock import JobsDirLock, acquire_jobs_dir_lock
from .pipeline.runner import chunk_length_budget_note

# 스레드 이름을 포맷에 포함한다 — 번역은 잡별 데몬 스레드로 **병렬** 실행되고
# (api.py: name=f"translate-{job_id}-{lang}") OCR 워커·sidecar 요청 스레드도 함께
# 돌아, 이름이 없으면 여러 잡의 로그가 한 줄씩 섞여 어느 잡·언어의 실패인지
# 사후에 분리할 수 없다. 로거별 코드 변경 없이 상관관계를 얻는 최소 조치.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s",
)
logger = logging.getLogger(__name__)

_GC_INTERVAL_S = 6 * 60 * 60  # 잡 TTL GC 주기 — 시작 시 1회 + 6시간마다

# ── 요청 본문 상한 (라우트 진입·파싱 전에 차단) ────────────────────────────
# POST /api/jobs의 MAX_UPLOAD_MB 검사는 Starlette가 폼을 파싱한 **뒤**에 돈다 —
# 그 시점엔 초과 본문이 이미 임시 스풀 파일에 전량 기록돼 있어, 인증이 없는 이
# 서비스에서 업로드 한 번으로 디스크를 소진할 수 있다. 그래서 파싱 이전 단계인
# ASGI 계층에서 먼저 끊는다.
# 같은 이유로 상한을 업로드 경로 하나에만 걸면 안 된다: POST /jobs/{id}/qa 같은
# JSON 라우트는 **유효한 잡 ID조차 없이** 본문을 통째로 메모리에 적재한다(실측
# uvicorn: 80MB 본문 1건에 RSS +422MB·동시 4건 +1.4GB, 게다가 422 응답이 80MB
# 원문을 되돌려준다. 상한 적용 후 같은 요청은 RSS +0MB·62바이트 413).
# 그래서 (경로 패턴 → 상한) 표로 넓히고, **표에 없는 경로는 작은 기본 상한**으로
# 기울인다 — 새 POST 라우트가 추가돼도 자동으로 보호된다.
_UPLOAD_PATH = "/api/jobs"
# 멀티파트 봉투(경계 문자열·파트 헤더·mode/dpi 필드) 여유분. 실제 봉투는 수백
# 바이트지만, 정확히 MAX_UPLOAD_MB인 PDF가 봉투 몇 바이트 때문에 거절되면 회귀이므로
# 넉넉히 잡는다 — 64KiB는 상한(기본 100MB) 대비 무시할 수 있고, 이 여유분 안으로
# 새어 들어온 본문은 라우트의 기존 스트리밍 검사가 413으로 잡는다.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
# /render-preview의 라우트 내부 상한(api._PREVIEW_MAX_BYTES)과 **같은 값** —
# 미들웨어가 더 빡빡하면 경계값(정확히 256KiB) 요청이 회귀로 거절된다. 예전 2MB는
# 인증 없는 요청 한 건에 수십 초짜리 렌더를 허용했다(감사 security-1).
_PREVIEW_LIMIT_BYTES = 256 * 1024
# 표에 없는 본문 있는 요청의 기본 상한. 남은 POST 라우트(cancel/translate/qa)는
# 전부 작은 JSON이라 64KiB로 충분하다.
_DEFAULT_BODY_LIMIT_BYTES = 64 * 1024
# 본문을 가질 수 있는 메서드만 검사한다 (GET/SSE/다운로드는 그대로 통과).
_GUARDED_METHODS = frozenset({"POST", "PUT", "PATCH"})


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
    """경로별 ASGI 본문 상한 — 라우트 진입(=스풀·파싱·메모리 적재) 전에 끊는다.

    (경로 패턴 → 상한) 표로 판정하고, 표에 없는 경로는 작은 기본 상한을 쓴다.
    GET(SSE 스트림·다운로드)은 검사 대상이 아니라 그대로 통과한다.
    """

    def __init__(self, app, max_bytes: int, max_mb: int) -> None:
        self.app = app
        # (패턴, 상한, 거절 사유) — 위에서부터 첫 일치를 쓴다
        self.rules: tuple[tuple[re.Pattern[str], int, str], ...] = (
            (
                re.compile(rf"^{re.escape(_UPLOAD_PATH)}$"),
                max_bytes + _MULTIPART_OVERHEAD_BYTES,
                f"업로드 상한({max_mb}MB)을 초과했습니다",
            ),
            (
                re.compile(r"^/api/jobs/[^/]+/render-preview$"),
                _PREVIEW_LIMIT_BYTES,
                "미리보기 본문이 너무 큽니다 (256KiB 초과)",
            ),
        )
        self.default_rule = (
            _DEFAULT_BODY_LIMIT_BYTES,
            f"요청 본문이 너무 큽니다 ({_DEFAULT_BODY_LIMIT_BYTES // 1024}KiB 초과)",
        )

    def _limit_for(self, path: str) -> tuple[int, str]:
        normalized = path.rstrip("/") or "/"
        for pattern, limit, detail in self.rules:
            if pattern.match(normalized):
                return limit, detail
        return self.default_rule

    async def __call__(self, scope, receive, send) -> None:
        if not (
            scope["type"] == "http" and scope.get("method") in _GUARDED_METHODS
        ):
            await self.app(scope, receive, send)
            return

        limit, detail = self._limit_for(_route_path(scope))

        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await self._reject(send, detail)  # 본문을 한 바이트도 읽지 않고 거절
            return

        state = {"received": 0, "exceeded": False, "started": False}

        async def guarded_receive():
            # 길이 미상(chunked)이면 누적 바이트를 세다가 상한에서 끊는다
            message = await receive()
            if message["type"] == "http.request" and not state["exceeded"]:
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["exceeded"] = True
                    if not state["started"]:
                        await self._reject(send, detail)
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

    async def _reject(self, send, detail: str) -> None:
        body = json.dumps({"detail": detail}, ensure_ascii=False).encode()
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


# ── HTML 응답 보안 헤더 · 정적 프런트엔드 캐시 정책 ─────────────────────────────
# OCR·텍스트 레이어 마크다운의 `![](https://…)`는 렌더러를 지나 그대로 <img>가 된다.
# 문서를 여는 순간 브라우저가 제3자·LAN 주소로 요청을 보내 열람 사실·IP·인스턴스
# 주소가 새고 내부망 GET이 유도됐다(감사 frontend-3·pipeline-ocr-4·sidecar-5 — 근본
# 수정은 렌더러 쪽). 그 심층 방어로 HTML 응답에 CSP를 붙여 외부 리소스 로드(이미지·
# 폰트·연결)를 같은 출처와 data:/blob:으로 묶는다. 리더는 /html·/layout 조각을 SPA에
# innerHTML로 넣으므로 실제 효력은 SPA 문서(index.html)의 정책에서 난다.
# - 인라인 style 속성(레이아웃 좌표·KaTeX)은 쓰므로 style-src에 'unsafe-inline'.
# - SPA 스크립트는 같은 출처 파일('self' — 테마 부트스트랩도 theme-init.js)이고, index.html에
#   인라인 스크립트가 생기면 그 해시로만 허용한다 — 주입된 인라인 스크립트·on* 속성은
#   막힌다. 해시는 index.html에서 계산하고 파일이 바뀌면 다시 계산한다.
# - API가 내보내는 HTML(document.html 내려받기 등)은 KaTeX를 인라인으로 품으므로
#   스크립트는 'unsafe-inline'을 두되 리소스 출처는 같은 규칙으로 묶는다.
_CSP_COMMON = (
    "default-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
)
_API_HTML_CSP = "; ".join(
    (_CSP_COMMON[0], "script-src 'self' 'unsafe-inline'", *_CSP_COMMON[1:])
)
# 외부로 나가는 링크 클릭에 인스턴스 주소(Referer)를 싣지 않는다 — 같은 출처끼리는 유지.
_REFERRER_POLICY = "same-origin"
_INLINE_SCRIPT = re.compile(
    r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>", re.IGNORECASE | re.DOTALL,
)
_SCRIPT_SRC_ATTR = re.compile(r"\bsrc\s*=", re.IGNORECASE)


def _inline_script_hashes(html: str) -> list[str]:
    """인라인 <script> 본문의 CSP 해시('sha256-…'). src가 있는 스크립트는 제외한다.

    브라우저는 스크립트 요소의 텍스트를 UTF-8 그대로 해시한다(개행은 HTML 파서가 LF로
    정규화 — read_text의 범용 개행 처리와 같다)."""
    hashes = []
    for match in _INLINE_SCRIPT.finditer(html):
        if _SCRIPT_SRC_ATTR.search(match.group("attrs")):
            continue
        digest = hashlib.sha256(match.group("body").encode("utf-8")).digest()
        hashes.append(f"'sha256-{base64.b64encode(digest).decode('ascii')}'")
    return hashes


class _SpaContentSecurityPolicy:
    """SPA 문서용 CSP — index.html이 바뀌면(배포·개발 중 수정) 해시를 다시 계산한다."""

    def __init__(self, frontend_dir: Path | None) -> None:
        self._index = frontend_dir / "index.html" if frontend_dir is not None else None
        self._key: tuple[int, int] | None = None
        self._value = self._build([])
        self._lock = threading.Lock()

    @staticmethod
    def _build(hashes: list[str]) -> str:
        script = " ".join(("'self'", *dict.fromkeys(hashes)))
        return "; ".join((_CSP_COMMON[0], f"script-src {script}", *_CSP_COMMON[1:]))

    def value(self) -> str:
        if self._index is None:
            return self._value
        try:
            stat = self._index.stat()
        except OSError:
            return self._value
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key != self._key:
                try:
                    html = self._index.read_text(encoding="utf-8")
                except (OSError, ValueError):
                    return self._value
                self._value = self._build(_inline_script_hashes(html))
                self._key = key
            return self._value


class SecurityHeadersMiddleware:
    """응답 헤더 정책 — 순수 ASGI라 SSE·파일 스트리밍을 버퍼링하지 않는다.

    - HTML 응답: Content-Security-Policy(위 설명) + Referrer-Policy.
    - 정적 프런트엔드(/api 밖): Cache-Control: no-cache. 예전에는 Cache-Control 없이
      Last-Modified만 나가 브라우저가 휴리스틱 신선도로 ES 모듈을 파일마다 다른 시점에
      재검증 없이 재사용했다 — 업그레이드 뒤 새 reader.js가 옛 viewer.js에서 새 export를
      import하면 SyntaxError로 앱 전체가 흰 화면이 됐다. no-cache는 '저장하되 매번
      재검증'이라 ETag가 같으면 304로 끝난다.
    라우트가 이미 정한 헤더는 덮어쓰지 않는다.
    """

    def __init__(self, app, spa_csp: _SpaContentSecurityPolicy) -> None:
        self.app = app
        self.spa_csp = spa_csp

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _route_path(scope)
        is_api = path == "/api" or path.startswith("/api/")

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if headers.get("content-type", "").startswith("text/html"):
                    if "content-security-policy" not in headers:
                        headers["Content-Security-Policy"] = (
                            _API_HTML_CSP if is_api else self.spa_csp.value()
                        )
                    if "referrer-policy" not in headers:
                        headers["Referrer-Policy"] = _REFERRER_POLICY
                if not is_api and "cache-control" not in headers:
                    headers["Cache-Control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ── 정상 종료 신호 → 열린 SSE 스트림 종료 ─────────────────────────────────────
# uvicorn은 SIGTERM/SIGINT를 받으면 새 연결을 막고 진행 중 응답에는 keep_alive=False만
# 세운 뒤, 연결이 **모두 닫힐 때까지** 기다리고 나서야 lifespan 종료를 보낸다(기본
# timeout_graceful_shutdown=None). 진행 중 잡·번역의 SSE 스트림은 done/error나 클라이언트
# 끊김으로만 끝나므로, 탭 하나만 열려 있어도 docker stop은 매번 SIGKILL(lifespan 종료
# 미실행)로 끝났고 make dev(--reload)는 잡이 끝날 때까지 재시작이 멈췄다. lifespan은
# 너무 늦게 불리므로, 서버가 설치한 신호 처리기를 감싸 종료 요청을 앱에도 알리고 SSE
# 루프가 다음 폴(≤1s)에서 스스로 끝나게 한다. EventSource는 retry로 재연결한다.
_SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class ShutdownSignal:
    """종료 요청 표식 — SSE 루프가 폴마다 읽는다.

    신호 처리기에서 세우므로 락·Event를 쓰지 않는다: 처리기는 메인 스레드의 바이트코드
    사이에서 돌아, 두 번째 신호가 첫 처리기의 Event.set() 내부(락 보유 중)를 끊으면
    같은 스레드가 같은 락을 다시 기다리는 교착이 될 수 있다. 속성 대입은 원자적이다."""

    def __init__(self) -> None:
        self.requested = False


def _install_shutdown_hooks(flag: ShutdownSignal) -> dict:
    """서버가 설치한 SIGINT/SIGTERM 파이썬 처리기를 감싸 종료 요청을 flag에도 남긴다.

    신호 처리기는 메인 스레드에서만 바꿀 수 있다(TestClient의 lifespan은 별도 스레드 —
    그때는 아무것도 하지 않는다). 파이썬 처리기가 없으면(기본 동작) 감쌀 대상도, 기다릴
    정상 종료도 없으므로 건드리지 않는다. 원래 처리기는 그대로 호출한다."""
    installed: dict = {}
    if threading.current_thread() is not threading.main_thread():
        return installed
    for sig in _SHUTDOWN_SIGNALS:
        previous = signal.getsignal(sig)
        if not callable(previous):
            continue

        def _hook(signum, frame, _previous=previous) -> None:
            flag.requested = True
            _previous(signum, frame)

        signal.signal(sig, _hook)
        installed[sig] = (previous, _hook)
    return installed


def _remove_shutdown_hooks(installed: dict) -> None:
    """감싼 처리기를 원래대로 — 그 사이 다른 쪽이 바꿨다면 건드리지 않는다."""
    if not installed or threading.current_thread() is not threading.main_thread():
        return
    for sig, (previous, hook) in installed.items():
        if signal.getsignal(sig) is hook:
            signal.signal(sig, previous)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    # load_existing()이 디스크 상태(running→error, work/ 삭제)를 바꾸기 **전에** 잡
    # 디렉터리의 단일 소유권을 잡는다 — 살아 있는 다른 백엔드가 쥐고 있으면 여기서
    # 기동을 거부한다. 앱 수명 동안 쥐고 lifespan 종료 시 놓는다(owner_lock.py).
    owner_lock = acquire_jobs_dir_lock(settings.jobs_dir)
    try:
        return _assemble_app(settings, owner_lock)
    except BaseException:
        owner_lock.release()  # 조립 실패(잘못된 OCR_DEVICE 등)에 락만 남지 않게
        raise


def _assemble_app(settings: Settings, owner_lock: JobsDirLock) -> FastAPI:
    store = JobStore(settings.jobs_dir)
    # 페이지 구분자를 기록하기 전에 만든 잡에는 지금 설정을 한 번 고정해 둔다(jobs.Job).
    restored = store.load_existing(default_page_separator=settings.page_separator)
    broker = EventBroker()
    engine = build_engine(settings)  # 잘못된 OCR_DEVICE/OCR_ENGINE은 여기서 즉시 실패
    # MAX_LENGTH < 청크 최악 길이는 설정의 성질이다 — 잡마다가 아니라 기동 시 한 번만,
    # 경고가 아닌 안내로 남긴다(잘린 청크는 페이지별로 복구돼 내용이 빠지지 않는다).
    budget_note = chunk_length_budget_note(settings, engine)
    if budget_note:
        logger.info("%s", budget_note)
    cancel_events: dict[str, threading.Event] = {}
    # 모델 로드 오류 — 프리로드 스레드와 워커(잡 시작 시 로드)가 함께 기록하고
    # /api/health의 model_load_error가 읽는다.
    load_state: dict = {"error": None}
    worker = Worker(store, broker, engine, settings, cancel_events, load_state=load_state)
    # 재시작 전 대기열에 들어갔지만 시작하지 못한 잡을 생성 순서대로 다시 제출한다
    # (실행 중이던 잡은 load_existing이 오류로 마감 — 크래시 루프를 피해 자동 재실행하지
    # 않는다). 워커는 lifespan에서 시작되므로 그때부터 차례로 처리된다.
    for job in restored:
        worker.submit(job)
    if restored:
        logger.info("재시작 전 대기 잡 %d개를 다시 대기열에 넣었습니다", len(restored))

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

    shutdown = ShutdownSignal()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        shutdown_hooks = _install_shutdown_hooks(shutdown)
        try:
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
        finally:
            # 신호 없이 끝나는 수명(TestClient 등)에서도 남은 스트림이 끝나게 한다.
            shutdown.requested = True
            _remove_shutdown_hooks(shutdown_hooks)
            # 닫힌 앱은 잡 디렉터리 소유권을 바로 놓는다 — 같은 DATA_DIR로 다음 앱
            # (재시작·테스트의 재생성)이 뜰 수 있게. 예외로 끝난 수명도 마찬가지다.
            owner_lock.release()

    app = FastAPI(
        title="Unlimited-OCR — PDF → Markdown", version=__version__, lifespan=lifespan,
    )
    frontend = settings.resolve_frontend_dir()
    # HTML 응답 CSP·Referrer-Policy와 정적 파일 재검증 정책 — 가장 안쪽 미들웨어.
    app.add_middleware(
        SecurityHeadersMiddleware, spa_csp=_SpaContentSecurityPolicy(frontend),
    )
    # 요청 본문 상한 — 라우트 진입(폼 파싱=임시 스풀 파일 기록, JSON 메모리 적재)
    # 이전에 끊는다. 경로별 표는 UploadBodyLimitMiddleware 참조.
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
    app.state.owner_lock = owner_lock
    app.state.store = store
    app.state.broker = broker
    app.state.engine = engine
    app.state.worker = worker
    app.state.cancel_events = cancel_events
    app.state.load_state = load_state
    # 정상 종료 요청 표식 — SSE 루프(api.py)가 폴마다 보고 스스로 끝난다.
    app.state.shutdown = shutdown
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

    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
        logger.info("프론트엔드 서빙: %s", frontend)
    else:
        logger.warning("프론트엔드 디렉터리를 찾지 못했습니다 (FRONTEND_DIR 설정 가능)")

    return app


# `app` 지연 생성을 한 번으로 묶는다 (동시에 처음 접근해도 앱은 하나).
_DEFAULT_APP_LOCK = threading.Lock()


def __getattr__(name: str) -> FastAPI:
    """PEP 562 지연 속성 — `uvicorn app.main:app`·`from app.main import app`이
    처음 `app`을 찾을 때만 기본 앱을 만들고, 이후에는 모듈 전역에 캐시한다.

    예전에는 모듈 끝에서 `app = create_app()`을 바로 불러 **import만으로** 부작용이
    났다. pytest 수집(conftest가 create_app을 import)이나 e2e_mock_app 같은 다른
    진입점도 Settings.from_env()로 개발자의 실제 .env(실키)를 프로세스 환경에
    주입했고, 개발 서버가 쓰는 DATA_DIR에 load_existing()을 돌려 실행 중 잡을
    error로 덮고 work/를 지웠다.
    """
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _DEFAULT_APP_LOCK:
        built = globals().get("app")
        if built is None:
            try:
                built = create_app()
            except AttributeError as e:
                # __getattr__에서 새어 나간 AttributeError는 호출자에게 '속성 없음'으로
                # 보인다(uvicorn: Attribute "app" not found) — 진짜 원인을 가리지 않게 바꾼다.
                raise RuntimeError(f"기본 앱 생성 실패: {e}") from e
            globals()["app"] = built
    return built
