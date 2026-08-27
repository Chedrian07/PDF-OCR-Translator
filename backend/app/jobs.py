"""잡 상태 저장(JobStore) · SSE 이벤트 브로커 · 단일 워커 스레드."""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .config import Settings
    from .engine.base import OCREngine

logger = logging.getLogger(__name__)

_META_NAME = "meta.json"
_EVENT_QUEUE_MAX = 2000
# EventSource 최초 연결·자동 재연결 전에 생성된 OCR 토큰을 복구한다. 페이지별
# decoded 문자 상한(16,384) × 최대 200페이지보다 넉넉하고, 단일 OCR 워커라
# 동시에 커지는 히스토리는 하나뿐이다. 터미널 이벤트에서 즉시 폐기한다.
_TOKEN_HISTORY_MAX_CHARS = 8 * 1024 * 1024
# 아직 워커 큐에 제출되지 않은(업로드 중) 잡의 정렬 키 — 제출된 잡보다 항상 뒤.
_UNSUBMITTED_SEQ = float("inf")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_progress() -> dict:
    return {"phase": "render", "current_page": 0, "total_pages": 0, "chunk": 0, "total_chunks": 0}


@dataclass
class Job:
    id: str
    filename: str
    mode: str
    dpi: int
    dir: Path
    status: str = "queued"  # queued|running|done|error|canceled
    created_at: str = field(default_factory=_now_iso)
    progress: dict = field(default_factory=_default_progress)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    delete_requested: bool = False
    # 변환에 사용된 엔진/모델 메타 — 완료 후에도 어떤 모델로 변환했는지 확인 가능.
    # 구버전 meta.json에는 없으므로 복원 시 None 허용 (필드 부재 = 알 수 없음).
    engine: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    provider: str | None = None
    # 워커 큐 제출 순번(런타임 전용, meta.json 미기록). 업로드 본문 수신이 끝난 뒤에야
    # submit()되므로 생성 순서와 어긋날 수 있다 — queue_position이 이 값을 쓴다.
    # 아직 제출 전(업로드 중)이면 None.
    submit_seq: int | None = None

    def _result_block(self, *, include_files: bool = True) -> dict | None:
        if self.status != "done":
            return None
        base = f"/api/jobs/{self.id}"

        def _urls(subdir: str) -> list[str]:
            # include_files=False면 디렉터리 스캔 자체를 건너뛴다 — 목록 폴링처럼
            # 전 페이지 URL이 필요 없는 호출부의 전수 스캔 비용을 없앤다.
            # 키는 항상 유지하므로 기존 클라이언트 계약은 불변.
            if not include_files:
                return []
            d = self.dir / subdir
            if not d.is_dir():
                return []
            # 이미지 파일만 — images/boxes.json 같은 메타 파일은 목록에서 제외
            return [
                f"{base}/files/{subdir}/{f.name}"
                for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

        return {
            "markdown_url": f"{base}/markdown",
            "html_url": f"{base}/html",
            "archive_url": f"{base}/archive",
            "viewer_manifest_url": f"{base}/viewer-manifest",
            "images": _urls("images"),
            "layouts": _urls("layout"),
            "pages": _urls("pages"),
            # 레이아웃 뷰/다운로드 가능 여부 — 레이아웃 기능(P14) 이전에 변환된
            # 잡에는 layout.json이 없어 /layout*이 404가 난다. 프런트가 이 플래그로
            # 버튼을 비활성화한다 (없으면 재변환 필요).
            "has_layout": (self.dir / "layout.json").is_file(),
        }

    def to_dict(
        self, queue_position: int | None = None, *, include_files: bool = True
    ) -> dict:
        d = {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status,
            "mode": self.mode,
            "created_at": self.created_at,
            "progress": dict(self.progress),
            "error": self.error,
            "warnings": list(self.warnings),
            "result": self._result_block(include_files=include_files),
            # 신규 필드(추가만 — 기존 필드 의미 불변). 구 잡은 null.
            "engine": self.engine,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "provider": self.provider,
        }
        # 선택 필드 — queued 잡에만 존재(계약). running/터미널 잡은 필드 자체가 없다.
        if queue_position is not None:
            d["queue_position"] = queue_position
        return d

    def meta(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "mode": self.mode,
            "dpi": self.dpi,
            "status": self.status,
            "created_at": self.created_at,
            "progress": self.progress,
            "error": self.error,
            "warnings": self.warnings,
            "engine": self.engine,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "provider": self.provider,
        }


class JobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._submit_seq = 0
        self._lock = threading.RLock()

    def create(
        self, filename: str, mode: str, dpi: int, engine_info: dict | None = None
    ) -> Job:
        job_id = f"j_{uuid.uuid4().hex[:12]}"
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, filename=filename, mode=mode, dpi=dpi, dir=job_dir)
        if engine_info:
            job.engine = engine_info.get("engine")
            job.model_id = engine_info.get("model_id")
            job.model_revision = engine_info.get("model_revision")
            job.provider = engine_info.get("provider")
        with self._lock:
            self._jobs[job_id] = job
        self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def queue_position(self, job: Job) -> int | None:
        """queued 잡의 대기열 위치(1-base): 워커 큐 제출 순서 기준.

        단일 워커 큐는 FIFO 제출 순서인데, 제출(submit)은 업로드 본문 수신이 끝난
        뒤라 생성 순서와 어긋날 수 있다(큰 파일을 먼저 올리기 시작해도 작은 파일이
        먼저 제출된다). 그래서 mark_submitted()가 매긴 submit_seq로 센다. 아직
        제출 전(업로드 중)인 잡은 이미 제출된 잡들 뒤에 오도록 두고, 동률은 _jobs
        삽입 순서(=create() 호출 순서)로 안정 정렬한다.
        running/터미널 잡은 None(직렬화 시 필드 생략)."""
        if job.status != "queued":
            return None
        with self._lock:
            if job.id not in self._jobs:
                return None  # 삭제 경합 — 목록에서 빠졌으면 위치 없음
            order = sorted(
                (j.submit_seq if j.submit_seq is not None else _UNSUBMITTED_SEQ, idx, j.id)
                for idx, j in enumerate(self._jobs.values())
                if j.status == "queued"
            )
        for pos, (_seq, _idx, jid) in enumerate(order, start=1):
            if jid == job.id:
                return pos
        return None

    def mark_submitted(self, job: Job) -> None:
        """워커 큐 제출 순번을 부여한다 — queue_position이 실제 처리 순서를 반영하게."""
        with self._lock:
            self._submit_seq += 1
            job.submit_seq = self._submit_seq

    def save(self, job: Job) -> None:
        tmp = job.dir / f".{_META_NAME}.tmp"
        try:
            tmp.write_text(json.dumps(job.meta(), ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, job.dir / _META_NAME)
        except OSError as e:
            # 메타 기록은 best-effort — 디스크 만원(ENOSPC)·권한 오류가 잡 처리
            # 흐름(특히 오류 마감 경로)이나 워커 스레드를 죽여서는 안 된다.
            # FileNotFoundError는 삭제 경합이라 정상 경로 — 로그도 남기지 않는다.
            if not isinstance(e, FileNotFoundError):
                logger.warning("잡 메타 기록 실패: %s (%s)", job.id, e)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def delete_dir(self, job: Job) -> None:
        shutil.rmtree(job.dir, ignore_errors=True)
        self.remove(job.id)

    def gc_expired(
        self, ttl_days: int, is_protected: "Callable[[str], bool] | None" = None
    ) -> int:
        """TTL 지난 터미널 잡 자동 정리 (JOB_TTL_DAYS — 0 이하면 무동작).

        마지막 활동 시각은 meta.json mtime(save()가 상태 변화마다 재기록)과
        translations/*/state.json mtime의 최댓값 — OCR이 오래전에 끝났어도 최근
        번역된 잡은 보존한다. queued/running 잡은 절대 삭제하지 않고, is_protected
        (번역 스레드 활성 등)는 삭제 직전에 잡별로 호출한다 — 스냅샷 방식이면 GC
        패스 도중 시작된 번역이 보호되지 않는다. 삭제는 DELETE 엔드포인트와 같은
        delete_dir 경로. 삭제 수 반환."""
        if ttl_days <= 0:
            return 0
        now = time.time()
        cutoff = now - ttl_days * 86400
        with self._lock:
            jobs = list(self._jobs.values())
        removed = 0
        for job in jobs:
            if job.status in ("queued", "running"):
                continue
            try:
                mtime = (job.dir / _META_NAME).stat().st_mtime
            except OSError:  # meta 유실 — 나이를 알 수 없으니 보수적으로 보존
                continue
            tdir = job.dir / "translations"
            if tdir.is_dir():
                for st in tdir.glob("*/state.json"):
                    try:
                        mtime = max(mtime, st.stat().st_mtime)
                    except OSError:
                        pass
            if mtime >= cutoff:
                continue
            if is_protected is not None and is_protected(job.id):
                continue
            logger.info("잡 GC: %s 삭제 (status=%s, %.1f일 경과 > TTL %d일)",
                        job.id, job.status, (now - mtime) / 86400, ttl_days)
            self.delete_dir(job)
            removed += 1
        return removed

    def load_existing(self) -> None:
        """서버 재시작 시 디스크의 잡 복원. 실행 중이던 잡은 오류로 마킹."""
        if not self.jobs_dir.is_dir():
            return
        for d in sorted(self.jobs_dir.iterdir()):
            meta_path = d / _META_NAME
            if not meta_path.is_file():
                continue
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                job = Job(
                    id=m["id"], filename=m["filename"], mode=m.get("mode", "multi"),
                    dpi=int(m.get("dpi", 200)), dir=d, status=m.get("status", "error"),
                    created_at=m.get("created_at", _now_iso()),
                    progress=m.get("progress") or _default_progress(),
                    error=m.get("error"), warnings=m.get("warnings") or [],
                    # 구버전 meta.json에는 없는 필드 — 없으면 None으로 안전 복원
                    engine=m.get("engine"), model_id=m.get("model_id"),
                    model_revision=m.get("model_revision"), provider=m.get("provider"),
                )
                changed = job.status in ("queued", "running")
                if changed:
                    job.status = "error"
                    job.error = "서버 재시작으로 중단되었습니다"
                with self._lock:
                    self._jobs[job.id] = job
                # 상태가 바뀐 잡만 재기록 — 터미널 잡의 meta.json mtime은 TTL GC의
                # "마지막 갱신" 시계라, 무조건 재저장하면 재시작마다 TTL이 리셋된다.
                if changed:
                    # 중단된 잡의 work/ 잔여물 정리 — runner의 finally(터미널 마감
                    # 시 rmtree)가 돌지 못하고 죽었고, 이 잡은 다시 실행되지 않아
                    # 어느 경로에서도 정리되지 않는다.
                    shutil.rmtree(job.dir / "work", ignore_errors=True)
                    self.save(job)
            except Exception:
                logger.exception("잡 메타 복원 실패: %s", d)


class EventBroker:
    """잡별 SSE 구독 큐 + 실행 중 OCR token 재연결 히스토리.

    느린 구독자의 큐가 가득 차면 token 이벤트를 버리되 그 구독자에 표식을 남긴다 —
    SSE 루프가 표식을 보고 누적 원문 replay로 재동기화한다(조용한 유실 금지:
    토큰 하나가 빠지면 <PAGE> 마커나 <|det|> 절반이 사라져 이후 페이지 귀속이
    영구히 어긋난다). 새 구독은 subscribe_with_replay()로 누적 원문을 한 번 받아
    중간 접속·재연결 갭을 복구한다.

    재처리(rewind)로 서버가 이미 보낸 출력을 폐기할 때는 truncate_token_history()로
    히스토리도 같은 지점까지 되돌린다 — 그러지 않으면 재연결 replay가 폐기된
    출력을 다시 실어 나른다.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[queue.Queue]] = {}
        self._token_history: dict[str, deque[str]] = {}
        self._token_history_chars: dict[str, int] = {}
        self._token_history_truncated: set[str] = set()
        # 잡 시작부터 지금까지 발행한 token 문자 수(절대 오프셋). 앞쪽 절단과
        # 무관하게 단조 증가하므로 rewind 지점을 절대 좌표로 지정할 수 있다.
        self._token_emitted: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _new_queue() -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_EVENT_QUEUE_MAX)
        # 느린 구독자에게서 token을 버렸다는 표식 — SSE 루프가 이 플래그를 보고
        # 누적 원문 replay로 재동기화한다. 버린 채 조용히 넘어가면 <PAGE> 마커나
        # <|det|> 절반이 사라져 이후 페이지 귀속이 영구히 어긋난다.
        q.token_dropped = False
        return q

    def subscribe(self, job_id: str) -> queue.Queue:
        q = self._new_queue()
        with self._lock:
            self._subs.setdefault(job_id, []).append(q)
        return q

    def subscribe_with_replay(self, job_id: str) -> tuple[queue.Queue, str, bool]:
        """구독 등록과 이전 token 스냅샷을 원자적으로 수행한다.

        락 안에서 먼저 히스토리를 복사하고 구독자를 등록한다. publish()도 같은
        락에서 히스토리 갱신과 구독자 스냅샷을 함께 하므로, 경계의 token은
        replay 또는 새 큐 중 정확히 한 곳에 들어간다(중복·유실 없음).
        """
        q = self._new_queue()
        with self._lock:
            replay = "".join(self._token_history.get(job_id, ()))
            truncated = job_id in self._token_history_truncated
            self._subs.setdefault(job_id, []).append(q)
        return q, replay, truncated

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(job_id)
            if subs and q in subs:
                subs.remove(q)
            if subs is not None and not subs:
                del self._subs[job_id]

    def publish(self, job_id: str, event: str, data: dict) -> None:
        # 히스토리 갱신과 구독자 배달을 같은 락 안에서 수행한다 — resync()가
        # "대기 중 token을 비우고 히스토리를 스냅샷"하는 사이에 새 token이 큐로
        # 들어가면 replay와 중복된다. put_nowait는 블로킹하지 않아 안전하다.
        with self._lock:
            if event == "token":
                text = data.get("text")
                if isinstance(text, str) and text:
                    history = self._token_history.setdefault(job_id, deque())
                    history.append(text)
                    self._token_emitted[job_id] = self._token_emitted.get(job_id, 0) + len(text)
                    total = self._token_history_chars.get(job_id, 0) + len(text)
                    while history and total > _TOKEN_HISTORY_MAX_CHARS:
                        total -= len(history.popleft())
                        self._token_history_truncated.add(job_id)
                    self._token_history_chars[job_id] = total
            subs = list(self._subs.get(job_id, ()))
            if event in ("done", "error"):
                self._token_history.pop(job_id, None)
                self._token_history_chars.pop(job_id, None)
                self._token_history_truncated.discard(job_id)
                self._token_emitted.pop(job_id, None)
            for q in subs:
                try:
                    q.put_nowait((event, data))
                except queue.Full:
                    if event == "token":
                        # 유실을 표식으로 남긴다 — SSE 루프가 replay로 되살린다
                        q.token_dropped = True
                        continue
                    try:  # 오래된 것 하나 버리고 재시도
                        evicted = q.get_nowait()
                        if evicted[0] == "token":
                            # 제어 이벤트 자리를 만드느라 밀어낸 token도 유실이다
                            q.token_dropped = True
                        q.put_nowait((event, data))
                    except (queue.Empty, queue.Full):  # pragma: no cover
                        pass

    def truncate_token_history(self, job_id: str, keep_chars: int) -> None:
        """재처리로 폐기한 출력을 재연결 히스토리에서도 되돌린다.

        keep_chars는 잡 시작부터의 **절대** 문자 오프셋이다(BrokerSink가 발행한
        누계와 같은 좌표계). 앞쪽이 상한으로 잘린 잡은 절대 좌표를 복원할 수
        없으므로 건드리지 않는다 — 그런 잡의 replay는 클라이언트가 이미 거부한다.
        """
        with self._lock:
            history = self._token_history.get(job_id)
            emitted = self._token_emitted.get(job_id, 0)
            if history is None or keep_chars >= emitted:
                return
            if job_id in self._token_history_truncated:
                return
            drop = emitted - keep_chars
            while drop > 0 and history:
                last = history[-1]
                if len(last) <= drop:
                    history.pop()
                    drop -= len(last)
                else:
                    history[-1] = last[: len(last) - drop]
                    drop = 0
            self._token_emitted[job_id] = keep_chars
            self._token_history_chars[job_id] = sum(len(x) for x in history)

    def resync(self, job_id: str, q: queue.Queue) -> tuple[str, bool]:
        """유실 표식이 붙은 구독자를 누적 원문으로 되살린다.

        큐에 남은 token 이벤트를 버리고(그 내용은 히스토리에 이미 있다) 히스토리
        스냅샷을 돌려준다. publish()가 같은 락에서 배달하므로 스냅샷과 배달 사이에
        새 token이 끼어들지 않는다 — 중복·유실 없이 정확히 한 번씩만 전달된다.
        """
        with self._lock:
            q.token_dropped = False
            keep: list[tuple[str, dict]] = []
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                # token은 히스토리에 이미 있다. reset도 마찬가지 — truncate가 publish
                # 보다 먼저 같은 락에서 일어나므로 스냅샷이 이미 절단된 상태다. 늦게
                # 배달하면 이미 되돌린 원문을 한 번 더 잘라 멀쩡한 페이지가 사라진다.
                if item[0] not in ("token", "reset"):
                    keep.append(item)
            for item in keep:
                try:
                    q.put_nowait(item)
                except queue.Full:  # pragma: no cover — 방금 비운 큐
                    break
            return (
                "".join(self._token_history.get(job_id, ())),
                job_id in self._token_history_truncated,
            )

    def publish_progress(self, job: Job) -> None:
        self.publish(job.id, "progress", {**job.progress, "status": job.status})


class Worker(threading.Thread):
    """단일 워커: 모델이 프로세스당 1개이므로 잡을 직렬 처리한다."""

    def __init__(
        self,
        store: JobStore,
        broker: EventBroker,
        engine: "OCREngine",
        settings: "Settings",
        cancel_events: dict[str, threading.Event],
    ) -> None:
        super().__init__(name="ocr-worker", daemon=True)
        self.store = store
        self.broker = broker
        self.engine = engine
        self.settings = settings
        self.cancel_events = cancel_events
        self._queue: queue.Queue = queue.Queue()

    def submit(self, job: Job) -> None:
        self.cancel_events.setdefault(job.id, threading.Event())
        self.store.mark_submitted(job)
        self._queue.put(job.id)

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        from .engine.base import JobCanceled
        from .pipeline.runner import execute_job

        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            # 잡 단위 예외 방벽 — execute_job이나 마감 경로(store.save의 OSError 등)에서
            # 예외가 새어 나와도 워커 스레드가 죽으면 안 된다. 죽으면 이후 제출되는
            # 모든 잡이 영구 queued로 남고 프로세스 재시작 외에 복구 수단이 없다.
            # cancel_events 정리는 finally로 일원화한다(모든 종료 경로 공통).
            try:
                job = self.store.get(job_id)
                if job is None:
                    # queued 상태에서 삭제돼 dequeue 시 이미 사라진 잡
                    continue
                cancel = self.cancel_events.setdefault(job_id, threading.Event())
                if job.delete_requested or cancel.is_set():
                    job.status = "canceled"
                    job.error = "사용자에 의해 취소되었습니다"
                    self.store.save(job)
                    if job.delete_requested:
                        self.store.delete_dir(job)
                    continue
                try:
                    if not self.engine.loaded:
                        def _on_wait(note: str, _jid: str = job_id) -> None:
                            # 모델 로딩 대기를 진행 상태로 알린다 — 프론트가 "모델 로딩
                            # 대기 중…"을 표시하고, 잡이 조용히 멈춘 것처럼 보이지 않게 한다.
                            self.broker.publish(_jid, "progress", {
                                "phase": "loading", "status": "queued", "note": note,
                                "current_page": 0, "total_pages": 0,
                                "chunk": 0, "total_chunks": 0,
                            })

                        _on_wait("모델 로딩 대기 중…")
                        self.engine.wait_until_ready(cancel, on_wait=_on_wait)
                except JobCanceled:
                    # 대기 중 사용자가 취소 — 오류가 아니라 취소로 마감
                    job.status = "canceled"
                    job.error = "사용자에 의해 취소되었습니다"
                    self.store.save(job)
                    if job.delete_requested:
                        self.store.delete_dir(job)
                    else:
                        self.broker.publish(
                            job_id, "error", {"message": job.error, "canceled": True}
                        )
                    continue
                except Exception as e:  # noqa: BLE001 — 로드 실패를 잡 오류로 변환
                    logger.exception("엔진 로드 실패")
                    job.status = "error"
                    job.error = f"모델 로드 실패: {e}"[:2000]
                    self.store.save(job)
                    self.broker.publish(job_id, "error", {"message": job.error})
                    continue
                execute_job(job, self.store, self.broker, self.engine, self.settings, cancel)
            except Exception:  # noqa: BLE001 — 워커 스레드 영구 정지 방지
                logger.exception("잡 처리 중 예기치 못한 오류: %s", job_id)
                # 메모리 상 running으로 남으면 DELETE도 거부돼(api의 running 가드)
                # 사용자가 치울 수 없다 — 터미널(error)로 마감한다.
                stuck = self.store.get(job_id)
                if stuck is not None and stuck.status in ("queued", "running"):
                    stuck.status = "error"
                    stuck.error = "잡 처리 중 내부 오류가 발생했습니다"
                    self.store.save(stuck)
                    self.broker.publish(job_id, "error", {"message": stuck.error})
            finally:
                self.cancel_events.pop(job_id, None)
