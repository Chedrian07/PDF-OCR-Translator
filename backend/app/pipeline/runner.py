"""잡 실행 오케스트레이션: 렌더 → 청크 OCR → 병합. 워커 스레드에서 호출된다."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..engine.base import EngineError, JobCanceled, OCREngine, RepetitiveOutputError
from .fidelity import PageFidelity, evaluate_layout_pages, evaluate_raw_page
from .layout import blocks_to_raw
from .merge import ChunkResult, IncrementalMerger
from .pdf import extract_embedded_page_markdown, render_pdf_pages

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..jobs import EventBroker, Job, JobStore

logger = logging.getLogger(__name__)

_TOKEN_FLUSH_CHARS = 256
_TOKEN_FLUSH_SECS = 0.1
_PAGE_MARKER = "<PAGE>"
# 잡 단위 경고 상한 — 엔진 경고는 청크(기본 1페이지)마다 나올 수 있어 상한이 없으면
# meta.json이 비대해지고 5초 주기 GET /api/jobs 응답까지 부풀린다.
_MAX_JOB_WARNINGS = 200
# 충실도 재처리 채택에 요구하는 최소 개선폭. 0.01 수준의 요동으로 페이지를
# 갈아끼우면 라이브 뷰만 흔들리고 얻는 것이 없다. 실측 회수 사례는 전부
# 0.45→0.97처럼 큰 폭이라 0.05는 넉넉히 통과한다.
_FIDELITY_ACCEPT_MARGIN = 0.05


def _retry_fidelity(source_pdf: Path, page_dir: Path, page: int) -> "PageFidelity | None":
    """단독 재처리 산출물(raw_pages.json)을 원본 대비로 판정한다.

    벤더는 `{"pages": ["<|det|>…"]}` 형식으로 원출력을 남긴다(패치 P14).
    파일이 없거나 깨졌으면 None — 채택하지 않는다(원래 결과를 지킨다).
    """
    import json

    raw_file = page_dir / "raw_pages.json"
    try:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        pages = payload.get("pages") if isinstance(payload, dict) else payload
        raw = str(pages[0]) if pages else ""
    except Exception:  # noqa: BLE001 — 판정 불가는 '미채택'으로 흡수된다
        return None
    if not raw.strip():
        return None
    return evaluate_raw_page(source_pdf, raw, page)


def _partial_marker_suffix(text: str) -> int:
    """<PAGE>의 진부분 접두사로 끝나면 그 길이 — 델타 경계에 걸친 마커 보류용."""
    for size in range(min(len(text), len(_PAGE_MARKER) - 1), 0, -1):
        if text.endswith(_PAGE_MARKER[:size]):
            return size
    return 0


def _page_span(start_page: int, num_pages: int) -> str:
    end = start_page + num_pages - 1
    return f"{start_page}페이지" if num_pages == 1 else f"{start_page}–{end}페이지"


class BrokerSink:
    """엔진 토큰 스트림 → SSE token 이벤트(코얼레싱) + **서버가 소유하는** 페이지 프레이밍.

    라이브 3패널(레이아웃 박스·RAW·미리보기)은 전부 이 스트림 하나만 보고 페이지를
    센다. 그래서 "스트림의 k번째 <PAGE> 세그먼트 == 글로벌 페이지 k" 불변식을
    모델이 아니라 서버가 보장한다:

    - multi 호출: 모델이 페이지마다 <PAGE>를 낸다 → 통과시키며 카운트
    - single 호출(per_page 모드·폴백 재처리): 모델은 마커를 내지 않는다 →
      서버가 페이지 시작에 하나 주입하고 모델이 흘린 마커는 제거한다
    - 합성 페이지(텍스트 레이어 복구·실패 플레이스홀더): emit_page()로 직접 프레이밍

    이 불변식이 없으면 폴백 재처리 동안 진행률이 멈추고(마커가 없어 페이지가
    올라가지 않음) 이후 모든 박스가 한 페이지에 쌓인다. 폐기(재처리)에는
    rewind_to()가 reset 이벤트 + 토큰 히스토리 절단을 함께 낸다.
    """

    def __init__(self, job: "Job", store: "JobStore", broker: "EventBroker") -> None:
        self._job = job
        self._store = store
        self._broker = broker
        self._buf: list[str] = []
        self._buf_len = 0
        self._last_flush = time.monotonic()
        self._marker_tail = ""
        self._chunk_start = 1
        self._chunk_pages = 1
        self._pages_seen = 0
        self._expect_markers = True
        self._need_marker = False
        # 발행한 token 문자 누계(EventBroker._token_emitted와 같은 좌표계) +
        # 페이지 시작 오프셋 — rewind_to가 정확히 그 지점까지 되돌린다.
        self._emitted = 0
        self._marks: dict[int, int] = {}

    def _drain_marker_tail(self) -> None:
        """완성되지 않은 채 남은 부분 마커 조각을 본문으로 확정한다."""
        if self._marker_tail:
            tail, self._marker_tail = self._marker_tail, ""
            self._append(tail)

    def set_chunk(
        self, start_page: int, *, expect_markers: bool = True, num_pages: int = 1
    ) -> None:
        """다음 엔진 호출이 만들 첫 글로벌 페이지와 페이지 수를 선언한다.

        expect_markers=False면 모델이 <PAGE>를 내지 않는 호출(run_single)이라
        서버가 대신 주입한다. num_pages는 마커 과부족을 보정하는 기준이다 —
        merge도 같은 기준으로 result.md 페이지 수를 맞추므로(split_pages 초과분
        병합·부족분 빈 페이지) 라이브와 최종 산출물의 페이지 경계가 일치한다."""
        self._drain_marker_tail()
        self.flush()
        self._chunk_start = start_page
        self._chunk_pages = max(1, num_pages)
        self._pages_seen = 0
        self._marker_tail = ""
        self._expect_markers = expect_markers
        self._need_marker = not expect_markers
        self._marks[start_page] = self._emitted

    def finish_chunk(self) -> None:
        """모델이 낸 마커가 청크의 페이지 수보다 적으면 모자란 만큼 채운다.

        merge가 부족분을 빈 페이지로 보정하는 것과 같은 계약 — 채우지 않으면
        이후 모든 페이지의 세그먼트 인덱스가 밀려 박스가 엉뚱한 페이지에 붙는다."""
        self._drain_marker_tail()
        while self._pages_seen < self._chunk_pages:
            self._pages_seen += 1
            self._begin_page(self._chunk_start + self._pages_seen - 1)
            self._append(_PAGE_MARKER)
        self.flush()

    def on_text(self, text: str) -> None:
        if not text:
            return
        # 델타 경계에 걸친 마커(`<PA` + `GE>`)는 완성될 때까지 보류한다 — 스트림에
        # 내보내지도, 마커로 세지도 않는다. 최대 5자라 지연은 무시할 수준이다.
        data = self._marker_tail + text
        self._marker_tail = ""
        hold = _partial_marker_suffix(data)
        if hold:
            self._marker_tail = data[-hold:]
            data = data[:-hold]
        if not data:
            return

        if not self._expect_markers:
            # single 출력에 섞여 나온 마커는 프레이밍을 깨뜨린다 — 제거하고
            # 페이지 시작 마커는 서버가 정확히 하나만 넣는다.
            data = data.replace(_PAGE_MARKER, "")
            if self._need_marker:
                self._need_marker = False
                self._pages_seen += 1  # finish_chunk가 또 채우지 않도록 함께 센다
                self._begin_page(self._chunk_start)
                self._append(_PAGE_MARKER)
            if data:
                self._append(data)
            return

        while True:
            idx = data.find(_PAGE_MARKER)
            if idx == -1:
                break
            head, data = data[:idx], data[idx + len(_PAGE_MARKER):]
            if head:
                self._append(head)
            if self._pages_seen >= self._chunk_pages:
                # 청크 페이지 수를 넘는 초과 마커는 스트림에서 지운다 — merge가
                # 초과분을 마지막 페이지에 합치는 것과 같은 계약이라 라이브와
                # 최종 result.md의 페이지 경계가 어긋나지 않는다.
                continue
            # 마커 앞의 꼬리는 **이전** 페이지 몫이다. 다음 페이지 선언(progress)
            # 보다 먼저 내보내야 클라이언트가 그 꼬리의 박스를 이전 페이지에 붙인다.
            self.flush()
            self._pages_seen += 1
            self._begin_page(self._chunk_start + self._pages_seen - 1)
            self._append(_PAGE_MARKER)
        if data:
            self._append(data)

    def _begin_page(self, page: int) -> None:
        """페이지 시작을 진행률로 알리고 **되감기 지점을 기록**한다.

        진행률 선언은 마커 토큰보다 먼저 나가야 한다 — 클라이언트는 선언 직후의
        첫 마커를 재확인(no-op)으로 소비하므로 이 순서가 곧 페이지 귀속 계약이다
        (frontend/js/core.js groundAnnounce).

        마크를 여기서 남기는 이유: 예전에는 `set_chunk`만 마크를 남겨 **청크의 첫
        페이지에만** 되감기 지점이 있었다. 그래서 8쪽 청크 한가운데(33–40쪽의 38쪽)를
        `rewind_to`로 물리려 하면 아무 이벤트도 내지 않고 조용히 no-op이 됐고, 그 뒤
        `emit_page`를 부르면 세그먼트가 하나 늘어 "k번째 <PAGE> == 페이지 k" 불변식이
        깨진다(이후 전 페이지의 박스 귀속이 밀린다). 충실도 게이트는 청크 중간
        페이지를 교체해야 하므로 모든 페이지에 마크가 필요하다.

        먼저 flush하는 이유: 마크는 "이 페이지가 시작하는 발행 오프셋"이어야 한다.
        버퍼가 남아 있으면 마크가 앞 페이지의 꼬리를 가리켜 되감기가 그것까지 지운다.
        (`finish_chunk`의 채움 경로가 정확히 그 상태로 이 함수를 부른다.)
        """
        self.flush()
        # setdefault — `set_chunk`가 이미 남긴 청크 시작 마크를 덮어쓰지 않는다.
        self._marks.setdefault(page, self._emitted)
        total = max(self._job.progress.get("total_pages", 1), 1)
        current = min(page, total)
        if current > self._job.progress.get("current_page", 0):
            self._job.progress["current_page"] = current
            self._broker.publish_progress(self._job)

    def _append(self, text: str) -> None:
        self._buf.append(text)
        self._buf_len += len(text)
        now = time.monotonic()
        if self._buf_len >= _TOKEN_FLUSH_CHARS or (now - self._last_flush) >= _TOKEN_FLUSH_SECS:
            self.flush()

    def emit_page(self, page: int, text: str) -> None:
        """스트리밍 없이 만들어진 페이지를 라이브 스트림에도 같은 프레이밍으로 넣는다.

        텍스트 레이어 복구·실패 플레이스홀더가 대상이다 — 세그먼트를 건너뛰면
        이후 모든 페이지의 프리뷰 경계와 박스 귀속이 한 칸씩 밀린다."""
        self.set_chunk(page, expect_markers=False)
        self.on_text(text or " ")
        self.finish_chunk()  # 본문이 비어도 세그먼트는 반드시 하나 남긴다

    def rewind_to(self, page: int, reason: str) -> None:
        """이미 보낸 page 이후의 출력을 폐기하라고 클라이언트에 알린다.

        서버가 work_dir를 지우고 같은 페이지를 다시 처리할 때 호출한다. 클라이언트의
        원문은 append-only라 알려주지 않으면 폐기된 출력이 영원히 남는다(박스 중복,
        잘린 <table>이 뒤 내용을 통째로 삼킨 미리보기, RAW 중복). 재연결 replay가
        같은 쓰레기를 다시 싣지 않도록 토큰 히스토리도 함께 되돌린다."""
        self._drain_marker_tail()
        self.flush()
        mark = self._marks.get(page)
        if mark is None or mark >= self._emitted:
            # 시작 지점을 모르거나(잘못 자르느니 그대로 둔다) 아직 이 페이지의
            # 출력이 나가지 않았다 — 물릴 것이 없으니 잡음도 내지 않는다.
            return
        for stale in [p for p in self._marks if p > page]:
            del self._marks[stale]
        self._pages_seen = 0
        self._need_marker = not self._expect_markers
        self._broker.truncate_token_history(self._job.id, mark)
        self._emitted = mark
        self._broker.publish(
            self._job.id, "reset", {"from_page": page, "reason": reason}
        )
        prev = max(0, page - 1)
        if self._job.progress.get("current_page", 0) > prev:
            self._job.progress["current_page"] = prev
            self._broker.publish_progress(self._job)

    def flush(self) -> None:
        if self._buf:
            payload = "".join(self._buf)
            self._emitted += len(payload)
            self._broker.publish(self._job.id, "token", {"text": payload})
            self._buf = []
            self._buf_len = 0
        self._last_flush = time.monotonic()


def _chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


_FAILED_PAGE_MD = "> ⚠️ 이 페이지는 변환에 실패했습니다"


def _empty_device_cache() -> None:
    """실패한 청크 재시도 전 디바이스 캐시 반환 — OOM류 실패 후 가용 메모리 복구.
    (unlimited.py의 _release_device_cache와 동일한 best-effort empty_cache 패턴)"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover — 방어적 (torch 미설치 등)
        pass


def _add_failed_chunk(
    merger: IncrementalMerger,
    work_dir: Path,
    start_page: int,
    num_pages: int,
    single: bool,
    err: Exception,
    sink: "BrokerSink | None" = None,
) -> None:
    """재시도까지 실패한 청크를 기대 페이지 수만큼의 플레이스홀더로 보정.

    merge의 <PAGE> 계약(청크당 num_pages개 페이지)을 그대로 지켜 글로벌 페이지
    번호 정합을 유지하고, warnings에 남긴 뒤 다음 청크로 계속하게 한다.
    라이브 스트림에도 같은 수의 페이지 세그먼트를 넣어 result.md와 프레이밍을
    맞춘다 — 건너뛰면 이후 페이지의 박스·프리뷰 경계가 통째로 밀린다."""
    work_dir.mkdir(parents=True, exist_ok=True)  # 엔진이 만들기 전에 실패했을 수 있음
    if single:
        md = _FAILED_PAGE_MD
    else:
        md = "<PAGE>\n" + "\n<PAGE>\n".join([_FAILED_PAGE_MD] * num_pages)
    merger.add_chunk(ChunkResult(work_dir, start_page, num_pages, md, single=single))
    if sink is not None:
        for offset in range(1 if single else num_pages):
            sink.emit_page(start_page + offset, _FAILED_PAGE_MD)
    span = _page_span(start_page, num_pages)
    merger.warnings.append(
        f"{span}: 변환 실패로 플레이스홀더 삽입 ({err.__class__.__name__}: {str(err)[:200]})"
    )


def execute_job(
    job: "Job",
    store: "JobStore",
    broker: "EventBroker",
    engine: OCREngine,
    settings: "Settings",
    cancel: threading.Event,
) -> None:
    sink = BrokerSink(job, store, broker)
    merger: IncrementalMerger | None = None
    try:
        job.status = "running"
        # 엔진/모델 메타 확정 — 업로드 시점에는 sidecar health(revision 등)를 모를 수
        # 있으므로, 엔진이 로드된 실행 시작 시점에 실제 사용값으로 갱신한다.
        start_caps = engine.capabilities()
        job.engine = engine.name
        job.model_id = start_caps.model_id or job.model_id
        job.model_revision = start_caps.model_revision or job.model_revision
        job.provider = start_caps.provider or job.provider
        job.progress.update(phase="render", current_page=0, chunk=0, total_chunks=0)
        store.save(job)
        broker.publish_progress(job)

        def _render_cb(done: int, total: int) -> None:
            # 렌더 단계에서도 페이지 단위로 취소/삭제에 반응한다 — 대형 문서(수백 p)
            # 렌더가 끝날 때까지 취소가 무시되지 않게. 예외는 render_pdf_pages를
            # 관통해 아래 JobCanceled 핸들러로 떨어진다.
            if cancel.is_set():
                raise JobCanceled()
            job.progress.update(current_page=done, total_pages=total)
            broker.publish_progress(job)

        pages = render_pdf_pages(
            job.dir / "source.pdf", job.dir / "pages", job.dpi, settings.max_pages, _render_cb
        )
        if cancel.is_set():
            raise JobCanceled()

        total = len(pages)
        caps = engine.capabilities()
        if job.mode == "per_page":
            chunk_size = 1
        else:
            # 엔진 capability가 청크 크기를 결정한다: 페이지 단위 sidecar 엔진은
            # preferred_chunk_size(=OCR_REMOTE_PAGE_CONCURRENCY, 기본 1),
            # Unlimited는 None → 기존 PAGES_PER_CHUNK.
            chunk_size = caps.preferred_chunk_size or settings.pages_per_chunk
        chunk_size = max(1, chunk_size)
        chunks = _chunked(pages, chunk_size)
        job.progress.update(total_pages=total, total_chunks=len(chunks), current_page=0)
        store.save(job)

        merger = IncrementalMerger(job.dir, settings.page_separator)
        # 충실도 게이트의 단독 재처리 예산. 스캔·손상 문서에서 전 페이지를 다시
        # 돌려 런타임이 폭발하지 않게 막는다. 짧은 문서도 복구할 수 있게 최소 2쪽.
        retry_budget = [
            max(2, int(total * max(0.0, settings.ocr_fidelity_max_retry_ratio)))
        ]
        engine.drain_warnings()  # 이전 잡의 잔여 경고 폐기 (엔진은 잡 간 공유된다)
        if job.mode != "per_page" and not caps.supports_multi_page:
            # 페이지 단위 모델 안내 — 한 번만 기록 (multi를 선택해도 정상 처리되지만
            # 내부적으로는 페이지별 추론이며 오류도 페이지 단위로 격리된다)
            merger.warnings.append(
                f"{engine.name} 엔진은 페이지 단위 모델이라 문서를 페이지별로 처리했습니다"
                " (결과는 동일하게 하나의 Markdown으로 병합됨)"
            )

        def _try_embedded_text_fallback(
            page_number: int,
            recovery_dir: Path,
            error: Exception,
        ) -> bool:
            """single OCR 최종 실패 페이지를 원본 PDF 텍스트 레이어로 복구."""
            if cancel.is_set():
                raise JobCanceled()
            # 실패한 single 호출이 만든 crop/layout/raw 산출물은 절대 병합하지 않는다.
            shutil.rmtree(recovery_dir, ignore_errors=True)
            page_md = extract_embedded_page_markdown(
                job.dir / "source.pdf", page_number
            )
            if cancel.is_set():
                raise JobCanceled()
            if page_md is None:
                return False
            recovery_dir.mkdir(parents=True, exist_ok=True)
            if cancel.is_set():
                raise JobCanceled()
            merger.add_chunk(
                ChunkResult(recovery_dir, page_number, 1, page_md, single=True)
            )
            # 실패한 OCR 시도가 라이브로 흘려보낸 출력을 물리고, 복구된 본문을
            # 같은 페이지 세그먼트로 다시 넣는다.
            sink.rewind_to(page_number, "PDF 텍스트 레이어로 복구")
            sink.emit_page(page_number, page_md)
            message = (
                f"{page_number}페이지: single OCR 실패 후 PDF 내장 텍스트 레이어로 복구 "
                f"(이미지·정밀 레이아웃 제외; {error.__class__.__name__}: "
                f"{str(error)[:160]})"
            )
            merger.warnings.append(message)
            logger.warning("%s", message)
            return True

        done_pages = 0
        failed_chunks = 0
        last_chunk_error: Exception | None = None
        for ci, chunk in enumerate(chunks):
            if cancel.is_set():
                raise JobCanceled()
            start_page = done_pages + 1
            job.progress.update(phase="ocr", chunk=ci + 1, current_page=start_page)
            store.save(job)
            broker.publish_progress(job)

            work_dir = job.dir / "work" / f"chunk_{ci:02d}"

            def _run_engine() -> str:
                # run_single은 <PAGE>를 내지 않는다 — 서버가 주입해야 라이브 뷰의
                # 페이지 프레이밍(진행률·박스 귀속·프리뷰 경계)이 유지된다.
                sink.set_chunk(
                    start_page,
                    expect_markers=job.mode != "per_page",
                    num_pages=1 if job.mode == "per_page" else len(chunk),
                )
                if job.mode == "per_page":
                    return engine.run_single(chunk[0], work_dir, sink, cancel)
                return engine.run_multi(chunk, work_dir, sink, cancel)

            def _run_with_retry(
                run: Callable[[], str],
                context: str,
                *,
                reset_output: Callable[[], None] | None = None,
                rewind_page: int | None = None,
            ) -> str:
                """엔진 호출을 1회 재시도하되 의미 반복은 즉시 호출자에게 넘긴다.

                반복 감지는 재시도해도 같은 내용에서 재발하므로 재시도 대상이
                아니다 — 복구(per_page 강등·텍스트 레이어 폴백)는 호출자 몫."""
                try:
                    return run()
                except JobCanceled:
                    raise
                except RepetitiveOutputError:
                    if cancel.is_set():
                        raise JobCanceled() from None
                    raise
                except Exception as error:  # noqa: BLE001 — 청크 단위 격리
                    first_error = error

                logger.warning(
                    "%s 실패 (%s: %s) — 캐시 해제 후 1회 재시도",
                    context,
                    first_error.__class__.__name__,
                    str(first_error)[:200],
                )
                _empty_device_cache()
                if cancel.is_set():
                    raise JobCanceled() from None
                if reset_output is not None:
                    reset_output()
                if rewind_page is not None:
                    # 산출물과 함께 라이브 스트림도 되돌린다 — 첫 시도가 흘려보낸
                    # 부분 출력이 남으면 재시도분과 중복된다.
                    sink.rewind_to(rewind_page, f"{context} 재시도")
                try:
                    return run()
                except JobCanceled:
                    raise
                except Exception as retry_error:  # noqa: BLE001 — 청크 단위 격리
                    if cancel.is_set():
                        raise JobCanceled() from None
                    logger.warning(
                        "%s 재시도 실패 (%s: %s)",
                        context,
                        retry_error.__class__.__name__,
                        str(retry_error)[:200],
                    )
                    raise

            def _repair_low_fidelity_pages(
                start_page: int, chunk_pages: list[Path], work_dir: Path,
            ) -> None:
                """청크 병합 직후, 원본 텍스트 레이어와 대조해 열화 페이지만 재처리.

                멀티페이지 추론은 처리량을 위해 8쪽을 한 컨텍스트에 넣는데, 모델은
                sliding_window=128의 12층 MoE라 그 범위에 걸친 구조 기록을 유지할
                수단이 없다. 실측(46쪽 논문): p34·p39가 프로덕션에서 **0자**,
                p38이 0.409였는데 **같은 페이지를 단독 실행하면 0.945/0.972/0.971**이다.
                해상도 문제가 아니다(타일 모드는 +0.01~0.02) — 청크에서 빼내는 것이 번다.

                born-digital PDF에서는 PyMuPDF 텍스트 레이어가 공짜 정답이므로,
                열화를 정확히 탐지할 수 있다. 재시도 결과가 **실제로 더 나을 때만**
                채택하므로 오탐의 대가는 정확성이 아니라 시간이다.
                """
                threshold = settings.ocr_fidelity_threshold
                if threshold <= 0 or job.mode == "per_page":
                    return
                # 게이트가 원리상 도움이 될 수 없는 엔진은 건너뛴다.
                #  · 텍스트 bbox를 안 주는 엔진(figure_only/none)은 블록 내용이 비어
                #    전 페이지가 0.00으로 나온다 — 전량 오탐이다.
                #  · 이미 페이지 단위로 도는 엔진은 재처리가 **같은 호출**이라 개선 여지가
                #    없다(중복 추론만 늘린다).
                if caps.layout_capability != "full":
                    return
                if not caps.supports_multi_page or (caps.preferred_chunk_size or 0) == 1:
                    return
                source_pdf = job.dir / "source.pdf"
                if not source_pdf.is_file():
                    return
                span_pages = range(start_page, start_page + len(chunk_pages))
                layout = [
                    lp for lp in merger.layout_pages
                    if int(lp.get("page") or 0) in span_pages
                ]
                if not layout:
                    return
                scored = {
                    r.page: r for r in evaluate_layout_pages(source_pdf, layout)
                }
                degraded = sorted(
                    pno for pno, r in scored.items()
                    if r.measurable and r.score < threshold
                )
                if not degraded:
                    return
                if retry_budget[0] <= 0:
                    merger.warnings.append(
                        f"{_page_span(start_page, len(chunk_pages))}: 충실도 미달 "
                        f"{len(degraded)}쪽을 발견했지만 재처리 예산을 모두 썼습니다 "
                        "(OCR_FIDELITY_MAX_RETRY_RATIO)"
                    )
                    return
                if len(degraded) > retry_budget[0]:
                    # 상한에 걸려 버리는 페이지를 조용히 넘기지 않는다 — 그러면
                    # "전부 검사했다"로 읽히지만 실제로는 손대지 못한 페이지가 남는다.
                    skipped = degraded[retry_budget[0] :]
                    merger.warnings.append(
                        f"충실도 미달이지만 재처리 예산이 모자라 건너뛴 페이지: "
                        f"{', '.join(str(pno) for pno in skipped)} "
                        "(OCR_FIDELITY_MAX_RETRY_RATIO로 상한 조정)"
                    )
                    degraded = degraded[: retry_budget[0]]
                first = degraded[0]

                def _relive_text(page_number: int) -> str:
                    """이미 병합한 페이지를 라이브에 **다시** 내보낼 때 쓸 본문.

                    병합된 마크다운을 그대로 흘리면 `<|det|>` 태그가 없어 그 페이지들의
                    레이아웃 박스가 라이브 뷰에서 사라진다(RAW·미리보기만 남는다).
                    layout 블록에서 grounding을 복원해 박스 귀속을 지킨다.
                    """
                    for entry in merger.layout_pages:
                        if int(entry.get("page") or 0) == page_number:
                            restored = blocks_to_raw(entry.get("blocks") or [])
                            if restored.strip():
                                return restored
                            break
                    return merger.pages_md[page_number - 1]

                # 라이브 뷰는 페이지 단위로만 되감을 수 있다. 첫 열화 페이지까지
                # 물린 뒤, 그 뒤 정상 페이지는 병합된 마크다운으로 다시 내보낸다.
                sink.rewind_to(first, "충실도 게이트 — 열화 페이지 재처리")
                end_page = start_page + len(chunk_pages)

                def _refill(from_page: int) -> None:
                    """되감은 구간 중 아직 못 채운 페이지를 병합본으로 복원한다."""
                    for rest in range(from_page, end_page):
                        sink.emit_page(rest, _relive_text(rest))

                for pno in range(first, end_page):
                    if cancel.is_set():
                        # 되감아 놓고 그냥 끝내면 라이브 뷰에서 그 뒤 페이지가 사라진다.
                        _refill(pno)
                        raise JobCanceled()
                    if pno not in degraded:
                        sink.emit_page(pno, _relive_text(pno))
                        continue
                    retry_budget[0] -= 1
                    before = scored[pno]
                    page_dir = work_dir / "fidelity" / f"page_{pno:04d}"
                    shutil.rmtree(page_dir, ignore_errors=True)
                    accepted = False
                    note = ""
                    try:
                        sink.set_chunk(pno, expect_markers=False)
                        page_md = engine.run_single(
                            chunk_pages[pno - start_page], page_dir, sink, cancel
                        )
                    except JobCanceled:
                        raise
                    except Exception as retry_error:  # noqa: BLE001 — 페이지 격리
                        note = (
                            f"재처리 실패({retry_error.__class__.__name__}) — 원래 결과 유지"
                        )
                        logger.warning(
                            "%d페이지 충실도 재처리 실패: %s", pno, retry_error
                        )
                    else:
                        after = _retry_fidelity(source_pdf, page_dir, pno)
                        if (
                            after is not None
                            and after.measurable
                            and after.score >= before.score + _FIDELITY_ACCEPT_MARGIN
                        ):
                            sink.finish_chunk()
                            merger.replace_page(
                                pno,
                                ChunkResult(page_dir, pno, 1, page_md, single=True),
                            )
                            accepted = True
                            note = (
                                f"충실도 {before.score:.2f} → {after.score:.2f}로 "
                                "개선되어 단독 재처리 결과를 채택"
                            )
                        else:
                            got = (
                                f"{after.score:.2f}"
                                if after is not None and after.measurable
                                else "판정 불가"
                            )
                            note = (
                                f"충실도 {before.score:.2f} → {got} — 개선되지 않아 "
                                "원래 결과 유지"
                            )
                    if not accepted:
                        shutil.rmtree(page_dir, ignore_errors=True)
                        sink.rewind_to(pno, "재처리 결과 미채택")
                        sink.emit_page(pno, _relive_text(pno))
                    merger.warnings.append(f"{pno}페이지: {note}")
                    logger.info("%d페이지 충실도 게이트: %s", pno, note)

            def _recover_unsafe_generation_chunk(
                generation_error: RepetitiveOutputError,
            ) -> tuple[bool, Exception]:
                """반복/상한 초과 multi 산출물을 버리고 페이지별 single 재처리."""
                if cancel.is_set():
                    raise JobCanceled()
                span = _page_span(start_page, len(chunk))
                logger.warning(
                    "%s multi OCR 비정상 생성 감지 — 페이지별 재처리: %s",
                    span,
                    str(generation_error)[:200],
                )
                merger.warnings.append(
                    f"{span}: 반복/출력 상한 감지로 페이지별 재처리 "
                    f"({str(generation_error)[:200]})"
                )

                # multi와 single은 이미지/레이아웃 파일명 규약이 다르다. 부분 multi
                # 결과를 병합하지 않도록 제거하고 페이지마다 독립 디렉터리를 쓴다.
                shutil.rmtree(work_dir, ignore_errors=True)
                # 라이브 스트림도 청크 시작으로 되돌린다 — 폐기한 multi 출력이
                # 남으면 재처리분과 중복되고(박스 2배), 상한에서 잘린 <table>이
                # 뒤 내용을 통째로 삼킨 미리보기가 그대로 굳는다.
                sink.rewind_to(start_page, "반복/출력 상한 감지 — 페이지별 재처리")
                failed_pages = 0
                last_error: Exception = generation_error
                for local_page, image_path in enumerate(chunk):
                    if cancel.is_set():
                        raise JobCanceled()
                    global_page = start_page + local_page
                    page_dir = work_dir / "fallback" / f"page_{local_page:02d}"

                    def _run_page() -> str:
                        sink.set_chunk(global_page, expect_markers=False)
                        return engine.run_single(image_path, page_dir, sink, cancel)

                    try:
                        page_md = _run_with_retry(
                            _run_page,
                            f"{global_page}페이지 fallback OCR",
                            reset_output=lambda directory=page_dir: shutil.rmtree(
                                directory, ignore_errors=True
                            ),
                            rewind_page=global_page,
                        )
                    except JobCanceled:
                        raise
                    except Exception as page_error:  # noqa: BLE001 — 페이지 단위 격리
                        last_error = page_error
                        if not _try_embedded_text_fallback(
                            global_page, page_dir, page_error
                        ):
                            failed_pages += 1
                            sink.rewind_to(global_page, "페이지 변환 실패")
                            _add_failed_chunk(
                                merger, page_dir, global_page, 1, True, page_error, sink
                            )
                    else:
                        # 토큰을 흘리지 않는 엔진에서도 세그먼트가 빠지지 않게
                        sink.finish_chunk()
                        merger.add_chunk(
                            ChunkResult(page_dir, global_page, 1, page_md, single=True)
                        )
                return failed_pages == len(chunk), last_error

            # 청크 단위 격리: 한 청크가 죽어도(OOM·벤더 예외 등) 잡 전체를 죽이지
            # 않는다 — 캐시 해제 후 1회 재시도, 그래도 실패하면 플레이스홀더로
            # 보정하고 계속. 취소(JobCanceled)는 절대 삼키지 않고 그대로 전파.
            # 재시도는 엔진 실행만 감싼다 — add_chunk는 비멱등(pages_md 확장)이라
            # 재시도에 포함하면 병합 도중 실패 시 페이지가 중복 병합될 수 있다.
            md: str | None = None
            try:
                md = _run_with_retry(
                    _run_engine,
                    f"청크 {ci + 1}/{len(chunks)}",
                    reset_output=lambda: shutil.rmtree(work_dir, ignore_errors=True),
                    rewind_page=start_page,
                )
            except JobCanceled:
                raise
            except RepetitiveOutputError as repetition_error:
                if job.mode != "per_page":
                    all_failed, last_error = _recover_unsafe_generation_chunk(
                        repetition_error
                    )
                    if all_failed:
                        failed_chunks += 1
                        last_chunk_error = last_error
                else:
                    if not _try_embedded_text_fallback(
                        start_page, work_dir, repetition_error
                    ):
                        failed_chunks += 1
                        last_chunk_error = repetition_error
                        sink.rewind_to(start_page, "반복/출력 상한 감지")
                        _add_failed_chunk(
                            merger, work_dir, start_page, len(chunk), True,
                            repetition_error, sink,
                        )
            except Exception as chunk_error:  # noqa: BLE001 — 청크 단위 격리
                recovered = job.mode == "per_page" and _try_embedded_text_fallback(
                    start_page, work_dir, chunk_error
                )
                if not recovered:
                    logger.warning(
                        "청크 %d/%d 최종 실패 (%s: %s) — 플레이스홀더로 보정 후 계속",
                        ci + 1,
                        len(chunks),
                        chunk_error.__class__.__name__,
                        str(chunk_error)[:200],
                    )
                    failed_chunks += 1
                    last_chunk_error = chunk_error
                    shutil.rmtree(work_dir, ignore_errors=True)
                    sink.rewind_to(start_page, "청크 변환 실패")
                    _add_failed_chunk(
                        merger,
                        work_dir,
                        start_page,
                        len(chunk),
                        job.mode == "per_page",
                        chunk_error,
                        sink,
                    )
            # 엔진이 남긴 정화/절단 경고를 잡 warnings로 승격 — 내용이 빠졌는데
            # 조용히 done이 되지 않게 한다 (sidecar 엔진의 bbox 폐기·상한 절단 등).
            # 페이지 범위는 여기서 붙인다 (엔진은 전역 번호를 모른다).
            span = _page_span(start_page, len(chunk))
            for warning in engine.drain_warnings():
                if len(merger.warnings) < _MAX_JOB_WARNINGS:
                    merger.warnings.append(f"{span}: {warning}")
                elif len(merger.warnings) == _MAX_JOB_WARNINGS:
                    merger.warnings.append(
                        f"경고가 {_MAX_JOB_WARNINGS}건을 넘어 이후 항목은 생략됩니다 "
                        "(서버 로그에서 전체 확인)"
                    )
                    logger.warning("잡 %s: 경고 상한 도달 — 이후 경고는 로그에만 남습니다", job.id)
                else:
                    logger.warning("잡 %s 경고(생략됨): %s: %s", job.id, span, warning)
            if md is not None:
                # 모델이 마커를 덜 냈으면 채워 세그먼트 수를 페이지 수에 맞춘다
                sink.finish_chunk()
                # 병합은 엔진 성공 시 1회만 — 병합 실패는 잡 레벨 IO 오류로 취급한다.
                merger.add_chunk(
                    ChunkResult(work_dir, start_page,
                                1 if job.mode == "per_page" else len(chunk),
                                md, single=job.mode == "per_page")
                )
                # 복구는 **선택적 개선**이다 — 여기서 난 IO 오류가 이미 완주한 잡을
                # error로 만들면 안 된다. 취소만 그대로 전파한다.
                try:
                    _repair_low_fidelity_pages(start_page, chunk, work_dir)
                except JobCanceled:
                    raise
                except Exception as gate_error:  # noqa: BLE001 — 게이트 단위 격리
                    logger.warning(
                        "충실도 게이트 실패 (%s: %s) — 원래 결과를 유지하고 계속",
                        gate_error.__class__.__name__, str(gate_error)[:200],
                    )
                    merger.warnings.append(
                        f"{_page_span(start_page, len(chunk))}: 충실도 게이트가 실패해 "
                        f"건너뛰었습니다 ({gate_error.__class__.__name__})"
                    )
            sink.flush()
            # 취소돼도 이 청크의 부분 출력까지는 병합 후에 중단한다
            if cancel.is_set():
                raise JobCanceled()

            done_pages += len(chunk)
            # 단조 가드 — 마커 과잉 생성으로 sink가 선행시킨 진행률을 되돌리지 않는다
            job.progress["current_page"] = max(
                job.progress.get("current_page", 0), done_pages
            )
            store.save(job)
            broker.publish_progress(job)

        # 전 청크 실패면 부분 성공이 없으므로 기존대로 잡 오류로 마감
        if chunks and failed_chunks == len(chunks):
            raise EngineError(
                f"모든 청크({len(chunks)}개) 변환에 실패했습니다: {last_chunk_error}"
            ) from last_chunk_error

        job.progress["phase"] = "merge"
        broker.publish_progress(job)
        merger.finalize()
        job.warnings = merger.warnings
        job.status = "done"
        job.error = None
        store.save(job)
        broker.publish(
            job.id,
            "done",
            {
                "markdown_url": f"/api/jobs/{job.id}/markdown",
                "archive_url": f"/api/jobs/{job.id}/archive",
            },
        )
        logger.info("잡 완료: %s (%d페이지)", job.id, total)

    except JobCanceled:
        sink.flush()
        job.status = "canceled"
        job.error = "사용자에 의해 취소되었습니다"
        # 취소·오류로 끝나도 그때까지 쌓인 경고(플레이스홀더·재처리·텍스트 레이어
        # 복구 등)는 남긴다 — 부분 결과는 보존되는데 왜 그런지가 사라지면
        # 사용자는 정상 변환된 부분과 구분할 수 없다.
        if merger is not None:
            job.warnings = merger.warnings
        store.save(job)
        broker.publish(job.id, "error", {"message": job.error, "canceled": True})
        logger.info("잡 취소: %s", job.id)
    except Exception as e:  # noqa: BLE001 — 잡 단위 격리
        sink.flush()
        logger.exception("잡 실패: %s", job.id)
        job.status = "error"
        job.error = str(e)[:2000] or e.__class__.__name__
        if merger is not None:
            job.warnings = merger.warnings
        store.save(job)
        broker.publish(job.id, "error", {"message": job.error})
    finally:
        if job.delete_requested:
            shutil.rmtree(job.dir, ignore_errors=True)
            store.remove(job.id)
        else:
            # 터미널 상태(done/error/canceled) 마감 시 work/ 정리 — 필요 산출물
            # (images/·layout/·result.md·layout.json)은 add_chunk 시점에 이미 잡
            # 루트로 이동/기록됐고, work/에는 boxes.json·raw_pages.json·실패 청크
            # 잔여물만 남아 잡마다 무한 축적된다.
            shutil.rmtree(job.dir / "work", ignore_errors=True)
