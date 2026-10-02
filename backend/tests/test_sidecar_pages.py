"""페이지마다 응답을 고르는 stub sidecar로 돌리는 잡 시나리오 (업로드 → runner → merge).

stub은 업로드 파일명(runner 규약 page_NNNN.png)에서 전역 페이지 번호를 읽어, 그 페이지의
원본 본문을 OCR 결과로 돌려준다 — 동시 요청(OCR_REMOTE_PAGE_CONCURRENCY>1)에서도 페이지와
응답이 어긋나지 않는다.

- 여러 쪽 figure_only 청크의 병합: 원출력(raw_pages.json)이 없다고 merge가 마커 불일치로
  오인해 원본 대조 재배치를 돌리고 거짓 '페이지 마커' 경고로 잡을 degraded로 만들던 회귀
  (감사 sidecar-1 — d4b667f).
- 잡 도중 sidecar 재시작(엔진 사망 → 503 → 재기동 → 모델 재로드): 정상 복구된 잡이
  대기 문구 경고로 degraded가 되던 문제(감사 sidecar-3), 엔진을 결정적으로 죽이는 페이지
  하나가 컨테이너 재시작을 3~5회 일으키던 문제(감사 sidecar-2).
- 여러 쪽 청크의 읽기 타임아웃: 페이지별 복구가 타임아웃 난 페이지를 곧바로 다시 보내고
  이미 끝난 형제 페이지까지 다시 추론하던 문제(감사 pipeline-6).
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest

from app.config import Settings
from app.engine.base import NullSink
from app.engine.registry import build_engine
from app.engine.sidecar import SidecarRestartLoopError
from app.main import create_app
from app.sidecar.client import SidecarTimeoutError

from tests.conftest import wait_done
from tests.test_sidecar_client import _health_body, _parse_body

_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
_PAGE_FILE_RE = re.compile(rb'filename="page_(\d+)\.png"')


def _page_text(page: int) -> str:
    """페이지마다 다른 본문 — 정합 대조·충실도 판정(정답 200자 이상)에 걸리는 길이."""
    w = _WORDS[(page - 1) % len(_WORDS)]
    return (
        f"Chapter {page} discusses the {w} protocol in great detail. "
        f"The {w} measurement campaign collected {100 + page * 7} samples across many sites. "
        f"Results for {w} show steady improvements over baseline number {page * 13 + 5}. "
        f"Every {w} table in this chapter is reproduced from the original {w} report."
    )


def _text_pdf(pages: int) -> bytes:
    import pymupdf as fitz

    doc = fitz.open()
    for page in range(1, pages + 1):
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(fitz.Rect(56, 56, 540, 800), _page_text(page), fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class PageStub:
    """페이지 번호로 응답을 고르는 stub sidecar.

    behavior(page) → (status, body dict) 또는 None(기본: 그 페이지 본문으로 정상 응답).
    health_fn() → health dict 또는 None(HTTP 503 — 재기동 중이라 응답하지 못하는 상태)."""

    def __init__(self) -> None:
        self.behavior = None
        self.health_fn = None
        self.seen: list[int] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: N802 — 테스트 소음 제거
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except BrokenPipeError:
                    pass

            def do_GET(self):  # noqa: N802
                health = outer.health_fn() if outer.health_fn else _health_body()
                if health is None:
                    self._send(503, {"detail": "sidecar 재기동 중"})
                else:
                    self._send(200, health)

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                m = _PAGE_FILE_RE.search(raw)
                page = int(m.group(1)) if m else -1
                with outer._lock:
                    outer.seen.append(page)
                result = outer.behavior(page) if outer.behavior else None
                self._send(*(result or (200, page_body(page))))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Lifecycle:
    """sidecar 수명 상태기계 — 추론 엔진이 죽으면 503을 돌려주고 스스로 재기동한다.

    ready → (엔진 사망) restarting(health restarting=true) → down(health도 503 — 컨테이너
    재기동) → loading(model_loaded=false) → ready. 추론은 락으로 직렬화한다(실제 sidecar의
    max_num_seqs=1+락·소유 스레드와 같다). kills(page, deaths)가 참이면 그 추론이 엔진을
    죽인다."""

    def __init__(self, kills, phase_s: float = 0.1) -> None:
        self.kills = kills
        self.phase_s = phase_s
        self.died_at: float | None = None
        self.deaths = 0
        self.inferences = 0
        self._lock = threading.Lock()

    def _state(self) -> str:
        if self.died_at is None:
            return "ready"
        elapsed = time.monotonic() - self.died_at
        for state, until in (("restarting", 1), ("down", 2), ("loading", 3)):
            if elapsed < until * self.phase_s:
                return state
        return "ready"

    def health(self) -> dict | None:
        state = self._state()
        if state == "down":
            return None
        if state == "restarting":
            return _health_body(model_loaded=False, restarting=True)
        if state == "loading":
            return _health_body(model_loaded=False)
        return _health_body()

    def parse(self, page: int):
        with self._lock:
            if self._state() != "ready":
                return 503, {"detail": "모델이 아직 로드되지 않았습니다"}
            self.inferences += 1
            if self.kills(page, self.deaths):
                self.deaths += 1
                self.died_at = time.monotonic()
                return 503, {"detail": "추론 엔진이 종료돼 sidecar를 재시작합니다"}
        return None


def page_body(page: int, markdown: str | None = None, **page_fields) -> dict:
    body = _parse_body()
    body["page"].update(
        markdown=_page_text(page) if markdown is None else markdown, blocks=[], **page_fields
    )
    return body


@pytest.fixture
def stub():
    s = PageStub()
    yield s
    s.close()


def _settings(tmp_path, stub, **kw) -> Settings:
    base = dict(
        engine="ovisocr2", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
        sidecar_url=stub.url, sidecar_connect_timeout_s=2.0,
        sidecar_read_timeout_s=10.0, sidecar_health_timeout_s=2.0,
    )
    base.update(kw)
    return Settings(**base)


def _run_job(tmp_path, stub, pages: int, **kw) -> tuple[dict, dict, list[str]]:
    """잡 하나를 끝까지 돌려 (잡 본문, viewer-manifest quality, 페이지별 markdown)을 돌려준다."""
    from fastapi.testclient import TestClient

    with TestClient(create_app(_settings(tmp_path, stub, **kw))) as c:
        r = c.post("/api/jobs", data={"mode": "multi"},
                   files={"file": ("doc.pdf", BytesIO(_text_pdf(pages)), "application/pdf")})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        body = wait_done(c, job_id, timeout=60)
        quality = c.get(f"/api/jobs/{job_id}/viewer-manifest").json()["quality"]
        md = c.get(f"/api/jobs/{job_id}/markdown").text
    return body, quality, [page.strip() for page in md.split("\n\n---\n\n")]


# ── 여러 쪽 figure_only 청크 (감사 sidecar-1) ─────────────────────────────────

def test_concurrent_figure_only_chunks_merge_without_marker_warnings(tmp_path, stub):
    """OCR_REMOTE_PAGE_CONCURRENCY=4(청크=4쪽)로 모든 페이지가 정상이면 경고가 없어야 한다 —
    예전에는 4쪽 청크마다 '페이지 마커 4개 (기대 4) — 원본 본문과 대조해…'가 쌓였다."""
    body, quality, pages = _run_job(tmp_path, stub, 8, remote_page_concurrency=4)

    assert body["status"] == "done", body
    assert body["warnings"] == [], body["warnings"]
    assert quality["state"] == "ok", quality
    assert [p.split(" discusses")[0] for p in pages] == [f"Chapter {n}" for n in range(1, 9)]
    assert body["result"]["has_layout"] is False  # 좌표 없는 엔진 — layout.json 없음


def test_truncated_page_keeps_leading_pages_without_a_marker_warning(tmp_path, stub):
    """동시성 4에서 3쪽이 출력 상한에 잘려 텍스트 레이어로 넘어가면, 끝까지 받은 앞 2쪽은
    그대로 병합된다 — 그 병합이 '페이지 마커 2개 (기대 2)' 경고를 남기면 안 된다."""
    def truncate_page_3(page: int):
        if page != 3:
            return None
        return 200, page_body(3, markdown="Chapter 3", truncated=True,
                              warnings=["출력 토큰 상한(8192)에 도달 — 페이지 끝부분이 잘렸을 수 있습니다"])

    stub.behavior = truncate_page_3
    body, quality, pages = _run_job(tmp_path, stub, 8, remote_page_concurrency=4)

    assert body["status"] == "done", body
    assert not any("페이지 마커" in w for w in body["warnings"]), body["warnings"]
    assert len(body["warnings"]) == 1, body["warnings"]
    assert body["warnings"][0].startswith("3페이지: ") and "텍스트 레이어로 복구" in body["warnings"][0]
    assert quality["warning_count"] == 1
    # 1·2쪽은 OCR 그대로, 3쪽은 텍스트 레이어 전문, 4쪽 이후는 다시 OCR
    assert len(pages) == 8
    assert pages[0] == _page_text(1) and pages[1] == _page_text(2)
    assert "PDF 내장 텍스트 레이어에서 복구" in pages[2]
    assert "Chapter 3 discusses the charlie protocol" in pages[2]
    assert pages[3:] == [_page_text(n) for n in range(4, 9)]
    assert stub.seen.count(3) == 1  # 잘린 페이지를 GPU에 다시 보내지 않았다


# ── 잡 도중 sidecar 재시작 (감사 sidecar-3) ──────────────────────────────────

@pytest.mark.parametrize("concurrency,pages,bad", [(1, 3, 2), (4, 4, 3)])
def test_job_recovered_from_a_sidecar_restart_is_not_degraded(
    tmp_path, stub, monkeypatch, concurrency, pages, bad
):
    """추론 엔진이 한 번 죽어 sidecar가 재기동돼도 모든 페이지가 정상 처리됐으면 잡은 'ok'다 —
    예전에는 '재시작/모델 재로드 대기 중…'·'모델 로딩 대기 중… (최초 기동은…)' 같은 대기
    문구가 경고로 쌓여 '주의 3건'(degraded)으로 끝났다. 경위는 참고 한 줄로 남는다."""
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.02)
    life = Lifecycle(kills=lambda page, deaths: page == bad and deaths == 0)
    stub.health_fn, stub.behavior = life.health, life.parse

    body, quality, merged = _run_job(tmp_path, stub, pages, remote_page_concurrency=concurrency)

    assert body["status"] == "done", body
    assert life.deaths == 1
    assert merged == [_page_text(n) for n in range(1, pages + 1)]
    assert body["warnings"] == [], body["warnings"]
    assert quality["state"] == "ok", quality
    recovered = [n for n in body["notices"] if "다시 보내 처리" in n]
    span = f"{bad}페이지" if concurrency == 1 else f"1–{pages}페이지"
    assert len(recovered) == 1 and recovered[0].startswith(f"{span}: "), body["notices"]
    assert not any("최초 기동" in m for m in body["warnings"] + body["notices"])


# ── 엔진을 죽이는 페이지 (감사 sidecar-2) ──────────────────────────────────────

def _job_pages(tmp_path, count: int) -> list:
    """runner 규약의 페이지 래스터 — stub이 파일명으로 페이지를 안다."""
    from PIL import Image

    pages = tmp_path / "job" / "pages"
    pages.mkdir(parents=True)
    paths = []
    for page in range(1, count + 1):
        path = pages / f"page_{page:04d}.png"
        Image.new("RGB", (400, 560), "white").save(path)
        paths.append(path)
    return paths


def test_page_that_kills_the_engine_again_after_recovery_is_not_retried(
    tmp_path, stub, monkeypatch
):
    """복귀 뒤 다시 보낸 요청에서도 sidecar가 내려가면 retry_same_page=False로 올린다 —
    runner가 같은 페이지를 또 보내지 않게. 페이지별 복구가 그 페이지를 한 번 더 보낼 때는
    또 내려가도 기다려 다시 보내지 않는다(형제 페이지는 그 한 번으로 정상 처리된다)."""
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.02)
    life = Lifecycle(kills=lambda page, deaths: page == 2)
    stub.health_fn, stub.behavior = life.health, life.parse
    eng = build_engine(_settings(tmp_path, stub, remote_page_concurrency=2))
    eng.load()
    p1, p2 = _job_pages(tmp_path, 2)

    with pytest.raises(SidecarRestartLoopError) as ei:
        eng.run_multi([p1, p2], tmp_path / "chunk", NullSink(), threading.Event())
    assert ei.value.retry_same_page is False
    assert life.deaths == 2            # 첫 요청 + 복귀 뒤 1회 재요청

    # runner의 페이지별 복구 — 1쪽은 재기동을 기다렸다가 정상 처리된다
    md = eng.run_single(p1, tmp_path / "f1", NullSink(), threading.Event())
    assert md.startswith("Chapter 1 discusses")
    # 2쪽은 한 번만 더 보낸다 — 또 내려가면 기다려 다시 보내지 않는다
    sent = stub.seen.count(2)
    with pytest.raises(SidecarRestartLoopError):
        eng.run_single(p2, tmp_path / "f2", NullSink(), threading.Event())
    assert stub.seen.count(2) == sent + 1 and life.deaths == 3
    assert eng.drain_warnings() == []  # 손실 경고는 runner가 페이지 격리에서 남긴다


def test_next_chunk_waits_for_the_restart_instead_of_sending_into_it(
    tmp_path, stub, monkeypatch
):
    """복귀 뒤 재요청까지 실패하면 health 캐시를 다시 무효화한다 — 복귀 대기가 남긴
    loaded=True를 믿으면 다음 청크가 재시작 중인 sidecar로 곧장 보내 503을 받고서야 기다렸다."""
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.02)
    life = Lifecycle(kills=lambda page, deaths: page == 2)
    stub.health_fn, stub.behavior = life.health, life.parse
    eng = build_engine(_settings(tmp_path, stub))
    eng.load()
    p1, p2 = _job_pages(tmp_path, 2)
    with pytest.raises(SidecarRestartLoopError):
        eng.run_multi([p2], tmp_path / "c0", NullSink(), threading.Event())
    assert not eng.loaded
    eng.drain_notices()

    md = eng.run_multi([p1], tmp_path / "c1", NullSink(), threading.Event())  # 다음 청크
    assert "Chapter 1 discusses" in md
    assert stub.seen.count(1) == 1     # 재기동이 끝난 뒤에 한 번만 보냈다
    assert eng.drain_notices() == ["sidecar 재시작/모델 재로드가 끝나기를 기다린 뒤 이어서 진행했습니다"]


def test_restart_loop_mark_does_not_outlive_the_next_chunk(tmp_path, stub, monkeypatch):
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.02)
    life = Lifecycle(kills=lambda page, deaths: page == 2)
    stub.health_fn, stub.behavior = life.health, life.parse
    eng = build_engine(_settings(tmp_path, stub))
    eng.load()
    p1, p2 = _job_pages(tmp_path, 2)
    with pytest.raises(SidecarRestartLoopError):
        eng.run_multi([p2], tmp_path / "c0", NullSink(), threading.Event())
    eng.run_multi([p1], tmp_path / "c1", NullSink(), threading.Event())  # 다음 청크
    with pytest.raises(SidecarRestartLoopError):
        eng.run_single(p2, tmp_path / "late", NullSink(), threading.Event())
    assert life.deaths == 4  # 표식이 지워져 다시 '복귀 뒤 1회 재요청'을 탔다


@pytest.mark.parametrize("concurrency,pages,bad,deaths", [(1, 3, 2, 2), (4, 4, 3, 3)])
def test_page_that_kills_the_engine_costs_few_sidecar_restarts(
    tmp_path, stub, monkeypatch, concurrency, pages, bad, deaths
):
    """같은 이미지에서 결정적으로 엔진이 죽는 페이지(EngineCore OOM·illegal memory access)
    하나가 예전에는 컨테이너 재시작·모델 재로드를 3회(동시성 1)~5회(동시성 4) 일으킨 뒤에야
    텍스트 레이어로 넘어갔다. 이제 첫 요청 + 복귀 뒤 1회(+동시성>1이면 페이지별 복구 1회)뿐이다."""
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.02)
    life = Lifecycle(kills=lambda page, _deaths: page == bad)
    stub.health_fn, stub.behavior = life.health, life.parse

    body, quality, merged = _run_job(tmp_path, stub, pages, remote_page_concurrency=concurrency)

    assert body["status"] == "done", body
    assert life.deaths == deaths, (life.deaths, stub.seen)
    assert len(body["warnings"]) == 1, body["warnings"]
    warning = body["warnings"][0]
    assert warning.startswith(f"{bad}페이지: ") and "텍스트 레이어로 복구" in warning
    assert "SidecarRestartLoopError" in warning
    assert "PDF 내장 텍스트 레이어에서 복구" in merged[bad - 1]
    # 범인이 아닌 페이지는 모두 OCR 결과 그대로다
    assert [m for n, m in enumerate(merged, 1) if n != bad] == [
        _page_text(n) for n in range(1, pages + 1) if n != bad
    ]


# ── 여러 쪽 청크의 읽기 타임아웃 (감사 pipeline-6) ────────────────────────────────

def _slow_page_1(page: int):
    """1쪽만 읽기 타임아웃보다 오래 추론한다 — sidecar는 끊긴 요청도 끝까지 돈다."""
    if page == 1:
        time.sleep(1.0)
    return None


def test_page_recovery_resends_neither_the_timed_out_page_nor_finished_siblings(tmp_path, stub):
    """동시성 2 청크에서 1쪽이 읽기 타임아웃이면 run_multi가 실패하고 runner는 청크의 모든 페이지를
    run_single로 다시 부른다. 1쪽은 sidecar에 보내지 않고 같은 SidecarTimeoutError로(텍스트
    레이어로 가게), 이미 끝난 2쪽은 받아 둔 결과로 답한다. 예전에는 1쪽을 곧바로 다시 보내 끊긴
    추론 뒤에 줄 세우고(또 타임아웃), 끝난 2쪽까지 GPU에서 다시 추론했다."""
    stub.behavior = _slow_page_1
    eng = build_engine(_settings(tmp_path, stub, remote_page_concurrency=2,
                                 sidecar_read_timeout_s=0.3, sidecar_retries=0))
    eng.load()
    p1, p2 = _job_pages(tmp_path, 2)
    with pytest.raises(SidecarTimeoutError):
        eng.run_multi([p1, p2], tmp_path / "chunk", NullSink(), threading.Event())
    assert sorted(stub.seen) == [1, 2]

    with pytest.raises(SidecarTimeoutError) as ei:
        eng.run_single(p1, tmp_path / "f1", NullSink(), threading.Event())
    assert ei.value.retry_same_page is False  # runner는 재시도 없이 텍스트 레이어로 간다
    assert eng.run_single(p2, tmp_path / "f2", NullSink(), threading.Event()) == _page_text(2)
    assert sorted(stub.seen) == [1, 2]  # 페이지별 복구는 sidecar에 아무것도 보내지 않았다

    # 1회용 — 같은 페이지를 또 부르면(다른 경로의 재처리) 이번에는 실제로 보낸다
    assert eng.run_single(p2, tmp_path / "f3", NullSink(), threading.Event()) == _page_text(2)
    assert stub.seen.count(2) == 2


def test_job_with_a_timed_out_page_sends_every_page_once(tmp_path, stub):
    """잡 단위: 1쪽은 텍스트 레이어로 복구되고 2쪽은 OCR 그대로 — 페이지마다 요청은 한 번이다
    (예전 {1: 2, 2: 2})."""
    stub.behavior = _slow_page_1
    body, quality, merged = _run_job(tmp_path, stub, 2, remote_page_concurrency=2,
                                     sidecar_read_timeout_s=0.3, sidecar_retries=0)

    assert body["status"] == "done", body
    assert sorted(stub.seen) == [1, 2], stub.seen
    assert len(body["warnings"]) == 1, body["warnings"]
    assert body["warnings"][0].startswith("1페이지: ") and "텍스트 레이어로 복구" in body["warnings"][0]
    assert "SidecarTimeoutError" in body["warnings"][0]
    assert "PDF 내장 텍스트 레이어에서 복구" in merged[0] and "Chapter 1 discusses" in merged[0]
    assert merged[1] == _page_text(2)
