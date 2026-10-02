"""페이지마다 응답을 고르는 stub sidecar로 돌리는 잡 시나리오 (업로드 → runner → merge).

stub은 업로드 파일명(runner 규약 page_NNNN.png)에서 전역 페이지 번호를 읽어, 그 페이지의
원본 본문을 OCR 결과로 돌려준다 — 동시 요청(OCR_REMOTE_PAGE_CONCURRENCY>1)에서도 페이지와
응답이 어긋나지 않는다.

- 여러 쪽 figure_only 청크의 병합: 원출력(raw_pages.json)이 없다고 merge가 마커 불일치로
  오인해 원본 대조 재배치를 돌리고 거짓 '페이지 마커' 경고로 잡을 degraded로 만들던 회귀
  (감사 sidecar-1 — d4b667f).
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest

from app.config import Settings
from app.main import create_app

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
