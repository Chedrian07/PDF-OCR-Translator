"""업로드 복잡도 게이트 — 렌더를 폭주시키는 PDF를 큐에 넣기 전에 400으로 거부한다.

감사 A11(security-2·gap3-mupdf-analysis-amplification-2): 3KB짜리 중첩 Form XObject PDF
(리프 그리기 10^12회)와 평면 대량 path 페이지가 페이지 수·크기만 보던 probe_pdf를 통과해
단일 OCR 워커를 영구히 점거하거나 서버를 OOM으로 죽였다. 픽스처는 전부 여기서 합성한다.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import pytest

from app.pipeline.pdf import probe_pdf

from conftest import make_pdf_bytes

REPO = Path(__file__).resolve().parents[2]


# ── 합성 픽스처 ───────────────────────────────────────────────────────────


def _fitz():
    import pymupdf

    return pymupdf


def _form_chain(doc, levels: int, fanout: int) -> int:
    """리프(사각형 1개)부터 levels단계 — 단계마다 아래 단계를 fanout번 부르는 Form 체인.
    맨 위 Form의 xref를 돌려준다(펼친 그리기 = fanout^levels)."""
    below = None
    for level in range(levels + 1):
        xref = doc.get_new_xref()
        if level == 0:
            stream, resources = b"0 0 1 1 re f", ""
        else:
            stream = b" ".join([b"q /X Do Q"] * fanout)
            resources = f"/Resources << /XObject << /X {below} 0 R >> >>"
        doc.update_object(
            xref, f"<< /Type /XObject /Subtype /Form /BBox [0 0 595 842] {resources} >>",
        )
        doc.update_stream(xref, stream)
        below = xref
    return below


def _page_with_content(doc, content: bytes, resources: str = "<< >>"):
    page = doc.new_page(width=595, height=842)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< >>")
    doc.update_stream(xref, content)
    doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    doc.xref_set_key(page.xref, "Resources", resources)
    return page


def nested_bomb(path: Path, levels: int = 12, fanout: int = 10) -> Path:
    """감사 재현과 같은 모양 — 페이지가 맨 위 Form을 한 번 부른다."""
    doc = _fitz().open()
    top = _form_chain(doc, levels, fanout)
    _page_with_content(doc, b"q /Top Do Q", f"<< /XObject << /Top {top} 0 R >> >>")
    doc.save(path)
    doc.close()
    return path


def flat_content(path: Path, lines: int) -> Path:
    """XObject 없이 짧은 path를 lines개 그리는 평면 콘텐츠 페이지."""
    body = b"".join(
        b"%d %d m %d %d l S\n" % (i % 500, i % 800, i % 500 + 3, i % 800 + 2)
        for i in range(lines)
    )
    doc = _fitz().open()
    _page_with_content(doc, body)
    doc.save(path, deflate=True)
    doc.close()
    return path


def flate_bomb(path: Path, megabytes: int) -> Path:
    """공백 N MB를 압축한 콘텐츠 스트림 — 파일은 수 KB지만 풀면 N MB."""
    doc = _fitz().open()
    page = doc.new_page(width=595, height=842)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< /Filter /FlateDecode >>")
    doc.update_stream(xref, zlib.compress(b" " * (megabytes * 1048576), 9), compress=False)
    doc.xref_set_key(xref, "Filter", "/FlateDecode")
    doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def small_limits(monkeypatch):
    """작은 픽스처로 상한을 넘기려고 상한을 낮춘다(1MB·1,000회)."""
    monkeypatch.setenv("PDF_MAX_PAGE_CONTENT_MB", "1")
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "1000")


# ── 거부되는 모양 ─────────────────────────────────────────────────────────


def test_audit_nested_xobject_bomb_is_rejected_quickly(tmp_path):
    bomb = nested_bomb(tmp_path / "bomb.pdf", levels=12, fanout=10)
    assert bomb.stat().st_size < 5_000  # 수 KB짜리 파일
    started = time.monotonic()
    with pytest.raises(ValueError, match="중첩 그리기 호출\\(Form XObject\\)") as info:
        probe_pdf(bomb, max_pages=10)
    assert time.monotonic() - started < 2  # 10^12를 렌더하지 않고 센다
    message = str(info.value)
    assert "PDF_MAX_PAGE_XOBJECT_CALLS" in message and str(tmp_path) not in message


def test_flat_content_over_the_page_limit_is_rejected(tmp_path, small_limits):
    with pytest.raises(ValueError, match="압축 해제"):
        probe_pdf(flat_content(tmp_path / "flat.pdf", 60_000), max_pages=10)  # ~1.5MB


def test_compression_bomb_is_stopped_at_the_limit(tmp_path, small_limits):
    bomb = flate_bomb(tmp_path / "zip.pdf", 64)
    assert bomb.stat().st_size < 200_000
    started = time.monotonic()
    with pytest.raises(ValueError, match="콘텐츠가 너무 큽니다"):
        probe_pdf(bomb, max_pages=10)
    assert time.monotonic() - started < 2  # 64MB를 다 풀지 않고 1MB에서 멈춘다


def test_image_draw_calls_count_too(tmp_path, small_limits):
    """이미지 XObject 호출도 한 번의 그리기다 — 같은 1px 이미지를 수천 번 찍는 페이지."""
    fitz = _fitz()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1, 1), False)
    image_xref = page.insert_image(fitz.Rect(0, 0, 1, 1), pixmap=pix)
    body = b" ".join(b"q 1 0 0 1 %d %d cm /Im0 Do Q" % (i % 500, i % 800) for i in range(2000))
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< >>")
    doc.update_stream(xref, body)
    doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    doc.xref_set_key(page.xref, "Resources", f"<< /XObject << /Im0 {image_xref} 0 R >> >>")
    path = tmp_path / "images.pdf"
    doc.save(path)
    doc.close()
    with pytest.raises(ValueError, match="중첩 그리기 호출"):
        probe_pdf(path, max_pages=10)


def test_too_deep_form_nesting_is_rejected(tmp_path):
    """팬아웃 1로 깊이만 쌓아 바닥에서 폭발시키는 체인 — 끝까지 세지 않고 깊이로 거부."""
    with pytest.raises(ValueError, match="64단계"):
        probe_pdf(nested_bomb(tmp_path / "deep.pdf", levels=70, fanout=1), max_pages=10)


def test_bombs_hidden_behind_other_entry_points_are_counted(tmp_path, small_limits):
    """페이지 리소스 상속(Pages 노드)·주석 외형·타일링 패턴·Type3 글리프 뒤의 중첩 Form."""
    fitz = _fitz()

    def _inherited(doc, top):
        _page_with_content(doc, b"q /Top Do Q", "<< >>")
        page_xref = doc.page_xref(0)
        doc.xref_set_key(page_xref, "Resources", "null")  # 페이지에는 리소스가 없다
        pages_xref = int(doc.xref_get_key(page_xref, "Parent")[1].split()[0])
        doc.xref_set_key(pages_xref, "Resources", f"<< /XObject << /Top {top} 0 R >> >>")

    def _annotation(doc, top):
        page = _page_with_content(doc, b"")
        annot = page.add_rect_annot(fitz.Rect(10, 10, 50, 50))
        doc.xref_set_key(annot.xref, "AP", f"<< /N {top} 0 R >>")

    def _pattern(doc, top):
        pattern = doc.get_new_xref()
        doc.update_object(
            pattern,
            "<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] /XStep 10 "
            f"/YStep 10 /Resources << /XObject << /X {top} 0 R >> >> >>",
        )
        doc.update_stream(pattern, b"/X Do")
        _page_with_content(
            doc, b"/Pattern cs /P0 scn 0 0 100 100 re f",
            f"<< /Pattern << /P0 {pattern} 0 R >> >>",
        )

    def _type3(doc, top):
        proc = doc.get_new_xref()
        doc.update_object(proc, "<< >>")
        doc.update_stream(proc, b"1000 0 d0 /X Do")
        font = doc.get_new_xref()
        doc.update_object(
            font,
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
            "/FontMatrix [0.001 0 0 0.001 0 0] "
            f"/CharProcs << /g {proc} 0 R >> /Encoding << /Differences [65 /g] >> "
            "/FirstChar 65 /LastChar 65 /Widths [1000] "
            f"/Resources << /XObject << /X {top} 0 R >> >> >>",
        )
        _page_with_content(doc, b"BT /F1 12 Tf 10 10 Td (A) Tj ET",
                           f"<< /Font << /F1 {font} 0 R >> >>")

    for build in (_inherited, _annotation, _pattern, _type3):
        doc = fitz.open()
        top = _form_chain(doc, 4, 10)  # 10^4회 > 상한 1,000회
        build(doc, top)
        path = tmp_path / f"{build.__name__}.pdf"
        doc.save(path)
        doc.close()
        with pytest.raises(ValueError, match="중첩 그리기 호출"):
            probe_pdf(path, max_pages=10)


def test_escaped_names_and_cycles_are_handled(tmp_path, small_limits):
    fitz = _fitz()
    # 이름 이스케이프(/X#31 = /X1)로 부른 체인도 센다
    doc = fitz.open()
    top = _form_chain(doc, 4, 10)
    _page_with_content(doc, b"q /X#31 Do Q", f"<< /XObject << /X1 {top} 0 R >> >>")
    escaped = tmp_path / "escaped.pdf"
    doc.save(escaped)
    doc.close()
    with pytest.raises(ValueError, match="중첩 그리기 호출"):
        probe_pdf(escaped, max_pages=10)

    # 서로를 부르는 Form(순환) — 끝없이 세지 않고 통과시킨다(MuPDF도 재귀를 끊는다)
    doc = fitz.open()
    a, b = doc.get_new_xref(), doc.get_new_xref()
    for this, other in ((a, b), (b, a)):
        doc.update_object(
            this, "<< /Type /XObject /Subtype /Form /BBox [0 0 10 10] "
            f"/Resources << /XObject << /X {other} 0 R >> >> >>",
        )
        doc.update_stream(this, b"0 0 1 1 re f /X Do")
    _page_with_content(doc, b"/A Do", f"<< /XObject << /A {a} 0 R >> >>")
    cycle = tmp_path / "cycle.pdf"
    doc.save(cycle)
    doc.close()
    assert probe_pdf(cycle, max_pages=10) == 1


# ── 통과하는 모양 ─────────────────────────────────────────────────────────


def test_ordinary_documents_pass(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf_bytes(pages=3))
    assert probe_pdf(path, max_pages=10) == 3
    # 펼쳐도 상한 아래인 작은 중첩·같은 Form을 여러 페이지가 공유하는 문서
    assert probe_pdf(nested_bomb(tmp_path / "small.pdf", levels=3, fanout=10), 10) == 1


def test_shared_streams_are_counted_once_per_page(tmp_path, small_limits):
    """같은 0.6MB Form을 두 페이지가 부른다 — 페이지마다 한 번(0.6MB)이라 1MB 상한 아래다."""
    fitz = _fitz()
    doc = fitz.open()
    form = doc.get_new_xref()
    doc.update_object(form, "<< /Type /XObject /Subtype /Form /BBox [0 0 595 842] >>")
    doc.update_stream(form, b"0 0 m 1 1 l S\n" * 45_000)
    for _ in range(2):
        _page_with_content(doc, b"/F Do /F Do", f"<< /XObject << /F {form} 0 R >> >>")
    path = tmp_path / "shared.pdf"
    doc.save(path, deflate=True)
    doc.close()
    assert probe_pdf(path, max_pages=10) == 2


@pytest.mark.parametrize("name", ["2504.19874v1.pdf", "unlimited-ocr-paper.pdf", "sample.pdf"])
def test_repository_samples_pass_with_default_limits(name):
    """정상 논문 표본(페이지당 최대 0.3MB·129회)은 기본 상한(64MB·2,000,000회)에 한참 못 미친다."""
    path = REPO / "sample" / name
    if not path.is_file():
        pytest.skip("저장소 표본 없음")
    assert probe_pdf(path, max_pages=200) >= 1


def test_zero_disables_each_check(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "0")
    assert probe_pdf(nested_bomb(tmp_path / "bomb.pdf", levels=12, fanout=10), 10) == 1
    monkeypatch.setenv("PDF_MAX_PAGE_CONTENT_MB", "0")
    assert probe_pdf(flate_bomb(tmp_path / "zip.pdf", 8), 10) == 1


# ── API ───────────────────────────────────────────────────────────────────


def _upload(client, data: bytes):
    return client.post("/api/jobs", files={"file": ("x.pdf", data, "application/pdf")})


def test_upload_of_a_bomb_is_a_400_and_leaves_no_job(client, settings, tmp_path):
    bomb = nested_bomb(tmp_path / "bomb.pdf").read_bytes()
    r = _upload(client, bomb)
    assert r.status_code == 400, r.text
    assert "Form XObject" in r.json()["detail"]
    assert client.get("/api/jobs").json()["jobs"] == []
    assert not [p for p in settings.jobs_dir.iterdir() if p.is_dir()]  # 잡 디렉터리도 정리됐다


def test_process_mode_probe_rejects_bombs_and_bounds_time(
    pdf_worker_processes, client, tmp_path, monkeypatch,
):
    bomb = nested_bomb(tmp_path / "bomb.pdf").read_bytes()
    r = _upload(client, bomb)
    assert r.status_code == 400 and "Form XObject" in r.json()["detail"]
    ok = _upload(client, make_pdf_bytes(pages=2))
    assert ok.status_code == 202, ok.text
    probe = pdf_worker_processes.pool_stats()["pools"]["probe"]
    assert probe["tasks"] == 2 and probe["crashes"] == 0

    # 검증이 시간 상한을 넘으면 그 PDF를 거부한다(워커 기동조차 1ms 안에 끝나지 않는다)
    pdf_worker_processes.shutdown_pools()
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "0.001")
    slow = _upload(client, make_pdf_bytes(pages=2))
    assert slow.status_code == 400, slow.text
    assert "시간 상한" in slow.json()["detail"]


def test_process_mode_probe_reports_busy_as_503(pdf_worker_processes, client, monkeypatch):
    import threading

    holders = [
        threading.Thread(
            target=pdf_worker_processes.run, args=("pdf_worker_tasks:sleep", (3,)),
            kwargs={"pool": "probe", "timeout": 30},
        )
        for _ in range(2)  # probe 풀 크기만큼 붙잡는다
    ]
    for holder in holders:
        holder.start()
    time.sleep(0.5)
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "0.5")  # 빈 워커 대기 상한 = 검증 상한
    r = _upload(client, make_pdf_bytes(pages=1))
    for holder in holders:
        holder.join()
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "5"
    assert client.get("/api/jobs").json()["jobs"] == []
