"""번역 PDF 내보내기 테스트 — pipeline/pdf_export.py + GET /api/jobs/{id}/pdf.

e2e는 FakeEngine 잡 위에 번역 아티팩트(layout.ko.json·result.ko.md·state.json)를
계약대로 심어 검증하고(test_api_translate.py의 페이크 관례), 리댁션 기하학처럼
좌표가 중요한 검증은 소스 PDF와 레이아웃을 직접 만드는 유닛 테스트로 고정한다.
"""

import io
import json
import os
from pathlib import Path

import pytest
from conftest import make_pdf_bytes, wait_done

from app.pipeline.pdf_export import (
    PdfExportError,
    _plain_text,
    build_translated_pdf,
)

KO_TEXT = "한국어 번역 텍스트 블록입니다"


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def _make_done_job(client, pages: int = 2) -> str:
    r = client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", io.BytesIO(make_pdf_bytes(pages=pages)),
                        "application/pdf")},
        data={"mode": "multi"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_done(client, job_id)
    return job_id


def _job_dir(client, job_id: str) -> Path:
    return client.app.state.store.get(job_id).dir


def _add_translation(job_dir: Path, lang: str = "ko") -> None:
    """FakeEngine이 만든 layout.json을 복제해 text 블록만 한국어로 바꾼 번역
    아티팩트를 심는다. title 블록은 원문 유지 → '변경 없음' 경로도 함께 검증."""
    pages = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
    for page in pages:
        for block in page.get("blocks", []):
            if block.get("type") == "text":
                block["content"] = f"{KO_TEXT} (p{page['page']})"
    (job_dir / f"layout.{lang}.json").write_text(
        json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    (job_dir / f"result.{lang}.md").write_text(
        f"# 번역본\n\n{KO_TEXT}\n", encoding="utf-8")
    tdir = job_dir / "translations" / lang
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "state.json").write_text(
        json.dumps({"lang": lang, "status": "done"}), encoding="utf-8")


def _pdf_text(data: bytes) -> str:
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        # 일부 폰트는 공백을 NBSP(U+00A0)로 추출한다 — 비교 전에 정규화
        return "\n".join(page.get_text() for page in doc).replace("\xa0", " ")


# ── API 계약 ──────────────────────────────────────────────────────────────
def test_pdf_export_e2e(client):
    job_id = _make_done_job(client, pages=2)
    _add_translation(_job_dir(client, job_id))

    r = client.get(f"/api/jobs/{job_id}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "sample.ko.pdf" in r.headers.get("content-disposition", "")
    assert r.content.startswith(b"%PDF")
    assert int(r.headers["x-uocr-pdf-replaced"]) >= 1
    assert int(r.headers["x-uocr-pdf-preserved"]) >= 0
    assert int(r.headers["x-uocr-pdf-warnings"]) >= 0
    assert int(r.headers["x-uocr-pdf-specialist-preserved"]) >= 0

    import fitz

    with fitz.open(stream=r.content, filetype="pdf") as doc:
        assert doc.page_count == 2
    text = _pdf_text(r.content)
    assert KO_TEXT in text          # 번역 블록 삽입됨
    # 캐시 파일 생성
    assert (_job_dir(client, job_id) / "export.ko.pdf").is_file()
    assert (_job_dir(client, job_id) / "export.ko.report.json").is_file()


def test_pdf_409_before_done(client):
    job_id = _make_done_job(client)
    job = client.app.state.store.get(job_id)
    job.status = "running"  # 결정적 재현 (test_api_translate 관례)
    try:
        assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 409
    finally:
        job.status = "done"


def test_pdf_404_without_translation(client):
    job_id = _make_done_job(client)
    r = client.get(f"/api/jobs/{job_id}/pdf")
    assert r.status_code == 404
    assert "번역" in r.json()["detail"]


def test_pdf_400_bad_lang(client):
    job_id = _make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/pdf?lang=jp").status_code == 400


def test_pdf_409_without_layout(client):
    job_id = _make_done_job(client)
    job_dir = _job_dir(client, job_id)
    _add_translation(job_dir)
    (job_dir / "layout.ko.json").unlink()
    r = client.get(f"/api/jobs/{job_id}/pdf")
    assert r.status_code == 409
    assert "HTML" in r.json()["detail"]


def test_pdf_cache_and_rebuild(client):
    job_id = _make_done_job(client)
    job_dir = _job_dir(client, job_id)
    _add_translation(job_dir)

    assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 200
    out = job_dir / "export.ko.pdf"
    first = out.stat().st_mtime_ns

    assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 200
    assert out.stat().st_mtime_ns == first  # 캐시 히트 — 재생성 없음

    # 번역 레이아웃이 갱신되면(더 새로운 mtime) 재생성
    newer = out.stat().st_mtime + 5
    os.utime(job_dir / "layout.ko.json", (newer, newer))
    assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 200
    assert out.stat().st_mtime_ns != first


def test_unknown_job_404(client):
    assert client.get("/api/jobs/j_missing000000/pdf").status_code == 404


# ── 빌더 유닛 (좌표를 직접 통제) ──────────────────────────────────────────
def _unit_job(tmp_path: Path, *, rotate: int = 0, block_type: str = "text",
              src_text: str = "Original English sentence", with_graphic: bool = False) -> Path:
    """source.pdf의 실제 글리프 위치와 레이아웃 bbox를 일치시킨 잡 디렉토리."""
    import fitz

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # bbox [100,100,900,300] (0–999) ↔ 표시 공간 (59.5,84.2)–(535.5,252.6)pt
    page.insert_textbox(fitz.Rect(60, 85, 535, 250), src_text, fontsize=12)
    if with_graphic:
        # 번역 대상 bbox 안에 완전히 포함된 벡터 도형. apply_redactions()의
        # graphics 기본값이면 제거되므로 레이아웃 보존 회귀를 잡아낸다.
        page.draw_rect(fitz.Rect(100, 150, 220, 190), color=(1, 0, 0), width=2)
    if rotate:
        page.set_rotation(rotate)
    doc.save(job_dir / "source.pdf")
    doc.close()

    blocks = [{"type": block_type, "bbox": [100, 100, 900, 300],
               "content": src_text, "fs": 2.0}]
    orig = [{"page": 1, "width": 1000, "height": 1414, "blocks": blocks}]
    trans = json.loads(json.dumps(orig))
    trans[0]["blocks"][0]["content"] = KO_TEXT
    (job_dir / "layout.json").write_text(json.dumps(orig), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(trans, ensure_ascii=False), encoding="utf-8")
    return job_dir


def test_build_replaces_and_redacts(tmp_path):
    job_dir = _unit_job(tmp_path)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1 and not result.warnings
    text = _pdf_text(result.path.read_bytes())
    assert KO_TEXT in text
    assert "Original English sentence" not in text  # 리댁션으로 원문 제거


def test_build_preserves_vector_graphics_inside_replaced_block(tmp_path):
    """텍스트 리댁션은 같은 bbox 안의 밑줄·도형·차트 선을 제거하지 않는다."""
    import fitz

    job_dir = _unit_job(tmp_path, with_graphic=True)
    with fitz.open(job_dir / "source.pdf") as source:
        assert len(source[0].get_drawings()) == 1
    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        assert len(exported[0].get_drawings()) == 1


def test_build_skips_unfittable_block_without_redacting_original(tmp_path):
    """번역문이 최소 글자 크기에도 안 들어가면 원문을 지우지 않고 보존한다."""
    job_dir = _unit_job(tmp_path)
    trans = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    trans[0]["blocks"][0]["content"] = "매우 긴 번역문 " * 20_000
    (job_dir / "layout.ko.json").write_text(
        json.dumps(trans, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0 and result.kept >= 1
    assert any("원문 보존" in warning for warning in result.warnings)
    assert "Original English sentence" in _pdf_text(result.path.read_bytes())


def test_build_rejects_misaligned_translation_layout_before_redaction(tmp_path):
    """content 이외의 구조가 달라지면 잘못된 사각형을 리댁션하지 않고 중단한다."""
    job_dir = _unit_job(tmp_path)
    trans = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    trans[0]["blocks"][0]["bbox"] = [0, 0, 999, 999]
    (job_dir / "layout.ko.json").write_text(
        json.dumps(trans, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(PdfExportError, match="블록이 일치하지 않습니다"):
        build_translated_pdf(job_dir, "ko")
    assert not (job_dir / "export.ko.pdf").exists()


def test_build_uses_source_font_metrics_without_layout_view(tmp_path, monkeypatch):
    """사용자가 레이아웃 탭을 먼저 열지 않아도 원본 12pt 실측값을 PDF에 쓴다."""
    import fitz

    job_dir = _unit_job(tmp_path)
    for name in ("layout.json", "layout.ko.json"):
        pages = json.loads((job_dir / name).read_text(encoding="utf-8"))
        pages[0]["blocks"][0].pop("fs", None)
        (job_dir / name).write_text(
            json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    # 실측 백필이 없다면 0.1cqw → 최소 4pt로 떨어지는 확실한 대조군.
    monkeypatch.setattr("app.pipeline.pdf_export.estimate_font_size_cqw", lambda *_: 0.1)

    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        spans = [
            span
            for block in exported[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if "한국어" in span.get("text", "")
        ]
    assert spans and spans[0]["size"] > 8, spans


def test_build_rotated_page(tmp_path):
    job_dir = _unit_job(tmp_path, rotate=90)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1
    assert KO_TEXT in _pdf_text(result.path.read_bytes())


def test_build_keeps_unreplaceable_types(tmp_path):
    job_dir = _unit_job(tmp_path, block_type="equation")
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0
    text = _pdf_text(result.path.read_bytes())
    assert "Original English sentence" in text  # 수식 블록은 원본 유지
    assert KO_TEXT not in text


def test_build_keeps_unchanged_blocks(tmp_path):
    job_dir = _unit_job(tmp_path)
    # 번역본 내용 = 원문 → 아무것도 바꾸지 않아야 한다
    (job_dir / "layout.ko.json").write_text(
        (job_dir / "layout.json").read_text(encoding="utf-8"), encoding="utf-8")
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0 and result.kept >= 1
    assert "Original English sentence" in _pdf_text(result.path.read_bytes())


def test_build_missing_inputs(tmp_path):
    job_dir = _unit_job(tmp_path)
    (job_dir / "layout.ko.json").unlink()
    with pytest.raises(PdfExportError):
        build_translated_pdf(job_dir, "ko")


def test_plain_text_strips_markup():
    assert _plain_text("<b>굵게</b>  두  칸") == "굵게 두 칸"
    assert _plain_text("줄1\n\n<i>줄2</i>") == "줄1\n줄2"
    assert _plain_text(r"가격은 \( E = mc^{2} \) 이다") == "가격은 E = mc² 이다"


def test_build_translates_table_cells_without_removing_grid(tmp_path):
    """구조가 같은 HTML 표는 셀 단위로 번역하고 원본 벡터 격자를 유지한다."""
    import fitz

    job_dir = tmp_path / "table-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((90, 120), "Header", fontsize=11)
    page.insert_text((330, 120), "Value", fontsize=11)
    page.insert_text((90, 170), "alpha", fontsize=11)
    page.insert_text((330, 170), "42", fontsize=11)
    for y in (90, 140, 190):
        page.draw_line(fitz.Point(60, y), fitz.Point(535, y))
    page.draw_line(fitz.Point(300, 90), fitz.Point(300, 190))
    doc.save(job_dir / "source.pdf")
    doc.close()

    original_html = (
        "<table><tr><td>Header</td><td>Value</td></tr>"
        "<tr><td>alpha</td><td>42</td></tr></table>"
    )
    translated_html = (
        "<table><tr><td>항목</td><td>값</td></tr>"
        "<tr><td>alpha</td><td>42</td></tr></table>"
    )
    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "table", "bbox": [100, 100, 900, 250], "content": original_html},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = translated_html
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    with fitz.open(job_dir / "source.pdf") as source:
        drawings = len(source[0].get_drawings())
    result = build_translated_pdf(job_dir, "ko")
    text = _pdf_text(result.path.read_bytes())
    assert result.table_cells_replaced == 2
    assert "항목" in text and "값" in text
    assert "Header" not in text and "Value" not in text
    assert "alpha" in text and "42" in text
    with fitz.open(result.path) as exported:
        assert len(exported[0].get_drawings()) == drawings
    report = json.loads((job_dir / "export.ko.report.json").read_text(encoding="utf-8"))
    assert report["table_cells_replaced"] == 2 and report["warning_count"] == 0


def test_build_expands_only_into_collision_free_space(tmp_path):
    """짧은 bbox의 번역문은 아래 빈 공간을 쓰되 다음 블록 전에서 멈춘다."""
    import fitz

    job_dir = tmp_path / "growth-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(60, 85, 300, 110), "Original", fontsize=12)
    page.insert_textbox(fitz.Rect(60, 253, 300, 278), "Footer obstacle", fontsize=12)
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [100, 100, 500, 130], "content": "Original", "fs": 2.0},
        {"type": "footer", "bbox": [100, 300, 500, 330], "content": "Footer obstacle", "fs": 2.0},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = "충돌을 피해서 아래 빈 공간에 배치하는 긴 번역 문장이다. " * 8
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    text = _pdf_text(result.path.read_bytes())
    assert result.relocated == 1 and result.replaced == 1
    assert "충돌을 피해서" in text
    assert "Footer obstacle" in text
