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
    PDF_EXPORT_FORMAT_VERSION,
    PdfExportError,
    _free_growth_rect,
    _metrics_font,
    _plain_text,
    _resolve_font,
    build_dual_pdf,
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
    report_path = _job_dir(client, job_id) / "export.ko.report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format_version"] == PDF_EXPORT_FORMAT_VERSION

    # 조판 규칙 버전이 오래된 캐시는 입력 mtime이 그대로여도 재생성한다.
    report["format_version"] = PDF_EXPORT_FORMAT_VERSION - 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    refreshed = client.get(f"/api/jobs/{job_id}/pdf")
    assert refreshed.status_code == 200
    refreshed_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert refreshed_report["format_version"] == PDF_EXPORT_FORMAT_VERSION

    # 같은 PDF를 기준면으로 쓰는 한국어 HTML/리더 페이지도 생성된다.
    page = client.get(f"/api/jobs/{job_id}/page/1?lang=ko")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("image/png")
    document = client.get(f"/api/jobs/{job_id}/document.html?lang=ko")
    assert document.status_code == 200
    assert "data:image/png;base64," in document.text
    assert "facsimile-text-block" in document.text
    assert (_job_dir(client, job_id) / "rendered" / "ko" / ".source.json").is_file()

    # UI가 쓰는 대조형은 한 장마다 좌측 원문·우측 한국어를 벡터로 붙인다.
    dual = client.get(f"/api/jobs/{job_id}/pdf?view=dual")
    assert dual.status_code == 200
    assert dual.headers["content-type"] == "application/pdf"
    assert (_job_dir(client, job_id) / "export.ko.dual.pdf").is_file()
    with fitz.open(stream=r.content, filetype="pdf") as single_doc, \
            fitz.open(stream=dual.content, filetype="pdf") as dual_doc:
        assert dual_doc.page_count == single_doc.page_count
        page = dual_doc[0]
        assert page.rect.width == pytest.approx(single_doc[0].rect.width * 2)
        assert page.rect.height == pytest.approx(single_doc[0].rect.height)
        page_text = page.get_text().replace("\xa0", " ")
        assert "Sample page 1" in page_text
        assert KO_TEXT in page_text
        center = single_doc[0].rect.width
        assert any(
            drawing["rect"].x0 == pytest.approx(center)
            and drawing["rect"].x1 == pytest.approx(center)
            and drawing["rect"].height == pytest.approx(page.rect.height)
            and drawing["width"] == pytest.approx(1)
            for drawing in page.get_drawings()
        )


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


def test_pdf_400_bad_view(client):
    job_id = _make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/pdf?view=stacked").status_code == 400


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

    assert client.get(f"/api/jobs/{job_id}/pdf?view=dual").status_code == 200
    dual = job_dir / "export.ko.dual.pdf"
    dual_first = dual.stat().st_mtime_ns

    assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 200
    assert out.stat().st_mtime_ns == first  # 캐시 히트 — 재생성 없음
    assert client.get(f"/api/jobs/{job_id}/pdf?view=dual").status_code == 200
    assert dual.stat().st_mtime_ns == dual_first

    # 번역 레이아웃이 갱신되면(더 새로운 mtime) 재생성
    newer = out.stat().st_mtime + 5
    os.utime(job_dir / "layout.ko.json", (newer, newer))
    assert client.get(f"/api/jobs/{job_id}/pdf").status_code == 200
    assert out.stat().st_mtime_ns != first
    assert client.get(f"/api/jobs/{job_id}/pdf?view=dual").status_code == 200
    assert dual.stat().st_mtime_ns != dual_first


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


def test_build_dual_pdf_keeps_each_page_as_a_source_translation_pair(tmp_path):
    import fitz

    job_dir = _unit_job(tmp_path)
    translated = build_translated_pdf(job_dir, "ko")
    out = build_dual_pdf(
        job_dir / "source.pdf",
        translated.path,
        job_dir / "export.ko.dual.pdf",
    )

    with fitz.open(out) as doc:
        page = doc[0]
        assert page.rect.width == pytest.approx(595 * 2)
        assert page.rect.height == pytest.approx(842)
        text = page.get_text().replace("\xa0", " ")
        assert "Original English sentence" in text
        assert KO_TEXT in text
        assert any(
            drawing["rect"].x0 == pytest.approx(595)
            and drawing["rect"].height == pytest.approx(842)
            for drawing in page.get_drawings()
        )


def test_build_dual_pdf_rejects_different_page_counts(tmp_path):
    import fitz

    job_dir = _unit_job(tmp_path)
    translated = build_translated_pdf(job_dir, "ko")
    mismatched = tmp_path / "mismatched.pdf"
    with fitz.open(translated.path) as doc:
        doc.new_page()
        doc.save(mismatched)

    with pytest.raises(PdfExportError, match="페이지 수가 일치하지"):
        build_dual_pdf(job_dir / "source.pdf", mismatched, job_dir / "dual.pdf")
    assert not (job_dir / "dual.pdf").exists()


def test_build_preserves_vector_graphics_inside_replaced_block(tmp_path):
    """텍스트 리댁션은 같은 bbox 안의 밑줄·도형·차트 선을 제거하지 않는다."""
    import fitz

    job_dir = _unit_job(tmp_path, with_graphic=True)
    with fitz.open(job_dir / "source.pdf") as source:
        assert len(source[0].get_drawings()) == 1
    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        assert len(exported[0].get_drawings()) == 1


def test_growth_stops_when_next_column_block_touches_current_bbox():
    """연속 문단(y1 == next.y0)은 확장 공간이 아니다 — 겹침 회귀 방지."""
    import fitz

    page = type("PageStub", (), {
        "rect": fitz.Rect(0, 0, 600, 800),
        "derotation_matrix": fitz.Identity,
    })()
    current = fitz.Rect(50, 100, 280, 200)
    touching = fitz.Rect(50, 200, 280, 310)
    grown = _free_growth_rect(page, current, [touching])
    assert grown == current


def test_같은_폰트_파일은_resource_이름이_달라도_한_번만_파싱된다(tmp_path):
    """캐시 키에 무시되는 fontname을 넣으면 18MB 파일이 최대 네 번 상주한다.

    한 번의 내보내기가 같은 파일을 uocr-serif·uocr-sans·uocr-table·uocr-serif-2로
    참조하므로 키를 정규화하지 않으면 파싱과 상주 비용이 그대로 곱해진다.
    """
    fontfile, _fontname = _resolve_font("")
    if not fontfile:
        pytest.skip("파일 한글 폰트가 없는 환경")
    _metrics_font.cache_clear()

    first = _metrics_font(fontfile, "uocr-serif")
    assert _metrics_font(fontfile, "uocr-sans") is first
    assert _metrics_font(fontfile, "uocr-table-2") is first
    # 내장 CJK는 fontname이 실제로 폰트를 결정하므로 별개로 캐시된다.
    assert _metrics_font(None, "korea") is not first
    assert _metrics_font.cache_info().currsize == 2


def test_폰트_파일을_같은_경로에_덮어쓰면_측정_캐시가_무효화된다(tmp_path):
    """PDF_EXPORT_FONT 인플레이스 교체가 프로세스 재시작 없이 반영돼야 한다."""
    import os

    source, _fontname = _resolve_font("")
    if not source:
        pytest.skip("파일 한글 폰트가 없는 환경")
    replaced = tmp_path / "deployed.ttf"
    replaced.write_bytes(Path(source).read_bytes())
    _metrics_font.cache_clear()

    before = _metrics_font(str(replaced), "uocr-serif")
    assert _metrics_font(str(replaced), "uocr-serif") is before
    stat = replaced.stat()
    os.utime(replaced, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert _metrics_font(str(replaced), "uocr-serif") is not before


def test_측정용_폰트를_재사용해도_조판_결과가_같다(tmp_path):
    """계획 함수의 fitz.Font는 블록마다 재파싱하지 않고 캐시를 공유한다."""
    job_dir = _unit_job(tmp_path)
    _metrics_font.cache_clear()
    cold = _pdf_text(build_translated_pdf(job_dir, "ko").path.read_bytes())
    warm = _pdf_text(build_translated_pdf(job_dir, "ko").path.read_bytes())
    assert KO_TEXT in cold
    assert warm == cold


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
    import fitz

    job_dir = _unit_job(tmp_path, rotate=90)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1
    exported_text = _pdf_text(result.path.read_bytes())
    assert KO_TEXT in exported_text
    # 회전 페이지에서도 원문 글리프가 남으면 번역문과 겹쳐 찍힌다.
    assert "Original English sentence" not in exported_text
    dual = build_dual_pdf(job_dir / "source.pdf", result.path, job_dir / "dual.pdf")
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(dual) as exported:
        page = exported[0]
        assert page.rect.width == pytest.approx(source[0].rect.width * 2)
        assert page.rect.height == pytest.approx(source[0].rect.height)
        assert "Original English sentence" in page.get_text()
        # PyMuPDF can expose CJK word separators as NBSP after show_pdf_page().
        # Normalize them before checking semantic text preservation.
        assert KO_TEXT in page.get_text().replace("\xa0", " ")


def test_build_keeps_unreplaceable_types(tmp_path):
    job_dir = _unit_job(tmp_path, block_type="equation")
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0
    text = _pdf_text(result.path.read_bytes())
    assert "Original English sentence" in text  # 수식 블록은 원본 유지
    assert KO_TEXT not in text


def test_build_preserves_reference_entry_and_url(tmp_path):
    job_dir = _unit_job(
        tmp_path,
        block_type="ref_text",
        src_text="[1] Author, Original Paper Title, https://example.test/paper.",
    )
    translated = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    translated[0]["blocks"][0]["content"] = (
        "[1] Author, 번역된 논문 제목, https://example.test/paper."
    )
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko")
    text = _pdf_text(result.path.read_bytes())
    assert result.replaced == 0
    assert result.specialist_kept["reference"] == 1
    assert "https://example.test/paper." in text
    assert "Original Paper Title" in text
    assert "번역된 논문 제목" not in text


def test_참고문헌_중복scheme_url은_미세교정으로_한번만_남는다(tmp_path):
    """보존 대상 ref_text에서도 allowlist된 조판 오류(`https://https://`)만 고친다.

    `_microfix_plan` 경로는 원문 한 행 안의 더 짧은 치환만 허용하므로 서지
    항목의 다른 글리프는 그대로 남아야 한다. 겹친 링크 annotation의 URI도 같은
    정상 주소로 맞춘다.
    """
    import fitz

    job_dir = tmp_path / "microfix-job"
    job_dir.mkdir()
    doubled = "https://https://"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 100), "[1] Author, Paper Title,", fontsize=9)
    # 중복 scheme만 별도 span이 되도록 다른 위치·폰트로 기록한다.
    page.insert_text((60, 114), doubled, fontsize=9, fontname="cour")
    for link_rect in (fitz.Rect(58, 104, 160, 118), fitz.Rect(58, 150, 160, 164)):
        # 첫 annotation은 리댁션에서 사라지지만, 다음 행으로 이어진 같은 URI의
        # 링크는 남는다 — 그 링크도 같은 정상 주소로 맞춰야 한다.
        page.insert_link({
            "kind": fitz.LINK_URI,
            "from": link_rect,
            "uri": f"{doubled}example.test/paper",
        })
    doc.save(job_dir / "source.pdf")
    doc.close()

    content = f"[1] Author, Paper Title, {doubled}example.test/paper"
    original = [{"page": 1, "width": 595, "height": 842, "blocks": [{
        "type": "ref_text",
        "bbox": [round(58 / 595 * 999), round(88 / 842 * 999),
                 round(300 / 595 * 999), round(120 / 842 * 999)],
        "content": content,
        "fs": 9 / 595 * 100,
    }]}]
    translated = json.loads(json.dumps(original))
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    text = _pdf_text(result.path.read_bytes())
    with fitz.open(result.path) as exported:
        uris = [link.get("uri") for link in exported[0].get_links()]

    assert result.specialist_kept["reference"] == 1
    assert result.replaced == 1, result.report()
    assert doubled not in text, text
    assert "https://" in text and "Paper Title" in text, text
    assert uris == ["https://example.test/paper"], uris


def test_build_redaction_includes_source_glyphs_past_ocr_bbox(tmp_path):
    """OCR bbox 밖으로 조금 나온 긴 원문 span 꼬리도 번역문 옆에 남기지 않는다."""
    import fitz

    job_dir = tmp_path / "short-bbox-job"
    job_dir.mkdir()
    original_text = "Original reference URL ending.pdf."
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 100), original_text, fontsize=12)
    doc.save(job_dir / "source.pdf")
    doc.close()

    # 실제 span은 x≈250pt까지 가지만 OCR bbox는 x=210pt에서 끝나는 상황.
    original = [{"page": 1, "width": 595, "height": 842, "blocks": [{
        "type": "text",
        "bbox": [round(60 / 595 * 999), round(84 / 842 * 999),
                 round(210 / 595 * 999), round(104 / 842 * 999)],
        "content": original_text,
        "fs": 12 / 595 * 100,
    }]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = "번역 참고문헌"
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko")
    text = _pdf_text(result.path.read_bytes())
    assert "번역 참고문헌" in text
    assert "Original" not in text
    assert "ending.pdf." not in text


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


def test_build_preserves_title_size_weight_and_center_alignment(tmp_path, monkeypatch):
    """CJK 행높이 때문에 제목을 축소하지 않고 빈 공간·굵기·가운데 정렬을 쓴다."""
    import fitz

    # 개발 머신의 시스템 한글 폰트 유무와 관계없이 Linux 최종 폴백을 검증한다.
    # PyMuPDF 내장 CJK 폰트는 Font.text_length()와 실제 삽입 폭이 다르다.
    monkeypatch.setattr(
        "app.pipeline.pdf_export._resolve_font",
        lambda *args, **kwargs: (None, "korea"),
    )

    job_dir = tmp_path / "styled-title-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((207, 78), "Original Paper Title", fontsize=18, fontname="hebo")
    page.insert_text((70, 125), "Next block must remain", fontsize=10)
    doc.save(job_dir / "source.pdf")
    doc.close()

    def norm_x(x):
        return round(x / 595 * 999)

    def norm_y(y):
        return round(y / 842 * 999)

    original = [{"page": 1, "width": 595, "height": 842, "blocks": [
        {
            "type": "title",
            "bbox": [norm_x(180), norm_y(59), norm_x(415), norm_y(79)],
            "content": "Original Paper Title",
        },
        {
            "type": "text",
            "bbox": [norm_x(70), norm_y(112), norm_x(250), norm_y(128)],
            "content": "Next block must remain",
        },
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = "한국어 논문 제목"
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        page = exported[0]
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if "한국어" in span.get("text", "")
        ]
        streams = b"\n".join(
            exported.xref_stream(xref) for xref in page.get_contents()
        )

    assert spans, "번역 제목이 삽입되어야 한다"
    title = spans[0]
    assert title["size"] >= 16, title
    assert abs((title["bbox"][0] + title["bbox"][2]) / 2 - 595 / 2) < 5, title
    assert b"2 Tr" in streams, "CJK 제목도 fill+stroke로 굵기 계층을 보존해야 한다"
    assert "Next block must remain" in _pdf_text(result.path.read_bytes())


# ── 폰트 폴백·이모지 이미지·그림 위 텍스트 (Docker 폰트 부재 회귀) ─────────
def test_resolve_font_falls_back_to_builtin_when_no_candidates(tmp_path, monkeypatch):
    """후보 폰트가 전부 없으면 (None, 'korea') 내장 폴백 계약을 지킨다.

    ⚠ candidates 기본값은 def 시점에 바인딩되므로 모듈 상수 monkeypatch 대신
    빈 튜플을 직접 전달한다. fc-list 런타임 탐색도 결정적으로 비활성화한다.
    """
    monkeypatch.setattr("app.pipeline.pdf_export._fontconfig_candidates", lambda: ())
    assert _resolve_font("", ()) == (None, "korea")
    assert _resolve_font(str(tmp_path / "missing.ttf"), ()) == (None, "korea")


def test_resolve_font_uses_fontconfig_discovery_as_last_resort(monkeypatch):
    """정적 후보 전멸 시 fc-list가 찾은 한글 폰트를 같은 has_glyph 검증으로 채택."""
    from app.pipeline.pdf_export import _SYSTEM_FONT_CANDIDATES

    real_font = next((p for p in _SYSTEM_FONT_CANDIDATES if Path(p).is_file()), None)
    if real_font is None:
        pytest.skip("시스템 한글 폰트가 없어 fc-list 탐색을 대리 검증할 수 없음")
    monkeypatch.setattr(
        "app.pipeline.pdf_export._fontconfig_candidates", lambda: (real_font,))
    assert _resolve_font("", ()) == (real_font, "uocr-ko")


def test_build_warns_when_only_builtin_cjk_font_available(tmp_path, monkeypatch):
    """내장 'korea' 폴백은 실패 대신 경고를 리포트로 드러낸다(방어적 폴백 유지)."""
    monkeypatch.setattr(
        "app.pipeline.pdf_export._resolve_font",
        lambda *args, **kwargs: (None, "korea"),
    )
    job_dir = _unit_job(tmp_path)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1  # 폴백에도 내보내기는 성공한다
    assert any("폰트" in w for w in result.warnings)
    report = json.loads((job_dir / "export.ko.report.json").read_text(encoding="utf-8"))
    assert report["warning_count"] >= 1
    assert any("폰트" in w for w in report["warnings"])


def _png_stamp(size: int = 8, value: int = 120) -> bytes:
    import fitz

    pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, size, size))
    pm.clear_with(value)
    return pm.tobytes("png")


def test_build_removes_inline_emoji_image_inside_replaced_block(tmp_path):
    """교체 블록 안에 완전히 포함된 소형 이미지(이모지의 '이미지 절반')는 제거된다.

    macOS Quartz 산출 PDF는 컬러 이모지를 보이지 않는 텍스트 글리프 + 이미지
    XObject로 이중 기록한다 — 텍스트 리댁션만으로는 이미지가 번역문 위에 남는다.
    같은 이미지라도 블록 밖 인스턴스는 보존되어야 한다(인스턴스 단위 제거).
    """
    import fitz

    job_dir = tmp_path / "emoji-job"
    job_dir.mkdir()
    stamp = _png_stamp()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(60, 85, 535, 250), "Original English sentence", fontsize=12)
    page.insert_image(fitz.Rect(100, 120, 112, 132), stream=stamp)  # 블록 안 '이모지'
    page.insert_image(fitz.Rect(100, 700, 112, 712), stream=stamp)  # 블록 밖 — 보존
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300],
         "content": "Original English sentence", "fs": 2.0},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = KO_TEXT
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1
    assert KO_TEXT in _pdf_text(result.path.read_bytes())
    with fitz.open(result.path) as exported:
        boxes = [fitz.Rect(info["bbox"]) for info in exported[0].get_image_info()]
    assert any(abs(box.y0 - 700) < 2.0 for box in boxes), boxes  # 블록 밖 보존
    assert not any(box.y0 < 200 for box in boxes), boxes         # 블록 안 제거


def test_build_keeps_letter_sized_limit_and_preserves_inline_figure(tmp_path):
    """이모지 2차 패스는 '글자 한 칸 크기'만 지운다 — 인라인 그림·아이콘은 보존.

    면적비(25%)만 보면 넓은 텍스트 블록 안의 100x100pt 로고·차트도 걸려 경고
    없이 사라진다. docs/ARCHITECTURE.md의 "리댁션은 블록 안의 그림을 제거하지
    않는다" 계약을 지키려면 절대 크기 상한이 함께 필요하다.
    """
    import fitz

    job_dir = tmp_path / "inline-figure-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(60, 85, 535, 250), "Original English sentence", fontsize=12)
    # 블록(59.5,84.2)–(535.5,252.6) 안에 완전히 든 100x100pt 인라인 그림.
    # 면적비는 약 12%라 기존 25% 조건만으로는 제거 대상이 된다.
    page.insert_image(
        fitz.Rect(400, 130, 500, 230), stream=_png_stamp(32, 90),
        keep_proportion=False)
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300],
         "content": "Original English sentence", "fs": 2.0},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = KO_TEXT
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1, result.report()
    with fitz.open(result.path) as exported:
        boxes = [fitz.Rect(info["bbox"]) for info in exported[0].get_image_info()]
    assert any(
        abs(box.width - 100) < 2.0 and abs(box.height - 100) < 2.0 for box in boxes
    ), boxes


def test_build_keeps_text_block_overlapping_figure_image(tmp_path):
    """래스터 그림과 크게 겹치는(≥30%) OCR 텍스트 블록은 교체하지 않는다.

    그림 속 텍스트는 OCR 오독이 잦고, 번역을 스탬프하면 리댁션되지 않은 원문
    그림·텍스트와 이중으로 겹쳐 보인다 — 원본 조판 보존이 항상 우월하다.
    """
    import fitz

    job_dir = tmp_path / "figure-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # keep_proportion=False: 블록 표시 영역(59.5,84.2..535.5,252.6)을 거의 다
    # 덮는 그림 — 겹침비 약 98%.
    page.insert_image(
        fitz.Rect(60, 85, 535, 250), stream=_png_stamp(32, 200),
        keep_proportion=False)
    page.insert_textbox(
        fitz.Rect(60, 85, 535, 250), "Original English sentence", fontsize=12)
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300],
         "content": "Original English sentence", "fs": 2.0},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = KO_TEXT
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0 and result.kept >= 1
    assert result.specialist_kept.get("figure_text") == 1
    assert any("그림" in w for w in result.warnings)
    text = _pdf_text(result.path.read_bytes())
    assert "Original English sentence" in text  # 원문·그림 그대로 보존
    assert KO_TEXT not in text


def test_build_replaces_text_over_full_page_scan_background(tmp_path):
    """전면 스캔 래스터(페이지 85% 이상)는 배경으로 간주 — 번역이 생략되지 않는다."""
    import fitz

    job_dir = tmp_path / "scan-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=_png_stamp(32, 250), keep_proportion=False)
    page.insert_textbox(
        fitz.Rect(60, 85, 535, 250), "Original English sentence", fontsize=12)
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300],
         "content": "Original English sentence", "fs": 2.0},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = KO_TEXT
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1
    assert "figure_text" not in result.specialist_kept
    assert KO_TEXT in _pdf_text(result.path.read_bytes())
    with fitz.open(result.path) as exported:
        # 전면 배경 이미지는 이모지 패스(완전 포함 + 소면적 한정)에 걸리지 않는다.
        assert exported[0].get_image_info(), "전면 스캔 배경은 보존되어야 한다"
