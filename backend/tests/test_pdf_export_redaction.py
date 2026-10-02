"""리댁션 영역이 보존해야 할 이웃 줄의 글리프를 지우지 않는다.

MuPDF는 리댁션 사각형에 **닿는** 글리프를 통째로 지운다. 글리프 상자는 폰트
ascender/descender 상자(PyMuPDF span bbox의 세로 범위)의 위·아래를 10%씩 깎은
것이라, span bbox 전체(+0.25pt)를 걸면 10pt 글자/12pt 행간에서 아래 줄 글리프
상자에 닿는다. 번역 블록 바로 아래·위에 붙은 보존 블록(번역이 원문과 같은 줄,
수식, 공간 부족으로 보존된 문단)의 한 줄이 리포트에 흔적 없이 사라졌다.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from app.pipeline.pdf_export import _SourceSpan, _source_span_records, build_translated_pdf
from app.pipeline.pdf_export.fitting import _microfix_plan, _plan_listing_lines
from app.pipeline.pdf_export.models import _LineSegment
from app.pipeline.pdf_export.spans import _source_text_rects, _span_redaction_band

PAGE_W, PAGE_H = 595.0, 842.0
_MYUNGJO = "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf"
_FONTS = [("tiro", None), ("helv", None), ("cour", None)]
if Path(_MYUNGJO).is_file():
    _FONTS.append(("myj", _MYUNGJO))


def _redact(page, rects) -> None:
    for rect in rects:
        page.add_redact_annot(rect)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )


@pytest.mark.parametrize(("fontname", "fontfile"), _FONTS)
@pytest.mark.parametrize("leading", [10.0, 12.0])
def test_baseline_band_removes_its_line_but_not_the_next(fontname, fontfile, leading):
    """띠는 자기 줄의 모든 글리프(밑줄·쉼표·따옴표 포함)를 지우고 이웃 줄은 남긴다."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    kwargs = {"fontsize": 10, "fontname": fontname}
    if fontfile:
        kwargs["fontfile"] = fontfile
    page.insert_text((40, 100 - leading), "Above line keeps every word", **kwargs)
    page.insert_text((40, 100), 'Target: under_score, "quote" gjpqy.', **kwargs)
    page.insert_text((40, 100 + leading), "Below line keeps every word", **kwargs)
    records = _source_span_records(fitz, page)
    target = [span for span in records if span.text.startswith("Target")]
    assert len(target) == 1

    _redact(page, [_span_redaction_band(fitz, target[0])])
    text = page.get_text()
    doc.close()

    assert "Target" not in text and "under" not in text and "quote" not in text, text
    assert "Above line keeps every word" in text, text
    assert "Below line keeps every word" in text, text


def _type3_page_pdf(font_bbox: str) -> bytes:
    """FontBBox를 마음대로 정한 Type3 글꼴로 세 줄(12pt 행간)을 쓴 한 쪽 PDF.

    글리프는 0~700 단위 높이의 채운 사각형이다. FontBBox가 그보다 낮으면(퇴화 메트릭)
    PyMuPDF는 span bbox를 1em으로 늘려 보고하지만 MuPDF 리댁션은 FontBBox로 글리프 상자를
    잡는다.
    """
    chars = "abcdeiklmnoprtuwxy "
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    procs = {}
    for char in chars:
        stream = b"250 0 d0" if char == " " else b"600 0 0 0 500 700 d1 50 0 400 700 re f"
        procs[char] = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    names = {char: "space" if char == " " else char for char in chars}
    charprocs = add(
        b"<< " + b" ".join(b"/%s %d 0 R" % (names[c].encode(), procs[c]) for c in chars) + b" >>"
    )
    first, last = min(map(ord, chars)), max(map(ord, chars))
    widths = " ".join(
        "250" if chr(code) == " " else ("600" if chr(code) in chars else "0")
        for code in range(first, last + 1)
    )
    differences = " ".join(f"{ord(c)} /{names[c]}" for c in chars)
    bfchars = "\n".join(f"<{ord(c):02X}> <{ord(c):04X}>" for c in chars)
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /T3 def "
        f"1 begincodespacerange <00> <FF> endcodespacerange {len(chars)} beginbfchar\n"
        f"{bfchars}\nendbfchar endcmap CMapName currentdict /CMap defineresource pop end end"
    ).encode()
    tounicode = add(b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream")
    font = add((
        f"<< /Type /Font /Subtype /Type3 /FontBBox [{font_bbox}] "
        "/FontMatrix [0.001 0 0 0.001 0 0] "
        f"/CharProcs {charprocs} 0 R /Encoding << /Type /Encoding /Differences [{differences}] >> "
        f"/FirstChar {first} /LastChar {last} /Widths [{widths}] /Resources << >> "
        f"/ToUnicode {tounicode} 0 R >>"
    ).encode())
    content = (
        b"BT /T3 10 Tf 50 140 Td (upper line kept) Tj 0 -12 Td (middle line redact) Tj "
        b"0 -12 Td (lower line kept) Tj ET"
    )
    contents = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    page_id = len(objects) + 1
    add((
        f"<< /Type /Page /Parent {page_id + 1} 0 R /MediaBox [0 0 300 200] "
        f"/Resources << /Font << /T3 {font} 0 R >> >> /Contents {contents} 0 R >>"
    ).encode())
    add(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode())
    catalog = add(f"<< /Type /Catalog /Pages {page_id + 1} 0 R >>".encode())
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, catalog, xref,
    )
    return out


@pytest.mark.parametrize("font_bbox", ["0 0 600 100", "0 -200 600 800"])
def test_band_still_removes_glyphs_of_fonts_with_degenerate_metrics(font_bbox):
    """원래 ascender−descender가 아주 작은 폰트도 자기 줄은 지우고 이웃 줄은 남긴다.

    PyMuPDF는 그런 span의 bbox를 1em으로 늘려 보고한다. 거기서 구한 띠가 MuPDF의 실제
    글리프 상자(FontBBox 0–0.1em)에 닿지 않아 원문이 남고 번역이 겹쳐 찍혔다(감사 pdf-8).
    """
    doc = fitz.open("pdf", _type3_page_pdf(font_bbox))
    page = doc[0]
    middle = [span for span in _source_span_records(fitz, page) if "middle" in span.text]
    assert len(middle) == 1

    _redact(page, [_span_redaction_band(fitz, middle[0])])
    text = page.get_text()
    doc.close()

    assert "middle" not in text, text
    assert "upper line kept" in text and "lower line kept" in text, text


def test_span_bbox_redaction_reaches_the_next_line_at_normal_leading():
    """대조군 — 예전 방식(span bbox + 0.25pt)은 12pt 행간에서 아래 줄을 지운다."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((40, 100), "Target line here", fontsize=10, fontname="tiro")
    page.insert_text((40, 112), "Below line keeps every word", fontsize=10, fontname="tiro")
    target = [s for s in _source_span_records(fitz, page) if s.text.startswith("Target")][0]
    _redact(page, [target.rect + (-0.25, -0.25, 0.25, 0.25)])
    text = page.get_text()
    doc.close()
    assert "Below line keeps every word" not in text


@pytest.mark.parametrize("rotation", [90, 270])
def test_band_follows_the_line_direction_on_rotated_text(rotation):
    """세로로 놓인 줄(회전 페이지의 화면상 가로 줄)도 줄 방향 기준 띠로 지운다."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.set_rotation(rotation)
    for offset, text in ((-12, "Above line keeps every word"), (0, "Target line to remove"),
                         (12, "Below line keeps every word")):
        point = fitz.Point(60, 100 + offset) * page.derotation_matrix
        page.insert_text(point, text, fontsize=10, fontname="tiro", rotate=rotation)
    target = [s for s in _source_span_records(fitz, page) if s.text.startswith("Target")][0]
    assert abs(target.dir[1]) > 0.99  # 비회전 공간에서는 세로로 진행한다

    band = _span_redaction_band(fitz, target)
    # 띠는 줄 방향으로 길고 그 수직 방향으로 얇다.
    assert band.height > band.width * 5, band
    _redact(page, [band])
    text = page.get_text()
    doc.close()
    assert "Target" not in text, text
    assert "Above line keeps every word" in text and "Below line keeps every word" in text, text


def test_rects_without_span_metadata_keep_the_legacy_padding():
    """사각형만 넘기는 호출부(호환 경로)는 예전과 같은 사각형(+0.25pt)을 쓴다."""
    page = type("PageStub", (), {"mediabox": fitz.Rect(0, 0, 500, 500)})()
    rect = fitz.Rect(10, 10, 100, 20)
    assert _source_text_rects(page, rect, [fitz.Rect(10, 10, 100, 20)]) == (
        fitz.Rect(9.75, 9.75, 100.25, 20.25),
    )


def _span(text: str, x0: float, baseline: float, size: float = 10.0) -> _SourceSpan:
    width = len(text) * size * 0.5
    return _SourceSpan(
        fitz.Rect(x0, baseline - 1.05 * size, x0 + width, baseline + 0.28 * size),
        text, size, 0, (x0, baseline),
    )


def test_listing_and_microfix_paths_share_the_baseline_band():
    """줄 단위 리스팅·참고문헌 미세 교정도 같은 띠 함수를 쓴다(span bbox 아님)."""
    span = _span("for i in range(n):", 72, 200)
    segment = _LineSegment(
        "i를 n까지 반복:", "for i in range(n):", (span,), 72, 300, 200, 10.0,
        (span.rect.y0, span.rect.y1),
    )
    page = fitz.open().new_page(width=PAGE_W, height=PAGE_H)
    planned, changed = _plan_listing_lines(page, (segment,), "korea", None, [], 0)
    assert changed == 1 and len(planned) == 1
    band = _span_redaction_band(fitz, span)
    assert planned[0].redact_rects == (band,)
    assert band.y1 < span.origin[1] and band.y0 > span.rect.y0

    scheme = _span("https://https://", 72, 300)
    fix = _microfix_plan(
        fitz, scheme.rect, scheme.origin, "https://", 10.0, "cour", None, (scheme,),
    )
    assert fix is not None
    assert fix.redact_rects == (_span_redaction_band(fitz, scheme),)


def _bbox(rect: fitz.Rect) -> list[float]:
    return [
        rect.x0 / PAGE_W * 999, rect.y0 / PAGE_H * 999,
        rect.x1 / PAGE_W * 999, rect.y1 / PAGE_H * 999,
    ]


def test_export_keeps_preserved_neighbour_lines_at_twelve_point_leading(tmp_path):
    """번역 블록 위·아래에 문단 간격 없이 붙은 보존 블록의 줄이 그대로 남는다."""
    job_dir = tmp_path / "tight-leading"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    above = ["Kept paragraph above line one stays.", "Kept paragraph above line two stays."]
    target = ["Translated paragraph first source line.", "Translated paragraph second line."]
    below = ["Kept paragraph below line one stays.", "Kept paragraph below line two stays."]
    y = 200.0
    rects = []
    for lines in (above, target, below):
        top = y - 10
        for line in lines:
            page.insert_text((72, y), line, fontsize=10, fontname="tiro")
            y += 12.0
        rects.append(fitz.Rect(70, top, 330, y - 12 + 3))
    doc.save(job_dir / "source.pdf")
    doc.close()

    blocks = [
        {"type": "text", "bbox": _bbox(rect), "content": "\n".join(lines),
         "fs": 10 / PAGE_W * 100}
        for rect, lines in zip(rects, (above, target, below))
    ]
    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": blocks}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][1]["content"] = "번역된 문단 첫 줄입니다.\n번역된 문단 둘째 줄입니다."
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1, result.report()
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
    for line in above + below:
        assert line in text, (line, text)
    assert "Translated paragraph" not in text, text
    assert "번역된 문단" in text, text
