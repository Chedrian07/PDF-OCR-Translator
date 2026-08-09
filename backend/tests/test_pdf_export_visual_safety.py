"""PDF export regressions for text that is readable and collision-free.

These checks intentionally inspect the text geometry in the written PDF.  A
successful ``insert_textbox`` return value only proves that PyMuPDF accepted a
layout; it does not prove that the selected CJK font's glyph boxes are
separated from adjacent lines or other page content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.pdf_export import (
    _plain_text,
    _resolve_font,
    _restore_title_prefix,
    _textbox_ink_rect,
    build_translated_pdf,
)


PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0


def _layout_bbox(rect) -> list[float]:
    """Convert a PDF-space rectangle to the layout contract's 0-999 space."""
    return [
        rect.x0 / PAGE_WIDTH * 999,
        rect.y0 / PAGE_HEIGHT * 999,
        rect.x1 / PAGE_WIDTH * 999,
        rect.y1 / PAGE_HEIGHT * 999,
    ]


def _write_layout_pair(
    job_dir: Path,
    original_blocks: list[dict],
    translated_text: list[str],
) -> None:
    assert len(original_blocks) == len(translated_text)
    original = [{
        "page": 1,
        "width": PAGE_WIDTH,
        "height": PAGE_HEIGHT,
        "blocks": original_blocks,
    }]
    translated = json.loads(json.dumps(original))
    for block, text in zip(translated[0]["blocks"], translated_text):
        block["content"] = text
    (job_dir / "layout.json").write_text(
        json.dumps(original), encoding="utf-8",
    )
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )


def _line_entries(page) -> list[tuple[str, object]]:
    """Return visible text and the actual bbox reported for each PDF line."""
    import fitz

    entries: list[tuple[str, object]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                entries.append((text, fitz.Rect(line["bbox"])))
    return entries


def _cover(rects: list[object]):
    assert rects
    rect = +rects[0]
    for other in rects[1:]:
        rect.include_rect(other)
    return rect


def _overlap_area(left, right) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height


@pytest.fixture
def real_cjk_fontfile() -> str:
    """Use AppleMyungjo, Noto, or another verified Korean font file."""
    fontfile, _fontname = _resolve_font("")
    if fontfile is None or not Path(fontfile).is_file():
        pytest.skip("no usable file-backed CJK font is installed")
    return fontfile


def test_plain_text_converts_common_raw_tex_to_readable_unicode():
    converted = _plain_text(
        r"\(P_{L}\), \(L_{max}\), \(\mathcal{M}\), "
        r"\(\oplus\), \(\ldots\), \(\beta_{1}\)"
    )

    assert converted == "P(L), L(max), M, ⊕, …, β(1)"
    assert not any(raw in converted for raw in ("\\", "{", "}", "_"))


def test_title_prefix_restore_does_not_treat_abstract_as_appendix_a():
    assert _restore_title_prefix("Abstract", "초록") == "초록"
    assert _restore_title_prefix("A Detailed Results", "상세 결과") == "A 상세 결과"


def test_textbox_dry_run_ink_rect_matches_real_cjk_spans(real_cjk_fontfile: str):
    """PyMuPDF spare가 놓치는 CJK descender까지 계획 bbox에 들어가야 한다."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    rect = fitz.Rect(60, 80, 420, 170)
    text = "첫줄 한국어\n둘째줄 한국어\n셋째줄 한국어\n넷째줄 한국어"
    font = fitz.Font(fontfile=real_cjk_fontfile)
    shape = page.new_shape()
    spare = shape.insert_textbox(
        rect,
        text,
        fontsize=10,
        fontname="visual-safety-cjk",
        fontfile=real_cjk_fontfile,
        lineheight=1.44,
    )
    assert spare >= 0
    planned = _textbox_ink_rect(page, rect, shape, 10, font, spare, False)
    shape.commit()
    actual = _cover([
        fitz.Rect(line["bbox"])
        for block in page.get_text("dict").get("blocks", [])
        for line in block.get("lines", [])
    ])
    doc.close()

    assert planned.y0 == pytest.approx(actual.y0, abs=0.05)
    assert planned.y1 == pytest.approx(actual.y1, abs=0.05)


def test_exported_cjk_body_lines_have_disjoint_bboxes(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """A real CJK font's adjacent glyph boxes must not overlap within one block."""
    import fitz

    job_dir = tmp_path / "multiline-body"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "ORIGINAL BODY", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    body_rect = fitz.Rect(60, 80, 320, 150)
    original = [{
        "type": "text",
        "bbox": _layout_bbox(body_rect),
        "content": "ORIGINAL BODY",
        "fs": 12 / PAGE_WIDTH * 100,
    }]
    lines = [
        "첫줄표식 한국어 본문",
        "둘줄표식 글리프 검사",
        "셋줄표식 줄 간격 검사",
    ]
    _write_layout_pair(job_dir, original, ["\n".join(lines)])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    assert result.replaced == 1
    with fitz.open(result.path) as exported:
        entries = _line_entries(exported[0])

    translated = sorted(
        (rect for text, rect in entries if any(marker in text for marker in lines)),
        key=lambda rect: rect.y0,
    )
    assert len(translated) == 3, entries
    for upper, lower in zip(translated, translated[1:]):
        assert upper.y1 <= lower.y0 + 0.01, (upper, lower)


@pytest.mark.parametrize("lower_y0", [100.0, 110.0], ids=["overlap", "touch"])
def test_overlapping_or_touching_layout_blocks_export_without_text_collision(
    tmp_path: Path,
    real_cjk_fontfile: str,
    lower_y0: float,
):
    """Conflicting layout blocks may translate or preserve, but never collide."""
    import fitz

    job_dir = tmp_path / f"neighbor-blocks-{lower_y0:g}"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 98), "SOURCE TOP", fontsize=12)
    page.insert_text((60, lower_y0 + 23), "SOURCE BOTTOM", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    upper_rect = fitz.Rect(60, 80, 320, 110)
    lower_rect = fitz.Rect(60, lower_y0, 320, lower_y0 + 30)
    original = [
        {
            "type": "text",
            "bbox": _layout_bbox(upper_rect),
            "content": "SOURCE TOP",
            "fs": 12 / PAGE_WIDTH * 100,
        },
        {
            "type": "text",
            "bbox": _layout_bbox(lower_rect),
            "content": "SOURCE BOTTOM",
            "fs": 12 / PAGE_WIDTH * 100,
        },
    ]
    _write_layout_pair(
        job_dir,
        original,
        ["TOPMARK 위쪽 번역\nTOPMARK 이어지는 번역", "BOTTOMMARK 아래쪽 번역"],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        text = page.get_text().replace("\xa0", " ")
        top = page.search_for("TOPMARK") or page.search_for("SOURCE TOP")
        bottom = page.search_for("BOTTOMMARK") or page.search_for("SOURCE BOTTOM")

    assert top and bottom, text
    top_rect = _cover(top)
    bottom_rect = _cover(bottom)
    assert _overlap_area(top_rect, bottom_rect) <= 0.01, (top_rect, bottom_rect)


def test_expanded_translation_does_not_cover_unlisted_source_pdf_span(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """Source text absent from layout.json is still an expansion obstacle."""
    import fitz

    job_dir = tmp_path / "unlisted-source-span"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 98), "LAYOUT SOURCE", fontsize=12)
    page.insert_text((60, 145), "UNLISTED SOURCE SPAN", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    layout_rect = fitz.Rect(60, 80, 320, 108)
    original = [{
        "type": "text",
        "bbox": _layout_bbox(layout_rect),
        "content": "LAYOUT SOURCE",
        "fs": 12 / PAGE_WIDTH * 100,
    }]
    translated_lines = [
        "EXPANDMARK 번역 첫 번째 줄",
        "EXPANDMARK 번역 두 번째 줄",
        "EXPANDMARK 번역 세 번째 줄",
        "EXPANDMARK 번역 네 번째 줄",
        "EXPANDMARK 번역 다섯 번째 줄",
    ]
    _write_layout_pair(job_dir, original, ["\n".join(translated_lines)])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        text = page.get_text().replace("\xa0", " ")
        translated = page.search_for("EXPANDMARK")
        unlisted = page.search_for("UNLISTED SOURCE SPAN")

    assert unlisted, text
    if not translated:
        assert "LAYOUT SOURCE" in text, text
    else:
        for translated_rect in translated:
            for source_rect in unlisted:
                assert _overlap_area(translated_rect, source_rect) <= 0.01, (
                    translated_rect,
                    source_rect,
                )
