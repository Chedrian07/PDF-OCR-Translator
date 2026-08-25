"""PDF export regressions for text that is readable and collision-free.

These checks intentionally inspect the text geometry in the written PDF.  A
successful ``insert_textbox`` return value only proves that PyMuPDF accepted a
layout; it does not prove that the selected CJK font's glyph boxes are
separated from adjacent lines or other page content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline import pdf_export as pdf_export_module
from app.pipeline.pdf_export import (
    _SourceSpan,
    _block_rect,
    _listing_segments,
    _load_pages,
    _plain_text,
    _noto_visible_ink_bounds,
    _plan_single_line,
    _protect_trailing_words,
    _reconstruct_rich_runs,
    _reflow_flattened_text,
    _resolve_font,
    _rich_prefix_markup,
    _restore_title_prefix,
    _table_cell_rects,
    _table_cells,
    _textbox_ink_rect,
    build_translated_pdf,
)


PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0


def _layout_bbox(
    rect,
    *,
    page_width: float = PAGE_WIDTH,
    page_height: float = PAGE_HEIGHT,
) -> list[float]:
    """Convert a PDF-space rectangle to the layout contract's 0-999 space."""
    return [
        rect.x0 / page_width * 999,
        rect.y0 / page_height * 999,
        rect.x1 / page_width * 999,
        rect.y1 / page_height * 999,
    ]


def _write_layout_pair(
    job_dir: Path,
    original_blocks: list[dict],
    translated_text: list[str],
    *,
    page_width: float = PAGE_WIDTH,
    page_height: float = PAGE_HEIGHT,
) -> None:
    assert len(original_blocks) == len(translated_text)
    original = [{
        "page": 1,
        "width": page_width,
        "height": page_height,
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
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).replace("\xa0", " ")
            if text.strip():
                entries.append((text, fitz.Rect(line["bbox"])))
    return entries


def _span_entries(page) -> list[dict]:
    return [
        span
        for block in page.get_text("dict").get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if str(span.get("text") or "").strip()
    ]


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


def test_dense_narrow_page_translates_all_blocks_without_greedy_starvation(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """An earlier body must not consume the following title's placement."""
    import fitz

    page_width = 280.0
    page_height = 360.0
    job_dir = tmp_path / "dense-one-column-page"
    job_dir.mkdir()

    source_specs = [
        (
            "SOURCE ALPHA PARAGRAPH WITH ENOUGH CHARACTERS TO DOMINATE",
            (24, 50),
            10,
        ),
        ("SOURCE HEADING", (24, 128), 14),
        ("SOURCE BETA PARAGRAPH", (24, 158), 10),
        ("SOURCE GAMMA PARAGRAPH", (24, 233), 10),
    ]
    source = fitz.open()
    page = source.new_page(width=page_width, height=page_height)
    for text, origin, fontsize in source_specs:
        page.insert_text(origin, text, fontsize=fontsize)
    source.save(job_dir / "source.pdf")
    source.close()

    rects = [
        fitz.Rect(24, 30, 256, 126),
        fitz.Rect(24, 105, 256, 136),
        fitz.Rect(24, 137, 256, 210),
        fitz.Rect(24, 212, 256, 285),
    ]
    block_types = ["text", "title", "text", "text"]
    original = [
        {
            "type": block_type,
            "bbox": _layout_bbox(
                rect,
                page_width=page_width,
                page_height=page_height,
            ),
            "content": source_text,
            "fs": fontsize / page_width * 100,
        }
        for (source_text, _origin, fontsize), rect, block_type in zip(
            source_specs, rects, block_types,
        )
    ]
    translated = [
        "\n".join(f"ALPHAMARK 번역 본문 {line}" for line in range(1, 7)),
        "TITLEMARK 중간 제목",
        "\n".join(f"BETAMARK 번역 본문 {line}" for line in range(1, 4)),
        "\n".join(f"GAMMAMARK 번역 본문 {line}" for line in range(1, 4)),
    ]
    _write_layout_pair(
        job_dir,
        original,
        translated,
        page_width=page_width,
        page_height=page_height,
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        text = page.get_text().replace("\xa0", " ")
        translated_lines = [
            (line_text, rect)
            for line_text, rect in _line_entries(page)
            if any(
                marker in line_text
                for marker in ("ALPHAMARK", "TITLEMARK", "BETAMARK", "GAMMAMARK")
            )
        ]

    assert result.replaced == len(original), (result.report(), text)
    assert all(marker in text for marker in (
        "ALPHAMARK", "TITLEMARK", "BETAMARK", "GAMMAMARK",
    )), text
    assert not any(marker in text for marker in (
        "SOURCE ALPHA", "SOURCE HEADING", "SOURCE BETA", "SOURCE GAMMA",
    )), text
    assert len(translated_lines) == 13, translated_lines
    translated_lines.sort(key=lambda item: item[1].y0)
    for (upper_text, upper), (lower_text, lower) in zip(
        translated_lines, translated_lines[1:],
    ):
        assert upper.y1 <= lower.y0 + 0.01, (
            upper_text,
            upper,
            lower_text,
            lower,
        )


def test_short_table_header_keeps_source_relative_readable_size(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """A short translated header should use measured source typography."""
    import fitz

    job_dir = tmp_path / "compact-table-header"
    job_dir.mkdir()
    rows = [
        ("Header", "Metric"),
        ("Alpha", "10"),
        ("Beta", "20"),
        ("Gamma", "30"),
    ]
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for (label, value), baseline in zip(rows, (94, 110, 126, 142)):
        page.insert_text((75, baseline), label, fontsize=11.5)
        page.insert_text((330, baseline), value, fontsize=10)
    for y in (80, 100, 116, 132, 150):
        page.draw_line((60, y), (535, y))
    for x in (60, 300, 535):
        page.draw_line((x, 80), (x, 150))
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = "<table>" + "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>"
        for label, value in rows
    ) + "</table>"
    translated_html = original_html.replace("<th>Header</th>", "<th>항목</th>")
    table_rect = fitz.Rect(60, 80, 535, 150)
    original = [{
        "type": "table",
        "bbox": _layout_bbox(table_rect),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    with fitz.open(job_dir / "source.pdf") as source_doc:
        source_header = next(
            span for span in _span_entries(source_doc[0]) if span["text"] == "Header"
        )
        source_size = float(source_header["size"])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
        translated_headers = [
            span for span in _span_entries(exported[0]) if span["text"] == "항목"
        ]

    expected_floor = min(source_size, max(9.0, source_size * 0.80))
    assert result.table_cells_replaced == 1
    assert translated_headers, text
    translated_size = float(translated_headers[0]["size"])
    assert translated_size >= expected_floor, (
        source_size,
        translated_size,
        expected_floor,
    )


def test_spanned_table_translates_ordinary_cells_when_structure_matches(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """rowspan/colspan must not force unchanged ordinary cells to stay English."""
    import fitz

    job_dir = tmp_path / "spanned-table"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for text, origin in (
        ("Category", (78, 122)),
        ("Measurements", (270, 102)),
        ("Left", (255, 137)),
        ("Right", (405, 137)),
        ("Sample", (78, 174)),
        ("10", (270, 174)),
        ("20", (410, 174)),
    ):
        page.insert_text(origin, text, fontsize=10)
    for y in (80, 150, 190):
        page.draw_line((60, y), (535, y))
    page.draw_line((220, 115), (535, 115))
    for x in (60, 220, 535):
        page.draw_line((x, 80), (x, 190))
    page.draw_line((380, 115), (380, 190))
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = (
        '<table><tr><th rowspan="2">Category</th>'
        '<th colspan="2">Measurements</th></tr>'
        "<tr><th>Left</th><th>Right</th></tr>"
        "<tr><td>Sample</td><td>10</td><td>20</td></tr></table>"
    )
    translated_html = (
        '<table><tr><th rowspan="2">Category</th>'
        '<th colspan="2">Measurements</th></tr>'
        "<tr><th>왼쪽</th><th>오른쪽</th></tr>"
        "<tr><td>시료</td><td>10</td><td>20</td></tr></table>"
    )
    table_rect = fitz.Rect(60, 80, 535, 190)
    original = [{
        "type": "table",
        "bbox": _layout_bbox(table_rect),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.table_cells_replaced == 3, result.report()
    assert all(label in text for label in ("왼쪽", "오른쪽", "시료")), text
    assert not any(label in text for label in ("Left", "Right", "Sample")), text
    assert "Category" in text and "Measurements" in text


def test_multiline_title_avoids_one_or_two_character_orphan(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """Prefer a modest shrink over accepting a two-character last title line."""
    import fitz

    job_dir = tmp_path / "balanced-title"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 105), "GENERIC TITLE", fontsize=18)
    source.save(job_dir / "source.pdf")
    source.close()

    title_rect = fitz.Rect(60, 80, 250, 140)
    original = [{
        "type": "title",
        "bbox": _layout_bbox(title_rect),
        "content": "GENERIC TITLE",
        "fs": 18 / PAGE_WIDTH * 100,
    }]
    translated_title = "가나다라마바사아자차카"
    _write_layout_pair(job_dir, original, [translated_title])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        translated_lines = [
            text.replace(" ", "")
            for text, _rect in _line_entries(exported[0])
            if any("가" <= char <= "힣" for char in text)
        ]

    assert result.replaced == 1
    assert "".join(translated_lines) == translated_title
    assert len(translated_lines) == 1 or len(translated_lines[-1]) >= 3, translated_lines


def test_rich_prefix_merges_same_line_pua_run_after_reconstruction():
    """A split Story placeholder must not leave a visual gap before a digit."""
    source = "질의 생성 예: Llama-3-Instruction"
    rebuilt = _reconstruct_rich_runs(
        source,
        (
            (10.0, 20.0, "질의\xa0생성", True),
            (45.0, 20.0, "\xa0예:\xa0Llama‑", False),
            (120.0, 20.0, "\uf63e‑Instruction", False),
        ),
    )

    assert rebuilt == (
        (10.0, 20.0, "질의 생성", True),
        (45.0, 20.0, " 예: Llama-3-Instruction", False),
    )


def test_incomplete_citation_tail_keeps_author_marker_together():
    text = "효과적인 장문맥 지시문에 필요한 다양성을 보장하지 못한다. 기존 프로젝트(Li et al., 2025;"

    protected = _protect_trailing_words(text)

    assert protected == (
        "효과적인 장문맥 지시문에"
        "\n필요한 다양성을 보장하지 못한다. 기존 프로젝트"
        "(Li et al., 2025;"
    )


def test_incomplete_citation_reflow_never_splits_latin_or_year_tokens(
    real_cjk_fontfile: str,
):
    """명시 행갈이는 좁은 Noto/Apple CJK 상자에서도 Li·2025를 쪼개지 않는다."""
    import fitz

    source = "효과적인 장문맥 지시문에 필요한 다양성을 보장하지 못한다. 기존 프로젝트(Li et al., 2025;"
    protected = _protect_trailing_words(source)
    for width in (140, 220, 260):
        doc = fitz.open()
        page = doc.new_page(width=400, height=500)
        spare = page.insert_textbox(
            fitz.Rect(20, 20, 20 + width, 480),
            protected,
            fontsize=10,
            fontname="uocr-test",
            fontfile=real_cjk_fontfile,
            lineheight=1.44,
        )
        assert spare >= 0
        lines = [text.replace("\xa0", " ") for text, _rect in _line_entries(page)]
        compact = "".join(re.sub(r"\s+", "", line) for line in lines)
        assert compact == re.sub(r"\s+", "", source), lines
        for token in ("Li", "et", "al.", "2025"):
            assert any(token in line for line in lines), (width, token, lines)
        doc.close()


def test_compact_noto_bounds_use_only_measured_character_classes():
    assert _noto_visible_ink_bounds("초기 모델 2025") == (0.86, -0.12)
    assert _noto_visible_ink_bounds("(한글), Q") == (0.86, -0.27)
    assert _noto_visible_ink_bounds("한글.") == (0.86, -0.12)
    assert _noto_visible_ink_bounds("한글,") == (0.86, -0.20)
    assert _noto_visible_ink_bounds("(한글)") == (0.86, -0.23)
    assert _noto_visible_ink_bounds("한글@") == (0.86, -0.20)
    assert _noto_visible_ink_bounds("한글|") is None
    assert _noto_visible_ink_bounds("Agyp") == (0.86, -0.27)
    assert _noto_visible_ink_bounds("² University, ⁴ Tsinghua") == (1.0, -0.27)
    assert _noto_visible_ink_bounds("Ắ") is None
    assert _noto_visible_ink_bounds("β") is None
    assert _noto_visible_ink_bounds("한글🙂") is None


def test_latin_descender_uses_conservative_noto_collision_bounds(
    monkeypatch: pytest.MonkeyPatch,
):
    """Agyp의 실제 descender를 한글 bbox로 줄여 인접 선 충돌을 통과시키지 않는다."""
    import fitz

    class FakeNotoFont:
        name = "Noto Serif CJK JP Regular"
        ascender = 1.151
        descender = -0.286

        @staticmethod
        def text_length(_text: str, *, fontsize: float) -> float:
            return fontsize * 2

    monkeypatch.setattr(
        pdf_export_module,
        "quiet_fitz",
        lambda: SimpleNamespace(
            Font=lambda **_kwargs: FakeNotoFont(),
            Rect=fitz.Rect,
        ),
    )
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    rect = fitz.Rect(20, 80, 200, 100)
    obstacle = fitz.Rect(20, 100.2, 200, 110)
    common = {
        "page": page,
        "rect": rect,
        "base_pt": 20,
        "fontname": "uocr-test",
        "fontfile": "/fonts/NotoSerifCJK-Regular.ttc",
        "max_rect": fitz.Rect(20, 80, 200, 120),
        "avoid_rects": [obstacle],
        "scales": (1.0,),
    }

    assert _plan_single_line(text="Agyp", **common) is None
    assert _plan_single_line(text="(한글), Q", **common) is None
    assert _plan_single_line(text="Ắ", **common) is None
    hangul = _plan_single_line(text="한글", **common)
    assert hangul is not None
    assert hangul.ink_rect.y1 <= obstacle.y0
    doc.close()


def test_rich_prefix_keeps_hangul_hyphen_compound_in_nowrap_span():
    markup = _rich_prefix_markup(
        "질의 생성 문서-질의 쌍을 생성한다.",
        ("", "질의 생성"),
    )

    assert '<span class="nowrap">문서-질의</span>' in markup


def test_rich_prefix_round_trip_keeps_compound_once_and_unbroken(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """Story 계획부터 실제 삽입까지 run-in label과 하이픈 합성어를 한 번만 쓴다."""
    import fitz

    job_dir = tmp_path / "rich-prefix-round-trip"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    prefix = "Query Generation"
    suffix = " documents produce query-answer pairs."
    origin = fitz.Point(60, 105)
    page.insert_text(origin, prefix, fontsize=11, fontname="hebo")
    prefix_width = fitz.get_text_length(prefix, fontname="hebo", fontsize=11)
    page.insert_text(
        fitz.Point(origin.x + prefix_width, origin.y),
        suffix,
        fontsize=11,
        fontname="helv",
    )
    source.save(job_dir / "source.pdf")
    source.close()

    original = [{
        "type": "text",
        "bbox": _layout_bbox(fitz.Rect(55, 82, 310, 155)),
        "content": prefix + suffix,
        "fs": 11 / PAGE_WIDTH * 100,
    }]
    translated = "질의 생성 각 문서는 다양한 문서-질의 쌍을 생성한다."
    _write_layout_pair(job_dir, original, [translated])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        lines = [
            text.replace("\xa0", " ").replace("‑", "-")
            for text, _rect in _line_entries(exported[0])
        ]
    output = " ".join(lines)

    assert result.replaced == 1, result.report()
    assert output.count("질의 생성") == 1, lines
    assert output.count("문서-질의") == 1, lines
    assert not any(
        line.endswith("문서-질") and next_line.startswith("의")
        for line, next_line in zip(lines, lines[1:])
    ), lines
    assert "Query Generation" not in output, output


def test_mid_sentence_bold_run_is_not_treated_as_a_structural_prefix():
    """일반 문장 중간 강조 때문에 번역문의 첫 단어가 굵어지면 안 된다."""
    import fitz

    spans = [
        pdf_export_module._SourceSpan(
            fitz.Rect(10, 10, 55, 22), "This is ", 10, 0, (10, 20),
        ),
        pdf_export_module._SourceSpan(
            fitz.Rect(55, 10, 105, 22), "important", 10, 16, (55, 20),
        ),
    ]

    assert pdf_export_module._leading_bold_prefix(spans, "이것은 중요하다") is None


def test_list_marker_can_still_precede_a_structural_bold_prefix():
    """문두 목록 marker 뒤의 run-in label 강조는 유지한다."""
    import fitz

    spans = [
        pdf_export_module._SourceSpan(
            fitz.Rect(10, 10, 16, 22), "•", 10, 0, (10, 20),
        ),
        pdf_export_module._SourceSpan(
            fitz.Rect(18, 10, 42, 22), "Note", 10, 16, (18, 20),
        ),
    ]

    assert pdf_export_module._leading_bold_prefix(spans, "• 주의 내용") == ("• ", "주의")


def test_repeated_scheme_link_normalization_preserves_path_and_scope():
    """선택한 URL의 중복 scheme만 접고 다른 링크와 path/query는 보존한다."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    target_rect = fitz.Rect(50, 80, 300, 100)
    wrapped_rect = fitz.Rect(50, 100, 300, 120)
    other_rect = fitz.Rect(50, 140, 300, 160)
    target_uri = "https://https://huggingface.co/datasets/acme/custom?rev=1"
    other_uri = "https://https://huggingface.co/datasets/other/untouched"
    page.insert_link({"kind": fitz.LINK_URI, "from": target_rect, "uri": target_uri})
    page.insert_link({"kind": fitz.LINK_URI, "from": wrapped_rect, "uri": target_uri})
    page.insert_link({"kind": fitz.LINK_URI, "from": other_rect, "uri": other_uri})
    page = doc.reload_page(page)

    pdf_export_module._normalize_repeated_scheme_links(page, [target_rect])
    page = doc.reload_page(page)
    links = sorted(page.get_links(), key=lambda link: link["from"].y0)
    doc.close()

    assert links[0]["uri"] == "https://huggingface.co/datasets/acme/custom?rev=1"
    assert links[1]["uri"] == "https://huggingface.co/datasets/acme/custom?rev=1"
    assert links[2]["uri"] == other_uri


def test_visible_uri_annotation_border_is_hidden_without_losing_link(
    tmp_path: Path,
):
    """논문 링크의 유색 사각 테두리를 숨겨도 URI와 클릭 영역은 보존한다."""
    import fitz

    job_dir = tmp_path / "borderless-link"
    job_dir.mkdir()
    source_path = job_dir / "source.pdf"
    uri = "https://api.semanticscholar.org/CorpusID:273098476"
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "REFERENCE", fontsize=10)
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(60, 85, 180, 105),
        "uri": uri,
    })
    source.save(source_path)
    source.close()
    with fitz.open(source_path) as source:
        xref = source[0].get_links()[0]["xref"]
        source.xref_set_key(xref, "Border", "[0 0 1]")
        source.xref_set_key(xref, "BS", "<< /W 1 /S /S >>")
        source.xref_set_key(xref, "C", "[0 1 1]")
        appearance = source.get_new_xref()
        source.update_object(
            appearance,
            "<< /Type /XObject /Subtype /Form /BBox [0 0 120 20] >>",
        )
        source.update_stream(appearance, b"0 1 1 RG 1 w 0 0 120 20 re S")
        source.xref_set_key(xref, "AP", f"<< /N {appearance} 0 R >>")
        source.saveIncr()

    original = [{
        "type": "reference",
        "bbox": _layout_bbox(fitz.Rect(60, 85, 180, 105)),
        "content": "REFERENCE",
        "fs": 10 / PAGE_WIDTH * 100,
    }]
    _write_layout_pair(job_dir, original, ["REFERENCE"])

    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        links = exported[0].get_links()
        border = exported.xref_get_key(links[0]["xref"], "Border")
        border_style = exported.xref_get_key(links[0]["xref"], "BS")
        appearance = exported.xref_get_key(links[0]["xref"], "AP")

    assert links[0]["uri"] == uri
    assert border == ("array", "[0 0 0]")
    assert border_style == ("dict", "<</W 0>>")
    assert appearance == ("null", "null")


def test_changed_table_cell_redaction_preserves_adjacent_unchanged_value(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """A left-cell translation must not erase the start of the right cell."""
    import fitz

    job_dir = tmp_path / "table-adjacent-value"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    # 두 행 중심의 중간 경계가 separator 바로 아래에 놓이는 실제 Table 6
    # geometry를 재현한다. 예전 탐지는 이처럼 셀 y0보다 0.1pt 위인 선을 놓쳤다.
    page.insert_text((278, 93), "training setting", fontsize=9)
    page.insert_text((205, 113), "Initial Model", fontsize=9)
    model_name = "Llama-3-8B-NExtLong-512K-Base"
    page.insert_text((279, 113), model_name, fontsize=9)
    for y in (80, 100, 122):
        page.draw_line((200, y), (410, y))
    for x in (200, 285, 410):
        page.draw_line((x, 80), (x, 122))
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = (
        '<table><tr><td colspan="2">training setting</td></tr>'
        f"<tr><td>Initial Model</td><td>{model_name}</td></tr></table>"
    )
    translated_html = (
        original_html
        .replace("training setting", "학습 설정")
        .replace("Initial Model", "초기 모델")
    )
    original = [{
        "type": "table",
        # OCR bbox가 실제 top rule보다 조금 위에서 시작하는 LaTeX 표를 재현.
        "bbox": _layout_bbox(fitz.Rect(200, 77.5, 410, 122)),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    with fitz.open(job_dir / "source.pdf") as source_doc:
        source_page = source_doc[0]
        table_rect = _block_rect(
            fitz,
            source_page,
            _load_pages(job_dir / "layout.json")[0]["blocks"][0]["bbox"],
        )
        parsed = _table_cells(original_html)
        assert table_rect is not None and parsed is not None
        cells, rows, cols = parsed
        measured_rects, grid_trusted = _table_cell_rects(
            source_page, table_rect, cells, rows, cols,
        )
        assert grid_trusted
        top_header_rect = next(
            rect
            for cell, rect in zip(cells, measured_rects)
            if cell.row == 0 and cell.col == 0
        )
        first_row_rect = next(
            rect
            for cell, rect in zip(cells, measured_rects)
            if cell.row == 1 and cell.col == 0
        )
        # 표 최상단 rule은 기존 0.25pt 여백을 유지한다.
        assert 80.74 <= top_header_rect.y0 <= 80.76, top_header_rect
        # y=100, width=1인 header rule의 하단에서 0.5pt를 띄운다.
        assert first_row_rect.y0 >= 101.0 - 0.01, first_row_rect

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        text = page.get_text().replace("\xa0", " ")
        translated_header = next(
            span for span in _span_entries(page) if span["text"] == "학습 설정"
        )

    assert result.table_cells_replaced == 2, result.report()
    assert float(translated_header["bbox"][1]) > 80.25, translated_header
    assert "초기 모델" in text, text
    assert model_name in text, text


def test_shared_pdf_span_preserves_every_ambiguous_ocr_block(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """한 source span을 공유한 블록은 고유 span이 따로 있어도 함께 보존한다."""
    import fitz

    job_dir = tmp_path / "shared-source-span"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "ALPHA BETA", fontsize=12)
    page.insert_text((60, 122), "UNIQUEA", fontsize=12)
    page.insert_text((160, 144), "UNIQUEB", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(50, 80, 150, 132)),
            "content": "ALPHA UNIQUEA",
            "fs": 12 / PAGE_WIDTH * 100,
        },
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(105, 80, 260, 154)),
            "content": "BETA UNIQUEB",
            "fs": 12 / PAGE_WIDTH * 100,
        },
    ]
    _write_layout_pair(job_dir, original, ["첫째 번역", "둘째 번역"])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.replaced == 0
    assert result.kept == 2
    assert all(value in text for value in ("ALPHA BETA", "UNIQUEA", "UNIQUEB")), text
    assert "첫째 번역" not in text and "둘째 번역" not in text, text


def test_blank_ocr_rect_can_translate_without_redacting_remote_source_text(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """페이지에 다른 text span이 있어도 실제 빈 OCR bbox는 안전하게 채운다."""
    import fitz

    job_dir = tmp_path / "blank-ocr-rect"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "REMOTE SOURCE", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [{
        "type": "text",
        "bbox": _layout_bbox(fitz.Rect(300, 300, 520, 345)),
        "content": "OCR ONLY SOURCE",
        "fs": 12 / PAGE_WIDTH * 100,
    }]
    _write_layout_pair(job_dir, original, ["빈 영역 번역"])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.replaced == 1, result.report()
    assert "빈 영역 번역" in text, text
    assert "REMOTE SOURCE" in text, text


def test_explicit_font_does_not_probe_an_unconfigured_compact_font(
    tmp_path: Path,
    real_cjk_fontfile: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """PDF_EXPORT_FONT가 유효하면 일반 문단과 표가 시스템 폰트를 우회하지 않는다."""
    import fitz

    job_dir = tmp_path / "explicit-font-contract"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "EXPLICIT SOURCE", fontsize=12)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [{
        "type": "text",
        "bbox": _layout_bbox(fitz.Rect(60, 80, 320, 120)),
        "content": "EXPLICIT SOURCE",
        "fs": 12 / PAGE_WIDTH * 100,
    }]
    _write_layout_pair(job_dir, original, ["명시 폰트 번역"])

    calls: list[str] = []
    original_resolve = pdf_export_module._resolve_font

    def tracked_resolve(explicit: str = "", *args, **kwargs):
        calls.append(explicit)
        return original_resolve(explicit, *args, **kwargs)

    monkeypatch.setattr(pdf_export_module, "_resolve_font", tracked_resolve)
    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)

    assert result.replaced == 1
    assert calls and all(call == real_cjk_fontfile for call in calls), calls


def test_inserted_font_resource_name_does_not_reuse_source_resource(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """원본의 /uocr-serif resource가 새 fontfile 삽입을 가로채면 안 된다."""
    import fitz

    job_dir = tmp_path / "font-resource-collision"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_font(fontname="uocr-serif", fontfile=real_cjk_fontfile)
    page.insert_text((60, 100), "SOURCE TARGET", fontsize=12, fontname="uocr-serif")
    page.insert_text((60, 180), "PRESERVED", fontsize=12, fontname="uocr-serif")
    source.save(job_dir / "source.pdf")
    source.close()

    original = [{
        "type": "text",
        "bbox": _layout_bbox(fitz.Rect(60, 80, 320, 120)),
        "content": "SOURCE TARGET",
        "fs": 12 / PAGE_WIDTH * 100,
    }]
    _write_layout_pair(job_dir, original, ["리소스 충돌 방지"])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        text = page.get_text().replace("\xa0", " ")
        resource_names = {font[4] for font in page.get_fonts() if len(font) > 4}

    assert result.replaced == 1
    assert "리소스 충돌 방지" in text and "PRESERVED" in text, text
    assert "uocr-serif" in resource_names
    assert any(name.startswith("uocr-serif-") for name in resource_names), resource_names


def test_failed_table_reports_every_changed_cell_as_preserved(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """원자적 표 계획이 실패하면 실패 이전 셀만이 아니라 전체 변경 수를 센다."""
    import fitz

    job_dir = tmp_path / "table-kept-count"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((65, 105), "Left", fontsize=10)
    page.insert_text((165, 105), "Right", fontsize=10)
    for y in (80, 120):
        page.draw_line((60, y), (260, y))
    for x in (60, 160, 260):
        page.draw_line((x, 80), (x, 120))
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = "<table><tr><td>Left</td><td>Right</td></tr></table>"
    translated_html = (
        "<table><tr><td>" + "매우긴번역" * 30 + "</td><td>오른쪽</td></tr></table>"
    )
    original = [{
        "type": "table",
        "bbox": _layout_bbox(fitz.Rect(60, 80, 260, 120)),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)

    assert result.table_cells_replaced == 0, result.report()
    assert result.kept == 2, result.report()


@pytest.mark.parametrize(
    ("found_cells", "expect_trusted"),
    [
        (("A1", "A2"), True),    # 4셀 중 2셀 = 정확히 50% → 신뢰
        (("A1",), False),        # 4셀 중 1셀 = 25% → 불신
        (("A1", "A2", "B1"), True),
    ],
)
def test_표_격자_신뢰_게이트의_경계는_원문_셀_검색_50퍼센트다(
    tmp_path: Path, found_cells: tuple[str, ...], expect_trusted: bool,
):
    """임계값을 아무 테스트도 고정하지 않으면 60%·40%로 바뀌어도 조용히 통과한다.

    50%는 "번역문이 엉뚱한 셀에 찍히고 인접 셀 원문이 리댁션되는" 사고와
    "정상 표를 통째로 보존해 미번역으로 보이는" 사고 사이의 경계다.
    """
    import fitz

    positions = {
        "A1": (65, 95), "A2": (300, 95), "B1": (65, 115), "B2": (300, 115),
    }
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for name in found_cells:
        page.insert_text(positions[name], name, fontsize=9)
    doc.save(tmp_path / "grid.pdf")
    doc.close()

    parsed = _table_cells(
        "<table><tr><td>A1</td><td>A2</td></tr>"
        "<tr><td>B1</td><td>B2</td></tr></table>"
    )
    assert parsed is not None
    cells, rows, cols = parsed
    with fitz.open(tmp_path / "grid.pdf") as source:
        _rects, grid_trusted = _table_cell_rects(
            source[0], fitz.Rect(60, 85, 530, 125), cells, rows, cols,
        )

    assert grid_trusted is expect_trusted


def test_원문_셀_검색이_전부_빗나간_표는_균등격자로_강행하지_않는다(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """OCR 오독으로 격자 추정이 실패하면 표를 통째로 보존한다.

    폭이 크게 다른 열(350pt vs 70pt)에서 균등 분할로 강행하면 번역문이 엉뚱한
    셀에 겹쳐 찍히고 인접 셀 원문이 경고 없이 리댁션된다.
    """
    import fitz

    job_dir = tmp_path / "table-untrusted-grid"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    # 1열 350pt, 2·3열 70pt인 불균등 표. 텍스트 레이어가 실제로 존재한다.
    page.insert_text((65, 95), "Method", fontsize=9)
    page.insert_text((400, 95), "Acc", fontsize=9)
    page.insert_text((470, 95), "F1", fontsize=9)
    page.insert_text((65, 115), "Ours", fontsize=9)
    page.insert_text((400, 115), "12.3", fontsize=9)
    page.insert_text((470, 115), "45.6", fontsize=9)
    for y in (85, 100, 122):
        page.draw_line((60, y), (530, y))
    source.save(job_dir / "source.pdf")
    source.close()

    # OCR이 셀 문자열을 조금씩 잘못 읽어 search_for가 전부 빗나가는 상황.
    original_html = (
        "<table>"
        "<tr><td>Methoo</td><td>Acc.</td><td>Fl</td></tr>"
        "<tr><td>Ourss</td><td>l2.3</td><td>45,6</td></tr>"
        "</table>"
    )
    translated_html = (
        "<table>"
        "<tr><td>방법</td><td>정확도</td><td>F1</td></tr>"
        "<tr><td>제안</td><td>l2.3</td><td>45,6</td></tr>"
        "</table>"
    )
    original = [{
        "type": "table",
        "bbox": _layout_bbox(fitz.Rect(60, 82, 530, 122)),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.table_cells_replaced == 0, result.report()
    assert result.specialist_kept.get("table") == 1, result.report()
    assert any("격자 추정 실패" in warning for warning in result.warnings), result.warnings
    for value in ("Method", "Acc", "F1", "12.3", "45.6"):
        assert value in text, text
    assert "정확도" not in text, text


def test_표_셀마다_페이지_텍스트를_재추출하지_않는다(
    tmp_path: Path,
    real_cjk_fontfile: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """셀 스타일 추정은 페이지에서 이미 읽어둔 span 목록을 재사용한다."""
    import fitz

    job_dir = tmp_path / "table-span-reuse"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((65, 95), "Method", fontsize=9)
    page.insert_text((400, 95), "Acc", fontsize=9)
    page.insert_text((470, 95), "F1", fontsize=9)
    page.insert_text((65, 115), "Ours", fontsize=9)
    page.insert_text((400, 115), "12.3", fontsize=9)
    page.insert_text((470, 115), "45.6", fontsize=9)
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = (
        "<table>"
        "<tr><td>Method</td><td>Acc</td><td>F1</td></tr>"
        "<tr><td>Ours</td><td>12.3</td><td>45.6</td></tr>"
        "</table>"
    )
    translated_html = (
        original_html.replace("Method", "방법").replace("Acc", "정확도")
        .replace("Ours", "제안")
    )
    original = [{
        "type": "table",
        "bbox": _layout_bbox(fitz.Rect(60, 82, 530, 122)),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    calls: list[int] = []
    extract_records = pdf_export_module._source_span_records

    def counted(fitz_module, page):
        calls.append(page.number)
        return extract_records(fitz_module, page)

    monkeypatch.setattr(pdf_export_module, "_source_span_records", counted)
    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)

    assert result.table_cells_replaced == 3, result.report()
    # 페이지당 1회 — 변경 셀 수만큼 곱해지면 안 된다.
    assert calls == [0], calls


def test_텍스트레이어_없는_스캔표는_균등격자로도_계속_번역된다(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """격자 신뢰도 게이트가 이 프로젝트의 주 대상인 스캔 표를 막으면 안 된다."""
    import fitz

    job_dir = tmp_path / "table-scanned-grid"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    # 스캔 표: 지울 원문 글리프가 없으므로 균등 격자가 무해하다.
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16))
    pixmap.clear_with(230)
    page.insert_image(
        fitz.Rect(60, 82, 530, 122), stream=pixmap.tobytes("png"),
        keep_proportion=False,
    )
    source.save(job_dir / "source.pdf")
    source.close()

    original_html = (
        "<table>"
        "<tr><td>Method</td><td>Acc</td><td>F1</td></tr>"
        "<tr><td>Ours</td><td>12.3</td><td>45.6</td></tr>"
        "</table>"
    )
    translated_html = original_html.replace("Method", "방법").replace("Ours", "제안")
    original = [{
        "type": "table",
        "bbox": _layout_bbox(fitz.Rect(60, 82, 530, 122)),
        "content": original_html,
    }]
    _write_layout_pair(job_dir, original, [translated_html])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.table_cells_replaced == 2, result.report()
    assert not any("격자 추정 실패" in warning for warning in result.warnings), result.warnings
    assert "방법" in text and "제안" in text, text


# ── CropBox/회전 페이지 가드 ─────────────────────────────────────────────
_CROP_BOX = (20.0, 30.0, 592.0, 700.0)
_CROP_WIDTH = _CROP_BOX[2] - _CROP_BOX[0]
_CROP_HEIGHT = _CROP_BOX[3] - _CROP_BOX[1]
_LONG_KO_TEXT = "이 문단은 페이지 하단에 놓인 긴 한국어 번역문입니다. " * 8


def _cropped_bottom_block_job(
    job_dir: Path,
    rotation: int,
    block_rect,
    translated: str,
    *,
    with_source_text: bool = True,
):
    """CropBox<MediaBox 페이지의 하단 텍스트 블록 한 개짜리 잡을 만든다.

    `block_rect`은 표시(회전 반영) 좌표다 — 레이아웃 계약의 0–999 bbox와 같은
    공간이므로 회전 여부와 무관하게 그대로 정규화할 수 있다.
    """
    import fitz

    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    page.set_cropbox(fitz.Rect(*_CROP_BOX))
    if rotation:
        page.set_rotation(rotation)
    if with_source_text:
        # 삽입 좌표는 비회전 내부 공간이다. `block_rect`(표시 좌표) 안에 실제로
        # 글리프가 놓이도록 derotation을 거치고, 표시 공간에서 똑바로 읽히게
        # 페이지 회전각만큼 글자도 돌린다.
        origin = fitz.Point(block_rect.x0 + 2, block_rect.y0 + 25)
        page.insert_text(
            origin * page.derotation_matrix,
            "BOTTOM SOURCE LINE",
            fontsize=10,
            rotate=rotation,
        )
    source.save(job_dir / "source.pdf")
    source.close()

    display_width, display_height = (
        (_CROP_HEIGHT, _CROP_WIDTH) if rotation in (90, 270)
        else (_CROP_WIDTH, _CROP_HEIGHT)
    )
    original = [{
        "type": "text",
        "bbox": _layout_bbox(
            block_rect, page_width=display_width, page_height=display_height,
        ),
        "content": "BOTTOM SOURCE LINE",
        "fs": 10 / display_width * 100,
    }]
    _write_layout_pair(
        job_dir,
        original,
        [translated],
        page_width=display_width,
        page_height=display_height,
    )


def test_크롭박스_페이지_하단_번역문이_표시영역_밖으로_나가지_않는다(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """CropBox<MediaBox 문서에서 확장 가드가 MediaBox를 쓰면 번역문이 사라진다."""
    import fitz

    job_dir = tmp_path / "cropbox-bottom-block"
    # 표시 좌표(page.rect = 572x670) 기준 하단 블록.
    _cropped_bottom_block_job(
        job_dir, 0, fitz.Rect(40, 615, 530, 655), _LONG_KO_TEXT,
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        entries = _line_entries(page)
        extracted = page.get_text().replace("\xa0", " ")
        page_rect = +page.rect

    if result.replaced == 0:
        # 공간이 없다고 판단해 원문을 보존하는 것도 안전한 결과다. 다만 원문을
        # 지워 놓고 번역문만 잃어버리는 조합은 허용되지 않는다.
        assert "BOTTOM SOURCE LINE" in extracted, extracted
        return
    for text, rect in entries:
        assert page_rect.contains(rect), (text, rect, page_rect)
    visible = re.sub(r"\s", "", extracted)
    expected = re.sub(r"\s", "", _LONG_KO_TEXT)
    assert len(visible) >= len(expected) * 0.95, (len(visible), len(expected))


def test_회전된_크롭박스_페이지도_같은_표시영역_가드를_쓴다(
    tmp_path: Path,
    real_cjk_fontfile: str,
):
    """`page.rect.y1` 직접 치환 회귀 방지 — 90도 페이지는 derotation이 필요하다.

    90도 페이지의 `page.rect`는 가로·세로가 뒤바뀌어 y1이 572다. 그 값을 하단
    가드로 쓰면 비회전 내부 y가 588~664인 정상 블록이 '공간 부족'으로 보존된다.
    """
    import fitz

    job_dir = tmp_path / "cropbox-rotated-block"
    # 표시 좌표 x가 작을수록 비회전 내부 y가 크다(내부 y = 670 - 표시 x).
    _cropped_bottom_block_job(
        job_dir, 90, fitz.Rect(6, 85, 82, 427), "하단 회전 번역",
        with_source_text=False,
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        entries = _line_entries(page)
        extracted = page.get_text().replace("\xa0", " ")
        # 회전 페이지의 span bbox는 비회전 내부 좌표로 보고된다 — 표시 영역도
        # 같은 좌표계(derotation)로 되돌려 비교한다.
        page_rect = page.rect * page.derotation_matrix
        page_rect.normalize()

    assert result.replaced == 1, result.report()
    assert "하단 회전 번역" in extracted, extracted
    for text, rect in entries:
        assert page_rect.contains(rect), (text, rect, page_rect)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_회전_페이지에서도_원문_글리프가_리댁션된다(
    tmp_path: Path,
    real_cjk_fontfile: str,
    rotation: int,
):
    """회전 페이지에서 번역문만 삽입되고 원문이 남으면 두 글이 겹쳐 찍힌다."""
    import fitz

    # 표시(회전 반영) 좌표 기준 하단 블록. 90/270도 페이지의 표시 크기는
    # 가로·세로가 바뀐 670x572다.
    block_rect = (
        fitz.Rect(40, 500, 530, 540) if rotation in (90, 270)
        else fitz.Rect(40, 600, 530, 640)
    )
    job_dir = tmp_path / f"rotated-redaction-{rotation}"
    _cropped_bottom_block_job(job_dir, rotation, block_rect, "하단 회전 번역")

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        page = exported[0]
        entries = _line_entries(page)
        extracted = page.get_text().replace("\xa0", " ")
        page_rect = page.rect * page.derotation_matrix
        page_rect.normalize()

    assert result.replaced == 1, result.report()
    assert "하단 회전 번역" in extracted, extracted
    assert "BOTTOM SOURCE LINE" not in extracted, extracted
    for text, rect in entries:
        assert page_rect.contains(rect), (text, rect, page_rect)


def _span(text: str, x0: float, baseline: float, size: float = 9.0):
    """`_reflow_flattened_text` 단위 테스트용 원문 span."""
    import fitz

    return _SourceSpan(
        fitz.Rect(x0, baseline - size, x0 + len(text) * size * 0.5, baseline + 2),
        text,
        size,
        0,
        (x0, baseline),
    )


def test_가로_평탄화_헤더는_원문_시각_줄_수로_리플로우된다():
    """한 줄에 나란히 놓인 셀이 OCR 줄바꿈으로 평탄화된 블록만 되접는다."""
    spans = [
        _span("Dataset", 60, 100),
        _span("Long Evaluation", 160, 100),
        _span("Short Evaluation", 300, 100),
    ]

    reflowed = _reflow_flattened_text(
        "Dataset\nLong Evaluation\nShort Evaluation",
        "데이터셋\n긴 평가\n짧은 평가",
        spans,
        9.3,
    )

    assert reflowed == "데이터셋 긴 평가 짧은 평가"


def test_가로_평탄화_리플로우가_표의_행_구조를_보존한다():
    """여러 행이 평탄화된 블록은 한 줄이 아니라 원래 행 수로 되접는다."""
    spans = [
        _span("Model", 60, 100),
        _span("Score", 200, 100),
        _span("Alpha", 60, 112),
        _span("1.0", 200, 112),
    ]

    reflowed = _reflow_flattened_text(
        "Model\nScore\nAlpha\n1.0",
        "모델\n점수\n알파\n1.0",
        spans,
        9.3,
    )

    assert reflowed == "모델 점수\n알파 1.0"


def test_정상적으로_줄바꿈된_산문은_리플로우하지_않는다():
    """진짜 여러 줄 문단을 한 줄로 뭉치면 원문보다 나빠진다 — 건드리지 않는다."""
    spans = [
        _span("This approach differs fundamentally", 60, 100),
        _span("from traditional prompt engineering", 60, 112),
        _span("as we are not instructing the model", 60, 124),
    ]

    reflowed = _reflow_flattened_text(
        "This approach differs fundamentally\n"
        "from traditional prompt engineering\n"
        "as we are not instructing the model",
        "이 접근은 근본적으로 다르다\n전통적인 프롬프트 공학과\n"
        "우리는 모델에 지시하지 않는다",
        spans,
        10.3,
    )

    assert reflowed is None


def test_원문_span이_없으면_리플로우하지_않는다():
    """소유한 원문 span이 없으면 시각 줄 수를 잴 수 없다 — 보수적으로 보존."""
    assert _reflow_flattened_text("A\nB", "가\n나", [], 9.3) is None


def test_리스팅_세그먼트가_들여쓰기와_열_경계를_보존한다():
    """의사코드는 줄 구조와 들여쓰기가 곧 의미다 — 원문 x 좌표를 그대로 쓴다."""
    spans = [
        _span("1:", 60, 100),
        _span("procedure Main", 80, 100),
        _span("2:", 60, 112),
        _span("x ←1", 100, 112),
        _span("▷comment", 300, 112),
    ]

    segments = _listing_segments(
        "1:\nprocedure Main\n2:\nx ←1\n▷comment",
        "1:\n프로시저 메인\n2:\nx ←1\n▷주석",
        spans,
        9.3,
        430.0,
    )

    # 두 번째 시각 줄의 코드가 더 깊게(60 → 100) 들여쓰여 있음이 보존된다.
    assert [round(segment.x0) for segment in segments] == [60, 80, 60, 100, 300]
    # 각 세그먼트의 오른쪽 한계는 같은 줄 다음 세그먼트의 시작 x(마지막은 블록 우변).
    assert [round(segment.x1) for segment in segments] == [80, 430, 100, 300, 430]
    assert [round(segment.baseline) for segment in segments] == [100, 100, 112, 112, 112]
    assert [segment.text for segment in segments] == [
        "1:", "프로시저 메인", "2:", "x ←1", "▷주석",
    ]


def test_원문_span_한가운데를_가르는_줄은_세그먼트로_만들지_않는다():
    """span을 통째로 리댁션하면 옆 줄 원문까지 지워진다 — 그 시각 줄만 버린다."""
    spans = [
        _span("AlphaBeta", 60, 100),
        _span("Gamma", 200, 112),
        _span("Delta", 300, 112),
    ]

    segments = _listing_segments(
        "Alpha\nBeta\nGamma\nDelta",
        "알파\n베타\n감마\n델타",
        spans,
        9.3,
        430.0,
    )

    # 첫 시각 줄은 span 하나를 두 OCR 줄이 나눠 가지므로 통째로 보존한다.
    assert [segment.text for segment in segments] == ["감마", "델타"]


def _flattened_header_job(
    job_dir: Path,
    translated: list[str],
    *,
    header_cells: tuple[str, ...] = ("Dataset", "Long Evaluation", "Short Evaluation"),
) -> None:
    """헤더 한 줄이 OCR에서 여러 줄로 평탄화된 얕은 블록 + 위아래 보존 블록.

    위아래를 보존 블록으로 막아야 flow planner가 위로 밀거나 아래로 늘려서
    세 줄을 그대로 조판하는 우회로가 없어진다 — 실제 표 헤더와 같은 상황.
    """
    import fitz

    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 84), "PRESERVED CAPTION LINE", fontsize=9)
    for text, x0 in zip(header_cells, (60, 160, 300)):
        page.insert_text((x0, 100), text, fontsize=9)
    page.insert_text((60, 118), "PRESERVED FOOTNOTE LINE", fontsize=9)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [
        {
            "type": "ref_text",
            "bbox": _layout_bbox(fitz.Rect(55, 74, 430, 88)),
            "content": "PRESERVED CAPTION LINE",
            "fs": 9 / PAGE_WIDTH * 100,
        },
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(55, 90, 430, 104)),
            "content": "\n".join(header_cells),
            "fs": 9 / PAGE_WIDTH * 100,
        },
        {
            "type": "ref_text",
            "bbox": _layout_bbox(fitz.Rect(55, 108, 430, 122)),
            "content": "PRESERVED FOOTNOTE LINE",
            "fs": 9 / PAGE_WIDTH * 100,
        },
    ]
    _write_layout_pair(job_dir, original, translated)


def test_평탄화된_표_헤더가_얕은_bbox에서도_번역된다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """사용자 신고의 직접 원인: 한 줄 높이 bbox + 세 줄 OCR → 통째로 미번역."""
    import fitz

    job_dir = tmp_path / "flattened-header"
    job_dir.mkdir()
    _flattened_header_job(
        job_dir,
        [
            "PRESERVED CAPTION LINE",
            "데이터셋\n긴 평가\n짧은 평가",
            "PRESERVED FOOTNOTE LINE",
        ],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    # 각 셀이 원문 열 좌표에 그대로 들어가므로 셀 수만큼 교체된다(이전에는 한
    # 상자에 흘려 넣어 1건이었고 열 정렬이 사라졌다).
    assert result.replaced == 3, result.report()
    assert result.listing_lines_replaced == 3, result.report()
    for cell in ("데이터셋", "긴 평가", "짧은 평가"):
        assert cell in text, text
    assert "Long Evaluation" not in text, text
    # 아래 보존 블록의 원문은 그대로 남아야 한다.
    assert "PRESERVED FOOTNOTE LINE" in text, text


def test_평탄화된_표_헤더가_원문_열_x좌표를_유지한다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """리플로우는 번역은 되지만 열 정렬을 잃고 왼쪽 한 줄로 흘렀다 — 열을 지킨다."""
    import fitz

    job_dir = tmp_path / "flattened-columns"
    job_dir.mkdir()
    _flattened_header_job(
        job_dir,
        [
            "PRESERVED CAPTION LINE",
            "데이터셋\n긴 평가\n짧은 평가",
            "PRESERVED FOOTNOTE LINE",
        ],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    starts: dict[str, float] = {}
    with fitz.open(result.path) as exported:
        for block in exported[0].get_text("dict").get("blocks", ()):
            for line in block.get("lines", ()):
                for span in line.get("spans", ()):
                    body = span.get("text", "").strip()
                    if body:
                        starts.setdefault(body, span["bbox"][0])

    assert result.replaced == 3, result.report()
    # 원문 셀은 x0 = 60/160/300에 있었다. 번역 셀이 같은 열에서 시작해야 한다.
    for cell, expected in (("데이터셋", 60.0), ("긴 평가", 160.0), ("짧은 평가", 300.0)):
        assert cell in starts, starts
        assert abs(starts[cell] - expected) <= 1.5, (cell, starts[cell], expected)


def test_한_열이_안_들어가도_나머지_열은_회수된다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """전부-아니면-전무를 없앤다 — 폭이 모자란 줄만 원문으로 남기고 사유를 남긴다."""
    import fitz

    job_dir = tmp_path / "flattened-partial"
    job_dir.mkdir()
    _flattened_header_job(
        job_dir,
        [
            "PRESERVED CAPTION LINE",
            "매우긴번역문장" * 60 + "\n두번째\n세번째",
            "PRESERVED FOOTNOTE LINE",
        ],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.replaced == 2, result.report()
    assert result.kept_reasons.get("listing_line_no_fit") == 1, result.report()
    assert any("줄 폭 부족" in warning for warning in result.warnings), result.warnings
    # 회수한 두 열은 번역되고, 못 들어간 첫 열만 원문으로 남는다.
    assert "두번째" in text and "세번째" in text, text
    assert "Dataset" in text, text
    assert "Long Evaluation" not in text, text


def test_의사코드_리스팅이_줄과_들여쓰기를_지킨_채_번역된다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """사용자 문서에 영어로 남던 알고리즘 블록.

    원문 4줄 ↔ OCR 10줄이라 상자 조판은 어떤 크기로도 실패하고, 한 줄로 뭉치면
    리스팅 구조가 무너진다. 줄·열 좌표를 그대로 두고 줄별로 조판해야 한다.
    """
    import fitz

    job_dir = tmp_path / "pseudocode"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    listing = (
        (100, (("1:", 60), ("procedure Build", 80))),
        (112, (("2:", 60), ("while i < n do", 95))),
        (124, (("3:", 60), ("x = f(i)", 115), ("comment here", 300))),
        (136, (("4:", 60), ("end while", 95))),
    )
    for baseline, cells in listing:
        for text, x0 in cells:
            page.insert_text((x0, baseline), text, fontsize=9)
    source.save(job_dir / "source.pdf")
    source.close()

    ocr_lines = [text for _, cells in listing for text, _ in cells]
    translated_lines = [
        "1:", "프로시저 빌드",
        "2:", "동안 i < n 반복",
        "3:", "x = f(i)", "여기 주석",
        "4:", "반복 끝",
    ]
    original = [{
        "type": "text",
        "bbox": _layout_bbox(fitz.Rect(55, 90, 430, 140)),
        "content": "\n".join(ocr_lines),
        "fs": 9 / PAGE_WIDTH * 100,
    }]
    _write_layout_pair(job_dir, original, ["\n".join(translated_lines)])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)

    placed: dict[str, tuple[float, float]] = {}
    with fitz.open(result.path) as exported:
        for block in exported[0].get_text("dict").get("blocks", ()):
            for line in block.get("lines", ()):
                for span in line.get("spans", ()):
                    body = span.get("text", "").strip()
                    if body:
                        placed.setdefault(body, (span["bbox"][0], span["origin"][1]))

    assert result.listing_lines_replaced == 4, result.report()
    assert result.kept_reasons.get("flattened_no_fit") is None, result.report()
    # 각 번역 줄이 원문 줄의 baseline과 들여쓰기 x에 그대로 놓인다.
    for text, expected_x, expected_y in (
        ("프로시저 빌드", 80.0, 100.0),
        ("동안 i < n 반복", 95.0, 112.0),
        ("여기 주석", 300.0, 124.0),
        ("반복 끝", 95.0, 136.0),
    ):
        assert text in placed, placed
        x0, baseline = placed[text]
        assert abs(x0 - expected_x) <= 1.5, (text, x0, expected_x)
        assert abs(baseline - expected_y) <= 1.5, (text, baseline, expected_y)
    # 줄 번호와 바뀌지 않은 코드 줄은 원문 그대로 남는다.
    assert "1:" in placed and "x = f(i)" in placed, placed


def test_어느_줄도_못_들어가는_평탄화_블록은_사유가_구분된다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """'공간 부족'과 '가로 평탄화'를 한 문구로 뭉뚱그리면 다음 신고를 진단할 수 없다."""
    job_dir = tmp_path / "flattened-no-fit"
    job_dir.mkdir()
    _flattened_header_job(
        job_dir,
        [
            "PRESERVED CAPTION LINE",
            "\n".join(["매우긴번역문장" * 60] * 3),
            "PRESERVED FOOTNOTE LINE",
        ],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)

    assert result.replaced == 0, result.report()
    assert result.kept_reasons.get("flattened_no_fit") == 1, result.report()
    assert any("가로 평탄화" in warning for warning in result.warnings), result.warnings


def test_보존_사유가_모든_kept_블록을_설명한다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """kept는 58인데 경고는 26이던 상태를 막는다 — 사유별 집계의 합이 kept와 같다."""
    import fitz

    job_dir = tmp_path / "keep-reasons"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "UNCHANGED LINE", fontsize=10)
    page.insert_text((60, 130), "REFERENCE ENTRY", fontsize=10)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(55, 88, 430, 104)),
            "content": "UNCHANGED LINE",
            "fs": 10 / PAGE_WIDTH * 100,
        },
        {
            "type": "ref_text",
            "bbox": _layout_bbox(fitz.Rect(55, 118, 430, 134)),
            "content": "REFERENCE ENTRY",
            "fs": 10 / PAGE_WIDTH * 100,
        },
    ]
    _write_layout_pair(job_dir, original, ["UNCHANGED LINE", "참고문헌 항목"])

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    report = result.report()

    assert result.kept == 2, report
    assert sum(result.kept_reasons.values()) == result.kept, report
    assert result.kept_reasons["unchanged"] == 1, report
    assert result.kept_reasons["preserve_type:ref_text"] == 1, report
    assert report["kept_reasons"] == result.kept_reasons


def test_한_블록이_안_들어가도_같은_단의_이웃은_번역된다(
    tmp_path: Path, real_cjk_fontfile: str,
):
    """flow 그룹이 전부 실패하면 개별 배치로 들어가는 만큼 회수한다."""
    import fitz

    job_dir = tmp_path / "partial-flow"
    job_dir.mkdir()
    source = fitz.open()
    page = source.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((60, 100), "FIRST HEADING LINE", fontsize=10)
    page.insert_text((60, 118), "SECOND DENSE LINE", fontsize=10)
    page.insert_text((60, 140), "PRESERVED TAIL LINE", fontsize=10)
    source.save(job_dir / "source.pdf")
    source.close()

    original = [
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(55, 88, 430, 104)),
            "content": "FIRST HEADING LINE",
            "fs": 10 / PAGE_WIDTH * 100,
        },
        {
            "type": "text",
            "bbox": _layout_bbox(fitz.Rect(55, 106, 430, 122)),
            "content": "SECOND DENSE LINE",
            "fs": 10 / PAGE_WIDTH * 100,
        },
        {
            "type": "ref_text",
            "bbox": _layout_bbox(fitz.Rect(55, 130, 430, 146)),
            "content": "PRESERVED TAIL LINE",
            "fs": 10 / PAGE_WIDTH * 100,
        },
    ]
    _write_layout_pair(
        job_dir,
        original,
        ["첫째 제목", "들어갈수없는아주긴번역" * 40, "PRESERVED TAIL LINE"],
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")

    assert result.replaced == 1, result.report()
    assert result.kept_reasons.get("no_fit") == 1, result.report()
    assert "첫째 제목" in text, text
    assert "SECOND DENSE LINE" in text, text
    assert "FIRST HEADING LINE" not in text, text
