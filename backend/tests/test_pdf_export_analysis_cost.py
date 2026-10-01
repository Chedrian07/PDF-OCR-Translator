"""내보내기의 MuPDF 분석 비용이 셀·패스 수만큼 곱해지지 않는다.

`page.search_for(clip=...)`는 textpage를 넘기지 않으면 호출마다 TextPage를 새로
만들고, 그때마다 clip과 무관하게 페이지 콘텐츠 스트림 전체를 다시 해석한다.
`page.get_drawings()`도 호출마다 페이지의 모든 path를 다시 파싱한다. 벡터가
많은 차트 페이지에서 이 곱이 표 하나에 수 초~수십 초가 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from app.pipeline.pdf_export import _table_cell_rects, _table_cells, build_translated_pdf
from app.pipeline.pdf_export.tables import (
    _drawing_horizontal_segments,
    _horizontal_table_rules,
)

PAGE_W, PAGE_H = 595.0, 842.0
TABLE_HTML = (
    "<table>"
    "<tr><td>Method</td><td>Acc</td><td>F1</td></tr>"
    "<tr><td>Ours</td><td>12.3</td><td>45.6</td></tr>"
    "<tr><td>Base</td><td>10.1</td><td>40.2</td></tr>"
    "</table>"
)


def _table_page(doc, top: float = 85.0):
    page = doc.new_page(width=PAGE_W, height=PAGE_H) if doc.page_count == 0 else doc[0]
    rows = [("Method", "Acc", "F1"), ("Ours", "12.3", "45.6"), ("Base", "10.1", "40.2")]
    for index, row in enumerate(rows):
        y = top + 12 + index * 20
        for x, value in zip((65, 400, 470), row):
            page.insert_text((x, y), value, fontsize=9)
    for y in (top, top + 16, top + 56):
        page.draw_line((60, y), (530, y), width=0.6)
    # 표와 무관한 벡터 잡음 — get_drawings가 페이지 전체를 순회한다는 것을 드러낸다.
    for k in range(40):
        page.draw_line((60 + k * 10, 700), (65 + k * 10, 760), width=0.3)
    return page


@pytest.fixture
def table_page(tmp_path: Path):
    doc = fitz.open()
    _table_page(doc)
    path = tmp_path / "table.pdf"
    doc.save(path)
    doc.close()
    with fitz.open(path) as source:
        yield source[0]


def _count_textpages(monkeypatch) -> list[object]:
    calls: list[object] = []
    original = fitz.Page.get_textpage

    def counted(self, *args, **kwargs):
        calls.append(kwargs.get("clip"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_textpage", counted)
    return calls


def test_cell_search_builds_one_textpage_per_table(table_page, monkeypatch):
    """셀 9개를 찾아도 TextPage는 하나다 — 예전에는 셀마다 + 원문 판정 1회를 더 만들었다."""
    parsed = _table_cells(TABLE_HTML)
    assert parsed is not None
    cells, rows, cols = parsed
    calls = _count_textpages(monkeypatch)

    rects, trusted = _table_cell_rects(
        table_page, fitz.Rect(60, 85, 530, 141), cells, rows, cols,
    )

    assert trusted is True
    assert len(rects) == 9
    assert len(calls) == 1, calls


def test_cached_analysis_is_reused_across_plan_passes(table_page, monkeypatch):
    """계획 패스가 같은 표를 다시 봐도 TextPage를 다시 만들지 않는다."""
    parsed = _table_cells(TABLE_HTML)
    assert parsed is not None
    cells, rows, cols = parsed
    table_rect = fitz.Rect(60, 85, 530, 141)
    cache: dict = {}
    calls = _count_textpages(monkeypatch)

    first = _table_cell_rects(table_page, table_rect, cells, rows, cols, cache=cache)
    second = _table_cell_rects(table_page, table_rect, cells, rows, cols, cache=cache)

    assert first == second
    assert len(calls) == 1, calls


def test_shared_analysis_gives_the_same_grid_as_fresh_extraction(table_page):
    """TextPage·가로 선분을 공유해도 셀 사각형과 신뢰 판정이 그대로다."""
    parsed = _table_cells(TABLE_HTML)
    assert parsed is not None
    cells, rows, cols = parsed
    table_rect = fitz.Rect(60, 85, 530, 141)
    segments = [
        segment
        for drawing in table_page.get_drawings()
        for segment in _drawing_horizontal_segments(drawing)
    ]

    fresh = _table_cell_rects(table_page, table_rect, cells, rows, cols)
    shared = _table_cell_rects(
        table_page, table_rect, cells, rows, cols,
        cache={"horizontal_segments": segments},
    )

    assert fresh == shared
    assert _horizontal_table_rules(table_page, table_rect) == _horizontal_table_rules(
        table_page, table_rect, segments,
    )


def test_export_reads_page_drawings_once_for_several_tables(tmp_path, monkeypatch):
    """표 블록 × 계획 패스마다 get_drawings를 다시 부르지 않는다(페이지당 1회)."""
    job_dir = tmp_path / "two-tables"
    job_dir.mkdir()
    doc = fitz.open()
    _table_page(doc, top=85.0)
    _table_page(doc, top=300.0)
    doc.save(job_dir / "source.pdf")
    doc.close()

    def bbox(x0, y0, x1, y1):
        return [x0 / PAGE_W * 999, y0 / PAGE_H * 999, x1 / PAGE_W * 999, y1 / PAGE_H * 999]

    translated_html = TABLE_HTML.replace("Method", "방법").replace("Ours", "제안")
    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": [
        {"type": "table", "bbox": bbox(60, 85, 530, 141), "content": TABLE_HTML},
        {"type": "table", "bbox": bbox(60, 300, 530, 356), "content": TABLE_HTML},
    ]}]
    translated = json.loads(json.dumps(original))
    for block in translated[0]["blocks"]:
        block["content"] = translated_html
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    calls: list[int] = []
    original_get_drawings = fitz.Page.get_drawings

    def counted(self, *args, **kwargs):
        calls.append(self.number)
        return original_get_drawings(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_drawings", counted)
    result = build_translated_pdf(job_dir, "ko")

    assert result.table_cells_replaced == 4, result.report()
    assert calls == [0], calls


def test_rich_prefix_trials_reuse_the_font_archive_and_extract_once(monkeypatch):
    """run-in 굵은 라벨 시행마다 폰트 Archive를 다시 만들고 scratch를 두 번 추출하던 비용.

    서브셋이 없는 환경에서는 시행마다 26MB 폰트를 다시 읽었다. 한 블록의 시행은
    같은 폰트를 쓰므로 Archive는 한 번, scratch 페이지 추출은 시행당 한 번이다.
    """
    import pymupdf

    from app.pipeline.pdf_export import _plan_rich_prefix, _resolve_font

    fontfile, _name = _resolve_font("")
    if fontfile is None:
        pytest.skip("파일 기반 한글 폰트가 없다")
    archives: list[tuple] = []

    class CountingArchive(pymupdf.Archive):
        # insert_htmlbox가 isinstance로 검사하므로 함수가 아니라 하위 클래스로 센다.
        def __init__(self, *args, **kwargs):
            archives.append(args)
            super().__init__(*args, **kwargs)

    trials: list[float] = []
    real_insert = pymupdf.Page.insert_htmlbox

    def counting_insert(self, *args, **kwargs):
        result = real_insert(self, *args, **kwargs)
        if result[0] >= 0 and result[1] >= 0.999:
            trials.append(result[0])
        return result

    extractions: list[int] = []
    real_get_text = pymupdf.Page.get_text

    def counting_get_text(self, *args, **kwargs):
        extractions.append(1)
        return real_get_text(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf, "Archive", CountingArchive)
    monkeypatch.setattr(pymupdf.Page, "insert_htmlbox", counting_insert)
    monkeypatch.setattr(pymupdf.Page, "get_text", counting_get_text)
    page = pymupdf.open().new_page(width=PAGE_W, height=PAGE_H)
    text = "굵은 라벨 다음에 이어지는 본문이 상자에 들어가려면 몇 번의 시행이 필요하다. " * 3
    plan = _plan_rich_prefix(
        page,
        pymupdf.Rect(72, 100, 300, 130),
        text.strip(),
        ("", "굵은 라벨"),
        10.0,
        fontfile,
        max_rect=pymupdf.Rect(72, 100, 300, 260),
        lineheights=(1.52, 1.48, 1.44),
    )

    assert plan is not None
    # Story가 내부에서 만드는 빈 Archive는 빼고, 폰트 파일을 읽은 것만 센다.
    font_reads = [args for args in archives if args and str(args[0]) == fontfile]
    assert len(font_reads) == 1, font_reads
    assert trials, trials
    assert len(extractions) == len(trials), (len(extractions), len(trials))
