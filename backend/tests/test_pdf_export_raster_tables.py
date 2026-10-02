"""스캔 표 픽셀 격자(raster_tables) 단위 테스트 — 픽셀 단위로 정확히 그린 합성 표.

분석 배율(2px/pt)과 같은 해상도로 그린 전면 이미지라 행·열 빈 띠와 괘선 위치를 픽셀까지
정할 수 있다. 글자는 괘선으로 오인되지 않게 좁은 막대(6px) 여러 개로 그린다.
"""

from __future__ import annotations

import io

import fitz
import pytest
from PIL import Image, ImageDraw

from app.pipeline.pdf_export.raster_tables import raster_table_grid
from app.pipeline.pdf_export.tables import _table_cells

PAGE_W, PAGE_H = 595, 842
SCALE = 2                                    # 이미지 px/pt = 분석 배율
TABLE = fitz.Rect(60, 100, 300, 150)
ROW_TOPS = (208, 264)                         # 글자 막대 윗변(px)
LETTER_H = 16
COLUMN_LEFTS = (140, 440)


def _page_with_table(rule_top: int | None):
    image = Image.new("L", (PAGE_W * SCALE, PAGE_H * SCALE), 250)
    draw = ImageDraw.Draw(image)
    for top in ROW_TOPS:
        for left in COLUMN_LEFTS:
            for letter in range(5):          # 'Alpha' 다섯 글자 — 6px 막대, 3px 간격
                x = left + letter * 9
                draw.rectangle([x, top, x + 5, top + LETTER_H - 1], fill=20)
    if rule_top is not None:
        draw.rectangle([TABLE.x0 * SCALE, rule_top, TABLE.x1 * SCALE - 1, rule_top + 1], fill=0)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return doc, page


def _cells():
    parsed = _table_cells(
        "<table><tr><td>Alpha</td><td>Alpha</td></tr><tr><td>Alpha</td><td>Alpha</td></tr></table>"
    )
    assert parsed is not None
    return parsed


@pytest.mark.parametrize("rule_top", range(238, 247))
def test_cell_rects_keep_clear_of_a_rule_wherever_the_row_gap_falls(rule_top):
    """행 사이 빈 띠 가운데(px 244)가 괘선 가장자리에 딱 붙어도 셀은 괘선과 떨어진다.

    예전에는 경계가 괘선 아랫변과 정확히 같으면 그 괘선을 '셀 밖'으로 보고 밀어내지 않아,
    셀 조판이 늘 괘선 장애물과 충돌해 표 전체가 '공간 부족'으로 보존됐다(글꼴에 따라 우연히).
    """
    doc, page = _page_with_table(rule_top)
    cells, rows, cols = _cells()

    grid = raster_table_grid(page, TABLE, cells, rows, cols)
    doc.close()

    assert grid is not None
    rules = [rect for rect in grid.rule_rects if rect.width > 100]
    assert len(rules) == 1, grid.rule_rects
    rule = rules[0]
    for rect in grid.cell_rects:
        # 조판 충돌 판정은 글자 상자를 위아래로 0.5pt 넓혀 본다 — 그보다 떨어져 있어야 한다.
        assert rect.y1 <= rule.y0 - 0.5 or rect.y0 >= rule.y1 + 0.5, (rect, rule)


def test_table_without_any_ink_is_not_trusted():
    """글자 픽셀이 하나도 없는 표 이미지는 격자를 확정할 수 없다 — 덮지 않고 보존한다."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    blank = Image.new("L", (64, 64), 235)
    buffer = io.BytesIO()
    blank.save(buffer, "PNG")
    page.insert_image(TABLE, stream=buffer.getvalue(), keep_proportion=False)
    cells, rows, cols = _cells()

    assert raster_table_grid(page, TABLE, cells, rows, cols) is None
    doc.close()


def test_grid_finds_columns_and_rows_from_the_pixels():
    """균등 분할이 아니라 실제 글자 자리에서 열·행이 나온다(열 폭 80pt vs 160pt)."""
    doc, page = _page_with_table(None)
    cells, rows, cols = _cells()

    grid = raster_table_grid(page, TABLE, cells, rows, cols)
    doc.close()

    assert grid is not None
    first, second = grid.cell_rects[0], grid.cell_rects[1]
    # 글자는 70pt·220pt에서 시작한다 — 균등 격자(60·180pt 경계)와 다르다.
    assert first.x0 == pytest.approx(70, abs=1.0) and second.x0 == pytest.approx(220, abs=1.0)
    assert first.x1 < 220 - 1 and second.x1 >= 299
    ink = grid.ink_rects[0]
    assert ink is not None and ink.contains(fitz.Rect(70, 104, 92, 112))
    assert ink.y1 <= 122                       # 아래 행 글자(132pt~)를 덮지 않는다
