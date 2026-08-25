"""HTML 표 파싱과 원문 PDF 위 셀 격자 복원."""
from __future__ import annotations

from html.parser import HTMLParser
from statistics import median

from ..pdf import quiet_fitz
from .constants import _MAX_TABLE_CELLS, _MIN_FONT_PT, _TABLE_RULE_TEXT_GAP_PT
from .geometry import _rect_horizontal_overlap
from .models import _RawTableCell, _SourceSpan, _TableCell
from .spans import _source_span_records
from .text import _plain_text


class _TableParser(HTMLParser):
    """번역 레이아웃의 HTML table을 span 정보와 함께 파싱한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_RawTableCell]] = []
        self._row: list[_RawTableCell] | None = None
        self._cell: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            values = {str(key).lower(): str(value) for key, value in attrs if value is not None}
            try:
                self._rowspan = max(1, int(values.get("rowspan", "1")))
                self._colspan = max(1, int(values.get("colspan", "1")))
            except ValueError:
                self._rowspan = self._colspan = 1
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_RawTableCell(
                _plain_text("".join(self._cell)),
                self._rowspan,
                self._colspan,
            ))
            self._cell = None
            self._rowspan = self._colspan = 1
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _table_cells(content: str) -> tuple[list[_TableCell], int, int] | None:
    """HTML 표를 rowspan/colspan을 보존한 논리 셀과 격자 크기로 변환한다."""
    parser = _TableParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:  # noqa: BLE001 — 손상 OCR HTML은 표 번역만 보수적으로 생략
        return None
    if not parser.rows or not parser.rows[0]:
        return None

    occupied: dict[tuple[int, int], _TableCell] = {}
    cells: list[_TableCell] = []
    max_col = 0
    for row_index, raw_row in enumerate(parser.rows):
        col_index = 0
        for raw in raw_row:
            if (
                len(cells) >= _MAX_TABLE_CELLS
                or raw.rowspan > _MAX_TABLE_CELLS
                or raw.colspan > _MAX_TABLE_CELLS
                or raw.rowspan * raw.colspan > _MAX_TABLE_CELLS
            ):
                return None
            while (row_index, col_index) in occupied:
                col_index += 1
            cell = _TableCell(
                row_index,
                col_index,
                raw.rowspan,
                raw.colspan,
                raw.text,
            )
            for rr in range(row_index, row_index + raw.rowspan):
                for cc in range(col_index, col_index + raw.colspan):
                    if (rr, cc) in occupied:
                        return None
                    occupied[(rr, cc)] = cell
            cells.append(cell)
            col_index += raw.colspan
            max_col = max(max_col, col_index)

    row_count = max((row + 1 for row, _ in occupied), default=0)
    if row_count == 0 or max_col == 0:
        return None
    if any((row, col) not in occupied
           for row in range(row_count) for col in range(max_col)):
        return None
    return cells, row_count, max_col


def _table_matrix(content: str) -> list[list[str]] | None:
    """호환용 펼친 행렬. span 셀은 점유하는 모든 격자 위치에 같은 값을 둔다."""
    parsed = _table_cells(content)
    if parsed is None:
        return None
    cells, rows, cols = parsed
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                matrix[row][col] = cell.text
    return matrix


def _grid_boundaries(centers: list[list[float]], lo: float, hi: float) -> list[float]:
    """행/열별 관측 중심값을 경계로 변환. 관측이 비면 균등 분할 폴백."""
    n = len(centers)
    values: list[float] = []
    for i, samples in enumerate(centers):
        values.append(median(samples) if samples else lo + (i + 0.5) * (hi - lo) / n)
    # OCR/search 오차로 순서가 뒤집히면 안전한 균등 격자로 폴백한다.
    if any(values[i] >= values[i + 1] for i in range(n - 1)):
        values = [lo + (i + 0.5) * (hi - lo) / n for i in range(n)]
    return [lo] + [(values[i] + values[i + 1]) / 2 for i in range(n - 1)] + [hi]


def _horizontal_table_rules(page, table_rect) -> list[tuple[float, float, float, float]]:
    """표 영역의 가로 rule을 `(x0, x1, y, width)`로 반환한다."""
    rules: list[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 — rule 보정은 품질 향상용
        return rules
    for drawing in drawings:
        width = max(0.1, float(drawing.get("width") or 0.1))
        for item in drawing.get("items", []):
            if not item or item[0] != "l" or len(item) < 3:
                continue
            start, end = item[1], item[2]
            if abs(float(start.y) - float(end.y)) > max(0.5, width):
                continue
            x0, x1 = sorted((float(start.x), float(end.x)))
            y = (float(start.y) + float(end.y)) / 2
            if (
                x1 < table_rect.x0 - 2
                or x0 > table_rect.x1 + 2
                or y < table_rect.y0 - 3
                or y > table_rect.y1 + 3
            ):
                continue
            rules.append((x0, x1, y, width))
    return rules


def _table_cell_rects(
    page,
    table_rect,
    original: list[_TableCell],
    rows: int,
    cols: int,
) -> tuple[list[object], bool]:
    """원문 셀 검색 중심으로 표 격자를 추정해 `(셀 사각형, 격자 신뢰 여부)`를 낸다.

    `grid_trusted=False`는 원문 셀을 거의 찾지 못해 균등 분할로 강행했다는 뜻이다
    — 호출부는 그런 표를 교체하지 말고 원문 그대로 보존해야 한다.
    """
    x_centers: list[list[float]] = [[] for _ in range(cols)]
    y_centers: list[list[float]] = [[] for _ in range(rows)]
    fitz = quiet_fitz()
    search_clip = +table_rect
    search_clip += (-2.0, -12.0, 2.0, 12.0)
    search_clip &= page.mediabox
    observed: list[object] = []
    for cell in original:
        if not cell.text:
            continue
        expected_x = table_rect.x0 + (
            cell.col + cell.colspan / 2
        ) * table_rect.width / cols
        expected_y = table_rect.y0 + (
            cell.row + cell.rowspan / 2
        ) * table_rect.height / rows
        hits = page.search_for(cell.text, clip=search_clip)
        if not hits:
            continue
        hit = min(hits, key=lambda r: abs((r.x0 + r.x1) / 2 - expected_x)
                  + abs((r.y0 + r.y1) / 2 - expected_y))
        observed.append(hit)
        if cell.colspan == 1:
            x_centers[cell.col].append((hit.x0 + hit.x1) / 2)
        if cell.rowspan == 1:
            y_centers[cell.row].append((hit.y0 + hit.y1) / 2)
    grid_x0 = min([table_rect.x0] + [hit.x0 - 2.0 for hit in observed])
    grid_x1 = max([table_rect.x1] + [hit.x1 + 2.0 for hit in observed])
    grid_y0 = min([table_rect.y0] + [hit.y0 - 1.0 for hit in observed])
    grid_y1 = max([table_rect.y1] + [hit.y1 + 1.0 for hit in observed])
    # 텍스트 레이어가 있는 표에서 원문 셀 검색이 절반도 맞지 않으면 아래 균등
    # 격자는 실제 열 폭과 무관하다 — 번역문이 엉뚱한 셀에 찍히고 인접 셀 원문이
    # 리댁션된다. 스캔 표(텍스트 레이어 없음)는 지울 원문이 없어 균등 격자가
    # 무해하므로 그대로 신뢰한다.
    try:
        has_source_text = bool(page.get_text("text", clip=search_clip).strip())
    except Exception:  # noqa: BLE001 — 추출 실패는 스캔 표와 같게 취급
        has_source_text = False
    grid_trusted = not has_source_text or (
        len(observed) * 2 >= sum(1 for cell in original if cell.text)
    )
    xs = _grid_boundaries(x_centers, grid_x0, grid_x1)
    ys = _grid_boundaries(y_centers, grid_y0, grid_y1)
    horizontal_rules = _horizontal_table_rules(page, table_rect)
    rects: list[object] = []
    for cell in original:
        rect = fitz.Rect(
            xs[cell.col],
            ys[cell.row],
            xs[cell.col + cell.colspan],
            ys[cell.row + cell.rowspan],
        )
        xpad = min(2.0, rect.width * 0.04)
        ypad = min(0.6, rect.height * 0.04)
        rect.x0 += xpad
        rect.x1 -= xpad
        rect.y0 += ypad
        rect.y1 -= ypad
        # OCR table bbox가 LaTeX `\toprule`보다 2–3pt 위에서 시작하면 CJK의
        # 큰 ascender가 rule에 관통된다. 셀 윗부분 *안쪽*에 들어온 rule만
        # 실제 상단으로 삼고, 가까운 다음 rule까지의 안전한 띠를 사용한다.
        spanning_rules = sorted(
            (y, width)
            for x0, x1, y, width in horizontal_rules
            if _rect_horizontal_overlap(
                fitz.Rect(x0, y - width / 2, x1, y + width / 2), rect,
            ) >= rect.width * 0.5
        )
        interior_top = next((
            (y, width)
            for y, width in spanning_rules
            if (
                rect.y0 - max(1.0, width / 2 + 0.25)
                <= y
                <= rect.y0 + min(3.5, rect.height * 0.35)
            )
        ), None)
        if interior_top is not None:
            top_y, top_width = interior_top
            # 기존처럼 rule이 셀 내부에 충분히 들어온 경우 0.25pt면 안전하다.
            # 표 최상단 rule도 기존 header 여백을 유지한다. 그 밖의 내부 rule을
            # OCR 중심 경계가 이미 지나친 경우에만 더 큰 간격을 쓴다.
            gap = (
                0.25
                if top_y <= table_rect.y0 + 4.0
                or top_y > rect.y0 + 0.25
                else _TABLE_RULE_TEXT_GAP_PT
            )
            rect.y0 = max(
                rect.y0,
                top_y + top_width / 2 + gap,
            )
            lower = next((
                (y, width) for y, width in spanning_rules
                if y > top_y + 1.0 and y <= rect.y1 + 2.0
            ), None)
            if lower is not None:
                lower_y, lower_width = lower
                rect.y1 = max(rect.y1, lower_y - lower_width / 2 - 0.1)
        rects.append(rect)
    return rects, grid_trusted


def _table_cell_source_style(
    page, cell_rect, records: list[_SourceSpan] | None = None,
) -> tuple[float, int, bool, object]:
    """원본 셀의 실제 span 크기·정렬·굵기와 텍스트 redaction 영역을 추정한다.

    `records`는 페이지에서 이미 한 번 읽어둔 span 목록이다. 넘기지 않으면 이
    함수가 직접 추출하지만, 셀마다 페이지 전체를 재추출하면 큰 표에서 비용이
    셀 수만큼 곱해진다.
    """
    if records is None:
        records = _source_span_records(quiet_fitz(), page)
    # `get_text(..., clip=...)`는 경계에 걸친 이웃 span을 잘라서 반환한다.
    # 그 잘린 조각까지 현재 셀의 redaction으로 합치면, 왼쪽 셀 번역이 오른쪽
    # 값의 첫 글자들을 지우는 일이 생긴다. 전체 span의 중심이 실제 셀 안에
    # 있는 경우만 소유하게 해 인접 셀 경계를 침범하지 않는다.
    spans = [
        span for span in records
        if cell_rect.contains((span.rect.tl + span.rect.br) / 2)
    ]

    if not spans:
        size = min(12.0, max(_MIN_FONT_PT, cell_rect.height * 0.65))
        return size, 0, False, +cell_rect

    sizes = [span.size for span in spans if span.size > 0]
    source_size = median(sizes) if sizes else cell_rect.height * 0.65
    ink = +spans[0].rect
    bold_chars = total_chars = 0
    for span in spans:
        ink.include_rect(span.rect)
        count = max(1, len(span.text.strip()))
        total_chars += count
        if span.flags & 16:
            bold_chars += count

    center = (ink.x0 + ink.x1) / 2
    if abs(center - (cell_rect.x0 + cell_rect.x1) / 2) <= cell_rect.width * 0.12:
        align = 1
    elif ink.x1 >= cell_rect.x1 - cell_rect.width * 0.08:
        align = 2
    else:
        align = 0
    ink += (-0.4, -0.3, 0.4, 0.3)
    ink &= cell_rect
    return min(12.0, max(_MIN_FONT_PT, source_size * 1.03)), align, (
        bold_chars > total_chars * 0.5
    ), ink
