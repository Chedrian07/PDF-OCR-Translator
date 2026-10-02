"""스캔(래스터) 표의 셀 격자 — 원문 픽셀의 빈 띠로 실제 열·행 경계를 찾는다.

텍스트 레이어가 없는 표 이미지에서는 셀 검색(search_for)이 아무것도 찾지 못해 격자가
균등 분할로 떨어진다. born-digital 표는 원문 span 단위로 지우므로 격자가 어긋나도 번역
위치만 틀리지만, 스캔 표는 바뀐 셀의 영역을 바탕색으로 **덮는다**. 균등 격자로 덮으면
바뀌지 않은 이웃 셀의 글자와 괘선이 영구히 지워졌다(감사 pdf-1: 열 폭 50/90/340pt 표에서
'DOI' 머리글이 사라지고 괘선이 끊겼으며, 번역은 엉뚱한 열에 찍혔다).

그래서 표 영역을 렌더해(표시 공간)
 1) 표 폭·높이의 상당 부분을 잇는 긴 어두운 선분을 괘선으로 떼어 내고,
 2) 남은 글자 잉크의 가로 투영으로 줄 띠(→ 행)를, 세로 투영의 빈 띠로 열 경계를 찾고,
 3) HTML 셀 구조와 맞는지 검증한다 — 글자가 있는 셀마다 잉크가 있고, 서로 다른 셀
    사이 경계를 가로지르는 잉크가 없고, 고른 경계가 단어 사이 간격(여러 줄 셀이면
    행간)보다 뚜렷이 넓어야 하며, 셀의 잉크 폭이 글자 수에 비해 터무니없이 넓지
    않아야 한다(이웃 칸 글자를 품었다는 뜻).
하나라도 어긋나면 None — 호출부는 표 전체를 덮지 않고 원문 그대로 보존한다.

덮개는 셀 사각형이 아니라 **셀 안 글자 잉크의 bbox(+여유)** 다. 괘선 픽셀은 잉크에서
빠져 있고 덮개도 괘선을 넘지 않게 자르므로, 바뀐 셀의 글자만 지워진다.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..pdf import quiet_fitz
from .models import _TableCell
from .text import _plain_text

# 분석 렌더 배율. 144dpi면 8pt 글자도 획이 2px 이상이라 빈 띠와 잉크가 갈린다.
_ZOOM = 2.0
# OCR bbox는 0–999 정수 좌표라 가장자리 글리프를 1pt쯤 자르기도 한다 — 그만큼 더 본다.
_CLIP_PAD_PT = 1.5
# 이보다 적은 어두운 픽셀만 있는 줄·열은 비어 있다고 본다(스캔 티끌).
_MIN_PROJ_PX = 2
# 바탕 밝기의 이 비율보다 어두우면 잉크다. 흰 종이(250)면 155, 바랜 종이(228)면 141 —
# 글자(20~80)와 옅은 셀 음영(≥180)이 둘 다 이 문턱의 반대편에 온다.
_DARK_RATIO = 0.62
# 괘선: 이만큼 이어진 어두운 런. 글자 획(가장 긴 것이 대시·'T'의 가로획 1em 남짓)보다
# 훨씬 길고, 부분 괘선(cmidrule)도 잡히도록 표 폭의 일부만 요구한다.
_HRULE_MIN_PT = 18.0
_HRULE_MIN_FRACTION = 0.15
_VRULE_MIN_PT = 16.0
_VRULE_MIN_FRACTION = 0.45
# i·j의 점, 악센트처럼 같은 줄인데 몇 px 떨어진 잉크를 한 줄 띠로 합치는 간격.
_BAND_MERGE_PT = 1.0
# 이보다 잉크가 적은 줄 띠는 티끌로 버린다.
_MIN_BAND_INK_PX = 6
# 고른 경계(열 사이·행 사이 빈 띠)는 고르지 않은 것(단어 사이·여러 줄 셀의 행간)보다
# 이만큼 넓어야 한다 — 그렇지 않으면 어느 쪽이 경계인지 픽셀만으로는 모른다.
_GAP_MARGIN = 1.25
_GAP_MARGIN_PX = 2
# 글자가 있는 셀이 가져야 할 최소 잉크(마침표 하나 남짓).
_MIN_CELL_INK_PX = 3
# 셀 잉크 폭/글자 수가 표의 중앙값보다 이 배수 넘게 크면 이웃 칸 글자를 품은 것이다.
_WIDTH_PER_CHAR_RATIO = 3.0
_WIDTH_CHECK_MIN_CHARS = 3
# 덮개 여유 — 안티앨리어싱 가장자리와 임계값 아래의 옅은 획 끝까지 덮는다.
_INK_PAD_PT = 1.0
# 번역 글자와 괘선·열 경계 사이에 남길 간격. 조판 충돌 판정이 글자 상자를 위아래로
# 0.5pt 넓혀 보므로 그보다 커야 괘선 바로 아래 셀이 늘 '충돌'로 떨어지지 않는다.
_RULE_TEXT_GAP_PT = 0.75
_CELL_EDGE_PAD_PT = 2.0
# baseline: 줄 띠에서 가로 투영이 정점의 이 비율 이상인 마지막 픽셀 줄. 그 아래는
# 디센더(g·p·y의 가는 획)뿐이라 투영이 뚝 떨어진다.
_BASELINE_DENSITY = 0.35
# 줄 띠 윗변~baseline(대문자·어센더·숫자의 높이) 대비 글자 크기. 셋 다 약 0.68em이고
# 144dpi 분석의 번짐(≈0.25pt)을 더하면 0.71em이다. 띠 전체 높이는 디센더가 있는 행과
# 없는 행이 0.68em/0.9em로 갈려 크기 추정에 쓰지 않는다(실측: 같은 표에서 ±15%).
_ASCENT_EM = 0.71


@dataclass(frozen=True)
class RasterTableGrid:
    """래스터 표 한 개의 격자. 사각형은 모두 비회전 PDF 좌표다(셀 순서 = HTML 셀 순서)."""

    # 번역을 앉힐 셀 사각형(괘선·열 경계 안쪽, 정렬에 맞춰 원문 글자 자리에 붙인다).
    cell_rects: tuple
    # 셀 안 원문 글자 잉크를 덮을 사각형. 잉크가 없는 셀은 None.
    ink_rects: tuple
    # 0=왼쪽, 1=가운데, 2=오른쪽 — 원문 잉크가 그 열 안에서 놓인 자리.
    aligns: tuple
    # 래스터 괘선 — 번역이 그 위에 찍히지 않게 장애물로 쓴다.
    rule_rects: tuple
    # 원문 글자 크기 추정(pt). 모르면 None.
    font_pt: float | None
    # 셀 첫 줄의 원문 baseline — **표시 공간** y(pt). 한 줄 번역을 같은 행의 남는 영어와
    # 같은 높이에 앉힐 때 쓴다. 잉크가 없는 셀은 None.
    baselines: tuple = ()


def raster_table_grid(
    page, table_rect, cells: list[_TableCell], rows: int, cols: int,
    *, cache: dict | None = None, key: object = None,
) -> RasterTableGrid | None:
    """래스터 표의 셀 격자. 픽셀로 확정할 수 없으면 None(호출부가 표를 보존한다).

    `cache`는 한 페이지 계획 동안 살아 있는 분석 캐시다 — 계획 패스마다 같은 표를
    다시 렌더하지 않는다. 분석 실패(렌더 오류 등)도 None이다.
    """
    cache_key = ("raster_table", key)
    if cache is not None and key is not None and cache_key in cache:
        return cache[cache_key]
    try:
        grid = _estimate_grid(page, table_rect, cells, rows, cols)
    except Exception:  # noqa: BLE001 — 분석 실패는 '확정 불가'와 같다(표 보존)
        grid = None
    if cache is not None and key is not None:
        cache[cache_key] = grid
    return grid


def _runs(mask) -> list[tuple[int, int]]:
    """1차원 bool 배열의 True 구간 [start, end) 목록."""
    import numpy as np

    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1).tolist()
    ends = np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, ends))


def _long_runs(mask, length: int):
    """각 행(axis=1)에서 길이 `length` 이상 이어진 True 런에 속한 픽셀만 True."""
    import numpy as np

    height, width = mask.shape
    if length <= 0 or length > width:
        return np.zeros_like(mask)
    prefix = np.zeros((height, width + 1), dtype=np.int32)
    np.cumsum(mask, axis=1, out=prefix[:, 1:])
    full = (prefix[:, length:] - prefix[:, :-length]) == length   # 시작점 s: [s, s+length)
    starts = np.zeros((height, full.shape[1] + 1), dtype=np.int32)
    np.cumsum(full, axis=1, out=starts[:, 1:])
    xs = np.arange(width)
    low = np.clip(xs - length + 1, 0, None)
    high = np.clip(xs, None, width - length) + 1
    return (starts[:, high] - starts[:, low]) > 0


def _rule_boxes(mask, horizontal: bool) -> list[tuple[int, int, int, int]]:
    """괘선 마스크 → 이어진 선분마다 px 상자 (x0, y0, x1, y1)."""
    out: list[tuple[int, int, int, int]] = []
    if horizontal:
        for y0, y1 in _runs(mask.any(axis=1)):
            for x0, x1 in _runs(mask[y0:y1].any(axis=0)):
                out.append((x0, y0, x1, y1))
    else:
        for x0, x1 in _runs(mask.any(axis=0)):
            for y0, y1 in _runs(mask[:, x0:x1].any(axis=1)):
                out.append((x0, y0, x1, y1))
    return out


def _choose_gaps(
    gaps: list[tuple[int, int]], need: int, forced: set[int] | frozenset = frozenset(),
) -> list[int] | None:
    """빈 띠 후보 `(인덱스, 폭)`에서 경계 `need`개를 고른다. 모호하면 None.

    `forced`(괘선이 지나가는 빈 띠)는 무조건 경계다. 나머지는 넓은 순으로 고르되, 고른
    것 중 가장 좁은 것이 고르지 않은 것 중 가장 넓은 것보다 뚜렷이 넓어야 한다.
    """
    if need < 0 or len(gaps) < need or len(forced) > need:
        return None
    rest = sorted(
        ((width, index) for index, width in gaps if index not in forced), reverse=True,
    )
    picked = rest[: need - len(forced)]
    left = rest[need - len(forced):]
    if picked and left and picked[-1][0] < left[0][0] * _GAP_MARGIN + _GAP_MARGIN_PX:
        return None
    return sorted(list(forced) + [index for _width, index in picked])


def _estimate_grid(page, table_rect, cells, rows: int, cols: int) -> RasterTableGrid | None:
    import numpy as np

    fitz = quiet_fitz()
    if rows <= 0 or cols <= 0 or not cells:
        return None
    # 분석은 표시 공간에서 한다 — 회전 페이지에서도 HTML의 행·열이 화면의 가로·세로다.
    shown = table_rect * page.rotation_matrix
    shown.normalize()
    clip = +shown
    clip += (-_CLIP_PAD_PT, -_CLIP_PAD_PT, _CLIP_PAD_PT, _CLIP_PAD_PT)
    clip &= page.rect
    if clip.is_empty or clip.width < 4 or clip.height < 4:
        return None
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(_ZOOM, _ZOOM), clip=clip, colorspace=fitz.csGRAY, alpha=False,
    )
    height, width = pixmap.height, pixmap.width
    if height < 4 or width < 4:
        return None
    gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        height, pixmap.stride,
    )[:, :width]
    origin_x, origin_y = pixmap.x, pixmap.y

    def to_rect(x0: float, y0: float, x1: float, y1: float):
        """px 상자 → 비회전 PDF 좌표 Rect."""
        rect = fitz.Rect(
            (origin_x + x0) / _ZOOM, (origin_y + y0) / _ZOOM,
            (origin_x + x1) / _ZOOM, (origin_y + y1) / _ZOOM,
        )
        rect = rect * page.derotation_matrix
        rect.normalize()
        return rect

    table_x0 = max(0, int(round(shown.x0 * _ZOOM - origin_x)))
    table_x1 = min(width, int(round(shown.x1 * _ZOOM - origin_x)))
    table_y0 = max(0, int(round(shown.y0 * _ZOOM - origin_y)))
    table_y1 = min(height, int(round(shown.y1 * _ZOOM - origin_y)))
    if table_x1 - table_x0 < 4 or table_y1 - table_y0 < 4:
        return None

    # 1) 바탕(가장 흔한 밝기)과 잉크. 어두운 바탕(흰 글자 표)은 다루지 않는다.
    background = int(np.bincount(gray.ravel(), minlength=256).argmax())
    if background < 128:
        return None
    dark = gray < background * _DARK_RATIO
    if not dark.any():
        return None

    # 2) 괘선을 떼어 낸다 — 글자 잉크의 투영을 흐리고, 덮개가 지워서는 안 되는 픽셀이다.
    hrule = _long_runs(dark, max(
        int(round(_HRULE_MIN_PT * _ZOOM)),
        int(round((table_x1 - table_x0) * _HRULE_MIN_FRACTION)),
    ))
    vrule = _long_runs(dark.T, max(
        int(round(_VRULE_MIN_PT * _ZOOM)),
        int(round((table_y1 - table_y0) * _VRULE_MIN_FRACTION)),
    )).T
    ink = dark & ~hrule & ~vrule
    hrules = _rule_boxes(hrule, horizontal=True)
    vrules = _rule_boxes(vrule, horizontal=False)

    # 3) 줄 띠 — 가로 투영. 점·악센트 간격은 합치고, 표 밖(여유 띠)에 중심이 있는 띠와
    #    티끌은 버린다.
    row_counts = ink.sum(axis=1)
    merge_px = max(1, int(round(_BAND_MERGE_PT * _ZOOM)))
    merged: list[list[int]] = []
    for y0, y1 in _runs(row_counts >= _MIN_PROJ_PX):
        if merged and y0 - merged[-1][1] <= merge_px:
            merged[-1][1] = y1
        else:
            merged.append([y0, y1])
    bands = [
        (y0, y1) for y0, y1 in merged
        if table_y0 <= (y0 + y1) / 2 <= table_y1
        and int(row_counts[y0:y1].sum()) >= _MIN_BAND_INK_PX
    ]
    if len(bands) < rows:
        return None
    in_band = np.zeros(height, dtype=bool)
    for y0, y1 in bands:
        in_band[y0:y1] = True
    ink &= in_band[:, None]

    # 4) 줄 띠 → 행. 괘선이 지나가는 빈 띠는 무조건 행 경계다.
    row_gaps = [(index, bands[index + 1][0] - bands[index][1]) for index in range(len(bands) - 1)]
    forced = {
        index for index, _width in row_gaps
        if hrule[bands[index][1]:bands[index + 1][0]].any()
    }
    row_breaks = _choose_gaps(row_gaps, rows - 1, forced)
    if row_breaks is None:
        return None
    groups: list[tuple[int, int]] = []
    start = 0
    for index in row_breaks + [len(bands) - 1]:
        groups.append((bands[start][0], bands[index][1]))
        start = index + 1
    if len(groups) != rows:
        return None
    row_bounds = [min(table_y0, groups[0][0])]
    row_bounds += [(groups[i][1] + groups[i + 1][0]) // 2 for i in range(rows - 1)]
    row_bounds.append(max(table_y1, groups[-1][1]))

    # 5) 열 경계 — 가로 병합 셀이 없는 행들의 세로 투영에서 빈 띠를 고른다. 표 밖(여유
    #    띠)에 중심이 있는 잉크 덩어리는 이웃 내용이므로 표에서 뺀다.
    simple_rows = [
        row for row in range(rows)
        if not any(
            cell.colspan > 1 and cell.row <= row < cell.row + cell.rowspan for cell in cells
        )
    ] or list(range(rows))
    use = np.zeros(height, dtype=bool)
    for row in simple_rows:
        use[groups[row][0]:groups[row][1]] = True
    column_ink = ink[use].sum(axis=0) >= _MIN_PROJ_PX
    kept_runs = [
        (x0, x1) for x0, x1 in _runs(column_ink)
        if table_x0 <= (x0 + x1) / 2 <= table_x1
    ]
    if not kept_runs:
        return None
    first, last = kept_runs[0][0], kept_runs[-1][1]
    left_edge, right_edge = min(table_x0, first), max(table_x1, last)
    ink[:, :left_edge] = False
    ink[:, right_edge:] = False
    column_ink[:first] = False
    column_ink[last:] = False
    column_gaps = [(x0 + first, x1 + first) for x0, x1 in _runs(~column_ink[first:last])]
    chosen = _choose_gaps(
        [(index, x1 - x0) for index, (x0, x1) in enumerate(column_gaps)], cols - 1,
    )
    if chosen is None:
        return None
    col_bounds = [left_edge]
    col_bounds += [(column_gaps[index][0] + column_gaps[index][1]) // 2 for index in chosen]
    col_bounds.append(right_edge)

    # 6) 서로 다른 셀 사이의 열 경계를 가로지르는 잉크가 없어야 한다 — 가로 병합 셀이
    #    있는 행은 위 빈 띠 계산에 들어가지 않았으므로 여기서 따로 본다.
    owner = {}
    for cell_index, cell in enumerate(cells):
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                owner[(row, col)] = cell_index
    for row in range(rows):
        y0, y1 = groups[row]
        for col in range(1, cols):
            if owner.get((row, col - 1)) == owner.get((row, col)):
                continue
            x = col_bounds[col]
            if int(ink[y0:y1, max(0, x - 1):x + 2].sum()) >= _MIN_PROJ_PX:
                return None

    # 7) 셀마다 잉크 상자 — 글자가 있는 셀에 잉크가 없으면 격자가 HTML과 다르다. 셀
    #    잉크 폭이 글자 수에 비해 지나치게 넓으면 이웃 칸 글자를 품은 것이다.
    boxes: list[tuple[int, int, int, int] | None] = []
    widths_per_char: list[float] = []
    for cell in cells:
        x0, x1 = col_bounds[cell.col], col_bounds[cell.col + cell.colspan]
        y0, y1 = row_bounds[cell.row], row_bounds[cell.row + cell.rowspan]
        ys, xs = np.nonzero(ink[y0:y1, x0:x1])
        text = _plain_text(cell.text).strip()
        if len(ys) < _MIN_CELL_INK_PX:
            if text:
                return None
            boxes.append(None)
            continue
        box = (
            x0 + int(xs.min()), y0 + int(ys.min()),
            x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1,
        )
        boxes.append(box)
        chars = len("".join(text.split()))
        if "\n" not in text and chars >= _WIDTH_CHECK_MIN_CHARS:
            widths_per_char.append((box[2] - box[0]) / chars)
    if len(widths_per_char) >= 3:
        typical = median(widths_per_char)
        if typical > 0 and max(widths_per_char) > typical * _WIDTH_PER_CHAR_RATIO:
            return None

    # 8) 정렬 — 원문 잉크가 열(병합 셀이면 그 셀이 덮는 열들)의 글자 범위 안 어디에 있나.
    column_extent: dict[int, tuple[int, int]] = {}
    for cell, box in zip(cells, boxes):
        if box is None or cell.colspan != 1:
            continue
        low, high = column_extent.get(cell.col, (box[0], box[2]))
        column_extent[cell.col] = (min(low, box[0]), max(high, box[2]))
    slack = 2.0 * _ZOOM
    aligns = []
    for cell, box in zip(cells, boxes):
        if box is None:
            aligns.append(0)
            continue
        spans = [column_extent[c] for c in range(cell.col, cell.col + cell.colspan) if c in column_extent]
        low = min((extent[0] for extent in spans), default=box[0])
        high = max((extent[1] for extent in spans), default=box[2])
        before, after = box[0] - low, high - box[2]
        if before > slack and after > slack and abs(before - after) <= max(slack, (high - low) * 0.15):
            aligns.append(1)
        elif after <= slack < before:
            aligns.append(2)
        else:
            aligns.append(0)

    # 9) 조판 사각형과 덮개. 괘선은 셀 안쪽으로 들어온 것만 잘라 낸다.
    rule_gap = _RULE_TEXT_GAP_PT * _ZOOM
    edge = _CELL_EDGE_PAD_PT * _ZOOM
    pad = _INK_PAD_PT * _ZOOM
    cell_rects = []
    ink_rects = []
    for cell, box, align in zip(cells, boxes, aligns):
        x0, x1 = col_bounds[cell.col], col_bounds[cell.col + cell.colspan]
        y0, y1 = row_bounds[cell.row], row_bounds[cell.row + cell.rowspan]
        center_x = (box[0] + box[2]) / 2 if box else (x0 + x1) / 2
        center_y = (box[1] + box[3]) / 2 if box else (y0 + y1) / 2
        # 열 경계(빈 띠 가운데)에서는 조금 띄우고, 표 바깥 가장자리는 그대로 쓴다.
        lo_x = float(x0) + (edge if cell.col > 0 else 0.0)
        hi_x = float(x1) - (edge if cell.col + cell.colspan < cols else 0.0)
        lo_y, hi_y = float(y0), float(y1)
        # 셀 경계(빈 띠 가운데)가 괘선 가장자리에 딱 붙는 경우도 있다 — 간격 안에 든 괘선도 민다.
        for rx0, ry0, rx1, ry1 in hrules:
            if (
                min(rx1, x1) - max(rx0, x0) < (x1 - x0) * 0.5
                or ry1 + rule_gap <= lo_y or ry0 - rule_gap >= hi_y
            ):
                continue
            if (ry0 + ry1) / 2 < center_y:
                lo_y = max(lo_y, ry1 + rule_gap)
            else:
                hi_y = min(hi_y, ry0 - rule_gap)
        for rx0, ry0, rx1, ry1 in vrules:
            if (
                min(ry1, y1) - max(ry0, y0) < (y1 - y0) * 0.5
                or rx1 + rule_gap <= lo_x or rx0 - rule_gap >= hi_x
            ):
                continue
            if (rx0 + rx1) / 2 < center_x:
                lo_x = max(lo_x, rx1 + rule_gap)
            else:
                hi_x = min(hi_x, rx0 - rule_gap)
        # 원문 글자 자리에 맞춘다 — 왼쪽 정렬이면 원문이 시작하던 x에서, 오른쪽이면 끝나던
        # x까지, 가운데면 원문 중심을 축으로(열 경계 안에서 좌우 대칭) 쓴다.
        if box is not None and align == 0:
            lo_x = max(lo_x, min(float(box[0]), hi_x - 2))
        elif box is not None and align == 2:
            hi_x = min(hi_x, max(float(box[2]), lo_x + 2))
        elif box is not None and align == 1:
            half = min(center_x - lo_x, hi_x - center_x)
            lo_x, hi_x = center_x - half, center_x + half
        if hi_x - lo_x < 2 or hi_y - lo_y < 2:
            return None
        cell_rects.append(to_rect(lo_x, lo_y, hi_x, hi_y))
        if box is None:
            ink_rects.append(None)
            continue
        cover = [
            max(float(x0), box[0] - pad), max(float(y0), box[1] - pad),
            min(float(x1), box[2] + pad), min(float(y1), box[3] + pad),
        ]
        for rx0, ry0, rx1, ry1 in hrules:
            if rx1 <= cover[0] or rx0 >= cover[2] or ry1 <= cover[1] or ry0 >= cover[3]:
                continue
            if (ry0 + ry1) / 2 < center_y:
                cover[1] = max(cover[1], float(ry1))
            else:
                cover[3] = min(cover[3], float(ry0))
        for rx0, ry0, rx1, ry1 in vrules:
            if rx1 <= cover[0] or rx0 >= cover[2] or ry1 <= cover[1] or ry0 >= cover[3]:
                continue
            if (rx0 + rx1) / 2 < center_x:
                cover[0] = max(cover[0], float(rx1))
            else:
                cover[2] = min(cover[2], float(rx0))
        ink_rects.append(
            to_rect(*cover) if cover[2] > cover[0] and cover[3] > cover[1] else None
        )

    def baseline_px(y0: int, y1: int, x0: int, x1: int) -> int:
        """줄 띠 [y0, y1)·열 [x0, x1)에서 투영이 짙은 마지막 픽셀 줄의 아랫변."""
        counts = ink[y0:y1, x0:x1].sum(axis=1)
        dense = np.flatnonzero(counts >= max(1, counts.max() * _BASELINE_DENSITY))
        return y0 + int(dense[-1]) + 1 if len(dense) else y1

    ascents = [
        (baseline_px(y0, y1, 0, width) - y0) / _ZOOM for y0, y1 in bands
    ]
    font_pt = median(ascents) / _ASCENT_EM if ascents else None
    baselines = []
    for box in boxes:
        # 셀 잉크가 시작하는 줄 띠에서 그 셀 글자의 baseline.
        band = None if box is None else next(
            ((y0, y1) for y0, y1 in bands if y1 > box[1]), None,
        )
        if band is None:
            baselines.append(None)
            continue
        baselines.append((origin_y + baseline_px(band[0], band[1], box[0], box[2])) / _ZOOM)
    rules = tuple(to_rect(*box) for box in hrules + vrules)
    return RasterTableGrid(
        tuple(cell_rects), tuple(ink_rects), tuple(aligns), rules, font_pt,
        tuple(baselines),
    )
