"""내보내기 오케스트레이션 — 계획을 모아 리댁션하고 번역문을 삽입한다.

페이지마다 (1) 모든 블록의 삽입 계획 수집 → (2) 원문 텍스트 일괄 리댁션 →
(3) 번역문 삽입 순서를 지킨다. 계획이 전부 끝나기 전에는 어떤 원문도 지우지
않으므로 조판이 실패한 블록은 원문 글리프가 그대로 남는다.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..layout import estimate_font_size_cqw
from ..pdf import quiet_fitz
from .constants import (
    _BODY_LINEHEIGHTS,
    _CAPTION_LINEHEIGHTS,
    _MAX_FONT_PT,
    _MAX_TABLE_CELLS,
    _MIN_FONT_PT,
    _MIN_TABLE_FONT_PT,
    _LASTRESORT_MIN_FONT_PT,
    _LASTRESORT_SHRINK_STEPS,
    _PRESERVE_TYPES,
    _REDACT_CHUNK,
    _REPLACEABLE_TYPES,
    _SPECIALIST_TYPES,
    _TITLE_LINEHEIGHTS,
    _VERTICAL_SKIP,
)
from .fitting import (
    _flow_components,
    _plan_flow_group,
    _plan_listing_lines,
    _plan_shrink_to_fit,
    _plan_single_line,
    _preserved_reference_microfixes,
)
from .fonts import (
    _SYSTEM_SANS_FONT_CANDIDATES,
    _balance_title_text,
    _document_font_resource_names,
    _portable_text_for_font,
    _resolve_font,
    _unique_font_resource_name,
)
from .geometry import _ink_collides, _rect_overlap_area
from .models import _FlowCandidate, _Replacement, _SourceSpan, _TableCell, _TextFitPlan
from .raster_tables import raster_table_grid
from .report import PdfExportError, PdfExportResult
from .spans import (
    _assign_source_spans,
    _block_rect,
    _hide_occluded_spans,
    _leading_bold_prefix,
    _listing_segments,
    _ownership_text,
    _reflow_flattened_text,
    _source_span_matches_rect,
    _source_span_records,
    _source_text_rects,
)
from .subset import drawable_charset, subset_font_files
from .tables import (
    _drawing_horizontal_segments,
    _table_cell_rects,
    _table_cell_source_style,
    _table_cells,
)
from .text import (
    match_paragraph_shape,
    strip_markdown,
    _TITLE_PREFIX_RE,
    _normalize_inline_spacing,
    _plain_text,
    _protect_trailing_words,
    _restore_title_prefix,
)

# 패키지로 쪼개기 전과 같은 로거 이름을 유지한다(핸들러·필터 설정 호환).
logger = logging.getLogger(__package__)


def _load_pages(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PdfExportError(f"레이아웃 파일을 읽을 수 없습니다: {path.name}") from e
    if not isinstance(data, list):
        raise PdfExportError(f"레이아웃 파일 형식이 올바르지 않습니다: {path.name}")
    return data


def _validate_layout_pair(orig_pages: list[dict], trans_pages: list[dict]) -> None:
    """번역 레이아웃이 원문 레이아웃의 content-only 사본인지 검증한다.

    PDF 내보내기는 두 파일의 같은 인덱스 블록을 서로 대응시킨다. 구조가 어긋난
    파일을 zip()으로 조용히 처리하면 엉뚱한 사각형의 원문을 영구 리댁션할 수
    있으므로, 페이지/블록 수와 블록의 안정 식별자(type, bbox, image)를 먼저
    전부 검증하고 단 하나라도 다르면 문서를 전혀 수정하지 않는다.
    """
    if len(orig_pages) != len(trans_pages):
        raise PdfExportError("원문과 번역 레이아웃의 페이지 수가 일치하지 않습니다")
    seen_pages: set[int] = set()
    for index, (opage, tpage) in enumerate(zip(orig_pages, trans_pages), start=1):
        if not isinstance(opage, dict) or not isinstance(tpage, dict):
            raise PdfExportError(f"레이아웃 {index}페이지 형식이 올바르지 않습니다")
        pno = opage.get("page")
        if not isinstance(pno, int) or pno in seen_pages or tpage.get("page") != pno:
            raise PdfExportError(f"원문과 번역 레이아웃의 {index}페이지 대응이 올바르지 않습니다")
        seen_pages.add(pno)
        oblocks = opage.get("blocks", [])
        tblocks = tpage.get("blocks", [])
        if not isinstance(oblocks, list) or not isinstance(tblocks, list):
            raise PdfExportError(f"레이아웃 {pno}페이지 블록 형식이 올바르지 않습니다")
        if len(oblocks) != len(tblocks):
            raise PdfExportError(f"원문과 번역 레이아웃의 {pno}페이지 블록 수가 일치하지 않습니다")
        for block_no, (ob, tb) in enumerate(zip(oblocks, tblocks), start=1):
            if not isinstance(ob, dict) or not isinstance(tb, dict):
                raise PdfExportError(f"레이아웃 {pno}페이지 {block_no}번 블록 형식이 올바르지 않습니다")
            for key in ("type", "bbox", "image"):
                if ob.get(key) != tb.get(key):
                    raise PdfExportError(
                        f"원문과 번역 레이아웃의 {pno}페이지 {block_no}번 블록이 일치하지 않습니다"
                    )


@dataclass
class _ExportFonts:
    """내보내기 한 번에 쓰는 serif/sans/표 폰트 파일과 PDF resource 이름."""

    serif_ff: str | None
    serif_name: str
    sans_ff: str | None
    sans_name: str
    table_ff: str | None
    table_name: str


def _resolve_export_fonts(fontfile: str) -> _ExportFonts:
    """본문 serif·sans와 표 보조 폰트를 한 번에 해석한다."""
    serif_ff, serif_name = _resolve_font(fontfile)
    sans_ff, sans_name = _resolve_font(
        fontfile, _SYSTEM_SANS_FONT_CANDIDATES, prefer_serif=False,
    )
    if serif_ff:
        serif_name = "uocr-serif"
    if sans_ff:
        sans_name = "uocr-sans"
    # 명시 폰트가 없을 때만 로컬의 조밀한 CJK serif를 표 보조로 허용한다.
    # PDF_EXPORT_FONT가 주어졌다면 표도 같은 파일을 사용해야 배포 환경과 결과가
    # 달라지지 않고, 호출자가 지정한 폰트 계약을 우회하지 않는다.
    compact_ff = None if fontfile else _resolve_font("")[0]
    table_ff = compact_ff or serif_ff
    table_name = "uocr-table" if compact_ff and compact_ff != serif_ff else serif_name
    return _ExportFonts(
        serif_ff, serif_name, sans_ff, sans_name, table_ff, table_name,
    )


def _subset_export_fonts(
    fonts: _ExportFonts, charset: str, out_dir: Path,
) -> _ExportFonts:
    """조판에 쓰는 글리프만 남긴 폰트로 갈아끼운다(실패하면 원본 그대로).

    임베드되는 것은 여기서 정해진 파일이다. 서브셋은 `has_glyph` 동치가 검증된
    경우에만 채택되므로 조판 결과는 바뀌지 않고 파일만 작아진다 — 실측 20.8MB →
    0.95MB. 부수 효과로 조판 시행(`_plan_rich_prefix`)이 폰트를 여는 비용도 함께
    줄어든다: 시행마다 26MB를 파싱하던 것이 180KB가 된다.
    """
    mapping = subset_font_files(
        [fonts.serif_ff, fonts.sans_ff, fonts.table_ff], charset, out_dir,
    )
    if not mapping:
        return fonts
    return replace(
        fonts,
        serif_ff=mapping.get(fonts.serif_ff, fonts.serif_ff),
        sans_ff=mapping.get(fonts.sans_ff, fonts.sans_ff),
        table_ff=mapping.get(fonts.table_ff, fonts.table_ff),
    )


def _reserve_font_resource_names(doc, fonts: _ExportFonts) -> None:
    """원본 page resource와 충돌하지 않는 삽입용 fontname을 예약한다."""
    # PyMuPDF는 fontname을 페이지 resource key로도 사용한다. 원본에 같은 key가
    # 있으면 새 fontfile 대신 기존 글꼴을 재사용할 수 있으므로 문서 전체에서
    # 충돌하지 않는 이름을 먼저 예약한다.
    used_font_names = _document_font_resource_names(doc)
    if fonts.serif_ff:
        fonts.serif_name = _unique_font_resource_name("uocr-serif", used_font_names)
    if fonts.sans_ff:
        fonts.sans_name = (
            fonts.serif_name
            if fonts.sans_ff == fonts.serif_ff
            else _unique_font_resource_name("uocr-sans", used_font_names)
        )
    if fonts.table_ff:
        fonts.table_name = (
            fonts.serif_name
            if fonts.table_ff == fonts.serif_ff
            else _unique_font_resource_name("uocr-table", used_font_names)
        )


def _enrich_source_fonts(src: Path, orig_page_list: list[dict]) -> None:
    """원본 PDF의 실측 폰트 메타를 레이아웃에 메모리 백필한다."""
    # /layout 탭을 먼저 열지 않아도 PDF 내보내기가 원본의 실측 폰트 크기와
    # 세로쓰기 정보를 사용해야 한다. 지연 백필 결과를 메모리에서만 활용하고,
    # 번역본에 없는 메타는 아래에서 원문 블록 값을 폴백으로 읽는다.
    try:
        from ..pdf_fonts import enrich_layout_fonts

        enrich_layout_fonts(src, orig_page_list)
    except Exception:  # noqa: BLE001 — 폰트 메타는 품질 향상용, 내보내기 필수 조건 아님
        logger.warning("PDF 내보내기용 원본 폰트 메타 추출 실패 — 면적 휴리스틱 사용")


@dataclass(frozen=True)
class _PageContext:
    """한 페이지의 계획 단계가 공유하는 읽기 전용 상태."""

    fitz: object
    page: object
    pno: int
    aspect: float
    oblocks: list
    tblocks: list
    block_rects: list
    source_records: list[_SourceSpan]
    source_ownership: dict[int, list[_SourceSpan]]
    unowned_source: list[_SourceSpan]
    ambiguous_blocks: set[int]
    image_regions: list
    fixed_visuals: list
    fonts: _ExportFonts
    # 이번 계획 패스에서 원문이 **지워질 것으로 가정**하는 블록 인덱스.
    # 계획 단계의 장애물 모델과 실제 리댁션이 어긋나면 두 방향으로 모두 깨진다:
    # 지워질 원문을 장애물로 남기면 들어갈 자리가 있는 번역이 "공간 부족"으로
    # 버려지고(실측 no_fit 81건 중 74건), 반대로 남을 원문을 장애물에서 빼면
    # 그 위에 번역이 찍힌다(실측 겹침 30건). _process_page가 수렴할 때까지
    # 이 집합을 줄여 가며 두 모델을 일치시킨다.
    cleared_indices: frozenset = frozenset()
    # fixed_visuals 중 벡터 도형 장애물(flow의 전폭 장식 예외 대상).
    drawing_visuals: list = field(default_factory=list)
    # 원문이 텍스트가 아니라 래스터 픽셀인 블록(스캔 페이지·표 이미지). span이 없으니
    # 남는 동안에는 블록 영역 전체가 장애물이고, 교체되면 그 픽셀을 덮어 지운다.
    raster_blocks: frozenset = frozenset()
    # 이번 패스에서 줄 단위로 확정한 리스팅 블록의 **남는** 원문 span(바뀌지 않은
    # 줄·정렬 실패 줄). 블록은 지워진다고 가정돼도 이 줄들은 남으므로 장애물이다.
    # 패스마다 새 dict다(`_plan_until_consistent`가 replace로 넣는다).
    residual_spans: dict = field(default_factory=dict)
    # 계획 패스 사이에 공유하는 페이지 분석 캐시(표 검색 TextPage, 가로 선분).
    # 패스마다 `replace()`로 새 컨텍스트를 만들어도 같은 dict를 가리킨다 — 리댁션
    # 전의 원본 페이지에서만 유효하므로 `_process_page`가 계획 직후 비운다.
    analysis: dict = field(default_factory=dict)

    def obstacle_spans(self, exclude: "set[int] | frozenset[int]" = frozenset()):
        """이번 패스에서 **남을** 원문 span들 — 계획의 장애물 집합."""
        for owner, spans in self.source_ownership.items():
            if owner in self.cleared_indices or owner in exclude:
                continue
            for span in spans:
                yield span.rect
        for owner, rects in self.residual_spans.items():
            if owner in self.cleared_indices and owner not in exclude:
                yield from rects
        yield from self.raster_obstacles(self.cleared_indices | frozenset(exclude))

    def raster_obstacles(self, skip: "set[int] | frozenset[int]" = frozenset()):
        """남는 래스터 원문 블록의 영역 — 스캔에는 장애물로 셀 span이 없다.

        이게 없으면 스캔 페이지의 보존 블록(수식·표·머리말 픽셀)은 장애물이 아니어서
        번역이 그 위로 자라 겹쳐 찍힌다.
        """
        for owner in self.raster_blocks:
            if owner in skip:
                continue
            rect = self.block_rects[owner]
            if rect is not None:
                yield rect


# 페이지 면적의 이 비율을 넘는 벡터 도형은 "큰 도형"으로 본다. 10%면 A4에서
# 대략 8x8cm — 그보다 큰 상자의 내부를 통째로 막으면 본문 한 단이 사라진다.
_LARGE_DRAWING_FRACTION = 0.10
# 이보다 밝은 단색 채움은 배경으로 간주한다(흰 상자·연회색 코드 배경).
_INVISIBLE_FILL_LEVEL = 0.90


def _drawing_interior_is_open(drawing: dict) -> bool:
    """이 도형의 **안쪽**에 글자를 놓아도 가려지지 않는가.

    채움이 없으면(테두리만) 안쪽은 비어 있고, 채움이 배경색이면 글자를 덮지
    않는다. 둘 다 아니면(그래프의 색 채움 등) 통째로 장애물이다.
    """
    fill = drawing.get("fill")
    if fill is None:
        return True
    try:
        return min(float(c) for c in fill) >= _INVISIBLE_FILL_LEVEL
    except (TypeError, ValueError):
        return False


def _rect_edge_bands(fitz, rect, stroke_width: float) -> list:
    """사각형의 네 변만 얇은 띠로 — 테두리는 피하되 내부는 쓸 수 있게."""
    band = max(1.0, float(stroke_width or 1.0)) + 0.5
    return [
        fitz.Rect(rect.x0, rect.y0, rect.x1, min(rect.y1, rect.y0 + band)),
        fitz.Rect(rect.x0, max(rect.y0, rect.y1 - band), rect.x1, rect.y1),
        fitz.Rect(rect.x0, rect.y0, min(rect.x1, rect.x0 + band), rect.y1),
        fitz.Rect(max(rect.x0, rect.x1 - band), rect.y0, rect.x1, rect.y1),
    ]


@dataclass(frozen=True)
class _PageVisuals:
    """리댁션 전에 1회만 수집한 페이지 시각 요소."""

    raster_rects: list
    image_regions: list
    fixed_visuals: list
    # fixed_visuals 중 벡터 도형에서 온 것 — flow가 '피할 수 없는 전폭 장식'으로
    # 뺄 수 있는 것은 이것뿐이다(원문 span·그림은 절대 빼지 않는다).
    drawing_rects: list
    # 표 rule 보정이 쓰는 가로 선분 — 같은 get_drawings() 결과에서 뽑는다.
    # None이면 수집 실패(표 쪽이 직접 다시 읽는다).
    horizontal_segments: list | None
    # 페이지 대부분을 덮는 래스터 — 스캔 배경. image_regions에서는 빠지지만 그 위
    # 블록의 원문은 이 픽셀이다.
    scan_rasters: list = field(default_factory=list)
    # get_image_info() 원본(상자·소프트 마스크 여부) — 이미지에 가려진 span 판정이 쓴다.
    image_infos: list = field(default_factory=list)


def _page_image_infos(page) -> list:
    """페이지의 래스터 인스턴스(get_image_info). 리댁션 '이전'에 1회만 읽는다 —
    apply_redactions 이후의 get_image_info()는 스테일 캐시를 반환할 수 있다(실측)."""
    try:
        return list(page.get_image_info())
    except Exception:  # noqa: BLE001 — 이미지 목록 실패가 텍스트 교체를 막지 않는다
        return []


# 여러 장으로 나뉜 스캔(가로 띠·타일)을 한 장으로 본다 — 이 거리 안에서 맞닿거나 겹치는
# 래스터는 한 덩어리다.
_SCAN_TILE_GAP_PT = 2.0
# 덩어리가 자기 bbox를 이만큼 채워야 한 장의 스캔이다(띠 사이 틈·떨어져 놓인 그림 제외).
_SCAN_FILL_RATIO = 0.9
# 여백을 두고 놓인 스캔(이미지→PDF 변환기의 인쇄 여백 — 30pt 여백이면 A4의 84%)은 85%
# 규칙에 걸리지 않는다. 페이지의 이만큼 이상을 채운 덩어리가 레이아웃 그림 블록이 아니고
# 그 안에 보이는 텍스트가 하나도 없으면 스캔 배경이다.
_SCAN_MARGIN_FRACTION = 0.5
# 묶기는 래스터 쌍을 모두 비교한다 — 이보다 많은 이미지 인스턴스(글자마다 이미지를 쓰는
# 생성기 등)가 있는 쪽은 묶지 않고 예전 규칙(한 장이 85% 이상)만 쓴다.
_SCAN_CLUSTER_MAX = 400


def _scan_backgrounds(fitz, page, raster_rects, block_rects, oblocks, source_records) -> list:
    """스캔 배경으로 볼 래스터 덩어리 — `(덩어리 bbox, 구성 래스터 인덱스들)` 목록.

    예전에는 래스터 **한 장**이 페이지의 85% 이상일 때만 스캔으로 봤다. 페이지 이미지를
    가로 띠 여러 장으로 나눠 넣은 스캔과 여백을 두고 놓인 스캔은 그 규칙을 피해 그림으로
    분류됐고, 모든 블록이 '그림 위 텍스트'로 보존돼 번역이 하나도 들어가지 않았다(감사 pdf-9).
    맞닿은 래스터를 한 덩어리로 묶어 (a) 덩어리가 페이지의 85% 이상을 채우거나 (b) 50%
    이상을 채우면서 레이아웃 그림이 아니고 안에 보이는 글자가 없으면 스캔 배경이다.
    """
    count = len(raster_rects)
    if not count:
        return []
    page_area = page.rect.width * page.rect.height or 1.0
    if count > _SCAN_CLUSTER_MAX:
        return [
            (+rect, [index]) for index, rect in enumerate(raster_rects)
            if rect.width * rect.height >= page_area * _SCAN_RASTER_FRACTION
        ]
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(count):
        grown = +raster_rects[first]
        grown += (-_SCAN_TILE_GAP_PT, -_SCAN_TILE_GAP_PT, _SCAN_TILE_GAP_PT, _SCAN_TILE_GAP_PT)
        for second in range(first + 1, count):
            if grown.intersects(raster_rects[second]):
                parent[find(first)] = find(second)
    clusters: dict[int, list[int]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(index)
    figures = [
        rect for rect, block in zip(block_rects, oblocks)
        if rect is not None
        and isinstance(block, dict)
        and (str(block.get("type") or "") == "image" or block.get("image"))
    ]
    out = []
    for members in clusters.values():
        bounds = +raster_rects[members[0]]
        for index in members[1:]:
            bounds.include_rect(raster_rects[index])
        area = bounds.width * bounds.height
        if area <= 0:
            continue
        filled = sum(raster_rects[i].width * raster_rects[i].height for i in members)
        if min(1.0, filled / area) < _SCAN_FILL_RATIO:
            continue
        if area >= page_area * _SCAN_RASTER_FRACTION:
            out.append((bounds, members))
            continue
        if area < page_area * _SCAN_MARGIN_FRACTION:
            continue
        if sum(_rect_overlap_area(bounds, figure) for figure in figures) >= area * 0.5:
            continue                          # 레이아웃이 그림으로 본 큰 이미지
        if any(
            span.visible and bounds.contains((span.rect.tl + span.rect.br) / 2)
            for span in source_records
        ):
            continue                          # 이미지 위·옆에 보이는 글자가 있는 born-digital 쪽
        out.append((bounds, members))
    return out


def _page_visual_obstacles(
    fitz, page, block_rects, oblocks, *, image_infos: list | None = None,
    source_records: list | None = None,
) -> _PageVisuals:
    """래스터 인스턴스·그림 영역·확장 장애물·가로 선분. 리댁션 전에 1회만 수집한다."""
    if image_infos is None:
        image_infos = _page_image_infos(page)
    raster_rects = [fitz.Rect(info["bbox"]) for info in image_infos]
    # 벡터 표·그래프·구분선도 번역문 확장 영역의 장애물이다. path의 rect가
    # 수평/수직 0폭 선이면 먼저 1pt 패딩해 유효한 사각형으로 만든다.
    #
    # 다만 **큰 도형의 bbox를 통째로** 장애물로 쓰면 안 된다. 흰 배경 채움이나
    # 코드 상자 테두리는 안쪽에 글자를 얼마든지 놓을 수 있는데도 페이지 절반을
    # 막아 버려, 들어갈 자리가 있는 번역이 "공간 부족"으로 버려진다. 큰 도형 중
    # 안쪽이 비었거나(테두리만) 배경색으로 칠해진 것은 **테두리 띠만** 남긴다.
    page_area_all = page.rect.width * page.rect.height or 1.0
    drawing_rects = []
    horizontal_segments: list | None = []
    try:
        for drawing in page.get_drawings():
            # 표 rule 보정용 가로 선분도 같은 순회에서 뽑는다 — 표 블록 × 계획 패스마다
            # 페이지 벡터 목록 전체를 다시 파싱하지 않게.
            horizontal_segments.extend(_drawing_horizontal_segments(drawing))
            bbox = drawing.get("rect")
            if bbox is None:
                continue
            drawing_rect = fitz.Rect(bbox)
            drawing_rect += (-0.5, -0.5, 0.5, 0.5)
            drawing_rect &= page.mediabox
            if drawing_rect.is_empty:
                continue
            area_fraction = (drawing_rect.width * drawing_rect.height) / page_area_all
            if area_fraction >= _LARGE_DRAWING_FRACTION and _drawing_interior_is_open(
                drawing,
            ):
                drawing_rects.extend(
                    _rect_edge_bands(fitz, drawing_rect, drawing.get("width") or 1.0)
                )
                continue
            drawing_rects.append(drawing_rect)
    except Exception:  # noqa: BLE001 — 벡터 목록 실패가 텍스트 교체를 막지 않는다
        drawing_rects = []
        horizontal_segments = None
    # 그림 위 텍스트 방어용 영역: layout image 블록 ∪ 래스터 인스턴스.
    # 스캔 배경(페이지의 85% 이상을 덮는 래스터 — 띠로 나뉜 스캔이면 그 덩어리, 여백 있는
    # 스캔도 포함: _scan_backgrounds)은 제외한다 — 스캔 문서에서 모든 블록 교체가 생략되는
    # 사고 방지. 대신 그 덩어리는 scan_rasters로 따로 넘겨, 그 위 블록을 교체할 때 원문
    # 픽셀을 덮게 한다.
    page_area = page.rect.width * page.rect.height or 1.0
    backgrounds = _scan_backgrounds(
        fitz, page, raster_rects, block_rects, oblocks, source_records or [],
    )
    scan_members = {index for _bounds, members in backgrounds for index in members}
    image_regions = [
        r
        for r, b in zip(block_rects, oblocks)
        if (
            r is not None
            and isinstance(b, dict)
            and (str(b.get("type") or "") == "image" or b.get("image"))
            and r.width * r.height < page_area * 0.85
        )
    ]
    image_regions += [
        r for index, r in enumerate(raster_rects)
        if index not in scan_members and 0 < r.width * r.height < page_area * 0.85
    ]
    fixed_visuals = image_regions + drawing_rects
    scan_rasters = [bounds for bounds, _members in backgrounds]
    return _PageVisuals(
        raster_rects, image_regions, fixed_visuals, drawing_rects, horizontal_segments,
        scan_rasters, image_infos,
    )


# image_regions의 85% 규칙과 같은 기준 — 이 비율 이상을 덮는 래스터가 스캔 배경이다.
_SCAN_RASTER_FRACTION = 0.85
# 래스터를 원문으로 보는 블록의 겹침 기준. 전면 스캔은 블록 면적의 절반 이상,
# 표 이미지는 표 블록이 한 래스터 안에 거의(80%) 들어 있어야 한다.
_SCAN_BLOCK_OVERLAP = 0.5
_RASTER_TABLE_OVERLAP = 0.8
# 레이아웃 그림 블록과 이만큼 겹치면 그림 속 글자다 — figure_text와 같은 30%.
_FIGURE_OVERLAP = 0.30


# 스캔에는 줄 방향(dir)을 알려 줄 텍스트 레이어가 없다. 레이아웃 뷰(layout.py)와
# 같은 기하 폴백 — 극단적으로 좁고 긴 블록은 세로쓰기(여백의 arXiv 식별자 등)다.
_VERTICAL_ASPECT = 6.0
_VERTICAL_MIN_CHARS = 12
# 세로쓰기는 **한 줄**이다. 세로 한 줄의 블록 폭은 줄 높이(≈글자 크기)라 글자 하나가
# 블록 폭의 약 0.45배를 차지하고, 글자 수 × 그 폭이 블록 높이와 맞아야 한다(여유 1.5배).
# 좁은 단에 가로 본문 여러 줄이 든 블록(신문·잡지 단)은 글자가 그 수십 배다 — 예전에는
# 종횡비만 봐서 74×574pt 단(57줄)을 세로쓰기로 보존해 번역이 통째로 빠졌다(감사 pdf-4).
_VERTICAL_ADVANCE_PER_WIDTH = 0.45
_VERTICAL_LENGTH_SLACK = 1.5


def _raster_block_looks_vertical(ctx: "_PageContext", block_index: int, block) -> bool:
    """래스터 원문 블록이 화면에서 세로로 놓인 글인가(덮고 가로로 다시 쓰면 안 된다).

    텍스트 레이어가 있으면 폰트 백필이 줄 방향으로 `vertical`을 심지만, 스캔은
    그럴 수 없어 세로 스탬프를 덮은 뒤 한국어를 한 글자씩 세로로 쌓아 찍었다.
    좁고 길어도 여러 줄 가로 본문(내용에 줄바꿈이 있거나 글자가 세로 한 줄에 들어갈
    양보다 훨씬 많은 블록)은 세로쓰기가 아니다.
    """
    if block_index not in ctx.raster_blocks or not isinstance(block, dict):
        return False
    rect = ctx.block_rects[block_index]
    if rect is None:
        return False
    shown = rect * ctx.page.rotation_matrix
    shown.normalize()
    if shown.width <= 0:
        return False
    content = str(block.get("content") or "").strip()
    return (
        shown.height / shown.width >= _VERTICAL_ASPECT
        and len(content) >= _VERTICAL_MIN_CHARS
        and "\n" not in content
        and len(content) * shown.width * _VERTICAL_ADVANCE_PER_WIDTH
        <= shown.height * _VERTICAL_LENGTH_SLACK
    )


def _raster_backed_blocks(block_rects, oblocks, source_records, visuals) -> frozenset:
    """원문이 텍스트가 아니라 래스터 픽셀인 블록 인덱스.

    스캔 페이지(또는 투명 OCR 텍스트만 얹힌 스캔)에서는 텍스트 리댁션으로 지울
    원문이 없어, 번역이 영어 스캔 글자 위에 그대로 겹쳐 찍혔다. 그런 블록은 교체할
    때 그 영역을 바탕색으로 덮는다. born-digital 그림을 건드리지 않도록 좁게 고른다:
    보이는 원문 span의 중심이 블록 안에 하나도 없고, (a) 전면 스캔 래스터와 블록
    면적 절반 이상이 겹치거나 (b) 표 블록이 한 래스터 안에 들어 있어야 하며(표
    이미지 — 표는 셀 단위로 번역하는 대상이다), 레이아웃 그림 블록 안의 글자는 뺀다.
    """
    if not visuals.raster_rects:
        return frozenset()
    figures = [
        rect for rect, block in zip(block_rects, oblocks)
        if rect is not None
        and isinstance(block, dict)
        and (str(block.get("type") or "") == "image" or block.get("image"))
    ]
    visible_centers = [
        (span.rect.tl + span.rect.br) / 2 for span in source_records if span.visible
    ]
    out: set[int] = set()
    for index, (rect, block) in enumerate(zip(block_rects, oblocks)):
        if rect is None or not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "image" or block.get("image"):
            continue
        area = rect.width * rect.height
        if area <= 0:
            continue
        on_scan = any(
            _rect_overlap_area(rect, raster) / area >= _SCAN_BLOCK_OVERLAP
            for raster in visuals.scan_rasters
        )
        table_image = block_type == "table" and any(
            _rect_overlap_area(rect, raster) / area >= _RASTER_TABLE_OVERLAP
            for raster in visuals.raster_rects
        )
        if not (on_scan or table_image):
            continue
        if any(_rect_overlap_area(rect, figure) / area >= _FIGURE_OVERLAP for figure in figures):
            continue
        if any(rect.contains(center) for center in visible_centers):
            continue
        out.add(index)
    return frozenset(out)


def _align_to_baseline(page, plan: _TextFitPlan, baseline, cell_rect, avoid) -> _TextFitPlan:
    """한 줄 계획을 원문 줄의 baseline(표시 공간 y)으로 옮긴다.

    셀 조판은 셀 위쪽에서 시작하므로, 같은 행에 남는 영어(스캔 픽셀)보다 번역이 떠
    보였다. 옮긴 글자 상자가 셀 안에 들고 남는 원문·괘선에 닿지 않을 때만 옮긴다.
    """
    if plan.origin is None or plan.ink_rect is None or baseline is None:
        return plan
    to_display, to_page = page.rotation_matrix, page.derotation_matrix
    ox, oy = plan.origin
    shown_x = ox * to_display.a + oy * to_display.c + to_display.e
    shown_y = ox * to_display.b + oy * to_display.d + to_display.f
    delta = float(baseline) - shown_y
    if abs(delta) < 0.25:
        return plan
    ink = plan.ink_rect * to_display
    ink.normalize()
    cell = cell_rect * to_display
    cell.normalize()
    if ink.y0 + delta < cell.y0 - 0.01 or ink.y1 + delta > cell.y1 + 0.01:
        return plan
    moved = quiet_fitz().Rect(ink.x0, ink.y0 + delta, ink.x1, ink.y1 + delta) * to_page
    moved.normalize()
    if _ink_collides(moved, avoid):
        return plan
    new_y = shown_y + delta
    origin = (
        float(shown_x * to_page.a + new_y * to_page.c + to_page.e),
        float(shown_x * to_page.b + new_y * to_page.d + to_page.f),
    )
    return replace(plan, origin=origin, first_origin=origin, ink_rect=moved)


def _plan_table_block(
    ctx: _PageContext, block_index: int, ob: dict, tb: dict,
    result: PdfExportResult,
) -> list[_Replacement]:
    """표 블록의 셀별 교체 계획. 한 셀이라도 실패하면 표 전체를 보존한다."""
    old_parsed = _table_cells(str(ob.get("content") or ""))
    new_parsed = _table_cells(str(tb.get("content") or ""))
    table_rect = ctx.block_rects[block_index]
    structure_matches = bool(
        old_parsed is not None
        and new_parsed is not None
        and table_rect is not None
        and old_parsed[1:] == new_parsed[1:]
        and len(old_parsed[0]) == len(new_parsed[0])
        and len(old_parsed[0]) <= _MAX_TABLE_CELLS
        and all(
            (old_cell.row, old_cell.col, old_cell.rowspan, old_cell.colspan)
            == (new_cell.row, new_cell.col, new_cell.rowspan, new_cell.colspan)
            for old_cell, new_cell in zip(old_parsed[0], new_parsed[0])
        )
    )
    if not structure_matches:
        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
        if str(ob.get("content") or "") != str(tb.get("content") or ""):
            result.warnings.append(
                f"p{ctx.pno}: 표 셀 구조 불일치 — 원문 표 보존"
            )
        return []
    old_cells, row_count, col_count = old_parsed
    new_cells = new_parsed[0]
    changed = [
        index
        for index, (old_cell, new_cell) in enumerate(zip(old_cells, new_cells))
        if (
            (new_text := _plain_text(new_cell.text))
            and new_text != (old_text := _plain_text(old_cell.text))
            and new_text.casefold() != old_text.casefold()
        )
    ]
    raster_grid = None
    if block_index in ctx.raster_blocks:
        if not changed:
            return []
        # 스캔 표는 바뀐 셀을 바탕색으로 **덮는다** — 원문 검색이 비어 균등 분할로
        # 떨어진 격자로 덮으면 이웃 셀 글자와 괘선까지 지워진다(감사 pdf-1). 픽셀의 빈
        # 띠로 실제 열·행 경계를 찾고, 확정하지 못하면 표를 덮지 않고 보존한다.
        raster_grid = raster_table_grid(
            ctx.page, table_rect, old_cells, row_count, col_count,
            cache=ctx.analysis, key=block_index,
        )
        if raster_grid is None:
            result.keep("table_grid_untrusted")
            result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
            result.warnings.append(
                f"p{ctx.pno}: 스캔 표의 열·행 경계를 픽셀에서 확정하지 못함 — 원문 표 보존"
            )
            return []
        cell_rects = list(raster_grid.cell_rects)
    else:
        cell_rects, grid_trusted = _table_cell_rects(
            ctx.page, table_rect, old_cells, row_count, col_count, cache=ctx.analysis,
        )
        if not grid_trusted:
            result.keep("table_grid_untrusted")
            result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
            result.warnings.append(
                f"p{ctx.pno}: 표 셀 격자 추정 실패(원문 검색 불일치) — 원문 표 보존"
            )
            return []
    table_targets: list[_Replacement] = []
    changed_cell_specs = [
        (index, old_cells[index], new_cells[index], cell_rects[index]) for index in changed
    ]
    changed_cells = len(changed_cell_specs)
    # 이 표에서 **지워지지 않는** 원문 span은 셀 조판의 장애물이다. 격자 추정은
    # 셀 경계를 대략만 맞추므로, 장애물 없이 조판하면 번역 셀이 옆 칸에 남은
    # 원문 글리프에 닿는다(실측 p3: 번역 "실세계"가 앞 칸 끝의 ")"와 겹침).
    cell_zone = [rect for _i, _o, _n, rect in changed_cell_specs]
    table_avoid = [
        span.rect
        for span in ctx.source_records
        if table_rect is not None
        and not (span.rect & table_rect).is_empty
        and not any(
            _rect_overlap_area(span.rect, rect)
            >= max(0.01, span.rect.width * span.rect.height) * 0.5
            for rect in cell_zone
        )
    ]
    if raster_grid is not None:
        # 스캔 표에는 span이 없다 — 남는 원문은 바뀌지 않은 셀의 글자 픽셀과 괘선이다.
        changed_set = set(changed)
        table_avoid.extend(raster_grid.rule_rects)
        table_avoid.extend(
            ink for index, ink in enumerate(raster_grid.ink_rects)
            if ink is not None and index not in changed_set
        )
    failed_cell: _TableCell | None = None
    for cell_index, old_cell, new_cell, cell_rect in changed_cell_specs:
        new_text = _portable_text_for_font(
            _plain_text(new_cell.text), ctx.fonts.table_ff,
        )
        base_pt, cell_align, cell_bold, source_redact = (
            _table_cell_source_style(ctx.page, cell_rect, ctx.source_records)
        )
        # 덮개 — 원문이 픽셀인 셀은 셀 사각형이 아니라 그 셀 글자 잉크만 덮는다.
        cover = cell_rect
        if raster_grid is not None:
            cover = raster_grid.ink_rects[cell_index]
            cell_align = raster_grid.aligns[cell_index]
            if raster_grid.font_pt:
                base_pt = min(12.0, max(_MIN_FONT_PT, raster_grid.font_pt * 1.03))
        plan = _plan_single_line(
            ctx.page,
            cell_rect,
            new_text,
            base_pt,
            ctx.fonts.table_name,
            ctx.fonts.table_ff,
            max_rect=cell_rect,
            align=cell_align,
            bold=cell_bold,
            avoid_rects=table_avoid,
        )
        if plan is None:
            plan = _plan_shrink_to_fit(
                ctx.page,
                cell_rect,
                new_text,
                base_pt,
                ctx.fonts.table_name,
                ctx.fonts.table_ff,
                max_rect=cell_rect,
                align=cell_align,
                bold=cell_bold,
                lineheights=_CAPTION_LINEHEIGHTS,
                avoid_rects=table_avoid,
            )
        source_size = base_pt / 1.03
        readable_floor = max(
            _MIN_TABLE_FONT_PT,
            min(source_size, source_size * 0.80),
        )
        if plan is None or plan.fontsize + 0.01 < readable_floor:
            failed_cell = old_cell
            break
        if raster_grid is not None and raster_grid.baselines:
            plan = _align_to_baseline(
                ctx.page, plan, raster_grid.baselines[cell_index], cell_rect, table_avoid,
            )
        table_targets.append(_Replacement(
            plan,
            new_text,
            "table",
            source_redact,
            ctx.fonts.table_name,
            ctx.fonts.table_ff,
            cover,
            block_index,
        ))
    if failed_cell is not None:
        result.keep("table_cell_no_fit", changed_cells)
        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
        result.warnings.append(
            f"p{ctx.pno}: 표 {failed_cell.row + 1}행 {failed_cell.col + 1}열 "
            "번역 생략(공간/가독성 부족) — 표 전체 원문 보존"
        )
        return []
    return table_targets


def _plan_text_block(
    ctx: _PageContext, block_index: int, block_type: str, ob: dict, tb: dict,
    targets: list[_Replacement], result: PdfExportResult,
) -> _FlowCandidate | None:
    """일반 텍스트 블록의 flow 후보. 줄 단위로 확정되면 targets에 직접 넣는다."""
    source_raw = str(ob.get("content") or "")
    old = _plain_text(source_raw)
    # 번역문의 마크다운 표기를 걷어내고, 원문이 한 문단이면 문단 구조도 맞춘다 —
    # 둘 다 지면에 그대로 찍히거나(마커) 높이를 부풀려 블록을 통째로 버리게 한다.
    # 원문을 함께 넘겨 원문에 실제로 있던 목록·`>`·`*`/`__` 표기는 지우지 않는다.
    new = _plain_text(
        strip_markdown(
            match_paragraph_shape(source_raw, str(tb.get("content") or "")),
            source_raw,
        )
    )
    if block_type == "title":
        new = _restore_title_prefix(old, new)
    if not new or new == old:
        result.keep("unchanged")
        return None
    rect = ctx.block_rects[block_index]
    if rect is None:
        result.keep("no_rect")
        return None
    # 그림 패널·로고 위 OCR 텍스트 블록은 교체하지 않는다. 그림 속
    # 텍스트는 OCR 오독이 잦고 원본 조판이 항상 우월하며, 번역을
    # 스탬프하면 원문 그림과 이중으로 겹쳐 보인다. 임계값 0.30은
    # 실측 분포(문제 블록 53.7~93% vs 정상 블록 ≤2%)의 빈 구간 안.
    rect_area = rect.width * rect.height
    if rect_area > 0 and any(
        _rect_overlap_area(rect, region) / rect_area >= 0.30
        for region in ctx.image_regions
    ):
        result.keep("figure_text")
        result.specialist_kept["figure_text"] = (
            result.specialist_kept.get("figure_text", 0) + 1
        )
        result.warnings.append(f"p{ctx.pno}: 그림 위 텍스트 — 원문 보존")
        return None
    fs_cqw = ob.get("fs") or tb.get("fs") or estimate_font_size_cqw(
        tb.get("bbox"), str(tb.get("content") or ""), ctx.aspect,
    ) or 1.8
    base_pt = min(_MAX_FONT_PT, max(
        _MIN_FONT_PT, fs_cqw / 100 * ctx.page.rect.width))
    # 같은 pt에서 AppleMyungjo/Noto Serif CJK는 Times 계열 영문보다
    # 시각적 몸통이 조금 작다. 제목은 계층을 잃지 않도록 더 보정하고,
    # 본문은 3%만 보정해 원문과 비슷한 잉크 밀도를 유지한다.
    base_pt *= 1.06 if block_type == "title" else 1.03
    if ob.get("font_style") == "sans":
        block_fontfile, block_fontname = ctx.fonts.sans_ff, ctx.fonts.sans_name
    else:
        block_fontfile, block_fontname = ctx.fonts.serif_ff, ctx.fonts.serif_name
    new = _normalize_inline_spacing(new)
    new = _portable_text_for_font(new, block_fontfile)
    if block_type != "title":
        new = _protect_trailing_words(new)
    if block_type == "title":
        # 줄 폭은 화면 폭이다 — 회전 페이지에서 비회전 폭은 화면의 높이다.
        shown = rect * ctx.page.rotation_matrix
        shown.normalize()
        new = _balance_title_text(
            new,
            shown.width,
            base_pt,
            block_fontname,
            block_fontfile,
        )
    if block_type == "title":
        lineheights = _TITLE_LINEHEIGHTS
    elif block_type in {
        "caption", "image_caption", "table_caption",
        "page_footnote", "footnote", "aside_text",
    }:
        lineheights = _CAPTION_LINEHEIGHTS
    else:
        lineheights = _BODY_LINEHEIGHTS
    align_value = str(ob.get("align") or "")
    if (
        not align_value
        and block_type == "title"
        and _TITLE_PREFIX_RE.match(old) is None
        and base_pt >= 15.0
    ):
        # 논문 표제처럼 큰 무번호 제목만 OCR의 누락된 center 정렬을
        # 복원한다. 절/부록 제목은 원 논문 관례대로 왼쪽 정렬한다.
        align_value = "center"
    align = {"center": 1, "right": 2, "justify": 3}.get(align_value, 0)
    bold = bool(ob.get("bold")) or block_type == "title"
    owned_records = ctx.source_ownership.get(block_index, [])
    local_source_records = [
        span for span in ctx.source_records
        if _source_span_matches_rect(span, rect)
    ]
    if block_index in ctx.ambiguous_blocks or (
        not owned_records and local_source_records
    ):
        result.keep("ambiguous_source")
        result.warnings.append(
            f"p{ctx.pno}: 블록 {block_index + 1} 교체 생략"
            "(안전한 원문 span 없음) — 원문 보존"
        )
        return None
    # 원문 PDF의 실제 baseline 수보다 OCR 줄 수가 많으면 그 줄바꿈은
    # 문단의 줄바꿈이 아니라 가로 배치(표 헤더·행)의 평탄화다.
    # bbox 높이는 원문 줄 수만큼뿐이라 축소로는 절대 들어가지 않는다.
    reflow_text = _reflow_flattened_text(
        old, new, owned_records, base_pt,
    )
    # 리스팅·표는 줄 구조와 열 위치 자체가 의미다. 원문 좌표에 줄별로
    # 그대로 조판할 수 있으면 흘려 넣기(리플로우)보다 항상 낫다.
    listing_segments = (
        _listing_segments(old, new, owned_records, base_pt, rect.x1)
        if reflow_text is not None
        else ()
    )
    listing_dropped = _listing_dropped_lines(old, new, listing_segments)
    if listing_segments:
        listing_avoid = [span.rect for span in ctx.unowned_source]
        listing_avoid.extend(
            span.rect
            for owner, spans in ctx.source_ownership.items()
            if owner != block_index
            for span in spans
        )
        listing_avoid.extend(ctx.fixed_visuals)
        listing_avoid.extend(ctx.raster_obstacles(skip={block_index}))
        listing_avoid.extend(
            target.plan.ink_rect
            for target in targets
            if target.plan.ink_rect is not None
        )
        listing_targets, listing_changed = _plan_listing_lines(
            ctx.page,
            listing_segments,
            block_fontname,
            block_fontfile,
            listing_avoid,
            block_index,
            bold=bold,
        )
        # 모든 줄을 제자리에 넣을 수 있을 때만 여기서 확정한다. 일부만
        # 되면 흘려 넣기가 더 많이 회수할 수 있으므로 flow에 맡기고,
        # 그마저 실패하면 아래 개별 배치에서 줄 단위로 부분 회수한다.
        if listing_changed and len(listing_targets) == listing_changed:
            targets.extend(listing_targets)
            _keep_dropped_listing_lines(ctx, block_index, listing_dropped, result)
            # 바뀌지 않은 줄·정렬 실패 줄은 지워지지 않는다 — 같은 패스의 flow가
            # 바로 장애물로 보게 기록한다(다음 패스까지 미루면 계획을 한 번 더 세운다).
            ctx.residual_spans[block_index] = _unplaced_listing_spans(
                ctx, block_index, listing_targets,
            )
            return None
    return _FlowCandidate(
        block_index,
        block_type,
        new,
        rect,
        base_pt,
        block_fontname,
        block_fontfile,
        align,
        bold,
        lineheights,
        # span bbox가 아니라 baseline 띠로 지운다 — 일반 행간에서 이웃 보존 줄을 지키려고.
        _source_text_rects(ctx.page, rect, owned_records),
        rect,
        None if bold else _leading_bold_prefix(owned_records, new),
        reflow_text,
        listing_segments,
        listing_dropped,
    )


def _listing_dropped_lines(old: str, new: str, segments) -> int:
    """번역이 바뀐 OCR 줄 중 세그먼트가 되지 못한(원문 시각 줄에 정렬 실패) 줄 수."""
    if not segments:
        return 0
    old_lines, new_lines = old.splitlines(), new.splitlines()
    if len(old_lines) != len(new_lines):
        return 0
    changed = sum(
        1 for before, after in zip(old_lines, new_lines)
        if after.strip() and after.strip() != before.strip()
    )
    covered = sum(1 for segment in segments if segment.text and segment.text != segment.original)
    return max(0, changed - covered)


def _keep_dropped_listing_lines(
    ctx: "_PageContext", block_index: int, dropped: int, result: PdfExportResult,
) -> None:
    """줄 단위 조판에서 빠진 줄(원문이 그대로 남는다)을 보존 사유로 남긴다."""
    if dropped <= 0:
        return
    result.keep("listing_line_unaligned", dropped)
    result.warnings.append(
        f"p{ctx.pno}: 블록 {block_index + 1}의 {dropped}줄 교체 생략"
        "(원문 줄 위치 정렬 실패) — 그 줄만 원문 보존"
    )


def _unplaced_listing_spans(ctx: "_PageContext", block_index: int, placed) -> list:
    """부분 배치된 리스팅 블록에서 리댁션되지 않고 **남는** 원문 span 사각형."""
    regions = [target.source_rect for target in placed if target.source_rect is not None]
    out = []
    for span in ctx.source_ownership.get(block_index, []):
        area = max(0.01, span.rect.width * span.rect.height)
        if not any(_rect_overlap_area(span.rect, region) >= area * 0.9 for region in regions):
            out.append(span.rect)
    return out


def _partially_redacted_blocks(ctx: "_PageContext", targets) -> set[int]:
    """원문 span 일부만 지워지는 교체 블록(줄 단위 부분 배치).

    계획이 '교체됨'으로 세면 그 블록의 남는 원문 줄이 다른 flow의 장애물에서
    빠진다 — 다음 패스에서 지워진다고 가정하지 않도록 돌려준다.
    """
    listing: dict[int, list] = {}
    for target in targets:
        if (
            target.kind == "listing"
            and target.block_index >= 0
            # 같은 패스에서 이미 남는 줄을 장애물로 세운 블록은 일관적이다.
            and target.block_index not in ctx.residual_spans
        ):
            listing.setdefault(target.block_index, []).append(target)
    return {
        index for index, placed in listing.items()
        if _unplaced_listing_spans(ctx, index, placed)
    }


def _plan_page_targets(ctx: _PageContext, result: PdfExportResult):
    """페이지의 모든 블록에서 (확정 계획, flow 후보, 링크 정상화 대상)을 모은다."""
    targets: list[_Replacement] = []
    flow_candidates: list[_FlowCandidate] = []
    repeated_scheme_link_rects: list[object] = []
    for block_index, (ob, tb) in enumerate(zip(ctx.oblocks, ctx.tblocks)):
        if not isinstance(ob, dict) or not isinstance(tb, dict):
            continue
        block_type = str(tb.get("type") or "")
        # 표는 HTML을 통째로 평문 삽입하지 않고 셀 구조가 원문과 정확히
        # 대응할 때만 셀별로 교체한다. 벡터 선은 redaction 옵션으로 보존된다.
        if block_type == "table":
            targets.extend(_plan_table_block(ctx, block_index, ob, tb, result))
            continue

        if block_type in _PRESERVE_TYPES:
            result.keep(f"preserve_type:{block_type}")
            preserve_kind = "reference" if block_type == "ref_text" else "running_text"
            result.specialist_kept[preserve_kind] = (
                result.specialist_kept.get(preserve_kind, 0) + 1
            )
            if block_type == "ref_text":
                microfixes = _preserved_reference_microfixes(
                    ctx.fitz,
                    ctx.source_ownership.get(block_index, []),
                    ctx.fonts.serif_name,
                    ctx.fonts.serif_ff,
                )
                targets.extend(microfixes)
                repeated_scheme_link_rects.extend(
                    target.source_rect
                    for target in microfixes
                    if target.text in {"http://", "https://"}
                    and target.source_rect is not None
                )
            continue
        # 번역 단계가 "번역하지 않기로 결정"한 블록(코드·CLI 트랜스크립트·식별자
        # 나열). 원문과 번역이 같아 아래 `unchanged`로 떨어지면 번역 결함과
        # 구분되지 않는다 — 의도적 보존으로 따로 집계한다.
        preserved = str(tb.get("preserved") or "")
        if preserved:
            result.keep(f"preserved:{preserved}")
            result.specialist_kept[preserved] = (
                result.specialist_kept.get(preserved, 0) + 1
            )
            continue
        if block_type not in _REPLACEABLE_TYPES:
            if block_type in _SPECIALIST_TYPES:
                result.specialist_kept[block_type] = (
                    result.specialist_kept.get(block_type, 0) + 1
                )
            continue
        declared_vertical = (tb.get("vertical") or ob.get("vertical")) in _VERTICAL_SKIP
        if declared_vertical or _raster_block_looks_vertical(ctx, block_index, ob):
            result.keep("vertical")
            result.specialist_kept["vertical"] = result.specialist_kept.get("vertical", 0) + 1
            if not declared_vertical and (
                str(tb.get("content") or "").strip() != str(ob.get("content") or "").strip()
            ):
                # 스캔의 세로쓰기는 줄 방향 정보 없이 모양으로만 추정한다 — 번역이 빠진
                # 이유를 리포트에서 알 수 있게 남긴다.
                result.warnings.append(
                    f"p{ctx.pno}: 블록 {block_index + 1}을 세로쓰기로 보고 원문 보존"
                    "(스캔 — 좁고 긴 한 줄 모양으로 추정)"
                )
            continue
        candidate = _plan_text_block(
            ctx, block_index, block_type, ob, tb, targets, result,
        )
        if candidate is not None:
            flow_candidates.append(candidate)
    return targets, flow_candidates, repeated_scheme_link_rects


class _DisplaySpace:
    """회전 페이지의 flow 계획을 표시(화면) 공간에서 한다.

    flow 계획(`_flow_components`·`_plan_flow_group`)은 '아래로 자란다'·'같은 단'을
    비회전 y·x축으로 계산한다. /Rotate 90·270 페이지에서 비회전 y는 화면의 가로라,
    번역 문단이 아래로 자라지 않고 옆 여백·옆 단으로 밀리고 쓸데없이 줄었다(감사
    pdf-5). 그래서 화면 크기와 같은 회전 없는 임시 페이지에서 표시 공간 좌표로
    계획하고, 결과의 사각형·원점만 비회전 좌표로 되돌린다.

    PyMuPDF textbox의 줄바꿈은 rotate=90·270이면 상자의 높이를 줄 폭으로, 폭을 쓸 수
    있는 높이로 쓴다 — 표시 공간 상자를 rotate=0으로 시험한 결과와 그 상자를 되돌린
    비회전 상자를 rotate=page.rotation으로 넣은 결과의 줄바꿈·남는 공간이 같다. 한 줄
    경로(`_plan_single_line`)는 원래부터 표시 공간에서 원점을 계산해 되돌린다.
    """

    def __init__(self, fitz, page) -> None:
        self.to_display = page.rotation_matrix
        self.to_page = page.derotation_matrix
        self._doc = fitz.open()
        self.page = self._doc.new_page(width=page.rect.width, height=page.rect.height)

    @classmethod
    def for_page(cls, fitz, page) -> "_DisplaySpace | None":
        return cls(fitz, page) if page.rotation % 360 else None

    def close(self) -> None:
        self._doc.close()

    def shown(self, rect):
        out = rect * self.to_display
        out.normalize()
        return out

    def unshown(self, rect):
        out = rect * self.to_page
        out.normalize()
        return out

    def unshown_point(self, x: float, y: float) -> tuple[float, float]:
        matrix = self.to_page
        return (
            float(x * matrix.a + y * matrix.c + matrix.e),
            float(x * matrix.b + y * matrix.d + matrix.f),
        )

    def candidate(self, candidate: _FlowCandidate) -> _FlowCandidate:
        # 배치 기하만 옮긴다 — 지울 원문(source_rect·redact_rects)은 비회전 좌표 그대로다.
        return replace(candidate, rect=self.shown(candidate.rect))

    def replacement(self, target: _Replacement) -> _Replacement:
        plan = target.plan
        return replace(target, plan=replace(
            plan,
            rect=self.unshown(plan.rect),
            ink_rect=None if plan.ink_rect is None else self.unshown(plan.ink_rect),
            origin=None if plan.origin is None else self.unshown_point(*plan.origin),
            first_origin=(
                None if plan.first_origin is None else self.unshown_point(*plan.first_origin)
            ),
            rich_runs=tuple(
                (*self.unshown_point(x, y), text, prefix)
                for x, y, text, prefix in plan.rich_runs
            ),
        ))


def _shown_flow_components(space: _DisplaySpace | None, candidates: list[_FlowCandidate]):
    """`_flow_components`를 화면 기준으로 — 같은 단·위아래 이웃은 화면에서 정한다."""
    if space is None:
        return _flow_components(candidates)
    shown = [space.candidate(candidate) for candidate in candidates]
    original = {id(item): candidate for item, candidate in zip(shown, candidates)}
    return [[original[id(item)] for item in component] for component in _flow_components(shown)]


def _shown_flow_group(
    page, space: _DisplaySpace | None, candidates: list[_FlowCandidate],
    fixed_rects: list, *, decorative_rects=None, **kwargs,
) -> list[_Replacement] | None:
    """`_plan_flow_group`을 화면 기준으로 — 회전 페이지면 표시 공간 임시 페이지에서 계획한다."""
    if space is None:
        return _plan_flow_group(
            page, candidates, fixed_rects, decorative_rects=decorative_rects, **kwargs,
        )
    planned = _plan_flow_group(
        space.page,
        [space.candidate(candidate) for candidate in candidates],
        [space.shown(rect) for rect in fixed_rects if rect is not None],
        decorative_rects=[space.shown(rect) for rect in decorative_rects or () if rect is not None],
        **kwargs,
    )
    return None if planned is None else [space.replacement(target) for target in planned]


def _plan_flow_targets(
    ctx: _PageContext, flow_candidates: list[_FlowCandidate],
    targets: list[_Replacement], result: PdfExportResult,
) -> None:
    """같은 단의 인접 본문을 원자적으로 reflow하고, 실패하면 단계적으로 회수한다.

    회전 페이지는 화면(표시 공간) 기준으로 계획한다(`_DisplaySpace`).
    """
    space = _DisplaySpace.for_page(ctx.fitz, ctx.page) if flow_candidates else None
    try:
        _plan_flow_components(ctx, flow_candidates, targets, result, space)
    finally:
        if space is not None:
            space.close()


def _plan_flow_components(
    ctx: _PageContext, flow_candidates: list[_FlowCandidate],
    targets: list[_Replacement], result: PdfExportResult, space: _DisplaySpace | None,
) -> None:
    # 일반 텍스트는 페이지에서 모두 수집한 뒤 같은 단의 인접 블록을
    # 원자적으로 reflow한다. 이 단계 전에는 어떤 원문도 redaction하지 않는다.
    for component in _shown_flow_components(space, flow_candidates):
        component_indices = {candidate.block_index for candidate in component}
        fixed_rects = [span.rect for span in ctx.unowned_source]
        # 이번 패스에서 지워질 블록의 원문은 장애물이 아니다 — 남을 블록만 센다.
        fixed_rects.extend(ctx.obstacle_spans(exclude=component_indices))
        fixed_rects.extend(ctx.image_regions)
        fixed_rects.extend(
            target.plan.ink_rect
            for target in targets
            if target.plan.ink_rect is not None
        )
        # 벡터 도형 장애물은 따로 넘긴다 — flow가 블록 안쪽 전폭 장식 띠를 뺄 때
        # 남는 원문 span까지 함께 빼면 번역이 보존 원문 위에 겹쳐 찍힌다.
        decorative = ctx.drawing_visuals
        # 가로 평탄화 블록은 OCR 줄바꿈을 그대로 조판하면 구조적으로
        # 들어갈 수 없다. 원문의 시각적 줄 수로 되돌린 대안을 함께 시도한다.
        variants = [component]
        if any(candidate.reflow_text for candidate in component):
            variants.append([
                replace(candidate, text=candidate.reflow_text)
                if candidate.reflow_text
                else candidate
                for candidate in component
            ])
        planned = None
        for variant in variants:
            planned = _shown_flow_group(
                ctx.page, space, variant, fixed_rects, decorative_rects=decorative,
            )
            if planned is not None:
                break
        if planned is None:
            # 최후 수단: 한 블록이 안 들어간다고 같은 단의 나머지 문단까지
            # 원문으로 되돌리면 사용자에게는 문단 대여섯 개가 통째로
            # 미번역으로 보인다. 위에서 아래로 개별 배치해 들어가는 만큼만
            # 회수하고, 아직 계획하지 않은 이웃의 원문은 보존될 수 있으므로
            # 장애물로 예약해 번역문이 그 위에 겹치지 않게 한다.
            planned = []
            # 아직 배치되지 않은 형제의 원문은 남을 수 있으므로 장애물로 예약한다.
            # 앞쪽(이미 실패한) 형제도 반드시 포함해야 한다 — 빼면 그 자리를 빈
            # 공간으로 보고 번역문을 최대 48pt 위로 끌어올려 지워지지 않은 영문
            # 위에 찍는다(실측: p2 "Main Contributions" 100% 피복, p8 5줄 겹침).
            pending = {candidate.block_index for candidate in component}
            # 줄 단위로 일부만 들어간 형제의 남는 원문 줄. pending에서 빠져도 그 줄은
            # 지워지지 않으므로, 뒤 형제가 그 위로 당겨지지 않게 계속 장애물로 둔다.
            residual: list = []
            for candidate in component:
                obstacles = list(fixed_rects)
                obstacles.extend(residual)
                obstacles.extend(
                    target.plan.ink_rect
                    for target in planned
                    if target.plan.ink_rect is not None
                )
                obstacles.extend(
                    span.rect
                    for other in pending
                    if other != candidate.block_index
                    for span in ctx.source_ownership.get(other, [])
                )
                # 래스터 원문 형제는 span이 없다 — 남을 수 있는 블록 영역을 예약한다.
                obstacles.extend(ctx.raster_obstacles(
                    frozenset(ctx.raster_blocks) - (pending - {candidate.block_index}),
                ))
                single = None
                for text in (candidate.text, candidate.reflow_text):
                    if text is None:
                        continue
                    single = _shown_flow_group(
                        ctx.page, space, [replace(candidate, text=text)], obstacles,
                        decorative_rects=decorative,
                    )
                    if single:
                        break
                if single:
                    planned.extend(single)
                    pending.discard(candidate.block_index)
                    continue
                # 흘려 넣기가 전부 실패해도 리스팅·표는 줄 단위로는 제자리에
                # 들어간다. 폭이 모자란 줄만 원문으로 남기고 나머지를 회수한다.
                listing_targets, listing_changed = _plan_listing_lines(
                    ctx.page,
                    candidate.listing_segments,
                    candidate.fontname,
                    candidate.fontfile,
                    obstacles + list(decorative),
                    candidate.block_index,
                    bold=candidate.bold,
                )
                if listing_targets:
                    planned.extend(listing_targets)
                    pending.discard(candidate.block_index)
                    missing = listing_changed - len(listing_targets)
                    if missing:
                        result.keep("listing_line_no_fit", missing)
                        result.warnings.append(
                            f"p{ctx.pno}: 블록 {candidate.block_index + 1}의 "
                            f"{missing}줄 교체 생략(줄 폭 부족) — 그 줄만 원문 보존"
                        )
                    _keep_dropped_listing_lines(
                        ctx, candidate.block_index, candidate.listing_dropped, result,
                    )
                    residual.extend(_unplaced_listing_spans(
                        ctx, candidate.block_index, listing_targets,
                    ))
                    continue
                # 최후 수단: 가독성 하한 아래로 축소해서라도 놓는다. 여기서
                # 포기하면 번역 면에 영문 원문이 그대로 남고, 그건 사용자가
                # 금지한 상태다(미번역·원문 혼재). 계층이 무너지는 편이 낫다.
                for text in (candidate.text, candidate.reflow_text):
                    if text is None:
                        continue
                    single = _shown_flow_group(
                        ctx.page, space, [replace(candidate, text=text)], obstacles,
                        scales=_LASTRESORT_SHRINK_STEPS,
                        min_pt=_LASTRESORT_MIN_FONT_PT,
                        decorative_rects=decorative,
                    )
                    if single:
                        break
                if single:
                    planned.extend(single)
                    pending.discard(candidate.block_index)
                    smallest = min(t.plan.fontsize for t in single)
                    result.warnings.append(
                        f"p{ctx.pno}: 블록 {candidate.block_index + 1} 축소 배치"
                        f"({smallest:.1f}pt) — 원문을 남기지 않으려 가독성 하한 아래로 조판"
                    )
                    continue
                flattened = candidate.reflow_text is not None
                result.keep("flattened_no_fit" if flattened else "no_fit")
                reason = (
                    "가로 평탄화 블록 — 리플로우 실패"
                    if flattened
                    else "공간 부족"
                )
                result.warnings.append(
                    f"p{ctx.pno}: 블록 {candidate.block_index + 1} 교체 생략"
                    f"({reason}) — 원문 보존"
                )
        targets.extend(planned)


def _normalize_repeated_scheme_links(page, source_rects: list[object]) -> None:
    """미세 교정한 중복 URL의 클릭 annotation도 같은 정상 URI로 맞춘다."""
    fitz = quiet_fitz()
    if not source_rects:
        return
    links = page.get_links()
    matched_uris: set[str] = set()
    for link in links:
        uri = str(link.get("uri") or "")
        match = re.match(r"^(https?://)(?:\1)+(.*)$", uri, re.IGNORECASE)
        if link.get("kind") != fitz.LINK_URI or match is None:
            continue
        raw_rect = link.get("from")
        if raw_rect is None:
            continue
        try:
            link_rect = fitz.Rect(raw_rect)
        except Exception:  # noqa: BLE001 — 손상 annotation은 건너뛴다
            continue
        if not any(_rect_overlap_area(link_rect, rect) > 0.01 for rect in source_rects):
            continue
        matched_uris.add(uri)
    for link in links:
        uri = str(link.get("uri") or "")
        if uri not in matched_uris:
            continue
        match = re.match(r"^(https?://)(?:\1)+(.*)$", uri, re.IGNORECASE)
        if match is None:
            continue
        normalized = match.group(1) + match.group(2)
        updated = dict(link)
        updated["uri"] = normalized
        page.update_link(updated)


def _hide_visible_link_borders(page) -> None:
    """URI 동작은 보존하고 논문 본문 위의 유색 annotation 테두리만 숨긴다."""
    doc = page.parent
    for link in page.get_links():
        xref = link.get("xref")
        if link.get("kind") != quiet_fitz().LINK_URI or not isinstance(xref, int) or xref <= 0:
            continue
        try:
            # PDF 기본 border width 1은 Semantic Scholar 링크처럼 본문 두 행을
            # 청록색 상자로 둘러싼다. Border와 우선순위가 높은 BS를 모두 0으로
            # 만들고 기존 appearance를 제거해도 URI와 클릭 영역은 그대로다.
            doc.xref_set_key(xref, "Border", "[0 0 0]")
            doc.xref_set_key(xref, "BS", "<< /W 0 >>")
            doc.xref_set_key(xref, "AP", "null")
        except (RuntimeError, ValueError, TypeError):
            logger.warning("PDF 링크 테두리를 숨기지 못했습니다: xref=%s", xref)


def _redact_in_chunks(page, rects, chunk: int = _REDACT_CHUNK, **apply_kwargs) -> None:
    """리댁션을 나눠 건다 — PyMuPDF의 annot 이름 부여가 페이지당 2차이기 때문.

    `page.add_redact_annot`는 삽입할 때마다 `JM_add_annot_id`가 페이지의 **기존
    annot을 전수 순회**해 `/NM`이 겹치지 않는지 본다. 그래서 한 페이지에 N개를
    쌓으면 비용이 N²로 큰다(실측: annot당 0.245ms@N=100 → 2.773ms@N=1000).
    실제 문서는 페이지당 최대 547개, 문서 전체 4,190개였다.

    `apply_redactions`가 지운 annot을 페이지에서 걷어내므로, 끊어서 add→apply를
    반복하면 순회 대상이 매번 chunk 크기로 리셋된다. 지우는 사각형 집합과 순서는
    같으므로 결과 문서는 달라지지 않는다(실측: 6.14s → 1.83s, 리포트·추출 텍스트
    동일). 빈 목록이면 apply_redactions를 아예 부르지 않는다 — 호출자는 지금
    targets가 빈 페이지에서 먼저 빠져나가지만, 여기서도 지울 게 없으면 페이지를
    건드리지 않는 편이 안전하다.
    """
    for start in range(0, len(rects), chunk):
        batch = rects[start:start + chunk]
        if not batch:
            continue
        for rect in batch:
            page.add_redact_annot(rect)
        page.apply_redactions(**apply_kwargs)


def _apply_page_redactions(fitz, page, targets, raster_rects, raster_covers=()) -> None:
    """원문 텍스트(그리고 이모지의 이미지 절반)만 지운다 — 그래픽은 보존.

    `raster_covers`는 `(영역, 바탕색)` 목록이다. 원문이 스캔 픽셀인 교체 블록의
    영역을 바탕색으로 덮는다 — 텍스트 리댁션으로는 지울 수 없는 원문이다.
    """
    # 2) 원문 텍스트 리댁션 (이미지·그래픽 보존) — 삽입 전에 일괄 적용
    source_rects = []
    text_rects = []
    for target in targets:
        # 삽입 bbox가 아래 빈 공간으로 커져도 실제 원문 bbox만 지운다.
        # 확장 사각형 전체를 리댁션하면 인접한 원문 글리프가 함께 사라질 수 있다.
        target_redactions = target.redact_rects or (
            target.redact_rect if target.redact_rect is not None else target.plan.rect,
        )
        text_rects.extend(target_redactions)
        source_rects.append((
            +(
                target.source_rect
                if target.source_rect is not None
                else target_redactions[0]
            ),
            # 이모지는 글자 한 칸 크기다. 블록 폰트의 2배를 넘는 인스턴스는
            # 인라인 그림·로고·아이콘이므로 제거 대상에서 뺀다.
            max(2.0 * target.plan.fontsize, 20.0),
        ))
    # 텍스트만 제거한다. graphics 기본값(REMOVE_IF_COVERED)을 그대로 두면
    # 블록 안의 밑줄·도형·차트 선까지 사라져 "레이아웃 보존"을 위반한다.
    _redact_in_chunks(
        page, text_rects,
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    # 2b) 이모지의 '이미지 절반' 제거. macOS Quartz 산출 PDF는 컬러
    # 이모지를 (보이지 않는 텍스트 글리프 + 이미지 XObject) 이중으로
    # 기록해 텍스트 리댁션만으로는 이미지가 번역문 위에 남는다. 교체
    # 사각형에 완전히 포함된 소형(25% 이하) 인스턴스만 별도 pass로
    # 지운다 — 부분 겹침 rect에 IMAGE_REMOVE를 쓰면 걸친 그림에 흰
    # 구멍이 나므로 절대 블록 rect 전체로 걸지 않는다. 면적비만으로는
    # 넓은 블록 안의 100pt 인라인 그림도 걸리므로 '글자 한 칸 크기의
    # 정사각형'이라는 이모지 고유 성질을 절대 크기·종횡비로 함께 건다.
    emoji_boxes = []
    for bbox in raster_rects:
        area = bbox.width * bbox.height
        for rr, size_limit in source_rects:
            # Quartz 반올림으로 이미지가 글리프 상자를 1pt 미만 벗어나는
            # 경우가 있어 1pt 허용 오차로 '완전 포함'을 판정한다.
            if (
                bbox.x0 >= rr.x0 - 1.0
                and bbox.y0 >= rr.y0 - 1.0
                and bbox.x1 <= rr.x1 + 1.0
                and bbox.y1 <= rr.y1 + 1.0
                and 0 < area <= rr.width * rr.height * 0.25
                and max(bbox.width, bbox.height) <= size_limit
                and 0.5 <= bbox.width / max(bbox.height, 0.01) <= 2.0
            ):
                emoji_boxes.append(bbox)
                break
    if emoji_boxes:
        _redact_in_chunks(
            page, emoji_boxes,
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_NONE,
        )
    # 2c) 스캔 픽셀 덮기. 리댁션의 채움(fill)은 적용 시 페이지 내용 맨 위에 그려지고,
    # 번역문은 그 뒤에 삽입되므로 '바탕색 사각형 위 한국어'가 된다. 이미지 객체와
    # 다른 영역의 픽셀은 그대로다. PDF_REDACT_IMAGE_PIXELS로 실제 픽셀까지 지우지
    # 않는 이유: 화면 결과는 같은데 MuPDF가 이미지를 디코드해 비압축(Flate)으로 다시
    # 써서 JPEG 스캔이 페이지당 0.92→1.73MB(회색)·0.96→2.69MB(컬러)로 불어났다(실측,
    # 300dpi). JPX 스캔이면 더 크다.
    covers = list(raster_covers)
    for start in range(0, len(covers), _REDACT_CHUNK):
        for rect, fill in covers[start:start + _REDACT_CHUNK]:
            page.add_redact_annot(rect, fill=fill)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_NONE,
        )


# 덮을 영역의 바탕색을 고를 때 쓰는 렌더 해상도. 50dpi면 A4가 413x585px라 페이지당
# 한 번 렌더하는 비용이 작고, 글자 획은 흐려져도 가장 흔한 색은 종이색이다.
_RASTER_SAMPLE_DPI = 50
# OCR bbox는 0–999 정수 좌표라 글리프 가장자리를 1pt 안쪽에서 자르기도 한다.
_RASTER_ERASE_PAD_PT = 0.8
# 바탕색이 이보다 어두우면(어두운 띠 위 흰 글자 등) 검은 번역 글자가 보이지 않으므로
# 흰색으로 덮는다.
_RASTER_DARK_LUMA = 0.5


def _subtract_rect(fitz, rect, hole) -> list:
    """`rect`에서 `hole`을 뺀 영역 — 겹치지 않으면 그대로, 겹치면 최대 네 조각."""
    overlap = fitz.Rect(
        max(rect.x0, hole.x0), max(rect.y0, hole.y0),
        min(rect.x1, hole.x1), min(rect.y1, hole.y1),
    )
    if overlap.x1 <= overlap.x0 or overlap.y1 <= overlap.y0:
        return [rect]
    pieces = []
    if overlap.y0 > rect.y0:
        pieces.append(fitz.Rect(rect.x0, rect.y0, rect.x1, overlap.y0))
    if overlap.y1 < rect.y1:
        pieces.append(fitz.Rect(rect.x0, overlap.y1, rect.x1, rect.y1))
    if overlap.x0 > rect.x0:
        pieces.append(fitz.Rect(rect.x0, overlap.y0, overlap.x0, overlap.y1))
    if overlap.x1 < rect.x1:
        pieces.append(fitz.Rect(overlap.x1, overlap.y0, rect.x1, overlap.y1))
    return pieces


# 이 두께 이하로 남은 덮개 조각은 버린다 — 덮개의 여유와 남는 블록 쪽 여유가 겹친
# 자리(예: 캡션 덮개가 그림 bbox 옆으로 삐져나온 1.6pt 띠)라 블록 자신의 글자가 아니라
# 이웃의 가장자리 픽셀(눈금 라벨 끝 등)만 지운다.
_RASTER_ERASE_MIN_PIECE_PT = 2 * _RASTER_ERASE_PAD_PT + 0.01


def _raster_erase_regions(ctx: _PageContext, targets) -> tuple[list, set[int]]:
    """원문이 래스터인 교체 대상마다 덮을 원래 영역(번역이 옮겨 가도 원래 자리).

    `(덮을 영역 목록, 덮을 곳이 남지 않은 블록 인덱스)`를 낸다. 표 셀의 `source_rect`는
    셀 사각형이 아니라 그 셀 글자 잉크의 상자다(raster_tables) — 괘선과 이웃 셀 글자를
    덮지 않으므로 여유를 더하지 않는다.

    OCR bbox는 0–999 격자라 캡션과 그림, 문단과 수식 상자가 몇 pt씩 겹친다. 덮개가 그
    겹침까지 덮으면 이웃 그림의 축 라벨·남는 수식·참고문헌 줄 픽셀이 지워졌다(감사
    pdf-3). 그래서 남아야 하는 영역 — 레이아웃 그림·표 블록과 교체되지 않는 블록 —
    을 덮개에서 뺀다. 경계가 맞닿은 정상 레이아웃이 예전과 같게, 이웃 쪽으로 여유
    (`_RASTER_ERASE_PAD_PT`)만큼은 들어가도 된다.
    """
    fitz = ctx.fitz
    replaced = {target.block_index for target in targets if target.block_index >= 0}
    protected: list[tuple[int, object]] = []
    for index, (rect, block) in enumerate(zip(ctx.block_rects, ctx.oblocks)):
        if rect is None or not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type in ("image", "table") or block.get("image"):
            keep = True
        else:
            # 내용이 빈 블록은 자식 블록을 감싼 컨테이너(빈 list 등)다 — 제 픽셀이 없으므로
            # 지키면 그 안의 교체 블록을 하나도 덮지 못한다(실측: 목록 항목 11개).
            keep = index not in replaced and bool(str(block.get("content") or "").strip())
        if keep:
            keep_out = +rect
            keep_out += (
                _RASTER_ERASE_PAD_PT, _RASTER_ERASE_PAD_PT,
                -_RASTER_ERASE_PAD_PT, -_RASTER_ERASE_PAD_PT,
            )
            if keep_out.x1 > keep_out.x0 and keep_out.y1 > keep_out.y0:
                protected.append((index, keep_out))
    regions: list = []
    uncovered: set[int] = set()
    seen: set[tuple] = set()
    for target in targets:
        if target.block_index not in ctx.raster_blocks:
            continue
        if target.kind not in ("text", "listing", "table") or target.source_rect is None:
            continue
        region = +target.source_rect
        if target.kind == "text":
            region += (
                -_RASTER_ERASE_PAD_PT, -_RASTER_ERASE_PAD_PT,
                _RASTER_ERASE_PAD_PT, _RASTER_ERASE_PAD_PT,
            )
        region &= ctx.page.mediabox
        if region.is_empty:
            continue
        pieces = [region]
        for index, keep_out in protected:
            if index != target.block_index:
                pieces = [
                    part for piece in pieces for part in _subtract_rect(fitz, piece, keep_out)
                ]
        pieces = [
            piece for piece in pieces
            if piece.width > _RASTER_ERASE_MIN_PIECE_PT
            and piece.height > _RASTER_ERASE_MIN_PIECE_PT
        ]
        if not pieces:
            uncovered.add(target.block_index)
            continue
        for piece in pieces:
            key = tuple(round(value, 2) for value in piece)
            if key in seen:
                continue
            seen.add(key)
            regions.append(piece)
    return regions, uncovered


def _raster_background_fills(fitz, page, regions) -> list:
    """덮을 영역마다 스캔 바탕색(그 영역에서 가장 흔한 색). 알 수 없으면 흰색.

    누렇게 바랜 스캔을 흰색으로 덮으면 블록마다 흰 사각형이 도드라진다. 페이지를
    한 번만 낮은 해상도로 렌더하고(표시 공간), 영역별 최빈색을 고른다.
    """
    white = (1.0, 1.0, 1.0)
    if not regions:
        return []
    zoom = _RASTER_SAMPLE_DPI / 72.0
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False,
        )
    except Exception:  # noqa: BLE001 — 렌더 실패는 흰색 덮기로 폴백
        return [white] * len(regions)
    bounds = fitz.IRect(0, 0, pixmap.width, pixmap.height)
    fills = []
    for region in regions:
        shown = region * page.rotation_matrix
        shown.normalize()
        clip = fitz.IRect(
            int(shown.x0 * zoom) - 1, int(shown.y0 * zoom) - 1,
            int(shown.x1 * zoom) + 2, int(shown.y1 * zoom) + 2,
        ) & bounds
        if clip.is_empty:
            fills.append(white)
            continue
        try:
            _ratio, color = pixmap.color_topusage(clip=clip)
        except Exception:  # noqa: BLE001
            fills.append(white)
            continue
        red, green, blue = (component / 255.0 for component in color[:3])
        luma = 0.299 * red + 0.587 * green + 0.114 * blue
        fills.append((red, green, blue) if luma >= _RASTER_DARK_LUMA else white)
    return fills


def _warn_unerased_raster_sources(ctx: _PageContext, targets, visuals, result) -> None:
    """보이는 원문 텍스트 없이 이미지 일부와 겹친 교체 블록 — 덮지 못하므로 알린다.

    전면 스캔·표 이미지는 덮지만, 그 밖의 래스터(그림 일부 등)는 born-digital 그림을
    지키려고 덮지 않는다. 그 위에 번역이 찍히면 이미지 속 글자와 겹쳐 보일 수 있다 —
    리포트가 '깨끗한 교체'로만 세지 않도록 경고로 남긴다.
    """
    if not visuals.raster_rects:
        return
    warned: set[int] = set()
    for target in targets:
        index = target.block_index
        if index < 0 or index in warned or index in ctx.raster_blocks:
            continue
        if target.kind not in ("text", "listing", "table"):
            continue
        if any(span.visible for span in ctx.source_ownership.get(index, [])):
            continue
        rect = ctx.block_rects[index] if index < len(ctx.block_rects) else None
        if rect is None or not any(
            _rect_overlap_area(rect, raster) > 1.0 for raster in visuals.raster_rects
        ):
            continue
        warned.add(index)
        result.warnings.append(
            f"p{ctx.pno}: 블록 {index + 1}의 원문이 이미지 픽셀이라 지우지 못함 — "
            "번역이 이미지 속 글자와 겹쳐 보일 수 있음"
        )


def _insert_fitted_text(
    page, plan: _TextFitPlan, text: str, fontname: str, fontfile: str | None,
    bold_prefix: tuple[str, str] | None = None,
) -> None:
    """검증된 계획을 적용한다. dry-run과 달라지면 손상 PDF를 저장하지 않고 중단."""
    if bold_prefix is not None:
        if not fontfile or not plan.rich_runs:
            raise PdfExportError("접두 강조용 한글 폰트가 없어 PDF 생성을 중단했습니다")
        for x, y, run_text, is_prefix in plan.rich_runs:
            run_kwargs = {
                "fontsize": plan.fontsize,
                "fontname": fontname,
                "fontfile": fontfile,
                "rotate": page.rotation,
                "color": (0, 0, 0),
            }
            if is_prefix:
                run_kwargs.update({
                    "render_mode": 2,
                    "fill": (0, 0, 0),
                    "border_width": 0.02,
                })
            page.insert_text(
                quiet_fitz().Point(x, y),
                run_text,
                **run_kwargs,
            )
        return
    kwargs = {
        "fontsize": plan.fontsize,
        "fontname": fontname,
        "fontfile": fontfile,
        "align": plan.align,
        "rotate": page.rotation,
        "color": (0, 0, 0),
        "lineheight": plan.lineheight,
    }
    if plan.bold:
        kwargs.update({
            "render_mode": 2,
            "fill": (0, 0, 0),
            "border_width": 0.02,
        })
    if plan.origin is not None:
        fitz = quiet_fitz()
        single_kwargs = dict(kwargs)
        single_kwargs.pop("align", None)
        page.insert_text(fitz.Point(*plan.origin), text, **single_kwargs)
    else:
        leftover = page.insert_textbox(plan.rect, text, **kwargs)
        if leftover < 0:
            raise PdfExportError("번역 텍스트 조판 결과가 사전 검증과 달라 PDF 생성을 중단했습니다")


def _insert_page_targets(page, targets, result: PdfExportResult) -> None:
    """계획대로 번역문을 삽입하고 결과 집계를 갱신한다."""
    # 3) 번역 텍스트 삽입
    for target in targets:
        _insert_fitted_text(
            page,
            target.plan,
            target.text,
            target.fontname,
            target.fontfile,
            target.bold_prefix,
        )
        result.replaced += 1
        if target.plan.expanded:
            result.relocated += 1
        if target.kind == "table":
            result.table_cells_replaced += 1
        elif target.kind == "listing":
            result.listing_lines_replaced += 1


# 판정에 쓸 블록의 최소 길이 — 짧은 라벨은 우연 일치가 잦다.
_REGISTRATION_MIN_CHARS = 20
_REGISTRATION_PROBE = 40
# 판정에 필요한 최소 프로브 수. 블록이 한둘뿐인 페이지는 비율이 요동쳐 오탐이 난다.
_REGISTRATION_MIN_PROBES = 3
# 이웃 페이지가 이만큼 더 잘 맞으면 "이 페이지의 layout이 아니다"로 본다.
# 절대 점수는 OCR 잡음 때문에 정상 페이지도 0.5까지 내려간다(실측 p7 0.55) —
# 판별력이 있는 건 **이웃과의 차이**다(실측: 어긋난 페이지는 이웃이 0.83~1.00,
# 자기 자리는 0.08~0.30으로 격차가 0.5 이상 벌어졌다).
_REGISTRATION_MARGIN = 0.25
_REGISTRATION_NEIGHBOUR_MIN = 0.50
_REGISTRATION_RADIUS = 2


def _registration_probes(oblocks) -> list[str]:
    probes = [
        _ownership_text(b.get("content"))[:_REGISTRATION_PROBE]
        for b in oblocks
        if (b.get("type") or "").lower() not in _SPECIALIST_TYPES
    ]
    return [t for t in probes if len(t) >= _REGISTRATION_MIN_CHARS]


def _registration_score(probes: list[str], haystack: str) -> float:
    if not probes or not haystack:
        return 0.0
    return sum(1 for t in probes if t in haystack) / len(probes)


def _unregistered_layout_pages(doc, orig_pages: dict) -> dict:
    """layout 페이지가 **다른** 물리 페이지를 설명하고 있는 경우를 찾는다.

    OCR 단계가 페이지를 밀어 매핑하면(모델이 페이지를 쪼개거나 건너뛴 경우)
    여기서 걸러야 한다. 걸러지지 않으면 `_assign_source_spans`가 순전히 기하로
    소유권을 배정해 **엉뚱한 페이지의 원문을 영구 리댁션**하고 그 자리에 다른
    페이지의 번역을 찍는다(실측: 46p 논문에서 23쪽 프로젝트명 29개가 삭제되고
    24쪽 캡션이 그 자리에 찍혔다).

    반환: {layout 페이지 번호: (더 잘 맞는 물리 페이지, 그 점수, 제자리 점수)}
    """
    texts: dict = {}

    def page_text(idx: int) -> str:
        if idx not in texts:
            texts[idx] = (
                _ownership_text(doc[idx].get_text())
                if 0 <= idx < doc.page_count else ""
            )
        return texts[idx]

    out: dict = {}
    for pno, opage in orig_pages.items():
        if not isinstance(pno, int):
            continue
        probes = _registration_probes(opage.get("blocks", []))
        if len(probes) < _REGISTRATION_MIN_PROBES:
            continue  # 판정 근거 부족 — 기존 동작 유지
        mine = _registration_score(probes, page_text(pno - 1))
        best_score, best_page = mine, pno
        for delta in range(-_REGISTRATION_RADIUS, _REGISTRATION_RADIUS + 1):
            if delta == 0:
                continue
            other = _registration_score(probes, page_text(pno - 1 + delta))
            if other > best_score:
                best_score, best_page = other, pno + delta
        if (
            best_page != pno
            and best_score >= _REGISTRATION_NEIGHBOUR_MIN
            and best_score >= mine + _REGISTRATION_MARGIN
        ):
            out[pno] = (best_page, best_score, mine)
    return out


# 계획 ↔ 리댁션 일치까지 허용하는 최대 패스 수. 매 패스마다 "지워진다고 가정한
# 블록" 집합이 **엄격히 줄어들므로** 블록 수 안에서 반드시 수렴한다. 상한은
# 병리적 입력의 CPU 방어일 뿐이다(실측 46p 논문: 페이지당 1~2패스).
_MAX_PLAN_PASSES = 6


def _optimistically_cleared(oblocks, tblocks) -> frozenset:
    """원문이 지워질 **후보** 블록 — 계획 1패스의 낙관적 가정.

    표는 셀 단위로 리댁션하므로 블록 통째로 지워진다고 가정하지 않는다.
    """
    out = set()
    for i, (ob, tb) in enumerate(zip(oblocks, tblocks)):
        if not isinstance(ob, dict) or not isinstance(tb, dict):
            continue
        btype = str(tb.get("type") or "")
        if btype not in _REPLACEABLE_TYPES:
            continue
        if (tb.get("vertical") or ob.get("vertical")) in _VERTICAL_SKIP:
            continue
        if str(tb.get("content") or "").strip() == str(ob.get("content") or "").strip():
            continue
        out.add(i)
    return frozenset(out)


def _plan_until_consistent(base_ctx: _PageContext, result: PdfExportResult):
    """계획의 장애물 모델과 실제 리댁션이 일치할 때까지 다시 계획한다.

    1패스는 교체 후보의 원문이 **전부 지워진다**고 가정한다 — 그래야 같은 단의
    이웃 원문이 자리를 막아 번역이 통째로 버려지는 일이 없다. 배치에 실패한
    블록은 원문이 남으므로 그 원문을 장애물로 되돌리고 다시 계획한다. 가정
    집합이 매 패스 줄어들어 수렴하며, 수렴 시점에는 "장애물로 본 것 = 실제로
    남는 것"이 되어 번역문이 남은 원문 위에 찍히는 일이 구조적으로 없다.
    """
    cleared = frozenset(
        index
        for index in _optimistically_cleared(base_ctx.oblocks, base_ctx.tblocks)
        if not _raster_block_looks_vertical(base_ctx, index, base_ctx.oblocks[index])
    )
    targets: list[_Replacement] = []
    links: list[object] = []
    trial = result
    for attempt in range(_MAX_PLAN_PASSES):
        # 마지막 패스가 아니면 집계를 버린다 — 중간 패스의 keep 사유가 리포트에
        # 섞이면 실제 산출물과 다른 수치가 남는다.
        trial = PdfExportResult(path=result.path)
        ctx = replace(base_ctx, cleared_indices=cleared, residual_spans={})
        targets, flow_candidates, links = _plan_page_targets(ctx, trial)
        _plan_flow_targets(ctx, flow_candidates, targets, trial)
        # 줄 단위로 일부만 들어간 블록은 '지워졌다'고 볼 수 없다 — 남는 줄이 있다.
        placed = {t.block_index for t in targets if t.block_index >= 0}
        placed -= _partially_redacted_blocks(ctx, targets)
        missing = cleared - placed
        if not missing:
            break
        if attempt == _MAX_PLAN_PASSES - 1:
            # 수렴하지 못한 계획은 장애물 모델이 실제 리댁션과 어긋날 수 있다. 아무것도
            # 지워진다고 가정하지 않는(가장 보수적인) 계획으로 한 번 더 세운다 — 모든
            # 원문이 장애물이므로 남는 원문 위에 번역이 찍힐 수 없다.
            logger.warning(
                "PDF 내보내기: p%d 계획이 %d패스 안에 수렴하지 않음 — 보수적 계획 사용",
                base_ctx.pno, _MAX_PLAN_PASSES,
            )
            trial = PdfExportResult(path=result.path)
            ctx = replace(base_ctx, cleared_indices=frozenset(), residual_spans={})
            targets, flow_candidates, links = _plan_page_targets(ctx, trial)
            _plan_flow_targets(ctx, flow_candidates, targets, trial)
            break
        cleared = cleared - missing
    result.merge(trial)
    return targets, links


def _process_page(
    fitz, page, pno, tpage, opage, fonts, result, misregistered=None,
) -> None:
    """한 페이지를 계획 → 리댁션 → 삽입 순서로 처리한다."""
    # 0) 페이지 등록 확인 — 다른 페이지를 설명하는 레이아웃이면 **한 글자도
    #    건드리지 않는다**. 건드리면 그 페이지의 원문이 영구 삭제된다.
    hit = (misregistered or {}).get(pno)
    if hit is not None:
        elsewhere, score, mine = hit
        blocks = opage.get("blocks", [])
        result.keep("page_source_mismatch", max(1, len(blocks)))
        result.warnings.append(
            f"p{pno}: 레이아웃이 {elsewhere}쪽 내용과 일치해 이 페이지는 건너뜀 "
            f"(대조 {score:.0%} vs 제자리 {mine:.0%}) — 재변환이 필요합니다"
        )
        logger.warning(
            "PDF 내보내기: p%d 레이아웃이 p%d와 일치(%.2f vs %.2f) — 페이지 미수정",
            pno, elsewhere, score, mine,
        )
        return

    _hide_visible_link_borders(page)
    width = tpage.get("width") or 1
    height = tpage.get("height") or 1
    aspect = height / width if width else 1.0

    # 1) 교체 대상 수집. 모든 블록 사각형을 먼저 만들고, 일반 텍스트가
    # 공간을 늘릴 때 같은 단의 다음 블록과 충돌하지 않는 하단을 계산한다.
    oblocks = opage.get("blocks", [])
    tblocks = tpage.get("blocks", [])
    block_rects = [_block_rect(fitz, page, b.get("bbox")) for b in oblocks]
    image_infos = _page_image_infos(page)
    # 나중에 그린 이미지에 가려진 글자는 원문이 아니라 그 이미지 픽셀이다(이미지 아래 텍스트
    # 형식의 스캔) — 소유권을 정하기 전에 고쳐야 래스터 원문 판정·경고가 같은 span을 본다.
    source_records = _hide_occluded_spans(
        fitz, page, _source_span_records(fitz, page), image_infos,
    )
    source_ownership, unowned_source, ambiguous_blocks = _assign_source_spans(
        page, block_rects, oblocks, source_records,
    )
    visuals = _page_visual_obstacles(
        fitz, page, block_rects, oblocks,
        image_infos=image_infos, source_records=source_records,
    )
    base_ctx = _PageContext(
        fitz, page, pno, aspect, oblocks, tblocks, block_rects,
        source_records, source_ownership, unowned_source, ambiguous_blocks,
        visuals.image_regions, visuals.fixed_visuals, fonts,
        drawing_visuals=visuals.drawing_rects,
        raster_blocks=_raster_backed_blocks(block_rects, oblocks, source_records, visuals),
        analysis={"horizontal_segments": visuals.horizontal_segments},
    )
    try:
        targets, repeated_scheme_link_rects = _plan_until_consistent(base_ctx, result)
    finally:
        # TextPage 등 원본 페이지 분석 결과는 리댁션 뒤에는 쓸모없고 메모리만 잡는다.
        base_ctx.analysis.clear()
    if not targets:
        return

    # 반복 scheme의 첫 링크 annotation은 아래 redaction에서 사라질 수
    # 있다. 겹친 annotation으로 bad URI 집합을 식별할 수 있을 때 같은
    # URI를 가진 wrapped 링크까지 먼저 정상화한다.
    if repeated_scheme_link_rects:
        _normalize_repeated_scheme_links(page, repeated_scheme_link_rects)

    # 래스터 원문을 덮을 영역과 그 바탕색은 페이지를 건드리기 **전에** 정한다 —
    # 리댁션 뒤 렌더는 이미 덮인 색을 샘플링한다.
    erase_regions, uncovered = _raster_erase_regions(base_ctx, targets)
    erase_fills = _raster_background_fills(fitz, page, erase_regions)
    _warn_unerased_raster_sources(base_ctx, targets, visuals, result)
    for index in sorted(uncovered):
        result.warnings.append(
            f"p{pno}: 블록 {index + 1}의 원문 픽셀이 이웃 그림·보존 블록과 겹쳐 덮지 못함 — "
            "번역이 원문과 겹쳐 보일 수 있음"
        )
    _apply_page_redactions(
        fitz, page, targets, visuals.raster_rects, list(zip(erase_regions, erase_fills)),
    )
    result.raster_blocks_erased += len({
        target.block_index for target in targets
        if target.block_index in base_ctx.raster_blocks
        and target.block_index not in uncovered
    })
    _insert_page_targets(page, targets, result)


_TOUNICODE_BFCHAR_RE = re.compile(
    rb"beginbfchar(.*?)endbfchar", re.DOTALL,
)


def _space_sharing_glyph(fontfile: str | None) -> int | None:
    """공백(U+0020)과 NBSP(U+00A0)가 **같은 글리프**인 폰트면 그 gid. 아니면 None."""
    if not fontfile:
        return None
    try:
        font = quiet_fitz().Font(fontfile=fontfile)
        space, nbsp = int(font.has_glyph(0x20)), int(font.has_glyph(0xA0))
    except Exception:  # noqa: BLE001 — 확인 불가면 손대지 않는다
        return None
    return space if space and space == nbsp else None


def _restore_space_tounicode(doc, fonts: _ExportFonts) -> int:
    """삽입 폰트의 ToUnicode에서 공백 글리프가 U+00A0으로 역매핑된 것을 U+0020으로.

    AppleSDGothicNeo처럼 공백과 NBSP가 같은 글리프인 폰트로 삽입하면 PyMuPDF가
    만드는 ToUnicode가 그 글리프를 U+00A0 하나로만 적어, 번역문의 **모든** 공백이
    NBSP로 추출된다(복사·검색·grep이 어긋난다). 화면은 같고 추출만 틀리므로, 같은
    글리프임을 폰트 cmap으로 확인한 경우에만 그 bfchar 한 줄을 U+0020으로 고친다.
    고친 ToUnicode 스트림 수를 돌려준다.
    """
    targets: dict[str, int] = {}
    for name, fontfile in (
        (fonts.serif_name, fonts.serif_ff),
        (fonts.sans_name, fonts.sans_ff),
        (fonts.table_name, fonts.table_ff),
    ):
        gid = _space_sharing_glyph(fontfile)
        if gid is not None:
            targets[name] = gid
    if not targets:
        return 0
    seen: set[int] = set()
    fixed = 0
    for page in doc:
        try:
            page_fonts = page.get_fonts(full=True)
        except Exception:  # noqa: BLE001
            continue
        for entry in page_fonts:
            xref, resource = entry[0], str(entry[4]) if len(entry) > 4 else ""
            gid = targets.get(resource)
            if gid is None or xref in seen:
                continue
            seen.add(xref)
            try:
                kind, value = doc.xref_get_key(xref, "ToUnicode")
                if kind != "xref":
                    continue
                cmap_xref = int(value.split()[0])
                stream = doc.xref_stream(cmap_xref)
            except Exception:  # noqa: BLE001 — 손상 font dict는 건너뛴다
                continue
            wrong = f"<{gid:04x}> <00a0>".encode()
            right = f"<{gid:04x}> <0020>".encode()
            updated = _TOUNICODE_BFCHAR_RE.sub(
                lambda match: match.group(0).replace(wrong, right)
                .replace(wrong.upper(), right), stream,
            )
            if updated != stream:
                doc.update_stream(cmap_xref, updated)
                fixed += 1
    return fixed


def _write_export_report(job_dir: Path, lang: str, result: PdfExportResult) -> None:
    """UI가 보존·재배치 정보를 읽을 수 있게 리포트를 원자적으로 저장한다."""
    # 캐시된 PDF 요청에서도 UI가 보존/재배치 정보를 읽을 수 있게 별도 리포트를
    # 원자적으로 저장한다. 본문·API 응답·비밀은 포함하지 않는다.
    report_path = job_dir / f"export.{lang}.report.json"
    report_tmp = job_dir / f".export.{lang}.report.{uuid.uuid4().hex}.tmp"
    try:
        report_tmp.write_text(json.dumps(result.report(), ensure_ascii=False), encoding="utf-8")
        report_tmp.replace(report_path)
    except OSError:
        logger.warning("PDF 내보내기 리포트 저장 실패: %s", report_path.name)
    finally:
        report_tmp.unlink(missing_ok=True)


# ── 빌드 격리: export 풀 워커에서 실행 ─────────────────────────────────────
# 빌드는 페이지마다 벡터·텍스트 분석·리댁션·저장을 하는 수십 초짜리 MuPDF 작업이다. 서버
# 프로세스 안에서 돌리면 GIL을 쥐어 같은 프로세스의 OCR 디코드가 31.7→1.0 tok/s로 굶고
# (감사 gap1-metal-real-e2e-2·concurrency-2), 빌드 스레드 N개는 코어 하나를 나눠 쓸 뿐이며
# (가속비 1.00 — concurrency-3·pdf-export-9), 손상 PDF의 MuPDF 크래시가 서버를 죽인다.
# 그래서 export 풀(PDF_EXPORT_MAX_CONCURRENT개)의 워커에서 돌리고 결과(리포트)만 받는다.
# 잡 락·캐시 판정은 호출부(derived)가 부모 프로세스에서 그대로 쥔다.


def _sweep_build_leftovers(directory: Path, patterns: tuple[str, ...]) -> None:
    """종료당한 빌드 워커는 finally를 못 돌린다 — 원자적 교체용 임시 파일을 부모가 지운다.
    같은 잡의 빌드는 잡 락으로 직렬이라 진행 중인 다른 빌드의 파일을 지우지 않는다."""
    for pattern in patterns:
        try:
            leftovers = list(directory.glob(pattern))
        except OSError:
            return
        for leftover in leftovers:
            leftover.unlink(missing_ok=True)


def _run_isolated_build(
    target: str, args: tuple, *, what: str, leftovers: tuple[Path, tuple[str, ...]],
):
    """빌드 작업을 export 풀 워커에서 — 상한 초과·워커 사망·옮길 수 없는 예외는 PdfExportError."""
    from .. import pdf_worker

    timeout = pdf_worker.export_build_timeout()
    try:
        return pdf_worker.run(target, args, pool=pdf_worker.POOL_EXPORT, timeout=timeout)
    except pdf_worker.PdfWorkerTimeout as error:
        _sweep_build_leftovers(*leftovers)
        limit = f"{timeout:g}초" if timeout else "상한"
        raise PdfExportError(
            f"{what}이 시간 상한({limit})을 넘어 중단했습니다 — 지나치게 크거나 복잡한 "
            "PDF입니다 (PDF_EXPORT_BUILD_TIMEOUT_S)"
        ) from error
    except pdf_worker.PdfWorkerCrashed as error:
        _sweep_build_leftovers(*leftovers)
        raise PdfExportError(
            f"{what} 중 처리 프로세스가 비정상 종료했습니다 — 손상되었거나 처리할 수 없는 "
            "PDF입니다"
        ) from error
    except pdf_worker.PdfWorkerRemoteError as error:
        raise PdfExportError(f"{what}에 실패했습니다 ({error.type_name})") from error


def build_translated_pdf(
    job_dir: Path, lang: str, *, fontfile: str = "",
) -> PdfExportResult:
    """source.pdf + layout.json + layout.{lang}.json → export.{lang}.pdf (원자적 교체).

    export 풀 워커에서 실행한다(위 '빌드 격리' — 시간 상한 PDF_EXPORT_BUILD_TIMEOUT_S).
    실패는 전부 PdfExportError(사용자에게 그대로 보여 줄 수 있는 문구)로 낸다.
    """
    return _run_isolated_build(
        "app.pipeline.pdf_export.build:translated_pdf_local",
        (job_dir, lang, fontfile),
        what="번역 PDF 생성",
        leftovers=(job_dir, (f".export.{lang}.*.tmp",)),
    )


def translated_pdf_local(job_dir: Path, lang: str, fontfile: str = "") -> PdfExportResult:
    """(PDF 워커) build_translated_pdf의 본체.

    실패는 전부 PdfExportError로 낸다. 예전에는 fitz.open만 감싸 페이지 조판·저장 중의
    MuPDF·조판 예외가 그대로 새어 /pdf·/page가 문구 없는 500을 내고 예열 스레드가
    traceback을 남겼다(호출부 derived._call_builder는 MuPDF 예외만 정규화한다). 원래 예외는
    로그와 예외 사슬(__cause__)에 남긴다.
    """
    try:
        return _build_translated_pdf(job_dir, lang, fontfile=fontfile)
    except PdfExportError:
        raise
    except Exception as error:  # noqa: BLE001 — MuPDF·조판 예외를 사용자 메시지로 정규화
        logger.warning("번역 PDF 생성 실패(lang=%s): %s", lang, type(error).__name__, exc_info=True)
        raise PdfExportError(
            f"번역 PDF를 만들 수 없습니다 — PDF 처리 중 오류가 났습니다 ({type(error).__name__})"
        ) from error


def _build_translated_pdf(job_dir: Path, lang: str, *, fontfile: str) -> PdfExportResult:
    fitz = quiet_fitz()
    src = job_dir / "source.pdf"
    orig_path = job_dir / "layout.json"
    trans_path = job_dir / f"layout.{lang}.json"
    for p, msg in (
        (src, "원본 PDF가 없습니다"),
        (orig_path, "레이아웃 정보가 없습니다"),
        (trans_path, "번역 레이아웃이 없습니다 — 먼저 번역을 실행하세요"),
    ):
        if not p.is_file():
            raise PdfExportError(msg)

    orig_page_list = _load_pages(orig_path)
    trans_pages = _load_pages(trans_path)
    _validate_layout_pair(orig_page_list, trans_pages)

    _enrich_source_fonts(src, orig_page_list)

    orig_pages = {p.get("page"): p for p in orig_page_list}
    fonts = _resolve_export_fonts(fontfile)

    result = PdfExportResult(path=job_dir / f"export.{lang}.pdf")
    if fonts.serif_ff is None and fonts.sans_ff is None:
        # 내장 CJK(비임베드 Dotum)는 라틴 포함 전 문자를 1em 전각으로 조판해
        # "R e i n f o r c e m e n t"처럼 자간이 찢어진다. 내보내기는 계속하되
        # (폰트 때문에 실패하지 않는다는 모듈 계약) 품질 열화를 리포트·UI 토스트
        # 파이프라인(report.json → X-UOCR-PDF-Warnings 헤더)으로 드러낸다.
        result.warnings.append(
            "한글 폰트 파일을 찾지 못해 PyMuPDF 내장 CJK(비임베드)로 대체합니다 — "
            "글자 간격 품질이 낮아집니다. 컨테이너에 fonts-noto-cjk를 설치하거나 "
            "PDF_EXPORT_FONT로 폰트 파일을 지정하세요"
        )
    try:
        doc = fitz.open(src)
    except Exception as e:  # noqa: BLE001 — mupdf 예외 타입이 다양함
        raise PdfExportError("원본 PDF를 열 수 없습니다") from e

    # 서브셋 폰트 파일은 doc.save()가 글리프를 임베드할 때까지 살아 있어야 한다.
    # 잡 디렉터리가 아니라 시스템 임시 경로에 둔다 — 강제 종료로 남더라도 잡
    # 산출물 목록을 더럽히지 않고 OS가 청소한다.
    with tempfile.TemporaryDirectory(prefix="uocr-font-") as fontdir:
        fonts = _subset_export_fonts(
            fonts,
            drawable_charset(trans_pages, orig_page_list),
            Path(fontdir),
        )
        _reserve_font_resource_names(doc, fonts)

        # 어느 페이지의 레이아웃이 **다른** 물리 페이지를 설명하는지 먼저 판정한다.
        # 리댁션은 되돌릴 수 없으므로 한 페이지라도 건드리기 전에 알아야 한다.
        misregistered = _unregistered_layout_pages(doc, orig_pages)
        try:
            for tpage in trans_pages:
                pno = tpage.get("page")
                opage = orig_pages.get(pno)
                if (
                    not isinstance(pno, int)
                    or not (1 <= pno <= doc.page_count)
                    or opage is None
                ):
                    continue
                try:
                    _process_page(
                        fitz, doc[pno - 1], pno, tpage, opage, fonts, result,
                        misregistered,
                    )
                except PdfExportError:
                    raise
                except Exception as error:  # noqa: BLE001 — 어느 페이지인지 문구에 남긴다
                    logger.warning("번역 PDF %d페이지 조판 실패", pno, exc_info=True)
                    raise PdfExportError(
                        f"{pno}페이지를 번역 PDF로 조판하지 못했습니다 ({type(error).__name__})"
                    ) from error
            tmp = job_dir / f".export.{lang}.{uuid.uuid4().hex}.tmp"
            try:
                _restore_space_tounicode(doc, fonts)
                doc.save(tmp, garbage=3, deflate=True)
                tmp.replace(result.path)
            except Exception as error:  # noqa: BLE001 — 디스크 만원·삭제된 잡 디렉터리 등
                logger.warning("번역 PDF 저장 실패(lang=%s)", lang, exc_info=True)
                raise PdfExportError(
                    f"번역 PDF를 저장하지 못했습니다 ({type(error).__name__})"
                ) from error
            finally:
                tmp.unlink(missing_ok=True)
        finally:
            doc.close()

    if result.warnings:
        for w in result.warnings[:5]:
            logger.warning("PDF 내보내기: %s", w)
    _write_export_report(job_dir, lang, result)
    return result


def build_dual_pdf(source_pdf: Path, translated_pdf: Path, out: Path) -> Path:
    """원본·번역 PDF를 페이지별 좌우 대조 스프레드로 원자적으로 묶는다.

    각 출력 페이지는 왼쪽에 원본, 오른쪽에 같은 번호의 번역 페이지를 원래 크기로
    배치한다. ``show_pdf_page``를 써서 래스터화하지 않으므로 텍스트 선택·벡터
    그림·원본 해상도를 보존한다. 두 입력의 페이지 수가 다르면 잘못 짝지은 대조본을
    만들지 않고 명시적으로 실패한다. export 풀 워커에서 실행한다(위 '빌드 격리').
    """
    return _run_isolated_build(
        "app.pipeline.pdf_export.build:dual_pdf_local",
        (source_pdf, translated_pdf, out),
        what="원문·번역 대조 PDF 생성",
        leftovers=(out.parent, (f".{out.stem}.*.tmp",)),
    )


def dual_pdf_local(source_pdf: Path, translated_pdf: Path, out: Path) -> Path:
    """(PDF 워커) build_dual_pdf의 본체."""
    for path, message in (
        (source_pdf, "원본 PDF가 없습니다"),
        (translated_pdf, "번역 PDF가 없습니다 — 먼저 번역 PDF를 생성하세요"),
    ):
        if not path.is_file():
            raise PdfExportError(message)
    # 출력 디렉터리(잡 디렉터리)는 만들지 않는다 — 삭제(DELETE·TTL GC)와 겹친 빌드가
    # parents=True로 meta.json 없는 잡 디렉터리를 되살려 영구 고아를 남겼다.
    if not out.parent.is_dir():
        raise PdfExportError("삭제된 작업입니다 — 대조 PDF를 만들 디렉터리가 없습니다")

    fitz = quiet_fitz()
    source = translated = dual = None
    tmp = out.parent / f".{out.stem}.{uuid.uuid4().hex}.tmp"
    try:
        source = fitz.open(str(source_pdf))
        translated = fitz.open(str(translated_pdf))
        if source.needs_pass or translated.needs_pass:
            raise PdfExportError("암호화된 PDF는 원문·번역 대조 내보내기를 지원하지 않습니다")
        if source.page_count != translated.page_count:
            raise PdfExportError(
                "원본과 번역 PDF의 페이지 수가 일치하지 않아 대조 PDF를 만들 수 없습니다"
            )
        if source.page_count == 0:
            raise PdfExportError("페이지가 없는 PDF는 대조 내보내기를 지원하지 않습니다")

        dual = fitz.open()
        for index in range(source.page_count):
            source_page = source[index]
            translated_page = translated[index]
            # show_pdf_page()는 원본 페이지의 /Rotate를 Form XObject에 자동으로
            # 승계하지 않는다. 저장본은 건드리지 않고 열린 문서 메모리에서만
            # 회전을 평탄화해, 회전된 원본·번역본도 각자 화면에 보이던 방향과
            # 크기로 대조 스프레드에 들어가게 한다.
            if source_page.rotation:
                source_page.remove_rotation()
            if translated_page.rotation:
                translated_page.remove_rotation()
            left = source_page.rect
            right = translated_page.rect
            left_width, left_height = float(left.width), float(left.height)
            right_width, right_height = float(right.width), float(right.height)
            if min(left_width, left_height, right_width, right_height) <= 0:
                raise PdfExportError(f"{index + 1}페이지 크기를 읽을 수 없습니다")

            page_height = max(left_height, right_height)
            page = dual.new_page(width=left_width + right_width, height=page_height)
            page.show_pdf_page(
                fitz.Rect(0, 0, left_width, left_height), source, index,
            )
            page.show_pdf_page(
                fitz.Rect(left_width, 0, left_width + right_width, right_height),
                translated,
                index,
            )
            # 대조할 두 면을 명확히 나누는 1pt 중앙선. 참조 Doclingo 대조 PDF와
            # 같은 구조이며 페이지 여백 안에만 있으므로 원문 콘텐츠를 가리지 않는다.
            page.draw_line(
                fitz.Point(left_width, 0),
                fitz.Point(left_width, page_height),
                color=(0, 0, 0),
                width=1,
            )

        dual.save(str(tmp), garbage=3, deflate=True)
        tmp.replace(out)
        return out
    except PdfExportError:
        raise
    except Exception as error:  # noqa: BLE001 — MuPDF 오류를 사용자 메시지로 정규화
        raise PdfExportError("원문·번역 대조 PDF를 만들 수 없습니다") from error
    finally:
        if dual is not None:
            dual.close()
        if translated is not None:
            translated.close()
        if source is not None:
            source.close()
        tmp.unlink(missing_ok=True)
