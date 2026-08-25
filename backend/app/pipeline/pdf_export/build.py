"""내보내기 오케스트레이션 — 계획을 모아 리댁션하고 번역문을 삽입한다.

페이지마다 (1) 모든 블록의 삽입 계획 수집 → (2) 원문 텍스트 일괄 리댁션 →
(3) 번역문 삽입 순서를 지킨다. 계획이 전부 끝나기 전에는 어떤 원문도 지우지
않으므로 조판이 실패한 블록은 원문 글리프가 그대로 남는다.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, replace
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
    _PRESERVE_TYPES,
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
from .geometry import _rect_overlap_area
from .models import _FlowCandidate, _Replacement, _SourceSpan, _TableCell, _TextFitPlan
from .report import PdfExportError, PdfExportResult
from .spans import (
    _assign_source_spans,
    _block_rect,
    _leading_bold_prefix,
    _listing_segments,
    _reflow_flattened_text,
    _source_span_matches_rect,
    _source_span_records,
    _source_text_rects,
)
from .tables import _table_cell_rects, _table_cell_source_style, _table_cells
from .text import (
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


def _page_visual_obstacles(fitz, page, block_rects, oblocks):
    """(래스터 인스턴스, 그림 영역, 확장 장애물). 리댁션 전에 1회만 수집한다."""
    # 래스터 인스턴스는 리댁션 '이전'에 1회만 수집한다 — apply_redactions
    # 이후의 get_image_info()는 스테일 캐시를 반환할 수 있다(실측).
    try:
        raster_rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
    except Exception:  # noqa: BLE001 — 이미지 목록 실패가 텍스트 교체를 막지 않는다
        raster_rects = []
    # 벡터 표·그래프·구분선도 번역문 확장 영역의 장애물이다. path의 rect가
    # 수평/수직 0폭 선이면 먼저 1pt 패딩해 유효한 사각형으로 만든다.
    drawing_rects = []
    try:
        for drawing in page.get_drawings():
            bbox = drawing.get("rect")
            if bbox is None:
                continue
            drawing_rect = fitz.Rect(bbox)
            drawing_rect += (-0.5, -0.5, 0.5, 0.5)
            drawing_rect &= page.mediabox
            if not drawing_rect.is_empty:
                drawing_rects.append(drawing_rect)
    except Exception:  # noqa: BLE001 — 벡터 목록 실패가 텍스트 교체를 막지 않는다
        drawing_rects = []
    # 그림 위 텍스트 방어용 영역: layout image 블록 ∪ 래스터 인스턴스.
    # 페이지의 85% 이상을 덮는 영역은 전면 스캔 배경으로 간주해 제외한다
    # — 스캔 문서에서 모든 블록 교체가 생략되는 사고 방지.
    page_area = page.rect.width * page.rect.height or 1.0
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
        r for r in raster_rects if 0 < r.width * r.height < page_area * 0.85
    ]
    fixed_visuals = image_regions + drawing_rects
    return raster_rects, image_regions, fixed_visuals


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
    cell_rects, grid_trusted = _table_cell_rects(
        ctx.page, table_rect, old_cells, row_count, col_count,
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
        (old_cell, new_cell, cell_rect)
        for old_cell, new_cell, cell_rect in zip(
            old_cells, new_cells, cell_rects,
        )
        if (
            (new_text := _plain_text(new_cell.text))
            and new_text != (old_text := _plain_text(old_cell.text))
            and new_text.casefold() != old_text.casefold()
        )
    ]
    changed_cells = len(changed_cell_specs)
    failed_cell: _TableCell | None = None
    for old_cell, new_cell, cell_rect in changed_cell_specs:
        old_text = _plain_text(old_cell.text)
        new_text = _portable_text_for_font(
            _plain_text(new_cell.text), ctx.fonts.table_ff,
        )
        base_pt, cell_align, cell_bold, source_redact = (
            _table_cell_source_style(ctx.page, cell_rect, ctx.source_records)
        )
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
            )
        source_size = base_pt / 1.03
        readable_floor = max(
            _MIN_TABLE_FONT_PT,
            min(source_size, source_size * 0.80),
        )
        if plan is None or plan.fontsize + 0.01 < readable_floor:
            failed_cell = old_cell
            break
        table_targets.append(_Replacement(
            plan,
            new_text,
            "table",
            source_redact,
            ctx.fonts.table_name,
            ctx.fonts.table_ff,
            cell_rect,
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
    old = _plain_text(str(ob.get("content") or ""))
    new = _plain_text(str(tb.get("content") or ""))
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
        new = _balance_title_text(
            new,
            rect.width,
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
    owned_rects = [span.rect for span in owned_records]
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
    if listing_segments:
        listing_avoid = [span.rect for span in ctx.unowned_source]
        listing_avoid.extend(
            span.rect
            for owner, spans in ctx.source_ownership.items()
            if owner != block_index
            for span in spans
        )
        listing_avoid.extend(ctx.fixed_visuals)
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
        _source_text_rects(ctx.page, rect, owned_rects),
        rect,
        None if bold else _leading_bold_prefix(owned_records, new),
        reflow_text,
        listing_segments,
    )


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
        if block_type not in _REPLACEABLE_TYPES:
            if block_type in _SPECIALIST_TYPES:
                result.specialist_kept[block_type] = (
                    result.specialist_kept.get(block_type, 0) + 1
                )
            continue
        if (tb.get("vertical") or ob.get("vertical")) in _VERTICAL_SKIP:
            result.keep("vertical")
            result.specialist_kept["vertical"] = result.specialist_kept.get("vertical", 0) + 1
            continue
        candidate = _plan_text_block(
            ctx, block_index, block_type, ob, tb, targets, result,
        )
        if candidate is not None:
            flow_candidates.append(candidate)
    return targets, flow_candidates, repeated_scheme_link_rects


def _plan_flow_targets(
    ctx: _PageContext, flow_candidates: list[_FlowCandidate],
    targets: list[_Replacement], result: PdfExportResult,
) -> None:
    """같은 단의 인접 본문을 원자적으로 reflow하고, 실패하면 단계적으로 회수한다."""
    # 일반 텍스트는 페이지에서 모두 수집한 뒤 같은 단의 인접 블록을
    # 원자적으로 reflow한다. 이 단계 전에는 어떤 원문도 redaction하지 않는다.
    for component in _flow_components(flow_candidates):
        component_indices = {candidate.block_index for candidate in component}
        fixed_rects = [span.rect for span in ctx.unowned_source]
        fixed_rects.extend(
            span.rect
            for owner, spans in ctx.source_ownership.items()
            if owner not in component_indices
            for span in spans
        )
        fixed_rects.extend(ctx.fixed_visuals)
        fixed_rects.extend(
            target.plan.ink_rect
            for target in targets
            if target.plan.ink_rect is not None
        )
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
            planned = _plan_flow_group(ctx.page, variant, fixed_rects)
            if planned is not None:
                break
        if planned is None:
            # 최후 수단: 한 블록이 안 들어간다고 같은 단의 나머지 문단까지
            # 원문으로 되돌리면 사용자에게는 문단 대여섯 개가 통째로
            # 미번역으로 보인다. 위에서 아래로 개별 배치해 들어가는 만큼만
            # 회수하고, 아직 계획하지 않은 이웃의 원문은 보존될 수 있으므로
            # 장애물로 예약해 번역문이 그 위에 겹치지 않게 한다.
            planned = []
            for position, candidate in enumerate(component):
                obstacles = list(fixed_rects)
                obstacles.extend(
                    target.plan.ink_rect
                    for target in planned
                    if target.plan.ink_rect is not None
                )
                obstacles.extend(
                    span.rect
                    for other in component[position + 1:]
                    for span in ctx.source_ownership.get(other.block_index, [])
                )
                single = None
                for text in (candidate.text, candidate.reflow_text):
                    if text is None:
                        continue
                    single = _plan_flow_group(
                        ctx.page, [replace(candidate, text=text)], obstacles,
                    )
                    if single:
                        break
                if single:
                    planned.extend(single)
                    continue
                # 흘려 넣기가 전부 실패해도 리스팅·표는 줄 단위로는 제자리에
                # 들어간다. 폭이 모자란 줄만 원문으로 남기고 나머지를 회수한다.
                listing_targets, listing_changed = _plan_listing_lines(
                    ctx.page,
                    candidate.listing_segments,
                    candidate.fontname,
                    candidate.fontfile,
                    obstacles,
                    candidate.block_index,
                    bold=candidate.bold,
                )
                if listing_targets:
                    planned.extend(listing_targets)
                    missing = listing_changed - len(listing_targets)
                    if missing:
                        result.keep("listing_line_no_fit", missing)
                        result.warnings.append(
                            f"p{ctx.pno}: 블록 {candidate.block_index + 1}의 "
                            f"{missing}줄 교체 생략(줄 폭 부족) — 그 줄만 원문 보존"
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


def _apply_page_redactions(fitz, page, targets, raster_rects) -> None:
    """원문 텍스트(그리고 이모지의 이미지 절반)만 지운다 — 그래픽은 보존."""
    # 2) 원문 텍스트 리댁션 (이미지·그래픽 보존) — 삽입 전에 일괄 적용
    source_rects = []
    for target in targets:
        # 삽입 bbox가 아래 빈 공간으로 커져도 실제 원문 bbox만 지운다.
        # 확장 사각형 전체를 리댁션하면 인접한 원문 글리프가 함께 사라질 수 있다.
        target_redactions = target.redact_rects or (
            target.redact_rect if target.redact_rect is not None else target.plan.rect,
        )
        for rr in target_redactions:
            page.add_redact_annot(rr)
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
    page.apply_redactions(
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
        for bbox in emoji_boxes:
            page.add_redact_annot(bbox)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_NONE,
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


def _process_page(fitz, page, pno, tpage, opage, fonts, result) -> None:
    """한 페이지를 계획 → 리댁션 → 삽입 순서로 처리한다."""
    _hide_visible_link_borders(page)
    width = tpage.get("width") or 1
    height = tpage.get("height") or 1
    aspect = height / width if width else 1.0

    # 1) 교체 대상 수집. 모든 블록 사각형을 먼저 만들고, 일반 텍스트가
    # 공간을 늘릴 때 같은 단의 다음 블록과 충돌하지 않는 하단을 계산한다.
    oblocks = opage.get("blocks", [])
    tblocks = tpage.get("blocks", [])
    block_rects = [_block_rect(fitz, page, b.get("bbox")) for b in oblocks]
    source_records = _source_span_records(fitz, page)
    source_ownership, unowned_source, ambiguous_blocks = _assign_source_spans(
        page, block_rects, oblocks, source_records,
    )
    raster_rects, image_regions, fixed_visuals = _page_visual_obstacles(
        fitz, page, block_rects, oblocks,
    )
    ctx = _PageContext(
        fitz, page, pno, aspect, oblocks, tblocks, block_rects,
        source_records, source_ownership, unowned_source, ambiguous_blocks,
        image_regions, fixed_visuals, fonts,
    )
    targets, flow_candidates, repeated_scheme_link_rects = _plan_page_targets(
        ctx, result,
    )
    _plan_flow_targets(ctx, flow_candidates, targets, result)
    if not targets:
        return

    # 반복 scheme의 첫 링크 annotation은 아래 redaction에서 사라질 수
    # 있다. 겹친 annotation으로 bad URI 집합을 식별할 수 있을 때 같은
    # URI를 가진 wrapped 링크까지 먼저 정상화한다.
    if repeated_scheme_link_rects:
        _normalize_repeated_scheme_links(page, repeated_scheme_link_rects)

    _apply_page_redactions(fitz, page, targets, raster_rects)
    _insert_page_targets(page, targets, result)


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


def build_translated_pdf(
    job_dir: Path, lang: str, *, fontfile: str = "",
) -> PdfExportResult:
    """source.pdf + layout.json + layout.{lang}.json → export.{lang}.pdf (원자적 교체)."""
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

    _reserve_font_resource_names(doc, fonts)

    try:
        for tpage in trans_pages:
            pno = tpage.get("page")
            opage = orig_pages.get(pno)
            if not isinstance(pno, int) or not (1 <= pno <= doc.page_count) or opage is None:
                continue
            _process_page(fitz, doc[pno - 1], pno, tpage, opage, fonts, result)
        tmp = job_dir / f".export.{lang}.{uuid.uuid4().hex}.tmp"
        try:
            doc.save(tmp, garbage=3, deflate=True)
            tmp.replace(result.path)
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
    만들지 않고 명시적으로 실패한다.
    """
    for path, message in (
        (source_pdf, "원본 PDF가 없습니다"),
        (translated_pdf, "번역 PDF가 없습니다 — 먼저 번역 PDF를 생성하세요"),
    ):
        if not path.is_file():
            raise PdfExportError(message)

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

        out.parent.mkdir(parents=True, exist_ok=True)
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
