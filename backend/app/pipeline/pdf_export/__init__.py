"""번역 PDF 내보내기 — 원본 PDF 위에 번역 텍스트를 재배치한다.

원리(레이아웃 보존 번역): layout.json(원문)과 layout.{lang}.json(번역본)을 블록
단위로 비교해 내용이 실제로 바뀐 텍스트 블록만 원본 페이지에서 리댁션(텍스트만
제거, 이미지·그래픽 보존)하고 같은 자리에 번역 텍스트를 삽입한다. 번역 엔진이
마스킹으로 원문을 유지한 블록(수식·식별자·페이지 번호)과 그림·표는 원본
글리프가 그대로 남으므로 시각 품질이 보존된다.

폰트: PDF_EXPORT_FONT(파일 경로) → 시스템 한글 폰트 후보 → fc-list 런타임
탐색 → PyMuPDF 내장 CJK("korea", Droid Sans Fallback) 순으로 폴백한다. 내장
폰트는 어떤 환경에서도 존재하므로 내보내기가 폰트 때문에 실패하지는 않지만,
전 문자를 1em 전각으로 조판하는 비임베드 폰트라 채택 시 경고를 리포트에 남긴다.
"""
from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from ..layout import estimate_font_size_cqw
from ..pdf import quiet_fitz
from . import (
    constants,
    report,
    models,
    text,
    fonts,
    geometry,
    spans,
    tables,
    fitting,
    subset,
    build,
)
from .constants import (
    _BLOCK_GAP_PT,
    _BODY_LINEHEIGHTS,
    _CAPTION_LINEHEIGHTS,
    _FLOW_JOIN_GAP_PT,
    _FLOW_OBSTACLE_GAP_PT,
    _FLOW_UPWARD_SLACK_PT,
    _LISTING_COLUMN_GAP_PT,
    _MAX_FONT_PT,
    _MAX_TABLE_CELLS,
    _MIN_BODY_FONT_PT,
    _MIN_FONT_PT,
    _MIN_TABLE_FONT_PT,
    _PRESERVE_TYPES,
    _REDACT_CHUNK,
    _REPLACEABLE_TYPES,
    _SHRINK_STEPS,
    _SINGLE_LINE_SCALES,
    _SPECIALIST_TYPES,
    _TABLE_RULE_TEXT_GAP_PT,
    _TITLE_LINEHEIGHTS,
    _VERTICAL_SKIP,
)
from .report import (
    PDF_EXPORT_FORMAT_VERSION,
    PdfExportError,
    PdfExportResult,
)
from .models import (
    _FlowCandidate,
    _LineSegment,
    _RawTableCell,
    _Replacement,
    _SourceSpan,
    _TableCell,
    _TextFitPlan,
)
from .text import (
    _LATEX_COMMANDS,
    _LATEX_COMMAND_RE,
    _LATEX_SUB_RE,
    _LATEX_SUP_RE,
    _LATEX_WRAPPER_RE,
    _LITERAL_LBRACE,
    _LITERAL_RBRACE,
    _PORTABLE_SYMBOL_FALLBACKS,
    _SUBSCRIPT_MAP,
    _SUPERSCRIPT_MAP,
    _TAG_RE,
    _TITLE_PREFIX_RE,
    _UNICODE_SUPERSCRIPT_ASCII,
    _WS_RE,
    _latex_command,
    _normalize_inline_spacing,
    _plain_text,
    _protect_trailing_words,
    _restore_title_prefix,
    _script_text,
)
from .fonts import (
    _SERIF_NAME_HINT,
    _SYSTEM_FONT_CANDIDATES,
    _SYSTEM_SANS_FONT_CANDIDATES,
    _balance_title_text,
    _document_font_resource_names,
    _fontconfig_candidates,
    _metrics_font,
    _metrics_font_cached,
    _portable_text_for_font,
    _resolve_font,
    _unique_font_resource_name,
)
from .geometry import (
    _TEXT_BASELINE_RE,
    _TEXT_HEX_RUN_RE,
    _free_growth_rect,
    _ink_collides,
    _page_bounds,
    _rect_horizontal_overlap,
    _rect_overlap_area,
    _shape_has_automatic_orphan,
    _textbox_ink_rect,
)
from .spans import (
    _align_ocr_lines,
    _align_ocr_spans,
    _assign_source_spans,
    _block_rect,
    _leading_bold_prefix,
    _listing_segments,
    _ownership_text,
    _reflow_flattened_text,
    _source_span_matches_rect,
    _source_span_records,
    _source_span_rects,
    _source_text_rect,
    _source_text_rects,
    _visual_line_clusters,
    _visual_lines,
)
from .tables import (
    _TableParser,
    _grid_boundaries,
    _horizontal_table_rules,
    _table_cell_rects,
    _table_cell_source_style,
    _table_cells,
    _table_matrix,
)
from .fitting import (
    _TEXT_ORIGIN_RE,
    _flow_components,
    _flow_gap,
    _microfix_plan,
    _noto_visible_ink_bounds,
    _plan_flow_group,
    _plan_listing_lines,
    _plan_rich_prefix,
    _plan_shrink_to_fit,
    _plan_single_line,
    _preserved_reference_microfixes,
    _reconstruct_rich_runs,
    _rich_prefix_css,
    _rich_prefix_markup,
)
from .build import (
    _hide_visible_link_borders,
    _insert_fitted_text,
    _load_pages,
    _normalize_repeated_scheme_links,
    _validate_layout_pair,
    build_dual_pdf,
    build_translated_pdf,
    logger,
)

# 단일 모듈 시절의 공개·사설 심볼을 전부 같은 이름으로 재노출한다.
# api.py와 테스트가 `from app.pipeline.pdf_export import ...`로 쓰는 경로다.
__all__ = [
    "estimate_font_size_cqw",
    "quiet_fitz",
    "logger",
    "PDF_EXPORT_FORMAT_VERSION",
    "_REPLACEABLE_TYPES",
    "_SPECIALIST_TYPES",
    "_PRESERVE_TYPES",
    "_VERTICAL_SKIP",
    "_TAG_RE",
    "_WS_RE",
    "_LATEX_WRAPPER_RE",
    "_LATEX_SUP_RE",
    "_LATEX_SUB_RE",
    "_LATEX_COMMAND_RE",
    "_TEXT_BASELINE_RE",
    "_TEXT_ORIGIN_RE",
    "_TEXT_HEX_RUN_RE",
    "_SUPERSCRIPT_MAP",
    "_SUBSCRIPT_MAP",
    "_UNICODE_SUPERSCRIPT_ASCII",
    "_LATEX_COMMANDS",
    "_PORTABLE_SYMBOL_FALLBACKS",
    "_LITERAL_LBRACE",
    "_LITERAL_RBRACE",
    "_TITLE_PREFIX_RE",
    "_SYSTEM_FONT_CANDIDATES",
    "_SYSTEM_SANS_FONT_CANDIDATES",
    "_SERIF_NAME_HINT",
    "_SHRINK_STEPS",
    "_SINGLE_LINE_SCALES",
    "_MIN_FONT_PT",
    "_MIN_BODY_FONT_PT",
    "_MAX_FONT_PT",
    "_MAX_TABLE_CELLS",
    "_MIN_TABLE_FONT_PT",
    "_TABLE_RULE_TEXT_GAP_PT",
    "_BODY_LINEHEIGHTS",
    "_CAPTION_LINEHEIGHTS",
    "_TITLE_LINEHEIGHTS",
    "_BLOCK_GAP_PT",
    "_REDACT_CHUNK",
    "_FLOW_JOIN_GAP_PT",
    "_LISTING_COLUMN_GAP_PT",
    "_FLOW_UPWARD_SLACK_PT",
    "_FLOW_OBSTACLE_GAP_PT",
    "PdfExportError",
    "PdfExportResult",
    "_TextFitPlan",
    "_Replacement",
    "_FlowCandidate",
    "_LineSegment",
    "_TableCell",
    "_RawTableCell",
    "_SourceSpan",
    "_TableParser",
    "_fontconfig_candidates",
    "_resolve_font",
    "_metrics_font_cached",
    "_metrics_font",
    "_document_font_resource_names",
    "_unique_font_resource_name",
    "_load_pages",
    "_script_text",
    "_latex_command",
    "_plain_text",
    "_protect_trailing_words",
    "_normalize_inline_spacing",
    "_portable_text_for_font",
    "_restore_title_prefix",
    "_balance_title_text",
    "_table_cells",
    "_table_matrix",
    "_page_bounds",
    "_rect_horizontal_overlap",
    "_rect_overlap_area",
    "_free_growth_rect",
    "_ink_collides",
    "_textbox_ink_rect",
    "_shape_has_automatic_orphan",
    "_grid_boundaries",
    "_horizontal_table_rules",
    "_table_cell_rects",
    "_table_cell_source_style",
    "_validate_layout_pair",
    "_block_rect",
    "_source_span_records",
    "_source_span_rects",
    "_ownership_text",
    "_source_span_matches_rect",
    "_assign_source_spans",
    "_leading_bold_prefix",
    "_visual_line_clusters",
    "_visual_lines",
    "_align_ocr_spans",
    "_align_ocr_lines",
    "_reflow_flattened_text",
    "_listing_segments",
    "_plan_listing_lines",
    "_microfix_plan",
    "_preserved_reference_microfixes",
    "_normalize_repeated_scheme_links",
    "_hide_visible_link_borders",
    "_source_text_rects",
    "_source_text_rect",
    "_plan_shrink_to_fit",
    "_noto_visible_ink_bounds",
    "_plan_single_line",
    "_rich_prefix_markup",
    "_rich_prefix_css",
    "_reconstruct_rich_runs",
    "_plan_rich_prefix",
    "_flow_components",
    "_flow_gap",
    "_plan_flow_group",
    "_insert_fitted_text",
    "build_translated_pdf",
    "build_dual_pdf",
]

_SUBMODULES = (
    constants, report, models, text, fonts, geometry, spans, tables, fitting,
    subset, build,
)


class _PdfExportPackage(_ModuleType):
    """monkeypatch 의미를 단일 모듈 시절과 동일하게 유지하는 패키지 모듈 타입.

    분할 전에는 `pdf_export._resolve_font`를 바꾸면 모듈 전역이 하나뿐이라 모든
    호출자가 즉시 대체 구현을 봤다. 패키지로 쪼갠 뒤에는 각 서브모듈이 자기
    전역을 참조하므로 패키지 속성만 바꾸면 아무 효과가 없다. 같은 이름을 가진
    서브모듈 전역까지 함께 갱신해 그 의미를 되돌린다 — monkeypatch의 undo도
    setattr이라 원상 복구까지 그대로 전파된다.
    """

    def __setattr__(self, name: str, value: object) -> None:
        _ModuleType.__setattr__(self, name, value)
        for sub in _SUBMODULES:
            if hasattr(sub, name):
                setattr(sub, name, value)


# 서브모듈 임포트가 끝난 뒤에 교체한다 — 초기화 중의 속성 대입은 전파 대상이 아니다.
_sys.modules[__name__].__class__ = _PdfExportPackage
