"""조판 계획 — 번역문이 원문 자리에 들어가는지 검증하고 계획만 만든다.

이 모듈의 어떤 함수도 페이지를 변경하지 않는다. 리댁션·삽입은 계획이 전부
성립한 뒤 `build`가 한 번에 적용한다.
"""
from __future__ import annotations

import re
from dataclasses import replace
from html import escape
from pathlib import Path
from statistics import median

from ..pdf import quiet_fitz
from .constants import (
    _BLOCK_GAP_PT,
    _FLOW_JOIN_GAP_PT,
    _FLOW_OBSTACLE_GAP_PT,
    _FLOW_UPWARD_SLACK_PT,
    _LISTING_COLUMN_GAP_PT,
    _MIN_BODY_FONT_PT,
    _MIN_FONT_PT,
    _SHRINK_STEPS,
    _SINGLE_LINE_SCALES,
)
from .fonts import _metrics_font
from .geometry import (
    _ink_collides,
    _page_bounds,
    _rect_horizontal_overlap,
    _shape_has_automatic_orphan,
    _textbox_ink_rect,
)
from .models import _FlowCandidate, _LineSegment, _Replacement, _SourceSpan, _TextFitPlan

_TEXT_ORIGIN_RE = re.compile(
    r"1 0 0 1 ([-+0-9.]+) ([-+0-9.]+) Tm"
)


def _plan_listing_lines(
    page,
    segments: tuple[_LineSegment, ...],
    fontname: str,
    fontfile: str | None,
    avoid_rects: list[object],
    block_index: int,
    *,
    bold: bool = False,
) -> tuple[list[_Replacement], int]:
    """세그먼트를 원문 좌표 그대로 조판한다. `(계획, 교체 대상 세그먼트 수)`.

    조판이 원문 자리를 벗어나지 않으므로 flow 재배치가 필요 없다. 폭이 모자란
    세그먼트만 건너뛰어 그 줄의 원문을 남기고 나머지는 회수한다.
    """
    fitz = quiet_fitz()
    try:
        font = _metrics_font(fontfile, fontname)
    except Exception:  # noqa: BLE001 — 메트릭을 못 읽으면 기존 경로에 맡긴다
        return [], 0
    page_limit = _page_bounds(page).y1 - _BLOCK_GAP_PT
    planned: list[_Replacement] = []
    changed = 0
    for segment in segments:
        if not segment.text or segment.text == segment.original:
            continue
        changed += 1
        segment_bold = bold or segment.bold
        available = segment.x1 - segment.x0 - _LISTING_COLUMN_GAP_PT
        if available <= 1.0:
            continue
        # 원문보다 커지는 축소는 없다. 원문 자체가 6pt 미만인 리스팅은 그
        # 크기까지만 허용해 본문 하한 때문에 통째로 놓치지 않게 한다.
        floor = min(segment.size, _MIN_BODY_FONT_PT)
        for scale in _SINGLE_LINE_SCALES:
            size = max(_MIN_FONT_PT, segment.size * scale)
            if size + 0.01 < floor:
                break
            width = (
                font.text_length(segment.text, fontsize=size)
                if fontfile
                else fitz.get_text_length(
                    segment.text, fontname=fontname, fontsize=size,
                )
            )
            if width > available:
                continue
            # 원문이 이미 점유하던 줄 띠 안에 머무는 대체는 이웃과 새 충돌을
            # 만들지 않는다. 첨자만 있는 얕은 span에서 띠가 0에 가까워지지
            # 않도록 한글 실측 잉크(0.88em/-0.18em)를 하한으로 더한다.
            ink = fitz.Rect(
                segment.x0,
                min(segment.band[0], segment.baseline - size * 0.88),
                segment.x0 + width,
                max(segment.band[1], segment.baseline + size * 0.18),
            )
            if segment_bold:
                ink += (-0.5, -0.5, 0.5, 0.5)
            if ink.y1 > page_limit or _ink_collides(ink, avoid_rects):
                continue
            source_rect = +segment.spans[0].rect
            for span in segment.spans[1:]:
                source_rect |= span.rect
            planned.append(_Replacement(
                _TextFitPlan(
                    +ink,
                    size,
                    False,
                    0,
                    segment_bold,
                    None,
                    (float(segment.x0), float(segment.baseline)),
                    ink,
                    (float(segment.x0), float(segment.baseline)),
                ),
                segment.text,
                "listing",
                None,
                fontname,
                fontfile,
                source_rect,
                block_index,
                None,
                tuple(span.rect for span in segment.spans),
            ))
            break
    return planned, changed


def _microfix_plan(
    fitz,
    rect,
    origin: tuple[float, float],
    text: str,
    fontsize: float,
    fontname: str,
    fontfile: str | None,
) -> _Replacement | None:
    """원문 한 행 안에서 더 짧은 안전 치환만 허용하는 보존 텍스트 계획."""
    try:
        font = _metrics_font(fontfile, fontname)
        width = (
            font.text_length(text, fontsize=fontsize)
            if fontfile
            else fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
        )
    except Exception:  # noqa: BLE001 — 미세 교정 실패 시 원문 보존
        return None
    if width > rect.width + 3.0:
        return None
    ink = fitz.Rect(
        origin[0],
        origin[1] - fontsize * font.ascender,
        origin[0] + width,
        origin[1] - fontsize * font.descender,
    )
    redact = +rect
    redact += (-0.3, -0.3, 0.3, 0.3)
    return _Replacement(
        _TextFitPlan(
            +redact,
            fontsize,
            origin=origin,
            ink_rect=ink,
            first_origin=origin,
        ),
        text,
        "microfix",
        redact,
        fontname,
        fontfile,
        rect,
    )


def _preserved_reference_microfixes(
    fitz,
    source_spans: list[_SourceSpan],
    fontname: str,
    fontfile: str | None,
) -> list[_Replacement]:
    """참고문헌 전체 reflow 없이 allowlist된 원문 조판 오류만 교정한다."""
    ordered = sorted(source_spans, key=lambda span: (span.origin[1], span.rect.x0))
    fixes: list[_Replacement] = []
    for index in range(len(ordered) - 2):
        first, slash, tail = ordered[index:index + 3]
        if (
            first.text == "T"
            and slash.text == "\\"
            and re.fullmatch(r'"\s*ulu\s+3:', tail.text, re.IGNORECASE)
            and max(
                abs(first.origin[1] - slash.origin[1]),
                abs(first.origin[1] - tail.origin[1]),
            ) <= 0.5
        ):
            union = +first.rect
            union.include_rect(slash.rect)
            union.include_rect(tail.rect)
            plan = _microfix_plan(
                fitz,
                union,
                first.origin,
                "Tülu 3:",
                max(_MIN_FONT_PT, median((first.size, slash.size, tail.size))),
                fontname,
                fontfile,
            )
            if plan is not None:
                fixes.append(plan)

    repeated_scheme = re.compile(r"^(https?://)\1$", re.IGNORECASE)
    for span in ordered:
        match = repeated_scheme.fullmatch(span.text)
        if match is None:
            continue
        plan = _microfix_plan(
            fitz,
            span.rect,
            span.origin,
            match.group(1),
            max(_MIN_FONT_PT, span.size),
            "cour",
            None,
        )
        if plan is not None:
            fixes.append(plan)
    return fixes


def _plan_shrink_to_fit(
    page, rect, text: str, base_pt: float, fontname: str, fontfile: str | None,
    *, max_rect=None, align: int = 0, bold: bool = False,
    lineheights: tuple[float | None, ...] = (None,),
    avoid_rects: list[object] | None = None,
    scales: tuple[float, ...] = _SHRINK_STEPS,
) -> _TextFitPlan | None:
    """Shape로 실제 삽입과 동일한 조판을 dry-run해 안전한 계획만 반환한다.

    원문 리댁션보다 먼저 실행하는 것이 핵심이다. 번역문이 아무 크기로도 들어가지
    않으면 None을 반환해 해당 원문 블록을 그대로 보존한다.
    """
    rot = page.rotation
    fitz = quiet_fitz()
    try:
        font = _metrics_font(fontfile, fontname)
    except Exception:  # noqa: BLE001 — 삽입기와 동일한 내장 CJK로 메트릭 폴백
        font = _metrics_font(None, "korea")

    page_limit = _page_bounds(page).y1 - _BLOCK_GAP_PT
    grown = +(max_rect if max_rect is not None else rect)
    if max_rect is None:
        grown.y1 = page_limit
    base = +rect
    base.y1 = min(base.y1, grown.y1)
    if base.y1 <= base.y0 + 0.5:
        return None
    candidates = [(base, False)]
    if grown.y1 > base.y1 + 0.5:
        candidates.append((grown, True))
    avoid = avoid_rects or []
    orphan_fallback: _TextFitPlan | None = None

    # OCR bbox는 원본 글리프에 딱 맞지만 CJK 폰트의 ascender/descender는 더 높다.
    # 같은 크기에서 원래 상자 → 충돌 없는 확장 상자 순으로 시도한 뒤에야 축소한다.
    # 그렇지 않으면 18pt 제목이 원래 상자의 12.6pt에 먼저 들어가 계층이 무너진다.
    for scale in scales:
        size = max(_MIN_FONT_PT, base_pt * scale)
        # 한국어 본문은 영문 기본 leading보다 넓은 1.44 이상 행간이 자연스럽다.
        # 같은 글자 크기에서 자연 행간 → 조밀한 행간 순으로 먼저 시도하고,
        # 그 뒤에만 폰트를 축소한다. 이 순서가 짧은 번역문 사이의 큰 흰 구멍과
        # 긴 번역문만 유난히 작아지는 현상을 동시에 줄인다.
        for candidate, expanded in candidates:
            for lineheight in lineheights:
                kwargs = {
                    "fontsize": size,
                    "fontname": fontname,
                    "fontfile": fontfile,
                    "align": align,
                    "rotate": rot,
                    "color": (0, 0, 0),
                    "lineheight": lineheight,
                }
                if bold:
                    # CJK 시스템 폰트가 단일 TTC/regular 파일인 환경에서도 제목 계층을
                    # 보존한다. fill+stroke는 글자 외곽만 약하게 굵게 하며 조판 폭은 같다.
                    kwargs.update({
                        "render_mode": 2,
                        "fill": (0, 0, 0),
                        # border_width는 pt가 아니라 fontsize 비율이다. 2%만 더해
                        # CJK 획이 서로 붙지 않는 얇은 합성 볼드를 만든다.
                        "border_width": 0.02,
                    })
                shape = page.new_shape()
                spare = shape.insert_textbox(candidate, text, **kwargs)
                if spare >= 0:
                    ink = _textbox_ink_rect(
                        page, candidate, shape, size, font, spare, bold,
                    )
                    if ink.y1 > page_limit:
                        continue
                    if _ink_collides(ink, avoid):
                        continue
                    first_origin = None
                    origins = _TEXT_ORIGIN_RE.findall(shape.text_cont)
                    if origins:
                        raw_x, raw_y = origins[0]
                        point = fitz.Point(float(raw_x), float(raw_y))
                        if page.rotation == 0:
                            point = point * page.transformation_matrix
                        first_origin = (float(point.x), float(point.y))
                    plan = _TextFitPlan(
                        +candidate,
                        size,
                        expanded,
                        align,
                        bold,
                        lineheight,
                        ink_rect=ink,
                        first_origin=first_origin,
                    )
                    if _shape_has_automatic_orphan(shape, text):
                        orphan_fallback = orphan_fallback or plan
                        continue
                    return plan
    # 단일 scale 호출은 상위 flow planner가 다음 scale을 시도하게 None을 준다.
    # 전체 scale 호출에서도 균형 해법이 전혀 없을 때만 읽을 수 있는 첫 계획을 쓴다.
    return orphan_fallback if len(scales) > 1 else None


def _noto_visible_ink_bounds(text: str) -> tuple[float, float] | None:
    """Noto CJK 문자열의 보수적 실측 ascender/descender를 반환한다."""
    if not text:
        return None
    ascender = 0.86
    descender = -0.12
    # Noto CJK에서 직접 실측한 흔한 논문용 문장부호만 허용한다. Unicode의
    # punctuation 전체를 같은 bbox로 취급하면 괄호·쉼표의 하단 잉크를 놓치며,
    # 드문 기호는 모양이 제각각이므로 전역 폰트 메트릭으로 폴백하는 편이 안전하다.
    measured_safe_punctuation = frozenset(
        ".:!?<>-\u2010\u2011\u2012\u2013\u2014\u2026+=*&%#'\""
    )
    measured_mid_punctuation = frozenset(",;[]_/@\\")
    measured_deep_punctuation = frozenset("(){}")
    measured_superscripts = frozenset("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
    for char in text:
        codepoint = ord(char)
        if (
            char.isspace()
            or "0" <= char <= "9"
            or 0x1100 <= codepoint <= 0x11FF
            or 0x3130 <= codepoint <= 0x318F
            or 0xA960 <= codepoint <= 0xA97F
            or 0xAC00 <= codepoint <= 0xD7A3
            or 0xD7B0 <= codepoint <= 0xD7FF
            or 0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
        ):
            continue
        if char.isascii() and char.isalpha():
            # 기본 Latin은 한글보다 높지 않지만 g/j/p/q/y의 descender는
            # 약 -0.26em까지 내려온다.
            if char.lower() in "gjpqy":
                descender = min(descender, -0.27)
            continue
        if char in measured_safe_punctuation:
            continue
        if char in measured_mid_punctuation:
            # 쉼표·세미콜론·대괄호는 약 -0.17em까지 내려간다.
            descender = min(descender, -0.20)
            continue
        if char in measured_deep_punctuation:
            # 괄호류는 실측 약 -0.206em이며 반올림 여유를 더한다.
            descender = min(descender, -0.23)
            continue
        if char in measured_superscripts:
            ascender = max(ascender, 1.00)
            descender = min(descender, -0.27)
            continue
        # 결합 악센트·Latin Extended·Greek 및 미실측 문장부호는 Noto의 전역
        # 1.151/-0.286em bbox로 되돌린다. 예: Ắ는 1.0em보다 위까지 올라간다.
        return None
    return ascender, descender


def _plan_single_line(
    page,
    rect,
    text: str,
    base_pt: float,
    fontname: str,
    fontfile: str | None,
    *,
    max_rect=None,
    align: int = 0,
    bold: bool = False,
    avoid_rects: list[object] | None = None,
    scales: tuple[float, ...] = _SINGLE_LINE_SCALES,
) -> _TextFitPlan | None:
    """한 줄 번역을 원문 baseline 크기로 배치한다.

    `insert_textbox()`는 CJK ascender/descender 전체가 얕은 OCR bbox 안에 들어가야
    성공으로 판정하므로, 실제로는 한 줄이 넉넉히 들어가는 제목·목록 항목도 60~70%로
    축소하는 문제가 있다. 줄바꿈이 필요 없고 가로 폭이 맞는 경우에는 폰트 메트릭으로
    baseline을 계산해 `insert_text()` 경로를 사용한다.
    """
    # insert_text()의 origin은 회전 전 PDF 좌표지만 표시 bbox는 회전 좌표다.
    # 여기서 단순 baseline을 계산하면 90/270도 페이지의 끝 글자가 재단된다.
    # 회전 페이지는 실제 textbox dry-run과 동일한 조판 경로만 사용한다.
    if page.rotation != 0 or not text or "\n" in text:
        return None
    fitz = quiet_fitz()
    try:
        font = _metrics_font(fontfile, fontname)
    except Exception:  # noqa: BLE001 — textbox 폴백이 있으므로 품질 경로만 포기
        return None
    vertical = +(max_rect if max_rect is not None else rect)
    working = +rect
    working.y1 = min(working.y1, vertical.y1)
    if working.y1 <= working.y0 + 0.5:
        return None
    avoid = avoid_rects or []
    for scale in scales:
        size = max(_MIN_FONT_PT, base_pt * scale)
        # PyMuPDF의 내장 CJK 폰트는 ``Font.text_length()``와 실제
        # ``Page.insert_text()``가 서로 다른 폭을 보고한다. 예를 들어 korea
        # 폰트의 Font API는 ASCII 공백을 약 0.26em으로 계산하지만 PDF 삽입기는
        # 모든 문자를 1em 전각으로 인코딩한다. 그 값을 그대로 가운데 정렬에 쓰면
        # Linux 폴백 환경에서 제목이 오른쪽으로 밀린다. 파일 폰트는 Font 메트릭을,
        # 내장 폰트는 실제 삽입기와 계약이 같은 get_text_length()를 사용한다.
        width = (
            font.text_length(text, fontsize=size)
            if fontfile
            else fitz.get_text_length(text, fontname=fontname, fontsize=size)
        )
        if width > working.width + 0.25:
            continue
        if align == 1:
            x = working.x0 + (working.width - width) / 2
        elif align == 2:
            x = working.x1 - width
        else:
            x = working.x0
        layout_ascender = float(font.ascender)
        layout_descender = float(font.descender)
        # Noto CJK는 실제 한글 잉크(약 0.85em/-0.12em)보다 em bbox를
        # 1.151em/-0.286em으로 크게 보고한다. 얕은 표 행과 저자 소속 한 줄에서
        # 그 전역 메트릭을 쓰면 보이는 글자는 충분히 들어가도 통째로 보존된다.
        # 다만 Latin descender·diacritic는 이 범위를 벗어나므로 글자 종류별
        # 보수적 실측 상한이 있는 문자열에만 줄인다. 알 수 없는 문자와 bold는
        # 전역 bbox로 충돌을 보수적으로 판정한다.
        visible_bounds = _noto_visible_ink_bounds(text)
        if (
            fontfile
            and not bold
            and layout_ascender - layout_descender > 1.30
            and "Noto" in font.name
            and "CJK" in font.name
            and visible_bounds is not None
        ):
            measured_ascender, measured_descender = visible_bounds
            layout_ascender = min(layout_ascender, measured_ascender)
            layout_descender = max(layout_descender, measured_descender)
        baseline = working.y0 + size * layout_ascender
        glyph_bottom = baseline - size * layout_descender
        # max_rect은 다음 블록 앞 5pt에서 끝난다. 폰트 bbox의 descender는 한글
        # 글리프가 실제로 쓰지 않는 하단까지 포함하므로 그 안전 여백만 허용한다.
        if glyph_bottom > vertical.y1:
            continue
        ink = fitz.Rect(
            x,
            baseline - size * layout_ascender,
            x + width,
            glyph_bottom,
        )
        if bold:
            ink += (-0.5, -0.5, 0.5, 0.5)
        if _ink_collides(ink, avoid):
            continue
        return _TextFitPlan(
            +working,
            size,
            False,
            align,
            bold,
            None,
            (float(x), float(baseline)),
            ink,
            (float(x), float(baseline)),
        )
    return None


def _rich_prefix_markup(text: str, bold_prefix: tuple[str, str]) -> str:
    """run-in prefix만 한 번 기록하는 HTML 조각을 만든다."""
    def rich_escape(value: str) -> str:
        """한국어 하이픈 합성어가 글자 중간에서 꺾이지 않는 HTML로 escape."""
        chunks: list[str] = []
        cursor = 0
        for match in re.finditer(r"(?<![가-힣])([가-힣]+(?:[-‐‑][가-힣]+)+)(?![가-힣])", value):
            chunks.append(escape(value[cursor:match.start()]))
            chunks.append(f'<span class="nowrap">{escape(match.group(1))}</span>')
            cursor = match.end()
        chunks.append(escape(value[cursor:]))
        return "".join(chunks)

    leading, prefix = bold_prefix
    first, separator, rest_lines = text.partition("\n")
    start = len(leading)
    if not first.startswith(leading) or first[start:start + len(prefix)] != prefix:
        return ""
    remainder = first[start + len(prefix):]
    first_html = f"{rich_escape(leading)}<b>{rich_escape(prefix)}</b>{rich_escape(remainder)}"
    if separator:
        first_html += "<br>" + "<br>".join(
            rich_escape(line) for line in rest_lines.splitlines()
        )
    return first_html


def _rich_prefix_css(
    fontfile: str,
    fontsize: float,
    align: int,
    lineheight: float,
) -> tuple[str, object]:
    fitz = quiet_fitz()
    font_path = Path(fontfile)
    archive = fitz.Archive(str(font_path), font_path.name)
    text_align = {1: "center", 2: "right", 3: "justify"}.get(align, "left")
    css = (
        f"@font-face {{font-family:uocr-rich;src:url('{font_path.name}');}}"
        "body,p{margin:0;padding:0;}"
        f"body{{font-family:uocr-rich;font-size:{fontsize:.6f}pt;"
        f"line-height:{lineheight:.6f};text-align:{text_align};color:#000;}}"
        # 1/255 blue is visually black but forces Story to split the prefix into
        # a distinct span. The scratch span is never committed; its origin is
        # used to insert each run exactly once with a synthetic bold stroke.
        "b{font-family:uocr-rich;font-weight:400;color:#000001;}"
        ".nowrap{white-space:nowrap;}"
    )
    return css, archive


def _reconstruct_rich_runs(
    source: str,
    raw_runs: tuple[tuple[float, float, str, bool], ...],
) -> tuple[tuple[float, float, str, bool], ...] | None:
    """Story가 NUL/비분리 하이픈으로 바꾼 글자를 원문 순서로 되돌린다."""
    source_index = 0
    rebuilt: list[tuple[float, float, str, bool]] = []

    def equivalent(left: str, right: str) -> bool:
        if left.isspace() and right.isspace():
            return True
        if left in {"-", "‐", "‑"} and right in {"-", "‐", "‑"}:
            return True
        return left == right

    for x, y, raw_text, is_prefix in raw_runs:
        output: list[str] = []
        for raw_char in raw_text:
            while source_index < len(source) and source[source_index] == "\n":
                source_index += 1
            # MuPDF Story는 폰트에 따라 숫자와 일부 기호를 NUL뿐 아니라
            # BMP private-use 문자로 돌려준다. 둘 다 원문의 같은 순서 글자로
            # 복원해 실제 PDF에 tofu가 들어가는 것을 막는다.
            if raw_char == "\x00" or 0xE000 <= ord(raw_char) <= 0xF8FF:
                while source_index < len(source) and source[source_index].isspace():
                    source_index += 1
                if source_index >= len(source):
                    return None
                output.append(source[source_index])
                source_index += 1
                continue
            if not raw_char.isspace():
                # 자동 줄바꿈에서 빠진 원문 공백은 다음 run의 origin이 대신한다.
                while source_index < len(source) and source[source_index].isspace():
                    source_index += 1
            if source_index >= len(source) or not equivalent(
                source[source_index], raw_char,
            ):
                return None
            output.append(source[source_index])
            source_index += 1
        rebuilt.append((x, y, "".join(output), is_prefix))
    while source_index < len(source) and source[source_index].isspace():
        source_index += 1
    if source_index != len(source):
        return None

    # Story는 폰트에 없는 숫자를 PUA run으로 따로 나눌 수 있다. 복원한 각 run을
    # 원래 x-origin에 따로 삽입하면 PUA의 넓은 advance가 남아 `Llama- 3`처럼
    # 보인다. 같은 baseline·스타일의 인접 run은 원문 순서대로 한 번에 삽입해
    # 실제 복원 문자열의 폭으로 자연스럽게 이어지게 한다.
    merged: list[tuple[float, float, str, bool]] = []
    for x, y, run_text, is_prefix in rebuilt:
        if (
            merged
            and abs(merged[-1][1] - y) <= 0.1
            and merged[-1][3] == is_prefix
        ):
            previous = merged[-1]
            merged[-1] = (
                previous[0], previous[1], previous[2] + run_text, is_prefix,
            )
        else:
            merged.append((x, y, run_text, is_prefix))
    return tuple(merged)


def _plan_rich_prefix(
    page,
    rect,
    text: str,
    bold_prefix: tuple[str, str],
    base_pt: float,
    fontfile: str | None,
    *,
    max_rect=None,
    align: int = 0,
    lineheights: tuple[float | None, ...] = (None,),
    avoid_rects: list[object] | None = None,
    scales: tuple[float, ...] = _SHRINK_STEPS,
) -> _TextFitPlan | None:
    """HTML rich text를 scratch page에서 실제 조판해 prefix-only bold를 검증한다."""
    if not fontfile or page.rotation != 0:
        return None
    markup = _rich_prefix_markup(text, bold_prefix)
    if not markup:
        return None
    fitz = quiet_fitz()
    page_limit = _page_bounds(page).y1 - _BLOCK_GAP_PT
    grown = +(max_rect if max_rect is not None else rect)
    if max_rect is None:
        grown.y1 = page_limit
    base = +rect
    base.y1 = min(base.y1, grown.y1)
    if base.y1 <= base.y0 + 0.5:
        return None
    candidates = [(base, False)]
    if grown.y1 > base.y1 + 0.5:
        candidates.append((grown, True))
    avoid = avoid_rects or []
    orphan_fallback: _TextFitPlan | None = None
    for scale in scales:
        size = max(_MIN_FONT_PT, base_pt * scale)
        for candidate, expanded in candidates:
            for requested_lineheight in lineheights:
                lineheight = float(requested_lineheight or 1.44)
                scratch = fitz.open()
                try:
                    scratch_page = scratch.new_page(
                        width=page.mediabox.width,
                        height=page.mediabox.height,
                    )
                    css, archive = _rich_prefix_css(
                        fontfile, size, align, lineheight,
                    )
                    spare, actual_scale = scratch_page.insert_htmlbox(
                        candidate,
                        markup,
                        css=css,
                        archive=archive,
                        scale_low=1,
                    )
                    if spare < 0 or actual_scale < 0.999:
                        continue
                    scratch_lines = [
                        line
                        for block in scratch_page.get_text("dict").get("blocks", [])
                        for line in block.get("lines", [])
                        if line.get("bbox")
                    ]
                    line_rects = [fitz.Rect(line.get("bbox")) for line in scratch_lines]
                    if not line_rects:
                        continue
                    raw_runs = tuple(
                        (
                            float((span.get("origin") or span["bbox"][:2])[0]),
                            float((span.get("origin") or span["bbox"][:2])[1]),
                            str(span.get("text") or ""),
                            bool(int(span.get("color") or 0)),
                        )
                        for block in scratch_page.get_text("dict").get("blocks", [])
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                        if str(span.get("text") or "")
                    )
                    rich_runs = _reconstruct_rich_runs(text, raw_runs)
                    if not rich_runs or not any(run[3] for run in rich_runs):
                        continue
                    ink = +line_rects[0]
                    for line_rect in line_rects[1:]:
                        ink.include_rect(line_rect)
                    ink += (-0.5, -0.5, 0.5, 0.5)
                    if ink.y1 > page_limit:
                        continue
                    if _ink_collides(ink, avoid):
                        continue
                    plan = _TextFitPlan(
                        +candidate,
                        size,
                        expanded,
                        align,
                        False,
                        lineheight,
                        ink_rect=ink,
                        rich_runs=rich_runs,
                    )
                    line_texts = [
                        "".join(str(span.get("text") or "") for span in line.get("spans", []))
                        for line in scratch_lines
                    ]
                    explicit_last = re.sub(
                        r"\s+", "", text.splitlines()[-1],
                    ) if text else ""
                    short_last_line = bool(
                        len(explicit_last) > 5
                        and len(line_texts) > 1
                        and 0 < len(re.sub(r"\s+", "", line_texts[-1])) <= 5
                    )
                    if short_last_line:
                        orphan_fallback = orphan_fallback or plan
                        continue
                    return plan
                finally:
                    scratch.close()
    return orphan_fallback if len(scales) > 1 else None


def _flow_components(candidates: list[_FlowCandidate]) -> list[list[_FlowCandidate]]:
    """같은 단에서 맞닿은 번역 블록을 위에서 아래로 묶는다."""
    components: list[list[_FlowCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: (item.rect.y0, item.rect.x0)):
        choices: list[tuple[float, float, float, int]] = []
        for index, component in enumerate(components):
            previous = component[-1]
            overlap = _rect_horizontal_overlap(previous.rect, candidate.rect)
            required = min(previous.rect.width, candidate.rect.width) * 0.35
            gap = candidate.rect.y0 - previous.rect.y1
            if overlap < required or gap > _FLOW_JOIN_GAP_PT:
                continue
            center_distance = abs(
                (previous.rect.x0 + previous.rect.x1)
                - (candidate.rect.x0 + candidate.rect.x1)
            )
            choices.append((overlap / max(1.0, required), -abs(gap), -center_distance, index))
        if choices:
            components[max(choices)[-1]].append(candidate)
        else:
            components.append([candidate])
    return components


def _flow_gap(previous: _FlowCandidate, current: _FlowCandidate) -> float:
    """제목 계층은 보존하면서 본문 흐름의 과도한 흰 구멍을 피한다."""
    if current.block_type == "title":
        return 8.0
    if previous.block_type == "title":
        return 5.0
    if current.block_type in {"caption", "image_caption", "table_caption"}:
        return 2.0
    return 3.0


def _plan_flow_group(
    page,
    candidates: list[_FlowCandidate],
    fixed_rects: list[object],
) -> list[_Replacement] | None:
    """인접 블록을 순차 reflow하고 전부 성공할 때만 계획을 확정한다."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.rect.y0, item.rect.x0))
    column_x0 = min(item.rect.x0 for item in ordered)
    column_x1 = max(item.rect.x1 for item in ordered)
    column_width = max(1.0, column_x1 - column_x0)
    original_start = min(item.rect.y0 for item in ordered)
    original_bottom = max(item.rect.y1 for item in ordered)

    relevant_fixed = [
        rect for rect in fixed_rects
        if rect is not None
        and not rect.is_empty
        and _rect_horizontal_overlap(
            quiet_fitz().Rect(column_x0, rect.y0, column_x1, rect.y1), rect,
        ) >= min(column_width, rect.width) * 0.15
    ]
    page_area_bounds = _page_bounds(page)
    lower_bound = page_area_bounds.y0 + _BLOCK_GAP_PT
    for obstacle in relevant_fixed:
        if obstacle.y1 <= original_start + 0.5:
            lower_bound = max(lower_bound, obstacle.y1 + _FLOW_OBSTACLE_GAP_PT)
    shifted_start = max(
        lower_bound,
        original_start - _FLOW_UPWARD_SLACK_PT,
    )
    preferred_start = max(original_start, lower_bound)
    starts = [preferred_start]
    if shifted_start < preferred_start - 0.5:
        starts.append(shifted_start)

    end = page_area_bounds.y1 - _BLOCK_GAP_PT
    for obstacle in relevant_fixed:
        if obstacle.y0 > original_bottom + 0.5:
            end = min(end, obstacle.y0 - _FLOW_OBSTACLE_GAP_PT)
    if end <= shifted_start + 0.5:
        return None

    # 이 flow에서 가장 작은 블록이 _MIN_BODY_FONT_PT 아래로 내려가는 축소는
    # 읽을 수 없는 번역이 된다 — 시도하지 않고 원문을 보존한다.
    smallest_pt = min(item.base_pt for item in ordered)
    readable_scales = tuple(
        scale for scale in _SHRINK_STEPS
        if smallest_pt * scale >= _MIN_BODY_FONT_PT
    ) or (_SHRINK_STEPS[0],)
    scale_sets = [(scale,) for scale in readable_scales]
    # 모든 블록이 같은 scale에서 불가능할 때만 개별 first-fit을 허용한다.
    scale_sets.append(readable_scales)
    for start in starts:
        for compact in (False, True):
            for scales in scale_sets:
                planned: list[_Replacement] = []
                reserved = list(relevant_fixed)
                cursor = start
                failed = False
                for index, candidate in enumerate(ordered):
                    gap = 0.0 if index == 0 else _flow_gap(ordered[index - 1], candidate)
                    preferred_y0 = candidate.rect.y0 if not compact else cursor + gap
                    y0 = max(start if index == 0 else cursor + gap, preferred_y0)
                    if y0 >= end - 0.5:
                        failed = True
                        break
                    placement = +candidate.rect
                    delta = y0 - placement.y0
                    placement.y0 = y0
                    placement.y1 = min(end, placement.y1 + delta)
                    if placement.y1 <= placement.y0 + 0.5:
                        placement.y1 = min(
                            end,
                            placement.y0 + max(candidate.rect.height, candidate.base_pt * 1.6),
                        )
                    vertical = +placement
                    vertical.y1 = end
                    rich_prefix_applied = False
                    if candidate.bold_prefix is not None:
                        plan = _plan_rich_prefix(
                            page,
                            placement,
                            candidate.text,
                            candidate.bold_prefix,
                            candidate.base_pt,
                            candidate.fontfile,
                            max_rect=vertical,
                            align=candidate.align,
                            lineheights=candidate.lineheights,
                            avoid_rects=reserved,
                            scales=scales,
                        )
                        rich_prefix_applied = plan is not None
                        if plan is None:
                            plan = _plan_single_line(
                                page,
                                placement,
                                candidate.text,
                                candidate.base_pt,
                                candidate.fontname,
                                candidate.fontfile,
                                max_rect=vertical,
                                align=candidate.align,
                                bold=candidate.bold,
                                avoid_rects=reserved,
                                scales=_SINGLE_LINE_SCALES,
                            )
                            if plan is None:
                                plan = _plan_shrink_to_fit(
                                    page,
                                    placement,
                                    candidate.text,
                                    candidate.base_pt,
                                    candidate.fontname,
                                    candidate.fontfile,
                                    max_rect=vertical,
                                    align=candidate.align,
                                    bold=candidate.bold,
                                    lineheights=candidate.lineheights,
                                    avoid_rects=reserved,
                                    scales=scales,
                                )
                    else:
                        plan = _plan_single_line(
                            page,
                            placement,
                            candidate.text,
                            candidate.base_pt,
                            candidate.fontname,
                            candidate.fontfile,
                            max_rect=vertical,
                            align=candidate.align,
                            bold=candidate.bold,
                            avoid_rects=reserved,
                            # 한 줄 원문이 1–2% 폭 차이로 두 줄이 되는 것보다
                            # 4–12% 완만한 축소로 한 줄 리듬을 지키는 편이 낫다.
                            scales=_SINGLE_LINE_SCALES,
                        )
                        if plan is None:
                            plan = _plan_shrink_to_fit(
                                page,
                                placement,
                                candidate.text,
                                candidate.base_pt,
                                candidate.fontname,
                                candidate.fontfile,
                                max_rect=vertical,
                                align=candidate.align,
                                bold=candidate.bold,
                                lineheights=candidate.lineheights,
                                avoid_rects=reserved,
                                scales=scales,
                            )
                    if plan is None or plan.ink_rect is None:
                        failed = True
                        break
                    moved = abs(placement.y0 - candidate.rect.y0) > 0.5
                    plan = replace(plan, expanded=plan.expanded or moved)
                    planned.append(_Replacement(
                        plan,
                        candidate.text,
                        "text",
                        candidate.redact_rects[0] if len(candidate.redact_rects) == 1 else None,
                        candidate.fontname,
                        candidate.fontfile,
                        candidate.source_rect,
                        candidate.block_index,
                        candidate.bold_prefix if rich_prefix_applied else None,
                        candidate.redact_rects,
                    ))
                    reserved.append(plan.ink_rect)
                    cursor = plan.ink_rect.y1
                if not failed and len(planned) == len(ordered):
                    return planned
    return None
