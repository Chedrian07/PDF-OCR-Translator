"""페이지 기하 — 사각형 겹침, 확장 가능한 여백, 실제 잉크 bbox."""
from __future__ import annotations

import re

from ..pdf import quiet_fitz
from .constants import _BLOCK_GAP_PT

_TEXT_BASELINE_RE = re.compile(r"1 0 0 1 [-+0-9.]+ ([-+0-9.]+) Tm")
_TEXT_HEX_RUN_RE = re.compile(r"<([0-9A-Fa-f]+)>")


def _page_bounds(page):
    """블록 좌표와 같은 비회전 내부 좌표계의 페이지 표시 영역(CropBox) Rect.

    `page.mediabox`는 CropBox를 무시한 PDF 원좌표라 CropBox<MediaBox인 문서에서
    실제 페이지 하단보다 아래를 가드 기준으로 삼는다. `_block_rect`가 만드는
    좌표와 같은 공간으로 맞추려면 `page.rect`를 derotation까지 거쳐야 한다
    (회전 페이지에서 `page.rect.y1`을 그대로 쓰면 가로/세로가 뒤바뀐다).
    """
    bounds = page.rect * page.derotation_matrix
    bounds.normalize()
    return bounds


def _rect_horizontal_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _rect_overlap_area(a, b) -> float:
    """두 Rect의 교차 면적. 겹치지 않으면 0 — fitz `&`의 빈 Rect 의존 없이 계산."""
    return _rect_horizontal_overlap(a, b) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _free_growth_rect(page, rect, obstacles: list[object]) -> object:
    """현재 블록 바로 아래의 빈 세로 영역까지만 확장 가능한 Rect를 반환한다.

    가로로 사실상 겹치지 않는 사이드바/다른 단은 장애물로 보지 않는다. 같은 단의
    다음 블록·표·그림·푸터 앞 5pt에서 멈추므로 번역문 확장이 이웃 내용을 덮지 않는다.
    """
    # 페이지 끝 자체도 장애물이다. 푸터 검출이 빠져도 마지막 행과 재단선 사이에
    # 최소 여백을 남긴다.
    limit = _page_bounds(page).y1 - _BLOCK_GAP_PT
    for other in obstacles:
        if other is rect or other.is_empty:
            continue
        overlap = _rect_horizontal_overlap(rect, other)
        if overlap < min(rect.width, other.width) * 0.15:
            continue
        # 현재 블록보다 위에서 시작한 장애물은 앞 단락/배경이다. 아래에서 시작한
        # 첫 장애물은 OCR bbox와 이미 겹쳐 있어도 안전 하단으로 사용한다. 예전처럼
        # 원 bbox를 그대로 반환하면 겹친 두 OCR bbox에 번역문도 그대로 겹쳐졌다.
        if other.y0 <= rect.y0 + 0.5:
            continue
        limit = min(limit, other.y0 - _BLOCK_GAP_PT)
    grown = +rect
    # 이 함수는 이름 그대로 *확장* 상한만 계산한다. OCR bbox가 다음 블록과
    # 닿거나 겹친다고 원래 상자를 위로 잘라내면 원문의 정상 한 줄조차 들어가지
    # 않는다. 실제 글리프 충돌은 planner의 avoid_rects가 별도로 판정한다.
    grown.y1 = max(rect.y1, limit)
    return grown


def _ink_collides(ink_rect, avoid_rects: list[object]) -> bool:
    """실제 사용 높이 추정치가 앞서 예약된 번역 글자와 닿는지 판정한다."""
    if ink_rect is None:
        return False
    padded = +ink_rect
    padded.y0 -= 0.5
    padded.y1 += 0.5
    return any(_rect_overlap_area(padded, other) > 0.01 for other in avoid_rects)


def _textbox_ink_rect(page, candidate, shape, size: float, font, spare: float, bold: bool):
    """dry-run Shape의 baseline들로 실제 textbox 글리프 bbox를 계산한다.

    ``insert_textbox``의 반환값은 실제 CJK span bbox가 아니라 조판 여유 공간이다.
    Shape가 아직 commit되지 않았어도 ``text_cont``에는 각 줄의 PDF-space baseline
    matrix가 들어 있으므로, 회전 없는 일반 논문 페이지에서는 폰트 ascender/
    descender와 결합해 실제 출력 bbox를 정확히 예측할 수 있다. 회전 페이지는
    보수적인 기존 추정으로 폴백한다.
    """
    fitz = quiet_fitz()
    if page.rotation != 0:
        # 회전 textbox의 진행축은 PDF y축이 아니다. `spare`를 세로 여백처럼
        # 계산하면 90도 문장의 실제 170pt 높이를 0.5pt로 오판할 수 있다.
        # insert_textbox가 성공한 candidate 전체는 실제 글리프를 반드시 포함하므로
        # 회전 페이지에서는 이 보수적 bbox를 예약해 후속 블록 충돌을 막는다.
        ink = +candidate
        if bold:
            ink += (-0.5, -0.5, 0.5, 0.5)
        return ink
    if page.rotation == 0:
        baselines = []
        for raw_y in _TEXT_BASELINE_RE.findall(shape.text_cont):
            point = fitz.Point(0, float(raw_y)) * page.transformation_matrix
            baselines.append(point.y)
        if baselines:
            ink = fitz.Rect(
                candidate.x0,
                min(y - size * font.ascender for y in baselines),
                candidate.x1,
                max(y - size * font.descender for y in baselines),
            )
            if bold:
                ink += (-0.5, -0.5, 0.5, 0.5)
            return ink

    # 회전/비표준 Shape command는 실제 하단을 과소평가하지 않는 보수적 폴백.
    glyph_height = max(1.0, float(font.ascender - font.descender))
    ink = +candidate
    ink.y1 = max(
        ink.y0 + 0.5,
        candidate.y1 - spare + size * glyph_height + 2.0,
    )
    return ink


def _shape_has_automatic_orphan(shape, text: str, *, max_glyphs: int = 5) -> bool:
    """파일 CJK textbox의 자동 생성된 매우 짧은 마지막 행을 판정한다."""
    explicit_last = re.sub(r"\s+", "", text.splitlines()[-1]) if text else ""
    if 0 < len(explicit_last) <= max_glyphs:
        return False
    counts: list[int] = []
    for command in shape.text_cont.splitlines():
        if "TJ" not in command:
            continue
        hex_runs = _TEXT_HEX_RUN_RE.findall(command)
        if not hex_runs:
            continue
        # Identity-H CJK font는 한 글자를 2-byte CID(4 hex digits)로 쓴다.
        counts.append(sum(len(run) // 4 for run in hex_runs))
    if len(counts) <= 1:
        return False
    return 0 < counts[-1] <= max_glyphs
