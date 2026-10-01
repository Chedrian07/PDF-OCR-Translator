"""PDF 텍스트 레이어 블록의 읽기 순서 — 다단 조판과 조각난 블록 처리.

textlayer 엔진과 OCR 실패 페이지의 텍스트 레이어 복구가 같이 쓴다.

## 왜 sort=True를 쓰지 않는가

PyMuPDF `get_text(sort=True)`는 블록을 (y, x) 좌표로 정렬한다. 2단 논문에서는
같은 높이의 왼쪽·오른쪽 단 블록이 번갈아 나와 문단이 교차로 뒤섞인다(실측: Keshav
'How to Read a Paper' 1쪽 — 'ABSTRACT' 바로 뒤에 오른쪽 단의 '4. Glance over the
references'가 온다). 한 단 문서에서도 여백의 arXiv 세로 스탬프가 본문 사이에 끼어든다.

## 방법

1. 기준은 **내용 스트림 순서**(sort=False)다 — LaTeX·워드프로세서 대부분이 읽기
   순서대로 그린다.
2. 단 구분(`_find_gutter`): 페이지 가운데 영역에서 "가로지르는 블록의 높이 합"이
   가장 작은 x를 찾는다. 양쪽에 본문이 충분하고(각 20% 이상) 가로지르는 블록이 적으면
   (25% 이하) 다단으로 본다. 가로지르는 블록(제목·저자·전폭 그림 캡션·쪽 번호)은
   단 흐름의 **경계**다 — 그 위·아래 띠(band)마다 왼쪽 단 → 오른쪽 단 순으로 읽는다.
   단 안은 내용 순서를 지키고, 3단 이상은 같은 방법을 재귀로 적용한다.
3. 조각 병합(`merge_fragments`): MuPDF 1.28은 가운데 정렬 제목·디스플레이 수식을
   더 잘게 쪼갠다(2504.19874v1.pdf: 블록 700 → 974). 한 문장이 블록 둘로 갈리면
   번역 유닛도 둘로 갈려 번역 품질이 떨어지므로, 읽기 순서상 이웃한 블록이
   (a) 같은 줄의 조각이거나 (b) 같은 문단의 다음 줄이면 합친다.

순수 기하 함수(`order_blocks`)는 PyMuPDF 없이 테스트할 수 있다.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, replace

# 다단 판정 임계값
_MIN_BLOCKS_FOR_COLUMNS = 4
_GUTTER_SEARCH = 0.25        # 텍스트 폭의 가운데 50%에서만 거터를 찾는다
_MIN_SIDE_SHARE = 0.20       # 거터 양쪽 블록 높이 합이 각각 전체의 20% 이상
_MAX_CROSS_SHARE = 0.25      # 거터를 가로지르는 블록 높이 합은 전체의 25% 이하
_MIN_GUTTER_PT = 3.0         # 거터 양쪽 블록 사이의 최소 빈 폭
_MAX_DEPTH = 3               # 재귀 상한(3단 이상 조판)
# 조각 병합 임계값 (글꼴 크기·줄 높이 대비 비율)
_SAME_LINE_OVERLAP = 0.4     # 같은 줄: 세로 겹침 / 낮은 쪽 높이 (근호·첨자는 줄 위로 솟는다)
_SAME_LINE_GAP = 0.6         # 같은 줄: 가로 간격 상한 = 글꼴 크기 × 0.6
_NEXT_LINE_GAP = 0.6         # 다음 줄: 세로 간격 범위 = ± 줄 높이 × 0.6
_COLUMN_OVERLAP = 0.5        # 다음 줄: 가로 겹침 / 좁은 쪽 폭
_SIZE_TOLERANCE = 0.08       # 다음 줄: 대표 글꼴 크기 상대 차이 상한
_CENTERED_PT = 3.0           # 대문자로 시작하는 다음 줄은 가운데 정렬 제목의 둘째 줄일 때만
_MAX_MERGE_BLOCKS = 4000     # 병적 페이지(블록 폭증)는 병합 없이 순서만 정한다

# 문단을 끝내는 꼬리(마침표·물음표·느낌표·콜론·세미콜론, 닫는 괄호·따옴표 허용)
_TERMINAL = re.compile(r"[.!?:;。！？][\"'”’)\]]*\s*$")
# 새 문단·항목을 여는 머리(글머리표, 번호 목록 `1.`·`(a)`·`[3]`)
_ITEM_START = re.compile(
    r"^\s*(?:[•◦▪▸►‣∙·\-–—*]\s|\(?\d{1,3}[.):]\s|\([a-zA-Z]\)\s|\[\d{1,3}\]\s)"
)


@dataclass(frozen=True)
class TextBlock:
    """표시(회전 반영) 좌표계의 텍스트 블록."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float = 0.0            # 대표 글꼴 크기(가장 많은 글자를 실은 크기)
    line_height: float = 0.0     # 평균 줄 높이
    last_y0: float = 0.0         # 마지막 줄의 세로 범위·끝 x (같은 줄 조각 판정용)
    last_y1: float = 0.0
    last_x1: float = 0.0
    horizontal: bool = True      # 줄이 표시 좌표에서 가로로 흐르는가
    tabular: bool = False        # 줄들이 옆으로 늘어선 표 행·셀 묶음인가
    order: int = 0               # 내용 스트림 순서

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


# ── 단 구분 ───────────────────────────────────────────────────────────────


def _find_gutter(blocks: list[TextBlock]) -> float | None:
    """다단 조판의 단 사이 x(거터). 한 단이면 None.

    가로지르는 블록 높이 합이 최소인 x를 고른다 — 제목·쪽 번호처럼 거터를 넘는
    블록이 몇 개 있어도 단 구조를 알아본다(빈 세로 띠만 찾으면 가운데 제목 하나에
    무너진다). 판정은 블록 경계의 정렬 + 누적합으로 O(n log n)이다.
    """
    if len(blocks) < _MIN_BLOCKS_FOR_COLUMNS:
        return None
    left_edge = min(b.x0 for b in blocks)
    right_edge = max(b.x1 for b in blocks)
    width = right_edge - left_edge
    if width <= 0:
        return None
    lo = left_edge + width * _GUTTER_SEARCH
    hi = right_edge - width * _GUTTER_SEARCH

    total = sum(b.height for b in blocks)
    by_x1 = sorted((b.x1, b.height) for b in blocks)
    x1s = [x for x, _ in by_x1]
    left_cum = [0.0]
    for _, h in by_x1:
        left_cum.append(left_cum[-1] + h)
    by_x0 = sorted((b.x0, b.height) for b in blocks)
    x0s = [x for x, _ in by_x0]
    right_cum = [0.0]
    for _, h in reversed(by_x0):
        right_cum.append(right_cum[-1] + h)
    right_cum.reverse()  # right_cum[i] = x0s[i:]의 높이 합

    edges = sorted({b.x0 for b in blocks} | {b.x1 for b in blocks})
    best: tuple[float, float, float] | None = None  # (가로지름, -빈 폭, 중앙 거리)
    best_x: float | None = None
    for a, c in zip(edges, edges[1:]):
        x = (a + c) / 2
        if not lo <= x <= hi:
            continue
        n_left = bisect.bisect_right(x1s, x)
        n_right = len(x0s) - bisect.bisect_left(x0s, x)
        left = left_cum[n_left]
        right = right_cum[len(x0s) - n_right]
        cross = total - left - right
        if left < total * _MIN_SIDE_SHARE or right < total * _MIN_SIDE_SHARE:
            continue
        if cross > total * _MAX_CROSS_SHARE or n_left < 2 or n_right < 2:
            continue
        gap = x0s[len(x0s) - n_right] - x1s[n_left - 1]
        if gap < _MIN_GUTTER_PT:
            continue
        key = (round(cross, 3), -gap, abs(x - (left_edge + right_edge) / 2))
        if best is None or key < best:
            best, best_x = key, x
    return best_x


def order_blocks(blocks: list[TextBlock], _depth: int = 0) -> list[TextBlock]:
    """내용 순서를 기본으로, 다단이면 띠별로 왼쪽 단 → 오른쪽 단 순서로 재배열.

    세로로 흐르는 블록(여백의 arXiv 스탬프·회전된 축 라벨)은 단 판정에서 빼고 페이지
    끝에 내용 순서대로 둔다. 페이지 대부분이 세로 텍스트면 내용 순서를 그대로 쓴다.
    """
    blocks = sorted(blocks, key=lambda b: b.order)
    if _depth >= _MAX_DEPTH:
        return blocks
    flowing = [b for b in blocks if b.horizontal]
    if len(flowing) * 2 <= len(blocks):
        return blocks
    rotated = [b for b in blocks if not b.horizontal]
    gutter = _find_gutter(flowing)
    if gutter is None:
        return blocks if not rotated else flowing + rotated
    blocks = flowing
    # 띠 경계는 가로지르는 블록의 세로 중심 — bisect가 쓰므로 중심 기준으로 정렬한다
    spanning = sorted(
        (b for b in blocks if b.x0 < gutter < b.x1), key=lambda b: (b.cy, b.order)
    )
    span_centers = [b.cy for b in spanning]
    bands_left: list[list[TextBlock]] = [[] for _ in range(len(spanning) + 1)]
    bands_right: list[list[TextBlock]] = [[] for _ in range(len(spanning) + 1)]
    for b in blocks:
        if b.x0 < gutter < b.x1:
            continue
        band = bisect.bisect_left(span_centers, b.cy)
        (bands_left if b.x1 <= gutter else bands_right)[band].append(b)
    out: list[TextBlock] = []
    for k in range(len(spanning) + 1):
        out.extend(order_blocks(bands_left[k], _depth + 1))
        out.extend(order_blocks(bands_right[k], _depth + 1))
        if k < len(spanning):
            out.append(spanning[k])
    return out + rotated


# ── 조각 병합 ─────────────────────────────────────────────────────────────


def _same_line_fragment(a: TextBlock, b: TextBlock) -> bool:
    """b가 a의 마지막 줄 바로 오른쪽에 이어지는 같은 줄 조각인가(디스플레이 수식 등)."""
    overlap = min(a.last_y1, b.y1) - max(a.last_y0, b.y0)
    low = min(max(a.last_y1 - a.last_y0, 1.0), b.height)
    if overlap < low * _SAME_LINE_OVERLAP:
        return False
    size = max(a.size, b.size, 1.0)
    gap = b.x0 - a.last_x1
    return -size <= gap <= size * _SAME_LINE_GAP


def _next_line_of_paragraph(a: TextBlock, b: TextBlock) -> bool:
    """b가 a와 같은 문단의 다음 줄인가(가운데 정렬 제목의 둘째 줄·수식에 끊긴 문장)."""
    line_h = max(a.line_height, 1.0)
    if not -line_h * _NEXT_LINE_GAP <= b.y0 - a.y1 <= line_h * _NEXT_LINE_GAP:
        return False
    narrow = min(a.width, b.width)
    if narrow <= 0 or min(a.x1, b.x1) - max(a.x0, b.x0) < narrow * _COLUMN_OVERLAP:
        return False
    if a.size <= 0 or b.size <= 0 or abs(a.size - b.size) > max(a.size, b.size) * _SIZE_TOLERANCE:
        return False
    if a.tabular or b.tabular:
        return False  # 표 행끼리 합치면 행 구조가 한 문단으로 뭉개진다
    if _TERMINAL.search(a.text) or _ITEM_START.match(b.text):
        return False
    first = b.text.lstrip()[:1]
    if first.isupper():
        # 왼쪽 정렬 본문에서 대문자로 시작하는 다음 블록은 새 문단(Proof. …)이나 새
        # 항목이다. 가운데 정렬 제목·저자 블록의 둘째 줄(가운데가 같고 왼쪽 끝은
        # 다르다)일 때만 이어 붙인다.
        centered = abs((a.x0 + a.x1) / 2 - (b.x0 + b.x1) / 2) <= _CENTERED_PT
        return centered and abs(a.x0 - b.x0) > _CENTERED_PT
    return True


def _join(a: TextBlock, b: TextBlock, same_line: bool) -> TextBlock:
    lines_a = max(a.y1 - a.y0, 1.0) / max(a.line_height, 1.0)
    lines_b = max(b.y1 - b.y0, 1.0) / max(b.line_height, 1.0)
    return replace(
        a,
        x0=min(a.x0, b.x0),
        y0=min(a.y0, b.y0),
        x1=max(a.x1, b.x1),
        y1=max(a.y1, b.y1),
        text=f"{a.text} {b.text}" if same_line else f"{a.text}\n{b.text}",
        size=a.size if len(a.text) >= len(b.text) else b.size,
        line_height=(
            a.line_height if same_line
            else (a.line_height * lines_a + b.line_height * lines_b) / (lines_a + lines_b)
        ),
        last_y0=min(a.last_y0, b.y0) if same_line else b.last_y0,
        last_y1=max(a.last_y1, b.y1) if same_line else b.last_y1,
        last_x1=b.x1 if same_line else b.last_x1,
        tabular=a.tabular or b.tabular,
    )


def merge_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """읽기 순서상 이웃한 같은 줄 조각·같은 문단 다음 줄을 한 블록으로 합친다."""
    if len(blocks) > _MAX_MERGE_BLOCKS:
        return list(blocks)
    out: list[TextBlock] = []
    for b in blocks:
        if out and out[-1].horizontal and b.horizontal:
            a = out[-1]
            if _same_line_fragment(a, b):
                out[-1] = _join(a, b, same_line=True)
                continue
            if _next_line_of_paragraph(a, b):
                out[-1] = _join(a, b, same_line=False)
                continue
        out.append(b)
    return out


# ── PyMuPDF 어댑터 ────────────────────────────────────────────────────────


def extract_text_blocks(page, fitz) -> list[TextBlock]:
    """페이지의 텍스트 블록(내용 순서, 표시 좌표). 이미지 블록·빈 블록은 뺀다.

    get_text("blocks")와 같은 플래그로 "dict"를 읽어 글꼴 크기·줄 정보를 함께 얻는다
    (블록 텍스트 = 줄마다 span을 이은 뒤 줄바꿈으로 연결 — "blocks" 출력과 같다).
    좌표는 `page.rotation_matrix`로 렌더 PNG와 같은 회전 공간에 사상한다.
    """
    matrix = page.rotation_matrix
    raw = page.get_text("dict", flags=fitz.TEXTFLAGS_BLOCKS, sort=False)
    out: list[TextBlock] = []
    for order, blk in enumerate(raw.get("blocks", [])):
        if blk.get("type", 0) != 0:
            continue
        lines = blk.get("lines") or []
        texts: list[str] = []
        sizes: dict[float, int] = {}
        horizontal = 0
        heights: list[float] = []
        last_rect = None
        side_by_side = 0
        for line in lines:
            spans = line.get("spans") or []
            text = "".join(str(s.get("text") or "") for s in spans)
            texts.append(text)
            for s in spans:
                size = round(float(s.get("size") or 0.0), 1)
                sizes[size] = sizes.get(size, 0) + len(str(s.get("text") or "").strip())
            dx, dy = line.get("dir") or (1.0, 0.0)
            ddx = matrix.a * dx + matrix.c * dy  # 표시 좌표에서의 줄 방향(선형부만)
            ddy = matrix.b * dx + matrix.d * dy
            if ddx > 0.9 and abs(ddy) < 0.2:
                horizontal += 1
            lr = fitz.Rect(line.get("bbox")) * matrix
            lr.normalize()
            heights.append(max(lr.height, 1.0))
            if last_rect is not None and (
                min(last_rect.y1, lr.y1) - max(last_rect.y0, lr.y0)
                >= 0.5 * min(last_rect.height, lr.height)
                and (lr.x0 >= last_rect.x1 or lr.x1 <= last_rect.x0)
            ):
                side_by_side += 1  # 앞 줄과 같은 높이에 나란히 놓인 줄(표 셀)
            last_rect = lr
        text = "\n".join(texts).strip()
        if not text:
            continue
        rect = fitz.Rect(blk.get("bbox")) * matrix
        rect.normalize()
        size = max(sizes.items(), key=lambda kv: (kv[1], kv[0]))[0] if sizes else 0.0
        out.append(TextBlock(
            x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
            text=text,
            size=size,
            line_height=sum(heights) / len(heights) if heights else rect.height,
            last_y0=last_rect.y0 if last_rect is not None else rect.y0,
            last_y1=last_rect.y1 if last_rect is not None else rect.y1,
            last_x1=last_rect.x1 if last_rect is not None else rect.x1,
            horizontal=bool(lines) and horizontal * 2 > len(lines),
            tabular=side_by_side >= 2,
            order=order,
        ))
    return out


def page_text_blocks(page, fitz) -> list[TextBlock]:
    """읽기 순서로 정렬·병합된 페이지 텍스트 블록 (textlayer·텍스트 레이어 복구 공용)."""
    return merge_fragments(order_blocks(extract_text_blocks(page, fitz)))
