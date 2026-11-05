"""반복 생성으로 실패한 단일 페이지를 영역 단위로 다시 읽어 살린다 (Unlimited-OCR-2 블록 출력).

## 왜 필요한가
모델은 빽빽한 페이지(실측: 고정폭 상자에 LaTeX 소스를 그대로 인쇄한 추론 기록)에서 같은 짧은
패턴을 끝없이 낸다. 공식 35-gram 억제는 같은 글자를 다른 토큰 분할로 내면 위상만 바뀔 뿐
막지 못한다. 그래서 8쪽 청크도, 페이지별 single 재처리도 같은 자리에서 루프에 빠져 페이지
전체가 자리표시자가 됐다. 116쪽 논문에서 16쪽이 그랬다. 같은 페이지를 공식 README의 300 DPI로
렌더해 더 작은 영역으로 나눠 읽으면, 15쪽은 인식 불가 줄 없이 충실도 0.89~1.0으로 돌아왔다.
나머지 1쪽은 약 80줄 중 5줄만 남았다(0.957). 디코딩을 바꾸는 방법은 쓰지 않는다:
n-gram 창·크기, 전역 뷰는 0/10이었고, repetition_penalty는 루프 없이 틀린 내용(0.22)을 냈다.
temperature>0은 이 SGLang 이미지의 sampling 커널 JIT 빌드가 실패해 서버를 죽인다.

## 알고리즘 (solve)
1. 영역을 OCR한다. 성공하면 그 블록들을 쓴다.
2. 루프·상한이면, 모델은 이미 끝낸 블록들과 루프에 빠진 블록의 머리(좌표 포함)를 낸 상태다.
   끝낸 블록이 기하적으로 온전하고(한 블록이 다른 블록을 삼키지 않음 — 상자 전체를 한
   블록으로 낸 뒤 같은 줄을 다시 내는 출력이 실측됐다) 루프 블록이 영역의 절반보다 작으면,
   끝낸 블록은 그대로 두고 루프 블록만 따로(여백을 둔 크롭) 읽은 뒤 그 아래부터 이어 읽는다.
3. 아니면 영역 가운데에 가장 가까운 빈 행(잉크 없는 행 — 줄 사이)에서 반으로 나눠 각각 푼다.
4. 줄이 MAX_MARKER_LINES 이하인 작은 영역이 끝까지 루프면 그 줄만 '인식 불가'로 남긴다.
   더 큰 영역을 자를 수 없거나 호출 예산을 넘기면 구조 전체를 실패로 돌린다(그 페이지는
   지금처럼 자리표시자가 된다) — 페이지 절반을 '인식 불가'로 비워 둔 채 살렸다고 하지 않는다.

## 인식 불가 줄
모델이 끝내 못 읽은 줄은 두 블록이 된다: 그 줄의 원본 이미지(`image` — 벤더 크롭 경로가 잘라
`![](images/…)`로 낸다)와 안내 블록(UNREADABLE_LABEL). 호출자가 line_text를 주면(원본 PDF에
텍스트 레이어가 있을 때) 그 줄 상자 안의 텍스트를 안내 블록에 함께 싣는다 — 이미지가 원본이고
텍스트는 검색·복사용이다(렌더된 수식은 텍스트 레이어의 구조가 깨져 있을 수 있다).

## 다단 페이지
이어 읽기(루프 블록 아래 전폭)와 전폭 자르기는 2단 페이지에서 단을 섞거나(왼위·오른위·왼아래…)
오른쪽 단 윗부분을 건너뛴다. 충실도 지표는 문자 bigram 잔존율이라 순서에 둔감해서(표 오탐을
막으려는 설계) 이를 잡지 못한다. 그래서 첫 시도가 실패한 페이지에 세로 단 간격(gutter)이 있으면
넓은 빈 띠로 가로 띠를 나누고, 띠마다 단으로 나눈 뒤 단마다 따로 푼다(읽기 순서: 띠는 위→아래,
단은 왼→오른).

좌표는 모두 모델 계약대로 이미지 대비 0~999로 정규화한다 — 영역 크롭의 좌표를 페이지 좌표로
옮겨 붙이므로, 합친 출력은 모델이 페이지 한 장을 읽은 것과 같은 형식이 된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

# 이보다 어두운 화소를 잉크로 본다 — 옅은 상자 배경(실측 ~242)과 여백(255)은 잉크가 아니다
INK_LEVEL = 200
# 영역 크롭에 두르는 흰 여백(px) — 가장자리에 붙은 줄을 모델이 잘린 줄로 읽지 않게
PAD = 48
# 끝까지 루프인 영역을 '인식 불가'로 남길 수 있는 최대 줄 수·높이(페이지 높이 비율)
MAX_MARKER_LINES = 2
MAX_MARKER_HEIGHT = 0.04
# 단 간격(gutter)의 최소 폭(페이지 폭 비율)과 단을 나누는 넓은 빈 띠의 최소 높이(페이지 높이 비율)
GUTTER_MIN_WIDTH = 0.012
BAND_MIN_GAP = 0.012
# 인식 불가 줄 상자에 두르는 최대 여백(페이지 높이 비율) — 잉크 행 상자는 위아래 획에 딱 붙는다.
# 줄 간격이 좁으면 이웃 줄 잉크까지 거리의 절반만 넓힌다(InkMap.line_band).
MARKER_MARGIN = 0.006
UNREADABLE_TEXT = "> ⚠️ 이 줄은 인식하지 못해 원본 이미지로 넣었습니다"
# 원본 PDF 텍스트 레이어에서 그 줄을 찾았을 때 — 텍스트는 이 뒤에 한 줄로 붙는다
UNREADABLE_FILLED = "> ⚠️ 인식하지 못한 줄 — 원본 PDF 텍스트: "
_FILLED_MAX_CHARS = 1_000
# 인식 불가 줄의 블록 라벨 — 번역 PDF(pdf_export)의 교체 대상(_REPLACEABLE_TYPES)이 아니라 원문 줄이
# 그대로 남고, 번역(segment.layout_units)에도 보내지 않는다. "text"로 두면 번역 PDF가 원문 줄을
# 지우고 이 안내문(의 번역)을 그 자리에 조판한다 — 페이지 자리표시자(블록 없음)보다 나빠진다.
UNREADABLE_LABEL = "unreadable"

_HEAD = re.compile(r"<BLOCK>\s*([\w-]+)((?:<LEVEL[1-6]>)?)\s*\[([^\]<>]{1,300})\]\s*<FORK>")
# 채운 텍스트가 블록·참조 문법(<BLOCK>·<FORK>·<PAGE>·<LEVELn>·<|…|>)으로 읽히지 않게 한다
_GRAMMAR = re.compile(r"<(?=\||/?(?:BLOCK|FORK|PAGE|LEVEL[1-6])>)")


class RescueFailed(Exception):
    """영역 구조로도 페이지를 온전히 살리지 못했다 — 호출자는 원래 실패를 그대로 처리한다."""


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Block:
    label: str
    level: str
    box: tuple[float, float, float, float] | None  # 페이지 대비 0~999
    text: str


@dataclass(frozen=True)
class RegionOutput:
    """영역 OCR 한 번의 결과 — ok가 아니면 raw는 루프·상한에서 멈춘 부분 출력이다."""

    ok: bool
    raw: str
    reason: str = "ok"


@dataclass
class RescueResult:
    raw: str  # 페이지 한 장 분량의 '<PAGE><BLOCK>…' 출력(페이지 좌표)
    unreadable: int = 0
    calls: int = 0
    regions: list[str] = field(default_factory=list)
    markdown: str = ""  # 엔진이 산출물을 쓴 뒤 채운다(run_single과 같은 처리된 마크다운)
    unreadable_share: float = 0.0  # 인식 불가로 남긴 줄들의 높이 합 / 페이지 높이
    filled: int = 0  # 인식 불가 줄 중 원본 PDF 텍스트를 함께 실은 줄 수


# (crop 이미지, 영역, 여백 여부) → RegionOutput. 엔진이 SGLang 호출로 채운다.
RegionOCR = Callable[[Image.Image, Rect, int], RegionOutput]
# 인식 불가 줄 상자들(페이지 대비 0~999) → 줄마다 원본 텍스트(없으면 ""). 한 번에 부른다.
LineText = Callable[[list[tuple[float, float, float, float]]], list[str]]


class InkMap:
    """페이지의 잉크 분포 — 줄(잉크 행 띠)·빈 행·단 간격을 찾는다."""

    def __init__(self, image: Image.Image) -> None:
        with image.convert("L") as gray:
            self.ink = np.asarray(gray) < INK_LEVEL
        self.height, self.width = self.ink.shape

    def _rows(self, rect: Rect) -> np.ndarray:
        area = self.ink[rect.y0:rect.y1, rect.x0:rect.x1]
        # 스캔 잡티(점 몇 개)는 줄로 치지 않는다
        tolerance = max(1, (rect.x1 - rect.x0) // 500)
        return area.sum(axis=1) > tolerance

    def lines(self, rect: Rect) -> list[tuple[int, int]]:
        """잉크가 있는 행의 띠(페이지 행 좌표) — 대략 텍스트 줄 하나가 띠 하나다."""
        rows = self._rows(rect)
        out, start = [], None
        for offset, inked in enumerate(rows):
            row = rect.y0 + offset
            if inked and start is None:
                start = row
            elif not inked and start is not None:
                out.append((start, row))
                start = None
        if start is not None:
            out.append((start, rect.y1))
        return out

    def middle_cut(self, rect: Rect) -> int | None:
        """영역 가운데에 가장 가까운 줄 사이 빈 행 — 줄이 둘 미만이면 None."""
        lines = self.lines(rect)
        if len(lines) < 2:
            return None
        target = (rect.y0 + rect.y1) / 2
        gaps = [(a[1] + b[0]) // 2 for a, b in zip(lines, lines[1:])]
        return min(gaps, key=lambda gap: abs(gap - target))

    def expand_to_blank(self, rect: Rect, y0: int, y1: int) -> tuple[int, int]:
        """[y0, y1)을 바깥으로 넓혀 위아래가 줄 사이 빈 행에서 끝나게 한다(rect 안)."""
        rows = self._rows(rect)
        top = min(max(y0, rect.y0), rect.y1)
        bottom = min(max(y1, top), rect.y1)
        while top > rect.y0 and rows[top - 1 - rect.y0]:
            top -= 1
        while bottom < rect.y1 and rows[bottom - rect.y0]:
            bottom += 1
        return top, bottom

    def line_band(self, x0: int, x1: int, top: int, bottom: int, margin: int) -> tuple[int, int]:
        """줄 띠 [top, bottom)을 원본 줄 크롭용으로 넓힌다 — 획 끝(한 점이라도 잉크인 행)까지 넣고,
        그 너머 빈 행은 이웃 줄 잉크까지 거리의 절반(이웃이 멀면 margin)만 넣는다. 줄 판정(_rows)은
        잡티 허용치가 있어 이웃 줄의 드문 획 끝을 빈 행으로 보므로 여기서는 쓰지 않는다."""
        lo, hi = max(0, top - 3 * margin), min(self.height, bottom + 3 * margin)
        inked = self.ink[lo:hi, x0:x1].any(axis=1)

        def inside(y: int) -> bool:
            return lo <= y < hi

        def reach(edge: int, step: int) -> int:
            y = edge
            while inside(y + step) and abs(y + step - edge) <= margin and inked[y + step - lo]:
                y += step
            gap = 0
            while inside(y + step * (gap + 1)) and gap < 2 * margin and not inked[y + step * (gap + 1) - lo]:
                gap += 1
            neighbour = inside(y + step * (gap + 1)) and gap < 2 * margin
            return y + step * (gap // 2 if neighbour else min(gap, margin))

        return reach(top, -1), reach(bottom - 1, 1) + 1

    def ink_x_extent(self, rect: Rect) -> tuple[int, int]:
        cols = np.where(self.ink[rect.y0:rect.y1, rect.x0:rect.x1].any(axis=0))[0]
        if not len(cols):
            return rect.x0, rect.x1
        return rect.x0 + int(cols[0]), rect.x0 + int(cols[-1]) + 1

    def gutter(self, rect: Rect) -> int | None:
        """영역 전체 높이를 가로지르는 세로 빈 띠(단 간격)의 가운데 열 — 없으면 None."""
        if rect.height <= 0 or rect.x1 - rect.x0 <= 0:
            return None
        area = self.ink[rect.y0:rect.y1, rect.x0:rect.x1]
        per_col = area.sum(axis=0)
        total = per_col.sum()
        if total == 0:
            return None
        blank = per_col <= max(1, rect.height // 500)
        width = rect.x1 - rect.x0
        min_run = max(12, int(self.width * GUTTER_MIN_WIDTH))
        best, best_run = None, 0
        start = None
        for col in range(width + 1):
            is_blank = col < width and blank[col]
            if is_blank and start is None:
                start = col
            elif not is_blank and start is not None:
                run = col - start
                center = (start + col) // 2
                left, right = per_col[:start].sum(), per_col[col:].sum()
                if (
                    run >= min_run and run > best_run
                    and 0.25 * width <= center <= 0.75 * width
                    and left >= 0.1 * total and right >= 0.1 * total
                ):
                    best, best_run = rect.x0 + center, run
                start = None
        return best

    def bands(self, rect: Rect) -> list[Rect]:
        """넓은 빈 행 띠(BAND_MIN_GAP 이상)로 나눈 가로 띠들 — 제목·본문 단·꼬리말 경계."""
        rows = self._rows(rect)
        min_gap = max(8, int(self.height * BAND_MIN_GAP))
        cuts, run_start = [], None
        for offset, inked in enumerate(list(rows) + [True]):
            if not inked and run_start is None:
                run_start = offset
            elif inked and run_start is not None:
                if offset - run_start >= min_gap and 0 < run_start and offset < len(rows):
                    cuts.append(rect.y0 + (run_start + offset) // 2)
                run_start = None
        edges = [rect.y0, *cuts, rect.y1]
        return [Rect(rect.x0, a, rect.x1, b) for a, b in zip(edges, edges[1:]) if b > a]


def column_regions(ink: InkMap, rect: Rect, depth: int = 0) -> list[Rect]:
    """다단 페이지면 읽기 순서대로 단 영역 목록, 단이 없으면 [rect]."""
    gutter = ink.gutter(rect)
    if gutter is not None:
        left = Rect(rect.x0, rect.y0, gutter, rect.y1)
        right = Rect(gutter, rect.y0, rect.x1, rect.y1)
        return column_regions(ink, left, depth + 1) + column_regions(ink, right, depth + 1)
    if depth >= 3:
        return [rect]
    bands = ink.bands(rect)
    if len(bands) <= 1:
        return [rect]
    out, split = [], False
    for band in bands:
        sub = column_regions(ink, band, depth + 1)
        split = split or len(sub) > 1
        out += sub
    return out if split else [rect]


def parse_blocks(raw: str, rect: Rect, crop_size: tuple[int, int], pad: int,
                 page_size: tuple[int, int]) -> list[Block]:
    """영역 크롭 출력의 블록들 — 좌표를 크롭 대비 0~999에서 페이지 대비 0~999로 옮긴다."""
    body = raw.replace("<PAGE>", "")
    heads = list(_HEAD.finditer(body))
    crop_w, crop_h = crop_size
    page_w, page_h = page_size
    out = []
    for index, match in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(body)
        text = body[match.end():end].rstrip("\n")
        box = None
        try:
            values = [float(v) for v in match[3].split(",")]
        except ValueError:
            values = []
        if len(values) == 4:
            x0, y0, x1, y1 = values
            box = (
                (x0 / 999 * crop_w - pad + rect.x0) / page_w * 999,
                (y0 / 999 * crop_h - pad + rect.y0) / page_h * 999,
                (x1 / 999 * crop_w - pad + rect.x0) / page_w * 999,
                (y1 / 999 * crop_h - pad + rect.y0) / page_h * 999,
            )
        out.append(Block(match[1], match[2], box, text))
    return out


def format_blocks(blocks: list[Block]) -> str:
    def header(block: Block) -> str:
        if block.box is None:
            return f"<BLOCK>{block.label}{block.level}[0, 0, 999, 999]<FORK>"
        coords = ", ".join(str(int(round(min(max(v, 0.0), 999.0)))) for v in block.box)
        return f"<BLOCK>{block.label}{block.level}[{coords}]<FORK>"

    parts = []
    for i, block in enumerate(blocks):
        text = header(block) + block.text
        # 인식 불가 줄(원본 줄 이미지 + 인용 안내)은 앞뒤를 빈 줄로 띄운다 — 붙이면 마크다운이 이미지를
        # 앞 문단에 잇고, 인용 뒤 줄을 인용 안으로 끌어들인다(lazy continuation)
        if block.label == UNREADABLE_LABEL:
            text += "\n"
        elif (
            parts and block.label == "image" and i + 1 < len(blocks)
            and blocks[i + 1].label == UNREADABLE_LABEL
        ):
            text = "\n" + text
        parts.append(text)
    return "<PAGE>" + "\n".join(parts)


def sane(blocks: list[Block]) -> bool:
    """한 블록이 다른 블록 면적의 절반 넘게 겹쳐 덮지 않는다 — 상자 통째 블록 + 같은 줄 재출력을 거른다."""
    boxes = [b.box for b in blocks if b.box is not None]

    def area(box):
        return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)

    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            ix = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
            iy = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
            smaller = min(area(first), area(second))
            if smaller > 0 and ix * iy > 0.5 * smaller:
                return False
    return True


class _Solver:
    def __init__(self, image: Image.Image, ocr: RegionOCR, max_calls: int,
                 deadline: Callable[[], bool]) -> None:
        self.image = image
        self.ocr = ocr
        self.ink = InkMap(image)
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.unreadable = 0
        self.unreadable_rows = 0
        self.trace: list[str] = []

    @property
    def page_size(self) -> tuple[int, int]:
        return self.image.width, self.image.height

    def _read(self, rect: Rect, pad: bool) -> tuple[RegionOutput, list[Block]]:
        if self.calls >= self.max_calls or self.deadline():
            raise RescueFailed("영역 재처리 예산 초과")
        crop = self.image.crop((rect.x0, rect.y0, rect.x1, rect.y1))
        try:
            if pad:
                padded = ImageOps.expand(crop, border=PAD, fill="white")
                crop.close()
                crop = padded
            self.calls += 1
            result = self.ocr(crop, rect, PAD if pad else 0)
            blocks = parse_blocks(result.raw, rect, crop.size, PAD if pad else 0, self.page_size)
        finally:
            crop.close()
        self.trace.append(f"{rect.y0}-{rect.y1}:{result.reason}")
        return result, blocks

    def _px(self, value: float) -> int:
        return int(round(value / 999 * self.image.height))

    def _marker(self, rect: Rect) -> list[Block]:
        lines = self.ink.lines(rect)
        if not lines:
            return []  # 빈 영역 — 잃을 내용이 없다
        if (
            len(lines) > MAX_MARKER_LINES
            or rect.height > MAX_MARKER_HEIGHT * self.image.height
        ):
            raise RescueFailed("자를 수 없는 큰 영역이 끝까지 반복 생성")
        self.unreadable += 1
        self.unreadable_rows += lines[-1][1] - lines[0][0]
        width, height = self.page_size
        margin = max(4, round(MARKER_MARGIN * height))
        x0, x1 = self.ink.ink_x_extent(rect)
        x0, x1 = max(rect.x0, x0 - margin), min(rect.x1, x1 + margin)
        top, bottom = self.ink.line_band(x0, x1, lines[0][0], lines[-1][1], margin)
        box = (x0 / width * 999, top / height * 999, x1 / width * 999, bottom / height * 999)
        # 원본 줄 이미지 + 안내 — 이미지는 벤더 크롭 경로가 잘라 낸다(내용 없는 image 블록)
        return [Block("image", "", box, ""), Block(UNREADABLE_LABEL, "", box, UNREADABLE_TEXT)]

    def solve(self, rect: Rect, pad: bool, first: tuple | None = None) -> list[Block]:
        if rect.height < 4 or not self.ink.lines(rect):
            return []
        result, blocks = first if first is not None else self._read(rect, pad)
        if result.ok:
            return blocks
        done, trap = blocks[:-1], (blocks[-1] if blocks else None)
        if trap is not None and trap.box is not None and done and sane(blocks):
            t0, t1 = self.ink.expand_to_blank(rect, self._px(trap.box[1]), self._px(trap.box[3]) + 1)
            above = all(b.box is None or b.box[1] <= trap.box[3] for b in done)
            if above and t1 > t0 and (t1 - t0) < rect.height / 2:
                trap_rect = Rect(rect.x0, max(t0, rect.y0), rect.x1, t1)
                rest = Rect(rect.x0, t1, rect.x1, rect.y1)
                return done + self.solve_isolated(trap_rect) + self.solve(rest, True)
        return self.split(rect)

    def solve_isolated(self, rect: Rect) -> list[Block]:
        if rect.height < 4 or not self.ink.lines(rect):
            return []
        result, blocks = self._read(rect, True)
        if result.ok:
            return blocks
        return self.split(rect)

    def split(self, rect: Rect) -> list[Block]:
        cut = self.ink.middle_cut(rect)
        if cut is None:
            return self._marker(rect)
        return (
            self.solve(Rect(rect.x0, rect.y0, rect.x1, cut), True)
            + self.solve(Rect(rect.x0, cut, rect.x1, rect.y1), True)
        )


def _filled(text: str) -> str:
    """원본 텍스트를 안내 블록 한 줄에 싣는다 — 공백을 접고 블록 문법을 무력화한다."""
    text = _GRAMMAR.sub("⟨", " ".join(str(text or "").split()))
    if len(text) > _FILLED_MAX_CHARS:
        text = text[:_FILLED_MAX_CHARS].rstrip() + " …"
    return UNREADABLE_FILLED + text if text else ""


def fill_unreadable(blocks: list[Block], line_text: LineText) -> tuple[list[Block], int]:
    """인식 불가 블록마다 line_text가 찾은 원본 텍스트를 싣는다. (블록, 채운 줄 수)."""
    targets = [i for i, b in enumerate(blocks) if b.label == UNREADABLE_LABEL and b.box]
    if not targets:
        return blocks, 0
    texts = line_text([blocks[i].box for i in targets])
    if len(texts) != len(targets):
        return blocks, 0
    out, filled = list(blocks), 0
    for i, text in zip(targets, texts):
        content = _filled(text)
        if content:
            b = out[i]
            out[i] = Block(b.label, b.level, b.box, content)
            filled += 1
    return out, filled


def rescue_page(
    image: Image.Image,
    ocr: RegionOCR,
    *,
    max_calls: int,
    deadline: Callable[[], bool] = lambda: False,
    line_text: LineText | None = None,
) -> RescueResult:
    """페이지 이미지 한 장을 영역 구조로 읽는다. 살리지 못하면 RescueFailed.

    line_text를 주면 인식 불가 줄에 원본 텍스트를 함께 싣는다(fill_unreadable)."""
    solver = _Solver(image, ocr, max_calls, deadline)
    page = Rect(0, 0, image.width, image.height)
    first = solver._read(page, False)
    if first[0].ok:
        blocks = first[1]
    else:
        columns = column_regions(solver.ink, page)
        if len(columns) > 1:
            # 다단 — 부분 출력의 읽기 순서를 믿고 이어 읽으면 단을 건너뛴다. 단마다 처음부터 푼다.
            blocks = []
            for column in columns:
                blocks += solver.solve(column, True)
        else:
            blocks = solver.solve(page, False, first=first)
    if not any(block.text.strip() for block in blocks):
        raise RescueFailed("영역 재처리 결과가 비었다")
    filled = 0
    if line_text is not None and solver.unreadable:
        blocks, filled = fill_unreadable(blocks, line_text)
    return RescueResult(
        raw=format_blocks(blocks), unreadable=solver.unreadable,
        calls=solver.calls, regions=solver.trace,
        unreadable_share=solver.unreadable_rows / max(image.height, 1), filled=filled,
    )
