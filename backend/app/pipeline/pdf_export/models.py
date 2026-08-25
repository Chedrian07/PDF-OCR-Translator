"""조판 계획·원문 span을 나르는 값 객체.

PyMuPDF Rect는 fitz를 지연 로드하는 모듈 계약 때문에 `object`로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _TextFitPlan:
    """리댁션 전에 검증을 끝낸 텍스트 삽입 계획.

    rect는 PyMuPDF Rect지만 이 모듈은 fitz를 quiet_fitz()로 지연 로드하므로
    구체 타입을 주석에 고정하지 않는다.
    """

    rect: object
    fontsize: float
    expanded: bool = False
    align: int = 0
    bold: bool = False
    lineheight: float | None = None
    origin: tuple[float, float] | None = None
    ink_rect: object | None = None
    first_origin: tuple[float, float] | None = None
    rich_runs: tuple[tuple[float, float, str, bool], ...] = ()


@dataclass(frozen=True)
class _Replacement:
    plan: _TextFitPlan
    text: str
    kind: str = "text"
    redact_rect: object | None = None
    fontname: str = "korea"
    fontfile: str | None = None
    source_rect: object | None = None
    block_index: int = -1
    bold_prefix: tuple[str, str] | None = None
    redact_rects: tuple[object, ...] = ()


@dataclass(frozen=True)
class _FlowCandidate:
    """한 페이지에서 원자적으로 재배치할 수 있는 번역 텍스트 블록."""

    block_index: int
    block_type: str
    text: str
    rect: object
    base_pt: float
    fontname: str
    fontfile: str | None
    align: int
    bold: bool
    lineheights: tuple[float | None, ...]
    redact_rects: tuple[object, ...]
    source_rect: object
    bold_prefix: tuple[str, str] | None = None
    # 가로로 나란히 놓였던 셀이 OCR 줄바꿈으로 평탄화된 블록을 원문의 시각 줄
    # 수로 되접은 대안 텍스트. 원래 조판이 실패할 때만 쓰며 평탄화가 아니면 None.
    reflow_text: str | None = None
    # 원문 줄·열 좌표에 그대로 조판할 수 있는 세그먼트. 리플로우까지 실패했을
    # 때 줄 단위 부분 회수에 쓴다(전부-아니면-전무를 피하는 마지막 단계).
    listing_segments: tuple[_LineSegment, ...] = ()


@dataclass(frozen=True)
class _LineSegment:
    """원문 시각 줄 안에서 OCR 논리 줄 하나가 차지하던 가로 구간.

    의사코드 리스팅과 가로 평탄화된 표에서 "몇 번째 줄의 어느 열"이 곧 의미다.
    번역문을 한 상자에 흘려 넣으면 그 구조가 사라지므로 세그먼트 단위로 원문
    좌표에 그대로 조판한다. `x1`은 같은 줄 다음 세그먼트가 시작하는 x(마지막
    세그먼트는 블록 우변)이며, `band`는 원문이 이미 점유하던 세로 띠다.
    """

    text: str
    original: str
    spans: tuple[_SourceSpan, ...]
    x0: float
    x1: float
    baseline: float
    size: float
    band: tuple[float, float]
    bold: bool = False


@dataclass(frozen=True)
class _TableCell:
    """row/col span을 펼치지 않고 보존한 HTML 표 셀."""

    row: int
    col: int
    rowspan: int
    colspan: int
    text: str


@dataclass(frozen=True)
class _RawTableCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class _SourceSpan:
    """원문 PDF 텍스트 span과 조판 메타데이터."""

    rect: object
    text: str
    size: float
    flags: int
    origin: tuple[float, float]
