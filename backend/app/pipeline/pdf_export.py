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

import json
import logging
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from statistics import median

from .layout import estimate_font_size_cqw
from .pdf import quiet_fitz

logger = logging.getLogger(__name__)

# 번역 텍스트로 교체할 수 있는 블록 타입. 이 밖의 타입(image·equation·table·
# algorithm 등)은 내용이 달라도 원본을 유지한다 — 표 HTML·수식 LaTeX를 평문으로
# 밀어 넣으면 오히려 품질이 나빠진다.
_REPLACEABLE_TYPES = frozenset({
    "text", "title", "list", "caption", "image_caption", "table_caption",
    "page_footnote", "footnote", "aside_text", "header", "footer",
})
_SPECIALIST_TYPES = frozenset({"table", "equation", "algorithm"})
# 참고문헌은 제목을 억지로 번역하면 저자명·학술지명·URL 사이에 서로 다른 문자 폭이
# 섞여 원문보다 훨씬 불안정하게 줄바꿈된다. 학술 번역 관례대로 서지 항목은 원문
# 조판을 그대로 보존한다(본문의 인용 번호와 참고문헌 제목은 계속 검색 가능).
_PRESERVE_TYPES = frozenset({"ref_text"})

# 세로쓰기 블록은 회전 조합이 페이지 회전과 얽혀 배치가 어긋나기 쉽다 — 원본 유지.
_VERTICAL_SKIP = ("up", "down")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}")
_LATEX_SUP_RE = re.compile(r"\^\{([0-9+\-=()n]+)\}")
_SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")

_SYSTEM_FONT_CANDIDATES = (
    # 논문 원본의 Times 계열 질감에 맞는 명조/serif를 우선한다. 고딕을 먼저
    # 쓰면 같은 pt라도 x-height와 폭이 커져 한국어 본문만 유난히 크고 빽빽해진다.
    "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-KR.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",          # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-KR-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)
_SYSTEM_SANS_FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",          # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-KR-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    # sans가 없으면 한글 누락보다 명조 폴백이 낫다.
    "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
)
# fc-list 탐색 결과에서 serif(명조) 계열을 판별하는 파일명 힌트 — serif 체인은
# 명조를 앞세우고 sans 체인은 뒤로 미뤄 원문 Times 질감 우선순위를 유지한다.
_SERIF_NAME_HINT = re.compile(r"serif|myeongjo|myungjo|batang", re.IGNORECASE)

# insert_textbox 축소 사다리 — layout-fit.js와 같은 정신: 최대 45%까지 줄여 본다.
_SHRINK_STEPS = (1.0, 0.9, 0.8, 0.7, 0.62, 0.55)
_SINGLE_LINE_SCALES = (1.0, 0.96, 0.92)
_MIN_FONT_PT = 4.0
_MAX_FONT_PT = 72.0
_MAX_TABLE_CELLS = 500  # search_for 셀별 탐색의 CPU 상한 + 비정상 HTML 표 방어
_BODY_LINEHEIGHTS = (1.36, 1.28, 1.20)
_CAPTION_LINEHEIGHTS = (1.30, 1.22, 1.16)
_TITLE_LINEHEIGHTS = (1.18, 1.12)


class PdfExportError(RuntimeError):
    """내보내기 불가(입력 파일 없음/손상). 사용자에게 그대로 보여줄 한국어 메시지."""


@dataclass
class PdfExportResult:
    path: Path
    replaced: int = 0
    kept: int = 0
    relocated: int = 0
    table_cells_replaced: int = 0
    specialist_kept: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict:
        """경로·본문 없이 UI에 안전하게 노출할 ASCII/숫자 중심 생성 리포트."""
        return {
            "replaced": self.replaced,
            "kept": self.kept,
            "relocated": self.relocated,
            "table_cells_replaced": self.table_cells_replaced,
            "specialist_kept": dict(sorted(self.specialist_kept.items())),
            "warning_count": len(self.warnings),
            "warnings": self.warnings[:50],
        }


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


@dataclass(frozen=True)
class _Replacement:
    plan: _TextFitPlan
    text: str
    kind: str = "text"
    redact_rect: object | None = None
    fontname: str = "korea"
    fontfile: str | None = None


class _TableParser(HTMLParser):
    """번역 레이아웃의 단순 HTML table을 행/셀 평문 행렬로 변환한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_plain_text("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _fontconfig_candidates() -> tuple[str, ...]:
    """fc-list로 한글 지원 폰트 파일을 탐색한다 — 정적 후보 전멸 시 최후 보조.

    _SYSTEM_FONT_CANDIDATES는 macOS·Debian 계열 경로만 알고 있어 Fedora/Arch
    같은 배포판에서는 전부 빗나갈 수 있다. fc-list가 없는 환경(macOS 기본,
    fontconfig 미설치 컨테이너)이나 실행 실패는 조용히 빈 목록으로 처리해
    기존 폴백 체인을 그대로 따른다.
    """
    fc = shutil.which("fc-list")
    if not fc:
        return ()
    try:
        proc = subprocess.run(
            [fc, "--format", "%{file}\n", ":lang=ko"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 — 탐색 실패는 조용히 정적 폴백 유지
        return ()
    return tuple(dict.fromkeys(
        line.strip() for line in proc.stdout.splitlines() if line.strip()
    ))


def _resolve_font(
    explicit: str = "",
    candidates: tuple[str, ...] = _SYSTEM_FONT_CANDIDATES,
    *,
    prefer_serif: bool = True,
) -> tuple[str | None, str]:
    """(fontfile|None, fontname) 반환. 파일 폰트는 로드 검증 후 채택한다.

    정적 후보가 전부 실패하면 fc-list(fontconfig) 런타임 탐색을 최후 보조로
    시도하고, 그마저 없으면 PyMuPDF 내장 CJK로 폴백한다 — 내보내기가 폰트
    때문에 실패하지는 않는다는 모듈 계약 유지.
    """
    fitz = quiet_fitz()

    def _verify(path: str) -> str | None:
        p = Path(path)
        if not p.is_file():
            if path == explicit:
                logger.warning("PDF_EXPORT_FONT 파일이 없습니다: %s — 폴백 사용", path)
            return None
        try:
            font = fitz.Font(fontfile=str(p))
            if font.has_glyph(ord("한")):
                return str(p)
            logger.warning("폰트에 한글 글리프가 없습니다: %s — 폴백 사용", path)
        except Exception:  # noqa: BLE001 — 손상/미지원 포맷은 다음 후보로
            logger.warning("폰트 로드 실패: %s — 폴백 사용", path)
        return None

    paths = ([explicit] if explicit else []) + list(candidates)
    for path in paths:
        found = _verify(path)
        if found is not None:
            return found, "uocr-ko"
    # 정적 후보 전멸 — fc-list가 찾은 한글 폰트를 같은 has_glyph 검증으로 시도.
    extras = [p for p in _fontconfig_candidates() if p not in paths]
    extras.sort(
        key=lambda p: bool(_SERIF_NAME_HINT.search(Path(p).name)) != prefer_serif
    )
    for path in extras:
        found = _verify(path)
        if found is not None:
            return found, "uocr-ko"
    return None, "korea"  # PyMuPDF 내장 CJK (Droid Sans Fallback) — 항상 존재


def _load_pages(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PdfExportError(f"레이아웃 파일을 읽을 수 없습니다: {path.name}") from e
    if not isinstance(data, list):
        raise PdfExportError(f"레이아웃 파일 형식이 올바르지 않습니다: {path.name}")
    return data


def _plain_text(content: str) -> str:
    """블록 내용 → 삽입용 평문.

    PDF textbox는 LaTeX를 조판하지 못하므로 흔한 inline 수식 표기를 읽을 수 있는
    유니코드 평문으로 낮춘다(`\\(E=mc^{2}\\)` → `E=mc²`). 복잡한 equation
    블록은 애초 교체 대상이 아니며 원본 조판을 유지한다.
    """
    text = _TAG_RE.sub(" ", content)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "").replace("$$", "")
    text = _LATEX_TEXT_RE.sub(lambda m: m.group(1), text)
    text = _LATEX_SUP_RE.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), text)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _table_matrix(content: str) -> list[list[str]] | None:
    parser = _TableParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:  # noqa: BLE001 — 손상 OCR HTML은 표 번역만 보수적으로 생략
        return None
    rows = parser.rows
    if not rows or not rows[0]:
        return None
    width = len(rows[0])
    return rows if all(len(row) == width for row in rows) else None


def _rect_horizontal_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _rect_overlap_area(a, b) -> float:
    """두 Rect의 교차 면적. 겹치지 않으면 0 — fitz `&`의 빈 Rect 의존 없이 계산."""
    return _rect_horizontal_overlap(a, b) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _free_growth_rect(page, rect, obstacles: list[object]) -> object:
    """현재 블록 바로 아래의 빈 세로 영역까지만 확장 가능한 Rect를 반환한다.

    가로로 사실상 겹치지 않는 사이드바/다른 단은 장애물로 보지 않는다. 같은 단의
    다음 블록·표·그림·푸터 앞 2pt에서 멈추므로 번역문 확장이 이웃 내용을 덮지 않는다.
    """
    limit = page.mediabox.y1
    for other in obstacles:
        if other is rect or other.y1 <= rect.y1 + 0.5:
            continue
        overlap = _rect_horizontal_overlap(rect, other)
        if overlap < min(rect.width, other.width) * 0.15:
            continue
        # OCR 문단 bbox는 연속 문단에서 y1 == 다음 y0인 경우가 흔하고 반올림
        # 때문에 1pt 안팎 겹치기도 한다. 이런 장애물을 "현재 bbox보다 위"로
        # 간주해 건너뛰면 그 다음 문단까지 확장되어 번역문끼리 겹친다.
        # 같은 단의 다음 블록이 현재 하단에 닿거나 걸치면 확장을 금지한다.
        if other.y0 <= rect.y1 + 0.5:
            return +rect
        limit = min(limit, other.y0 - 2.0)
    grown = +rect
    grown.y1 = max(rect.y1, limit)
    return grown


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


def _table_cell_rects(page, table_rect, original: list[list[str]]) -> list[list[object]]:
    """원문 셀 검색 중심으로 표 격자를 추정하고, 실패 셀은 균등 격자로 보완한다."""
    rows, cols = len(original), len(original[0])
    x_centers: list[list[float]] = [[] for _ in range(cols)]
    y_centers: list[list[float]] = [[] for _ in range(rows)]
    for ri, row in enumerate(original):
        for ci, text in enumerate(row):
            if not text:
                continue
            expected_x = table_rect.x0 + (ci + 0.5) * table_rect.width / cols
            expected_y = table_rect.y0 + (ri + 0.5) * table_rect.height / rows
            hits = page.search_for(text, clip=table_rect)
            if not hits:
                continue
            hit = min(hits, key=lambda r: abs((r.x0 + r.x1) / 2 - expected_x)
                      + abs((r.y0 + r.y1) / 2 - expected_y))
            x_centers[ci].append((hit.x0 + hit.x1) / 2)
            y_centers[ri].append((hit.y0 + hit.y1) / 2)
    xs = _grid_boundaries(x_centers, table_rect.x0, table_rect.x1)
    ys = _grid_boundaries(y_centers, table_rect.y0, table_rect.y1)
    fitz = quiet_fitz()
    matrix: list[list[object]] = []
    for ri in range(rows):
        out_row = []
        for ci in range(cols):
            rect = fitz.Rect(xs[ci], ys[ri], xs[ci + 1], ys[ri + 1])
            pad = min(2.0, rect.width * 0.06, rect.height * 0.10)
            rect.x0 += pad
            rect.x1 -= pad
            rect.y0 += pad
            rect.y1 -= pad
            out_row.append(rect)
        matrix.append(out_row)
    return matrix


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


def _block_rect(fitz, page, bbox):
    """0–999 정규화 bbox → 페이지 내부 좌표 Rect|None (표시 공간 경유, 회전 보정)."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    w, h = page.rect.width, page.rect.height  # 표시(회전 반영) 공간
    rect = fitz.Rect(x1 / 999 * w, y1 / 999 * h, x2 / 999 * w, y2 / 999 * h)
    rect = rect * page.derotation_matrix  # 내부(비회전) 좌표로
    rect.normalize()
    return rect if not rect.is_empty else None


def _source_span_rects(fitz, page) -> list[object]:
    """페이지 텍스트 span 사각형을 한 번만 읽어 반환한다."""
    out: list[object] = []
    try:
        blocks = page.get_text("dict").get("blocks", ())
    except Exception:  # noqa: BLE001 — 텍스트 레이어가 없으면 빈 목록
        return out
    for block in blocks:
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                bbox = span.get("bbox")
                if bbox and len(bbox) == 4 and str(span.get("text") or "").strip():
                    out.append(fitz.Rect(bbox))
    return out


def _source_text_rect(page, rect, source_spans: list[object]):
    """OCR bbox에 속한 실제 원문 span까지 포함한 안전한 리댁션 사각형.

    검출 bbox가 URL·긴 단어의 끝을 몇 pt 잘라내는 경우 OCR 사각형만 지우면 `f.`
    같은 원문 꼬리가 번역문 옆에 남는다. span 중심이 OCR bbox 안에 든 경우에만
    실제 글리프 bbox를 합치므로 다음 문단까지 무차별 확장하지 않는다.
    """
    actual = +rect
    for source in source_spans:
        center = (source.tl + source.br) / 2
        if (
            rect.x0 - 1 <= center.x <= rect.x1 + 1
            and rect.y0 - 1 <= center.y <= rect.y1 + 1
        ):
            actual.include_rect(source)
    return actual & page.mediabox


def _plan_shrink_to_fit(
    page, rect, text: str, base_pt: float, fontname: str, fontfile: str | None,
    *, max_rect=None, align: int = 0, bold: bool = False,
    lineheights: tuple[float | None, ...] = (None,),
) -> _TextFitPlan | None:
    """Shape로 실제 삽입과 동일한 조판을 dry-run해 안전한 계획만 반환한다.

    원문 리댁션보다 먼저 실행하는 것이 핵심이다. 번역문이 아무 크기로도 들어가지
    않으면 None을 반환해 해당 원문 블록을 그대로 보존한다.
    """
    rot = page.rotation

    grown = +(max_rect if max_rect is not None else rect)
    if max_rect is None:
        grown.y1 = page.mediabox.y1

    # OCR bbox는 원본 글리프에 딱 맞지만 CJK 폰트의 ascender/descender는 더 높다.
    # 같은 크기에서 원래 상자 → 충돌 없는 확장 상자 순으로 시도한 뒤에야 축소한다.
    # 그렇지 않으면 18pt 제목이 원래 상자의 12.6pt에 먼저 들어가 계층이 무너진다.
    for scale in _SHRINK_STEPS:
        size = max(_MIN_FONT_PT, base_pt * scale)
        # 한국어 본문은 영문 기본 leading보다 넓은 1.3대 행간이 자연스럽다.
        # 같은 글자 크기에서 자연 행간 → 조밀한 행간 순으로 먼저 시도하고,
        # 그 뒤에만 폰트를 축소한다. 이 순서가 짧은 번역문 사이의 큰 흰 구멍과
        # 긴 번역문만 유난히 작아지는 현상을 동시에 줄인다.
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
            for candidate, expanded in (
                (rect, False),
                (grown, True),
            ):
                if expanded and grown.y1 <= rect.y1 + 0.5:
                    continue
                shape = page.new_shape()
                if shape.insert_textbox(candidate, text, **kwargs) >= 0:
                    return _TextFitPlan(
                        +candidate, size, expanded, align, bold, lineheight,
                    )
    return None


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
) -> _TextFitPlan | None:
    """한 줄 번역을 원문 baseline 크기로 배치한다.

    `insert_textbox()`는 CJK ascender/descender 전체가 얕은 OCR bbox 안에 들어가야
    성공으로 판정하므로, 실제로는 한 줄이 넉넉히 들어가는 제목·목록 항목도 60~70%로
    축소하는 문제가 있다. 줄바꿈이 필요 없고 가로 폭이 맞는 경우에는 폰트 메트릭으로
    baseline을 계산해 `insert_text()` 경로를 사용한다.
    """
    if not text or "\n" in text:
        return None
    fitz = quiet_fitz()
    try:
        font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(fontname=fontname)
    except Exception:  # noqa: BLE001 — textbox 폴백이 있으므로 품질 경로만 포기
        return None
    vertical = max_rect if max_rect is not None else rect
    for scale in _SINGLE_LINE_SCALES:
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
        if width > rect.width + 0.25:
            continue
        if align == 1:
            x = rect.x0 + (rect.width - width) / 2
        elif align == 2:
            x = rect.x1 - width
        else:
            x = rect.x0
        baseline = rect.y0 + size * font.ascender
        glyph_bottom = baseline - size * font.descender
        # max_rect은 다음 블록 앞 2pt에서 끝난다. 폰트 bbox의 descender는 한글
        # 글리프가 실제로 쓰지 않는 하단까지 포함하므로 그 안전 여백만 허용한다.
        if glyph_bottom > vertical.y1 + 2.5:
            continue
        return _TextFitPlan(
            +rect,
            size,
            False,
            align,
            bold,
            None,
            (float(x), float(baseline)),
        )
    return None


def _insert_fitted_text(
    page, plan: _TextFitPlan, text: str, fontname: str, fontfile: str | None,
) -> None:
    """검증된 계획을 적용한다. dry-run과 달라지면 손상 PDF를 저장하지 않고 중단."""
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
        return
    leftover = page.insert_textbox(plan.rect, text, **kwargs)
    if leftover < 0:
        raise PdfExportError("번역 텍스트 조판 결과가 사전 검증과 달라 PDF 생성을 중단했습니다")


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

    # /layout 탭을 먼저 열지 않아도 PDF 내보내기가 원본의 실측 폰트 크기와
    # 세로쓰기 정보를 사용해야 한다. 지연 백필 결과를 메모리에서만 활용하고,
    # 번역본에 없는 메타는 아래에서 원문 블록 값을 폴백으로 읽는다.
    try:
        from .pdf_fonts import enrich_layout_fonts

        enrich_layout_fonts(src, orig_page_list)
    except Exception:  # noqa: BLE001 — 폰트 메타는 품질 향상용, 내보내기 필수 조건 아님
        logger.warning("PDF 내보내기용 원본 폰트 메타 추출 실패 — 면적 휴리스틱 사용")

    orig_pages = {p.get("page"): p for p in orig_page_list}
    serif_ff, serif_name = _resolve_font(fontfile)
    sans_ff, sans_name = _resolve_font(
        fontfile, _SYSTEM_SANS_FONT_CANDIDATES, prefer_serif=False,
    )

    result = PdfExportResult(path=job_dir / f"export.{lang}.pdf")
    if serif_ff is None and sans_ff is None:
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

    try:
        for tpage in trans_pages:
            pno = tpage.get("page")
            opage = orig_pages.get(pno)
            if not isinstance(pno, int) or not (1 <= pno <= doc.page_count) or opage is None:
                continue
            page = doc[pno - 1]
            width = tpage.get("width") or 1
            height = tpage.get("height") or 1
            aspect = height / width if width else 1.0

            # 1) 교체 대상 수집. 모든 블록 사각형을 먼저 만들고, 일반 텍스트가
            # 공간을 늘릴 때 같은 단의 다음 블록과 충돌하지 않는 하단을 계산한다.
            oblocks = opage.get("blocks", [])
            tblocks = tpage.get("blocks", [])
            block_rects = [_block_rect(fitz, page, b.get("bbox")) for b in oblocks]
            source_spans = _source_span_rects(fitz, page)
            # 래스터 인스턴스는 리댁션 '이전'에 1회만 수집한다 — apply_redactions
            # 이후의 get_image_info()는 스테일 캐시를 반환할 수 있다(실측).
            try:
                raster_rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
            except Exception:  # noqa: BLE001 — 이미지 목록 실패가 텍스트 교체를 막지 않는다
                raster_rects = []
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
            targets: list[_Replacement] = []
            for block_index, (ob, tb) in enumerate(zip(oblocks, tblocks)):
                if not isinstance(ob, dict) or not isinstance(tb, dict):
                    continue
                block_type = str(tb.get("type") or "")

                # 표는 HTML을 통째로 평문 삽입하지 않고 셀 구조가 원문과 정확히
                # 대응할 때만 셀별로 교체한다. 벡터 선은 redaction 옵션으로 보존된다.
                if block_type == "table":
                    old_matrix = _table_matrix(str(ob.get("content") or ""))
                    new_matrix = _table_matrix(str(tb.get("content") or ""))
                    table_rect = block_rects[block_index]
                    if (old_matrix is None or new_matrix is None or table_rect is None
                            or len(old_matrix) != len(new_matrix)
                            or any(len(a) != len(b) for a, b in zip(old_matrix, new_matrix))
                            or len(old_matrix) * len(old_matrix[0]) > _MAX_TABLE_CELLS):
                        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
                        if str(ob.get("content") or "") != str(tb.get("content") or ""):
                            result.warnings.append(
                                f"p{pno}: 표 셀 구조 불일치 — 원문 표 보존"
                            )
                        continue
                    cell_rects = _table_cell_rects(page, table_rect, old_matrix)
                    table_failed = False
                    for ri, (old_row, new_row) in enumerate(zip(old_matrix, new_matrix)):
                        for ci, (old_cell, new_cell) in enumerate(zip(old_row, new_row)):
                            old_text = _plain_text(old_cell)
                            new_text = _plain_text(new_cell)
                            if not new_text or new_text == old_text:
                                continue
                            cell = cell_rects[ri][ci]
                            base_pt = min(12.0, max(_MIN_FONT_PT, cell.height * 0.45))
                            plan = _plan_shrink_to_fit(
                                page,
                                cell,
                                new_text,
                                base_pt,
                                serif_name,
                                serif_ff,
                                max_rect=cell,
                                lineheights=_CAPTION_LINEHEIGHTS,
                            )
                            if plan is None:
                                result.kept += 1
                                table_failed = True
                                result.warnings.append(
                                    f"p{pno}: 표 {ri + 1}행 {ci + 1}열 번역 생략(공간 부족)"
                                    " — 원문 셀 보존"
                                )
                                continue
                            targets.append(_Replacement(
                                plan,
                                new_text,
                                "table",
                                cell,
                                serif_name,
                                serif_ff,
                            ))
                    if table_failed:
                        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
                    continue

                if block_type in _PRESERVE_TYPES:
                    result.kept += 1
                    result.specialist_kept["reference"] = (
                        result.specialist_kept.get("reference", 0) + 1
                    )
                    continue
                if block_type not in _REPLACEABLE_TYPES:
                    if block_type in _SPECIALIST_TYPES:
                        result.specialist_kept[block_type] = (
                            result.specialist_kept.get(block_type, 0) + 1
                        )
                    continue
                if (tb.get("vertical") or ob.get("vertical")) in _VERTICAL_SKIP:
                    result.kept += 1
                    result.specialist_kept["vertical"] = result.specialist_kept.get("vertical", 0) + 1
                    continue
                new = _plain_text(str(tb.get("content") or ""))
                old = _plain_text(str(ob.get("content") or ""))
                if not new or new == old:
                    result.kept += 1
                    continue
                rect = block_rects[block_index]
                if rect is None:
                    result.kept += 1
                    continue
                # 그림 패널·로고 위 OCR 텍스트 블록은 교체하지 않는다. 그림 속
                # 텍스트는 OCR 오독이 잦고 원본 조판이 항상 우월하며, 번역을
                # 스탬프하면 원문 그림과 이중으로 겹쳐 보인다. 임계값 0.30은
                # 실측 분포(문제 블록 53.7~93% vs 정상 블록 ≤2%)의 빈 구간 안.
                rect_area = rect.width * rect.height
                if rect_area > 0 and any(
                    _rect_overlap_area(rect, region) / rect_area >= 0.30
                    for region in image_regions
                ):
                    result.kept += 1
                    result.specialist_kept["figure_text"] = (
                        result.specialist_kept.get("figure_text", 0) + 1
                    )
                    result.warnings.append(f"p{pno}: 그림 위 텍스트 — 원문 보존")
                    continue
                fs_cqw = tb.get("fs") or ob.get("fs") or estimate_font_size_cqw(
                    tb.get("bbox"), str(tb.get("content") or ""), aspect,
                ) or 1.8
                base_pt = min(_MAX_FONT_PT, max(
                    _MIN_FONT_PT, fs_cqw / 100 * page.rect.width))
                # 같은 pt에서 AppleMyungjo/Noto Serif CJK는 Times 계열 영문보다
                # 시각적 몸통이 조금 작다. 제목은 계층을 잃지 않도록 더 보정하고,
                # 본문은 3%만 보정해 원문과 비슷한 잉크 밀도를 유지한다.
                base_pt *= 1.06 if block_type == "title" else 1.03
                obstacles = [
                    r
                    for i, (r, block) in enumerate(zip(block_rects, oblocks))
                    if (
                        i != block_index
                        and r is not None
                        and (
                            str(block.get("content") or "").strip()
                            or block.get("image")
                        )
                    )
                ]
                max_rect = _free_growth_rect(page, rect, obstacles)
                if ob.get("font_style") == "sans":
                    block_fontfile, block_fontname = sans_ff, sans_name
                else:
                    block_fontfile, block_fontname = serif_ff, serif_name
                if block_type == "title":
                    lineheights = _TITLE_LINEHEIGHTS
                elif block_type in {
                    "caption", "image_caption", "table_caption",
                    "page_footnote", "footnote", "aside_text",
                }:
                    lineheights = _CAPTION_LINEHEIGHTS
                else:
                    lineheights = _BODY_LINEHEIGHTS
                align = {"center": 1, "right": 2, "justify": 3}.get(
                    str(ob.get("align") or ""), 0,
                )
                bold = bool(ob.get("bold"))
                plan = _plan_single_line(
                    page,
                    rect,
                    new,
                    base_pt,
                    block_fontname,
                    block_fontfile,
                    max_rect=max_rect,
                    align=align,
                    bold=bold,
                )
                if plan is None:
                    plan = _plan_shrink_to_fit(
                        page,
                        rect,
                        new,
                        base_pt,
                        block_fontname,
                        block_fontfile,
                        max_rect=max_rect,
                        align=align,
                        bold=bold,
                        lineheights=lineheights,
                    )
                if plan is None:
                    result.kept += 1
                    result.warnings.append(
                        f"p{pno}: 블록 교체 생략(공간 부족) — 원문 보존"
                    )
                    continue
                targets.append(_Replacement(
                    plan,
                    new,
                    "text",
                    _source_text_rect(page, rect, source_spans),
                    block_fontname,
                    block_fontfile,
                ))

            if not targets:
                continue

            # 2) 원문 텍스트 리댁션 (이미지·그래픽 보존) — 삽입 전에 일괄 적용
            redact_rects = []
            for target in targets:
                # 삽입 bbox가 아래 빈 공간으로 커져도 실제 원문 bbox만 지운다.
                # 확장 사각형 전체를 리댁션하면 인접한 원문 글리프가 함께 사라질 수 있다.
                rr = target.redact_rect if target.redact_rect is not None else target.plan.rect
                redact_rects.append(+rr)
                page.add_redact_annot(rr)
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
            # 구멍이 나므로 절대 블록 rect 전체로 걸지 않는다.
            emoji_boxes = []
            for bbox in raster_rects:
                area = bbox.width * bbox.height
                for rr in redact_rects:
                    # Quartz 반올림으로 이미지가 글리프 상자를 1pt 미만 벗어나는
                    # 경우가 있어 1pt 허용 오차로 '완전 포함'을 판정한다.
                    if (
                        bbox.x0 >= rr.x0 - 1.0
                        and bbox.y0 >= rr.y0 - 1.0
                        and bbox.x1 <= rr.x1 + 1.0
                        and bbox.y1 <= rr.y1 + 1.0
                        and 0 < area <= rr.width * rr.height * 0.25
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

            # 3) 번역 텍스트 삽입
            for target in targets:
                _insert_fitted_text(
                    page,
                    target.plan,
                    target.text,
                    target.fontname,
                    target.fontfile,
                )
                result.replaced += 1
                if target.plan.expanded:
                    result.relocated += 1
                if target.kind == "table":
                    result.table_cells_replaced += 1

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
