"""번역 PDF 내보내기 — 원본 PDF 위에 번역 텍스트를 재배치한다.

원리(레이아웃 보존 번역): layout.json(원문)과 layout.{lang}.json(번역본)을 블록
단위로 비교해 내용이 실제로 바뀐 텍스트 블록만 원본 페이지에서 리댁션(텍스트만
제거, 이미지·그래픽 보존)하고 같은 자리에 번역 텍스트를 삽입한다. 번역 엔진이
마스킹으로 원문을 유지한 블록(수식·식별자·페이지 번호)과 그림·표는 원본
글리프가 그대로 남으므로 시각 품질이 보존된다.

폰트: PDF_EXPORT_FONT(파일 경로) → 시스템 한글 폰트 후보 → PyMuPDF 내장
CJK("korea", Droid Sans Fallback) 순으로 폴백한다. 내장 폰트는 어떤 환경에서도
존재하므로 내보내기가 폰트 때문에 실패하지는 않는다.
"""
from __future__ import annotations

import json
import logging
import re
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
    "page_footnote", "footnote", "aside_text", "header", "footer", "ref_text",
})
_SPECIALIST_TYPES = frozenset({"table", "equation", "algorithm"})

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

# insert_textbox 축소 사다리 — layout-fit.js와 같은 정신: 최대 45%까지 줄여 본다.
_SHRINK_STEPS = (1.0, 0.9, 0.8, 0.7, 0.62, 0.55)
_MIN_FONT_PT = 4.0
_MAX_FONT_PT = 72.0
_MAX_TABLE_CELLS = 500  # search_for 셀별 탐색의 CPU 상한 + 비정상 HTML 표 방어


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


@dataclass(frozen=True)
class _Replacement:
    plan: _TextFitPlan
    text: str
    kind: str = "text"
    redact_rect: object | None = None


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


def _resolve_font(explicit: str = "") -> tuple[str | None, str]:
    """(fontfile|None, fontname) 반환. 파일 폰트는 로드 검증 후 채택한다."""
    fitz = quiet_fitz()
    candidates = ([explicit] if explicit else []) + list(_SYSTEM_FONT_CANDIDATES)
    for path in candidates:
        p = Path(path)
        if not p.is_file():
            if path == explicit:
                logger.warning("PDF_EXPORT_FONT 파일이 없습니다: %s — 폴백 사용", path)
            continue
        try:
            font = fitz.Font(fontfile=str(p))
            if font.has_glyph(ord("한")):
                return str(p), "uocr-ko"
            logger.warning("폰트에 한글 글리프가 없습니다: %s — 폴백 사용", path)
        except Exception:  # noqa: BLE001 — 손상/미지원 포맷은 다음 후보로
            logger.warning("폰트 로드 실패: %s — 폴백 사용", path)
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


def _free_growth_rect(page, rect, obstacles: list[object]) -> object:
    """현재 블록 바로 아래의 빈 세로 영역까지만 확장 가능한 Rect를 반환한다.

    가로로 사실상 겹치지 않는 사이드바/다른 단은 장애물로 보지 않는다. 같은 단의
    다음 블록·표·그림·푸터 앞 2pt에서 멈추므로 번역문 확장이 이웃 내용을 덮지 않는다.
    """
    limit = page.mediabox.y1
    for other in obstacles:
        if other is rect or other.y0 < rect.y1 + 0.5:
            continue
        overlap = _rect_horizontal_overlap(rect, other)
        if overlap < min(rect.width, other.width) * 0.15:
            continue
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
        kwargs = {
            "fontsize": size,
            "fontname": fontname,
            "fontfile": fontfile,
            "align": align,
            "rotate": rot,
            "color": (0, 0, 0),
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
                return _TextFitPlan(+candidate, size, expanded, align, bold)
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
    }
    if plan.bold:
        kwargs.update({
            "render_mode": 2,
            "fill": (0, 0, 0),
            "border_width": 0.02,
        })
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
    ff, fname = _resolve_font(fontfile)

    result = PdfExportResult(path=job_dir / f"export.{lang}.pdf")
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
                                page, cell, new_text, base_pt, fname, ff, max_rect=cell,
                            )
                            if plan is None:
                                result.kept += 1
                                table_failed = True
                                result.warnings.append(
                                    f"p{pno}: 표 {ri + 1}행 {ci + 1}열 번역 생략(공간 부족)"
                                    " — 원문 셀 보존"
                                )
                                continue
                            targets.append(_Replacement(plan, new_text, "table", cell))
                    if table_failed:
                        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
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
                fs_cqw = tb.get("fs") or ob.get("fs") or estimate_font_size_cqw(
                    tb.get("bbox"), str(tb.get("content") or ""), aspect,
                ) or 1.8
                base_pt = min(_MAX_FONT_PT, max(
                    _MIN_FONT_PT, fs_cqw / 100 * page.rect.width))
                obstacles = [r for i, r in enumerate(block_rects)
                             if i != block_index and r is not None]
                max_rect = _free_growth_rect(page, rect, obstacles)
                plan = _plan_shrink_to_fit(
                    page, rect, new, base_pt, fname, ff, max_rect=max_rect,
                    align={"center": 1, "right": 2, "justify": 3}.get(
                        str(ob.get("align") or ""), 0,
                    ),
                    bold=bool(ob.get("bold")),
                )
                if plan is None:
                    result.kept += 1
                    result.warnings.append(
                        f"p{pno}: 블록 교체 생략(공간 부족) — 원문 보존"
                    )
                    continue
                targets.append(_Replacement(
                    plan, new, "text", _source_text_rect(page, rect, source_spans),
                ))

            if not targets:
                continue

            # 2) 원문 텍스트 리댁션 (이미지·그래픽 보존) — 삽입 전에 일괄 적용
            for target in targets:
                # 삽입 bbox가 아래 빈 공간으로 커져도 실제 원문 bbox만 지운다.
                # 확장 사각형 전체를 리댁션하면 인접한 원문 글리프가 함께 사라질 수 있다.
                page.add_redact_annot(
                    target.redact_rect if target.redact_rect is not None else target.plan.rect
                )
            # 텍스트만 제거한다. graphics 기본값(REMOVE_IF_COVERED)을 그대로 두면
            # 블록 안의 밑줄·도형·차트 선까지 사라져 "레이아웃 보존"을 위반한다.
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

            # 3) 번역 텍스트 삽입
            for target in targets:
                _insert_fitted_text(page, target.plan, target.text, fname, ff)
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
