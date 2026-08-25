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

import functools
import json
import logging
import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from statistics import median

from .layout import estimate_font_size_cqw
from .pdf import quiet_fitz

logger = logging.getLogger(__name__)

# 캐시된 export PDF가 이전 조판 규칙으로 생성됐는지 판별하는 공개 포맷 버전.
# 조판 결과가 달라지는 변경에서는 반드시 올려 기존 잡도 다음 요청 때 재생성한다.
PDF_EXPORT_FORMAT_VERSION = 5

# 번역 텍스트로 교체할 수 있는 블록 타입. 이 밖의 타입(image·equation·table·
# algorithm 등)은 내용이 달라도 원본을 유지한다 — 표 HTML·수식 LaTeX를 평문으로
# 밀어 넣으면 오히려 품질이 나빠진다.
_REPLACEABLE_TYPES = frozenset({
    "text", "title", "list", "caption", "image_caption", "table_caption",
    "page_footnote", "footnote", "aside_text",
})
_SPECIALIST_TYPES = frozenset({"table", "equation", "algorithm"})
# 참고문헌은 제목을 억지로 번역하면 저자명·학술지명·URL 사이에 서로 다른 문자 폭이
# 섞여 원문보다 훨씬 불안정하게 줄바꿈된다. 학술 번역 관례대로 서지 항목은 원문
# 조판을 그대로 보존한다(본문의 인용 번호와 참고문헌 제목은 계속 검색 가능).
_PRESERVE_TYPES = frozenset({"ref_text", "header", "footer"})

# 세로쓰기 블록은 회전 조합이 페이지 회전과 얽혀 배치가 어긋나기 쉽다 — 원본 유지.
_VERTICAL_SKIP = ("up", "down")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|mathrm|operatorname|mathbf|mathit|mathsf|mathtt|mathcal)\s*\{([^{}]*)\}"
)
_LATEX_SUP_RE = re.compile(r"\^(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_SUB_RE = re.compile(r"_(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_TEXT_BASELINE_RE = re.compile(r"1 0 0 1 [-+0-9.]+ ([-+0-9.]+) Tm")
_TEXT_ORIGIN_RE = re.compile(
    r"1 0 0 1 ([-+0-9.]+) ([-+0-9.]+) Tm"
)
_TEXT_HEX_RUN_RE = re.compile(r"<([0-9A-Fa-f]+)>")
_SUPERSCRIPT_MAP = str.maketrans({
    **dict(zip("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")),
    "i": "ⁱ", "n": "ⁿ",
})
_SUBSCRIPT_MAP = str.maketrans({
    **dict(zip("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")),
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
})
_UNICODE_SUPERSCRIPT_ASCII = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ",
    "0123456789+-=()in",
)
_LATEX_COMMANDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω",
    "oplus": "⊕", "otimes": "⊗", "times": "×", "pm": "±", "mp": "∓",
    "in": "∈", "notin": "∉", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "neq": "≠", "approx": "≈", "sim": "∼", "to": "→",
    "rightarrow": "→", "leftarrow": "←", "ldots": "…", "cdots": "…",
    "dots": "…", "infty": "∞", "partial": "∂", "nabla": "∇",
    "forall": "∀", "exists": "∃", "cup": "∪", "cap": "∩",
    "left": "", "right": "", "quad": " ", "qquad": "  ",
}
_LITERAL_LBRACE = "\uf000"
_LITERAL_RBRACE = "\uf001"
_TITLE_PREFIX_RE = re.compile(r"^([A-Z]|\d+(?:\.\d+)*)(?=\s)")

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

# 과도한 축소는 한 블록만 각주처럼 작아지는 계층 붕괴를 만든다. 76%에서도
# 들어가지 않으면 원문을 보존하고 리포트에 남기는 편이 읽을 수 없는 번역보다 낫다.
# 65%까지 열어 실측한 결과 회수는 0건이고 본문 최소 한글 크기만 7.02pt→6.00pt로
# 줄었다(16p 논문 재현본). 미번역의 주 원인은 축소 부족이 아니라 OCR의 가로
# 평탄화와 flow 그룹의 전부-아니면-전무 실패다(보존 26건 → 4건: 개별 배치 13건,
# `_reflow_flattened_text` 9건 회수). 축소 하한은 76%로 유지한다.
_SHRINK_STEPS = (1.0, 0.94, 0.88, 0.82, 0.76)
_SINGLE_LINE_SCALES = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76)
_MIN_FONT_PT = 4.0
# 본문 흐름 조판의 가독성 절대 하한. 논문 인쇄에서 6pt는 표 각주·판권 표기의
# 최소 크기이고 그 아래는 한글 받침이 뭉친다. 알고리즘 리스팅처럼 base_pt가
# 7.9pt 미만인 블록은 76% 축소만으로도 6pt 밑으로 내려가므로 비율이 아니라 실제
# pt로 막는다(실측 회수 손실 0건). 하한에 걸린 블록은 원문을 보존한다(사유: no_fit).
_MIN_BODY_FONT_PT = 6.0
_MAX_FONT_PT = 72.0
_MAX_TABLE_CELLS = 500  # search_for 셀별 탐색의 CPU 상한 + 비정상 HTML 표 방어
_MIN_TABLE_FONT_PT = 6.0
# booktabs rule의 안티앨리어싱 끝과 CJK 실제 잉크 사이에 남길 최소 간격.
# 0.5pt는 1200dpi에서 약 8px라 선과 첫 행 글자가 하나의 component로 붙지 않는다.
_TABLE_RULE_TEXT_GAP_PT = 0.5
# AppleMyungjo 실측에서 1.36까지 실제 span bbox가 겹쳤다. 1.44는 10/12pt에서
# 0.65/0.78pt의 여유가 있어 합성 bold stroke와 렌더 반올림도 견딘다.
_BODY_LINEHEIGHTS = (1.52, 1.48, 1.44)
_CAPTION_LINEHEIGHTS = (1.48, 1.44)
_TITLE_LINEHEIGHTS = (1.48, 1.44)
_BLOCK_GAP_PT = 5.0
# 서로 인접한 본문을 개별적으로 탐욕 배치하면 앞 블록이 여백을 먼저 차지해 뒤
# 제목/문단이 원문으로 되돌아간다. 이 거리 안의 같은 단 블록은 실패 시 하나의
# flow로 다시 계획한다. 글자 크기와 1.44 행간 하한은 그대로 유지한다.
_FLOW_JOIN_GAP_PT = 22.0
_FLOW_UPWARD_SLACK_PT = 48.0
_FLOW_OBSTACLE_GAP_PT = 0.9


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
    kept_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def keep(self, reason: str, count: int = 1) -> None:
        """보존 블록 수와 사유를 함께 기록한다.

        경고는 사용자에게 보여줄 만한 이상 징후만 남기므로 `kept`의 일부만
        설명한다. 다음 미번역 신고를 코드 없이 진단하려면 보존된 *모든* 블록의
        사유가 필요하다. `kept_reasons`의 합은 항상 `kept`와 같다.
        """
        self.kept += count
        self.kept_reasons[reason] = self.kept_reasons.get(reason, 0) + count

    def report(self) -> dict:
        """경로·본문 없이 UI에 안전하게 노출할 ASCII/숫자 중심 생성 리포트."""
        return {
            "format_version": PDF_EXPORT_FORMAT_VERSION,
            "replaced": self.replaced,
            "kept": self.kept,
            "relocated": self.relocated,
            "table_cells_replaced": self.table_cells_replaced,
            "specialist_kept": dict(sorted(self.specialist_kept.items())),
            # 교체 대상 타입이 아닌 블록(image/equation/algorithm 등)은 애초에
            # kept로 세지 않고 specialist_kept로만 집계한다.
            "kept_reasons": dict(sorted(self.kept_reasons.items())),
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


class _TableParser(HTMLParser):
    """번역 레이아웃의 HTML table을 span 정보와 함께 파싱한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_RawTableCell]] = []
        self._row: list[_RawTableCell] | None = None
        self._cell: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            values = {str(key).lower(): str(value) for key, value in attrs if value is not None}
            try:
                self._rowspan = max(1, int(values.get("rowspan", "1")))
                self._colspan = max(1, int(values.get("colspan", "1")))
            except ValueError:
                self._rowspan = self._colspan = 1
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_RawTableCell(
                _plain_text("".join(self._cell)),
                self._rowspan,
                self._colspan,
            ))
            self._cell = None
            self._rowspan = self._colspan = 1
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


@functools.lru_cache(maxsize=8)
def _metrics_font(fontfile: str | None, fontname: str):
    """측정 전용 fitz.Font 재사용 캐시.

    Font 생성은 폰트 파일 전체를 파싱한다(실측: AppleMyungjo 18MB 3.1ms,
    AppleSDGothicNeo 55MB 8.6ms). 계획 함수들은 text_length/has_glyph/ascender
    같은 읽기 전용 측정에만 쓰므로 내보내기 한 번에 serif/sans/table 폰트를
    공유해도 조판 결과가 같다. `_resolve_font`는 손상 폰트 검증이 목적이고
    빌드당 몇 회뿐이라 캐시하지 않는다.
    """
    fitz = quiet_fitz()
    return fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(fontname=fontname)


def _document_font_resource_names(doc) -> set[str]:
    """원본 문서에 이미 존재하는 페이지 font resource 이름을 모은다."""
    names: set[str] = set()
    for page_number in range(doc.page_count):
        try:
            fonts = doc.load_page(page_number).get_fonts()
        except Exception:  # noqa: BLE001 — 충돌 회피용 보조 정보
            continue
        for font in fonts:
            if len(font) > 4 and font[4]:
                names.add(str(font[4]))
    return names


def _unique_font_resource_name(base: str, used: set[str]) -> str:
    """원본 page resource를 재사용하지 않는 삽입용 fontname을 예약한다."""
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _load_pages(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PdfExportError(f"레이아웃 파일을 읽을 수 없습니다: {path.name}") from e
    if not isinstance(data, list):
        raise PdfExportError(f"레이아웃 파일 형식이 올바르지 않습니다: {path.name}")
    return data


def _script_text(value: str, table: dict[int, str], marker: str) -> str:
    """TeX 위/아래첨자 그룹을 Unicode로 낮추고 불가 문자는 명시적으로 감싼다."""
    value = value.strip()
    lowered = value.lower()
    # Noto Serif CJK를 포함한 흔한 CJK PDF 폰트는 아래첨자 글리프를 일부만
    # 제공한다. 실제 대상 폰트도 ₗ뿐 아니라 ₁/₂까지 누락했다. P(L), β(1)처럼
    # 읽을 수 있는 ASCII 괄호 표기가 빈 네모(tofu)보다 이식성과 검색성이 높다.
    if marker == "_":
        return f"({value})"
    # CJK 본문 폰트는 숫자 위첨자는 대체로 포함하지만 n 같은 라틴 위첨자
    # 글리프는 빠진 경우가 많다. Vⁿ이 NUL/빈 네모가 되는 대신 검색 가능한
    # ASCII 표기 V^(n)을 사용한다.
    if marker == "^" and any(ch.isalpha() for ch in value):
        return f"^({value})"
    if lowered and all(ord(ch) in table for ch in lowered):
        return lowered.translate(table)
    return f"{marker}({value})"


def _latex_command(match: re.Match[str]) -> str:
    command = match.group(1)
    # 모르는 명령도 역슬래시 원문을 그대로 노출하지 않는다. 명령 이름은 남겨
    # 손실을 최소화하고 PDF에서 제어 문자열처럼 보이는 시각 결함만 제거한다.
    return _LATEX_COMMANDS.get(command, command)


def _plain_text(content: str) -> str:
    """블록 내용 → 삽입용 평문.

    PDF textbox는 LaTeX를 조판하지 못하므로 흔한 inline 수식 표기를 읽을 수 있는
    유니코드 평문으로 낮춘다(`\\(E=mc^{2}\\)` → `E=mc²`). 복잡한 equation
    블록은 애초 교체 대상이 아니며 원본 조판을 유지한다.
    """
    text = _TAG_RE.sub(" ", content)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "").replace("$$", "")
    # literal set braces는 TeX grouping brace 제거와 구분해 끝까지 보존한다.
    text = text.replace("\\{", _LITERAL_LBRACE).replace("\\}", _LITERAL_RBRACE)
    # wrapper가 중첩되지 않은 일반 inline 표현을 여러 번 벗긴다.
    for _ in range(3):
        updated = _LATEX_WRAPPER_RE.sub(lambda m: m.group(1), text)
        if updated == text:
            break
        text = updated
    text = _LATEX_SUP_RE.sub(
        lambda m: _script_text(m.group(1) or m.group(2), _SUPERSCRIPT_MAP, "^"), text,
    )
    text = _LATEX_SUB_RE.sub(
        lambda m: _script_text(m.group(1) or m.group(2), _SUBSCRIPT_MAP, "_"), text,
    )
    text = _LATEX_COMMAND_RE.sub(_latex_command, text)
    # 남은 grouping braces는 평문에서 의미가 없고 줄 폭만 늘린다. literal set은 복원.
    text = text.replace("{", "").replace("}", "")
    text = text.replace(_LITERAL_LBRACE, "{").replace(_LITERAL_RBRACE, "}")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _protect_trailing_words(text: str) -> str:
    """자동 줄바꿈에서 짧은 마지막 한 단어가 고아행이 되지 않게 묶는다."""
    protected: list[str] = []
    for line in text.splitlines():
        # 페이지 끝에서 다음 페이지로 이어지는 미완결 인용은 `(Li et` /
        # `al., 2025;`처럼 저자 표기 한가운데가 갈라지기 쉽다. 닫는 괄호가 없는
        # 짧은 인용 꼬리만 한 덩어리로 묶어 일반 본문의 줄바꿈에는 영향이 없게 한다.
        citation_tail = re.search(
            r"(\([^()\n]{0,60}\bet\s+al\.,\s*\d{4}[a-z]?;\s*)$",
            line,
            re.IGNORECASE,
        )
        if citation_tail:
            start, end = citation_tail.span(1)
            preceding = list(re.finditer(r"\S+", line[:start]))
            if preceding:
                # 인용이 붙은 명사와 앞 절의 짧은 꼬리까지 함께 보내 마지막 행이
                # `al., 2025;` 또는 `프로젝트(...)` 한 조각만 남지 않게 한다.
                start = preceding[max(0, len(preceding) - 6)].start()
            # 긴 NBSP 묶음은 좁은 상자에서 `Li`나 `2025` 자체를 강제로 쪼갤 수
            # 있다. 자연 공백은 그대로 두고 안전한 단어 경계에 명시 행갈이만 둔다.
            before = line[:start].rstrip()
            tail = line[start:end].lstrip()
            protected.append(
                (before + "\n" if before else "") + tail + line[end:]
            )
            continue
        tokens = list(re.finditer(r"\S+", line))
        if len(tokens) < 4:
            protected.append(line)
            continue
        last = tokens[-1].group()
        if len(last) > 16 or "://" in last or "@" in last:
            protected.append(line)
            continue
        gap_start = tokens[-2].end()
        gap_end = tokens[-1].start()
        protected.append(line[:gap_start] + "\xa0" + line[gap_end:])
    return "\n".join(protected)


def _normalize_inline_spacing(text: str) -> str:
    """각주 위첨자와 뒤 문장부호 사이의 번역기 삽입 공백을 제거한다."""
    return re.sub(
        r"\s+([¹²³⁴⁵⁶⁷⁸⁹]+)\s*([.,;:!?])",
        r"\1\2",
        text,
    )


def _portable_text_for_font(text: str, fontfile: str | None) -> str:
    """선택 폰트가 빠뜨린 글리프만 검색 가능한 ASCII로 안전하게 낮춘다."""
    if not fontfile or not text:
        return text
    try:
        font = _metrics_font(fontfile, "")
    except Exception:  # noqa: BLE001 — 실제 삽입기의 기존 폴백을 유지한다
        return text
    output: list[str] = []
    for char in text:
        if char.isspace() or font.has_glyph(ord(char)):
            output.append(char)
            continue
        mapped = char.translate(_UNICODE_SUPERSCRIPT_ASCII)
        if mapped != char and all(font.has_glyph(ord(part)) for part in mapped):
            output.append(mapped)
            continue
        # macOS 명조처럼 Hangul은 있지만 precomposed Latin만 빠진 폰트에서도
        # tofu 대신 검색 가능한 기본 문자를 남긴다. 컨테이너 Noto는 ö/ü를 직접
        # 지원하므로 이 경로를 타지 않고 원 철자를 보존한다.
        decomposed = unicodedata.normalize("NFKD", char)
        ascii_base = "".join(part for part in decomposed if part.isascii())
        if ascii_base and all(font.has_glyph(ord(part)) for part in ascii_base):
            output.append(ascii_base)
        else:
            output.append(char)
    return "".join(output)


def _restore_title_prefix(original: str, translated: str) -> str:
    """번역 모델이 떨군 절/부록 식별자(A, 2.1 등)를 제목 앞에 복구한다."""
    source = _TITLE_PREFIX_RE.match(original)
    if source is None or _TITLE_PREFIX_RE.match(translated):
        return translated
    return f"{source.group(1)} {translated}"


def _balance_title_text(
    text: str,
    width: float,
    base_pt: float,
    fontname: str,
    fontfile: str | None,
) -> str:
    """허용 축소에서도 한 줄이 안 되는 제목을 1–2자 고아 없이 나눈다."""
    if not text or "\n" in text or width <= 1:
        return text
    fitz = quiet_fitz()
    try:
        font = _metrics_font(fontfile, fontname)
    except Exception:  # noqa: BLE001 — 기존 textbox wrapping으로 폴백
        return text

    def text_width(value: str, size: float) -> float:
        return (
            font.text_length(value, fontsize=size)
            if fontfile
            else fitz.get_text_length(value, fontname=fontname, fontsize=size)
        )

    if any(text_width(text, base_pt * scale) <= width + 0.25
           for scale in _SINGLE_LINE_SCALES):
        return text
    breakpoints = [match.start() for match in re.finditer(r"\s+", text)]
    if not breakpoints:
        breakpoints = list(range(3, len(text) - 2))
    for scale in _SHRINK_STEPS:
        size = max(_MIN_FONT_PT, base_pt * scale)
        options: list[tuple[float, int, str, str]] = []
        for point in breakpoints:
            left, right = text[:point].rstrip(), text[point:].lstrip()
            if len(re.sub(r"\s+", "", left)) < 3 or len(re.sub(r"\s+", "", right)) < 3:
                continue
            left_width, right_width = text_width(left, size), text_width(right, size)
            if max(left_width, right_width) > width + 0.25:
                continue
            options.append((
                abs(left_width - right_width),
                point,
                left,
                right,
            ))
        if options:
            _score, _point, left, right = min(options)
            return f"{left}\n{right}"
    return text


def _table_cells(content: str) -> tuple[list[_TableCell], int, int] | None:
    """HTML 표를 rowspan/colspan을 보존한 논리 셀과 격자 크기로 변환한다."""
    parser = _TableParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:  # noqa: BLE001 — 손상 OCR HTML은 표 번역만 보수적으로 생략
        return None
    if not parser.rows or not parser.rows[0]:
        return None

    occupied: dict[tuple[int, int], _TableCell] = {}
    cells: list[_TableCell] = []
    max_col = 0
    for row_index, raw_row in enumerate(parser.rows):
        col_index = 0
        for raw in raw_row:
            if (
                len(cells) >= _MAX_TABLE_CELLS
                or raw.rowspan > _MAX_TABLE_CELLS
                or raw.colspan > _MAX_TABLE_CELLS
                or raw.rowspan * raw.colspan > _MAX_TABLE_CELLS
            ):
                return None
            while (row_index, col_index) in occupied:
                col_index += 1
            cell = _TableCell(
                row_index,
                col_index,
                raw.rowspan,
                raw.colspan,
                raw.text,
            )
            for rr in range(row_index, row_index + raw.rowspan):
                for cc in range(col_index, col_index + raw.colspan):
                    if (rr, cc) in occupied:
                        return None
                    occupied[(rr, cc)] = cell
            cells.append(cell)
            col_index += raw.colspan
            max_col = max(max_col, col_index)

    row_count = max((row + 1 for row, _ in occupied), default=0)
    if row_count == 0 or max_col == 0:
        return None
    if any((row, col) not in occupied
           for row in range(row_count) for col in range(max_col)):
        return None
    return cells, row_count, max_col


def _table_matrix(content: str) -> list[list[str]] | None:
    """호환용 펼친 행렬. span 셀은 점유하는 모든 격자 위치에 같은 값을 둔다."""
    parsed = _table_cells(content)
    if parsed is None:
        return None
    cells, rows, cols = parsed
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                matrix[row][col] = cell.text
    return matrix


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


def _horizontal_table_rules(page, table_rect) -> list[tuple[float, float, float, float]]:
    """표 영역의 가로 rule을 `(x0, x1, y, width)`로 반환한다."""
    rules: list[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 — rule 보정은 품질 향상용
        return rules
    for drawing in drawings:
        width = max(0.1, float(drawing.get("width") or 0.1))
        for item in drawing.get("items", []):
            if not item or item[0] != "l" or len(item) < 3:
                continue
            start, end = item[1], item[2]
            if abs(float(start.y) - float(end.y)) > max(0.5, width):
                continue
            x0, x1 = sorted((float(start.x), float(end.x)))
            y = (float(start.y) + float(end.y)) / 2
            if (
                x1 < table_rect.x0 - 2
                or x0 > table_rect.x1 + 2
                or y < table_rect.y0 - 3
                or y > table_rect.y1 + 3
            ):
                continue
            rules.append((x0, x1, y, width))
    return rules


def _table_cell_rects(
    page,
    table_rect,
    original: list[_TableCell],
    rows: int,
    cols: int,
) -> tuple[list[object], bool]:
    """원문 셀 검색 중심으로 표 격자를 추정해 `(셀 사각형, 격자 신뢰 여부)`를 낸다.

    `grid_trusted=False`는 원문 셀을 거의 찾지 못해 균등 분할로 강행했다는 뜻이다
    — 호출부는 그런 표를 교체하지 말고 원문 그대로 보존해야 한다.
    """
    x_centers: list[list[float]] = [[] for _ in range(cols)]
    y_centers: list[list[float]] = [[] for _ in range(rows)]
    fitz = quiet_fitz()
    search_clip = +table_rect
    search_clip += (-2.0, -12.0, 2.0, 12.0)
    search_clip &= page.mediabox
    observed: list[object] = []
    for cell in original:
        if not cell.text:
            continue
        expected_x = table_rect.x0 + (
            cell.col + cell.colspan / 2
        ) * table_rect.width / cols
        expected_y = table_rect.y0 + (
            cell.row + cell.rowspan / 2
        ) * table_rect.height / rows
        hits = page.search_for(cell.text, clip=search_clip)
        if not hits:
            continue
        hit = min(hits, key=lambda r: abs((r.x0 + r.x1) / 2 - expected_x)
                  + abs((r.y0 + r.y1) / 2 - expected_y))
        observed.append(hit)
        if cell.colspan == 1:
            x_centers[cell.col].append((hit.x0 + hit.x1) / 2)
        if cell.rowspan == 1:
            y_centers[cell.row].append((hit.y0 + hit.y1) / 2)
    grid_x0 = min([table_rect.x0] + [hit.x0 - 2.0 for hit in observed])
    grid_x1 = max([table_rect.x1] + [hit.x1 + 2.0 for hit in observed])
    grid_y0 = min([table_rect.y0] + [hit.y0 - 1.0 for hit in observed])
    grid_y1 = max([table_rect.y1] + [hit.y1 + 1.0 for hit in observed])
    # 텍스트 레이어가 있는 표에서 원문 셀 검색이 절반도 맞지 않으면 아래 균등
    # 격자는 실제 열 폭과 무관하다 — 번역문이 엉뚱한 셀에 찍히고 인접 셀 원문이
    # 리댁션된다. 스캔 표(텍스트 레이어 없음)는 지울 원문이 없어 균등 격자가
    # 무해하므로 그대로 신뢰한다.
    try:
        has_source_text = bool(page.get_text("text", clip=search_clip).strip())
    except Exception:  # noqa: BLE001 — 추출 실패는 스캔 표와 같게 취급
        has_source_text = False
    grid_trusted = not has_source_text or (
        len(observed) * 2 >= sum(1 for cell in original if cell.text)
    )
    xs = _grid_boundaries(x_centers, grid_x0, grid_x1)
    ys = _grid_boundaries(y_centers, grid_y0, grid_y1)
    horizontal_rules = _horizontal_table_rules(page, table_rect)
    rects: list[object] = []
    for cell in original:
        rect = fitz.Rect(
            xs[cell.col],
            ys[cell.row],
            xs[cell.col + cell.colspan],
            ys[cell.row + cell.rowspan],
        )
        xpad = min(2.0, rect.width * 0.04)
        ypad = min(0.6, rect.height * 0.04)
        rect.x0 += xpad
        rect.x1 -= xpad
        rect.y0 += ypad
        rect.y1 -= ypad
        # OCR table bbox가 LaTeX `\toprule`보다 2–3pt 위에서 시작하면 CJK의
        # 큰 ascender가 rule에 관통된다. 셀 윗부분 *안쪽*에 들어온 rule만
        # 실제 상단으로 삼고, 가까운 다음 rule까지의 안전한 띠를 사용한다.
        spanning_rules = sorted(
            (y, width)
            for x0, x1, y, width in horizontal_rules
            if _rect_horizontal_overlap(
                fitz.Rect(x0, y - width / 2, x1, y + width / 2), rect,
            ) >= rect.width * 0.5
        )
        interior_top = next((
            (y, width)
            for y, width in spanning_rules
            if (
                rect.y0 - max(1.0, width / 2 + 0.25)
                <= y
                <= rect.y0 + min(3.5, rect.height * 0.35)
            )
        ), None)
        if interior_top is not None:
            top_y, top_width = interior_top
            # 기존처럼 rule이 셀 내부에 충분히 들어온 경우 0.25pt면 안전하다.
            # 표 최상단 rule도 기존 header 여백을 유지한다. 그 밖의 내부 rule을
            # OCR 중심 경계가 이미 지나친 경우에만 더 큰 간격을 쓴다.
            gap = (
                0.25
                if top_y <= table_rect.y0 + 4.0
                or top_y > rect.y0 + 0.25
                else _TABLE_RULE_TEXT_GAP_PT
            )
            rect.y0 = max(
                rect.y0,
                top_y + top_width / 2 + gap,
            )
            lower = next((
                (y, width) for y, width in spanning_rules
                if y > top_y + 1.0 and y <= rect.y1 + 2.0
            ), None)
            if lower is not None:
                lower_y, lower_width = lower
                rect.y1 = max(rect.y1, lower_y - lower_width / 2 - 0.1)
        rects.append(rect)
    return rects, grid_trusted


def _table_cell_source_style(
    page, cell_rect, records: list[_SourceSpan] | None = None,
) -> tuple[float, int, bool, object]:
    """원본 셀의 실제 span 크기·정렬·굵기와 텍스트 redaction 영역을 추정한다.

    `records`는 페이지에서 이미 한 번 읽어둔 span 목록이다. 넘기지 않으면 이
    함수가 직접 추출하지만, 셀마다 페이지 전체를 재추출하면 큰 표에서 비용이
    셀 수만큼 곱해진다.
    """
    if records is None:
        records = _source_span_records(quiet_fitz(), page)
    # `get_text(..., clip=...)`는 경계에 걸친 이웃 span을 잘라서 반환한다.
    # 그 잘린 조각까지 현재 셀의 redaction으로 합치면, 왼쪽 셀 번역이 오른쪽
    # 값의 첫 글자들을 지우는 일이 생긴다. 전체 span의 중심이 실제 셀 안에
    # 있는 경우만 소유하게 해 인접 셀 경계를 침범하지 않는다.
    spans = [
        span for span in records
        if cell_rect.contains((span.rect.tl + span.rect.br) / 2)
    ]

    if not spans:
        size = min(12.0, max(_MIN_FONT_PT, cell_rect.height * 0.65))
        return size, 0, False, +cell_rect

    sizes = [span.size for span in spans if span.size > 0]
    source_size = median(sizes) if sizes else cell_rect.height * 0.65
    ink = +spans[0].rect
    bold_chars = total_chars = 0
    for span in spans:
        ink.include_rect(span.rect)
        count = max(1, len(span.text.strip()))
        total_chars += count
        if span.flags & 16:
            bold_chars += count

    center = (ink.x0 + ink.x1) / 2
    if abs(center - (cell_rect.x0 + cell_rect.x1) / 2) <= cell_rect.width * 0.12:
        align = 1
    elif ink.x1 >= cell_rect.x1 - cell_rect.width * 0.08:
        align = 2
    else:
        align = 0
    ink += (-0.4, -0.3, 0.4, 0.3)
    ink &= cell_rect
    return min(12.0, max(_MIN_FONT_PT, source_size * 1.03)), align, (
        bold_chars > total_chars * 0.5
    ), ink


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


def _source_span_records(fitz, page) -> list[_SourceSpan]:
    """페이지 텍스트 span과 소유권·스타일 판정에 필요한 메타를 읽는다."""
    out: list[_SourceSpan] = []
    try:
        blocks = page.get_text("dict").get("blocks", ())
    except Exception:  # noqa: BLE001 — 텍스트 레이어가 없으면 빈 목록
        return out
    for block in blocks:
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                bbox = span.get("bbox")
                text = str(span.get("text") or "")
                if bbox and len(bbox) == 4 and text.strip():
                    origin = span.get("origin") or (bbox[0], bbox[3])
                    out.append(_SourceSpan(
                        fitz.Rect(bbox),
                        text,
                        float(span.get("size") or 0.0),
                        int(span.get("flags") or 0),
                        (float(origin[0]), float(origin[1])),
                    ))
    return out


def _source_span_rects(fitz, page) -> list[object]:
    """호환용 페이지 텍스트 span 사각형 목록."""
    return [span.rect for span in _source_span_records(fitz, page)]


def _ownership_text(value: object) -> str:
    """source span과 OCR block의 느슨한 내용 대응을 위한 정규화 문자열."""
    return re.sub(r"[^\w]+", "", _plain_text(str(value or "")).casefold())


def _source_span_matches_rect(span: _SourceSpan, rect) -> bool:
    """OCR 오차를 허용하면서 source span이 블록에 실질적으로 속하는지 판정한다."""
    center = (span.rect.tl + span.rect.br) / 2
    inside = (
        rect.x0 - 1 <= center.x <= rect.x1 + 1
        and rect.y0 - 1 <= center.y <= rect.y1 + 1
    )
    span_area = max(0.01, span.rect.width * span.rect.height)
    overlap = _rect_overlap_area(span.rect, rect) / span_area
    return inside or overlap >= 0.35


def _assign_source_spans(
    page,
    block_rects: list[object | None],
    blocks: list[dict],
    source_spans: list[_SourceSpan],
) -> tuple[dict[int, list[_SourceSpan]], list[_SourceSpan], set[int]]:
    """각 PDF span을 최대 한 OCR block에만 배정한다.

    OCR bbox는 서로 겹칠 수 있다. 블록마다 독립적으로 center 포함 여부를 검사하면
    같은 원문 span을 두 교체가 모두 redaction하여, 뒤 블록이 fallback해도 제목이
    사라질 수 있다. 내용 일치와 기하 점수를 함께 사용해 유일 소유자를 정한다.
    """
    owned: dict[int, list[_SourceSpan]] = {
        index: [] for index in range(len(block_rects))
    }
    unowned: list[_SourceSpan] = []
    ambiguous_blocks: set[int] = set()
    block_texts = [
        _ownership_text(block.get("content")) if isinstance(block, dict) else ""
        for block in blocks
    ]
    for span in source_spans:
        center = (span.rect.tl + span.rect.br) / 2
        span_area = max(0.01, span.rect.width * span.rect.height)
        span_text = _ownership_text(span.text)
        span_fragments = {
            _ownership_text(token)
            for token in re.findall(r"\w+", _plain_text(span.text).casefold())
            if len(_ownership_text(token)) >= 3
        }
        fragment_owners = {
            fragment: {
                index
                for index, rect in enumerate(block_rects)
                if rect is not None
                and block_texts[index]
                and _source_span_matches_rect(span, rect)
                and fragment in block_texts[index]
            }
            for fragment in span_fragments
        }
        choices: list[tuple[tuple[float, ...], int]] = []
        for index, rect in enumerate(block_rects):
            if rect is None:
                continue
            # list/table 같은 구조 컨테이너가 자식 텍스트 bbox를 넓게 감싸더라도
            # 빈 컨테이너가 실제 글리프를 소유해서는 안 된다.
            if not block_texts[index]:
                continue
            inside = (
                rect.x0 - 1 <= center.x <= rect.x1 + 1
                and rect.y0 - 1 <= center.y <= rect.y1 + 1
            )
            overlap = _rect_overlap_area(span.rect, rect) / span_area
            if not inside and overlap < 0.35:
                continue
            block_text = block_texts[index]
            content_match = bool(
                block_text
                and (
                    (
                        min(len(span_text), len(block_text)) >= 3
                        and (span_text in block_text or block_text in span_text)
                    )
                    or any(
                        owners == {index}
                        for owners in fragment_owners.values()
                    )
                )
            )
            distance = abs(center.x - (rect.x0 + rect.x1) / 2) + abs(
                center.y - (rect.y0 + rect.y1) / 2
            )
            area = max(0.01, rect.width * rect.height)
            choices.append((
                (
                    float(content_match),
                    float(inside),
                    min(1.0, overlap),
                    -distance / max(1.0, page.rect.width + page.rect.height),
                    -area / max(1.0, page.rect.width * page.rect.height),
                    -float(index),
                ),
                index,
            ))
        if not choices:
            unowned.append(span)
            continue
        matching_blocks = {
            index for score, index in choices if score[0] > 0.5
        }
        if len(matching_blocks) > 1:
            # 하나의 PDF span이 여러 OCR 블록의 문자열을 함께 품으면 어느 한쪽이
            # 전체 span을 지우는 순간 다른 블록의 원문도 손실된다. 해당 span과
            # 연관된 블록을 모두 원문 보존 대상으로 표시한다.
            unowned.append(span)
            ambiguous_blocks.update(matching_blocks)
            continue
        _score, owner = max(choices)
        owned[owner].append(span)
    return owned, unowned, ambiguous_blocks


def _leading_bold_prefix(
    source_spans: list[_SourceSpan], translated: str,
) -> tuple[str, str] | None:
    """원문 첫 행의 짧은 bold run을 번역문 접두 라벨에 대응시킨다."""
    if not source_spans or not translated:
        return None
    ordered = sorted(source_spans, key=lambda span: (span.origin[1], span.rect.x0))
    first_y = ordered[0].origin[1]
    first_line = [span for span in ordered if abs(span.origin[1] - first_y) <= 1.25]
    first_line.sort(key=lambda span: span.rect.x0)

    bold_started = False
    bold_text: list[str] = []
    leading_source: list[str] = []
    for span in first_line:
        is_bold = bool(span.flags & 16)
        if is_bold:
            bold_started = True
            bold_text.append(span.text)
        elif bold_started:
            break
        else:
            leading_source.append(span.text)
    source_prefix = "".join(bold_text).strip()
    word_count = len(re.findall(r"\S+", source_prefix))
    if not source_prefix or not (1 <= word_count <= 6) or len(source_prefix) > 80:
        return None
    source_lead = "".join(leading_source).strip()
    marker_re = re.compile(r"(?:[-•·–—]|\(?[A-Za-z0-9]{1,3}[.)])")
    if source_lead and marker_re.fullmatch(source_lead) is None:
        # 문장 중간의 강조(`This is ` + bold `important`)를 run-in label로
        # 오인하면 번역문의 첫 단어가 임의로 굵어진다. 첫 bold run은 문두 또는
        # 목록/번호 marker 바로 뒤에 있을 때만 구조적 접두 라벨로 취급한다.
        return None

    translated_first = translated.splitlines()[0].strip()
    if not translated_first:
        return None
    # 번역기가 run-in label 뒤에 명시적 줄바꿈을 남긴 경우 첫 행 전체가 라벨이다.
    if "\n" in translated:
        return "", translated_first

    tokens = list(re.finditer(r"\S+", translated_first))
    if not tokens:
        return None
    leading = ""
    start = 0
    if source_lead and marker_re.fullmatch(tokens[0].group()):
        leading = translated_first[:tokens[0].end()] + " "
        start = 1
    if len(tokens) < start + word_count:
        return None
    prefix_start = tokens[start].start()
    prefix_end = tokens[start + word_count - 1].end()
    if not leading:
        leading = translated_first[:prefix_start]
    prefix = translated_first[prefix_start:prefix_end]
    return (leading, prefix) if prefix else None


def _visual_lines(spans: list[_SourceSpan], base_pt: float) -> list[str]:
    """원문 PDF에서 이 블록이 실제로 차지한 시각 줄의 텍스트를 위→아래로 만든다.

    같은 줄의 span은 baseline이 정확히 같지 않다. 아래/위 첨자(β1, NS)는 약
    0.1em, 표 헤더의 세로 가운데 맞춤 셀은 0.5em까지 어긋난다. 반면 다음 행은
    1.0em 이상 떨어진다(실측 /tmp/p9repro 9.24pt 표: 첨자 1.0pt, 가운데 맞춤
    헤더 4.98pt, 행 간격 8.96~10.36pt). 0.55em 경계가 그 빈 구간 안이다.
    """
    if not spans:
        return []
    tolerance = max(1.25, base_pt * 0.55)
    clusters: list[list[_SourceSpan]] = []
    anchor: float | None = None
    for span in sorted(spans, key=lambda item: item.origin[1]):
        if anchor is None or span.origin[1] - anchor > tolerance:
            clusters.append([])
            anchor = span.origin[1]
        clusters[-1].append(span)
    return [
        "".join(span.text for span in sorted(cluster, key=lambda item: item.rect.x0))
        for cluster in clusters
    ]


def _align_ocr_lines(ocr_lines: list[str], visual_texts: list[str]) -> list[int] | None:
    """각 OCR 논리 줄이 원문의 몇 번째 시각 줄에서 왔는지 단조 정렬로 찾는다."""
    normalized = [_ownership_text(text) for text in visual_texts]
    if not normalized:
        return None
    groups: list[int] = []
    index, remaining = 0, normalized[0]
    for line in ocr_lines:
        target = _ownership_text(line)
        if not target:
            groups.append(index)
            continue
        # OCR은 합자·첨자·특수기호를 원문 span과 다르게 펼치므로 줄 전체가 항상
        # 일치하지는 않는다. 앞 10자로 위치만 찾고 일치하는 만큼만 소비한다.
        probe = target[:10]
        cursor, rest = index, remaining
        while probe not in rest and cursor + 1 < len(normalized):
            cursor += 1
            rest = normalized[cursor]
        if probe not in rest:
            return None
        consumed = (
            rest.index(target) + len(target)
            if target in rest
            else rest.index(probe) + len(probe)
        )
        index, remaining = cursor, rest[consumed:]
        groups.append(index)
    return groups


def _reflow_flattened_text(
    original: str, translated: str, spans: list[_SourceSpan], base_pt: float,
) -> str | None:
    """가로 배치가 OCR 줄바꿈으로 평탄화된 블록의 번역문을 원문 줄 수로 되접는다.

    원문에서 한 줄이던 표 헤더 `Dataset | Long Evaluation | Short Evaluation`이
    OCR에서 세 줄로 평탄화되면 bbox 높이는 한 줄뿐인데 textbox 조판은 세 줄을
    요구한다 — 어떤 크기로 축소해도 구조적으로 들어가지 않는다. 원문 span의
    baseline으로 실제 줄 수를 재고 그만큼만 줄바꿈을 남기면 원문의 시각적 모습에
    가까우면서 조판도 가능해진다.

    진짜 여러 줄 문단을 뭉개지 않도록 조건을 좁게 건다: 원문 시각 줄 수가 OCR
    줄 수보다 적고, 번역문의 줄 수가 원문과 1:1로 대응하며, 모든 OCR 줄을 원문
    시각 줄에 단조 정렬할 수 있을 때만 리플로우한다. 하나라도 어긋나면 None을
    반환해 기존 보존 경로를 그대로 탄다.
    """
    original_lines = original.splitlines()
    translated_lines = translated.splitlines()
    if len(original_lines) < 2 or len(original_lines) != len(translated_lines):
        return None
    visual_texts = _visual_lines(spans, base_pt)
    if not visual_texts or len(visual_texts) >= len(original_lines):
        return None
    groups = _align_ocr_lines(original_lines, visual_texts)
    if groups is None:
        return None
    merged = [
        " ".join(
            translated_lines[index].strip()
            for index, group in enumerate(groups)
            if group == line and translated_lines[index].strip()
        )
        for line in range(groups[-1] + 1)
    ]
    reflowed = "\n".join(line for line in merged if line)
    return reflowed if reflowed and reflowed != translated else None


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


def _source_text_rects(page, rect, source_spans: list[object]) -> tuple[object, ...]:
    """원문 span을 행별 비연속 redaction 사각형으로 반환한다.

    검출 bbox가 URL·긴 단어의 끝을 몇 pt 잘라내는 경우 OCR 사각형만 지우면 `f.`
    같은 원문 꼬리가 번역문 옆에 남는다. 반대로 여러 행의 bounding union을 하나로
    지우면 그 사이에 낀 다른 owner의 제목까지 사라지므로 같은 행에서 닿는 span만
    병합하고 각 행을 별도 annotation으로 처리한다.
    """
    if not source_spans:
        fallback = +rect
        fallback &= page.mediabox
        return (fallback,) if not fallback.is_empty else ()
    ordered = sorted(source_spans, key=lambda span: (span.y0, span.x0))
    groups: list[object] = []
    for source in ordered:
        merged = False
        for group in reversed(groups):
            vertical = max(0.0, min(group.y1, source.y1) - max(group.y0, source.y0))
            required = min(group.height, source.height) * 0.5
            horizontal_gap = max(0.0, source.x0 - group.x1, group.x0 - source.x1)
            if vertical >= required and horizontal_gap <= 1.5:
                group.include_rect(source)
                merged = True
                break
            if source.y0 > group.y1 + 1.5:
                break
        if not merged:
            groups.append(+source)
    output: list[object] = []
    for group in groups:
        group += (-0.25, -0.25, 0.25, 0.25)
        group &= page.mediabox
        if not group.is_empty:
            output.append(group)
    return tuple(output)


def _source_text_rect(page, rect, source_spans: list[object]):
    """호환용 union. 실제 export redaction은 `_source_text_rects`를 사용한다."""
    rects = _source_text_rects(page, rect, source_spans)
    if not rects:
        return +rect
    actual = +rects[0]
    for source in rects[1:]:
        actual.include_rect(source)
    return actual


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

    # PyMuPDF는 fontname을 페이지 resource key로도 사용한다. 원본에 같은 key가
    # 있으면 새 fontfile 대신 기존 글꼴을 재사용할 수 있으므로 문서 전체에서
    # 충돌하지 않는 이름을 먼저 예약한다.
    used_font_names = _document_font_resource_names(doc)
    if serif_ff:
        serif_name = _unique_font_resource_name("uocr-serif", used_font_names)
    if sans_ff:
        sans_name = (
            serif_name
            if sans_ff == serif_ff
            else _unique_font_resource_name("uocr-sans", used_font_names)
        )
    if table_ff:
        table_name = (
            serif_name
            if table_ff == serif_ff
            else _unique_font_resource_name("uocr-table", used_font_names)
        )

    try:
        for tpage in trans_pages:
            pno = tpage.get("page")
            opage = orig_pages.get(pno)
            if not isinstance(pno, int) or not (1 <= pno <= doc.page_count) or opage is None:
                continue
            page = doc[pno - 1]
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
            targets: list[_Replacement] = []
            flow_candidates: list[_FlowCandidate] = []
            repeated_scheme_link_rects: list[object] = []
            for block_index, (ob, tb) in enumerate(zip(oblocks, tblocks)):
                if not isinstance(ob, dict) or not isinstance(tb, dict):
                    continue
                block_type = str(tb.get("type") or "")

                # 표는 HTML을 통째로 평문 삽입하지 않고 셀 구조가 원문과 정확히
                # 대응할 때만 셀별로 교체한다. 벡터 선은 redaction 옵션으로 보존된다.
                if block_type == "table":
                    old_parsed = _table_cells(str(ob.get("content") or ""))
                    new_parsed = _table_cells(str(tb.get("content") or ""))
                    table_rect = block_rects[block_index]
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
                                f"p{pno}: 표 셀 구조 불일치 — 원문 표 보존"
                            )
                        continue
                    old_cells, row_count, col_count = old_parsed
                    new_cells = new_parsed[0]
                    cell_rects, grid_trusted = _table_cell_rects(
                        page, table_rect, old_cells, row_count, col_count,
                    )
                    if not grid_trusted:
                        result.keep("table_grid_untrusted")
                        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
                        result.warnings.append(
                            f"p{pno}: 표 셀 격자 추정 실패(원문 검색 불일치) — 원문 표 보존"
                        )
                        continue
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
                            _plain_text(new_cell.text), table_ff,
                        )
                        base_pt, cell_align, cell_bold, source_redact = (
                            _table_cell_source_style(page, cell_rect, source_records)
                        )
                        plan = _plan_single_line(
                            page,
                            cell_rect,
                            new_text,
                            base_pt,
                            table_name,
                            table_ff,
                            max_rect=cell_rect,
                            align=cell_align,
                            bold=cell_bold,
                        )
                        if plan is None:
                            plan = _plan_shrink_to_fit(
                                page,
                                cell_rect,
                                new_text,
                                base_pt,
                                table_name,
                                table_ff,
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
                            table_name,
                            table_ff,
                            cell_rect,
                            block_index,
                        ))
                    if failed_cell is not None:
                        result.keep("table_cell_no_fit", changed_cells)
                        result.specialist_kept["table"] = result.specialist_kept.get("table", 0) + 1
                        result.warnings.append(
                            f"p{pno}: 표 {failed_cell.row + 1}행 {failed_cell.col + 1}열 "
                            "번역 생략(공간/가독성 부족) — 표 전체 원문 보존"
                        )
                    else:
                        targets.extend(table_targets)
                    continue

                if block_type in _PRESERVE_TYPES:
                    result.keep(f"preserve_type:{block_type}")
                    preserve_kind = "reference" if block_type == "ref_text" else "running_text"
                    result.specialist_kept[preserve_kind] = (
                        result.specialist_kept.get(preserve_kind, 0) + 1
                    )
                    if block_type == "ref_text":
                        microfixes = _preserved_reference_microfixes(
                            fitz,
                            source_ownership.get(block_index, []),
                            serif_name,
                            serif_ff,
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
                old = _plain_text(str(ob.get("content") or ""))
                new = _plain_text(str(tb.get("content") or ""))
                if block_type == "title":
                    new = _restore_title_prefix(old, new)
                if not new or new == old:
                    result.keep("unchanged")
                    continue
                rect = block_rects[block_index]
                if rect is None:
                    result.keep("no_rect")
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
                    result.keep("figure_text")
                    result.specialist_kept["figure_text"] = (
                        result.specialist_kept.get("figure_text", 0) + 1
                    )
                    result.warnings.append(f"p{pno}: 그림 위 텍스트 — 원문 보존")
                    continue
                fs_cqw = ob.get("fs") or tb.get("fs") or estimate_font_size_cqw(
                    tb.get("bbox"), str(tb.get("content") or ""), aspect,
                ) or 1.8
                base_pt = min(_MAX_FONT_PT, max(
                    _MIN_FONT_PT, fs_cqw / 100 * page.rect.width))
                # 같은 pt에서 AppleMyungjo/Noto Serif CJK는 Times 계열 영문보다
                # 시각적 몸통이 조금 작다. 제목은 계층을 잃지 않도록 더 보정하고,
                # 본문은 3%만 보정해 원문과 비슷한 잉크 밀도를 유지한다.
                base_pt *= 1.06 if block_type == "title" else 1.03
                if ob.get("font_style") == "sans":
                    block_fontfile, block_fontname = sans_ff, sans_name
                else:
                    block_fontfile, block_fontname = serif_ff, serif_name
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
                owned_records = source_ownership.get(block_index, [])
                local_source_records = [
                    span for span in source_records
                    if _source_span_matches_rect(span, rect)
                ]
                if block_index in ambiguous_blocks or (
                    not owned_records and local_source_records
                ):
                    result.keep("ambiguous_source")
                    result.warnings.append(
                        f"p{pno}: 블록 {block_index + 1} 교체 생략"
                        "(안전한 원문 span 없음) — 원문 보존"
                    )
                    continue
                owned_rects = [span.rect for span in owned_records]
                # 원문 PDF의 실제 baseline 수보다 OCR 줄 수가 많으면 그 줄바꿈은
                # 문단의 줄바꿈이 아니라 가로 배치(표 헤더·행)의 평탄화다.
                # bbox 높이는 원문 줄 수만큼뿐이라 축소로는 절대 들어가지 않는다.
                reflow_text = _reflow_flattened_text(
                    old, new, owned_records, base_pt,
                )
                flow_candidates.append(_FlowCandidate(
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
                    _source_text_rects(page, rect, owned_rects),
                    rect,
                    None if bold else _leading_bold_prefix(owned_records, new),
                    reflow_text,
                ))

            # 일반 텍스트는 페이지에서 모두 수집한 뒤 같은 단의 인접 블록을
            # 원자적으로 reflow한다. 이 단계 전에는 어떤 원문도 redaction하지 않는다.
            for component in _flow_components(flow_candidates):
                component_indices = {candidate.block_index for candidate in component}
                fixed_rects = [span.rect for span in unowned_source]
                fixed_rects.extend(
                    span.rect
                    for owner, spans in source_ownership.items()
                    if owner not in component_indices
                    for span in spans
                )
                fixed_rects.extend(fixed_visuals)
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
                    planned = _plan_flow_group(page, variant, fixed_rects)
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
                            for span in source_ownership.get(other.block_index, [])
                        )
                        single = None
                        for text in (candidate.text, candidate.reflow_text):
                            if text is None:
                                continue
                            single = _plan_flow_group(
                                page, [replace(candidate, text=text)], obstacles,
                            )
                            if single:
                                break
                        if single:
                            planned.extend(single)
                            continue
                        flattened = candidate.reflow_text is not None
                        result.keep("flattened_no_fit" if flattened else "no_fit")
                        reason = (
                            "가로 평탄화 블록 — 리플로우 실패"
                            if flattened
                            else "공간 부족"
                        )
                        result.warnings.append(
                            f"p{pno}: 블록 {candidate.block_index + 1} 교체 생략"
                            f"({reason}) — 원문 보존"
                        )
                targets.extend(planned)

            if not targets:
                continue

            # 반복 scheme의 첫 링크 annotation은 아래 redaction에서 사라질 수
            # 있다. 겹친 annotation으로 bad URI 집합을 식별할 수 있을 때 같은
            # URI를 가진 wrapped 링크까지 먼저 정상화한다.
            if repeated_scheme_link_rects:
                _normalize_repeated_scheme_links(page, repeated_scheme_link_rects)

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
