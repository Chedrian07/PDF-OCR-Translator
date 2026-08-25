"""폰트 해석·메트릭 캐시와 폰트에 의존하는 텍스트 보정.

폴백 체인: PDF_EXPORT_FONT(파일 경로) → 시스템 한글 폰트 후보 → fc-list 런타임
탐색 → PyMuPDF 내장 CJK. 자세한 배경은 패키지 `__init__`의 모듈 문서를 보라.
"""
from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import unicodedata
import re
from pathlib import Path

from ..pdf import quiet_fitz
from .constants import _MIN_FONT_PT, _SHRINK_STEPS, _SINGLE_LINE_SCALES
from .text import _PORTABLE_SYMBOL_FALLBACKS, _UNICODE_SUPERSCRIPT_ASCII

# 패키지로 쪼개기 전과 같은 로거 이름을 유지한다(핸들러·필터 설정 호환).
logger = logging.getLogger(__package__)

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
def _metrics_font_cached(_key: tuple, fontfile: str | None, fontname: str):
    """`_metrics_font`의 캐시 본체. 키 정규화는 호출자가 끝내고 들어온다."""
    fitz = quiet_fitz()
    return fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(fontname=fontname)


def _metrics_font(fontfile: str | None, fontname: str):
    """측정 전용 fitz.Font 재사용 캐시.

    Font 생성은 폰트 파일 전체를 파싱한다(실측: AppleMyungjo 18MB 3.1ms,
    AppleSDGothicNeo 55MB 8.6ms). 계획 함수들은 text_length/has_glyph/ascender
    같은 읽기 전용 측정에만 쓰므로 내보내기 한 번에 serif/sans/table 폰트를
    공유해도 조판 결과가 같다. `_resolve_font`는 손상 폰트 검증이 목적이고
    빌드당 몇 회뿐이라 캐시하지 않는다.

    캐시 키는 *실제로 Font를 결정하는 값*만 쓴다. fontfile이 있으면 fontname은
    PDF resource 이름일 뿐 무시되므로, 키에 넣으면 같은 18MB 파일이
    uocr-serif/uocr-sans/uocr-table/uocr-serif-2로 최대 네 번 파싱·상주한다.
    파일 경로만으로는 `PDF_EXPORT_FONT`를 같은 경로에 덮어써 교체하는 배포에서
    프로세스가 살아 있는 한 옛 글리프 메트릭을 계속 쓰므로 mtime·크기까지 키에
    넣어 인플레이스 교체를 자동으로 무효화한다.
    """
    if not fontfile:
        return _metrics_font_cached((None, fontname), None, fontname)
    try:
        stat = Path(fontfile).stat()
        signature = (fontfile, stat.st_mtime_ns, stat.st_size)
    except OSError:  # 경로가 사라졌으면 로드 시점에서 실패하도록 그대로 넘긴다
        signature = (fontfile, None, None)
    return _metrics_font_cached(signature, fontfile, "")


# 테스트·운영에서 캐시를 다루는 인터페이스는 lru_cache와 동일하게 유지한다.
_metrics_font.cache_clear = _metrics_font_cached.cache_clear
_metrics_font.cache_info = _metrics_font_cached.cache_info


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
        # NFKD 분해가 없는 수학 기호는 아래 경로로도 구제되지 않는다.
        # (실측: AppleMyungjo에 U+2212 MINUS SIGN이 없어 `1 −PL`이 tofu가 됐다)
        fallback = _PORTABLE_SYMBOL_FALLBACKS.get(char)
        if fallback and all(font.has_glyph(ord(part)) for part in fallback):
            output.append(fallback)
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
