"""임베드 폰트 서브셋 — 실제로 조판에 쓰이는 글리프만 남긴다.

내보내기 PDF의 부피는 거의 전부가 임베드된 CJK 폰트 한 벌이다. 실측(17페이지
논문): source.pdf 693KB → export.ko.pdf 20.77MB이고, 그 중 20.00MB(96.3%)가
서브셋되지 않은 Noto Serif CJK 한 벌(65,535 글리프)이었다. 정작 그 문서가 그리는
한글·기호는 500자 남짓이라, 쓰는 글리프만 남기면 같은 폰트가 ~180KB로 줄어
다운로드가 20배 이상 작아진다(20.8MB → ~0.95MB).

서브셋은 **조판 결과를 바꾸면 안 된다**. 조판기는 `has_glyph`로 글리프 유무를
물어 없으면 ASCII로 낮추므로(`_portable_text_for_font`), 서브셋에서 빠진 글자가
있으면 조용히 다른 문자가 찍힌다. 그래서 두 겹으로 막는다.

1. 문자 집합을 레이아웃 텍스트에서만 뽑지 않는다. 조판 파이프라인이 **만들어낼
   수 있는** 문자 — 위/아래첨자 맵, LaTeX 명령 치환, 기호 폴백, NFKD ASCII
   분해 — 의 치역을 전부 더해 닫힌 집합으로 만든다. 치환표가 유한하므로 이
   닫힘은 완전하다.
2. 서브셋을 만든 뒤 그 집합 전체에 대해 원본 폰트와 `has_glyph` 답이 같은지
   확인한다. 하나라도 어긋나면 서브셋을 버리고 원본을 그대로 임베드한다 —
   느리고 큰 것이 조용히 틀린 것보다 낫다.

동치의 기준은 "블록별 조판 결정과 그려지는 글자가 같다"이다(리포트의
replaced/kept/relocated·경고가 같고, 공백을 무시한 추출 텍스트가 같고, 추출
텍스트에 U+0000이 없다). 재조판된 블록 **안에서** 줄바꿈 위치가 한 글자
움직이는 것은 동치로 본다 — 어차피 우리가 다시 흘려 넣는 줄이고, 실측상
서브셋 폰트로 셰이핑하면 몇 줄에서 그만큼 달라진다.

fontTools가 없거나 서브셋이 실패하면 조용히 원본 경로를 돌려준다. 내보내기가
폰트 최적화 **때문에** 실패하지는 않는다는 것이 이 모듈의 계약이다.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from ..pdf import quiet_fitz
from .text import (
    _LATEX_COMMANDS,
    _PORTABLE_SYMBOL_FALLBACKS,
    _SUBSCRIPT_MAP,
    _SUPERSCRIPT_MAP,
    _UNICODE_SUPERSCRIPT_ASCII,
)

logger = logging.getLogger(__package__)

# 서브셋을 시도할 최소 크기. 이미 작은 폰트(서브셋된 본문 폰트, 나눔고딕 등)는
# 파싱·직렬화 비용만 늘고 얻는 게 없다.
_SUBSET_MIN_BYTES = 2 * 1024 * 1024
# 치환표와 무관하게 항상 넣는 바닥 집합 — ASCII 인쇄 가능 문자.
_ASCII_PRINTABLE = "".join(chr(code) for code in range(0x20, 0x7F))
# 조판기가 **원문에 없던 공백류를 스스로 만들어 넣는다**. `_protect_trailing_words`
# (text.py)는 마지막 낱말이 홀로 떨어지지 않게 일반 공백을 U+00A0으로 바꾼다.
# 이 글자가 서브셋에서 빠지면 화면은 같아 보이는데 추출 텍스트에 U+0000이
# 박혀 복사·검색이 깨진다(실측: 17페이지에서 7곳). 개별 글자를 쫓지 않고
# 공백·보이지 않는 서식 문자 부류를 통째로 넣는다 — 글리프 몇 개 값이면
# 같은 종류의 회귀를 앞으로도 막는다.
_INVISIBLE_FORMATTING = (
    "\u00a0"                                          # NO-BREAK SPACE
    + "".join(chr(c) for c in range(0x2000, 0x200C))  # EN/EM/THIN SPACE … ZWJ
    + "\u202f\u205f\u2060\u3000\ufeff"                    # NARROW NBSP, MMSP, WJ, IDEOGRAPHIC SP, BOM
)


def _translation_targets(table: dict) -> str:
    """str.maketrans 결과의 치역(만들어질 수 있는 문자)을 모은다."""
    out: list[str] = []
    for value in table.values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, int):
            out.append(chr(value))
    return "".join(out)


def _substitution_range() -> str:
    """조판 파이프라인이 원문에 없던 문자를 만들어내는 모든 경로의 치역.

    `text._normalize_text`는 LaTeX 명령을 유니코드 기호로, 위/아래첨자를 유니코드
    첨자로 바꾼다. `fonts._portable_text_for_font`는 반대로 유니코드 첨자·기호를
    ASCII로 낮춘다. 두 방향의 결과가 모두 실제로 그려질 수 있으므로 전부 넣는다.
    """
    return "".join((
        _ASCII_PRINTABLE,
        _INVISIBLE_FORMATTING,
        _translation_targets(_SUPERSCRIPT_MAP),
        _translation_targets(_SUBSCRIPT_MAP),
        _translation_targets(_UNICODE_SUPERSCRIPT_ASCII),
        "".join(_LATEX_COMMANDS.values()),
        "".join(_PORTABLE_SYMBOL_FALLBACKS),
        "".join(_PORTABLE_SYMBOL_FALLBACKS.values()),
    ))


def drawable_charset(*payloads: object) -> str:
    """조판이 그릴 수 있는 문자의 닫힌 집합.

    payload는 레이아웃 페이지 리스트처럼 텍스트를 품은 임의의 JSON 직렬화 가능
    객체다. 블록 구조를 따라다니며 텍스트 필드만 골라내는 대신 통째로 직렬화해
    문자를 훑는다 — 키 이름 같은 ASCII가 몇 개 더 들어올 뿐이고, 새로운 텍스트
    필드가 생겨도 여기를 고칠 필요가 없다(빠뜨리면 조용히 틀리는 쪽이 위험하다).
    """
    chars: set[str] = set(_substitution_range())
    for payload in payloads:
        if payload is None:
            continue
        if isinstance(payload, str):
            blob = payload
        else:
            blob = json.dumps(payload, ensure_ascii=False, default=str)
        chars.update(blob)
    # NFKD ASCII 분해는 `_portable_text_for_font`의 마지막 폴백 경로다.
    for char in list(chars):
        for part in unicodedata.normalize("NFKD", char):
            if part.isascii():
                chars.add(part)
    chars.discard("\n")
    chars.discard("\r")
    chars.discard("\t")
    return "".join(sorted(chars))


def _has_glyph_map(fontfile: str, chars: str) -> dict[str, bool] | None:
    """폰트가 각 문자의 글리프를 가지는지 — 서브셋 전후 동치 확인용."""
    fitz = quiet_fitz()
    try:
        font = fitz.Font(fontfile=fontfile)
    except Exception:  # noqa: BLE001 — 확인 불가는 '서브셋 포기'로 처리한다
        return None
    try:
        return {char: bool(font.has_glyph(ord(char))) for char in chars}
    except Exception:  # noqa: BLE001
        return None


def subset_font(src: str, chars: str, out_dir: Path) -> str | None:
    """`src` 폰트에서 `chars`에 필요한 글리프만 남긴 파일 경로. 실패하면 None.

    .ttc(폰트 컬렉션)는 PyMuPDF가 face 0을 쓰므로 같은 face를 서브셋한다.
    """
    try:
        from fontTools import subset as ft_subset
        from fontTools.ttLib import TTFont
    except ImportError:
        # fontTools 미설치 — 서브셋 없이 예전처럼 전체 폰트를 임베드한다.
        return None

    # fontTools는 서브셋 진행 상황을 자기 로거로 쏟아낸다("... NOT subset;
    # don't know how to subset; dropped"). 우리 로그에는 결과 한 줄만 남긴다.
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    source = Path(src)
    try:
        if source.stat().st_size < _SUBSET_MIN_BYTES:
            return None
    except OSError:
        return None

    out = out_dir / f"{source.stem}.subset.otf"
    try:
        font = TTFont(str(source), fontNumber=0, lazy=True)
        options = ft_subset.Options()
        # 글리프 번호를 원본 그대로 둔다. cmap에 없고 GSUB로만 닿는 글리프
        # (합자·문맥 대체)는 PyMuPDF가 ToUnicode를 역매핑할 때 기댈 곳이
        # 원본과 같은 번호뿐이다. 실측상 최종 PDF 크기는 이 옵션으로 달라지지
        # 않았다(둘 다 0.79MB — 남는 것은 빈 글리프라 압축에서 사라진다).
        options.retain_gids = True
        # CFF 서브셋에서 로컬 subr 참조가 깨지는 폰트가 있어 인라인화한다.
        options.desubroutinize = True
        # 폰트에 없는 문자를 넣어도 실패하지 않게 한다 — 닫힌 집합에는 이
        # 폰트가 모르는 기호(다른 폰트가 담당)도 섞여 들어온다.
        options.ignore_missing_glyphs = True
        options.ignore_missing_unicodes = True
        # DSIG는 서브셋 후 무효가 된다.
        options.drop_tables += ["DSIG"]
        subsetter = ft_subset.Subsetter(options=options)
        subsetter.populate(text=chars)
        subsetter.subset(font)
        font.save(str(out))
        font.close()
    except Exception:  # noqa: BLE001 — 어떤 폰트 결함도 원본 폴백으로 흡수한다
        logger.warning("폰트 서브셋 실패 — 전체 폰트를 임베드합니다: %s", source.name)
        out.unlink(missing_ok=True)
        return None

    # 동치 확인: 서브셋이 조판 결정을 바꾸지 않는지 문자 단위로 검증한다.
    original_map = _has_glyph_map(str(source), chars)
    subset_map = _has_glyph_map(str(out), chars)
    if original_map is None or subset_map is None or original_map != subset_map:
        missing = (
            [c for c in chars if original_map.get(c) and not subset_map.get(c)][:8]
            if original_map and subset_map else []
        )
        logger.warning(
            "폰트 서브셋 글리프 검증 실패 — 전체 폰트를 임베드합니다: %s%s",
            source.name,
            f" (누락 예: {''.join(missing)})" if missing else "",
        )
        out.unlink(missing_ok=True)
        return None

    try:
        before, after = source.stat().st_size, out.stat().st_size
    except OSError:
        return str(out)
    logger.info(
        "폰트 서브셋: %s %.1fMB → %.2fMB (%d자)",
        source.name, before / 1e6, after / 1e6, len(chars),
    )
    return str(out)


def subset_font_files(
    fontfiles: list[str | None], chars: str, out_dir: Path,
) -> dict[str, str]:
    """서로 다른 폰트 파일마다 한 번씩만 서브셋해 {원본경로: 서브셋경로}를 만든다.

    serif·sans·표 폰트가 같은 파일로 해석되는 배포가 흔하다(컨테이너 기본값이
    그렇다) — 같은 26MB 파일을 세 번 파싱하지 않는다. 서브셋에 실패한 폰트는
    맵에 넣지 않으므로 호출자는 원본 경로를 그대로 쓰게 된다.
    """
    mapping: dict[str, str] = {}
    for fontfile in fontfiles:
        if not fontfile or fontfile in mapping:
            continue
        subset = subset_font(fontfile, chars, out_dir)
        if subset:
            mapping[fontfile] = subset
    return mapping
