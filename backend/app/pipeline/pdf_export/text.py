"""번역 텍스트 정규화 — 마크업·LaTeX 제거와 조판용 공백 보정."""
from __future__ import annotations

import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|mathrm|operatorname|mathbf|mathit|mathsf|mathtt|mathcal)\s*\{([^{}]*)\}"
)
_LATEX_SUP_RE = re.compile(r"\^(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_SUB_RE = re.compile(r"_(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
# LaTeX의 비알파벳 이스케이프. `_LATEX_COMMAND_RE`는 알파벳 명령만 잡아 `\\%`가
# 번역 면에 그대로 찍혔다(실측 8건: "96\\% 정밀도"). 평문에서는 기호 자체가 답이다.
_LATEX_ESCAPE_RE = re.compile(r"\\([%&#$])")
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
# NFKD \ud638\ud658 \ubd84\ud574\uac00 \uc5c6\uc5b4 `_portable_text_for_font`\uc758 \uc77c\ubc18 \uacbd\ub85c\ub85c\ub294 ASCII\uae4c\uc9c0
# \ub0b4\ub824\uac00\uc9c0 \uc54a\ub294 \uae30\ud638\ub4e4. \ubaa8\uc591\uc774 \uc0ac\uc2e4\uc0c1 \uac19\uc740 ASCII \ub300\uccb4\ub9cc \ub123\ub294\ub2e4.
_PORTABLE_SYMBOL_FALLBACKS = {
    "\u2212": "-",  # MINUS SIGN
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u00d7": "x",  # MULTIPLICATION SIGN
    "\u2044": "/",  # FRACTION SLASH
    # `\sim`\uc774 \ub9cc\ub4e4\uc5b4\ub0b4\ub294 \uae00\uc790. \ucee8\ud14c\uc774\ub108 \uae30\ubcf8 \ud3f0\ud2b8(Noto Serif/Sans CJK) \ub458 \ub2e4
    # \uc774 \uae00\ub9ac\ud504\uac00 \uc5c6\uc5b4 `\( \sim \) 8M\uac1c\uc758 \ud1a0\ud070`\uc774 \uc2e4\uc81c \uc0b0\ucd9c\ubb3c\uc5d0\uc11c tofu\ub85c \ub098\uc654\ub2e4
    # (\uc2e4\uce21: j_afea33c8b77a p4). ASCII \ubb3c\uacb0\ud45c\uac00 \ub73b\ub3c4 \ud1b5\ud558\uace0 \uac80\uc0c9\ub3c4 \ub41c\ub2e4.
    "\u223c": "~",  # TILDE OPERATOR
}
_LITERAL_LBRACE = "\uf000"
_LITERAL_RBRACE = "\uf001"
# `\_`·`\^`는 위/아래 첨자 표기가 아니라 **글자 그대로의 밑줄·캐럿**이다.
# 첨자 정규식보다 먼저 봉인하지 않으면 `snake\_case`가 첨자로 해석된다.
_LITERAL_UNDERSCORE = "\uf002"
_LITERAL_CARET = "\uf003"
_TITLE_PREFIX_RE = re.compile(r"^([A-Z]|\d+(?:\.\d+)*)(?=\s)")


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


# 번역 단계가 흘리는 마크다운 표기. 레이아웃 경로에는 마크다운 렌더러가 없어
# 그대로 조판되면 "### 테스크 입력 및 출력"처럼 마커가 지면에 찍힌다(실측 p4).
# 마커는 폭·높이도 잡아먹어 "공간 부족" 오판을 늘린다.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_MD_CODESPAN_RE = re.compile(r"`+([^`]+)`+")


def strip_markdown(content: str) -> str:
    """번역문에서 마크다운 구조 표기를 걷어낸다 (내용은 그대로 둔다).

    원문 OCR 블록에는 적용하지 않는다 — 원문의 `-`는 실제 글머리표 글리프이고,
    양쪽을 다르게 정규화해도 비교는 `_ownership_text`가 따로 담당한다.
    """
    text = _MD_HEADING_RE.sub("", content)
    text = _MD_QUOTE_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub(lambda m: m.group(2), text)
    text = _MD_CODESPAN_RE.sub(lambda m: m.group(1), text)
    return text


_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def match_paragraph_shape(source_raw: str, translated_raw: str) -> str:
    """원문이 한 문단인데 번역이 여러 **문단**으로 쪼개졌으면 다시 잇는다.

    번역 유닛은 기하 정보를 갖지 않아 LLM이 run-in 제목을 "제목 / 빈 줄 / 본문"
    두 문단으로 재구성한다. 원문 bbox는 한 문단 분량이라 그대로 조판하면 높이가
    모자라 블록이 통째로 '공간 부족'으로 버려지고 영문 원문이 남는다
    (실측: "OSS-Fuzz에서의 수집\n\nOSS-Fuzz가 탐지한 …" 형태).

    빈 줄(문단 경계)만 대상이다 — 단순 줄바꿈은 원문의 시각적 줄 구조일 수
    있으므로 건드리지 않는다. 원문에 이미 문단 경계가 있으면 손대지 않는다.
    """
    if _BLANK_LINE_RE.search(source_raw or ""):
        return translated_raw
    return _BLANK_LINE_RE.sub(" ", translated_raw or "")


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
    text = text.replace("\\_", _LITERAL_UNDERSCORE).replace("\\^", _LITERAL_CARET)
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
    text = _LATEX_ESCAPE_RE.sub(r"\1", text)
    # 남은 grouping braces는 평문에서 의미가 없고 줄 폭만 늘린다. literal set은 복원.
    text = text.replace("{", "").replace("}", "")
    text = text.replace(_LITERAL_LBRACE, "{").replace(_LITERAL_RBRACE, "}")
    text = text.replace(_LITERAL_UNDERSCORE, "_").replace(_LITERAL_CARET, "^")
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


def _restore_title_prefix(original: str, translated: str) -> str:
    """번역 모델이 떨군 절/부록 식별자(A, 2.1 등)를 제목 앞에 복구한다."""
    source = _TITLE_PREFIX_RE.match(original)
    if source is None or _TITLE_PREFIX_RE.match(translated):
        return translated
    return f"{source.group(1)} {translated}"
