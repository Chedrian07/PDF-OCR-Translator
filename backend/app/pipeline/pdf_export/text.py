"""번역 텍스트 정규화 — 마크업·LaTeX 제거와 조판용 공백 보정."""
from __future__ import annotations

import re

# 실제 HTML 태그만 지운다 — 부등호 사이의 산문·특수 토큰은 보존(layout과 같은 정의).
from ..layout import HTML_TAG_RE as _TAG_RE

_WS_RE = re.compile(r"[ \t ]+")
# 인자만 남기고 벗겨 낼 서식 명령. 아래 구조 변환(`_latex_structures`)이 중첩 인자까지
# 처리하고, 이 정규식은 남은 단순 형태를 한 번 더 벗기는 안전망이다.
_LATEX_WRAPPER_NAMES = (
    "text", "textrm", "textbf", "textit", "textsf", "texttt", "textnormal", "textup",
    "emph", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathscr",
    "mathfrak", "mathnormal", "operatorname", "boldsymbol", "bm", "pmb", "underline",
    "mbox", "hbox",
)
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:" + "|".join(_LATEX_WRAPPER_NAMES) + r")\*?\s*\{([^{}]*)\}"
)
_LATEX_SUP_RE = re.compile(r"\^(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_SUB_RE = re.compile(r"_(?:\{([^{}]+)\}|([A-Za-z0-9+\-=()]+))")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
# LaTeX의 비알파벳 이스케이프. `_LATEX_COMMAND_RE`는 알파벳 명령만 잡아 `\\%`가
# 번역 면에 그대로 찍혔다(실측 8건: "96\\% 정밀도"). 평문에서는 기호 자체가 답이다.
_LATEX_ESCAPE_RE = re.compile(r"\\([%&#$])")
# `\,` `\;` `\:` `\>` `\ `(역슬래시+공백)은 TeX 간격 — 평문에서는 공백 하나다.
_LATEX_SPACING_RE = re.compile(r"\\[,;:> ]")
_HTML_SCRIPT_RE = re.compile(r"<(sup|sub)>([^<>{}\n]{1,24})</\1>", re.IGNORECASE)
# `<|im_start|>`·`<|endoftext|>` 같은 모델 특수 토큰은 논문 본문의 **내용**이다. 첨자
# 변환이 `_start`를 '(start)'로 바꾸지 않게 통째로 보관했다가 되돌린다.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\n]{1,64}\|>")
_TOKEN_OPEN = "\ue002"
_TOKEN_CLOSE = "\ue003"
_TOKEN_SLOT_RE = re.compile(_TOKEN_OPEN + r"(\d+)" + _TOKEN_CLOSE)
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
    # 아래는 흔한 인라인 수식에서 이름 그대로 새던 명령들('a cdot b', 'ell(2)', 'eta').
    # 대상 글자는 KS X 1001에 들어 있어 macOS 명조/고딕과 Noto CJK KR가 모두 그린다.
    "zeta": "ζ", "eta": "η", "iota": "ι", "kappa": "κ", "nu": "ν", "xi": "ξ",
    "upsilon": "υ", "chi": "χ", "psi": "ψ", "Xi": "Ξ", "Upsilon": "Υ", "Psi": "Ψ",
    "varepsilon": "ε", "vartheta": "θ", "varphi": "φ", "varrho": "ρ",
    "varsigma": "σ", "varpi": "π",
    "cdot": "·", "ell": "ℓ", "mid": "|", "vert": "|", "lvert": "|", "rvert": "|",
    "Vert": "‖", "lVert": "‖", "rVert": "‖", "sum": "∑", "prod": "∏", "int": "∫",
    "propto": "∝", "prime": "′", "dagger": "†", "neg": "¬", "lnot": "¬",
    "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "implies": "⇒",
    "iff": "⇔", "leftrightarrow": "↔", "uparrow": "↑", "downarrow": "↓",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇", "ni": "∋",
    "equiv": "≡", "ll": "≪", "gg": "≫", "cong": "≅", "simeq": "≃",
    "angle": "∠", "perp": "⊥", "bot": "⊥", "therefore": "∴", "because": "∵",
    "triangle": "△", "colon": ":", "ast": "*",
    # 내적·쌍 괄호(`\langle y, x \rangle`). 예전에는 'langle y, xrangle'처럼 이름이 그대로
    # 샜다(실서버 25쪽 논문: 번역 블록 44개). ⟨⟩가 없는 macOS 명조·고딕은 조판 직전에
    # KS X 1001의 〈〉로 낮춘다(_PORTABLE_SYMBOL_FALLBACKS).
    "langle": "⟨", "rangle": "⟩",
    # 전치 기호 `W^\top`은 'W^T'가 논문 독자에게 가장 익숙한 평문이다.
    "top": "T", "intercal": "T",
    # 크기·스타일 지정은 평문에서 의미가 없다.
    "big": "", "Big": "", "bigg": "", "Bigg": "", "bigl": "", "bigr": "",
    "Bigl": "", "Bigr": "", "biggl": "", "biggr": "", "Biggl": "", "Biggr": "",
    "displaystyle": "", "textstyle": "", "scriptstyle": "", "limits": "",
    "nolimits": "",
}
# `\mathbb{R}` 등 이중선 대문자. 유니코드 BMP에 있는 것만 바꾸고(나머지는 원 글자),
# 폰트에 없으면 `_portable_text_for_font`의 NFKD 경로가 'R'로 낮춘다.
_LATEX_BLACKBOARD = {
    "C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ",
}
# `\hat{y}` 같은 한 글자 악센트 → 결합 문자. 폰트에 결합 글리프가 없으면 조판 직전에
# 빠지고 기본 글자만 남는다(`fonts._portable_text_for_font`). 'haty'보다는 'y'가 낫다.
_LATEX_ACCENTS = {
    "hat": "\u0302", "widehat": "\u0302", "tilde": "\u0303", "widetilde": "\u0303",
    "bar": "\u0304", "overline": "\u0304", "dot": "\u0307", "ddot": "\u0308",
    "vec": "\u20d7",
}
_LATEX_FRACTIONS = frozenset({"frac", "dfrac", "tfrac", "cfrac"})
_LATEX_BINOMIALS = frozenset({"binom", "dbinom", "tbinom"})
# 조판 파이프라인이 **원문에 없던** 문자를 만들어 내는 나머지 경로의 치역 — 서브셋
# 문자 집합(subset._substitution_range)이 이것을 포함해야 서브셋이 조판을 바꾸지 않는다.
# 참고문헌 미세 교정이 원문(`T\"ulu 3:`로 깨진 bib)을 바로잡아 그리는 문자열.
_TULU_MICROFIX = "Tülu 3:"
_GENERATED_SYMBOLS = "".join((
    "√•",
    "".join(_LATEX_BLACKBOARD.values()),
    "".join(_LATEX_ACCENTS.values()),
    _TULU_MICROFIX,
))
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
    # `\langle`\u00b7`\rangle`\uc758 \uce58\ud658 \uacb0\uacfc. AppleMyungjo\u00b7AppleSDGothicNeo\uc5d0 \uc5c6\ub2e4(\uc2e4\uce21).
    "\u27e8": "\u3008",  # MATHEMATICAL LEFT ANGLE BRACKET \u2192 \u3008
    "\u27e9": "\u3009",  # MATHEMATICAL RIGHT ANGLE BRACKET \u2192 \u3009
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


# 구조 명령(분수·근호·이중선·악센트·서식) 변환. 이름만 남기던 예전 처리는 인라인
# 수식을 'frac1N', 'sqrtd(k)', 'mathbbR^(d)', 'haty'처럼 읽을 수 없게 만들었다.
# 정규식의 `[^{}]`로는 `\frac{1}{\sqrt{d_k}}` 같은 중첩 인자를 못 잡으므로 괄호 짝을
# 세는 작은 파서로 인자를 읽고 안쪽부터 재귀 변환한다.
_LATEX_NAME_RE = re.compile(r"\\([A-Za-z]+)")
_LATEX_TOKEN_RE = re.compile(r"\\(?:[A-Za-z]+|.)")
# 분수의 분자·분모를 괄호 없이 써도 되는 '원자' — 숫자·변수·명령 하나에 첨자가
# 붙은 꼴(`1`, `N`, `d_k`, `x^{2}`, `\alpha_i`, `√d`). 그 밖(`a+b`, `n-1`)은 괄호로 감싼다.
_LATEX_ATOM_RE = re.compile(
    r"^√?(?:\\[A-Za-z]+|[^\W_]|[.′'\u0300-\u036f\u20d0-\u20ff])+"
    r"(?:[_^](?:\{[^{}]*\}|\\[A-Za-z]+|[^\W_]))*$"
)
_LATEX_MAX_DEPTH = 8


def _latex_argument(text: str, start: int) -> tuple[str, int] | None:
    """`start`부터 공백을 건너뛴 TeX 인자 하나 — `{그룹}`·`\\명령`·글자 한 개."""
    index = start
    while index < len(text) and text[index] in " \t":
        index += 1
    if index >= len(text):
        return None
    char = text[index]
    if char == "{":
        depth = 0
        for end in range(index, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    return text[index + 1:end], end + 1
        return None  # 짝이 없는 그룹 — 변환하지 않고 원문을 남긴다
    if char == "\\":
        token = _LATEX_TOKEN_RE.match(text, index)
        if token is not None:
            return token.group(0), token.end()
        return None
    if char in "{}^_":
        return None
    return char, index + 1


def _latex_atom(value: str) -> str:
    value = value.strip()
    return value if _LATEX_ATOM_RE.match(value) else f"({value})"


def _latex_structures(text: str, depth: int = 0) -> str:
    """분수·근호·이중선·악센트·서식 명령을 읽을 수 있는 평문으로 바꾼다.

    처리하지 못한 명령은 그대로 남겨 뒤의 일반 명령 치환(`_latex_command`)에 맡긴다.
    """
    if depth > _LATEX_MAX_DEPTH or "\\" not in text:
        return text
    out: list[str] = []
    index = 0
    while index < len(text):
        match = _LATEX_NAME_RE.match(text, index) if text[index] == "\\" else None
        if match is None:
            out.append(text[index])
            index += 1
            continue
        name = match.group(1)
        cursor = match.end()
        if name == "operatorname" and text.startswith("*", cursor):
            cursor += 1
        converted: str | None = None
        if name in _LATEX_FRACTIONS or name in _LATEX_BINOMIALS:
            first = _latex_argument(text, cursor)
            second = _latex_argument(text, first[1]) if first else None
            if first and second:
                top = _latex_structures(first[0], depth + 1)
                bottom = _latex_structures(second[0], depth + 1)
                converted = (
                    f"{_latex_atom(top)}/{_latex_atom(bottom)}"
                    if name in _LATEX_FRACTIONS
                    else f"C({top.strip()}, {bottom.strip()})"
                )
                cursor = second[1]
        elif name == "sqrt":
            index_text = ""
            probe = cursor
            while probe < len(text) and text[probe] in " \t":
                probe += 1
            if probe < len(text) and text[probe] == "[":
                close = text.find("]", probe)
                if close != -1:
                    index_text = text[probe + 1:close].strip()
                    cursor = close + 1
            argument = _latex_argument(text, cursor)
            if argument:
                radicand = _latex_structures(argument[0], depth + 1)
                prefix = ""
                if index_text:
                    prefix = (
                        index_text.translate(_SUPERSCRIPT_MAP)
                        if index_text.isdigit()
                        else f"{index_text}"
                    )
                converted = f"{prefix}√{_latex_atom(radicand)}"
                cursor = argument[1]
        elif name == "mathbb":
            argument = _latex_argument(text, cursor)
            if argument:
                inner = _latex_structures(argument[0], depth + 1)
                converted = "".join(_LATEX_BLACKBOARD.get(char, char) for char in inner)
                cursor = argument[1]
        elif name in _LATEX_ACCENTS:
            argument = _latex_argument(text, cursor)
            if argument:
                inner = _latex_structures(argument[0], depth + 1).strip()
                single = len(inner) == 1 or bool(re.fullmatch(r"\\[A-Za-z]+", inner))
                converted = inner + _LATEX_ACCENTS[name] if single else inner
                cursor = argument[1]
        elif name in _LATEX_WRAPPER_NAMES:
            argument = _latex_argument(text, cursor)
            if argument:
                converted = _latex_structures(argument[0], depth + 1)
                cursor = argument[1]
        if converted is None:
            out.append(match.group(0))
            index = match.end()
            continue
        out.append(converted)
        index = cursor
    return "".join(out)


# 번역 단계가 흘리는 마크다운 표기. 레이아웃 경로에는 마크다운 렌더러가 없어
# 그대로 조판되면 "### 테스크 입력 및 출력"처럼 마커가 지면에 찍힌다(실측 p4).
# 마커는 폭·높이도 잡아먹어 "공간 부족" 오판을 늘린다.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^(\s{0,3})[-*+]\s+", re.MULTILINE)
# 원문 블록이 목록인지 — OCR content는 목록 항목을 '- …'로, 텍스트 레이어는 실제
# 글머리표 글리프('•' 등)로 담는다.
_SOURCE_BULLET_RE = re.compile(r"^\s{0,3}[-*+•·▪◦‣●○■□]\s+", re.MULTILINE)
_SOURCE_QUOTE_RE = re.compile(r"^\s{0,3}>", re.MULTILINE)
# 강조 표기는 **단어 경계**에서만 벗긴다. 여는 표기 앞에 ASCII 영숫자가 붙어 있으면
# 곱셈·식별자다(`4*8*16`, `p*q`, `*args, **kwargs`) — 예전에는 '4816 = 512로'처럼
# 숫자가 조용히 바뀌었다. 닫는 표기 뒤의 한글 조사('**강조**된')는 흔한 LLM 출력이라
# 허용한다. 밑줄 강조는 거의 쓰이지 않고 `__init__` 같은 dunder와 겹치므로, 안쪽이
# ASCII 식별자 한 덩어리면 벗기지 않는다.
_MD_STAR_EMPHASIS_RE = re.compile(
    r"(?<![A-Za-z0-9*\\])(\*{1,3})(?=[^\s*])(.+?)(?<=[^\s*\\])\1(?![A-Za-z0-9*])",
    re.DOTALL,
)
_MD_UNDERSCORE_EMPHASIS_RE = re.compile(
    r"(?<![A-Za-z0-9_\\])(_{2,3})(?=[^\s_])(.+?)(?<=[^\s_])\1(?![A-Za-z0-9_])",
    re.DOTALL,
)
_ASCII_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MD_CODESPAN_RE = re.compile(r"`+([^`]+)`+")
# 코드 스팬 내용을 강조 제거에서 보호하는 자리표시자(사용자 영역 문자 — 본문에 없다).
_CODE_OPEN = "\ue000"
_CODE_CLOSE = "\ue001"
_CODE_SLOT_RE = re.compile(_CODE_OPEN + r"(\d+)" + _CODE_CLOSE)
# 번역문에 다시 넣는 글머리표. '•'는 KS X 1001 글자라 명조/고딕·Noto CJK KR가 그린다.
_BULLET = "• "


def strip_markdown(content: str, source: str | None = None) -> str:
    """번역문에서 마크다운 구조 표기를 걷어낸다 (내용은 그대로 둔다).

    원문 OCR 블록에는 적용하지 않는다 — 원문의 `-`는 실제 글머리표 글리프이고,
    양쪽을 다르게 정규화해도 비교는 `_ownership_text`가 따로 담당한다.

    `source`(원문 블록 content)를 주면 원문에 실제로 있던 구조는 지우지 않는다.
    원문 글머리표 글리프는 블록 소유로 리댁션되므로, 원문이 목록인데 번역문의
    '- '까지 지우면 목록이 글머리표 없는 평문단이 된다 — 그런 줄은 '• '로 바꾼다.
    원문 줄이 `>`로 시작하면(예: '> 0.5 임계값') 인용 표기가 아니라 내용이다.
    원문에 같은 길이의 강조 표기(`*`, `**`, `__`)가 있으면 그 표기는 글자 그대로라
    보존한다.
    """
    text = _MD_HEADING_RE.sub("", content)
    if source is None or not _SOURCE_QUOTE_RE.search(source):
        text = _MD_QUOTE_RE.sub("", text)
    keep_bullets = source is not None and bool(_SOURCE_BULLET_RE.search(source))
    text = _MD_BULLET_RE.sub(
        (lambda m: m.group(1) + _BULLET) if keep_bullets else "", text,
    )
    # 코드 스팬 안의 `*`·`__`는 강조가 아니다 — 자리표시자로 빼 두고 마지막에 되돌린다.
    spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return f"{_CODE_OPEN}{len(spans) - 1}{_CODE_CLOSE}"

    text = _MD_CODESPAN_RE.sub(_stash, text)
    # 원문에 같은 길이의 표기가 있으면 그 표기는 글자 그대로다(각주 '*', `__x__`).
    # 길이별로 따져 원문의 '*' 하나 때문에 LLM의 `**굵게**`까지 남기지는 않는다.
    source_stars = {len(run) for run in re.findall(r"\*+", source or "")}
    source_underscores = {len(run) for run in re.findall(r"_{2,}", source or "")}
    text = _MD_STAR_EMPHASIS_RE.sub(
        lambda m: m.group(0) if len(m.group(1)) in source_stars else m.group(2),
        text,
    )
    text = _MD_UNDERSCORE_EMPHASIS_RE.sub(
        lambda m: m.group(0)
        if len(m.group(1)) in source_underscores
        or _ASCII_IDENTIFIER_RE.fullmatch(m.group(2))
        else m.group(2),
        text,
    )
    return _CODE_SLOT_RE.sub(lambda m: spans[int(m.group(1))], text)


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
    tokens: list[str] = []

    def _stash_token(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"{_TOKEN_OPEN}{len(tokens) - 1}{_TOKEN_CLOSE}"

    text = _SPECIAL_TOKEN_RE.sub(_stash_token, content)
    # HTML 위/아래첨자는 TeX 첨자와 같은 경로로 낮춘다(`x<sup>2</sup>` → `x²`).
    text = _HTML_SCRIPT_RE.sub(
        lambda m: ("^{" if m.group(1).lower() == "sup" else "_{") + m.group(2) + "}",
        text,
    )
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "").replace("$$", "")
    # literal set braces는 TeX grouping brace 제거와 구분해 끝까지 보존한다.
    text = text.replace("\\{", _LITERAL_LBRACE).replace("\\}", _LITERAL_RBRACE)
    text = text.replace("\\_", _LITERAL_UNDERSCORE).replace("\\^", _LITERAL_CARET)
    # 비알파벳 간격·구분 명령: `3\,GB` → '3 GB', `\|x\|` → '‖x‖'. 역슬래시째 찍히던 것들.
    text = _LATEX_SPACING_RE.sub(" ", text).replace("\\!", "")
    text = text.replace("\\|", "‖")
    text = _latex_structures(text)
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
    if tokens:
        text = _TOKEN_SLOT_RE.sub(lambda m: tokens[int(m.group(1))], text)
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
