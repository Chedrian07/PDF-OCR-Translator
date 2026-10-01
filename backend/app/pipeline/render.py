"""마크다운 → HTML 프래그먼트 (서버사이드 렌더).

raw HTML은 비활성(html=False)이라 OCR 결과에 악성 태그가 섞여도 이스케이프된다.
예외적으로 **신뢰 경로에서 서버가 생성**하는 것만 복원/주입한다:
- 표: 모델이 HTML `<table>`로 출력 → 구조 태그만(숫자 colspan/rowspan 포함) 복원
- 수식: 모델의 `\\( … \\)` / `\\[ … \\]`(실측 형태, 아래 참조)를 $-델리미터로
  정규화한 뒤 dollarmath로 파싱하고, tex를 **이스케이프한** `.math-inline` /
  `.math-display` 요소로 출력 — 최종 타이포셋은 클라이언트 KaTeX가 수행한다.
- 그림: 잡이 만든 `images/…` 상대 참조와 인라인 `data:` 래스터만 <img>로 낸다.
  본문에 실린 외부·내부망 주소는 자동으로 불러오지 않는 링크로 낮춘다(`_render_image`).

정규화는 렌더 레이어에서만 일어난다. result.md(다운로드 소스)는 모델 원본
LaTeX 델리미터를 그대로 유지한다 (포터빌리티 계약, ARCHITECTURE.md 전역 제약).
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from mdit_py_plugins.dollarmath import dollarmath_plugin


def _render_math_inline(self, tokens, idx, options, env) -> str:
    return f'<span class="math-inline">{escapeHtml(tokens[idx].content)}</span>'


def _render_math_block(self, tokens, idx, options, env) -> str:
    return f'<div class="math-display">{escapeHtml(tokens[idx].content)}</div>'


# 자동으로 불러와도 되는 그림 주소: 잡이 만든 그림(`images/<파일명>`, 렌더 끝에서 잡
# 파일 URL로 재작성된다)과 인라인 data: 래스터뿐이다. 마크다운 본문은 OCR 모델이나
# PDF 텍스트 레이어(보이지 않는 텍스트 포함)가 그대로 옮겨 적은 **비신뢰 입력**이라,
# `![](https://tracker/…)`·`![](//host/…)`·`![](http://192.168.0.1/…)`를 <img>로 두면
# 문서를 여는 순간 브라우저가 클릭 없이 제3자·내부망으로 요청을 보낸다(열람 사실·IP·
# Referer 유출, 내부망 GET 유도). `/api/…` 같은 같은 출처 절대 경로도 비싼 GET(PDF
# 빌드 등)을 유발할 수 있어 막는다. 막힌 그림은 클릭해야 열리는 링크로만 남긴다.
_SAFE_IMAGE_SRC = re.compile(
    r"images/[A-Za-z0-9_-][A-Za-z0-9._-]*\Z|(?i:data:image/(?:png|jpeg|gif|webp);)"
)


def _render_image(self, tokens, idx, options, env) -> str:
    token = tokens[idx]
    src = str(token.attrGet("src") or "")
    if _SAFE_IMAGE_SRC.match(src):
        return self.image(tokens, idx, options, env)
    alt = self.renderInlineAsText(token.children or [], options, env).strip()
    label = escapeHtml(alt) if alt else "이미지"
    return (
        f'<a class="blocked-image" href="{escapeHtml(src)}" '
        f'rel="noopener noreferrer nofollow">[외부 이미지 — 자동으로 불러오지 않음: {label}]</a>'
    )


_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
_md.enable(["table", "strikethrough"])
# allow_space=True: 모델이 `\( [10, 30] \)`처럼 공백을 끼워 넣는 실측 케이스 허용
_md.use(dollarmath_plugin, allow_space=True, double_inline=False)
_md.add_render_rule("math_inline", _render_math_inline)
_md.add_render_rule("math_block", _render_math_block)
_md.add_render_rule("image", _render_image)

# 표 구조 태그만 복원한다. 여는 태그의 **임의 속성**(border/style/class/onclick 등)은
# 전부 버리고 colspan/rowspan(숫자)만 유지한다 — OvisOCR2처럼 모델이 `<table border="1">`
# 로 속성을 붙여도 여는 태그가 통째로 이스케이프돼 표가 깨지던 것을 고친다.
# 태그명 **직후에 경계**(공백/`/`/`&gt;`)를 룩어헤드로 강제한다 — 이게 없으면 `<threshold>`·
# `<trace>` 같은 본문/코드 플레이스홀더의 접두 `th`/`tr`가 표 태그로 오인돼 가운데 텍스트가
# 소리없이 삭제된다. 속성은 태그 경계(&gt;/&lt;)를 넘지 않는 tempered-dot으로 300자까지
# 소거 대상으로 잡는다(백트래킹·폭탄 방어). 원본 속성이 그대로 통과하지 않으므로 XSS-safe.
_TABLE_TAG = re.compile(
    r"&lt;(/?)(table|thead|tbody|tr|th|td)(?=[\s/]|&gt;)"
    r"((?:(?!&gt;|&lt;).){0,300}?)"
    r"\s*/?&gt;"
)
# 진짜 속성은 앞에 공백이 있다 — data-colspan/x-rowspan 같은 접미 속성을 오승격하지 않게
# 선행 공백을 요구한다(구분자 뒤 워드경계만으로는 `-colspan`도 매칭됐다).
_SAFE_TABLE_ATTR = re.compile(r"(?<=\s)(colspan|rowspan)=&quot;(\d{1,3})&quot;")


def _restore_table_tags(html: str) -> str:
    def _repl(m: re.Match) -> str:
        slash, tag, attrs = m.groups()
        safe = ""
        if not slash and attrs:  # 닫는 태그엔 속성이 없다
            for name, val in _SAFE_TABLE_ATTR.findall(attrs):
                safe += f' {name}="{val}"'
        return f"<{slash}{tag}{safe}>"

    return _balance_table_tags(_TABLE_TAG.sub(_repl, html))


# 표 구조 태그 균형 맞추기. 모델 출력이 잘리면(페이지 출력 상한·스트리밍 꼬리)
# `</table>` 없는 표가 남는데, HTML 파서는 그 뒤의 문서 전체를 마지막 `<td>` 안으로
# 빨아들인다 — 실제로 라이브 미리보기가 "빈 표 격자 + 오른쪽 끝으로 밀린 본문"이
# 되던 원인이다. 열린 표 구조는 그 표가 속한 블록(문단·목록 항목·인용·제목·코드
# 블록)이 끝나는 지점에서 닫고, 짝 없는 닫는 태그는 버린다.
# 정상(균형 잡힌) 표는 스택이 비므로 무변경이다.
_TABLE_STRUCT = re.compile(
    r"<(/?)(table|thead|tbody|tr|th|td)\b[^>]*>"
    r"|</(?:p|li|blockquote|pre|h[1-6])>"
)


def _balance_table_tags(html: str) -> str:
    stack: list[str] = []
    out: list[str] = []
    pos = 0
    for m in _TABLE_STRUCT.finditer(html):
        if not stack and m.group(2) is None:
            continue  # 표 밖의 블록 경계 — 볼 일 없다
        if m.group(2) is None:  # 블록 경계 — 열린 표를 여기서 닫는다
            out.append(html[pos:m.start()])
            out.extend(f"</{t}>" for t in reversed(stack))
            stack.clear()
            pos = m.start()
            continue
        closing, tag = m.group(1), m.group(2)
        if not closing:
            stack.append(tag)
            continue
        if tag not in stack:
            # 짝 없는 닫는 태그 — 파서가 무시하지만 원문에 남기지 않는다
            out.append(html[pos:m.start()])
            pos = m.end()
            continue
        out.append(html[pos:m.start()])
        pos = m.start()
        while stack and stack[-1] != tag:
            out.append(f"</{stack.pop()}>")
        stack.pop()
    out.append(html[pos:])
    out.extend(f"</{t}>" for t in reversed(stack))
    return "".join(out)


# ── 수식 델리미터 정규화 (렌더 전용) ──────────────────────────────────
# 엔진별 수식 표기가 다르다 (둘 다 지원해야 한다):
#   Unlimited-OCR: 인라인 `\( … \)`, 디스플레이 `\[ … \]`
#   OvisOCR2·PaddleOCR-VL: 인라인 `$ … $`, 디스플레이 `$$ … $$` (표준 LaTeX)
# `$`는 통화($5)와 수식이 모두 쓰는 모호한 문자다 — `$$`는 항상 수식으로, `$…$`는
# 내용이 LaTeX스러울 때(\^_{} 포함)만 수식으로, 그 외 bare `$`는 통화로 이스케이프한다.
# 결과 md(result.md)는 원본 표기를 그대로 보존하고, 변환은 렌더에서만 한다.
#
# **비용 상한(보안)**: 이 정규화는 인증 없는 POST /render-preview(본문 최대 2MB)가
# 그대로 호출하고, 정규식은 GIL을 놓지 않는다. 그래서 모든 패턴이 입력 길이에
# **선형**이어야 한다 — 짝 없는 여는 델리미터마다 문단 끝까지 다시 훑는 패턴은
# 개행 없는 `\[ a ` 반복 200KiB에 61초(크기 2배→시간 4배) 동안 프로세스 전체를
# 멈췄다. 원칙: (1) 수식 본문은 다음 **짝 없는(이스케이프되지 않은) 여는 델리미터**나
# 빈 줄에서 멈춘다 — 어느 여는 델리미터든 다음 여는 델리미터까지만 보므로 전체 합이
# 선형이다. (2) 코드펜스는 줄 단위 상태 기계로 찾는다(`_mask_code_regions`).
# (3) 마스크 복원은 한 번의 패스로 한다.
_FENCE_CLOSE = {
    "```": re.compile(r"```[ \t]*"),
    "~~~": re.compile(r"~~~[ \t]*"),
}
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# 이스케이프되지 않은 여는 델리미터. `\\[2pt]`(LaTeX 줄바꿈 간격 — aligned·array 안에서
# 흔하다)의 `\[`는 디스플레이 수식의 시작이 아니고, 수식 본문 안에 나와도 본문을
# 끊지 않는다.
_OPEN_DISPLAY = r"(?<!\\)\\\["
_OPEN_INLINE = r"(?<!\\)\\\("
_PARA_BREAK = r"\n[ \t]*\n"
# 디스플레이 수식은 **문단 경계를 넘지 않는다**: 짝이 어긋난 여는 델리미터 하나가
# 다음 델리미터까지의 문단들을 통째로 tex로 삼켜(그 안의 헤딩·이미지가 소실)
# KaTeX 오류 덩어리로 바뀌던 것을 막는다 — 본문에 빈 줄(문단 경계)과 다음 여는
# 델리미터를 금지하는 tempered-dot으로 폭주 범위를 가둔다(표 태그 정규식과 같은 방식).
_MATH_DISPLAY = re.compile(
    rf"{_OPEN_DISPLAY}((?:(?!{_OPEN_DISPLAY}|{_PARA_BREAK}).)+?)\\\]", re.DOTALL
)
# 인라인 수식도 같은 울타리를 친다 — 짝 없는 `\(` 하나(잘린 수식·코드의 리터럴 `\(`)가
# 다음 수식의 `\)`까지 본문 전체를 수식 스팬으로 삼키지 않게.
_MATH_INLINE = re.compile(
    rf"{_OPEN_INLINE}((?:(?!{_OPEN_INLINE}|{_PARA_BREAK}).)+?)\\\)", re.DOTALL
)
_MATH_DOLLAR_DISPLAY = re.compile(rf"\$\$((?:(?!{_PARA_BREAK}).)+?)\$\$", re.DOTALL)
_MATH_DOLLAR_INLINE = re.compile(r"\$([^$\n]+?)\$")
_MATH_LIKE = re.compile(r"[\\^_{}]")  # LaTeX 명령/첨자
_MASK_FMT = "\x00MDMASK{}\x00"
_MASK_TOKEN = re.compile("\x00MDMASK(\\d+)\x00")


def _mask_code_regions(md_text: str, mask) -> str:
    """코드펜스(``` / ~~~)와 인라인 코드를 mask(text)→토큰으로 바꾼다 — 선형 시간.

    의미는 예전 정규식 `^```.*?^```[ \\t]*$|^~~~.*?^~~~[ \\t]*$|`[^`\\n]+``과 같다:
    0열에서 시작하는 펜스 줄은 그 뒤 첫 '닫는 줄'(같은 펜스 문자 + 공백뿐)까지를
    통째로 덮고, 닫는 줄이 없으면 덮지 않는다. 정규식은 닫히지 않은 여는 줄마다
    문서 끝까지 다시 훑어 `"```a\\n" × N`에서 O(N²)이었다 — 여기서는 '다음 닫는 줄'
    위치를 뒤에서부터 한 번에 구해 둔다. 펜스는 줄 전체를 덮고 인라인 코드는 줄을
    넘지 못하므로, 펜스를 먼저 덮고 남은 본문에 인라인 패턴을 돌려도 결과가 같다.
    """
    lines = md_text.split("\n")
    n = len(lines)
    next_close: dict[str, list[int]] = {}
    for fence, closer in _FENCE_CLOSE.items():
        nxt = [-1] * n
        found = -1
        for i in range(n - 1, -1, -1):
            nxt[i] = found  # i보다 **뒤**의 첫 닫는 줄
            if closer.fullmatch(lines[i]):
                found = i
        next_close[fence] = nxt
    out: list[str] = []
    i = 0
    while i < n:
        line = lines[i]
        fence = line[:3]
        j = next_close[fence][i] if fence in next_close else -1
        if j != -1:
            out.append(mask("\n".join(lines[i : j + 1])))
            i = j + 1
        else:
            out.append(line)
            i += 1
    return _INLINE_CODE.sub(lambda m: mask(m.group(0)), "\n".join(out))


def _is_inline_dollar_math(tex: str) -> bool:
    """`$…$` 내용이 수식인가 — LaTeX스럽거나(단항 포함), 통화가 아닌 짧은 변수식.

    구분: 수식 변수($T·$x·$\\tau)는 문자/기호로 시작하고, 통화($5·$10·$5 그리고·
    $5 million)는 숫자로 시작한다. 숫자+변수($2x, LaTeX 없음)는 드물어 리터럴로 두는
    편이 안전하다 — KaTeX 오류보다 원문 텍스트가 낫다. 긴 산문은 길이로 배제."""
    tex = tex.strip()
    if _MATH_LIKE.search(tex):
        return True
    return bool(tex) and not tex[0].isdigit() and len(tex) <= 40


def _normalize_math_delimiters(md_text: str) -> str:
    """엔진별 수식 표기(`\\(..\\)`/`\\[..\\]` 및 `$..$`/`$$..$$`)를 dollarmath 대상으로
    정규화한다. 코드 구간은 마스킹, 통화용 bare `$`는 이스케이프해 오탐을 막는다."""
    masked: list[str] = []

    def _mask_literal(text: str) -> str:
        masked.append(text)
        return _MASK_FMT.format(len(masked) - 1)

    # 0) NUL은 CommonMark가 어차피 U+FFFD로 바꾼다(markdown-it도 동일) — 미리 바꿔
    #    본문이 마스크 토큰을 위조해 복원 단계를 교란하지 못하게 한다.
    md_text = md_text.replace("\x00", "\ufffd")

    # 1) 코드펜스/인라인 코드 보호
    md_text = _mask_code_regions(md_text, _mask_literal)

    # 2) 모델이 `$$`/`$`로 낸 수식을 **통화 이스케이프 전에** 마스킹(Ovis/Paddle).
    #    $$는 항상 수식, $…$는 LaTeX스러운 내용일 때만(그 외는 통화로 남겨 이스케이프).
    def _mask_dollar_display(m: re.Match) -> str:
        tex = m.group(1).strip()
        return _mask_literal(f"\n\n$$\n{tex}\n$$\n\n") if tex else ""

    def _mask_dollar_inline(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex or not _is_inline_dollar_math(tex):
            return m.group(0)  # 통화 등 — 마스킹하지 않고 아래에서 이스케이프되게 둔다
        return _mask_literal(f"${tex}$")

    md_text = _MATH_DOLLAR_DISPLAY.sub(_mask_dollar_display, md_text)
    md_text = _MATH_DOLLAR_INLINE.sub(_mask_dollar_inline, md_text)

    # 3) 남은 bare `$`(통화)는 이스케이프 — 이 함수가 만든 $-델리미터만 수식이 된다
    md_text = md_text.replace("$", "\\$")

    # 4) Unlimited의 `\(..\)`/`\[..\]` → `$..$`/`$$..$$`
    def _display(m: re.Match) -> str:
        tex = m.group(1).strip()
        return f"\n\n$$\n{tex}\n$$\n\n" if tex else ""

    def _inline(m: re.Match) -> str:
        tex = m.group(1).strip()
        return f"${tex}$" if tex else ""

    md_text = _MATH_DISPLAY.sub(_display, md_text)
    md_text = _MATH_INLINE.sub(_inline, md_text)

    # 5) 마스킹 복원 (코드 + $$/$ 수식) — 한 번의 패스. 토큰마다 전체 문자열을
    #    다시 훑는 replace 반복은 마스크 수 × 길이(인라인 코드 50만 개짜리 2MB 본문에서
    #    10^12)라 이것만으로도 프로세스를 멈춘다. `$…$` 마스크 안에 먼저 만든 코드
    #    마스크가 들어 있을 수 있어 **더 앞 번호만** 재귀로 펼친다(종료·크기 보장 —
    #    토큰마다 원문은 한 곳에만 있다).
    def _restore(text: str, limit: int) -> str:
        def _repl(m: re.Match) -> str:
            idx = int(m.group(1))
            if idx >= limit:
                return m.group(0)
            return _restore(masked[idx], idx)

        return _MASK_TOKEN.sub(_repl, text)

    return _restore(md_text, len(masked))


# ── 플레인 텍스트 + 수식 스팬 (마크다운이 아닌 문맥용 — 레이아웃 뷰 등) ──
# 마크다운 경로(_MATH_DISPLAY/_MATH_INLINE)와 같은 울타리: 이스케이프되지 않은 다음
# 여는 델리미터나 빈 줄을 넘지 않는다(짝 없는 `\(`가 뒤 본문을 삼키지 않고, 비용도 선형).
_MATH_ANY = re.compile(
    rf"{_OPEN_DISPLAY}((?:(?!{_OPEN_DISPLAY}|{_PARA_BREAK}).)+?)\\\]"
    rf"|{_OPEN_INLINE}((?:(?!{_OPEN_INLINE}|{_PARA_BREAK}).)+?)\\\)",
    re.DOTALL,
)


def text_with_math_html(text: str) -> str:
    """플레인 텍스트를 전부 이스케이프하되 `\\(..\\)`/`\\[..\\]` 구간은
    KaTeX 대상 `.math-inline`/`.math-display` 스팬으로 변환한다."""
    out: list[str] = []
    pos = 0
    for m in _MATH_ANY.finditer(text):
        out.append(escapeHtml(text[pos:m.start()]))
        display_tex, inline_tex = m.group(1), m.group(2)
        tex = (display_tex if display_tex is not None else inline_tex).strip()
        if tex:
            cls = "math-display" if display_tex is not None else "math-inline"
            out.append(f'<span class="{cls}">{escapeHtml(tex)}</span>')
        pos = m.end()
    out.append(escapeHtml(text[pos:]))
    return "".join(out)


# ── figure 상대 폭 주입 (렌더 후처리 — result.md/원문 불변) ───────────
# 벤더 P13이 export한 boxes.json(픽셀 bbox + 페이지 크기)으로 각 figure를
# 원본 페이지 대비 상대 폭으로 표시. 값은 전부 서버가 계산한 숫자라 안전하다.
_IMG_TAG = re.compile(r'<img src="([^"]+/images/([^"/]+))" alt="([^"]*)"\s*/?>')
_CENTER_THRESHOLD = 0.6
_MIN_REL_W = 0.08


def _inject_figure_widths(html: str, figure_boxes: dict) -> str:
    def _repl(m: re.Match) -> str:
        src, name, alt = m.groups()
        meta = figure_boxes.get(name)
        if not isinstance(meta, dict):
            return m.group(0)
        try:
            rel_w = (float(meta["x2"]) - float(meta["x1"])) / float(meta["image_width"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return m.group(0)
        if not (0 < rel_w <= 1.5):  # 비정상 메타는 무시하고 풀폭 폴백
            return m.group(0)
        rel_w = min(max(rel_w, _MIN_REL_W), 1.0)
        style = f"width:{rel_w * 100:.1f}%;height:auto;"
        if rel_w < _CENTER_THRESHOLD:
            style += "display:block;margin-left:auto;margin-right:auto;"
        return f'<img src="{src}" alt="{alt}" style="{style}">'

    return _IMG_TAG.sub(_repl, html)


def render_markdown_html(
    markdown_text: str, files_base_url: str, figure_boxes: dict | None = None
) -> str:
    """`![](images/...)` 상대 참조를 잡 파일 서빙 URL로 재작성해 렌더.
    figure_boxes(images/boxes.json)가 있으면 figure에 원본 상대 폭을 주입한다."""
    html = _md.render(_normalize_math_delimiters(markdown_text))
    html = _restore_table_tags(html)
    html = html.replace('src="images/', f'src="{files_base_url}/images/')
    if figure_boxes:
        html = _inject_figure_widths(html, figure_boxes)
    return html


def render_document_html(
    markdown_text: str,
    files_base_url: str,
    figure_boxes: dict | None = None,
    page_separator: str = "\n\n---\n\n",
) -> str:
    """최종 문서 렌더(/html 전용): 페이지 경계를 `<section class="doc-page">`로 승격.

    소스(result.md)는 포터빌리티를 위해 `---` 구분자를 유지하고, 경계 해석은
    렌더에서만 한다. 본문이 우연히 구분자와 동일한 텍스트를 포함하면 초과
    분할될 수 있는 best-effort 휴리스틱 (실측 코퍼스에서 미관측).
    라이브 프리뷰(/render-preview)는 기존 flat 렌더를 그대로 쓴다.
    """
    if not markdown_text.strip():
        return ""
    segments = markdown_text.split(page_separator) if page_separator else [markdown_text]
    if len(segments) == 1:
        return render_markdown_html(markdown_text, files_base_url, figure_boxes)
    parts = []
    for i, seg in enumerate(segments, start=1):
        inner = render_markdown_html(seg, files_base_url, figure_boxes)
        parts.append(f'<section class="doc-page" data-page="{i}">\n{inner}</section>')
    return "\n".join(parts)
