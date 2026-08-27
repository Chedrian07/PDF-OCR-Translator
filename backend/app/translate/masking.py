"""비언어 토큰 마스킹 — 번역 전 치환, 번역 후 복원·검증.

수식·코드·이미지·URL·인용·참조·HTML 태그는 번역해서는 안 되는 불변 토큰이다.
이들을 `<m1 v="…"/>` 형태의 플레이스홀더로 바꿔 LLM에 넘기고(v는 조사 선택용
미리보기), 번역문에서 원문으로 복원한다. 복원 실패(누락·중복)는 검증에서 보고되며
엔진이 해당 유닛을 원문 유지로 처리한다.

설계 핵심:
  * **단일 패스 결합 정규식** — 우선순위 순서의 alternation 하나로 스캔한다.
    re.sub는 치환 결과를 재스캔하지 않으므로 플레이스홀더가 자기 자신을
    다시 매칭하는 문제가 원천적으로 없다.
  * 플레이스홀더 번호는 종류를 가로질러 전역 1-based로 증가한다 (예: m1 c2 f3).
  * 복원은 관용적 — 슬래시 누락·속성 변형·공백 삽입을 모두 허용한다.
"""

from __future__ import annotations

import re

# ── 결합 토큰 정규식 (우선순위 = alternation 순서) ────────────────────────
# 그룹명 첫 글자가 종류 코드다: m=수식 k=코드 g=이미지 t=태그 u=URL/DOI/이메일
# c=인용 f=참조. DOTALL은 `.`에만 영향 → 펜스/디스플레이 수식만 개행을 넘는다.
_TOKEN_RE = re.compile(
    r"(?P<k1>```.*?```)"                                    # 1 펜스 코드
    r"|(?P<m1>\$\$.*?\$\$)"                                 # 2 디스플레이 수식
    # 3 인라인 수식(개행 1개 허용, 비어있지 않음). 통화 표기 오인 방지 3규칙:
    #   여는 $ 뒤 공백 금지 / 닫는 $ 앞 공백 금지 / 닫는 $ 뒤 숫자 금지.
    #   "costs $5 and $7 total"은 종전에 "$5 and $"를 수식으로 잡아 그 사이 산문을
    #   통째로 마스킹했다(영어 그대로 잔존). 세 규칙 모두 실 LaTeX(`$x_i$`)엔 무해하다.
    r"|(?P<m2>\$(?![\s$])[^$\n]*(?:\n[^$\n]*)?(?<![\s$])\$(?!\d))"
    # 3b LaTeX 델리미터 — render.py 포터빌리티 계약상 result.md는 \(..\)/\[..\]를
    #    원본 그대로 유지하므로 $-정규화 여부와 무관하게 수식으로 보호한다.
    r"|(?P<m3>\\\[.*?\\\])"                                 # 디스플레이 \[..\]
    r"|(?P<m4>\\\(.*?\\\))"                                 # 인라인 \(..\)
    r"|(?P<k2>`[^`\n]+`)"                                   # 4 인라인 코드
    r"|(?P<g1>!\[[^\]]*\]\([^)\s]*\))"                      # 5 이미지
    r"|(?P<t1></?[a-zA-Z][^>]*>)"                           # 6 HTML 태그
    r"|(?P<u1>https?://\S+|\b10\.\d{4,}/\S+|[\w.+%-]+@[\w-]+\.[\w.-]+)"  # 7 URL/DOI/이메일
    r"|(?P<c1>\[\d+(?:\s*[,–-]\s*\d+)*\])"             # 8 인용 [1] [1, 2] [3-5]
    r"|(?P<f1>\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eqs?\.?|Section|Sec\.?"
    r"|Appendix|Algorithm|Alg\.?)\s*\(?\d+(?:\.\d+)*\)?)"  # 9 Fig/Table/Eq/Sec 참조
    # 10 셸 프롬프트 줄 · 11 단계 마커 · 12 `ls -l` 타임스탬프.
    # 부록의 에이전트 트랜스크립트는 명령과 산문이 한 유닛에 섞여 있어 블록 단위
    # skip(should_skip)이 (오탐을 피하려고 올바르게) 거부한다. 그러면 명령까지
    # 번역돼 재현 불가능한 산출물이 된다 — 실측 result.ko.md:
    #   `$ condensation` → `$ 응축`,  `May 6 10:26` → `5월 6일 10:26`,
    #   `[Step 85/100]` → `[단계 85/100]`.
    # 줄 단위로 불변 토큰으로 묶으면 산문은 번역되고 명령은 그대로 지나간다.
    r"|(?P<k3>^[ \t]*[$%>][ \t]+[^\n]*)"
    r"|(?P<k4>^[ \t]*\[?[ \t]*Step[ \t]*:?[ \t]*\d+[ \t]*/[ \t]*\d+[ \t]*\]?[ \t]*(?=\n|$))"
    # 크기 + 월 + 일 + 시각 + 파일명이 이어지는 `ls -l` 행의 타임스탬프만. 산문 속
    # 날짜(`in May 2024 we …`)는 크기·파일명이 없어 걸리지 않는다.
    r"|(?P<k5>(?<=\s)\d+[ \t]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[ \t]+\d{1,2}[ \t]+(?:\d{1,2}:\d{2}|\d{4})(?=[ \t]+\S))",
    re.DOTALL | re.MULTILINE,
)

# 복원·잔여 검사에 쓰는 플레이스홀더 인식 패턴 (id 접두 = 종류 코드)
_PLACEHOLDER_RE = re.compile(r"<[mkgucft]\d+\b[^>]*>")
_RESIDUAL_RE = re.compile(r"<[mkgucft]\d+")


def _preview(s: str) -> str:
    """원문 앞 12자 미리보기 — 따옴표·개행·꺾쇠 제거(플레이스홀더 문법 보호)."""
    s = re.sub(r"[\"'\n\r<>]", "", s)
    return s.strip()[:12]


# 프롬프트 줄을 "프롬프트+명령 이름" / "나머지 인자"로 가르는 패턴.
# `$ think "There is no evidence that …"`처럼 인자가 자연어인 명령이 실제로 있고,
# 줄 전체를 불변으로 묶으면 그 문장이 번역면에 영어로 남는다(사용자 규칙 3 위반).
# 반대로 줄 전체를 번역하면 명령 이름까지 번역된다(`$ condense` → `$ 응축`).
# 그래서 명령 이름까지만 불변으로 묶고 자연어 인자는 번역 대상으로 남긴다.
_K3_HEAD = re.compile(r"^([ \t]*[$%][ \t]+\S+)([ \t]+.*)?$")
# `>`는 셸 연속 프롬프트이기도 하고 마크다운 인용 기호이기도 하다. 뒤따르는 명령
# 이름이 없으므로 기호만 보호하고 나머지는 전부 번역 대상으로 넘긴다 — 그래야
# `> The rest of the paper is organized as follows…` 같은 인용문이 번역된다.
_K3_MARK = re.compile(r"^([ \t]*>)([ \t]+.*)?$")


def mask(text: str) -> tuple[str, dict[str, str]]:
    """비언어 토큰을 플레이스홀더로 치환. (masked, {placeholder_id → 원문}) 반환."""
    mapping: dict[str, str] = {}
    counter = [0]
    depth = [0]

    def _place(kind: str, original: str) -> str:
        counter[0] += 1
        pid = f"{kind}{counter[0]}"
        mapping[pid] = original
        return f'<{pid} v="{_preview(original)}"/>'

    def _repl(m: re.Match) -> str:
        kind = m.lastgroup[0]  # 그룹명 첫 글자 = 종류 코드
        original = m.group()
        # 프롬프트 줄의 인자가 자연어면 명령 이름까지만 보호하고 인자는 번역시킨다.
        # depth 가드: 잘라낸 꼬리를 재스캔할 때 이 분기를 다시 타지 않게 한다.
        if m.lastgroup == "k3" and depth[0] == 0:
            pat = _K3_MARK if original.lstrip()[:1] == ">" else _K3_HEAD
            head = pat.match(original)
            tail = (head.group(2) or "") if head else ""
            if tail.strip() and _prose_line(tail):
                out = _place("k", head.group(1))
                depth[0] += 1
                try:
                    return out + _TOKEN_RE.sub(_repl, tail)
                finally:
                    depth[0] -= 1
        return _place(kind, original)

    return _TOKEN_RE.sub(_repl, text), mapping


def _lenient_re(pid: str) -> re.Pattern:
    """관용 복원 패턴 — `<m1>`, `< m1 />`, 속성 변형·슬래시 누락 모두 허용."""
    return re.compile(r"<\s*" + re.escape(pid) + r"\b[^>]*?/?\s*>")


def unmask(translated: str, mapping: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """플레이스홀더를 원문으로 복원. (복원문, missing_ids, dup_ids) 반환.

    각 id는 정확히 1회 등장이 정상 — 0회는 missing, 2회 이상은 dup(전부 복원하되
    실패로 보고). 복원 후에도 남은 `<m1`류 잔여물이 있으면 dup에 추가한다.
    """
    missing: list[str] = []
    dup: list[str] = []
    out = translated
    for pid, original in mapping.items():
        pat = _lenient_re(pid)
        n = len(pat.findall(out))
        if n == 0:
            missing.append(pid)
        elif n >= 2:
            dup.append(pid)
        # 개수와 무관하게 전부 복원 (lambda로 원문의 백슬래시·그룹참조를 리터럴 취급)
        out = pat.sub(lambda _m, _o=original: _o, out)
    for m in _RESIDUAL_RE.finditer(out):
        dup.append(m.group())
    return out, missing, dup


# ── 모델 발명 딜리미터·페이지 마커 새니타이즈 (unmask 이전 원출력에 적용) ─────
# 마스킹된 원문에는 수식 딜리미터가 전혀 없다(전부 플레이스홀더). 따라서 모델
# 출력에 나타나는 `\(` `\[` `$$` 등은 전부 모델이 만들어낸 것이며, 남겨두면
# 수식 카운트 불일치·원문 오염을 일으킨다. 리터럴 <PAGE>도 모델 발명품이다.
# 플레이스홀더 태그의 v 속성 내부에서 치환돼도 무해하다 — unmask는 id로 매칭하고
# 이 치환들은 `<` `>` `/` `=` `"`를 건드리지 않아 태그 구조를 깨지 않는다.
_SANITIZE_SUBS: tuple[tuple[str, str], ...] = (
    ("\\(", "("),
    ("\\)", ")"),
    ("\\[", "["),
    ("\\]", "]"),
    ("$$", ""),      # 이중 달러 제거 (실 수식은 마스킹으로 보호됨)
    ("<PAGE>", ""),  # 리터럴 페이지 마커 제거
)


def sanitize_translation(raw: str) -> tuple[str, int]:
    """모델 발명 수식 딜리미터·리터럴 <PAGE>를 제거. (정리문, 치환 건수) 반환.

    엔진의 모든 complete() 출력 경로(최초·repair·분할 반쪽)에 unmask 직전 적용한다.
    치환들은 서로 겹치는 문자열을 만들지 않으므로 순서·연쇄 재매칭 문제가 없다.
    치환은 전체에 적용하되 **카운트는 플레이스홀더 태그 밖만** 센다 — v 속성의
    수식 미리보기(v="\\( E=mc…")까지 세면 리포트가 실측(25p 784건)처럼 부풀려진다.
    """
    outside = _PLACEHOLDER_RE.sub("", raw)  # 카운트 전용 — 태그(속성 포함) 제거본
    count = 0
    out = raw
    for needle, repl in _SANITIZE_SUBS:
        if needle in out:
            count += outside.count(needle)
            out = out.replace(needle, repl)
    return out, count


# 참고문헌 항목 줄 — "[12] Gersho, A. …" 형태. 헤딩("References") 기반 스킵은
# OCR 출력이 헤딩을 안 뽑으면(실측 2504.19874v1 — 굵은 텍스트/부재) 무력하므로,
# 항목 모양으로 페이지·뷰(md/layout) 무관하게 감지한다.
_REF_LINE_RE = re.compile(r"^\s*\[\d+\]\s+\S")
# 단일 항목일 때 서지 정보 증거 — 연도·arXiv·이니셜("Smith, J.")·권/쪽 표기
_REF_EVIDENCE_RE = re.compile(r"\b(?:19|20)\d{2}\b|arXiv|,\s*[A-Z]\.|\bpp\.\s*\d|\bvol\.\s*\d", re.I)


def _looks_reference_list(text: str) -> bool:
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return False
    hits = sum(1 for ln in lines if _REF_LINE_RE.match(ln))
    if hits >= 2:
        return True  # [n] 항목이 여럿 → 참고문헌 목록
    # 단일 [n] 줄은 서지 증거가 있을 때만 (본문 "[1] shows that…" 오탐 방지)
    return hits == len(lines) == 1 and bool(_REF_EVIDENCE_RE.search(text))


# ── 코드·CLI 트랜스크립트 판별 ────────────────────────────────────────────
# 부록의 에이전트 트랜스크립트·소스 리스팅은 번역 대상이 아니다. 그런데 셸 명령과
# 코드에는 알파벳이 많아 아래 non-linguistic 규칙(문자 비율 0.3)을 그대로 통과한다.
# 그 결과 왕복 한 번을 쓰고, 출력 게이트(hangul-ratio)가 거부하고, 래더를 소진한 뒤
# 원문 유지로 떨어진다 — PDF에는 "미번역"으로 집계된다. 더 나쁜 경우는 게이트를
# 통과해 **명령이 실제로 번역되는 것**이다(실측: `$ condensation` → `$ 응축`,
# `ls -l` 출력의 `May 6 10:26` → `5월 6일 10:26`). 그러면 재현 불가능한 명령이
# 산출물에 실린다. 그래서 입력 단계에서 걸러 "의도적 원문 유지"로 만든다.
#
# 오탐이 미탐보다 나쁘다: 건너뛴 산문은 재시도도 경고도 없이 영문으로 굳는다.
# 그래서 **산문 줄이 하나라도 있으면 건너뛰지 않는다**. 실측(46p 논문 556블록):
# 코드로 판정 61건 · 그중 산문 오탐 0건 · 현재 실패하는 47건 중 39건 회수.
# 프롬프트 문자 뒤에는 **공백**이 있어야 한다 — `$E = mc^2$`(인라인 수식)를
# 셸 명령으로 오인하지 않기 위해서다. `#`은 아예 뺀다: 마크다운 제목(`# Deep
# Learning`)과 구분할 수 없고, 제목을 건너뛰면 산문이 통째로 영문으로 남는다.
_SHELL_PROMPT = re.compile(r"^\s*[$>%]\s+\S")
_STEP_MARKER = re.compile(r"^\s*#?\s*\[?\s*(?:Step|단계)\s*:?\s*\d+\s*/\s*\d+\s*\]?\s*$")
# OCR이 권한 문자열에 가짜 하이픈을 끼워 넣는다(`-r-w-r--r--`) — 길이에 여유를 둔다.
_LS_LONG = re.compile(r"^\s*[-dlbcps][-rwxsStT]{8,12}\s+\d+\s+\S+")
_JSONISH = re.compile(r"""^\s*[\{\[].*["'][A-Za-z_]+["']\s*[:=]""")
# 셸 정형 출력. 로그 레벨(INFO/ERROR/MESSAGE …)은 넣지 않는다 — 필드 라벨 뒤의
# 산문을 통째로 삼켜 커밋 메시지 16건을 오탐했다(실측 p19).
_SHELL_OUTPUT = re.compile(
    r"^\s*(?:total\s+\d+\s*$"
    r"|\w[\w.-]*:\s*(?:cannot |No such |command not found|Permission denied))"
)
_DECL_KEYWORD = re.compile(
    r"^\s*(?:use|import|from|package|require|#include|#define|def|class|fn|func|"
    r"impl|struct|pub|let|const|var|module|namespace)\s+\S"
)
_CODE_OPS = re.compile(r"::|->|=>|!==|===|==|!=|\+=|-=|&&|\|\||\bself\b")
_ASSIGN = re.compile(r"^\s*[A-Za-z_][\w.\[\]'\"+:-]*\s*=\s*\S")
_BLOCK_OPEN = re.compile(r"[:{]\s*$")
_PATHY = re.compile(
    r"(?:^|\s)(?:\.{0,2}/[\w./-]+|[\w-]+\.(?:py|rs|sh|c|cc|cpp|h|js|ts|json|jpg|md))"
)
_CALLISH = re.compile(r"\w\([^)]*\)")
_IDENT_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_PROMPT_OR_COMMENT = re.compile(r"^\s*(?:[$#>%]+|//|--|;;)\s*")
# 산문 판정 전용 기능어. `with`/`as`/`in`/`for`/`if`/`not`은 파이썬·러스트 키워드
# 이기도 해서 코드 한 줄을 산문으로 오인시킨다(`with open(...) as f:`) — 뺐다.
_PROSE_WORDS = frozenset("""
the of to this that these those it its we our you your they them their there here
was were been being has have had does did would could should might must
which what when where why how because however therefore although while whereas
than such more most also both each any every other another same different
""".split())


def _identifier_list(text: str) -> bool:
    """공백 없는 소문자 식별자를 쉼표로 나열한 블록 (프로젝트명 목록 등)."""
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 8:
        return False
    return sum(1 for p in parts if _IDENT_TOKEN.match(p)) / len(parts) >= 0.9


def _code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if (_SHELL_PROMPT.match(s) or _STEP_MARKER.match(s) or _LS_LONG.match(s)
            or _JSONISH.match(s) or _SHELL_OUTPUT.match(s)):
        return True
    if _DECL_KEYWORD.match(s) and not _SENTENCE_END.search(s):
        return True
    if _ASSIGN.match(s) and not _SENTENCE_END.search(s):
        return True   # 단독 대입문은 그 자체로 코드다 (`data[sos+4] = 2`)
    if _SENTENCE_END.search(s):
        return False
    signals = sum((
        bool(_CODE_OPS.search(s)), bool(_BLOCK_OPEN.search(s)), s.endswith(";"),
        bool(_PATHY.search(s)), bool(_CALLISH.search(s)),
    ))
    return signals >= 2


def _prose_line(line: str) -> bool:
    """자연어 문장으로 볼 만한 줄인가 — 하나라도 있으면 블록을 건너뛰지 않는다.

    OCR이 산문 줄 앞에 프롬프트 기호를 잘못 붙이기도 하므로(`$ Since there are no
    pre-built binaries, …`) 기호를 벗겨 낸 본문으로 판정한다.
    """
    s = line.strip()
    if (not s or _LS_LONG.match(s) or _JSONISH.match(s) or _STEP_MARKER.match(s)
            or _SHELL_OUTPUT.match(s)):
        return False
    body = _PROMPT_OR_COMMENT.sub("", s)
    words = [w.lower() for w in _WORD.findall(body)]
    if len(words) < 6:
        return False
    if _DECL_KEYWORD.match(body) or body.rstrip().endswith(";"):
        return False
    if sum(1 for w in words if w in _PROSE_WORDS) < 2:
        return False
    return len(re.findall(r"[^\w\s]", body)) / max(1, len(body)) < 0.18


def looks_like_code(text: str) -> str:
    """코드·CLI 트랜스크립트면 사유 슬러그, 아니면 빈 문자열."""
    if re.search(r"[가-힣]", text or ""):
        return ""                       # 이미 한국어가 섞였으면 판단하지 않는다
    if _identifier_list(text):
        return "identifier-list"
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    if any(_prose_line(ln) for ln in lines):
        return ""                       # 산문이 섞였다 — 번역 대상
    return "code" if sum(1 for ln in lines if _code_line(ln)) * 2 > len(lines) else ""


def should_skip(text: str) -> str:
    """번역 불필요 사유를 반환(빈 문자열이면 번역 대상).

    already-korean: 한글이 비공백 문자의 과반 → 이미 번역됨.
    identifier: 전체가 arXiv id 패턴뿐.
    references: 참고문헌 항목 목록 — 서지 정보는 원문 유지가 정책.
    code / identifier-list: 셸 트랜스크립트·소스 코드·식별자 나열 — 번역하면 안 된다.
    non-linguistic: 마스킹 후 잔여에 2자+ 알파벳 단어가 없거나 영문자 비율 < 0.3.
    """
    stripped = text.strip()
    if not stripped:
        return "non-linguistic"

    if _looks_reference_list(text):
        return "references"

    non_ws = len(re.findall(r"\S", text))
    hangul = len(re.findall(r"[가-힣]", text))
    if non_ws and hangul / non_ws > 0.5:
        return "already-korean"

    if re.fullmatch(r"(?:arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*)+", stripped):
        return "identifier"

    code_reason = looks_like_code(text)
    if code_reason:
        return code_reason

    masked, _ = mask(text)
    residual = _PLACEHOLDER_RE.sub(" ", masked)
    if not re.search(r"[A-Za-z]{2,}", residual):
        return "non-linguistic"
    letters = len(re.findall(r"[A-Za-z]", residual))
    non_ws_r = len(re.findall(r"\S", residual))
    if non_ws_r == 0 or letters / non_ws_r < 0.3:
        return "non-linguistic"
    return ""


# 모델 거부문 — 길이·한글 비율 검사를 빠져나가는 짧은 유닛까지 잡는 마지막 그물.
# 원문에 같은 표현이 있으면(거부를 다루는 문서) 정상 번역이므로 호출부에서 제외한다.
_REFUSAL_RE = re.compile(
    r"(?:cannot|can'?t|unable to|won'?t)\s+(?:translate|assist|help|comply|process)"
    r"|as an ai(?:\s+language)?\s+model"
    r"|i'?m (?:sorry|unable)"
    r"|i am (?:sorry|unable)"
    # 거부문만 잡도록 좁힌다 — 종전 `번역…없`는 "번역 없이"·"번역이 없는" 같은
    # 정상 산문까지 거부문으로 판정했다(원문이 영어라 src 가드도 무력하다).
    # "수 없"/"불가"를 반드시 요구하고 사이 어절도 거부문 관용구로 한정한다.
    r"|번역(?:을|이|은|할|해)?\s*(?:드릴|제공할|수행할)?\s*수\s*없"
    r"|번역(?:을|이|은)?\s*불가"
    r"|죄송(?:합니다|해요)"
    r"|도와드릴\s*수\s*없",
    re.IGNORECASE,
)


# 프롬프트 구조 마커 — 번역 결과에 절대 나올 수 없는 문자열이다. 모델이 프롬프트를
# 되돌려주거나(echo 계열 장애) repair 프롬프트를 그대로 출력하면 이 마커가 산출물에
# 박힌다. 실측: fault=echo 실행에서 repair 프롬프트가 result.ko.md에 20벌 들어갔다.
# 문구는 prompts.py가 소유하며, 드리프트는 test_translate_masking.py의
# test_스캐폴딩_마커가_실제_프롬프트와_일치한다 가 잡는다(import 결합 없이 계약만 고정).
_SCAFFOLD_RE = re.compile(
    r"\[번역할 원문|\[원문 유지|\[용어집|\[직전 문맥|\[첫 등장 병기|\[블록 유형"
    r"|\[수정할 번역문|\[원문 — 아래 꺾쇠 태그|\[논문 개요|\[섹션 제목|\[용어 후보"
    r"|다음 꺾쇠 태그가"
)


def looks_untranslated(src: str, out: str, mapping: dict) -> bool:
    """출력 측 최소 검증 — 거부문·요약·원문 echo면 True(엔진이 래더로 보낸다).

    판정 사유가 필요하면 untranslated_reason()을 쓴다 (이 함수는 그 얇은 래퍼다).
    """
    return bool(untranslated_reason(src, out, mapping))


def untranslated_reason(src: str, out: str, mapping: dict) -> str:
    """게이트 판정 사유 — 통과면 "", 거부면 어느 규칙이 걸었는지 나타내는 슬러그.

    사유: scaffold / refusal / hangul-ratio / length-ratio.

    **오탐 관측용이다.** 게이트는 오탐해도 조용하다 — 정상 번역이 거부되면 래더
    왕복이 늘고, 래더가 소진되면 그 문단이 영어로 남는다(kept_reason=gate-rejected).
    총합만으로는 어떤 규칙이 몇 건을 걸었는지 알 수 없어, 임계값을 조정해야 하는지
    공급자가 고장 난 것인지 운영자가 구분할 수 없었다. 엔진이 사유별로 집계해
    report.json의 gate_reasons에 남긴다.

    입력 측 should_skip()과 대칭인 게이트다. 플레이스홀더 정합만으로는 모델
    거부문("I cannot translate this text.")·한 줄 요약·영문 echo가 문단을 통째로
    대체해도 '성공'으로 통과해 units.json에 캐시된다(무손실 원칙 위반).

    **오탐도 미탐과 마찬가지로 번역 손실이다** — 거절된 출력은 래더(repair→분할)로
    가고 래더가 소진되면 "kept"(원문 보존)로 떨어진다. 즉 정상 번역을 오탐하면
    그 유닛은 영어 원문 그대로 PDF에 남는다. 따라서 임계값은 "실측 분포 밖"에만
    둔다 — 아래 수치는 data/jobs 실번역 쌍 169건으로 측정했다
    (tests/fixtures/real_translation_pairs.json에 고정).
    """
    # 거부문은 길이·한글 비율 검사보다 먼저 본다. 아래 "짧은 원문 면제"는 고유명사가
    # 그대로 되돌아오는 echo를 허용하려는 것이지 원문이 임의 문장으로 대체되는 것을
    # 허용하려는 게 아니다. 면제가 앞서면 'Abstract' 같은 한 단어 제목이 거부문으로
    # 통째로 바뀌어도 통과한다(실 PDF 하네스에서 실측된 유출 경로).
    # 프롬프트 스캐폴딩이 섞인 출력은 어떤 경우에도 번역이 아니다. 원문 길이·언어와
    # 무관하므로 모든 면제보다 앞에 둔다(원문에 같은 문구가 있으면 정상이므로 제외).
    if _SCAFFOLD_RE.search(out) and not _SCAFFOLD_RE.search(src):
        return "scaffold"

    if _REFUSAL_RE.search(out) and not _REFUSAL_RE.search(src):
        return "refusal"

    residual = _PLACEHOLDER_RE.sub(" ", mask(src)[0])
    src_words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", residual)]
    if len(src_words) < 2:
        return ""  # 고유명사·짧은 라벨은 원문 그대로 나와도 정상
    # 복원된 불변 토큰(수식·URL·코드)은 한글일 수 없으므로 비율 계산에서 뺀다 —
    # 넣고 세면 수식·표가 많은 정상 번역이 거부문으로 오탐된다.
    out_text = out
    for original in mapping.values():
        out_text = out_text.replace(original, " ")
    non_ws = len(re.findall(r"\S", out_text))
    hangul_ratio = len(re.findall(r"[가-힣]", out_text)) / non_ws if non_ws else 0.0
    if non_ws and hangul_ratio < 0.15:
        # 한글이 적다고 곧바로 echo는 아니다 — 사사문·저자 소속처럼 고유명사가
        # 대부분인 정상 번역은 실측 한글 비율이 0.09~0.13까지 내려간다. 진짜 echo는
        # 원문 영단어가 그대로 다 남는다(잔존율 1.0). 두 신호를 함께 본다.
        out_words = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", out)}
        retention = sum(1 for w in src_words if w in out_words) / len(src_words)
        if hangul_ratio < 0.05 or retention >= 0.9:
            return "hangul-ratio"  # 한글이 사실상 없음 / 원문 영단어 그대로 → 거부문·echo
    # 한국어 거부문·한 줄 요약은 한글 비율을 통과하므로 길이비로 잡는다.
    # 하한은 **원문 길이로 나눈다** — 한국어는 짧은 명사구일수록 압축이 극단적이다
    # ('Writing the introduction'→'서론 쓰기' 0.208). 실측 165쌍 분포:
    #   len(src)<=80  min 0.208 / p5 0.227 / p10 0.296
    #   len(src)>80   min 0.444 / p10 0.474
    # 하한을 각 구간 분포 밖(0.12 / 0.25)에 둬 오탐 0을 확보한다.
    lo = 0.12 if len(src) <= 80 else 0.25
    hi = 4.0 if mapping else 3.0  # 수식·표 유닛은 상한 완화
    if lo * len(src) <= len(out) <= hi * len(src):
        return ""
    return "length-ratio"
