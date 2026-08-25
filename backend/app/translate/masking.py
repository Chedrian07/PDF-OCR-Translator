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
    r"|Appendix|Algorithm|Alg\.?)\s*\(?\d+(?:\.\d+)*\)?)",  # 9 Fig/Table/Eq/Sec 참조
    re.DOTALL,
)

# 복원·잔여 검사에 쓰는 플레이스홀더 인식 패턴 (id 접두 = 종류 코드)
_PLACEHOLDER_RE = re.compile(r"<[mkgucft]\d+\b[^>]*>")
_RESIDUAL_RE = re.compile(r"<[mkgucft]\d+")


def _preview(s: str) -> str:
    """원문 앞 12자 미리보기 — 따옴표·개행·꺾쇠 제거(플레이스홀더 문법 보호)."""
    s = re.sub(r"[\"'\n\r<>]", "", s)
    return s.strip()[:12]


def mask(text: str) -> tuple[str, dict[str, str]]:
    """비언어 토큰을 플레이스홀더로 치환. (masked, {placeholder_id → 원문}) 반환."""
    mapping: dict[str, str] = {}
    counter = [0]

    def _repl(m: re.Match) -> str:
        counter[0] += 1
        kind = m.lastgroup[0]  # 그룹명 첫 글자 = 종류 코드
        original = m.group()
        pid = f"{kind}{counter[0]}"
        mapping[pid] = original
        return f'<{pid} v="{_preview(original)}"/>'

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


def should_skip(text: str) -> str:
    """번역 불필요 사유를 반환(빈 문자열이면 번역 대상).

    already-korean: 한글이 비공백 문자의 과반 → 이미 번역됨.
    identifier: 전체가 arXiv id 패턴뿐.
    references: 참고문헌 항목 목록 — 서지 정보는 원문 유지가 정책.
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
        return True

    if _REFUSAL_RE.search(out) and not _REFUSAL_RE.search(src):
        return True

    residual = _PLACEHOLDER_RE.sub(" ", mask(src)[0])
    src_words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", residual)]
    if len(src_words) < 2:
        return False  # 고유명사·짧은 라벨은 원문 그대로 나와도 정상
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
            return True  # 한글이 사실상 없음 / 원문 영단어 그대로 → 거부문·echo
    # 한국어 거부문·한 줄 요약은 한글 비율을 통과하므로 길이비로 잡는다.
    # 하한은 **원문 길이로 나눈다** — 한국어는 짧은 명사구일수록 압축이 극단적이다
    # ('Writing the introduction'→'서론 쓰기' 0.208). 실측 165쌍 분포:
    #   len(src)<=80  min 0.208 / p5 0.227 / p10 0.296
    #   len(src)>80   min 0.444 / p10 0.474
    # 하한을 각 구간 분포 밖(0.12 / 0.25)에 둬 오탐 0을 확보한다.
    lo = 0.12 if len(src) <= 80 else 0.25
    hi = 4.0 if mapping else 3.0  # 수식·표 유닛은 상한 완화
    return not (lo * len(src) <= len(out) <= hi * len(src))
