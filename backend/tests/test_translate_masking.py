"""마스킹 — 토큰 왕복·관용 복원·검증·should_skip."""

import json
import re
import pathlib

import pytest

from app.translate.masking import mask, sanitize_translation, should_skip, unmask


def _roundtrip(text: str) -> tuple[str, list, list]:
    masked, mapping = mask(text)
    return unmask(masked, mapping)


@pytest.mark.parametrize("text", [
    "The energy is $E=mc^2$ here.",                       # 인라인 수식
    "Display $$\\sum_{i=1}^n x_i$$ end.",                 # 디스플레이 수식
    "Inline latex \\( a^2 + b^2 \\) here.",                # LaTeX 인라인 (render.py 실측 형태)
    "Block \\[ E = mc^2 \\] shown.",                       # LaTeX 디스플레이
    "See ```py\nx = 1\ny = 2\n``` block.",                # 펜스 코드
    "Use `inline_code()` please.",                        # 인라인 코드
    "Figure ![cap](images/p0001_0.jpg) shown.",           # 이미지
    "Visit https://example.com/path?a=1 now.",            # URL
    "DOI 10.1145/1234567.890 reference.",                 # DOI
    "Mail me at user.name@example.co.kr today.",          # 이메일
    "Bold <b>text</b> and <br/> tag.",                    # HTML 태그
    "As in [1] and [2, 3] and [4-6].",                    # 인용
    "See Figure 2 and Table 1 and Eq. (5).",              # 참조
])
def test_각_토큰_종류_왕복(text):
    restored, missing, dup = _roundtrip(text)
    assert restored == text
    assert missing == [] and dup == []


def test_수식_혼합_인라인_디스플레이():
    text = "식 $a$와 $$b$$ 혼합"
    masked, mapping = mask(text)
    # 인라인·디스플레이 각각 다른 플레이스홀더로 마스킹됨 (v 미리보기엔 원문 조각이 남음)
    assert set(mapping.values()) == {"$a$", "$$b$$"}
    assert len(mapping) == 2
    assert "$a$와" not in masked  # 구조상 원문 수식이 본문에서 치환됨
    restored, missing, dup = unmask(masked, mapping)
    assert restored == text and not missing and not dup


def test_플레이스홀더_전역_번호():
    masked, mapping = mask("$x$ and Figure 1 and [2]")
    # 종류를 가로질러 1,2,3 전역 증가
    assert set(mapping) == {"m1", "f2", "c3"}


def test_관용_복원_슬래시_속성_공백():
    _, mapping = mask("value $x$ end")
    pid = next(iter(mapping))
    for variant in (f"값 <{pid}> 끝", f"값 < {pid} /> 끝", f"값 <{pid} v=\"바뀜\"/> 끝"):
        restored, missing, dup = unmask(variant, mapping)
        assert restored == "값 $x$ 끝"
        assert not missing and not dup


def test_missing_보고():
    _, mapping = mask("$x$ and $y$")
    restored, missing, dup = unmask("플레이스홀더 없는 번역", mapping)
    assert set(missing) == {"m1", "m2"} and dup == []


def test_dup_보고_전부_복원():
    masked, mapping = mask("only $x$")
    pid = next(iter(mapping))
    restored, missing, dup = unmask(f"<{pid}/> 그리고 <{pid}/>", mapping)
    assert restored == "$x$ 그리고 $x$"  # 전부 복원
    assert pid in dup and missing == []


def test_잔여_플레이스홀더_dup():
    _, mapping = mask("text $x$ here")
    # LLM이 만들어낸 존재하지 않는 플레이스홀더가 남음
    restored, missing, dup = unmask("복원됨 $x$ 그런데 <m9/> 잔여", mapping)
    assert any("m9" in d for d in dup)


def test_sanitize_수식딜리미터_치환_카운트():
    clean, n = sanitize_translation("번역 \\(a\\) 그리고 \\[b\\] 끝")
    assert clean == "번역 (a) 그리고 [b] 끝"
    assert n == 4  # \( \) \[ \]


def test_sanitize_이중달러_페이지마커_제거():
    clean, n = sanitize_translation("앞 $$수식$$ 여기 <PAGE> 뒤")
    assert "$$" not in clean and "<PAGE>" not in clean
    assert "수식" in clean and "앞" in clean and "뒤" in clean
    assert n == 3  # $$ 2건 + <PAGE> 1건


def test_sanitize_치환없으면_무변화_0건():
    text = "일반 번역문에는 $단일$ 달러와 \\sum 같은 게 있어도 무관하다."
    clean, n = sanitize_translation(text)
    assert clean == text and n == 0  # 단일 $, \sum(\s)은 대상 아님


def test_sanitize_플레이스홀더_v속성_치환돼도_복원_무해():
    # v 미리보기 안에 \( \)가 들어간 실제 마스킹 출력을 그대로 새니타이즈
    masked, mapping = mask("see \\( a \\) end")
    clean, n = sanitize_translation(masked)
    assert n == 0  # v 속성 안의 \( \)는 치환은 되지만 카운트하지 않는다 (리포트 부풀림 방지)
    restored, missing, dup = unmask(clean, mapping)
    assert restored == "see \\( a \\) end"  # 원문 그대로 복원 (id 매칭이라 v 치환 무해)
    assert not missing and not dup


def test_sanitize_strips_dollar_signs_invented_around_math_placeholders():
    """소형 모델이 수식 플레이스홀더를 `$…$`로 다시 감싸 번역 PDF에 '$〈 y, x̃ 〉$를'이 찍혔다
    (P4 실앱). 원문 `$x$`는 달러까지 한 플레이스홀더라 양옆 달러는 언제나 지어낸 것이다."""
    masked, mapping = mask("We define \\( \\langle y, x \\rangle \\) as the product.")
    raw = masked.replace("We define ", "우리는 $").replace(" as the product.", "$를 내적으로 정의한다.")
    clean, n = sanitize_translation(raw)
    assert n == 1
    restored, missing, dup = unmask(clean, mapping)
    assert restored == "우리는 \\( \\langle y, x \\rangle \\)를 내적으로 정의한다."
    assert not missing and not dup
    assert sanitize_translation("값은 $ <m2/> $이다") == ("값은 <m2/>이다", 1)


def test_sanitize_keeps_currency_and_one_sided_dollars_next_to_placeholders():
    for text in (
        "가격은 $5 <m1/> $10이다",          # 통화 — 달러 바로 뒤가 플레이스홀더가 아니다
        "<m1/>$만 남았다",                   # 한쪽만
        "$<c1/>$ 코드 플레이스홀더는 그대로",  # 수식(m)이 아닌 플레이스홀더
    ):
        assert sanitize_translation(text) == (text, 0), text


def test_sanitize_strips_one_sided_dollars_the_source_does_not_have():
    """한쪽에만 붙은 '$<m…/>'·'<m…/>$'도 모델이 지어낸 것이다 — 원문(마스킹본)의 같은 자리에
    '$'가 없을 때만 지운다. 마크다운 리더는 짝 없는 '$'를 뒤쪽의 다른 '$'와 묶어 산문을 수식으로
    조판할 수 있다(delta-pdf-translate-5)."""
    masked, mapping = mask("Then \\( \\tilde{x}_{mse} \\) is the estimate of \\( x \\).")
    ids = [m.group(1) for m in re.finditer(r"<(m\d+)", masked)]
    raw = f"그러면 $<{ids[0]}/>에 대한 <{ids[1]}/>$의 추정이다."
    clean, n = sanitize_translation(raw, masked)
    assert clean == f"그러면 <{ids[0]}/>에 대한 <{ids[1]}/>의 추정이다." and n == 2
    # 원문에 원래 있던 한쪽 달러(산문 '$' 바로 옆의 수식)는 그대로 둔다
    priced, _ = mask("It costs $\\( 5 \\) per unit.")
    pid = re.search(r"<(m\d+)", priced).group(1)
    assert sanitize_translation(f"단위당 $<{pid}/>이다.", priced) == (f"단위당 $<{pid}/>이다.", 0)
    # 통화는 건드리지 않는다
    assert sanitize_translation(f"<{ids[0]}/> $10이다", masked) == (f"<{ids[0]}/> $10이다", 0)


def test_cached_translation_drops_dollars_around_restored_math():
    """원출력 단계 정리가 생기기 전에 캐시된 번역은 복원된 수식 옆에 '$'가 남아 있다 — 재번역·
    재빌드해도 '$〈y, x̃〉$를'이 찍혔다. 캐시 적중 때 복원문에서 같은 정리를 한다."""
    from app.translate.masking import strip_math_dollars

    src = r"We define \( \langle y, \tilde{x} \rangle \) and \( \tilde{x}_{\mathrm{mse}} \) here."
    _masked, mapping = mask(src)
    cached = (r"우리는 $\( \langle y, \tilde{x} \rangle \)$를 정의하고 "
              r"$\( \tilde{x}_{\mathrm{mse}} \)에 대해 쓴다. 가격은 $5다.")
    assert strip_math_dollars(cached, mapping, src) == (
        r"우리는 \( \langle y, \tilde{x} \rangle \)를 정의하고 "
        r"\( \tilde{x}_{\mathrm{mse}} \)에 대해 쓴다. 가격은 $5다."
    )
    priced = r"It costs $\( 5 \) per unit."
    assert strip_math_dollars(r"단위당 $\( 5 \)이다.", mask(priced)[1], priced) == (
        r"단위당 $\( 5 \)이다."                                   # 원문 표기는 그대로
    )


def test_should_skip_수식뿐():
    assert should_skip("$E = mc^2$") == "non-linguistic"
    assert should_skip("[1, 2, 3]") == "non-linguistic"


def test_should_skip_식별자():
    assert should_skip("arXiv:2504.19874") == "identifier"
    assert should_skip("arXiv:1908.07836v1") == "identifier"


def test_should_skip_한국어_과반():
    assert should_skip("이것은 이미 한국어로 된 문장이다.") == "already-korean"


def test_should_skip_번역대상():
    assert should_skip("This is a normal English sentence about models.") == ""
    # 수식 섞였어도 자연어가 있으면 번역 대상
    assert should_skip("The loss is $L = \\sum x$ over samples.") == ""


def test_should_skip_참고문헌_모양감지():
    """헤딩 없는 참고문헌(실측 2504.19874v1: 헤딩이 markdown으로 안 뽑힘)도
    [n] 항목 모양으로 스킵한다 — md·layout 유닛 공통 경로."""
    multi = "[41] Liu, Z., and Hu, X. Kivi: KV cache quantization. 2024.\n[42] Kim, J. Paper title. arXiv:2401.00001."
    assert should_skip(multi) == "references"
    single = "[26] Gersho, A. On the structure of vector quantizers. IEEE Trans., 28(2):157-166, 1982."
    assert should_skip(single) == "references"
    # 본문이 인용으로 시작하는 산문은 스킵하면 안 됨 (서지 증거 없음)
    prose = "[1] shows that the method improves accuracy over strong baselines."
    assert should_skip(prose) == ""


def test_통화_달러는_인라인수식으로_오인하지_않는다():
    """문장 안의 통화 $ 두 개가 수식으로 잡히면 그 사이 산문이 통째로 마스킹돼
    번역되지 않고 영어로 남는다 — 여는/닫는 $ 공백 금지 + 닫는 $ 뒤 숫자 금지로 차단."""
    text = "The device costs $5 and the matching case costs $7 in total."
    masked, mapping = mask(text)
    assert mapping == {} and masked == text
    assert should_skip(text) == ""                 # 여전히 번역 대상

    # 붙어 있는 범위 표기도 마찬가지 ("$5-$10")
    _, span_mapping = mask("Prices range from $5-$10 per unit at this store.")
    assert span_mapping == {}

    # 실제 인라인 수식은 그대로 보호된다
    _, math_mapping = mask("The loss $L = \\sum_i x_i$ is minimized here.")
    assert list(math_mapping.values()) == ["$L = \\sum_i x_i$"]


def test_looks_untranslated_거부문_요약_영문echo_감지():
    """출력 측 검증 — should_skip과 대칭인 최소 게이트."""
    from app.translate.masking import looks_untranslated

    src = "The accuracy improved on every benchmark dataset that we evaluated."
    assert looks_untranslated(src, "I cannot translate this text.", {}) is True
    assert looks_untranslated(src, src, {}) is True                 # 영문 echo
    assert looks_untranslated(src, "죄송합니다, 번역할 수 없습니다.", {}) is True
    assert looks_untranslated(
        src, "우리가 평가한 모든 벤치마크 데이터셋에서 정확도가 향상되었다.", {},
    ) is False
    # 영단어 2개 미만(고유명사·짧은 라벨)은 원문 유지가 정상 → 통과
    assert looks_untranslated("Adam", "Adam", {}) is False
    # 플레이스홀더가 있는 유닛은 길이비를 완화한다
    masked, mapping = mask("See $E = mc^2$ and https://example.com/a/very/long/path here.")
    assert mapping
    assert looks_untranslated(
        "See $E = mc^2$ and https://example.com/a/very/long/path here.",
        "$E = mc^2$와 https://example.com/a/very/long/path 를 참고하라.",
        mapping,
    ) is False


def test_looks_untranslated_짧은_유닛_거부문도_잡는다():
    """영단어 2개 미만 면제가 거부문까지 통과시키면 안 된다.

    실 PDF 하네스(scripts/verify_e2e.py)의 결함 주입에서 'Abstract' 같은 한 단어
    제목이 거부문으로 통째로 대체돼도 kept_original에 남지 않고 result.ko.md로
    유출되던 경로의 회귀 테스트다.
    """
    from app.translate.masking import looks_untranslated

    for short_src in ("Abstract", "Introduction", "Adam", "1"):
        assert looks_untranslated(short_src, "I cannot translate this text.", {}) is True
        assert looks_untranslated(short_src, "죄송합니다, 번역할 수 없습니다.", {}) is True
        # 면제의 본래 목적(원문 그대로 되돌아오는 echo 허용)은 유지돼야 한다.
        assert looks_untranslated(short_src, short_src, {}) is False

    # 짧은 원문의 정상 번역은 통과한다.
    assert looks_untranslated("Abstract", "초록", {}) is False
    assert looks_untranslated("Introduction", "서론", {}) is False
    assert looks_untranslated("GPU", "그래픽 처리 장치", {}) is False


def test_looks_untranslated_거부문을_다루는_원문은_오탐하지_않는다():
    """원문 자체가 거부 표현을 담고 있으면 그 번역도 담는 것이 정상이다."""
    from app.translate.masking import looks_untranslated

    src = "The model responds with 'I cannot translate this text.' when the policy filter fires."
    assert looks_untranslated(
        src, "정책 필터가 작동하면 모델은 'I cannot translate this text.'라고 응답한다.", {},
    ) is False


# ── 실번역 픽스처 회귀 — "오탐 0" 계약 ─────────────────────────────────────
_PAIRS_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "real_translation_pairs.json"


def _real_pairs() -> list[dict]:
    return json.loads(_PAIRS_FIXTURE.read_text(encoding="utf-8"))


def test_looks_untranslated_실번역_픽스처_오탐_0():
    """실 LLM 번역 쌍 168건 전부가 게이트를 통과해야 한다(오탐 = 번역 손실).

    픽스처는 data/jobs의 layout.json ↔ layout.ko.json 블록 쌍에서 뽑은 실번역이다
    (should_skip 대상·원문 보존분·원문 영단어 잔존율 0.9 이상인 echo 등가 1건 제외).
    오탐은 래더 소진 후 "kept"(원문 보존)로 귀결되므로 사용자 눈에는 "번역 안 된
    문단"으로 보인다 — 따라서 이 목록은 전건 False여야 한다.
    """
    from app.translate.masking import looks_untranslated, mask

    pairs = _real_pairs()
    assert len(pairs) >= 150, "픽스처가 축소됐다 — 계약의 통계적 의미가 사라진다"
    bad = [
        (p["src"][:60], p["out"][:40], round(len(p["out"]) / max(1, len(p["src"])), 3))
        for p in pairs
        if looks_untranslated(p["src"], p["out"], mask(p["src"])[1])
    ]
    assert bad == [], f"실번역 오탐 {len(bad)}건: {bad[:5]}"


@pytest.mark.parametrize(("src", "out"), [
    ("Writing the introduction", "서론 쓰기"),          # 길이비 0.208 (실측 최소)
    ("Experimental Results", "실험 결과"),              # 0.250
    ("Topic sentence", "주제문"),                       # 0.214
    ("The roadmap", "로드맵"),                          # 0.273
    ("Writing the conclusion", "결론 쓰기"),            # 0.227
    ("• An overview of a debate, the positions on both sides",
     "• 논쟁의 개요와 양측의 입장"),                     # 0.296
    ("2. THE THREE-PASS APPROACH", "2. 3회독 접근법"),  # 0.385
])
def test_looks_untranslated_짧은_제목의_압축_번역은_통과한다(src, out):
    """한국어는 짧은 명사구에서 영어 대비 0.2배까지 압축된다 — 하한 0.3은 오탐이었다."""
    from app.translate.masking import looks_untranslated

    assert looks_untranslated(src, out, {}) is False


def test_looks_untranslated_긴_원문은_하한이_그대로_엄격하다():
    """길이비 하한 완화는 짧은 유닛 한정 — 긴 산문의 한 줄 요약은 계속 잡는다."""
    from app.translate.masking import looks_untranslated

    src = (
        "Researchers spend a great deal of time reading research papers. However, this "
        "skill is rarely taught, leading to much wasted effort. This article outlines a "
        "practical and efficient three-pass method for reading research papers."
    )
    assert len(src) > 80
    assert looks_untranslated(src, "논문 읽기 방법에 대한 글이다.", {}) is True   # 한 줄 요약
    # 같은 길이비(0.11)라도 원문이 짧으면 통과 대상이 아님을 대비로 확인
    assert looks_untranslated("Writing the introduction", "서", {}) is True


def test_refusal_re_번역_없이_같은_정상_표현을_오탐하지_않는다():
    """`번역…없` 패턴이 정상 산문("번역 없이", "번역이 없는")을 거부문으로 판정했다.

    원문이 영어라 `_REFUSAL_RE.search(src)` 가드는 구조적으로 무력하므로 패턴 자체를
    좁혔다 — "수 없"/"불가"를 반드시 요구한다.
    """
    from app.translate.masking import looks_untranslated

    src = (
        "The pipeline forwards the paragraph verbatim when no translation is available, "
        "so downstream consumers still receive the original English text."
    )
    for ok in (
        "번역 없이 원문을 그대로 전달하므로 하위 소비자는 영어 원문을 받는다.",
        "번역이 없는 문단은 원문 그대로 전달되어 하위 소비자가 영어 원문을 받는다.",
        "번역이 필요 없는 문단은 원문 그대로 전달되며 하위 소비자가 영어 원문을 받는다.",
    ):
        assert looks_untranslated(src, ok, {}) is False, ok

    # 진짜 거부문은 여전히 잡는다.
    for refusal in (
        "죄송합니다, 번역할 수 없습니다.",
        "이 문서는 번역을 제공할 수 없습니다.",
        "요청하신 번역해 드릴 수 없습니다.",
        "해당 텍스트는 번역이 불가능합니다.",
        "I cannot translate this text.",
        "As an AI language model, I'm unable to help with that request.",
    ):
        assert looks_untranslated(src, refusal, {}) is True, refusal


def test_looks_untranslated_고유명사_많은_번역과_echo를_구분한다():
    """한글 비율 하한만으로는 사사문·저자 블록이 오탐된다 — 원문 영단어 잔존율을 함께 본다."""
    from app.translate.masking import looks_untranslated

    src = (
        "This work was supported by grants from the National Science and Engineering "
        "Council of Canada and by a gift from the Cisco University Research Program."
    )
    # 실측 한글 비율 0.127 — 고유명사가 대부분인 정상 번역이다.
    ok = (
        "이 연구는 National Science and Engineering Council of Canada의 지원과 "
        "Cisco University Research Program의 기부로 수행되었다."
    )
    assert looks_untranslated(src, ok, {}) is False
    # 원문 영단어가 그대로 다 남는 echo는 계속 잡는다.
    assert looks_untranslated(src, src, {}) is True
    assert looks_untranslated(src, src + "  \n", {}) is True


def test_스캐폴딩이_섞인_출력은_번역으로_채택하지_않는다():
    """프롬프트가 산출물에 새는 경로 차단.

    fault=echo 실행에서 repair 프롬프트가 result.ko.md에 20벌 박힌 실측 사례의
    회귀 테스트다. 원문 길이·언어와 무관하므로 짧은 원문 면제보다 앞서야 한다.
    """
    from app.translate.masking import looks_untranslated

    src = "The accuracy improved on every benchmark dataset that we evaluated."
    for leaked in (
        "[번역할 원문]\n무언가",
        "[수정할 번역문]\n무언가",
        "[원문 — 아래 꺾쇠 태그가 정답이다]\n무언가",
        "다음 꺾쇠 태그가 누락되었거나 중복되었다: <m1/>",
        "[용어집 — 반드시 이 역어 사용]\n- bias → 편향",
    ):
        assert looks_untranslated(src, leaked, {}) is True
        # 짧은 원문(영단어 2개 미만) 면제 경로도 뚫리면 안 된다
        assert looks_untranslated("Abstract", leaked, {}) is True

    # 원문 자체가 그 문구를 담고 있으면(프롬프트를 다루는 문서) 정상이다
    meta = "The template starts with [번역할 원문] and ends there."
    assert looks_untranslated(meta, "템플릿은 [번역할 원문]으로 시작한다.", {}) is False


def test_스캐폴딩_마커가_실제_프롬프트와_일치한다():
    """masking._SCAFFOLD_RE는 prompts.py 문구를 복사한 것이다 — 드리프트를 잡는다.

    import 결합 대신 계약만 고정한다. prompts.py가 문구를 바꾸면 여기서 실패해
    게이트가 조용히 무력화되는 것을 막는다.
    """
    from app.translate.masking import _SCAFFOLD_RE
    from app.translate.prompts import build_repair_prompt, build_unit_prompt

    unit = build_unit_prompt(
        "SRC", [("a", "b")], [("c", "d")],
        context_tail="CTX", keep_terms=["K"], unit_kind="title",
    )
    repair = build_repair_prompt("SRC", "OUT", ["<m1/>"])
    for prompt in (unit, repair):
        headers = [ln for ln in prompt.splitlines() if ln.startswith("[")]
        assert headers, "프롬프트에 대괄호 헤더가 없다 — 마커 추출 전제가 깨졌다"
        for header in headers:
            assert _SCAFFOLD_RE.search(header), (
                f"프롬프트 헤더 {header!r}가 _SCAFFOLD_RE에 없다 — 마커를 갱신하라"
            )


def test_untranslated_reason이_규칙별_사유를_돌려준다():
    """게이트 오탐 관측의 전제 — 어느 규칙이 걸었는지 구분할 수 있어야 한다."""
    from app.translate.masking import looks_untranslated, untranslated_reason

    src = "The accuracy improved on the benchmark dataset that we evaluated."
    cases = {
        "[번역할 원문]\n무언가": "scaffold",
        "죄송합니다, 이 텍스트는 번역할 수 없습니다.": "refusal",
        src: "hangul-ratio",
    }
    for out, expected in cases.items():
        assert untranslated_reason(src, out, {}) == expected, out
        assert looks_untranslated(src, out, {}) is True

    # 한 줄 요약은 한글 비율을 통과하므로 길이비가 잡는다 (원문 80자 초과 → 하한 0.25)
    long_src = (
        "The accuracy improved on every benchmark dataset that we evaluated, and the "
        "ablation study confirms that each component contributes to the final result."
    )
    assert untranslated_reason(long_src, "정확도가 올랐다.", {}) == "length-ratio"

    ok = "우리가 평가한 벤치마크 데이터셋에서 정확도가 향상되었다."
    assert untranslated_reason(src, ok, {}) == ""
    assert looks_untranslated(src, ok, {}) is False


_ECHO_SRC = (
    r"By pigeonhole principle there exist an index \( j \in [d] \) such that "
    r"\( \mathbb{E}\left[\left|\langle \boldsymbol{e}_j, \boldsymbol{x} \rangle\right|^2\right] "
    r"\geq \frac{1}{d} \cdot \frac{1}{4^b} \), which completes the proof."
)


def test_한국어_메타_문장으로_감싼_원문_echo는_거부된다():
    """실서버(로컬 MLX 0.8B) 실측 — 원문을 통째로 품고 앞뒤만 한국어인 출력이 번역 PDF에 실렸다.

    한글 비율(0.15 이상)·길이비(2배 남짓)를 모두 통과해 게이트가 놓쳤다. 진짜 번역은 원문
    문장을 통째로 옮겨 적지 않는다(실번역 쌍의 최장 연속 원문 단어열은 8단어 이상 원문에서 최대 39%).
    """
    from app.translate.masking import untranslated_reason

    masked, mapping = mask(_ECHO_SRC)
    meta = "이제부터는 모든 문맥을 고려하여 번역하지 않습니다."
    out = f"{meta}\n\n문서:\n{_ECHO_SRC}\n\n번역:\n{meta}"
    assert untranslated_reason(_ECHO_SRC, out, mapping) == "echo"
    # 앞에 번역을 두고 원문을 덧붙인 이중 출력도 같은 echo다
    both = "비둘기집 원리에 의해 다음을 만족하는 인덱스가 존재하며, 이로써 증명이 끝난다. " + _ECHO_SRC
    assert untranslated_reason(_ECHO_SRC, both, mapping) == "echo"


@pytest.mark.parametrize(("src", "out"), [
    # 긴 고유명사 덩어리는 그대로 남는 것이 정상 — 원문의 57%
    ("This work was supported by the National Science and Engineering Research "
     "Council of Canada.",
     "이 연구는 National Science and Engineering Research Council of Canada의 지원을 받았다."),
    # 8단어 미만 원문은 이 규칙 밖(고유명사·짧은 라벨 echo 면제)
    ("Simone A. Fried, TF Spring 2021", "Simone A. Fried, TF 2021년 봄"),
])
def test_원문_일부가_남는_정상_번역은_echo가_아니다(src, out):
    from app.translate.masking import untranslated_reason

    assert untranslated_reason(src, out, mask(src)[1]) == ""


_KEYWORDS_SRC = (
    "Keywords: large language models, retrieval augmented generation, long context, "
    "instruction tuning, data synthesis"
)


def test_역어_원어_병기로_번역한_키워드는_echo가_아니다():
    """프롬프트 규칙 4의 '역어(원어)' 병기로 Keywords·Index Terms를 옮기면 원어가 원문 순서대로
    모두 남는다. 영단어만 뽑아 연속을 보던 때는 사이에 낀 한글 역어가 연속을 끊지 못해 원문
    단어 100%가 한 덩어리로 보여 echo로 거부됐고, 문장 경계가 없어 분할도 못 하고 영어
    원문으로 남았다(delta-pdf-translate-4). 한글 어절은 연속을 끊는다."""
    from app.translate.masking import untranslated_reason

    out = (
        "키워드: 대규모 언어 모델(large language models), 검색 증강 생성(retrieval augmented "
        "generation), 긴 문맥(long context), 지시 튜닝(instruction tuning), "
        "데이터 합성(data synthesis)"
    )
    assert untranslated_reason(_KEYWORDS_SRC, out, mask(_KEYWORDS_SRC)[1]) == ""
    # 한국어 메타 문장 사이에 원문을 통째로 넣은 진짜 echo는 그대로 잡는다
    echo = f"다음 키워드는 번역하지 않습니다. {_KEYWORDS_SRC} 이상입니다."
    assert untranslated_reason(_KEYWORDS_SRC, echo, mask(_KEYWORDS_SRC)[1]) == "echo"


# ── 축퇴(반복 루프) 게이트 (probe:MLX-02, mlx-integration-4) ─────────────────

_LOOP_SRC = (
    "Unlike prior work, our method does not require any additional training data, and it "
    "removes the unnecessary regions before the attention computation is performed."
)


def test_반복_루프_출력은_repetition으로_거부된다():
    """'…영역을만만만만…'처럼 잘린 루프가 길이비 하한만 넘으면 채택·캐시되던 경로."""
    from app.translate.masking import untranslated_reason

    loop = "선행 연구와 달리 제안 방법은 추가 학습 데이터가 필요 없으며 불필요한 영역을" + "만" * 60
    assert 0.25 * len(_LOOP_SRC) <= len(loop) <= 3 * len(_LOOP_SRC)  # 길이비로는 못 잡는다
    assert untranslated_reason(_LOOP_SRC, loop, {}) == "repetition"

    phrase = "제안 방법은 영역을 제거하였다. " + "그리고 영역을 제거하였다. " * 9
    assert untranslated_reason(_LOOP_SRC, phrase, {}) == "repetition"

    # 짧은 유닛의 글자 하나 루프 — 80자 기준에 못 미쳐도 정상 문장에는 없다
    assert untranslated_reason("Results on the test set", "테스트 결과" + "과" * 30, {}) == (
        "repetition"
    )


def test_원문에도_있는_반복과_구분선은_축퇴가_아니다():
    from app.translate.masking import is_degenerate_repetition, longest_repetition

    row = "| 0 " * 30 + "|"
    assert longest_repetition(row) >= 80
    assert not is_degenerate_repetition(row, row)            # 원문 표 행 그대로
    assert longest_repetition("목차" + "." * 200 + "1") == 0  # 말줄임 리더
    assert longest_repetition("구분선\n" + "-" * 120) == 0
    assert not is_degenerate_repetition("짧은 반복 하하하하하하하하", "")  # 80자 미만


def test_반복_검사는_복원된_불변_토큰을_세지_않는다():
    """같은 수식이 여러 번 나오는 정상 번역을 루프로 오인하면 안 된다."""
    from app.translate.masking import untranslated_reason

    src = "We repeat the term $x$ " * 12 + "in the proof of the main theorem."
    masked, mapping = mask(src)
    out = "증명에서 " + "항 $x$를 반복하고 " * 12 + "주정리를 보인다."
    assert untranslated_reason(src, out, mapping) == ""


# ── 아주 짧은 유닛 게이트 (Phase-0 replay, translate-llm-11) ──────────────────

@pytest.mark.parametrize(("src", "out", "reason"), [
    ("Abstract", "요약입니다.", "label-sentence"),            # 캔드 응답 — 라벨이 문장으로
    ("Related Work", "관련 연구입니다", "label-sentence"),
    ("arXiv:2504.19874v1  [cs.LG]  28 Apr 2025", "요약입니다.", "number-mismatch"),
    ("Introduction", "모델이 직전 문맥까지 번역했다. " * 6 + "서론", "length-ratio"),
])
def test_짧은_유닛의_캔드_대체는_거부된다(src, out, reason):
    from app.translate.masking import untranslated_reason

    assert untranslated_reason(src, out, mask(src)[1]) == reason


@pytest.mark.parametrize(("src", "out"), [
    ("Abstract", "초록"),
    ("Abstract", "Abstract"),                                      # 고유명사식 echo 면제 유지
    ("Thank you", "감사합니다."),                                  # 합쇼체지만 '-입니다'가 아니다
    ("arXiv:2504.19874v1  [cs.LG]  28 Apr 2025", "arXiv:2504.19874v1 [cs.LG] 2025년 4월 28일"),
    ("In 2019, we collected 10000 samples", "2019년에 10,000개 샘플을 수집하였다"),  # 구분 쉼표
    ("The answer is yes.", "답은 그렇습니다."),                     # 문장 원문은 라벨 규칙 밖
])
def test_짧은_유닛의_정상_번역은_통과한다(src, out):
    from app.translate.masking import untranslated_reason

    assert untranslated_reason(src, out, mask(src)[1]) == ""


# ── unmask 단일 패스·닫는 태그 (translate-llm-9, translate-llm-10) ──────────────

@pytest.mark.parametrize("src", [
    "Loop `for (i=0; i<k1; i++)` until done.",              # 코드 안의 <k1
    "for (i=0; i<k1; i++) we loop `for (i=0; i<k1; i++)` here.",   # 산문에도 <k1
    "The bound $0<t1$ holds for all inputs.",
    "Inline \\( u<u2 \\) appears here.",
    "The value 0<t1 stays in plain prose.",                   # 마스킹 안 된 산문
    "If i<k1 and k1>0 then $x$ wins.",
])
def test_원문에_플레이스홀더_비슷한_문자열이_있어도_완벽한_번역은_통과(src):
    """종전 unmask는 복원된 텍스트를 다시 훑어 영구 placeholder-mismatch였다."""
    masked, mapping = mask(src)
    restored, missing, dup = unmask(masked, mapping, masked)
    assert restored == src and missing == [] and dup == []


def test_원문_코드의_c2_문자열이_빠진_인용을_가리지_않는다():
    masked, mapping = mask("uses `x<c2>y` in [1] here")
    dropped = masked.replace('<c2 v="[1]"/>', "")
    restored, missing, dup = unmask(dropped, mapping, masked)
    assert missing == ["c2"]                       # 종전: missing=[] dup=[] — 인용 조용히 소실
    assert restored == "uses `x<c2>y` in  here"    # 코드는 변조되지 않는다


def test_모델이_만든_쌍_태그는_빈쌍은_접고_내용이_든_쌍은_dup():
    masked, mapping = mask("As shown in Figure 2, the loss $L$ is low.")
    restored, missing, dup = unmask(
        '<f1 v="Figure 2">그림 2</f1>에서 보듯이, 손실 <m2 v="L"></m2>은 낮다.', mapping, masked,
    )
    assert "</f1>" in dup and missing == []        # 내용이 든 쌍 → repair로
    assert "</m2>" not in restored                 # 빈 쌍은 자기 닫힘으로 접힘
    restored2, missing2, dup2 = unmask(
        "<f1 v='Figure 2'/>에서, 손실 <m2 v=L/>은 낮다.", mapping, masked,
    )
    assert restored2 == "Figure 2에서, 손실 $L$은 낮다." and not missing2 and not dup2


# ── HTML 태그 마스킹 범위 (translate-llm-14) ────────────────────────────────

@pytest.mark.parametrize("text", [
    "If x<y then the algorithm stops early. Otherwise we update the mapping f: A -> B.",
    "The ordering a<b holds for every pair, and by symmetry observe that c>d, as required.",
])
def test_부등호_사이_산문은_태그로_삼키지_않는다(text):
    masked, mapping = mask(text)
    assert not any(k.startswith("t") for k in mapping)
    assert "algorithm" in masked or "symmetry" in masked


@pytest.mark.parametrize(("text", "tags"), [
    ('<table border="1"><tr><td colspan=2>A</td></tr></table>',
     ['<table border="1">', "<tr>", "<td colspan=2>", "</td>", "</tr>", "</table>"]),
    ("Bold <b>text</b> and <br/> tag.", ["<b>", "</b>", "<br/>"]),
    ('A <span style="color: red">red</span> word.', ['<span style="color: red">', "</span>"]),
    ("Tokens <s> and <unk> and <think> stay.", ["<s>", "<unk>", "<think>"]),
])
def test_진짜_태그는_계속_보호한다(text, tags):
    _masked, mapping = mask(text)
    assert [v for k, v in mapping.items() if k.startswith("t")] == tags


# ── 거부문 오탐 — MT·NLP 논문 (translate-llm-8) ─────────────────────────────

@pytest.mark.parametrize(("src", "out"), [
    ("Some idioms cannot be translated literally, so we paraphrase them.",
     "일부 관용구는 문자 그대로 번역할 수 없으므로 의역하였다."),
    ("Translation of these rare named entities is not possible without context.",
     "문맥 없이는 이러한 희귀 개체명의 번역이 불가능하다."),
    ("Such puns are impossible to translate into other languages.",
     "이러한 말장난은 다른 언어로 번역할 수 없다."),
])
def test_번역_불가능성을_다루는_정상_번역은_거부문이_아니다(src, out):
    from app.translate.masking import untranslated_reason

    assert untranslated_reason(src, out, mask(src)[1]) == ""


def test_번역을_다루는_원문이어도_사과_거부문은_잡는다():
    from app.translate.masking import untranslated_reason

    src = "Some idioms cannot be translated literally, so we paraphrase them."
    assert untranslated_reason(src, "죄송합니다, 번역할 수 없습니다.", {}) == "refusal"
    plain = "The accuracy improved on the benchmark dataset that we evaluated."
    assert untranslated_reason(plain, "이 텍스트는 번역할 수 없습니다.", {}) == "refusal"


def test_출력_맨앞의_원문_헤더_한_줄만_걷어낸다():
    """실측(mlx_lm, Qwen3.5-0.8B): '[번역할 원문]\\n' + 정상 번역 → 종전 scaffold 거부."""
    from app.translate.masking import untranslated_reason

    src = "The code is available at https://github.com/example/repo and in Table 2."
    masked, mapping = mask(src)
    raw = '[번역할 원문]\n코드는 <u1 v="https://gith"/> 와 <f2 v="Table 2"/> 에서 제공된다.'
    clean, n = sanitize_translation(raw)
    restored, missing, dup = unmask(clean, mapping, masked)
    assert n == 1 and not missing and not dup
    assert untranslated_reason(src, restored, mapping) == ""
    # 다른 스캐폴딩 echo·본문 중간의 헤더는 그대로 남겨 게이트가 거부한다
    echo = "[용어집 — 반드시 이 역어 사용]\n- learning rate → 학습률\n\n[번역할 원문]\n번역"
    assert sanitize_translation(echo) == (echo, 0)
    assert sanitize_translation("앞 [번역할 원문]\n뒤")[1] == 0
