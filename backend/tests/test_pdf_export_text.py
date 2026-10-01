"""번역 PDF 평문화 회귀 — 태그·마크다운·LaTeX 정리가 내용을 지우거나 깨뜨리지 않는다.

번역 블록은 원문이 리댁션된 자리에 평문으로 다시 찍힌다. 정리 단계가 문장을
삼키거나 숫자를 바꾸면 독자는 원문과 대조할 수 없어 손실을 알아채지 못한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.layout import HTML_TAG_RE, estimate_font_size_cqw
from app.pipeline.pdf_export import (
    _plain_text,
    _portable_text_for_font,
    _resolve_font,
    build_translated_pdf,
)
from app.pipeline.pdf_export.subset import drawable_charset
from app.pipeline.pdf_export.text import strip_markdown


# ── HTML 태그 화이트리스트 ─────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "p < 0.05 이면 유의하며 n > 30 인 경우에만 적용한다.",
    "x<y and z>w",
    "a<b and c>d",
    "a <= b and c >= d",
    "Use the <think> token",
    "끝에 <eos> 토큰을 붙인다",
    "<threshold> 값을 넘으면",
])
def test_plain_text_keeps_prose_between_angle_brackets(text):
    """`<[^>]+>`가 부등호 사이의 문장·특수 토큰을 통째로 지우던 회귀."""
    assert _plain_text(text) == text


def test_plain_text_keeps_model_special_tokens_intact():
    """`<|im_start|>`는 내용이다 — 지워지지도, 첨자로 바뀌지도 않는다."""
    assert _plain_text("예를 들어 <|im_start|>user 형식") == "예를 들어 <|im_start|>user 형식"
    assert _plain_text("D(1) <|doc_sep|> D(2)") == "D(1) <|doc_sep|> D(2)"


def test_plain_text_still_strips_real_html_tags():
    assert _plain_text("<b>굵게</b>  두  칸") == "굵게 두 칸"
    assert _plain_text('<td colspan="2">셀</td><br/>다음') == "셀 다음"
    assert _plain_text('<span style="color:red">빨강</span>') == "빨강"
    # 위/아래첨자 태그는 TeX 첨자와 같은 평문으로 낮춘다.
    assert _plain_text("x<sup>2</sup> 와 H<sub>2</sub>O") == "x² 와 H(2)O"


def test_layout_font_estimate_counts_text_between_angle_brackets():
    """폰트 크기 추정도 같은 화이트리스트를 쓴다 — 부등호 사이 글자를 빼면 과대추정한다."""
    bbox = (100, 100, 900, 160)
    with_inequality = estimate_font_size_cqw(bbox, "a" * 40 + " < " + "b" * 40 + " > c", 1.4)
    plain = estimate_font_size_cqw(bbox, "a" * 40 + " x " + "b" * 40 + " x c", 1.4)
    assert with_inequality == pytest.approx(plain)
    assert HTML_TAG_RE.sub(" ", "<table><tr><td>a</td></tr></table>").split() == ["a"]


# ── LaTeX 구조 명령 ───────────────────────────────────────────────────────
@pytest.mark.parametrize(("raw", "expected"), [
    (r"\( \frac{1}{N} \)", "1/N"),
    (r"\( \frac{a+b}{2} \)", "(a+b)/2"),
    (r"\( \frac12 \)", "1/2"),
    (r"\( \frac{1}{\sqrt{d_k}} \)", "1/√d(k)"),
    (r"\( \sqrt{d_k} \)", "√d(k)"),
    (r"\( \sqrt[3]{x+1} \)", "³√(x+1)"),
    (r"\( \mathbb{R}^{d} \)", "ℝ^(d)"),
    (r"\( \boldsymbol{x} \)", "x"),
    (r"\( \bm{W} \)", "W"),
    (r"\( a \cdot b \)", "a · b"),
    (r"\( \ell_2 \)", "ℓ(2)"),
    (r"\( 3\,\mathrm{GB} \)", "3 GB"),
    (r"\( \|x\|_2 \)", "‖x‖(2)"),
    (r"\( \binom{n}{k} \)", "C(n, k)"),
    (r"\( \eta \)", "η"),
    (r"\( W^\top \)", "W^T"),
    (r"\( \operatorname*{arg\,max}_{\theta} \)", "arg max(θ)"),
    (r"\( \underline{m} \)", "m"),
])
def test_plain_text_converts_structural_latex(raw, expected):
    """이름만 남기던 처리('frac1N', 'mathbbR^(d)', '3\\,GB')를 읽을 수 있는 평문으로."""
    assert _plain_text(raw) == expected


def test_plain_text_turns_single_letter_accents_into_combining_marks():
    assert _plain_text(r"\( \hat{y} \)") == "y\u0302"
    assert _plain_text(r"\( \bar{x} \)") == "x\u0304"
    assert _plain_text(r"\( \hat{\theta} \)") == "θ\u0302"
    # 여러 글자 인자에는 악센트를 붙이지 않는다(인자만 남긴다).
    assert _plain_text(r"\( \hat{xy} \)") == "xy"


def test_plain_text_never_leaves_backslashes_for_known_structures():
    converted = _plain_text(
        r"\( \frac{QK^{\top}}{\sqrt{d_k}} \), \( \mathbb{E}[x] \), \( \tilde{x} \)"
    )
    assert "\\" not in converted and "{" not in converted, converted


class _GlyphlessFont:
    def __init__(self, missing: str):
        self._missing = {ord(ch) for ch in missing}

    def has_glyph(self, code: int) -> bool:
        return code not in self._missing


def test_missing_combining_accent_is_dropped_instead_of_tofu(monkeypatch):
    """결합 악센트 글리프가 없는 폰트에서는 빈 네모 대신 기본 글자만 남긴다."""
    monkeypatch.setattr(
        "app.pipeline.pdf_export._metrics_font",
        lambda fontfile, fontname: _GlyphlessFont("\u0302\u0304"),
    )
    assert _portable_text_for_font("y\u0302 와 x\u0304", "/fake/font.ttf") == "y 와 x"
    # 폰트가 결합 글리프를 가지면 그대로 둔다.
    monkeypatch.setattr(
        "app.pipeline.pdf_export._metrics_font",
        lambda fontfile, fontname: _GlyphlessFont(""),
    )
    assert _portable_text_for_font("y\u0302", "/fake/font.ttf") == "y\u0302"


def test_subset_charset_covers_structural_substitutions():
    """구조 변환이 만드는 글자가 서브셋에서 빠지면 조판 결과가 바뀐다(동치 계약)."""
    charset = drawable_charset([{"blocks": [{"content": "가"}]}])
    for char in ("•", "√", "ℝ", "R", "\u0302", "·", "ℓ", "‖", "∑"):
        assert char in charset, char


# ── 마크다운 정리 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "배치 크기는 4*8*16 = 512로 설정한다",
    "*args, **kwargs를 받는다",
    "p*q 와 r*s",
    "a - b 그리고 2*3",
])
def test_emphasis_stripping_never_touches_multiplication_or_identifiers(text):
    """여는 표기 앞에 영숫자가 붙은 `*`는 곱셈·식별자다('4816 = 512로' 회귀)."""
    assert strip_markdown(text) == text


def test_code_spans_and_dunders_survive_emphasis_stripping():
    assert strip_markdown("`__init__` 메서드에서 __call__") == "__init__ 메서드에서 __call__"
    assert strip_markdown("`a*b*c` 계산") == "a*b*c 계산"
    # 실제 LLM 강조 표기는 계속 걷어낸다(닫는 표기 뒤 한글 조사 포함).
    assert strip_markdown("**강조**된 문장") == "강조된 문장"
    assert strip_markdown("**BERT**는 모델이다") == "BERT는 모델이다"
    assert strip_markdown("__중요__ 문장") == "중요 문장"


def test_emphasis_markers_present_in_the_source_are_kept_literally():
    assert strip_markdown("이 *방법*은 좋다", "This *method* is good") == "이 *방법*은 좋다"
    assert strip_markdown("__x__ 는 변수", "the __x__ variable") == "__x__ 는 변수"


def test_list_bullets_survive_when_the_source_block_is_a_list():
    """원문 글머리표 글리프는 블록 소유로 리댁션된다 — 번역문 쪽 표기를 '• '로 남긴다."""
    assert strip_markdown(
        "- 우리는 X를 제안한다\n- 우리는 Y를 보인다", "- We introduce X\n- We show Y",
    ) == "• 우리는 X를 제안한다\n• 우리는 Y를 보인다"
    assert strip_markdown("* 우리는 X를 제안한다", "• We introduce X") == "• 우리는 X를 제안한다"
    # 원문이 목록이 아니면(LLM이 지어낸 표기) 지금처럼 걷어낸다.
    assert strip_markdown("- 항목 하나\n- 항목 둘", "Item one. Item two.") == "항목 하나\n항목 둘"
    assert strip_markdown("- 항목 하나") == "항목 하나"


def test_leading_comparison_is_content_when_the_source_starts_with_it():
    assert strip_markdown("> 0.5 임계값", "> 0.5 threshold") == "> 0.5 임계값"
    assert strip_markdown("> 인용문") == "인용문"


# ── 내보내기 전 구간 ───────────────────────────────────────────────────────
@pytest.fixture
def real_cjk_fontfile() -> str:
    fontfile, _fontname = _resolve_font("")
    if fontfile is None or not Path(fontfile).is_file():
        pytest.skip("no usable file-backed CJK font is installed")
    return fontfile


def test_exported_list_keeps_bullets_and_inequality_sentences(tmp_path, real_cjk_fontfile):
    """번역 PDF에서 목록 글머리표와 부등호 사이 문장이 모두 남는다."""
    import fitz

    job_dir = tmp_path / "text-job"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "• We introduce a new benchmark.", fontsize=10)
    page.insert_text((72, 134), "• We show it is hard.", fontsize=10)
    page.insert_text((72, 200), "The effect holds when p < 0.05 and n > 30 only.", fontsize=10)
    doc.save(job_dir / "source.pdf")
    doc.close()

    def bbox(x0, y0, x1, y1):
        return [x0 / 595 * 999, y0 / 842 * 999, x1 / 595 * 999, y1 / 842 * 999]

    original = [{"page": 1, "width": 595, "height": 842, "blocks": [
        {"type": "list", "bbox": bbox(70, 108, 400, 138),
         "content": "- We introduce a new benchmark.\n- We show it is hard.",
         "fs": 10 / 595 * 100},
        {"type": "text", "bbox": bbox(70, 188, 400, 204),
         "content": "The effect holds when p < 0.05 and n > 30 only.",
         "fs": 10 / 595 * 100},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = (
        "- 우리는 새 벤치마크를 소개한다.\n- 우리는 그것이 어렵다는 것을 보인다."
    )
    translated[0]["blocks"][1]["content"] = "효과는 p < 0.05 이고 n > 30 일 때만 성립한다."
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko", fontfile=real_cjk_fontfile)
    assert result.replaced == 2, result.report()
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
    assert text.count("•") == 2, text
    assert "p < 0.05 이고 n > 30 일 때만" in " ".join(text.split()), text
    assert "We introduce" not in text and "effect holds" not in text, text
