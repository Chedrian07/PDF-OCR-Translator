"""flow 재배치가 남는 원문과 겹치지 않고, 읽기 순서를 바꾸지 않는다.

같은 단의 두 번역 문단 사이에 남는 보존 콘텐츠(OCR이 놓친 줄, 번역이 원문과 같은
줄, 수식)가 끼어 있고 앞 문단 번역이 길어 재배치가 필요한 구성이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from app.pipeline.pdf_export import build_translated_pdf

PAGE_W, PAGE_H = 595.0, 842.0
LEADING = 14.0
X0, X1 = 72.0, 300.0
A_LINES = ["Short paragraph A one line long text here ok", "and its second line ends right here now."]
KEEP_LINE = "KEEPLINE: this full width line must stay readable okay ok"
B_LINES = ["Paragraph B starts here with some words and", "finishes on this second line for the test."]
A_KO = (
    "짧은 문단 A의 번역은 원문보다 훨씬 길어서 원래 상자보다 더 많은 줄을 차지하게 "
    "되며 아래로 자라야 한다. 계속 이어지는 설명 문장이 하나 더 있다."
)
B_KO = "문단 B의 번역 문장입니다. 두 번째 줄도 있습니다."


def _bbox(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0 / PAGE_W * 999, y0 / PAGE_H * 999, x1 / PAGE_W * 999, y1 / PAGE_H * 999]


def _korean_lines(page) -> list[fitz.Rect]:
    return [
        fitz.Rect(line["bbox"])
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        if any("가" <= char <= "힣" for span in line["spans"] for char in span["text"])
    ]


def _line_rect(page, needle: str) -> fitz.Rect:
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if needle in "".join(span["text"] for span in line["spans"]):
                return fitz.Rect(line["bbox"])
    raise AssertionError(f"{needle!r} 줄이 없다")


def _job_with_kept_middle(tmp_path: Path, kept_kind: str, middle: str) -> Path:
    """A(번역) – 남는 한 줄 – B(번역). kept_kind: unowned(레이아웃에 없음)·unchanged."""
    job_dir = tmp_path / f"flow-{kept_kind}"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = 200.0

    def write(text: str) -> None:
        nonlocal y
        page.insert_text((X0, y), text, fontsize=10, fontname="tiro")
        y += LEADING

    a0 = y - 10
    for line in A_LINES:
        write(line)
    a1 = y - LEADING + 4
    k0 = y - 10
    write(middle)
    k1 = y - LEADING + 4
    b0 = y - 10
    for line in B_LINES:
        write(line)
    b1 = y - LEADING + 4
    doc.save(job_dir / "source.pdf")
    doc.close()

    blocks = [{"type": "text", "bbox": _bbox(X0, a0, X1, a1), "content": "\n".join(A_LINES)}]
    if kept_kind == "unchanged":
        blocks.append({"type": "text", "bbox": _bbox(X0, k0, X1, k1), "content": middle})
    blocks.append({"type": "text", "bbox": _bbox(X0, b0, X1, b1), "content": "\n".join(B_LINES)})
    original = [{"page": 1, "width": 1000, "height": 1414, "blocks": blocks}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = A_KO
    translated[0]["blocks"][-1]["content"] = B_KO
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )
    return job_dir


@pytest.mark.parametrize("kept_kind", ["unowned", "unchanged"])
def test_translation_never_overprints_a_kept_full_width_source_line(tmp_path, kept_kind):
    """전폭 보존 줄이 '피할 수 없는 장식'으로 장애물에서 빠져 한국어가 겹쳐 찍히던 회귀.

    그 예외는 코드 상자 테두리 같은 **벡터 도형 띠**용이다. 남는 원문 span에
    적용하면 모듈이 막으려던 '번역·원문 겹침'이 그대로 생긴다.
    """
    job_dir = _job_with_kept_middle(tmp_path, kept_kind, KEEP_LINE)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 2, result.report()
    with fitz.open(result.path) as exported:
        page = exported[0]
        keep = _line_rect(page, "KEEPLINE")
        korean = _korean_lines(page)
        text = page.get_text()

    assert "KEEPLINE: this full width line must stay readable okay ok" in text, text
    overlapping = [line for line in korean if (line & keep).get_area() > 0.5]
    assert not overlapping, (keep, overlapping)


@pytest.mark.parametrize("kept_kind", ["unowned", "unchanged"])
def test_compact_reflow_keeps_reading_order_around_kept_content(tmp_path, kept_kind):
    """앞 문단이 위로 당겨져도 뒤 문단은 사이에 있던 보존 줄을 건너뛰어 올라가지 않는다."""
    job_dir = _job_with_kept_middle(tmp_path, kept_kind, KEEP_LINE)
    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 2, result.report()
    with fitz.open(result.path) as exported:
        page = exported[0]
        keep = _line_rect(page, "KEEPLINE")
        a_first = _line_rect(page, "짧은 문단 A")
        b_first = _line_rect(page, "문단 B의")

    assert a_first.y0 < keep.y0, (a_first, keep)
    assert b_first.y0 >= keep.y1, (b_first, keep)


def test_equation_explanation_stays_after_its_equation(tmp_path):
    """'where x is…' 설명 문단이 그 위의 수식보다 먼저 놓이던 재배치 회귀."""
    job_dir = tmp_path / "flow-equation"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = 200.0
    for line in A_LINES:
        page.insert_text((X0, y), line, fontsize=10, fontname="tiro")
        y += LEADING
    a_rect = fitz.Rect(X0, 190, X1, y - LEADING + 4)
    # 단 폭의 1/3짜리 가운데 수식 — 전폭 예외와 무관하게 장애물로 남는다.
    page.insert_text((150, y), "y = f(x) + b", fontsize=10, fontname="tiro")
    eq_rect = fitz.Rect(148, y - 10, 222, y + 4)
    y += LEADING
    b_top = y - 10
    explanation = ["where x is the input and y is the output of", "the model under consideration in this work."]
    for line in explanation:
        page.insert_text((X0, y), line, fontsize=10, fontname="tiro")
        y += LEADING
    b_rect = fitz.Rect(X0, b_top, X1, y - LEADING + 4)
    doc.save(job_dir / "source.pdf")
    doc.close()

    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": [
        {"type": "text", "bbox": _bbox(*a_rect), "content": "\n".join(A_LINES),
         "fs": 10 / PAGE_W * 100},
        {"type": "equation", "bbox": _bbox(*eq_rect), "content": r"\[ y = f(x) + b \]"},
        {"type": "text", "bbox": _bbox(*b_rect), "content": "\n".join(explanation),
         "fs": 10 / PAGE_W * 100},
    ]}]
    translated = json.loads(json.dumps(original))
    translated[0]["blocks"][0]["content"] = A_KO
    translated[0]["blocks"][2]["content"] = "여기서 x는 입력, y는 모델의 출력이다."
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 2, result.report()
    with fitz.open(result.path) as exported:
        page = exported[0]
        equation = _line_rect(page, "y = f(x) + b")
        explanation_ko = _line_rect(page, "여기서 x는")
        korean = _korean_lines(page)

    assert explanation_ko.y0 >= equation.y1, (explanation_ko, equation)
    assert not [line for line in korean if (line & equation).get_area() > 0.5]


def _listing_job(tmp_path: Path, name: str, *, single_span_second_row: bool) -> Path:
    """두 줄짜리 평탄화 리스팅(2열) + 바로 아래 번역 형제 문단 + 위아래 보존 줄."""
    job_dir = tmp_path / name
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((60, 84), "PRESERVED CAPTION LINE", fontsize=9)
    page.insert_text((60, 100), "Alpha", fontsize=9)
    page.insert_text((200, 100), "Beta", fontsize=9)
    if single_span_second_row:
        # 한 span에 두 OCR 줄이 들어 있어 세그먼트로 되짚을 수 없는 시각 줄.
        page.insert_text((60, 112), "Gamma          Delta", fontsize=9)
    else:
        page.insert_text((60, 112), "Gamma", fontsize=9)
        page.insert_text((200, 112), "Delta", fontsize=9)
    page.insert_text((60, 128), "Sibling paragraph short line.", fontsize=9)
    page.insert_text((60, 160), "PRESERVED FOOTNOTE LINE", fontsize=9)
    doc.save(job_dir / "source.pdf")
    doc.close()
    fs = 9 / PAGE_W * 100
    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": [
        {"type": "ref_text", "bbox": _bbox(55, 74, 430, 88),
         "content": "PRESERVED CAPTION LINE", "fs": fs},
        {"type": "text", "bbox": _bbox(55, 90, 430, 115),
         "content": "Alpha\nBeta\nGamma\nDelta", "fs": fs},
        {"type": "text", "bbox": _bbox(55, 118, 430, 132),
         "content": "Sibling paragraph short line.", "fs": fs},
        {"type": "ref_text", "bbox": _bbox(55, 150, 430, 164),
         "content": "PRESERVED FOOTNOTE LINE", "fs": fs},
    ]}]
    translated = json.loads(json.dumps(original))
    return job_dir, original, translated


def _write(job_dir: Path, original, translated) -> None:
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )


def test_sibling_is_not_pulled_over_rows_a_partial_listing_left(tmp_path):
    """줄 단위로 일부만 들어간 리스팅의 남는 영문 행 위로 뒤 형제 번역이 당겨지던 회귀.

    부분 배치 블록이 pending에서 빠지면서 남는 줄이 장애물에서도 빠졌다.
    """
    job_dir, original, translated = _listing_job(tmp_path, "partial", single_span_second_row=False)
    translated[0]["blocks"][1]["content"] = (
        "알파\n베타\n" + "매우긴감마번역" * 40 + "\n" + "매우긴델타번역" * 40
    )
    translated[0]["blocks"][2]["content"] = (
        "형제 문단의 번역은 원문보다 훨씬 길어서 한 줄에 들어가지 않고 여러 줄이 "
        "필요하며 위로 당겨질 수 있다. " * 3
    ).strip()
    _write(job_dir, original, translated)

    result = build_translated_pdf(job_dir, "ko")
    assert result.kept_reasons.get("listing_line_no_fit") == 2, result.report()
    with fitz.open(result.path) as exported:
        page = exported[0]
        gamma = _line_rect(page, "Gamma")
        delta = _line_rect(page, "Delta")
        sibling = [
            fitz.Rect(line["bbox"])
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if "번역은" in "".join(span["text"] for span in line["spans"])
            or "형제" in "".join(span["text"] for span in line["spans"])
        ]

    assert sibling
    overlapping = [
        line for line in sibling
        if (line & gamma).get_area() > 0.5 or (line & delta).get_area() > 0.5
    ]
    assert not overlapping, (gamma, delta, overlapping)


def test_listing_rows_that_cannot_be_aligned_are_reported_as_kept(tmp_path):
    """세그먼트로 되짚지 못해 원문으로 남는 줄도 보존 사유로 센다(조용한 미번역 금지)."""
    job_dir, original, translated = _listing_job(tmp_path, "unaligned", single_span_second_row=True)
    translated[0]["blocks"][1]["content"] = "알파\n베타\n감마\n델타"
    _write(job_dir, original, translated)

    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(result.path) as exported:
        text = exported[0].get_text()
    # 한 span에 담긴 두 번째 행은 원문으로 남고, 그 사실이 리포트에 드러난다.
    assert "Gamma" in text and "알파" in text, text
    assert result.kept_reasons.get("listing_line_unaligned") == 2, result.report()
    assert any("정렬 실패" in warning for warning in result.warnings), result.warnings
    assert sum(result.kept_reasons.values()) == result.kept, result.report()


def test_plan_falls_back_to_a_conservative_pass_when_it_cannot_converge(tmp_path, monkeypatch):
    """패스 상한에서 불일치 계획을 그대로 쓰지 않고, 아무것도 지워진다고 가정하지 않고 다시 계획한다."""
    from app.pipeline.pdf_export import build as build_mod

    job_dir, original, translated = _listing_job(tmp_path, "cap", single_span_second_row=False)
    translated[0]["blocks"][1]["content"] = (
        "알파\n베타\n" + "매우긴감마번역" * 40 + "\n" + "매우긴델타번역" * 40
    )
    translated[0]["blocks"][2]["content"] = "형제 문단 번역"
    _write(job_dir, original, translated)

    seen: list[frozenset] = []
    plan_targets = build_mod._plan_page_targets

    def recording(ctx, result):
        seen.append(ctx.cleared_indices)
        return plan_targets(ctx, result)

    monkeypatch.setattr(build_mod, "_MAX_PLAN_PASSES", 1)
    monkeypatch.setattr(build_mod, "_plan_page_targets", recording)
    result = build_translated_pdf(job_dir, "ko")

    assert len(seen) == 2, seen          # 1패스(불일치) + 보수적 재계획
    assert seen[0] and seen[-1] == frozenset()
    assert result.replaced >= 1, result.report()


def test_unchanged_listing_rows_are_obstacles_within_the_same_pass(tmp_path, monkeypatch):
    """줄 단위로 확정된 리스팅의 바뀌지 않은 행은 같은 패스의 flow에서 바로 장애물이다.

    블록은 '교체됨'이어도 그 행의 원문은 남는다. 다음 패스로 미루면 계획을 한 번 더
    세워야 해서(실측 25쪽 논문에서 10쪽이 2패스) 빌드가 느려진다.
    """
    from app.pipeline.pdf_export import build as build_mod

    job_dir, original, translated = _listing_job(tmp_path, "residual", single_span_second_row=False)
    translated[0]["blocks"][1]["content"] = "알파\n베타\nGamma\nDelta"   # 두 번째 행은 그대로
    translated[0]["blocks"][2]["content"] = (
        "형제 문단의 번역은 원문보다 훨씬 길어서 한 줄에 들어가지 않고 여러 줄이 "
        "필요하며 위로 당겨질 수 있다. " * 3
    ).strip()
    _write(job_dir, original, translated)

    passes: list[frozenset] = []
    plan_targets = build_mod._plan_page_targets

    def counting(ctx, result):
        passes.append(ctx.cleared_indices)
        return plan_targets(ctx, result)

    monkeypatch.setattr(build_mod, "_plan_page_targets", counting)
    result = build_translated_pdf(job_dir, "ko")
    assert result.listing_lines_replaced == 2, result.report()
    assert len(passes) == 1, passes
    with fitz.open(result.path) as exported:
        page = exported[0]
        gamma = _line_rect(page, "Gamma")
        sibling = [
            fitz.Rect(line["bbox"])
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if "형제" in "".join(span["text"] for span in line["spans"])
            or "번역은" in "".join(span["text"] for span in line["spans"])
        ]
    assert sibling
    assert not [line for line in sibling if (line & gamma).get_area() > 0.5], (gamma, sibling)
