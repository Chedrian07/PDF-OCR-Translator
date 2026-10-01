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
