"""번역 PDF — 텍스트 블록 안 인라인 수식 선(분수선·근호 윗선) 처리.

TeX 계열 PDF의 분수선은 글리프가 아니라 얇은 가로 선이다. 텍스트 리댁션은 글리프만
지우므로 예전에는 (1) 번역문 옆에 원문 수식의 가로줄만 떠 있었고 (2) 그 선이 고정
장애물이라 자기 블록 안에서 번역문의 자리를 막아 가독성 하한 아래로 축소됐다
(실서버 25쪽 논문 + 로컬 MLX 번역: 축소 경고 6건 중 5건이 블록 안 선 2–5개).
블록이 소유한 선은 원문 글리프처럼 다룬다 — 블록이 남으면 보존, 교체되면 함께 지운다.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from app.pipeline.pdf_export import build_translated_pdf

PAGE_W, PAGE_H = 595.0, 842.0
LINES = [
    "We bound the distortion of the quantizer for every bit width b and",
    "show that the error decays like over the whole range, which is",
    "within a small constant factor of the information theoretic bound.",
]
KO = (
    "우리는 모든 비트 폭 b에 대해 양자화기의 왜곡을 상한으로 묶고, 오차가 전 범위에서 "
    "1/d처럼 줄어듦을 보이며, 이는 정보 이론적 하한의 작은 상수배 이내다."
)


def _bbox(rect: fitz.Rect) -> list[float]:
    return [
        rect.x0 / PAGE_W * 999, rect.y0 / PAGE_H * 999,
        rect.x1 / PAGE_W * 999, rect.y1 / PAGE_H * 999,
    ]


def _paragraph_with_fraction(page) -> fitz.Rect:
    """세 줄 문단 + 둘째 줄의 인라인 분수(작은 분자·분모 글자와 0.4pt 분수선)."""
    y = 200.0
    for line in LINES:
        page.insert_text((72, y), line, fontsize=10, fontname="tiro")
        y += 12.0
    # 둘째 줄(기준선 212) 'like' 뒤의 1/d — 분수선은 수식 축(기준선 위 ~2.5pt)에 있다
    page.insert_text((268, 208), "1", fontsize=7, fontname="tiro")
    page.insert_text((268, 215), "d", fontsize=7, fontname="tiro")
    page.draw_line((267, 209.5), (273, 209.5), width=0.4)
    return fitz.Rect(70, 190, 420, 229)


def _write_job(tmp_path: Path, name: str, blocks: list[dict], translations: dict) -> Path:
    job_dir = tmp_path / name
    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": blocks}]
    translated = json.loads(json.dumps(original))
    for index, content in translations.items():
        translated[0]["blocks"][index]["content"] = content
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )
    return job_dir


def _new_source(tmp_path: Path, name: str):
    job_dir = tmp_path / name
    job_dir.mkdir()
    doc = fitz.open()
    return job_dir, doc, doc.new_page(width=PAGE_W, height=PAGE_H)


def _drawings(path: Path) -> list:
    with fitz.open(path) as exported:
        return [fitz.Rect(d["rect"]) for d in exported[0].get_drawings()]


def _block(rect: fitz.Rect, content: str, block_type: str = "text") -> dict:
    return {"type": block_type, "bbox": _bbox(rect), "content": content,
            "fs": 10 / PAGE_W * 100}


def _korean_font_sizes(path: Path) -> set[float]:
    with fitz.open(path) as exported:
        return {
            round(span["size"], 1)
            for block in exported[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if any("가" <= ch <= "힣" for ch in span["text"])
        }


def test_translated_paragraph_drops_its_fraction_bar_and_keeps_full_size(tmp_path):
    """위·아래 문단이 붙어 있어 옮길 자리가 없는 문단 — 자기 분수선이 자리를 막으면 안 된다.

    예전: 분수선이 장애물이라 6.6pt로 축소 배치(경고)되고 분수선은 그대로 남았다.
    """
    above = "Previous paragraph line that stays in English above the target."
    below = "Next paragraph line that stays in English below the target block."
    job_dir, doc, page = _new_source(tmp_path, "fraction")
    page.insert_text((72, 188), above, fontsize=10, fontname="tiro")
    _paragraph_with_fraction(page)
    page.insert_text((72, 236), below, fontsize=10, fontname="tiro")
    doc.save(job_dir / "source.pdf")
    doc.close()
    blocks = [
        _block(fitz.Rect(70, 179, 420, 191.5), above),
        _block(fitz.Rect(70, 192, 420, 227), "\n".join(LINES)),
        _block(fitz.Rect(70, 227.5, 420, 239), below),
    ]
    _write_job(tmp_path, "fraction", blocks, {1: KO})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.warnings == [], result.warnings
    assert min(_korean_font_sizes(result.path)) >= 8.0
    # 원문 분수선이 번역문 옆에 남지 않는다
    assert _drawings(result.path) == []
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
    assert "양자화기의 왜곡" in text and "We bound" not in text
    assert above in text and below in text


def test_list_item_owns_the_bar_not_the_empty_list_container(tmp_path):
    """OCR 'list' 컨테이너(빈 content)는 주인이 아니다 — 항목 블록이 교체되면 선도 지운다."""
    job_dir, doc, page = _new_source(tmp_path, "list-item")
    rect = _paragraph_with_fraction(page)
    doc.save(job_dir / "source.pdf")
    doc.close()
    container = fitz.Rect(rect.x0 - 4, rect.y0 - 4, rect.x1 + 4, rect.y1 + 4)
    blocks = [_block(container, "", "list"), _block(rect, "- " + " ".join(LINES))]
    _write_job(tmp_path, "list-item", blocks, {1: "- " + KO})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert _drawings(result.path) == []


def test_kept_paragraph_keeps_its_fraction_bar(tmp_path):
    """번역이 원문과 같아 남는 블록은 수식 선도 그대로 둔다."""
    job_dir, doc, page = _new_source(tmp_path, "kept")
    rect = _paragraph_with_fraction(page)
    doc.save(job_dir / "source.pdf")
    doc.close()
    content = "\n".join(LINES)
    _write_job(tmp_path, "kept", [_block(rect, content)], {0: content})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 0, result.report()
    assert len(_drawings(result.path)) == 1


def test_qed_box_and_long_rules_inside_a_translated_paragraph_survive(tmp_path):
    """네 선으로 그린 QED 상자와 긴 밑줄·구분선(160pt 초과)은 인라인 수식 선이 아니다."""
    job_dir, doc, page = _new_source(tmp_path, "box")
    y = 200.0
    for line in LINES:
        page.insert_text((72, y), line, fontsize=10, fontname="tiro")
        y += 12.0
    # 마지막 줄 끝의 □ (6.5 × 7pt, 선 네 개)
    x0, y0, x1, y1 = 400.0, 218.0, 406.5, 225.0
    for start, end in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)),
                       ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))):
        page.draw_line(start, end, width=0.4)
    # 문단 폭 대부분을 가로지르는 긴 선 (장식·밑줄)
    page.draw_line((72, 203.5), (372, 203.5), width=0.4)
    doc.save(job_dir / "source.pdf")
    doc.close()
    rect = fitz.Rect(70, 190, 420, 229)
    _write_job(tmp_path, "box", [_block(rect, "\n".join(LINES))], {0: KO})

    with fitz.open(job_dir / "source.pdf") as source:
        before = len(source[0].get_drawings())
    result = build_translated_pdf(job_dir, "ko")

    assert before == 5
    assert len(_drawings(result.path)) == 5
