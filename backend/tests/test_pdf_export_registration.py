"""내보내기의 안전 계약 — 대조되지 않는 페이지는 건드리지 않는다.

OCR 단계가 페이지를 밀어 매핑하면(모델이 페이지를 쪼개거나 건너뛴 경우) 내보내기는
그 좌표를 믿고 **엉뚱한 페이지의 원문을 영구 리댁션**한다. 실측(j_a6df80d1d8ea):
layout 6개 페이지가 한 칸씩 밀려 PDF 23쪽의 프로젝트명 29개가 삭제되고 24쪽 캡션이
그 자리에 찍혔다. 리댁션은 되돌릴 수 없으므로 이 게이트가 마지막 방어선이다.
"""

import json
from pathlib import Path

import fitz
import pytest

from app.pipeline.pdf_export import build_translated_pdf
from app.pipeline.pdf_export.build import _unregistered_layout_pages
from app.pipeline.pdf_export.text import match_paragraph_shape, strip_markdown

PAGE_W, PAGE_H = 360.0, 480.0
MARKS = ["ALPHAPAGE", "BRAVOPAGE", "CHARLIEPAGE", "DELTAPAGE"]


def _bbox(rect: fitz.Rect) -> list[int]:
    return [
        int(rect.x0 / PAGE_W * 999), int(rect.y0 / PAGE_H * 999),
        int(rect.x1 / PAGE_W * 999), int(rect.y1 / PAGE_H * 999),
    ]


# 페이지마다 서로 구별되는 문단 4개 — 게이트는 프로브가 3개 미만이면 판정하지
# 않는다(블록이 한둘뿐인 페이지는 비율이 요동쳐 오탐이 나므로).
_PARAS = 4


def _para(mark: str, idx: int) -> str:
    return " ".join([f"{mark}{idx}"] * 10)


def _make_doc(job: Path) -> None:
    doc = fitz.open()
    for mark in MARKS:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for idx in range(_PARAS):
            for line in range(3):
                page.insert_text(
                    (30, 60 + idx * 100 + line * 16), _para(mark, idx), fontsize=8,
                )
    doc.save(str(job / "source.pdf"))
    doc.close()


def _layout(shift: int) -> list[dict]:
    """shift=0이면 제자리, shift=1이면 layout p가 물리 p+1을 설명한다."""
    pages = []
    for pno in range(1, len(MARKS) + 1):
        src = min(pno - 1 + shift, len(MARKS) - 1)
        blocks = []
        for idx in range(_PARAS):
            top = 50 + idx * 100
            blocks.append({
                "type": "text",
                "bbox": _bbox(fitz.Rect(30, top, 330, top + 70)),
                "content": _para(MARKS[src], idx),
            })
        pages.append({
            "page": pno, "width": PAGE_W, "height": PAGE_H, "blocks": blocks,
        })
    return pages


def _write_pair(job: Path, shift: int) -> None:
    (job / "layout.json").write_text(json.dumps(_layout(shift)), encoding="utf-8")
    ko = _layout(shift)
    for page in ko:
        for block in page["blocks"]:
            block["content"] = "번역된 본문 " * 20
    (job / "layout.ko.json").write_text(json.dumps(ko), encoding="utf-8")


@pytest.fixture()
def job(tmp_path: Path) -> Path:
    d = tmp_path / "job"
    d.mkdir()
    _make_doc(d)
    return d


def test_aligned_layout_is_not_refused(job: Path):
    """제자리에 있는 레이아웃은 게이트에 걸리지 않는다 (오탐 없음)."""
    _write_pair(job, shift=0)
    with fitz.open(job / "source.pdf") as doc:
        pages = {p["page"]: p for p in json.loads(
            (job / "layout.json").read_text(encoding="utf-8"))}
        assert _unregistered_layout_pages(doc, pages) == {}


def test_shifted_layout_is_detected_and_the_page_is_left_untouched(job: Path):
    """한 칸 밀린 레이아웃은 검출되고, 그 페이지의 원문은 한 글자도 지워지지 않는다."""
    _write_pair(job, shift=1)
    with fitz.open(job / "source.pdf") as doc:
        pages = {p["page"]: p for p in json.loads(
            (job / "layout.json").read_text(encoding="utf-8"))}
        flagged = _unregistered_layout_pages(doc, pages)
    assert set(flagged) >= {1, 2, 3}, flagged
    for pno, (elsewhere, score, mine) in flagged.items():
        assert elsewhere == pno + 1 and score > mine

    result = build_translated_pdf(job, "ko")
    assert result.kept_reasons.get("page_source_mismatch", 0) >= 3
    with fitz.open(result.path) as out:
        for i, mark in enumerate(MARKS[:3]):
            assert f"{mark}0" in out[i].get_text(), (
                f"p{i + 1}의 원문이 삭제됐다 — 리댁션은 되돌릴 수 없다"
            )
    assert any("건너뜀" in w for w in result.warnings), result.warnings


def test_markdown_markers_never_reach_the_page():
    """번역 단계가 흘린 마크다운 표기는 조판 전에 걷어낸다 (실측 p4의 '###')."""
    assert strip_markdown("### 테스크 입력 및 출력") == "테스크 입력 및 출력"
    assert strip_markdown("- 항목 하나\n- 항목 둘") == "항목 하나\n항목 둘"
    assert strip_markdown("**강조**된 문장") == "강조된 문장"
    assert strip_markdown("`코드` 스팬") == "코드 스팬"
    assert strip_markdown("> 인용문") == "인용문"
    # 본문 안의 하이픈·별표는 건드리지 않는다
    assert strip_markdown("a - b 그리고 2*3") == "a - b 그리고 2*3"


def test_paragraph_shape_follows_the_source_block():
    """원문이 한 문단이면 번역의 문단 분리를 잇는다 — 한 줄짜리 상자에 두 문단이
    들어오면 높이가 모자라 블록이 통째로 버려진다."""
    joined = match_paragraph_shape("Sourcing from OSS-Fuzz The lifecycle …",
                                   "OSS-Fuzz에서의 수집\n\nOSS-Fuzz가 탐지한 …")
    assert "\n" not in joined and "수집 OSS-Fuzz가" in joined
    # 의미 있는 줄바꿈(빈 줄 아님)은 보존한다
    assert match_paragraph_shape("one line", "첫 줄\n둘째 줄") == "첫 줄\n둘째 줄"
    # 원문에도 문단 경계가 있으면 손대지 않는다
    assert match_paragraph_shape("a\n\nb", "가\n\n나") == "가\n\n나"


def test_preserved_blocks_are_reported_as_intentional_not_as_failure(job: Path):
    """번역 단계가 '번역 안 함'을 결정한 블록은 실패가 아니라 의도적 보존으로 집계된다.

    표식이 없으면 원문과 번역이 같아 kept_reason=unchanged로 떨어져 번역 결함과
    구분되지 않는다 — 운영자가 리포트만 보고 "왜 이 페이지가 미번역인가"를
    판단할 수 없다.
    """
    _write_pair(job, shift=0)
    ko_path = job / "layout.ko.json"
    ko = json.loads(ko_path.read_text(encoding="utf-8"))
    for page in ko:
        page["blocks"][0]["content"] = _para(MARKS[page["page"] - 1], 0)  # 원문 그대로
        page["blocks"][0]["preserved"] = "code"
    ko_path.write_text(json.dumps(ko), encoding="utf-8")

    result = build_translated_pdf(job, "ko")
    assert result.kept_reasons.get("preserved:code", 0) == len(MARKS)
    assert result.specialist_kept.get("code", 0) == len(MARKS)
    assert result.kept_reasons.get("unchanged", 0) == 0, result.kept_reasons
    # 원문은 그대로 남는다 (교체를 시도하지 않는다)
    with fitz.open(result.path) as out:
        assert f"{MARKS[0]}0" in out[0].get_text()
