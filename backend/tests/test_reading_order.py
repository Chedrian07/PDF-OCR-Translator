"""텍스트 레이어 읽기 순서(pipeline/reading_order.py) 검증.

`get_text(sort=True)`는 (y, x) 좌표 정렬이라 2단 논문의 문단을 왼쪽·오른쪽 단 교차로
뒤섞었다(textlayer 엔진 결과·번역·Q&A 문맥의 문장 흐름이 끊긴다). 내용 스트림 순서를
기준으로 다단을 인식하고, MuPDF 1.28이 잘게 쪼갠 같은 문장·같은 줄 조각을 합친다.
"""

import threading
from collections import Counter
from pathlib import Path

import pymupdf as fitz
import pytest

from app.pipeline.reading_order import (
    TextBlock,
    extract_text_blocks,
    merge_fragments,
    order_blocks,
    page_text_blocks,
)

SAMPLE = Path(__file__).resolve().parents[2] / "sample" / "2504.19874v1.pdf"


def _b(order, x0, y0, x1, y1, text="", **kw) -> TextBlock:
    kw.setdefault("size", 10.0)
    kw.setdefault("line_height", 12.0)
    return TextBlock(
        x0=x0, y0=y0, x1=x1, y1=y1, text=text or f"b{order}", order=order,
        last_y0=kw.pop("last_y0", y1 - 12), last_y1=kw.pop("last_y1", y1),
        last_x1=kw.pop("last_x1", x1), **kw,
    )


def _texts(blocks) -> list[str]:
    return [b.text for b in blocks]


# ── 순수 기하: 단 구분 ────────────────────────────────────────────────────


def _two_column_page(interleaved: bool) -> list[TextBlock]:
    title = _b(0, 200, 60, 410, 80, "TITLE")
    left = [_b(0, 54, 120 + i * 150, 293, 250 + i * 150, f"LEFT-{i + 1}") for i in range(3)]
    right = [_b(0, 317, 120 + i * 150, 556, 250 + i * 150, f"RIGHT-{i + 1}") for i in range(3)]
    footer = _b(0, 300, 740, 312, 750, "7")
    if interleaved:   # 좌표 순(y→x)으로 그린 문서: 같은 높이의 좌·우 블록이 번갈아 나온다
        stream = [title] + [b for pair in zip(left, right) for b in pair] + [footer]
    else:             # LaTeX식: 왼쪽 단 전체 → 오른쪽 단 전체
        stream = [title] + left + right + [footer]
    return [
        TextBlock(**{**b.__dict__, "order": i}) for i, b in enumerate(stream)
    ]


@pytest.mark.parametrize("interleaved", [False, True])
def test_two_columns_read_left_then_right_between_spanning_blocks(interleaved):
    ordered = order_blocks(_two_column_page(interleaved))
    assert _texts(ordered) == [
        "TITLE", "LEFT-1", "LEFT-2", "LEFT-3", "RIGHT-1", "RIGHT-2", "RIGHT-3", "7",
    ]


def test_full_width_block_splits_the_column_flow_into_bands():
    """전폭 그림 캡션처럼 거터를 가로지르는 블록은 단 흐름의 경계다 — 그 위의 두 단을
    다 읽고, 경계를 지나, 아래의 두 단을 읽는다."""
    stream = [
        _b(0, 54, 100, 293, 300, "L-top"), _b(1, 317, 100, 556, 300, "R-top"),
        _b(2, 54, 320, 556, 360, "FIGURE CAPTION"),
        _b(3, 54, 380, 293, 600, "L-bottom"), _b(4, 317, 380, 556, 600, "R-bottom"),
    ]
    # 내용 순서를 일부러 뒤섞는다 (아래 띠가 먼저 그려진 문서)
    shuffled = [TextBlock(**{**b.__dict__, "order": o}) for b, o in zip(stream, [3, 4, 2, 0, 1])]
    assert _texts(order_blocks(shuffled)) == [
        "L-top", "R-top", "FIGURE CAPTION", "L-bottom", "R-bottom",
    ]


def test_single_column_keeps_content_order_and_moves_rotated_stamps_last():
    """한 단 문서는 내용 순서를 그대로 쓴다. 여백의 세로 스탬프(arXiv)는 본문 사이에
    끼어들지 않고 끝에 온다(sort=True는 저자 블록 사이에 넣었다)."""
    stream = [
        _b(0, 96, 100, 516, 140, "TITLE"),
        _b(1, 11, 200, 38, 560, "arXiv:2504.19874v1", horizontal=False),
        _b(2, 98, 300, 514, 550, "ABSTRACT BODY"),
        _b(3, 71, 600, 541, 680, "INTRO BODY"),
    ]
    assert _texts(order_blocks(stream)) == [
        "TITLE", "ABSTRACT BODY", "INTRO BODY", "arXiv:2504.19874v1",
    ]


def test_formula_fragments_do_not_look_like_columns():
    """디스플레이 수식 조각처럼 작은 블록이 옆으로 늘어서도 본문이 거터를 가로지르면
    단으로 보지 않는다(내용 순서 유지)."""
    stream = [_b(0, 71, 100, 541, 300, "BODY-1")]
    stream += [_b(i, 87 + 30 * i, 310, 110 + 30 * i, 330, f"f{i}") for i in range(1, 9)]
    stream += [_b(9, 71, 340, 541, 600, "BODY-2")]
    assert _texts(order_blocks(stream)) == _texts(stream)


# ── 순수 기하: 조각 병합 ──────────────────────────────────────────────────


def test_centered_title_lines_merge_into_one_unit():
    a = _b(0, 96, 108, 516, 125, "TurboQuant: Online Vector Quantization with Near-optimal",
           size=17.2, line_height=17)
    b = _b(1, 252, 130, 360, 147, "Distortion Rate", size=17.2, line_height=17)
    merged = merge_fragments([a, b])
    assert _texts(merged) == [
        "TurboQuant: Online Vector Quantization with Near-optimal\nDistortion Rate"
    ]
    assert (merged[0].x0, merged[0].y0, merged[0].x1, merged[0].y1) == (96, 108, 516, 147)


def test_sentence_broken_by_inline_math_is_rejoined():
    # 실측(2504.19874v1.pdf 5쪽): 인라인 수식 때문에 줄 높이가 23pt이고 두 블록이 9pt 겹친다
    a = _b(0, 70.9, 594.3, 540.6, 617.1, "In Theorem 2 we prove that Qprod : Rd →{0, 1}b·d",
           size=10.9, line_height=22.9, last_y0=594.3)
    b = _b(1, 70.9, 607.7, 469.2, 630.7, "achieves the following distortion for any x:",
           size=10.9, line_height=23.0, last_y0=607.7)
    assert len(merge_fragments([a, b])) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("We conclude the proof.", "next paragraph starts here"),    # 문단 끝 마침표
        ("followed by the bound", "Proof. The claim follows"),       # 왼쪽 정렬 대문자 = 새 문단
        ("the steps are", "2. Read the section headings"),           # 번호 목록 항목
        ("Algorithm 1 TurboQuant", "5: y ←Π · x"),                   # 알고리즘 줄 번호
        ("the list is", "• second bullet"),                          # 글머리표
    ],
)
def test_new_paragraphs_and_items_stay_separate(first, second):
    a = _b(0, 71, 100, 541, 124, first, line_height=12)
    b = _b(1, 71, 126, 541, 150, second, line_height=12)
    assert len(merge_fragments([a, b])) == 2


def test_heading_followed_by_body_is_not_merged():
    heading = _b(0, 71, 100, 300, 112, "1.3 Overview of Techniques", size=12.0, line_height=12)
    body = _b(1, 71, 114, 541, 200, "our first algorithm is designed", size=10.0, line_height=11)
    assert len(merge_fragments([heading, body])) == 2


def test_table_rows_are_not_merged_into_one_paragraph():
    row1 = _b(0, 71, 100, 541, 110, "Method\nBits\nScore", tabular=True)
    row2 = _b(1, 71, 111, 541, 121, "full cache\n16\n50.06", tabular=True)
    assert len(merge_fragments([row1, row2])) == 2


def test_same_line_formula_fragments_merge_left_to_right():
    a = _b(0, 87, 390, 228, 414, "• Dmse(Qmse) := E", last_y0=390, last_y1=414, last_x1=228,
           size=10.9)
    b = _b(1, 218, 395, 279, 408, "mse (Qmse(x))", size=10.9)
    c = _b(2, 310, 386, 317, 394, "√", size=8.0)   # 줄 위로 솟은 근호 — 간격이 넓다
    d = _b(3, 281, 390, 306, 414, "≤", size=10.9)
    merged = merge_fragments([a, b, d, c])
    assert _texts(merged) == ["• Dmse(Qmse) := E mse (Qmse(x)) ≤ √"]


def test_side_by_side_author_blocks_stay_separate():
    a = _b(0, 108, 171, 219, 211, "Amir Zandieh\nGoogle Research", last_y0=199, last_y1=211)
    b = _b(1, 248, 171, 371, 211, "Majid Daliri\nNYU", last_y0=199, last_y1=211)
    assert len(merge_fragments([a, b])) == 2


# ── 실제 PDF ──────────────────────────────────────────────────────────────


def _two_column_pdf(path: Path, interleaved: bool) -> None:
    """LaTeX식(왼쪽 단 → 오른쪽 단) 또는 좌표 순(행마다 좌·우 교차)으로 그린 2단 PDF."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((230, 70), "A Two Column Paper", fontsize=16)
    para = "word " * 40
    cells = []
    for i in range(3):
        y = 120 + i * 180
        cells.append(((54, y, 293, y + 150), f"LEFT-{i + 1} " + para))
        cells.append(((317, y, 556, y + 150), f"RIGHT-{i + 1} " + para))
    if not interleaved:
        cells = cells[0::2] + cells[1::2]
    for rect, text in cells:
        page.insert_textbox(fitz.Rect(*rect), text, fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.mark.parametrize("interleaved", [False, True])
def test_textlayer_engine_reads_two_column_pdf_in_column_order(tmp_path, interleaved):
    from app.config import Settings
    from app.engine.base import NullSink
    from app.engine.textlayer import TextLayerEngine
    from app.pipeline.pdf import render_pdf_pages

    job = tmp_path / "job"
    job.mkdir()
    _two_column_pdf(job / "source.pdf", interleaved)
    with fitz.open(str(job / "source.pdf")) as doc:
        sorted_order = [
            b[4].split()[0] for b in doc[0].get_text("blocks", sort=True) if "-" in b[4]
        ]
    assert sorted_order[:2] == ["LEFT-1", "RIGHT-1"]  # 예전 sort=True가 낸 교차 순서

    pages = render_pdf_pages(job / "source.pdf", job / "pages", 72, 10)
    engine = TextLayerEngine(Settings(engine="textlayer", data_dir=tmp_path / "d"))
    md = engine.run_multi(pages, job / "work", NullSink(), threading.Event())

    marks = ["LEFT-1", "LEFT-2", "LEFT-3", "RIGHT-1", "RIGHT-2", "RIGHT-3"]
    positions = [md.index(m) for m in marks]
    assert positions == sorted(positions), md[:400]
    assert md.index("A Two Column Paper") < positions[0]


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample/2504.19874v1.pdf 없음")
def test_sample_paper_blocks_are_merged_without_losing_text():
    """MuPDF 1.28이 쪼갠 제목·수식 조각을 합쳐 번역 유닛 수를 줄이되 글자는 하나도
    잃지 않는다(합치기 전후의 비공백 문자 다중집합이 같다)."""
    with fitz.open(str(SAMPLE)) as doc:
        raw_total = merged_total = 0
        for page in doc:
            raw = extract_text_blocks(page, fitz)
            merged = page_text_blocks(page, fitz)
            raw_total += len(raw)
            merged_total += len(merged)
            raw_chars = Counter("".join(b.text for b in raw).replace(" ", "").replace("\n", ""))
            got_chars = Counter("".join(b.text for b in merged).replace(" ", "").replace("\n", ""))
            assert raw_chars == got_chars, f"{page.number + 1}쪽에서 글자가 바뀌었다"
        first = page_text_blocks(doc[0], fitz)

    assert merged_total < raw_total * 0.8, (raw_total, merged_total)
    texts = [b.text for b in first]
    # 두 줄로 쪼개진 제목이 한 유닛이 되고, 세로 arXiv 스탬프는 저자 사이가 아니라 끝에 온다
    assert texts[0] == "TurboQuant: Online Vector Quantization with Near-optimal\nDistortion Rate"
    assert texts[-1].startswith("arXiv:2504.19874v1")
    assert texts.index("Abstract") < next(i for i, t in enumerate(texts) if "Introduction" in t)
