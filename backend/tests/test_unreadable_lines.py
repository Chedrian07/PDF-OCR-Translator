"""영역 재처리로도 모델이 끝내 못 읽은 줄 — 원본 텍스트 조회 · 충실도 게이트 · 레이아웃.

region_rescue는 그 줄을 원본 줄 이미지(`image`) + 안내 블록(`unreadable`, 같은 상자)으로 낸다.
안내 블록에는 원본 PDF 텍스트 레이어에서 그 상자 안의 줄을 싣는다(text_reference.page_line_texts).
"""

import pymupdf
import pytest

from app.pipeline.layout import render_layout_html
from app.pipeline.text_reference import line_texts_local, page_line_texts

_LINES = [
    "First line of the derivation stays above.",
    "$C'D''^2 = 7^2 + 13^2 - 2 \\cdot 7 \\cdot 13$",
    "Third line of the derivation stays below.",
]


def _three_line_pdf(path, rotation=0):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(_LINES):
        page.insert_text((60, 120 + i * 18), line, fontsize=11)
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()


def _box_of(page, needle, grow=(0.0, 0.0)):
    """렌더 공간(회전 반영) 0~999 상자 — needle 줄의 상자를 grow(pt)만큼 넓혀서."""
    rect = page.search_for(needle)[0] * page.rotation_matrix
    rect.normalize()
    dx, dy = grow
    w, h = page.rect.width, page.rect.height
    return (
        (rect.x0 - dx) / w * 999, (rect.y0 - dy) / h * 999,
        (rect.x1 + dx) / w * 999, (rect.y1 + dy) / h * 999,
    )


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("grow", [(0.0, -2.0), (4.0, 6.0)])  # 잉크보다 좁은 상자 · 줄 사이까지 넓힌 상자
def test_the_line_inside_the_box_is_read_without_its_neighbours(tmp_path, rotation, grow):
    path = tmp_path / "lines.pdf"
    _three_line_pdf(path, rotation)
    with pymupdf.open(path) as doc:
        box = _box_of(doc[0], _LINES[1], grow=grow)
    assert line_texts_local(path, 0, [box]) == [_LINES[1]]


def test_each_box_gets_its_own_line_and_an_empty_box_gets_nothing(tmp_path):
    path = tmp_path / "lines.pdf"
    _three_line_pdf(path)
    with pymupdf.open(path) as doc:
        first = _box_of(doc[0], "First line", grow=(300, 1))
        third = _box_of(doc[0], "Third line", grow=(300, 1))
    blank = (100, 900, 900, 950)
    assert line_texts_local(path, 0, [third, blank, first]) == [_LINES[2], "", _LINES[0]]


def test_an_unopenable_source_fills_nothing(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    assert line_texts_local(path, 0, [(0, 0, 999, 999)]) == [""]


class _FakePage:
    rect = pymupdf.Rect(0, 0, 999, 999)
    derotation_matrix = pymupdf.Matrix(1, 1)
    rotation_matrix = pymupdf.Matrix(1, 1)

    def __init__(self, lines):
        self._lines = lines

    def get_text(self, kind):
        assert kind == "dict"
        return {"blocks": [{"lines": [
            {"bbox": bbox, "spans": [{"text": text}]} for bbox, text in self._lines
        ]}]}


def test_superscripts_extracted_as_separate_lines_join_their_row_in_order():
    page = _FakePage([
        ((150, 96, 158, 106), "2"),           # 위첨자 — 위로 솟아 먼저 정렬된다
        ((100, 100, 149, 114), "C'D''"),
        ((162, 100, 260, 114), "= 49 + 169"),
        ((100, 130, 300, 144), "next row"),   # 다음 줄
    ])
    assert page_line_texts(pymupdf, page, [(90, 90, 300, 150)]) == ["C'D''2 = 49 + 169 next row"]


def test_a_broken_text_layer_fills_nothing():
    pua = "".join(chr(0xE000 + i % 300) for i in range(400))
    page = _FakePage([((100, 100, 900, 114), pua), ((100, 130, 900, 144), "few words")])
    assert page_line_texts(pymupdf, page, [(90, 90, 910, 150)]) == [""]
    # 페이지는 멀쩡해도 깨진 글리프가 든 줄은 싣지 않는다
    page = _FakePage([((100, 100, 900, 114), "good text " * 40), ((100, 130, 900, 144), "x�y")])
    assert page_line_texts(pymupdf, page, [(90, 125, 910, 150)]) == [""]


def test_extracted_text_cannot_inject_page_or_ref_grammar():
    page = _FakePage([((100, 100, 900, 114), "a <|ref|>text<|/ref|> b <PAGE> c " * 10)])
    (text,) = page_line_texts(pymupdf, page, [(90, 90, 910, 120)])
    assert "<|ref|>" not in text and "<PAGE>" not in text


# ── 충실도 게이트: 채운 텍스트는 모델의 전사가 아니다 ─────────────────────────


def test_the_gate_counts_an_unreadable_line_as_missing_even_when_filled(tmp_path):
    """채운 원본 텍스트를 후보에 넣으면 게이트가 정답을 정답과 비교한다. 줄 크롭(`image`)도
    배경 상자가 그려진 줄에서 '믿을 수 있는 그림'이 되어 그 줄을 정답에서 지우면 안 된다."""
    from app.pipeline.fidelity import page_fidelity_blocks

    paragraph = (
        "The model reads every other line of this derivation without trouble and the gate "
        "compares those lines with the embedded text layer of the original document page. "
    ) * 3
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(60, 60, 535, 400), paragraph, fontsize=11)
    page.draw_rect(pymupdf.Rect(55, 445, 540, 470), color=None, fill=(0.95, 0.95, 0.95))
    page.insert_text((60, 462), "an unreadable line of the derivation 7^2 + 13^2", fontsize=11)
    path = tmp_path / "gate.pdf"
    doc.save(path)
    doc.close()

    box = [90, 527, 910, 560]
    read = {"type": "text", "bbox": [90, 60, 910, 480], "content": paragraph}
    marked = [
        read,
        {"type": "image", "bbox": box, "content": ""},
        {"type": "unreadable", "bbox": box,
         "content": "> ⚠️ 인식하지 못한 줄 — 원본 PDF 텍스트: an unreadable line of the derivation 7^2 + 13^2"},
    ]
    with pymupdf.open(path) as opened:
        without = page_fidelity_blocks(pymupdf, opened[0], [read], 1)
        filled = page_fidelity_blocks(pymupdf, opened[0], marked, 1)
        figure = page_fidelity_blocks(pymupdf, opened[0], marked[:2], 1)
    assert without.measurable and without.score < 0.99
    assert filled.score == pytest.approx(without.score)
    # 안내 블록 없는 같은 상자의 image는 모델의 그림 분류다 — 배경이 그려져 있어 그 줄을 정답에서 뺀다
    assert figure.score > without.score


# ── 레이아웃: 원본 줄 이미지가 그 줄을 보인다 ──────────────────────────────


@pytest.mark.parametrize("facsimile", [False, True])
def test_layout_views_show_the_original_line_not_the_note(facsimile):
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "image", "bbox": [90, 500, 910, 530], "content": "", "image": "p0001_0.jpg"},
        {"type": "unreadable", "bbox": [90, 500, 910, 530], "content": "> ⚠️ 인식하지 못한 줄"},
    ]}]
    html = render_layout_html(pages, "/b", page_src=lambda n: f"/pages/{n}.png", facsimile=facsimile)
    assert "인식하지 못한 줄" not in html
    assert ("p0001_0.jpg" in html) is (not facsimile)  # facsimile은 페이지 이미지가 그 줄을 보인다
