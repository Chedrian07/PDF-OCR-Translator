"""영역 구조(region_rescue) 단위 테스트 — 합성 페이지 + 가짜 영역 OCR.

검은 막대가 텍스트 줄이다. 가짜 OCR은 크롭에 완전히 들어온 줄을 모델처럼 읽기 순서대로
`<BLOCK>text[크롭 대비 0~999]<FORK>줄 이름`으로 낸다. 함정 줄이 크롭에 있으면 그 앞 줄들과
함정 줄의 머리까지 내고 루프로 멈춘다 — 실측 SGLang 출력과 같은 모양이다.
"""

import pytest
from PIL import Image, ImageDraw

from app.engine.region_rescue import (
    UNREADABLE_FILLED,
    UNREADABLE_TEXT,
    Block,
    InkMap,
    Rect,
    RegionOutput,
    RescueFailed,
    column_regions,
    parse_blocks,
    rescue_page,
    sane,
)


def _page(lines=20, columns=1, width=1000, line_h=20, gap=20, top=60, figure=None):
    height = top * 2 + lines * (line_h + gap)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    geometry = []  # (name, box px), in the model's reading order
    spans = (
        [(80, width - 80)] if columns == 1
        else [(60, width // 2 - 40), (width // 2 + 40, width - 60)]
    )
    for column, (x0, x1) in enumerate(spans):
        for i in range(lines):
            y0 = top + i * (line_h + gap)
            box = (x0, y0, x1, y0 + line_h)
            draw.rectangle(box, fill="black")
            geometry.append((f"{'LR'[column]}{i}", box))
    if figure is not None:
        draw.rectangle(figure, fill="black")
    return image, geometry


class FakeRegionOCR:
    def __init__(self, geometry, traps=(), readable_alone=(), whole_box_first=False):
        self.geometry = geometry
        self.traps = set(traps)
        self.readable_alone = set(readable_alone)
        self.whole_box_first = whole_box_first
        self.calls = []

    def __call__(self, crop, rect, pad):
        self.calls.append(rect)
        inside = [
            (name, box) for name, box in self.geometry
            if box[0] >= rect.x0 and box[2] <= rect.x1 and box[1] >= rect.y0 and box[3] <= rect.y1
        ]
        width, height = crop.size

        def header(box):
            x0 = (box[0] - rect.x0 + pad) / width * 999
            y0 = (box[1] - rect.y0 + pad) / height * 999
            x1 = (box[2] - rect.x0 + pad) / width * 999
            y1 = (box[3] - rect.y0 + pad) / height * 999
            return f"<BLOCK>text[{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}]<FORK>"

        names = [name for name, _ in inside]
        trap = next((name for name in names if name in self.traps), None)
        if trap is None or (names == [trap] and trap in self.readable_alone):
            return RegionOutput(True, "<PAGE>" + "\n".join(header(b) + n for n, b in inside))
        out = []
        if self.whole_box_first and len(inside) > 1:
            out.append(header((inside[0][1][0], inside[0][1][1], inside[-1][1][2], inside[-1][1][3]))
                       + "everything at once")
        for name, box in inside:
            if name == trap:
                out.append(header(box) + "x x x x x x")
                break
            out.append(header(box) + name)
        return RegionOutput(False, "<PAGE>" + "\n".join(out), "loop")


def _texts(result):
    return [b.split("<FORK>", 1)[1] for b in result.raw.split("\n") if "<FORK>" in b]


def test_a_page_that_reads_cleanly_is_returned_as_is():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry)
    result = rescue_page(image, ocr, max_calls=10)
    assert _texts(result) == [name for name, _ in geometry]
    assert result.calls == 1 and result.unreadable == 0


def test_a_small_looping_block_is_isolated_and_reading_resumes_below_it():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L7"}, readable_alone={"L7"})
    result = rescue_page(image, ocr, max_calls=10)
    assert _texts(result) == [name for name, _ in geometry]  # 순서 그대로, 빠짐 없음
    assert result.unreadable == 0
    # 전체 → 함정 줄 단독 → 그 아래 나머지: 이미 끝낸 L0~L6은 다시 읽지 않는다
    assert result.calls == 3
    # 함정 줄 단독 크롭은 그 줄의 잉크를 통째로 덮는다(위아래 흰 여백은 크롭에 두른다)
    assert ocr.calls[1].y0 <= geometry[7][1][1] and ocr.calls[1].y1 > geometry[7][1][3]


def _marker_lines(result):
    lines = result.raw.split("\n")
    i = next(i for i, b in enumerate(lines) if "<BLOCK>unreadable[" in b)
    return lines[i - 1], lines[i]


def _box(line):
    return [int(v) for v in line.split("[", 1)[1].split("]", 1)[0].split(",")]


def test_a_line_that_still_loops_alone_becomes_its_original_image_and_a_note():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L12"})
    result = rescue_page(image, ocr, max_calls=20)
    expected = []
    for name, _ in geometry:
        expected += ["", UNREADABLE_TEXT] if name == "L12" else [name]
    assert _texts(result) == expected
    assert result.unreadable == 1 and result.filled == 0
    assert 0 < result.unreadable_share < 0.05
    # 원본 줄 이미지(내용 없는 image — 벤더 크롭 경로가 잘라 낸다) + 전용 라벨 안내, 같은 상자.
    # 전용 라벨이라 번역 PDF가 원문 줄을 지우고 안내문을 조판하지 않는다(pdf_export 교체 대상 아님)
    crop, note = _marker_lines(result)
    assert crop.startswith("<BLOCK>image[") and _box(crop) == _box(note)
    # 상자는 그 줄의 잉크를 여백과 함께 덮되 이웃 줄(L11 아래 520, L13 위 580)에 닿지 않고,
    # 가로는 잉크 폭만 덮는다(전폭 0~999가 아니다)
    x0, y0, x1, y1 = _box(note)
    top, bottom = y0 / 999 * image.height, y1 / 999 * image.height
    assert 520 < top <= 537 and 563 <= bottom < 580
    assert 70 <= x0 < 80 and 920 < x1 / 999 * image.width < 935


def test_an_unreadable_line_crop_stops_halfway_to_a_tight_neighbour():
    # 줄 간격 8px, 위 줄의 드문 획 끝 두 점이 아래 줄 3px 위까지 내려온다(줄 판정의 잡티 허용치
    # 아래라 '빈 행'으로 보인다) — 크롭에 들면 안 된다
    image, geometry = _page(gap=8)
    l12, l13 = (box for name, box in geometry if name in ("L12", "L13"))
    stroke = l12[1] - 3
    ImageDraw.Draw(image).line([(300, l12[1] - 8), (300, stroke)], fill="black", width=1)
    ocr = FakeRegionOCR(geometry, traps={"L12"})
    result = rescue_page(image, ocr, max_calls=20)
    _, note = _marker_lines(result)
    _, y0, _, y1 = _box(note)
    top, bottom = y0 / 999 * image.height, y1 / 999 * image.height
    assert stroke < top + 0.5 <= l12[1] and l12[3] <= bottom - 0.5 <= l13[1] - 2


def test_markers_are_set_apart_from_their_neighbours_by_blank_lines():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L12"})
    raw = rescue_page(image, ocr, max_calls=20).raw
    # 앞 문단 뒤 빈 줄 → 이미지 → 안내 인용 → 빈 줄 → 다음 줄 (인용이 다음 줄을 삼키지 않게)
    assert "L11\n\n<BLOCK>image[" in raw
    assert f"{UNREADABLE_TEXT}\n\n<BLOCK>text[" in raw


def test_the_source_text_of_an_unreadable_line_rides_on_its_note():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L12"})
    asked = []

    def line_text(boxes):
        asked.append(boxes)
        return ["$a <FORK> b$\n  <PAGE>c"]

    result = rescue_page(image, ocr, max_calls=20, line_text=line_text)
    assert len(asked) == 1 and len(asked[0]) == 1  # 인식 불가 줄 상자만, 한 번에
    crop, note = _marker_lines(result)
    assert list(asked[0][0]) == pytest.approx(_box(note), abs=1)
    assert result.filled == 1
    # 한 줄로 접고 블록 문법을 무력화한다 — 채운 텍스트가 새 블록·페이지를 만들지 않는다
    assert note.split("<FORK>", 1)[1] == UNREADABLE_FILLED + "$a ⟨FORK> b$ ⟨PAGE>c"
    assert result.raw.count("<FORK>") == 21 and result.raw.count("<PAGE>") == 1


@pytest.mark.parametrize("answer", [[""], [], ["a", "b"]])
def test_a_line_without_usable_source_text_keeps_the_image_only_note(answer):
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L12"})
    result = rescue_page(image, ocr, max_calls=20, line_text=lambda boxes: answer)
    assert result.filled == 0 and _marker_lines(result)[1].endswith(UNREADABLE_TEXT)


def test_the_source_text_hook_is_not_called_when_every_line_was_read():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L7"}, readable_alone={"L7"})

    def line_text(boxes):
        raise AssertionError("called")

    result = rescue_page(image, ocr, max_calls=10, line_text=line_text)
    assert result.unreadable == 0 and result.filled == 0


def test_a_whole_box_block_that_overlaps_its_own_lines_is_not_trusted():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L5"}, readable_alone={"L5"}, whole_box_first=True)
    result = rescue_page(image, ocr, max_calls=20)
    texts = _texts(result)
    assert "everything at once" not in texts            # 겹치는 통째 블록은 버리고 나눠 읽었다
    assert texts == [name for name, _ in geometry]


def test_a_large_region_that_cannot_be_cut_fails_the_rescue():
    # 줄 사이에 빈 행이 없는 큰 덩어리(그림·잡티 스캔) 안의 루프 — 절반을 비운 채 살리지 않는다
    figure = (80, 300, 920, 700)
    image, geometry = _page(lines=20, figure=figure)
    geometry = [g for g in geometry if g[1][3] < figure[1] or g[1][1] > figure[3]]
    geometry.insert(next(i for i, g in enumerate(geometry) if g[1][1] > figure[3]), ("FIG", figure))
    ocr = FakeRegionOCR(geometry, traps={"FIG"})
    with pytest.raises(RescueFailed):
        rescue_page(image, ocr, max_calls=30)


def test_the_call_budget_fails_the_rescue_instead_of_blanking_lines():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L3", "L9", "L15"})
    with pytest.raises(RescueFailed):
        rescue_page(image, ocr, max_calls=3)
    assert len(ocr.calls) == 3


def test_the_deadline_fails_the_rescue():
    image, geometry = _page()
    ocr = FakeRegionOCR(geometry, traps={"L3"})
    with pytest.raises(RescueFailed):
        rescue_page(image, ocr, max_calls=50, deadline=lambda: len(ocr.calls) >= 1)


def test_two_column_pages_are_solved_column_by_column_in_reading_order():
    # 왼쪽 단 중간의 루프 — 전폭 이어 읽기면 오른쪽 단 윗부분(R0~R6)을 건너뛰고,
    # 전폭 자르기면 L·R가 섞인다. 단마다 따로 풀어 왼쪽 전부 → 오른쪽 전부.
    image, geometry = _page(columns=2)
    ocr = FakeRegionOCR(geometry, traps={"L7"}, readable_alone={"L7"})
    result = rescue_page(image, ocr, max_calls=20)
    assert _texts(result) == [name for name, _ in geometry]


def test_column_regions_only_splits_pages_with_a_gutter():
    single, _ = _page()
    double, _ = _page(columns=2)
    assert len(column_regions(InkMap(single), Rect(0, 0, *single.size))) == 1
    regions = column_regions(InkMap(double), Rect(0, 0, *double.size))
    assert len(regions) == 2 and regions[0].x1 <= regions[1].x0


def test_middle_cut_lands_between_lines():
    image, geometry = _page(lines=6)
    ink = InkMap(image)
    cut = ink.middle_cut(Rect(0, 0, *image.size))
    assert all(not (box[1] <= cut < box[3]) for _, box in geometry)


def test_parse_blocks_maps_padded_crop_boxes_to_page_coordinates():
    rect = Rect(0, 500, 1000, 700)
    crop_size = (1000 + 2 * 48, 200 + 2 * 48)
    # 페이지 px (100, 550)-(900, 570)을 크롭(여백 48) 대비 0~999로 쓴 출력
    x0, y0 = (100 + 48) / crop_size[0] * 999, (50 + 48) / crop_size[1] * 999
    x1, y1 = (900 + 48) / crop_size[0] * 999, (70 + 48) / crop_size[1] * 999
    raw = f"<PAGE><BLOCK>text[{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}]<FORK>hello"
    (block,) = parse_blocks(raw, rect, crop_size, 48, (1000, 2000))
    assert block.text == "hello"
    assert block.box[0] == pytest.approx(100 / 1000 * 999, abs=2)
    assert block.box[1] == pytest.approx(550 / 2000 * 999, abs=2)
    assert block.box[3] == pytest.approx(570 / 2000 * 999, abs=2)


def test_sane_rejects_a_block_that_swallows_another():
    whole = Block("other", "", (0, 0, 999, 999), "all")
    line = Block("text", "", (10, 10, 900, 30), "line")
    assert sane([line, Block("text", "", (10, 40, 900, 60), "next")])
    assert not sane([whole, line])


def test_an_empty_result_is_a_failure():
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle((50, 50, 350, 70), fill="black")

    def nothing(crop, rect, pad):
        return RegionOutput(True, "<PAGE>")

    with pytest.raises(RescueFailed):
        rescue_page(image, nothing, max_calls=5)
