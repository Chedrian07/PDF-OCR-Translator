"""스캔(래스터) 원문 위 번역 — 원문 픽셀을 덮고 나서 한국어를 넣는다.

텍스트 레이어가 없거나 투명 OCR 텍스트만 얹힌 스캔에서는 텍스트 리댁션으로 지울
원문이 없다. 예전에는 모든 블록이 교체 성공으로 집계되면서 영어 스캔 글자 위에
한국어가 겹쳐 찍혀 둘 다 읽을 수 없었다. 회귀 테스트는 렌더한 PNG만으로 만든
페이지로 '블록 영역의 영어 픽셀이 번역 글자 자리 말고는 남지 않는다'를 검사한다.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.pipeline.pdf_export import build_translated_pdf

PAGE_W, PAGE_H = 595.0, 842.0
SCAN_DPI = 150
PAPER = (247, 243, 228)
INK = (25, 25, 25)
LINES = [
    "Scanned papers are a core input for OCR products, and",
    "the translated PDF must not print Korean on top of",
    "the original English pixels of the scanned page.",
]
BLOCK = fitz.Rect(70, 108, 350, 152)
EQUATION = fitz.Rect(148, 178, 330, 194)
KO = (
    "스캔 논문은 OCR 제품의 핵심 입력이며, 번역 PDF는 스캔 페이지의 원래 영어 픽셀 위에 "
    "한국어를 겹쳐 찍어서는 안 된다. 이 문장은 원문보다 조금 더 길어서 아래로 자랄 수 있다."
)


def _font(size: float):
    for path in (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, int(size))
    return ImageFont.load_default(size=int(size))


def _scan_png(width=PAGE_W, height=PAGE_H, *, lines=LINES, top=120.0, extra=None) -> bytes:
    """영어 문단(과 수식 한 줄)을 그려 넣은 스캔 이미지 — 텍스트 레이어 없음."""
    scale = SCAN_DPI / 72
    image = Image.new("RGB", (int(width * scale), int(height * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    font = _font(11 * scale)
    for index, line in enumerate(lines):
        draw.text((72 * scale, (top + index * 15 - 9) * scale), line, fill=INK, font=font)
    for x, y, text in extra or ():
        draw.text((x * scale, (y - 9) * scale), text, fill=INK, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _bbox(rect: fitz.Rect) -> list[float]:
    return [
        rect.x0 / PAGE_W * 999, rect.y0 / PAGE_H * 999,
        rect.x1 / PAGE_W * 999, rect.y1 / PAGE_H * 999,
    ]


def _write_layout(job_dir: Path, blocks: list[dict], translations: dict[int, str]) -> None:
    original = [{"page": 1, "width": PAGE_W, "height": PAGE_H, "blocks": blocks}]
    translated = json.loads(json.dumps(original))
    for index, text in translations.items():
        translated[0]["blocks"][index]["content"] = text
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )


def _scan_job(
    tmp_path: Path, name: str, *, invisible_text: bool = False, translation: str = KO,
    font_pt: float | None = None,
) -> Path:
    job_dir = tmp_path / name
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(
        page.rect, stream=_scan_png(extra=[(150, 190, "E = mc^2 (kept equation)")]),
    )
    if invisible_text:
        # Acrobat/ABBYY/ocrmypdf 식 투명 OCR 레이어(렌더 모드 3).
        for index, line in enumerate(LINES):
            page.insert_text(
                (72, 120 + index * 15), line, fontsize=11, fontname="tiro", render_mode=3,
            )
    doc.save(job_dir / "source.pdf")
    doc.close()
    text_block = {"type": "text", "bbox": _bbox(BLOCK), "content": "\n".join(LINES)}
    if font_pt is not None:
        text_block["fs"] = font_pt / PAGE_W * 100
    _write_layout(job_dir, [
        text_block,
        {"type": "equation", "bbox": _bbox(EQUATION), "content": r"\[E = mc^2\]"},
    ], {0: translation})
    return job_dir


def _dark_pixels(page, clip: fitz.Rect, dpi: int = 100) -> set[tuple[int, int]]:
    pixmap = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY, alpha=False)
    samples = pixmap.samples
    return {
        (x, y)
        for y in range(pixmap.height)
        for x in range(pixmap.width)
        if samples[y * pixmap.stride + x] < 110
    }


def _ink_beyond_translation(source_page, exported_path: Path, region: fitz.Rect) -> tuple[int, int]:
    """번역 글자 말고 영역에 남은 어두운 픽셀 수(와 원문의 어두운 픽셀 수).

    같은 내보내기 결과를 스캔 이미지째 그린 것과 이미지를 지우고(번역·덮개만) 그린
    것의 어두운 픽셀 수 차이다. 덮지 않았다면 차이가 영어 픽셀 수만큼 크고, 덮었다면
    0에 가깝다. 좌표를 맞대 비교하지 않는 것은 MuPDF가 텍스트를 먼저 추출한 페이지를
    1px 어긋나게 그리기도 하기 때문이다.
    """
    before = len(_dark_pixels(source_page, region))
    with fitz.open(exported_path) as exported:
        full = len(_dark_pixels(exported[0], region))
    with fitz.open(exported_path) as exported:
        page = exported[0]
        for info in page.get_images(full=True):
            page.delete_image(info[0])
        translation_only = len(_dark_pixels(page, region))
    return full - translation_only, before


@pytest.mark.parametrize("invisible_text", [False, True])
def test_scanned_page_translation_covers_the_english_pixels(tmp_path, invisible_text):
    job_dir = _scan_job(tmp_path, f"scan-{invisible_text}", invisible_text=invisible_text)

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    assert result.report()["raster_blocks_erased"] == 1
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, BLOCK)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
        images = exported[0].get_image_info()
    assert before > 500, before                       # 원문에 영어 픽셀이 실제로 있었다
    assert left <= before * 0.02, (left, before)      # 번역 글자 말고는 남지 않는다
    assert "스캔 논문은" in text, text
    assert "Scanned papers" not in text, text         # 투명 OCR 텍스트도 함께 지운다
    assert len(images) == 1                           # 스캔 이미지 객체는 그대로다


def test_scan_cover_uses_the_paper_colour_and_keeps_the_file_small(tmp_path):
    """흰 사각형이 도드라지지 않게 바탕색으로 덮고, 스캔 이미지를 다시 쓰지 않는다."""
    job_dir = _scan_job(tmp_path, "scan-colour")
    result = build_translated_pdf(job_dir, "ko")
    paper = tuple(component / 255 for component in PAPER)
    with fitz.open(result.path) as exported:
        covers = [
            drawing for drawing in exported[0].get_drawings()
            if drawing.get("fill") is not None
            and drawing["rect"].contains(BLOCK)
        ]
    assert covers, "블록 영역을 덮는 채움 사각형이 없다"
    fill = covers[0]["fill"]
    assert all(abs(a - b) <= 6 / 255 for a, b in zip(fill, paper)), fill
    # 이미지 픽셀을 다시 인코딩하지 않으므로 파일이 스캔 크기만큼 불어나지 않는다.
    assert result.path.stat().st_size < (job_dir / "source.pdf").stat().st_size + 200_000


def test_kept_scan_block_is_an_obstacle_for_growing_translations(tmp_path):
    """스캔의 보존 블록(수식 픽셀)은 span이 없어도 장애물이다 — 번역이 그 위로 자라지 않는다.

    번역이 원문보다 훨씬 길어 아래로 자라야 하는 구성이다. 장애물이 없으면 바로
    아래 수식 줄의 픽셀 위까지 한국어가 흘러내린다.
    """
    # 원문 크기(11pt)를 아는 블록이라 번역은 줄이지 않고 아래로 자라려 한다.
    job_dir = _scan_job(tmp_path, "scan-obstacle", translation=KO + " " + KO, font_pt=11)
    result = build_translated_pdf(job_dir, "ko")
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(result.path) as exported:
        equation_before = _dark_pixels(source[0], EQUATION)
        equation_after = _dark_pixels(exported[0], EQUATION)
        korean = [
            fitz.Rect(line["bbox"])
            for block in exported[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if any("가" <= ch <= "힣" for span in line["spans"] for ch in span["text"])
        ]
    assert korean
    assert not [line for line in korean if (line & EQUATION).get_area() > 0.5], korean
    # 수식 픽셀은 덮이지도, 번역 글자가 더해지지도 않았다.
    assert equation_before
    assert abs(len(equation_after) - len(equation_before)) <= len(equation_before) * 0.05


def test_visible_text_over_a_full_page_background_is_not_covered(tmp_path):
    """born-digital 텍스트가 보이는 블록은 텍스트 리댁션만 한다(배경 이미지를 덮지 않는다)."""
    job_dir = tmp_path / "visible-over-background"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=_scan_png(lines=[]))
    page.insert_text((72, 120), "Visible born-digital sentence here.", fontsize=11)
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(fitz.Rect(70, 108, 350, 124)),
         "content": "Visible born-digital sentence here."},
    ], {0: "보이는 본문 문장."})

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 0, result.report()
    assert not result.warnings, result.warnings


def test_visible_text_drawn_under_a_later_scan_image_is_treated_as_raster(tmp_path):
    """'이미지 아래 텍스트' 스캔 — 보이는 모드로 쓴 OCR 텍스트 위에 전면 스캔 이미지를 덮은 쪽.

    칠하기 비트만 보면 그 글자는 '보여서' 래스터 원문으로 잡히지 않았고, 한국어가 영어
    스캔 픽셀 위에 그대로 겹쳐 찍혔다(감사 pdf-2). 그려진 순서상 이미지에 가려진 글자는
    원문이 아니다 — 스캔 픽셀을 덮고 번역을 넣는다.
    """
    job_dir = tmp_path / "text-under-scan"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for index, line in enumerate(LINES):
        page.insert_text((72, 120 + index * 15), line, fontsize=11, fontname="tiro")
    page.insert_image(page.rect, stream=_scan_png())          # 글자 위에 덮는 스캔
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(BLOCK), "content": "\n".join(LINES)},
    ], {0: KO})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, BLOCK)
    with fitz.open(result.path) as exported:
        text = exported[0].get_text().replace("\xa0", " ")
    assert before > 500 and left <= before * 0.02, (left, before)
    assert "Scanned papers" not in text, text


def test_text_inside_a_layout_figure_on_a_scan_is_left_alone(tmp_path):
    """스캔 위라도 레이아웃 그림 블록 안의 글자는 원문 보존 — 그림 픽셀을 덮지 않는다."""
    job_dir = tmp_path / "scan-figure"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=_scan_png())
    doc.save(job_dir / "source.pdf")
    doc.close()
    figure = fitz.Rect(60, 100, 380, 170)
    _write_layout(job_dir, [
        {"type": "image", "bbox": _bbox(figure), "content": "", "image": "p0001_0.jpg"},
        {"type": "text", "bbox": _bbox(BLOCK), "content": "\n".join(LINES)},
    ], {1: KO})

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 0, result.report()
    assert result.raster_blocks_erased == 0, result.report()
    assert result.specialist_kept.get("figure_text") == 1, result.report()


def test_textless_block_partly_over_a_figure_image_is_reported(tmp_path):
    """덮지 않는(그림 일부와 겹친) 래스터 원문 위 교체는 경고로 드러난다."""
    job_dir = tmp_path / "partial-figure"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(
        fitz.Rect(60, 60, 400, 116), stream=_scan_png(400, 140, lines=["Figure label"], top=20),
        keep_proportion=False,
    )
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(fitz.Rect(70, 110, 350, 150)), "content": "Some caption"},
    ], {0: "그림 설명"})

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 0, result.report()
    assert any("이미지 픽셀이라 지우지 못함" in warning for warning in result.warnings), result.warnings


def test_table_image_cells_are_covered_before_translation(tmp_path):
    """텍스트 레이어 없는 표 이미지의 바뀐 셀은 덮은 뒤 번역을 넣는다."""
    job_dir = tmp_path / "table-image"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    table = fitz.Rect(60, 82, 530, 122)
    cells = Image.new("RGB", (470 * 2, 40 * 2), PAPER)
    draw = ImageDraw.Draw(cells)
    font = _font(18)
    for col, text in enumerate(("Method", "Acc", "F1")):
        draw.text((10 + col * 313, 6), text, fill=INK, font=font)
    for col, text in enumerate(("Ours", "12.3", "45.6")):
        draw.text((10 + col * 313, 46), text, fill=INK, font=font)
    buffer = io.BytesIO()
    cells.save(buffer, "PNG")
    page.insert_image(table, stream=buffer.getvalue(), keep_proportion=False)
    doc.save(job_dir / "source.pdf")
    doc.close()
    html = (
        "<table><tr><td>Method</td><td>Acc</td><td>F1</td></tr>"
        "<tr><td>Ours</td><td>12.3</td><td>45.6</td></tr></table>"
    )
    _write_layout(job_dir, [{"type": "table", "bbox": _bbox(table), "content": html}],
                  {0: html.replace("Method", "방법").replace("Ours", "제안")})

    result = build_translated_pdf(job_dir, "ko")
    assert result.table_cells_replaced == 2, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    with fitz.open(job_dir / "source.pdf") as source:
        method_cell = fitz.Rect(62, 83, 210, 101)
        left, before = _ink_beyond_translation(source[0], result.path, method_cell)
    assert before > 20 and left <= before * 0.05, (left, before)


def test_tall_narrow_scan_stamp_is_kept_as_vertical_text(tmp_path):
    """스캔 여백의 세로 스탬프(arXiv 식별자)는 덮고 가로로 다시 쓰지 않는다.

    텍스트 레이어가 없어 줄 방향을 모르는 스캔에서는 레이아웃 뷰와 같은 기하
    폴백(폭 대비 높이 6배 이상 + 12자 이상)으로 세로쓰기를 판정해 원문을 보존한다.
    예전에는 그 영역을 덮은 뒤 한국어를 한 글자씩 세로로 쌓아 찍었다.
    """
    job_dir = tmp_path / "scan-stamp"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    page_image = Image.open(io.BytesIO(_scan_png()))
    stamp = Image.new("RGB", (int(260 * scale), int(14 * scale)), PAPER)
    ImageDraw.Draw(stamp).text((0, 0), "arXiv:2504.19874v1 [cs.LG] 28 Apr 2025",
                               fill=INK, font=_font(11 * scale))
    page_image.paste(stamp.rotate(90, expand=True), (int(20 * scale), int(300 * scale)))
    buffer = io.BytesIO()
    page_image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(job_dir / "source.pdf")
    doc.close()
    stamp_rect = fitz.Rect(19, 299, 36, 562)
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(stamp_rect),
         "content": "arXiv:2504.19874v1 [cs.LG] 28 Apr 2025"},
        {"type": "text", "bbox": _bbox(BLOCK), "content": "\n".join(LINES)},
    ], {0: "아카이브 식별자 번역문 스물여덟 사월", 1: KO})

    result = build_translated_pdf(job_dir, "ko")
    assert result.kept_reasons.get("vertical") == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()   # 본문 블록만 덮었다
    # 모양으로만 추정한 세로쓰기라 번역이 빠진 이유를 리포트에 남긴다.
    assert any("세로쓰기" in warning for warning in result.warnings), result.warnings
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, stamp_rect)
    assert before > 100 and left >= before * 0.9, (left, before)   # 스탬프 픽셀은 그대로


@pytest.mark.parametrize("joiner", ["\n", " "])
def test_narrow_multi_line_scan_column_is_not_taken_for_vertical_text(tmp_path, joiner):
    """좁고 긴 가로쓰기 단(신문 단)은 세로쓰기가 아니다 — 덮고 번역한다.

    예전에는 종횡비(6 이상)만 봐서 74×574pt 단의 57줄 본문을 세로쓰기로 보존해 번역이
    경고 없이 통째로 빠졌다(감사 pdf-4). OCR이 단을 한 문단(줄바꿈 없음)으로 내도 글자
    수가 세로 한 줄에 들어갈 양의 수십 배라 세로쓰기가 아니다.
    """
    job_dir = tmp_path / f"scan-narrow-{joiner == ' '}"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    font = _font(8 * scale)
    words = (
        "City council approved the new budget on Monday after a long debate about "
        "public transit, school funding and road repairs. "
    ).split() * 8
    lines: list[str] = []
    for word in words:                       # 단 폭(70pt)에 맞춰 실제 글꼴 폭으로 줄바꿈
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and font.getlength(candidate) <= 70 * scale:
            lines[-1] = candidate
        else:
            lines.append(word)
    image = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((40 * scale, (60 + index * 10) * scale), line, fill=INK, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(job_dir / "source.pdf")
    doc.close()
    column = fitz.Rect(38, 58, 112, 60 + len(lines) * 10 + 2)
    assert column.height / column.width >= 6          # 종횡비만으로는 세로쓰기 후보다
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(column), "content": joiner.join(lines)},
    ], {0: "시의회는 대중교통, 학교 재정, 도로 보수를 둘러싼 긴 논쟁 끝에 월요일 새 예산을 승인했다. " * 6})

    result = build_translated_pdf(job_dir, "ko")

    assert "vertical" not in result.kept_reasons, result.report()
    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    assert not any("세로쓰기" in warning for warning in result.warnings), result.warnings


# ── 스캔 표: 픽셀로 찾은 실제 열·행 경계 ─────────────────────────────────────
# 열 폭이 50/90/340pt로 다른 표(감사 pdf-1 재현). 균등 격자(160pt씩)로 덮으면 바뀌지
# 않은 'DOI' 머리글과 괘선이 지워지고 '제목'이 엉뚱한 열(x≈382)에 찍혔다.
TABLE_COLUMNS = (60, 110, 200, 540)
TABLE_ROWS = (100, 118, 136)
TABLE_RULES = (100, 118, 154)
TABLE_RECT = fitz.Rect(60, 100, 540, 154)
TABLE_HEADER = ("Year", "DOI", "Title")
TABLE_DATA = ("2021", "10.1145/3442", "Learning to translate scanned tables")


def _table_html(rows) -> str:
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    ) + "</table>"


def _rotated_copy(doc, rotation: int):
    """화면에 보이는 모습은 그대로 두고 /Rotate만 건 사본 — 회전 스캔 페이지."""
    if not rotation:
        return doc
    rotated = fitz.open()
    page = rotated.new_page(width=PAGE_H, height=PAGE_W)
    page.show_pdf_page(page.rect, doc, 0, rotate=rotation)
    page.set_rotation(rotation)
    return rotated


def _uneven_table_job(tmp_path: Path, rotation: int = 0) -> Path:
    job_dir = tmp_path / f"scan-table-{rotation}"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    image = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    font = _font(10 * scale)
    for row, values in zip(TABLE_ROWS, (TABLE_HEADER, TABLE_DATA)):
        for x, text in zip(TABLE_COLUMNS, values):
            draw.text(((x + 4) * scale, (row + 3) * scale), text, fill=INK, font=font)
    for y in TABLE_RULES:
        draw.line(
            [(TABLE_COLUMNS[0] * scale, y * scale), (TABLE_COLUMNS[-1] * scale, y * scale)],
            fill=(0, 0, 0), width=2,
        )
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    _rotated_copy(doc, rotation).save(job_dir / "source.pdf")
    original = _table_html([TABLE_HEADER, TABLE_DATA])
    translated = _table_html([
        ("연도", "DOI", "제목"), ("2021", "10.1145/3442", "스캔한 표를 번역하는 법 배우기"),
    ])
    _write_layout(job_dir, [{"type": "table", "bbox": _bbox(TABLE_RECT), "content": original}],
                  {0: translated})
    return job_dir


def _korean_line_rects(page) -> dict[str, fitz.Rect]:
    """한글 span의 화면(표시 공간) 사각형 — 회전 페이지에서도 화면 기준으로 비교한다."""
    out: dict[str, fitz.Rect] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].replace("\xa0", " ").strip()
                if any("가" <= ch <= "힣" for ch in text):
                    shown = fitz.Rect(span["bbox"]) * page.rotation_matrix
                    shown.normalize()
                    out[text] = shown
    return out


@pytest.mark.parametrize("rotation", [0, 90])
def test_scan_table_with_uneven_columns_covers_only_the_changed_cells(tmp_path, rotation):
    """바뀌지 않은 셀 글자와 괘선은 그대로, 바뀐 셀의 영어만 지우고 제 열에 번역을 넣는다."""
    job_dir = _uneven_table_job(tmp_path, rotation)

    result = build_translated_pdf(job_dir, "ko")

    assert result.table_cells_replaced == 3, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    assert not result.warnings, result.warnings
    doi_column = fitz.Rect(110, 101, 198, 135)        # 'DOI'·'10.1145/3442' — 바뀌지 않는다
    rules = [fitz.Rect(60, y - 1.5, 540, y + 1.5) for y in TABLE_RULES]
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(result.path) as exported:
        for region in [doi_column, *rules]:
            before = len(_dark_pixels(source[0], region))
            after = len(_dark_pixels(exported[0], region))
            assert before > 20, (region, before)
            assert abs(after - before) <= before * 0.03, (region, before, after)
        placed = _korean_line_rects(exported[0])
        year_left, year_before = _ink_beyond_translation(
            source[0], result.path, fitz.Rect(60, 101, 105, 117),
        )
        title_left, title_before = _ink_beyond_translation(
            source[0], result.path, fitz.Rect(200, 119, 540, 153),
        )
    assert year_left <= year_before * 0.02, (year_left, year_before)
    assert title_left <= title_before * 0.02, (title_left, title_before)
    # 번역은 원문이 있던 열에 앉는다(균등 격자면 '제목'이 x≈382에 찍혔다).
    assert 196 <= placed["제목"].x0 <= 215, placed
    assert 56 <= placed["연도"].x0 <= 75, placed


def test_scan_table_whose_columns_cannot_be_told_from_word_gaps_is_kept(tmp_path):
    """열 사이 빈 띠가 단어 사이 간격과 구분되지 않으면 덮지 않고 원문 표를 보존한다."""
    job_dir = tmp_path / "scan-table-ambiguous"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    image = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    font = _font(10 * scale)
    left = "Alpha beta"
    # 둘째 열을 첫 열 끝에서 정확히 공백 한 칸 뒤에 둔다 — 픽셀만으로는 어느 빈 띠가 열
    # 경계인지('Alpha'|'beta' vs 'beta'|'Gamma') 알 수 없다.
    second_x = 64 * scale + font.getlength(left + " ")
    draw.text((64 * scale, 103 * scale), left, fill=INK, font=font)
    draw.text((second_x, 103 * scale), "Gamma", fill=INK, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(job_dir / "source.pdf")
    doc.close()
    table = fitz.Rect(60, 100, 200, 118)
    _write_layout(job_dir, [{
        "type": "table", "bbox": _bbox(table), "content": _table_html([(left, "Gamma")]),
    }], {0: _table_html([("알파 베타", "감마")])})

    result = build_translated_pdf(job_dir, "ko")

    assert result.table_cells_replaced == 0, result.report()
    assert result.raster_blocks_erased == 0, result.report()
    assert result.kept_reasons.get("table_grid_untrusted") == 1, result.report()
    assert any("스캔 표" in warning for warning in result.warnings), result.warnings
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(result.path) as exported:
        assert not [d for d in exported[0].get_drawings() if d.get("fill") is not None]
        before = len(_dark_pixels(source[0], table))
        after = len(_dark_pixels(exported[0], table))
    assert before > 50 and after == before, (before, after)


def test_scan_table_vertical_rules_survive_the_cell_covers(tmp_path):
    """세로 괘선이 있는 격자 표 — 덮개는 셀 글자만 지우고 세로·가로 괘선은 남긴다."""
    job_dir = tmp_path / "scan-table-boxed"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    xs, ys = (60, 200, 300, 420), (100, 122, 144)
    image = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    font = _font(10 * scale)
    for x in xs:
        draw.line([(x * scale, ys[0] * scale), (x * scale, ys[-1] * scale)], fill=(0, 0, 0), width=2)
    for y in ys:
        draw.line([(xs[0] * scale, y * scale), (xs[-1] * scale, y * scale)], fill=(0, 0, 0), width=2)
    rows = (("Dataset", "Size", "Language"), ("WMT14", "4.5M", "English-German"))
    for row, values in zip(ys, rows):
        for x, text in zip(xs, values):
            draw.text(((x + 6) * scale, (row + 5) * scale), text, fill=INK, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(job_dir / "source.pdf")
    doc.close()
    table = fitz.Rect(60, 100, 420, 144)
    _write_layout(job_dir, [{
        "type": "table", "bbox": _bbox(table), "content": _table_html(rows),
    }], {0: _table_html((("데이터셋", "크기", "언어"), ("WMT14", "4.5M", "영어-독일어")))})

    result = build_translated_pdf(job_dir, "ko")

    assert result.table_cells_replaced == 4, result.report()
    assert not result.warnings, result.warnings
    verticals = [fitz.Rect(x - 1.5, ys[0] + 2, x + 1.5, ys[-1] - 2) for x in xs]
    horizontals = [fitz.Rect(xs[0] + 2, y - 1.5, xs[-1] - 2, y + 1.5) for y in ys]
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(result.path) as exported:
        for region in verticals + horizontals:
            before = len(_dark_pixels(source[0], region))
            after = len(_dark_pixels(exported[0], region))
            assert before > 20 and abs(after - before) <= before * 0.03, (region, before, after)
        # 'English-German' 셀 안쪽(괘선 제외) — 영어 픽셀은 번역 글자 말고 남지 않는다.
        left, before = _ink_beyond_translation(
            source[0], result.path, fitz.Rect(302, 124, 418, 142),
        )
    assert before > 20 and left <= before * 0.02, (left, before)


def test_scan_cover_spares_an_overlapping_figure_and_its_axis_labels(tmp_path):
    """캡션 bbox가 그림 bbox와 몇 pt 겹쳐도 덮개는 그림 픽셀(눈금 라벨)을 지우지 않는다.

    OCR bbox는 0–999 격자라 캡션·그림이 겹치기 쉽다. 예전 덮개는 캡션 bbox(+0.8pt)를
    그대로 덮어 그림 안쪽 x축 눈금 라벨의 아랫부분을 지웠다(감사 pdf-3: 261→77px).
    """
    job_dir = tmp_path / "scan-figure-caption"
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    image = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), PAPER)
    draw = ImageDraw.Draw(image)
    font = _font(9 * scale)
    draw.rectangle([80 * scale, 110 * scale, 360 * scale, 285 * scale], outline=(0, 0, 0), width=2)
    for index, label in enumerate(("0", "10", "20", "30", "40", "50")):
        draw.text(((80 + index * 56) * scale, 288 * scale), label, fill=INK, font=font)
    caption = "Figure 1: Accuracy of the scanned model over epochs."
    draw.text((120 * scale, 304 * scale), caption, fill=INK, font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "image", "bbox": _bbox(fitz.Rect(70, 100, 370, 300)), "content": "",
         "image": "p0001_0.jpg"},
        # 캡션 bbox가 그림 bbox와 6pt 겹친다(그림 안쪽 294~300pt에 눈금 라벨이 있다).
        {"type": "image_caption", "bbox": _bbox(fitz.Rect(70, 294, 370, 318)), "content": caption},
    ], {1: "그림 1: 에포크에 따른 스캔 모델의 정확도."})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    assert not result.warnings, result.warnings
    labels = fitz.Rect(78, 286, 372, 300)
    with fitz.open(job_dir / "source.pdf") as source, fitz.open(result.path) as exported:
        before = len(_dark_pixels(source[0], labels))
        after = len(_dark_pixels(exported[0], labels))
        left, english = _ink_beyond_translation(
            source[0], result.path, fitz.Rect(70, 301, 370, 318),
        )
    assert before > 50 and after == before, (before, after)           # 눈금 라벨 그대로
    assert english > 100 and left <= english * 0.02, (left, english)  # 캡션 영어는 덮였다


def test_scan_block_inside_a_kept_block_is_reported_instead_of_silently_overprinted(tmp_path):
    """덮을 영역이 전부 남는 블록 안이면 덮지 않는다 — 겹쳐 보일 수 있다고 알린다."""
    job_dir = _scan_job(tmp_path, "scan-nested")
    layout = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
    translated = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    # 바깥 블록은 번역이 원문과 같아 남는다(교체되지 않는다).
    outer = {"type": "text", "bbox": _bbox(fitz.Rect(60, 96, 380, 164)), "content": "Outer note"}
    layout[0]["blocks"].append(outer)
    translated[0]["blocks"].append(dict(outer))
    (job_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")

    assert result.raster_blocks_erased == 0, result.report()
    assert any("덮지 못함" in warning for warning in result.warnings), result.warnings


def test_scan_blocks_inside_an_empty_list_container_are_still_covered(tmp_path):
    """내용이 빈 컨테이너(자식 항목을 감싼 list)는 지킬 픽셀이 없다 — 자식 교체는 그대로 덮는다."""
    job_dir = _scan_job(tmp_path, "scan-container")
    layout = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
    translated = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    container = {"type": "list", "bbox": _bbox(fitz.Rect(60, 96, 380, 164)), "content": ""}
    layout[0]["blocks"].append(container)
    translated[0]["blocks"].append(dict(container))
    (job_dir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")

    result = build_translated_pdf(job_dir, "ko")

    assert result.raster_blocks_erased == 1, result.report()
    assert not result.warnings, result.warnings
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, BLOCK)
    assert before > 500 and left <= before * 0.02, (left, before)


def _tiled_scan_job(tmp_path: Path, name: str, *, strips: int = 1, margin: float = 0.0) -> Path:
    """같은 스캔을 가로 띠 여러 장으로 나눠 넣거나, 여백을 두고 한 장으로 놓은 쪽."""
    job_dir = tmp_path / name
    job_dir.mkdir()
    scale = SCAN_DPI / 72
    image = Image.open(io.BytesIO(_scan_png()))
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    if margin:
        # 여백만큼 줄여 놓되 글자는 페이지의 같은 자리에 오게 그 영역만 잘라 넣는다.
        crop = image.crop((
            int(margin * scale), int(margin * scale),
            int((PAGE_W - margin) * scale), int((PAGE_H - margin) * scale),
        ))
        buffer = io.BytesIO()
        crop.save(buffer, "PNG")
        page.insert_image(
            fitz.Rect(margin, margin, PAGE_W - margin, PAGE_H - margin),
            stream=buffer.getvalue(), keep_proportion=False,
        )
    else:
        step = image.height // strips
        for index in range(strips):
            bottom = image.height if index == strips - 1 else (index + 1) * step
            buffer = io.BytesIO()
            image.crop((0, index * step, image.width, bottom)).save(buffer, "PNG")
            page.insert_image(
                fitz.Rect(0, PAGE_H * index * step / image.height, PAGE_W, PAGE_H * bottom / image.height),
                stream=buffer.getvalue(), keep_proportion=False,
            )
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(BLOCK), "content": "\n".join(LINES)},
    ], {0: KO})
    return job_dir


@pytest.mark.parametrize(("strips", "margin"), [(2, 0.0), (4, 0.0), (1, 30.0), (1, 60.0)])
def test_tiled_or_margined_scans_are_still_scan_backgrounds(tmp_path, strips, margin):
    """띠로 나뉜 스캔·여백을 둔 스캔(면적 85% 미만)도 스캔 배경이다 — 덮고 번역한다.

    예전에는 래스터 한 장이 페이지의 85% 이상일 때만 스캔으로 봐서, 이런 쪽은 모든 블록이
    '그림 위 텍스트'로 보존되고 번역이 하나도 들어가지 않았다(감사 pdf-9).
    """
    job_dir = _tiled_scan_job(tmp_path, f"tiled-{strips}-{int(margin)}", strips=strips, margin=margin)

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.raster_blocks_erased == 1, result.report()
    assert "figure_text" not in result.kept_reasons, result.report()
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, BLOCK)
    assert before > 500 and left <= before * 0.02, (left, before)


def test_large_image_with_visible_text_over_it_is_still_a_figure(tmp_path):
    """보이는 글자가 위에 얹힌 큰 이미지(born-digital 그림)는 스캔 배경이 아니다 — 그대로 보존."""
    job_dir = tmp_path / "big-figure"
    job_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    figure = fitz.Rect(40, 80, 555, 600)                      # 페이지의 53%
    page.insert_image(figure, stream=_scan_png(lines=[]), keep_proportion=False)
    page.insert_text((72, 120), "Encoder block with attention", fontsize=11)
    doc.save(job_dir / "source.pdf")
    doc.close()
    _write_layout(job_dir, [
        {"type": "text", "bbox": _bbox(fitz.Rect(70, 108, 350, 124)),
         "content": "Encoder block with attention"},
    ], {0: "어텐션이 있는 인코더 블록"})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 0, result.report()
    assert result.kept_reasons.get("figure_text") == 1, result.report()
