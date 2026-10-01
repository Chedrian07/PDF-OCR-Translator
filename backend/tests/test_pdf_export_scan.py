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
    with fitz.open(job_dir / "source.pdf") as source:
        left, before = _ink_beyond_translation(source[0], result.path, stamp_rect)
    assert before > 100 and left >= before * 0.9, (left, before)   # 스탬프 픽셀은 그대로
