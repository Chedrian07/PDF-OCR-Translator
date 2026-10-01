"""pdf_fonts.enrich_layout_fonts — 원본 PDF 텍스트 레이어에서 실측 폰트 크기 주입.

합성 PDF(612×792pt)를 테스트 안에서 만든다:
- 본문: (100,200)에 11pt로 삽입 → span bbox 세로 ≈188–203pt, 중심 ≈(356,196)
- 제목: (100,100)에 16pt Helvetica-Bold(hebo)로 삽입 → 중심 ≈(136,94)
insert_text의 y는 **베이스라인**이므로 det 사각형은 넉넉히 잡는다.
det bbox는 0–999 정규화 = (x/612×999, y/792×999).
"""

from pathlib import Path

from app.pipeline.pdf_fonts import _font_style, enrich_layout_fonts


def _make_pdf(tmp_path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 200), "body text ..." * 10, fontsize=11, fontname="tiro")
    page.insert_text((100, 100), "Bold Title", fontsize=16, fontname="hebo")  # Helvetica-Bold
    p = tmp_path / "source.pdf"
    doc.save(str(p))
    doc.close()
    return p


def _norm(x_pt: float, y_pt: float) -> tuple[int, int]:
    return round(x_pt / 612 * 999), round(y_pt / 792 * 999)


def test_font_style_recognizes_urw_nimbus_sans_alias():
    """실제 샘플의 대표 제목 폰트명(NimbusSanL)을 sans로 보존한다."""
    assert _font_style("HIYFFY+NimbusSanL-Bold") == "sans"


def test_enrich_injects_measured_font_sizes(tmp_path):
    pdf = _make_pdf(tmp_path)
    # 제목 사각형 (80,80)–(400,120)pt, 본문 (80,180)–(600,215)pt
    tx1, ty1 = _norm(80, 80)
    tx2, ty2 = _norm(400, 120)
    bx1, by1 = _norm(80, 180)
    bx2, by2 = _norm(600, 215)
    # 빈 영역 (80,400)–(500,500)pt, 이미지 블록은 임의 bbox + image 키
    ex1, ey1 = _norm(80, 400)
    ex2, ey2 = _norm(500, 500)

    pages = [{
        "page": 1, "width": 1000, "height": 1294,  # 픽셀 크기(무관 — enrich는 pt 사용)
        "blocks": [
            {"type": "title", "bbox": [tx1, ty1, tx2, ty2], "content": "Bold Title"},
            {"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "body text"},
            {"type": "image", "bbox": [tx1, ty1, tx2, ty2], "content": "", "image": "p0001_0.jpg"},
            {"type": "text", "bbox": [ex1, ey1, ex2, ey2], "content": "빈 영역"},
        ],
    }]

    changed = enrich_layout_fonts(pdf, pages)
    assert changed is True

    title, body, image, empty = pages[0]["blocks"]

    # 본문 11pt → 11/612×100 = 1.80cqw ±0.15
    assert "fs" in body
    assert abs(body["fs"] - 11 / 612 * 100) < 0.15, body["fs"]
    assert body.get("font_style") == "serif"

    # 제목 16pt → 16/612×100 = 2.61cqw ±0.2, 볼드
    assert "fs" in title
    assert abs(title["fs"] - 16 / 612 * 100) < 0.2, title["fs"]
    assert title.get("bold") is True
    assert title.get("font_style") == "sans"

    # 이미지 블록은 손대지 않는다
    assert "fs" not in image and "bold" not in image

    # 빈 영역 블록엔 span이 없어 fs 미주입
    assert "fs" not in empty and "bold" not in empty


def test_enrich_empty_page_stamps_version_without_fs(tmp_path):
    """텍스트 레이어 없는(스캔) 페이지: fs는 못 심지만 fonts_v는 스탬프한다 —
    매 요청 재스캔을 막기 위해 True(변경됨)를 반환하는 것이 계약."""
    import fitz

    from app.pipeline.pdf_fonts import ENRICH_VERSION

    doc = fitz.open()
    doc.new_page(width=612, height=792)  # 텍스트 없음
    p = tmp_path / "empty.pdf"
    doc.save(str(p))
    doc.close()

    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [100, 100, 900, 300], "content": "무엇이든"},
    ]}]
    assert enrich_layout_fonts(p, pages) is True   # 스탬프만으로도 저장 필요
    assert "fs" not in pages[0]["blocks"][0]
    assert pages[0]["fonts_v"] == ENRICH_VERSION


def test_enrich_corrupt_pdf_returns_false(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [0, 0, 999, 999], "content": "x"},
    ]}]
    assert enrich_layout_fonts(bad, pages) is False


def test_enrich_page_index_out_of_range(tmp_path):
    from app.pipeline.pdf_fonts import ENRICH_VERSION

    pdf = _make_pdf(tmp_path)
    pages = [{"page": 99, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [131, 227, 979, 271], "content": "x"},
    ]}]
    # 범위를 벗어난 페이지는 아무것도 주입하지 않지만 스탬프는 찍는다 — 스탬프가
    # 없으면 백필 호출부가 매 요청 전 문서를 재스캔한다. 스탬프만으로도 저장이
    # 필요하므로 True다.
    assert enrich_layout_fonts(pdf, pages) is True
    assert "fs" not in pages[0]["blocks"][0]
    assert pages[0]["fonts_v"] == ENRICH_VERSION


def test_enrich_stamps_pages_it_had_to_skip(tmp_path, monkeypatch):
    """텍스트 추출이 예외를 던진 페이지·번호가 깨진 페이지도 스탬프한다.

    스탬프 없이 빠져나가면 `_backfill_layout_fonts`의 미스탬프 검사가 매 요청
    참이 되어, 전 문서 재스캔 → layout 재기록 → 내보내기 PDF 캐시 무효화가
    끝없이 반복된다(손상 PDF로 재현된 루프). 정상 페이지의 주입은 그대로다.
    """
    import fitz

    from app.pipeline.pdf_fonts import ENRICH_VERSION

    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((100, 200), "first page", fontsize=11)
    doc.new_page(width=612, height=792).insert_text((100, 200), "second page", fontsize=11)
    pdf = tmp_path / "two.pdf"
    doc.save(pdf)
    doc.close()

    real_get_text = fitz.Page.get_text

    def flaky_get_text(self, *args, **kwargs):
        if self.number == 1:
            raise RuntimeError("MuPDF가 이 페이지에서 예외를 낸다")
        return real_get_text(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_text", flaky_get_text)
    bx1, by1 = _norm(80, 180)
    bx2, by2 = _norm(600, 215)
    pages = [
        {"page": 1, "blocks": [{"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "x"}]},
        {"page": 2, "blocks": [{"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "y"}]},
        {"page": "not-a-number", "blocks": []},
    ]
    assert enrich_layout_fonts(pdf, pages) is True
    assert "fs" in pages[0]["blocks"][0]
    assert "fs" not in pages[1]["blocks"][0]
    assert [page["fonts_v"] for page in pages] == [ENRICH_VERSION] * 3

    # 백필 호출부의 재실행 조건(api._backfill_layout_fonts와 같은 식)이 꺼진다.
    assert not any(int(page.get("fonts_v") or 0) < ENRICH_VERSION for page in pages)
    # 다시 불러도 바뀔 것이 없으면 저장을 요구하지 않는다(스탬프가 이미 최신).
    assert enrich_layout_fonts(pdf, pages[1:]) is False


def test_enrich_detects_vertical_text_and_stamps_version(tmp_path):
    """90° 회전 텍스트(arXiv 여백 스탬프)는 줄 dir로 감지 → vertical="up".
    처리된 페이지에는 fonts_v 버전이 스탬프된다."""
    import fitz

    from app.pipeline.pdf_fonts import ENRICH_VERSION

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # rotate=90: (30, 700)에서 위쪽으로 진행하는 세로쓰기
    page.insert_text((30, 700), "arXiv:1908.07836v1 [cs.CL] 16 Aug 2019", fontsize=9, rotate=90)
    page.insert_text((100, 200), "normal horizontal body text " * 5, fontsize=11)
    p = tmp_path / "source.pdf"
    doc.save(str(p))
    doc.close()

    x1, y1 = _norm(15, 380)
    x2, y2 = _norm(45, 720)
    bx1, by1 = _norm(80, 170)
    bx2, by2 = _norm(520, 220)
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [x1, y1, x2, y2], "content": "arXiv:1908.07836v1"},
        {"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "normal body"},
    ]}]
    assert enrich_layout_fonts(p, pages) is True
    vert, horiz = pages[0]["blocks"]
    assert vert.get("vertical") == "up", vert
    assert abs(vert["fs"] - 9 / 612 * 100) < 0.15
    assert "vertical" not in horiz
    assert pages[0]["fonts_v"] == ENRICH_VERSION


def test_enrich_detects_centered_and_justified_blocks(tmp_path):
    """저자/대표 제목의 가운데 정렬과 본문 양쪽 정렬을 원문 기하에서 복원한다."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    centered_rect = fitz.Rect(120, 70, 492, 145)
    page.insert_textbox(
        centered_rect,
        "Short author\nA considerably longer affiliation line\nCenter",
        fontsize=11,
        align=1,
    )
    body_rect = fitz.Rect(80, 220, 532, 340)
    page.insert_textbox(
        body_rect,
        "A justified paragraph has enough words to fill several complete lines. "
        "The final line is intentionally shorter while the preceding lines reach "
        "both edges of the source text box. Additional material makes the paragraph "
        "long enough for reliable multi-line alignment detection in this wide box. "
        "One more sentence leaves only the final source line intentionally short.",
        fontsize=11,
        align=3,
    )
    p = tmp_path / "align.pdf"
    doc.save(p)
    doc.close()

    cx1, cy1 = _norm(centered_rect.x0, centered_rect.y0)
    cx2, cy2 = _norm(centered_rect.x1, centered_rect.y1)
    bx1, by1 = _norm(body_rect.x0, body_rect.y0)
    bx2, by2 = _norm(body_rect.x1, body_rect.y1)
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [cx1, cy1, cx2, cy2], "content": "author"},
        {"type": "text", "bbox": [bx1, by1, bx2, by2], "content": "paragraph"},
    ]}]

    assert enrich_layout_fonts(p, pages) is True
    assert pages[0]["blocks"][0].get("align") == "center"
    assert pages[0]["blocks"][1].get("align") == "justify"


def _display_bbox(page, rect) -> list[int]:
    """표시(회전 반영) 공간 pt 사각형 → det와 같은 0–999 좌표."""
    width, height = page.rect.width, page.rect.height
    return [
        round(rect.x0 / width * 999), round(rect.y0 / height * 999),
        round(rect.x1 / width * 999), round(rect.y1 / height * 999),
    ]


def test_enrich_rotated_page_compares_in_display_space(tmp_path):
    """/Rotate 90 페이지: det bbox(표시 공간)와 span(비회전 공간)을 같은 좌표계에서 비교한다.

    예전에는 두 공간을 섞어 회전 페이지의 모든 블록에서 fs·정렬이 빠지고, 화면에서
    가로로 읽히는 줄을 세로쓰기로 오판해 번역 PDF가 그 블록을 원문으로 남기거나
    최후 수단 크기(≈0.7배)로 찍었다. 화면에서 세로인 여백 스탬프는 여전히 세로다.
    """
    import fitz

    from app.pipeline.pdf_fonts import ENRICH_VERSION

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.set_rotation(90)
    lines = [
        (fitz.Point(60, 80), "Table 3 reports accuracy across all evaluated datasets."),
        (fitz.Point(60, 120), "Our approach consistently outperforms the strongest baseline."),
    ]
    blocks = []
    for point, text in lines:
        # 표시 공간의 점을 비회전 좌표로 옮기고 페이지 회전만큼 돌려 쓰면 화면에서 똑바로 보인다.
        page.insert_text(point * page.derotation_matrix, text, fontsize=10, fontname="tiro", rotate=90)
        width = fitz.get_text_length(text, "tiro", 10)
        shown = fitz.Rect(point.x - 1, point.y - 10, point.x + width + 1, point.y + 3)
        blocks.append({"type": "text", "bbox": _display_bbox(page, shown), "content": text})
    # 화면에서 아래→위로 읽히는 여백 스탬프(arXiv 식별자): 비회전 공간에서는 180° 진행.
    stamp_origin = fitz.Point(30, 560)
    page.insert_text(
        stamp_origin * page.derotation_matrix, "arXiv:2504.19874v1 [cs.CL] 28 Apr 2025",
        fontsize=9, fontname="tiro", rotate=180,
    )
    stamp = fitz.Rect(18, 330, 34, 565)
    blocks.append({"type": "text", "bbox": _display_bbox(page, stamp), "content": "arXiv:2504.19874v1"})
    pdf = tmp_path / "rotated.pdf"
    doc.save(pdf)
    doc.close()

    pages = [{"page": 1, "blocks": blocks}]
    assert enrich_layout_fonts(pdf, pages) is True
    body, second, margin = pages[0]["blocks"]
    for block in (body, second):
        # cqw는 **표시** 폭(842pt)의 1% 단위다.
        assert abs(block["fs"] - 10 / 842 * 100) < 0.1, block
        assert block.get("font_style") == "serif", block
        assert "vertical" not in block, block
    assert margin.get("vertical") == "up", margin
    assert pages[0]["fonts_v"] == ENRICH_VERSION


def test_enrich_version_was_bumped_for_rotated_page_fix():
    """회전 페이지 메타가 바뀌었으므로 기존 잡도 다시 백필되도록 버전을 올렸다."""
    from app.pipeline.pdf_fonts import ENRICH_VERSION

    assert ENRICH_VERSION >= 6
