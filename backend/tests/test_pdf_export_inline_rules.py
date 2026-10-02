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
import pytest

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


def test_zero_thickness_rules_inside_a_figure_are_not_owned():
    """get_drawings는 스트로크 선('s'·'l')을 높이 0인 사각형으로 보고한다. 빈 사각형의 intersects는
    늘 거짓이라 '그림 영역에 걸친 선은 건드리지 않는다' 가드가 가장 흔한 TeX·matplotlib 선에서
    꺼져 있었다 — 같은 자리의 0.4pt 채움 사각형만 걸렸다(delta-pdf-translate-6)."""
    from app.pipeline.pdf_export import build as build_mod

    figure = fitz.Rect(148, 158, 452, 398)
    caption = fitz.Rect(140, 385, 460, 420)         # OCR bbox가 그림 아래쪽에 몇 pt 겹친 캡션
    blocks = [{"type": "text", "content": "Figure 1: results"}]
    legend = fitz.Rect(160, 392, 185, 392)           # 그림 안 범례 선 — 높이 0
    assert legend.is_empty and not legend.intersects(figure) and figure.contains(legend)
    assert build_mod._inline_rule_owner(legend, [caption], blocks, [figure]) is None
    filled = fitz.Rect(160, 391.8, 185, 392.2)       # 대조군: 두께 있는 같은 선
    assert build_mod._inline_rule_owner(filled, [caption], blocks, [figure]) is None
    # 그림 밖(캡션 글줄 안)의 두께 0 분수선은 여전히 캡션이 주인이다
    fraction = fitz.Rect(300, 410, 306, 410)
    assert build_mod._inline_rule_owner(fraction, [caption], blocks, [figure]) == 0


def test_translating_a_caption_keeps_the_figure_legend_line(tmp_path):
    """그림 아래쪽에 OCR bbox가 몇 pt 겹친 캡션을 번역해도 그림 속 범례 선(높이 0 스트로크)은
    남는다. 예전에는 캡션이 그 선을 '자기 수식 선'으로 소유해 캡션 교체와 함께 지웠다."""
    job_dir, doc, page = _new_source(tmp_path, "legend")
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(150, 160, 450, 380))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()
    shape = page.new_shape()
    shape.draw_line(fitz.Point(160, 392), fitz.Point(185, 392))          # 범례 표본 선
    shape.finish(color=(0.1, 0.3, 0.8), width=1.0, closePath=False)
    shape.commit()
    page.insert_text(fitz.Point(190, 395), "TurboQuant", fontsize=8, fontname="helv")
    caption_text = ("Figure 2: Mean squared error versus bit width for the proposed quantizer "
                    "and the baselines on synthetic data.")
    page.insert_textbox(fitz.Rect(72, 404, 540, 430), caption_text, fontsize=9, fontname="tiro")
    doc.save(job_dir / "source.pdf")
    doc.close()
    figure = fitz.Rect(148, 158, 452, 398)
    caption = fitz.Rect(70, 386, 542, 432)        # 캡션 글자보다 8pt 위(그림의 범례 줄)에서 시작
    blocks = [
        {"type": "image", "bbox": _bbox(figure), "content": ""},
        {**_block(caption, caption_text), "type": "image_caption"},
    ]
    _write_job(tmp_path, "legend", blocks,
               {1: "그림 2: 합성 데이터에서 제안한 양자화기와 기준선의 비트 폭에 따른 평균 제곱 오차."})

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1                    # 캡션은 번역문으로 바뀌었다
    legend_lines = [
        rect for rect in _drawings(result.path)
        if abs(rect.y0 - 392) < 1 and rect.x0 < 165 and rect.x1 > 180 and rect.height < 2
    ]
    assert legend_lines, _drawings(result.path)


# ── 회전 페이지(/Rotate 90·270, 내용은 돌려 그려 화면에서 똑바로) ─────────────────
# pdflscape 가로 쪽·회전 출력된 쪽은 내용 스트림을 돌려 그리고 /Rotate로 화면을 바로 세운다.
# get_drawings·span은 회전 전 좌표라 화면의 가로 분수선이 세로선으로, span 방향이 (0,-1)로
# 보고된다 — 모양 판정을 그 좌표로 하던 때는 회전 쪽의 분수선을 하나도 소유하지 못해, 번역
# 문단이 5–6pt로 축소되고 원문 분수선이 번역문 사이에 떠 있었다(delta-pdf-translate-2).


def rotate_display_preserving(path: Path, rot: int) -> None:
    """화면은 그대로 두고 내용 스트림을 돌려 그린 뒤 /Rotate로 바로 세운다(레이아웃 bbox 유효)."""
    doc = fitz.open(path)
    for page in doc:
        width, height = page.mediabox.width, page.mediabox.height
        if rot == 90:
            matrix, box = (0, 1, -1, 0, height, 0), (0, 0, height, width)
        elif rot == 270:
            matrix, box = (0, -1, 1, 0, 0, width), (0, 0, height, width)
        else:
            raise ValueError(rot)
        pre = doc.get_new_xref()
        doc.update_object(pre, "<<>>")
        doc.update_stream(pre, ("q %g %g %g %g %g %g cm\n" % matrix).encode())
        post = doc.get_new_xref()
        doc.update_object(post, "<<>>")
        doc.update_stream(post, b"\nQ\n")
        contents = " ".join("%d 0 R" % xref for xref in page.get_contents())
        doc.xref_set_key(page.xref, "Contents", "[%d 0 R %s %d 0 R]" % (pre, contents, post))
        doc.xref_set_key(page.xref, "MediaBox", "[%g %g %g %g]" % box)
        doc.xref_set_key(page.xref, "CropBox", "[%g %g %g %g]" % box)
        doc.xref_set_key(page.xref, "Rotate", str(rot))
    rotated = path.with_name(path.stem + f".rot{rot}.pdf")
    doc.save(rotated)
    doc.close()
    rotated.replace(path)


@pytest.mark.parametrize("rot", [90, 270])
def test_rotated_page_paragraph_drops_its_fraction_bar_and_keeps_full_size(tmp_path, rot):
    above = "Previous paragraph line that stays in English above the target."
    below = "Next paragraph line that stays in English below the target block."
    job_dir, doc, page = _new_source(tmp_path, f"fraction-{rot}")
    page.insert_text((72, 188), above, fontsize=10, fontname="tiro")
    _paragraph_with_fraction(page)
    page.insert_text((72, 236), below, fontsize=10, fontname="tiro")
    doc.save(job_dir / "source.pdf")
    doc.close()
    rotate_display_preserving(job_dir / "source.pdf", rot)
    with fitz.open(job_dir / "source.pdf") as source:
        assert source[0].rotation == rot
        [bar] = [fitz.Rect(d["rect"]) for d in source[0].get_drawings()]
        assert bar.width < 1 < bar.height        # 회전 전 좌표에서는 세로선으로 보고된다
    blocks = [
        _block(fitz.Rect(70, 179, 420, 191.5), above),
        _block(fitz.Rect(70, 192, 420, 227), "\n".join(LINES)),
        _block(fitz.Rect(70, 227.5, 420, 239), below),
    ]
    _write_job(tmp_path, f"fraction-{rot}", blocks, {1: KO})

    result = build_translated_pdf(job_dir, "ko")

    assert result.replaced == 1, result.report()
    assert result.warnings == [], result.warnings         # 예전: 5–6pt 축소 경고
    assert min(_korean_font_sizes(result.path)) >= 8.0
    assert _drawings(result.path) == []                   # 원문 분수선이 떠 있지 않다
