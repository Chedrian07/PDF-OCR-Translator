"""textlayer 읽기 순서 — 줄바꿈 셀로 된 표·그림 격자를 2단으로 오인하지 않는다.

셀 텍스트가 줄바꿈되는 표는 MuPDF가 셀마다 블록을 만들어 가운데에 빈 세로 띠가 생기므로
다단 판정(`_find_gutter`)이 단으로 보고 '키 전부 → 값 전부'로 재배열했다. 내용 스트림과
예전 sort=True는 행 순서였다(감사 pipeline-3). 2×3 그림 격자도 (a)(c)(e) 뒤에 (b)(d)(f)가
왔다. 좌·우 블록이 행 단위로 마주 보는 짧은 블록이면 그 띠는 내용 순서를 지킨다 — 2단 본문
(행이 우연히 맞아도 문단은 줄이 많다)은 예전처럼 왼쪽 단 → 오른쪽 단이다.
"""

from __future__ import annotations

import threading

import pymupdf as fitz

from app.pipeline.reading_order import TextBlock, order_blocks, page_text_blocks


def _b(order, x0, y0, x1, y1, text, line_height=12.0) -> TextBlock:
    return TextBlock(
        x0=x0, y0=y0, x1=x1, y1=y1, text=text, size=10.0, line_height=line_height,
        last_y0=y1 - line_height, last_y1=y1, last_x1=x1, order=order,
    )


def _texts(blocks) -> list[str]:
    return [b.text for b in blocks]


def _reorder(blocks: list[TextBlock], stream: list[str]) -> list[TextBlock]:
    """stream(텍스트 순서)대로 내용 순서를 매긴다."""
    by_text = {b.text: b for b in blocks}
    return [TextBlock(**{**by_text[t].__dict__, "order": i}) for i, t in enumerate(stream)]


# ── 순수 기하 ─────────────────────────────────────────────────────────────


def _table_page(rows: int = 8) -> list[TextBlock]:
    """1단 페이지: 문단 2개 → 2열 표(셀은 두 줄) → 문단 1개. 내용 순서는 행 순서.

    표가 페이지 텍스트 높이 대부분을 차지해 가로지르는 문단이 25% 이하다 — 다단 판정이
    거터를 찾는 모양이다."""
    blocks = [_b(0, 72, 60, 540, 84, "P1"), _b(1, 72, 90, 540, 114, "P2")]
    for r in range(rows):
        y = 130 + r * 50
        blocks.append(_b(len(blocks), 90, y, 210, y + 24, f"KEY-{r}"))
        blocks.append(_b(len(blocks), 320, y, 450, y + 24, f"VALUE-{r}"))
    blocks.append(_b(len(blocks), 72, 540, 540, 564, "P3"))
    return blocks


def test_wrapped_cell_table_keeps_its_row_order():
    ordered = _texts(order_blocks(_table_page()))
    assert ordered[:2] == ["P1", "P2"] and ordered[-1] == "P3"
    cells = ordered[2:-1]
    assert cells[:4] == ["KEY-0", "VALUE-0", "KEY-1", "VALUE-1"], cells
    assert all(cells.index(f"KEY-{r}") + 1 == cells.index(f"VALUE-{r}") for r in range(8))


def test_figure_grid_reads_subcaptions_row_by_row_then_the_caption():
    blocks = []
    for r, y in enumerate((200, 430, 660)):
        for c, x in enumerate((60, 330)):
            label = "abcdef"[r * 2 + c]
            blocks.append(_b(len(blocks), x, y, x + 220, y + 24, f"({label})"))
    blocks.append(_b(len(blocks), 60, 720, 300, 732, "Figure 7"))
    assert _texts(order_blocks(blocks)) == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "Figure 7"]


def test_grid_band_keeps_content_order_rather_than_coordinates():
    """격자로 본 띠는 내용 순서를 그대로 쓴다 — LaTeX식 2단 참고문헌처럼 짧은 블록의 행이
    우연히 맞아도 그 문서의 내용 순서(왼쪽 단 → 오른쪽 단)는 깨지지 않는다."""
    refs = []
    for r in range(6):
        y = 120 + r * 40
        refs.append(_b(0, 54, y, 293, y + 24, f"[{r + 1}]"))
        refs.append(_b(0, 317, y, 556, y + 24, f"[{r + 7}]"))
    column_major = [f"[{i}]" for i in range(1, 13)]
    assert _texts(order_blocks(_reorder(refs, column_major))) == column_major


def test_row_aligned_paragraphs_are_still_read_as_columns():
    """행이 맞더라도 블록이 긴 문단(셀보다 줄이 많다)이면 2단 본문이다 — 좌표 순으로 그린
    문서도 왼쪽 단 → 오른쪽 단으로 읽는다."""
    blocks = []
    for i in range(3):
        y = 120 + i * 150
        blocks.append(_b(0, 54, y, 293, y + 130, f"LEFT-{i + 1}"))
        blocks.append(_b(0, 317, y, 556, y + 130, f"RIGHT-{i + 1}"))
    interleaved = [b.text for b in blocks]
    assert _texts(order_blocks(_reorder(blocks, interleaved))) == [
        "LEFT-1", "LEFT-2", "LEFT-3", "RIGHT-1", "RIGHT-2", "RIGHT-3",
    ]


# ── 실제 PDF ──────────────────────────────────────────────────────────────

_PARAGRAPH = (
    "The proposed method is evaluated on three benchmarks and compared against strong "
    "baselines under identical compute budgets and data mixtures in every experiment we run"
)


def _table_pdf(path) -> None:
    """감사 재현: 1쪽 문단 2개 + 8행×2열 표(셀 텍스트가 줄바꿈된다) + 문단."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(72, 60, 540, 110), _PARAGRAPH, fontsize=10)
    page.insert_textbox(fitz.Rect(72, 115, 540, 165), _PARAGRAPH, fontsize=10)
    y = 180
    for r in range(8):
        page.insert_textbox(
            fitz.Rect(90, y, 210, y + 40), f"Setting {r}: learning rate schedule", fontsize=10,
        )
        page.insert_textbox(
            fitz.Rect(320, y, 450, y + 40), f"cosine decay to {r} percent of peak", fontsize=10,
        )
        y += 50
    page.insert_textbox(fitz.Rect(72, y + 10, 540, y + 60), _PARAGRAPH, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_textlayer_engine_keeps_key_value_rows_together(tmp_path):
    from app.config import Settings
    from app.engine.base import NullSink
    from app.engine.textlayer import TextLayerEngine
    from app.pipeline.pdf import render_pdf_pages

    job = tmp_path / "job"
    job.mkdir()
    _table_pdf(job / "source.pdf")
    pages = render_pdf_pages(job / "source.pdf", job / "pages", 72, 10)
    engine = TextLayerEngine(Settings(engine="textlayer", data_dir=tmp_path / "d"))
    md = engine.run_multi(pages, job / "work", NullSink(), threading.Event())

    marks = []
    for r in range(8):
        marks += [f"Setting {r}:", f"cosine decay to {r} percent"]
    positions = [md.index(m) for m in marks]
    assert positions == sorted(positions), md[:600]


def test_html_table_with_wrapped_cells_reads_row_by_row(tmp_path):
    """현실적인 표 렌더러(Story — HTML table)로 만든 페이지도 행 순서다."""
    rows = "".join(
        f"<tr><td>Hyperparameter {i}: the learning rate warmup schedule used for stage {i}</td>"
        f"<td>Linear warmup over {i * 100} steps followed by cosine decay to ten percent of "
        "peak value</td></tr>"
        for i in range(12)
    )
    html = (
        "<h3>Table 5: Training hyperparameters</h3><p>We list all settings below. These were "
        "held fixed across all runs reported in the paper unless stated otherwise.</p>"
        f'<table border="1" style="width:100%">{rows}</table>'
    )
    path = tmp_path / "story.pdf"
    story = fitz.Story(html=html)
    writer = fitz.DocumentWriter(str(path))
    mediabox = fitz.paper_rect("letter")
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(mediabox + (54, 54, -54, -54))
        story.draw(device)
        writer.end_page()
    writer.close()

    with fitz.open(str(path)) as doc:
        texts = [b.text for b in page_text_blocks(doc[0], fitz)]
    keys = [next(i for i, t in enumerate(texts) if t.startswith(f"Hyperparameter {r}:"))
            for r in range(12)]
    values = [next(i for i, t in enumerate(texts) if t.startswith(f"Linear warmup over {r * 100} "))
              for r in range(12)]
    assert all(k + 1 == v for k, v in zip(keys, values)), texts[:8]
