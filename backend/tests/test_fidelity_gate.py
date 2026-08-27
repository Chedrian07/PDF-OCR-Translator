"""페이지 OCR 충실도 게이트 — 열화 탐지 · 단독 재처리 · 채택 판정.

멀티페이지 추론은 8쪽을 한 컨텍스트에 넣는데, 모델(sliding_window=128, 12층 MoE)은
그 범위에 걸친 구조 기록을 유지할 수단이 없다. 46쪽 논문 실측:

    p34, p39 → 프로덕션 **0자**, 같은 페이지를 단독 실행하면 0.945 / 0.972
    p38      → 프로덕션 0.409, 단독 0.971

여기 테스트는 그 결함의 최소 재현이다: multi에서 특정 페이지를 유실하고 single에서는
정상으로 읽는 가짜 엔진을 두고, 게이트가 (1) 그 페이지만 골라 (2) 다시 돌리고
(3) 실제로 나아졌을 때만 채택하며 (4) 라이브 프레이밍을 깨지 않는지 본다.
"""

import json
import queue
import threading

import pytest

from app.config import Settings
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore
from app.pipeline.fidelity import (
    MIN_TRUTH_CHARS,
    evaluate_layout_pages,
    normalize,
    score,
)
from app.pipeline.runner import execute_job

PAGE_MARKER = "<PAGE>"

# 게이트는 정답 텍스트가 MIN_TRUTH_CHARS 이상일 때만 판정한다. conftest의
# make_pdf_bytes는 쪽당 ~60자라 판정 불가로 빠지므로 전용 픽스처를 쓴다.
_PARAGRAPH = (
    "CyberGym evaluates agentic security by asking models to reproduce known "
    "vulnerabilities from real open source projects. Each instance ships the "
    "pre-patch codebase, a textual description of the flaw, and a reference "
    "proof of concept that triggers the sanitizer. The benchmark reports the "
    "success rate over instances where the agent produced a crashing input."
)


def make_texty_pdf(pages: int) -> bytes:
    """쪽마다 충분한 텍스트 레이어를 가진 born-digital PDF."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 60, 545, 760),
            f"Page {i + 1} of the evaluation appendix. {_PARAGRAPH}",
            fontsize=11,
        )
    data = doc.tobytes()
    doc.close()
    return data


def page_truth(index: int) -> str:
    return f"Page {index} of the evaluation appendix. {_PARAGRAPH}"


class PageDroppingEngine(FakeEngine):
    """multi에서 지정 페이지를 유실하고, single에서는 정상으로 읽는 엔진.

    실제 모델의 결함을 그대로 흉내낸다 — 모델이 그 페이지를 **못 읽는 게 아니라**
    청크 안에서 놓친다. `single_calls`로 게이트가 어떤 페이지를 다시 돌렸는지 본다.
    """

    def __init__(self, drop_pages=(), *, single_also_bad=(), single_raises=()):
        super().__init__(delay=0.0)
        self.drop_pages = set(drop_pages)
        self.single_also_bad = set(single_also_bad)
        self.single_raises = set(single_raises)
        self.single_calls: list[int] = []
        self.multi_calls = 0

    @staticmethod
    def _page_of(image_path) -> int:
        return int(str(image_path.stem).rsplit("_", 1)[-1])

    def _raw(self, page: int, *, degraded: bool) -> tuple[str, str]:
        if degraded:
            # 페이지 유실 — 머리글 한 줄만 남는다(프로덕션 p34/p39의 형태).
            body = ""
        else:
            body = page_truth(page)
        md = f"# page {page}\n\n{body}\n"
        raw = f"<|det|>text [60, 60, 930, 900]<|/det|>{body}"
        return md, raw

    # 실엔진은 **모델의 원출력**(grounding 태그 포함)을 스트리밍하고 후처리된
    # 마크다운을 반환한다. 라이브 박스는 그 태그에서 나오므로 그 형태를 지킨다.
    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        parts, raws = [], []
        for image_path in image_paths:
            page = self._page_of(image_path)
            md, raw = self._raw(page, degraded=page in self.drop_pages)
            sink.on_text(PAGE_MARKER + "\n")
            sink.on_text(raw)
            parts.append(md)
            raws.append(raw)
        self._write_raw_pages(out_dir, raws)
        return PAGE_MARKER + "\n" + f"\n{PAGE_MARKER}\n".join(parts)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = self._page_of(image_path)
        self.single_calls.append(page)
        if page in self.single_raises:
            raise RuntimeError(f"모의 단독 재처리 실패 (page {page})")
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        md, raw = self._raw(page, degraded=page in self.single_also_bad)
        self._write_raw_pages(out_dir, [raw])
        sink.on_text(raw)
        return md


def run_job(tmp_path, engine, *, pages=4, pages_per_chunk=4, **overrides):
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_texty_pdf(pages))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=pages_per_chunk,
        **overrides,
    )
    q = broker.subscribe(job.id)
    engine.load()
    execute_job(job, store, broker, engine, settings, threading.Event())
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return job, events


def stream_text(events) -> str:
    return "".join(d["text"] for e, d in events if e == "token" and "text" in d)


# ── 지표 ────────────────────────────────────────────────────────────────


def test_score_is_insensitive_to_global_reordering():
    """표는 셀 순서가 PDF 읽기 순서와 다르다 — 순서 민감 지표는 표에서 오탐한다.

    실측 p18은 정상인데 difflib ratio가 0.292였다. 문자 bigram 다중집합은 같은 내용을
    다르게 배열해도 점수를 유지한다.
    """
    cells = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    truth = normalize(" ".join(cells))
    shuffled = normalize(" ".join(reversed(cells)))
    assert score(truth, shuffled) > 0.90


def test_score_catches_missing_and_inflated_content():
    truth = normalize(_PARAGRAPH)
    assert score(truth, "") == pytest.approx(0.0)
    assert score(truth, normalize(_PARAGRAPH[: len(_PARAGRAPH) // 2])) < 0.75
    # 같은 문장을 네 번 반복 — 과생성도 점수를 떨어뜨려야 한다(팽창 페널티)
    assert score(truth, normalize(_PARAGRAPH * 4)) < 0.75
    assert score(truth, truth) == pytest.approx(1.0)


def test_score_is_sensitive_to_loss_where_dice_is_not():
    """대칭 지표(Dice)를 버린 이유 — 3분의 1이 빠져도 Dice는 임계값을 통과한다.

    실측(게이트 적용 실행의 p18): 표 내용 33% 유실. Dice 0.702 vs containment 0.587.
    """
    truth = normalize(_PARAGRAPH * 3)
    kept = normalize(_PARAGRAPH * 2)          # 3분의 1 유실
    got = score(truth, kept)
    dice = 2 * 0.667 / (1 + 0.667)            # 같은 상황의 Dice 근사
    assert got < 0.70 < dice, (got, dice)


def test_html_whitelist_keeps_angle_brackets_in_code():
    """범용 `<[^>]*>`는 **정답**을 파괴한다 — 코드의 꺾쇠는 태그가 아니다.

    실측 p33: 정답 2,532자 → 866자(65.8% 증발). 줄을 이어 붙인 뒤 적용하면 한 줄의
    `<`가 수십 줄 뒤의 `>`까지 삼킨다. 태그 이름 화이트리스트로 묶어야 한다.
    """
    code = "#include <AK/Debug.h> Vector<Component> read_value<BigEndian<u16>>(x)"
    kept = normalize(code)
    assert "AK/Debug.h" in kept and "Component" in kept and "BigEndian" in kept
    # 모델이 내는 표 마크업은 그대로 제거된다
    assert normalize("<table><tr><td>a</td></tr></table>") == "a"
    # 여러 줄을 가로지르는 삼킴이 없다
    assert "keepme" in normalize("x < 3 and\nkeepme\ny > 4")


def test_chart_blocks_are_excluded_from_the_truth(tmp_path):
    """`image`만 보면 차트가 빠진다 — 실측 이 논문은 image 12개 + chart 7개다.

    chart 4개로만 이루어진 p25는 전사가 완벽한데도 차트 안 텍스트가 정답에 남아
    0.729로 떨어졌다(임계 0.70과 간격 0.029).
    """
    import fitz
    import pymupdf

    from app.pipeline.fidelity import truth_text

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 60, 545, 300), _PARAGRAPH, fontsize=11)
    page.insert_textbox(fitz.Rect(50, 500, 545, 700), "AXIS LABEL IN CHART", fontsize=11)
    page.draw_rect(fitz.Rect(60, 480, 540, 720), color=(0, 0, 0), fill=(0.95, 0.95, 0.95))
    path = tmp_path / "chart.pdf"
    doc.save(path)
    doc.close()

    opened = pymupdf.open(path)
    try:
        masked = truth_text(pymupdf, opened[0], [{"type": "chart", "bbox": [50, 560, 950, 850]}])
        assert "AXIS LABEL" not in masked
        assert "CyberGym" in masked
    finally:
        opened.close()


def test_short_pages_are_not_judged(tmp_path):
    """정답이 빈약한 페이지(표지·백지)는 판정하지 않는다 — None은 실패가 아니다."""
    import fitz

    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_text((72, 80), "Title", fontsize=24)
    path = tmp_path / "short.pdf"
    doc.save(path)
    doc.close()
    layout = [{"page": 1, "blocks": [{"type": "text", "content": "Title"}]}]
    (result,) = evaluate_layout_pages(path, layout)
    assert not result.measurable
    assert result.truth_chars < MIN_TRUTH_CHARS


def test_missing_pdf_disables_the_gate(tmp_path):
    """스캔 PDF·원본 부재는 '판정 불가'지 실패가 아니다."""
    layout = [{"page": 1, "blocks": [{"type": "text", "content": "x" * 500}]}]
    (result,) = evaluate_layout_pages(tmp_path / "nope.pdf", layout)
    assert not result.measurable


def test_figure_text_is_excluded_from_the_truth(tmp_path):
    """모델이 그림으로 분류한 영역의 PDF 텍스트는 정답에서 빠져야 한다.

    빼지 않으면 그림이 큰 페이지가 전부 열화로 보인다.
    """
    import fitz
    import pymupdf

    from app.pipeline.fidelity import truth_text

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 60, 545, 300), _PARAGRAPH, fontsize=11)
    page.insert_textbox(fitz.Rect(50, 500, 545, 700), "LEGEND INSIDE FIGURE", fontsize=11)
    # 원본에 **실제 그림**을 그린다 — 게이트는 모델의 그림 분류를 PDF로 검증한다
    # (그리지 않으면 "놓친 페이지를 전면 그림이라 우기는" 경우와 구별되지 않는다).
    page.draw_rect(fitz.Rect(60, 480, 540, 720), color=(0, 0, 1), fill=(0.9, 0.9, 1))
    path = tmp_path / "fig.pdf"
    doc.save(path)
    doc.close()

    opened = pymupdf.open(path)
    try:
        # 그림 블록 없이는 범례가 정답에 들어간다
        plain = truth_text(pymupdf, opened[0], [])
        assert "LEGEND" in plain
        # 아래쪽을 image 블록으로 선언하면 빠진다 (0-999 정규화 좌표)
        figure = [{"type": "image", "bbox": [50, 560, 950, 850]}]
        masked = truth_text(pymupdf, opened[0], figure)
        assert "LEGEND" not in masked
        assert "CyberGym" in masked
    finally:
        opened.close()


# ── 게이트 통합 ─────────────────────────────────────────────────────────


def test_dropped_page_is_detected_and_repaired(tmp_path):
    """청크가 통째로 놓친 페이지를 게이트가 골라내 단독으로 복구한다."""
    engine = PageDroppingEngine(drop_pages={3})
    job, events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)

    assert engine.single_calls == [3], engine.single_calls
    result = (job.dir / "result.md").read_text(encoding="utf-8")
    assert page_truth(3) in result
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    page3 = [p for p in layout if p["page"] == 3][0]
    assert any(page_truth(3) in (b.get("content") or "") for b in page3["blocks"])
    assert any("충실도" in w for w in job.warnings), job.warnings


def test_only_degraded_pages_are_rerun(tmp_path):
    """정상 페이지는 건드리지 않는다 — 재처리는 비싸다."""
    engine = PageDroppingEngine(drop_pages={2})
    job, _events = run_job(tmp_path, engine, pages=5, pages_per_chunk=5)
    assert engine.single_calls == [2], engine.single_calls


def test_no_degraded_pages_means_no_reruns(tmp_path):
    engine = PageDroppingEngine()
    job, _events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    assert engine.single_calls == []
    assert not any("충실도" in w for w in job.warnings), job.warnings


def client_view(events) -> str:
    """클라이언트가 최종적으로 보게 되는 스트림 — reset을 실제로 적용해 재구성한다.

    프런트(frontend/js/core.js truncateRawToPage)는 reset{from_page}에서 그 페이지의
    마커 **앞**까지 잘라낸다. 서버 프레이밍이 옳은지는 이 재구성본에서만 드러난다.
    """
    text = ""
    for name, data in events:
        if name == "token" and "text" in data:
            text += data["text"]
        elif name == "reset":
            keep = int(data["from_page"]) - 1
            parts = text.split(PAGE_MARKER)
            text = PAGE_MARKER.join(parts[: keep + 1])
    return text


def test_repair_keeps_one_segment_per_page(tmp_path):
    """복구가 라이브 프레이밍을 깨면 안 된다 — 이후 전 페이지의 박스가 밀린다.

    reset을 실제로 적용한 뒤의 스트림이 "k번째 <PAGE> == 페이지 k"를 지켜야 한다.
    """
    engine = PageDroppingEngine(drop_pages={2})
    job, events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    resets = [d for e, d in events if e == "reset"]
    assert resets, "열화 페이지를 물리는 reset이 없다"
    assert resets[0]["from_page"] == 2, resets

    segments = client_view(events).split(PAGE_MARKER)
    assert len(segments) == 5, [s[:40] for s in segments]
    assert not segments[0].strip(), segments[0][:80]
    for page in range(1, 5):
        # 되감긴 뒤 재발행된 페이지는 마크다운이 아니라 grounding 원문으로 온다
        # (박스 귀속을 지키기 위해) — 두 형식 모두 그 페이지의 본문을 담는다.
        assert page_truth(page) in segments[page], (page, segments[page][:120])
        for other in range(1, 5):
            if other != page:
                assert page_truth(other) not in segments[page], (
                    f"세그먼트 {page}가 페이지 {other} 내용을 함께 담고 있다"
                )
    # 복구된 2쪽 본문이 라이브에도 들어가 있어야 한다
    assert page_truth(2) in segments[2]
    # 최종 산출물의 페이지 수 계약도 함께
    assert len((job.dir / "result.md").read_text(encoding="utf-8").split("\n\n---\n\n")) == 4


def test_rejected_retry_keeps_the_original_page(tmp_path):
    """재처리가 나아지지 않으면 원래 결과를 지킨다 — 게이트는 손해를 끼치면 안 된다."""
    engine = PageDroppingEngine(drop_pages={3}, single_also_bad={3})
    job, _events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    assert engine.single_calls == [3]
    assert any("개선되지 않아" in w for w in job.warnings), job.warnings
    # 재처리 산출물을 병합하지 않았으므로 3쪽은 여전히 비어 있다(원래 상태).
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    page3 = [p for p in layout if p["page"] == 3][0]
    assert not any(page_truth(3) in (b.get("content") or "") for b in page3["blocks"])


def test_failed_retry_does_not_break_the_job(tmp_path):
    """재처리 자체가 터져도 잡은 완주하고 원래 결과를 지킨다."""
    engine = PageDroppingEngine(drop_pages={2}, single_raises={2})
    job, _events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    assert job.status == "done"
    assert any("재처리 실패" in w for w in job.warnings), job.warnings


def test_threshold_zero_disables_the_gate(tmp_path):
    engine = PageDroppingEngine(drop_pages={3})
    job, _events = run_job(
        tmp_path, engine, pages=4, pages_per_chunk=4, ocr_fidelity_threshold=0.0
    )
    assert engine.single_calls == []
    assert job.status == "done"


def test_retry_budget_caps_the_work(tmp_path):
    """손상 문서가 전 페이지 재처리로 런타임을 폭발시키지 않게 막는다."""
    engine = PageDroppingEngine(drop_pages={1, 2, 3, 4, 5, 6})
    job, _events = run_job(
        tmp_path, engine, pages=6, pages_per_chunk=6,
        ocr_fidelity_max_retry_ratio=0.0,   # 하한 2쪽만 허용
    )
    assert len(engine.single_calls) == 2, engine.single_calls
    assert any("예산" in w for w in job.warnings), job.warnings


def test_scanned_pdf_without_text_layer_is_skipped(tmp_path):
    """텍스트 레이어가 없으면 정답이 없다 — 게이트를 걸지 않는다."""
    import fitz

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("scan.pdf", "multi", dpi=72)
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=595, height=842)
    (job.dir / "source.pdf").write_bytes(doc.tobytes())
    doc.close()
    engine = PageDroppingEngine(drop_pages={2})
    engine.load()
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=3,
    )
    execute_job(job, store, broker, engine, settings, threading.Event())
    assert engine.single_calls == []
    assert job.status == "done"


# ── 머저의 페이지 교체 API ──────────────────────────────────────────────


def test_replace_page_swaps_markdown_layout_and_images(tmp_path):
    from app.pipeline.merge import ChunkResult, IncrementalMerger

    job_dir = tmp_path / "job"
    (job_dir / "pages").mkdir(parents=True)
    (job_dir / "source.pdf").write_bytes(make_texty_pdf(2))

    merger = IncrementalMerger(job_dir, "\n\n---\n\n")
    chunk_dir = job_dir / "work" / "chunk_00"
    (chunk_dir / "images").mkdir(parents=True)
    (chunk_dir / "raw_pages.json").write_text(
        json.dumps({"pages": [
            "<|det|>text [60, 60, 930, 400]<|/det|>first",
            "<|det|>text [60, 60, 930, 400]<|/det|>second",
        ]}), encoding="utf-8",
    )
    merger.add_chunk(ChunkResult(chunk_dir, 1, 2, "first\n\n<PAGE>\n\nsecond"))
    assert len(merger.pages_md) == 2

    page_dir = job_dir / "work" / "fix"
    (page_dir / "images").mkdir(parents=True)
    (page_dir / "images" / "0.jpg").write_bytes(b"jpg")
    (page_dir / "raw_pages.json").write_text(
        json.dumps({"pages": ["<|det|>text [60, 60, 930, 400]<|/det|>second-fixed"]}),
        encoding="utf-8",
    )
    assert merger.replace_page(2, ChunkResult(page_dir, 2, 1, "second-fixed", single=True))

    assert merger.pages_md == ["first", "second-fixed"]
    assert len(merger.layout_pages) == 2
    replaced = [p for p in merger.layout_pages if p["page"] == 2][0]
    assert any("second-fixed" in (b.get("content") or "") for b in replaced["blocks"])
    assert (job_dir / "images" / "p0002_0.jpg").is_file()
    assert "second-fixed" in (job_dir / "result.md").read_text(encoding="utf-8")


def test_replace_page_rejects_non_single_chunks(tmp_path):
    from app.pipeline.merge import ChunkResult, IncrementalMerger

    merger = IncrementalMerger(tmp_path, "\n\n---\n\n")
    with pytest.raises(ValueError):
        merger.replace_page(1, ChunkResult(tmp_path, 1, 2, "x", single=False))


def test_replace_page_drops_stale_figures(tmp_path):
    """재처리가 그림을 더 적게 내면 옛 크롭이 유령으로 남으면 안 된다."""
    from app.pipeline.merge import ChunkResult, IncrementalMerger

    job_dir = tmp_path / "job"
    (job_dir / "pages").mkdir(parents=True)
    merger = IncrementalMerger(job_dir, "\n\n---\n\n")
    merger.pages_md = ["one"]
    merger.images_dir.mkdir(parents=True, exist_ok=True)
    for k in (0, 1):
        (merger.images_dir / f"p0001_{k}.jpg").write_bytes(b"old")
        merger.figure_boxes[f"p0001_{k}.jpg"] = {"bbox": [0, 0, 1, 1]}

    page_dir = job_dir / "work" / "fix"
    (page_dir / "images").mkdir(parents=True)
    (page_dir / "images" / "0.jpg").write_bytes(b"new")
    (page_dir / "raw_pages.json").write_text(
        json.dumps({"pages": ["<|det|>text [1, 1, 900, 900]<|/det|>fixed"]}),
        encoding="utf-8",
    )
    merger.replace_page(1, ChunkResult(page_dir, 1, 1, "fixed", single=True))

    assert (merger.images_dir / "p0001_0.jpg").read_bytes() == b"new"
    assert not (merger.images_dir / "p0001_1.jpg").exists()
    assert "p0001_1.jpg" not in merger.figure_boxes


def test_fake_full_page_figure_does_not_silence_the_gate(tmp_path):
    """놓친 페이지를 전면 `image`로 부르면 정답이 사라져 게이트가 침묵했다.

    OCR 모델이 판독 실패 영역을 그림으로 분류하는 것은 흔한 동작이라, 게이트가
    존재 이유인 바로 그 실패에서 조용해지는 구멍이었다. 원본 PDF가 거기에 아무것도
    그리지 않았다면 그 분류를 믿지 않는다.
    """
    import fitz
    import pymupdf

    from app.pipeline.fidelity import page_fidelity_blocks

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 60, 545, 760), _PARAGRAPH * 2, fontsize=11)
    path = tmp_path / "lost.pdf"
    doc.save(path)
    doc.close()

    opened = pymupdf.open(path)
    try:
        lost = [{"type": "image", "bbox": [0, 0, 999, 999]}]
        result = page_fidelity_blocks(pymupdf, opened[0], lost, 1)
        assert result.measurable, "가짜 전면 그림이 게이트를 무력화한다"
        assert result.truth_chars > MIN_TRUTH_CHARS
        assert result.score is not None and result.score < 0.2
    finally:
        opened.close()


def test_display_equations_are_dropped_from_both_sides(tmp_path):
    """PDF 글리프와 모델 LaTeX는 같은 내용인데 길이가 크게 다르다.

    그대로 두면 수식이 많은 페이지가 과생성으로 오탐된다. 양쪽에서 함께 빼면 대칭이라
    수식이 없는 문서에는 no-op이다.
    """
    import fitz
    import pymupdf

    from app.pipeline.fidelity import ocr_text, page_fidelity_blocks

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 60, 545, 600), _PARAGRAPH, fontsize=11)
    page.insert_textbox(fitz.Rect(50, 650, 545, 720), "E = mc2", fontsize=14)
    path = tmp_path / "eq.pdf"
    doc.save(path)
    doc.close()

    latex = r"\[ E = mc^{2} \quad \text{with} \quad \gamma = \frac{1}{\sqrt{1 - v^2/c^2}} \]"
    blocks = [
        {"type": "text", "content": _PARAGRAPH},
        {"type": "equation", "bbox": [80, 760, 920, 860], "content": latex},
    ]
    assert latex not in ocr_text(blocks)          # 후보에서 빠진다
    assert _PARAGRAPH in ocr_text(blocks)

    opened = pymupdf.open(path)
    try:
        result = page_fidelity_blocks(pymupdf, opened[0], blocks, 1)
        assert result.measurable and result.score is not None
        assert result.score > 0.90, result.score   # 수식이 점수를 끌어내리지 않는다
    finally:
        opened.close()


def test_equation_covering_most_of_the_page_is_not_trusted(tmp_path):
    """수식 블록도 페이지 절반을 넘게 덮으면 분류를 믿지 않는다 (그림과 같은 이유)."""
    import fitz
    import pymupdf

    from app.pipeline.fidelity import _equation_rects

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    path = tmp_path / "big.pdf"
    doc.save(path)
    doc.close()

    opened = pymupdf.open(path)
    try:
        small = [{"type": "equation", "bbox": [100, 100, 900, 300]}]
        assert _equation_rects(pymupdf, opened[0], small)
        whole = [{"type": "equation", "bbox": [0, 0, 999, 999]}]
        assert _equation_rects(pymupdf, opened[0], whole) == []
    finally:
        opened.close()


def test_reemitted_pages_keep_grounding_so_live_boxes_survive(tmp_path):
    """되감긴 뒤 재발행하는 정상 페이지도 `<|det|>`를 실어야 한다.

    병합된 마크다운을 그대로 흘리면 그 페이지들의 레이아웃 박스가 라이브 뷰에서
    통째로 사라진다 — 열화 페이지 하나 때문에 뒤따르는 정상 페이지 전부가 영향을 받는다.
    """
    engine = PageDroppingEngine(drop_pages={2})
    _job, events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    view = client_view(events)
    segments = view.split(PAGE_MARKER)
    assert len(segments) == 5, [s[:40] for s in segments]
    for page in (2, 3, 4):          # 되감기(from_page=2) 이후 재발행된 구간
        assert "<|det|>" in segments[page], (page, segments[page][:120])


def test_gate_is_skipped_for_engines_without_text_bboxes(tmp_path):
    """텍스트 bbox를 안 주는 엔진(figure_only)에서는 전 페이지가 0.00으로 오탐된다."""
    from dataclasses import replace as dc_replace

    class FigureOnlyEngine(PageDroppingEngine):
        def capabilities(self):
            return dc_replace(super().capabilities(), layout_capability="figure_only")

    engine = FigureOnlyEngine(drop_pages={2, 3})
    job, _events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    assert engine.single_calls == [], engine.single_calls
    assert job.status == "done"


def test_gate_is_skipped_for_page_level_engines(tmp_path):
    """이미 페이지 단위로 도는 엔진은 재처리가 같은 호출이라 개선 여지가 없다."""
    from dataclasses import replace as dc_replace

    class PageLevelEngine(PageDroppingEngine):
        def capabilities(self):
            return dc_replace(
                super().capabilities(), supports_multi_page=False, preferred_chunk_size=1
            )

    engine = PageLevelEngine(drop_pages={1, 2})
    job, _events = run_job(tmp_path, engine, pages=3, pages_per_chunk=3)
    assert engine.single_calls == [], engine.single_calls
    assert job.status == "done"


def test_gate_failure_does_not_fail_the_job(tmp_path):
    """복구는 선택적 개선이다 — 게이트의 IO 오류가 완주한 잡을 죽이면 안 된다."""
    import app.pipeline.runner as runner_mod

    engine = PageDroppingEngine(drop_pages={2})
    original = runner_mod.evaluate_layout_pages

    def _boom(*_a, **_k):
        raise OSError("모의 IO 실패")

    runner_mod.evaluate_layout_pages = _boom
    try:
        job, _events = run_job(tmp_path, engine, pages=3, pages_per_chunk=3)
    finally:
        runner_mod.evaluate_layout_pages = original

    assert job.status == "done"
    assert engine.single_calls == []
    assert any("게이트가 실패" in w for w in job.warnings), job.warnings


def framing(events, pages) -> tuple[int, list[bool]]:
    """(마커 수, 페이지별 본문 귀속 여부) — reset을 적용한 최종 스트림 기준."""
    segments = client_view(events).split(PAGE_MARKER)
    return len(segments) - 1, [
        page_truth(p) in segments[p] if p < len(segments) else False
        for p in range(1, pages + 1)
    ]


def test_degraded_page_in_a_later_chunk_keeps_framing(tmp_path):
    """열화가 **두 번째 청크** 한가운데일 때. 되감기 지점이 앞 청크를 건드리면 안 된다."""
    engine = PageDroppingEngine(drop_pages={6})
    _job, events = run_job(tmp_path, engine, pages=8, pages_per_chunk=4)
    assert engine.single_calls == [6], engine.single_calls
    markers, ok = framing(events, 8)
    assert markers == 8, markers
    assert all(ok), ok


def test_consecutive_degraded_pages_keep_framing(tmp_path):
    """연속한 두 페이지가 모두 열화면 한 번의 되감기로 둘 다 처리한다."""
    engine = PageDroppingEngine(drop_pages={2, 3})
    _job, events = run_job(tmp_path, engine, pages=5, pages_per_chunk=5)
    assert engine.single_calls == [2, 3], engine.single_calls
    markers, ok = framing(events, 5)
    assert markers == 5, markers
    assert all(ok), ok


def test_budget_truncation_keeps_framing(tmp_path):
    """예산으로 잘려 재처리하지 못한 페이지도 되감긴 자리를 반드시 다시 채워야 한다."""
    engine = PageDroppingEngine(drop_pages={1, 2, 3, 4, 5, 6})
    _job, events = run_job(
        tmp_path, engine, pages=6, pages_per_chunk=6, ocr_fidelity_max_retry_ratio=0.0
    )
    assert len(engine.single_calls) == 2, engine.single_calls
    markers, _ok = framing(events, 6)
    assert markers == 6, markers


def test_rejected_retries_keep_framing(tmp_path):
    """전부 미채택이어도 세그먼트 수와 본문 귀속이 유지된다."""
    engine = PageDroppingEngine(drop_pages={2, 3}, single_also_bad={2, 3})
    _job, events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    markers, ok = framing(events, 4)
    assert markers == 4, markers
    # 2·3쪽은 원래 열화 상태(본문 없음)로 되돌아간다 — 1·4쪽은 온전해야 한다
    assert ok[0] and ok[3], ok


def test_retry_that_dies_mid_stream_leaves_no_partial_output(tmp_path):
    """재처리가 **일부를 흘린 뒤** 터지면 그 조각이 라이브에 남으면 안 된다.

    거부 경로의 rewind_to가 이 부분 출력을 물어야 한다 — 남으면 페이지 본문이
    두 번 보이고 프리뷰의 열린 태그가 뒤 내용을 삼킨다.
    """
    class LateRaiseEngine(PageDroppingEngine):
        def run_single(self, image_path, out_dir, sink, cancel):
            self.single_calls.append(self._page_of(image_path))
            sink.on_text("PARTIAL-RETRY-OUTPUT ")
            raise RuntimeError("재처리 도중 실패")

    engine = LateRaiseEngine(drop_pages={2})
    job, events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    view = client_view(events)
    assert "PARTIAL-RETRY-OUTPUT" not in view, "폐기한 부분 출력이 라이브에 남았다"
    markers, _ok = framing(events, 4)
    assert markers == 4, markers
    assert job.status == "done"
    assert any("재처리 실패" in w for w in job.warnings), job.warnings


class FiguredDroppingEngine(PageDroppingEngine):
    """열화 재현 + **실엔진의 이미지/오버레이 파일 규약**까지 지키는 엔진.

    multi는 `images/page_{모델인덱스}_{크롭}.jpg`·`result_with_boxes_{i}.jpg`,
    single은 `images/{크롭}.jpg`·`result_with_boxes.jpg`로 쓴다. 규약이 다르므로
    복구 경로가 이미지 귀속을 깨는지는 이 엔진으로만 드러난다.
    """

    def _page_payload(self, page: int, *, degraded: bool) -> tuple[str, str]:
        body = "" if degraded else page_truth(page)
        md = f"# page {page}\n\n{body}\n\n![](images/{{img}})\n"
        raw = (
            f"<|det|>text [60, 60, 930, 700]<|/det|>{body}\n"
            f"<|det|>image [100, 720, 900, 960]<|/det|>"
        )
        return md, raw

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        parts, raws = [], []
        for i, image_path in enumerate(image_paths):
            page = self._page_of(image_path)
            md, raw = self._page_payload(page, degraded=page in self.drop_pages)
            (out_dir / "images" / f"page_{i}_0.jpg").write_bytes(b"crop")
            (out_dir / f"result_with_boxes_{i}.jpg").write_bytes(b"overlay")
            md = md.format(img=f"page_{i}_0.jpg")
            sink.on_text(PAGE_MARKER + "\n")
            sink.on_text(raw)
            parts.append(md)
            raws.append(raw)
        self._write_raw_pages(out_dir, raws)
        return PAGE_MARKER + "\n" + f"\n{PAGE_MARKER}\n".join(parts)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = self._page_of(image_path)
        self.single_calls.append(page)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        md, raw = self._page_payload(page, degraded=page in self.single_also_bad)
        (out_dir / "images" / "0.jpg").write_bytes(b"crop-single")
        (out_dir / "result_with_boxes.jpg").write_bytes(b"overlay-single")
        self._write_raw_pages(out_dir, [raw])
        sink.on_text(raw)
        return md.format(img="0.jpg")


def test_repair_keeps_image_references_resolvable(tmp_path):
    """복구가 그림 귀속을 깨면 안 된다 — multi와 single은 파일명 규약이 다르다.

    마크다운·레이아웃이 가리키는 크롭이 실제로 존재해야 하고, 아무도 참조하지
    않는 고아 파일이 남아서도 안 된다.
    """
    import re

    engine = FiguredDroppingEngine(drop_pages={3})
    job, _events = run_job(tmp_path, engine, pages=4, pages_per_chunk=4)
    assert engine.single_calls == [3], engine.single_calls

    on_disk = {p.name for p in (job.dir / "images").iterdir() if p.suffix == ".jpg"}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    md_refs = set(re.findall(r"images/([^)\s]+)", md))
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    lay_refs = {b["image"] for p in layout for b in p["blocks"] if b.get("image")}

    assert md_refs <= on_disk, f"마크다운이 없는 파일을 가리킨다: {md_refs - on_disk}"
    assert lay_refs <= on_disk, f"레이아웃이 없는 파일을 가리킨다: {lay_refs - on_disk}"
    assert on_disk <= md_refs | lay_refs, f"고아 크롭: {on_disk - md_refs - lay_refs}"
    # 복구된 3쪽의 크롭은 single 규약으로 다시 놓였어야 한다
    assert "p0003_0.jpg" in on_disk, sorted(on_disk)
    # 오버레이도 그 페이지 것으로 교체된다
    assert (job.dir / "layout" / "page_0003.jpg").read_bytes() == b"overlay-single"


def test_cancel_during_repair_leaves_no_hole_in_the_live_stream(tmp_path):
    """재처리 도중 취소하면 되감은 자리를 반드시 다시 채워야 한다.

    채우지 않으면 라이브 뷰에서 그 뒤 페이지가 사라진 채 잡이 끝난다.
    """
    cancel = threading.Event()

    class CancelingEngine(PageDroppingEngine):
        def run_single(self, image_path, out_dir, sink, cancel_event):
            cancel.set()          # 재처리 **직전**에 취소가 들어온 상황
            return super().run_single(image_path, out_dir, sink, cancel_event)

    engine = CancelingEngine(drop_pages={2})
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_texty_pdf(4))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=4,
    )
    q = broker.subscribe(job.id)
    engine.load()
    execute_job(job, store, broker, engine, settings, cancel)
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break

    assert job.status == "canceled", job.status
    markers = client_view(events).count(PAGE_MARKER)
    assert markers == 4, f"되감은 자리가 비었다 — 마커 {markers}개"
