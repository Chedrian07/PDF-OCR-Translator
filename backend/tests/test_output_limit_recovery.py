"""OutputLimitError(MAX_LENGTH 도달 = 잘린 출력)가 runner의 기존 복구 경로를 탄다.

UnlimitedEngine은 EOS 없이 MAX_LENGTH에 닿은 생성을 OutputLimitError로 올린다
(audit decode-correctness-1, gap1-metal-real-e2e-5). RepetitiveOutputError의 하위
클래스라 runner는 잘린 multi 청크를 채택하지 않고 페이지별 single로 다시 처리하며,
per_page 모드에서는 재시도 없이 텍스트 레이어 폴백으로 간다.
"""

from pathlib import Path

from test_runner_failures import FAILED_MARK, _run_job

from app.engine.base import OutputLimitError
from app.engine.fake import FakeEngine

LIMIT_MESSAGE = "생성 길이 상한(MAX_LENGTH=32,768토큰)에 도달해 EOS 없이 출력이 잘렸습니다"


class LengthCappedEngine(FakeEngine):
    """첫 multi 호출(또는 지정 페이지의 single)이 길이 상한에 닿은 것처럼 동작."""

    def __init__(self, *, capped_single_pages=()):
        super().__init__(delay=0.0)
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}
        self.capped_single_pages = set(capped_single_pages)

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        if self.multi_calls == 1:
            sink.on_text("<PAGE>\n잘린 출력")  # 스트림은 이미 흘렀다(엔진은 flush 뒤 예외)
            raise OutputLimitError(LIMIT_MESSAGE)
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = int(Path(image_path).stem.rsplit("_", 1)[-1])
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        if page in self.capped_single_pages:
            raise OutputLimitError(LIMIT_MESSAGE)
        return super().run_single(image_path, out_dir, sink, cancel)


def test_length_capped_multi_chunk_is_reprocessed_page_by_page(tmp_path):
    engine = LengthCappedEngine()
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 2  # 잘린 청크를 같은 multi로 재시도하지 않음
    assert engine.single_calls == {1: 1, 2: 1}  # 그 청크만 페이지별 single
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert "잘린 출력" not in md
    assert len(md.split("\n\n---\n\n")) == 4
    # 잘린 청크의 재처리 경위는 참고다 — 페이지별 single로 모두 살려 품질 저하가 없다
    assert any("MAX_LENGTH" in note and "페이지별 재처리" in note for note in job.notices)
    assert job.warnings == []


def test_length_capped_single_page_falls_back_to_text_layer(tmp_path):
    """single 폴백까지 상한에 닿으면 재시도 없이 그 페이지만 텍스트 레이어로 복구."""
    engine = LengthCappedEngine(capped_single_pages={2})
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 1}  # 상한 오류는 재시도 대상이 아니다
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 2" in md  # 텍스트 레이어 복구


# ── 잘린 청크의 앞부분은 세그먼트 수가 아니라 원본 대조로 정한다 ─────────────────
# 잘리기 전에 모델이 한 페이지를 둘로 쪼개거나 건너뛰면 '마지막을 뺀 k개 = 앞 k쪽'이
# 틀린다. 예전에는 잘린 페이지 직전 물리 페이지의 본문이 통째로 사라지고(아무 경고 없이)
# 남긴 페이지와 layout이 한 칸씩 밀렸다 — 앞부분을 정확히 keep개로 병합하므로 병합기의
# 마커 수 정합도 걸리지 않았다(감사 pipeline-1).

_WORDS = ["alpha", "bravo", "charlie", "delta"]


def _distinct_text(index: int) -> str:
    """페이지마다 다른 본문(정합 대조가 페이지를 구분할 수 있는 실제 분량)."""
    word = _WORDS[index]
    return " ".join(
        f"{word}{j} sentence number {j} about topic {word} unique{index}x{j}" for j in range(25)
    )


def _distinct_text_pdf(pages: int) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 50, 545, 800), _distinct_text(i), fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def _raw(text: str) -> str:
    return f"<|det|>text [50, 50, 900, 900]<|/det|>{text}"


class MarkerSkewEngine(FakeEngine):
    """첫 multi가 segments_of(본문들)대로 출력하다 마지막 세그먼트에서 MAX_LENGTH로 잘린다.

    single은 그 페이지의 본문을 정상으로 읽는다(single_calls에 기록)."""

    def __init__(self, segments_of):
        super().__init__(delay=0.0)
        self.segments_of = segments_of
        self.multi_calls = 0
        self.single_calls: list[int] = []

    @staticmethod
    def _page(image_path) -> int:
        return int(Path(image_path).stem.rsplit("_", 1)[-1])

    def run_multi(self, image_paths, out_dir, sink, cancel):
        import json

        self.multi_calls += 1
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        segments = self.segments_of([_distinct_text(self._page(p) - 1) for p in image_paths])
        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": [_raw(s) for s in segments]}), encoding="utf-8"
        )
        for segment in segments:
            sink.on_text("<PAGE>\n" + _raw(segment) + "\n")
        raise OutputLimitError(
            LIMIT_MESSAGE, partial_output="<PAGE>\n" + "\n<PAGE>\n".join(segments)
        )

    def run_single(self, image_path, out_dir, sink, cancel):
        import json

        page = self._page(image_path)
        self.single_calls.append(page)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        text = _distinct_text(page - 1)
        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": [_raw(text)]}), encoding="utf-8"
        )
        sink.on_text(_raw(text))
        return text


def _run_skewed(tmp_path, segments_of, *, pages=4, threshold=0.0):
    """게이트를 끈 채(threshold=0) 돌린다 — 정합이 게이트에 기대지 않는지 본다."""
    import queue
    import threading

    from app.config import Settings
    from app.jobs import EventBroker, JobStore
    from app.pipeline.runner import execute_job

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(_distinct_text_pdf(pages))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data", preload_model=False,
        fake_delay=0.0, pages_per_chunk=pages, ocr_fidelity_threshold=threshold,
    )
    engine = MarkerSkewEngine(segments_of)
    engine.load()
    q = broker.subscribe(job.id)
    execute_job(job, store, broker, engine, settings, threading.Event())
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return job, engine, events


def _assert_every_page_in_place(job, events, pages=4):
    import json

    from tests.test_fidelity_gate import client_view

    segments = (job.dir / "result.md").read_text(encoding="utf-8").split("\n\n---\n\n")
    assert len(segments) == pages
    for index in range(pages):
        head, tail = f"{_WORDS[index]}0 ", f"{_WORDS[index]}24 "
        assert head in segments[index] and tail in segments[index], (
            f"{index + 1}페이지 자리에 그 페이지 본문이 없다: {segments[index][:60]!r}"
        )
        for other in range(pages):
            if other != index:
                assert f"{_WORDS[other]}3 " not in segments[index], (index, other)
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == list(range(1, pages + 1))
    for index, page in enumerate(layout):
        text = " ".join(b.get("content") or "" for b in page["blocks"])
        assert f"{_WORDS[index]}0 " in text, f"layout {index + 1}쪽이 밀렸다: {text[:60]!r}"
    live = client_view(events).split("<PAGE>")
    assert len(live) == pages + 1, [s[:40] for s in live]
    for index in range(pages):
        assert f"{_WORDS[index]}0 " in live[index + 1], (index, live[index + 1][:60])


def test_split_page_before_truncation_keeps_every_page_in_place(tmp_path):
    """1쪽을 둘로 쪼갠 뒤 3쪽에서 잘림 — 세그먼트는 4개지만 끝까지 생성된 물리 페이지는 2개다.

    예전에는 keep=3이 되어 [1쪽 앞절반, 1쪽 뒷절반, 2쪽, 4쪽]이 됐다 — 3쪽 본문이 사라졌다."""
    def segments_of(texts):
        half = len(texts[0]) // 2
        return [texts[0][:half], texts[0][half:], texts[1], texts[2][:200]]

    job, engine, events = _run_skewed(tmp_path, segments_of)

    assert job.status == "done"
    assert engine.multi_calls == 1
    assert engine.single_calls == [3, 4]  # 잘린 3쪽부터만 다시 처리한다
    _assert_every_page_in_place(job, events)
    assert any("앞 2쪽은 유지" in n for n in job.notices), job.notices


def test_skipped_page_before_truncation_is_reprocessed_from_the_gap(tmp_path):
    """2쪽을 건너뛴 뒤 4쪽에서 잘림 — 건너뛴 페이지부터 다시 처리해 본문을 되찾는다."""
    job, engine, events = _run_skewed(
        tmp_path, lambda texts: [texts[0], texts[2], texts[3][:200]],
    )

    assert job.status == "done"
    assert engine.single_calls == [2, 3, 4]
    _assert_every_page_in_place(job, events)


def test_honest_truncation_with_a_text_layer_keeps_the_completed_pages(tmp_path):
    """마커가 정직하면 예전처럼 끝까지 생성된 앞 페이지를 모두 지키고 잡음도 없다."""
    job, engine, events = _run_skewed(
        tmp_path, lambda texts: [texts[0], texts[1], texts[2][:200]],
    )

    assert job.status == "done"
    assert engine.single_calls == [3, 4]
    _assert_every_page_in_place(job, events)
    assert job.warnings == []


def test_contradicting_weak_evidence_reprocesses_the_whole_chunk(tmp_path):
    """대조 근거가 약한데(4개 중 매칭 1개) 그 매칭이 위치와 어긋나면 아무것도 믿지 않는다."""
    noise = "zz " * 60
    job, engine, events = _run_skewed(
        tmp_path, lambda texts: [noise, texts[2], noise, noise[:40]],
    )

    assert job.status == "done"
    assert engine.single_calls == [1, 2, 3, 4]
    _assert_every_page_in_place(job, events)
