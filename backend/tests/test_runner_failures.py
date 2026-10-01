"""청크 실패 격리(runner.py) 검증 — 한 청크가 죽어도 잡 전체가 죽지 않는다.

FlakyEngine: FakeEngine을 상속해 지정한 호출 번호(1-based)에서만 예외를 던진다.
호출 카운트로 재시도 횟수까지 검증한다.
"""

import json
import threading
from pathlib import Path

from app.config import Settings
from app.engine.base import JobCanceled, RepetitiveOutputError
from app.engine.fake import FakeEngine
from app.jobs import EventBroker, JobStore
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes

FAILED_MARK = "이 페이지는 변환에 실패했습니다"


class FlakyEngine(FakeEngine):
    """지정한 호출(1-based)에서 예외를 던지는 가짜 엔진 — 청크 격리/재시도 검증용."""

    def __init__(self, fail_calls=(), fail_all=False, exc=RuntimeError):
        super().__init__(delay=0.0)
        self.calls = 0
        self.fail_calls = set(fail_calls)
        self.fail_all = fail_all
        self.exc = exc

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_all or self.calls in self.fail_calls:
            raise self.exc(f"모의 실패 (call {self.calls})")

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self._maybe_fail()
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        self._maybe_fail()
        return super().run_single(image_path, out_dir, sink, cancel)


class LoopFallbackEngine(FakeEngine):
    """첫 multi 호출을 반복 오류로 만들고 single 폴백 호출을 기록한다."""

    def __init__(
        self,
        *,
        fail_single_pages=(),
        fail_single_once_pages=(),
        loop_single_pages=(),
        cancel_on_multi_loop=False,
    ):
        super().__init__(delay=0.0)
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}
        self.fail_single_pages = set(fail_single_pages)
        self.fail_single_once_pages = set(fail_single_once_pages)
        self.loop_single_pages = set(loop_single_pages)
        self.cancel_on_multi_loop = cancel_on_multi_loop

    @staticmethod
    def _page_number(image_path) -> int:
        return int(Path(image_path).stem.rsplit("_", 1)[-1])

    @staticmethod
    def _write_single_poison(out_dir) -> None:
        images = out_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "99.jpg").write_bytes(b"partial single output")
        (out_dir / "result_with_boxes.jpg").write_bytes(b"partial layout")
        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": ["partial repeated layout"]}), encoding="utf-8"
        )

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        if self.multi_calls == 1:
            poison = out_dir / "images" / "page_0_99.jpg"
            poison.parent.mkdir(parents=True, exist_ok=True)
            poison.write_bytes(b"partial multi output")
            if self.cancel_on_multi_loop:
                cancel.set()
            raise RepetitiveOutputError("모의 의미 반복")
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = self._page_number(image_path)
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        if page in self.loop_single_pages:
            self._write_single_poison(out_dir)
            raise RepetitiveOutputError(f"{page}페이지 모의 의미 반복")
        if page in self.fail_single_pages or (
            page in self.fail_single_once_pages and self.single_calls[page] == 1
        ):
            self._write_single_poison(out_dir)
            raise RuntimeError(f"{page}페이지 모의 실패")
        return super().run_single(image_path, out_dir, sink, cancel)


def _run_job(
    tmp_path,
    engine,
    pages=4,
    mode="multi",
    pages_per_chunk=2,
    *,
    embedded_text=True,
):
    """execute_job을 워커 없이 직접 구동 (4페이지 × 청크 2 → 청크 2개 구성)."""
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", mode, dpi=72)
    if embedded_text:
        pdf_bytes = make_pdf_bytes(pages=pages, with_image=False)
    else:
        import fitz

        doc = fitz.open()
        for _ in range(pages):
            doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
    (job.dir / "source.pdf").write_bytes(pdf_bytes)
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=pages_per_chunk,
    )
    engine.load()
    execute_job(job, store, broker, engine, settings, threading.Event())
    return job


def test_failed_multi_chunk_is_reprocessed_page_by_page(tmp_path):
    """청크1의 multi가 재시도까지 실패해도 청크 전체를 플레이스홀더로 확정하지 않는다
    — 페이지별 single로 다시 처리한다(8쪽 prefill OOM은 1쪽씩이면 대개 통과한다).

    예전 기대값(1–2쪽 플레이스홀더)은 감사에서 결함으로 확인된 동작이었다: 같은
    실패가 per_page 모드에서는 복구되는데 multi(업로드 기본값)에서만 내용이 사라졌다."""
    engine = FlakyEngine(fail_calls={1, 2})  # 청크1: 최초 + 재시도 모두 실패
    job = _run_job(tmp_path, engine)

    assert job.status == "done"
    assert engine.calls == 5  # 청크1 multi ×2 + single ×2(1·2쪽) + 청크2 multi ×1
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    for page in range(1, 5):
        assert f"![](images/p{page:04d}_0.jpg)" in md
    assert len(md.split("\n\n---\n\n")) == 4  # 글로벌 페이지 수 정합 유지
    assert len(job.warnings) == 1
    assert "1–2페이지" in job.warnings[0] and "페이지별 재처리" in job.warnings[0]
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done"
    assert meta["warnings"] == job.warnings


class PageFailingEngine(FakeEngine):
    """multi는 늘 실패하고, single은 지정 페이지에서만 결정적으로 실패하는 엔진."""

    def __init__(self, fail_single_pages=()):
        super().__init__(delay=0.0)
        self.fail_single_pages = set(fail_single_pages)
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        raise RuntimeError("MPS backend out of memory (모의 8쪽 prefill)")

    def run_single(self, image_path, out_dir, sink, cancel):
        page = int(Path(image_path).stem.rsplit("_", 1)[-1])
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        if page in self.fail_single_pages:
            raise RuntimeError(f"{page}페이지 모의 벤더 예외")
        return super().run_single(image_path, out_dir, sink, cancel)


def test_one_deterministically_failing_page_does_not_take_its_chunk_down(tmp_path):
    """한 페이지만 결정적으로 실패하면 그 페이지만 텍스트 레이어로 복구되고 나머지는
    OCR 결과를 지킨다 — 4쪽 단일 청크 문서가 잡 전체 error로 끝나던 회귀 방지."""
    engine = PageFailingEngine(fail_single_pages={3})
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=4)

    assert job.status == "done"
    assert engine.multi_calls == 2
    assert engine.single_calls == {1: 1, 2: 1, 3: 2, 4: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    for page in (1, 2, 4):
        assert f"![](images/p{page:04d}_0.jpg)" in md
    assert "Sample page 3" in md and "PDF 내장 텍스트 레이어" in md
    assert any("텍스트 레이어로 복구" in w for w in job.warnings)


def test_single_page_multi_chunk_tries_the_text_layer_first(tmp_path):
    """1쪽 청크(sidecar 엔진의 기본 구성)는 재시도까지 실패하면 플레이스홀더 대신
    텍스트 레이어부터 시도한다 — 무거운 페이지가 타임아웃을 두 번 내면 텍스트
    레이어가 있는데도 빈 페이지가 됐다."""
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=1)

    assert job.status == "done"
    assert engine.calls == 3  # 1쪽 ×2(실패) + 2쪽 ×1 — 같은 페이지를 또 OCR하지 않는다
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md
    assert "![](images/p0002_0.jpg)" in md


def test_retry_recovers_without_placeholder(tmp_path):
    """최초 실패 후 재시도가 성공하면 플레이스홀더/warnings 없이 정상 완료.
    호출 카운트로 재시도 1회가 실제 일어났음을 확인한다."""
    engine = FlakyEngine(fail_calls={1})
    job = _run_job(tmp_path, engine)

    assert job.status == "done"
    assert engine.calls == 3  # 청크1 실패 1 + 재시도 1 + 청크2 1
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg", "p0004_0.jpg"):
        assert f"![](images/{name})" in md
    assert job.warnings == []


def test_all_chunks_failed_is_error(tmp_path):
    """복구할 길이 하나도 없으면(페이지별 single도 실패, 텍스트 레이어 없음) 부분
    성공이 없으므로 기존대로 status=error."""
    engine = FlakyEngine(fail_all=True)
    job = _run_job(tmp_path, engine, embedded_text=False)

    assert job.status == "error"
    assert "모든 청크" in job.error
    # 청크 2개 × (multi 최초 + 재시도 + 페이지 2쪽 × single 최초·재시도)
    assert engine.calls == 12


def test_job_canceled_is_not_swallowed(tmp_path):
    """JobCanceled는 청크 격리에 삼켜지지 않고 그대로 전파 — 재시도도 없다."""
    engine = FlakyEngine(fail_calls={1}, exc=JobCanceled)
    job = _run_job(tmp_path, engine)

    assert job.status == "canceled"
    assert engine.calls == 1  # 재시도 없이 즉시 취소 처리


def test_per_page_mode_failed_page_recovers_from_embedded_text(tmp_path):
    """per_page single 최종 실패도 원본 PDF 텍스트 레이어로 복구한다."""
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(tmp_path, engine, pages=2, mode="per_page")

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "PDF 내장 텍스트 레이어" in md and "Sample page 1" in md
    assert "![](images/p0002_0.jpg)" in md
    assert len(job.warnings) == 1 and "텍스트 레이어로 복구" in job.warnings[0]


def test_per_page_mode_without_text_layer_still_uses_placeholder(tmp_path):
    engine = FlakyEngine(fail_calls={1, 2})
    job = _run_job(
        tmp_path,
        engine,
        pages=2,
        mode="per_page",
        embedded_text=False,
    )

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert md.count(FAILED_MARK) == 1
    assert len(job.warnings) == 1 and "플레이스홀더" in job.warnings[0]


def test_repetitive_multi_chunk_falls_back_to_single_pages(tmp_path):
    engine = LoopFallbackEngine()
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 2  # 반복 난 청크는 같은 multi로 재시도하지 않음
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert len(md.split("\n\n---\n\n")) == 4
    for page in range(1, 5):
        assert f"![](images/p{page:04d}_0.jpg)" in md
    assert not (job.dir / "images" / "p0001_99.jpg").exists()
    assert any("반복/출력 상한 감지로 페이지별 재처리" in warning for warning in job.warnings)


def test_single_fallback_failure_recovers_only_that_page_from_pdf_text(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={2})
    job = _run_job(tmp_path, engine, pages=4, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "![](images/p0001_0.jpg)" in md
    assert "![](images/p0003_0.jpg)" in md
    assert "![](images/p0004_0.jpg)" in md
    assert "![](images/p0002_0.jpg)" not in md
    assert "Sample page 2" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()
    assert not (job.dir / "layout" / "page_0002.jpg").exists()
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    # 복구된 페이지도 layout.json에 자리를 유지한다(좌표는 없으므로 blocks는 비운다).
    # 통째로 빠지면 facsimile 내보내기에서 페이지가 사라지고 /page/2가 404가 되며
    # pdf_export의 원문↔번역 페이지 정렬 불변식이 깨진다.
    assert {page["page"] for page in layout} == {1, 2, 3, 4}
    assert layout[1]["blocks"] == []


def test_single_fallback_repetition_goes_directly_to_pdf_text_without_retry(tmp_path):
    engine = LoopFallbackEngine(loop_single_pages={2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 2" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()
    assert any("RepetitiveOutputError" in warning for warning in job.warnings)


def test_single_fallback_retry_discards_first_attempt_artifacts(tmp_path):
    engine = LoopFallbackEngine(fail_single_once_pages={2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "![](images/p0002_0.jpg)" in md
    assert not (job.dir / "images" / "p0002_99.jpg").exists()


def test_all_single_fallback_pages_can_all_recover_from_pdf_text(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={1, 2})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "done"
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 2, 2: 2}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert md.count("PDF 내장 텍스트 레이어") == 2
    assert "Sample page 1" in md and "Sample page 2" in md


def test_all_single_fallback_pages_without_text_keep_all_failed_contract(tmp_path):
    engine = LoopFallbackEngine(fail_single_pages={1, 2})
    job = _run_job(
        tmp_path,
        engine,
        pages=2,
        pages_per_chunk=2,
        embedded_text=False,
    )

    assert job.status == "error"
    assert "모든 청크" in job.error
    assert engine.multi_calls == 1
    assert engine.single_calls == {1: 2, 2: 2}


def test_per_page_repetition_goes_directly_to_embedded_text_without_retry(tmp_path):
    engine = LoopFallbackEngine(loop_single_pages={1})
    job = _run_job(tmp_path, engine, pages=2, mode="per_page")

    assert job.status == "done"
    assert engine.multi_calls == 0
    assert engine.single_calls == {1: 1, 2: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md
    assert not (job.dir / "images" / "p0001_99.jpg").exists()
    assert not (job.dir / "layout" / "page_0001.jpg").exists()


def test_cancel_wins_over_multi_repetition_fallback(tmp_path):
    engine = LoopFallbackEngine(cancel_on_multi_loop=True)
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)

    assert job.status == "canceled"
    assert engine.multi_calls == 1
    assert engine.single_calls == {}


def test_canceled_job_keeps_the_warnings_it_accumulated(tmp_path):
    """취소로 끝나도 그때까지의 경고는 남는다 — 부분 결과만 있고 이유가 없으면
    사용자는 정상 변환된 부분과 구분할 수 없다."""

    class FailThenCancelEngine(FakeEngine):
        def __init__(self):
            super().__init__(delay=0.0)
            self.calls = 0

        def run_multi(self, image_paths, out_dir, sink, cancel):
            self.calls += 1
            if self.calls <= 2:  # 청크1: 최초 + 재시도 실패 → 플레이스홀더 + 경고
                raise RuntimeError("모의 실패")
            cancel.set()  # 청크2 진입 시 취소
            raise JobCanceled()

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=4, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=2,
    )
    engine = FailThenCancelEngine()
    engine.load()
    execute_job(job, store, broker, engine, settings, threading.Event())

    assert job.status == "canceled"
    assert job.warnings, "취소 전에 쌓인 경고가 사라졌다"
    assert any("페이지별 재처리" in w for w in job.warnings)
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["warnings"] == job.warnings


class _DeviceTensor:
    """실패 시도의 KV·활성화 텐서 대역 — 살아 있는지 weakref로 본다."""


class TensorPinningEngine(FakeEngine):
    """run_multi 프레임 지역 변수에 '텐서'를 쥔 채 실패하는 엔진.

    retry_alive[k]: k번째 호출 시점에 그 이전 실패 시도들의 텐서가 살아 있었는가.
    """

    def __init__(self, fail_calls):
        super().__init__(delay=0.0)
        self.fail_calls = set(fail_calls)
        self.calls = 0
        self.refs: list = []
        self.retry_alive: list[bool] = []

    def _attempt(self):
        import weakref

        self.calls += 1
        self.retry_alive.append(any(r() is not None for r in self.refs))
        tensor = _DeviceTensor()  # noqa: F841 — 프레임 지역 변수로 붙잡히는 것이 핵심
        self.refs.append(weakref.ref(tensor))
        if self.calls in self.fail_calls:
            raise RuntimeError("MPS backend out of memory (모의)")

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self._attempt()
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        self._attempt()
        return super().run_single(image_path, out_dir, sink, cancel)


def test_failed_attempt_tensors_are_released_before_the_retry(tmp_path):
    """예외 객체를 보관하면 traceback → 실패 프레임 → 텐서가 살아 있어, 재시도 직전의
    캐시 해제가 아무것도 돌려받지 못하고 같은 OOM이 재발한다(순환 GC 전까지 유지)."""
    import gc

    engine = TensorPinningEngine(fail_calls={1})
    gc.disable()  # 순환 GC에 기대지 않는다 — 참조를 직접 끊어야 한다
    try:
        job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2)
    finally:
        gc.enable()

    assert job.status == "done"
    assert engine.calls == 2
    assert engine.retry_alive == [False, False], "재시도 시점에 실패 시도의 텐서가 살아 있다"


def test_final_chunk_errors_do_not_pin_tensors_after_the_job(tmp_path):
    import gc

    engine = TensorPinningEngine(fail_calls=set(range(1, 7)))  # multi ×2 + single 2쪽 ×2
    gc.disable()
    try:
        job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=2, embedded_text=False)
        alive = [r() is not None for r in engine.refs]
    finally:
        gc.enable()

    assert job.status == "error" and "모든 청크" in job.error
    assert not any(alive), alive


# ── MAX_LENGTH 도달(OutputLimitError) ────────────────────────────────────


class OutputLimitEngine(FakeEngine):
    """첫 multi 호출이 MAX_LENGTH에서 잘리는 엔진.

    complete_pages쪽까지는 끝까지 생성하고 그다음 페이지 중간에서 잘린다. 산출물
    (크롭·오버레이·raw_pages.json)은 잘린 페이지 것까지 out_dir에 남긴다 — 벤더가
    생성이 끝난 뒤 save_results를 하는 것과 같다. attach_partial이면 엔진 계약대로
    OutputLimitError.partial_output에 run_multi 형식의 잘린 출력을 싣는다.
    """

    def __init__(self, complete_pages: int, *, attach_partial: bool = True):
        super().__init__(delay=0.0)
        self.complete_pages = complete_pages
        self.attach_partial = attach_partial
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}

    def run_multi(self, image_paths, out_dir, sink, cancel):
        from app.engine.base import OutputLimitError
        from app.pipeline.merge import split_pages

        self.multi_calls += 1
        if self.multi_calls > 1:
            return super().run_multi(image_paths, out_dir, sink, cancel)
        full = super().run_multi(image_paths[: self.complete_pages + 1], out_dir, sink, cancel)
        pages = split_pages(full)
        pages[-1] = pages[-1][: len(pages[-1]) // 3]  # 마지막 페이지는 중간에서 잘렸다
        partial = "<PAGE>\n" + "\n<PAGE>\n".join(pages) if self.attach_partial else None
        raise OutputLimitError(
            "생성이 총 길이 상한(MAX_LENGTH=32768)에 도달해 출력이 잘림", partial_output=partial,
        )

    def run_single(self, image_path, out_dir, sink, cancel):
        page = int(Path(image_path).stem.rsplit("_", 1)[-1])
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        return super().run_single(image_path, out_dir, sink, cancel) + "\n\nSINGLE-RUN"


def _run_job_events(tmp_path, engine, *, pages, pages_per_chunk, mode="multi"):
    import queue

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", mode, dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=pages, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=pages_per_chunk,
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


def test_output_limit_keeps_completed_pages_and_reprocesses_the_rest(tmp_path):
    """MAX_LENGTH에서 잘린 8쪽 청크를 통째로 다시 돌리지 않는다 — 끝까지 생성된 앞
    페이지는 multi 결과를 지키고, 잘린 페이지부터만 페이지별로 다시 처리한다."""
    from tests.test_fidelity_gate import client_view

    engine = OutputLimitEngine(complete_pages=2)
    job, events = _run_job_events(tmp_path, engine, pages=4, pages_per_chunk=4)

    assert job.status == "done"
    assert engine.multi_calls == 1           # 잘린 청크를 같은 multi로 다시 돌리지 않는다
    assert engine.single_calls == {3: 1, 4: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    segments = md.split("\n\n---\n\n")
    assert len(segments) == 4
    assert "SINGLE-RUN" not in segments[0] and "SINGLE-RUN" not in segments[1]
    assert "SINGLE-RUN" in segments[2] and "SINGLE-RUN" in segments[3]
    for page in range(1, 5):
        assert f"![](images/p{page:04d}_0.jpg)" in segments[page - 1]
    # 잘린 페이지의 multi 크롭이 마지막 보존 페이지로 접혀 들어가지 않는다
    assert not list((job.dir / "images").glob("p0002_x*"))
    layout = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == [1, 2, 3, 4]
    assert any("MAX_LENGTH 도달" in w and "앞 2쪽은 유지" in w for w in job.warnings), job.warnings
    # 라이브 스트림도 페이지당 세그먼트 하나 — 잘린 페이지 출력은 물려졌다
    assert client_view(events).count("<PAGE>") == 4


def test_output_limit_without_partial_output_reprocesses_every_page(tmp_path):
    engine = OutputLimitEngine(complete_pages=2, attach_partial=False)
    job, _events = _run_job_events(tmp_path, engine, pages=4, pages_per_chunk=4)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 1, 3: 1, 4: 1}
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert md.count("SINGLE-RUN") == 4
    assert any("MAX_LENGTH 도달" in w for w in job.warnings), job.warnings


def test_output_limit_in_per_page_mode_names_the_cause(tmp_path):
    """single 호출이 MAX_LENGTH에 닿아도 잘린 출력을 채택하지 않고 텍스트 레이어로
    복구하며, 경고에 원인(MAX_LENGTH 도달)을 밝힌다."""
    from app.engine.base import OutputLimitError

    class SingleLimitEngine(FakeEngine):
        def run_single(self, image_path, out_dir, sink, cancel):
            raise OutputLimitError("생성이 MAX_LENGTH에 도달해 출력이 잘림")

    job = _run_job(tmp_path, SingleLimitEngine(delay=0.0), pages=1, mode="per_page")

    assert job.status == "done"
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md
    assert any("MAX_LENGTH 도달" in w and "텍스트 레이어로 복구" in w for w in job.warnings)


def test_max_length_smaller_than_chunk_budget_is_logged(tmp_path, caplog):
    import logging

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data", preload_model=False,
        fake_delay=0.0, pages_per_chunk=8, max_length=32768, max_page_output_tokens=6144,
    )
    engine = FakeEngine(delay=0.0)
    engine.load()
    with caplog.at_level(logging.WARNING, logger="app.pipeline.runner"):
        execute_job(job, store, broker, engine, settings, threading.Event())
    assert job.status == "done"
    assert any("MAX_LENGTH(32768)" in r.message and "8쪽" in r.message for r in caplog.records)

    caplog.clear()
    job2 = store.create("doc.pdf", "multi", dpi=72)
    (job2.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=2, with_image=False))
    settings.max_length = 8 * 6144 + 4096
    with caplog.at_level(logging.WARNING, logger="app.pipeline.runner"):
        execute_job(job2, store, broker, engine, settings, threading.Event())
    assert not any("MAX_LENGTH(" in r.message for r in caplog.records)


# ── 취소 ─────────────────────────────────────────────────────────────────


class CancelMidChunkEngine(FakeEngine):
    """multi 도중 취소가 들어오면(엔진은 부분 출력을 정상 반환한다) 앞 두 쪽만 낸다."""

    def __init__(self, cancel_event):
        super().__init__(delay=0.0)
        self.cancel_event = cancel_event
        self.single_calls = 0

    def run_multi(self, image_paths, out_dir, sink, cancel):
        md = super().run_multi(image_paths[:2], out_dir, sink, cancel)
        self.cancel_event.set()
        return md

    def run_single(self, image_path, out_dir, sink, cancel):
        self.single_calls += 1
        return super().run_single(image_path, out_dir, sink, cancel)


def test_canceled_chunk_skips_marker_correction_and_the_fidelity_gate(tmp_path):
    """취소한 잡에 '페이지 마커 2개(기대 8)'·'충실도 미달·예산 부족' 같은 품질 경고가
    영구 기록되고, 진행률이 청크 끝(8쪽)으로 뛰고, 게이트가 되감기·재처리를 하던 문제."""
    import queue

    from tests.test_fidelity_gate import make_texty_pdf

    cancel = threading.Event()
    engine = CancelMidChunkEngine(cancel)
    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_texty_pdf(8))  # 게이트가 판정할 수 있는 문서
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, pages_per_chunk=8,
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

    assert job.status == "canceled"
    assert engine.single_calls == 0                  # 게이트 재처리 없음
    assert [e for e, _ in events if e == "reset"] == []
    assert job.warnings == ["1–8페이지: 취소로 중단된 청크 — 생성된 부분까지만 병합했습니다"]
    assert job.progress["current_page"] <= 2         # 실제로 처리한 페이지까지만
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert "페이지 1" in md and "페이지 2" in md     # 부분 결과는 보존된다


# ── 페이지 단위 엔진·타임아웃 재시도 정책 ──────────────────────────────────


class _NoSamePageRetry(RuntimeError):
    """sidecar 읽기 타임아웃 대역 — 같은 페이지 즉시 재요청은 무의미하다."""

    retry_same_page = False


class PageUnitEngine(FakeEngine):
    """sidecar처럼 페이지 단위로 도는 엔진(동시성 2 → 청크 2쪽). bad_pages는 늘 실패."""

    def __init__(self, bad_pages=(), exc=RuntimeError, chunk=2):
        super().__init__(delay=0.0)
        self.bad_pages = set(bad_pages)
        self.exc = exc
        self.chunk = chunk
        self.multi_calls = 0
        self.single_calls: dict[int, int] = {}

    def capabilities(self):
        from app.engine.base import EngineCapabilities

        return EngineCapabilities(
            model_id="page-unit", supports_multi_page=False,
            preferred_chunk_size=self.chunk, stream_granularity="page",
        )

    @staticmethod
    def _page(image_path) -> int:
        return int(Path(image_path).stem.rsplit("_", 1)[-1])

    def run_multi(self, image_paths, out_dir, sink, cancel):
        self.multi_calls += 1
        bad = [self._page(p) for p in image_paths if self._page(p) in self.bad_pages]
        if bad:
            raise self.exc(f"{bad[0]}페이지 sidecar 추론 실패 (HTTP 502)")
        return super().run_multi(image_paths, out_dir, sink, cancel)

    def run_single(self, image_path, out_dir, sink, cancel):
        page = self._page(image_path)
        self.single_calls[page] = self.single_calls.get(page, 0) + 1
        if page in self.bad_pages:
            raise self.exc(f"{page}페이지 sidecar 추론 실패 (HTTP 502)")
        return super().run_single(image_path, out_dir, sink, cancel)


def test_page_unit_engine_isolates_a_bad_page_without_rerunning_the_chunk(tmp_path):
    """동시성>1에서 한 페이지의 실패가 같은 청크의 정상 페이지를 GPU에서 다시 추론시키고
    전부 플레이스홀더로 만들던 문제 — 청크 재시도 없이 페이지별로 내린다."""
    engine = PageUnitEngine(bad_pages={1})
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=8)

    assert job.status == "done"
    assert engine.multi_calls == 1                 # 청크 통째 재시도 없음
    assert engine.single_calls == {1: 2, 2: 1}     # 정상 페이지는 한 번, 실패 페이지만 재시도
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert FAILED_MARK not in md
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md
    assert "![](images/p0002_0.jpg)" in md


def test_no_same_page_retry_errors_go_straight_to_page_isolation(tmp_path):
    """읽기 타임아웃처럼 같은 페이지 즉시 재요청이 무의미한 실패는 재시도하지 않는다 —
    재요청은 버려진 추론 뒤에 줄을 서 다시 타임아웃이 나고 시간만 2배가 된다."""
    engine = PageUnitEngine(bad_pages={1}, exc=_NoSamePageRetry, chunk=1)
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=8)

    assert job.status == "done"
    assert engine.multi_calls == 2                 # 1쪽(실패, 재시도 없음) + 2쪽
    assert engine.single_calls == {}               # 1쪽 청크는 텍스트 레이어로 바로 간다
    md = (job.dir / "result.md").read_text(encoding="utf-8")
    assert "Sample page 1" in md and "PDF 내장 텍스트 레이어" in md


def test_no_same_page_retry_also_applies_to_per_page_fallbacks(tmp_path):
    engine = PageUnitEngine(bad_pages={1}, exc=_NoSamePageRetry, chunk=2)
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=8)

    assert job.status == "done"
    assert engine.single_calls == {1: 1, 2: 1}     # 실패 페이지도 한 번만


def test_retry_and_rerun_policies_are_declared_on_the_base_contracts(tmp_path):
    """덕 타이핑이던 runner 계약을 base.py가 선언한다 — EngineError.retry_same_page(기본
    True), OCREngine.deterministic_rerun(기본 False). 선언된 속성만으로 정책이 바뀐다."""
    from app.engine.base import EngineError, OCREngine
    from app.engine.textlayer import TextLayerEngine

    assert EngineError("x").retry_same_page is True
    assert OCREngine.deterministic_rerun is False
    assert FakeEngine.deterministic_rerun is False
    assert TextLayerEngine.deterministic_rerun is True

    class _TimeoutLike(EngineError):
        retry_same_page = False

    engine = PageUnitEngine(bad_pages={1}, exc=_TimeoutLike, chunk=1)
    job = _run_job(tmp_path, engine, pages=2, pages_per_chunk=8)

    assert job.status == "done"
    assert engine.multi_calls == 2                 # 1쪽(실패, 재시도 없음) + 2쪽
    assert engine.single_calls == {}
