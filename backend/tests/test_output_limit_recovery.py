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
