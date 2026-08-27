"""라이브 스트림 프레이밍 계약 — "k번째 <PAGE> 세그먼트 == 글로벌 페이지 k".

라이브 3패널(레이아웃 박스·RAW·미리보기)은 전부 이 토큰 스트림 하나만 보고
페이지를 센다. 모델의 마커에만 의존하면 run_single(마커 없음)과 재처리 폴백에서
프레이밍이 무너져 진행률이 멈추고 이후 모든 박스가 한 페이지에 쌓인다 —
실제로 46페이지 논문에서 107개 박스가 18페이지 하나에 몰리고 19–24페이지는
통째로 사라졌다. 이 스위트는 서버가 그 불변식을 지키는지 검증한다.
"""

import queue
import re
import threading

from app.config import Settings
from app.engine.base import RepetitiveOutputError
from app.jobs import EventBroker, JobStore
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes
from test_runner_failures import FlakyEngine, LoopFallbackEngine

PAGE_MARKER = "<PAGE>"


def run_with_events(
    tmp_path, engine, *, pages=4, mode="multi", pages_per_chunk=2, embedded_text=True
):
    """execute_job을 구동하며 브로커 이벤트를 전부 수집한다 (구독 → 실행 → 배수)."""
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
    q = broker.subscribe(job.id)
    engine.load()
    execute_job(job, store, broker, engine, settings, threading.Event())
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return job, broker, events


def stream_text(events) -> str:
    """클라이언트가 최종적으로 갖게 되는 원문 — token을 이어 붙이고 reset을 적용한다.

    reset은 과거 이벤트를 지우는 게 아니라 "여기부터 물려라"라는 지시다
    (frontend/js/core.js truncateRawToPage와 같은 규칙: page N의 마커 지점에서 자른다)."""
    raw = ""
    for event, data in events:
        if event == "token":
            raw += data["text"]
        elif event == "reset":
            pos = -1
            for _ in range(max(1, int(data["from_page"]))):
                pos = raw.find(PAGE_MARKER, pos + 1)
                if pos == -1:
                    break
            if pos != -1:
                raw = raw[:pos]
    return raw


def published_text(events) -> str:
    """물림을 적용하지 않은 원시 발행 스트림 (reset 자체를 검증할 때만 쓴다)."""
    return "".join(d["text"] for e, d in events if e == "token")


def segments(text) -> list[str]:
    """마커 기준 세그먼트 — 인덱스 k가 곧 글로벌 페이지 k (0은 첫 마커 앞 서두)."""
    return text.split(PAGE_MARKER)


def ocr_pages(events) -> list[int]:
    """phase=ocr 진행 이벤트가 알린 current_page 순서."""
    return [d["current_page"] for e, d in events if e == "progress" and d.get("phase") == "ocr"]


def assert_framing(text, pages):
    """세그먼트 k의 본문이 페이지 k의 원본 이미지에서 나왔는지 (FakeEngine은
    md에 원본 stem `page_000k`를 적는다)."""
    segs = segments(text)
    assert len(segs) == pages + 1, f"마커 {len(segs) - 1}개 (기대 {pages}): {[s[:40] for s in segs]}"
    assert not segs[0].strip(), f"첫 마커 앞에 본문이 있다: {segs[0][:80]!r}"
    for page in range(1, pages + 1):
        stem = f"page_{page:04d}"
        assert stem in segs[page], f"세그먼트 {page}에 {stem}이 없다: {segs[page][:120]!r}"
        for other in range(1, pages + 1):
            if other != page:
                assert f"page_{other:04d}" not in segs[page], (
                    f"세그먼트 {page}가 페이지 {other} 내용을 함께 담고 있다"
                )


def test_multi_stream_is_framed_one_segment_per_page(tmp_path):
    """정상 multi 경로 — 기존 계약이 그대로 유지된다(회귀 가드)."""
    job, _broker, events = run_with_events(tmp_path, FlakyEngine(), pages=4)
    assert job.status == "done"
    assert_framing(stream_text(events), 4)
    assert ocr_pages(events) == [1, 2, 2, 3, 4, 4]


def test_per_page_mode_gets_server_injected_markers(tmp_path):
    """per_page 모드는 run_single만 쓴다 — 모델이 마커를 내지 않으므로 서버가 주입한다.

    주입 전에는 스트림 전체가 세그먼트 1개라 미리보기가 확정 페이지를 영영 만들지
    못하고, 누적 꼬리를 600ms마다 통째로 재전송했다(O(n²) → 2MB 413)."""
    job, _broker, events = run_with_events(
        tmp_path, FlakyEngine(), pages=4, mode="per_page", pages_per_chunk=1
    )
    assert job.status == "done"
    assert_framing(stream_text(events), 4)


def test_repetition_fallback_rewinds_and_reframes(tmp_path):
    """multi 청크가 반복/상한으로 폐기되면 reset으로 되돌리고 페이지별로 다시 프레이밍한다.

    이것이 사용자 신고의 직접 원인이었다: 폐기 통보가 없어 폐기된 출력이 클라이언트에
    영원히 남고, 재처리 출력에는 마커가 없어 진행률이 멈춘 채 모든 박스가 한 페이지에
    쌓였다."""
    class StreamThenLoopEngine(LoopFallbackEngine):
        """실엔진처럼 청크의 앞부분을 흘려보낸 뒤 상한에 걸린다.

        (아무것도 흘리지 않고 실패하면 물릴 것이 없어 reset도 나오지 않는다 —
        그건 별도 케이스로 아래에서 검증한다.)"""

        def run_multi(self, image_paths, out_dir, sink, cancel):
            if self.multi_calls == 0:
                sink.on_text("<PAGE>\n버려질 1페이지 출력 <table><tr><td>잘린 표")
            return super().run_multi(image_paths, out_dir, sink, cancel)

    engine = StreamThenLoopEngine()
    job, _broker, events = run_with_events(tmp_path, engine, pages=4, pages_per_chunk=2)
    assert job.status == "done"
    assert engine.multi_calls >= 1 and set(engine.single_calls) == {1, 2}
    assert "버려질" not in stream_text(events), "폐기한 출력이 스트림에 남았다"

    resets = [d for e, d in events if e == "reset"]
    assert resets, "폐기를 알리는 reset 이벤트가 없다"
    assert resets[0]["from_page"] == 1

    text = stream_text(events)
    assert_framing(text, 4)
    # 진행률이 재처리 페이지마다 다시 올라간다 (멈춘 채 방치되지 않는다)
    assert ocr_pages(events)[-1] == 4
    assert 1 in ocr_pages(events) and 2 in ocr_pages(events)


def test_rewind_also_truncates_reconnect_replay(tmp_path):
    """폐기한 출력은 재연결 replay 히스토리에서도 사라져야 한다 —
    그러지 않으면 재연결 한 번에 쓰레기가 그대로 되살아난다."""
    engine = LoopFallbackEngine()
    job, broker, _events = run_with_events(tmp_path, engine, pages=4, pages_per_chunk=2)
    assert job.status == "done"
    # done 이벤트가 히스토리를 비우므로, 실행 중 스냅샷 대신 브로커 내부 계약을
    # 직접 검증한다: 절단 후 발행 누계가 되돌아간 지점과 일치해야 한다.
    fresh = EventBroker()
    fresh.publish("j", "token", {"text": "<PAGE>one"})
    fresh.publish("j", "token", {"text": "<PAGE>two"})
    fresh.truncate_token_history("j", len("<PAGE>one"))
    _q, replay, truncated = fresh.subscribe_with_replay("j")
    assert replay == "<PAGE>one" and not truncated


def test_chunk_retry_rewinds_the_partial_stream(tmp_path):
    """첫 시도가 흘려보낸 부분 출력은 재시도 전에 되돌린다 (중복 방지)."""
    engine = FlakyEngine(fail_calls={1})  # 청크1 최초 실패 → 재시도 성공
    job, _broker, events = run_with_events(tmp_path, engine, pages=4, pages_per_chunk=2)
    assert job.status == "done"
    assert_framing(stream_text(events), 4)


def test_embedded_text_recovery_keeps_one_segment_per_page(tmp_path):
    """텍스트 레이어 복구 페이지도 라이브 스트림에 자기 세그먼트를 갖는다 —
    건너뛰면 이후 페이지의 프리뷰 경계·박스 귀속이 한 칸씩 밀린다."""
    engine = LoopFallbackEngine(fail_single_pages={2})
    job, _broker, events = run_with_events(tmp_path, engine, pages=4, pages_per_chunk=2)
    assert job.status == "done"
    text = stream_text(events)
    assert len(segments(text)) - 1 == 4, "복구 페이지 세그먼트가 빠졌다"


def test_failed_page_placeholder_keeps_one_segment_per_page(tmp_path):
    """OCR·텍스트 레이어 둘 다 실패한 페이지도 세그먼트를 차지한다."""
    engine = LoopFallbackEngine(fail_single_pages={2})
    job, _broker, events = run_with_events(
        tmp_path, engine, pages=4, pages_per_chunk=2, embedded_text=False
    )
    assert job.status == "done"
    text = stream_text(events)
    assert len(segments(text)) - 1 == 4
    assert "변환에 실패했습니다" in text


def test_single_output_page_markers_are_stripped(tmp_path):
    """모델이 single 출력에 마커를 흘려도 서버 프레이밍이 깨지지 않는다."""

    class MarkerLeakEngine(FlakyEngine):
        def run_single(self, image_path, out_dir, sink, cancel):
            md = super().run_single(image_path, out_dir, sink, cancel)
            sink.on_text("<PAGE>잘못된 마커<PAGE>")
            return md

    job, _broker, events = run_with_events(
        tmp_path, MarkerLeakEngine(), pages=3, mode="per_page", pages_per_chunk=1
    )
    assert job.status == "done"
    assert stream_text(events).count(PAGE_MARKER) == 3


def test_dropped_token_flags_the_subscriber_for_resync(tmp_path):
    """느린 구독자에게서 token을 버릴 때 표식을 남긴다 — 조용한 유실은 페이지
    귀속을 영구히 어긋나게 한다."""
    broker = EventBroker()
    q = broker.subscribe("j")
    for _ in range(3000):  # _EVENT_QUEUE_MAX(2000) 초과
        broker.publish("j", "token", {"text": "x"})
    assert q.token_dropped is True
    text, truncated = broker.resync("j", q)
    assert q.token_dropped is False and not truncated
    assert len(text) == 3000, "히스토리는 유실 없이 전부 보존한다"
    assert q.qsize() == 0, "대기 중 token은 replay와 중복되므로 버린다"


def test_repetition_error_class_is_still_the_trigger():
    """폴백 트리거 계약 회귀 가드 — 이 예외가 곧 '재처리 + rewind' 경로다."""
    assert issubclass(RepetitiveOutputError, Exception)
    assert re.search(r"\d", "16384")


def test_excess_page_markers_are_capped_to_the_chunk(tmp_path):
    """모델이 마커를 더 내도 라이브 세그먼트 수는 청크 페이지 수를 넘지 않는다.

    merge는 초과분을 마지막 페이지에 합쳐 result.md를 N페이지로 유지한다 —
    라이브가 더 세면 그 뒤 모든 페이지의 박스·프리뷰 경계가 밀린다.
    (실측: 46페이지 잡의 1청크가 8페이지에 마커 9개를 냈다.)"""

    class ExtraMarkerEngine(FlakyEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            md = super().run_multi(image_paths, out_dir, sink, cancel)
            sink.on_text("<PAGE>덤으로 나온 마커")
            return md

    job, _broker, events = run_with_events(
        tmp_path, ExtraMarkerEngine(), pages=4, pages_per_chunk=2
    )
    assert job.status == "done"
    assert stream_text(events).count(PAGE_MARKER) == 4


def test_missing_page_markers_are_padded_to_the_chunk(tmp_path):
    """모델이 마커를 덜 내면 청크 끝에서 채워 세그먼트 수를 맞춘다."""

    class SwallowMarkerEngine(FlakyEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            real = sink.on_text
            seen = {"n": 0}

            def gated(text: str) -> None:
                # 청크의 두 번째 마커를 삼킨다 (모델이 페이지 경계를 놓친 상황)
                if PAGE_MARKER in text:
                    seen["n"] += 1
                    if seen["n"] == 2:
                        text = text.replace(PAGE_MARKER, "")
                real(text)

            sink.on_text = gated
            try:
                return super().run_multi(image_paths, out_dir, sink, cancel)
            finally:
                del sink.on_text

    job, _broker, events = run_with_events(
        tmp_path, SwallowMarkerEngine(), pages=4, pages_per_chunk=2
    )
    assert job.status == "done"
    assert stream_text(events).count(PAGE_MARKER) == 4
    assert ocr_pages(events)[-1] == 4


def test_page_marker_split_across_deltas_is_framed_once(tmp_path):
    """마커가 델타 경계에 쪼개져 도착해도(`<PA` + `GE>`) 정확히 한 번만 센다."""

    class SplitMarkerEngine(FlakyEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            real = sink.on_text

            def split(text: str) -> None:
                while PAGE_MARKER in text:
                    head, _, rest = text.partition(PAGE_MARKER)
                    real(head + PAGE_MARKER[:3])
                    real(PAGE_MARKER[3:])
                    text = rest
                if text:
                    real(text)

            sink.on_text = split
            try:
                return super().run_multi(image_paths, out_dir, sink, cancel)
            finally:
                del sink.on_text

    job, _broker, events = run_with_events(
        tmp_path, SplitMarkerEngine(), pages=4, pages_per_chunk=2
    )
    assert job.status == "done"
    assert_framing(stream_text(events), 4)


def test_page_tail_is_flushed_before_the_next_page_announcement(tmp_path):
    """페이지 k의 마지막 토큰은 페이지 k+1 선언보다 **먼저** 나간다.

    순서가 뒤집히면 클라이언트가 페이지 끝 블록의 박스를 다음 페이지 오버레이에
    그린다 (applyProgress가 선언 전에 drain하기 때문)."""
    job, _broker, events = run_with_events(tmp_path, FlakyEngine(), pages=4)
    assert job.status == "done"
    seen = ""
    announced: set[int] = set()
    for event, data in events:
        if event == "token":
            seen += data["text"]
        elif event == "progress" and data.get("phase") == "ocr":
            page = data["current_page"]
            # 각 페이지의 **첫** 선언만 본다 — 청크 종료 시의 재발행은 이미
            # 그 페이지 본문이 나간 뒤라 순서 계약의 대상이 아니다.
            if page > 1 and page not in announced:
                announced.add(page)
                # 선언 시점에 이전 페이지의 본문은 이미 전부 전달돼 있어야 한다
                assert f"page_{page - 1:04d}" in seen, (
                    f"페이지 {page} 선언이 페이지 {page - 1} 본문보다 먼저 나갔다"
                )
                assert f"page_{page:04d}" not in seen, (
                    f"페이지 {page} 본문이 선언보다 먼저 나갔다"
                )


def test_rewind_is_silent_when_nothing_was_streamed_yet(tmp_path):
    """흘려보낸 출력이 없으면 reset을 내지 않는다 — 클라이언트가 물릴 것이 없는데
    "재처리합니다" 경고만 남기는 잡음을 막는다."""
    engine = LoopFallbackEngine()  # 아무것도 스트리밍하지 않고 첫 multi에서 실패
    job, _broker, events = run_with_events(tmp_path, engine, pages=4, pages_per_chunk=2)
    assert job.status == "done"
    assert [d for e, d in events if e == "reset"] == []
    assert_framing(stream_text(events), 4)


def _sink_with_events(tmp_path):
    """BrokerSink 하나와 그 브로커 구독 큐 — 프레이밍 단위 테스트용."""
    from app.pipeline.runner import BrokerSink

    store = JobStore(tmp_path / "jobs")
    broker = EventBroker()
    job = store.create("doc.pdf", "multi", dpi=72)
    job.progress["total_pages"] = 8
    q = broker.subscribe(job.id)
    return BrokerSink(job, store, broker), q


def _drain(q) -> list[tuple[str, object]]:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_rewind_to_a_mid_chunk_page_actually_rewinds(tmp_path):
    """청크 **한가운데** 페이지도 되감을 수 있어야 한다.

    예전에는 `set_chunk`만 되감기 마크를 남겨 청크의 **첫 페이지에만** 지점이
    있었다. 그래서 8쪽 청크의 중간 페이지를 `rewind_to`로 물리려 하면 아무
    이벤트도 내지 않고 조용히 no-op이 됐고, 그 뒤 `emit_page`를 부르면 세그먼트가
    하나 늘어 "k번째 <PAGE> == 페이지 k" 불변식이 깨졌다. 충실도 게이트는 청크
    중간 페이지를 교체하므로 이 성질에 의존한다.
    """
    sink, q = _sink_with_events(tmp_path)
    sink.set_chunk(1, expect_markers=True, num_pages=4)
    sink.on_text("<PAGE>alpha<PAGE>bravo<PAGE>charlie<PAGE>delta")
    sink.finish_chunk()
    sink.flush()
    before = published_text(_drain(q))
    assert len(segments(before)) == 5, before

    sink.rewind_to(3, "테스트 — 중간 페이지 재처리")
    events = _drain(q)
    resets = [d for e, d in events if e == "reset"]
    assert resets, "중간 페이지 되감기가 reset을 내지 않았다 (무음 no-op)"
    assert resets[0]["from_page"] == 3, resets

    # 되감은 뒤 3·4쪽을 다시 채우면 세그먼트 수가 그대로여야 한다.
    sink.emit_page(3, "charlie-fixed")
    sink.emit_page(4, "delta")
    sink.flush()
    after = published_text(_drain(q))
    tail_segments = after.split(PAGE_MARKER)
    # 3쪽부터 다시 채웠으므로 새 스트림 조각에는 마커가 정확히 2개 있다.
    assert len(tail_segments) - 1 == 2, tail_segments
    assert "charlie-fixed" in after and "delta" in after
    assert "charlie" not in after.replace("charlie-fixed", "")


def test_every_page_gets_a_rewind_mark_not_only_chunk_starts(tmp_path):
    """마크는 페이지마다 남아야 한다 — 청크 시작에만 남기면 중간 교체가 불가능하다."""
    sink, _q = _sink_with_events(tmp_path)
    sink.set_chunk(5, expect_markers=True, num_pages=3)
    sink.on_text("<PAGE>five<PAGE>six<PAGE>seven")
    sink.finish_chunk()
    marks = sink._marks
    assert set(marks) >= {5, 6, 7}, marks
    # 마크는 단조 증가해야 한다 (페이지 k의 시작 오프셋)
    assert marks[5] <= marks[6] <= marks[7], marks


def test_filler_page_marks_do_not_point_into_the_previous_page(tmp_path):
    """`finish_chunk`의 채움 경로는 flush 전에 페이지를 열었다 — 마크가 앞
    페이지의 꼬리를 가리키면 되감기가 그것까지 지운다."""
    sink, _q = _sink_with_events(tmp_path)
    sink.set_chunk(1, expect_markers=True, num_pages=3)
    sink.on_text("<PAGE>only-page-one-content")
    sink.finish_chunk()  # 2·3쪽을 빈 세그먼트로 채운다
    marks = sink._marks
    assert set(marks) >= {1, 2, 3}, marks
    assert marks[2] >= len("<PAGE>only-page-one-content"), (
        f"채움 페이지 마크가 1쪽 본문 안을 가리킨다: {marks}"
    )
