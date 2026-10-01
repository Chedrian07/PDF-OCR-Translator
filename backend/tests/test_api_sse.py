"""SSE 스트림 회귀 테스트 — 재동기화(resync) 경로와 구독자 상한.

TestClient는 SSE 응답 전체를 버퍼링하므로 라이브 경로는 body_iterator를 직접 돌린다
(tests/test_api_translate.py의 같은 패턴).
"""

import asyncio
import json


def _drive(app, make_response, on_chunk) -> list[str]:
    """body_iterator를 끝까지 돌린다. on_chunk(index, chunk)는 청크마다 불린다."""
    from starlette.requests import Request

    async def drive():
        request = Request({"type": "http", "app": app, "headers": []})

        async def receive():
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        request._receive = receive
        response = await make_response(request)
        chunks: list[str] = []
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                on_chunk(len(chunks) - 1, chunk)
        finally:
            await response.body_iterator.aclose()
        return chunks

    return asyncio.run(asyncio.wait_for(drive(), timeout=10))


def _events(chunks: list[str]) -> list[tuple[str, dict]]:
    out = []
    for chunk in chunks:
        if chunk.startswith("event: "):
            head, data = chunk.split("\n", 1)
            out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# ── tests-baseline-2: resync 직후 이미 꺼낸 reset을 다시 보내지 않는다 ─────────────
def test_resync_does_not_redeliver_a_reset_it_already_dequeued(client, monkeypatch):
    """느린 구독자의 큐가 넘쳐 token을 버린 상태(token_dropped)에서 큐 머리의 reset을
    꺼내면, 루프는 누적 원문 replay(이미 절단 반영)를 보낸 뒤 그 reset을 또 보냈다 —
    클라이언트가 방금 받은 재처리 페이지 마커를 잘라 이후 페이지 귀속이 밀렸다."""
    from app.api import job_events

    store = client.app.state.store
    broker = client.app.state.broker
    job = store.create("resync.pdf", "multi", 200)
    job.status = "running"
    job.progress.update(phase="ocr", current_page=2, total_pages=2)
    # rewind 뒤의 히스토리: 2페이지는 이미 재처리분으로 갈아끼워져 있다
    broker.publish(job.id, "token", {"text": "<PAGE>\n1쪽\n<PAGE>\n2쪽 재처리"})
    captured: dict = {}
    real_subscribe = broker.subscribe_with_replay

    def _capture(job_id):
        q, replay, truncated = real_subscribe(job_id)
        captured["q"] = q
        return q, replay, truncated

    monkeypatch.setattr(broker, "subscribe_with_replay", _capture)

    def _on_chunk(index: int, chunk: str) -> None:
        if index == 2:                      # retry · progress · 접속 replay 다음
            q = captured["q"]
            q.put_nowait(("reset", {"page": 2}))   # 큐 머리에 남은 (이미 반영된) reset
            q.token_dropped = True                 # 그 사이 token 유실
            q.put_nowait(("done", {"markdown_url": "m", "archive_url": "a"}))

    try:
        chunks = _drive(client.app, lambda request: job_events(request, job.id), _on_chunk)
    finally:
        store.delete_dir(job)
    names = [name for name, _data in _events(chunks)]
    assert names == ["progress", "replay", "replay", "done"], names
    resynced = _events(chunks)[2][1]
    assert resynced["text"].endswith("2쪽 재처리")


# ── security-4: SSE 구독자 상한 ──────────────────────────────────────────────
def test_broker_caps_subscribers_per_channel_and_in_total():
    import pytest

    from app.jobs import EventBroker, SubscriberLimitError

    broker = EventBroker(max_per_channel=2, max_total=3)
    a1 = broker.subscribe("job-a")
    broker.subscribe_with_replay("job-a")
    assert not broker.has_room("job-a")
    with pytest.raises(SubscriberLimitError):
        broker.subscribe("job-a")                      # 채널 상한
    with pytest.raises(SubscriberLimitError):
        broker.subscribe_with_replay("job-a")
    broker.subscribe("job-b")
    with pytest.raises(SubscriberLimitError):
        broker.subscribe("job-c")                      # 전체 상한
    broker.unsubscribe("job-a", a1)                    # 끊긴 연결은 자리를 돌려준다
    assert broker.has_room("job-a") and broker.has_room("job-c") is True
    broker.subscribe("job-c")


def test_sse_routes_answer_503_with_retry_after_when_capped(client, sample_pdf):
    """상한 초과는 '잠시 뒤 재시도'다(프런트 계약: 503 + Retry-After). EventSource는 200이
    아니면 재연결하지 않으므로 프런트는 폴링으로 넘어간다."""
    from conftest import wait_done

    broker = client.app.state.broker
    jid = client.post(
        "/api/jobs", files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    ).json()["job_id"]
    wait_done(client, jid)
    held = [broker.subscribe(jid) for _ in range(broker.max_per_channel)]
    try:
        busy = client.get(f"/api/jobs/{jid}/events")
        assert busy.status_code == 503, busy.text
        assert busy.headers["Retry-After"] == "5"
    finally:
        for q in held:
            broker.unsubscribe(jid, q)
    with client.stream("GET", f"/api/jobs/{jid}/events") as stream:   # 자리가 나면 정상
        assert stream.status_code == 200

    channel = f"{jid}:translate:ko"
    held = [broker.subscribe(channel) for _ in range(broker.max_per_channel)]
    try:
        client.app.state.translate_tasks[(jid, "ko")] = {"thread": None, "cancel": None}
        busy = client.get(f"/api/jobs/{jid}/translate/events?lang=ko")
        assert busy.status_code == 503 and busy.headers["Retry-After"] == "5"
    finally:
        client.app.state.translate_tasks.pop((jid, "ko"), None)
        for q in held:
            broker.unsubscribe(channel, q)


def test_sse_stream_closes_quietly_when_the_cap_is_hit_after_the_check(client, monkeypatch):
    """판정과 구독 사이 경합 — 이벤트 없이 재연결 간격만 늘리고 닫는다(구독 누수 없음)."""
    from app.api import job_events
    from app.jobs import SubscriberLimitError

    store = client.app.state.store
    broker = client.app.state.broker
    job = store.create("race.pdf", "multi", 200)

    def _full(job_id):
        raise SubscriberLimitError(job_id)

    monkeypatch.setattr(broker, "subscribe_with_replay", _full)
    try:
        chunks = _drive(client.app, lambda request: job_events(request, job.id), lambda i, c: None)
    finally:
        store.delete_dir(job)
    assert chunks == ["retry: 5000\n\n"]
