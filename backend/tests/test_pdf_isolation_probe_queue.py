"""업로드 검증(probe) 대기열 — 바쁠 때 서버 스레드를 오래 붙잡지 않는다.

API는 probe_pdf를 공용 anyio 스레드풀(모든 동기 라우트와 공유, 기본 40)에서 돌린다. 예전에는
빈 probe 워커를 기다리는 상한이 검증 상한(PDF_PAGE_TIMEOUT_S, 기본 60초)과 같고 대기 수에도
제한이 없어, 조금 무거운 업로드 수십 건이 그 스레드를 60초씩 물고 /api/health까지 멈췄다(감사
security-1 — 40건 동시 업로드에 health 21초, 80건에 59초 정지). 동시에 검증을 기다리는 업로드
수와 대기 시간을 묶어 넘치는 업로드는 곧바로 바쁨(503 + Retry-After)으로 끝낸다.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.pipeline import pdf as pdf_mod
from app.pipeline import pdf_worker

from conftest import make_pdf_bytes


@pytest.mark.parametrize(
    ("page_timeout", "expected_wait"),
    [("60", 5.0), ("0", 5.0), ("0.5", 0.5)],
)
def test_probe_waits_briefly_for_a_worker_not_the_whole_page_limit(
    monkeypatch, tmp_path, page_timeout, expected_wait,
):
    calls = []
    monkeypatch.setenv("PDF_WORKER_MODE", "process")
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", page_timeout)
    monkeypatch.setattr(pdf_worker, "run", lambda *a, **kw: calls.append(kw) or 1)
    assert pdf_mod.probe_pdf(tmp_path / "x.pdf", 10) == 1
    assert calls[0]["wait"] == expected_wait  # 검증 상한(최대 60초·무제한)이 아니다
    assert calls[0]["timeout"] == (float(page_timeout) or None)  # 검증 자체의 상한은 그대로


def test_excess_concurrent_probes_are_refused_without_waiting(
    pdf_worker_processes, monkeypatch, tmp_path,
):
    """probe 워커 둘이 모두 바쁠 때 동시에 올라온 업로드 8건 — 대기 슬롯(4)을 넘는 것은
    기다리지 않고 바로 바쁨으로 끝나 스레드를 놓는다. 대기한 것도 짧은 대기 상한에서 끝난다."""
    sample = tmp_path / "doc.pdf"
    sample.write_bytes(make_pdf_bytes(pages=1, with_image=False))
    holders = [
        threading.Thread(
            target=pdf_worker_processes.run, args=("pdf_worker_tasks:sleep", (3,)),
            kwargs={"pool": "probe", "timeout": 30},
        )
        for _ in range(2)  # probe 풀 크기만큼 붙잡는다
    ]
    for holder in holders:
        holder.start()
    deadline = time.monotonic() + 10
    while pdf_worker_processes.pool_stats()["pools"].get("probe", {}).get("in_use", 0) < 2:
        assert time.monotonic() < deadline, "probe 워커를 붙잡지 못했다"
        time.sleep(0.05)
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "1.0")  # 대기 상한 = min(검증 상한, 5초) = 1초

    outcomes: list[tuple[float, str]] = []
    lock = threading.Lock()

    def upload() -> None:
        started = time.monotonic()
        try:
            pdf_mod.probe_pdf(sample, 10)
            result = "ok"
        except pdf_worker.PdfWorkerBusy:
            result = "busy"
        with lock:
            outcomes.append((time.monotonic() - started, result))

    uploads = [threading.Thread(target=upload) for _ in range(8)]
    for thread in uploads:
        thread.start()
    for thread in uploads:
        thread.join(20)
    for holder in holders:
        holder.join(20)

    assert [result for _, result in outcomes] == ["busy"] * 8
    waited = [elapsed for elapsed, _ in outcomes if elapsed >= 0.5]
    assert len(waited) <= pdf_mod._PROBE_MAX_IN_FLIGHT, sorted(outcomes)
    assert all(elapsed < 3 for elapsed, _ in outcomes), sorted(outcomes)
    # 슬롯은 모두 반납됐다 — 바쁨이 풀리면 다시 검증한다
    assert pdf_mod.probe_pdf(sample, 10) == 1
