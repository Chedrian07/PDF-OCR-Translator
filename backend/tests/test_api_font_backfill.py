"""리더 라우트의 폰트 메타 지연 백필(api._backfill_layout_fonts) — 감사 api-1.

ENRICH_VERSION이 오르면 옛 잡은 리더를 처음 열 때 layout[.lang].json을 백필한다. 실측은 export
풀의 PDF 워커에서 도는데, 요청 스레드가 그 풀의 빈 워커를 상한 없이 기다려 번역·대조 PDF
빌드가 도는 동안 /viewer/pages·/alignment·/outline·/page/{n}이 빌드가 끝날 때까지 멈췄고,
같은 잡의 동시 요청은 저마다 전 문서 백필을 따로 돌았다. 이제 백필은 산출물마다 하나뿐인
백그라운드 스레드가 맡고, 요청은 짧은 유예만 기다린다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from conftest import wait_done


def _upload(client, pdf_bytes: bytes):
    return client.post(
        "/api/jobs", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )


def _stale_job(client, sample_pdf) -> tuple[str, object, Path]:
    """완료 잡의 layout.json을 구버전 폰트 메타(fonts_v=1)로 되돌린다 — 업그레이드 직후의 옛 잡."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    job = client.app.state.store.get(jid)
    target = job.dir / "layout.json"
    pages = json.loads(target.read_text(encoding="utf-8"))
    for page in pages:
        page["fonts_v"] = 1
    target.write_text(json.dumps(pages), encoding="utf-8")
    return jid, job, target


def _stamps(pages_or_path) -> list:
    pages = pages_or_path
    if isinstance(pages_or_path, Path):
        pages = json.loads(pages_or_path.read_text(encoding="utf-8"))
    return [page.get("fonts_v") for page in pages]


def _wait_until(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("조건이 시간 내에 성립하지 않음")


def _no_backfill_in_flight() -> bool:
    import app.api as api_mod

    return not getattr(api_mod, "_FONT_BACKFILLS", {})


def _in_thread(target, *args) -> tuple[threading.Thread, dict]:
    result: dict = {}

    def _run():
        result["value"] = target(*args)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


def _gate_enrich(monkeypatch) -> dict:
    """enrich_layout_fonts를 게이트에 묶은 대역(실측 대신 fs를 심고 스탬프한다).

    잡을 만든 **뒤에** 건다 — 병합(merge)도 같은 함수를 불러 OCR이 게이트에 걸린다."""
    from app.pipeline import pdf_fonts

    gate = threading.Event()
    entered = threading.Event()
    calls: list[str] = []

    def _fake(pdf_path, pages):
        calls.append(threading.current_thread().name)
        entered.set()
        assert gate.wait(15), "게이트가 열리지 않았다"
        for page in pages:
            for block in page.get("blocks", ()):
                block["fs"] = 1.5
            page["fonts_v"] = pdf_fonts.ENRICH_VERSION
        return True

    monkeypatch.setattr(pdf_fonts, "enrich_layout_fonts", _fake)
    return {"gate": gate, "entered": entered, "calls": calls}


def _hold_export_worker(pdf_worker, seconds: float) -> None:
    """번역 PDF 빌드처럼 export 워커 하나를 붙잡는다(풀 종료로 끊겨도 조용히 끝난다)."""
    try:
        pdf_worker.run("pdf_worker_tasks:sleep", (seconds,), pool="export", timeout=60)
    except pdf_worker.PdfWorkerError:
        pass


def test_reader_routes_answer_while_a_build_holds_the_export_pool(
    pdf_worker_processes, client, sample_pdf, monkeypatch,
):
    """빌드가 export 워커를 쥔 동안에도 리더 라우트는 바로 답하고(지금 layout), 백필은 워커가
    비자 뒤에서 끝나 산출물을 갱신한다. 예전에는 라우트가 빌드가 끝날 때까지 멈췄다."""
    import app.api as api_mod
    from app.pipeline.pdf_fonts import ENRICH_VERSION

    monkeypatch.setenv("PDF_WORKER_MODE", "inline")       # 업로드·OCR은 빠르게
    jid, _job, target = _stale_job(client, sample_pdf)
    monkeypatch.setenv("PDF_WORKER_MODE", "process")      # 백필만 실제 워커로
    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "1")  # export 워커 1개 — 빌드 하나로 가득
    monkeypatch.setattr(api_mod, "_FONT_BACKFILL_GRACE_S", 0.2)
    stale = _stamps(target)

    holder = threading.Thread(target=_hold_export_worker, args=(pdf_worker_processes, 4.0))
    holder.start()
    _wait_until(
        lambda: pdf_worker_processes.pool_stats()["pools"].get("export", {}).get("in_use") == 1,
    )
    for url in (
        f"/api/jobs/{jid}/viewer/pages?start=1&limit=4",
        f"/api/jobs/{jid}/alignment?page=1",
        f"/api/jobs/{jid}/outline",
        f"/api/jobs/{jid}/page/1",
    ):
        assert client.get(url).status_code == 200, url
    assert holder.is_alive(), "리더 라우트가 export 풀이 빌 때까지 기다렸다"
    assert _stamps(target) == stale                       # 백필은 아직 워커를 기다린다

    holder.join(60)
    _wait_until(lambda: _stamps(target) == [ENRICH_VERSION] * len(stale))
    _wait_until(_no_backfill_in_flight)


def test_concurrent_reader_requests_share_one_backfill(client, sample_pdf, monkeypatch):
    """리더가 한꺼번에 부르는 라우트들이 저마다 전 문서 백필을 돌았다 — 이제 산출물마다 백필
    하나를 함께 기다리고, 유예 안에 끝나면 모두 실측 메타가 담긴 layout을 본다. 페이지
    이미지(/page/{n})는 폰트 메타가 필요 없어 그 백필을 기다리지 않는다."""
    import app.api as api_mod
    from app.pipeline.pdf_fonts import ENRICH_VERSION

    jid, job, target = _stale_job(client, sample_pdf)
    monkeypatch.setattr(api_mod, "_FONT_BACKFILL_GRACE_S", 30.0)
    gated = _gate_enrich(monkeypatch)
    st = client.app.state
    try:
        loaders = [_in_thread(api_mod._load_layout_pages, job, None, st) for _ in range(3)]
        assert gated["entered"].wait(5)
        time.sleep(0.3)                                   # 나머지 요청도 백필 지점에 닿는다

        page_thread, page = _in_thread(client.get, f"/api/jobs/{jid}/page/1")
        page_thread.join(5)
        assert not page_thread.is_alive(), "/page/1이 진행 중인 폰트 백필을 기다렸다"
        assert page["value"].status_code == 200
    finally:
        gated["gate"].set()
    for thread, _result in loaders:
        thread.join(10)
    assert len(gated["calls"]) == 1, gated["calls"]
    assert gated["calls"][0].startswith("font-backfill-")
    for _thread, result in loaders:
        pages = result["value"]
        assert _stamps(pages) == [ENRICH_VERSION] * len(pages)
        assert all(block.get("fs") == 1.5 for page in pages for block in page["blocks"])
    assert _stamps(target) == [ENRICH_VERSION] * len(_stamps(target))
    _wait_until(_no_backfill_in_flight)
    assert not list(job.dir.glob(".layout.*.tmp"))


def test_backfill_does_not_overwrite_a_layout_replaced_while_it_waited(
    client, sample_pdf, monkeypatch,
):
    """백필이 워커를 기다리는 사이 다른 쓰기(재병합·재번역)가 산출물을 바꾸면, 낡은 입력의
    결과로 덮지 않는다 — 빌드 뒤로 길어진 대기가 새 내용을 지우지 않게."""
    import app.api as api_mod

    _jid, job, target = _stale_job(client, sample_pdf)
    monkeypatch.setattr(api_mod, "_FONT_BACKFILL_GRACE_S", 0.1)
    gated = _gate_enrich(monkeypatch)
    try:
        loader, _ = _in_thread(api_mod._load_layout_pages, job, None, client.app.state)
        assert gated["entered"].wait(5)
        newer = json.loads(target.read_text(encoding="utf-8"))
        newer[0]["blocks"][0]["content"] = "재병합된 새 내용"
        staged = target.with_name("layout.json.new")
        staged.write_text(json.dumps(newer), encoding="utf-8")
        os.replace(staged, target)
        expected = target.read_bytes()
    finally:
        gated["gate"].set()
    loader.join(10)
    _wait_until(_no_backfill_in_flight)
    assert target.read_bytes() == expected
    assert not list(job.dir.glob(".layout.*.tmp"))


def test_backfill_cut_short_by_shutdown_is_not_saved(
    pdf_worker_processes, client, sample_pdf, monkeypatch,
):
    """종료가 풀을 닫으면 enrich_layout_fonts는 남은 페이지를 실측 없이 스탬프만 해 돌려준다
    — 그것을 저장하면 그 잡은 다음 ENRICH_VERSION 상향 전까지 실측 메타를 얻지 못한다."""
    import app.api as api_mod

    monkeypatch.setenv("PDF_WORKER_MODE", "inline")
    _jid, job, target = _stale_job(client, sample_pdf)
    monkeypatch.setenv("PDF_WORKER_MODE", "process")
    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "1")
    monkeypatch.setattr(api_mod, "_FONT_BACKFILL_GRACE_S", 0.1)
    stale = _stamps(target)

    holder = threading.Thread(target=_hold_export_worker, args=(pdf_worker_processes, 30.0))
    holder.start()
    _wait_until(
        lambda: pdf_worker_processes.pool_stats()["pools"].get("export", {}).get("in_use") == 1,
    )
    loader, _ = _in_thread(api_mod._load_layout_pages, job, None, client.app.state)
    time.sleep(0.5)                                       # 백필이 빈 워커를 기다리는 중
    # main.py lifespan 종료와 같은 순서 — 표식을 세운 뒤 풀을 닫는다
    client.app.state.shutdown.requested = True
    pdf_worker_processes.shutdown_pools()
    holder.join(30)
    loader.join(30)
    _wait_until(_no_backfill_in_flight)
    assert _stamps(target) == stale
    assert not list(job.dir.glob(".layout.*.tmp"))
