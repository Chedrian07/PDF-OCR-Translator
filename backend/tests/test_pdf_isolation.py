"""PyMuPDF 프로세스 격리 — 적대적 페이지·내보내기 빌드가 서버를 멈추지 못한다 (감사 A11).

업로드 게이트(test_pdf_complexity_gate)가 1차 방어선이고, 여기서는 게이트를 끄거나 빠져나간
경우의 백스톱을 실제 spawn 워커로 확인한다:
- security-2: 중첩 Form XObject 페이지가 단일 OCR 워커를 영구 점거하고 GIL로 health·취소를
  멈췄다 → 페이지 시간 상한 뒤 흰 페이지 + 경고, health는 내내 응답, 취소는 즉시.
- gap1-metal-real-e2e-2·concurrency-2/3·pdf-export-9: 같은 프로세스의 빌드가 GIL을 쥐어
  OCR 디코드가 31.7→1.0 tok/s로 굶었다 → 빌드가 워커에서 도는 동안 부모의 CPU 루프는
  제 속도를 낸다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_pdf_bytes, wait_done
from test_pdf_complexity_gate import _form_chain, _page_with_content


def _bomb_page(doc, levels: int = 12, fanout: int = 10) -> None:
    top = _form_chain(doc, levels, fanout)
    _page_with_content(doc, b"q /Top Do Q", f"<< /XObject << /Top {top} 0 R >> >>")


def _mixed_pdf() -> bytes:
    """1·3쪽은 정상, 2쪽은 감사 재현과 같은 10^12 중첩 그리기 페이지."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=595, height=842).insert_text((72, 80), "Normal page one", fontsize=18)
    _bomb_page(doc)
    doc.new_page(width=595, height=842).insert_text((72, 80), "Normal page three", fontsize=18)
    data = doc.tobytes()
    doc.close()
    return data


def _bomb_only_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    _bomb_page(doc)
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client, data: bytes) -> str:
    r = client.post("/api/jobs", files={"file": ("x.pdf", data, "application/pdf")})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


@pytest.fixture
def backstop_only(monkeypatch):
    """업로드 게이트를 끄고 페이지 시간 상한(백스톱)만 남긴다."""
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "0")
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "2")


def test_hostile_page_becomes_a_blank_page_while_health_stays_responsive(
    pdf_worker_processes, backstop_only, client,
):
    jid = _upload(client, _mixed_pdf())
    started = time.monotonic()
    health_latency: list[float] = []
    seen_running_job = False
    body: dict = {}
    while time.monotonic() - started < 60:
        t = time.monotonic()
        health = client.get("/api/health")
        health_latency.append(time.monotonic() - t)
        assert health.status_code == 200
        if health.json()["worker_job_id"] == jid:
            seen_running_job = True
            assert health.json()["worker_last_progress_at"]
        body = client.get(f"/api/jobs/{jid}").json()
        if body["status"] in ("done", "error", "canceled"):
            break
        time.sleep(0.05)
    elapsed = time.monotonic() - started
    assert body["status"] == "done", body
    assert elapsed < 30  # 10^12회 그리기를 끝까지 기다리지 않았다(상한 2초)
    # 렌더 중에도 서버는 GIL을 쥐지 않는다 — health가 상한 시간 내내 즉시 응답했다
    assert max(health_latency) < 1.0, max(health_latency)
    assert seen_running_job

    warnings = body["warnings"]
    assert any("렌더 시간 상한" in w and "(2)" in w for w in warnings), warnings
    # 같은 페이지의 충실도 분석은 상한을 다시 기다리지 않고 건너뛰었다는 참고를 남긴다
    assert any("충실도 분석이 시간 상한" in n for n in body["notices"]), body["notices"]

    job_dir = client.app.state.store.get(jid).dir
    with Image.open(job_dir / "pages" / "page_0002.png") as blank:
        assert blank.convert("RGB").getextrema() == ((255, 255), (255, 255), (255, 255))
    with Image.open(job_dir / "pages" / "page_0001.png") as normal:
        assert normal.convert("L").getextrema()[0] < 128  # 정상 페이지는 그대로 렌더

    after = client.get("/api/health").json()
    assert after["worker_job_id"] is None
    pools = after["pdf_workers"]
    assert pools["mode"] == "process"
    assert pools["pools"]["ocr"]["timeouts"] >= 1


@pytest.mark.parametrize("action", ["cancel", "delete"])
def test_cancel_and_delete_do_not_wait_for_a_hostile_page(
    pdf_worker_processes, client, monkeypatch, action,
):
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "0")
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "120")  # 상한이 아니라 취소로 끝나야 한다
    jid = _upload(client, _bomb_only_pdf())
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{jid}").json()
        if body["status"] == "running" and body["progress"].get("phase") == "render":
            break
        time.sleep(0.05)
    time.sleep(0.5)  # 워커가 그 페이지를 렌더하는 중이다
    started = time.monotonic()
    if action == "cancel":
        assert client.post(f"/api/jobs/{jid}/cancel").status_code == 202
        body = wait_done(client, jid, timeout=10)
        assert body["status"] == "canceled"
    else:
        assert client.delete(f"/api/jobs/{jid}").status_code in (200, 202, 204)
        while client.get(f"/api/jobs/{jid}").status_code != 404:
            assert time.monotonic() - started < 10, "삭제가 렌더를 기다렸다"
            time.sleep(0.05)
    assert time.monotonic() - started < 5
    assert pdf_worker_processes.pool_stats()["pools"]["ocr"]["canceled"] == 1


def _heavy_export_job(root: Path, pages: int = 6, paths: int = 10_000) -> Path:
    """벡터가 많은 원본 + 번역 레이아웃 — 빌드가 페이지마다 벡터 분석·리댁션을 한다."""
    import pymupdf

    job_dir = root / "heavy-job"
    job_dir.mkdir()
    doc = pymupdf.open()
    blocks_by_page = []
    for number in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            pymupdf.Rect(60, 85, 535, 250), f"Original English paragraph {number}. " * 6,
            fontsize=11,
        )
        body = b"".join(
            b"%d %d m %d %d l S\n" % (60 + i % 470, 300 + (i // 470) % 500,
                                       63 + i % 470, 302 + (i // 470) % 500)
            for i in range(paths)
        )
        xref = doc.get_new_xref()
        doc.update_object(xref, "<< >>")
        doc.update_stream(xref, body)
        page_contents = page.get_contents()
        doc.xref_set_key(
            page.xref, "Contents",
            "[" + " ".join(f"{x} 0 R" for x in [*page_contents, xref]) + "]",
        )
        blocks_by_page.append([{
            "type": "text", "bbox": [100, 100, 900, 300],
            "content": f"Original English paragraph {number}.", "fs": 1.8,
        }])
    doc.save(job_dir / "source.pdf", deflate=True)
    doc.close()
    original = [
        {"page": n + 1, "width": 1000, "height": 1414, "blocks": blocks}
        for n, blocks in enumerate(blocks_by_page)
    ]
    translated = json.loads(json.dumps(original))
    for page in translated:
        page["blocks"][0]["content"] = "번역된 한국어 문단입니다. " * 4
    (job_dir / "layout.json").write_text(json.dumps(original), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(translated, ensure_ascii=False), encoding="utf-8",
    )
    return job_dir


# GIL을 놓는 스텝 하나의 입력 — hashlib은 2KiB를 넘는 입력을 해시하는 동안 GIL을 놓는다.
_GIL_STEP_BLOCK = b"x" * 65536
# 빌드 중 스텝 속도의 하한(빌드 전 대비). 실측(M4 Max): 워커 프로세스 빌드 0.96~0.97배, 같은
# 프로세스(inline) 빌드 0.003배. 다른 프로세스와 CPU 하나를 나눠 쓰는 러너(2 vCPU·SMT·x86_64
# 에뮬레이션에서 torch를 올린 부모와 자식이 한 vCPU에 묶인 경우 — P4 Linux CI 재현)는 ~0.5배다.
_MIN_STEP_RATIO = 0.3


def _gil_step_rate(seconds: float) -> float:
    """GIL을 놓았다 다시 잡는 스텝의 초당 횟수 — OCR 디코드(토치 커널마다 GIL을 놓고 다시
    잡는다)의 대역. 같은 프로세스에서 MuPDF가 GIL을 쥐면 스텝마다 GIL을 다시 얻으려 줄을 서서
    수백 분의 1로 떨어진다. 예전의 순수 파이썬 루프는 GIL 전환 주기마다 반을 얻어 같은 프로세스
    빌드에서도 ~0.55배라, CPU를 나눠 쓰는 러너(~0.5배)와 구분하지 못했다."""
    count = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        hashlib.sha256(_GIL_STEP_BLOCK).digest()
        count += 1
    return count / seconds


def test_parent_keeps_its_cpu_while_an_export_build_runs(pdf_worker_processes, tmp_path):
    """빌드가 export 워커 프로세스에서 도는 동안 서버 프로세스의 디코드 대역 스텝이 굶지 않는다.
    같은 프로세스의 스레드였을 때는 MuPDF가 GIL을 쥐어 디코드가 31.7→1.0 tok/s로 굶었다(감사
    gap1-metal-real-e2e-2)."""
    from app.pipeline.pdf_export import build_translated_pdf

    job_dir = _heavy_export_job(tmp_path)
    build_translated_pdf(job_dir, "ko")  # 워커 기동·폰트 탐색 비용을 측정 밖으로
    baseline = _gil_step_rate(0.6)

    stop = threading.Event()
    builds: list[int] = []
    errors: list[BaseException] = []

    def _keep_building() -> None:
        try:
            while not stop.is_set():
                result = build_translated_pdf(job_dir, "ko")
                builds.append(result.replaced)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    builder = threading.Thread(target=_keep_building)
    builder.start()
    time.sleep(0.2)
    during = _gil_step_rate(1.5)
    stop.set()
    builder.join(120)
    assert not errors, errors
    assert builds and builds[0] == 6  # 측정 구간 내내 실제 빌드가 돌았다(페이지 6개 교체)
    assert during >= baseline * _MIN_STEP_RATIO, (during, baseline)


def test_export_build_time_limit_is_a_409_and_leaves_no_temp_files(
    pdf_worker_processes, client, monkeypatch,
):
    jid = _upload(client, make_pdf_bytes(pages=2))
    assert wait_done(client, jid)["status"] == "done"
    job_dir = client.app.state.store.get(jid).dir
    (job_dir / "result.ko.md").write_text(
        (job_dir / "result.md").read_text(encoding="utf-8"), encoding="utf-8",
    )
    shutil.copyfile(job_dir / "layout.json", job_dir / "layout.ko.json")

    monkeypatch.setenv("PDF_EXPORT_BUILD_TIMEOUT_S", "0.001")
    r = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert r.status_code == 409, r.text
    assert "시간 상한" in r.json()["detail"] and "PDF_EXPORT_BUILD_TIMEOUT_S" in r.json()["detail"]
    assert not list(job_dir.glob(".export.ko.*.tmp"))

    monkeypatch.setenv("PDF_EXPORT_BUILD_TIMEOUT_S", "120")
    ok = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert ok.status_code == 200 and ok.content.startswith(b"%PDF-")
    stats = pdf_worker_processes.pool_stats()["pools"]
    assert stats["export"]["timeouts"] == 1 and stats["ocr"]["tasks"] > 0


def test_textlayer_engine_skips_a_page_that_exceeded_the_limit(
    pdf_worker_processes, backstop_only, tmp_path, monkeypatch,
):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.engine import textlayer as textlayer_mod
    from app.main import create_app

    monkeypatch.setattr(textlayer_mod, "find_tesseract", lambda: None)
    settings = Settings(
        engine="textlayer", device="cpu", data_dir=tmp_path / "data", preload_model=False,
        frontend_dir=tmp_path / "no-frontend", native_text_threshold=1,
    )
    with TestClient(create_app(settings)) as client:
        jid = _upload(client, _mixed_pdf())
        body = wait_done(client, jid, timeout=60)
    assert body["status"] == "done", body
    md_warnings = " ".join(body["warnings"])
    assert "렌더 시간 상한" in md_warnings
    assert "텍스트 레이어 추출을 건너뛰었습니다" in md_warnings
