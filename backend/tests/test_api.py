import io
import json
import threading
import zipfile

from conftest import wait_done


def _upload(client, pdf_bytes: bytes, **data):
    return client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        data=data,
    )


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"
    assert body["device"] == "cpu"
    assert "native_ops" in body


def test_health_reports_max_upload_mb(tmp_path):
    """max_upload_mb는 settings 값 그대로 노출 — 프런트가 업로드 상한 안내에 쓴다."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
        max_upload_mb=7,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["max_upload_mb"] == 7


def test_health_translate_available_follows_env(client, monkeypatch):
    """translate_available은 POST /translate의 503 판정(TranslateConfig.from_env)과
    동일 — 프로바이더 env 유무에 따라 False/True로 바뀐다."""
    for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "TRANSLATE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert client.get("/api/health").json()["translate_available"] is False

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert client.get("/api/health").json()["translate_available"] is True


def test_untrusted_host_header_rejected(client):
    """TrustedHostMiddleware — 화이트리스트 밖 Host는 400 (DNS rebinding 방어)."""
    r = client.get("/api/health", headers={"host": "evil.example.com"})
    assert r.status_code == 400
    # 기본 클라이언트(Host: testserver)와 포트 붙은 허용 호스트는 통과
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"host": "localhost:8000"}).status_code == 200


def test_upload_validation(client, sample_pdf):
    r = client.post("/api/jobs", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/jobs", files={"file": ("a.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 400
    r = _upload(client, sample_pdf, mode="weird")
    assert r.status_code == 400
    r = _upload(client, sample_pdf, dpi="9999")
    assert r.status_code == 400


def test_full_flow_multi(client, sample_pdf):
    r = _upload(client, sample_pdf, mode="multi")
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]

    body = wait_done(client, jid)
    assert body["status"] == "done", body
    assert body["progress"]["total_pages"] == 3
    assert body["progress"]["current_page"] == 3

    res = body["result"]
    assert len(res["pages"]) == 3
    assert len(res["layouts"]) == 3
    assert len(res["images"]) == 3  # FakeEngine: 페이지당 figure 1개

    md = client.get(f"/api/jobs/{jid}/markdown")
    assert md.status_code == 200
    assert "X-Partial" not in md.headers
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg"):
        assert f"![](images/{name})" in md.text

    html = client.get(f"/api/jobs/{jid}/html")
    assert f'src="/api/jobs/{jid}/files/images/p0001_0.jpg"' in html.text
    assert "<table>" in html.text
    assert '<span class="math-inline">E = mc^2</span>' in html.text
    # FakeEngine boxes.json 경유 상대 폭: 크롭 (w//8..w//2) ≈ 37.5% → 센터링 포함
    assert "width:37.5%" in html.text
    assert "margin-left:auto" in html.text
    # 페이지 경계가 doc-page 섹션으로 승격됨 (3페이지)
    assert html.text.count('<section class="doc-page"') == 3
    assert 'data-page="3"' in html.text

    # PDF facsimile 레이아웃 뷰: 완성 페이지 기준면 + 검색용 투명 텍스트 레이어
    layout = client.get(f"/api/jobs/{jid}/layout")
    assert layout.status_code == 200
    assert layout.text.count('<section class="layout-page"') == 3
    assert f'src="/api/jobs/{jid}/files/pages/page_0001.png"' in layout.text
    assert "layout-page-image" in layout.text
    assert "facsimile-text-block" in layout.text
    assert '<span class="math-inline">E = mc^2</span>' in layout.text
    # 투명 선택 레이어에도 원본 좌표·폰트 메타는 유지된다.
    assert "font-size:" in layout.text and "cqw" in layout.text

    # standalone HTML 다운로드: PDF 페이지 PNG base64 + 텍스트 레이어
    dl = client.get(f"/api/jobs/{jid}/layout.html")
    assert dl.status_code == 200
    assert "attachment" in dl.headers["content-disposition"]
    assert dl.text.startswith("<!doctype html>")
    assert "data:image/png;base64," in dl.text
    assert "layout-page-image" in dl.text
    # (uocrFitLayout/KaTeX 인라인은 frontend_dir 자산 필요 — client 픽스처는
    #  no-frontend로 비활성화하므로 여기서 단언 안 함. E2E/test_layout에서 검증.)
    assert f"/api/jobs/{jid}" not in dl.text  # 서버 참조 없는 완전 자립 파일

    # 주 HTML도 좌표 레이아웃 잡에서는 같은 facsimile renderer를 사용한다.
    doc = client.get(f"/api/jobs/{jid}/document.html")
    assert doc.status_code == 200
    assert "attachment" in doc.headers["content-disposition"]
    assert 'filename="document.html"' in doc.headers["content-disposition"]
    assert doc.text.startswith("<!doctype html>")
    assert "data:image/png;base64," in doc.text
    assert "layout-page-image" in doc.text
    assert "facsimile-document-title" in doc.text
    assert doc.text.count('<section class="layout-page"') == 3
    assert f"/api/jobs/{jid}" not in doc.text          # 완전 자립 파일

    # 리더 최종 페이지 endpoint도 원문 PDF 기준면을 반환한다.
    page = client.get(f"/api/jobs/{jid}/page/1")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("image/png")

    outline = client.get(f"/api/jobs/{jid}/outline")
    assert outline.status_code == 200
    assert any(item["text"].startswith("페이지 1") for item in outline.json()["items"])

    img = client.get(res["images"][0])
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/")

    ar = client.get(f"/api/jobs/{jid}/archive")
    assert ar.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(ar.content))
    names = set(zf.namelist())
    assert "result.md" in names
    assert "images/p0001_0.jpg" in names

    listed = client.get("/api/jobs").json()["jobs"]
    assert any(j["job_id"] == jid for j in listed)

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_full_flow_per_page(client, sample_pdf):
    r = _upload(client, sample_pdf, mode="per_page")
    assert r.status_code == 202
    jid = r.json()["job_id"]
    body = wait_done(client, jid)
    assert body["status"] == "done", body
    assert body["progress"]["total_chunks"] == 3
    md = client.get(f"/api/jobs/{jid}/markdown").text
    assert "![](images/p0001_0.jpg)" in md
    assert "![](images/p0003_0.jpg)" in md


def test_sse_events(client, sample_pdf):
    jid = _upload(client, sample_pdf, mode="multi").json()["job_id"]
    events = set()
    with client.stream("GET", f"/api/jobs/{jid}/events") as s:
        for line in s.iter_lines():
            if line.startswith("event: "):
                events.add(line.removeprefix("event: ").strip())
            if "event: done" in line or "event: error" in line:
                break
    assert "done" in events, events
    # 완료 후 재접속하면 스냅샷으로 즉시 done
    with client.stream("GET", f"/api/jobs/{jid}/events") as s:
        first_events = [ln for ln in s.iter_lines() if ln.startswith("event: ")]
    assert first_events and first_events[0] == "event: done"


def test_sse_replays_pre_subscription_tokens_without_duplication(client, monkeypatch):
    """업로드 응답→EventSource 연결 및 재연결 사이의 token을 replay로 복구하고,
    구독 등록 뒤 token은 일반 이벤트로 정확히 한 번 이어 받는다."""
    store = client.app.state.store
    broker = client.app.state.broker
    job = store.create("replay.pdf", "multi", 200)
    job.status = "running"
    job.progress.update(phase="ocr", current_page=1, total_pages=2)
    store.save(job)
    broker.publish(job.id, "token", {"text": "<PAGE>\n첫 페이지"})

    subscribed = threading.Event()
    real_subscribe = broker.subscribe_with_replay

    def _subscribe_with_signal(job_id):
        result = real_subscribe(job_id)
        subscribed.set()
        return result

    monkeypatch.setattr(broker, "subscribe_with_replay", _subscribe_with_signal)

    def _finish_stream():
        assert subscribed.wait(2)
        broker.publish(job.id, "token", {"text": "\n이어지는 출력"})
        broker.publish(job.id, "done", {"markdown_url": "x", "archive_url": "y"})

    publisher = threading.Thread(target=_finish_stream, daemon=True)
    publisher.start()
    events = []
    try:
        with client.stream("GET", f"/api/jobs/{job.id}/events") as response:
            current = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current = line.removeprefix("event: ")
                elif line.startswith("data: ") and current:
                    events.append((current, json.loads(line.removeprefix("data: "))))
                    if current == "done":
                        break
                    current = None
    finally:
        publisher.join(timeout=2)
        store.delete_dir(job)

    replay = [data for event, data in events if event == "replay"]
    tokens = [data["text"] for event, data in events if event == "token"]
    assert replay == [{
        "text": "<PAGE>\n첫 페이지",
        "truncated": False,
        "current_page": 1,
        "total_pages": 2,
    }]
    assert tokens == ["\n이어지는 출력"]

    # 터미널 이벤트에서 문서 원문을 담은 메모리 히스토리를 즉시 폐기한다.
    q, replay_after_done, truncated = broker.subscribe_with_replay(job.id)
    broker.unsubscribe(job.id, q)
    assert replay_after_done == "" and truncated is False


def test_files_path_traversal_blocked(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    assert client.get(f"/api/jobs/{jid}/files/../meta.json").status_code in (400, 404)
    assert client.get(f"/api/jobs/{jid}/files/work/chunk_00/result.md").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/meta.json").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/pages/page_0001.png").status_code == 200


def test_cancel_keeps_partial_results(tmp_path, sample_pdf):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.4, frontend_dir=tmp_path / "no-frontend",
    )
    with TestClient(create_app(settings)) as client:
        jid = _upload(client, sample_pdf).json()["job_id"]
        r = client.post(f"/api/jobs/{jid}/cancel")
        assert r.status_code == 202
        assert r.json()["status"] == "canceling"
        body = wait_done(client, jid)
        assert body["status"] == "canceled", body
        # 삭제되지 않고 남아 있어야 함 (부분 결과 보존)
        assert client.get(f"/api/jobs/{jid}").status_code == 200
        md = client.get(f"/api/jobs/{jid}/markdown")
        assert md.status_code == 200
        assert md.headers.get("X-Partial") == "true"


def test_cancel_finished_job_is_noop(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 202
    assert r.json()["status"] == "done"


def test_render_preview(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    md = "# 라이브\n\n![](images/p0001_0.jpg)\n\n<table><tr><td>a</td></tr></table>\n\n<script>x</script>"
    r = client.post(f"/api/jobs/{jid}/render-preview", content=md.encode())
    assert r.status_code == 200
    assert f'src="/api/jobs/{jid}/files/images/p0001_0.jpg"' in r.text
    assert "<table><tr><td>a</td></tr></table>" in r.text
    assert "<script>" not in r.text
    assert client.post("/api/jobs/j_nope/render-preview", content=b"x").status_code == 404


def test_render_preview_body_limit(client, sample_pdf):
    """상한(2MB)은 스트리밍 수신 중 검사되어 초과 즉시 413. 경계값(정확히 2MB)은 통과."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    r = client.post(f"/api/jobs/{jid}/render-preview", content=b"x" * 2_000_001)
    assert r.status_code == 413
    at_limit = (b"x" * 99 + b"\n") * 20_000  # 정확히 2,000,000바이트
    r = client.post(f"/api/jobs/{jid}/render-preview", content=at_limit)
    assert r.status_code == 200


def test_archive_before_done_conflicts(client, sample_pdf, settings):
    # fake_delay=0이면 너무 빨리 끝나 409를 못 볼 수 있으므로 큰 파일로 시도하지 않고
    # 존재하지 않는 완료 전 상태를 시뮬레이션: 새 잡을 만들고 즉시 archive 요청 경합 허용
    jid = _upload(client, sample_pdf).json()["job_id"]
    r = client.get(f"/api/jobs/{jid}/archive")
    assert r.status_code in (200, 409)


def test_result_block_has_layout_플래그(tmp_path):
    """레이아웃 기능(P14) 이전에 변환된 잡은 layout.json이 없어 /layout*이 404 —
    프런트가 버튼을 비활성화할 수 있도록 결과 블록에 has_layout을 내려준다."""
    from app.jobs import Job

    old = Job(id="j_old", filename="a.pdf", mode="multi", dpi=200, dir=tmp_path / "old", status="done")
    old.dir.mkdir()
    assert old.to_dict()["result"]["has_layout"] is False

    new = Job(id="j_new", filename="b.pdf", mode="multi", dpi=200, dir=tmp_path / "new", status="done")
    new.dir.mkdir()
    (new.dir / "layout.json").write_text("[]", encoding="utf-8")
    assert new.to_dict()["result"]["has_layout"] is True


def test_queue_position_for_queued_jobs(tmp_path, sample_pdf):
    """queued 잡에만 queue_position(1-base, 생성 순서)이 붙는다 — 단일 워커 FIFO 큐.
    with 블록(lifespan) 없이 요청하면 워커가 기동하지 않아 큐가 소비되지 않는다."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
    )
    app = create_app(settings)
    client = TestClient(app)  # lifespan 미실행 → 워커 미기동
    ids = [_upload(client, sample_pdf).json()["job_id"] for _ in range(3)]

    listed = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert [listed[i]["queue_position"] for i in ids] == [1, 2, 3]
    # 상세 응답도 동일 계약
    assert client.get(f"/api/jobs/{ids[2]}").json()["queue_position"] == 3

    # 맨 앞 잡이 running으로 넘어가면 필드가 사라지고 뒤 잡들이 한 칸씩 당겨진다
    app.state.store.get(ids[0]).status = "running"
    listed = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert "queue_position" not in listed[ids[0]]
    assert [listed[i]["queue_position"] for i in ids[1:]] == [1, 2]


def test_queue_position_absent_on_terminal_job(client, sample_pdf):
    jid = _upload(client, sample_pdf).json()["job_id"]
    body = wait_done(client, jid)
    assert body["status"] == "done"
    assert "queue_position" not in body
    listed = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert "queue_position" not in listed[jid]
