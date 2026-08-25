import io
import json
import threading
import zipfile
from types import SimpleNamespace

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


def test_health_reports_worker_alive(client):
    """워커가 죽으면 잡이 영원히 queued로 남는다 — 운영 가시성용 필드."""
    assert client.get("/api/health").json()["worker_alive"] is True

    class _Dead:
        def is_alive(self):
            return False

    real = client.app.state.worker
    client.app.state.worker = _Dead()
    try:
        assert client.get("/api/health").json()["worker_alive"] is False
    finally:
        client.app.state.worker = real


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
    assert res["viewer_manifest_url"] == f"/api/jobs/{jid}/viewer-manifest"

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

    # 구 레이아웃 HTML URL은 중복 파일을 만들지 않고 정식 document.html로 통합한다.
    legacy = client.get(f"/api/jobs/{jid}/layout.html", follow_redirects=False)
    assert legacy.status_code == 307
    assert legacy.headers["location"] == f"/api/jobs/{jid}/document.html"

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

    alignment = client.get(f"/api/jobs/{jid}/alignment?page=1")
    assert alignment.status_code == 200
    aligned = alignment.json()
    assert aligned["page"] == 1
    assert aligned["lang"] == "orig"
    assert aligned["bbox_space"] == 1000
    assert aligned["blocks"]
    assert all(len(block["bbox"]) == 4 for block in aligned["blocks"])
    assert all(block["target"] == block["source"] for block in aligned["blocks"])

    outline = client.get(f"/api/jobs/{jid}/outline")
    assert outline.status_code == 200
    assert any(item["text"].startswith("페이지 1") for item in outline.json()["items"])

    manifest = client.get(f"/api/jobs/{jid}/viewer-manifest")
    assert manifest.status_code == 200
    viewer = manifest.json()
    assert viewer["schema_version"] == 1
    assert viewer["document"]["page_count"] == 3
    assert viewer["document"]["ready_page_count"] == 3
    assert viewer["capabilities"]["source_page_image"] is True
    assert viewer["capabilities"]["alignment"] is True
    assert viewer["links"]["source_page_template"].startswith(
        f"/api/jobs/{jid}/page/{{page}}"
    )
    assert manifest.headers["cache-control"] == "private, no-cache"
    assert client.get(
        f"/api/jobs/{jid}/viewer-manifest",
        headers={"if-none-match": manifest.headers["etag"]},
    ).status_code == 304

    batch = client.get(
        f"/api/jobs/{jid}/viewer/pages?start=1&limit=2&include=alignment"
    )
    assert batch.status_code == 200
    viewer_pages = batch.json()
    assert viewer_pages["total"] == 3
    assert viewer_pages["next_start"] == 3
    assert [item["page"] for item in viewer_pages["items"]] == [1, 2]
    assert all(item["alignment"]["blocks"] for item in viewer_pages["items"])
    assert client.get(
        f"/api/jobs/{jid}/viewer/pages?start=0&limit=2"
    ).status_code == 422
    assert client.get(
        f"/api/jobs/{jid}/viewer/pages?start=1&limit=17"
    ).status_code == 422

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


_TRAVERSAL_PATHS = (
    "pages/../source.pdf",
    "pages/./../source.pdf",
    "pages/%2e%2e/source.pdf",
    "pages/../../../etc/passwd",
    "images/../../etc/passwd",
    "rendered/../meta.json",
    "layout/../result.md",
    "pages/subdir/../../source.pdf",
    "pages/../translations/ko/units.json",
)


def test_files_allowlist_not_bypassable_by_parent_refs(client, sample_pdf):
    """허용 디렉터리로 시작하기만 하면 상위참조로 잡 디렉터리 안 임의 파일(원본
    업로드 PDF·meta.json·번역 산출물)을 받아갈 수 있었다 — 정규화 후 검사한다.

    httpx는 URL의 `..`를 전송 전에 정규화하므로(실제 브라우저·curl --path-as-is는
    그대로 보낸다) 라우트 함수를 직접 호출해 계약을 고정하고, HTTP 경로로도
    차단되는지 함께 확인한다."""
    import pytest
    from fastapi import HTTPException

    import app.api as api_mod

    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    job = client.app.state.store.get(jid)
    assert (job.dir / "source.pdf").is_file()   # 실제로 존재해야 회귀가 의미 있다

    request = SimpleNamespace(app=client.app)   # job_file이 쓰는 표면은 app.state뿐
    for path in _TRAVERSAL_PATHS:
        with pytest.raises(HTTPException) as excinfo:
            api_mod.job_file(request, jid, path)
        assert excinfo.value.status_code == 404, path
        assert client.get(f"/api/jobs/{jid}/files/{path}").status_code in (400, 404), path

    # 정상 경로는 그대로 200
    assert api_mod.job_file(request, jid, "pages/page_0001.png").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/pages/page_0001.png").status_code == 200


def test_files_symlink_escape_blocked(client, sample_pdf):
    """허용 디렉터리 안의 심볼릭 링크로도 잡 디렉터리 안팎을 빠져나갈 수 없다."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    job = client.app.state.store.get(jid)
    inside = job.dir / "pages" / "leak.pdf"
    inside.symlink_to(job.dir / "source.pdf")
    outside = job.dir / "pages" / "outside.json"
    outside.symlink_to(job.dir.parent)

    assert client.get(f"/api/jobs/{jid}/files/pages/leak.pdf").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/pages/outside.json").status_code == 404
    assert client.get(f"/api/jobs/{jid}/files/pages/page_0001.png").status_code == 200


def test_job_list_omits_file_urls(client, sample_pdf):
    """목록 폴링은 잡마다 pages/·images/ 전수 스캔이 필요 없다 — 키는 유지하되
    빈 배열. 단건 조회는 기존 계약대로 전체 URL을 준다."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)

    listed = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}[jid]
    assert listed["result"]["images"] == []
    assert listed["result"]["layouts"] == []
    assert listed["result"]["pages"] == []
    assert listed["result"]["has_layout"] is True

    single = client.get(f"/api/jobs/{jid}").json()
    assert len(single["result"]["pages"]) == 3
    assert len(single["result"]["images"]) == 3
    assert len(single["result"]["layouts"]) == 3


def test_upload_failure_leaves_no_ghost_job(client, sample_pdf, monkeypatch):
    """업로드 중 HTTPException이 아닌 예외(디스크 오류 등)에서도 잡 디렉터리와
    목록 항목이 남으면 안 된다 — 남으면 영원히 queued인 유령 잡이 된다."""
    import pytest

    def _boom(*args, **kwargs):
        raise RuntimeError("디스크 오류")

    monkeypatch.setattr("app.api.probe_pdf", _boom)
    with pytest.raises(RuntimeError):
        _upload(client, sample_pdf)

    assert client.get("/api/jobs").json()["jobs"] == []


def test_job_render_lock_is_per_job(client):
    """전역 락이면 200페이지 잡 하나의 첫 렌더가 다른 잡의 /layout·/page까지 막는다."""
    import app.api as api_mod

    lock_a = api_mod._job_render_lock("job-a")
    assert api_mod._job_render_lock("job-a") is lock_a
    assert api_mod._job_render_lock("job-b") is not lock_a
    api_mod._forget_job_caches("job-a")
    api_mod._forget_job_caches("job-b")


def test_page_image_does_not_reparse_layout_each_request(client, sample_pdf, monkeypatch):
    """페이지 이미지 요청마다 layout.json 전체를 재파싱하지 않는다(크기·mtime 캐시)."""
    import app.api as api_mod

    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    assert client.get(f"/api/jobs/{jid}/page/1").status_code == 200  # 캐시 워밍(폰트 백필 포함)

    calls = []
    real = api_mod._load_layout_pages
    monkeypatch.setattr(
        "app.api._load_layout_pages",
        lambda job, lang=None: (calls.append(lang), real(job, lang))[1],
    )
    for page in (1, 2, 3, 1):
        assert client.get(f"/api/jobs/{jid}/page/{page}").status_code == 200
    assert calls == []


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


# ── 업로드 본문 상한: 멀티파트 파싱 전에 차단 (UploadBodyLimitMiddleware) ──
def _limited_app(tmp_path, max_upload_mb: int = 1):
    """MAX_UPLOAD_MB가 작은 앱 — 상한 경계를 실제 업로드로 확인하기 위해."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, frontend_dir=tmp_path / "no-frontend",
        max_upload_mb=max_upload_mb,
    )
    return TestClient(create_app(settings)), settings


def test_upload_over_limit_rejected_before_spooling(tmp_path, sample_pdf, monkeypatch):
    """상한 초과 Content-Length는 폼 파싱 전에 413 — 스풀 임시 파일도, 잡 디렉터리도
    만들어지지 않는다(무인증 서비스의 디스크 소진 벡터)."""
    import starlette.formparsers as fp

    spooled = []
    real_spool = fp.SpooledTemporaryFile
    monkeypatch.setattr(
        fp, "SpooledTemporaryFile",
        lambda *a, **kw: (spooled.append(1), real_spool(*a, **kw))[1],
    )

    client, settings = _limited_app(tmp_path)
    with client:
        r = _upload(client, sample_pdf + b"\n" * (2 * 1024 * 1024))
        assert r.status_code == 413
        assert "1MB" in r.json()["detail"]
        assert spooled == []                                  # 본문이 디스크에 닿지 않았다
        assert client.get("/api/jobs").json()["jobs"] == []   # 유령 잡 없음
        assert list(settings.jobs_dir.iterdir()) == []


def test_upload_exactly_at_limit_is_accepted(tmp_path, sample_pdf):
    """멀티파트 봉투 여유분이 없으면 정확히 MAX_UPLOAD_MB인 파일이 거절된다 — 회귀 방지."""
    client, settings = _limited_app(tmp_path)
    at_limit = sample_pdf + b"\n" * (settings.max_upload_bytes - len(sample_pdf))
    assert len(at_limit) == settings.max_upload_bytes
    with client:
        r = _upload(client, at_limit)
        assert r.status_code == 202, r.text
        wait_done(client, r.json()["job_id"])


def test_upload_over_limit_without_content_length(tmp_path, sample_pdf):
    """길이를 알 수 없는(chunked) 본문도 누적 바이트로 판정해 413."""
    boundary = "----limit-test"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    payload = sample_pdf + b"\n" * (2 * 1024 * 1024)

    def gen():
        yield head
        yield payload
        yield f"\r\n--{boundary}--\r\n".encode()

    client, settings = _limited_app(tmp_path)
    with client:
        r = client.post(
            "/api/jobs",
            content=gen(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        assert r.status_code == 413
        assert "content-length" not in {k.lower() for k in r.request.headers}
        assert list(settings.jobs_dir.iterdir()) == []


def test_upload_body_limit_middleware_cuts_streaming_body():
    """chunked 경로는 앱이 본문을 다 받기 전에 끊긴다 — 누적 바이트가 상한을 넘는
    즉시 413을 보내고 뒤따르는 청크는 읽지 않는다."""
    import anyio

    from app.main import UploadBodyLimitMiddleware

    chunk = b"x" * 4096
    total_chunks = 64
    reads = {"n": 0}
    seen: list[int] = []
    sent: list[dict] = []

    async def receive():
        reads["n"] += 1
        if reads["n"] <= total_chunks:
            return {"type": "http.request", "body": chunk, "more_body": True}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    async def app(scope, receive_, send_):
        while True:
            msg = await receive_()
            if msg["type"] == "http.disconnect":
                raise RuntimeError("client disconnected")  # 폼 파서와 같은 반응
            seen.append(len(msg["body"]))
            if not msg.get("more_body"):
                break
        await send_({"type": "http.response.start", "status": 202, "headers": []})
        await send_({"type": "http.response.body", "body": b"{}"})

    # max_bytes=0 → 상한은 멀티파트 여유분(64KiB)뿐
    mw = UploadBodyLimitMiddleware(app, max_bytes=0, max_mb=0)
    scope = {"type": "http", "method": "POST", "path": "/api/jobs", "headers": []}
    anyio.run(mw, scope, receive, send)

    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [413]
    assert sum(seen) <= 64 * 1024 + len(chunk)   # 상한 언저리에서 멈췄다
    assert reads["n"] < total_chunks             # 나머지 청크는 읽지 않았다


def test_upload_limit_does_not_affect_other_routes(tmp_path, sample_pdf):
    """경로별 상한이라 /render-preview는 업로드 상한이 아니라 자기 2MB 상한을 받는다."""
    client, settings = _limited_app(tmp_path)
    with client:
        jid = _upload(client, sample_pdf).json()["job_id"]
        # MAX_UPLOAD_MB(1MB)보다 크지만 프리뷰 상한(2MB) 안이면 통과해야 한다
        big = b"# preview\n" + b"x" * 1_500_000
        assert client.post(f"/api/jobs/{jid}/render-preview", content=big).status_code == 200
        # SSE·다운로드 등 GET 경로도 그대로
        assert client.get("/api/health").status_code == 200


def test_qa_body_limit_cuts_before_route(tmp_path):
    """POST /jobs/{id}/qa는 잡 존재 확인조차 하기 전에 무제한 본문을 메모리에 적재했다.
    (실측 uvicorn: 80MB 본문 1건에 RSS +414MB, 동시 4건 1.46GB.)
    이제 기본 상한(64KiB)이 **라우트 진입 전에** 413으로 끊는다 — 잡이 없는 ID인데도
    404가 아니라 413이 나오는 것이 '라우트가 돌지 않았다'는 증거다."""
    client, _ = _limited_app(tmp_path)
    with client:
        big = json.dumps({"question": "q" * 200_000}).encode()
        r = client.post(
            "/api/jobs/j_nope/qa", content=big,
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413, r.text
        assert "64KiB" in r.json()["detail"]
        assert len(r.content) < 1024          # 422처럼 원문을 되돌려주지 않는다
        # 상한 안의 본문은 그대로 라우트에 도달한다 (없는 잡 → 404)
        small = json.dumps({"question": "안녕"}).encode()
        r = client.post("/api/jobs/j_nope/qa", content=small,
                        headers={"content-type": "application/json"})
        assert r.status_code == 404, r.text


def test_body_limit_default_covers_unlisted_post_routes(tmp_path):
    """표에 없는 POST 경로도 기본 상한(64KiB)으로 자동 보호된다 — 새 라우트가
    추가돼도 무제한 적재가 생기지 않게."""
    client, _ = _limited_app(tmp_path)
    with client:
        for path in ("/api/jobs/j_nope/cancel", "/api/jobs/j_nope/translate",
                     "/api/some-future-route"):
            r = client.post(path, content=b"x" * (64 * 1024 + 1))
            assert r.status_code == 413, (path, r.status_code)


def test_render_preview_limit_enforced_in_middleware(tmp_path, sample_pdf):
    """프리뷰 상한은 미들웨어에서도 같은 값(2MB)으로 걸린다 — 경계값은 여전히 통과."""
    client, _ = _limited_app(tmp_path)
    with client:
        jid = _upload(client, sample_pdf).json()["job_id"]
        r = client.post(f"/api/jobs/{jid}/render-preview", content=b"x" * 2_000_001)
        assert r.status_code == 413
        assert "미리보기" in r.json()["detail"]
        at_limit = (b"x" * 99 + b"\n") * 20_000  # 정확히 2,000,000바이트
        assert client.post(
            f"/api/jobs/{jid}/render-preview", content=at_limit
        ).status_code == 200


def test_body_limit_ignores_get_routes(tmp_path, sample_pdf):
    """본문 없는 메서드(GET/SSE/다운로드)는 검사 대상이 아니다."""
    client, _ = _limited_app(tmp_path)
    with client:
        jid = _upload(client, sample_pdf).json()["job_id"]
        wait_done(client, jid)
        assert client.get(f"/api/jobs/{jid}/markdown").status_code == 200
        assert client.get("/api/jobs").status_code == 200


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
    """queued 잡에만 queue_position(1-base, 워커 큐 제출 순서)이 붙는다 — 단일 워커
    FIFO 큐. 업로드 본문 수신이 끝난 뒤 submit되므로 생성 순서와 어긋날 수 있고
    (여기처럼 순차 업로드면 두 순서가 같다), with 블록(lifespan) 없이 요청하면
    워커가 기동하지 않아 큐가 소비되지 않는다."""
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
