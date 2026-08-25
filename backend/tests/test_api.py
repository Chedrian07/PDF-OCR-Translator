import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
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


def test_worker_death_is_logged_not_only_exposed(client, caplog):
    """worker_alive 필드는 아무도 폴링하지 않으면 무성이다 — 사망은 서버 로그에도
    남아야 한다. 다만 health 폴링마다 찍으면 로그가 잠기므로 전이에서만 남긴다."""

    class _Dead:
        def is_alive(self):
            return False

    real = client.app.state.worker
    client.app.state.worker = _Dead()
    try:
        with caplog.at_level(logging.ERROR, logger="app.api"):
            for _ in range(3):
                assert client.get("/api/health").json()["worker_alive"] is False
        dead = [r for r in caplog.records if "워커" in r.getMessage()]
        assert len(dead) == 1, [r.getMessage() for r in dead]
    finally:
        client.app.state.worker = real
        client.app.state.worker_alive_last = None


def test_rate_limit_key_trusts_proxy_only_when_opted_in(monkeypatch):
    """리버스 프록시 뒤에서는 client.host가 프록시 IP라 전원이 한 버킷으로 붕괴한다.
    그렇다고 X-Forwarded-For를 무조건 믿으면 헤더 위조로 우회된다 — 신뢰 홉 수를
    명시한 배포에서만 헤더를 읽고, 기본값(미설정)은 현행대로 헤더를 무시한다."""
    import app.api as api_mod

    request = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.9"),
        # client(203.0.113.7) → 프록시A → 프록시B → 앱: 각 홉이 한 항목씩 덧붙인다
        headers={"x-forwarded-for": "203.0.113.7, 10.0.0.9"},
    )
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    assert api_mod._client_key(request) == "10.0.0.9"        # 기본: 헤더 무시

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    assert api_mod._client_key(request) == "10.0.0.9"        # 신뢰 홉 1개 = 프록시A 주소
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert api_mod._client_key(request) == "203.0.113.7"     # 실제 클라이언트
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "3")            # 체인이 홉 수보다 짧다
    assert api_mod._client_key(request) == "10.0.0.9"        # → 헤더를 믿지 않는다
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "abc")          # 오타는 안전한 쪽으로
    assert api_mod._client_key(request) == "10.0.0.9"

    # 헤더가 없거나 비어 있어도 직접 주소로 폴백한다
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    bare = SimpleNamespace(client=SimpleNamespace(host="10.0.0.9"), headers={})
    assert api_mod._client_key(bare) == "10.0.0.9"
    # 키가 무한히 길어지지 않는다 (레이트리밋 dict 메모리 방어)
    long_header = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.9"),
        headers={"x-forwarded-for": "9" * 500},
    )
    assert len(api_mod._client_key(long_header)) == api_mod._XFF_KEY_MAX


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
    # 진입 컨텍스트도 같은 락을 재사용한다 (중복 빌드 방지의 실제 경로)
    with api_mod._job_render_guard("job-a"):
        assert api_mod._FACSIMILE_LOCKS["job-a"] is lock_a
    api_mod._forget_job_caches("job-a")
    api_mod._forget_job_caches("job-b")


def test_job_render_lock_cache_is_bounded_but_never_drops_in_use(monkeypatch):
    """락 dict은 잡마다 커지고 TTL GC로 사라진 잡은 DELETE 훅도 타지 않는다 —
    상한을 두되, **사용 중** 항목은 절대 버리지 않는다(버리면 같은 잡에 락이 둘
    생겨 중복 빌드가 되살아난다)."""
    import app.api as api_mod
    from app.pipeline import derived  # 상한 상수의 소유자 (api는 같은 dict을 재노출)

    monkeypatch.setattr(derived, "_FACSIMILE_LOCKS_MAX", 8)
    api_mod._FACSIMILE_LOCKS.clear()
    api_mod._JOB_LOCK_REFS.clear()
    entered = threading.Event()
    release = threading.Event()

    def _hold():
        with api_mod._job_render_guard("job-held"):
            entered.set()
            release.wait(5)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    try:
        assert entered.wait(5)
        held_lock = api_mod._FACSIMILE_LOCKS["job-held"]
        for index in range(50):
            with api_mod._job_render_guard(f"job-{index}"):
                pass
        assert len(api_mod._FACSIMILE_LOCKS) <= 8
        assert api_mod._FACSIMILE_LOCKS.get("job-held") is held_lock
    finally:
        release.set()
        holder.join(5)
        api_mod._FACSIMILE_LOCKS.clear()
        api_mod._JOB_LOCK_REFS.clear()


def test_facsimile_memo_is_bounded_and_expires(monkeypatch):
    """검증 메모는 상한이 있고(잡이 늘어도 무한 증가하지 않는다) TTL이 지나면
    디스크 재검증으로 되돌아간다 — marker는 그대로인데 PNG만 사라진 경우 대비."""
    import app.api as api_mod
    from app.pipeline import derived  # 상한·TTL 상수의 소유자 (api는 같은 dict을 재노출)

    monkeypatch.setattr(derived, "_FACSIMILE_VERIFIED_MAX", 4)
    api_mod._FACSIMILE_VERIFIED.clear()
    try:
        for index in range(20):
            api_mod._facsimile_memo_set((f"job-{index}", "ko"), {"pages": index}, (1, index))
        assert len(api_mod._FACSIMILE_VERIFIED) <= 4
        assert api_mod._facsimile_memo_get(("job-19", "ko")) == ({"pages": 19}, (1, 19))
        monkeypatch.setattr(derived, "_FACSIMILE_MEMO_TTL_S", 0.0)
        assert api_mod._facsimile_memo_get(("job-19", "ko")) is None
    finally:
        api_mod._FACSIMILE_VERIFIED.clear()


def test_forget_job_caches_survives_concurrent_memo_writes():
    """DELETE가 캐시를 버리는 동안 다른 잡의 facsimile 준비가 같은 dict에 쓰면
    락 없는 순회는 RuntimeError('dictionary changed size during iteration')로
    204여야 할 DELETE를 500으로 만든다 (실측: 3000회 중 216회)."""
    import app.api as api_mod

    stop = threading.Event()
    errors: list[str] = []

    def _writer():
        index = 0
        while not stop.is_set():
            api_mod._facsimile_memo_set((f"other-{index}", "ko"), {"n": index}, (1, index))
            index = (index + 1) % 5000

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        for _ in range(3000):
            try:
                api_mod._forget_job_caches("job-x")
            except RuntimeError as e:  # noqa: PERF203 — 회귀 재현이 목적
                errors.append(str(e))
    finally:
        stop.set()
        writer.join(5)
        api_mod._FACSIMILE_VERIFIED.clear()
    assert errors == []


def test_stale_facsimile_staging_is_swept(tmp_path):
    """staging 임시 디렉터리는 정상 경로에서만 지워진다 — 강제 종료로 남은 잔해가
    어느 경로에서도 청소되지 않아 영구 잔존했다."""
    import app.api as api_mod

    rendered = tmp_path / "rendered"
    rendered.mkdir()
    orphan = rendered / ".ko.deadbeef.tmp"
    orphan.mkdir()
    (orphan / "page_0001.png").write_bytes(b"x")
    fresh = rendered / ".ko.feedface.tmp"
    fresh.mkdir()
    old = time.time() - api_mod._STAGING_STALE_S - 60
    os.utime(orphan, (old, old))

    assert api_mod._sweep_stale_staging(rendered, "ko") == 1
    assert not orphan.exists()
    assert fresh.is_dir()                       # 진행 중일 수 있는 최근 것은 남긴다
    assert api_mod._sweep_stale_staging(rendered, "en") == 0   # 다른 언어는 무관
    assert api_mod._sweep_stale_staging(tmp_path / "없음", "ko") == 0


def test_missing_file_at_send_time_is_404_not_500(tmp_path):
    """is_file() 검사와 실제 전송 사이에 잡이 삭제되면(DELETE·TTL GC) Starlette
    FileResponse는 RuntimeError를 던져 500이 된다 — 이 경합의 답은 404다."""
    import anyio
    import pytest
    from starlette.responses import FileResponse

    import app.api as api_mod

    target = tmp_path / "page_0001.png"

    async def _drive(response):
        sent: list[dict] = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await response({"type": "http", "method": "GET", "headers": []}, receive, send)
        return sent

    target.write_bytes(b"x")
    stock = FileResponse(target)
    target.unlink()
    with pytest.raises(RuntimeError):           # 기존 동작(=500)을 고정
        anyio.run(_drive, stock)

    target.write_bytes(b"x")
    guarded = api_mod._JobFileResponse(target)
    target.unlink()
    sent = anyio.run(_drive, guarded)
    starts = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert starts == [404]

    # 정상 파일은 그대로 200
    target.write_bytes(b"x")
    sent = anyio.run(_drive, api_mod._JobFileResponse(target))
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]


def test_dual_pdf_is_built_once_under_concurrent_first_requests(tmp_path, monkeypatch):
    """대조(dual) PDF 빌드만 잡 단위 락 밖에 남아, 프런트 기본 다운로드 경로의
    동시 첫 요청이 같은 PDF를 중복으로 만들었다 (실측: 3스레드 → 3회 빌드)."""
    import app.api as api_mod

    job = SimpleNamespace(id="job-dual", dir=tmp_path)
    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.4 source")
    translated = tmp_path / "export.ko.pdf"
    translated.write_bytes(b"%PDF-1.4 translated")

    builds: list[int] = []

    def _slow_build(source_pdf, translated_pdf, out):
        builds.append(1)
        time.sleep(0.2)                          # 동시 진입 창을 넓힌다
        out.write_bytes(b"%PDF-1.4 dual")
        return out

    monkeypatch.setattr(api_mod, "build_dual_pdf", _slow_build)
    results: list[Path] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(api_mod._ensure_dual_pdf(job, "ko", translated))
        )
        for _ in range(3)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert len(builds) == 1, builds
        assert results == [tmp_path / "export.ko.dual.pdf"] * 3
    finally:
        api_mod._forget_job_caches("job-dual")


def _dual_job(tmp_path: Path, name: str) -> tuple[SimpleNamespace, Path]:
    """대조 PDF 빌드에 필요한 최소 입력만 갖춘 가짜 잡."""
    job_dir = tmp_path / name
    job_dir.mkdir()
    (job_dir / "source.pdf").write_bytes(b"%PDF-1.4 source")
    translated = job_dir / "export.ko.pdf"
    translated.write_bytes(b"%PDF-1.4 translated")
    return SimpleNamespace(id=name, dir=job_dir), translated


def test_pdf_export_builds_are_globally_capped(tmp_path, monkeypatch):
    """잡 단위 락은 **같은 잡**의 중복 빌드만 막는다 — 서로 다른 잡 N개가 동시에
    요청되면 N개 빌드가 함께 돌았다(1건당 실측 9.4s/16p, CPU 포화).
    PDF_EXPORT_FORMAT_VERSION이 오르면 기존 배포의 전 캐시가 한꺼번에 무효화되므로
    업그레이드 직후 이 폭주가 실제로 일어난다."""
    import app.api as api_mod

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "2")
    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", "30")

    live = 0
    peak = 0
    guard = threading.Lock()

    def _slow_build(source_pdf, translated_pdf, out):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        time.sleep(0.15)                         # 동시 진입 창을 넓힌다
        with guard:
            live -= 1
        out.write_bytes(b"%PDF-1.4 dual")
        return out

    monkeypatch.setattr(api_mod, "build_dual_pdf", _slow_build)
    jobs = [_dual_job(tmp_path, f"capped-{index}") for index in range(6)]
    threads = [
        threading.Thread(
            target=lambda j=job, t=translated: api_mod._ensure_dual_pdf(j, "ko", t)
        )
        for job, translated in jobs
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert peak == 2, peak                   # 상한 없이는 6 (잡마다 락이 다르다)
        assert live == 0
        for job, _translated in jobs:
            assert (job.dir / "export.ko.dual.pdf").is_file()
    finally:
        for job, _translated in jobs:
            api_mod._forget_job_caches(job.id)


def test_pdf_export_cache_hit_does_not_take_a_build_slot(tmp_path, monkeypatch):
    """정상 사용(캐시 적중)은 전역 상한의 영향을 받지 않아야 한다 — 슬롯을 잡으면
    상한이 낮은 배포에서 이미 만들어 둔 PDF 다운로드까지 줄을 선다."""
    import app.api as api_mod
    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "1")
    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", "0.2")
    job, translated = _dual_job(tmp_path, "cached")
    dual = job.dir / "export.ko.dual.pdf"
    dual.write_bytes(b"%PDF-1.4 dual")           # 입력보다 나중 = 최신 캐시

    def _never(*args, **kwargs):
        raise AssertionError("캐시가 최신인데 다시 빌드했다")

    monkeypatch.setattr(api_mod, "build_dual_pdf", _never)
    try:
        with derived.export_build_slot():        # 유일한 슬롯을 점유한 채로
            assert api_mod._ensure_dual_pdf(job, "ko", translated) == dual
    finally:
        api_mod._forget_job_caches(job.id)


def test_pdf_export_queue_timeout_is_503_with_retry_after(client, sample_pdf, monkeypatch):
    """대기가 길어져도 요청이 무한정 매달리면 안 된다 — 재시도 가능한 과부하이므로
    잡 상태 오류(4xx)나 서버 결함(500)이 아니라 503 + Retry-After다."""
    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "1")
    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", "0.2")
    jid, _job_dir = _ko_layout_job(client, sample_pdf)

    with derived.export_build_slot():            # 유일한 슬롯을 점유한 채로
        busy = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert busy.status_code == 503, busy.text
    assert int(busy.headers["Retry-After"]) >= 1

    # 슬롯이 풀리면 같은 요청이 그대로 성공한다 (일시적 거절이지 영구 실패가 아니다)
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200


def _ko_layout_job(client, sample_pdf) -> tuple[str, Path]:
    """번역 산출물(result.ko.md·layout.ko.json)만 갖춘 잡 — 번역 엔진 없이
    한국어 facsimile 경로(export.ko.pdf → rendered/ko/*.png)를 실제로 태운다."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    job_dir = client.app.state.store.get(jid).dir
    (job_dir / "result.ko.md").write_text(
        (job_dir / "result.md").read_text(encoding="utf-8"), encoding="utf-8",
    )
    shutil.copyfile(job_dir / "layout.json", job_dir / "layout.ko.json")
    return jid, job_dir


def test_translated_page_recovers_from_lost_png(client, sample_pdf):
    """marker는 그대로인데 PNG만 사라지면, 프로세스 내 메모가 '검증됨'으로 굳어
    페이지 이미지가 프로세스 수명 내내 404로 남았다 — 재검증으로 자가 복구한다.
    같은 경로에서 고아 staging 디렉터리도 청소된다."""
    import app.api as api_mod

    jid, job_dir = _ko_layout_job(client, sample_pdf)
    assert client.get(f"/api/jobs/{jid}/page/1?lang=ko").status_code == 200
    png = job_dir / "rendered" / "ko" / "page_0001.png"
    assert png.is_file()
    assert (job_dir / "rendered" / "ko" / ".source.json").is_file()

    orphan = job_dir / "rendered" / ".ko.deadbeef.tmp"
    orphan.mkdir()
    old = time.time() - api_mod._STAGING_STALE_S - 60
    os.utime(orphan, (old, old))

    png.unlink()                                  # marker는 남겨 둔다 (부분 유실 재현)
    r = client.get(f"/api/jobs/{jid}/page/1?lang=ko")
    assert r.status_code == 200, r.text
    assert png.is_file()
    assert not orphan.exists()
    assert list((job_dir / "rendered").glob(".ko.*.tmp")) == []


def test_translated_layout_font_backfill_follows_enrich_version(client, sample_pdf):
    """ENRICH_VERSION 상향은 파생 산출물 layout.{lang}.json으로도 전파되어야 한다 —
    아니면 다음 상향 때 한국어 뷰만 구버전 폰트 메타로 고정된다."""
    import app.api as api_mod
    from app.pipeline.pdf_fonts import ENRICH_VERSION

    jid, job_dir = _ko_layout_job(client, sample_pdf)
    job = client.app.state.store.get(jid)
    stale = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    for page in stale:
        page["fonts_v"] = 1                       # 구버전 enrichment 결과 재현
    (job_dir / "layout.ko.json").write_text(json.dumps(stale), encoding="utf-8")
    original_before = (job_dir / "layout.json").read_bytes()

    pages = api_mod._load_layout_pages(job, "ko")
    assert [page.get("fonts_v") for page in pages] == [ENRICH_VERSION] * len(pages)
    saved = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    assert [page.get("fonts_v") for page in saved] == [ENRICH_VERSION] * len(saved)
    # 번역본 백필이 원본 layout.json을 덮어쓰지 않는다 (경로 혼동 회귀 방지)
    assert (job_dir / "layout.json").read_bytes() == original_before


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
