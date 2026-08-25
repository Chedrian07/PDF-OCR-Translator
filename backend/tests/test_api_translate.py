"""번역 API 레이어 테스트.

API 레이어의 상태·SSE·산출물 계약을 고립해 검증하도록 app.api.run_translation을
몽키패치로 대체한다. 페이크는 계약대로 동작한다: progress 콜백 호출,
translations/{lang}/state.json·result.ko.md·layout.ko.json 기록, TranslateResult 반환.
"""

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest
from conftest import wait_done

from app.config import Settings
from app.translate import TranslateResult
from app.main import create_app


# ── 페이크 run_translation ────────────────────────────────────────────────
def _make_fake(*, gate: threading.Event | None = None, wait_cancel: bool = False,
               total: int = 2):
    """계약을 지키는 페이크 run_translation을 만든다.

    gate: 주어지면 running 상태를 쓴 뒤 이 이벤트가 set될 때까지 블록(진행 관찰용).
    wait_cancel: True면 cancel 이벤트를 기다렸다가 취소로 종료.
    """
    md = "# 번역본\n\n안녕하세요. 번역된 문서입니다.\n"
    def fake(job_dir, lang, cfg, *, page_separator="\n\n---\n\n",
             progress=None, cancel=None, force=False, client=None):
        job_dir = Path(job_dir)
        tdir = job_dir / "translations" / lang
        tdir.mkdir(parents=True, exist_ok=True)

        def write_state(**over):
            base = {
                "lang": lang, "status": "running", "current": 0, "total": total,
                "error": None, "model": getattr(cfg, "model", ""),
                "api_mode": getattr(cfg, "api_mode", ""), "prompt_v": "1",
                "started_at": "2026-07-07T00:00:00+00:00", "finished_at": None,
            }
            base.update(over)
            (tdir / "state.json").write_text(
                json.dumps(base, ensure_ascii=False), encoding="utf-8")

        write_state(status="running", current=0, total=total)

        if wait_cancel:
            if cancel is not None:
                cancel.wait(timeout=10)
                if cancel.is_set():
                    write_state(status="canceled", current=0, error="번역이 취소되었습니다",
                                finished_at="2026-07-07T00:00:01+00:00")
                    return TranslateResult(status="canceled", total=total)

        if gate is not None:
            gate.wait(timeout=10)

        if progress is not None:
            progress(1, total)
            write_state(status="running", current=1, total=total)
            progress(total, total)

        (job_dir / f"result.{lang}.md").write_text(md, encoding="utf-8")
        # 실제 번역 계약처럼 원문 레이아웃을 깊은 복사하고 content만 바꾼다.
        layout = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
        first = True
        for page in layout:
            for block in page.get("blocks", []):
                content = str(block.get("content") or "").strip()
                if not content:
                    continue
                block["content"] = "안녕하세요" if first else f"번역본 {content}"
                first = False
        (job_dir / f"layout.{lang}.json").write_text(
            json.dumps(layout, ensure_ascii=False), encoding="utf-8")
        write_state(status="done", current=total, total=total,
                    finished_at="2026-07-07T00:00:02+00:00")
        return TranslateResult(status="done", total=total, translated=total, cached=0)

    return fake


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
@pytest.fixture
def provider_env(monkeypatch):
    """번역 프로바이더 env 설정 (TranslateConfig.from_env 통과용)."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    return monkeypatch


def _done_job(client, sample_pdf) -> str:
    r = client.post("/api/jobs", files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
                    data={"mode": "multi"})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    return jid


def _tstate(client, jid, lang="ko") -> dict:
    return client.get(f"/api/jobs/{jid}/translate/state?lang={lang}").json()


def _wait_until_status(client, jid, want, lang="ko", timeout=5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = _tstate(client, jid, lang)
        if body.get("status") == want:
            return body
        time.sleep(0.02)
    raise AssertionError(f"상태가 {want}가 되지 않음: {_tstate(client, jid, lang)}")


def _wait_no_task(client, jid, lang="ko", timeout=10.0) -> None:
    """번역 데몬 스레드가 완전히 끝날 때까지 대기(레지스트리에서 제거 = finally 실행 완료)."""
    tasks = client.app.state.translate_tasks
    lock = client.app.state.translate_lock
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with lock:
            present = (jid, lang) in tasks
        if not present:
            return
        time.sleep(0.02)
    raise AssertionError("번역 스레드가 시간 내에 종료되지 않음")


def _collect_sse(client, url, max_lines=500):
    """SSE 스트림을 (event, data_dict) 목록으로 수집. done/error에서 종료."""
    out = []
    with client.stream("GET", url) as s:
        cur = None
        seen = 0
        for line in s.iter_lines():
            seen += 1
            if seen > max_lines:
                break
            if line.startswith("event: "):
                cur = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                out.append((cur, json.loads(line.removeprefix("data: "))))
                if cur in ("done", "error"):
                    break
    return out


def test_translate_global_concurrency가_프로세스_세마포어_용량을_결정(settings):
    settings.translate_global_concurrency = 2
    slots = create_app(settings).state.translate_api_slots
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    assert not slots.acquire(blocking=False)
    slots.release()
    slots.release()


# ── 1. POST → 202 → events SSE progress…done ──────────────────────────────
def test_translate_global_concurrency_빈값은_잡당_설정으로_fallback(monkeypatch):
    """Compose가 빈 전역 키를 전달해도 잡당 값을 따라가며, 명시 값은 분리된다."""
    monkeypatch.setenv("TRANSLATE_CONCURRENCY", "3")
    monkeypatch.setenv("TRANSLATE_GLOBAL_CONCURRENCY", "")
    assert Settings.from_env().translate_global_concurrency == 3

    monkeypatch.setenv("TRANSLATE_GLOBAL_CONCURRENCY", "2")
    assert Settings.from_env().translate_global_concurrency == 2


def test_translate_post_then_events_stream(client, sample_pdf, provider_env, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr("app.api.run_translation", _make_fake(gate=gate))
    jid = _done_job(client, sample_pdf)

    r = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "running"

    # 페이크가 running을 기록하고 게이트에서 대기할 때까지 (state가 있어야 events가 404 안 남)
    _wait_until_status(client, jid, "running")

    events = []
    with client.stream("GET", f"/api/jobs/{jid}/translate/events?lang=ko") as s:
        for line in s.iter_lines():
            if line.startswith("event: "):
                ev = line.removeprefix("event: ").strip()
                events.append(ev)
                # 스냅샷 progress를 받은 뒤 페이크를 해제 → live progress/done이 이어진다
                if ev == "progress" and not gate.is_set():
                    gate.set()
            if "event: done" in line or "event: error" in line:
                break

    assert "progress" in events, events
    assert "done" in events, events
    _wait_no_task(client, jid)
    # done 이벤트 시점에 산출물이 존재한다
    assert client.get(f"/api/jobs/{jid}/markdown?lang=ko").status_code == 200


# ── 2. done 후 재-POST 200; force=true 재실행 ─────────────────────────────
def test_translate_idempotent_and_force(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"

    # 재-POST (force 없음) → 재실행 없이 200 done
    r2 = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "done"

    # force=true → 재실행 (게이트로 running 확정 관찰)
    gate = threading.Event()
    monkeypatch.setattr("app.api.run_translation", _make_fake(gate=gate))
    r3 = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko", "force": True})
    assert r3.status_code == 202
    assert r3.json()["status"] == "running"
    _wait_until_status(client, jid, "running")
    gate.set()
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"


# ── 3. 한국어 결과 라우트 — 전 404 / 후 200 ─────────────────────────────
def test_translated_output_routes(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    for path in (f"/api/jobs/{jid}/markdown?lang=ko",
                 f"/api/jobs/{jid}/html?lang=ko",
                 f"/api/jobs/{jid}/layout?lang=ko",
                 f"/api/jobs/{jid}/layout.html?lang=ko",
                 f"/api/jobs/{jid}/document.html?lang=ko"):
        assert client.get(path).status_code == 404, path

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)
    assert _tstate(client, jid)["status"] == "done"

    md = client.get(f"/api/jobs/{jid}/markdown?lang=ko")
    assert md.status_code == 200
    assert "번역본" in md.text
    assert "X-Partial" not in md.headers

    html = client.get(f"/api/jobs/{jid}/html?lang=ko")
    assert html.status_code == 200
    assert "X-Partial" not in html.headers

    layout = client.get(f"/api/jobs/{jid}/layout?lang=ko")
    assert layout.status_code == 200
    assert 'lang="ko"' in layout.text
    assert 'class="doclayout-body"' in layout.text

    legacy = client.get(
        f"/api/jobs/{jid}/layout.html?lang=ko",
        follow_redirects=False,
    )
    assert legacy.status_code == 307
    assert legacy.headers["location"] == f"/api/jobs/{jid}/document.html?lang=ko"

    document = client.get(f"/api/jobs/{jid}/document.html?lang=ko")
    assert document.status_code == 200
    assert document.text.startswith("<!doctype html>")
    assert 'lang="ko"' in document.text
    assert "번역본" in document.text
    assert ".ko.html" in document.headers["content-disposition"]
    assert f"/api/jobs/{jid}" not in document.text

    alignment = client.get(f"/api/jobs/{jid}/alignment?page=1&lang=ko")
    assert alignment.status_code == 200
    blocks = alignment.json()["blocks"]
    assert blocks
    assert any(block["translated"] for block in blocks)
    assert all(block["source"] for block in blocks)
    assert all(block["target"] for block in blocks)

    # 좌표가 달라진 손상 번역 레이아웃은 잘못 매핑하지 않고 명시적으로 거부한다.
    job = client.app.state.store.get(jid)
    translated_layout = json.loads(
        (job.dir / "layout.ko.json").read_text(encoding="utf-8")
    )
    translated_layout[0]["blocks"][0]["bbox"][0] += 1
    (job.dir / "layout.ko.json").write_text(
        json.dumps(translated_layout, ensure_ascii=False),
        encoding="utf-8",
    )
    broken = client.get(f"/api/jobs/{jid}/alignment?page=1&lang=ko")
    assert broken.status_code == 409


# ── 4. 400 잘못된 lang / 409 미완료 잡 / 503 env 미설정 ────────────────────
def test_translate_validation_errors(client, sample_pdf, monkeypatch):
    jid = _done_job(client, sample_pdf)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_MODEL", "m")

    # 400 — 지원하지 않는 언어
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "fr"}).status_code == 400

    # 409 — 완료되지 않은 잡 (상태를 강제로 비-done 처리)
    job = client.app.state.store.get(jid)
    old = job.status
    job.status = "running"
    try:
        assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 409
    finally:
        job.status = old

    # 503 — 프로바이더 env 미설정
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATE_MODEL", raising=False)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 503


# ── 5. cancel → error 이벤트 canceled:true ────────────────────────────────
def test_translate_cancel(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake(wait_cancel=True))
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_until_status(client, jid, "running")

    c = client.post(f"/api/jobs/{jid}/translate/cancel?lang=ko")
    assert c.status_code == 202
    assert c.json()["status"] == "canceling"

    assert _wait_until_status(client, jid, "canceled")["status"] == "canceled"
    _wait_no_task(client, jid)

    evs = _collect_sse(client, f"/api/jobs/{jid}/translate/events?lang=ko")
    canceled_errors = [d for e, d in evs if e == "error" and d.get("canceled") is True]
    assert canceled_errors, evs


# ── 5b. 잡 삭제가 실행 중 번역에 취소를 전파 ───────────────────────────────
def test_delete_job_cancels_running_translation(client, sample_pdf, provider_env, monkeypatch):
    """DELETE /jobs/{id}는 실행 중 번역 스레드의 cancel 이벤트를 set한다 —
    삭제된 디렉터리에 유료 API 호출·기록을 계속하지 않게."""
    monkeypatch.setattr("app.api.run_translation", _make_fake(wait_cancel=True))
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_until_status(client, jid, "running")
    with client.app.state.translate_lock:
        task = client.app.state.translate_tasks[(jid, "ko")]

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert task["cancel"].is_set()          # 삭제가 번역 취소를 전파했다
    _wait_no_task(client, jid)              # 스레드가 곧 종료 (레지스트리 정리)


# ── 6. stale 조정: running인데 태스크 없음 → error 재기록 ──────────────────
def test_translate_state_stale_adjusted(client, sample_pdf, settings):
    jid = _done_job(client, sample_pdf)
    tdir = settings.jobs_dir / jid / "translations" / "ko"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "state.json").write_text(json.dumps({
        "lang": "ko", "status": "running", "current": 1, "total": 3, "error": None,
    }), encoding="utf-8")

    body = client.get(f"/api/jobs/{jid}/translate/state?lang=ko").json()
    assert body["status"] == "error"
    assert "서버가 재시작" in body["error"]
    # 원자적으로 재기록되어 재조회해도 error 유지
    assert _tstate(client, jid)["status"] == "error"


# ── 6b. 레이스 회귀: state 읽기와 레지스트리 확인 사이 완료 → done 보존 ─────
def test_translate_state_race_done_preserved(client, sample_pdf, settings, monkeypatch):
    """read→check 순서 레이스 회귀: running을 읽은 직후 스레드가 완료(최종 done 기록
    → 레지스트리 제거)되어도 done을 error로 덮어쓰면 안 된다. _read_translate_state를
    감싸 읽기 직후에 스레드 완료 시퀀스를 그대로 재현한다."""
    import app.api as api_mod

    jid = _done_job(client, sample_pdf)
    tdir = settings.jobs_dir / jid / "translations" / "ko"
    tdir.mkdir(parents=True, exist_ok=True)
    state_path = tdir / "state.json"
    state_path.write_text(json.dumps({
        "lang": "ko", "status": "running", "current": 1, "total": 3, "error": None,
    }), encoding="utf-8")

    # 스레드가 아직 실행 중인 것처럼 레지스트리에 태스크를 넣어둔다
    st = client.app.state
    st.translate_tasks[(jid, "ko")] = {"thread": None, "cancel": threading.Event()}

    real_read = api_mod._read_translate_state
    fired = threading.Event()

    def racy_read(job, lang):
        state = real_read(job, lang)
        if not fired.is_set():
            fired.set()
            # 스레드 완료 시퀀스 재현: 최종 state 기록 → 레지스트리 제거 (이 순서 그대로).
            # 호출자가 translate_lock을 쥔 채일 수 있으므로 락 없이 직접 pop한다.
            state_path.write_text(json.dumps({
                "lang": "ko", "status": "done", "current": 3, "total": 3, "error": None,
            }), encoding="utf-8")
            st.translate_tasks.pop((jid, "ko"), None)
        return state

    monkeypatch.setattr("app.api._read_translate_state", racy_read)

    body = _tstate(client, jid)
    assert fired.is_set()
    assert body["status"] != "error", body  # 완료 직전 running을 stale로 오판 금지
    # 디스크의 최종 done이 보존된다 (error 덮어쓰기 금지)
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "done"
    # 재조회(레지스트리 확인 전에 이미 done이 써진 경우): 터미널 상태 그대로 반환
    assert _tstate(client, jid)["status"] == "done"


# ── 6c. api.py → 엔진 결선: 실 client가 앱 전역 세마포어를 들고 간다 ────────
def test_translate_스레드가_실클라이언트를_전역슬롯과_함께_주입(
    client, sample_pdf, provider_env, monkeypatch,
):
    """`request_semaphore` 키워드나 client 주입이 리팩터링에서 끊기면 프로덕션
    POST /translate만 죽고 페이크 기반 테스트는 전부 초록으로 남는다."""
    from app.translate.client import OpenAICompatClient

    seen = {}
    fake = _make_fake()

    def _capture(*args, **kwargs):
        seen["client"] = kwargs.get("client")
        return fake(*args, **kwargs)

    monkeypatch.setattr("app.api.run_translation", _capture)
    jid = _done_job(client, sample_pdf)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)

    assert isinstance(seen["client"], OpenAICompatClient)
    assert seen["client"]._request_semaphore is client.app.state.translate_api_slots


# ── 6d. 202 직후 events가 404가 아니다 (state.json 기록 전 창) ──────────────
def test_translate_events_open_right_after_202(client, sample_pdf, provider_env, monkeypatch):
    """POST가 202를 준 직후에는 워커가 아직 state.json을 쓰기 전일 수 있다.
    그 창에서 404를 주면 프런트가 SSE를 포기하고 폴백도 못 한다."""
    started = threading.Event()
    release = threading.Event()

    def _slow_start(job_dir, lang, cfg, **kwargs):
        started.set()
        release.wait(timeout=10)          # state.json을 쓰기 전에 멈춰 있는다
        return _make_fake()(job_dir, lang, cfg, **kwargs)

    monkeypatch.setattr("app.api.run_translation", _slow_start)
    jid = _done_job(client, sample_pdf)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    assert started.wait(5)
    assert not (
        settings_state_path := client.app.state.store.get(jid).dir
        / "translations" / "ko" / "state.json"
    ).exists(), settings_state_path

    try:
        with client.stream(
            "GET", f"/api/jobs/{jid}/translate/events?lang=ko"
        ) as stream:
            assert stream.status_code == 200
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    assert line.removeprefix("event: ").strip() == "progress"
                    break
    finally:
        release.set()
        _wait_no_task(client, jid)


# ── 6e. 남용 방어: 번역 레이트리밋 / 동시 실행 상한 → 429 ───────────────────
def test_translate_rate_limit_returns_429(tmp_path, sample_pdf, provider_env, monkeypatch):
    """무인증 서비스에서 200페이지 번역을 반복 트리거하지 못하게 한다.
    상한 안의 정상 사용은 통과하고, 초과분만 429 + Retry-After."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr("app.api.run_translation", _make_fake())
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, frontend_dir=tmp_path / "no-frontend",
        translate_rate_limit_per_min=2,
    )
    with TestClient(create_app(settings)) as client:
        jid = _done_job(client, sample_pdf)
        assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
        _wait_no_task(client, jid)
        assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 200
        r = client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1


def test_translate_concurrent_job_cap_returns_429(client, sample_pdf, provider_env, monkeypatch):
    """동시에 도는 번역 스레드 수에도 상한이 있다 — 초과 요청은 429."""
    # 가드는 첫 번역 요청에서 지연 생성되므로 그 전에 Settings 상한만 낮추면 된다
    client.app.state.settings.translate_max_active = 1
    monkeypatch.setattr("app.api.run_translation", _make_fake(wait_cancel=True))
    jid = _done_job(client, sample_pdf)

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_until_status(client, jid, "running")
    # 다른 잡(다른 (job,lang) 키)의 번역 시작이 상한에 걸린다
    other = _done_job(client, sample_pdf)
    r = client.post(f"/api/jobs/{other}/translate", json={"lang": "ko"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "30"

    client.post(f"/api/jobs/{jid}/translate/cancel?lang=ko")
    _wait_no_task(client, jid)
    # 슬롯이 비면 정상 사용 재개
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    assert client.post(f"/api/jobs/{other}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, other)


# ── 6f. 번역 페이지 이미지 캐시: 원자적 재생성 / 폰트 무효화 / 4xx ───────────
def _translate_and_render(client, sample_pdf, monkeypatch) -> tuple[str, Path]:
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)
    assert client.get(f"/api/jobs/{jid}/page/1?lang=ko").status_code == 200
    return jid, client.app.state.store.get(jid).dir


def test_translated_page_cache_regeneration_is_atomic(
    client, sample_pdf, provider_env, monkeypatch,
):
    """재생성 실패가 이미 서빙 중인 PNG를 지우면 안 된다 — 예전에는 렌더 전에
    기존 파일을 먼저 지워서, 실패하는 동안 /files가 404를 돌려줬다."""
    from app.pipeline.pdf_export import PdfExportError

    jid, job_dir = _translate_and_render(client, sample_pdf, monkeypatch)
    rendered = job_dir / "rendered" / "ko"
    stale = rendered / "page_0099.png"
    stale.write_bytes(b"stale")
    (rendered / ".source.json").unlink()        # 다음 요청에서 재생성되도록 무효화

    def _boom(*args, **kwargs):
        raise PdfExportError("렌더 실패")

    monkeypatch.setattr("app.api.render_pdf_pages", _boom)
    assert client.get(f"/api/jobs/{jid}/page/1?lang=ko").status_code == 409
    # 실패했어도 기존 캐시는 그대로 서빙된다
    assert (rendered / "page_0001.png").is_file()
    assert client.get(f"/api/jobs/{jid}/files/rendered/ko/page_0001.png").status_code == 200
    assert list((job_dir / "rendered").glob(".ko.*.tmp")) == []   # 임시 렌더 디렉터리 미잔존

    monkeypatch.undo()
    assert client.get(f"/api/jobs/{jid}/page/1?lang=ko").status_code == 200
    assert not stale.exists()                   # 세대에 없는 옛 페이지는 정리된다
    assert list((job_dir / "rendered").glob(".ko.*.tmp")) == []


def test_concurrent_first_requests_build_translated_pdf_once(
    client, sample_pdf, provider_env, monkeypatch,
):
    """같은 잡의 첫 진입이 동시에 들어와도 export.ko.pdf는 한 번만 만든다
    (예전에는 PDF 빌드가 락 밖이라 수십 초짜리 작업이 중복 실행됐다)."""
    import app.api as api_mod

    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)
    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)

    builds = []
    real_build = api_mod.build_translated_pdf

    def _slow_build(*args, **kwargs):
        builds.append(1)
        time.sleep(0.2)                     # 동시 진입 창을 넓힌다
        return real_build(*args, **kwargs)

    monkeypatch.setattr("app.api.build_translated_pdf", _slow_build)
    codes = []
    threads = [
        threading.Thread(
            target=lambda: codes.append(
                client.get(f"/api/jobs/{jid}/page/1?lang=ko").status_code
            )
        )
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert codes == [200, 200, 200]
    assert len(builds) == 1, builds


def test_translated_pdf_cache_invalidated_by_font_setting(
    client, sample_pdf, provider_env, monkeypatch,
):
    """PDF_EXPORT_FONT는 입력 파일이 아니라 설정이라 mtime 비교로는 잡히지 않는다 —
    바꾸면 예전 폰트로 조판된 export.ko.pdf가 계속 나갔다."""
    import app.api as api_mod

    builds = []
    real_build = api_mod.build_translated_pdf

    def _counting_build(*args, **kwargs):
        builds.append(kwargs.get("fontfile"))
        return real_build(*args, **kwargs)

    monkeypatch.setattr("app.api.build_translated_pdf", _counting_build)
    jid, _job_dir = _translate_and_render(client, sample_pdf, monkeypatch)
    assert len(builds) == 1

    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert len(builds) == 1                      # 설정이 그대로면 캐시 재사용

    client.app.state.settings.pdf_export_font = str(Path("/does/not/exist/ko.ttf"))
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert builds[-1] == "/does/not/exist/ko.ttf"
    assert len(builds) == 2                      # 폰트 설정 변경 → 재빌드


def test_translated_page_export_failure_is_4xx(client, sample_pdf, provider_env, monkeypatch):
    """내보내기 불가(입력 누락)는 서버 결함이 아니다 — 500이 아니라 4xx."""
    jid, job_dir = _translate_and_render(client, sample_pdf, monkeypatch)
    (job_dir / "source.pdf").unlink()
    (job_dir / "export.ko.pdf").unlink()
    (job_dir / "rendered" / "ko" / ".source.json").unlink()

    r = client.get(f"/api/jobs/{jid}/page/1?lang=ko")
    assert r.status_code == 409, r.text
    assert "원본 PDF가 없습니다" in r.json()["detail"]


# ── 7. /archive에 result.ko.md 포함 (캐시 삭제 후 재생성) ──────────────────
def test_archive_includes_translation(client, sample_pdf, provider_env, monkeypatch):
    monkeypatch.setattr("app.api.run_translation", _make_fake())
    jid = _done_job(client, sample_pdf)

    ar0 = client.get(f"/api/jobs/{jid}/archive")
    assert ar0.status_code == 200
    names0 = set(zipfile.ZipFile(io.BytesIO(ar0.content)).namelist())
    assert "result.md" in names0
    assert "result.ko.md" not in names0

    assert client.post(f"/api/jobs/{jid}/translate", json={"lang": "ko"}).status_code == 202
    _wait_no_task(client, jid)  # 스레드가 archive.zip 캐시를 삭제할 때까지
    assert _tstate(client, jid)["status"] == "done"

    ar1 = client.get(f"/api/jobs/{jid}/archive")
    assert ar1.status_code == 200
    names1 = set(zipfile.ZipFile(io.BytesIO(ar1.content)).namelist())
    assert "result.md" in names1
    assert "result.ko.md" in names1


# ── 8. 사유별 집계 노출 (report 조회 경로 + state 병합) ─────────────────────
_REPORT = {
    "kept_original": ["md:0:1"],
    "retried": 1, "repaired": 0, "split": 0, "sanitized": 0,
    "skipped": 3,
    "skip_reasons": {"references": 2, "already-korean": 1},
    "kept_reasons": {"gate-rejected": 1},
    "reference_rule": {"md_only": 0, "layout_only": 4, "sample_units": ["lay:9:2"]},
    "cached": 0, "translated": 5, "api_mode": "chat", "warnings": ["참고문헌 규칙 불일치"],
}


def _write_report(settings, jid, lang="ko") -> Path:
    tdir = settings.jobs_dir / jid / "translations" / lang
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "state.json").write_text(json.dumps({
        "lang": lang, "status": "done", "current": 5, "total": 5, "error": None,
    }), encoding="utf-8")
    (tdir / "report.json").write_text(json.dumps(_REPORT, ensure_ascii=False), encoding="utf-8")
    return tdir


def test_translate_report_없으면_404(client, sample_pdf, settings):
    jid = _done_job(client, sample_pdf)
    r = client.get(f"/api/jobs/{jid}/translate/report?lang=ko")
    assert r.status_code == 404


def test_translate_report는_사유별_집계를_노출한다(client, sample_pdf, settings):
    jid = _done_job(client, sample_pdf)
    _write_report(settings, jid)
    r = client.get(f"/api/jobs/{jid}/translate/report?lang=ko")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == jid and body["lang"] == "ko"
    assert body["skip_reasons"] == {"references": 2, "already-korean": 1}
    assert body["kept_reasons"] == {"gate-rejected": 1}
    assert body["reference_rule"]["layout_only"] == 4
    assert body["kept_original"] == ["md:0:1"]


def test_translate_state에_사유별_집계가_병합된다(client, sample_pdf, settings):
    """프런트가 상태 폴링 한 번으로 '왜 원문이 남았는지'를 알 수 있어야 한다."""
    jid = _done_job(client, sample_pdf)
    _write_report(settings, jid)
    body = _tstate(client, jid)
    assert body["status"] == "done"
    assert body["skip_reasons"] == {"references": 2, "already-korean": 1}
    assert body["kept_reasons"] == {"gate-rejected": 1}
    assert body["reference_rule"]["md_only"] == 0


def test_translate_report_지원하지_않는_언어는_400(client, sample_pdf, settings):
    jid = _done_job(client, sample_pdf)
    assert client.get(f"/api/jobs/{jid}/translate/report?lang=zz").status_code == 400
