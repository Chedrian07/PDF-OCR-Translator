"""파생 산출물(번역 PDF·facsimile) 보장 계층 — 실제 HTTP 라우트를 거치는 회귀 테스트.

derived를 직접 부르는 테스트만으로는 라우트가 먼저 연 대기 예산 같은 결선 결함을
놓친다(예열 대기 상한이 /pdf에서 한 번도 적용되지 않았다). 여기서는 TestClient로
/pdf·/page·/layout을 그대로 태운다.
"""

import json
import shutil
import threading
import time
from pathlib import Path

import pytest

from conftest import wait_done


def _upload(client, pdf_bytes: bytes):
    return client.post(
        "/api/jobs", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )


def _ko_layout_job(client, sample_pdf) -> tuple[str, Path]:
    """번역 산출물(result.ko.md·layout.ko.json)만 갖춘 완료 잡 — 번역 엔진 없이
    한국어 내보내기 경로(export.ko.pdf → rendered/ko/*.png)를 실제로 태운다."""
    jid = _upload(client, sample_pdf).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    job_dir = client.app.state.store.get(jid).dir
    (job_dir / "result.ko.md").write_text(
        (job_dir / "result.md").read_text(encoding="utf-8"), encoding="utf-8",
    )
    shutil.copyfile(job_dir / "layout.json", job_dir / "layout.ko.json")
    return jid, job_dir


def _wait_warm_idle(jid: str, lang: str = "ko", timeout: float = 30.0) -> None:
    from app.pipeline import derived

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with derived._WARM_GUARD:
            if (jid, lang) not in derived._WARM_INFLIGHT:
                return
        time.sleep(0.02)
    raise AssertionError("예열 스레드가 시간 내에 끝나지 않음")


@pytest.fixture
def gated_build(monkeypatch):
    """첫 빌드를 게이트에 묶어 둔다 — 예열이 잡 락·슬롯을 쥔 채 '빌드 중'인 상태 재현."""
    import app.api as api_mod

    real_build = api_mod.build_translated_pdf
    gate = threading.Event()
    entered = threading.Event()
    calls: list[str] = []

    def _gated(*args, **kwargs):
        calls.append(threading.current_thread().name)
        entered.set()
        assert gate.wait(15), "게이트가 열리지 않았다"
        return real_build(*args, **kwargs)

    monkeypatch.setattr(api_mod, "build_translated_pdf", _gated)
    state = {"gate": gate, "entered": entered, "calls": calls}
    try:
        yield state
    finally:
        gate.set()


def _export_env(monkeypatch, *, queue: str = "0.3", warm: str = "30", slots: str = "2"):
    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", slots)
    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", queue)   # 일반 대기열은 짧게
    monkeypatch.setenv("PDF_EXPORT_WARM_WAIT_S", warm)        # 예열 대기는 넉넉히
    derived._PDF_EXPORT_SLOTS = None                          # 상한 재적용


def _request_in_thread(client, url: str) -> tuple[threading.Thread, dict]:
    result: dict = {}

    def _run():
        started = time.monotonic()
        response = client.get(url)
        result.update(
            status=response.status_code,
            elapsed=time.monotonic() - started,
            headers=dict(response.headers),
            text=response.text if response.status_code != 200 else "",
            body=response.content,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


# ── F1-1: 예열 대기 상한이 실제 라우트에서 적용된다 ─────────────────────────
@pytest.mark.parametrize(
    "path",
    ["/pdf?lang=ko", "/pdf?lang=ko&view=dual", "/page/1?lang=ko", "/layout?lang=ko"],
)
def test_export_routes_wait_for_inflight_prewarm_instead_of_503(
    client, sample_pdf, monkeypatch, gated_build, path,
):
    """번역 직후 예열이 빌드 중일 때 들어온 요청은 일반 대기열 상한(여기 0.3s)이 아니라
    예열 대기 상한으로 기다렸다가 예열 결과를 받는다 — 503이 아니다.

    예전에는 /pdf가 먼저 연 30s 예산 때문에 예열 대기 연장이 버려졌고(중첩 예산),
    /page·/layout은 예열 여부를 보기도 전에 잡 락을 30s만 기다려 503이 됐다."""
    import app.api as api_mod

    _export_env(monkeypatch)
    jid, _job_dir = _ko_layout_job(client, sample_pdf)
    st = client.app.state
    job = st.store.get(jid)
    try:
        assert api_mod._warm_export_pdf(st, job, "ko") is True
        assert gated_build["entered"].wait(5), "예열이 빌드에 들어가지 않았다"

        thread, result = _request_in_thread(client, f"/api/jobs/{jid}{path}")
        # 일반 상한(0.3s)이었다면 이미 503이 돌아왔을 시간
        thread.join(1.2)
        assert thread.is_alive(), f"예열 대기 중인데 조기 응답: {result}"

        gated_build["gate"].set()
        thread.join(30)
        assert not thread.is_alive()
        assert result["status"] == 200, result
        # 요청은 예열이 만든 PDF를 재사용했다 — 같은 PDF를 두 번 빌드하지 않는다.
        assert len(gated_build["calls"]) == 1, gated_build["calls"]
        assert gated_build["calls"][0].startswith("pdf-warm-")
        if path.startswith("/layout"):
            assert f"/api/jobs/{jid}/files/rendered/ko/page_0001.png" in result["body"].decode()
    finally:
        gated_build["gate"].set()
        _wait_warm_idle(jid)


def test_layout_does_not_503_behind_the_prewarm_it_started_itself(
    client, sample_pdf, monkeypatch, gated_build,
):
    """ENRICH_VERSION 상향 뒤 /layout?lang=ko는 폰트 백필로 layout.ko.json을 다시 쓰고
    예열을 띄운다 — 그 직후 facsimile 준비가 **자기가 띄운** 예열 뒤에서 30s를 다 쓰고
    503이 됐다. 이제는 예열 대기 상한으로 기다려 같은 PDF를 받는다."""
    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    stale = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    for page in stale:
        page["fonts_v"] = 1                    # 구버전 enrichment 결과 → 백필 + 예열 유발
    (job_dir / "layout.ko.json").write_text(json.dumps(stale), encoding="utf-8")
    try:
        thread, result = _request_in_thread(client, f"/api/jobs/{jid}/layout?lang=ko")
        assert gated_build["entered"].wait(5), "백필이 예열을 띄우지 않았다"
        thread.join(1.2)
        assert thread.is_alive(), f"자기 예열 뒤에서 조기 응답: {result}"

        gated_build["gate"].set()
        thread.join(30)
        assert result["status"] == 200, result
        assert "rendered/ko/page_0001.png" in result["body"].decode()
        assert len(gated_build["calls"]) == 1, gated_build["calls"]
    finally:
        gated_build["gate"].set()
        _wait_warm_idle(jid)


@pytest.mark.parametrize("path", ["/pdf?lang=ko", "/page/1?lang=ko", "/layout?lang=ko"])
def test_busy_without_prewarm_is_503_with_retry_after(client, sample_pdf, monkeypatch, path):
    """예열이 없는데 슬롯이 막혀 있으면 일시적 과부하다 — 프런트 계약상 '재시도'인
    503 + Retry-After로 알린다(404/409는 '없음'이라 재시도하지 않는다)."""
    from app.pipeline import derived

    _export_env(monkeypatch, queue="0.2", slots="1")
    jid, _job_dir = _ko_layout_job(client, sample_pdf)
    with derived.export_build_slot():           # 유일한 슬롯을 다른 빌드가 점유
        busy = client.get(f"/api/jobs/{jid}{path}")
    assert busy.status_code == 503, busy.text
    assert int(busy.headers["Retry-After"]) >= 1
    assert client.get(f"/api/jobs/{jid}{path}").status_code == 200   # 일시적 거절


# ── 대기 예산 의미론 ──────────────────────────────────────────────────────
def test_nested_wait_budget_extends_for_a_larger_request_and_never_shrinks(monkeypatch):
    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", "5")
    with derived.export_wait_budget():
        assert derived._EXPORT_WAIT.remaining == 5.0
        with derived.export_wait_budget(1.0):            # 더 작은 요청 — 줄이지 않는다
            assert derived._EXPORT_WAIT.remaining == 5.0
        with derived.export_wait_budget(60.0):           # 예열 대기 — 늘린다
            assert derived._EXPORT_WAIT.remaining == 60.0
        assert derived._EXPORT_WAIT.remaining == 60.0    # 요청 단위로 유지된다
    assert derived._EXPORT_WAIT.remaining is None


def test_budget_extension_counts_time_already_waited(monkeypatch):
    """연장은 '이미 기다린 시간'을 빼고 잡는다 — 요청 전체 대기 합이 큰 쪽 상한을
    넘지 않는다(예열이 연달아 돌아도 180s를 두 번 기다리지 않는다)."""
    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_QUEUE_TIMEOUT_S", "0.5")
    lock = threading.Lock()
    lock.acquire()
    releaser = threading.Timer(0.2, lock.release)
    releaser.start()
    try:
        with derived.export_wait_budget():
            assert derived._acquire_within_budget(lock.acquire)
            waited = derived._EXPORT_WAIT.waited
            assert 0.1 < waited < 0.5
            with derived.export_wait_budget(1.0):
                assert derived._EXPORT_WAIT.remaining == pytest.approx(1.0 - waited, abs=0.01)
    finally:
        releaser.join()
        if lock.locked():
            lock.release()


def test_prewarm_budget_is_pinned_at_zero():
    """예열 스레드도 자기 (job, lang)이 '예열 중'이라 그대로면 예열 대기 연장이 자기
    예산을 0 → 180s로 늘려, 사용자 클릭을 밀어내지 않는다는 계약이 깨진다."""
    from app.pipeline import derived

    with derived._warm_budget():
        with derived.export_wait_budget(180.0):
            assert derived._EXPORT_WAIT.remaining == 0.0
    assert getattr(derived._EXPORT_WAIT, "remaining", None) is None
    assert getattr(derived._EXPORT_WAIT, "pinned", False) is False
