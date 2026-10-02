"""파생 산출물(번역 PDF·facsimile) 보장 계층 — 실제 HTTP 라우트를 거치는 회귀 테스트.

derived를 직접 부르는 테스트만으로는 라우트가 먼저 연 대기 예산 같은 결선 결함을
놓친다(예열 대기 상한이 /pdf에서 한 번도 적용되지 않았다). 여기서는 TestClient로
/pdf·/page·/layout을 그대로 태운다.
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from conftest import wait_done


@pytest.fixture(autouse=True)
def _skip_font_subsetting(monkeypatch):
    """여기 테스트는 라우트·캐시 의미론만 본다 — 실제 빌드마다 시스템 CJK 폰트를
    서브셋하는 비용(빌드당 ~1.3s, 이 파일 전체로 수십 초)은 건너뛴다. 서브셋 자체의
    정확성은 tests/test_pdf_export_subset.py가 지킨다."""
    monkeypatch.setattr(
        "app.pipeline.pdf_export.build.subset_font_files", lambda *args, **kwargs: {},
    )


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


# ── F1-2: 캐시 판정은 기록된 입력 지문의 일치로 ─────────────────────────────
_NEW_TRANSLATION = "갱신된 번역 문장입니다"


def _retranslate_layout(job_dir: Path, text: str = _NEW_TRANSLATION) -> None:
    """재번역 완료를 흉내낸다 — layout.ko.json을 원자적으로 교체(새 inode)."""
    pages = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
    for page in pages:
        for block in page.get("blocks", ()):
            if block.get("type") in ("text", "title"):
                block["content"] = text
    tmp = job_dir / ".layout.ko.retranslate.tmp"
    tmp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_dir / "layout.ko.json")


def _pdf_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        # NBSP 등 공백 변형은 하나로 정규화한다(조판기가 단어 사이에 넣을 수 있다)
        return " ".join(" ".join(page.get_text() for page in doc).split())


def test_translation_replaced_during_build_is_not_frozen_as_current(
    client, sample_pdf, monkeypatch,
):
    """빌드 도중 재번역이 끝나면(layout.ko.json 교체) 그 빌드는 옛 번역으로 만든 PDF다.

    예전에는 출력 mtime이 입력보다 늦다는 이유로 그 PDF가 '최신'으로 굳어, 다음
    무효화 전까지 모든 다운로드가 옛 번역을 받았다(결정적 재현). 이제는 빌드 전
    지문으로 표식을 남겨 캐시로 확정하지 않고, 새 입력으로 다시 예열한다."""
    import app.api as api_mod
    from app.pipeline import derived

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    job = client.app.state.store.get(jid)
    real_build = api_mod.build_translated_pdf
    calls: list[str] = []

    def _build_then_retranslate(*args, **kwargs):
        result = real_build(*args, **kwargs)        # 옛 번역을 읽어 조판했다
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            _retranslate_layout(job_dir)            # 그 사이 재번역이 끝났다
            # 실제 경합의 순서 재현: 빌더는 입력을 먼저 읽고 출력을 **마지막에** 쓴다 —
            # 출력 mtime이 교체된 입력보다 늦다(예전 mtime 판정이 '최신'으로 오판한 조건).
            later = time.time_ns() + 1_000_000_000
            os.utime(result.path, ns=(later, later))
        return result

    monkeypatch.setattr(api_mod, "build_translated_pdf", _build_then_retranslate)
    first = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert first.status_code == 200, first.text
    assert _NEW_TRANSLATION not in _pdf_text(first.content)   # 요청 시점 입력의 결과

    font_id = derived._pdf_export_font_id(client.app.state.settings)
    _wait_warm_idle(jid)                            # 입력 변경 감지 → 재예열
    current, _out, _report = derived._translated_pdf_cache(job, "ko", font_id)
    assert current, "재예열이 새 번역으로 캐시를 다시 만들어야 한다"
    assert len(calls) == 2 and calls[1].startswith("pdf-warm-"), calls

    second = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert second.status_code == 200
    assert _NEW_TRANSLATION in _pdf_text(second.content)
    assert len(calls) == 2                          # 재예열 결과를 그대로 받았다


def test_cache_is_stale_when_an_input_is_replaced_within_the_same_mtime_tick(
    client, sample_pdf, monkeypatch,
):
    """mtime 해상도가 거친 파일시스템(같은 틱의 교체)에서도 교체를 놓치지 않는다 —
    지문에 inode가 들어 있어 크기·mtime이 같아도 다른 파일로 판정한다."""
    from app.pipeline import derived

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    job = client.app.state.store.get(jid)
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    font_id = derived._pdf_export_font_id(client.app.state.settings)
    assert derived._translated_pdf_cache(job, "ko", font_id)[0]

    layout = job_dir / "layout.ko.json"
    before = layout.stat()
    payload = layout.read_bytes()
    tmp = job_dir / ".layout.ko.same.tmp"
    tmp.write_bytes(payload)                        # 같은 내용·크기
    os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))   # 같은 mtime
    tmp.replace(layout)
    assert layout.stat().st_mtime_ns == before.st_mtime_ns
    assert not derived._translated_pdf_cache(job, "ko", font_id)[0]


def test_legacy_font_marker_is_rebuilt_once_into_a_build_stamp(
    client, sample_pdf, monkeypatch,
):
    """업그레이드 직후 예전 폰트 표식(문자열)만 있는 캐시는 한 번 다시 만든다."""
    import app.api as api_mod
    from app.pipeline import artifacts

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    artifacts.export_font_marker(job_dir, "ko").write_text("auto", encoding="utf-8")

    builds: list[int] = []
    real_build = api_mod.build_translated_pdf
    monkeypatch.setattr(
        api_mod, "build_translated_pdf",
        lambda *a, **kw: (builds.append(1), real_build(*a, **kw))[1],
    )
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert builds == [1]
    stamp = json.loads(artifacts.export_font_marker(job_dir, "ko").read_text(encoding="utf-8"))
    assert set(stamp["inputs"]) == {"source.pdf", "layout.json", "layout.ko.json"}


def test_prewarm_requested_while_one_is_running_runs_again_afterwards(tmp_path, monkeypatch):
    """진행 중 예열에 들어온 부탁을 버리면, 그 부탁을 만든 새 입력(재번역 완료)이
    반영되지 않은 채 끝난다 — dirty로 남겨 같은 스레드가 한 번 더 돈다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    gate = threading.Event()
    entered = threading.Event()
    rounds: list[int] = []
    real_warm = derived.warm_translated_pdf

    def _counting_warm(*args, **kwargs):
        rounds.append(1)
        return real_warm(*args, **kwargs)

    def _slow_build(job_dir, lang, *, fontfile=""):
        entered.set()
        assert gate.wait(10)
        raise derived.PdfExportError("입력 없음")   # 결과는 상관없다 — 회차만 센다

    monkeypatch.setattr(derived, "warm_translated_pdf", _counting_warm)
    job = SimpleNamespace(id="dirty-warm", dir=tmp_path)
    settings = SimpleNamespace(pdf_export_font="")
    try:
        assert derived.warm_translated_pdf_async(job, "ko", settings, build=_slow_build)
        assert entered.wait(5)
        # 진행 중 — 새로 띄우지 않지만(False) 버리지도 않는다
        assert derived.warm_translated_pdf_async(job, "ko", settings, build=_slow_build) is False
        assert derived.warm_translated_pdf_async(job, "ko", settings, build=_slow_build) is False
        gate.set()
        _wait_warm_idle("dirty-warm")
    finally:
        gate.set()
    assert len(rounds) == 2      # 부탁 두 번은 한 회차로 합쳐진다
    assert ("dirty-warm", "ko") not in derived._WARM_DIRTY


def test_prewarm_thread_start_failure_does_not_leave_it_inflight(tmp_path, monkeypatch):
    """스레드를 못 띄웠는데 '예열 중' 표식이 남으면 이후 예열이 전부 건너뛰어진다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    def _no_threads(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _no_threads)
    job = SimpleNamespace(id="warm-start-fail", dir=tmp_path)
    with pytest.raises(RuntimeError):
        derived.warm_translated_pdf_async(job, "ko", SimpleNamespace(pdf_export_font=""))
    assert not derived._warm_inflight("warm-start-fail", "ko")


# ── pdf-export-16: 자동 폰트 해석·서브셋 가능 여부도 캐시 정체성이다 ─────────
@pytest.fixture
def fake_font_env(tmp_path, monkeypatch):
    """시스템 폰트 후보를 임시 경로로 바꾸고 폰트 환경 메모를 매번 다시 계산하게 한다."""
    from app.pipeline import derived
    from app.pipeline.pdf_export import fonts

    candidate = tmp_path / "fonts" / "NotoSerifCJK-KR.otf"
    monkeypatch.setattr(fonts, "_SYSTEM_FONT_CANDIDATES", (str(candidate),))
    monkeypatch.setattr(fonts, "_SYSTEM_SANS_FONT_CANDIDATES", ())
    monkeypatch.setattr(fonts, "_fontconfig_candidates", lambda: ())
    monkeypatch.setattr(derived, "_FONT_ENV", None)
    monkeypatch.setattr(derived, "_FONT_ENV_TTL_S", 0.0)
    return candidate


def test_auto_font_identity_changes_when_a_korean_font_is_installed(fake_font_env):
    """폰트 없는 환경의 '1em 전각 자간' PDF가 fonts-noto-cjk 설치 뒤에도 캐시로 나갔다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    settings = SimpleNamespace(pdf_export_font="")
    without = derived._pdf_export_font_id(settings)
    fake_font_env.parent.mkdir(parents=True)
    fake_font_env.write_bytes(b"fake-otf")
    with_font = derived._pdf_export_font_id(settings)
    assert with_font != without
    assert derived._pdf_export_font_id(settings) == with_font   # 같은 환경 → 같은 값


def test_font_identity_tracks_fonttools_availability(fake_font_env, monkeypatch):
    """fontTools 설치 전 만든 비서브셋(대용량) PDF도 설치 뒤 다시 만든다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    for explicit in ("", str(fake_font_env)):
        settings = SimpleNamespace(pdf_export_font=explicit)
        monkeypatch.setattr(derived, "_fonttools_available", lambda: False)
        full = derived._pdf_export_font_id(settings)
        monkeypatch.setattr(derived, "_fonttools_available", lambda: True)
        assert derived._pdf_export_font_id(settings) != full, explicit


def test_installing_fonttools_rebuilds_a_cached_export(client, sample_pdf, monkeypatch):
    """라우트 경로 — 폰트 환경이 바뀌면 다음 다운로드가 캐시 대신 다시 만든다."""
    import app.api as api_mod
    from app.pipeline import derived

    _export_env(monkeypatch)
    monkeypatch.setattr(derived, "_FONT_ENV", None)
    monkeypatch.setattr(derived, "_FONT_ENV_TTL_S", 0.0)
    monkeypatch.setattr(derived, "_fonttools_available", lambda: False)
    jid, _job_dir = _ko_layout_job(client, sample_pdf)
    builds: list[int] = []
    real_build = api_mod.build_translated_pdf
    monkeypatch.setattr(
        api_mod, "build_translated_pdf",
        lambda *a, **kw: (builds.append(1), real_build(*a, **kw))[1],
    )
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert builds == [1]                                   # 같은 환경 → 캐시 적중
    monkeypatch.setattr(derived, "_fonttools_available", lambda: True)
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200
    assert builds == [1, 1]                                # 환경 변화 → 재빌드


def test_font_environment_is_memoized_between_requests(monkeypatch):
    """캐시 판정은 /page?lang 요청마다 돈다 — 후보 stat·fc-list를 매번 하지 않는다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    calls: list[int] = []
    monkeypatch.setattr(derived, "_FONT_ENV", None)
    monkeypatch.setattr(
        derived, "_system_font_digest", lambda: (calls.append(1), "digest")[1],
    )
    settings = SimpleNamespace(pdf_export_font="")
    for _ in range(5):
        derived._pdf_export_font_id(settings)
    assert calls == [1]


# ── gap3-…-7: 예열은 사용자 클릭 몫의 빌드 슬롯을 남긴다 ─────────────────────
def _bare_job(tmp_path: Path, name: str):
    """빌드 입력만 갖춘 가짜 잡 (빌더는 주입한다)."""
    from types import SimpleNamespace

    job_dir = tmp_path / name
    job_dir.mkdir()
    for file_name in ("source.pdf", "layout.json", "layout.ko.json"):
        (job_dir / file_name).write_text("[]", encoding="utf-8")
    return SimpleNamespace(id=name, dir=job_dir)


def _fake_build_factory(gate: threading.Event | None = None, entered: list | None = None):
    from types import SimpleNamespace

    from app.pipeline.pdf_export import PDF_EXPORT_FORMAT_VERSION

    def _build(job_dir, lang, *, fontfile=""):
        if entered is not None:
            entered.append(job_dir.name)
        if gate is not None:
            assert gate.wait(10), "게이트가 열리지 않았다"
        out = job_dir / f"export.{lang}.pdf"
        out.write_bytes(b"%PDF-1.4 built")
        report = {"format_version": PDF_EXPORT_FORMAT_VERSION}
        (job_dir / f"export.{lang}.report.json").write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(path=out, report=lambda: dict(report))

    return _build


def test_prewarms_leave_one_build_slot_for_user_clicks(tmp_path, monkeypatch):
    """번역이 끝난 두 잡의 예열이 빌드 슬롯(기본 2)을 다 채우면, 아무도 누르지 않은
    백그라운드 작업 때문에 세 번째 잡의 진짜 클릭이 대기열 상한 뒤 503을 받았다
    (주석의 불변식 '예열은 사용자 요청을 밀어내지 않는다' 위반, 재현됨)."""
    from types import SimpleNamespace

    from app.pipeline import derived

    _export_env(monkeypatch, queue="0.3", slots="2")
    derived._WARM_SLOTS = None
    gate = threading.Event()
    entered: list[str] = []
    gated = _fake_build_factory(gate, entered)
    settings = SimpleNamespace(pdf_export_font="")
    warm_a, warm_b, clicked = (_bare_job(tmp_path, n) for n in ("warm-a", "warm-b", "click-c"))
    outcomes: dict[str, object] = {}

    def _warm(job):
        outcomes[job.id] = derived.warm_translated_pdf(job, "ko", settings, build=gated)

    first = threading.Thread(target=_warm, args=(warm_a,), daemon=True)
    second = threading.Thread(target=_warm, args=(warm_b,), daemon=True)
    try:
        first.start()
        deadline = time.monotonic() + 5
        while "warm-a" not in entered and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered == ["warm-a"]
        second.start()
        second.join(5)
        # 두 번째 예열은 예열 몫이 없어 기다리지 않고 포기한다(빌드에 들어가지 않는다)
        assert not second.is_alive() and outcomes["warm-b"] is False
        assert entered == ["warm-a"]

        started = time.monotonic()
        path, _report = derived._ensure_translated_pdf(
            clicked, "ko", settings, build=_fake_build_factory(),
        )
        assert path.name == "export.ko.pdf"
        assert time.monotonic() - started < 2.0          # 남겨 둔 클릭 몫 슬롯으로 즉시 빌드
    finally:
        gate.set()
        first.join(10)
        for job in (warm_a, warm_b, clicked):
            derived._forget_job_caches(job.id)
    assert outcomes["warm-a"] is True


def test_single_slot_deployment_does_not_prewarm(tmp_path, monkeypatch):
    """상한 1이면 클릭 몫을 남길 여유가 없다 — 예열은 빌드하지 않고 클릭이 직접 만든다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    _export_env(monkeypatch, queue="0.3", slots="1")
    derived._WARM_SLOTS = None
    entered: list[str] = []
    settings = SimpleNamespace(pdf_export_font="")
    job = _bare_job(tmp_path, "single-slot")
    try:
        assert derived.warm_translated_pdf(
            job, "ko", settings, build=_fake_build_factory(entered=entered),
        ) is False
        assert entered == []
        derived._ensure_translated_pdf(job, "ko", settings, build=_fake_build_factory(entered=entered))
        assert entered == ["single-slot"]
    finally:
        derived._forget_job_caches(job.id)


# ── 테스트 격리: 앞 테스트가 남긴 예열 빌드가 다음 테스트의 예열을 막지 않는다 ─────────
# 예열은 TestClient보다 오래 사는 데몬 스레드이고 빌드 슬롯은 모듈 전역이다. conftest의
# _fresh_pdf_export_slots가 없을 때는 CI 러너 속도에서 test_archive_includes_translation의 예열이
# 예열 몫(기본 1)을 쥔 채 남아, 바로 다음 test_번역이_끝나면_내보내기_PDF를_미리_만들어_둔다의
# 예열이 조용히 포기하고 실패했다(P4 Linux CI 재현). 아래 두 테스트는 그 순서를 그대로 재현한다.
_LEFTOVER_WARM: dict = {}


def test_a_warm_build_left_running_past_its_test(tmp_path, monkeypatch):
    """(다음 테스트의 전제) 끝나지 않은 예열 빌드를 남긴 채 끝난다 — 다음 테스트가 정리한다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "2")
    gate = threading.Event()
    entered: list[str] = []
    job = _bare_job(tmp_path, "leftover-warm")
    thread = threading.Thread(
        target=derived.warm_translated_pdf,
        args=(job, "ko", SimpleNamespace(pdf_export_font="")),
        kwargs={"build": _fake_build_factory(gate, entered)},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not entered and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entered == ["leftover-warm"]                 # 예열 몫을 쥐고 빌드 중
    _LEFTOVER_WARM.update(gate=gate, thread=thread, job=job)


def test_the_next_test_still_prewarms_while_a_leftover_warm_build_runs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.pipeline import derived

    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "2")
    job = _bare_job(tmp_path, "next-test-warm")
    try:
        assert derived.warm_translated_pdf(
            job, "ko", SimpleNamespace(pdf_export_font=""), build=_fake_build_factory(),
        ) is True, "앞 테스트가 남긴 예열이 예열 몫을 쥐고 있어 새 예열이 포기했다"
        assert (job.dir / "export.ko.pdf").is_file()
    finally:
        derived._forget_job_caches(job.id)
        gate = _LEFTOVER_WARM.pop("gate", None)
        if gate is not None:
            gate.set()
            _LEFTOVER_WARM.pop("thread").join(10)
            derived._forget_job_caches(_LEFTOVER_WARM.pop("job").id)


# ── concurrency-6: MuPDF 예외도 사용자용 내보내기 오류로 정규화된다 ─────────────
def _mupdf_failure(*args, **kwargs):
    import pymupdf

    # 사라진 디렉터리에 doc.save → 실제로 나는 예외 (FzErrorBase 직계, OSError 아님)
    raise pymupdf.mupdf.FzErrorSystem("code=2: cannot open file: No such file or directory")


def test_mupdf_error_during_export_keeps_each_route_contract(client, sample_pdf, monkeypatch):
    """예전에는 MuPDF 예외가 정규화를 우회해 /pdf·/page가 메시지 없는 500, /layout·
    document.html은 문서화된 '좌표 텍스트 렌더 폴백' 대신 500, 예열 스레드는 죽었다."""
    import app.api as api_mod
    from app.pipeline import derived

    _export_env(monkeypatch)
    monkeypatch.setattr(api_mod, "build_translated_pdf", _mupdf_failure)
    jid, _job_dir = _ko_layout_job(client, sample_pdf)

    for path in ("/pdf?lang=ko", "/pdf?lang=ko&view=dual", "/page/1?lang=ko"):
        r = client.get(f"/api/jobs/{jid}{path}")
        assert r.status_code == 409, (path, r.status_code, r.text)
        assert "번역 PDF를 만들 수 없습니다" in r.json()["detail"]

    layout = client.get(f"/api/jobs/{jid}/layout?lang=ko")
    assert layout.status_code == 200                    # 좌표 텍스트 렌더 폴백
    assert "rendered/ko" not in layout.text
    assert client.get(f"/api/jobs/{jid}/document.html?lang=ko").status_code == 200

    job = client.app.state.store.get(jid)
    settings = client.app.state.settings
    assert derived.warm_translated_pdf(job, "ko", settings, build=_mupdf_failure) is False


def test_page_render_failure_is_409_not_500(client, sample_pdf, monkeypatch):
    """래스터 실패(전 페이지 실패·암호화·페이지 상한 = ValueError)도 잡 상태 문제다."""
    import app.api as api_mod

    _export_env(monkeypatch)
    jid, _job_dir = _ko_layout_job(client, sample_pdf)

    def _all_pages_failed(*args, **kwargs):
        raise ValueError("모든 페이지 렌더에 실패했습니다")

    monkeypatch.setattr(api_mod, "render_pdf_pages", _all_pages_failed)
    r = client.get(f"/api/jobs/{jid}/page/1?lang=ko")
    assert r.status_code == 409, r.text
    assert "모든 페이지 렌더에 실패했습니다" in r.json()["detail"]


def test_pdf_layout_mismatch_is_409_with_the_reason(client, sample_pdf, monkeypatch):
    """레이아웃 대응 불일치는 결정적인 잡 상태 오류다 — /page처럼 409 + 원인 문구."""
    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    pages = json.loads((job_dir / "layout.ko.json").read_text(encoding="utf-8"))
    pages[0]["blocks"] = pages[0]["blocks"][:-1]          # 블록 하나가 사라진 번역
    (job_dir / "layout.ko.json").write_text(json.dumps(pages), encoding="utf-8")

    r = client.get(f"/api/jobs/{jid}/pdf?lang=ko")
    assert r.status_code == 409, r.text
    assert "블록 수가 일치하지 않습니다" in r.json()["detail"]


# ── concurrency-8: 삭제와 겹친 빌드가 잡 디렉터리를 되살리지 않는다 ─────────────
def test_dual_build_finishing_after_delete_does_not_resurrect_the_job(
    client, sample_pdf, monkeypatch,
):
    """대조 PDF 빌드 중 잡이 삭제되면, 빌더의 out.parent.mkdir(parents=True)가 meta.json
    없는 잡 디렉터리를 되살려 export.ko.dual.pdf만 든 영구 고아가 됐다(재현)."""
    import app.api as api_mod

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    store = client.app.state.store
    job = store.get(jid)
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200

    def _build_racing_delete(source_pdf, translated_pdf, out):
        store.delete_dir(job)                      # 빌드 도중 DELETE(또는 TTL GC)
        out.parent.mkdir(parents=True, exist_ok=True)   # 빌더가 하는 그대로
        out.write_bytes(b"%PDF-1.4 dual")
        return out

    monkeypatch.setattr(api_mod, "build_dual_pdf", _build_racing_delete)
    r = client.get(f"/api/jobs/{jid}/pdf?lang=ko&view=dual")
    assert r.status_code in (404, 409), r.text
    assert not job_dir.exists(), sorted(p.name for p in job_dir.iterdir())


def test_facsimile_render_finishing_after_delete_does_not_resurrect_the_job(
    client, sample_pdf, monkeypatch,
):
    import app.api as api_mod

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    store = client.app.state.store
    job = store.get(jid)

    def _render_racing_delete(pdf_path, out_dir, *, dpi, max_pages):
        store.delete_dir(job)
        out_dir.mkdir(parents=True, exist_ok=True)      # render_pdf_pages가 하는 그대로
        (out_dir / "page_0001.png").write_bytes(b"\x89PNG")

    monkeypatch.setattr(api_mod, "render_pdf_pages", _render_racing_delete)
    r = client.get(f"/api/jobs/{jid}/page/1?lang=ko")
    assert r.status_code in (404, 409), r.text
    assert not job_dir.exists()


def test_export_on_an_already_deleted_job_dir_does_not_recreate_it(tmp_path):
    """삭제가 먼저 끝났으면 빌드·렌더를 아예 시작하지 않는다."""
    from types import SimpleNamespace

    from app.pipeline import derived

    job = SimpleNamespace(id="gone", dir=tmp_path / "gone", dpi=72)
    settings = SimpleNamespace(pdf_export_font="", max_pages=10)
    with pytest.raises(derived.PdfExportError):
        derived._ensure_facsimile_pages(job, [1], "ko", settings, build=_mupdf_failure)
    with pytest.raises(derived.PdfExportError):
        derived._ensure_dual_pdf(job, "ko", job.dir / "export.ko.pdf", build=_mupdf_failure)
    assert not job.dir.exists()


def test_translate_state_write_does_not_recreate_a_deleted_job_dir(tmp_path):
    from types import SimpleNamespace

    import app.api as api_mod

    job = SimpleNamespace(id="gone", dir=tmp_path / "gone")
    with pytest.raises(OSError):
        api_mod._write_translate_state(job, "ko", {"status": "error"})
    assert not job.dir.exists()


# ── api-jobs-15 · concurrency-7: archive.zip은 내용 지문으로 판정한다 ─────────────
def _zip_names(data: bytes) -> set[str]:
    import io
    import zipfile

    return set(zipfile.ZipFile(io.BytesIO(data)).namelist())


def test_archive_built_across_a_translation_finish_is_not_served_stale(
    client, sample_pdf, monkeypatch,
):
    """번역이 끝나기 직전에 시작된 zip 빌드가 무효화(unlink) 뒤에 옛 zip을 써 넣으면,
    존재 여부만 보던 캐시가 다음 번역 전까지 번역본 없는 zip을 계속 내보냈다(재현)."""
    import zipfile

    from app.pipeline import artifacts

    jid = _upload(client, sample_pdf).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    job_dir = client.app.state.store.get(jid).dir
    real_write = zipfile.ZipFile.write
    finished = []

    def _write_while_translation_finishes(self, filename, arcname=None, *args, **kwargs):
        if str(arcname).startswith("images/") and not finished:
            finished.append(1)
            # _run_translate_thread가 하는 그대로: 번역본 기록 → 캐시 무효화
            (job_dir / "result.ko.md").write_text("# 번역본", encoding="utf-8")
            artifacts.archive(job_dir).unlink(missing_ok=True)
        return real_write(self, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", _write_while_translation_finishes)
    first = client.get(f"/api/jobs/{jid}/archive")
    assert first.status_code == 200
    assert "result.ko.md" in _zip_names(first.content)    # 빌드 중 바뀐 입력 → 같은 요청에서 재빌드
    monkeypatch.undo()
    second = client.get(f"/api/jobs/{jid}/archive")
    assert "result.ko.md" in _zip_names(second.content)


def test_archive_is_reused_until_its_contents_change(client, sample_pdf):
    from app.pipeline import artifacts

    jid = _upload(client, sample_pdf).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    job_dir = client.app.state.store.get(jid).dir
    first = client.get(f"/api/jobs/{jid}/archive")
    zip_path = artifacts.archive(job_dir)
    built = zip_path.stat()
    assert client.get(f"/api/jobs/{jid}/archive").content == first.content
    assert zip_path.stat().st_ino == built.st_ino         # 캐시 적중 — 다시 만들지 않는다

    # 무효화 없이(외부 수정·누락된 무효화) 내용만 바뀌어도 지문이 달라 다시 만든다
    (job_dir / "result.ko.md").write_text("# 늦게 도착한 번역본", encoding="utf-8")
    third = client.get(f"/api/jobs/{jid}/archive")
    assert "result.ko.md" in _zip_names(third.content)


# ── PDF 생성 리포트(JSON) — 헤더에는 숫자만, 상세는 /pdf/report로 ─────────────────
def test_pdf_report_route_serves_the_report_of_the_last_build(client, sample_pdf, monkeypatch):
    """프런트(reader.js)는 PDF 다운로드 뒤 /pdf/report를 불러 원문 보존 사유·스캔 픽셀
    지움·주의 문장을 그린다. 라우트가 없으면 매 다운로드가 404였다."""
    from app.pipeline import artifacts

    _export_env(monkeypatch)
    jid, job_dir = _ko_layout_job(client, sample_pdf)
    url = f"/api/jobs/{jid}/pdf/report?lang=ko"

    missing = client.get(url)                      # 빌드 전 — 리포트 없음
    assert missing.status_code == 404
    assert "PDF 생성 리포트가 없습니다" in missing.json()["detail"]

    pdf = client.get(f"/api/jobs/{jid}/pdf?lang=ko&view=dual")
    assert pdf.status_code == 200, pdf.text
    r = client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == jid and body["lang"] == "ko"
    stored = json.loads(artifacts.export_report(job_dir, "ko").read_text(encoding="utf-8"))
    assert {key: body[key] for key in stored} == stored    # report() 그대로
    for key in ("replaced", "kept", "raster_blocks_erased", "listing_lines_replaced",
                "kept_reasons", "warning_count", "warnings"):
        assert key in body, key
    assert pdf.headers["X-UOCR-PDF-Replaced"] == str(body["replaced"])   # 같은 빌드
    assert pdf.headers["X-UOCR-PDF-Warnings"] == str(body["warning_count"])

    # 손상된 리포트·번역 갱신 무효화 뒤에는 404 — 낡거나 깨진 리포트를 내지 않는다
    artifacts.export_report(job_dir, "ko").write_text("{깨짐", encoding="utf-8")
    assert client.get(url).status_code == 404
    assert client.get(f"/api/jobs/{jid}/pdf?lang=ko").status_code == 200    # 다시 빌드
    assert client.get(url).status_code == 200
    artifacts.invalidate_language_artifacts(job_dir, "ko")
    assert client.get(url).status_code == 404

    assert client.get(f"/api/jobs/{jid}/pdf/report?lang=zz").status_code == 400
    assert client.get("/api/jobs/j_000000000000/pdf/report?lang=ko").status_code == 404
