"""PDF 워커 풀 계약 — 별도 프로세스 실행·시간 상한·비정상 종료 격리·취소·종료.

MuPDF 작업을 서버 프로세스 밖으로 옮긴 이유(감사 A11: security-2·concurrency-2/3·
pdf-export-9·gap1-metal-real-e2e-2)가 실제로 성립하는지 실제 spawn 워커로 확인한다.
보조 작업은 tests/pdf_worker_tasks.py에 있다(워커가 'pdf_worker_tasks:함수'로 임포트).
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.pipeline import pdf_worker
from app.pipeline.pdf_worker import (
    PdfPageQuarantined,
    PdfWorkerBusy,
    PdfWorkerCanceled,
    PdfWorkerCrashed,
    PdfWorkerRemoteError,
    PdfWorkerTimeout,
)


@dataclass(frozen=True)
class _Payload:
    name: str
    path: Path
    data: bytes


def _pdf_children() -> list:
    return [p for p in multiprocessing.active_children() if p.name.startswith("pdf-")]


def _wait_no_children(timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left = _pdf_children()
        if not left:
            return []
        time.sleep(0.05)
    return _pdf_children()


# ── inline 모드(테스트 세션 기본) ─────────────────────────────────────────


def test_inline_mode_runs_in_the_calling_process():
    assert pdf_worker.mode() == pdf_worker.MODE_INLINE
    assert pdf_worker.run("pdf_worker_tasks:pid", timeout=5) == os.getpid()
    assert pdf_worker.run("pdf_worker_tasks:in_worker", timeout=5) is False
    # 작업이 자기 루프에서 취소를 확인하도록 콜백을 키워드로 넘긴다(프로세스 모드에서는
    # 부모가 워커를 끝내 같은 효과를 낸다)
    assert pdf_worker.run(
        "pdf_worker_tasks:sees_cancel", timeout=5, cancel=lambda: True,
        cancel_kwarg="should_cancel",
    ) is True
    assert pdf_worker.pool_stats()["pools"] == {}  # 프로세스를 하나도 띄우지 않았다


def test_mode_env_typo_falls_back_to_process(monkeypatch, caplog):
    monkeypatch.setenv("PDF_WORKER_MODE", "inlnie")
    assert pdf_worker.mode() == pdf_worker.MODE_PROCESS


def test_timeouts_and_limits_follow_env(monkeypatch):
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "12.5")
    monkeypatch.setenv("PDF_EXPORT_BUILD_TIMEOUT_S", "0")
    monkeypatch.setenv("PDF_MAX_PAGE_CONTENT_MB", "8")
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "abc")
    assert pdf_worker.page_timeout() == 12.5
    assert pdf_worker.export_build_timeout() is None  # 0 이하 = 상한 없음
    assert pdf_worker.max_page_content_bytes() == 8 * 1024 * 1024
    assert pdf_worker.max_page_xobject_calls() == 2_000_000  # 오타는 기본값으로


# ── process 모드 ──────────────────────────────────────────────────────────


def test_tasks_run_in_a_persistent_child_process(pdf_worker_processes):
    first = pdf_worker.run("pdf_worker_tasks:pid", timeout=30)
    second = pdf_worker.run("pdf_worker_tasks:pid", timeout=30)
    assert first == second != os.getpid()  # 같은 상주 워커를 재사용한다
    assert pdf_worker.run("pdf_worker_tasks:in_worker", timeout=30) is True
    stats = pdf_worker.pool_stats()
    assert stats["mode"] == "process"
    ocr = stats["pools"]["ocr"]
    assert ocr["spawned"] == 1 and ocr["tasks"] == 3 and ocr["workers"] == 1


def test_arguments_results_and_exceptions_round_trip(pdf_worker_processes, tmp_path):
    payload = _Payload("한글", tmp_path / "a.pdf", b"\x00\xff" * 10)
    assert pdf_worker.run("pdf_worker_tasks:echo", (payload,), timeout=30) == payload

    with pytest.raises(ValueError, match="사용자 메시지") as info:
        pdf_worker.run("pdf_worker_tasks:raise_value_error", ("사용자 메시지",), timeout=30)
    # 워커 쪽 traceback이 원인 사슬로 붙는다(서버 로그에서 추적 가능)
    assert "raise_value_error" in str(info.value.__cause__)

    with pytest.raises(PdfWorkerRemoteError, match="_NeedsTwoArgs: left/right"):
        pdf_worker.run("pdf_worker_tasks:raise_unpicklable", timeout=30)
    with pytest.raises(PdfWorkerRemoteError, match="부모 프로세스로 보낼 수 없습니다"):
        pdf_worker.run("pdf_worker_tasks:return_unpicklable", timeout=30)

    # 작업 예외는 워커를 죽이지 않는다 — 같은 워커가 계속 쓰인다
    ocr = pdf_worker.pool_stats()["pools"]["ocr"]
    assert ocr["spawned"] == 1 and ocr["crashes"] == 0


def test_timeout_kills_the_worker_and_the_next_call_respawns(pdf_worker_processes):
    before = pdf_worker.run("pdf_worker_tasks:pid", timeout=30)
    started = time.monotonic()
    with pytest.raises(PdfWorkerTimeout):
        pdf_worker.run("pdf_worker_tasks:sleep", (30,), timeout=0.5)
    assert time.monotonic() - started < 5  # 30초 작업을 끝까지 기다리지 않는다
    after = pdf_worker.run("pdf_worker_tasks:pid", timeout=30)
    assert after != before
    ocr = pdf_worker.pool_stats()["pools"]["ocr"]
    assert ocr["timeouts"] == 1 and ocr["spawned"] == 2


@pytest.mark.parametrize(
    ("target", "args", "exitcode"),
    [("pdf_worker_tasks:exit_now", (3,), 3), ("pdf_worker_tasks:segfault", (), -11)],
)
def test_crash_is_isolated_to_the_worker(pdf_worker_processes, target, args, exitcode):
    with pytest.raises(PdfWorkerCrashed) as info:
        pdf_worker.run(target, args, timeout=30)
    assert info.value.exitcode == exitcode
    # 서버(이 프로세스)는 멀쩡하고 다음 작업은 새 워커가 처리한다
    assert pdf_worker.run("pdf_worker_tasks:echo", ("ok",), timeout=30) == "ok"
    assert pdf_worker.pool_stats()["pools"]["ocr"]["crashes"] == 1


def test_cancel_stops_a_running_task_promptly(pdf_worker_processes):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(PdfWorkerCanceled):
        pdf_worker.run("pdf_worker_tasks:sleep", (30,), timeout=60, cancel=cancel.is_set)
    assert time.monotonic() - started < 3
    assert pdf_worker.run("pdf_worker_tasks:echo", (1,), timeout=30) == 1


def test_pools_are_separate_and_run_in_parallel(pdf_worker_processes, monkeypatch):
    monkeypatch.setenv("PDF_EXPORT_MAX_CONCURRENT", "2")
    pdf_worker.run("pdf_worker_tasks:pid", pool="export", timeout=30)  # 워커 기동 비용 제외
    pdf_worker.run("pdf_worker_tasks:pid", pool="ocr", timeout=30)
    results: list[int] = []

    def _job(pool: str) -> None:
        results.append(pdf_worker.run("pdf_worker_tasks:sleep", (1.0,), pool=pool, timeout=30))

    threads = [threading.Thread(target=_job, args=(p,)) for p in ("export", "export", "ocr")]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started
    assert len(set(results)) == 3  # 세 작업이 세 프로세스에서
    assert elapsed < 2.5, elapsed  # 직렬이면 3초 — 서로 기다리지 않았다
    pools = pdf_worker.pool_stats()["pools"]
    assert pools["export"]["workers"] == 2 and pools["ocr"]["workers"] == 1


def test_a_full_pool_makes_callers_wait_or_reports_busy(pdf_worker_processes):
    holder = threading.Thread(
        target=pdf_worker.run, args=("pdf_worker_tasks:sleep", (1.5,)),
        kwargs={"pool": "ocr", "timeout": 30},
    )
    holder.start()
    time.sleep(0.3)
    with pytest.raises(PdfWorkerBusy):
        pdf_worker.run("pdf_worker_tasks:pid", pool="ocr", timeout=30, wait=0.2)
    # 기다리면(대기 상한 없음) 앞 작업이 끝난 뒤 같은 워커로 처리된다
    assert pdf_worker.run("pdf_worker_tasks:echo", (7,), pool="ocr", timeout=30) == 7
    holder.join()
    stats = pdf_worker.pool_stats()["pools"]["ocr"]
    assert stats["in_use"] == 0 and stats["rejected_busy"] == 1


@pytest.mark.parametrize("settle", [0.5, 0.05, 0.2])
def test_shutdown_stops_idle_and_busy_workers(pdf_worker_processes, settle):
    """작업 중인 워커를 다른 스레드가 종료해도 호출자는 PdfWorkerCrashed만 본다 — 정리가 두
    스레드에서 겹쳐도(Process.close 이중 호출 → AttributeError) 새지 않는다. 시점을 바꿔 반복."""
    pdf_worker.run("pdf_worker_tasks:pid", pool="probe", timeout=30)
    errors: list[BaseException] = []

    def _long() -> None:
        try:
            pdf_worker.run("pdf_worker_tasks:sleep", (30,), pool="ocr", timeout=60)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=_long)
    thread.start()
    time.sleep(settle)
    started = time.monotonic()
    pdf_worker.shutdown_pools()
    thread.join(10)
    assert not thread.is_alive()
    assert time.monotonic() - started < 5
    assert len(errors) == 1 and isinstance(errors[0], (PdfWorkerCrashed, pdf_worker.PdfWorkerError))
    assert _wait_no_children() == []
    # 닫은 뒤 다시 쓰면 새 풀이 lazily 만들어진다
    assert pdf_worker.run("pdf_worker_tasks:echo", ("again",), timeout=30) == "again"


def test_workers_are_recycled_after_many_tasks(pdf_worker_processes, monkeypatch):
    monkeypatch.setattr(pdf_worker, "_MAX_TASKS_PER_WORKER", 2)
    pids = [pdf_worker.run("pdf_worker_tasks:pid", timeout=30) for _ in range(4)]
    assert pids[0] == pids[1] != pids[2] == pids[3]
    assert pdf_worker.pool_stats()["pools"]["ocr"]["recycled"] == 2


def test_peak_rss_is_the_workers_own_high_water_mark(monkeypatch, tmp_path):
    """Linux의 ru_maxrss는 fork·exec를 넘어 부모의 최댓값을 물려받는다 — 모델을 올린 서버(수 GB)가
    띄운 워커가 첫 작업부터 1.5GB 상한을 넘은 것으로 보여 작업마다 폐기·재생성됐다(감사
    infra-docs-1). 워커 자신의 VmHWM을 읽어야 한다."""
    import resource
    from types import SimpleNamespace

    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython3\nVmPeak:\t  999999 kB\nVmHWM:\t   14336 kB\nVmRSS:\t   12000 kB\n",
        encoding="ascii",
    )
    inherited = 13 * 1024 * 1024  # 13GB(Linux ru_maxrss 단위 KB) — 부모에게서 물려받은 값
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=inherited))
    monkeypatch.setattr(pdf_worker, "_PROC_STATUS", str(status))
    assert pdf_worker._peak_rss_bytes() == 14336 * 1024
    # procfs가 없으면 ru_maxrss — 기동 시점 값(물려받은 몫)을 넘지 않았으면 이 워커의 값이 아니다
    monkeypatch.setattr(pdf_worker, "_PROC_STATUS", str(tmp_path / "missing"))
    monkeypatch.setattr(pdf_worker, "_START_MAXRSS", pdf_worker._ru_maxrss_bytes())
    assert pdf_worker._peak_rss_bytes() == 0
    monkeypatch.setattr(pdf_worker, "_START_MAXRSS", 1024)
    assert pdf_worker._peak_rss_bytes() == pdf_worker._ru_maxrss_bytes() > 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ru_maxrss 상속은 Linux 커널 동작")
def test_workers_spawned_by_a_big_parent_are_reused(pdf_worker_processes, monkeypatch):
    """부모의 최대 RSS가 교체 기준을 넘어도 워커가 작업마다 교체되지 않는다(1.5GB를 할당하지
    않고 기준을 부모 최댓값 바로 아래로 내려 같은 상황을 만든다)."""
    import resource

    blob = b"\x01" * (256 << 20)  # 부모 최댓값을 워커의 실제 사용량보다 확실히 크게
    del blob
    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    monkeypatch.setattr(pdf_worker, "_RECYCLE_RSS_BYTES", parent_peak - (16 << 20))
    pids = [pdf_worker.run("pdf_worker_tasks:pid", timeout=30) for _ in range(3)]
    assert len(set(pids)) == 1
    ocr = pdf_worker.pool_stats()["pools"]["ocr"]
    assert ocr["spawned"] == 1 and ocr["recycled"] == 0


def test_each_task_arms_a_self_destruct_alarm(pdf_worker_processes):
    """부모가 SIGKILL로 사라져도 적대적 작업이 영원히 돌지 않게 — 커널이 끝내는 SIGALRM."""
    remaining = pdf_worker.run("pdf_worker_tasks:alarm_remaining", timeout=5)
    assert 5 < remaining <= 5 + pdf_worker._ALARM_GRACE_S
    # 작업이 끝나면 해제된다(쉬는 워커가 자폭하지 않는다)
    assert pdf_worker.run("pdf_worker_tasks:alarm_remaining", timeout=None) == 0


def test_linux_only_memory_guards(pdf_worker_processes, monkeypatch):
    monkeypatch.setenv("PDF_WORKER_MEM_LIMIT_MB", "3072")
    soft, _hard = pdf_worker.run("pdf_worker_tasks:rlimit_as", timeout=30)
    adj = pdf_worker.run("pdf_worker_tasks:oom_score_adj", timeout=30)
    if sys.platform.startswith("linux"):
        assert soft == 3072 * 1024 * 1024
        assert adj == "1000"  # 메모리 압박 시 서버보다 워커가 먼저 정리된다
    else:  # macOS는 RLIMIT_AS를 강제하지 않는다 — 걸지 않는다
        import resource

        assert soft == resource.RLIM_INFINITY
        assert adj is None


def test_worker_reuses_the_open_document_between_page_tasks(pdf_worker_processes, tmp_path):
    from conftest import make_pdf_bytes

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf_bytes(pages=2, with_image=False))
    first = pdf_worker.run("pdf_worker_tasks:cached_document_identity", (pdf,), timeout=30)
    second = pdf_worker.run("pdf_worker_tasks:cached_document_identity", (pdf,), timeout=30)
    assert first == second and first[1] == 2
    # 파일이 바뀌면(새 inode·크기) 캐시를 버리고 다시 연다
    replacement = tmp_path / "new.pdf"
    replacement.write_bytes(make_pdf_bytes(pages=3, with_image=False))
    os.replace(replacement, pdf)
    _doc_id, pages = pdf_worker.run(
        "pdf_worker_tasks:cached_document_identity", (pdf,), timeout=30,
    )
    assert pages == 3


def test_a_page_that_timed_out_is_not_retried(pdf_worker_processes, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(PdfWorkerTimeout):
        pdf_worker.run_page("pdf_worker_tasks:sleep_page", pdf, 2, timeout=0.3)
    spawned = pdf_worker.pool_stats()["pools"]["ocr"]["spawned"]
    started = time.monotonic()
    with pytest.raises(PdfPageQuarantined) as info:
        pdf_worker.run_page("pdf_worker_tasks:sleep_page", pdf, 2, timeout=30)
    assert time.monotonic() - started < 0.5
    assert isinstance(info.value, PdfWorkerTimeout)  # 호출부는 시간 초과와 같게 다룬다
    assert pdf_worker.pool_stats()["pools"]["ocr"]["spawned"] == spawned  # 새로 띄우지 않았다
    # 다른 페이지·바뀐 파일은 영향이 없다
    assert pdf_worker.is_quarantined(pdf, 2)
    assert not pdf_worker.is_quarantined(pdf, 1)
    pdf.write_bytes(b"%PDF-1.4\n% changed\n")
    assert not pdf_worker.is_quarantined(pdf, 2)


def test_child_import_graph_stays_light(pdf_worker_processes, tmp_path):
    """워커는 PDF 작업 모듈만 싣는다 — torch·mlx·앱 진입점·설정/LLM 계층은 없어야 한다."""
    from conftest import make_pdf_bytes

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf_bytes(pages=1))
    pdf_worker.run("pdf_worker_tasks:cached_document_identity", (pdf,), timeout=30)
    modules = set(pdf_worker.run("pdf_worker_tasks:import_task_modules", timeout=60))
    assert "pymupdf" in modules  # 실제로 MuPDF를 실었다
    forbidden = {
        "torch", "mlx", "transformers", "fastapi", "starlette", "httpx", "anyio",
        "app.main", "app.api", "app.jobs", "app.config", "app.llm", "app.engine.registry",
        "app.translate",
    }
    assert not forbidden & modules, sorted(forbidden & modules)


def test_exceptions_carry_no_unpicklable_state():
    """부모로 돌아오는 상한·종료 예외 자체도 피클 가능해야 한다(다른 계층이 다시 옮길 수 있게)."""
    for error in (
        PdfWorkerTimeout(3.0), PdfPageQuarantined(4), PdfWorkerCrashed(-11),
        PdfWorkerCanceled(), PdfWorkerBusy("export", 1.0), PdfWorkerRemoteError("X", "y"),
    ):
        assert str(pickle.loads(pickle.dumps(error)))


def test_temp_files_of_a_killed_worker_are_removed(pdf_worker_processes, tmp_path):
    """상한 초과로 종료된 워커는 finally를 못 돌린다 — 워커 전용 임시 디렉터리를 부모가 지운다."""
    report = tmp_path / "made.txt"
    with pytest.raises(PdfWorkerTimeout):
        pdf_worker.run("pdf_worker_tasks:temp_dir_then_sleep", (report,), timeout=1.0)
    made = Path(report.read_text(encoding="utf-8"))
    assert made.name.startswith("uocr-font-")
    assert made.parent.name.startswith("pdfocr-worker-ocr-")  # 워커 전용 디렉터리 안
    assert not made.parent.exists()


def test_orphan_worker_temp_dirs_are_swept_once(monkeypatch, tmp_path):
    import subprocess
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(pdf_worker, "_SWEPT_SCRATCH", False)
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    orphan = tmp_path / f"pdfocr-worker-export-{finished.pid}"
    alive = tmp_path / f"pdfocr-worker-export-{os.getpid()}"
    for path in (orphan, alive):
        (path / "font").mkdir(parents=True)
    pdf_worker._sweep_orphan_scratch()
    assert not orphan.exists()  # 주인이 없는(이전 서버가 SIGKILL로 남긴) 디렉터리
    assert alive.exists()  # 살아 있는 프로세스의 것은 건드리지 않는다


def test_workers_do_not_rerun_the_parents_main_script(tmp_path):
    """spawn은 기본적으로 자식에서 부모의 메인 스크립트를 다시 실행한다 — uvicorn 콘솔 스크립트면
    워커마다 uvicorn·click·watchfiles·anyio를 싣고, 가드 없는 스크립트면 자식이 부팅 중 다시
    워커를 띄우려다 죽는다. 워커는 메인 모듈을 건드리지 않아야 한다."""
    import subprocess

    backend = Path(__file__).resolve().parents[1]
    marker = tmp_path / "main-runs.txt"
    script = tmp_path / "no_guard.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(backend)!r})\n"
        f"open({str(marker)!r}, 'a').write(f'{{os.getpid()}}\\n')\n"
        "from app.pipeline import pdf_worker\n"
        "print(pdf_worker.run('os:getpid', timeout=30) != os.getpid())\n"
        "pdf_worker.shutdown_pools()\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PDF_WORKER_MODE": "process"}
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "True"
    assert len(marker.read_text().split()) == 1  # 메인 스크립트는 부모에서 한 번만 돌았다


def test_workers_do_not_see_credentials(pdf_worker_processes, monkeypatch):
    """워커는 비밀이 필요 없다 — MuPDF 결함으로 워커가 장악돼도 API 키·토큰을 읽지 못하게 지운다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-for-workers")
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "sk-test-not-for-workers")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("SOME_DB_PASSWORD", "x")
    monkeypatch.setenv("PDF_PAGE_TIMEOUT_S", "45")
    names = set(pdf_worker.run("pdf_worker_tasks:env_names", timeout=30))
    assert not {"OPENAI_API_KEY", "LLM_OPENAI_API_KEY", "HF_TOKEN", "SOME_DB_PASSWORD"} & names
    assert {"PDF_PAGE_TIMEOUT_S", "PATH"} <= names  # 실행 환경과 노브는 그대로
