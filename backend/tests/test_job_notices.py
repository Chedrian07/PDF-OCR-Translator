"""잡 메시지 두 갈래 — warnings(실제 품질 저하)와 notices(정보성 메모).

예전에는 둘이 한 목록(job.warnings)에 섞여, 페이지 단위 엔진 안내('…페이지별로
처리했습니다')·복구에 성공한 재처리 경위('…개선되어 단독 재처리 결과를 채택')까지
quality.state='degraded'를 만들었다 — sidecar 엔진의 모든 잡과 게이트가 고친 잡이
'주의'로 보여 신호가 의미를 잃었다(감사 pipeline-ocr-10, frontend-2).

- runner가 분류한다: 남은 품질 문제(플레이스홀더·텍스트 레이어 복구·충실도 미달 잔존·
  예산 소진·게이트 실패·마커 보정·엔진 정화 경고)는 warnings, 처리 경위는 notices.
- jobs가 meta.json에 notices를 남기고, 'notices' 키가 없는 옛 meta는 고정 문구로 가른다.
- API는 notices(문자열 목록)를 잡 응답에 싣고, quality.state는 warnings로만 정한다.
"""

import json
import os
import threading

from app.config import Settings
from app.engine.fake import FakeEngine
from app.jobs import LEGACY_NOTICE_MARKERS, EventBroker, JobStore, split_legacy_warnings
from app.pipeline.runner import execute_job

from conftest import make_pdf_bytes, wait_done


def _run(tmp_path, engine, *, pages=4, pages_per_chunk=2, **overrides):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    (job.dir / "source.pdf").write_bytes(make_pdf_bytes(pages=pages, with_image=False))
    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data", preload_model=False,
        fake_delay=0.0, pages_per_chunk=pages_per_chunk, **overrides,
    )
    engine.load()
    execute_job(job, store, EventBroker(), engine, settings, threading.Event())
    return job


# ── runner 분류 ↔ 옛 meta 분리 규칙의 일치 ────────────────────────────────


def _scenarios(tmp_path):
    """runner가 실제로 남기는 메시지를 모은다 — 참고와 경고가 골고루 나오는 잡들."""
    from tests.test_fidelity_gate import PageDroppingEngine, PartialDroppingEngine
    from tests.test_fidelity_gate import run_job as run_gate_job
    from tests.test_runner_failures import (
        FlakyEngine,
        LoopFallbackEngine,
        OutputLimitEngine,
        PageUnitEngine,
    )

    jobs = [
        _run(tmp_path / "a", PageUnitEngine(bad_pages={1}), pages=2, pages_per_chunk=8),
        _run(tmp_path / "b", FlakyEngine(fail_calls={1, 2})),
        _run(tmp_path / "c", LoopFallbackEngine(fail_single_pages={2})),
        _run(tmp_path / "d", OutputLimitEngine(complete_pages=2), pages=4, pages_per_chunk=4),
    ]
    gate_job, _events = run_gate_job(
        tmp_path / "e", PageDroppingEngine(drop_pages={3}), pages=4, pages_per_chunk=4,
    )
    breaker_job, _events = run_gate_job(
        tmp_path / "f",
        PartialDroppingEngine(
            partial={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            single_keeps_partial=True, drop_pages={7},
        ),
        pages=8, pages_per_chunk=2, ocr_fidelity_max_retry_ratio=1.0,
    )
    return jobs + [gate_job, breaker_job]


def test_runner_puts_process_notes_in_notices_and_degradation_in_warnings(tmp_path):
    jobs = _scenarios(tmp_path)
    notices = [n for job in jobs for n in job.notices]
    warnings = [w for job in jobs for w in job.warnings]

    def has(messages, text):
        return any(text in m for m in messages)

    # 처리 경위 = 참고
    assert has(notices, "페이지 단위 모델이라 문서를 페이지별로 처리했습니다")
    assert has(notices, "청크 변환 실패로 페이지별 재처리")
    assert has(notices, "반복/출력 상한 감지로 페이지별 재처리")
    assert has(notices, "끝까지 생성된 앞 2쪽은 유지")
    assert has(notices, "개선되어 단독 재처리 결과를 채택")
    assert has(notices, "측정 한계로 판단해 원래 결과 유지")
    assert has(notices, "재처리를 멈춥니다")
    # 남은 품질 문제 = 경고
    assert has(warnings, "텍스트 레이어로 복구")
    for text in ("페이지별", "채택", "측정 한계"):
        assert not has(warnings, text), (text, warnings)

    # 복구에 성공한 잡은 경고가 없다 — quality.state가 'ok'로 남는다
    flaky = jobs[1]
    assert flaky.status == "done" and flaky.warnings == [] and flaky.notices


def test_legacy_split_reproduces_the_runner_classification(tmp_path):
    """notices 이전 meta는 한 목록이었다 — 고정 문구로 가르면 runner의 분류와 같아야 한다.
    runner 문구를 바꿔 이 대조가 깨지면 LEGACY_NOTICE_MARKERS도 함께 고친다."""
    for job in _scenarios(tmp_path):
        assert split_legacy_warnings(job.warnings + job.notices) == (job.warnings, job.notices), (
            job.warnings, job.notices,
        )
    assert all(isinstance(marker, str) and marker for marker in LEGACY_NOTICE_MARKERS)


def test_engine_notices_reach_the_job_with_the_page_span(tmp_path):
    """엔진이 drain_notices로 넘긴 정보성 메모는 청크 페이지 범위를 붙여 notices로 간다."""

    class NotingEngine(FakeEngine):
        def run_multi(self, image_paths, out_dir, sink, cancel):
            self._pending = ["일시 중단 뒤 재시도해 정상 처리"]
            return super().run_multi(image_paths, out_dir, sink, cancel)

        def drain_notices(self):
            drained, self._pending = getattr(self, "_pending", []), []
            return drained

    job = _run(tmp_path, NotingEngine(delay=0.0), pages=4, pages_per_chunk=2)
    assert job.status == "done"
    assert job.notices == [
        "1–2페이지: 일시 중단 뒤 재시도해 정상 처리",
        "3–4페이지: 일시 중단 뒤 재시도해 정상 처리",
    ]
    assert job.warnings == []


# ── jobs: meta 기록·복원 ──────────────────────────────────────────────────


def _write_meta(job_dir, **fields):
    job_dir.mkdir(parents=True)
    (job_dir / "source.pdf").write_bytes(b"%PDF-1.4\n")
    meta = {
        "id": job_dir.name, "filename": "doc.pdf", "mode": "multi", "dpi": 72,
        "status": "done", "created_at": "2026-09-01T00:00:00+00:00",
        "progress": {}, "error": None, **fields,
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


LEGACY_MESSAGES = [
    "ovisocr2 엔진은 페이지 단위 모델이라 문서를 페이지별로 처리했습니다 (결과는 동일하게 하나의 Markdown으로 병합됨)",
    "1–8페이지: 반복/출력 상한 감지로 페이지별 재처리 (모의)",
    "3페이지: 충실도 0.40 → 0.97로 개선되어 단독 재처리 결과를 채택",
    "5페이지: 충실도 0.50 → 0.50 — 측정 한계로 판단해 원래 결과 유지",
    "6페이지: 충실도 0.30 → 0.31 — 개선되지 않아 원래 결과 유지",
    "2페이지: OCR 실패 후 PDF 내장 텍스트 레이어로 복구 (이미지·정밀 레이아웃 제외; RuntimeError: x)",
]


def test_old_meta_without_notices_is_split_in_memory_only(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "j_0123456789ab"
    _write_meta(job_dir, warnings=LEGACY_MESSAGES)
    meta_path = job_dir / "meta.json"
    os.utime(meta_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    before = meta_path.read_bytes()

    store = JobStore(jobs_dir)
    store.load_existing()
    job = store.get("j_0123456789ab")
    assert job.notices == LEGACY_MESSAGES[:4]
    assert job.warnings == LEGACY_MESSAGES[4:]
    # 옛 터미널 잡의 meta는 다시 쓰지 않는다 — mtime은 TTL GC의 '마지막 활동' 시계다
    assert meta_path.read_bytes() == before
    assert meta_path.stat().st_mtime_ns == 1_700_000_000_000_000_000


# 0.1.0(a9a4400) sidecar 엔진이 복구에 성공한 재시작 경위를 warnings에 남긴 문구 — runner가
# '{쪽 범위}: …'를 붙였다. 지금 코드는 같은 상황을 notices(_RECOVERED_NOTICE·_RESUMED_NOTICE)로
# 남긴다. 표식에서 빠져 정상 복구된 옛 잡이 '주의 2건'·degraded로 보였다(migration-3).
SIDECAR_010_NOTES = [
    "1–4페이지: sidecar 재시작/모델 재로드 대기 중… (해당 페이지는 복귀 후 재시도)",
    "1–4페이지: 모델 로딩 대기 중… (최초 기동은 다운로드·컴파일로 수 분 소요)",
]


def test_sidecar_recovery_notes_from_0_1_0_count_as_notices():
    lost = "3페이지: OCR 실패 후 PDF 내장 텍스트 레이어로 복구 (이미지·정밀 레이아웃 제외; x)"
    degraded = "1–4페이지: sidecar가 이상 상태를 보고했습니다(x) — 모델은 로드돼 있어 그대로 진행합니다."
    warnings, notices = split_legacy_warnings([*SIDECAR_010_NOTES, lost, degraded])
    assert notices == SIDECAR_010_NOTES
    assert warnings == [lost, degraded]          # 실제 손실·이상 신고는 여전히 경고다


def test_pinning_the_separator_keeps_the_legacy_warnings_list(tmp_path):
    """앱 기동(load_existing(default_page_separator=…))은 page_separator가 없는 옛 meta에 지금
    값을 고정해 다시 쓴다. 예전에는 그때 분리 결과(notices 키)까지 써 넣어, 나중에 표식을 고쳐도
    이미 이식된 잡은 다시 가르지 못했다(migration-3). 원래의 합쳐진 warnings를 그대로 두고
    notices 키는 쓰지 않는다 — 기동마다 지금 표식으로 다시 가른다(mtime 보존)."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "j_0123456789ab"
    combined = [*LEGACY_MESSAGES, *SIDECAR_010_NOTES]
    _write_meta(job_dir, warnings=combined)
    meta_path = job_dir / "meta.json"
    os.utime(meta_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    store = JobStore(jobs_dir)
    store.load_existing(default_page_separator="\n\n---\n\n")
    job = store.get("j_0123456789ab")
    assert job.page_separator == "\n\n---\n\n"
    assert job.notices == LEGACY_MESSAGES[:4] + SIDECAR_010_NOTES
    assert job.warnings == LEGACY_MESSAGES[4:]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["page_separator"] == "\n\n---\n\n"       # 이식은 기록된다
    assert meta["warnings"] == combined and "notices" not in meta   # 분리는 기록하지 않는다
    assert meta_path.stat().st_mtime_ns == 1_700_000_000_000_000_000

    again = JobStore(jobs_dir)                               # 다음 기동도 같은 분류
    again.load_existing(default_page_separator="\n\n---\n\n")
    restored = again.get("j_0123456789ab")
    assert (restored.warnings, restored.notices) == (job.warnings, job.notices)


def test_meta_with_notices_is_restored_verbatim(tmp_path):
    """notices 키가 있으면 기록된 분류를 그대로 믿는다(문구로 다시 가르지 않는다)."""
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "j_aaaaaaaaaaaa"
    warnings = ["1–2페이지: 청크 변환 실패로 페이지별 재처리 — 운영자가 경고로 남긴 사례"]
    _write_meta(job_dir, warnings=warnings, notices=["참고 1"])
    store = JobStore(jobs_dir)
    store.load_existing()
    job = store.get("j_aaaaaaaaaaaa")
    assert job.warnings == warnings and job.notices == ["참고 1"]


def test_notices_survive_a_save_and_restart(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create("doc.pdf", "multi", dpi=72)
    job.status = "done"
    job.warnings = ["경고 1"]
    job.notices = ["참고 1", "참고 2"]
    store.save(job)
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["warnings"] == ["경고 1"] and meta["notices"] == ["참고 1", "참고 2"]

    again = JobStore(tmp_path / "jobs")
    again.load_existing()
    restored = again.get(job.id)
    assert restored.warnings == ["경고 1"] and restored.notices == ["참고 1", "참고 2"]


# ── API ──────────────────────────────────────────────────────────────────


def test_api_serializes_notices_and_bases_quality_on_warnings_only(client, settings, sample_pdf):
    jid = client.post(
        "/api/jobs", files={"file": ("a.pdf", sample_pdf, "application/pdf")},
    ).json()["job_id"]
    body = wait_done(client, jid)
    assert body["status"] == "done"
    assert body["warnings"] == [] and body["notices"] == []
    listed = client.get("/api/jobs").json()["jobs"]
    assert listed[0]["notices"] == []

    st = client.app.state
    job = st.store.get(jid)
    job.notices = ["1–3페이지: 반복/출력 상한 감지로 페이지별 재처리 (모의)"]
    manifest = client.get(f"/api/jobs/{jid}/viewer-manifest").json()
    assert manifest["quality"]["state"] == "ok"           # 참고만 있으면 품질 정상
    assert manifest["quality"]["warning_count"] == 0
    assert manifest["quality"]["notice_count"] == 1
    assert client.get(f"/api/jobs/{jid}").json()["notices"] == job.notices

    job.warnings = ["2페이지: OCR 실패 후 PDF 내장 텍스트 레이어로 복구"]
    st.store.save(job)  # meta가 바뀌어야 ETag(artifact_revision)가 바뀐다
    manifest = client.get(f"/api/jobs/{jid}/viewer-manifest").json()
    assert manifest["quality"]["state"] == "degraded"
    assert manifest["quality"]["warnings"] == job.warnings
