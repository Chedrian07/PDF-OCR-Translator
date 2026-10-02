"""빌드 도중 입력이 바뀐 번역 PDF의 재예열 — facsimile 경로(잡 락 중첩)에서도 실제로 돈다.

`_ensure_facsimile_pages`는 잡 락(RLock)을 쥔 채 `_ensure_translated_pdf`를 부른다. 그 안에서
빌드 도중 입력이 바뀌면 재예열을 띄우는데, 예열은 대기 0이라 아직 쥐고 있는 바깥 락을 얻지
못하고 즉시 포기했다(감사 pdf-7: 'PDF 예열 건너뜀(경합)', 빌드 1회, 캐시 낡음). 재예열은
가장 바깥 잡 락을 놓은 뒤에 시작해야 한다.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline import derived
from app.pipeline.pdf_export import PDF_EXPORT_FORMAT_VERSION


def _job(tmp_path: Path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    for name in ("source.pdf", "layout.json", "layout.ko.json"):
        (job_dir / name).write_text("[]", encoding="utf-8")
    job = SimpleNamespace(
        id=f"rewarm-{uuid.uuid4().hex[:8]}", dir=job_dir, dpi=72, delete_requested=False,
    )
    return job, SimpleNamespace(pdf_export_font="", max_pages=10)


def _builder(builds: list[str], second_build: threading.Event):
    """첫 빌드 도중 번역 레이아웃이 바뀐다(재번역 완료·폰트 백필이 새 파일로 교체)."""

    def build(job_dir, lang, *, fontfile=""):
        builds.append(threading.current_thread().name)
        if len(builds) == 1:
            replacement = job_dir / ".layout.tmp"
            replacement.write_text("[1]", encoding="utf-8")
            os.replace(replacement, job_dir / f"layout.{lang}.json")
        out = job_dir / f"export.{lang}.pdf"
        out.write_bytes(b"%PDF-1.4 build " + str(len(builds)).encode())
        report = {"format_version": PDF_EXPORT_FORMAT_VERSION}
        (job_dir / f"export.{lang}.report.json").write_text(json.dumps(report), encoding="utf-8")
        if len(builds) >= 2:
            second_build.set()
        return SimpleNamespace(path=out, report=lambda: dict(report))

    return build


def _wait_warm_idle(job_id: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not derived._warm_inflight(job_id, "ko"):
            return
        time.sleep(0.02)
    raise AssertionError("예열 스레드가 시간 내에 끝나지 않음")


def test_facsimile_path_rewarms_after_releasing_the_job_lock(tmp_path):
    job, settings = _job(tmp_path)
    builds: list[str] = []
    second_build = threading.Event()

    def render(pdf_path, out_dir, *, dpi, max_pages):
        (Path(out_dir) / "page_0001.png").write_bytes(b"png")

    derived._ensure_facsimile_pages(
        job, [1], "ko", settings, render=render, build=_builder(builds, second_build),
    )

    assert second_build.wait(5), builds          # 재예열이 새 입력으로 실제로 다시 빌드했다
    _wait_warm_idle(job.id)
    assert builds[0] == threading.current_thread().name
    assert builds[1].startswith("pdf-warm-"), builds
    current, _out, _report = derived._translated_pdf_cache(
        job, "ko", derived._pdf_export_font_id(settings),
    )
    assert current                                # 캐시가 바뀐 입력 기준으로 최신이 됐다


def test_rewarm_still_starts_when_the_facsimile_raster_fails(tmp_path):
    """잡 락 안의 래스터가 실패해도 이미 감지한 입력 변경의 재예열은 미루다 버리지 않는다."""
    job, settings = _job(tmp_path)
    builds: list[str] = []
    second_build = threading.Event()

    def render(pdf_path, out_dir, *, dpi, max_pages):
        raise ValueError("래스터 실패")

    with pytest.raises(derived.PdfExportError):
        derived._ensure_facsimile_pages(
            job, [1], "ko", settings, render=render, build=_builder(builds, second_build),
        )

    assert second_build.wait(5), builds
    _wait_warm_idle(job.id)
    assert not derived._held_job_locks()          # 이 스레드에 남은 잡 락 깊이가 없다
