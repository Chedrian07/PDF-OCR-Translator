"""PDF 빌더의 실패 계약 — 사용자에게 보여 줄 PdfExportError로만 실패하고, 잡 디렉터리를
되살리지 않는다.

build_translated_pdf는 예전에 fitz.open만 감싸, 페이지 조판·저장 중의 MuPDF·조판 예외가
그대로 새어 /pdf·/page가 문구 없는 500을 내고 예열 스레드가 traceback을 남겼다.
build_dual_pdf는 out.parent.mkdir(parents=True)로 삭제와 겹친 빌드에서 meta.json 없는 잡
디렉터리를 되살렸다(감사 concurrency-6/8).
"""

import pytest

from app.pipeline.pdf_export import PdfExportError, build_dual_pdf, build_translated_pdf
from app.pipeline.pdf_export import build as build_mod

from tests.test_pdf_export import _unit_job


def _leftovers(job_dir):
    return sorted(p.name for p in job_dir.iterdir() if p.name.endswith(".tmp"))


def test_page_failures_become_a_page_numbered_export_error(tmp_path, monkeypatch):
    job_dir = _unit_job(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("MuPDF 내부 오류 흉내")

    monkeypatch.setattr(build_mod, "_process_page", _boom)
    with pytest.raises(PdfExportError, match="1페이지를 번역 PDF로 조판하지 못했습니다") as info:
        build_translated_pdf(job_dir, "ko")
    assert isinstance(info.value.__cause__, RuntimeError)  # 원인은 사슬에 남는다
    assert not (job_dir / "export.ko.pdf").exists()
    assert _leftovers(job_dir) == []


def test_save_failures_become_an_export_error(tmp_path, monkeypatch):
    job_dir = _unit_job(tmp_path)

    def _boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(build_mod, "_restore_space_tounicode", _boom)
    with pytest.raises(PdfExportError, match="번역 PDF를 저장하지 못했습니다"):
        build_translated_pdf(job_dir, "ko")
    assert not (job_dir / "export.ko.pdf").exists()
    assert _leftovers(job_dir) == []


def test_other_unexpected_failures_are_normalized_too(tmp_path, monkeypatch):
    job_dir = _unit_job(tmp_path)

    def _boom(*args, **kwargs):
        raise KeyError("bbox")

    monkeypatch.setattr(build_mod, "_unregistered_layout_pages", _boom)
    with pytest.raises(PdfExportError, match="PDF 처리 중 오류가 났습니다 \\(KeyError\\)"):
        build_translated_pdf(job_dir, "ko")


def test_export_errors_pass_through_unchanged(tmp_path):
    """이미 사용자용 문구인 PdfExportError는 감싸지 않는다(문구가 그대로 409 본문이 된다)."""
    job_dir = _unit_job(tmp_path)
    (job_dir / "layout.ko.json").unlink()
    with pytest.raises(PdfExportError) as info:
        build_translated_pdf(job_dir, "ko")
    assert str(info.value) == "번역 레이아웃이 없습니다 — 먼저 번역을 실행하세요"
    assert info.value.__cause__ is None


def test_dual_build_never_creates_the_output_directory(tmp_path):
    job_dir = _unit_job(tmp_path)
    translated = build_translated_pdf(job_dir, "ko")
    gone = tmp_path / "deleted-job"
    with pytest.raises(PdfExportError, match="삭제된 작업"):
        build_dual_pdf(job_dir / "source.pdf", translated.path, gone / "export.ko.dual.pdf")
    assert not gone.exists()

    # 출력 디렉터리가 있으면 그대로 만든다(대조군)
    out = build_dual_pdf(job_dir / "source.pdf", translated.path, job_dir / "export.ko.dual.pdf")
    assert out.is_file()
