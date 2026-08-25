"""잡 디렉터리 산출물 **이름의 단일 소유자**.

예전에는 같은 경로 리터럴(`export.{lang}.pdf`, `rendered/{lang}/.source.json` …)이
라우트 계층 곳곳에 흩어져 있었다. 한 곳에서 이름을 바꾸고 다른 곳을 빠뜨리면
"캐시 무효화는 했는데 아무것도 안 지워지는" 조용한 결함이 된다 — 이름을 여기로
모아 무효화 전략과 생성 전략이 반드시 같은 문자열을 보게 한다.

모든 함수는 `job_dir`(잡 루트)를 받아 Path를 돌려주는 순수 함수다. 파일 존재
여부는 확인하지 않는다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

__all__ = [
    "archive",
    "archive_tmp",
    "export_dual_pdf",
    "export_font_marker",
    "export_font_marker_tmp",
    "export_pdf",
    "export_report",
    "facsimile_marker",
    "facsimile_marker_tmp",
    "facsimile_staging",
    "facsimile_staging_glob",
    "figure_boxes",
    "images_dir",
    "invalidate_language_artifacts",
    "layout",
    "layout_tmp",
    "markdown",
    "meta",
    "page_image",
    "pages_dir",
    "rendered_dir",
    "rendered_root",
    "source_pdf",
    "translate_dir",
    "translate_report",
    "translate_state",
    "translate_state_tmp",
]


# ── 입력 ──────────────────────────────────────────────────────────────────
def source_pdf(job_dir: Path) -> Path:
    """업로드된 원본 PDF."""
    return job_dir / "source.pdf"


def meta(job_dir: Path) -> Path:
    """변환 메타(엔진/모델/경고)."""
    return job_dir / "meta.json"


# ── OCR/번역 본문 ─────────────────────────────────────────────────────────
def markdown(job_dir: Path, lang: str | None = None) -> Path:
    """lang=None이면 원본 result.md, lang이면 번역본 result.{lang}.md."""
    return job_dir / (f"result.{lang}.md" if lang else "result.md")


def layout(job_dir: Path, lang: str | None = None) -> Path:
    """lang=None이면 원본 layout.json, lang이면 번역본 layout.{lang}.json."""
    return job_dir / (f"layout.{lang}.json" if lang else "layout.json")


def layout_tmp(job_dir: Path) -> Path:
    """layout 백필용 요청별 고유 tmp — 동시 백필이 같은 tmp에 겹쳐 쓰는 레이스 차단.
    (병합 워커의 .layout.json.tmp와도 이름이 겹치지 않는다.)"""
    return job_dir / f".layout.{uuid.uuid4().hex}.tmp"


# ── 페이지 이미지·그림 ────────────────────────────────────────────────────
def pages_dir(job_dir: Path) -> Path:
    """OCR 입력에 쓰인 원본 페이지 PNG 디렉터리."""
    return job_dir / "pages"


def images_dir(job_dir: Path) -> Path:
    """추출된 그림 파일 디렉터리."""
    return job_dir / "images"


def figure_boxes(job_dir: Path) -> Path:
    """벤더 P13 → merge가 통합한 images/boxes.json."""
    return images_dir(job_dir) / "boxes.json"


def page_image(directory: Path, page_number: int) -> Path:
    """페이지 PNG 파일명 규약 — pages/와 rendered/{lang}/이 같은 규약을 쓴다."""
    return directory / f"page_{page_number:04d}.png"


# ── 번역 PDF에서 만든 facsimile 래스터 ────────────────────────────────────
def rendered_root(job_dir: Path) -> Path:
    return job_dir / "rendered"


def rendered_dir(job_dir: Path, lang: str) -> Path:
    """번역 PDF를 job.dpi로 렌더한 페이지 PNG 디렉터리."""
    return rendered_root(job_dir) / lang


def facsimile_marker(job_dir: Path, lang: str) -> Path:
    """rendered/{lang}/의 세대 표식 — PDF 크기·mtime·DPI·페이지 수를 담는다."""
    return rendered_dir(job_dir, lang) / ".source.json"


def facsimile_marker_tmp(job_dir: Path, lang: str) -> Path:
    return rendered_dir(job_dir, lang) / f".source.{uuid.uuid4().hex}.tmp"


def facsimile_staging(job_dir: Path, lang: str) -> Path:
    """원자적 교체용 staging 디렉터리 — 렌더가 끝난 뒤 os.replace로 갈아끼운다."""
    return rendered_root(job_dir) / f".{lang}.{uuid.uuid4().hex}.tmp"


def facsimile_staging_glob(lang: str) -> str:
    """rendered_root 안에서 staging 잔해를 찾는 glob 패턴."""
    return f".{lang}.*.tmp"


# ── PDF 내보내기 ──────────────────────────────────────────────────────────
def export_pdf(job_dir: Path, lang: str) -> Path:
    """레이아웃 보존 번역 PDF."""
    return job_dir / f"export.{lang}.pdf"


def export_dual_pdf(job_dir: Path, lang: str) -> Path:
    """원본·번역 좌우 대조 PDF."""
    return job_dir / f"export.{lang}.dual.pdf"


def export_report(job_dir: Path, lang: str) -> Path:
    """내보내기 리포트(치환/보존 수, format_version) — 파이프라인이 쓴다."""
    return job_dir / f"export.{lang}.report.json"


def export_font_marker(job_dir: Path, lang: str) -> Path:
    """어떤 폰트 설정으로 만든 PDF인지 남기는 표식(리포트는 파이프라인 소유)."""
    return job_dir / f"export.{lang}.font.txt"


def export_font_marker_tmp(job_dir: Path, lang: str) -> Path:
    return job_dir / f".export.{lang}.font.{uuid.uuid4().hex}.tmp"


# ── 아카이브 ──────────────────────────────────────────────────────────────
def archive(job_dir: Path) -> Path:
    return job_dir / "archive.zip"


def archive_tmp(job_dir: Path) -> Path:
    """요청별 고유 tmp — 동시 요청 둘이 같은 tmp에 겹쳐 써 손상 zip이 캐시되는
    레이스 차단(sync 핸들러는 스레드풀 병렬)."""
    return job_dir / f".archive.{uuid.uuid4().hex}.tmp"


# ── 번역 진행 상태 ────────────────────────────────────────────────────────
def translate_dir(job_dir: Path, lang: str) -> Path:
    return job_dir / "translations" / lang


def translate_state(job_dir: Path, lang: str) -> Path:
    return translate_dir(job_dir, lang) / "state.json"


def translate_state_tmp(job_dir: Path, lang: str) -> Path:
    return translate_dir(job_dir, lang) / ".state.json.tmp"


def translate_report(job_dir: Path, lang: str) -> Path:
    return translate_dir(job_dir, lang) / "report.json"


# ── 무효화 ────────────────────────────────────────────────────────────────
def invalidate_language_artifacts(job_dir: Path, lang: str) -> None:
    """한 언어의 번역이 갱신됐을 때 버려야 하는 파생 산출물을 **한 곳에서** 지운다.

    번역 PDF·대조 PDF·리포트는 번역 본문에서 파생되고, rendered/{lang}/.source.json
    은 그 PDF에서 파생된 HTML 기준면 캐시다. 표식을 지우면 다음 요청이 다시 렌더한다.
    (폰트 표식 export.{lang}.font.txt는 PDF와 함께 재기록되므로 남겨 둔다.)
    """
    export_pdf(job_dir, lang).unlink(missing_ok=True)
    export_report(job_dir, lang).unlink(missing_ok=True)
    export_dual_pdf(job_dir, lang).unlink(missing_ok=True)
    facsimile_marker(job_dir, lang).unlink(missing_ok=True)
