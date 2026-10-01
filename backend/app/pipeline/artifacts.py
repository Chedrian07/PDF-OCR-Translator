"""잡 디렉터리 산출물 **이름의 단일 소유자**.

예전에는 같은 경로 리터럴(`export.{lang}.pdf`, `rendered/{lang}/.source.json` …)이
라우트 계층 곳곳에 흩어져 있었다. 한 곳에서 이름을 바꾸고 다른 곳을 빠뜨리면
"캐시 무효화는 했는데 아무것도 안 지워지는" 조용한 결함이 된다 — 이름을 여기로
모아 무효화 전략과 생성 전략이 반드시 같은 문자열을 보게 한다.

모든 함수는 `job_dir`(잡 루트)를 받아 Path를 돌려주는 순수 함수다. 파일 존재
여부는 확인하지 않는다. 예외는 `has_usable_layout` 하나다 — "이 layout을 좌표
내보내기에 쓸 수 있는가"는 파일 이름이 아니라 **내용**으로만 판정되므로, 그 판정도
이름과 같은 곳에 둬서 생산자(merge)와 소비자(API·번역)가 같은 기준을 보게 한다.
"""

from __future__ import annotations

import functools
import json
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
    "has_usable_layout",
    "images_dir",
    "invalidate_language_artifacts",
    "layout",
    "layout_has_text_blocks",
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


# 본문 텍스트를 싣지 않는 그림 계열 블록 타입 (fidelity._FIGURE_TYPES와 같은 어휘).
_FIGURE_BLOCK_TYPES = frozenset({"image", "chart", "figure", "diagram"})


def layout_has_text_blocks(pages) -> bool:
    """layout 페이지 목록(`[{page, blocks}]`)에 그림이 아닌 블록이 하나라도 있는가.

    좌표 layout의 존재 이유는 텍스트 블록이다 — facsimile HTML·번역 PDF·한국어
    레이아웃은 전부 텍스트 블록의 좌표에 번역을 얹는다. image 블록만 있거나
    blocks가 전부 빈 layout(figure_only 엔진인 OvisOCR2의 옛 잡, 전면 스캔을
    Tesseract로 읽은 textlayer 잡)은 그 소비자들에게 '원문 래스터만 있는 문서'가
    되므로, '레이아웃 없음'과 똑같이 취급해야 텍스트가 사라지지 않는다.
    """
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "").strip().lower() not in _FIGURE_BLOCK_TYPES:
                return True
    return False


@functools.lru_cache(maxsize=1024)
def _usable_layout_file(path: str, mtime_ns: int, size: int) -> bool:
    """(경로, mtime, 크기)별 판정 캐시 — 요청마다 layout 전체를 재파싱하지 않는다.

    잡 목록(GET /api/jobs — 5초 주기 폴링, 한 쪽 최대 500건)도 잡마다 has_layout을 이
    판정으로 낸다. 캐시가 목록보다 작으면 같은 순서로 도는 폴링이 LRU를 매번 전부
    밀어내 모든 layout.json을 다시 파싱한다 — 한 쪽 상한보다 넉넉히 둔다(항목은 수백 B)."""
    try:
        pages = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(pages, list) and layout_has_text_blocks(pages)


def has_usable_layout(job_dir: Path, lang: str | None = None) -> bool:
    """layout(.lang).json이 있고, 좌표 내보내기에 쓸 텍스트 블록을 담고 있는가.

    파일 존재(`layout(...).is_file()`)만 보면 figure_only 엔진의 옛 잡(image 블록만)이
    has_layout으로 보여 document.html이 OCR 텍스트 없는 facsimile이 되고,
    /pdf?lang=ko가 409 대신 번역되지 않은 원문을 낸다. 새 잡은 생산 단계(merge)가
    애초에 그런 layout.json을 쓰지 않지만, 디스크에 남은 옛 잡은 이 판정으로 거른다.
    """
    path = layout(job_dir, lang)
    try:
        st = path.stat()
    except OSError:
        return False
    return _usable_layout_file(str(path), st.st_mtime_ns, st.st_size)


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
    """번역 PDF의 빌드 표식(build stamp) — 이 PDF를 무엇으로 만들었는가.

    내용은 JSON `{"v": 2, "font": <폰트 정체성>, "inputs": {파일명: [ino, size, mtime_ns]}}`다
    (pipeline/derived.py `_write_build_stamp`가 빌드 **직전**의 입력 지문으로 쓰고,
    `_translated_pdf_cache`가 현재 지문·폰트 정체성과 같을 때만 캐시를 최신으로 본다).
    inputs는 source.pdf·layout.json·layout.{lang}.json이다. 파일 이름은 예전 폰트 표식
    (폰트 정체성 문자열 한 줄) 그대로라 옛 형식은 JSON이 아니어서 한 번 재빌드된다.
    리포트(export_report)는 파이프라인이 쓰는 별도 파일이다."""
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
    (빌드 표식 export.{lang}.font.txt는 PDF와 함께 재기록되고, PDF가 없으면 표식만으로는
    캐시가 최신으로 판정되지 않으므로 남겨 둔다.)
    """
    export_pdf(job_dir, lang).unlink(missing_ok=True)
    export_report(job_dir, lang).unlink(missing_ok=True)
    export_dual_pdf(job_dir, lang).unlink(missing_ok=True)
    facsimile_marker(job_dir, lang).unlink(missing_ok=True)
