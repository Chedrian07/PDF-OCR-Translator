"""Embedded PDF text/reference helpers; never an OCR output engine."""
from __future__ import annotations
import logging
import math
import re
from pathlib import Path
logger = logging.getLogger(__name__)
_SPECIAL_SPAN = re.compile(r"<\|[^|>]{0,64}\|>")
_SANITIZE_ROUNDS = 8
def sanitize_text(text: str) -> str:
    """문서 추출 텍스트에서 스트림 계약을 깨는 토큰을 제거·치환.

    - ``<|...|>`` 토큰형 스팬: 제거로 새 스팬이 드러날 수 있어(중첩 조작 문서)
      고정점까지 반복하되 **상한을 둔다** — 무상한 고정점 반복은 순수 중첩
      ``<|<|…|>|>``(패스당 최심 스팬 1개만 제거)에서 O(n²)라, 조작 PDF의
      텍스트 레이어 한 장이 단일 워커를 시간 단위로 잠글 수 있다. 상한 안에
      수렴하지 않는 병적 입력은 ``<``를 치환해 스팬·마커 문법의 생성 자체를
      차단한다 — sidecar/protocol.strip_special_tokens와 동일한 상한·최후수단.
    - 리터럴 ``<PAGE>``: 페이지 마커 문법(진행률·병합·반복 감지)의 예약어라
      시각적 등가 문자로 치환한다 — 내용은 보존하되 마커로는 절대 세지 않게.
    """
    out = text
    for _ in range(_SANITIZE_ROUNDS):
        stripped = _SPECIAL_SPAN.sub("", out)
        if stripped == out:
            break
        out = stripped
    if _SPECIAL_SPAN.search(out):
        # 비정상적으로 깊은 중첩 — 정상 추출 텍스트에서는 도달하지 않는 경로
        out = out.replace("<", " ")
    return out.replace("<PAGE>", "⟨PAGE⟩")


def _norm_bbox(bbox, rect) -> tuple[int, int, int, int] | None:
    """pt 좌표 bbox → 0–999 정규화 (fake.py/벤더 det 문법과 동일 좌표계).

    페이지 rect 기준으로 클램프하고, 퇴화(폭·높이 0 이하) 박스는 None."""
    width = float(rect.width)
    height = float(rect.height)
    if width <= 0 or height <= 0:
        return None

    def norm(value: float, origin: float, size: float) -> int:
        return max(0, min(999, round((value - origin) / size * 999)))

    x1 = norm(float(bbox[0]), float(rect.x0), width)
    y1 = norm(float(bbox[1]), float(rect.y0), height)
    x2 = norm(float(bbox[2]), float(rect.x0), width)
    y2 = norm(float(bbox[3]), float(rect.y0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _fitz():
    """pipeline.pdf와 동일한 PyMuPDF 지연 임포트 (quiet_fitz 우선 — stderr 억제)."""
    try:
        from ..pipeline.pdf import quiet_fitz
    except ImportError:  # pragma: no cover — 패키지 밖 단독 사용 대비
        import pymupdf as fitz  # 레거시 fitz 임포트는 1.28+에서 stdout 폐지 경고

        return fitz
    return quiet_fitz()


def _drain_mupdf(context: str) -> None:
    """MuPDF 내부 경고 버퍼 요약 로깅 (best-effort — pipeline.pdf와 동일 패턴)."""
    try:
        from ..pipeline.pdf import drain_mupdf_warnings
    except ImportError:  # pragma: no cover
        return
    drain_mupdf_warnings(context)


def native_page_blocks(
    fitz, doc, page_number: int,
) -> list[tuple[tuple[int, int, int, int] | None, str]]:
    """열린 문서의 1-based 페이지 텍스트 레이어 블록 — (0–999 정규화 bbox, 정화된 텍스트).

    블록은 **읽기 순서**다(reading_order.page_text_blocks — 내용 스트림 순서 기준, 다단이면
    띠별로 왼쪽 단 → 오른쪽 단, 한 문장·한 줄 조각은 병합). 좌표는 렌더 PNG와 같은 회전
    반영 공간이다. 범위 밖 페이지는 빈 목록. 추출 예외는 호출자가 다룬다."""
    if not 1 <= page_number <= doc.page_count:
        return []
    from ..pipeline.reading_order import page_text_blocks

    page = doc[page_number - 1]
    rect = page.rect
    blocks: list[tuple[tuple[int, int, int, int] | None, str]] = []
    for block in page_text_blocks(page, fitz):
        text = sanitize_text(block.text).strip()
        if not text:
            continue
        blocks.append((_norm_bbox((block.x0, block.y0, block.x1, block.y1), rect), text))
    return blocks


def native_blocks_local(source_pdf: Path, page_index: int) -> list:
    """(PDF 워커) 원본 PDF 페이지 하나의 텍스트 레이어 블록 — 열 수 없거나 암호화·추출 실패면
    빈 목록(그 페이지는 OCR 경로로 간다)."""
    from ..pipeline import pdf_worker

    try:
        with pdf_worker.open_document(source_pdf) as doc:
            if doc.needs_pass:  # 업로드 probe가 거르지만 방어적으로
                return []
            return native_page_blocks(_fitz(), doc, page_index + 1)
    except Exception as e:  # noqa: BLE001 — 손상 페이지·원본은 OCR 경로로 격리
        logger.warning("%d페이지 텍스트 레이어 추출 실패 (%s: %s)",
                       page_index + 1, e.__class__.__name__, str(e)[:200])
        return []
    finally:
        _drain_mupdf("textlayer 추출")


_MAX_PAGE_FIGURES = 256


def native_page_figures(fitz, page) -> list[tuple[float, float, float, float]]:
    """Displayed raster occurrences in rotated page space, normalized to [0, 1].

    Rendered-page crops preserve masks, transparency and rotation. This is not
    vector figure detection. Full-page scans are represented by the page PNG.
    """
    rect = page.rect
    figures = []
    seen = set()
    for info in page.get_image_info(hashes=False, xrefs=False):
        bbox = info.get("bbox", ())
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            continue
        box = (fitz.Rect(bbox) * page.rotation_matrix) & rect
        if box.is_empty or box.is_infinite or not all(math.isfinite(v) for v in box):
            continue
        if box.get_area() >= rect.get_area() * 0.95:
            continue
        normalized = ((box.x0 - rect.x0) / rect.width,
                      (box.y0 - rect.y0) / rect.height,
                      (box.x1 - rect.x0) / rect.width,
                      (box.y1 - rect.y0) / rect.height)
        key = tuple(round(v, 7) for v in normalized)
        if key in seen:
            continue
        seen.add(key)
        figures.append(normalized)
        if len(figures) > _MAX_PAGE_FIGURES:
            raise ValueError(f"페이지 그림 수가 상한({_MAX_PAGE_FIGURES})을 초과했습니다")
    return figures


def native_page_local(source_pdf: Path, page_index: int) -> dict:
    """Inspect text and raster placement together inside the bounded PDF worker."""
    from ..pipeline import pdf_worker

    result = {"blocks": [], "figures": [], "figure_warning": None}
    try:
        with pdf_worker.open_document(source_pdf) as doc:
            if doc.needs_pass or not 0 <= page_index < doc.page_count:
                return result
            fitz = _fitz()
            try:
                result["blocks"] = native_page_blocks(fitz, doc, page_index + 1)
            except Exception as exc:
                logger.warning("텍스트 레이어 추출 실패: %s", exc)
            try:
                result["figures"] = native_page_figures(fitz, doc[page_index])
            except Exception:
                result["figure_warning"] = "내장 이미지 위치를 추출하지 못했습니다"
    except Exception:
        result["figure_warning"] = "원본 PDF에서 내장 이미지를 확인하지 못했습니다"
    finally:
        _drain_mupdf("textlayer 페이지 검사")
    return result
