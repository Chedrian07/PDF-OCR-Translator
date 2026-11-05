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


def _row_text(pieces: list) -> str:
    """같은 시각 행의 텍스트 레이어 줄 조각들을 왼→오른으로 잇는다(위첨자 조각 포함)."""
    pieces.sort(key=lambda piece: piece[0].x0)
    out, last = "", None
    for rect, text in pieces:
        if last is not None and rect.x0 - last.x1 > 0.25 * max(rect.height, last.height):
            out += " "
        out += text
        last = rect
    return out


def page_line_texts(fitz, page, boxes: list) -> list[str]:
    """페이지 대비 0~999 상자(렌더 공간)마다 그 안의 텍스트 레이어 줄을 읽기 순서로 한 줄에 잇는다.

    줄은 **중심이 상자 안에 있으면** 고른다 — 잉크에 맞춘 상자는 글꼴 ascent/descent까지 든
    줄 상자보다 작아 면적 비율로는 놓치고, 이웃 줄의 중심은 줄 사이 빈 행 너머에 있다. 회전
    페이지도 같은 규칙이다(비회전 좌표에서 비교). 페이지 텍스트 레이어가 깨진 글리프·기호
    위주면(fidelity 정답과 같은 기준) 전부 "", 깨진 글리프가 든 줄도 ""."""
    from .fidelity import _block_rect, _broken_glyph, _untrusted_truth, normalize

    lines = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            text = "".join(span.get("text") or "" for span in line.get("spans", []))
            if bbox and text.strip():
                lines.append((fitz.Rect(bbox), text))
    page_text = " ".join(text for _, text in lines)
    if not page_text.strip() or _untrusted_truth(page_text, normalize(page_text)):
        return ["" for _ in boxes]
    out = []
    for box in boxes:
        rect = _block_rect(fitz, page, {"bbox": list(box)})
        picked = [] if rect is None else [
            (line_rect * page.rotation_matrix, text) for line_rect, text in lines
            if rect.contains(fitz.Point((line_rect.x0 + line_rect.x1) / 2,
                                        (line_rect.y0 + line_rect.y1) / 2))
        ]
        picked.sort(key=lambda piece: piece[0].y0)
        # 세로로 충분히 겹치는 조각은 한 시각 행이다 — 위·아래첨자는 별도 줄로 추출된다
        rows: list[tuple[list, float, float]] = []
        for piece in picked:
            r = piece[0]
            if rows:
                items, top, bottom = rows[-1]
                if min(r.y1, bottom) - max(r.y0, top) > 0.3 * min(r.height, bottom - top):
                    items.append(piece)
                    rows[-1] = (items, min(top, r.y0), max(bottom, r.y1))
                    continue
            rows.append(([piece], r.y0, r.y1))
        text = sanitize_text(" ".join(_row_text(items) for items, _, _ in rows)).strip()
        out.append("" if any(_broken_glyph(c) for c in text if not c.isspace()) else text)
    return out


def line_texts_local(source_pdf: Path, page_index: int, boxes: list) -> list[str]:
    """(PDF 워커) 원본 페이지 하나에서 상자마다 텍스트 레이어 줄(page_line_texts).

    영역 재처리로도 모델이 끝내 읽지 못한 줄(engine.region_rescue.fill_unreadable)에만 쓴다 —
    OCR 결과를 대체하지 않는다. 열 수 없거나 추출에 실패하면 전부 ""(이미지만 남는다)."""
    from ..pipeline import pdf_worker

    empty = ["" for _ in boxes]
    try:
        with pdf_worker.open_document(source_pdf) as doc:
            if doc.needs_pass or not 0 <= page_index < doc.page_count:
                return empty
            return page_line_texts(_fitz(), doc[page_index], boxes)
    except Exception as e:  # noqa: BLE001 — 채우지 못하면 원본 줄 이미지만 남는다
        logger.warning("%d페이지 인식 불가 줄 텍스트 추출 실패 (%s: %s)",
                       page_index + 1, e.__class__.__name__, str(e)[:200])
        return empty
    finally:
        _drain_mupdf("인식 불가 줄 텍스트")
