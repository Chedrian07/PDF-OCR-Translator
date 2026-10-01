"""layout.json 블록에 원본 PDF 텍스트 레이어의 **실측** 폰트 크기를 주입.

면적 휴리스틱(layout.py `estimate_font_size_cqw`)은 어디까지나 폴백이다. 원본
PDF에 텍스트 레이어가 있으면 그 안의 span 크기를 그대로 읽어 훨씬 정확한
폰트 크기를 심을 수 있다.

동작:
- det bbox(0–999 정규화)를 PDF 페이지의 pt 사각형으로 사상(x_pt = x/999×W,
  y_pt = y/999×H; W·H는 fitz 페이지의 pt 크기). ±3pt 여유로 확장.
- 그 사각형 안에 **중심점**이 드는 span들을 모아 글자수 가중 중앙값 크기를 구하고
  block["fs"] = size_pt / page_width_pt × 100 (cqw = 페이지 폭의 1%)로 심는다.
- 볼드 글자가 과반이면 block["bold"] = True.
- 원문 span의 주 글꼴 계열을 block["font_style"] = "serif"|"sans"로 심어,
  번역 PDF의 대표 제목이 원문 sans인데 한글만 명조로 바뀌는 현상을 막는다.
- 원문 줄의 폭과 중심을 비교해 block["align"] = "center"|"justify"를 심는다.
  PDF 번역 내보내기가 모든 블록을 좌측 정렬로 평탄화하지 않게 하기 위함이다.
- 줄 방향(dir)이 세로인 글자가 과반이면 block["vertical"] = "up"|"down"
  (arXiv 왼쪽 여백 스탬프 같은 90° 회전 텍스트 — 렌더러가 writing-mode로 재현).
- 처리한 페이지에는 page["fonts_v"] 버전을 스탬프한다 — 백필이 구버전
  enrichment 결과를 감지해 1회 재실행할 수 있게 (매 요청 재스캔 방지).
  건너뛴 페이지(번호 오류·범위 밖·텍스트 추출 예외)도 똑같이 스탬프한다.
- span이 하나도 없거나 텍스트 레이어가 없는 블록은 건드리지 않는다(폴백에 위임).

실패(텍스트 레이어 없음·손상 PDF·페이지 범위 초과)는 조용히 무시한다 —
enrichment은 절대 잡·렌더를 깨뜨리면 안 된다.

텍스트 추출(get_text)은 PDF 워커 프로세스에서 페이지마다 돈다(pdf_worker — 호출 맥락의
풀: 병합은 ocr, API 백필은 export, 빌드 안에서는 그 워커 그대로). 시간 상한을 넘거나
워커를 죽인 페이지가 나오면 그 호출의 나머지 페이지는 실측 없이 스탬프만 한다.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from statistics import median

logger = logging.getLogger(__name__)

_BOLD_FLAG = 16  # fitz span flags: bit 4 == bold

# enrichment 스키마 버전 — 필드 추가 시 올리면 기존 잡이 /layout 요청 때 재백필된다
# 6: /Rotate 페이지에서 span을 표시(회전 반영) 공간으로 옮겨 bbox와 비교한다. 그 전에는
#    표시 공간 bbox와 비회전 공간 span을 섞어 fs·정렬이 전멸하고 가로 글을 세로쓰기로
#    오판했다 — 기존 잡의 회전 페이지 메타를 다시 계산하려고 올린다.
ENRICH_VERSION = 6


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    """(size, chars) 목록의 글자수 가중 중앙값 크기.

    크기 오름차순으로 정렬하고 누적 글자수가 전체의 절반에 처음 도달하는
    지점의 크기를 반환한다 — 큰 폰트의 짧은 조각(각주 번호 등)에 휘둘리지 않게."""
    total = sum(c for _, c in pairs)
    half = total / 2
    acc = 0
    for size, chars in sorted(pairs):
        acc += chars
        if acc >= half:
            return size
    return pairs[-1][0]  # 방어 — 정상 입력에선 도달하지 않음


def _span_is_bold(span: dict) -> bool:
    if int(span.get("flags", 0)) & _BOLD_FLAG:
        return True
    return "bold" in str(span.get("font", "")).lower()


def _font_style(font_name: str) -> str | None:
    """PDF 내장/subset 폰트 이름을 큰 serif/sans 계열로만 보수적으로 분류한다."""
    name = font_name.lower().replace("-", "").replace("_", "")
    # URW의 Helvetica 호환 글꼴은 PDF에 ``NimbusSanL``로 들어오기도 한다.
    # 일반적인 ``Sans`` 철자와 달라 대표 제목이 serif 폴백으로 빠지지 않게
    # 별도 토큰으로 분류한다.
    if any(token in name for token in ("sans", "nimbussan", "helv", "arial", "cmss")):
        return "sans"
    if any(token in name for token in (
        "serif", "times", "roman", "nimbusrom", "cmr", "cmbx", "cmti", "cmmi",
    )):
        return "serif"
    return None


def _infer_alignment(
    block: dict,
    block_rect: tuple[float, float, float, float],
    line_rects: list[tuple[float, float, float, float]],
    page_width: float,
) -> str | None:
    """원문 줄 기하에서 가운데/양쪽 정렬을 보수적으로 추론한다.

    OCR bbox는 대체로 실제 글리프에 딱 맞으므로 한 줄짜리 본문을 가운데 정렬로
    오인하기 쉽다. 한 줄 제목은 페이지 중앙에 놓인 경우만 center로 보고, 여러
    줄은 줄 중심이 일정하면서 줄 폭 중앙값이 블록보다 충분히 짧을 때만 center다.
    대부분의 줄이 블록 폭을 채우는 문단은 justify로 분류한다.
    """
    if not line_rects:
        return None
    x0, _y0, x1, _y1 = block_rect
    width = max(1.0, x1 - x0)
    center = (x0 + x1) / 2
    line_widths = [max(0.0, r[2] - r[0]) for r in line_rects]
    line_centers = [(r[0] + r[2]) / 2 for r in line_rects]
    ratios = [w / width for w in line_widths]

    if len(line_rects) == 1:
        if (
            str(block.get("type") or "") == "title"
            and abs(center - page_width / 2) <= page_width * 0.08
        ):
            return "center"
        return None

    centered = sum(abs(c - center) <= max(3.0, width * 0.04) for c in line_centers)
    if centered / len(line_rects) >= 0.75 and median(ratios) < 0.72:
        return "center"

    # 마지막 줄은 짧은 것이 정상이다. 그 전 줄 중 절반 이상이 블록 폭을 거의
    # 채우면 원문의 양쪽 정렬을 번역 PDF에서도 유지한다.
    paragraph_lines = ratios[:-1] or ratios
    if len(line_rects) >= 3 and (
        sum(r >= 0.82 for r in paragraph_lines) / len(paragraph_lines) >= 0.5
    ):
        return "justify"
    return None


def enrich_layout_fonts(pdf_path: Path, pages: list[dict]) -> bool:
    """layout.json 페이지 블록에 원본 PDF 실측 폰트 크기(cqw)를 주입.

    pages 엔트리: {"page": N(1-based), "width", "height", "blocks": [...]}.
    블록을 제자리(in-place)로 수정하고, 하나라도 주입했거나 fonts_v 스탬프를 새로
    찍었으면 True를 돌려준다(스탬프만으로도 저장이 필요하다). 원본 PDF를 열 수 없으면
    아무것도 찍지 않고 False.

    페이지마다 PDF 워커 작업 하나로 돈다(enrich_page_local) — 페이지 dict는 피클로 오가므로
    결과를 원래 dict에 제자리로 되돌려 호출자가 쥔 참조(병합기의 layout_pages)를 지킨다.
    시간 상한·워커 비정상 종료가 나면 그 페이지부터는 실측을 포기하고 스탬프만 찍는다 —
    건너뛴 페이지를 스탬프하지 않으면 백필이 매 요청 같은 상한을 다시 기다린다."""
    from . import pdf_worker

    changed = False
    gave_up = False
    for page in pages:
        if not isinstance(page, dict):
            continue
        if gave_up:
            if page.get("fonts_v") != ENRICH_VERSION:
                page["fonts_v"] = ENRICH_VERSION
                changed = True
            continue
        try:
            page_index = int(page.get("page", 0)) - 1
        except (TypeError, ValueError):
            page_index = -1
        try:
            result = pdf_worker.run_page(
                "app.pipeline.pdf_fonts:enrich_page_local", pdf_path, page_index, (page,),
            )
        except pdf_worker.PdfWorkerError as error:
            logger.warning("폰트 실측 주입 중단 — %d페이지부터 스탬프만 남김 (%s)",
                           page_index + 1, error)
            gave_up = True
            result = (False, page)
        except Exception:  # noqa: BLE001 — 한 페이지 실패가 나머지 백필을 막지 않는다
            result = (False, page)
        if result is None:
            return changed  # 원본 PDF를 열 수 없다 — 예전 계약대로 찍지 않는다
        page_changed, updated = result
        if updated is not page:  # 워커가 돌려준 사본 — 제자리로 되돌린다
            page.clear()
            page.update(updated)
        if page_changed:
            changed = True
        # 건너뛴 페이지(번호 오류·범위 밖·크기 0·추출 예외·상한 초과)도 이 버전으로 "시도했고
        # 얻을 것이 없었다"고 스탬프한다. 빼먹으면 백필 호출부가 미스탬프 페이지를 보고 매
        # 요청 전 문서를 재스캔하고 layout을 다시 써서 내보내기 PDF 캐시까지 무효화하는 루프에
        # 빠진다(손상 PDF로 재현).
        if page.get("fonts_v") != ENRICH_VERSION:
            page["fonts_v"] = ENRICH_VERSION
            changed = True
    return changed


def enrich_page_local(pdf_path: Path, page_index: int, page: dict) -> tuple[bool, dict] | None:
    """(PDF 워커) 페이지 하나에 실측 메타를 심고 fonts_v를 찍는다 → (바뀌었나, 페이지).

    원본 PDF를 열 수 없으면 None. 페이지 처리 예외는 삼키고 스탬프만 찍는다."""
    from . import pdf_worker
    from .pdf import drain_mupdf_warnings, quiet_fitz

    try:
        fitz = quiet_fitz()
        with contextlib.ExitStack() as stack:
            try:
                doc = stack.enter_context(pdf_worker.open_document(pdf_path))
            except Exception:  # noqa: BLE001 — 손상·없는 원본
                return None
            changed = False
            try:
                changed = _enrich_page(fitz, doc, page)
            except Exception:  # noqa: BLE001 — 한 페이지 실패가 나머지 백필을 막지 않는다
                changed = False
            if page.get("fonts_v") != ENRICH_VERSION:
                page["fonts_v"] = ENRICH_VERSION
                changed = True
            return changed, page
    finally:
        drain_mupdf_warnings("폰트 추출")


def _enrich_page(fitz, doc, page: dict) -> bool:
    """한 페이지의 블록에 실측 메타를 심는다. 블록을 바꿨으면 True.

    스탬프는 호출부가 결과와 무관하게 찍는다 — 여기서 일찍 빠져나가도 된다.
    """
    try:
        pno = int(page.get("page", 0))
    except (TypeError, ValueError):
        return False
    if pno < 1 or pno > doc.page_count:  # 페이지 범위 방어
        return False
    fpage = doc[pno - 1]
    pw = float(fpage.rect.width)
    ph = float(fpage.rect.height)
    if pw <= 0 or ph <= 0:
        return False
    try:
        text_dict = fpage.get_text("dict")
    except Exception:  # noqa: BLE001 — 손상 페이지는 폴백 휴리스틱에 맡긴다
        return False
    # det bbox는 렌더 이미지와 같은 **표시(회전 반영) 공간**의 0–999 좌표이고(pw·ph도
    # fpage.rect), get_text("dict")의 bbox·dir은 /Rotate 이전의 비회전 공간이다.
    # span을 rotation_matrix로 표시 공간에 옮겨야 비교·정렬 추론·세로쓰기 판정이 같은
    # 좌표계에서 이뤄진다(회전 0이면 항등행렬이라 기존과 같다). 방향 벡터는 평행이동
    # 없이 선형 부분만 돌린다 — 화면에서 가로로 읽히는 줄을 세로쓰기로 오판하지 않게.
    to_display = fpage.rotation_matrix
    # 페이지의 모든 span을 평면화 (표시 공간 bbox·dir과 원본 span dict)
    spans: list[tuple[dict, tuple, tuple, int]] = []
    line_no = 0
    for tb in text_dict.get("blocks", ()):
        for line in tb.get("lines", ()):
            dx, dy = tuple(line.get("dir") or (1, 0))[:2]
            ldir = (
                dx * to_display.a + dy * to_display.c,
                dx * to_display.b + dy * to_display.d,
            )
            for sp in line.get("spans", ()):
                raw_bbox = sp.get("bbox")
                if not raw_bbox or len(raw_bbox) != 4:
                    continue
                shown = fitz.Rect(raw_bbox) * to_display
                shown.normalize()
                spans.append((sp, (shown.x0, shown.y0, shown.x1, shown.y1), ldir, line_no))
            line_no += 1
    if not spans:
        return False

    changed = False
    for block in page.get("blocks", ()):
        if block.get("image"):
            continue  # 이미지 블록 — 폰트 크기 없음
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # 0–999 정규화 → pt, ±3pt 확장
        rx1 = x1 / 999 * pw - 3
        ry1 = y1 / 999 * ph - 3
        rx2 = x2 / 999 * pw + 3
        ry2 = y2 / 999 * ph + 3

        pairs: list[tuple[float, int]] = []
        bold_chars = 0
        total_chars = 0
        serif_chars = 0
        sans_chars = 0
        vert_up = vert_down = 0
        matched_lines: dict[int, list[float]] = {}
        for sp, sb, ldir, source_line_no in spans:
            cx = (sb[0] + sb[2]) / 2
            cy = (sb[1] + sb[3]) / 2
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            n = len((sp.get("text") or "").strip())
            if n <= 0:
                continue
            size = float(sp.get("size", 0) or 0)
            if size <= 0:
                continue
            pairs.append((size, n))
            total_chars += n
            bounds = matched_lines.setdefault(
                source_line_no, [float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])],
            )
            bounds[0] = min(bounds[0], float(sb[0]))
            bounds[1] = min(bounds[1], float(sb[1]))
            bounds[2] = max(bounds[2], float(sb[2]))
            bounds[3] = max(bounds[3], float(sb[3]))
            if _span_is_bold(sp):
                bold_chars += n
            style = _font_style(str(sp.get("font") or ""))
            if style == "serif":
                serif_chars += n
            elif style == "sans":
                sans_chars += n
            if abs(ldir[1]) > 0.7:  # 세로쓰기 줄 (y축 진행)
                if ldir[1] < 0:
                    vert_up += n    # 아래→위 (arXiv 스탬프 방향)
                else:
                    vert_down += n

        if not pairs or total_chars <= 0:
            continue  # 매칭 span 없음 — 블록 미변경(폴백에 위임)
        block["fs"] = _weighted_median(pairs) / pw * 100
        if bold_chars > total_chars * 0.5:
            block["bold"] = True
        else:
            block.pop("bold", None)
        if sans_chars > serif_chars and sans_chars > 0:
            block["font_style"] = "sans"
        elif serif_chars > 0:
            block["font_style"] = "serif"
        else:
            block.pop("font_style", None)
        align = _infer_alignment(
            block,
            (rx1 + 3, ry1 + 3, rx2 - 3, ry2 - 3),
            [tuple(values) for values in matched_lines.values()],
            pw,
        )
        if align:
            block["align"] = align
        else:
            block.pop("align", None)
        if (vert_up + vert_down) > total_chars * 0.5:
            block["vertical"] = "up" if vert_up >= vert_down else "down"
        else:
            block.pop("vertical", None)
        changed = True
    return changed
