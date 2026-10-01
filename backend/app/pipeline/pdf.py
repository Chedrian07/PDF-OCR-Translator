"""PDF → 페이지 PNG 렌더링 (pymupdf)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ── 리소스 상한 (렌더 OOM 방어) ────────────────────────────────────────────
# 페이지 한 변(pt) 상한. A0 = 2384×3370pt이므로 5400pt(≈1.9m)면 포스터·도면류
# 정상 문서까지 넉넉히 통과한다. PDF 스펙은 MediaBox를 14400pt(200인치)까지
# 허용하는데, 그런 페이지 1장(파일 수 KB)이 기본 200dpi에서 40000×40000px
# ≈ RGB 4.8GB를 할당해 워커를 OOM으로 죽인다 — 업로드 검증(probe)에서 거부.
MAX_PAGE_SIDE_PT = 5400

# 페이지당 렌더 픽셀 수 상한. 50M px ≈ RGB 150MB(pixmap 버퍼)로 워커 메모리
# 안에서 안전하다. A4@400dpi(≈15.5M px)는 그대로 통과하지만, probe를 통과한
# 대형 페이지도 고 dpi에선 초과할 수 있으므로(예: 5400pt² @400dpi ≈ 900M px)
# 거부가 아니라 비율 유지 축소로 처리한다. 그라운딩 좌표는 0–999 정규화라
# 균일 축소에 영향 없다 (layout.py/_pct, pdf_fonts.py 참조).
MAX_RENDER_PIXELS = 50_000_000

# Pillow 압축 폭탄 상한을 렌더 상한에 맞춘다. 앱이 PIL로 여는 이미지는 전부 이 모듈이
# 만든 페이지 PNG(MAX_RENDER_PIXELS 이하 — 축소 시 반올림으로 수천 px 넘을 수 있어
# 5% 여유)와 거기서 자른 그림뿐이다. Pillow 기본값(약 8,950만 px, 2배에서 오류)이면
# 조작된 이미지 헤더나 모델이 낸 거대한 crop 좌표(벤더 draw_bounding_boxes는 좌표를
# 검증하지 않는다)가 수백 MB를 할당한 뒤에야 막힌다 — 상한을 넘으면 경고, 2배를 넘으면
# DecompressionBombError로 할당 전에 끊긴다. 페이지 렌더가 늘 이 모듈을 거치므로
# 엔진(벤더 크롭 포함)이 이미지를 열기 전에 적용된다.
PIL_MAX_IMAGE_PIXELS = MAX_RENDER_PIXELS + MAX_RENDER_PIXELS // 20


def _bound_pil_decompression() -> None:
    """프로세스 전역 Pillow 상한을 낮춘다(이미 더 낮게 잡혀 있으면 그대로 둔다)."""
    from PIL import Image

    current = Image.MAX_IMAGE_PIXELS
    if current is None or current > PIL_MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS


_bound_pil_decompression()

# OCR fallback에서 한 페이지의 숨은/중복 텍스트 레이어가 결과 파일을 폭증시키지
# 못하게 하는 독립 상한. 정상 논문/문서 페이지의 텍스트는 이보다 훨씬 작다.
MAX_EMBEDDED_TEXT_CHARS = 100_000
_UNSAFE_TEXT_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMBEDDED_RECOVERY_NOTE = "> ℹ️ PDF 내장 텍스트 레이어에서 복구한 plain text입니다."
# 복구 텍스트에서 마크다운으로 해석되는 인라인 문자 — 백슬래시로 글자 그대로 남긴다.
# 대괄호는 백슬래시로 이스케이프하지 않는다: 렌더러가 `\[ … \]`를 디스플레이 수식으로
# 바꾸므로(render._MATH_DISPLAY) `[1]`이 수식이 된다. 대괄호가 문제인 경우(그림 `![`,
# 링크 참조 정의 `[x]: …` — 줄째 사라진다)만 따로 끊는다.
_MD_INLINE_SPECIALS = re.compile(r"([\\`*_<>|~])")
_MD_IMAGE_OPEN = re.compile(r"!(?=\[)")
_MD_LINK_TARGET = re.compile(r"\]\(")
_MD_LINK_DEFINITION = re.compile(r"^\[(?=[^\]]*\]:)")
_MD_ORDERED_ITEM = re.compile(r"^(\d{1,9})([.)])(?=\s|$)")

# 렌더 단계 경고의 인계 파일. 렌더는 잡 경고 채널(merge.IncrementalMerger)이
# 생기기 **전**에 끝나므로, 흰 페이지 대체 같은 품질 저하 사실을 여기에 남겨야
# merge가 승계해 job.warnings(→ quality.state="degraded")에 반영할 수 있다.
RENDER_WARNINGS_NAME = "render_warnings.json"
_MAX_LISTED_FAILED_PAGES = 10


def quiet_fitz():
    """fitz(pymupdf) 지연 임포트 + MuPDF 에러의 stderr 직접 출력 차단(프로세스 1회).

    MuPDF C 라이브러리는 복구 가능한 파싱 문제(예: 손상 CID 폰트의
    "syntax error: unknown cid font type")를 텍스트 객체마다 stderr에 직접 찍는다
    — 한 페이지에서 수십 줄씩 서버 콘솔을 뒤덮지만 렌더 자체는 폰트 폴백으로
    정상 진행된다(실측: 27p 문서에서 p5 하나가 49줄). 표시만 끄면 동작·예외는
    불변이고 메시지는 내부 버퍼에 계속 쌓이므로, 호출부가 작업 단위로
    drain_mupdf_warnings()로 요약해 로거에 남긴다."""
    # PyMuPDF 1.28+는 레거시 `fitz` 모듈을 임포트하면 stdout에 폐지 경고를 찍는다.
    # 같은 객체를 내보내는 pymupdf를 fitz 이름으로 받아 호출부는 그대로 둔다.
    import pymupdf as fitz

    if fitz.TOOLS.mupdf_display_errors():
        fitz.TOOLS.mupdf_display_errors(False)
    return fitz


def drain_mupdf_warnings(context: str) -> None:
    """MuPDF 내부 경고 버퍼를 비우고 종류별 건수로 요약해 한 줄 로깅.

    버퍼는 프로세스 전역이라 동시 사용 시 다른 작업의 메시지가 섞일 수 있으나
    (잡 러너는 단일 워커) 진단용 요약이므로 best-effort로 충분하다."""
    try:
        import pymupdf as fitz

        text = fitz.TOOLS.mupdf_warnings()
    except Exception:  # pragma: no cover - 방어적
        return
    if not text:
        return
    counts = Counter(text.splitlines())
    top = [f"{m} (x{c})" if c > 1 else m for m, c in counts.most_common(3)]
    extra = f" 외 {len(counts) - 3}종" if len(counts) > 3 else ""
    logger.info("MuPDF 복구성 경고 %d건 (%s — 처리는 계속됨): %s%s",
                sum(counts.values()), context, " · ".join(top), extra)


# PDF를 열지 못했을 때의 사용자 메시지. MuPDF 예외 문자열은 붙이지 않는다 — 그 안에
# 서버의 절대 경로('Failed to open file \'/…/jobs/j_…/source.pdf\'')가 실려, 무인증
# 업로드 한 번으로 DATA_DIR 위치(로컬이면 계정 이름까지)가 400 응답·잡 오류로 새어 나간다.
# 원문은 서버 로그에만 남긴다.
_OPEN_FAILED = "PDF를 열 수 없습니다 — 손상되었거나 지원하지 않는 형식입니다"


def _open_pdf(fitz, pdf_path: Path, context: str):
    try:
        return fitz.open(str(pdf_path))
    except Exception as e:
        drain_mupdf_warnings(context)
        logger.warning("%s: PDF 열기 실패 (%s: %s)", context, e.__class__.__name__, str(e)[:300])
        raise ValueError(_OPEN_FAILED) from e


def probe_pdf(pdf_path: Path, max_pages: int) -> int:
    """업로드 검증: 열 수 있는 PDF인지 확인하고 페이지 수를 돌려준다.
    문제가 있으면 사용자 메시지를 담은 ValueError(서버 경로 등 내부 정보 없음)."""
    fitz = quiet_fitz()

    doc = _open_pdf(fitz, pdf_path, "업로드 검증")
    try:
        if doc.needs_pass:
            raise ValueError("암호화된 PDF는 지원하지 않습니다")
        n = doc.page_count
        if n == 0:
            raise ValueError("페이지가 없는 PDF입니다")
        if n > max_pages:
            raise ValueError(f"페이지 수({n})가 상한({max_pages})을 초과합니다")
        for i in range(n):
            try:
                r = doc[i].rect
            except Exception:  # 페이지 로드 실패는 렌더 단계가 흰색 페이지로 격리
                continue
            if max(r.width, r.height) > MAX_PAGE_SIDE_PT:
                raise ValueError(
                    f"페이지 {i + 1}의 크기({r.width:.0f}×{r.height:.0f}pt)가 "
                    f"한 변 상한({MAX_PAGE_SIDE_PT}pt)을 초과합니다"
                )
        return n
    finally:
        doc.close()
        drain_mupdf_warnings("업로드 검증")


def _escape_markdown_line(line: str) -> str:
    """한 줄을 마크다운 문법이 아니라 **글자 그대로** 렌더되도록 이스케이프한다.

    인라인 문법 문자(`\\` `` ` `` `*` `_` `<` `>` `|` `~`, 그림·링크의 `![`·`](`)는 어디서든,
    줄 머리에서만 의미가 생기는 블록 문법(`#` 제목, `-`·`+` 목록·구분선, `=` setext 밑줄,
    `1.`·`1)` 번호 목록, `[x]:` 링크 참조 정의)은 줄 머리에서 끊는다. 이스케이프된
    `---`는 페이지 구분자로도 해석되지 않고, `\\(`·`\\[`는 수식으로 바뀌지 않는다.
    """
    line = _MD_INLINE_SPECIALS.sub(r"\\\1", line.strip())
    line = _MD_IMAGE_OPEN.sub(r"\\!", line)
    # 링크·그림 대상 `](`의 여는 괄호는 문자 참조로 — `\(`는 렌더러의 인라인 수식이다
    line = _MD_LINK_TARGET.sub("]&#40;", line)
    line = _MD_LINK_DEFINITION.sub("&#91;", line)
    ordered = _MD_ORDERED_ITEM.match(line)
    if ordered:
        return f"{ordered.group(1)}\\{ordered.group(2)}{line[ordered.end():]}"
    if line[:1] in "#+-=":
        return "\\" + line
    return line


def extract_embedded_page_markdown(pdf_path: Path, page_number: int) -> str | None:
    """PDF의 1-based 페이지 텍스트 레이어를 안전한 plain-text Markdown으로 추출.

    OCR이 최종 실패한 페이지의 복구 경로다. 텍스트 블록을 **읽기 순서**(다단 인식 —
    reading_order.page_text_blocks)대로 문단 하나씩 내보내고, 줄마다 마크다운 문법을
    이스케이프해 원문의 ``#``/``---``/``|``가 제목·페이지 구분자·표로 해석되지 않게
    한다. 예전에는 페이지 전체를 4칸 들여쓴 코드 블록으로 냈는데, 코드 블록은 번역
    유닛이 되지 않아 복구 페이지가 한국어 번역본·번역 PDF에 영어로 남았고,
    ``get_text("text", sort=True)``가 같은 높이의 좌·우 단 줄을 한 줄로 섞었다.
    스캔 문서처럼 유효한 텍스트가 없거나 MuPDF 추출이 실패하면 ``None``을 반환한다.
    """
    from ..engine.textlayer import sanitize_text
    from .reading_order import page_text_blocks

    fitz = quiet_fitz()
    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        if doc.needs_pass or not 1 <= page_number <= doc.page_count:
            return None
        paragraphs: list[str] = []
        budget = MAX_EMBEDDED_TEXT_CHARS
        truncated = False
        for block in page_text_blocks(doc[page_number - 1], fitz):
            text = block.text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
            # `<PAGE>`·`<|…|>`는 파이프라인 제어 문법이라 textlayer 엔진과 같은 정화를 거친다
            text = sanitize_text(_UNSAFE_TEXT_CONTROLS.sub("", text)).strip()
            if not text:
                continue
            if budget <= 0:
                truncated = True
                break
            if len(text) > budget:
                text = text[:budget].rstrip()
                truncated = True
            budget -= len(text)
            lines = [_escape_markdown_line(line) for line in text.split("\n")]
            paragraph = "\n".join(line for line in lines if line)
            if paragraph:
                paragraphs.append(paragraph)
            if truncated:
                break
        body = "\n\n".join(paragraphs)
        if not body or not any(char.isalnum() for char in body):
            return None
        if truncated:
            logger.warning(
                "%d페이지 PDF 텍스트 레이어가 상한(%d자)을 초과해 절단",
                page_number,
                MAX_EMBEDDED_TEXT_CHARS,
            )
            body += "\n\n(텍스트 레이어 절단됨)"
        return f"{_EMBEDDED_RECOVERY_NOTE}\n\n{body}"
    except Exception as error:  # noqa: BLE001 — OCR 실패 뒤의 best-effort 복구
        logger.warning(
            "%d페이지 PDF 텍스트 레이어 추출 실패 (%s: %s)",
            page_number,
            error.__class__.__name__,
            str(error)[:200],
        )
        return None
    finally:
        if doc is not None:
            doc.close()
        drain_mupdf_warnings(f"{page_number}페이지 텍스트 복구")


def _capped_scale(w_pt: float, h_pt: float, dpi: int) -> tuple[float, int]:
    """dpi 배율의 목표 픽셀 수를 구하고, MAX_RENDER_PIXELS 초과면 비율을
    유지한 채 줄인 배율을 돌려준다. 반환: (배율, 축소 전 목표 픽셀 수)."""
    scale = dpi / 72
    target = int(w_pt * scale) * int(h_pt * scale)
    if target > MAX_RENDER_PIXELS:
        scale *= (MAX_RENDER_PIXELS / target) ** 0.5
    return scale, target


def _write_blank_page(doc, index: int, path: Path, dpi: int) -> None:
    """렌더 실패 페이지의 대체 흰색 PNG — 페이지 크기를 못 읽으면 A4(pt) 기준.
    정상 렌더와 같은 픽셀 상한을 지켜 대체 경로도 OOM을 못 일으키게 한다."""
    from PIL import Image

    try:
        rect = doc[index].rect
        w_pt, h_pt = float(rect.width), float(rect.height)
    except Exception:  # 페이지 객체 자체가 깨진 경우
        w_pt, h_pt = 595.0, 842.0
    scale, _ = _capped_scale(w_pt, h_pt, dpi)
    size = (max(1, round(w_pt * scale)), max(1, round(h_pt * scale)))
    Image.new("RGB", size, "white").save(path)


def _write_render_warnings(pages_dir: Path, messages: list[str]) -> None:
    """렌더 경고를 pages_dir에 기록(없으면 삭제 — 재실행 시 이전 세대가 남지 않게).
    기록 실패는 렌더 자체를 죽이지 않는다(품질 고지는 best-effort)."""
    path = pages_dir / RENDER_WARNINGS_NAME
    try:
        if messages:
            path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    except OSError as e:  # pragma: no cover - 디스크 이상
        logger.warning("렌더 경고 파일 기록 실패 (%s): %s", path, e)


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    max_pages: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """모든 페이지를 pages_dir/page_%04d.png (1-based)로 렌더.

    한 페이지가 깨져도(get_pixmap 예외) 잡 전체를 죽이지 않는다 — 흰색 페이지로
    대체하고 계속한다. 전 페이지 실패 시에만 ValueError.
    페이지당 픽셀 수가 MAX_RENDER_PIXELS를 넘으면 비율을 유지한 채 축소한다."""
    fitz = quiet_fitz()

    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = _open_pdf(fitz, pdf_path, "페이지 렌더")  # 잡 오류 메시지로도 경로가 새지 않게
    try:
        if doc.needs_pass:
            raise ValueError("암호화된 PDF는 지원하지 않습니다")
        n = doc.page_count
        if n == 0:
            raise ValueError("페이지가 없는 PDF입니다")
        if n > max_pages:
            raise ValueError(f"페이지 수({n})가 상한({max_pages})을 초과합니다")
        out: list[Path] = []
        failed_pages: list[int] = []
        last_err: Exception | None = None
        for i in range(n):
            p = pages_dir / f"page_{i + 1:04d}.png"
            try:
                page = doc[i]
                rect = page.rect
                # pix 생성 전에 목표 픽셀 수를 계산 — 상한 초과 시 비율 유지 축소
                # (probe의 치수 검사를 통과한 페이지도 고 dpi에선 넘을 수 있다)
                scale, target = _capped_scale(rect.width, rect.height, dpi)
                if target > MAX_RENDER_PIXELS:
                    logger.warning(
                        "페이지 %d/%d: 렌더 %dpx가 페이지당 상한(%dpx)을 초과 — "
                        "비율 유지 축소 (배율 %.3f→%.3f)",
                        i + 1, n, target, MAX_RENDER_PIXELS, dpi / 72, scale)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                pix.save(str(p))
            except Exception as e:  # noqa: BLE001 — 페이지 단위 격리
                failed_pages.append(i + 1)
                last_err = e
                logger.warning("페이지 %d/%d 렌더 실패 (%s: %s) — 흰색 페이지로 대체",
                               i + 1, n, e.__class__.__name__, str(e)[:200])
                _write_blank_page(doc, i, p, dpi)
            out.append(p)
            if progress_cb:
                progress_cb(i + 1, n)
        if len(failed_pages) == n:
            raise ValueError(f"모든 페이지({n}) 렌더에 실패했습니다: {last_err}") from last_err
        # 흰 페이지 대체는 조용한 품질 저하다 — 로그만으로는 사용자가 알 수 없어
        # 병합기가 승계할 수 있게 파일로 남긴다 (없으면 파일 삭제 = 경고 없음).
        messages: list[str] = []
        if failed_pages:
            listed = ", ".join(str(p) for p in failed_pages[:_MAX_LISTED_FAILED_PAGES])
            more = (
                f" 외 {len(failed_pages) - _MAX_LISTED_FAILED_PAGES}쪽"
                if len(failed_pages) > _MAX_LISTED_FAILED_PAGES
                else ""
            )
            messages.append(
                f"{len(failed_pages)}/{n}페이지 렌더에 실패해 흰 페이지로 대체했습니다 "
                f"({listed}{more}) — 해당 페이지의 인식 결과가 비어 있거나 부정확할 수 있습니다"
            )
        _write_render_warnings(pages_dir, messages)
        return out
    finally:
        doc.close()
        drain_mupdf_warnings("페이지 렌더")
