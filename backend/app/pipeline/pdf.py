"""PDF → 페이지 PNG 렌더링 (pymupdf)."""

from __future__ import annotations

import json
import logging
import re
import threading
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


# 업로드 검증을 기다리며 서버 스레드를 붙잡을 수 있는 요청 수(probe 워커 2 + 짧은 대기 2)와
# 빈 워커를 기다리는 상한. API는 probe를 공용 anyio 스레드풀(모든 동기 라우트와 공유, 기본 40)
# 에서 돌린다 — 예전에는 대기 상한이 검증 상한(60초)과 같아 적대적 업로드 수십 건이 그 스레드를
# 60초씩 물고 /api/health·/api/jobs까지 굶겼다(감사 security-1). 넘치는 업로드는 기다리지 않고
# 곧바로 바쁨(PdfWorkerBusy → API 503 + Retry-After)으로 끝낸다.
_PROBE_MAX_IN_FLIGHT = 4
_PROBE_QUEUE_WAIT_S = 5.0
_PROBE_SLOTS = threading.BoundedSemaphore(_PROBE_MAX_IN_FLIGHT)


def probe_pdf(pdf_path: Path, max_pages: int) -> int:
    """업로드 검증: 열 수 있는 PDF인지 확인하고 페이지 수를 돌려준다.
    문제가 있으면 사용자 메시지를 담은 ValueError(서버 경로 등 내부 정보 없음).

    검증 자체(손상 xref repair·복잡도 게이트)도 MuPDF 작업이라 서버 프로세스가 아니라 probe
    풀 워커에서 돌린다(pdf_worker). 업로드가 수 분짜리 내보내기 빌드나 적대적 페이지를 처리
    중인 OCR 워커 뒤에 줄서지 않도록 풀을 따로 둔다. 시간 상한(PDF_PAGE_TIMEOUT_S)을 넘거나
    워커가 죽은 PDF는 거부한다. 동시에 검증 중인 업로드가 _PROBE_MAX_IN_FLIGHT를 넘거나 빈
    워커를 _PROBE_QUEUE_WAIT_S 안에 얻지 못하면 PdfWorkerBusy(→ API 503)."""
    from . import pdf_worker

    timeout = pdf_worker.page_timeout()
    isolated = pdf_worker.mode() == pdf_worker.MODE_PROCESS
    if isolated and not _PROBE_SLOTS.acquire(blocking=False):
        raise pdf_worker.PdfWorkerBusy(pdf_worker.POOL_PROBE, 0.0)
    try:
        return pdf_worker.run(
            "app.pipeline.pdf:probe_pdf_local",
            (
                pdf_path, max_pages,
                pdf_worker.max_page_content_bytes(), pdf_worker.max_page_xobject_calls(),
            ),
            pool=pdf_worker.POOL_PROBE, timeout=timeout,
            wait=min(timeout, _PROBE_QUEUE_WAIT_S) if timeout else _PROBE_QUEUE_WAIT_S,
        )
    except pdf_worker.PdfWorkerTimeout as error:
        raise ValueError(
            f"PDF 검증이 시간 상한({_limit_text(timeout)})을 넘었습니다 — 지나치게 복잡하거나 "
            "손상된 PDF입니다"
        ) from error
    except pdf_worker.PdfWorkerCrashed as error:
        # 빈 작업도 못 끝내는 워커면 서버 환경 탓이다 — 사용자 파일을 '손상'으로 몰지 않는다
        if not pdf_worker.workers_can_start(pdf_worker.POOL_PROBE):
            raise pdf_worker.PdfWorkerUnavailable() from error
        raise ValueError(
            "PDF 검증 중 처리 프로세스가 비정상 종료했습니다 — 손상되었거나 지원하지 않는 "
            "PDF입니다"
        ) from error
    finally:
        if isolated:
            _PROBE_SLOTS.release()


def probe_pdf_local(
    pdf_path: Path, max_pages: int, max_content_bytes: int = 0, max_xobject_calls: int = 0,
) -> int:
    """probe_pdf의 본체 — probe 워커(또는 inline 모드의 호출 스레드)에서 실행한다.
    복잡도 상한은 부모가 읽어 넘긴다(워커는 기동 시점의 환경을 물려받으므로)."""
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
        _check_page_complexity(fitz, doc, n, max_content_bytes, max_xobject_calls)
        return n
    finally:
        doc.close()
        drain_mupdf_warnings("업로드 검증")


def _check_page_complexity(
    fitz, doc, page_count: int, max_bytes: int, max_calls: int,
) -> None:
    """업로드 복잡도 게이트 — 페이지가 그리게 하는 콘텐츠와 중첩 XObject 호출을 렌더 없이 센다.

    페이지 수·한 변 길이만 보던 검증을 수 KB짜리 중첩 Form XObject PDF(리프 그리기 10^12회)와
    평면 대량 path 페이지가 그대로 통과해 렌더·분석이 폭주했다(감사 security-2·gap3-…-2).
    상한(PDF_MAX_PAGE_CONTENT_MB·PDF_MAX_PAGE_XOBJECT_CALLS — pdf_worker, 0 = 끄기)을 넘으면
    사용자 메시지를 담은 ValueError(ContentTooComplex)로 거부한다 → API 400. 분석 자체가 실패한
    페이지는 거부하지 않는다 — 깨진 페이지는 렌더 단계가 흰 페이지로 격리하는 기존 계약이다."""
    from .pdf_complexity import ComplexityScanner, ContentTooComplex

    if not (max_bytes or max_calls) or not doc.is_pdf:
        return
    try:
        scanner = ComplexityScanner(
            fitz, doc, max_content_bytes=max_bytes, max_xobject_calls=max_calls,
        )
    except Exception as error:  # noqa: BLE001 — 저수준 API 변화가 업로드를 막지 않게
        logger.warning("업로드 복잡도 검사를 건너뜁니다 (%s: %s)",
                       error.__class__.__name__, str(error)[:200])
        return
    for index in range(page_count):
        try:
            scanner.check(index)
        except ContentTooComplex:
            raise
        except Exception as error:  # noqa: BLE001 — 페이지 단위 격리
            logger.info("%d페이지 복잡도 검사 실패 — 렌더 단계 격리에 맡김 (%s: %s)",
                        index + 1, error.__class__.__name__, str(error)[:200])


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


def extract_embedded_page_markdown(
    pdf_path: Path, page_number: int, *, should_cancel: Callable[[], bool] | None = None,
) -> str | None:
    """PDF의 1-based 페이지 텍스트 레이어를 안전한 plain-text Markdown으로 추출.

    OCR이 최종 실패한 페이지의 복구 경로다. 텍스트 블록을 **읽기 순서**(다단 인식 —
    reading_order.page_text_blocks)대로 문단 하나씩 내보내고, 줄마다 마크다운 문법을
    이스케이프해 원문의 ``#``/``---``/``|``가 제목·페이지 구분자·표로 해석되지 않게
    한다. 예전에는 페이지 전체를 4칸 들여쓴 코드 블록으로 냈는데, 코드 블록은 번역
    유닛이 되지 않아 복구 페이지가 한국어 번역본·번역 PDF에 영어로 남았고,
    ``get_text("text", sort=True)``가 같은 높이의 좌·우 단 줄을 한 줄로 섞었다.
    스캔 문서처럼 유효한 텍스트가 없거나 MuPDF 추출이 실패하면 ``None``을 반환한다.

    추출은 PDF 워커에서 페이지 작업으로 돈다(pdf_worker.run_page) — 시간 상한을 넘었거나
    앞서 렌더·분석이 상한을 넘은 페이지는 기다리지 않고 None이다. should_cancel이 참이 되면
    진행 중인 추출의 워커를 끝내고 JobCanceled.
    """
    from . import pdf_worker

    try:
        return pdf_worker.run_page(
            "app.pipeline.pdf:embedded_page_markdown_local", pdf_path, page_number - 1,
            cancel=should_cancel,
        )
    except pdf_worker.PdfWorkerCanceled:
        _raise_job_canceled()
    except pdf_worker.PdfWorkerError as error:  # 시간 상한·워커 비정상 종료·격리 메모
        logger.warning("%d페이지 PDF 텍스트 레이어 복구를 건너뜀 (%s)", page_number, error)
        return None


def embedded_page_markdown_local(pdf_path: Path, page_index: int) -> str | None:
    """(PDF 워커) extract_embedded_page_markdown의 본체 — page_index는 0-based."""
    from ..engine.textlayer import sanitize_text
    from . import pdf_worker
    from .reading_order import page_text_blocks

    page_number = page_index + 1
    fitz = quiet_fitz()
    try:
        with pdf_worker.open_document(pdf_path) as doc:
            if doc.needs_pass or not 1 <= page_number <= doc.page_count:
                return None
            blocks = page_text_blocks(doc[page_number - 1], fitz)
        paragraphs: list[str] = []
        budget = MAX_EMBEDDED_TEXT_CHARS
        truncated = False
        for block in blocks:
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
        drain_mupdf_warnings(f"{page_number}페이지 텍스트 복구")


def page_text_local(pdf_path: Path, page_index: int) -> str:
    """(PDF 워커) 페이지 텍스트 레이어 원문(get_text) — 범위 밖이면 빈 문자열."""
    from . import pdf_worker

    try:
        with pdf_worker.open_document(pdf_path) as doc:
            if not 0 <= page_index < doc.page_count:
                return ""
            return doc[page_index].get_text()
    finally:
        drain_mupdf_warnings(f"{page_index + 1}페이지 텍스트")


def page_plain_texts(pdf_path: Path, page_indices: list[int]) -> list[str] | None:
    """여러 페이지(0-based)의 텍스트 레이어 원문 — 페이지마다 PDF 워커 작업.

    하나라도 실패하거나 시간 상한을 넘으면 None — 호출자(병합기의 페이지 정합)는 위치 기반
    배치로 돌아간다. 원문은 정합 대조용이라 일부만으로 판단하지 않는다. 단, 앞서 상한을 넘은
    페이지(격리 메모)는 기다리지 않으므로 빈 원문(텍스트 없는 페이지와 같다)으로 두고 계속한다."""
    from . import pdf_worker

    texts: list[str] = []
    for index in page_indices:
        try:
            texts.append(pdf_worker.run_page(
                "app.pipeline.pdf:page_text_local", pdf_path, index,
            ))
        except pdf_worker.PdfPageQuarantined:
            texts.append("")
        except Exception as error:  # noqa: BLE001 — 정합은 선택적 개선이다
            logger.info("%d페이지 텍스트 레이어를 읽지 못해 정합을 건너뜀 (%s: %s)",
                        index + 1, error.__class__.__name__, str(error)[:200])
            return None
    return texts


def _capped_scale(w_pt: float, h_pt: float, dpi: int) -> tuple[float, int]:
    """dpi 배율의 목표 픽셀 수를 구하고, MAX_RENDER_PIXELS 초과면 비율을
    유지한 채 줄인 배율을 돌려준다. 반환: (배율, 축소 전 목표 픽셀 수)."""
    scale = dpi / 72
    target = int(w_pt * scale) * int(h_pt * scale)
    if target > MAX_RENDER_PIXELS:
        scale *= (MAX_RENDER_PIXELS / target) ** 0.5
    return scale, target


def _write_blank_page(path: Path, size_pt: tuple[float, float] | None, dpi: int) -> None:
    """렌더 실패 페이지의 대체 흰색 PNG — 페이지 크기를 못 읽었으면 A4(pt) 기준.
    정상 렌더와 같은 픽셀 상한을 지켜 대체 경로도 OOM을 못 일으키게 한다.
    MuPDF를 쓰지 않는다(크기는 워커가 먼저 알려 준다) — 상한을 넘긴 페이지를 대신할 때도
    서버 프로세스에서 안전하게 만든다."""
    from PIL import Image

    w_pt, h_pt = size_pt if size_pt else (595.0, 842.0)
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


# 페이지 렌더 시간 상한 초과가 이만큼 쌓이면 남은 페이지를 기다리지 않고 잡을 끝낸다 —
# 업로드 게이트를 빠져나간 적대적 문서가 페이지마다 상한을 다 써서 단일 OCR 워커를 몇 시간씩
# 붙잡지 않게(200쪽 × 60초). 정상 문서는 상한에 한 번도 닿지 않는다.
_MAX_RENDER_TIMEOUTS = 3


def _limit_text(seconds: float | None) -> str:
    return f"{seconds:g}초" if seconds else "상한"


def _listed_pages(pages: list[int]) -> str:
    listed = ", ".join(str(p) for p in pages[:_MAX_LISTED_FAILED_PAGES])
    more = (
        f" 외 {len(pages) - _MAX_LISTED_FAILED_PAGES}쪽"
        if len(pages) > _MAX_LISTED_FAILED_PAGES
        else ""
    )
    return listed + more


def render_open_local(pdf_path: Path, max_pages: int) -> list[tuple[float, float] | None]:
    """(워커) 렌더할 문서를 검증하고 페이지별 크기(pt)를 돌려준다 — 못 읽은 페이지는 None.
    열 수 없거나 암호화·빈 문서·페이지 상한 초과면 사용자 메시지 ValueError."""
    from . import pdf_worker

    try:
        with pdf_worker.open_document(pdf_path) as doc:
            if doc.needs_pass:
                raise ValueError("암호화된 PDF는 지원하지 않습니다")
            n = doc.page_count
            if n == 0:
                raise ValueError("페이지가 없는 PDF입니다")
            if n > max_pages:
                raise ValueError(f"페이지 수({n})가 상한({max_pages})을 초과합니다")
            sizes: list[tuple[float, float] | None] = []
            for i in range(n):
                try:
                    rect = doc[i].rect
                    sizes.append((float(rect.width), float(rect.height)))
                except Exception:  # noqa: BLE001 — 깨진 페이지는 그 페이지 렌더가 격리한다
                    sizes.append(None)
            return sizes
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — 열기 실패(경로가 든 MuPDF 문구는 로그에만)
        logger.warning("페이지 렌더: PDF 열기 실패 (%s: %s)", e.__class__.__name__, str(e)[:300])
        raise ValueError(_OPEN_FAILED) from e
    finally:
        drain_mupdf_warnings("페이지 렌더")


def render_page_local(
    pdf_path: Path, page_index: int, out_path: Path, dpi: int, page_count: int,
) -> None:
    """(워커) 페이지 하나를 PNG로 — MAX_RENDER_PIXELS를 넘으면 비율을 유지해 줄인다."""
    from . import pdf_worker

    fitz = quiet_fitz()
    try:
        with pdf_worker.open_document(pdf_path) as doc:
            page = doc[page_index]
            rect = page.rect
            # pix 생성 전에 목표 픽셀 수를 계산 — 상한 초과 시 비율 유지 축소
            # (probe의 치수 검사를 통과한 페이지도 고 dpi에선 넘을 수 있다)
            scale, target = _capped_scale(rect.width, rect.height, dpi)
            if target > MAX_RENDER_PIXELS:
                logger.warning(
                    "페이지 %d/%d: 렌더 %dpx가 페이지당 상한(%dpx)을 초과 — "
                    "비율 유지 축소 (배율 %.3f→%.3f)",
                    page_index + 1, page_count, target, MAX_RENDER_PIXELS, dpi / 72, scale)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            pix.save(str(out_path))
    finally:
        drain_mupdf_warnings(f"{page_index + 1}페이지 렌더")


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    max_pages: int,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> list[Path]:
    """모든 페이지를 pages_dir/page_%04d.png (1-based)로 렌더.

    MuPDF 작업은 PDF 워커 프로세스에서 페이지마다 하나씩 돈다(pdf_worker.run_page — 풀은
    호출 맥락을 따른다: OCR 입력은 ocr, facsimile은 export). 그래서 렌더가 서버의 GIL을 쥐지
    않고, 한 페이지가 시간 상한(PDF_PAGE_TIMEOUT_S)을 넘거나 워커를 죽여도 그 페이지만 잃는다.
    한 페이지가 깨져도(예외·상한 초과·워커 비정상 종료) 잡 전체를 죽이지 않는다 — 흰색 페이지로
    대체하고 사용자 경고(render_warnings.json → job.warnings)를 남긴 뒤 계속한다. 전 페이지
    실패, 또는 상한 초과가 _MAX_RENDER_TIMEOUTS번 쌓이면 ValueError. 페이지당 픽셀 수가
    MAX_RENDER_PIXELS를 넘으면 비율을 유지한 채 축소한다. progress_cb의 예외는 그대로
    전파된다(러너의 페이지 사이 취소). should_cancel이 참이 되면 진행 중인 페이지의 워커까지
    끝내고 JobCanceled — 페이지 하나가 수십 초 걸려도 취소가 기다리지 않는다."""
    from . import pdf_worker

    pages_dir.mkdir(parents=True, exist_ok=True)
    pool = pdf_worker.current_pool()
    timeout = pdf_worker.page_timeout()
    try:
        sizes = pdf_worker.run(
            "app.pipeline.pdf:render_open_local", (pdf_path, max_pages),
            pool=pool, timeout=timeout, cancel=should_cancel,
        )
    except pdf_worker.PdfWorkerCanceled:
        _raise_job_canceled()
    except pdf_worker.PdfWorkerTimeout as e:
        raise ValueError(
            f"PDF를 여는 데 시간 상한({_limit_text(timeout)})을 넘었습니다 — 손상되었거나 "
            "지나치게 복잡한 PDF입니다"
        ) from e
    except pdf_worker.PdfWorkerCrashed as e:
        raise ValueError(_OPEN_FAILED) from e
    n = len(sizes)
    out: list[Path] = []
    failed_pages: list[int] = []
    timed_out: list[int] = []
    last_err: Exception | None = None
    for i in range(n):
        p = pages_dir / f"page_{i + 1:04d}.png"
        try:
            pdf_worker.run_page(
                "app.pipeline.pdf:render_page_local", pdf_path, i, (p, dpi, n),
                pool=pool, cancel=should_cancel,
            )
        except pdf_worker.PdfWorkerCanceled:
            _raise_job_canceled()
        except pdf_worker.PdfWorkerTimeout as e:  # 격리 메모(PdfPageQuarantined) 포함
            timed_out.append(i + 1)
            last_err = e
            logger.warning("페이지 %d/%d 렌더가 시간 상한을 넘어 흰색 페이지로 대체 (%s)",
                           i + 1, n, e)
            _write_blank_page(p, sizes[i], dpi)
        except Exception as e:  # noqa: BLE001 — 페이지 단위 격리(워커 비정상 종료 포함)
            failed_pages.append(i + 1)
            last_err = e
            logger.warning("페이지 %d/%d 렌더 실패 (%s: %s) — 흰색 페이지로 대체",
                           i + 1, n, e.__class__.__name__, str(e)[:200])
            _write_blank_page(p, sizes[i], dpi)
        out.append(p)
        if progress_cb:
            progress_cb(i + 1, n)
        if len(timed_out) >= _MAX_RENDER_TIMEOUTS and len(timed_out) < n:
            raise ValueError(
                f"페이지 {len(timed_out)}개의 렌더가 시간 상한({_limit_text(timeout)})을 넘어 처리를 "
                f"중단했습니다 ({_listed_pages(timed_out)}) — 지나치게 복잡한 PDF입니다 "
                "(PDF_PAGE_TIMEOUT_S)"
            ) from last_err
    if len(failed_pages) + len(timed_out) == n:
        raise ValueError(f"모든 페이지({n}) 렌더에 실패했습니다: {last_err}") from last_err
    # 흰 페이지 대체는 조용한 품질 저하다 — 로그만으로는 사용자가 알 수 없어
    # 병합기가 승계할 수 있게 파일로 남긴다 (없으면 파일 삭제 = 경고 없음).
    messages: list[str] = []
    if failed_pages:
        messages.append(
            f"{len(failed_pages)}/{n}페이지 렌더에 실패해 흰 페이지로 대체했습니다 "
            f"({_listed_pages(failed_pages)}) — 해당 페이지의 인식 결과가 비어 있거나 "
            "부정확할 수 있습니다"
        )
    if timed_out:
        messages.append(
            f"{len(timed_out)}/{n}페이지가 렌더 시간 상한({_limit_text(timeout)})을 넘어 흰 페이지로 "
            f"대체했습니다 ({_listed_pages(timed_out)}) — 지나치게 복잡한 페이지라 인식 결과가 "
            "비어 있습니다 (PDF_PAGE_TIMEOUT_S)"
        )
    _write_render_warnings(pages_dir, messages)
    return out


def _raise_job_canceled():
    """워커 취소 → 러너의 취소 계약(JobCanceled). 엔진 계층은 이 경로에서만 지연 임포트한다."""
    from ..engine.base import JobCanceled

    raise JobCanceled() from None
