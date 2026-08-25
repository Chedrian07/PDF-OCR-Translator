"""텍스트 레이어 + 로컬 Tesseract OCR 엔진 (Localight 이식) — torch·모델 불필요.

Localight의 auto 추출(app/services/documents.py의 _native_text/_meaningful_length)과
로컬 Tesseract 실행(app/services/ocr.py)을 B의 엔진 계약(base.py)으로 옮긴
1급 엔진이다. 페이지별 동작:

1) 원본 PDF의 텍스트 레이어를 PyMuPDF ``get_text("blocks", sort=True)``로 추출
2) 영숫자 수가 ``settings.native_text_threshold`` 이상이면 그대로 사용
3) 미만(스캔/이미지 페이지)이면 **이미 렌더된** 페이지 PNG를 로컬 Tesseract로
   OCR (stdin/stdout 파이프 — 네트워크 없음). Tesseract 미설치면 희박한 텍스트
   레이어로 폴백하고 잡당 1회 경고를 남긴다.

출력 규약은 FakeEngine과 동일(base.py 모듈 docstring 참조):
- run_multi는 ``<PAGE>`` 마커로 페이지를 구분해 스트림/반환 (진행률·병합 계약)
- raw_pages.json: 텍스트 블록 bbox(0–999 정규화)로 det 문법을 합성 → 레이아웃 뷰
- figure 크롭 없음 — images/는 빈 디렉터리, boxes.json은 빈 dict
- result_with_boxes 오버레이는 만들지 않는다 (merge가 결측을 조용히 스킵)

문서에서 추출한 텍스트는 스트림 계약을 지키도록 정화한다: 리터럴 ``<PAGE>``와
``<|...|>`` 토큰형 스팬은 마커 카운트(BrokerSink 진행률)·페이지 병합·라이브
파서를 오염시키므로 sanitize_text()로 제거·치환한다
(sidecar/protocol.strip_special_tokens와 같은 이유의 엔진 내 등가물).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .base import EngineCapabilities, EngineError, JobCanceled, OCREngine, StreamSink

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

logger = logging.getLogger(__name__)

_TESSERACT_TIMEOUT_S = 180          # Localight ocr.py와 동일한 페이지당 상한
_VERSION_TIMEOUT_S = 10
_MAX_WARNINGS = 40                  # sidecar.py와 동일한 잡당 경고 적재 상한
_PAGE_NAME = re.compile(r"^page_(\d+)\.png$")   # render_pdf_pages의 page_%04d.png
_SPECIAL_SPAN = re.compile(r"<\|[^|>]{0,64}\|>")
_NO_TESSERACT_WARNING = (
    "Tesseract 미설치 — 텍스트 레이어만 사용했습니다. "
    "brew install tesseract tesseract-lang"
)


# ── 모듈 레벨 재바인딩 심 (테스트가 monkeypatch로 대체한다) ────────────────


def find_tesseract() -> str | None:
    """로컬 tesseract 실행 파일 경로 (없으면 None). 호출 시점마다 조회한다."""
    return shutil.which("tesseract")


def run_tesseract(executable: str, languages: str, png_bytes: bytes) -> str:
    """렌더된 페이지 PNG 바이트를 stdin으로 넘겨 Tesseract OCR 실행.

    Localight app/services/ocr.py의 LocalTesseractOcr.recognize 이식 —
    stdin/stdout 파이프, ``--oem 1``(LSTM), ``--psm 3``(자동 페이지 분할),
    페이지당 180초 상한. 실패는 사용자 노출 가능한 EngineError로 변환한다.
    """
    try:
        process = subprocess.run(
            [executable, "stdin", "stdout", "-l", languages, "--oem", "1", "--psm", "3"],
            input=png_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_TESSERACT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise EngineError(
            f"Tesseract OCR가 페이지당 제한시간({_TESSERACT_TIMEOUT_S}초)을 초과했습니다"
        ) from e
    except OSError as e:  # 실행 파일이 조회 후 사라진 경우 등
        raise EngineError(f"Tesseract 실행 실패: {e}") from e
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise EngineError(f"Tesseract OCR 실패: {message[:200] or '알 수 없는 오류'}")
    return process.stdout.decode("utf-8", errors="replace").strip()


def _probe_tesseract_version() -> str:
    """``tesseract --version`` 첫 줄 — 미설치/실행 실패면 'no-tesseract'."""
    executable = find_tesseract()
    if not executable:
        return "no-tesseract"
    try:
        process = subprocess.run(
            [executable, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_VERSION_TIMEOUT_S,
        )
        # tesseract는 버전에 따라 stdout(5.x) 또는 stderr(3.x)에 출력한다
        lines = (process.stdout or process.stderr).decode("utf-8", errors="replace").strip()
        first = lines.splitlines()[0].strip() if lines else ""
        if first:
            return first[:80]
    except Exception:  # noqa: BLE001 — 버전 조회 실패가 health/잡 생성을 죽이면 안 된다
        pass
    return "no-tesseract"


# ── 텍스트 정화 유틸 ───────────────────────────────────────────────────────

_SANITIZE_ROUNDS = 8  # protocol._STRIP_ROUNDS와 동일 — 정상 문서는 1~2회에 고정점 도달


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


def _meaningful_length(text: str) -> int:
    """공백·구두점을 뺀 실질 글자 수 (Localight documents._meaningful_length 이식)."""
    return sum(character.isalnum() for character in text)


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
        import fitz

        return fitz
    return quiet_fitz()


def _drain_mupdf(context: str) -> None:
    """MuPDF 내부 경고 버퍼 요약 로깅 (best-effort — pipeline.pdf와 동일 패턴)."""
    try:
        from ..pipeline.pdf import drain_mupdf_warnings
    except ImportError:  # pragma: no cover
        return
    drain_mupdf_warnings(context)


class TextLayerEngine(OCREngine):
    """PDF 텍스트 레이어 우선 + Tesseract 폴백 — CPU 전용, 모델 로드 없음."""

    name = "textlayer"

    def __init__(self, settings: "Settings") -> None:
        self.device = "cpu"
        self.dtype_name = "-"
        self._settings = settings
        self._warn_lock = threading.Lock()
        self._warnings: list[str] = []
        # Tesseract 미설치 경고는 **잡당 1회**. 워커는 loaded=True인 엔진의
        # wait_until_ready()를 건너뛰므로 load()를 잡 시작 훅으로 쓸 수 없다 —
        # 대신 잡 디렉터리(이미지 경로에서 유도)를 키로 삼는다. 단일 워커가
        # 잡을 직렬 처리하므로 '마지막으로 경고한 잡'만 기억하면 충분하다.
        self._warned_job_dir: Path | None = None
        self._revision_cache: str | None = None

    # ── 상태/메타 ──────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        # 로드할 모델이 없다 — 항상 준비 상태 (health의 model_loaded=true 즉시)
        return True

    def load(self) -> None:
        """no-op — 멱등·스레드 세이프(상태 변경 없음). 프리로드/워커가 호출해도 무해."""

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            model_id="pymupdf-textlayer+tesseract",
            model_revision=self._tesseract_revision(),
            provider="in-process",
            supports_multi_page=True,       # 청크를 스스로 페이지별 처리한다
            preferred_chunk_size=None,      # settings.pages_per_chunk 그대로 사용
            stream_granularity="page",      # 페이지 완료 시 일괄 발행 (가짜 토큰 스트림 없음)
            layout_capability="full",       # 텍스트 레이어 bbox로 raw_pages.json 합성
            figure_capability=False,        # figure 크롭 없음 (boxes.json은 빈 dict)
        )

    def _tesseract_revision(self) -> str:
        """capabilities()는 health 폴링·잡 생성마다 불린다 — 프로세스 기동을 매번
        하지 않도록 1회 계산 후 캐시 (설치 변경 반영은 서버 재시작)."""
        if self._revision_cache is None:
            self._revision_cache = _probe_tesseract_version()
        return self._revision_cache

    # ── 경고 채널 (sidecar.py와 동일 패턴) ─────────────────────

    def _note(self, message: str) -> None:
        with self._warn_lock:
            if len(self._warnings) < _MAX_WARNINGS:
                self._warnings.append(message)

    def drain_warnings(self) -> list[str]:
        """쌓인 경고를 비우며 반환 — 반복된 동일 경고는 1건으로 접는다 (순서 보존)."""
        with self._warn_lock:
            drained, self._warnings = self._warnings, []
        seen: set[str] = set()
        unique: list[str] = []
        for warning in drained:
            if warning not in seen:
                seen.add(warning)
                unique.append(warning)
        return unique

    def _warn_no_tesseract(self, job_dir: Path) -> None:
        """미설치 경고 적재 — 같은 잡(디렉터리)에서는 한 번만."""
        with self._warn_lock:
            if self._warned_job_dir == job_dir:
                return
            self._warned_job_dir = job_dir
        self._note(_NO_TESSERACT_WARNING)

    # ── 페이지 추출 ────────────────────────────────────────────

    def _open_source(self, source_pdf: Path):
        """원본 PDF 열기 — 실패해도 잡을 죽이지 않는다 (텍스트 레이어 없이 진행)."""
        if not source_pdf.is_file():
            logger.warning("source.pdf가 없어 텍스트 레이어를 건너뜁니다: %s", source_pdf)
            return None
        fitz = _fitz()
        try:
            doc = fitz.open(str(source_pdf))
        except Exception as e:  # noqa: BLE001 — 렌더는 이미 성공했다, best-effort
            logger.warning("source.pdf 열기 실패 (%s) — 텍스트 레이어 없이 진행", str(e)[:200])
            return None
        if doc.needs_pass:  # 업로드 probe가 거르지만 방어적으로
            doc.close()
            return None
        return doc

    def _native_blocks(self, doc, image_path: Path) -> list[tuple[tuple[int, int, int, int] | None, str]]:
        """텍스트 레이어 블록 추출 (Localight documents._native_text 이식) + bbox 정규화.

        전역 페이지 번호(1-based)는 렌더 파일명 page_%04d.png에서 파싱한다.
        반환: (0–999 정규화 bbox — 퇴화면 None, 정화된 블록 텍스트) 목록.
        """
        if doc is None:
            return []
        match = _PAGE_NAME.match(image_path.name)
        if match is None:  # 규약 밖 파일명 — 텍스트 레이어 매칭 불가, OCR 경로로
            logger.warning("페이지 번호를 파일명에서 파싱하지 못했습니다: %s", image_path.name)
            return []
        page_number = int(match.group(1))
        if not 1 <= page_number <= doc.page_count:
            return []
        try:
            page = doc[page_number - 1]
            rect = page.rect
            # get_text("blocks") 좌표는 /Rotate 이전(비회전) 공간이고, rect와
            # 렌더된 page_%04d.png는 회전 반영 공간이다 — rotation_matrix로
            # 사상해야 레이아웃 뷰의 det 박스가 렌더 이미지와 정렬된다
            # (회전 0이면 항등행렬이라 무해).
            matrix = page.rotation_matrix
            raw_blocks = page.get_text("blocks", sort=True)
        except Exception as e:  # noqa: BLE001 — 손상 페이지는 OCR 경로로 격리
            logger.warning("%d페이지 텍스트 레이어 추출 실패 (%s: %s)",
                           page_number, e.__class__.__name__, str(e)[:200])
            return []
        fitz = _fitz()
        blocks: list[tuple[tuple[int, int, int, int] | None, str]] = []
        for block in raw_blocks:
            if len(block) < 7 or block[6] != 0:  # 6번 필드 0 = 텍스트 블록 (이미지 제외)
                continue
            text = sanitize_text(str(block[4])).strip()
            if not text:
                continue
            blocks.append((_norm_bbox(fitz.Rect(block[:4]) * matrix, rect), text))
        return blocks

    def _extract_page(self, doc, image_path: Path, job_dir: Path) -> tuple[str, str]:
        """페이지 1장 → (페이지 마크다운 텍스트, 레이아웃 뷰용 raw det 문법).

        Localight extract_page의 auto 모드: 텍스트 레이어의 영숫자 수가
        native_text_threshold 이상이면 그대로, 미만이면 Tesseract OCR.
        레이아웃 raw는 어느 경로든 텍스트 레이어 블록에서만 합성한다
        (Tesseract 출력에는 bbox가 없다 — 스캔 문서의 레이아웃 뷰는 빈 페이지).
        """
        blocks = self._native_blocks(doc, image_path)
        native_text = "\n\n".join(text for _, text in blocks).strip()
        raw = "\n".join(
            f"<|det|>text [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]<|/det|>{text}"
            for bbox, text in blocks
            if bbox is not None
        )
        if _meaningful_length(native_text) >= self._settings.native_text_threshold:
            return native_text, raw

        # 텍스트 레이어가 희박함(스캔/이미지 페이지) → 렌더된 PNG를 Tesseract로 OCR
        executable = find_tesseract()
        if executable is None:
            self._warn_no_tesseract(job_dir)
            return native_text, raw
        try:
            ocr_text = run_tesseract(
                executable, self._settings.ocr_languages, image_path.read_bytes()
            )
        except (EngineError, OSError) as e:
            # 페이지 단위 강등 — 예외를 청크로 흘리면 같은 청크의 정상 페이지까지
            # 통째로 플레이스홀더가 된다(언어팩 누락·페이지 타임아웃·PNG 손상).
            self._note(
                f"{image_path.name} Tesseract 실패로 텍스트 레이어만 사용했습니다 "
                f"({str(e)[:160]}) — 언어팩 확인: brew install tesseract-lang / "
                "apt install tesseract-ocr-kor"
            )
            return native_text, raw
        # OCR 텍스트를 쓰는 페이지의 레이아웃 raw는 버린다 — 그 det 문법은 본문으로
        # 채택되지 않은 희박 텍스트 레이어에서 합성된 것이라 result.md와 어긋난다
        # (Tesseract 출력에는 bbox가 없다 — 스캔 페이지의 레이아웃 뷰는 빈 페이지).
        return sanitize_text(ocr_text).strip(), ""

    # ── OCREngine 구현 ─────────────────────────────────────────

    def _run_pages(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
        single: bool,
    ) -> list[str]:
        # figure를 만들지 않지만 산출물 규약은 유지: 빈 images/ + 빈 boxes.json
        # (merge._move_chunk_files/_load_chunk_boxes가 조용히 스킵한다)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "boxes.json").write_text("{}", encoding="utf-8")

        # 잡 디렉터리 규약 가정: runner는 항상 {job.dir}/pages/page_%04d.png 경로를
        # 넘긴다(render_pdf_pages 산출물) — 원본 PDF는 그 두 단계 위인
        # {job.dir}/source.pdf. per-page fallback(run_single)도 이미지 경로는
        # pages/ 그대로라 동일하게 성립한다.
        job_dir = image_paths[0].parent.parent
        source_pdf = job_dir / "source.pdf"

        doc = self._open_source(source_pdf)
        texts: list[str] = []
        raws: list[str] = []
        try:
            for image_path in image_paths:
                if cancel.is_set():
                    raise JobCanceled()
                page_text, raw = self._extract_page(doc, image_path, job_dir)
                if single:
                    sink.on_text(page_text)
                else:
                    # 페이지 단위 스트림: 페이지당 정확히 마커 1개 + 전체 텍스트
                    # (BrokerSink의 <PAGE> 진행률 계약 — fake.py와 동일 형식)
                    sink.on_text("<PAGE>\n" + page_text + "\n")
                texts.append(page_text)
                raws.append(raw)
        finally:
            if doc is not None:
                doc.close()
            _drain_mupdf("textlayer 추출")
        (out_dir / "raw_pages.json").write_text(
            json.dumps({"pages": raws}, ensure_ascii=False), encoding="utf-8"
        )
        return texts

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        pages = self._run_pages(image_paths, out_dir, sink, cancel, single=False)
        return "<PAGE>\n" + "\n<PAGE>\n".join(pages)

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        return self._run_pages([image_path], out_dir, sink, cancel, single=True)[0]
