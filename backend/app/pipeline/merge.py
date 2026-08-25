"""청크 단위 모델 출력을 잡 단위 최종 결과로 병합.

입력 규약(엔진 출력, base.py 참조):
- multi 청크: `<PAGE>` 구분 마크다운 + chunk_dir/images/page_{i}_{k}.jpg
  + chunk_dir/result_with_boxes_{i}.jpg  (i는 청크 내 0-based)
- single 청크: 페이지 1장 마크다운 + chunk_dir/images/{k}.jpg
  + chunk_dir/result_with_boxes.jpg

출력(잡 디렉터리, ARCHITECTURE.md §4):
- images/p{글로벌페이지:04d}_{k}.jpg  (1-based)
- layout/page_{글로벌페이지:04d}.jpg
- result.md — 페이지들을 page_separator로 join (청크 완료 시마다 부분 갱신)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_IMG_MULTI = re.compile(r"!\[\]\(images/page_(\d+)_(\d+)\.jpg\)")
_IMG_SINGLE = re.compile(r"!\[\]\(images/(\d+)\.jpg\)")
_FILE_MULTI = re.compile(r"^page_(\d+)_(\d+)\.jpg$")
_FILE_SINGLE = re.compile(r"^(\d+)\.jpg$")
_BOXES_FILE = re.compile(r"^result_with_boxes_(\d+)\.jpg$")
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]{0,64}\|>")


def split_pages(markdown: str) -> list[str]:
    """`<PAGE>` 마커로 분리. 첫 마커 이전의 공백은 버린다."""
    parts = markdown.split("<PAGE>")
    if parts and not parts[0].strip():
        parts = parts[1:]
    return [p.strip() for p in parts]


def _global_image_name(global_page: int, k: int | str) -> str:
    return f"p{global_page:04d}_{k}.jpg"


def _neutralized_marker(core: str) -> str:
    """구분자 코어 줄을 렌더 결과는 유지하되 리터럴 일치는 깨는 형태로 바꾼다."""
    body = core.strip()
    if len(body) >= 3 and body[0] in "-*_" and set(body) == {body[0]}:
        # 마크다운 구분선 — 다른 구분선 문자로 바꿔도 렌더 결과(수평선)가 같다
        alt = "*" if body[0] != "*" else "-"
        return core.replace(body, alt * len(body))
    return core + " "  # 그 외 구분자: 후행 공백으로 리터럴 일치만 깬다


def _neutralize_separator(page_md: str, page_separator: str) -> str:
    """페이지 본문에서 page_separator와 리터럴로 충돌하는 줄을 무해화한다.

    result.md는 "N페이지 = 구분자 N-1개" 계약을 지켜야 한다 — qa.get_page_context,
    render_document_html의 doc-page 분할, 번역 조립이 전부 이 split에 의존한다.
    OCR이 각주선·구분선을 `---` 한 줄로 뱉기만 해도 그 뒤 모든 페이지 인덱스가
    경고 없이 밀려 엉뚱한 페이지가 Q&A 컨텍스트로 들어간다.

    실제로 충돌할 수 있는 줄만 바꾼다 — 구분자가 요구하는 빈 줄 패딩까지 갖춘
    경우. 덕분에 setext 제목의 밑줄(`제목` 다음 줄의 `---`)은 건드리지 않는다.
    """
    core = page_separator.strip("\n")
    if not core or "\n" in core:
        # 여러 줄짜리 코어를 가진 구분자는 줄 단위 무해화 대상이 아니다
        return page_md
    safe = _neutralized_marker(core)
    if safe == core:
        return page_md
    lines = page_md.split("\n")
    prev_needs_blank = page_separator.startswith("\n\n")
    next_needs_blank = page_separator.endswith("\n\n")
    out = list(lines)
    for i, line in enumerate(lines):
        if line != core:
            continue
        # 페이지 양끝은 join이 구분자의 빈 줄을 붙여 주므로 "빈 줄"로 친다
        prev_ok = not prev_needs_blank or i == 0 or not lines[i - 1].strip()
        next_ok = not next_needs_blank or i == len(lines) - 1 or not lines[i + 1].strip()
        if prev_ok and next_ok:
            out[i] = safe
    return "\n".join(out)


def _clean(page_md: str, page_separator: str = "") -> str:
    page_md = _SPECIAL_TOKEN.sub("", page_md).strip()
    if page_separator:
        page_md = _neutralize_separator(page_md, page_separator)
    return page_md


def _atomic_write_json(path: Path, obj, indent: int | None = None) -> None:
    """tmp에 쓰고 원자적으로 교체 — 크래시 타이밍에도 파손 파일이 남지 않는다.
    단일 워커 스레드 전용이라 고정 tmp 이름으로 충분하다 (API 스레드의 폰트
    백필은 요청별 고유 tmp를 쓰므로 이름이 겹치지 않는다)."""
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class ChunkResult:
    chunk_dir: Path
    start_page: int  # 글로벌 1-based
    num_pages: int
    markdown: str
    single: bool = False


class IncrementalMerger:
    def __init__(self, job_dir: Path, page_separator: str) -> None:
        self.job_dir = job_dir
        self.page_separator = page_separator
        self.images_dir = job_dir / "images"
        self.layout_dir = job_dir / "layout"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.layout_dir.mkdir(parents=True, exist_ok=True)
        self.pages_md: list[str] = []
        self.warnings: list[str] = []
        # 글로벌 이미지명 → figure bbox 메타 (벤더 P13의 boxes.json — 렌더 폭 계산용)
        self.figure_boxes: dict[str, dict] = {}
        # 레이아웃 뷰용 페이지 블록 (벤더 P14의 raw_pages.json → layout.json)
        self.layout_pages: list[dict] = []
        # 좌표 데이터를 실제로 받은 청크가 하나라도 있었는가 (없으면 layout.json 미생성)
        self.has_layout_data = False

    # ── 파일 이동 ──────────────────────────────────────────────

    def _load_chunk_boxes(self, chunk: ChunkResult) -> dict:
        p = chunk.chunk_dir / "boxes.json"
        if not p.is_file():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _move_chunk_files(self, chunk: ChunkResult) -> None:
        chunk_boxes = self._load_chunk_boxes(chunk)
        img_src = chunk.chunk_dir / "images"
        if img_src.is_dir():
            for f in sorted(img_src.iterdir()):
                if chunk.single:
                    m = _FILE_SINGLE.match(f.name)
                    if not m:
                        continue
                    dest = self.images_dir / _global_image_name(chunk.start_page, m.group(1))
                else:
                    m = _FILE_MULTI.match(f.name)
                    if not m:
                        continue
                    local_page, k = int(m.group(1)), m.group(2)
                    dest = self.images_dir / _global_image_name(chunk.start_page + local_page, k)
                meta = chunk_boxes.get(f.name)
                if isinstance(meta, dict):
                    self.figure_boxes[dest.name] = meta
                shutil.move(str(f), str(dest))
        self._write_boxes()

        if chunk.single:
            boxes = chunk.chunk_dir / "result_with_boxes.jpg"
            if boxes.is_file():
                shutil.move(str(boxes), str(self.layout_dir / f"page_{chunk.start_page:04d}.jpg"))
        else:
            for f in sorted(chunk.chunk_dir.iterdir()):
                m = _BOXES_FILE.match(f.name)
                if m:
                    g = chunk.start_page + int(m.group(1))
                    shutil.move(str(f), str(self.layout_dir / f"page_{g:04d}.jpg"))

    def _write_boxes(self) -> None:
        if self.figure_boxes:
            # 원자적 교체 — API 스레드(/html의 _load_figure_boxes)가 기록 중인 파일을
            # 반쯤 읽는 일이 없게 (비원자 기록은 torn read → 풀폭 폴백을 유발했다)
            _atomic_write_json(self.images_dir / "boxes.json", self.figure_boxes, indent=1)

    # ── 레이아웃 뷰 (Phase B — 부가 산출물, 결측 시 조용히 스킵) ─────────

    def _page_size(self, global_page: int) -> tuple[int, int]:
        p = self.job_dir / "pages" / f"page_{global_page:04d}.png"
        try:
            from PIL import Image

            with Image.open(p) as im:
                return im.size
        except Exception:
            return (1000, 1414)  # A4 비율 폴백

    def _load_raw_pages(self, chunk: ChunkResult) -> list:
        raw_path = chunk.chunk_dir / "raw_pages.json"
        if not raw_path.is_file():
            return []
        try:
            raw_pages = json.loads(raw_path.read_text(encoding="utf-8"))["pages"]
        except Exception:
            return []
        if not isinstance(raw_pages, list) or not raw_pages:
            return []
        self.has_layout_data = True
        return raw_pages

    def _ingest_layout(self, chunk: ChunkResult) -> None:
        """청크의 모든 페이지를 layout_pages에 반영한다.

        raw_pages.json이 없거나(텍스트 레이어 복구 페이지 등) 페이지 수가 모자라면
        빈 블록 페이지로 채운다 — layout.json의 페이지 목록이 result.md의 페이지와
        1:1로 대응해야 facsimile 내보내기에서 페이지가 통째로 사라지지 않고
        Q&A·문서 뷰의 페이지 번호와도 일치한다.
        """
        from .layout import parse_page_blocks

        raw_pages = self._load_raw_pages(chunk)
        new_pages: list[dict] = []
        for local in range(chunk.num_pages):
            g = chunk.start_page + (0 if chunk.single else local)
            raw = raw_pages[local] if local < len(raw_pages) else None
            blocks = parse_page_blocks(str(raw)) if raw is not None else []
            for b in blocks:
                if "crop_index" in b:
                    # 벤더 크롭 순서 == boxes/이미지 저장 순서 → 글로벌 이미지명 매핑
                    b["image"] = _global_image_name(g, b.pop("crop_index"))
            w, h = self._page_size(g)
            page = {"page": g, "width": w, "height": h, "blocks": blocks}
            new_pages.append(page)
            self.layout_pages.append(page)
        # 원본 PDF 텍스트 레이어의 실측 폰트 크기를 이번 청크 페이지들에 주입.
        # (청크마다 pdf 재오픈 — ms 수준이라 무방. enrichment 실패는 잡을 깨지 않음.)
        try:
            from .pdf_fonts import enrich_layout_fonts

            enrich_layout_fonts(self.job_dir / "source.pdf", new_pages)
        except Exception:
            pass
        if self.has_layout_data and self.layout_pages:
            # 좌표 데이터를 한 번도 받지 못한 잡(figure_only 엔진)은 layout.json을
            # 만들지 않는다 — has_layout=false로 남아야 레이아웃 뷰/PDF 내보내기가
            # 빈 캔버스를 제안하지 않는다.
            # 원자적 교체 — 크래시/재시작 타이밍에 layout.json이 파손된 채 남아
            # /layout이 500을 내는 일이 없게 (result.md의 _write_partial과 동일 패턴)
            _atomic_write_json(self.job_dir / "layout.json", self.layout_pages)

    # ── 마크다운 재작성 ────────────────────────────────────────

    def _rewrite_refs(self, page_md: str, chunk: ChunkResult) -> str:
        if chunk.single:
            return _IMG_SINGLE.sub(
                lambda m: f"![](images/{_global_image_name(chunk.start_page, m.group(1))})",
                page_md,
            )
        return _IMG_MULTI.sub(
            lambda m: f"![](images/{_global_image_name(chunk.start_page + int(m.group(1)), m.group(2))})",
            page_md,
        )

    # ── 공개 API ───────────────────────────────────────────────

    def add_chunk(self, chunk: ChunkResult) -> None:
        pages = [chunk.markdown] if chunk.single else split_pages(chunk.markdown)

        if len(pages) > chunk.num_pages:
            # 마커가 초과 생성됨 — 초과분을 마지막 페이지에 합침
            self.warnings.append(
                f"{chunk.start_page}페이지 청크: 페이지 마커 {len(pages)}개 (기대 {chunk.num_pages}) — 초과분 병합"
            )
            head = pages[: chunk.num_pages - 1]
            tail = "\n\n".join(pages[chunk.num_pages - 1 :])
            pages = head + [tail]
        elif len(pages) < chunk.num_pages:
            self.warnings.append(
                f"{chunk.start_page}페이지 청크: 페이지 마커 {len(pages)}개 (기대 {chunk.num_pages}) — 빈 페이지로 보정"
            )
            pages = pages + [""] * (chunk.num_pages - len(pages))

        pages = [self._rewrite_refs(p, chunk) for p in pages]
        self._move_chunk_files(chunk)
        self._ingest_layout(chunk)
        self.pages_md.extend(_clean(p, self.page_separator) for p in pages)
        self._write_partial()

    @property
    def markdown(self) -> str:
        # 전체 strip() 금지: 선두/말미 페이지가 비면(스캔 문서의 빈 표지 등)
        # 구분자의 공백 절반까지 먹혀 페이지가 통째로 사라진다 — result.md의
        # 페이지 수 계약(N페이지 = 구분자 N-1개)이 깨져 Q&A·번역·/html 문서 뷰의
        # 페이지 인덱스가 밀린다. 페이지별 양끝 공백은 _clean()이 이미 제거했다.
        return self.page_separator.join(self.pages_md) + "\n"

    def _write_partial(self) -> None:
        tmp = self.job_dir / ".result.md.tmp"
        tmp.write_text(self.markdown, encoding="utf-8")
        os.replace(tmp, self.job_dir / "result.md")

    def finalize(self) -> str:
        self._write_partial()
        md = self.markdown
        # 페이지 경계 계약 검증(N페이지 = 구분자 N-1개). _clean의 무해화가 놓친
        # 케이스를 조용히 넘기지 않고 경고로 남긴다 — 밀린 인덱스는 Q&A·문서 뷰가
        # "확신에 찬 오답"을 내는 형태로만 드러나기 때문.
        if self.page_separator and self.pages_md:
            got = len(md.split(self.page_separator))
            if got != len(self.pages_md):
                self.warnings.append(
                    f"result.md 페이지 경계 불일치: 분할 {got}개 ≠ 페이지 {len(self.pages_md)}개"
                )
        return md
