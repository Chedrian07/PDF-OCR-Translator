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
# 코드펜스(``` / ~~~) 여닫이 줄 — 최대 3칸 들여쓰기까지 펜스로 인정(CommonMark)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# 렌더 단계가 남긴 경고를 잡 경고로 승계할 때의 상한 (runner의 40칸 예산 보호)
_MAX_RENDER_WARNINGS = 5

# ── 모델 페이지 → 물리 페이지 정합 ────────────────────────────────────────
# 마커 과부족은 꼬리에서만 생기지 않는다. 모델이 한 페이지를 둘로 쪼개거나 한
# 페이지를 통째로 건너뛰면 그 **뒤의 모든 페이지**가 한 칸씩 밀린다. 개수만 맞추면
# result.md와 layout.json이 엉뚱한 물리 페이지를 가리키고, PDF 내보내기는 그
# 좌표를 믿고 원문을 영구 리댁션한다(실측: 46p 논문에서 layout 6개 페이지가 밀려
# 23쪽의 프로젝트명 29개가 삭제되고 24쪽 캡션이 그 자리에 찍혔다).
# 그래서 원본 PDF 텍스트 레이어와 내용을 대조해 위치를 정한다.
_ALIGN_PROBE_CHARS = 32     # 대조 프로브 길이
_ALIGN_MIN_PROBE_SRC = 40   # 이보다 짧은 모델 페이지는 정합 판정 대상이 아니다
_ALIGN_MIN_SCORE = 0.34     # 프로브 적중률이 이 미만이면 매칭으로 인정하지 않는다
_ALIGN_MIN_PAGE_CHARS = 80  # 물리 페이지 텍스트가 이보다 짧으면 텍스트 레이어 없음
_NON_WORD = re.compile(r"\W+", re.UNICODE)


def _align_norm(text: str) -> str:
    """대조용 정규화 — 공백·구두점·대소문자 차이를 지운다."""
    return _NON_WORD.sub("", _SPECIAL_TOKEN.sub("", text or "")).lower()


def _probe_score(model_text: str, page_text: str) -> float:
    """모델 페이지 본문이 이 물리 페이지에서 얼마나 확인되는가 (0..1)."""
    if len(model_text) < _ALIGN_MIN_PROBE_SRC or not page_text:
        return 0.0
    step = _ALIGN_PROBE_CHARS
    probes = [
        model_text[i : i + step]
        for i in range(0, max(1, len(model_text) - step + 1), step)
    ]
    probes = [p for p in probes if len(p) == step][:24]  # 앞부분 표본으로 충분
    if not probes:
        return 0.0
    return sum(1 for p in probes if p in page_text) / len(probes)


def align_model_pages(model_texts: list[str], page_texts: list[str]) -> list[int | None]:
    """모델이 낸 k번째 페이지 → 물리 페이지의 로컬 인덱스 (없으면 None).

    두 수열 모두 문서 순서라 단조 정합이다. 양쪽 건너뛰기를 허용하는 DP로
    총 적중률을 최대화한다. 순수 함수 — tests/test_merge.py에서 직접 검증한다.
    """
    n, m = len(model_texts), len(page_texts)
    if not n or not m:
        return [None] * n
    score = [
        [_probe_score(model_texts[k], page_texts[i]) for i in range(m)] for k in range(n)
    ]
    NEG = float("-inf")
    # best[k][i] = 모델 0..k-1을 물리 0..i-1에 단조 배치했을 때의 최대 점수
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        for i in range(m + 1):
            if k == 0 and i == 0:
                continue
            cand = []
            if k > 0:
                cand.append((best[k - 1][i], ("skip_model", k - 1, None)))  # 모델 페이지 미배치
            if i > 0:
                cand.append((best[k][i - 1], ("skip_page", None, i - 1)))   # 물리 페이지 비움
            if k > 0 and i > 0 and score[k - 1][i - 1] >= _ALIGN_MIN_SCORE:
                cand.append((
                    best[k - 1][i - 1] + score[k - 1][i - 1],
                    ("match", k - 1, i - 1),
                ))
            if not cand:
                best[k][i] = NEG
                continue
            best[k][i], back[k][i] = max(cand, key=lambda c: c[0])
    out: list[int | None] = [None] * n
    k, i = n, m
    while k > 0 or i > 0:
        step = back[k][i]
        if step is None:
            break
        kind, mk, pi = step
        if kind == "match":
            out[mk] = pi
            k, i = k - 1, i - 1
        elif kind == "skip_model":
            k -= 1
        else:
            i -= 1
    return out


def place_by_alignment(
    items: list, mapping: list[int | None], num_pages: int, joiner: str
) -> list:
    """정합 결과대로 모델 페이지들을 물리 페이지 자리에 배치한다.

    매칭되지 않은 모델 페이지는 **버리지 않고** 바로 앞의 매칭 페이지에 붙인다
    (앞이 없으면 첫 페이지). 내용 손실 없이 물리 페이지 수를 정확히 지킨다.
    """
    slots: list[list] = [[] for _ in range(num_pages)]
    cursor = 0
    for k, item in enumerate(items):
        target = mapping[k] if k < len(mapping) else None
        if target is None:
            target = cursor
        else:
            cursor = target
        if 0 <= target < num_pages:
            slots[target].append(item)
    return [joiner.join(str(x) for x in s) if s else "" for s in slots]


def split_pages(markdown: str) -> list[str]:
    """`<PAGE>` 마커로 분리. 첫 마커 이전의 공백은 버린다."""
    parts = markdown.split("<PAGE>")
    if parts and not parts[0].strip():
        parts = parts[1:]
    return [p.strip() for p in parts]


def _global_image_name(global_page: int, k: int | str) -> str:
    return f"p{global_page:04d}_{k}.jpg"


def _fold_local_page(chunk: "ChunkResult", local_page: int) -> tuple[int, str]:
    """청크 내 로컬 페이지 → (글로벌 페이지, 파일명 접두사).

    모델이 마커를 초과 생성하면 add_chunk가 초과 페이지를 마지막 페이지로 합친다.
    파일 이름도 같은 규칙으로 접어야 한다 — 접지 않으면 초과분이
    `start_page + local`로 계산돼 **다음 청크의 글로벌 페이지 네임스페이스**를
    침범하고, 다음 청크가 같은 이름으로 덮어써 엉뚱한 그림이 표시된다.
    접힌 파일은 접두사(`x{local}_`)로 구분해 청크 안에서도 충돌하지 않게 한다.
    """
    last = max(0, chunk.num_pages - 1)
    if local_page <= last:
        return chunk.start_page + local_page, ""
    return chunk.start_page + last, f"x{local_page}_"


def _neutralized_marker(core: str) -> str:
    """구분자 코어 줄을 렌더 결과는 유지하되 리터럴 일치는 깨는 형태로 바꾼다."""
    body = core.strip()
    if len(body) >= 3 and body[0] in "-*_" and set(body) == {body[0]}:
        # 마크다운 구분선 — 다른 구분선 문자로 바꿔도 렌더 결과(수평선)가 같다
        alt = "*" if body[0] != "*" else "-"
        return core.replace(body, alt * len(body))
    return core + " "  # 그 외 구분자: 후행 공백으로 리터럴 일치만 깬다


def _fenced_lines(lines: list[str]) -> list[bool]:
    """각 줄이 코드펜스 **내부**인지 여부 (펜스 줄 자체는 False).

    펜스 안의 텍스트는 렌더 결과가 곧 원문이라 `---`→`***` 치환이 곧 내용 변조다
    (YAML 문서 구분자·구분선 예제가 조용히 깨진다). 페이지 단위로 판정하므로
    닫히지 않은 펜스는 그 페이지 끝까지만 영향을 준다.
    """
    inside: list[bool] = []
    opener: str | None = None
    for line in lines:
        m = _FENCE.match(line)
        if opener is None:
            # 여는 펜스: 백틱 펜스의 info string에는 백틱이 올 수 없다(CommonMark)
            if m is not None and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                opener = m.group(1)
            inside.append(False)
            continue
        closing = (
            m is not None
            and m.group(1)[0] == opener[0]
            and len(m.group(1)) >= len(opener)
            and not m.group(2).strip()      # 닫는 펜스는 info string을 갖지 않는다
        )
        inside.append(not closing)
        if closing:
            opener = None
    return inside


def _neutralize_separator(page_md: str, page_separator: str) -> str:
    """페이지 본문에서 page_separator와 리터럴로 충돌하는 줄을 무해화한다.

    result.md는 "N페이지 = 구분자 N-1개" 계약을 지켜야 한다 — qa.get_page_context,
    render_document_html의 doc-page 분할, 번역 조립이 전부 이 split에 의존한다.
    OCR이 각주선·구분선을 `---` 한 줄로 뱉기만 해도 그 뒤 모든 페이지 인덱스가
    경고 없이 밀려 엉뚱한 페이지가 Q&A 컨텍스트로 들어간다.

    실제로 충돌할 수 있는 줄만 바꾼다 — 구분자가 요구하는 빈 줄 패딩까지 갖춘
    경우. 덕분에 setext 제목의 밑줄(`제목` 다음 줄의 `---`)은 건드리지 않는다.

    코드펜스 **안**에서는 구분선 문자 치환(`---`→`***`)을 쓰지 않는다 — 펜스 안은
    렌더 결과가 곧 원문이라 문자를 바꾸면 YAML 문서 구분자·구분선 예제가 조용히
    깨진다(result.md 포터빌리티 계약 위반). 대신 후행 공백만 붙여 리터럴 일치를
    깬다 — 렌더·YAML 의미는 보존되면서 페이지 경계 불변식은 그대로 지켜진다.
    """
    core = page_separator.strip("\n")
    if not core or "\n" in core:
        # 여러 줄짜리 코어를 가진 구분자는 줄 단위 무해화 대상이 아니다
        return page_md
    safe = _neutralized_marker(core)
    if safe == core:
        return page_md
    lines = page_md.split("\n")
    fenced = _fenced_lines(lines)
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
            out[i] = core + " " if fenced[i] else safe
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


def _render_warnings(job_dir: Path) -> list[str]:
    """렌더 단계(pdf.render_pdf_pages)가 남긴 경고를 잡 경고로 승계한다.

    페이지 렌더 실패는 흰 페이지로 격리되는데, 렌더는 잡 경고 채널(=이 병합기)이
    만들어지기 **전**에 끝나므로 pages/render_warnings.json이 유일한 인계 지점이다.
    승계하지 않으면 흰 페이지에서 나온 빈 결과가 quality.state="ok"로 남아
    사용자가 정상 변환으로 오인한다.
    """
    path = job_dir / "pages" / "render_warnings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(m) for m in data[:_MAX_RENDER_WARNINGS]]


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
        # 렌더 단계의 경고(흰 페이지 대체 등)를 잡 경고의 첫 항목으로 승계한다
        self.warnings: list[str] = _render_warnings(job_dir)
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
                    global_page, prefix = _fold_local_page(chunk, local_page)
                    dest = self.images_dir / _global_image_name(global_page, f"{prefix}{k}")
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
                    local_page = int(m.group(1))
                    if local_page >= chunk.num_pages:
                        continue  # 초과 생성분 — 마지막 페이지의 오버레이를 덮지 않는다
                    shutil.move(
                        str(f),
                        str(self.layout_dir / f"page_{chunk.start_page + local_page:04d}.jpg"),
                    )

    def _write_boxes(self) -> None:
        # 비었을 때도 파일이 이미 있으면 갱신해야 한다 — 마지막 그림이 사라진 뒤
        # (replace_page의 _drop_page_artifacts) 옛 항목이 그대로 남으면 레이아웃
        # 뷰가 없는 크롭을 그리려 한다.
        if self.figure_boxes or (self.images_dir / "boxes.json").is_file():
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

    def _source_page_texts(self, chunk: "ChunkResult") -> list[str]:
        """이 청크가 덮는 물리 페이지들의 원본 텍스트 (정합 대조용, 정규화됨).

        텍스트 레이어가 없는 스캔 PDF면 빈 목록 — 호출자는 기존 위치 기반
        동작으로 안전하게 되돌아간다."""
        try:
            import fitz

            with fitz.open(self.job_dir / "source.pdf") as doc:
                texts = []
                for local in range(chunk.num_pages):
                    idx = chunk.start_page - 1 + local
                    raw = doc[idx].get_text() if 0 <= idx < doc.page_count else ""
                    texts.append(_align_norm(raw))
        except Exception:
            return []
        return texts if any(len(t) >= _ALIGN_MIN_PAGE_CHARS for t in texts) else []

    def _align_chunk_pages(
        self, chunk: "ChunkResult", model_pages: list[str]
    ) -> list[int | None] | None:
        """모델 페이지 → 물리 페이지 정합. 정합할 수 없으면 None(기존 동작 유지)."""
        if chunk.single or chunk.num_pages <= 1 or len(model_pages) <= 1:
            return None
        page_texts = self._source_page_texts(chunk)
        if not page_texts:
            return None
        mapping = align_model_pages([_align_norm(p) for p in model_pages], page_texts)
        if not any(m is not None for m in mapping):
            return None  # 대조가 전부 실패 — 근거 없는 재배치는 하지 않는다
        return mapping

    def _ingest_layout(
        self, chunk: ChunkResult, raw_pages: list, slot_source: list | None = None,
    ) -> None:
        """청크의 모든 페이지를 layout_pages에 반영한다.

        raw_pages는 물리 페이지 자리에 이미 정합된 목록이다(add_chunk가 markdown과
        **같은 매핑**으로 배치한다) — layout.json의 페이지가 result.md의 페이지와,
        그리고 원본 PDF의 물리 페이지와 1:1로 대응해야 facsimile 내보내기가 엉뚱한
        페이지의 원문을 지우지 않는다.
        """
        from .layout import parse_page_blocks

        new_pages: list[dict] = []
        for local in range(chunk.num_pages):
            g = chunk.start_page + (0 if chunk.single else local)
            raw = raw_pages[local] if local < len(raw_pages) else None
            blocks = parse_page_blocks(str(raw)) if raw else []
            # 크롭 파일명은 **모델 페이지 인덱스**로 지어졌다(_move_chunk_files와 동일
            # 기준). 정렬이 페이지를 옮겼으면 물리 슬롯이 아니라 그 인덱스를 따라야
            # layout의 참조와 실제 파일이 일치한다.
            source = local if slot_source is None else slot_source[local] if (
                local < len(slot_source)
            ) else None
            image_page = g if chunk.single or source is None else (
                _fold_local_page(chunk, source)[0]
            )
            image_prefix = "" if chunk.single or source is None else (
                _fold_local_page(chunk, source)[1]
            )
            for b in blocks:
                if "crop_index" in b:
                    # 벤더 크롭 순서 == boxes/이미지 저장 순서 → 글로벌 이미지명 매핑
                    b["image"] = _global_image_name(
                        image_page, f"{image_prefix}{b.pop('crop_index')}"
                    )
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

    # ── 페이지 단위 교체 (충실도 게이트의 복구 경로) ─────────────

    def _drop_page_artifacts(self, page_number: int) -> None:
        """이 페이지가 앞서 남긴 그림·박스 항목을 지운다.

        재실행 결과가 원본보다 그림을 **적게** 낼 수 있다. 지우지 않으면 아무도
        참조하지 않는 `p0038_1.jpg`가 남아 boxes.json과 레이아웃 뷰에 유령
        항목으로 뜬다.
        """
        prefix = f"p{page_number:04d}_"
        if self.images_dir.is_dir():
            for f in list(self.images_dir.iterdir()):
                if f.name.startswith(prefix):
                    f.unlink(missing_ok=True)
                    self.figure_boxes.pop(f.name, None)

    def replace_page(self, page_number: int, chunk: ChunkResult) -> bool:
        """이미 병합된 페이지 하나를 단독 재실행 결과로 교체한다.

        멀티페이지 청크를 쪼개 재사용할 수는 없다 — `_move_chunk_files`가 파일을
        **꺼내 가고**, multi의 local→global 매핑이 `chunk.start_page`에 묶여 있어
        하위 청크로 나누면 이미지 귀속이 깨진다. 그래서 청크는 정상 경로로 이미
        병합해 두고, 열화가 확인된 페이지만 이 API로 사후 교체한다.

        chunk는 `single=True`, `num_pages=1`, `start_page=page_number`여야 한다.
        교체했으면 True. 페이지 번호가 범위 밖이면 아무것도 하지 않고 False.
        """
        if not chunk.single or chunk.num_pages != 1 or chunk.start_page != page_number:
            raise ValueError(
                f"replace_page는 단일 페이지 청크만 받는다 "
                f"(single={chunk.single}, num_pages={chunk.num_pages}, "
                f"start_page={chunk.start_page} vs page={page_number})"
            )
        index = page_number - 1
        if not 0 <= index < len(self.pages_md):
            return False

        raw_pages = self._load_raw_pages(chunk)
        self._drop_page_artifacts(page_number)
        self._move_chunk_files(chunk)

        page_md = _clean(self._rewrite_refs(chunk.markdown, chunk), self.page_separator)
        self.pages_md[index] = page_md

        raw = str(raw_pages[0]) if raw_pages else ""
        self._replace_layout_page(page_number, raw)
        self._write_partial()
        return True

    def _replace_layout_page(self, page_number: int, raw: str) -> None:
        """layout_pages에서 해당 물리 페이지의 블록만 새 원출력으로 갈아끼운다."""
        from .layout import parse_page_blocks

        blocks = parse_page_blocks(raw) if raw else []
        for b in blocks:
            if "crop_index" in b:
                b["image"] = _global_image_name(page_number, b.pop("crop_index"))
        width, height = self._page_size(page_number)
        page = {"page": page_number, "width": width, "height": height, "blocks": blocks}
        try:
            from .pdf_fonts import enrich_layout_fonts

            enrich_layout_fonts(self.job_dir / "source.pdf", [page])
        except Exception:  # noqa: BLE001 — enrichment 실패가 교체를 막지 않는다
            pass
        replaced = False
        for i, existing in enumerate(self.layout_pages):
            if existing.get("page") == page_number:
                self.layout_pages[i] = page
                replaced = True
                break
        if not replaced:
            # 좌표를 한 번도 못 받은 페이지였다면 순서를 지켜 끼워 넣는다.
            at = len(self.layout_pages)
            for i, existing in enumerate(self.layout_pages):
                if (existing.get("page") or 0) > page_number:
                    at = i
                    break
            self.layout_pages.insert(at, page)
        if blocks:
            self.has_layout_data = True
        if self.has_layout_data and self.layout_pages:
            _atomic_write_json(self.job_dir / "layout.json", self.layout_pages)

    # ── 마크다운 재작성 ────────────────────────────────────────

    def _rewrite_refs(self, page_md: str, chunk: ChunkResult) -> str:
        if chunk.single:
            return _IMG_SINGLE.sub(
                lambda m: f"![](images/{_global_image_name(chunk.start_page, m.group(1))})",
                page_md,
            )
        def _multi(m: re.Match) -> str:
            global_page, prefix = _fold_local_page(chunk, int(m.group(1)))
            return f"![](images/{_global_image_name(global_page, f'{prefix}{m.group(2)}')})"

        return _IMG_MULTI.sub(_multi, page_md)

    # ── 공개 API ───────────────────────────────────────────────

    def add_chunk(self, chunk: ChunkResult) -> None:
        pages = [chunk.markdown] if chunk.single else split_pages(chunk.markdown)
        raw_pages = self._load_raw_pages(chunk)

        # 개수가 어긋나면 **위치**부터 원본 PDF 본문과 대조해 정한다. 꼬리에서만
        # 보정하면 중간에서 쪼개지거나 건너뛴 경우 그 뒤가 전부 밀린다.
        # 물리 슬롯 L에 들어간 **모델 페이지 인덱스** k. 크롭 이미지 파일은 벤더가
        # `page_{k}_{j}.jpg`로 저장하고 `_move_chunk_files`·`_rewrite_refs`가 k로
        # 이름을 짓는다. 정렬이 페이지를 옮기면 `_ingest_layout`만 L로 이름을 지어
        # **layout이 없는 파일을 가리키고 실제 파일은 고아가 된다.** 같은 k를 쓴다.
        slot_source: list[int | None] = list(range(chunk.num_pages))
        if len(pages) != chunk.num_pages or len(raw_pages) != chunk.num_pages:
            mapping = self._align_chunk_pages(chunk, pages)
            if mapping is not None:
                slot_source = [None] * chunk.num_pages
                for model_index, slot in enumerate(mapping):
                    if slot is None or not 0 <= slot < chunk.num_pages:
                        continue
                    if slot_source[slot] is None:   # 합쳐진 경우 첫 모델 페이지 기준
                        slot_source[slot] = model_index
                placed = sum(1 for m in mapping if m is not None)
                pages = place_by_alignment(pages, mapping, chunk.num_pages, "\n\n")
                if raw_pages:
                    padded = list(raw_pages[: len(mapping)])
                    padded += [""] * (len(mapping) - len(padded))
                    raw_pages = place_by_alignment(
                        padded, mapping, chunk.num_pages, "\n"
                    )
                self.warnings.append(
                    f"{chunk.start_page}페이지 청크: 페이지 마커 {len(mapping)}개 "
                    f"(기대 {chunk.num_pages}) — 원본 본문과 대조해 "
                    f"{placed}개 페이지를 제자리에 배치"
                )

        if len(pages) > chunk.num_pages:
            # 마커가 초과 생성됨 — 초과분을 마지막 페이지에 합침
            self.warnings.append(
                f"{chunk.start_page}페이지 청크: 페이지 마커 {len(pages)}개 "
                f"(기대 {chunk.num_pages}) — 초과분을 마지막 페이지에 병합 "
                "(초과분의 레이아웃 좌표는 제외됩니다)"
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
        self._ingest_layout(chunk, raw_pages, slot_source)
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
