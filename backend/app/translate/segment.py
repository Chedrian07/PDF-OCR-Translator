"""번역 단위(유닛) 분리·재조립 — 마크다운과 레이아웃 두 소스.

마크다운은 페이지 구분자로 나눈 뒤 페이지별로 markdown-it 블록 토큰의 줄 범위를
유닛으로 삼는다. 재조립은 **원문 바이트를 최대한 보존**한다: 유닛 줄 범위만
번역문으로 교체하고 나머지(빈 줄·수평선 등)는 그대로 둔다.

핵심 골든 불변식:
  translations가 모든 유닛을 unit.src 그대로 매핑하면
  assemble_markdown 출력은 원본 md와 **바이트 동일**하다.

references 섹션은 skip_reason="references"로 표시해 번역에서 제외한다(문서 끝까지,
같은 레벨 이하의 다음 heading 전까지).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

# 세그먼트 전용 파서 — commonmark + table (render.py와 별개 인스턴스, dollarmath 불필요:
# 수식은 마스킹이 처리하고 여기선 블록 줄 범위만 필요).
_md = MarkdownIt("commonmark").enable("table")

# level 0 블록 오프너 → 유닛 kind
_OPENERS = {
    "paragraph_open": "paragraph",
    "heading_open": "heading",
    "table_open": "table",
    "fence": "fence",
    "html_block": "html",
    "blockquote_open": "blockquote",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
}

_REF_HEADING_RE = re.compile(r"(?i)^(references?|bibliography|acknowledg\w*)$")
_HR_LINE_RE = re.compile(r"^\s*-{3,}\s*$")


@dataclass
class Unit:
    id: str  # "md:{page}:{i}" | "lay:{page}:{i}"
    kind: str
    page: int
    src: str
    skip_reason: str = ""


def _page_blocks(page_text: str) -> list[dict]:
    """한 페이지의 level-0 블록들 → [{i, kind, s, e, level?, text?}] (문서 순서).

    i는 페이지 내 블록 인덱스(유닛 id에 사용), [s,e)는 0-based 줄 반열림 범위.
    heading은 level(int)과 inline 텍스트를 함께 싣는다(references 판별용).
    """
    tokens = _md.parse(page_text)
    blocks: list[dict] = []
    i = 0
    for idx, t in enumerate(tokens):
        if t.level != 0 or t.type not in _OPENERS or not t.map:
            continue
        b = {"i": i, "kind": _OPENERS[t.type], "s": t.map[0], "e": t.map[1]}
        if t.type == "heading_open":
            tag = t.tag[1:]
            b["level"] = int(tag) if tag.isdigit() else 1
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            b["text"] = nxt.content if nxt is not None and nxt.type == "inline" else ""
        blocks.append(b)
        i += 1
    return blocks


def _mark_references(annotated: list[tuple[Unit, dict]]) -> None:
    """references/bibliography/acknowledgments heading부터 같은 레벨 이하의 다음
    heading 전까지 skip_reason="references"로 표시(문서 전역, 페이지 넘나듦)."""
    ref_level: int | None = None
    for unit, b in annotated:
        if unit.kind == "heading":
            level = b.get("level", 1)
            # 활성 references 구간을 닫는 heading(같은 레벨 이하 = 레벨 번호 ≤ 기준)
            if ref_level is not None and level <= ref_level:
                ref_level = None
            htext = (b.get("text") or "").strip().strip("#").strip()
            if _REF_HEADING_RE.match(htext):
                ref_level = level
                unit.skip_reason = "references"
                continue
        if ref_level is not None:
            unit.skip_reason = "references"


def split_markdown(md_text: str, page_separator: str) -> list[Unit]:
    """result.md를 페이지별 블록 유닛으로 분리(문서 순서)."""
    pages = md_text.split(page_separator)
    annotated: list[tuple[Unit, dict]] = []
    for page_idx, page in enumerate(pages):
        lines = page.split("\n")
        for b in _page_blocks(page):
            src = "\n".join(lines[b["s"]:b["e"]])
            unit = Unit(id=f"md:{page_idx}:{b['i']}", kind=b["kind"], page=page_idx, src=src)
            annotated.append((unit, b))
    _mark_references(annotated)
    return [u for u, _ in annotated]


def _sanitize_unit(text: str) -> str:
    """번역문 새니타이즈 — 페이지 구분자 오염 방지.

    유닛 내부의 `---`(3+ 대시만 있는 줄)를 "⸻"로 바꾸고 앞뒤 빈 줄을 제거한다.
    (identity 케이스에서 유닛 src는 대시 전용 줄·앞뒤 빈 줄을 포함하지 않으므로 무변화.)
    """
    lines = text.split("\n")
    lines = ["⸻" if _HR_LINE_RE.match(ln) else ln for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def assemble_markdown(md_text: str, page_separator: str, translations: dict[str, str]) -> str:
    """원문에서 유닛 줄 범위만 번역문으로 교체(페이지별 뒤→앞), 나머지 보존.

    페이지 수가 원본과 달라지면 ValueError(최후 방어). 새니타이즈와 유닛 단위
    page_separator 검사가 선방어한다.
    """
    pages = md_text.split(page_separator)
    out_pages: list[str] = []
    for page_idx, page in enumerate(pages):
        lines = page.split("\n")
        # 뒤에서 앞으로 교체 → 앞선 유닛의 줄 인덱스가 밀리지 않는다
        for b in sorted(_page_blocks(page), key=lambda x: x["s"], reverse=True):
            uid = f"md:{page_idx}:{b['i']}"
            if uid not in translations:
                continue
            new_text = _sanitize_unit(translations[uid])
            if page_separator and page_separator in new_text:
                continue  # 유닛 단위 선방어 — 구분자 유발 유닛은 원문 유지
            lines[b["s"]:b["e"]] = new_text.split("\n")
        out_pages.append("\n".join(lines))
    result = page_separator.join(out_pages)
    if len(result.split(page_separator)) != len(pages):
        raise ValueError(
            f"조립 후 페이지 수 불일치: {len(result.split(page_separator))} != {len(pages)}"
        )
    return result


def layout_units(pages: list) -> list[Unit]:
    """layout.json 페이지들에서 번역 대상 블록 유닛만 (content 있고 image 키 없음)."""
    units: list[Unit] = []
    for page in pages:
        pno = page.get("page")
        for i, block in enumerate(page.get("blocks", [])):
            if "image" in block:
                continue
            content = block.get("content")
            if not content or not str(content).strip():
                continue
            kind = str(block.get("type") or "text")
            units.append(
                Unit(
                    id=f"lay:{pno}:{i}",
                    kind=kind,
                    page=pno,
                    src=content,
                    # 서지 항목은 저자명·학술지명·URL의 원문 표기를 유지한다.
                    # Markdown references 정책 및 PDF 내보내기 정책과 동일한 계약이다.
                    skip_reason="references" if kind == "ref_text" else "",
                )
            )
    return units


def _nonempty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def reference_rule_mismatch(md_units: list[Unit], lay_units: list[Unit]) -> dict:
    """md·layout 두 참고문헌 규칙의 불일치를 센다 (정책 변경 없이 관측만).

    md 경로는 heading 스윕(`_REF_HEADING_RE` 제목 이후 구간)을, layout 경로는 블록
    `type=="ref_text"`를 쓴다. 두 규칙은 **입력이 달라 하나로 합칠 수 없다** —
    layout 블록만 레이아웃 엔진이 준 타입을 갖고, Markdown에는 그 타입이 없다.
    반대로 heading 스윕은 타입을 주지 않는 엔진에서도 참고문헌을 보호한다. 그래서
    규칙을 통일하는 대신, 같은 원문 줄이 **한쪽에서만** 원문 유지되는 경우를 세어
    리포트 경고로 남긴다(같은 영역이 result.ko.md에선 번역, PDF에선 영어로 남는 사례).

    반환: {"md_only": n, "layout_only": n, "sample_units": [유닛 id ...]}
      md_only     — md는 references로 건너뛰는데 layout은 번역 대상인 블록 수
      layout_only — layout은 ref_text로 건너뛰는데 md는 번역 대상인 블록 수
    """
    md_ref: set[str] = set()
    md_plain: set[str] = set()
    for unit in md_units:
        target = md_ref if unit.skip_reason == "references" else md_plain
        for line in _nonempty_lines(unit.src):
            target.add(line)

    md_only = 0
    layout_only = 0
    sample_units: list[str] = []
    for unit in lay_units:
        is_ref = unit.skip_reason == "references"
        for line in _nonempty_lines(unit.src):
            if is_ref and line in md_plain and line not in md_ref:
                layout_only += 1
            elif not is_ref and line in md_ref and line not in md_plain:
                md_only += 1
            else:
                continue
            # 블록당 1건만 센다 — 줄 수가 많은 블록이 집계를 왜곡하지 않게.
            # 표본은 유닛 id만 남긴다(문서 원문은 리포트에 싣지 않는다).
            if len(sample_units) < 5:
                sample_units.append(unit.id)
            break
    return {"md_only": md_only, "layout_only": layout_only, "sample_units": sample_units}


def apply_layout(
    pages: list,
    translations: dict[str, str],
    preserved: dict[str, str] | None = None,
) -> list:
    """deep copy 후 content만 교체 — bbox/fs/bold/vertical/fonts_v 등은 그대로.

    preserved는 "번역하지 않기로 **결정한**" 블록의 사유(code / identifier-list /
    references …)다. 블록에 그대로 실어 두면 PDF 내보내기가 그 블록을 실패가 아니라
    의도적 보존으로 집계한다 — 그러지 않으면 원문과 번역이 같아 `unchanged`로
    떨어져 번역 결함과 구분되지 않는다.
    """
    out = copy.deepcopy(pages)
    marks = preserved or {}
    for page in out:
        pno = page.get("page")
        for i, block in enumerate(page.get("blocks", [])):
            uid = f"lay:{pno}:{i}"
            if uid in translations:
                block["content"] = translations[uid]
            elif uid in marks:
                block["preserved"] = marks[uid]
    return out


def _layout_line_candidates(source_pages: list) -> list[tuple[int, int, str]]:
    """reconcile이 줄 매핑 후보로 삼는 layout 블록들 → [(페이지 idx, 블록 idx, 원문)].

    필터는 reconcile_markdown_with_layout과 **같은 규칙**이다: ref_text 제외,
    단일 줄·비어있지 않은 content만. 번역문 쪽 조건은 여기서 알 수 없으므로 뺀다.
    """
    out: list[tuple[int, int, str]] = []
    if not isinstance(source_pages, list):
        return out
    for page_idx, source_page in enumerate(source_pages):
        blocks = source_page.get("blocks", []) if isinstance(source_page, dict) else []
        for block_idx, source_block in enumerate(blocks):
            if not isinstance(source_block, dict):
                continue
            if str(source_block.get("type") or "") == "ref_text":
                continue
            source = str(source_block.get("content") or "").strip()
            if not source or "\n" in source:
                continue
            out.append((page_idx, block_idx, source))
    return out


def layout_line_sources(source_pages: list) -> set[str]:
    """reconcile이 성공하면 layout 번역으로 덮어쓸 수 있는 md 원문 줄 집합.

    엔진이 md 유닛 1차 번역을 미루는(deferred) 판단에만 쓴다. 같은 원문이 여러
    블록에 등장하면 번역이 상충할 수 있어(reconcile도 그때 매핑에서 뺀다)
    보수적으로 제외한다.
    """
    seen: dict[str, int] = {}
    for _page_idx, _block_idx, source in _layout_line_candidates(source_pages):
        seen[source] = seen.get(source, 0) + 1
    return {source for source, n in seen.items() if n == 1}


def reconcile_markdown_with_layout(
    md_text: str,
    assembled: str,
    source_pages: list,
    translated_pages: list,
    page_separator: str,
    *,
    min_coverage: float = 0.7,
) -> str:
    """레이아웃과 Markdown의 동일 원문 줄을 하나의 번역으로 맞춘다.

    OCR merge 결과는 보통 각 layout 블록을 result.md의 한 줄로도 기록한다. 이때
    Markdown 유닛과 layout 유닛을 각각 LLM에 보내면 같은 문장이 서로 다르게 번역돼
    PDF·개요·읽기 텍스트가 어색하게 갈라질 수 있다. 원문 한 줄과 layout 블록이
    정확히 대응하고, 같은 원문이 항상 같은 번역으로 귀결될 때 layout 번역을 단일
    기준으로 사용한다.

    대응률이 낮은 비정형 Markdown은 기존 assembled 결과를 그대로 반환한다. 복수
    줄 블록·중복 원문의 상충 번역·ref_text는 보수적으로 매핑에서 제외한다.
    """
    if not isinstance(source_pages, list) or not isinstance(translated_pages, list):
        return assembled

    candidates: dict[str, set[str]] = {}
    for page_idx, block_idx, source in _layout_line_candidates(source_pages):
        if page_idx >= len(translated_pages):
            continue
        translated_page = translated_pages[page_idx]
        translated_blocks = (
            translated_page.get("blocks", []) if isinstance(translated_page, dict) else []
        )
        if block_idx >= len(translated_blocks):
            continue
        translated_block = translated_blocks[block_idx]
        if not isinstance(translated_block, dict):
            continue
        translated = str(translated_block.get("content") or "").strip()
        if not translated or "\n" in translated:
            continue
        candidates.setdefault(source, set()).add(translated)

    mapping = {
        source: next(iter(values))
        for source, values in candidates.items()
        if len(values) == 1
    }
    if not mapping:
        return assembled

    lines = md_text.splitlines(keepends=True)
    eligible = 0
    matched = 0
    out: list[str] = []
    separator_line = page_separator.strip()
    for line in lines:
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        stripped = body.strip()
        if stripped and stripped != separator_line:
            eligible += 1
        translated = mapping.get(stripped)
        if translated is None:
            out.append(line)
            continue
        matched += 1
        leading = body[:len(body) - len(body.lstrip())]
        trailing = body[len(body.rstrip()):]
        out.append(f"{leading}{translated}{trailing}{ending}")

    if eligible <= 0 or matched / eligible < min_coverage:
        return assembled
    reconciled = "".join(out)
    if page_separator and len(reconciled.split(page_separator)) != len(md_text.split(page_separator)):
        return assembled
    return reconciled
