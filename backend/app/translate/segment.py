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


def _trailing_blank_lines(lines: list[str], start: int, end: int) -> int:
    """[start, end) 줄 범위 끝에 붙은 빈 줄 수 — 번역문으로 바꿀 때 그대로 되살린다.

    markdown-it은 목록(bullet·ordered) 토큰의 줄 범위에 뒤따르는 빈 줄까지 넣는다. 번역문은
    _sanitize_unit이 앞뒤 빈 줄을 걷어내므로, 범위를 통째로 바꾸면 목록과 다음 블록을 가르는
    빈 줄이 사라져 다음 문단이 마지막 목록 항목(lazy continuation)으로 흡수됐다(fresh-user-1).
    유닛 src·캐시 키는 그대로 두고 조립에서만 원문의 빈 줄을 보존한다."""
    count = 0
    while end - count > start + 1 and not lines[end - count - 1].strip():
        count += 1
    return count


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
            lines[b["s"]:b["e"]] = new_text.split("\n") + [""] * _trailing_blank_lines(
                lines, b["s"], b["e"],
            )
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


def reference_rule_mismatch(
    md_units: list[Unit],
    lay_units: list[Unit],
    *,
    reasons: dict[str, str] | None = None,
    deferred: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """md·layout 두 참고문헌 규칙의 불일치를 센다 (정책 변경 없이 관측만).

    md 경로는 heading 스윕(`_REF_HEADING_RE` 제목 이후 구간)을, layout 경로는 블록
    `type=="ref_text"`를 쓴다. 두 규칙은 **입력이 달라 하나로 합칠 수 없다** —
    layout 블록만 레이아웃 엔진이 준 타입을 갖고, Markdown에는 그 타입이 없다.
    반대로 heading 스윕은 타입을 주지 않는 엔진에서도 참고문헌을 보호한다. 그래서
    규칙을 통일하는 대신, 같은 원문 줄이 **한쪽에서만** 원문 유지되는 경우를 세어
    리포트 경고로 남긴다(같은 영역이 result.ko.md에선 번역, PDF에선 영어로 남는 사례).

    엔진은 `reasons`(유닛 id → 실제 건너뜀 사유: heading 스윕 또는 should_skip의 내용
    판정)와 `deferred`(layout 줄에 전부 덮여 layout 번역·보존을 그대로 받는 md 유닛 id)를
    넘긴다. 둘 다 실제 산출물 기준이다 — 예전에는 유닛의 heading 스윕 표시만 봐서,
    Unlimited-OCR처럼 result.md에 제목 표기(#)가 없는 잡은 참고문헌 목록을 내용으로
    건너뛰거나 layout 보존을 그대로 받아 두 산출물이 같아도 매번 '불일치' 경고를 냈다
    (실서버 25쪽 논문: layout만 유지 66건 경고, 실제 result.ko.md 참고문헌 한글 0자).
    `reasons`가 없으면 예전처럼 유닛의 `skip_reason`만 본다.

    반환: {"md_only": n, "layout_only": n, "sample_units": [유닛 id ...]}
      md_only     — md는 references로 건너뛰는데 layout은 번역 대상인 블록 수
      layout_only — layout은 ref_text로 건너뛰는데 md는 번역 대상인 블록 수
    """

    def _reason(unit: Unit) -> str:
        return unit.skip_reason if reasons is None else reasons.get(unit.id, unit.skip_reason)

    md_ref: set[str] = set()
    md_plain: set[str] = set()
    for unit in md_units:
        reason = _reason(unit)
        if reason == "references":
            target = md_ref
        elif reasons is None or (not reason and unit.id not in deferred):
            # 실제 판정 기준에서는 md 쪽이 **스스로 번역하는** 줄만 센다 — 다른 사유로
            # 건너뛰거나 layout 결과를 받는(deferred) 유닛은 갈라질 수 없다.
            target = md_plain
        else:
            continue
        for line in _nonempty_lines(unit.src):
            target.add(line)

    md_only = 0
    layout_only = 0
    sample_units: list[str] = []
    for unit in lay_units:
        reason = _reason(unit)
        is_ref = reason == "references"
        # 실제 판정 기준에서 '번역 대상'은 아무 사유로도 건너뛰지 않는 블록뿐이다(쪽 번호 등 제외).
        translated = not is_ref if reasons is None else not reason
        for line in _nonempty_lines(unit.src):
            if is_ref and line in md_plain and line not in md_ref:
                layout_only += 1
            elif translated and line in md_ref and line not in md_plain:
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


def _layout_line_candidates(
    source_pages: list, *, multiline: bool = False,
) -> list[tuple[int, int, str]]:
    """reconcile이 매핑 후보로 삼는 layout 블록들 → [(페이지 idx, 블록 idx, 원문)].

    필터: ref_text 제외, 비어있지 않은 content만. multiline=False면 단일 줄 블록만
    (줄 매핑용), True면 여러 줄 블록도(유닛 전체 매핑용). 번역문 쪽 조건은 여기서
    알 수 없으므로 뺀다.
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
            if not source or ("\n" in source and not multiline):
                continue
            out.append((page_idx, block_idx, source))
    return out


def layout_line_sources(source_pages: list, *, multiline: bool = False) -> set[str]:
    """layout 번역으로 md를 덮을 수 있는 원문 집합 (엔진의 md 유닛 지연 판단용).

    기본은 단일 줄 블록 원문(줄 매핑). multiline=True면 여러 줄 블록의 원문 전체도
    넣는다 — md 유닛 하나가 layout 블록 하나와 통째로 같은 경우(textlayer 잡의 저자
    블록·여러 줄 문단, 실측 456개 중 241개)도 그 번역을 그대로 쓴다.
    같은 원문이 여러 블록에 등장하면 번역이 상충할 수 있어(layout_line_map도 그때
    매핑하지 않는다) 보수적으로 제외한다.
    """
    seen: dict[str, int] = {}
    for _page_idx, _block_idx, source in _layout_line_candidates(
        source_pages, multiline=multiline,
    ):
        seen[source] = seen.get(source, 0) + 1
    return {source for source, n in seen.items() if n == 1}


def layout_line_map(
    source_pages: list,
    translated_pages: list,
    final_ids: set[str] | None = None,
) -> dict[str, str]:
    """md 원문 한 줄 → layout 번역 한 줄 매핑 (layout을 단일 기준으로 쓰기 위한 재료).

    OCR merge 결과는 보통 각 layout 블록을 result.md의 한 줄로도 기록한다. md 유닛과
    layout 유닛을 따로 번역하면 같은 문장이 PDF와 Markdown에서 다르게 번역되므로,
    원문 한 줄과 layout 블록이 정확히 대응하면 layout 번역을 그대로 쓴다.

    final_ids가 주어지면 그 lay 유닛(번역 성공 또는 의도적 보존)의 블록만 쓴다 —
    번역에 실패해 원문이 남은 블록이 md 줄을 영어 그대로 '매핑'하지 않게 한다.
    복수 줄 블록·같은 원문의 상충 번역·ref_text는 매핑하지 않는다.
    """
    if not isinstance(source_pages, list) or not isinstance(translated_pages, list):
        return {}
    candidates: dict[str, set[str]] = {}
    for page_idx, block_idx, source in _layout_line_candidates(source_pages, multiline=True):
        if page_idx >= len(translated_pages):
            continue
        if final_ids is not None:
            pno = source_pages[page_idx].get("page")
            if f"lay:{pno}:{block_idx}" not in final_ids:
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
        # 한 줄 원문은 한 줄 번역만(줄 단위 치환이 줄 구조를 바꾸지 않게). 여러 줄 블록은
        # 유닛 전체 매핑에만 쓰이므로 번역의 줄 수가 달라도 된다.
        if not translated or ("\n" in translated and "\n" not in source):
            continue
        candidates.setdefault(source, set()).add(translated)
    return {source: next(iter(values)) for source, values in candidates.items() if len(values) == 1}


def _line_runs(sources) -> dict[str, list[tuple[tuple[str, ...], str]]]:
    """여러 줄 layout 원문 → {첫 줄: [(줄 튜플, 원문 키), …]}(긴 것부터).

    md 유닛 하나가 layout 블록 여럿(단일 줄 제목·초록 + 여러 줄 저자 블록)을 이어 붙인 경우가
    있다 — Unlimited-OCR 25쪽 논문의 1쪽 md 유닛(19줄)은 7줄만 단일 줄 블록과 같고 나머지 12줄은
    3줄짜리 저자 블록 넷이었다. 연속한 md 줄 묶음이 여러 줄 블록과 줄 단위로 같으면 그 블록의
    번역으로 덮는다. 블록 안에 빈 줄이 있으면 묶음 매칭에 쓰지 않는다(유닛 전체 매칭만)."""
    runs: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for source in sources:
        if "\n" not in source:
            continue
        lines = tuple(line.strip() for line in source.split("\n"))
        if len(lines) < 2 or not all(lines):
            continue
        runs.setdefault(lines[0], []).append((lines, source))
    for candidates in runs.values():
        candidates.sort(key=lambda candidate: len(candidate[0]), reverse=True)
    return runs


def _line_segments(lines: list[str], keys, runs) -> list[tuple[int, int, str | None]]:
    """md 유닛 줄을 layout 원문으로 덮은 구간 [(시작, 끝, 원문 키)] — 빈 줄은 '', 못 덮은 줄은 None.

    여러 줄 블록(연속 줄 묶음)이 같은 자리의 단일 줄 블록보다 먼저다."""
    segments: list[tuple[int, int, str | None]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            segments.append((index, index + 1, ""))
            index += 1
            continue
        for run, source in runs.get(stripped, ()):
            end = index + len(run)
            if tuple(line.strip() for line in lines[index:end]) == run:
                segments.append((index, end, source))
                index = end
                break
        else:
            segments.append((index, index + 1, stripped if stripped in keys else None))
            index += 1
    return segments


def layout_covers_unit(src: str, sources) -> bool:
    """md 유닛이 layout 원문으로 완전히 덮이는가 — 유닛 전체가 블록 하나와 같거나, 비어 있지 않은
    모든 줄이 단일 줄 블록 또는 여러 줄 블록의 연속 줄 묶음과 같다(map_unit_lines와 같은 규칙)."""
    if src.strip() in sources:
        return True
    segments = _line_segments(src.split("\n"), sources, _line_runs(sources))
    return any(key for _start, _end, key in segments) and all(
        key is not None for _start, _end, key in segments
    )


def _keep_spacing(line: str, text: str) -> str:
    leading = line[:len(line) - len(line.lstrip())]
    trailing = line[len(line.rstrip()):]
    return f"{leading}{text}{trailing}"


def map_unit_lines(src: str, mapping: dict[str, str], *, partial: bool = False) -> str | None:
    """md 유닛이 layout 번역으로 완전히 덮이면 그 결과, 아니면 None.

    유닛 전체가 블록 하나와 같으면 그 번역을 통째로, 아니면 비어 있지 않은 **모든**
    줄이 단일 줄 블록 또는 여러 줄 블록의 연속 줄 묶음으로 덮일 때 그 번역으로 바꾼 결과를
    돌려준다. 줄 앞뒤 공백은 보존한다(여러 줄 블록은 번역 줄 수가 같을 때 줄마다).

    reconcile을 유닛 단위로 한다(translate-llm-1). 종전 줄 단위 reconcile은 원문 줄로
    출력을 다시 만들어 매핑되지 않은 줄(여러 줄 블록·상충 중복·수식 정의 줄)을 영어로
    남기고, 1차에서 번역한 md 유닛 결과는 통째로 버렸다. 이제 한 줄이라도 매핑이
    없으면 그 유닛은 자기 번역(md 유닛 번역)을 쓴다.

    partial=True면 덮이지 않는 줄은 원문 그대로 두고 덮이는 줄만 바꾼다(하나도 없으면 None) —
    자기 번역이 실패해 원문으로 남은 유닛이 같은 문장의 layout 번역까지 버리지 않게 한다.
    """
    whole = mapping.get(src.strip())
    if whole is not None:
        # 유닛 전체가 layout 블록 하나와 같다(여러 줄 블록 포함) — 그 번역을 통째로 쓴다
        lead = src[:len(src) - len(src.lstrip())]
        trail = src[len(src.rstrip()):]
        return f"{lead}{whole}{trail}"
    lines = src.split("\n")
    out: list[str] = []
    mapped_any = False
    for start, end, key in _line_segments(lines, mapping, _line_runs(mapping)):
        if key == "":
            out.append(lines[start])
            continue
        if key is None:
            if not partial:
                return None
            out.extend(lines[start:end])
            continue
        mapped_any = True
        translated = mapping[key]
        if end - start == 1:
            out.append(_keep_spacing(lines[start], translated))
            continue
        pieces = translated.split("\n")
        if len(pieces) == end - start:
            out.extend(_keep_spacing(line, piece.strip()) for line, piece in zip(lines[start:end], pieces))
        else:
            lead = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
            out.extend(f"{lead}{piece.strip()}" for piece in pieces if piece.strip())
    return "\n".join(out) if mapped_any else None
