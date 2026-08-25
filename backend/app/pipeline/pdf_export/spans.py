"""원문 PDF span 추출·블록 소유권·시각 줄 복원.

OCR 레이아웃의 논리 줄과 원문 PDF의 시각 줄이 어긋나는 블록(가로 평탄화된 표,
의사코드 리스팅)을 여기서 판별하고 되접는다.
"""
from __future__ import annotations

import re
from statistics import median

from .geometry import _rect_overlap_area
from .models import _LineSegment, _SourceSpan
from .text import _plain_text


def _block_rect(fitz, page, bbox):
    """0–999 정규화 bbox → 페이지 내부 좌표 Rect|None (표시 공간 경유, 회전 보정)."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    w, h = page.rect.width, page.rect.height  # 표시(회전 반영) 공간
    rect = fitz.Rect(x1 / 999 * w, y1 / 999 * h, x2 / 999 * w, y2 / 999 * h)
    rect = rect * page.derotation_matrix  # 내부(비회전) 좌표로
    rect.normalize()
    return rect if not rect.is_empty else None


def _source_span_records(fitz, page) -> list[_SourceSpan]:
    """페이지 텍스트 span과 소유권·스타일 판정에 필요한 메타를 읽는다."""
    out: list[_SourceSpan] = []
    try:
        blocks = page.get_text("dict").get("blocks", ())
    except Exception:  # noqa: BLE001 — 텍스트 레이어가 없으면 빈 목록
        return out
    for block in blocks:
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                bbox = span.get("bbox")
                text = str(span.get("text") or "")
                if bbox and len(bbox) == 4 and text.strip():
                    origin = span.get("origin") or (bbox[0], bbox[3])
                    out.append(_SourceSpan(
                        fitz.Rect(bbox),
                        text,
                        float(span.get("size") or 0.0),
                        int(span.get("flags") or 0),
                        (float(origin[0]), float(origin[1])),
                    ))
    return out


def _source_span_rects(fitz, page) -> list[object]:
    """호환용 페이지 텍스트 span 사각형 목록."""
    return [span.rect for span in _source_span_records(fitz, page)]


def _ownership_text(value: object) -> str:
    """source span과 OCR block의 느슨한 내용 대응을 위한 정규화 문자열.

    `_`도 지운다. `\\w`에는 밑줄이 포함되지만 `_plain_text`는 `current_length`를
    TeX 아래첨자로 보고 `current(length)`로 낮추므로, 밑줄을 남기면 원문을 한
    조각으로 읽었을 때와 span(`current` `_` `length`)별로 읽었을 때의 정규화
    결과가 갈라진다 — 실측: 의사코드 리스팅 30줄 중 1줄에서 정렬이 깨졌다.
    """
    return re.sub(r"[^\w]+|_", "", _plain_text(str(value or "")).casefold())


def _source_span_matches_rect(span: _SourceSpan, rect) -> bool:
    """OCR 오차를 허용하면서 source span이 블록에 실질적으로 속하는지 판정한다."""
    center = (span.rect.tl + span.rect.br) / 2
    inside = (
        rect.x0 - 1 <= center.x <= rect.x1 + 1
        and rect.y0 - 1 <= center.y <= rect.y1 + 1
    )
    span_area = max(0.01, span.rect.width * span.rect.height)
    overlap = _rect_overlap_area(span.rect, rect) / span_area
    return inside or overlap >= 0.35


def _assign_source_spans(
    page,
    block_rects: list[object | None],
    blocks: list[dict],
    source_spans: list[_SourceSpan],
) -> tuple[dict[int, list[_SourceSpan]], list[_SourceSpan], set[int]]:
    """각 PDF span을 최대 한 OCR block에만 배정한다.

    OCR bbox는 서로 겹칠 수 있다. 블록마다 독립적으로 center 포함 여부를 검사하면
    같은 원문 span을 두 교체가 모두 redaction하여, 뒤 블록이 fallback해도 제목이
    사라질 수 있다. 내용 일치와 기하 점수를 함께 사용해 유일 소유자를 정한다.
    """
    owned: dict[int, list[_SourceSpan]] = {
        index: [] for index in range(len(block_rects))
    }
    unowned: list[_SourceSpan] = []
    ambiguous_blocks: set[int] = set()
    block_texts = [
        _ownership_text(block.get("content")) if isinstance(block, dict) else ""
        for block in blocks
    ]
    for span in source_spans:
        center = (span.rect.tl + span.rect.br) / 2
        span_area = max(0.01, span.rect.width * span.rect.height)
        span_text = _ownership_text(span.text)
        span_fragments = {
            _ownership_text(token)
            for token in re.findall(r"\w+", _plain_text(span.text).casefold())
            if len(_ownership_text(token)) >= 3
        }
        fragment_owners = {
            fragment: {
                index
                for index, rect in enumerate(block_rects)
                if rect is not None
                and block_texts[index]
                and _source_span_matches_rect(span, rect)
                and fragment in block_texts[index]
            }
            for fragment in span_fragments
        }
        choices: list[tuple[tuple[float, ...], int]] = []
        for index, rect in enumerate(block_rects):
            if rect is None:
                continue
            # list/table 같은 구조 컨테이너가 자식 텍스트 bbox를 넓게 감싸더라도
            # 빈 컨테이너가 실제 글리프를 소유해서는 안 된다.
            if not block_texts[index]:
                continue
            inside = (
                rect.x0 - 1 <= center.x <= rect.x1 + 1
                and rect.y0 - 1 <= center.y <= rect.y1 + 1
            )
            overlap = _rect_overlap_area(span.rect, rect) / span_area
            if not inside and overlap < 0.35:
                continue
            block_text = block_texts[index]
            content_match = bool(
                block_text
                and (
                    (
                        min(len(span_text), len(block_text)) >= 3
                        and (span_text in block_text or block_text in span_text)
                    )
                    or any(
                        owners == {index}
                        for owners in fragment_owners.values()
                    )
                )
            )
            distance = abs(center.x - (rect.x0 + rect.x1) / 2) + abs(
                center.y - (rect.y0 + rect.y1) / 2
            )
            area = max(0.01, rect.width * rect.height)
            choices.append((
                (
                    float(content_match),
                    float(inside),
                    min(1.0, overlap),
                    -distance / max(1.0, page.rect.width + page.rect.height),
                    -area / max(1.0, page.rect.width * page.rect.height),
                    -float(index),
                ),
                index,
            ))
        if not choices:
            unowned.append(span)
            continue
        matching_blocks = {
            index for score, index in choices if score[0] > 0.5
        }
        if len(matching_blocks) > 1:
            # 하나의 PDF span이 여러 OCR 블록의 문자열을 함께 품으면 어느 한쪽이
            # 전체 span을 지우는 순간 다른 블록의 원문도 손실된다. 해당 span과
            # 연관된 블록을 모두 원문 보존 대상으로 표시한다.
            unowned.append(span)
            ambiguous_blocks.update(matching_blocks)
            continue
        _score, owner = max(choices)
        owned[owner].append(span)
    return owned, unowned, ambiguous_blocks


def _leading_bold_prefix(
    source_spans: list[_SourceSpan], translated: str,
) -> tuple[str, str] | None:
    """원문 첫 행의 짧은 bold run을 번역문 접두 라벨에 대응시킨다."""
    if not source_spans or not translated:
        return None
    ordered = sorted(source_spans, key=lambda span: (span.origin[1], span.rect.x0))
    first_y = ordered[0].origin[1]
    first_line = [span for span in ordered if abs(span.origin[1] - first_y) <= 1.25]
    first_line.sort(key=lambda span: span.rect.x0)

    bold_started = False
    bold_text: list[str] = []
    leading_source: list[str] = []
    for span in first_line:
        is_bold = bool(span.flags & 16)
        if is_bold:
            bold_started = True
            bold_text.append(span.text)
        elif bold_started:
            break
        else:
            leading_source.append(span.text)
    source_prefix = "".join(bold_text).strip()
    word_count = len(re.findall(r"\S+", source_prefix))
    if not source_prefix or not (1 <= word_count <= 6) or len(source_prefix) > 80:
        return None
    source_lead = "".join(leading_source).strip()
    marker_re = re.compile(r"(?:[-•·–—]|\(?[A-Za-z0-9]{1,3}[.)])")
    if source_lead and marker_re.fullmatch(source_lead) is None:
        # 문장 중간의 강조(`This is ` + bold `important`)를 run-in label로
        # 오인하면 번역문의 첫 단어가 임의로 굵어진다. 첫 bold run은 문두 또는
        # 목록/번호 marker 바로 뒤에 있을 때만 구조적 접두 라벨로 취급한다.
        return None

    translated_first = translated.splitlines()[0].strip()
    if not translated_first:
        return None
    # 번역기가 run-in label 뒤에 명시적 줄바꿈을 남긴 경우 첫 행 전체가 라벨이다.
    if "\n" in translated:
        return "", translated_first

    tokens = list(re.finditer(r"\S+", translated_first))
    if not tokens:
        return None
    leading = ""
    start = 0
    if source_lead and marker_re.fullmatch(tokens[0].group()):
        leading = translated_first[:tokens[0].end()] + " "
        start = 1
    if len(tokens) < start + word_count:
        return None
    prefix_start = tokens[start].start()
    prefix_end = tokens[start + word_count - 1].end()
    if not leading:
        leading = translated_first[:prefix_start]
    prefix = translated_first[prefix_start:prefix_end]
    return (leading, prefix) if prefix else None


def _visual_line_clusters(
    spans: list[_SourceSpan], base_pt: float,
) -> list[list[_SourceSpan]]:
    """원문 PDF에서 이 블록이 실제로 차지한 시각 줄을 위→아래, 좌→우로 묶는다.

    같은 줄의 span은 baseline이 정확히 같지 않다. 아래/위 첨자(β1, NS)는 약
    0.1em, 표 헤더의 세로 가운데 맞춤 셀은 0.5em까지 어긋난다. 반면 다음 행은
    1.0em 이상 떨어진다(실측 /tmp/p9repro 9.24pt 표: 첨자 1.0pt, 가운데 맞춤
    헤더 4.98pt, 행 간격 8.96~10.36pt). 0.55em 경계가 그 빈 구간 안이다.
    """
    if not spans:
        return []
    tolerance = max(1.25, base_pt * 0.55)
    clusters: list[list[_SourceSpan]] = []
    anchor: float | None = None
    for span in sorted(spans, key=lambda item: item.origin[1]):
        if anchor is None or span.origin[1] - anchor > tolerance:
            clusters.append([])
            anchor = span.origin[1]
        clusters[-1].append(span)
    return [
        sorted(cluster, key=lambda item: item.rect.x0) for cluster in clusters
    ]


def _visual_lines(spans: list[_SourceSpan], base_pt: float) -> list[str]:
    """시각 줄의 텍스트만 필요한 호출부용 얇은 래퍼."""
    return [
        "".join(span.text for span in cluster)
        for cluster in _visual_line_clusters(spans, base_pt)
    ]


def _align_ocr_spans(
    ocr_lines: list[str], normalized: list[str],
) -> list[tuple[int, int, int]] | None:
    """OCR 논리 줄마다 `(시각 줄 index, 정규화 시작 offset, 끝 offset)`을 낸다.

    offset은 시각 줄을 정규화한 문자열 안의 구간이다. 줄 번호만 필요한 호출부는
    `_align_ocr_lines`를, 원문 span까지 되짚어야 하는 호출부는 이 함수를 쓴다.
    """
    if not normalized:
        return None
    spanned: list[tuple[int, int, int]] = []
    index, consumed = 0, 0
    for line in ocr_lines:
        target = _ownership_text(line)
        if not target:
            spanned.append((index, consumed, consumed))
            continue
        # OCR은 합자·첨자·특수기호를 원문 span과 다르게 펼치므로 줄 전체가 항상
        # 일치하지는 않는다. 앞 10자로 위치만 찾고 일치하는 만큼만 소비한다.
        probe = target[:10]
        cursor, offset = index, consumed
        while probe not in normalized[cursor][offset:] and cursor + 1 < len(normalized):
            cursor += 1
            offset = 0
        rest = normalized[cursor][offset:]
        if probe not in rest:
            return None
        matched = target if target in rest else probe
        start = offset + rest.index(matched)
        index, consumed = cursor, start + len(matched)
        spanned.append((index, start, consumed))
    return spanned


def _align_ocr_lines(ocr_lines: list[str], visual_texts: list[str]) -> list[int] | None:
    """각 OCR 논리 줄이 원문의 몇 번째 시각 줄에서 왔는지 단조 정렬로 찾는다."""
    aligned = _align_ocr_spans(
        ocr_lines, [_ownership_text(text) for text in visual_texts],
    )
    return None if aligned is None else [item[0] for item in aligned]


def _reflow_flattened_text(
    original: str, translated: str, spans: list[_SourceSpan], base_pt: float,
) -> str | None:
    """가로 배치가 OCR 줄바꿈으로 평탄화된 블록의 번역문을 원문 줄 수로 되접는다.

    원문에서 한 줄이던 표 헤더 `Dataset | Long Evaluation | Short Evaluation`이
    OCR에서 세 줄로 평탄화되면 bbox 높이는 한 줄뿐인데 textbox 조판은 세 줄을
    요구한다 — 어떤 크기로 축소해도 구조적으로 들어가지 않는다. 원문 span의
    baseline으로 실제 줄 수를 재고 그만큼만 줄바꿈을 남기면 원문의 시각적 모습에
    가까우면서 조판도 가능해진다.

    진짜 여러 줄 문단을 뭉개지 않도록 조건을 좁게 건다: 원문 시각 줄 수가 OCR
    줄 수보다 적고, 번역문의 줄 수가 원문과 1:1로 대응하며, 모든 OCR 줄을 원문
    시각 줄에 단조 정렬할 수 있을 때만 리플로우한다. 하나라도 어긋나면 None을
    반환해 기존 보존 경로를 그대로 탄다.
    """
    original_lines = original.splitlines()
    translated_lines = translated.splitlines()
    if len(original_lines) < 2 or len(original_lines) != len(translated_lines):
        return None
    visual_texts = _visual_lines(spans, base_pt)
    if not visual_texts or len(visual_texts) >= len(original_lines):
        return None
    groups = _align_ocr_lines(original_lines, visual_texts)
    if groups is None:
        return None
    merged = [
        " ".join(
            translated_lines[index].strip()
            for index, group in enumerate(groups)
            if group == line and translated_lines[index].strip()
        )
        for line in range(groups[-1] + 1)
    ]
    reflowed = "\n".join(line for line in merged if line)
    return reflowed if reflowed and reflowed != translated else None


def _listing_segments(
    original: str,
    translated: str,
    spans: list[_SourceSpan],
    base_pt: float,
    right_limit: float,
) -> tuple[_LineSegment, ...]:
    """평탄화된 블록의 OCR 줄을 원문의 줄·열 좌표에 1:1로 되짚는다.

    의사코드 리스팅과 표는 *줄 구조와 열 위치 자체가 의미*다. 번역문을 한 상자에
    흘려 넣는 리플로우는 들여쓰기와 열 정렬을 모두 잃고(리스팅이 무너지고 표가
    왼쪽 한 줄로 흐른다), 애초에 줄 수가 많으면 상자에 들어가지도 않는다.
    `_align_ocr_spans`가 준 정규화 offset을 원문 span 경계로 되돌려 각 OCR 줄이
    차지하던 가로 구간을 그대로 계산한다 — 들여쓰기는 span의 x 좌표로,
    열 경계는 다음 세그먼트의 시작 x로 보존된다.

    OCR 줄 경계가 원문 span 한가운데를 가르거나(그 span을 통째로 리댁션하면 옆
    줄 원문까지 지운다) 어디에도 정렬되지 않은 원문이 사이에 남으면 그 *시각
    줄만* 버린다 — 블록 전체를 포기하지 않는 부분 회수.
    """
    original_lines = original.splitlines()
    translated_lines = translated.splitlines()
    if len(original_lines) < 2 or len(original_lines) != len(translated_lines):
        return ()
    clusters = _visual_line_clusters(spans, base_pt)
    if not clusters:
        return ()
    normalized: list[str] = []
    edges: list[list[int]] = []
    for cluster in clusters:
        parts = [_ownership_text(span.text) for span in cluster]
        offsets = [0]
        for part in parts:
            offsets.append(offsets[-1] + len(part))
        normalized.append("".join(parts))
        edges.append(offsets)
    aligned = _align_ocr_spans(original_lines, normalized)
    if aligned is None:
        return ()

    per_line: dict[int, list[tuple[int, int, int]]] = {}
    for line_index, (cluster_index, start, end) in enumerate(aligned):
        if end <= start or not translated_lines[line_index].strip():
            continue
        per_line.setdefault(cluster_index, []).append((line_index, start, end))

    segments: list[_LineSegment] = []
    for cluster_index, entries in sorted(per_line.items()):
        cluster = clusters[cluster_index]
        offsets = edges[cluster_index]
        cores: list[tuple[int, int, int]] = []
        for line_index, start, end in entries:
            if start not in offsets or end not in offsets:
                cores = []
                break
            cores.append((line_index, offsets.index(start), offsets.index(end) - 1))
        if not cores:
            continue
        line_segments: list[_LineSegment] = []
        for position, (line_index, first, last) in enumerate(cores):
            owned_end = (
                cores[position + 1][1] - 1
                if position + 1 < len(cores)
                else len(cluster) - 1
            )
            if owned_end < last or any(
                _ownership_text(cluster[i].text)
                for i in range(last + 1, owned_end + 1)
            ):
                line_segments = []
                break
            owned = tuple(cluster[first:owned_end + 1])
            x0 = min(span.rect.x0 for span in owned)
            x1 = (
                cluster[cores[position + 1][1]].rect.x0
                if position + 1 < len(cores)
                else right_limit
            )
            if x1 <= x0 + 1.0:
                line_segments = []
                break
            sizes = [span.size for span in owned if span.size > 0]
            # 굵기는 블록이 아니라 세그먼트 단위로 판정한다. run-in 라벨과 표
            # 헤더 셀은 같은 블록 안에서 자기만 bold다(원문 span flags bit 4).
            inked = [span for span in owned if span.text.strip()]
            bold_chars = sum(
                len(span.text.strip()) for span in inked if span.flags & 16
            )
            all_chars = sum(len(span.text.strip()) for span in inked)
            line_segments.append(_LineSegment(
                # 단일 행 조판에는 자동 줄바꿈이 없으므로 고아행 방지용 NBSP는
                # 폰트에 따라 tofu만 남길 뿐이다 — 보통 공백으로 되돌린다.
                translated_lines[line_index].replace("\xa0", " ").strip(),
                original_lines[line_index].replace("\xa0", " ").strip(),
                owned,
                float(x0),
                float(min(x1, right_limit)),
                float(median([span.origin[1] for span in owned])),
                float(median(sizes)) if sizes else base_pt,
                (
                    float(min(span.rect.y0 for span in owned)),
                    float(max(span.rect.y1 for span in owned)),
                ),
                bool(all_chars and bold_chars * 5 >= all_chars * 3),
            ))
        segments.extend(line_segments)
    return tuple(segments)


def _source_text_rects(page, rect, source_spans: list[object]) -> tuple[object, ...]:
    """원문 span을 행별 비연속 redaction 사각형으로 반환한다.

    검출 bbox가 URL·긴 단어의 끝을 몇 pt 잘라내는 경우 OCR 사각형만 지우면 `f.`
    같은 원문 꼬리가 번역문 옆에 남는다. 반대로 여러 행의 bounding union을 하나로
    지우면 그 사이에 낀 다른 owner의 제목까지 사라지므로 같은 행에서 닿는 span만
    병합하고 각 행을 별도 annotation으로 처리한다.
    """
    if not source_spans:
        fallback = +rect
        fallback &= page.mediabox
        return (fallback,) if not fallback.is_empty else ()
    ordered = sorted(source_spans, key=lambda span: (span.y0, span.x0))
    groups: list[object] = []
    for source in ordered:
        merged = False
        for group in reversed(groups):
            vertical = max(0.0, min(group.y1, source.y1) - max(group.y0, source.y0))
            required = min(group.height, source.height) * 0.5
            horizontal_gap = max(0.0, source.x0 - group.x1, group.x0 - source.x1)
            if vertical >= required and horizontal_gap <= 1.5:
                group.include_rect(source)
                merged = True
                break
            if source.y0 > group.y1 + 1.5:
                break
        if not merged:
            groups.append(+source)
    output: list[object] = []
    for group in groups:
        group += (-0.25, -0.25, 0.25, 0.25)
        group &= page.mediabox
        if not group.is_empty:
            output.append(group)
    return tuple(output)


def _source_text_rect(page, rect, source_spans: list[object]):
    """호환용 union. 실제 export redaction은 `_source_text_rects`를 사용한다."""
    rects = _source_text_rects(page, rect, source_spans)
    if not rects:
        return +rect
    actual = +rects[0]
    for source in rects[1:]:
        actual.include_rect(source)
    return actual
