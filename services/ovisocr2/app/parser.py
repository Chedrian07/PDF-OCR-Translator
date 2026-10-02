"""OvisOCR2 raw 출력 파서 — 모델·vLLM 없이 임포트 가능한 순수 모듈 (표준 라이브러리만).

모델 카드 공식 출력 규약:
- figure: `<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />`,
  좌표는 [0, 1000) 정규화 정수 (공식 파서 정규식과 동일 형태만 인정)
- 표: HTML `<table>…</table>` / 수식: LaTeX / 나머지: 표준 Markdown
- 잘린 반복 suffix 정리: 모델 카드의 `_clean_truncated_repeats` 알고리즘

이 모듈은 모델 출력을 **비신뢰 입력**으로 취급한다:
- 좌표는 자릿수 제한(≤4자리) 정규식으로만 추출 — 경로 문자열은 절대 만들지 않는다
- 유효 figure 태그는 `[[FIGURE:n]]` placeholder로 치환 (파일명 결정은 메인 backend 몫)
- 그 외 모든 `<img …>` 태그는 제거 (외부 URL·경로 탈출·비정상 속성 무력화)
- figure 수·태그 길이·좌표 범위·중복·퇴화 bbox 검증
- 문서 본문에 리터럴로 실린 `[[FIGURE:`는 `&#91;&#91;FIGURE:`(숫자 문자 참조)로 바꾼다 —
  placeholder는 이 파서만 만든다
"""

from __future__ import annotations

import re

BBOX_MAX = 999          # 내부 프로토콜 정규화 상한 (0–999)
COORD_LIMIT = 1000      # 모델 출력 좌표 상한 [0, 1000) — 1000은 999로 clamp
MAX_FIGURES = 64
MIN_BBOX_SIDE = 2       # 정규화 단위 최소 변 — 미만은 퇴화 bbox로 거부
MAX_RAW_CHARS = 400_000

# 공식 규약 + 안전한 변형만: 공백 유연화, self-closing slash 생략 허용.
# \d{1,4}로 태그 길이가 상한되고, 경로는 숫자 4개 외 어떤 문자열도 매치되지 않는다.
FIGURE_TAG_RE = re.compile(
    r'<img\s+src="images/bbox_(\d{1,4})_(\d{1,4})_(\d{1,4})_(\d{1,4})\.jpg"\s*/?>'
)
# 유효 figure 추출 후 남은 모든 img 태그(외부 URL·트래버설·비정상 속성)는 제거
_ANY_IMG_TAG_RE = re.compile(r"<img\b[^>]{0,500}?/?>", re.IGNORECASE)
# 닫히지 않은 <img … (태그 종결 없이 줄이 끝나는 잔여물)도 정리
_UNCLOSED_IMG_RE = re.compile(r"<img\b[^>\n]{0,500}", re.IGNORECASE)

# 앱이 소유한 figure 참조 문법의 여는 부분과 그 리터럴 표기. 문서 본문에 `[[FIGURE:0]]`이
# 글자로 실려 있으면 모델이 그대로 옮겨 적고, 같은 index의 진짜 placeholder와 겹쳐
# figure가 문장 한가운데로 옮겨 붙었다(감사 sidecar-9). `\[`로 이스케이프하면 렌더러가
# 디스플레이 수식(`\[ … \]`)으로 읽으므로 숫자 문자 참조를 쓴다 — 렌더하면 `[[`로 보인다
# (backend protocol의 리터럴 표기와 같다).
_FIGURE_OPEN = "[[FIGURE:"
_FIGURE_OPEN_LITERAL = "&#91;&#91;FIGURE:"


def escape_literal_placeholders(text: str) -> str:
    """본문의 리터럴 `[[FIGURE:`를 문자 참조로 바꾼다 — placeholder 정규식에 걸리지 않게.

    치환문에는 `[`가 없어 결과에 `[[FIGURE:`가 새로 생길 수 없다(1회 치환으로 충분)."""
    return text.replace(_FIGURE_OPEN, _FIGURE_OPEN_LITERAL)


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """모델 카드 공식 반복 suffix 정리 알고리즘 (의미론 동일 구현).

    텍스트 끝에서 주기 1–200의 반복을 찾아, 5회 이상 & 100자 이상 반복이면
    한 주기 + 잘린 꼬리만 남긴다. 8000자 미만 텍스트는 건드리지 않는다.
    """
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]

    return text


def _validate_bbox(
    left: int, top: int, right: int, bottom: int
) -> tuple[int, int, int, int] | None:
    """[0,1000) 좌표 → [0,999] 정규화. 위반 시 None."""
    coords = (left, top, right, bottom)
    if any(v < 0 or v > COORD_LIMIT for v in coords):
        return None
    # [0,1000) 규약의 경계값 1000만 999로 clamp (0.1% 오차)
    left, top, right, bottom = (min(v, BBOX_MAX) for v in coords)
    if right - left < MIN_BBOX_SIDE or bottom - top < MIN_BBOX_SIDE:
        return None
    return (left, top, right, bottom)


def parse_page(raw: str) -> dict:
    """raw 모델 출력 → 프로토콜 page dict (markdown/blocks/warnings).

    markdown의 유효 figure 태그는 순서대로 `[[FIGURE:n]]`으로 치환되고,
    같은 순서로 image 블록(figure_index=n, bbox [0,999])이 생성된다.
    heading/본문/HTML 표/LaTeX/코드/목록은 그대로 보존된다.

    **채택된** figure 태그 사이의 본문 조각마다 비정상 img 태그를 지운 뒤 리터럴
    `[[FIGURE:`를 이스케이프한다. 치환을 마친 전체 문자열에서 img 태그를 지우면
    `[<img …>[FIGURE:0]]`처럼 지운 자리 양쪽이 이어져 placeholder가 새로 생기고, 바깥
    잔여물(`<img alt="`)이 뒤따르는 진짜 placeholder까지 삼켰다 — 조각 단위로 처리하면 둘 다
    일어나지 않는다. 검증에서 거부된 figure 태그(좌표 이상·퇴화·중복·개수 상한 초과)는
    분할점이 아니라 **조각 안에서** 지운다 — 조각 경계에서 빈 문자열로 지우면 이스케이프가
    이미 끝난 양쪽 조각이 이어져 같은 위조가 생겼다(감사 sidecar-4).
    """
    warnings: list[str] = []
    if len(raw) > MAX_RAW_CHARS:
        warnings.append(f"모델 출력이 상한({MAX_RAW_CHARS}자)을 초과해 절단됨")
        raw = raw[:MAX_RAW_CHARS]

    blocks: list[dict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    counter = {"n": 0}

    def _accept(m: re.Match) -> str | None:
        """유효 figure 태그 → placeholder(image 블록 생성). 거부하면 None — 태그는 본문
        조각에 남아 _text가 지운다."""
        try:
            left, top, right, bottom = (int(g) for g in m.groups())
        except ValueError:  # pragma: no cover — \d 정규식상 불가, 방어적
            warnings.append("figure 태그 좌표 파싱 실패 — 제거")
            return None
        bbox = _validate_bbox(left, top, right, bottom)
        if bbox is None:
            warnings.append(f"figure bbox 좌표 이상({left},{top},{right},{bottom}) — 제거")
            return None
        if bbox in seen_boxes:
            warnings.append(f"중복 figure bbox{bbox} — 제거")
            return None
        if counter["n"] >= MAX_FIGURES:
            warnings.append(f"figure 수 상한({MAX_FIGURES}) 초과 — 이후 태그 제거")
            return None
        seen_boxes.add(bbox)
        n = counter["n"]
        counter["n"] += 1
        blocks.append({
            "type": "image",
            "bbox": list(bbox),
            "content": "",
            "order": n,
            "figure_index": n,
            "confidence": None,
        })
        return f"[[FIGURE:{n}]]"

    removed = {"tags": 0, "unclosed": 0}

    def _text(segment: str) -> str:
        # 거부된 figure 태그(경고는 _accept가 남겼다)를 먼저 지운다 — 공백이 길어 아래
        # 일반 img 정규식의 길이 상한을 넘는 태그도 통째로 지워진다
        segment = FIGURE_TAG_RE.sub("", segment)
        # 유효 figure 외의 img 태그는 전부 제거 — 어떤 경로/URL도 통과시키지 않는다
        segment, n_tags = _ANY_IMG_TAG_RE.subn("", segment)
        segment, n_unclosed = _UNCLOSED_IMG_RE.subn("", segment)
        removed["tags"] += n_tags
        removed["unclosed"] += n_unclosed
        # 제거로 이어 붙은 자리까지 본 뒤에 이스케이프한다
        return escape_literal_placeholders(segment)

    # 채택된 태그만 분할점이다 — 그 사이(거부된 태그 포함)는 한 조각으로 정리·이스케이프한다
    parts: list[str] = []
    pos = 0
    for m in FIGURE_TAG_RE.finditer(raw):
        placeholder = _accept(m)
        if placeholder is None:
            continue
        parts.append(_text(raw[pos:m.start()]))
        parts.append(placeholder)
        pos = m.end()
    parts.append(_text(raw[pos:]))
    markdown = "".join(parts)
    if removed["tags"]:
        warnings.append(f"비정상 img 태그 {removed['tags']}개 제거")
    if removed["unclosed"]:
        warnings.append(f"닫히지 않은 img 태그 잔여물 {removed['unclosed']}개 제거")

    # 반복 suffix 정리는 placeholder 치환 후 적용 (공식 순서: 태그 필터 → 정리).
    cleaned = clean_truncated_repeats(markdown)
    if len(cleaned) != len(markdown):
        warnings.append("잘린 반복 suffix 정리됨 (모델 카드 알고리즘)")
        markdown = cleaned
        # 정리로 placeholder가 사라진 figure는 본문 참조 없는 crop이 된다 —
        # 메인 backend materializer가 페이지 끝에 붙인다 (내용 손실 없음)

    return {
        "markdown": markdown.strip(),
        "blocks": blocks,
        "warnings": warnings,
    }
