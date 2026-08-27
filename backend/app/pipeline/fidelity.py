"""페이지 OCR 충실도 — born-digital PDF의 텍스트 레이어를 정답으로 쓰는 게이트.

## 왜 필요한가

멀티페이지 추론(`PAGES_PER_CHUNK`, 기본 8)은 처리량을 위해 여러 페이지를 한
컨텍스트에 넣는다. 그런데 모델(`baidu/Unlimited-OCR`)은 `sliding_window=128`의
12층 MoE라 8쪽에 걸친 구조 기록을 유지할 구조적 수단이 없다. 실측(46쪽 논문):

    p34  프로덕션 0자   → 그 페이지만 단독 재실행하면 일치도 0.945
    p39  프로덕션 0자   → 단독 0.972
    p38  프로덕션 0.409 → 단독 0.971 (gundam 타일 모드 0.988)

즉 **모델이 그 페이지를 못 읽는 게 아니라, 청크 안에서 놓친다.** 해상도 문제도
아니다 — 타일 모드가 벌어 주는 건 0.01~0.02뿐이고, 청크에서 빼내는 것이 0.5~0.97을
번다.

born-digital PDF에서는 PyMuPDF가 뽑는 텍스트가 **공짜 정답**이다. 페이지별로
대조해 열화를 탐지하고, 그 페이지만 단독 재실행한다.

## 지표 설계

세 가지 실패 양태를 잡아야 한다: (1) 페이지 통째 유실, (2) 일부 유실,
(3) 반복·중복 전사로 인한 과생성. 그리고 **정상인데 오탐하면 안 되는** 셋이 있다:

  * **표** — 모델은 `<table>` HTML을 낸다. 태그를 벗기면 길이는 맞지만 셀 순서가
    PDF 읽기 순서와 다르다. 실측 p18은 difflib ratio 0.292인데 정상 페이지다.
    → **순서 민감 지표(LCS·difflib)를 쓰면 표에서 오탐한다.**
  * **그림·차트** — 모델이 `image`/`chart`로 분류한 영역의 PDF 텍스트는 의도적으로
    본문으로 뽑지 않는다. → 정답에서 그 영역을 빼야 한다(`_FIGURE_TYPES`).
  * **코드 리스팅** — 꺾쇠(`Vector<Component>`)는 HTML 태그가 아니다.
    → 태그 제거는 이름 화이트리스트로만(`_MARKUP_RE`).

그래서 정답·후보를 정규화한 뒤 **정답 문자 bigram의 잔존율 × 과생성 페널티**로 잰다
(`score()` 참조). 전역 순서에 둔감하고(표 안전), 유실과 과생성을 각각 잡는다.

## 계약

`page_fidelity()`는 0.0~1.0 또는 None(판정 불가)을 돌려준다. **None은 실패가
아니다** — 스캔 PDF처럼 정답이 없으면 게이트를 걸지 않는다는 뜻이다.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# 모델이 내는 **HTML 태그만** 벗긴다. 표 마크업을 남기면 표 페이지가 통째로 열화로
# 보이지만(실측 p18: 12,716자 중 12,427자가 정상 <table> 마크업), 범용 `<[^>]*>`는
# 반대로 **정답을 파괴한다** — 코드 리스팅의 `#include <AK/Debug.h>`·`Vector<Component>`가
# 태그로 오인되고, 줄을 이어 붙인 뒤 적용하므로 한 줄의 `<`가 수십 줄 뒤의 `>`까지
# 삼킨다(실측 p33: 정답 2,532자 → 866자, 65.8% 증발 / 46쪽 합계 2,877자).
# 그래서 태그 이름 화이트리스트 + 줄바꿈 금지로 묶는다. 실측: 정답 삭제 0건,
# 모델의 표 마크업 8,209건은 그대로 제거된다.
_HTML_TAGS = (
    "table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col"
    "|b|i|u|em|strong|sup|sub|br|hr|p|div|span|ul|ol|li|dl|dt|dd"
    "|code|pre|a|img|h[1-6]|figure|figcaption|blockquote"
)
# 태그 이름 뒤에는 공백·`/`·`>` 중 하나만 온다. `\b`로 두면 수학 표기 `<a, b>`·
# `<u, v>`·`<p|H|p>`가 태그로 먹힌다(내적·쌍대쌍·기댓값을 그렇게 쓰는 논문이 있다).
_MARKUP_RE = re.compile(
    rf"</?(?:{_HTML_TAGS})(?=[\s/>])[^>\n]*>", re.IGNORECASE
)
_WS_RE = re.compile(r"\s+")
# 정답에 없고 모델만 내는 마크다운 장식(**굵게**, `코드`, # 제목, 표의 `|`).
# 양쪽에 같이 적용하므로 대칭이고, 실측 기여는 마진 +0.003으로 작다 — 그래도
# 모델 전용 장식을 빼는 쪽이 지표의 의미에 맞아 남긴다.
_MD_NOISE_RE = re.compile(r"[|*_`~#]+")

# 판정에 필요한 최소 정답 분량. 이보다 짧은 페이지(표지·백지·전면 그림)는
# 정답 자체가 빈약해 지표가 요동친다 — 게이트를 걸지 않는다.
MIN_TRUTH_CHARS = 200
# 과생성 허용 배수. 1.0이면 정답보다 조금만 길어도 깎여 LaTeX 전개가 많은 문서가
# 억울해진다. 실측 최적은 1.2(마진 0.357)지만 그 여유를 사서 1.3을 쓴다(마진 0.319).
# 이 값이 발동하는 페이지는 46쪽 중 중복 전사된 p37 하나뿐이다.
_INFLATION_SLACK = 1.3
# 그림 블록으로 덮인 줄을 정답에서 뺄 때의 면적 기준.
_IMAGE_LINE_COVER = 0.5
# 모델의 그림 분류를 믿기 위해 원본 PDF가 그 영역에 그려야 하는 최소 면적 비율.
# 10%면 실제 도표·래스터는 넉넉히 넘고, 머리글 괘선 하나만 걸친 "가짜 전면 그림"은
# 걸러진다(실측: 이 논문의 그림·차트 19개는 전부 통과, 본문 영역은 전부 탈락).
_GRAPHIC_MIN_COVER = 0.10
# 모델이 "본문 텍스트로 뽑지 않는" 것이 정상인 블록 타입. `image`만 보면 차트가 빠진다
# — 실측 이 논문의 그림 블록은 image 12개 + **chart 7개**이고, chart 4개로만 이루어진
# p25는 전사가 완벽한데도 차트 안 텍스트가 정답에 남아 0.729로 떨어졌다(임계 0.70과
# 간격 0.029). equation은 LaTeX로 전사되므로 대상이 아니다.
_FIGURE_TYPES = frozenset({"image", "chart", "figure", "diagram"})
# 표시수식 블록. PDF 텍스트 레이어의 글리프(`E = mc2`)와 모델의 LaTeX
# (`\\(E = mc^{2}\\)`)는 **같은 내용인데 길이가 크게 다르다** — 그대로 두면 수식이 많은
# 페이지가 과생성으로 오탐된다(리뷰 지적: truth 504자 vs LaTeX 972자 → 0.594).
# 양쪽에서 함께 빼면 대칭이라 수식이 없는 문서에는 무해한 no-op이다.
# 남은 위험: `text` 블록 **안**의 인라인 수식은 뺄 수 없다. 실측 이 논문에서는
# 인라인 수식이 페이지의 6% 미만이라 슬랙 1.3 안에 들어간다.
_EQUATION_TYPES = frozenset({
    "equation", "formula", "isolate_formula", "interline_equation",
})
# 수식 블록이 페이지를 이만큼 넘게 덮으면 분류를 믿지 않는다 — 그림과 같은 이유로
# (놓친 페이지를 통째로 한 블록으로 부르는 경우) 정답이 사라지는 것을 막는다.
_EQUATION_MAX_COVER = 0.5


def normalize(text: str) -> str:
    """비교용 정규화 — 마크업·마크다운 장식·공백 제거."""
    text = _MARKUP_RE.sub(" ", text or "")
    text = _MD_NOISE_RE.sub(" ", text)
    return _WS_RE.sub("", text)


def _bigrams(text: str) -> Counter:
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def containment(truth: str, candidate: str) -> float:
    """정답 bigram 중 후보에 살아남은 비율 (0~1) — "몇 %가 남았나"."""
    if not truth:
        return 1.0
    a = _bigrams(truth)
    total = sum(a.values())
    if total == 0:
        return 1.0
    return sum((a & _bigrams(candidate)).values()) / total


def inflation_penalty(truth: str, candidate: str) -> float:
    """후보가 정답보다 `_INFLATION_SLACK`배 넘게 길면 그 비율만큼 깎는다.

    containment만으로는 **과생성**을 못 잡는다 — 모델이 같은 페이지를 두 번 전사하거나
    앞 페이지까지 삼켜도 정답 bigram은 전부 들어 있어 1.0이 나온다(실측 p37: 인쇄
    36·37쪽을 한 페이지에 중복 전사, containment 0.974). 페널티를 곱하면 0.49다.
    """
    if not candidate:
        return 1.0
    return min(1.0, _INFLATION_SLACK * len(truth) / len(candidate))


def score(truth: str, candidate: str) -> float:
    """충실도 = 정답 잔존율 × 과생성 페널티.

    **대칭 지표(Dice)를 쓰지 않는 이유**: 게이트가 잡아야 하는 것은 *손실*인데 Dice는
    "빠졌다"와 "더 썼다"를 한 숫자에 섞어 둘 다 둔감해진다. 실측(게이트 적용 실행의
    p18): 표 내용의 33%가 빠졌는데 Dice는 0.702로 임계값 0.70을 통과했고 containment는
    0.587로 잡았다. 손실률과 과생성을 분리하면 각각 따로 조절할 수도 있다.

    전역 순서에는 여전히 둔감해야 한다 — 모델은 표를 `<table>` HTML로 내고 셀 순서가
    PDF 읽기 순서와 다르다. 그래서 단어가 아니라 **문자 bigram 다중집합**을 센다
    (단어 토큰은 띄어쓰기가 없는 중국어·일본어에서 무너진다).

    실측 분리 마진(46쪽, 열화 4쪽이 확인된 실행): 대상 최고 0.494 vs 비대상 최저 0.813.
    """
    return containment(truth, candidate) * inflation_penalty(truth, candidate)


@dataclass(frozen=True)
class PageFidelity:
    """한 페이지의 판정 결과. score is None이면 정답이 없어 판정하지 않았다."""

    page: int
    score: float | None
    truth_chars: int
    ocr_chars: int
    reason: str = ""

    @property
    def measurable(self) -> bool:
        return self.score is not None


def _pdf_graphic_area(fitz, page, rect) -> float:
    """PDF가 그 사각형 안에 실제로 그린 그림·벡터의 면적 합(중복 허용).

    모델의 그림 분류를 **원본으로 검증**하는 데 쓴다. 중복 계산은 의도적이다 —
    과대평가는 "모델 분류를 믿는다"는 기존 동작으로 떨어질 뿐이라 안전하다.
    """
    total = 0.0
    try:
        for info in page.get_image_info():
            total += abs((fitz.Rect(info["bbox"]) & rect).get_area())
    except Exception:  # noqa: BLE001 — 그림 목록 실패는 '검증 불가'
        pass
    try:
        for drawing in page.get_drawings():
            box = drawing.get("rect")
            if box is None:
                continue
            total += abs((fitz.Rect(box) & rect).get_area())
    except Exception:  # noqa: BLE001
        pass
    return total


def _image_rects(fitz, page, blocks: list[dict]) -> list:
    """모델이 그림으로 분류한 블록 중 **원본에 실제로 그림이 있는** 것의 사각형.

    검증이 필요한 이유: 청크 안에서 페이지를 놓친 모델이 빈 출력 대신 전면 `image`
    블록 하나를 내면(판독 실패 영역을 그림으로 분류하는 것은 OCR 모델의 흔한 동작),
    정답 텍스트가 통째로 가려져 `truth_chars=0` → '정답 부족' → **게이트가 존재
    이유인 바로 그 실패에서 침묵한다.** 원본 PDF가 거기에 아무것도 그리지 않았다면
    그 분류는 틀린 것이므로 무시한다.
    """
    out = []
    w, h = page.rect.width, page.rect.height
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if str(b.get("type") or "") not in _FIGURE_TYPES and not b.get("image"):
            continue
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        rect = fitz.Rect(x1 / 999 * w, y1 / 999 * h, x2 / 999 * w, y2 / 999 * h)
        rect = rect * page.derotation_matrix
        rect.normalize()
        if rect.is_empty:
            continue
        area = abs(rect.get_area())
        if area > 0 and _pdf_graphic_area(fitz, page, rect) < area * _GRAPHIC_MIN_COVER:
            continue        # 원본에 그림이 없다 — 모델의 그림 분류를 믿지 않는다
        out.append(rect)
    return out


def _equation_rects(fitz, page, blocks: list[dict]) -> list:
    """표시수식 블록의 사각형. 페이지의 절반을 넘게 덮으면 믿지 않는다."""
    out = []
    w, h = page.rect.width, page.rect.height
    for b in blocks:
        if not isinstance(b, dict) or str(b.get("type") or "") not in _EQUATION_TYPES:
            continue
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        rect = fitz.Rect(x1 / 999 * w, y1 / 999 * h, x2 / 999 * w, y2 / 999 * h)
        rect = rect * page.derotation_matrix
        rect.normalize()
        if not rect.is_empty:
            out.append(rect)
    page_area = abs(page.rect.get_area()) or 1.0
    if sum(abs(r.get_area()) for r in out) > page_area * _EQUATION_MAX_COVER:
        return []
    return out


def ocr_text(blocks: list[dict]) -> str:
    """비교용 후보 텍스트 — 정답에서 영역을 빼는 블록은 후보에서도 뺀다.

    한쪽만 빼면 비대칭이 된다: 그림 영역을 정답에서 지우면서 모델이 그 그림에 대해
    쓴 글자를 후보에 남기면, 완벽한 전사가 과생성으로 보인다. 실측 이 모델의
    그림·차트 블록은 내용이 비어 있어(19개 중 0개) 지금은 no-op이지만, 차트 안
    텍스트를 뽑는 모델로 바뀌어도 지표가 무너지지 않게 여기서 막는다.
    """
    skip = _EQUATION_TYPES | _FIGURE_TYPES
    return " ".join(
        str(b.get("content") or "")
        for b in blocks
        if str(b.get("type") or "") not in skip
    )


def truth_text(fitz, page, blocks: list[dict]) -> str:
    """페이지의 정답 텍스트 — 그림·표시수식 블록에 덮인 줄은 뺀다.

    모델은 그림 안의 글자를 본문으로 뽑지 않는 것이 정상 동작이다. 정답에서 빼지
    않으면 그림이 큰 페이지가 전부 열화로 보인다. 수식은 `ocr_text`가 후보에서도
    빼므로 대칭이다.
    """
    rects = _image_rects(fitz, page, blocks) + _equation_rects(fitz, page, blocks)
    kept: list[str] = []
    try:
        raw = page.get_text("dict")
    except Exception:  # noqa: BLE001 — 텍스트 추출 실패는 '정답 없음'으로 처리
        return ""
    for blk in raw.get("blocks", []):
        for line in blk.get("lines", []):
            bbox = line.get("bbox")
            if not bbox:
                continue
            lr = fitz.Rect(bbox)
            area = lr.get_area()
            if area > 0 and any(
                (lr & ir).get_area() / area > _IMAGE_LINE_COVER for ir in rects
            ):
                continue
            kept.append("".join(sp.get("text") or "" for sp in line.get("spans", [])))
    return " ".join(kept)


def page_fidelity_blocks(fitz, page, blocks: list, page_number: int) -> PageFidelity:
    """파싱된 블록 목록을 원본 PDF 텍스트 레이어와 대조한다.

    **파싱 뒤**의 블록을 재는 것이 핵심이다 — 페이지 정렬·초과 마커 보정까지
    거친 '실제로 병합된 결과'를 판정해야 게이트가 사용자가 보는 것과 같은
    대상을 본다.
    """
    blocks = [b for b in (blocks or []) if isinstance(b, dict)]
    ocr = normalize(ocr_text(blocks))
    truth = normalize(truth_text(fitz, page, blocks))
    if len(truth) < MIN_TRUTH_CHARS:
        return PageFidelity(page_number, None, len(truth), len(ocr), "정답 텍스트 부족")
    return PageFidelity(page_number, score(truth, ocr), len(truth), len(ocr))


def page_fidelity(fitz, page, raw_page: str, page_number: int) -> PageFidelity:
    """벤더 원출력 문자열(`<|det|>type [bbox]<|/det|>내용`)을 대조한다."""
    from .layout import parse_page_blocks

    try:
        blocks = parse_page_blocks(raw_page or "")
    except Exception:  # noqa: BLE001 — 파싱 실패해도 원문 대조는 가능하다
        blocks = []
    if not blocks and raw_page:
        # 파서가 아무것도 못 얻었어도 원출력에 글자가 있으면 그대로 쓴다 —
        # 문법이 깨졌을 뿐 내용은 있을 수 있고, 그건 열화가 아니다.
        blocks = [{"type": "text", "content": raw_page}]
    return page_fidelity_blocks(fitz, page, blocks, page_number)


def _open_pdf(pdf_path):
    """(fitz, doc) 또는 (None, 사유). 열 수 없으면 게이트를 걸지 않는다."""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover — 배포 환경엔 항상 있다
        return None, "pymupdf 없음"
    try:
        return fitz, fitz.open(pdf_path)
    except Exception:  # noqa: BLE001 — 원본 PDF 접근 실패는 '판정 불가'
        return None, "원본 PDF 열기 실패"


def evaluate_layout_pages(pdf_path, layout_pages: list) -> list[PageFidelity]:
    """layout.json 형태의 페이지들(`{page, blocks}`)을 한 번에 판정한다."""
    fitz, doc = _open_pdf(pdf_path)
    if fitz is None:
        return [
            PageFidelity(int(p.get("page") or 0), None, 0, 0, doc)
            for p in layout_pages
        ]
    out: list[PageFidelity] = []
    try:
        for p in layout_pages:
            pno = int(p.get("page") or 0)
            if pno < 1 or pno > doc.page_count:
                out.append(PageFidelity(pno, None, 0, 0, "페이지 범위 밖"))
                continue
            out.append(
                page_fidelity_blocks(fitz, doc[pno - 1], p.get("blocks") or [], pno)
            )
    finally:
        doc.close()
    return out


def evaluate_raw_page(pdf_path, raw_page: str, page_number: int) -> PageFidelity:
    """단독 재실행 결과 한 장을 같은 잣대로 판정한다(채택 여부 판단용)."""
    fitz, doc = _open_pdf(pdf_path)
    if fitz is None:
        return PageFidelity(page_number, None, 0, 0, doc)
    try:
        if page_number < 1 or page_number > doc.page_count:
            return PageFidelity(page_number, None, 0, 0, "페이지 범위 밖")
        return page_fidelity(fitz, doc[page_number - 1], raw_page, page_number)
    finally:
        doc.close()
