"""수식 델리미터 정규화의 비용 상한(보안)과 짝 없는 델리미터 의미 검증.

POST /render-preview(인증 없음, 본문 최대 2MB)가 이 렌더러를 그대로 호출하고,
정규식은 GIL을 놓지 않는다. 예전 패턴은 짝 없는 여는 델리미터마다 문단 끝까지
다시 훑어 개행 없는 `\\[ a ` 반복 200KiB에 61초 동안 프로세스 전체(SSE·health 포함)를
멈췄다(크기 2배 → 시간 4배). 아래 입력은 예전 코드로는 각각 수 초~수십 초가 걸린다.
"""

import time

import pytest

from app.pipeline.render import (
    _normalize_math_delimiters,
    render_markdown_html,
    text_with_math_html,
)

_SIZE = 200 * 1024
# 선형이면 수 ms~수백 ms(M4 Max 0.7s 이하) — 이 안이면 바로 통과한다.
_BUDGET_S = 3.0
# 느린 러너나 커버리지 추적(CI backend 잡의 pytest --cov는 순수 파이썬 렌더를 수 배 느리게
# 한다 — x86_64 CI 근사 실측 3.8~4.2s)에서 절대 시간을 넘으면, 같은 입력의 1/4 크기와 걸린
# 시간의 비로 판정한다. 선형이면 ~4배, 예전 2차 비용이면 ~16배다(200KiB 61s).
_MAX_SCALING = 8.0

_PATHOLOGICAL = {
    "짝 없는 디스플레이": "\\[ a ",
    "짝 없는 인라인": "\\( a ",
    "닫히지 않은 펜스": "```a\n",
    "닫히지 않은 물결 펜스": "~~~a\n",
    "인라인 코드 다수(마스크 복원)": "`a` ",
    "디스플레이 달러": "$$ a ",
    "이스케이프된 여는 괄호": "\\\\[ a ",
}


def _seconds(render, text: str, clock) -> float:
    start = clock()
    render(text)
    return clock() - start


def _assert_linear(render, build, clock=time.perf_counter) -> None:
    """절대 예산 안이면 통과, 넘으면 1/4 크기 대비 시간 비로 2차 비용만 잡는다."""
    elapsed = _seconds(render, build(_SIZE), clock)
    if elapsed < _BUDGET_S:
        return
    quarter = _seconds(render, build(_SIZE // 4), clock)
    ratio = elapsed / max(quarter, 1e-6)
    assert ratio < _MAX_SCALING, (
        f"{_SIZE}B 렌더 {elapsed:.2f}s, 1/4 크기 {quarter:.2f}s(×{ratio:.1f}) — 2차 비용 회귀"
    )


def _render_preview(text: str) -> None:
    _normalize_math_delimiters(text)
    render_markdown_html(text, "/api/jobs/x/files")


@pytest.mark.parametrize("unit", list(_PATHOLOGICAL.values()), ids=list(_PATHOLOGICAL))
def test_pathological_preview_input_renders_in_linear_time(unit):
    _assert_linear(_render_preview, lambda size: "\\[" + unit * (size // len(unit)))


def test_layout_text_math_spans_are_linear_too():
    _assert_linear(text_with_math_html, lambda size: "\\( a " * (size // 5))


def test_linearity_check_still_catches_quadratic_cost():
    """예산을 넘는 환경에서도 2차 비용은 비로 잡힌다 — 판정 장치 자체의 회귀 방지."""
    sizes: list[int] = []

    def _record(text: str) -> None:
        sizes.append(len(text))

    def _fake_clock(*readings: float):
        return iter(readings).__next__

    # 200KiB 61s(실측 2차 회귀) · 50KiB 3.8s — 비 16배
    with pytest.raises(AssertionError, match="2차 비용 회귀"):
        _assert_linear(_record, lambda size: "x" * size, _fake_clock(0.0, 61.0, 100.0, 103.8))
    assert sizes == [_SIZE, _SIZE // 4]

    # 커버리지 추적으로 느려진 선형 렌더(4.2s · 1.05s, 비 4배)는 통과한다
    sizes.clear()
    _assert_linear(_record, lambda size: "x" * size, _fake_clock(0.0, 4.2, 10.0, 11.05))
    assert sizes == [_SIZE, _SIZE // 4]

    # 예산 안이면 1/4 크기를 다시 재지 않는다
    sizes.clear()
    _assert_linear(_record, lambda size: "x" * size, _fake_clock(0.0, 0.7))
    assert sizes == [_SIZE]


def test_unmatched_inline_opener_does_not_swallow_prose():
    """짝 없는 `\\(`(잘린 수식·코드의 리터럴) 하나가 다음 수식의 `\\)`까지 본문을
    수식 스팬으로 삼키지 않는다 — 실제 수식은 그대로 조판된다."""
    md = (
        "is defined as \\( p(x) when truncated, so we continue with plain prose here and\n"
        "then the next formula \\(y = 2x\\) appears."
    )
    html = render_markdown_html(md, "/f")
    assert '<span class="math-inline">y = 2x</span>' in html
    assert "plain prose here" in html.split('<span class="math-inline">')[0]
    layout = text_with_math_html(md)
    assert '<span class="math-inline">y = 2x</span>' in layout
    assert "when truncated" not in layout.split("math-inline")[1]


def test_inline_math_does_not_cross_a_paragraph_break():
    html = render_markdown_html("open \\( a\n\nnext paragraph \\(b\\)", "/f")
    assert '<span class="math-inline">b</span>' in html
    assert "next paragraph" in html


def test_display_math_keeps_latex_line_break_spacing():
    """`\\\\[4pt]`(aligned 안의 줄바꿈 간격)은 여는 델리미터가 아니다 — 디스플레이
    수식을 끊거나 새로 열지 않는다."""
    md = "\\[\\begin{aligned} a &= b \\\\[4pt] c &= d \\end{aligned}\\]"
    html = render_markdown_html(md, "/f")
    assert html.count('<div class="math-display">') == 1
    assert "\\\\[4pt] c &amp;= d" in html


def test_escaped_opener_outside_math_is_literal():
    html = render_markdown_html("line break \\\\[2pt] then text \\] end", "/f")
    assert "math-display" not in html


def test_code_span_inside_dollar_math_is_restored():
    """`$…$` 마스크 안에 먼저 만든 코드 마스크가 들어 있어도 복원 토큰이 새지 않는다."""
    out = _normalize_math_delimiters("see $a `x` b$ here")
    assert "\x00" not in out and "MDMASK" not in out
    assert "`x`" in out


def test_forged_mask_tokens_in_input_are_inert():
    """본문의 NUL·위조 마스크 토큰이 복원을 무한 재귀·폭증시키지 못한다."""
    forged = "`\x00MDMASK0\x00` and $\x00MDMASK1\x00\x00MDMASK1\x00$"
    out = _normalize_math_delimiters(forged)
    assert "\x00" not in out
    render_markdown_html(forged * 200, "/f")


def test_existing_math_semantics_hold():
    html = render_markdown_html(
        "inline \\(E = mc^{2}\\) and\n\n\\[\n\\int_0^1 x\\,dx\n\\]\n\n`\\(code\\)` $x^2$ costs $5",
        "/f",
    )
    assert '<span class="math-inline">E = mc^{2}</span>' in html
    assert '<div class="math-display">' in html and "\\int_0^1" in html
    assert "<code>\\(code\\)</code>" in html
    assert '<span class="math-inline">x^2</span>' in html and "$5" in html
