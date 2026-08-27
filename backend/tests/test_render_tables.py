"""잘린 표가 렌더 결과의 나머지를 삼키지 못하게 하는 계약.

모델은 표를 HTML `<table>`로 낸다. 페이지 출력 상한이나 스트리밍 꼬리에서 표가
잘리면 `</table>`이 없는데, HTML 파서는 그 뒤 문서 전체를 마지막 `<td>` 안으로
빨아들인다 — 실측 라이브 미리보기가 "빈 표 격자 + 오른쪽 끝으로 밀린 본문"이 되고,
같은 일이 /html 문서 뷰에서도 일어난다.
"""

import re

from app.pipeline.render import _balance_table_tags, render_markdown_html


def _counts(html: str, tag: str) -> tuple[int, int]:
    return len(re.findall(rf"<{tag}[\s>]", html)), html.count(f"</{tag}>")


def test_balanced_table_is_untouched():
    ok = "<p><table><tr><td>a</td><td>b</td></tr></table></p>"
    assert _balance_table_tags(ok) == ok


def test_truncated_table_is_closed_at_the_paragraph_boundary():
    got = _balance_table_tags("<p><table><tr><td>a</td><td>b</p>\n<p>after</p>\n<h2>H</h2>")
    assert got == "<p><table><tr><td>a</td><td>b</td></tr></table></p>\n<p>after</p>\n<h2>H</h2>"


def test_truncated_table_is_closed_in_lists_quotes_and_headings():
    for open_tag, close_tag in (("li", "li"), ("blockquote", "blockquote"), ("h3", "h3")):
        got = _balance_table_tags(f"<{open_tag}><table><tr><td>x</{close_tag}><p>after</p>")
        assert got == (
            f"<{open_tag}><table><tr><td>x</td></tr></table></{close_tag}><p>after</p>"
        ), got


def test_unmatched_closing_tag_is_dropped():
    assert _balance_table_tags("<p>x</td>y</p>") == "<p>xy</p>"


def test_open_table_at_end_of_document_is_closed():
    assert _balance_table_tags("<table><tr><td>a") == "<table><tr><td>a</td></tr></table>"


def test_truncated_model_table_cannot_swallow_later_blocks():
    """실제 렌더 경로 — 잘린 표 뒤의 문단·제목이 표 안으로 빨려들지 않는다."""
    md = (
        "Table 8: Success rates\n\n"
        "<table><tr><td>Crash type</td><td>458\n\n"
        "After the table\n\n"
        "## Heading\n"
    )
    html = render_markdown_html(md, "/api/jobs/x/files")
    assert _counts(html, "table") == (1, 1)
    assert _counts(html, "td")[0] == _counts(html, "td")[1]
    assert "After the table" in html
    assert "<h2>Heading</h2>" in html
    # 뒤 본문이 표 안에 들어가지 않았다 (표가 먼저 닫힌다)
    assert html.index("</table>") < html.index("After the table")


def test_normal_markdown_pipe_table_still_renders():
    html = render_markdown_html("| a | b |\n| --- | --- |\n| 1 | 2 |\n", "/f")
    assert _counts(html, "table") == (1, 1)
    assert "<td>1</td>" in html
