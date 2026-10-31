"""Presentation recovery must add delimiters without inventing mathematical text."""

import json
import re

import pytest
from PIL import Image

from app.engine.math_delimiters import MathDelimiterStream, recover_math_delimiters
from app.engine.postprocess import (
    BlockStream,
    normalize_blocks,
    save_results_multi,
    save_results_single,
)
from app.pipeline.layout import parse_page_blocks, render_layout_html
from app.pipeline.render import render_markdown_html


class Sink:
    def __init__(self):
        self.text = ""

    def on_text(self, text):
        self.text += text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "sigma = sqrt((1/N) * sum(x_i - mu)^2)",
            r"\(\mathrm{sigma} = \operatorname{sqrt}((1/N) * \operatorname{sum}(x_i - \mathrm{mu})^2)\)",
        ),
        (
            "rho_i = sum((x_i - mu)^2)",
            r"\(\mathrm{rho}_i = \operatorname{sum}((x_i - \mathrm{mu})^2)\)",
        ),
        ("r = sqrt(x^2 + y^2)", r"\(r = \operatorname{sqrt}(x^2 + y^2)\)"),
        (
            "theta = sin(x)^2 + cos(x)^2",
            r"\(\mathrm{theta} = \operatorname{sin}(x)^2 + \operatorname{cos}(x)^2\)",
        ),
        ("Gamma = exp(-x^2)", r"\(\mathrm{Gamma} = \operatorname{exp}(-x^2)\)"),
        ("tau = log(N^2)/N", r"\(\mathrm{tau} = \operatorname{log}(N^2)/N\)"),
        (
            " alpha = sqrt (x_{i}^{-2})\r\n",
            " " + r"\(\mathrm{alpha} = \operatorname{sqrt} (x_{i}^{-2})\)" + "\r\n",
        ),
    ],
)
def test_whole_line_named_math_preserves_identifiers_operators_and_grouping(
    source, expected
):
    assert recover_math_delimiters(source) == expected
    assert recover_math_delimiters(expected) == expected
    restored = re.sub(r"\\(?:mathrm|operatorname)\{([A-Za-z]+)\}", r"\1", expected)
    assert restored.replace(r"\(", "").replace(r"\)", "") == source
    assert 'class="math-inline"' in render_markdown_html(expected, "/files")


@pytest.mark.parametrize(
    "source",
    [
        "sigma = x_i^2",  # no explicit mathematical function
        "sigma = sqrt(x_i)",  # no numeric exponent
        "sigma = sqrt(x_i^j)",
        "sigma = sqrt(x_mu^2)",
        "sigma = sqrt(x_{mu}^2)",
        "sigma = sqrt(data^2)",
        "result = sqrt(x_i^2)",
        "sigma = unknown(x_i)^2",
        "sigma = sqrt.__call__(x_i)^2",
        "sigma = sqrt(x_i = 2)^2",
        "sigma = sqrt(x_i, y_i)^2",
        "sigma = sqrt(x_i)**2",
        "sigma = sqrt(x_i^12)",
        "sigma = sqrt(x_i^2);",
        "sigma = sqrt(x_i^2).",
        "The formula is sigma = sqrt(x_i^2)",
        "sigma = sqrt(x_i^2) in this notation",
        "sigma=sqrt(x_i^2)",
        "sigma := sqrt(x_i^2)",
        "sigma == sqrt(x_i^2)",
        '"sigma = sqrt(x_i^2)"',
        "`sigma = sqrt(x_i^2)`",
        "    sigma = sqrt(x_i^2)",
        "```python\nsigma = sqrt(x_i^2)\n```",
        "<code>sigma = sqrt(x_i^2)</code>",
        '<a title="sigma = sqrt(x_i^2)">label</a>',
        "[sigma = sqrt(x_i^2)](url)",
        "https://example.org/?sigma = sqrt(x_i^2)",
        r"\(sigma = sqrt(x_i^2)\)",
        "sigma = sqrt(" + "(" * 10 + "x_i^2" + ")" * 11,
        "sigma = sqrt(" + "x_i^2 + " * 40 + "x_i^2)",
        "<|ref|>code<|/ref|><|det|>[0,0,999,999]<|/det|>sigma = sqrt(x_i^2)",
    ],
)
def test_named_math_extension_abstains_on_unsupported_or_literal_whole_lines(source):
    assert recover_math_delimiters(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "The energy is E = mc^2 in this notation.",
            r"The energy is \(E = mc^2\) in this notation.",
        ),
        ("a^2 + b^2 = c^2", r"\(a^2 + b^2 = c^2\)"),
        ("y = 3*x_1 + x_2", r"\(y = 3*x_1 + x_2\)"),
        ("(a + b)^2 = c^2", r"\((a + b)^2 = c^2\)"),
        ("v = x_{12}/t^2", r"\(v = x_{12}/t^2\)"),
        ("x^2 >= y + 1", r"\(x^2 >= y + 1\)"),
        ("x = y^{-2}.", r"\(x = y^{-2}\)."),
        ("(E = mc^2), then y = x^3.", r"(\(E = mc^2\)), then \(y = x^3\)."),
        ("| formula | y = x^2 |", r"| formula | \(y = x^2\) |"),
        ("F = kx^2 and z = y_2.", r"\(F = kx^2\) and \(z = y_2\)."),
    ],
)
def test_recovers_a_small_algebra_language_without_rewriting_source(source, expected):
    assert recover_math_delimiters(source) == expected
    assert recover_math_delimiters(expected) == expected
    assert expected.replace(r"\(", "").replace(r"\)", "") == source
    assert 'class="math-inline"' in render_markdown_html(expected, "/files")


@pytest.mark.parametrize(
    "source",
    [
        "A version has 3.3B parameters, base_size=1024 and image_size=640.",
        "A = B is an ordinary label; C is a category.",
        "E=mc^2",  # compact assignment/query syntax is ambiguous
        "value = x^2",  # prose or a program identifier, not a short math symbol
        "object.x = y^2",
        "foo_bar = x^2",
        "x = version^2",
        "x = y^2.property",
        "x = y^2file",
        "x = y_1234",
        "x = y_12",  # unbraced multi-digit scripts change meaning in TeX
        "x = y^12",
        "x = y^-2",
        "x = y^{1234}",
        "x = y^2 + unknown",
        "x = y^2 /invalid/path",
        "x = y^2 = z",
        "x^2 + length = y^2",
        "const x = y^2;",
        "int x = y^2;",
        "if x = y^2:",
        "    x = y^2\n\tx = z^2\n",
        "<|ref|>text<|/ref|><|det|>[1,2,3,4]<|/det|>    x = y^2\n",
        "`x = y^2`",
        "``x = y^2 ` and a = b^2``",
        "```python\nx = y^2\n```\n",
        "   ~~~~\nx = y^2\n   ~~~~\n",
        "```unclosed\nx = y^2\n",
        "> ```python\n> x = y^2\n> ```still code\n> x = z^3\n> ```\n",
        "<|ref|>text<|/ref|><|det|>[1,2,3,4]<|/det|>```python\nx = y^2\n"
        "```still code\nx = z^3\n```\n",
        "<PRE>\nx = y^2\n</pRe >\n",
        "<code>x = y^2</code>",
        "'''x = y^2\nz = w^2\n'''",
        "<|ref|>code<|/ref|><|det|>[1,2,3,4]<|/det|>x = y^2\nx = z^2\n",
        "<|det|>algorithm [1,2,3,4]<|/det|>x = y^2\nx = z^2\n",
        '<a href="x = y^2">link</a>',
        '<a title="x > 0; y = z^2">label</a>',
        '<a title="x > 0;\ny = z^2">label</a>',
        "<!-- x = y^2 -->",
        "\"x = y^2\" and 'a = b^2'",
        "https://example.org/?q=E = mc^2",
        "www.example.org/?q=E = mc^2",
        "mailto:E = mc^2@example.org",
        "/tmp/E = mc^2",
        r"C:\E = mc^2",
        "E = mc^2@example.org",
        "[x = y^2](https://example.org/?q=x = y^2)",
        "[x = y^2][equation]",
        r"\(x = y^2\)",
        r"\[x = y^2\]",
        "$x = y^2$",
        "$$\nx = y^2\n$$",
        "$5 is the price and x = y^2",
        "\\`x = y^2",  # escaped literal delimiter is retained
    ],
)
def test_preserves_non_math_ambiguous_syntax_and_literal_regions(source):
    assert recover_math_delimiters(source) == source


def test_only_literal_regions_are_protected_and_surrounding_math_is_recovered():
    source = "`x = y^2` and a = b^2; [x = y^2](url) then c = d^3\n"
    expected = r"`x = y^2` and \(a = b^2\); [x = y^2](url) then \(c = d^3\)" + "\n"
    assert recover_math_delimiters(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "Formula: a^2 + b^2 = c^2.\nThen F = kx^2.",
        "`x = y^2\nstill x = z^2` and a = b^2\n",
        "``a = b^2 ` then x = y^2\n`` and z = w^3\n",
        "~~~python\nx = y^2\n~~~\nx = z^3\n",
        "<PRE>\nx = y^2\n</pRe >\nx = z^3\n",
        "<!--\nx = y^2\n-->\nx = z^3\n",
        "'''x = y^2\nz = w^2\n''' and a = b^2\n",
        '<a title="x > 0;\ny = z^2">label</a> then x = w^2\n',
        r"\[x = y^2" + "\nstill z = w^2" + r"\] then a = b^2",
        "[x = y^2](https://example.org/(nested)/?x = y^2) and z = w^3\n",
        "<|ref|>code<|/ref|><|det|>[1,2,3,4]<|/det|>x = y^2\n"
        "<|ref|>text<|/ref|><|det|>[1,2,3,4]<|/det|>z = w^3\n",
        "x = y^2" + " " * 4100 + "\nx = z^3\n",
        "```" + " " * 4100 + "\nx = y^2\n```\nx = z^3\n",
        "```python\nx = y^2\n```" + " " * 4100 + "still code\nx = z^3\n```\n",
        "x" * 4100 + "<code>\ny = z^2\n</code>\n",
        "<PAGE>`unclosed\n<PAGE>x = y^2\n",
        "<PAGE>" + "x" * 4100 + "<PAGE>y = z^2\n",
        "<PAGE>`literal <PAGE> x = y^2`\n",
        "<PAGE>sigma = sqrt((1/N) * sum(x_i - mu)^2)\n",
        "<PAGE>```python\nsigma = sqrt(x_i^2)\n```\n<PAGE>rho_i = sum(x_i^2)\n",
    ],
)
def test_stream_is_invariant_across_every_boundary_and_character_deltas(source):
    expected = recover_math_delimiters(source)
    for split in range(len(source) + 1):
        sink = Sink()
        stream = MathDelimiterStream(sink)
        stream.on_text(source[:split])
        stream.on_text(source[split:])
        stream.end()
        assert sink.text == expected
    sink = Sink()
    stream = MathDelimiterStream(sink)
    for char in source:
        stream.on_text(char)
    stream.end()
    assert sink.text == expected


def test_long_lines_are_bounded_and_pass_through():
    sink = Sink()
    stream = MathDelimiterStream(sink)
    source = "x = y^2 " + "a" * 10000
    for char in source:
        stream.on_text(char)
        assert len(stream.pending) <= 4096
    stream.end()
    assert sink.text == source


def test_uncertain_long_line_context_stays_literal_until_next_ocr_page():
    source = "<PAGE>" + "x" * 4100 + "<code>\ny = z^2\n<PAGE>y = z^2\n"
    expected = "<PAGE>" + "x" * 4100 + "<code>\ny = z^2\n<PAGE>" + r"\(y = z^2\)" + "\n"
    assert recover_math_delimiters(source) == expected


@pytest.mark.parametrize("opener", ["`unclosed", "<code>", r"\[", "```"])
def test_ocr_pages_have_identical_literal_context_in_saved_streamed_and_layout_text(
    opener,
):
    header = "<|ref|>text<|/ref|><|det|>[0,0,999,999]<|/det|>"
    pages = [header + opener + "\n", header + "y = x^2\n"]
    recovered = recover_math_delimiters("<PAGE>" + "<PAGE>".join(pages))
    for raw, presented in zip(pages, recovered.split("<PAGE>")[1:]):
        assert parse_page_blocks(raw) == parse_page_blocks(presented)
    assert r"\(y = x^2\)" in recovered


def test_v2_block_stream_and_final_artifacts_use_the_same_math_without_altering_raw(
    tmp_path,
):
    raw = "<PAGE><BLOCK>text[10,20,900,200]<FORK>The energy is E = mc^2.\n"
    raw += "<BLOCK>text[10,210,900,300]<FORK>y = x_1^2\n"
    normalized = recover_math_delimiters(normalize_blocks(raw))
    for split in range(len(raw) + 1):
        sink = Sink()
        stream = BlockStream(sink)
        stream.on_text(raw[:split])
        stream.on_text(raw[split:])
        stream.end()
        assert sink.text == normalized

    with Image.new("RGB", (100, 100)) as image:
        final = save_results_multi(raw, [image], str(tmp_path / "multi"))
        single = save_results_single(
            raw.removeprefix("<PAGE>"), image, str(tmp_path / "single")
        )
    assert r"\(E = mc^2\)" in final and r"\(y = x_1^2\)" in final
    assert r"\(E = mc^2\)" in single
    assert (tmp_path / "multi/result.md").read_text() == final
    assert (tmp_path / "single/result.md").read_text() == single
    raw_page = json.loads((tmp_path / "multi/raw_pages.json").read_text())["pages"][0]
    assert raw_page == normalize_blocks(raw).removeprefix("<PAGE>").strip()
    assert r"\(" not in raw_page
    blocks = parse_page_blocks(raw_page)
    assert r"\(E = mc^2\)" in blocks[0]["content"]
    assert 'class="math-inline"' in render_layout_html(
        [{"page": 1, "blocks": blocks}], "/files"
    )


def test_whole_line_named_math_is_identical_in_artifacts_stream_and_layout(tmp_path):
    formula = "sigma = sqrt((1/N) * sum(x_i - mu)^2)"
    raw = "<PAGE><BLOCK>text[10,20,900,200]<FORK>" + formula + "\n"
    normalized = recover_math_delimiters(normalize_blocks(raw))
    for split in range(len(raw) + 1):
        sink = Sink()
        stream = BlockStream(sink)
        stream.on_text(raw[:split])
        stream.on_text(raw[split:])
        stream.end()
        assert sink.text == normalized
    with Image.new("RGB", (100, 100)) as image:
        final = save_results_multi(raw, [image], str(tmp_path / "multi"))
        single = save_results_single(
            raw.removeprefix("<PAGE>"), image, str(tmp_path / "single")
        )
    expected = recover_math_delimiters(formula)
    assert final.strip() == "<PAGE>\n" + expected
    assert single.strip() == expected
    page = json.loads((tmp_path / "multi/raw_pages.json").read_text())["pages"][0]
    assert page == normalize_blocks(raw).removeprefix("<PAGE>").strip()
    assert parse_page_blocks(page)[0]["content"] == expected
    assert r"\(" not in page and r"\operatorname" not in page
