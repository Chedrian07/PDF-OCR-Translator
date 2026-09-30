"""Impossible PDF layouts are rejected before expensive native word wrapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.pipeline.pdf_export import _plan_shrink_to_fit, _plan_single_line, _resolve_font
from app.pipeline.pdf_export import fitting
from app.pipeline.pdf_export.fitting import _text_exceeds_box_capacity


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_oversized_translation_skips_native_textbox(monkeypatch, rotation):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.set_rotation(rotation)

    def unexpected_textbox(*_args, **_kwargs):
        pytest.fail("an impossible oversized block reached native word wrapping")

    monkeypatch.setattr(fitz.Shape, "insert_textbox", unexpected_textbox)
    plan = _plan_shrink_to_fit(
        page,
        fitz.Rect(60, 85, 535, 250),
        "매우 긴 번역문 " * 20_000,
        12,
        "korea",
        None,
        max_rect=fitz.Rect(60, 85, 535, 837),
        scales=(1.0, 0.46),
        lineheights=(None, 1.44),
    )
    assert plan is None
    doc.close()


def test_oversized_single_line_skips_full_string_measurement(monkeypatch):
    import fitz

    fontfile, fontname = _resolve_font("")
    if fontfile is None:
        pytest.skip("no file-backed Korean font is installed")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    original_measure = fitz.Font.text_length

    def measure_glyphs_only(font, text, **kwargs):
        assert len(text) <= 1, "an impossible oversized block measured its full text"
        return original_measure(font, text, **kwargs)

    monkeypatch.setattr(fitz.Font, "text_length", measure_glyphs_only)
    assert _plan_single_line(
        page,
        fitz.Rect(60, 85, 535, 250),
        "매우 긴 번역문 " * 20_000,
        12,
        fontname,
        fontfile,
        max_rect=fitz.Rect(60, 85, 535, 837),
    ) is None
    doc.close()


@pytest.mark.parametrize("text", ["Ordinary translation", "A" * 50, "  A\t B\n C  "])
def test_capacity_guard_retains_successful_native_layout(text):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    font = fitz.Font(fontname="helv")
    width = max(10, font.text_length(text.replace("\n", ""), fontsize=10) + 1)
    height = 10 * (font.ascender - 2 * font.descender) + 0.1
    if "\n" in text:
        height *= 3
    box = fitz.Rect(60, 85, 60 + width, 85 + height)
    assert not _text_exceeds_box_capacity(
        text, box, font, "helv", None, 10, (1.0,), (None,), 0,
    )
    plan = _plan_shrink_to_fit(
        page, box, text, 10, "helv", None, max_rect=box, scales=(1.0,),
    )
    assert plan is not None
    doc.close()


@pytest.mark.parametrize("planner", [_plan_shrink_to_fit, _plan_single_line])
def test_capacity_guard_includes_wider_original_candidate(planner):
    """A narrower growth rectangle must not reject the usable original box."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    original = fitz.Rect(60, 85, 460, 110)
    growth = fitz.Rect(60, 85, 70, 110)
    plan = planner(
        page,
        original,
        "A" * 50,
        10,
        "helv",
        None,
        max_rect=growth,
        scales=(1.0,),
    )
    assert plan is not None
    doc.close()


def test_capacity_guard_respects_smaller_explicit_lineheight():
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    font = fitz.Font(fontname="helv")
    box = fitz.Rect(60, 85, 110, 185)
    text = "aaaa " * 40
    assert not _text_exceeds_box_capacity(
        text, box, font, "helv", None, 10, (1.0,), (0.4,), 0,
    )
    assert page.new_shape().insert_textbox(
        box, text, fontsize=10, fontname="helv", lineheight=0.4,
    ) >= 0
    doc.close()


def test_capacity_guard_defers_missing_glyphs():
    import fitz

    font = SimpleNamespace(
        ascender=1.0,
        descender=-0.2,
        has_glyph=lambda _codepoint: False,
        text_length=lambda _text, **_kwargs: 10.0,
    )
    assert not _text_exceeds_box_capacity(
        "A" * 20_000,
        fitz.Rect(0, 0, 10, 10),
        font,
        "fake-font",
        "/fake-font.ttf",
        10,
        (1.0,),
        (None,),
        0,
    )


def test_capacity_guard_defers_failed_font_load(monkeypatch):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    original_metrics = fitting._metrics_font
    native_calls = []

    def metrics(fontfile, fontname):
        if fontfile:
            raise OSError("unreadable font")
        return original_metrics(fontfile, fontname)

    def unexpected_precheck(*_args, **_kwargs):
        pytest.fail("substitute font metrics were used to reject native layout")

    def native_layout(*_args, **_kwargs):
        native_calls.append(True)
        return -1

    monkeypatch.setattr(fitting, "_metrics_font", metrics)
    monkeypatch.setattr(fitting, "_text_exceeds_box_capacity", unexpected_precheck)
    monkeypatch.setattr(fitz.Shape, "insert_textbox", native_layout)
    assert _plan_shrink_to_fit(
        page,
        fitz.Rect(60, 85, 535, 250),
        "A" * 20_000,
        10,
        "unreadable-font",
        "/unreadable-font.ttf",
    ) is None
    assert native_calls
    doc.close()


def test_capacity_guard_ignores_discardable_whitespace_and_zero_width_glyphs():
    import fitz

    font = SimpleNamespace(
        ascender=1.0,
        descender=-0.2,
        has_glyph=lambda _codepoint: True,
        text_length=lambda _text, **_kwargs: 0.0,
    )
    assert not _text_exceeds_box_capacity(
        " \t\n\u200b" * 20_000,
        fitz.Rect(0, 0, 10, 10),
        font,
        "fake-zero-width",
        "/fake-font.ttf",
        10,
        (1.0,),
        (None,),
        0,
    )


@pytest.mark.parametrize("width", [float("nan"), -1.0])
def test_capacity_guard_defers_uncertain_font_metrics(width):
    import fitz

    font = SimpleNamespace(
        ascender=1.0,
        descender=-0.2,
        has_glyph=lambda _codepoint: True,
        text_length=lambda _text, **_kwargs: width,
    )
    assert not _text_exceeds_box_capacity(
        "A" * 20_000,
        fitz.Rect(0, 0, 10, 10),
        font,
        "fake-font",
        "/fake-font.ttf",
        10,
        (1.0,),
        (None,),
        0,
    )
