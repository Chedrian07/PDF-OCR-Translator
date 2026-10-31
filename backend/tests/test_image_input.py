"""Model-input contrast recovery must preserve source pixels and faint details."""

import pytest
from PIL import Image, ImageDraw

from app.engine.image_input import normalize_model_image


def paper(ink=80, rows=10, mode="RGB"):
    image = Image.new(mode, (100, 100), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 99, rows - 1), fill=ink if mode == "L" else (ink,) * 3)
    return image


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_washed_ink_recovers_without_mutating_source_or_geometry(mode):
    image = paper(mode=mode)
    faint = 230 if mode == "L" else (230,) * 3
    image.putpixel((20, 30), faint)
    before = image.tobytes()
    normalized = normalize_model_image(image)

    assert normalized is not image
    assert normalized.size == image.size and normalized.mode == image.mode
    assert image.tobytes() == before
    black = 0 if mode == "L" else (0,) * 3
    white = 255 if mode == "L" else (255,) * 3
    assert normalized.getpixel((20, 5)) == black
    assert normalized.getpixel((20, 50)) == white
    detail = normalized.getpixel((20, 30))
    assert 0 < (detail if mode == "L" else detail[0]) < 230
    assert normalize_model_image(normalized) is normalized


@pytest.mark.parametrize("ink", [0, 48, 129, 200])
def test_strong_black_and_faint_only_ink_are_unchanged(ink):
    image = paper(ink)
    assert normalize_model_image(image) is image


@pytest.mark.parametrize("ink", [49, 128])
def test_washed_ink_blackpoint_bounds(ink):
    image = paper(ink)
    assert normalize_model_image(image) is not image


@pytest.mark.parametrize("background", [0, 127, 245, 255])
def test_blank_pages_are_unchanged(background):
    image = Image.new("L", (100, 100), background)
    assert normalize_model_image(image) is image


def test_white_paper_fraction_requires_at_least_85_percent():
    image = paper(rows=15)
    assert normalize_model_image(image) is not image
    image.putpixel((0, 15), (80,) * 3)
    assert normalize_model_image(image) is image


def test_dark_specks_do_not_override_the_one_percent_blackpoint():
    image = paper()
    ImageDraw.Draw(image).line((0, 0, 99, 0), fill="black")
    normalized = normalize_model_image(image)
    assert normalized is not image
    assert normalized.getpixel((0, 1)) == (0, 0, 0)


def test_upper_tail_preserves_near_white_distinctions():
    image = Image.new("L", (100, 100), 245)
    ImageDraw.Draw(image).rectangle((0, 0, 99, 9), fill=80)
    image.putpixel((0, 50), 250)
    image.putpixel((1, 50), 255)
    normalized = normalize_model_image(image)
    assert normalized is not image
    assert 0 < normalized.getpixel((2, 50)) < normalized.getpixel((0, 50)) < 255
    assert normalized.getpixel((1, 50)) == 255


def test_photographic_gradient_has_insufficient_white_paper():
    image = Image.new("RGB", (256, 100))
    for x in range(256):
        ImageDraw.Draw(image).line((x, 0, x, 99), fill=(x, 255 - x, x // 2))
    assert normalize_model_image(image) is image


def test_saturated_colors_do_not_trigger_repeated_normalization():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 99, 9), fill=(200, 31, 0))
    assert normalize_model_image(image) is image


@pytest.mark.parametrize("mode", ["1", "P", "RGBA", "CMYK"])
def test_unsupported_modes_preserve_the_original(mode):
    image = Image.new(mode, (100, 100))
    assert normalize_model_image(image) is image


@pytest.mark.parametrize("single", [False, True])
def test_mlx_model_views_leave_original_crop_pixels_intact(tmp_path, single):
    from app.vendor.unlimited_ocr_mlx.processing import prepare_multi, prepare_single

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [2] * len(text)

    original = paper()
    path = tmp_path / "page.png"
    original.save(path)
    inputs = (
        prepare_single(Tokenizer(), "<image>Read.", path, base_size=100, image_size=100)
        if single
        else prepare_multi(Tokenizer(), "<image>Read.", [path], image_size=100)
    )
    assert inputs.images[0].tobytes() == original.tobytes()
    assert inputs.global_views[0, 5, 20, 0] == -1.0
