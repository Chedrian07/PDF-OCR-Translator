"""Pixel-only figure refinements must abstain without a complete rectangle."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.engine.figure_geometry import refine_figure_box


VISIBLE = (120, 100, 520, 360)
PROPOSAL = (116, 96, 524, 364)


def _rectangle(*, outline=None, fill=None, width=1):
    image = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(image).rectangle((120, 100, 519, 359), outline=outline, fill=fill, width=width)
    return image


@pytest.mark.parametrize("color", [(20, 35, 60), (80, 110, 210)])
def test_finds_unboxed_raster_color_boundary(color):
    with _rectangle(fill=color) as image:
        assert refine_figure_box(image, PROPOSAL) == VISIBLE
        assert refine_figure_box(image, VISIBLE) == VISIBLE


@pytest.mark.parametrize("color", [(180, 180, 190), (75, 100, 205)])
@pytest.mark.parametrize("proposal", [PROPOSAL, (125, 105, 515, 355)])
def test_retains_the_entire_outer_frame_stroke(color, proposal):
    with _rectangle(outline=color, width=3) as image:
        assert refine_figure_box(image, proposal) == VISIBLE


def test_content_and_nearby_text_do_not_move_a_supported_outer_boundary():
    with _rectangle(outline=(90, 100, 180), width=3) as image:
        draw = ImageDraw.Draw(image)
        for x, top in [(180, 200), (250, 150), (320, 230), (390, 180)]:
            draw.rectangle((x, top, x + 40, 330), fill=(90, 120, 210))
        draw.text((125, 78), "Neighboring title and a chart label", fill="black")
        draw.text((125, 368), "Caption outside the visible frame", fill="black")
        assert refine_figure_box(image, PROPOSAL) == VISIBLE


@pytest.mark.parametrize("color", ["black", (235, 235, 235)])
def test_visible_material_outside_a_frame_is_not_discarded(color):
    with _rectangle(outline="black", width=3) as image:
        ImageDraw.Draw(image).line((118, 190, 118, 210), fill=color)
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


@pytest.mark.parametrize("color", ["white", "black", (220, 220, 220)])
def test_uniform_page_has_no_boundary_evidence(color):
    with Image.new("RGB", (640, 480), color) as image:
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


@pytest.mark.parametrize("missing_side", ["top", "bottom", "left", "right"])
def test_incomplete_frame_and_unboxed_axes_are_not_tightened(missing_side):
    with Image.new("RGB", (640, 480), "white") as image:
        draw = ImageDraw.Draw(image)
        for side, line in {
            "top": (120, 100, 519, 100), "bottom": (120, 359, 519, 359),
            "left": (120, 100, 120, 359), "right": (519, 100, 519, 359),
        }.items():
            if side != missing_side:
                draw.line(line, fill="black", width=2)
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


def test_text_and_rules_do_not_count_as_a_rectangular_image():
    with Image.new("RGB", (640, 480), "white") as image:
        draw = ImageDraw.Draw(image)
        for y in range(110, 350, 18):
            draw.text((120, y), "Repeated text across a table row " * 2, fill="black")
        draw.line((110, 100, 530, 100), fill="black", width=2)
        draw.line((110, 359, 530, 359), fill="black", width=2)
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


def test_competing_nearby_frames_are_ambiguous():
    with _rectangle(outline="black") as image:
        ImageDraw.Draw(image).rectangle((124, 104, 515, 355), outline="black")
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


def test_a_thick_frame_keeps_its_outer_edge():
    with _rectangle(outline="black", width=14) as image:
        assert refine_figure_box(image, PROPOSAL) == VISIBLE


def test_a_proposal_inside_a_thick_frame_cannot_snap_to_its_inner_edge():
    proposal = (130, 110, 510, 350)
    with _rectangle(outline="black", width=14) as image:
        assert refine_figure_box(image, proposal) == proposal


def test_large_required_movement_is_rejected():
    proposal = (100, 80, 540, 380)
    with _rectangle(fill="navy") as image:
        assert refine_figure_box(image, proposal) == proposal


def test_skewed_frame_is_preserved():
    with _rectangle(outline="black", width=3) as image:
        with image.rotate(1.4, resample=Image.Resampling.BICUBIC, fillcolor="white") as skewed:
            assert refine_figure_box(skewed, PROPOSAL) == PROPOSAL


def test_faint_noise_and_page_gradient_do_not_fabricate_edges():
    rng = np.random.default_rng(12)
    gray = np.clip(np.linspace(230, 250, 640)[None, :] + rng.integers(-3, 4, (480, 640)), 0, 255)
    with Image.fromarray(gray.astype(np.uint8)).convert("RGB") as image:
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


def test_page_without_a_consistent_background_is_not_refined():
    with _rectangle(fill="navy") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 7, 7), fill="red")
        draw.rectangle((632, 0, 639, 7), fill="green")
        draw.rectangle((0, 472, 7, 479), fill="blue")
        assert refine_figure_box(image, PROPOSAL) == PROPOSAL


@pytest.mark.parametrize("box", [(0, 0, 640, 480), (-1, 96, 524, 364), (120, 100, 120, 360), (10, 10, 25, 25)])
def test_page_edges_invalid_or_tiny_proposals_are_left_unchanged(box):
    with _rectangle(fill="navy") as image:
        assert refine_figure_box(image, box) == box
