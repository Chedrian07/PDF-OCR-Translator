"""Conservative pixel-only refinement of already detected rectangular figures.

No content recognition is performed. A proposal moves only when four nearby,
nearly continuous color transitions identify an unambiguous rectangle. Outlined
frames retain their outer stroke; unsupported figures keep the detector's box.
Coordinates use PIL's right/bottom-exclusive crop convention.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_MIN_SPAN = 16
_MIN_CONTRAST = 24
_MIN_SUPPORT = 0.90
_BACKGROUND_TOLERANCE = 16


def _page_background(image):
    """Require agreement among flat page corners before interpreting an edge."""
    colors = []
    for x, y in ((0, 0), (image.width - 8, 0), (0, image.height - 8), (image.width - 8, image.height - 8)):
        with image.crop((x, y, x + 8, y + 8)) as cropped, cropped.convert("RGB") as rgb:
            pixels = np.asarray(rgb, dtype=np.int16).reshape(-1, 3)
        color = np.median(pixels, axis=0)
        if np.mean(np.max(np.abs(pixels - color), axis=1) <= 12) >= _MIN_SUPPORT:
            colors.append(color)
    if len(colors) < 3:
        return None
    background = np.median(colors, axis=0)
    return background if np.count_nonzero(np.max(np.abs(colors - background), axis=1) <= 12) >= 3 else None


def _transitions(image, axis, lower, upper, start, end, outer_low, background):
    """Return boundary coordinates and full-span transition support."""
    size = image.height if axis == 0 else image.width
    lower, upper = max(1, lower), min(size - 1, upper)
    if lower > upper or end - start < _MIN_SPAN:
        return np.array([], dtype=int), np.array([], dtype=float)
    region = (
        (start, lower - 1, end, upper + 1)
        if axis == 0 else (lower - 1, start, upper + 1, end)
    )
    with image.crop(region) as cropped, cropped.convert("RGB") as rgb:
        pixels = np.asarray(rgb, dtype=np.int16)
    change = np.max(np.abs(np.diff(pixels, axis=axis)), axis=2)
    outside = (
        pixels[:-1] if outer_low else pixels[1:]
    ) if axis == 0 else (pixels[:, :-1] if outer_low else pixels[:, 1:])
    matches_background = np.max(np.abs(outside - background), axis=2) <= _BACKGROUND_TOLERANCE
    support = np.mean((change >= _MIN_CONTRAST) & matches_background, axis=1 if axis == 0 else 0)
    return np.arange(lower, upper + 1), support


def _edge(image, axis, predicted, radius, start, end, outer_low, border_gap, background):
    coordinates, support = _transitions(
        image, axis, predicted - radius, predicted + radius, start, end, outer_low, background,
    )
    coordinates = coordinates[support >= _MIN_SUPPORT]
    if not len(coordinates):
        return None
    groups = np.split(coordinates, np.flatnonzero(np.diff(coordinates) > 1) + 1)
    if len(groups) != 1:
        return None
    group = groups[0]
    # Outward pixels already match the page background, so separated candidates
    # are competing outer edges, rather than opposite sides of one stroke.
    if group[-1] - group[0] > border_gap:
        return None
    return int(group[0] if outer_low else group[-1])


def _iou(left, right):
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]),
    )
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    return overlap / max(1, area(left) + area(right) - overlap)


def _removes_visible_pixels(image, original, refined, background):
    """A nearby frame must not discard labels or other ink outside its border."""
    x0, y0, x1, y1 = original
    left, top, right, bottom = refined
    for region in (
        (x0, y0, min(left, x1), y1), (max(right, x0), y0, x1, y1),
        (x0, y0, x1, min(top, y1)), (x0, max(bottom, y0), x1, y1),
    ):
        if region[2] <= region[0] or region[3] <= region[1]:
            continue
        with image.crop(region) as cropped, cropped.convert("RGB") as rgb:
            pixels = np.asarray(rgb, dtype=np.int16)
        if np.any(np.max(np.abs(pixels - background), axis=2) > _BACKGROUND_TOLERANCE):
            return True
    return False


def refine_figure_box(
    image: Image.Image, box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Refine a clamped detector box only with strong four-sided pixel evidence.

    Search is limited to 3.5% of the shorter side, at most 24 pixels. The entire
    replacement must preserve at least 90% IoU with the proposal. White figures,
    unboxed plots, partial/skewed frames, and ambiguous edges remain unchanged.
    Only narrow edge strips are materialized, rather than a page-sized array.
    """
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
        return box
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        return box
    if min(width, height) < 32:
        return box
    background = _page_background(image)
    if background is None:
        return box
    radius = max(1, min(24, int(min(width, height) * 0.035)))
    border_gap = max(2, min(8, round(min(width, height) * 0.015)))
    inset = radius + 2
    horizontal = (x0 + inset, x1 - inset)
    vertical = (y0 + inset, y1 - inset)
    edges = (
        _edge(image, 1, x0, radius, *vertical, True, border_gap, background),
        _edge(image, 0, y0, radius, *horizontal, True, border_gap, background),
        _edge(image, 1, x1, radius, *vertical, False, border_gap, background),
        _edge(image, 0, y1, radius, *horizontal, False, border_gap, background),
    )
    if any(edge is None for edge in edges):
        return box
    left, top, right, bottom = edges
    if (
        left >= right or top >= bottom or _iou(box, edges) < 0.90
        or _removes_visible_pixels(image, box, edges, background)
    ):
        return box
    # Validate all four boundaries against the completed rectangle too. This
    # prevents combining strong lines whose apparent spans do not meet.
    for axis, edge, start, end, outer_low in (
        (1, left, top + 1, bottom - 1, True), (1, right, top + 1, bottom - 1, False),
        (0, top, left + 1, right - 1, True), (0, bottom, left + 1, right - 1, False),
    ):
        _, support = _transitions(image, axis, edge, edge, start, end, outer_low, background)
        if not len(support) or support[0] < _MIN_SUPPORT:
            return box
    return left, top, right, bottom
