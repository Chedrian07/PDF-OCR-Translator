"""Refined figure pixels, saved crop metadata, and layout geometry must agree."""

import json

import pytest
from PIL import Image, ImageDraw

from app.pipeline.merge import ChunkResult, IncrementalMerger, SlotAlignment


SEP = "\n\n---\n\n"
MODEL_BBOX = [116, 96, 524, 364]
RAW = "<|det|>text [0,0,999,80]<|/det|>Body\n<|det|>image [116,96,524,364]<|/det|>"
META = {
    "x1": 120,
    "y1": 100,
    "x2": 520,
    "y2": 360,
    "image_width": 640,
    "image_height": 480,
    "refined": True,
}


def _chunk(job, name, meta, *, single=False):
    directory = job / "work" / name
    (directory / "images").mkdir(parents=True)
    image_name = "0.jpg" if single else "page_0_0.jpg"
    with Image.new("RGB", (400, 260), "navy") as image:
        image.save(directory / "images" / image_name)
    (directory / "boxes.json").write_text(json.dumps({image_name: meta}))
    (directory / "raw_pages.json").write_text(json.dumps({"pages": [RAW]}))
    markdown = f"Body\n![](images/{image_name})"
    return directory, markdown if single else "<PAGE>\n" + markdown


def _figure(merger, page=1):
    layout_page = next(p for p in merger.layout_pages if p["page"] == page)
    return next(b for b in layout_page["blocks"] if b["type"] == "image")


@pytest.mark.parametrize("single", [False, True])
def test_merge_uses_refined_pixel_geometry_after_global_rename_and_preserves_raw(
    tmp_path, single
):
    merger = IncrementalMerger(tmp_path, SEP)
    directory, markdown = _chunk(tmp_path, "initial", META, single=single)
    evidence = (directory / "raw_pages.json").read_bytes()
    merger.add_chunk(ChunkResult(directory, 1, 1, markdown, single=single))

    expected = [120 / 640 * 999, 100 / 480 * 999, 520 / 640 * 999, 360 / 480 * 999]
    assert _figure(merger)["bbox"] == pytest.approx(expected)
    assert _figure(merger)["image"] == "p0001_0.jpg"
    assert (
        json.loads((tmp_path / "layout.json").read_text())[0]["blocks"][1]["bbox"]
        == expected
    )
    assert merger.figure_boxes["p0001_0.jpg"] == META
    assert (directory / "raw_pages.json").read_bytes() == evidence


def test_replace_page_refreshes_refined_geometry_instead_of_reusing_old_crop(tmp_path):
    merger = IncrementalMerger(tmp_path, SEP)
    initial, markdown = _chunk(tmp_path, "initial", {**META, "refined": False})
    merger.add_chunk(ChunkResult(initial, 1, 1, markdown))
    assert _figure(merger)["bbox"] == MODEL_BBOX

    replacement, markdown = _chunk(tmp_path, "replacement", META, single=True)
    evidence = (replacement / "raw_pages.json").read_bytes()
    assert merger.replace_page(1, ChunkResult(replacement, 1, 1, markdown, single=True))
    assert _figure(merger)["bbox"] == pytest.approx(
        [120 / 640 * 999, 100 / 480 * 999, 520 / 640 * 999, 360 / 480 * 999]
    )
    assert merger.figure_boxes["p0001_0.jpg"] == META
    assert (replacement / "raw_pages.json").read_bytes() == evidence


def test_refined_metadata_follows_distinct_global_names_when_model_pages_share_a_slot(
    tmp_path,
):
    merger = IncrementalMerger(tmp_path, SEP)
    directory, first = _chunk(tmp_path, "initial", META)
    with Image.new("RGB", (400, 260), "red") as image:
        image.save(directory / "images/page_1_0.jpg")
    second_meta = {**META, "x1": 130, "x2": 530}
    (directory / "boxes.json").write_text(
        json.dumps(
            {
                "page_0_0.jpg": META,
                "page_1_0.jpg": second_meta,
            }
        )
    )
    (directory / "raw_pages.json").write_text(json.dumps({"pages": [RAW, RAW]}))
    second = "<PAGE>\nBody\n![](images/page_1_0.jpg)"
    merger.add_chunk(
        ChunkResult(
            directory,
            1,
            1,
            first + second,
            alignment=SlotAlignment((0, 0), 2),
        )
    )
    figures = [b for b in merger.layout_pages[0]["blocks"] if b["type"] == "image"]
    assert [b["image"] for b in figures] == ["p0001_0.jpg", "p0001_x1_0.jpg"]
    assert figures[0]["bbox"][0] == pytest.approx(120 / 640 * 999)
    assert figures[1]["bbox"][0] == pytest.approx(130 / 640 * 999)


@pytest.mark.parametrize(
    "update",
    [
        {"refined": False},
        {"refined": "true"},
        {"refined": 1},
        {"x1": True},
        {"x1": "120"},
        {"x1": None},
        {"x1": float("nan")},
        {"x2": float("inf")},
        {"image_width": 0},
        {"image_height": -1},
        {"image_width": float("inf")},
        {"image_height": False},
        {"x1": -1},
        {"x2": 641},
        {"y2": 481},
        {"x1": 520},
        {"y1": 400},
    ],
)
def test_non_refined_or_invalid_metadata_retains_model_bbox(tmp_path, update):
    merger = IncrementalMerger(tmp_path, SEP)
    directory, markdown = _chunk(tmp_path, "initial", {**META, **update})
    merger.add_chunk(ChunkResult(directory, 1, 1, markdown))
    assert _figure(merger)["bbox"] == MODEL_BBOX


def test_legacy_crop_metadata_without_flag_retains_model_bbox(tmp_path):
    merger = IncrementalMerger(tmp_path, SEP)
    directory, markdown = _chunk(
        tmp_path, "initial", {k: v for k, v in META.items() if k != "refined"}
    )
    merger.add_chunk(ChunkResult(directory, 1, 1, markdown))
    assert _figure(merger)["bbox"] == MODEL_BBOX


@pytest.mark.parametrize("visible_rectangle", [True, False])
def test_realigned_page_recrops_and_refines_destination_pixels_not_source_metadata(
    tmp_path, visible_rectangle
):
    (tmp_path / "pages").mkdir()
    with Image.new("RGB", (999, 999), "white") as image:
        if visible_rectangle:
            ImageDraw.Draw(image).rectangle((120, 100, 519, 359), fill="navy")
        image.save(tmp_path / "pages/page_0002.png")
    merger = IncrementalMerger(tmp_path, SEP)
    # Source refinement cannot be inherited: alignment changes the raster.
    directory, markdown = _chunk(tmp_path, "initial", META)
    evidence = (directory / "raw_pages.json").read_bytes()
    merger.add_chunk(
        ChunkResult(
            directory,
            1,
            2,
            markdown,
            alignment=SlotAlignment((1,), 1),
        )
    )

    expected = [120, 100, 520, 360] if visible_rectangle else MODEL_BBOX
    meta = merger.figure_boxes["p0002_0.jpg"]
    assert [meta[k] for k in ("x1", "y1", "x2", "y2")] == expected
    assert meta["image_width"] == meta["image_height"] == 999
    assert (meta.get("refined") is True) == visible_rectangle
    assert _figure(merger, 2)["bbox"] == expected
    assert _figure(merger, 2)["image"] == "p0002_0.jpg"
    with Image.open(tmp_path / "images/p0002_0.jpg") as crop:
        assert crop.size == (expected[2] - expected[0], expected[3] - expected[1])
        if not visible_rectangle:
            assert min(crop.getpixel((crop.width // 2, crop.height // 2))) > 240
    assert (directory / "raw_pages.json").read_bytes() == evidence
