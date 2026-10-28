"""Raster occurrence extraction and engine output contracts."""
import io
import json
import threading

import fitz
import pytest
from PIL import Image

from app.config import Settings
from app.engine import textlayer
from app.engine.base import JobCanceled


def document(tmp_path, rotation=0, fullpage=False, raster=True):
    pages = tmp_path / 'pages'
    pages.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((20, 30), 'Native text remains available for this page')
    if raster:
        png = io.BytesIO()
        Image.new('RGBA', (40, 40), (255, 0, 0, 128)).save(png, format='PNG')
        box = page.rect if fullpage else fitz.Rect(40, 100, 120, 180)
        xref = page.insert_image(box, stream=png.getvalue(), keep_proportion=False)
        if not fullpage:
            page.insert_image(fitz.Rect(200, 300, 280, 380), xref=xref)
            page.insert_image(box, xref=xref)  # same position deduplicates
    else:
        page.draw_rect(fitz.Rect(40, 100, 120, 180), color=(1, 0, 0))
    page.set_rotation(rotation)
    source = tmp_path / 'source.pdf'
    doc.save(source)
    image = pages / 'page_0001.png'
    page.get_pixmap().save(image)
    doc.close()
    return source, image


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_raster_occurrences_rotation_and_transparency(tmp_path, rotation):
    source, image = document(tmp_path, rotation)
    inspected = textlayer.native_page_local(source, 0)
    assert len(inspected['figures']) == 2
    assert inspected['figure_warning'] is None
    assert 'Native text' in inspected['blocks'][0][1]
    with fitz.open(source) as doc:
        expected = fitz.Rect(40, 100, 120, 180) * doc[0].rotation_matrix
        rect = doc[0].rect
        assert inspected['figures'][0] == pytest.approx(
            (expected.x0 / rect.width, expected.y0 / rect.height,
             expected.x1 / rect.width, expected.y1 / rect.height))


@pytest.mark.parametrize('kwargs', [{'fullpage': True}, {'raster': False}])
def test_fullpage_and_vectors_are_not_figures(tmp_path, kwargs):
    source, _ = document(tmp_path, **kwargs)
    assert textlayer.native_page_local(source, 0)['figures'] == []


@pytest.mark.parametrize('single', [False, True])
def test_engine_stream_links_crops_and_boxes(tmp_path, single):
    source, image = document(tmp_path, rotation=90)
    engine = textlayer.TextLayerEngine(Settings(native_text_threshold=1))
    streamed = []
    class Sink:
        on_text = staticmethod(streamed.append)
    output = tmp_path / 'output'
    if single:
        markdown = engine.run_single(image, output, Sink(), threading.Event())
        name = '0.jpg'
    else:
        markdown = engine.run_multi([image], output, Sink(), threading.Event())
        name = 'page_0_0.jpg'
    assert f'![](images/{name})' in markdown
    assert f'![](images/{name})' in ''.join(streamed)
    boxes = json.loads((output / 'boxes.json').read_text())
    assert len(boxes) == 2
    assert boxes[name]['image_width'] == 600
    with Image.open(output / 'images' / name) as cropped:
        assert cropped.size == (80, 80)
        r, g, b = cropped.getpixel((40, 40))
        assert r > 240 and 30 < g < 230 and abs(g-b) < 5
    assert 'image [' in (output / 'raw_pages.json').read_text()


def test_figure_failure_preserves_text_and_warns(tmp_path, monkeypatch):
    source, image = document(tmp_path)
    engine = textlayer.TextLayerEngine(Settings(native_text_threshold=1))
    result = textlayer.native_page_local(source, 0)
    monkeypatch.setattr(engine, '_isolated_page', lambda *args: result)
    image.write_bytes(b'broken png')
    class Sink:
        def on_text(self, text):
            pass
    markdown = engine.run_single(image, tmp_path / 'out', Sink(), threading.Event())
    assert 'Native text' in markdown
    assert '⚠️' in markdown
    assert engine.drain_warnings()


def test_worker_cancellation_is_not_swallowed(tmp_path, monkeypatch):
    from app.pipeline import pdf_worker
    def canceled(*args, **kwargs):
        raise pdf_worker.PdfWorkerCanceled()
    monkeypatch.setattr(pdf_worker, 'run_page', canceled)
    engine = textlayer.TextLayerEngine(Settings())
    with pytest.raises(JobCanceled):
        engine._isolated_page(tmp_path / 'source.pdf', tmp_path / 'page_0001.png', threading.Event())


def test_placement_filter_clips_invalid_and_bounds_count(monkeypatch):
    class Page:
        rect = fitz.Rect(0, 0, 100, 100)
        rotation_matrix = fitz.Matrix(1, 1)
        def get_image_info(self, **kwargs):
            assert kwargs == {'hashes': False, 'xrefs': False}
            return [{'bbox': b} for b in [
                (float('nan'), 0, 20, 20), (110, 110, 120, 120),
                (-10, -10, 20, 20), (-10, -10, 20, 20), (30, 30, 40, 40),
            ]]
    assert textlayer.native_page_figures(fitz, Page()) == [
        (0, 0, .2, .2), (.3, .3, .4, .4)]
    monkeypatch.setattr(textlayer, '_MAX_PAGE_FIGURES', 1)
    with pytest.raises(ValueError):
        textlayer.native_page_figures(fitz, Page())
