"""렌더된 그림의 자동 로드 범위 — 잡이 만든 그림과 data: 래스터만 <img>.

마크다운 본문은 OCR 모델이나 PDF 텍스트 레이어(보이지 않는 텍스트 포함)가 그대로
옮겨 적은 비신뢰 입력이다. `![](https://tracker/…)`가 <img>로 나가면 문서를 여는
순간 브라우저가 클릭 없이 제3자·내부망으로 요청한다(열람 사실·IP·Referer 유출).
"""

import re
import threading

import pytest

from app.pipeline.render import render_document_html, render_markdown_html

BASE = "/api/jobs/j_1/files"


def _img_srcs(html: str) -> list[str]:
    return re.findall(r'<img src="([^"]*)"', html)


def test_job_figures_still_render_as_images():
    html = render_markdown_html("본문 ![](images/p0001_0.jpg) 끝", BASE)
    assert _img_srcs(html) == [f"{BASE}/images/p0001_0.jpg"]


def test_inline_raster_data_uri_still_renders():
    html = render_markdown_html("![](data:image/png;base64,iVBORw0KGgo=)", BASE)
    assert _img_srcs(html) == ["data:image/png;base64,iVBORw0KGgo="]


@pytest.mark.parametrize(
    "src",
    [
        "https://tracker.example/p.png?doc=42",
        "http://192.168.0.1/cgi-bin/luci?reboot=1",
        "//evil.example/a.gif",
        "/api/jobs/j_1/pdf?lang=ko",           # 같은 출처의 비싼 GET도 자동으로 부르지 않는다
        "images/../../api/jobs",               # 경로 탈출
        "images/sub/x.jpg",                    # images/ 바로 아래 파일만
        "IMAGES/x.jpg",
        "x.jpg",
    ],
)
def test_external_or_unexpected_image_sources_never_autoload(src):
    html = render_markdown_html(f"앞 ![그림 설명]({src}) 뒤", BASE)
    assert _img_srcs(html) == [], html
    # 내용은 사라지지 않는다 — 클릭해야 열리는 링크와 대체 텍스트로 남는다
    assert 'class="blocked-image"' in html and "그림 설명" in html
    assert 'rel="noopener noreferrer nofollow"' in html
    assert "앞" in html and "뒤" in html


def test_document_html_applies_the_same_policy_per_page():
    md = "1쪽 ![](images/p0001_0.jpg)\n\n---\n\n2쪽 ![](https://tracker.example/b.gif)"
    html = render_document_html(md, BASE, page_separator="\n\n---\n\n")
    assert _img_srcs(html) == [f"{BASE}/images/p0001_0.jpg"]
    assert html.count('class="doc-page"') == 2


def test_attribute_injection_through_the_url_is_escaped():
    html = render_markdown_html('![x](https://e.example/a.png"onerror="alert(1))', BASE)
    assert "onerror=\"alert" not in html
    assert _img_srcs(html) == []


def test_invisible_text_layer_beacon_does_not_autoload(tmp_path):
    """감사 재현 경로 그대로: 렌더 모드 3(투명) 텍스트 레이어의 이미지 문법이
    textlayer 엔진을 거쳐 마크다운이 돼도 <img>로 나가지 않는다."""
    import pymupdf as fitz

    from app.config import Settings
    from app.engine.base import NullSink
    from app.engine.textlayer import TextLayerEngine
    from app.pipeline.pdf import render_pdf_pages

    job = tmp_path / "job"
    job.mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Visible body text of the paper " * 6, fontsize=9)
    page.insert_text(
        (72, 140), "![](https://tracker.example/p.png?doc=42)", fontsize=2, render_mode=3
    )
    doc.save(str(job / "source.pdf"))
    doc.close()
    pages = render_pdf_pages(job / "source.pdf", job / "pages", 72, 10)
    engine = TextLayerEngine(Settings(engine="textlayer", data_dir=tmp_path / "d"))
    md = engine.run_multi(pages, job / "work", NullSink(), threading.Event())

    assert "tracker.example" in md  # 텍스트 레이어는 그대로 옮겨진다(원문 보존)
    html = render_markdown_html(md, BASE)
    assert not any("tracker.example" in s for s in _img_srcs(html))
