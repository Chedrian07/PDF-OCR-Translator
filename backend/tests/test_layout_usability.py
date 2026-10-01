"""좌표 layout '사용 가능' 판정(artifacts.has_usable_layout) 검증.

figure_only 엔진(OvisOCR2)의 옛 잡은 image 블록만 든 layout.json을 남겼다. 파일
존재만 보면 그런 잡이 has_layout으로 보여 document.html이 OCR 텍스트 없는
facsimile이 되고 /pdf?lang=ko가 번역 안 된 원문을 낸다 — 판정은 내용으로 한다.
"""

import json
import os

from app.pipeline import artifacts


def _write(path, pages) -> None:
    path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")


def test_text_blocks_make_a_layout_usable():
    assert artifacts.layout_has_text_blocks([
        {"page": 1, "blocks": [{"type": "image", "bbox": [0, 0, 9, 9]}]},
        {"page": 2, "blocks": [{"type": "text", "bbox": [0, 0, 9, 9], "content": "본문"}]},
    ])
    # 대소문자·표 같은 비그림 타입도 텍스트 블록이다
    assert artifacts.layout_has_text_blocks([{"blocks": [{"type": "Table"}]}])


def test_image_only_or_empty_layouts_are_not_usable():
    assert not artifacts.layout_has_text_blocks([])
    assert not artifacts.layout_has_text_blocks(None)
    assert not artifacts.layout_has_text_blocks([{"page": 1, "blocks": []}])
    assert not artifacts.layout_has_text_blocks([
        {"page": 1, "blocks": [{"type": "image"}, {"type": "chart"}]},
        {"page": 2, "blocks": [{"type": "figure"}, "garbage", None]},
        "not-a-page",
    ])


def test_has_usable_layout_reads_the_file_per_language(tmp_path):
    assert not artifacts.has_usable_layout(tmp_path)  # 파일 없음

    # Ovis 옛 잡: image 블록만 든 layout.json + 번역이 만든 layout.ko.json
    image_only = [{"page": 1, "width": 10, "height": 10,
                   "blocks": [{"type": "image", "bbox": [1, 1, 9, 9], "image": "p0001_0.jpg"}]}]
    _write(artifacts.layout(tmp_path), image_only)
    _write(artifacts.layout(tmp_path, "ko"), image_only)
    assert not artifacts.has_usable_layout(tmp_path)
    assert not artifacts.has_usable_layout(tmp_path, "ko")

    text = [{"page": 1, "blocks": [{"type": "text", "bbox": [1, 1, 9, 9], "content": "a"}]}]
    _write(artifacts.layout(tmp_path, "ko"), text)
    assert artifacts.has_usable_layout(tmp_path, "ko")
    assert not artifacts.has_usable_layout(tmp_path)


def test_has_usable_layout_follows_rewrites_and_tolerates_garbage(tmp_path):
    path = artifacts.layout(tmp_path)
    _write(path, [{"page": 1, "blocks": [{"type": "image"}]}])
    assert not artifacts.has_usable_layout(tmp_path)

    # 같은 경로를 텍스트 layout으로 다시 쓰면 캐시가 아니라 새 내용으로 판정한다
    _write(path, [{"page": 1, "blocks": [{"type": "title", "content": "제목"}]}])
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert artifacts.has_usable_layout(tmp_path)

    path.write_text("{broken", encoding="utf-8")
    assert not artifacts.has_usable_layout(tmp_path)
    path.write_text('{"page": 1}', encoding="utf-8")  # 리스트가 아닌 JSON
    assert not artifacts.has_usable_layout(tmp_path)
