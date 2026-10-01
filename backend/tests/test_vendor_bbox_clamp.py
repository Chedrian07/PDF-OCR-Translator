"""[vendor patch P22] draw_bounding_boxes 좌표 clamp — 모델 출력 좌표를 그대로 crop하지 않는다.

모델 출력은 PDF 내용으로 유도할 수 있다. 거대·음수·뒤집힌 좌표가 crop/draw로 가면
거대한 검정 크롭(메모리)이나 Pillow 좌표 산술 오버플로(GHSA-6r8x-57c9-28j4)에 닿는다
(audit gap2-dependency-vuln-reachability-2). 규칙(MLX 엔진 이식과 공유):
x/y = int(v/999*size), x는 [0, W]·y는 [0, H]로 clamp, x2<=x1 또는 y2<=y1이면 건너뜀.
건너뛴 image 상자도 번호를 소비해 마크다운 참조(images/{prefix}{idx}.jpg)와 정렬된다.
"""

import json

import pytest

pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from app.vendor.unlimited_ocr.modeling_unlimitedocr import (  # noqa: E402
    _clamp_box,
    draw_bounding_boxes,
    re_match,
)

W, H = 200, 100


@pytest.mark.parametrize(
    "points,expected",
    [
        ([0, 0, 999, 999], (0, 0, 200, 100)),
        ([100, 200, 500, 800], (20, 20, 100, 80)),
        ([1_298_337_167, 0, 1_298_337_167 * 2, 999], None),  # 둘 다 W로 clamp → 퇴화
        ([0, 0, 1_298_337_167, 1_298_337_167], (0, 0, 200, 100)),  # 거대 끝점 → 이미지 경계
        ([-500, -10, 500, 500], (0, 0, 100, 50)),  # 음수 → 0
        ([500, 500, 100, 100], None),  # 뒤집힌 상자
        ([300, 300, 300, 600], None),  # 폭 0
        ([10**400, 0, 10**401, 999], None),  # float 변환 오버플로
        ([1e999, 0, 999, 999], None),  # inf
        ([True, 0, 999, 999], None),  # bool은 좌표가 아니다
        (["0", 0, 999, 999], None),  # 문자열
        ([0, 0, 999], None),  # 개수 오류
        ([[0, 0, 999, 999]], None),  # 중첩
    ],
)
def test_clamp_box_rule(points, expected):
    assert _clamp_box(points, W, H) == expected


def _draw(tmp_path, text: str):
    (tmp_path / "images").mkdir(exist_ok=True)
    refs, image_matches, _ = re_match(text)
    out = draw_bounding_boxes(Image.new("RGB", (W, H), "white"), refs, str(tmp_path), "page_0_")
    boxes_path = tmp_path / "boxes.json"
    boxes = json.loads(boxes_path.read_text()) if boxes_path.exists() else {}
    return out, image_matches, boxes


def _image_ref(coords: str) -> str:
    return f"<|ref|>image<|/ref|><|det|>{coords}<|/det|>\n"


def test_hostile_coordinates_never_produce_giant_crops(tmp_path):
    text = (
        _image_ref("[[0, 0, 1298337167, 1298337167]]")  # 거대 → 페이지 전체로 clamp
        + _image_ref("[[-999999, -999999, 500, 500]]")  # 음수 → 0
        + "<|ref|>title<|/ref|><|det|>[[-5, -5, 99999999, 99999999]]<|/det|># T\n"
        + "<|det|>text [[999, 999, -999, -999]]<|/det|>뒤집힌 텍스트 상자\n"
    )
    out, image_matches, boxes = _draw(tmp_path, text)

    assert out.size == (W, H)  # 오버레이 결과는 원본 크기 그대로, 예외 없음
    assert len(image_matches) == 2
    assert sorted(boxes) == ["page_0_0.jpg", "page_0_1.jpg"]
    for name, box in boxes.items():
        assert 0 <= box["x1"] < box["x2"] <= W and 0 <= box["y1"] < box["y2"] <= H, (name, box)
        with Image.open(tmp_path / "images" / name) as crop:
            assert crop.size == (box["x2"] - box["x1"], box["y2"] - box["y1"])
            assert crop.size[0] <= W and crop.size[1] <= H
    assert (boxes["page_0_0.jpg"]["x2"], boxes["page_0_0.jpg"]["y2"]) == (W, H)
    assert (boxes["page_0_1.jpg"]["x1"], boxes["page_0_1.jpg"]["y1"]) == (0, 0)


def test_skipped_image_boxes_keep_markdown_numbering(tmp_path):
    """퇴화·해석 불가 image 상자는 파일을 만들지 않지만 번호는 소비한다 — 뒤 그림의 파일
    번호가 마크다운 참조 번호(enumerate(image_matches))와 같아야 한다."""
    text = (
        _image_ref("[[500, 500, 100, 100]]")  # 0: 뒤집힘 → 건너뜀
        + _image_ref("[[0, 0, abc, 1]]")  # 1: 좌표 해석 불가 → 건너뜀
        + _image_ref("[[100, 100, 400, 600]]")  # 2: 정상
    )
    _, image_matches, boxes = _draw(tmp_path, text)

    assert len(image_matches) == 3
    assert sorted(boxes) == ["page_0_2.jpg"]  # 정상 그림은 마크다운 3번째 참조와 같은 번호
    assert not (tmp_path / "images" / "page_0_0.jpg").exists()
    assert not (tmp_path / "images" / "page_0_1.jpg").exists()
    assert (tmp_path / "images" / "page_0_2.jpg").exists()
    assert boxes["page_0_2.jpg"] == {
        "x1": 20, "y1": 10, "x2": 80, "y2": 60, "image_width": W, "image_height": H,
    }


def test_boxes_json_only_lists_files_that_exist(tmp_path):
    text = "".join(
        _image_ref(coords)
        for coords in (
            "[[0, 0, 999, 999]]",
            "[[999, 0, 999, 999]]",  # 폭 0
            "[[10, 10, 20, 20]]",
            "[[1e999, 0, 999, 999]]",  # inf
        )
    )
    _, _, boxes = _draw(tmp_path, text)
    on_disk = sorted(p.name for p in (tmp_path / "images").iterdir())
    assert sorted(boxes) == on_disk == ["page_0_0.jpg", "page_0_2.jpg"]
