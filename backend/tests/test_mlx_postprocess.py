"""MLX 포팅 후처리(app/vendor/unlimited_ocr_mlx/postprocess.py) — torch 벤더와의 패리티.

후처리는 torch·mlx 어느 쪽도 임포트하지 않는다(PIL/numpy만) — 그래서 이 파일은 mlx가
없는 Linux CI에서도 돈다. torch 벤더와의 흐름 패리티는 torch(+torchvision)가 있을 때만
비교한다: torch ``infer``/``infer_multi``를 모델 없이(스텁 self + P15 generate_fn이 정해진
토큰을 돌려줌) 실행해 같은 출력 문자열에 대한 save_results 산출물을 MLX 포팅과 대조한다.

비교 대상: 반환 마크다운, 파일 이름 집합, result.md, raw_pages.json, boxes.json,
figure 크롭 jpg 바이트. 오버레이(result_with_boxes*.jpg)는 업스트림이 상자 색을 난수로
뽑아 존재만 비교한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.vendor.unlimited_ocr_mlx import postprocess as pp
from app.vendor.unlimited_ocr_mlx.config import STOP_STR

BACKEND = Path(__file__).resolve().parents[1]
TORCH_MODELING = BACKEND / "app" / "vendor" / "unlimited_ocr" / "modeling_unlimitedocr.py"

PAGE_A = (
    "<|det|>title [100, 50, 900, 120]<|/det|>Title One\n"
    "<|det|>text [100, 130, 900, 300]<|/det|>Some text with \\coloneqq and \\eqqcolon.\n"
    "<|ref|>image<|/ref|><|det|>[[120, 320, 880, 700]]<|/det|>\n"
    "<|det|>image [50, 720, 950, 990]<|/det|>\n"
    "<|ref|>image<|/ref|><|det|>[[10, 10, 200, 200], [300, 300, 600, 600]]<|/det|>\n"
    "<|det|>text [1, 2, 3]<|/det|>three coords skipped\n"
    "<|ref|>text<|/ref|><|det|>[[a, b, c, d]]<|/det|>bad literal\n"
    "<|det|>table [10, 10, 20, 20]<|/det|>| a | b |\n"
)
PAGE_B = (
    "<|det|>text [0, 0, 999, 999]<|/det|>Page two text\n"
    "<|det|>image [100, 100, 500, 400]<|/det|>\n"
)
# 경계 밖·퇴화 좌표 — [P22] 대상
PAGE_OOB = (
    "<|det|>image [-50, 100, 1200, 600]<|/det|>\n"
    "<|det|>image [500, 500, 400, 700]<|/det|>\n"
    "<|det|>image [100, 100, 100, 300]<|/det|>\n"
    "<|det|>image [200, 200, 700, 800]<|/det|>\n"
    "<|det|>title [-10, -10, 1500, 50]<|/det|>Big title\n"
)


class PieceTokenizer:
    """generate_fn이 돌려줄 생성 토큰을 정해진 문자열 조각으로 디코드하는 가짜 토크나이저.

    id 1 = EOS 문자열, 10+ = 조각. 프롬프트 인코딩 값은 디코드되지 않으므로 아무 값이나."""

    eos_token_id = 1

    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 7) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            i = int(i)
            out.append(STOP_STR if i == 1 else self.pieces[i - 10])
        return "".join(out)

    def ids_for(self, with_eos: bool = True) -> list[int]:
        return list(range(10, 10 + len(self.pieces))) + ([1] if with_eos else [])


def _write_images(tmp_path: Path, sizes) -> list[str]:
    rng = np.random.default_rng(0)
    paths = []
    for i, (w, h) in enumerate(sizes):
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        p = tmp_path / f"page{i}.png"
        Image.fromarray(a).save(p)
        paths.append(str(p))
    return paths


def _snapshot(out_dir: Path) -> dict[str, bytes | None]:
    """파일 이름 → 바이트 (오버레이 jpg는 존재만)."""
    snap: dict[str, bytes | None] = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_dir():
            snap[str(p.relative_to(out_dir)) + "/"] = None
            continue
        rel = str(p.relative_to(out_dir))
        snap[rel] = None if p.name.startswith("result_with_boxes") else p.read_bytes()
    return snap


def _torch_vendor_has_p22() -> bool:
    return "vendor patch P22" in TORCH_MODELING.read_text(encoding="utf-8")


# ── torch 없이도 도는 단위 검증 ──


def test_postprocess_imports_neither_torch_nor_mlx():
    code = (
        "import sys\n"
        "import app.vendor.unlimited_ocr_mlx.postprocess, app.vendor.unlimited_ocr_mlx.processing\n"
        "print(int('torch' in sys.modules), int('mlx.core' in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["0", "0"]


def test_det_coordinates_are_parsed_without_eval(tmp_path):
    marker = tmp_path / "pwned"
    evil = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    ref = (f"<|det|>text {evil}<|/det|>", "text", evil)
    assert pp.extract_coordinates_and_label(ref, 100, 100) is None
    assert not marker.exists()
    assert pp.extract_coordinates_and_label(("", "image", "[1, 2, 3, 4]"), 10, 10) == (
        "image", [[1, 2, 3, 4]],
    )


def test_decode_outputs_strips_eos_and_whitespace():
    tok = PieceTokenizer(["  hello ", "world  "])
    assert pp.decode_outputs(tok, [10, 11, 1]) == "hello world"
    assert pp.decode_outputs(tok, [10, 11]) == "hello world"


def test_p22_clamps_out_of_range_and_skips_degenerate_boxes(tmp_path):
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (400, 300, 3), dtype=np.uint8))
    out = tmp_path / "out"
    md = pp.save_results_multi("<PAGE>" + PAGE_OOB, [img], str(out))
    W, H = img.size
    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    # 0번: x를 [0, W]로 클램프, 1번(뒤집힘)·2번(폭 0)은 크롭·bbox 없음, 3번은 정상
    assert sorted(boxes) == ["page_0_0.jpg", "page_0_3.jpg"]
    assert boxes["page_0_0.jpg"] == {
        "x1": 0, "y1": int(100 / 999 * H), "x2": W, "y2": int(600 / 999 * H),
        "image_width": W, "image_height": H,
    }
    assert boxes["page_0_3.jpg"]["x2"] == int(700 / 999 * W)
    crop0 = Image.open(out / "images" / "page_0_0.jpg")
    assert crop0.size == (W, int(600 / 999 * H) - int(100 / 999 * H))
    assert not (out / "images" / "page_0_1.jpg").exists()
    assert not (out / "images" / "page_0_2.jpg").exists()
    # 마크다운 번호는 업스트림처럼 매치 순서 그대로 (크롭 실패와 같은 규칙)
    for k in range(4):
        assert f"![](images/page_0_{k}.jpg)" in md
    assert "Big title" in md and "<|det|>" not in md
    assert (out / "result_with_boxes_0.jpg").is_file()


def test_raw_pages_and_result_md_layout(tmp_path):
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    out = tmp_path / "o"
    md = pp.save_results_multi("<PAGE>" + PAGE_B + "<PAGE>extra page\n", [img], str(out))
    raw = json.loads((out / "raw_pages.json").read_text(encoding="utf-8"))
    assert raw == {"pages": [PAGE_B.strip(), "extra page"]}
    # 이미지가 없는 초과 페이지는 원문 그대로 붙는다 (torch와 동일)
    assert md.endswith("\n<PAGE>\nextra page")
    assert (out / "result.md").read_text(encoding="utf-8") == md
    assert (out / "images").is_dir()


# ── [torch vendor patch P23] 페이지 분할 — 첫 <PAGE> 앞을 버리지 않는다 ──

PAGE_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


def _solid_pages(n: int, size=(300, 400)) -> list[Image.Image]:
    """페이지마다 다른 단색 — 크롭이 어느 래스터에서 잘렸는지 픽셀로 드러난다."""
    return [Image.new("RGB", size, PAGE_COLORS[i % 3]) for i in range(n)]


def _crop_color(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return rgb.getpixel((rgb.size[0] // 2, rgb.size[1] // 2))


def _near(color, expected, tol: int = 8) -> bool:  # JPEG 손실 허용
    return max(abs(a - b) for a, b in zip(color, expected, strict=True)) <= tol


@pytest.mark.parametrize(
    "outputs,expected",
    [
        ("<PAGE>\na\n<PAGE>\nb", ["a", "b"]),  # 정상(선행 마커) — 업스트림과 동일
        ("a\n<PAGE>\nb", ["a", "b"]),  # 선행 마커 생략 → 1쪽 보존
        ("only page", ["only page"]),  # 마커 0개 → 전체가 1쪽
        ("  \n<PAGE>\nx", ["x"]),  # 앞이 공백뿐이면 버림
        ("", []),
        ("<PAGE>", [""]),
    ],
)
def test_p23_split_keeps_content_before_first_marker(outputs, expected):
    assert [p.strip() for p in pp._split_multi_pages(outputs)] == expected


def test_p23_split_matches_app_merge_rule():
    """벤더 산출물(raw_pages·크롭 번호)과 병합(merge.split_pages)이 같은 페이지로 나눈다."""
    import random

    from app.pipeline.merge import split_pages

    rng = random.Random(4)
    pieces = ["<PAGE>", "\n", " ", "text", "표", "<|ref|>image<|/ref|>"]
    for _ in range(500):
        outputs = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 12)))
        assert [p.strip() for p in pp._split_multi_pages(outputs)] == split_pages(outputs), outputs


def test_p23_missing_leading_marker_keeps_page_one(tmp_path):
    """모델이 첫 <PAGE>를 생략해도 1쪽이 남고, 크롭은 각자 자기 페이지 래스터에서 잘린다.

    업스트림 분할(split[1:])은 1쪽을 버려 2쪽 내용·좌표가 1쪽 자리로 밀렸다 — page_0_0.jpg가
    2쪽 좌표로 1쪽 래스터를 잘랐다(audit mlx-1)."""
    out = tmp_path / "o"
    page1 = "Page one text\n<|det|>image [100, 100, 500, 400]<|/det|>"
    page2 = "Page two text\n<|det|>image [100, 100, 600, 600]<|/det|>"
    md = pp.save_results_multi(f"{page1}\n<PAGE>{page2}", _solid_pages(2), str(out))

    assert md == (
        "<PAGE>\nPage one text\n![](images/page_0_0.jpg)\n\n"
        "<PAGE>\nPage two text\n![](images/page_1_0.jpg)\n"
    )
    assert (out / "result.md").read_text(encoding="utf-8") == md
    raw = json.loads((out / "raw_pages.json").read_text(encoding="utf-8"))
    assert raw == {"pages": [page1, page2]}
    assert _near(_crop_color(out / "images" / "page_0_0.jpg"), PAGE_COLORS[0])
    assert _near(_crop_color(out / "images" / "page_1_0.jpg"), PAGE_COLORS[1])
    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    assert (boxes["page_0_0.jpg"]["x2"], boxes["page_1_0.jpg"]["x2"]) == (
        int(500 / 999 * 300), int(600 / 999 * 300),
    )
    assert (out / "result_with_boxes_0.jpg").is_file() and (out / "result_with_boxes_1.jpg").is_file()


def test_p23_output_without_any_marker_is_one_page(tmp_path):
    """마지막 1쪽 청크에서 마커를 아예 안 내도 그 쪽이 통째로 1쪽이다 (예전: 빈 결과)."""
    out = tmp_path / "o"
    page = "Only page text\n<|det|>image [100, 100, 500, 400]<|/det|>"
    md = pp.save_results_multi(page, _solid_pages(1), str(out))
    assert md == "<PAGE>\nOnly page text\n![](images/page_0_0.jpg)\n"
    assert json.loads((out / "raw_pages.json").read_text(encoding="utf-8")) == {"pages": [page]}
    assert _near(_crop_color(out / "images" / "page_0_0.jpg"), PAGE_COLORS[0])


# ── [P22] 쓸 수 없는 image 좌표도 크롭 번호를 소비한다 (torch와 같은 규칙) ──

# 좌표를 못 읽거나(literal_eval 실패) 상자로 쓸 수 없는 image det — 실제 모델의 inline 문법
# 기준. 상자 목록이 아닌 det 값(문자열·빈 목록·0)과 기형 평평 목록도 image 매치당 번호 1개다
# (torch-2). 뒤에 정상 그림이 오면 그 그림의 파일 번호가 마크다운 자리 번호와 같아야 한다.
UNUSABLE_IMAGE_DETS = {
    "missing-comma": "<|det|>image [100, 120 500, 420]<|/det|>",
    "three-coords": "<|det|>image [100, 120, 500]<|/det|>",
    "five-coords": "<|det|>image [100, 120, 500, 600, 700]<|/det|>",
    "inf": "<|det|>image [1e400, 0, 500, 500]<|/det|>",
    "float-overflow": "<|det|>image [" + "9" * 400 + ", 0, 500, 500]<|/det|>",
    "int-digit-limit": "<|det|>image [" + "9" * 5000 + ", 0, 500, 500]<|/det|>",
    "bool": "<|det|>image [True, 0, 500, 500]<|/det|>",
    "ref-non-literal": "<|ref|>image<|/ref|><|det|>[[a, b, c, d]]<|/det|>",
    "ref-three-coords": "<|ref|>image<|/ref|><|det|>[[0, 0, 999]]<|/det|>",
    "flat-strings": "<|det|>image ['x', 0, 500, 500]<|/det|>",
    "ref-string-literal": "<|ref|>image<|/ref|><|det|>'abcd'<|/det|>",
    "ref-empty-list": "<|ref|>image<|/ref|><|det|>[]<|/det|>",
    "ref-zero": "<|ref|>image<|/ref|><|det|>0<|/det|>",
}
GOOD_IMAGE_DET = "<|det|>image [100, 450, 800, 900]<|/det|>"


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
        ([0, 0, 999, 999, 5], None),
        ([[0, 0, 999, 999]], None),  # 중첩
        (7, None),  # 상자가 아닌 스칼라
    ],
)
def test_p22_clamp_box_rule(points, expected):
    """torch 벤더 _clamp_box와 같은 규칙 (tests/test_vendor_bbox_clamp.py와 같은 표)."""
    assert pp._clamp_box(points, 200, 100) == expected


@pytest.mark.parametrize("bad", list(UNUSABLE_IMAGE_DETS.values()), ids=list(UNUSABLE_IMAGE_DETS))
def test_p22_unusable_image_box_still_consumes_its_figure_number(tmp_path, bad):
    """예전 MLX 포팅은 이런 image det에서 번호를 소비하지 않아(예외 → ref 통째로 건너뜀) 뒤
    그림이 한 칸 앞 번호로 저장됐다 — 마크다운 첫 자리에 둘째 그림이 붙고 둘째 자리는 깨졌다
    (audit mlx-2·torch-1). bool 좌표는 크롭까지 만들었다."""
    W, H = 300, 400
    out = tmp_path / "o"
    md = pp.save_results_multi(
        f"<PAGE>{bad}\nfig A\n{GOOD_IMAGE_DET}\nfig B", _solid_pages(1, (W, H)), str(out)
    )
    assert md == "<PAGE>\n![](images/page_0_0.jpg)\n\nfig A\n![](images/page_0_1.jpg)\n\nfig B"
    # 쓸 수 없는 그림은 파일이 없고, 정상 그림은 마크다운 둘째 자리 번호로 저장된다
    assert sorted(p.name for p in (out / "images").iterdir()) == ["page_0_1.jpg"]
    assert json.loads((out / "boxes.json").read_text(encoding="utf-8")) == {
        "page_0_1.jpg": {
            "x1": int(100 / 999 * W), "y1": int(450 / 999 * H),
            "x2": int(800 / 999 * W), "y2": int(900 / 999 * H),
            "image_width": W, "image_height": H,
        }
    }


def test_p22_unusable_box_does_not_drop_the_rest_of_its_ref(tmp_path):
    """여러 상자 ref에서 앞 상자를 못 써도 뒤 상자는 잘린다(번호는 상자마다 1개 — 업스트림
    규칙). 예전 포팅은 첫 상자의 예외로 ref 전체를 건너뛰었다."""
    out = tmp_path / "o"
    pp.save_results_multi(
        "<PAGE><|ref|>image<|/ref|><|det|>[[0, 0, 999], [100, 450, 800, 900]]<|/det|>\n"
        + GOOD_IMAGE_DET,
        _solid_pages(1),
        str(out),
    )
    assert sorted(p.name for p in (out / "images").iterdir()) == ["page_0_1.jpg", "page_0_2.jpg"]


# ── torch 벤더 흐름 패리티 ──


@pytest.fixture
def torch_vendor():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from app.vendor.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM

    return UnlimitedOCRForCausalLM


def _stub_self():
    import torch

    class Stub:
        config = types.SimpleNamespace(sliding_window_size=128, sliding_window=128)

        def parameters(self):
            yield torch.zeros(1)

        def disable_torch_init(self):
            pass

    return Stub()


def _gen_fn(generated):
    import torch

    def fn(model, gen_kwargs):
        prompt = gen_kwargs["input_ids"]
        return torch.cat([prompt, torch.tensor([generated], dtype=prompt.dtype)], dim=1)

    return fn


def _run_torch_multi(cls, tok, paths, out_dir, generated):
    return cls.infer_multi(
        _stub_self(), tok, prompt="<image>Multi page parsing.", image_files=paths,
        output_path=str(out_dir), image_size=1024, save_results=True,
        generate_fn=_gen_fn(generated),
    )


def _run_mlx_multi(tok, paths, out_dir, generated):
    from app.vendor.unlimited_ocr_mlx.processing import load_pil_images

    outputs = pp.decode_outputs(tok, generated)
    n_tokens = pp.count_output_tokens(tok, outputs)
    md = pp.save_results_multi(outputs, load_pil_images(paths), str(out_dir))
    return md, n_tokens


@pytest.mark.parametrize("with_eos", [True, False])
def test_multi_flow_matches_torch_infer_multi(torch_vendor, tmp_path, with_eos):
    paths = _write_images(tmp_path, [(340, 440), (500, 260)])
    tok = PieceTokenizer(["<PAGE>", PAGE_A, "<PAGE>", PAGE_B])
    generated = tok.ids_for(with_eos)
    t_md, t_tokens = _run_torch_multi(torch_vendor, tok, paths, tmp_path / "torch", generated)
    m_md, m_tokens = _run_mlx_multi(tok, paths, tmp_path / "mlx", generated)
    assert m_md == t_md
    assert m_tokens == t_tokens
    assert _snapshot(tmp_path / "mlx") == _snapshot(tmp_path / "torch")
    # 내용 점검: 두 박스 ref는 img_idx를 2개 소비, figure 크롭 4+1개
    names = sorted(p.name for p in (tmp_path / "mlx" / "images").glob("*.jpg"))
    assert names == [
        "page_0_0.jpg", "page_0_1.jpg", "page_0_2.jpg", "page_0_3.jpg", "page_1_0.jpg",
    ]


def test_single_flow_matches_torch_infer(torch_vendor, tmp_path):
    (path,) = _write_images(tmp_path, [(1280, 620)])  # 640 초과 → gundam 타일 경로
    tok = PieceTokenizer([PAGE_A])
    generated = tok.ids_for(True)
    t_md = torch_vendor.infer(
        _stub_self(), tok, prompt="<image>document parsing.", image_file=path,
        output_path=str(tmp_path / "torch"), base_size=1024, image_size=640, crop_mode=True,
        save_results=True, generate_fn=_gen_fn(generated),
    )
    from app.vendor.unlimited_ocr_mlx.processing import load_pil_images

    outputs = pp.decode_outputs(tok, generated)
    m_md = pp.save_results_single(outputs, load_pil_images([path])[0], str(tmp_path / "mlx"))
    assert m_md == t_md
    assert _snapshot(tmp_path / "mlx") == _snapshot(tmp_path / "torch")
    assert sorted(p.name for p in (tmp_path / "mlx" / "images").glob("*.jpg")) == [
        "0.jpg", "1.jpg", "2.jpg", "3.jpg",
    ]
    assert (tmp_path / "mlx" / "result_with_boxes.jpg").is_file()


def test_out_of_range_boxes_match_torch_once_p22_lands(torch_vendor, tmp_path):
    if not _torch_vendor_has_p22():
        pytest.skip("torch 벤더에 P22(좌표 클램프)가 아직 없음 — lane b-torch 통합 후 비교")
    paths = _write_images(tmp_path, [(300, 400)])
    tok = PieceTokenizer(["<PAGE>", PAGE_OOB])
    generated = tok.ids_for(True)
    t_md, _ = _run_torch_multi(torch_vendor, tok, paths, tmp_path / "torch", generated)
    m_md, _ = _run_mlx_multi(tok, paths, tmp_path / "mlx", generated)
    assert m_md == t_md
    assert _snapshot(tmp_path / "mlx") == _snapshot(tmp_path / "torch")


def test_re_match_and_coordinates_match_torch(torch_vendor):
    from app.vendor.unlimited_ocr import modeling_unlimitedocr as tv

    for text in (PAGE_A, PAGE_B, PAGE_OOB, "", "plain text only"):
        assert pp.re_match(text) == tv.re_match(text)
        for ref in pp.re_match(text)[0]:
            assert pp.extract_coordinates_and_label(ref, 640, 480) == tv.extract_coordinates_and_label(
                ref, 640, 480
            )


@pytest.mark.parametrize(
    "text",
    [
        # [P23] 선행 마커 생략 · 마커 0개 · 빈 출력 · 앞이 공백뿐 · 마커만 · 이미지보다 많은 쪽
        "Page one\n<|det|>image [100, 100, 500, 400]<|/det|>\n<PAGE>" + PAGE_B,
        "Only page\n<|det|>image [100, 100, 500, 400]<|/det|>\n",
        "",
        " \n<PAGE>" + PAGE_B,
        "<PAGE>",
        "lead\n<PAGE><PAGE>" + PAGE_B + "<PAGE>third page text",
    ],
    ids=["no-leading-marker", "no-marker", "empty", "blank-lead", "marker-only", "extra-pages"],
)
def test_page_split_edge_cases_match_torch_infer_multi(torch_vendor, tmp_path, text):
    paths = _write_images(tmp_path, [(340, 440), (500, 260)])
    tok = PieceTokenizer([text])
    generated = tok.ids_for(True)
    t_md, t_tokens = _run_torch_multi(torch_vendor, tok, paths, tmp_path / "torch", generated)
    m_md, m_tokens = _run_mlx_multi(tok, paths, tmp_path / "mlx", generated)
    assert m_md == t_md
    assert m_tokens == t_tokens
    assert _snapshot(tmp_path / "mlx") == _snapshot(tmp_path / "torch")


def test_page_split_matches_torch_on_random_outputs(torch_vendor):
    import random

    from app.vendor.unlimited_ocr import modeling_unlimitedocr as tv

    rng = random.Random(23)
    pieces = ["<PAGE>", "<PAGE", "PAGE>", "\n", " ", "\t", "본문", "<|det|>image [1, 2, 3, 4]<|/det|>"]
    for _ in range(1000):
        outputs = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 14)))
        assert pp._split_multi_pages(outputs) == tv._split_multi_pages(outputs), outputs


# 단일 상자 규칙 밖의 패리티 전용 모서리 — 공백 라벨 image ref는 좌표가 정상이라 크롭되고
# (번호 판정은 마크다운 치환과 같은 _is_image_ref), 여러 상자 ref는 업스트림대로 상자마다
# 번호 1개를 쓴다(마크다운 자리는 1개). 규칙이 바뀌면 두 벤더가 함께 바뀌어야 한다.
QUIRK_IMAGE_DETS = {
    "ref-label-spaces": "<|ref|> image <|/ref|><|det|>[[100, 100, 500, 400]]<|/det|>",
    "ref-two-boxes-bad-first": "<|ref|>image<|/ref|><|det|>[[0, 0, 999], [100, 100, 500, 400]]<|/det|>",
    "ref-two-boxes": "<|ref|>image<|/ref|><|det|>[[0, 0, 400, 400], [500, 500, 999, 999]]<|/det|>",
}


@pytest.mark.parametrize(
    "bad",
    list(UNUSABLE_IMAGE_DETS.values()) + list(QUIRK_IMAGE_DETS.values()),
    ids=list(UNUSABLE_IMAGE_DETS) + list(QUIRK_IMAGE_DETS),
)
def test_unusable_image_boxes_match_torch_infer_multi(torch_vendor, tmp_path, bad):
    """[P22] 쓸 수 없는 image 좌표 뒤의 그림 번호·크롭·boxes.json이 torch 흐름과 바이트 동일."""
    paths = _write_images(tmp_path, [(300, 400)])
    tok = PieceTokenizer(["<PAGE>", f"{bad}\nfig A\n{GOOD_IMAGE_DET}\nfig B\n"])
    generated = tok.ids_for(True)
    t_md, _ = _run_torch_multi(torch_vendor, tok, paths, tmp_path / "torch", generated)
    m_md, _ = _run_mlx_multi(tok, paths, tmp_path / "mlx", generated)
    assert m_md == t_md
    assert _snapshot(tmp_path / "mlx") == _snapshot(tmp_path / "torch")


def _random_points(rng) -> object:
    atoms = [
        0, 1, 5, 450, 998, 999, 1000, -50, 1500, 2.5, -0.0, 1e-3, 10**400, -(10**400),
        float("inf"), float("-inf"), float("nan"), True, False, None, "0", "x", [0, 0, 1, 1], (),
    ]
    kind = rng.randrange(4)
    if kind == 0:  # 정상 범위 4좌표
        return [rng.randrange(-100, 1200) for _ in range(4)]
    if kind == 1:  # 이상 원소가 섞인 0~6좌표
        return [rng.choice(atoms) for _ in range(rng.randrange(7))]
    if kind == 2:  # 목록이 아닌 값
        return rng.choice([7, "abcd", None, {1: 2, 3: 4, 5: 6, 7: 8}, b"\x00\x01\x02\x03", (1, 2, 3, 4)])
    return [[rng.randrange(1000) for _ in range(4)]]  # 중첩


def test_clamp_box_matches_torch_on_random_points(torch_vendor):
    import random

    from app.vendor.unlimited_ocr import modeling_unlimitedocr as tv

    rng = random.Random(22)
    for _ in range(3000):
        points = _random_points(rng)
        size = (rng.randrange(1, 2000), rng.randrange(1, 2000))
        assert pp._clamp_box(points, *size) == tv._clamp_box(points, *size), (points, size)


def _random_page(rng) -> str:
    """라벨·문법·좌표가 뒤섞인 한 페이지 원문 — 정상·경계 밖·퇴화·해석 불가가 섞인다."""
    odd = ["-50", "1200", "2.5", "1e400", "9" * 400, "True", "'x'", "a"]

    def flat(k):
        if k == 4 and rng.random() < 0.6:  # 정상 상자 (경계 밖 끝점도 가끔)
            x1, y1 = rng.randrange(0, 800), rng.randrange(0, 800)
            nums = [x1, y1, x1 + rng.randrange(1, 400), y1 + rng.randrange(1, 400)]
            atoms = [str(v) for v in nums]
        else:  # 이상 원소가 섞인 좌표
            atoms = [rng.choice(odd) if rng.random() < 0.3 else str(rng.randrange(0, 1000)) for _ in range(k)]
        sep = ", " if rng.random() < 0.9 else " "  # 가끔 쉼표 누락
        return "[" + sep.join(atoms) + "]"

    blocks = []
    for _ in range(rng.randrange(1, 7)):
        label = rng.choice(["image", "image", "image", "text", "title", "table"])
        k = rng.choice([4, 4, 4, 3, 5, 0])
        if rng.random() < 0.5:
            blocks.append(f"<|det|>{label} {flat(k)}<|/det|>caption {rng.randrange(100)}")
        else:
            boxes = ", ".join(flat(rng.choice([4, 4, 3])) for _ in range(rng.randrange(0, 3)))
            payload = rng.choice([f"[{boxes}]", flat(k), "[[a, b, c, d]]", "'abcd'", ""])
            blocks.append(f"<|ref|>{label}<|/ref|><|det|>{payload}<|/det|>body {rng.randrange(100)}")
    return "\n".join(blocks)


def test_draw_bounding_boxes_matches_torch_on_random_pages(torch_vendor, tmp_path):
    """크롭 파일 이름·바이트·boxes.json이 torch draw_bounding_boxes와 같다 — 번호 소비 규칙
    전체(P22)를 무작위 페이지로 대조한다."""
    import random

    from app.vendor.unlimited_ocr import modeling_unlimitedocr as tv

    rng = random.Random(2210)
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (90, 120, 3), dtype=np.uint8))
    for i in range(150):
        text = _random_page(rng)
        refs = pp.re_match(text)[0]
        assert refs == tv.re_match(text)[0]
        snaps = []
        for name, mod in (("torch", tv), ("mlx", pp)):
            out = tmp_path / f"{i}_{name}"
            (out / "images").mkdir(parents=True)
            mod.draw_bounding_boxes(image.copy(), refs, str(out), image_prefix="page_0_")
            snaps.append(_snapshot(out))
        assert snaps[0] == snaps[1], text
