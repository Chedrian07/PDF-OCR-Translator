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
# 경계 밖·퇴화 좌표 — [local patch P22] 대상
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
