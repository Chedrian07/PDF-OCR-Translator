"""[vendor patch P23] infer_multi 페이지 분할 — 첫 <PAGE> 앞의 출력을 버리지 않는다.

업스트림은 ``split('<PAGE>')[1:]``로 첫 마커 앞을 항상 버려, 모델이 선행 마커를 생략하면
1쪽 내용이 사라지고 뒤 페이지 크롭이 한 칸 앞 이미지로 잘렸다(audit decode-correctness-7).
앱의 merge.split_pages와 같은 규칙이어야 벤더 산출물(raw_pages·크롭)과 병합이 어긋나지
않는다.
"""

import inspect
import random

import pytest

pytest.importorskip("torch")

from app.pipeline.merge import split_pages  # noqa: E402
from app.vendor.unlimited_ocr.modeling_unlimitedocr import (  # noqa: E402
    UnlimitedOCRForCausalLM,
    _split_multi_pages,
)


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
def test_split_keeps_content_before_first_marker(outputs, expected):
    assert [p.strip() for p in _split_multi_pages(outputs)] == expected


def test_split_matches_app_merge_rule():
    rng = random.Random(4)
    pieces = ["<PAGE>", "\n", " ", "text", "표", "<|ref|>image<|/ref|>"]
    for _ in range(500):
        outputs = "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 12)))
        assert [p.strip() for p in _split_multi_pages(outputs)] == split_pages(outputs), outputs


def test_infer_multi_uses_the_p23_split():
    src = inspect.getsource(UnlimitedOCRForCausalLM.infer_multi)
    assert "_split_multi_pages(outputs)" in src
    assert "split('<PAGE>')[1:]" not in src
