"""벤더 보안 패치 P8·P9 회귀 테스트 — 모델 출력은 절대 eval()되지 않는다.

모델 출력은 PDF 내용으로 유도할 수 있다(문서에 적힌 ``<|det|>text [...]<|/det|>``를
OCR이 그대로 전사). 업스트림은 그 출력을 ``eval()``에 넘겼다:
- P9: det 좌표 파싱 ``eval`` → ``ast.literal_eval`` (리터럴만 허용)
- P8: geo 플로팅 분기(eval 6곳)를 ``if False and …``로 비활성화
재동기화·패치 재적용 실수로 되돌아가면 무인증·0.0.0.0 서비스에서 원격 코드 실행이
되는데, ruff는 app/vendor를 제외하고 S307도 없어 정적 검사로 못 잡는다
(audit tests-baseline-6). 소스 AST와 실제 동작 양쪽으로 고정한다.

같은 출력이 det 자리에 상자 목록이 아닌 값(문자열·빈 목록·None)을 내도 P22의 그림 번호
계약 — image 매치마다 크롭 번호 1개, 마크다운 ``images/{prefix}{k}.jpg``와 같은 번호 — 은
지켜져야 한다(audit torch-2). 끝 절이 그 경로를 고정한다.
"""

import ast
import json
from pathlib import Path

import pytest

VENDOR_DIR = Path(__file__).resolve().parents[1] / "app" / "vendor" / "unlimited_ocr"
DANGEROUS = {"eval", "exec", "compile", "__import__"}


def _parents(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _inside_disabled_branch(node: ast.AST, parents: dict) -> bool:
    """node가 ``if False and …:``(P8) 블록의 본문 안에 있는가."""
    cur = node
    while cur in parents:
        parent = parents[cur]
        if isinstance(parent, ast.If) and cur in parent.body:
            test = parent.test
            if (
                isinstance(test, ast.BoolOp)
                and isinstance(test.op, ast.And)
                and isinstance(test.values[0], ast.Constant)
                and test.values[0].value is False
            ):
                return True
        cur = parent
    return False


def _dangerous_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DANGEROUS:
            yield node, _inside_disabled_branch(node, parents)


def test_vendor_sources_never_eval_reachable_code():
    """벤더 전 파일: eval/exec/compile/__import__ 호출은 P8의 ``if False and`` 블록
    안에만 있을 수 있다(도달 불가 죽은 코드)."""
    files = sorted(VENDOR_DIR.glob("*.py"))
    assert files, "벤더 디렉터리를 찾지 못함 — 경로 계약 재검토"
    reachable = []
    disabled = 0
    for path in files:
        for node, is_disabled in _dangerous_calls(path):
            if is_disabled:
                disabled += 1
            else:
                reachable.append(f"{path.name}:{node.lineno} {node.func.id}()")
    assert reachable == [], f"도달 가능한 eval/exec 호출: {reachable}"
    assert disabled >= 1, "P8 비활성 블록이 사라짐 — 삭제했다면 이 단언을 함께 갱신할 것"


def test_det_coordinate_parser_uses_literal_eval_only():
    """P9: extract_coordinates_and_label은 ast.literal_eval만 쓴다."""
    src = (VENDOR_DIR / "modeling_unlimitedocr.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "extract_coordinates_and_label"
    )
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    names = {
        (n.func.value.id + "." + n.func.attr)
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
        else getattr(n.func, "id", None)
        for n in calls
    }
    assert "ast.literal_eval" in names
    assert not names & DANGEROUS


@pytest.fixture
def vendor_mod():
    pytest.importorskip("torch")
    from app.vendor.unlimited_ocr import modeling_unlimitedocr

    return modeling_unlimitedocr


def _payload(marker: Path) -> str:
    return f"__import__('pathlib').Path({str(marker)!r}).touch()"


def test_malicious_det_payload_is_not_executed(vendor_mod, tmp_path):
    marker = tmp_path / "pwned"
    result = vendor_mod.extract_coordinates_and_label(("", "text", _payload(marker)), 100, 100)
    assert result is None
    assert not marker.exists()


def test_malicious_model_output_through_postprocessing(vendor_mod, tmp_path):
    """전사된 악성 태그가 re_match → draw_bounding_boxes(후처리 전 경로)를 지나도 실행되지
    않고 예외도 없다. det 패턴은 ']'가 없는 페이로드를 그대로 캡처하므로 실제로 도달한다."""
    from PIL import Image

    marker = tmp_path / "pwned"
    text = (
        f"<|det|>text [{_payload(marker)}]<|/det|>본문\n"
        f"<|ref|>image<|/ref|><|det|>[{_payload(marker)}]<|/det|>\n"
        "<|ref|>text<|/ref|><|det|>[[10, 10, 500, 500]]<|/det|>정상 블록\n"
    )
    refs, images, _ = vendor_mod.re_match(text)
    assert any(_payload(marker) in r[2] for r in refs)  # 페이로드가 파서까지 도달하는 경로
    (tmp_path / "images").mkdir()
    out = vendor_mod.process_image_with_refs(
        Image.new("RGB", (100, 100), "white"), refs, str(tmp_path)
    )
    assert out.size == (100, 100)
    assert not marker.exists()
    assert len(images) == 1


# ── P22: 상자 목록이 아닌 det 페이로드도 image 매치당 그림 번호 1개 (audit torch-2) ──


@pytest.mark.parametrize("payload", ['"abcd"', '""', "[]", "()", "None", "0", "5", "{0: 1}", "b'abcd'"])
def test_det_payload_that_is_not_a_box_list_is_unparseable(vendor_mod, payload):
    """문자열은 글자마다, 빈 목록·None은 0개의 번호를 소비했다 — 해석 불가(None)로 돌려
    호출자의 '해석 불가 image ref는 번호 1개' 분기를 태운다."""
    ref = (f"<|ref|>image<|/ref|><|det|>{payload}<|/det|>", "image", payload)
    assert vendor_mod.extract_coordinates_and_label(ref, 200, 100) is None


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("[1, 2, 3, 4]", [[1, 2, 3, 4]]),  # 평평한 상자 하나 → 목록으로 감싼다(업스트림)
        ("(1, 2, 3, 4)", [(1, 2, 3, 4)]),
        ("[[1, 2, 3, 4], [5, 6, 7, 8]]", [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ("[[0, 0, 999]]", [[0, 0, 999]]),  # 상자 단위 검증은 _clamp_box 몫(번호 1개 소비)
    ],
)
def test_box_lists_still_parse_as_before(vendor_mod, payload, expected):
    ref = (f"<|det|>image {payload}<|/det|>", "image", payload)
    assert vendor_mod.extract_coordinates_and_label(ref, 200, 100) == ("image", expected)


# 200×100 페이지에서 픽셀 (20,10)-(80,60)인 정상 그림 — ref 문법과 실가중치가 쓰는 인라인 문법
_REF_FIGURE = "<|ref|>image<|/ref|><|det|>[[100, 100, 400, 600]]<|/det|>"
_INLINE_FIGURE = "<|det|>image [100, 100, 400, 600]<|/det|>"


@pytest.mark.parametrize(
    "first,figure,first_cropped",
    [
        ('<|ref|>image<|/ref|><|det|>"abcd"<|/det|>', _REF_FIGURE, False),  # 구: 뒤 그림이 page_0_4
        ('<|ref|>image<|/ref|><|det|>""<|/det|>', _REF_FIGURE, False),
        ("<|ref|>image<|/ref|><|det|>[]<|/det|>", _REF_FIGURE, False),  # 구: 뒤 그림이 page_0_0
        ("<|ref|>image<|/ref|><|det|>()<|/det|>", _REF_FIGURE, False),
        ("<|ref|>image<|/ref|><|det|>None<|/det|>", _REF_FIGURE, False),
        ("<|ref|>image<|/ref|><|det|>0<|/det|>", _REF_FIGURE, False),
        ("<|det|>image [ ]<|/det|>", _INLINE_FIGURE, False),  # 인라인 문법의 빈 목록
        # re_match는 라벨을 strip해(또는 전체 매치의 <|ref|>image<|/ref|>로) 그림으로 치환한다 —
        # 크롭·번호 판정도 같은 기준이어야 그 그림 파일이 생기고 뒤 번호가 밀리지 않는다
        ("<|ref|> image <|/ref|><|det|>[[0, 0, 500, 500]]<|/det|>", _REF_FIGURE, True),
        ("<|ref|>note<|/ref|>본문<|ref|>image<|/ref|><|det|>[[0, 0, 500, 500]]<|/det|>", _REF_FIGURE, True),
        # 기존 P22 규칙 그대로
        ("<|ref|>image<|/ref|><|det|>[[0, 0, abc, 1]]<|/det|>", _REF_FIGURE, False),
        ("<|ref|>image<|/ref|><|det|>[[500, 500, 100, 100]]<|/det|>", _REF_FIGURE, False),
        ("<|ref|>image<|/ref|><|det|>[[0, 0, 500, 500]]<|/det|>", _REF_FIGURE, True),
    ],
    ids=[
        "string", "empty-string", "empty-list", "empty-tuple", "none", "zero", "inline-empty-list",
        "padded-label", "ref-without-det", "unparseable", "inverted", "valid",
    ],
)
def test_each_image_match_consumes_exactly_one_figure_number(
    vendor_mod, tmp_path, first, figure, first_cropped
):
    """마크다운은 image 매치마다 ``images/{prefix}{k}.jpg`` 하나(k = 매치 순번)를 쓴다 —
    크롭 번호가 어긋나면 정상 그림은 없는 파일이나 남의 크롭을 가리키고 실제 크롭은 고아가 된다."""
    from PIL import Image

    refs, images, _ = vendor_mod.re_match(f"{first}\n{figure}\n")
    assert len(images) == 2
    md_files = [f"page_0_{k}.jpg" for k in range(len(images))]  # infer/infer_multi의 치환 규칙
    (tmp_path / "images").mkdir()
    vendor_mod.draw_bounding_boxes(Image.new("RGB", (200, 100), "white"), refs, str(tmp_path), "page_0_")

    boxes = json.loads((tmp_path / "boxes.json").read_text(encoding="utf-8"))
    figure_file = md_files[images.index(figure)]
    assert boxes[figure_file] == {
        "x1": 20, "y1": 10, "x2": 80, "y2": 60, "image_width": 200, "image_height": 100,
    }
    on_disk = sorted(p.name for p in (tmp_path / "images").iterdir())
    assert on_disk == sorted(boxes) == (md_files if first_cropped else [figure_file])
