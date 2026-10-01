"""벤더 보안 패치 P8·P9 회귀 테스트 — 모델 출력은 절대 eval()되지 않는다.

모델 출력은 PDF 내용으로 유도할 수 있다(문서에 적힌 ``<|det|>text [...]<|/det|>``를
OCR이 그대로 전사). 업스트림은 그 출력을 ``eval()``에 넘겼다:
- P9: det 좌표 파싱 ``eval`` → ``ast.literal_eval`` (리터럴만 허용)
- P8: geo 플로팅 분기(eval 6곳)를 ``if False and …``로 비활성화
재동기화·패치 재적용 실수로 되돌아가면 무인증·0.0.0.0 서비스에서 원격 코드 실행이
되는데, ruff는 app/vendor를 제외하고 S307도 없어 정적 검사로 못 잡는다
(audit tests-baseline-6). 소스 AST와 실제 동작 양쪽으로 고정한다.
"""

import ast
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
