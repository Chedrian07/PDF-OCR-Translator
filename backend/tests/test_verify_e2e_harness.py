"""E2E 하네스(scripts/verify_e2e.py · scripts/mock_llm.py) 판정 로직 회귀 테스트.

하네스는 서버를 띄워야 돌지만 **판정 로직 자체는 순수 함수**다. 하네스가 거짓 안심을
주는 회귀(공허한 비교식, 길이 보존 목, 페이지 단위 유실 미탐)를 유닛 수준에서 잡는다.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

from app.translate.masking import looks_untranslated
from app.translate.prompts import build_repair_prompt, build_unit_prompt

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_harness_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


verify_e2e = _load("verify_e2e")
mock_llm = _load("mock_llm")


# ───────────────────────── 스캐폴딩 누출 탐지 ─────────────────────────

def test_scaffolding_markers_read_from_prompts_module():
    """문구를 하드코딩하면 prompts.py가 바뀔 때 검사만 조용히 죽는다."""
    markers = verify_e2e.scaffolding_markers()
    assert "[번역할 원문" in markers
    # repair 프롬프트 문구가 포함돼야 한다 (echo 결함에서 실제로 새던 경로).
    assert any("수정할 번역문" in m for m in markers)


@pytest.mark.parametrize("prompt", [
    build_unit_prompt("Source text here.", [("model", "모델")], [("token", "토큰")],
                      context_tail="앞 문단", keep_terms=["BERT"], unit_kind="title"),
    build_repair_prompt("Source <m1/> text.", "번역문 일부", ["<m1/>"]),
])
def test_scaffolding_hits_catches_real_prompt_leak(prompt):
    assert verify_e2e.scaffolding_hits(prompt)


def test_scaffolding_hits_clean_translation():
    clean = "본 논문은 새로운 방법을 제안한다. 우리는 이를 세 개의 데이터셋에서 평가하였다.\n\n표 2에 결과를 정리하였다."
    assert verify_e2e.scaffolding_hits(clean) == []


def test_echo_growth_shape_is_detectable():
    """줄 수 단언 — repair 스캐폴딩 25줄이 박히면 부풀어 오르는 것을 잡는다."""
    src = "\n".join(f"line {i}" for i in range(108))
    polluted = src + "\n" + "\n".join(["[수정할 번역문]"] * 25 + ["잔여"] * 37)
    assert (polluted.count("\n") + 1) > (src.count("\n") + 1) * 1.15 + 3
    assert verify_e2e.scaffolding_hits(polluted)


# ───────────────── 번역은 있는데 PDF가 버린 페이지 (신고 결함) ─────────────────

def _page(text: str) -> dict:
    return {"blocks": [{"content": text}]}


KO = "본 논문은 새로운 방법을 제안한다. 우리는 이를 세 개의 데이터셋에서 평가하였다."
EN = "This paper proposes a new method. We evaluate it on three datasets."


def test_dropped_translation_pages_flags_untranslated_pdf_page():
    pages = [_page(KO), _page(KO), _page(KO)]
    pdf = [KO, EN, KO + " " + EN * 2]  # p2 완전 유실, p3 부분 유실
    assert verify_e2e.dropped_translation_pages(pages, pdf) == [
        (2, 1.0, 0.0),
        (3, 1.0, 0.23),
    ]


def test_dropped_translation_pages_ignores_by_design_original_pages():
    """참고문헌·수치 표처럼 번역면 자체가 원문인 페이지는 결함이 아니다."""
    pages = [_page(EN)]
    assert verify_e2e.dropped_translation_pages(pages, [EN]) == []


def test_dropped_translation_pages_accepts_faithful_export():
    pages = [_page(KO), _page(KO)]
    assert verify_e2e.dropped_translation_pages(pages, [KO, KO + " Table 2"]) == []


def test_kept_reason_summary_groups_by_reason():
    report = {
        "replaced": 165,
        "kept": 58,
        "specialist_kept": {"vertical": 1, "table": 2},
        "warning_count": 3,
        "warnings": [
            "p6: 블록 1 교체 생략(공간 부족) — 원문 보존",
            "p15: 블록 4 교체 생략(공간 부족) — 원문 보존",
            "p3: 그림 위 텍스트 — 원문 보존",
        ],
    }
    summary = verify_e2e.kept_reason_summary(report)
    assert summary["warning: 블록 N 교체 생략(공간 부족) — 원문 보존"] == 2
    assert summary["warning: 그림 위 텍스트 — 원문 보존"] == 1
    assert summary["specialist_kept.table"] == 2


def test_kept_reason_summary_absorbs_new_reason_fields():
    """fitting 그룹이 사유 집계 필드를 추가해도 이름을 고정하지 않고 흡수한다."""
    summary = verify_e2e.kept_reason_summary({"kept_reasons": {"multiline_cell": 15}})
    assert summary["kept_reasons.multiline_cell"] == 15


# ───────────────────── 목 LLM 압축률 (게이트 하한 회귀 탐지) ─────────────────────

PARA = (
    "We propose a new evaluation method for long-context language models. "
    "The results in Table 2 show that our approach improves accuracy on three datasets, "
    "and the training cost remains comparable to the baseline model."
)


def test_mock_compresses_like_real_korean():
    out = mock_llm._translate(PARA, ratio=0.4)
    ratio = len(out) / len(PARA)
    # 실제 한국어는 영어의 0.3~0.5배. 길이 보존 목이면 이 단언이 깨진다.
    assert 0.3 < ratio < 0.6, ratio
    assert ratio < len(mock_llm._translate(PARA, ratio=0.0)) / len(PARA)


def test_mock_output_still_passes_current_output_gate():
    """압축 목이 정상 경로를 깨면 안 된다 — 현행 하한(0.3)은 통과해야 한다."""
    out = mock_llm._translate(PARA, ratio=0.4)
    assert not looks_untranslated(PARA, out, {})


def test_mock_output_would_fail_a_raised_length_floor():
    """길이비 하한을 0.55로 올리는 회귀가 하네스에서 '실패로' 드러나는 근거."""
    out = mock_llm._translate(PARA, ratio=0.4)
    assert len(out) < 0.55 * len(PARA)


def test_mock_preserves_placeholders_and_line_structure():
    src = "# Introduction\n\nThe model <m1 v=\"E=mc^2\"/> is fast.\n\n- first item\n- second item"
    out = mock_llm._translate(src, ratio=0.4)
    assert '<m1 v="E=mc^2"/>' in out
    assert out.count("\n") == src.count("\n")
    assert out.startswith("# ")


def test_mock_short_function_word_unit_falls_back_to_length_preserving():
    """전부 기능어인 짧은 유닛까지 압축하면 목이 파이프라인을 깨뜨린다."""
    out = mock_llm._translate("The a of and", ratio=0.4)
    assert len(out) >= mock_llm._MIN_SAFE_RATIO * len("The a of and")
    assert any("가" <= c <= "힣" for c in out)


def test_mock_ratio_env_zero_restores_legacy_behaviour(monkeypatch):
    monkeypatch.setenv("MOCK_TRANSLATE_RATIO", "0")
    assert mock_llm._target_ratio() == 0.0
    assert mock_llm._translate(PARA) == mock_llm._translate(PARA, ratio=0.0)
    monkeypatch.setenv("MOCK_TRANSLATE_RATIO", "nonsense")
    assert mock_llm._target_ratio() == mock_llm._DEFAULT_RATIO


def test_fault_expectations_are_substring_checks_not_equality():
    """6,491자 파일 전체를 6자와 == 비교하던 공허한 검사의 회귀 방지."""
    for fault, (_, needles) in verify_e2e._FAULT_EXPECT.items():
        injected = mock_llm._apply_fault(fault, "Some source text.")
        assert injected is not None, fault
        if not needles:
            continue
        # 손상 출력이 큰 문서 안에 묻혀 있어도 잡혀야 한다.
        haystack = ("정상 문단입니다.\n" * 200) + injected + ("\n다른 문단입니다." * 200)
        assert any(n in haystack.lower() or n in haystack for n in needles), fault


def test_layout_pages_accepts_both_shapes():
    assert verify_e2e.layout_pages([{"blocks": []}]) == [{"blocks": []}]
    assert verify_e2e.layout_pages(json.loads('{"pages": [1]}')) == [1]
