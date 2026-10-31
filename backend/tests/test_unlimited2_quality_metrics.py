"""Independent evaluation annotations must expose changed cells and operators."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def metric(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_score_unlimited2", ROOT / "scripts/score_unlimited2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", [
    "The pricing follows E = mc^2 style notation.",
    r"\(E = mc^{2}\)",
    r"$$ \mathrm{E} = \mathrm{m c}^{2} $$",
    "(E = mc^2), then the explanation.",
])
def test_math_formatting_differences_preserve_token_transcription(metric, source):
    assert metric.equation_transcription_matches(source, "E = mc^2")


@pytest.mark.parametrize("source", [
    r"\[\text {sigma} = \operatorname{sqrt} ((1 / \mathrm{N}) * \text {sum} (x _ {i} - m u) ^ {2})\]",
    r"$$\mathrm{sigma}=\sqrt{(1/N)\cdot\operatorname{sum}(x_{i}-\mathrm{mu})^{2}}$$",
])
def test_standard_latex_wrappers_and_numeric_script_braces_match(metric, source):
    assert metric.equation_transcription_matches(source, "sigma=sqrt((1/N)*sum(x_i-mu)^2)")


@pytest.mark.parametrize("source", [
    r"\(E = mc^{*2}\)", r"\(E = mc^{*}\)2", r"\(E = mc^2 + x\)",
    r"\(e = mc^2\)", "E = mc^3", "E = mc^2 + x", "E = mc^2 / 2",
    "E = mc^2x", "E = mc^2 x", "E = mc^2 = d", "E = mc^2 % 3",
    "E = mc^2 != d",
    "E=mc^2.1", r"E=mc^2 \div 2",
    "valueE = mc^2", "https://example.org/?E=mc^2", r"\(E = mc^2\text{ style}\)",
])
def test_changed_operators_or_larger_expressions_do_not_receive_math_credit(metric, source):
    assert not metric.equation_transcription_matches(source, "E = mc^2")


def test_malformed_exponent_star_remains_a_korean_equation_mismatch(metric):
    expected = "sigma=sqrt((1/N)*sum(x_i-mu)^2)"
    actual = r"\[\text{sigma}=\operatorname{sqrt}((1/N)^{*}\text{sum}(x_{i}-mu)^{2})\]"
    assert not metric.equation_transcription_matches(actual, expected)
    assert metric.math_tokens("E=mc^{*2}") is None
    assert metric.math_tokens("E=mc^{2}") == "E=mc^2"


def _annotations(pages=None):
    return {"schema_version": 1, "pages": pages or [{
        "page": 1, "tables": [{"cells": [["Item", "Value"], ["base_size", "1,234"]]}],
        "equations": ["E=mc^2"],
    }]}


def _document(last_cell="1, 2 3 4", row_attributes="", extra_cell=""):
    return (
        "<PAGE><table><tr><th>ITEM</th><th>Value</th></tr>"
        f"<tr><td {row_attributes}>base_size</td><td>{last_cell}</td>{extra_cell}</tr></table>"
        r"\(E=mc^{2}\)"
    )


def test_cells_normalize_formatting_while_preserving_numeric_punctuation(metric):
    result = metric.annotated_quality(_document(), _annotations())
    assert result["table_structure_exact_match"]
    assert result["correct_cells"] == result["expected_cells"] == 4
    assert result["matched_equations"] == 1
    wrong = metric.annotated_quality(_document(last_cell="1,235"), _annotations())
    assert wrong["table_structure_exact_match"]
    assert wrong["correct_cells"] == 3
    assert wrong["exact_tables"] == 0
    missing_comma = metric.annotated_quality(_document(last_cell="1234"), _annotations())
    assert missing_comma["correct_cells"] == 3


@pytest.mark.parametrize("source", [
    _document(extra_cell="<td>extra</td>"),
    _document(row_attributes='colspan="2"'),
    _document().replace("<td>1, 2 3 4</td>", ""),
])
def test_ragged_rows_spans_and_missing_cells_fail_structure(metric, source):
    assert not metric.annotated_quality(source, _annotations())["table_structure_exact_match"]


def test_page_alignment_prevents_duplicate_page_credit(metric):
    annotations = _annotations([
        {"page": 1, "tables": [], "equations": ["E=mc^2"]},
        {"page": 2, "tables": [], "equations": ["E=mc^2"]},
    ])
    result = metric.annotated_quality(r"<PAGE>\(E=mc^2\)", annotations)
    assert not result["page_count_match"]
    assert result["matched_equations"] == 1
    assert result["expected_equations"] == 2


def test_annotations_are_independent_and_bound_to_frozen_pdfs():
    for stem in ("en-mixed", "ko-report", "scan", "long-32", "columns"):
        folder = ROOT / "benchmark_docs/unlimited2"
        sidecar = json.loads((folder / f"{stem}.eval-v1.json").read_text())
        assert sidecar["schema_version"] == 1
        assert sidecar["input_sha256"] == hashlib.sha256((folder / sidecar["document"]).read_bytes()).hexdigest()
        assert [page["page"] for page in sidecar["pages"]] == list(range(1, len(sidecar["pages"]) + 1))
        if stem == "long-32":
            assert len(sidecar["pages"]) == 32
            assert sum(len(page["equations"]) for page in sidecar["pages"]) == 16


def test_optional_annotations_leave_all_legacy_metrics_unchanged(metric):
    folder = ROOT / "benchmark_docs/unlimited2"
    truth = json.loads((folder / "scan.truth.json").read_text())
    annotations = json.loads((folder / "scan.eval-v1.json").read_text())
    legacy = metric.score(folder / "results/scan/candidate", truth)
    annotated = metric.score(folder / "results/scan/candidate", truth, annotations)
    assert annotated.pop("annotated_quality")["matched_equations"] == 0
    assert annotated == legacy


def test_annotation_cli_writes_supplemental_metrics_without_replacing_legacy_scores(tmp_path):
    folder = ROOT / "benchmark_docs/unlimited2"
    output = tmp_path / "annotated-quality.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/score_unlimited2.py"),
        "--truth", str(folder / "scan.truth.json"),
        "--annotations", str(folder / "scan.eval-v1.json"),
        "--baseline", str(folder / "results/scan/baseline"),
        "--candidate", str(folder / "results/scan/candidate"),
        "--out", str(output),
    ], check=True)
    result = json.loads(output.read_text())
    assert result["baseline"]["normalized_alphanumeric_cer"] == 0.066667
    assert result["candidate"]["normalized_alphanumeric_cer"] == 0.071111
    assert result["candidate"]["annotated_quality"]["correct_cells"] == 12
    assert result["candidate"]["annotated_quality"]["matched_equations"] == 0
