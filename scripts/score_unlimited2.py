#!/usr/bin/env python3
"""Diagnostic quality metrics for the generated evaluation set (no hidden pass rule)."""

import argparse
import hashlib
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from benchmark_ocr_engines import edit_distance


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.rows = None
        self.row = None
        self.spans = []
        self.table_spans = None
        self.row_spans = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
            self.table_spans = []
        if tag == "tr" and self.rows is not None:
            self.row = []
            self.row_spans = []
        if tag in ("td", "th") and self.row is not None:
            self.row.append("")
            attributes = dict(attrs)
            try:
                span = [int(attributes.get(name, "1")) for name in ("rowspan", "colspan")]
            except (TypeError, ValueError):
                span = [0, 0]
            self.row_spans.append(span)

    def handle_data(self, data):
        if self.row:
            self.row[-1] += data

    def handle_endtag(self, tag):
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.table_spans.append(self.row_spans)
            self.row = None
        if tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.spans.append(self.table_spans)
            self.rows = None


def normalized(text):
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    return "".join(c.lower() for c in text if c.isalnum())


_MATH_SPANS = re.compile(
    r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
    re.DOTALL,
)
_FONT_WRAPPER = re.compile(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}")


def _math_presentation(text):
    """Normalize a small set of presentation conventions; retain every operator."""
    text = html.unescape(text).replace("−", "-").replace("×", "*")
    text = re.sub(r"\\(?:times|cdot)\b", "*", text)
    for _ in range(16):
        unwrapped = _FONT_WRAPPER.sub(lambda m: re.sub(r"\s+", "", m[1]), text)
        if unwrapped == text:
            break
        text = unwrapped
    text = re.sub(r"\\(?:left|right)\b", "", text)
    # Braced numeric scripts and single symbolic indices have the same token
    # transcription. Other groups, including ^{*2}, remain visible mismatches.
    text = re.sub(r"([_^])\s*\{\s*([+-]?\s*\d+|[A-Za-z])\s*\}", r"\1\2", text)
    for _ in range(16):
        root = re.search(r"\\sqrt\s*\{", text)
        if root is None:
            break
        depth, end = 1, root.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            break
        text = text[:root.start()] + "sqrt(" + text[root.end():end - 1] + ")" + text[end:]
    return text


def math_tokens(text):
    """Exact lexical transcription, deliberately not algebraic equivalence."""
    text = re.sub(r"\s+", "", _math_presentation(text))
    if not text or re.search(r"[^A-Za-z0-9.+*/=^_(),<>≤≥≈-]", text):
        return None
    depth = 0
    for char in text:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return None
    return text if depth == 0 else None


def equation_transcription_matches(page, expected):
    """Compare complete marked math, or a bounded bare expression in prose.

    A bare match cannot be a prefix of a larger expression or an identifier.
    Explicitly delimited math is always compared as a complete span.
    """
    target = math_tokens(expected)
    if target is None:
        raise ValueError(f"Unsupported annotation equation: {expected!r}")
    for span in _MATH_SPANS.finditer(page):
        if math_tokens(next(value for value in span.groups() if value is not None)) == target:
            return True
    bare = _MATH_SPANS.sub(" ", page)
    bare = re.sub(r"<[^>]*>", " ", bare)
    bare = _math_presentation(bare)
    pattern = r"\s*".join(re.escape(char) for char in target)
    for match in re.finditer(pattern, bare):
        before, after = bare[:match.start()], bare[match.end():]
        if before and (before[-1].isalnum() or before[-1] in "_\\/@.:?#%&=^+-*"):
            continue
        if before.rstrip(" \t")[-1:] in ("=", "+", "-", "*", "/", "^", "_"):
            continue
        if after and (after[0].isalnum() or after[0] in "_\\/@?#%&=^+*({<>"):
            continue
        if re.match(r"\.\d", after):
            continue  # the expected exponent may only prefix a decimal number
        tail = after.split("\n", 1)[0].lstrip(" \t")
        if tail.startswith(("=", "+", "-", "*", "/", "^", "_", "%", "&", "|", "!", "<", ">", "≤", "≥", "≈", "\\")):
            continue
        if re.match(r"[A-Za-z]{1,2}\b|\d", tail):
            continue  # possible implicit product, rather than following prose
        return True
    return False


def _cell_text(text):
    text = unicodedata.normalize("NFKC", _math_presentation(text)).casefold()
    text = re.sub(r"\\[()[\]]|\$", "", text)
    return re.sub(r"\s+", "", text)


def annotated_quality(text, annotations):
    """Independent page/cell annotations supplement the frozen legacy metrics."""
    if annotations.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation annotation schema")
    parts = text.split("<PAGE>")
    pages = parts[1:] if not parts[0].strip() else parts
    if not text.strip():
        pages = []
    expected_pages = annotations["pages"]
    totals = {
        "expected_tables": 0, "predicted_tables": 0, "exact_tables": 0,
        "correct_cells": 0, "expected_cells": 0, "matched_equations": 0, "expected_equations": 0,
    }
    details = []
    structures_match = len(pages) == len(expected_pages)
    for index in range(max(len(pages), len(expected_pages))):
        expected = expected_pages[index] if index < len(expected_pages) else {"tables": [], "equations": []}
        if expected.get("page", index + 1) != index + 1:
            raise ValueError("Evaluation annotations must list consecutive 1-based physical pages")
        page = pages[index] if index < len(pages) else ""
        parser = Tables()
        parser.feed(page)
        tables = expected["tables"]
        totals["expected_tables"] += len(tables)
        totals["predicted_tables"] += len(parser.tables)
        structures_match &= len(tables) == len(parser.tables)
        table_details = []
        for table_index, table in enumerate(tables):
            cells = table["cells"]
            predicted = parser.tables[table_index] if table_index < len(parser.tables) else []
            spans = parser.spans[table_index] if table_index < len(parser.spans) else []
            shape = [len(row) for row in cells]
            predicted_shape = [len(row) for row in predicted]
            structure = shape == predicted_shape and all(span == [1, 1] for row in spans for span in row)
            matches = [[
                row < len(predicted) and column < len(predicted[row])
                and _cell_text(cell) == _cell_text(predicted[row][column])
                for column, cell in enumerate(values)
            ] for row, values in enumerate(cells)]
            correct = sum(sum(row) for row in matches)
            count = sum(shape)
            totals["correct_cells"] += correct
            totals["expected_cells"] += count
            totals["exact_tables"] += structure and correct == count
            structures_match &= structure
            table_details.append({"expected_row_lengths": shape, "predicted_row_lengths": predicted_shape,
                                  "structure_match": structure, "cell_matches": matches})
        equations = [{"expected": equation, "token_transcription_match": equation_transcription_matches(page, equation)}
                     for equation in expected["equations"]]
        totals["expected_equations"] += len(equations)
        totals["matched_equations"] += sum(item["token_transcription_match"] for item in equations)
        details.append({"page": index + 1, "tables": table_details, "equations": equations,
                        "unexpected_tables": max(0, len(parser.tables) - len(tables))})
    return {
        "annotation_schema_version": 1,
        "limits": "Cells use NFKC, case folding and whitespace normalization while retaining punctuation. Equations compare exact operator-preserving token transcriptions after supported LaTeX presentation normalization; matches do not establish semantic equivalence. Unsupported notation is a mismatch.",
        "page_count_match": len(pages) == len(expected_pages),
        "table_structure_exact_match": bool(structures_match),
        **totals,
        "cell_accuracy": totals["correct_cells"] / totals["expected_cells"] if totals["expected_cells"] else None,
        "equation_transcription_accuracy": totals["matched_equations"] / totals["expected_equations"] if totals["expected_equations"] else None,
        "pages": details,
    }


def box_iou(a, b):
    def area(r):
        return max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    intersection = area(
        [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    )
    return intersection / max(area(a) + area(b) - intersection, 1e-12)


def figure_iou(folder, truth, chunk_size):
    expected = truth.get("figure_boxes")
    if expected is None:
        return None
    predicted = {}
    run = folder / "run-1"
    if not run.is_dir():
        run = folder
    for path in run.glob("**/boxes.json"):
        offset = (
            int(path.parent.name.split("-")[1]) * chunk_size
            if path.parent.name.startswith("chunk-")
            else 0
        )
        for name, box in json.loads(path.read_text()).items():
            match = re.match(r"page_(\d+)_", name)
            page = offset + (int(match[1]) if match else 0)
            predicted.setdefault(page, []).append(
                [
                    box["x1"] / box["image_width"],
                    box["y1"] / box["image_height"],
                    box["x2"] / box["image_width"],
                    box["y2"] / box["image_height"],
                ]
            )
    scores = []
    for page, boxes in enumerate(expected):
        available = list(predicted.get(page, []))
        for box in boxes:
            if not available:
                scores.append(0)
                continue
            best = max(range(len(available)), key=lambda i: box_iou(box, available[i]))
            scores.append(box_iou(box, available.pop(best)))
    return sum(scores) / len(scores) if scores else None


def score(folder, truth, annotations=None):
    path = folder / "run-1/document.md"
    if not path.exists():
        path = folder / "run-1/result.md"
    if not path.exists():
        path = folder / "output.md"
    text = path.read_text()
    ref = "\n".join(truth["reference_text"])
    a, b = normalized(text), normalized(ref)
    # Long fixture repeats the same known pages; report the complete document,
    # with a bounded row DP. No clipping or approximate matching.
    cer = edit_distance(a, b) / max(1, len(b))
    parser = Tables()
    parser.feed(text)
    shapes = [[len(t), len(t[0]) if t else 0] for t in parser.tables]
    landmarks = [normalized(x) for x in truth["reading_order_landmarks"]]
    positions = [a.find(x) for x in landmarks]
    found = [p for p in positions if p >= 0]
    pairs = len(found) * (len(found) - 1) // 2
    ordered = sum(x < y for i, x in enumerate(found) for y in found[i + 1 :])
    measurements = json.loads((folder / "measurements.json").read_text())
    result = {
        "normalized_alphanumeric_cer": round(cer, 6),
        "table_shapes": shapes,
        "table_shapes_match": shapes == truth["table_shapes"],
        "formula_markup_count": len(
            re.findall(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)", text, re.DOTALL)
        ),
        "reading_order_landmarks_found": len(found),
        "reading_order_landmarks_total": len(landmarks),
        "reading_order_pair_accuracy": ordered / pairs if pairs else None,
        "page_markers": measurements["runs"][1]["page_markers"],
        "figures": measurements["runs"][1]["figures"],
        "expected_figures": sum(truth["expected_figures"]),
        "median_ocr_s": measurements["median_ocr_s"],
        "figure_mean_iou": figure_iou(
            folder, truth, measurements.get("pages_per_chunk", 8)
        ),
    }
    if annotations is not None:
        result["annotated_quality"] = annotated_quality(text, annotations)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, help="Optional versioned table/equation annotation sidecar")
    args = ap.parse_args()
    truth = json.loads(args.truth.read_text())
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    if annotations is not None:
        pdf = args.annotations.parent / annotations["document"]
        if args.truth.name != pdf.with_suffix(".truth.json").name or hashlib.sha256(pdf.read_bytes()).hexdigest() != annotations["input_sha256"]:
            raise ValueError("Evaluation annotation does not match the frozen input PDF")
    result = {
        "metric_limits": "CER ignores whitespace, punctuation and common LaTeX wrappers. It is a diagnostic, not a formula-semantic or full layout score. Synthetic reference text includes text within drawn figures. Full quality non-regression needs visual review and additional annotations.",
        "baseline": score(args.baseline, truth, annotations),
        "candidate": score(args.candidate, truth, annotations),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
