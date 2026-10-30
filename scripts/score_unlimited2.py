#!/usr/bin/env python3
"""Diagnostic quality metrics for the generated evaluation set (no hidden pass rule)."""

import argparse
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

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        if tag == "tr" and self.rows is not None:
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.row.append("")

    def handle_data(self, data):
        if self.row:
            self.row[-1] += data

    def handle_endtag(self, tag):
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None


def normalized(text):
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    return "".join(c.lower() for c in text if c.isalnum())


def box_iou(a, b):
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
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


def score(folder, truth):
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
    return {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    truth = json.loads(args.truth.read_text())
    result = {
        "metric_limits": "CER ignores whitespace, punctuation and common LaTeX wrappers. It is a diagnostic, not a formula-semantic or full layout score. Synthetic reference text includes text within drawn figures. Full quality non-regression needs visual review and additional annotations.",
        "baseline": score(args.baseline, truth),
        "candidate": score(args.candidate, truth),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
