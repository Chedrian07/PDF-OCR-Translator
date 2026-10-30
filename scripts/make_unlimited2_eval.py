#!/usr/bin/env python3
"""Generate deterministic, redistribution-safe OCR evaluation PDFs and references."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz


def stable(path):
    with fitz.open(path) as doc:
        doc.xref_set_key(-1, "ID", "null")
        data = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("benchmark_docs/unlimited2"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    maker = Path(__file__).with_name("make_sample_pdf.py")
    for name, flags in [
        ("en-mixed", []),
        ("ko-report", ["--korean"]),
        ("scan", ["--scan"]),
    ]:
        subprocess.run(
            [sys.executable, str(maker), str(out / f"{name}.pdf"), *flags], check=True
        )
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 50), "Two-column reading order", fontsize=20)
    for x, label in [(40, "LEFT"), (315, "RIGHT")]:
        for i in range(8):
            page.insert_textbox(
                fitz.Rect(x, 90 + i * 65, x + 230, 145 + i * 65),
                f"{label} section {i + 1}\nThe reference value is {100 + i}.\nThis paragraph belongs to one column.",
                fontsize=11,
            )
    doc.save(out / "columns.pdf")
    doc.close()
    doc = fitz.open()
    with fitz.open(out / "en-mixed.pdf") as source:
        for i in range(16):
            doc.insert_pdf(source)
            for p in doc.pages(i * 2, i * 2 + 2):
                p.insert_text(
                    (420, 800), f"Physical page {p.number + 1:02d}", fontsize=10
                )
    doc.save(out / "long-32.pdf")
    doc.close()
    manifest = {"dpi": 200, "pages_per_chunk": 8, "documents": []}
    for path in sorted(out.glob("*.pdf")):
        stable(path)
        with fitz.open(path) as pdf:
            texts = [p.get_text("text", sort=False) for p in pdf]
            figures = [len(p.get_image_info()) for p in pdf]
            if path.stem in ("ko-report", "scan"):
                figures = [1]
            if path.stem == "scan":
                with fitz.open(out / "ko-report.pdf") as reference:
                    texts = [reference[0].get_text("text", sort=False)]
            figure_boxes = [
                [
                    [
                        box["bbox"][0] / p.rect.width,
                        box["bbox"][1] / p.rect.height,
                        box["bbox"][2] / p.rect.width,
                        box["bbox"][3] / p.rect.height,
                    ]
                    for box in p.get_image_info()
                ]
                for p in pdf
            ]
            if path.stem == "ko-report":
                figure_boxes = [[[72 / 595, 450 / 842, 400 / 595, 640 / 842]]]
            if path.stem == "scan":
                figure_boxes = None  # Rotated raster ROI needs a separate annotation.
            truth = {
                "reference_text": texts,
                "figure_boxes": figure_boxes,
                "expected_figures": figures,
                "reference_origin": "Text inserted by the synthetic generator; scan inherits the clean Korean source.",
                "reading_order_landmarks": (
                    [
                        f"{side} section {i + 1}"
                        for side in ("LEFT", "RIGHT")
                        for i in range(8)
                    ]
                    if path.stem == "columns"
                    else []
                ),
                "table_shapes": [[3, 4]] * (len(pdf) // 2)
                if path.stem in ("en-mixed", "long-32")
                else [[4, 3]]
                if path.stem in ("ko-report", "scan")
                else [],
            }
            (out / f"{path.stem}.truth.json").write_text(
                json.dumps(truth, ensure_ascii=False, indent=2) + "\n"
            )
            manifest["documents"].append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "pages": len(pdf),
                }
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
