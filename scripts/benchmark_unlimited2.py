#!/usr/bin/env python3
"""Fixed-input warmup + three measured OCR runs; run v1 from the baseline checkout.

Example: PYTHONPATH=backend python scripts/benchmark_unlimited2.py --pdf sample/sample.pdf
Use the same GPU, PDF, DPI, precision and concurrency for both revisions.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pages-per-chunk", type=int, default=8)
    ap.add_argument("--runtime-url", default="http://127.0.0.1:10000")
    args = ap.parse_args()
    import pymupdf as fitz
    from app.config import Settings
    from app.engine.registry import build_engine

    args.out.mkdir(parents=True, exist_ok=True)
    paths = []
    with fitz.open(args.pdf) as doc:
        for i, page in enumerate(doc):
            p = args.out / f"page_{i + 1:04}.png"
            page.get_pixmap(
                matrix=fitz.Matrix(args.dpi / 72, args.dpi / 72), alpha=False
            ).save(p)
            paths.append(p)
    kw = {"device": "cuda", "engine": "unlimited" if args.baseline else "unlimited2"}
    if not args.baseline:
        kw["sidecar_url"] = args.runtime_url
    engine = build_engine(Settings(**kw))
    engine.load()
    caps = engine.capabilities()
    info = {
        "model": caps.model_id,
        "revision": caps.model_revision,
        "runtime": getattr(caps, "runtime", "torch"),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "image_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        "dpi": args.dpi,
        "pages": len(paths),
        "concurrency": 1,
        "mode": "multi/base-1024",
        "pages_per_chunk": args.pages_per_chunk,
        "gpu_index": os.environ.get("BENCH_GPU", "0"),
        "runs": [],
    }

    class Sink:
        first = None

        def on_text(self, text):
            if text and self.first is None:
                self.first = time.perf_counter()

    for run in range(4):
        sink = Sink()
        stop = threading.Event()
        memory = []

        def sample(stop=stop, memory=memory):
            while not stop.is_set():
                try:
                    raw = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            info["gpu_index"],
                            "--query-gpu=name,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=3,
                        text=True,
                    )
                    name, used = raw.strip().rsplit(",", 1)
                    info["gpu"] = name
                    memory.append(int(used))
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                stop.wait(0.1)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t = time.perf_counter()
        row = {"run": run, "warmup": run == 0}
        try:
            outputs = []
            for start in range(0, len(paths), args.pages_per_chunk):
                target = args.out / f"run-{run}"
                if len(paths) > args.pages_per_chunk:
                    target = target / f"chunk-{start // args.pages_per_chunk}"
                outputs.append(
                    engine.run_multi(
                        paths[start : start + args.pages_per_chunk],
                        target,
                        sink,
                        threading.Event(),
                    )
                )
            output = "\n".join(outputs)
            (args.out / f"run-{run}" / "document.md").write_text(output)
            row["seconds"] = time.perf_counter() - t
            row["first_text_s"] = None if sink.first is None else sink.first - t
            row["pages_per_s"] = len(paths) / row["seconds"]
            row["page_markers"] = output.count("<PAGE>")
            row["figures"] = len(
                list((args.out / f"run-{run}").glob("**/images/*.jpg"))
            )
            row["warnings"] = engine.drain_warnings()
        except Exception as exc:  # noqa: BLE001 - record failed real-model runs, never claim success
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            stop.set()
            sampler.join(timeout=5)
            row["peak_vram_mib"] = max(memory) if memory else None
        info["runs"].append(row)
        measured = [
            r["seconds"] for r in info["runs"] if not r["warmup"] and "seconds" in r
        ]
        info["median_ocr_s"] = (
            statistics.median(measured) if len(measured) == 3 else None
        )
        (args.out / "measurements.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if "error" in row:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
