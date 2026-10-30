# Fixed Unlimited-OCR-2 evaluation set

The committed PDFs are the authoritative inputs (`manifest.json` records their hashes).
They are synthetic and contain no user documents. `scripts/make_unlimited2_eval.py`
generates equivalent documents; the original two PDFs were created before canonical
PDF-ID normalization was added. Use the committed files to reproduce exact byte hashes.

- `en-mixed.pdf`: English, table, inline equation, raster chart and picture, two pages.
- `ko-report.pdf`: Korean/English, Hangul jamo, table, equation and vector chart.
- `scan.pdf`: deterministic rotation, contrast/noise and JPEG degradation of the Korean page.
- `columns.pdf`: two columns with sixteen labelled reading-order landmarks.
- `long-32.pdf`: thirty-two pages with unique physical page numbers and repeated mixed content.

`results/<document>/{baseline,candidate}/measurements.json` includes one warmup and three
measured runs on the same RTX 5070 Ti. Raster hashes match between both models.
Baseline is commit `cc97954`, model `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead`;
candidate uses model `c14071b549f4fcab10909f3c4acc6a2edb366947` and the pinned official SGLang wheel.
Both use BF16, 200 DPI, 1024 global views, concurrency one, eight pages per chunk.

The first measured output and figure coordinates are retained. `quality.json` records
normalized alphanumeric CER, table shape, formula markup count, landmark order and figure
IoU (missing figures count as zero). These are diagnostics, not a complete formula-semantic
or layout benchmark. Scan figure IoU is unannotated. The synthetic reference text includes
labels inside drawn figures; CER is consequently not a certified recognition score.

```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/benchmark_unlimited2.py \
  --pdf benchmark_docs/unlimited2/en-mixed.pdf --out /tmp/uocr2-reproduce
backend/.venv/bin/python scripts/score_unlimited2.py \
  --truth benchmark_docs/unlimited2/en-mixed.truth.json \
  --baseline benchmark_docs/unlimited2/results/en-mixed/baseline \
  --candidate benchmark_docs/unlimited2/results/en-mixed/candidate \
  --out /tmp/uocr2-quality.json
```

Run baseline measurements from the old checkout using its CUDA environment and the same
benchmark script with `--baseline`. Do not run another OCR workload on the measured GPU.
See [validation status](../../docs/UNLIMITED_OCR_2_VALIDATION.md) for the unmet gates.
