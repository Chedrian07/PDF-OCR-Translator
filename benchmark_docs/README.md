# OCR evaluation

Current model: `baidu/Unlimited-OCR-2`, revision
`c14071b549f4fcab10909f3c4acc6a2edb366947`.

The [fixed evaluation set](unlimited2/README.md) contains English/Korean, scan, table,
equation, figure, two-column and 32-page inputs, references and measured v1/v2 results.
See [validation status](../docs/UNLIMITED_OCR_2_VALIDATION.md) for the unmet quality gates.

```bash
# Real CUDA runtime must already be running.
PYTHONPATH=backend backend/.venv/bin/python scripts/benchmark_unlimited2.py \
  --pdf benchmark_docs/unlimited2/en-mixed.pdf --out /tmp/uocr2-benchmark
# Generate additional equivalent synthetic inputs:
backend/.venv/bin/python scripts/make_unlimited2_eval.py --out /tmp/uocr2-inputs
```

The benchmark warms up once and records three runs with input/raster hashes, first text,
OCR duration, page throughput, GPU memory and artifact counts. Use the same GPU, input,
DPI, precision and concurrency for comparisons. `--baseline` requires the old checkout
and its environment; the current product cannot select the old model.

The generic HTTP harness `scripts/benchmark_ocr_engines.py` remains available for
comparing running backend deployments. Mock/FakeEngine results are not real OCR evidence.
