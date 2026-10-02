# Changelog

## Unreleased

Performance figures below were measured on an Apple M4 Max during this work, partly while other jobs shared the machine; they will be re-measured before release (see `docs/OCR_BENCHMARK.md`).

### Apple Silicon / MLX

- New in-process MLX OCR engine for Apple Silicon (`OCR_DEVICE=mlx`), the default on Macs through `OCR_DEVICE=auto`. It ports the Unlimited-OCR model code from mlx-vlm 0.7.4 without depending on mlx-vlm or torch, and reads the same pinned weights. About 3.9 s per page on 8-page chunks, versus 34 s per page on the previous torch MPS default, with the same text-layer recall.
- Optional 8-bit decoder quantization with `OCR_MLX_QUANT_BITS=8` (about 2.7 s per page, same recall).
- `make setup-mlx` installs the MLX engine, the torch MPS fallback and the C++ module together; `make setup-metal` now does the same, and `make dev-metal` runs the torch MPS fallback.
- Faster, leaner torch MPS fallback: fused MoE decode, host-side ring slots and a fixed-size n-gram window take an 8-page chunk from 34 to 9.8 s per page, and Objective-C autorelease pools stop memory from growing over long sessions.
- The device badge shows MLX with the chip name.

### OCR pipeline

- `OCR_DEVICE=auto` picks MLX, then CUDA, then torch MPS, then CPU, and logs the choice; Compose services keep their pinned devices.
- A chunk that hits `MAX_LENGTH` keeps its completed leading pages and reprocesses only from the truncated page. Failed multi-page chunks are recovered page by page (single page, then PDF text layer, then placeholder) instead of becoming placeholders.
- Job messages are split into `warnings` (real quality loss) and `notices` (informational). The viewer reports a degraded result only for warnings. Jobs record `started_at` and `finished_at`.
- The fidelity gate compares letters and digits only, skips untrustworthy text layers and the deterministic textlayer engine, retries lost pages first and stops after three unaccepted retries.
- The textlayer engine reads in content-stream order with column detection and merges fragments, so two-column papers no longer interleave. Text-layer recovery pages render as paragraphs and get translated.
- Merging realigns skipped or split model pages to the right physical page and isolates a page whose layout cannot be parsed.
- Queued jobs survive a restart, and each job keeps the page separator it was built with.
- Sidecars escape literal `[[FIGURE:` text, recover pages truncated at OvisOCR2's output limit from the text layer (or keep them with a warning), retry transient model-load failures with backoff, and restart their container when the inference engine dies. Health and job wait notes show load retries and restarts.

### Translation

- Chat requests stream by default (`TRANSLATE_STREAM`). Cancelling stops generation on the server, and `TRANSLATE_TIMEOUT_S` now bounds the gap between tokens.
- `TRANSLATE_REASONING=off` works on local servers: `TRANSLATE_REASONING_STYLE=auto` sends `chat_template_kwargs` to loopback and private hosts. New `TRANSLATE_EXTRA_BODY` and `TRANSLATE_MAX_RESPONSE_MB` settings.
- Truncated, empty and timed-out responses are never used or cached; the unit stays in the original language with a recorded reason. Stricter output checks catch repetition loops, labels turned into sentences, dropped numbers and canned outputs repeated across units, and old cache entries are re-checked.
- Markdown and layout translations are matched per unit, so partially covered paragraphs no longer keep English lines, with far fewer LLM calls.
- Responses API requests send `store: false`.
- The last translation's warnings appear under the result as "번역 참고 사항".
- README guide for local MLX translation servers (oMLX, LM Studio, mlx_lm.server) with a sample `.env`.

### Page Q&A

- New `local-openai` provider for loopback OpenAI-compatible servers (`LLM_LOCAL_OPENAI_BASE_URL`, `LLM_LOCAL_OPENAI_MODEL`, `LLM_LOCAL_OPENAI_MODELS`, `LLM_LOCAL_OPENAI_API_KEY`).

### PDF export

- On scanned and image pages, translations are drawn over a cover in the page's background colour instead of over the original pixels.
- Redaction removes a band per text line, so neighbouring lines survive. Rotated pages, list bullets, emphasis, LaTeX fractions, roots and accents, and HTML-like text such as `p < 0.05` are flattened correctly, and copied spaces are no longer non-breaking.
- `GET /api/jobs/{id}/pdf/report` returns the build report; after a download the UI lists preserved-block reasons and per-page warnings.
- Builds run in separate worker processes, truly in parallel, without slowing OCR. Export failures return 409 with a reason instead of 500, and caches are validated by a build stamp.
- `PDF_EXPORT_FORMAT_VERSION` is now 10 and `ENRICH_VERSION` 6.

### API and UI

- Job list paging (`limit`, `before`, `has_more`, `total`) with a "더 보기" button.
- Cancelling a queued job takes effect immediately; deleting a job sends `deleted: true` and closes it in open tabs.
- `/api/health` adds `config_warnings`, OCR worker progress (`worker_job_id`, `worker_last_progress_at`, `worker_progress_age_s`) and PDF worker pool counters.
- A 503 with `Retry-After` consistently means "retry later" for exports, event streams, uploads and translation starts; the Korean view and PDF downloads wait and retry instead of falling back.
- Live preview backs off on 429, and event streams fall back to polling as soon as the first connection fails.
- Reader highlights and citations are saved per job in the browser, with Markdown export.
- A "주의 N건" chip (or "참고 N건") lists job warnings and notices with page links; the job list shows a warning badge; health badges show model load failures and a stopped worker.
- `/viewer/pages` answers 304, and `archive.zip` is cached by a content signature.
- `verify_e2e` gains port options and a paired-tag fault; the OCR benchmark gains warm-up runs and process-time columns.

### Security and operations

- PyMuPDF work (rendering, analysis and export) runs in isolated worker processes with per-page and per-build time limits (`PDF_PAGE_TIMEOUT_S`, `PDF_EXPORT_BUILD_TIMEOUT_S`, optional `PDF_WORKER_MEM_LIMIT_MB`). Workers drop credential-like environment variables.
- An upload complexity gate rejects compression and nested-XObject bombs before queuing (`PDF_MAX_PAGE_CONTENT_MB`, `PDF_MAX_PAGE_XOBJECT_CALLS`).
- Content Security Policy on all HTML responses, including `frame-ancestors 'none'`; the SPA meta policy matches the header, and downloaded HTML carries its own offline-only policy. External images are never loaded automatically.
- KaTeX 0.18.10 with size and macro-expansion limits.
- `TRUSTED_PROXY_IPS` limits which peers may set `X-Forwarded-For`; rate limits are checked atomically; live-preview renders and event-stream subscribers are capped.
- Unknown keys in a `.env` the app reads itself (local runs) are reported by name in the log and `/api/health`. Containers have no `.env`, so `config_warnings` stays empty under Docker; the README shows how to check a Compose `.env` from the host.
- Only one backend may own a job store; `.env` is parsed with python-dotenv; numeric settings are validated at startup.
- Dependency upgrades: PyMuPDF 1.28.2 (MuPDF CVE-2026-3308), Pillow 12.3.0, urllib3 2.8.0, anyio 4.14.2. Sidecars install from hash-checked locks.
- CI audits dependencies and builds and smoke-tests the CPU image on amd64 and arm64. Releases wait for CI, gate images on `trivy`, and attach offline image tarballs with `SHA256SUMS`. Actions are pinned to commit SHAs and Dependabot proposes updates.
- Compose drops all capabilities, runs CPU-image backends with a read-only root filesystem and a `/tmp` tmpfs, and limits processes. Application code in the image is root-owned, and uvicorn shuts down within 5 seconds.
- New `make test-mps`, `make test-mlx-real` and `make audit` targets.

### Breaking and behaviour changes

- Local runs default to `OCR_DEVICE=auto` (MLX on Apple Silicon, CUDA on Linux with the cu129 extra). Set `OCR_DEVICE=cpu` for the old behaviour.
- Rename `REASONING_EFFORT` in `.env` to `TRANSLATE_REASONING` (translation) or `LLM_REASONING_EFFORT` (Q&A); the old key was never read.
- Translation streams by default in chat mode; set `TRANSLATE_STREAM=0` for gateways that cannot stream.
- Layouts with only image blocks (old OvisOCR2 jobs) count as no layout: coordinate routes return 404 and `/pdf` returns 409.
- Export failures return 409 instead of 500.
- After upgrading, every cached export PDF and `archive.zip` is rebuilt once, and layout fonts are re-measured once.
- `PDF_EXPORT_MAX_CONCURRENT=1` disables export prewarming. Behind a proxy that is not on loopback, set `TRUSTED_PROXY_IPS`.
- Uploads with pages over the complexity limits are rejected with 400.
- Out-of-range numeric settings fail at startup instead of being clamped.
- The per-job `MAX_LENGTH` budget warning is replaced by a single INFO note at startup.
- A second backend on the same data directory refuses to start.
- The PaddleOCR-VL sidecar image is built for linux/amd64 only.

## 0.1.0 - 2026-09-30

First release of the self-hosted PDF OCR and translation reader.

- Convert PDFs to Markdown with Unlimited-OCR, OvisOCR2, PaddleOCR-VL, or the model-free textlayer engine.
- Read original pages alongside Korean translations, with linked blocks, outlines, and page Q&A.
- Export Markdown, ZIP, HTML, and original/translated comparison PDFs.
- Keep translation requests and progress subscriptions attached to the current document, including rapid document switches.
- Publish a fresh translation state before accepting a rerun and recover cleanly when a worker cannot start.
- Report invalid numeric translation settings as configuration errors.
- Preserve download filenames when the user changes documents during a PDF transfer.
- Close progress streams for jobs canceled while still queued.
- Reserve actual Noto CJK glyph bounds to prevent collisions and keep roomy table headers inside their cells.
- Reject text that cannot fit even at the smallest supported size before expensive native word wrapping, while preserving the original block.
- Identify the release version in the API schema.
- Build and smoke-test CPU Docker images for Linux amd64 and arm64 before publishing to GHCR. Publishing requires successful CI on the exact release commit.

GPU engines remain available through the source Compose profiles; the published container image uses CPU dependencies. Browser and translation E2E checks use local mock LLM providers. Real GPU model inference and external translation providers require deployment-specific validation.
