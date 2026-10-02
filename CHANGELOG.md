# Changelog

## Unreleased

Performance figures below were measured on an Apple M4 Max on a quiet machine on 2026-10-02, with nothing else running but the desktop (see `docs/OCR_BENCHMARK.md`).

### Apple Silicon / MLX

- New in-process MLX OCR engine for Apple Silicon (`OCR_DEVICE=mlx`), the default on Macs through `OCR_DEVICE=auto`. It ports the Unlimited-OCR model code from mlx-vlm 0.7.4 without depending on mlx-vlm or torch, and reads the same pinned weights. About 3.8 s per page on 8-page chunks (a whole 25-page job takes 96 s, 3.85 s per page), versus 34 s per page on the previous torch MPS default, with the same text-layer recall.
- Optional 8-bit decoder quantization with `OCR_MLX_QUANT_BITS=8` (about 2.6 s per page, same recall).
- `make setup-mlx` installs the MLX engine, the torch MPS fallback and the C++ module together; `make setup-metal` now does the same, and `make dev-metal` runs the torch MPS fallback.
- Faster, leaner torch MPS fallback: fused MoE decode, host-side ring slots and a fixed-size n-gram window take an 8-page chunk from 34 to 9.1 s per page, and Objective-C autorelease pools stop memory from growing over long sessions.
- The device badge shows MLX with the chip name.

### OCR pipeline

- `OCR_DEVICE=auto` picks MLX, then CUDA, then torch MPS, then CPU, and logs the choice; Compose services keep their pinned devices.
- Loading the pinned model no longer contacts huggingface.co when the snapshot is already in the cache (the torch path used to probe missing optional files and the MLX path the revision API on every load), so offline and air-gapped hosts start without network timeouts. Branch or tag revisions still check the Hub.
- A chunk that hits `MAX_LENGTH` keeps its completed leading pages and reprocesses only from the truncated page. With a text layer, the kept pages are first checked against the source, so a page the model split or skipped before the cut is not lost; scans still trust the model's page markers. Failed multi-page chunks are recovered page by page (single page, then PDF text layer, then placeholder) instead of becoming placeholders, after the failed attempt's memory is released.
- Cancelling during page-by-page recovery or a fidelity retry stops cleanly; a partially generated page is merged with a warning instead of looking complete.
- Job messages are split into `warnings` (real quality loss) and `notices` (informational). The viewer reports a degraded result only for warnings. Jobs record `started_at` and `finished_at`.
- The fidelity gate compares letters and digits only, skips untrustworthy text layers and the deterministic textlayer engine, retries lost pages first and stops after three unaccepted retries.
- The textlayer engine reads in content-stream order with column detection and merges fragments, so two-column papers no longer interleave. Text-layer recovery pages render as paragraphs and get translated.
- Merging realigns skipped or split model pages to the right physical page and isolates a page whose layout cannot be parsed. Chunks whose page markers already match are no longer realigned or flagged.
- Figures after a malformed image box (a string, empty list, `None` or malformed flat list) or a padded image label point at the right crop file in both the torch and MLX engines, and the layout view numbers crops the same way.
- On Macs without MLX, `OCR_DEVICE=auto` (torch MPS) keeps the `PYTORCH_ENABLE_MPS_FALLBACK` safety net: the default is set before torch is first imported, and the metal engine warns when that is too late.
- The textlayer engine keeps wrapped-cell tables and figure grids in row order.
- Queued jobs survive a restart in their original queue order (even when uploaded in the same second), and each job keeps the page separator it was built with.
- Sidecars escape literal `[[FIGURE:` text, recover pages truncated at OvisOCR2's output limit from the text layer (or keep them with a warning), retry transient model-load failures with backoff, and restart their container when the inference engine dies. Health and job wait notes show load retries and restarts. A page that keeps killing the inference engine is re-sent at most once after the restart (no extra runner retry), restarts that recover are recorded as notices instead of warnings, OvisOCR2 jobs with `OCR_REMOTE_PAGE_CONCURRENCY` > 1 no longer get false page-marker warnings, rejected OvisOCR2 figure tags can no longer forge figure placeholders, and truncated pages that could not be checked name the real reason.
- When a multi-page sidecar chunk fails, page-by-page recovery reuses the pages that already finished and does not resend a page that timed out.

### Translation

- Chat requests stream by default (`TRANSLATE_STREAM`). Cancelling stops generation on the server, and `TRANSLATE_TIMEOUT_S` now bounds the gap between tokens.
- `TRANSLATE_REASONING=off` works on local servers: `TRANSLATE_REASONING_STYLE=auto` sends `chat_template_kwargs` to loopback and private hosts. New `TRANSLATE_EXTRA_BODY` and `TRANSLATE_MAX_RESPONSE_MB` settings.
- Truncated, empty and timed-out responses are never used or cached; the unit stays in the original language with a recorded reason. Stricter output checks catch repetition loops, labels turned into sentences, dropped numbers, canned outputs repeated across units and the source copied verbatim inside Korean filler text (`echo`), and old cache entries are re-checked.
- Markdown and layout translations are matched per unit, so partially covered paragraphs no longer keep English lines, with far fewer LLM calls.
- Responses API requests send `store: false`.
- The response size cap is enforced while reading, including length-less HTTP/1.0 streams (mlx_lm.server), declared `Content-Length` on streams and compressed bodies; long stream lines parse in linear time.
- A translation endpoint that stops responding fails the job after a wave of timed-out units instead of finishing hours later with English text.
- Dollar signs a model wraps around a formula (`$〈y, x〉$`) are dropped before the formula is restored, so they no longer show up in the translation and the translated PDF.
- Streamed translations stop a runaway output early: once the answer is twice as long as the prompt, a repetition loop or an answer over four times the prompt ends the request and closes the connection, instead of generating to `max_tokens` (small local models spent most of their time on such loops).
- Transient mid-stream provider errors (OpenRouter-style error events) are retried like HTTP 5xx, and streamed requests reuse keep-alive connections.
- A truncation retry that exceeds the response cap is treated as a truncated unit, not a failed job.
- `TRANSLATE_REASONING_STYLE=auto` recognises regional OpenAI hosts (`eu.api.openai.com`) and Podman, LAN and `.internal` host names.
- Connection errors no longer show the base URL or its query string, and `TRANSLATE_EXTRA_BODY` values are no longer written to `state.json`.
- The last translation's warnings appear under the result as "번역 참고 사항".
- The "참고문헌 규칙 불일치" warning only appears when the translated Markdown and the PDF really treat a reference line differently; Unlimited-OCR papers no longer get it on every translation.
- README guide for local MLX translation servers (oMLX, LM Studio, mlx_lm.server) with a sample `.env`.

### Page Q&A

- New `local-openai` provider for loopback OpenAI-compatible servers (`LLM_LOCAL_OPENAI_BASE_URL`, `LLM_LOCAL_OPENAI_MODEL`, `LLM_LOCAL_OPENAI_MODELS`, `LLM_LOCAL_OPENAI_API_KEY`).
- `local-openai` answers cut off at the token limit are reported as an error instead of being shown as complete.
- The question tab starts on a provider that is actually available (for example `local-openai` when no OpenAI key is set) instead of the unconfigured server default, and Thinking starts off for local providers (Ollama, `local-openai`), whose thinking models otherwise spent the whole answer budget thinking. A choice you made yourself is kept.

### PDF export

- On scanned and image pages, translations are drawn over a cover in the page's background colour instead of over the original pixels.
- Redaction removes a band per text line, so neighbouring lines survive. Rotated pages, list bullets, emphasis, LaTeX fractions, roots and accents, and HTML-like text such as `p < 0.05` are flattened correctly, and copied spaces are no longer non-breaking.
- `GET /api/jobs/{id}/pdf/report` returns the build report; after a download the UI lists preserved-block reasons and per-page warnings. The report also counts preserved blocks per page (`kept_pages`) and lists every page with a warning (`warning_pages`), beyond the 50-warning sample, and `verify_e2e` uses them to tell explained from silent translation loss.
- Builds run in separate worker processes, truly in parallel, without slowing OCR (a 25-page translated PDF builds in about 8 s; MLX decoding stays at about 290 tok/s meanwhile). Export failures return 409 with a reason instead of 500, and caches are validated by a build stamp.
- Scanned tables are translated from a pixel grid: only the changed cells' text is covered, rules and untouched cells stay intact, and a table whose columns cannot be located is kept with a warning.
- Rotated pages lay out multi-line translations in screen orientation.
- Translated PDF builds are about a third faster on dense papers (25-page paper 40 s → 26 s): layout trials run on a blank page with the same geometry instead of rescanning the source page's fonts on every attempt. The output is unchanged.
- List items left untranslated (the unit kept its original text) keep their original typesetting instead of being redrawn as plain text with stray fraction bars.
- Inline math rules (fraction bars, root overlines) inside a translated paragraph are removed with its text and no longer block its space, so translations are not shrunk below readable size and no stray bars remain (a 25-page paper went from seven shrink/no-fit warnings to one).
- Inner-product brackets (`\langle`, `\rangle`) are drawn as ⟨ ⟩ (〈 〉 on fonts without them) instead of the words `langle`/`rangle`, and a symbol command right before `\boldsymbol{…}` or `\frac{…}{…}` no longer fuses with it (`langley`, `cdotx`).
- Every piece of a large TeX delimiter (stacked `|`/`‖` bars) is removed with its paragraph, so no stray bars overlap the translation. Export format version 15 rebuilds cached PDFs.
- A short symbol that sticks out of its line and only grazes the OCR box (a radical `√`, a tall bracket, an accent) is removed with the paragraph it touches when that paragraph is translated, instead of staying on top of the translation. Export format version 16 rebuilds cached PDFs.
- Scan covers leave overlapping figures and kept blocks intact; tiled and margined scans and searchable scans with the text layer under the image are recognised; narrow multi-line scan columns are no longer kept as vertical text; lines set in fonts with degenerate metrics are removed.
- `PDF_EXPORT_FORMAT_VERSION` is now 16 and `ENRICH_VERSION` 6.

### API and UI

- Job list paging (`limit`, `before`, `has_more`, `total`) with a "더 보기" button.
- Cancelling a queued job takes effect immediately; deleting a job sends `deleted: true` and closes it in open tabs.
- `/api/health` adds `config_warnings`, OCR worker progress (`worker_job_id`, `worker_last_progress_at`, `worker_progress_age_s`) and PDF worker pool counters.
- A 503 with `Retry-After` consistently means "retry later" for exports, event streams, uploads and translation starts; the Korean view and PDF downloads wait and retry instead of falling back.
- Live preview backs off on 429, and event streams fall back to polling as soon as the first connection fails.
- Reader highlights and citations are saved per job in the browser, with Markdown export. Notes merge with what other tabs saved instead of overwriting it, stay in sync across tabs, free space from the oldest other documents when browser storage is full (the save message says so), and keep keyboard focus after a delete. A language switch clears a stale selection, so citations keep the right language.
- Re-measuring layout fonts after an upgrade runs in the background, once per layout; reader routes wait at most 2 seconds for it instead of stalling behind PDF builds.
- Health badges update only what changed, so screen readers no longer re-read them on every poll.
- The full-screen viewer makes everything outside it inert, including the translation and PDF report lists.
- Negative KaTeX sizes are clamped and oversized formulas fall back to their TeX source, in the app and in downloaded HTML. The front end no longer uses regex lookbehind, which left Safari 16.0–16.3 with a blank page.
- A "주의 N건" chip (or "참고 N건") lists job warnings and notices with page links; the job list shows a warning badge; health badges show model load failures and a stopped worker.
- `/viewer/pages` answers 304, and `archive.zip` is cached by a content signature.
- `HEAD` works on the API's download and page routes (same status and headers as `GET`, no body) instead of falling through to the front end and returning 404; event streams still take `GET` only.
- `verify_e2e` gains port options and a paired-tag fault; the OCR benchmark gains warm-up runs and process-time columns. The mock browser E2E fails on uncaught page errors in every browser context.

### Security and operations

- PyMuPDF work (rendering, analysis and export) runs in isolated worker processes with per-page and per-build time limits (`PDF_PAGE_TIMEOUT_S`, `PDF_EXPORT_BUILD_TIMEOUT_S`, optional `PDF_WORKER_MEM_LIMIT_MB`). Workers drop credential-like environment variables from their own environment (rotate keys if a worker is ever compromised), use private scratch directories, and on Linux are recycled by their own peak memory rather than the server's.
- An upload complexity gate rejects compression and nested-XObject bombs before queuing (`PDF_MAX_PAGE_CONTENT_MB`, `PDF_MAX_PAGE_XOBJECT_CALLS`). It decodes names the way MuPDF does and charges uncertain form calls at the most expensive form, so escaped or non-ASCII names cannot hide a bomb, and it rejects tangled form cycles. At most 4 uploads are validated at once; extra uploads get 503 with `Retry-After` instead of holding server threads.
- Content Security Policy on all HTML responses, including `frame-ancestors 'none'`; the SPA meta policy matches the header, and downloaded HTML carries its own offline-only policy. External images are never loaded automatically.
- KaTeX 0.18.10 with size and macro-expansion limits.
- `TRUSTED_PROXY_IPS` limits which peers may set `X-Forwarded-For`; proxy hops are counted correctly when uvicorn has already applied `X-Forwarded-For` from a loopback proxy, every `X-Forwarded-For` line is read, and 429 `Retry-After` is rounded up. Rate limits are checked atomically; live-preview renders and event-stream subscribers are capped.
- Unknown keys in a `.env` the app reads itself (local runs) are reported by name in the log and `/api/health` (MLX, Metal, Objective-C and Tesseract runtime keys and `OLLAMA_MEM_LIMIT` are recognised). Containers have no `.env`, so `config_warnings` stays empty under Docker; the README shows how to check a Compose `.env` from the host.
- Only one backend may own a job store; `.env` is parsed with python-dotenv; numeric settings are validated at startup.
- When a PDF worker cannot run even an empty self-check (for example `PDF_WORKER_MEM_LIMIT_MB` on an x86_64 image under emulation), uploads fail with a 500 that names the server configuration instead of calling every PDF corrupt (400). `.env.example` warns against that limit under emulation.
- Dependency upgrades: PyMuPDF 1.28.2 (MuPDF CVE-2026-3308), Pillow 12.3.0, urllib3 2.8.0, anyio 4.14.2. Sidecars install from hash-checked locks, and sidecar CI installs its web layer from each service's `requirements.lock`.
- The Docker build context excludes `backend/data` and `.env`, key and log files at any depth.
- CI audits dependencies and builds and smoke-tests the CPU image on amd64 and arm64. Releases wait for CI, gate images on `trivy`, and attach offline image tarballs with `SHA256SUMS`. Actions are pinned to commit SHAs and Dependabot proposes updates.
- Services in `docker-compose.yml` drop all capabilities, the backends have a process limit, and the CPU-image backends run with a read-only root filesystem and a `/tmp` tmpfs. The optional Ollama overlay keeps the upstream image's root user and default capabilities. Application code in the image is root-owned, and uvicorn shuts down within 5 seconds.
- Stopping the server while OCR is running (for example `docker stop`) exits with code 0 instead of aborting (exit 133 in containers): after cleanup, the native runtime teardown under the still-running inference thread is skipped. The interrupted job is still marked as interrupted on the next start, and jobs still waiting in the queue are no longer started during shutdown, so they run after the restart.
- New `make test-mps`, `make test-mlx-real` and `make audit` targets. `make test-mlx-real` reads the Hugging Face cache location from `.env` like `make dev` and stops with exit code 2 before pytest when the pinned snapshot is missing.

### Breaking and behaviour changes

- Local runs default to `OCR_DEVICE=auto` (MLX on Apple Silicon, CUDA on Linux with the cu129 extra). Set `OCR_DEVICE=cpu` for the old behaviour.
- Rename `REASONING_EFFORT` in `.env` to `TRANSLATE_REASONING` (translation) or `LLM_REASONING_EFFORT` (Q&A); the old key was never read.
- Translation streams by default in chat mode; set `TRANSLATE_STREAM=0` for gateways that cannot stream.
- Layouts with only image blocks (old OvisOCR2 jobs) count as no layout: coordinate routes return 404 and `/pdf` returns 409.
- Export failures return 409 instead of 500.
- After upgrading, every cached export PDF and `archive.zip` is rebuilt once, and layout fonts are re-measured once.
- `PDF_EXPORT_MAX_CONCURRENT=1` disables export prewarming. Behind a proxy that is not on loopback, set `TRUSTED_PROXY_IPS`.
- Uploads with pages over the complexity limits are rejected with 400, and more than 4 concurrent uploads get 503 with `Retry-After`.
- A translation job fails when a whole wave of units times out with no success; for slow non-streaming endpoints raise `TRANSLATE_TIMEOUT_S` or lower `TRANSLATE_CONCURRENCY`.
- Jobs translated with `TRANSLATE_EXTRA_BODY`, or with `TRANSLATE_REASONING` behind `*.api.openai.com` or `*.localhost`, `*.lan`, `*.home.arpa` and `*.internal` hosts, re-translate once. An internal gateway that forwards to OpenAI must set `TRANSLATE_REASONING_STYLE` explicitly.
- Out-of-range numeric settings fail at startup instead of being clamped.
- The per-job `MAX_LENGTH` budget warning is replaced by a single INFO note at startup.
- A second backend on the same data directory refuses to start.
- The PaddleOCR-VL sidecar image is built for linux/amd64 only.
- The automatic API pages `/docs` and `/redoc` are turned off: they load scripts from a CDN that the Content Security Policy blocks, so they only showed a blank page. The schema stays at `/openapi.json`.

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
