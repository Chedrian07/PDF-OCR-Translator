# Security

## Supported version

Security fixes are applied to the default branch. Pin the documented model revision and
the checked-in lock files when deploying.

## Reporting

Do not open a public issue containing credentials, private PDFs, extracted page text, or
provider responses. Report a vulnerability privately to the repository owner through
GitHub's private vulnerability reporting or another agreed private channel.

## Deployment boundaries

- OCR runs locally with `baidu/Unlimited-OCR` — in-process with torch (CPU, CUDA or
  Apple MPS) or MLX (Apple Silicon), or in the local GPU sidecars (OvisOCR2,
  PaddleOCR-VL). PDF files, rendered pages, and figures are not sent to an OCR API.
- Translation and Q&A may send extracted text to the explicitly configured LLM provider.
  Review the provider and model before enabling these opt-in features. Requests to the
  Responses API (translation and Q&A) carry `store: false`.
- The features use separate credentials and never fall back to one another:
  - `OPENAI_API_KEY` belongs to the translation provider and may point at any
    OpenAI-compatible base URL.
  - `LLM_OPENAI_API_KEY` is the Q&A key and is always sent to the official
    `https://api.openai.com` host that `LLM_OPENAI_BASE_URL` is pinned to. Never reuse a
    third-party gateway key as `LLM_OPENAI_API_KEY` — it would be handed to OpenAI. Q&A
    is unavailable (advertised as `available: false`) rather than falling back to the
    translation key.
  - `LLM_LOCAL_OPENAI_API_KEY` is the only key the `local-openai` Q&A provider sends.
    Its `LLM_LOCAL_OPENAI_BASE_URL` must be `127.0.0.1`, `localhost`, `::1` or
    `host.docker.internal`; anything else (DNS names, other IPs, userinfo, query strings)
    fails at startup, and redirects are not followed. The `local-openai` and Ollama
    clients ignore `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` and system proxy settings, so a
    proxy configured for model downloads never receives page text, questions or the
    local key.
- Keep `.env`, private keys, certificates, job data, and model caches outside Git and the
  Docker build context. The supplied ignore files enforce the common cases. Unknown
  `.env` keys are reported by name only (never by value) in the server log and in the
  `config_warnings` field of the unauthenticated `GET /api/health`.
- The application is unauthenticated and has no CSRF protection. The Compose default
  binds ports to `0.0.0.0` with `ALLOWED_HOSTS=*`, assuming a trusted network
  (VPN/Tailscale, firewalled LAN). Put authentication and TLS at a reverse proxy before
  exposing beyond that. To restore loopback-only operation set **both** values in
  `.env`: `BIND_HOST=127.0.0.1` (port binding) and `ALLOWED_HOSTS=localhost,127.0.0.1`
  (Host header allowlist). Setting `BIND_HOST` alone leaves the wildcard `ALLOWED_HOSTS`
  in place, so the DNS rebinding path stays open.
- Ports that Docker publishes are not filtered by host firewall front-ends such as `ufw`
  or `firewalld`: Docker inserts its own forwarding rules ahead of them. "Firewalled"
  must therefore mean a network firewall, rules in the `DOCKER-USER` chain, or a
  `BIND_HOST` that is not reachable from outside (loopback, or the VPN interface address).
- Keep `ALLOWED_HOSTS` narrow when clients reach the service via a stable hostname/IP.
  Do not rely on the wildcard default on an untrusted network.
- Behind a reverse proxy, set `TRUSTED_PROXY_HOPS` to the number of proxies and
  `TRUSTED_PROXY_IPS` to the proxies' addresses or CIDRs (default: loopback only; for a
  host proxy reaching a container through a published port, the Docker bridge gateway,
  e.g. `172.17.0.1`). `X-Forwarded-For` is honoured only when the direct peer is in that
  list; other peers are rate limited by their own address and the spoofed header is
  ignored (logged once).
- `POST /api/jobs/{id}/qa` and `POST /api/jobs/{id}/translate` are rate limited per job
  and per client IP over a 60s sliding window, and capped on concurrent execution.
  Requests over a cap are rejected with `429` and a `Retry-After` header. Defaults:
  `QA_RATE_LIMIT_PER_MIN=30`, `QA_MAX_CONCURRENT=4`, `TRANSLATE_RATE_LIMIT_PER_MIN=12`,
  `TRANSLATE_MAX_ACTIVE=4`; a value of 0 or less disables that cap. These bound the cost
  of mistakes and casual abuse of the operator's paid LLM key — they are **not a
  substitute for authentication**, and anyone who can reach the service can still read
  and delete documents. `docker-compose.yml` passes all four variables to every backend
  service, so set them in `.env` to tighten the caps for container deployments too
  (`backend/tests/test_ci_ops_contracts.py` keeps that wiring from regressing). A key is
  checked against all of its buckets at once, so a request refused by the per-IP limit
  does not use up the per-job budget.
- Other fixed limits: request bodies are capped before parsing (`MAX_UPLOAD_MB` plus
  64 KiB for uploads, 256 KiB for `POST /render-preview`, 64 KiB elsewhere);
  `/render-preview` is also rate limited (16 KiB cost units, 600 units per minute per job
  and per IP, 4 concurrent renders); server-sent event streams are capped at 8
  subscribers per job and 64 in total (`503` with `Retry-After: 5` beyond that).

## Untrusted PDFs and model output

Uploaded PDFs, and the text an OCR model produces from them, are treated as hostile.

- **PyMuPDF runs outside the server process.** Rendering, text-layer analysis, font
  measurement and translated-PDF builds run in spawned worker processes (pools `ocr`,
  `export` and `probe`). Each task has a wall-clock limit (`PDF_PAGE_TIMEOUT_S`, default
  60 s per page; `PDF_EXPORT_BUILD_TIMEOUT_S`, default 900 s per build). On a timeout,
  crash or cancel only that worker is killed; a page that timed out or crashed is skipped
  by later analysis steps instead of being retried. Workers arm `SIGALRM` at their limit
  plus 10 s, so a runaway task ends even if the server itself was killed. On Linux,
  workers set `oom_score_adj=1000` so the kernel kills a worker before the server, and
  `PDF_WORKER_MEM_LIMIT_MB` (off by default) adds an address-space limit.
- Workers remove credential-like environment variables (names containing `KEY`,
  `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL`) and variables whose values can carry
  credentials (names ending in `_URL`/`_URI` such as `OPENAI_BASE_URL`, proxy variables
  such as `HTTPS_PROXY`, and `TRANSLATE_EXTRA_BODY`) when they start. This only cleans the
  Python environment and processes the worker starts; the kernel keeps the original
  environment block (`/proc/self/environ`) and the server's own environment is readable by
  the same user, so treat API keys as exposed if a worker is ever compromised and rotate
  them. Each worker gets a private scratch directory (mode `0700`, unpredictable name)
  created by the server. A worker is a fault and resource boundary, **not a privilege
  sandbox**: it runs as the same user with the same files, and the server trusts the
  results it sends back.
- **Upload complexity gate.** Before a job is queued, each page is measured without
  rendering: the decompressed size of every stream it draws (content, form XObjects,
  annotation appearances, tiling patterns, Type3 glyphs) and the number of draw calls
  with nested form XObjects expanded. Pages over `PDF_MAX_PAGE_CONTENT_MB` (64) or
  `PDF_MAX_PAGE_XOBJECT_CALLS` (2,000,000), or with forms nested more than 64 levels
  deep, are rejected with `400` and a message naming the setting. A 3 KB file whose
  nested forms expand to 10^12 draw calls and a 1 GB flate bomb are both rejected in
  milliseconds. Names are decoded the way MuPDF reads them, and a `Do` whose target cannot
  be confirmed is charged the most expensive form in its resource dictionary, so escaped
  or non-ASCII names cannot hide a bomb. Strings and comments are followed with MuPDF's
  lexer rules, so the word "Do" in page text is not taken for an operator; after an
  inline image, or where a content stream ends inside a string or comment, every `Do`
  is still counted. At most 4 uploads are measured at once; excess
  uploads get `503` with `Retry-After: 5` instead of holding server threads. Costs the
  gate does not count (huge images, shadings, repeated Type3 glyphs) are bounded by the
  per-page time limit.
- Corrupt PDFs are rejected with a fixed message that contains no server paths. Pillow's
  decompression-bomb limit is lowered to 5% above the largest page render the app
  produces (52.5 million pixels).
- Model output never reaches `eval()` (vendor patches P8/P9 use `ast.literal_eval`), and
  model bounding boxes are clamped to the page before cropping (vendor patch P22, shared
  by the MLX port). Sidecar responses are schema-checked and sanitized; see
  `docs/OCR_ENGINE_PROTOCOL.md`.

## Browser-side protections

- **Content Security Policy, in layers.** Every HTML response carries a CSP header:
  `default-src 'self'`, `img-src 'self' data: blob:`, `font-src 'self' data:`,
  `style-src 'self' 'unsafe-inline'`, `object-src 'none'`, `base-uri 'self'`,
  `form-action 'self'` and `frame-ancestors 'none'` (no framing, against clickjacking).
  The SPA gets `script-src 'self'` plus hashes of any inline scripts in `index.html`
  (there are none today — the theme bootstrap lives in `theme-init.js`); API HTML such as
  `document.html` gets `script-src 'self' 'unsafe-inline'` for its inline KaTeX.
  `index.html` repeats the same policy in a `<meta>` tag for every directive a meta tag
  can express (`frame-ancestors` is header-only). Edit both layers together;
  `backend/tests/test_security_headers.py` fails if they drift.
- `Referrer-Policy: same-origin` (header and `index.html` meta) keeps the instance address
  out of requests to other origins.
- `X-Content-Type-Options: nosniff` on every response, so browsers never reinterpret a
  response outside its declared `Content-Type`.
- **Standalone downloads** (`document.html`, both the facsimile and the semantic export)
  are self-contained and carry their own meta CSP — `default-src 'none'; img-src data:
  blob:; font-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'` — and
  `<meta name="referrer" content="no-referrer">`. A file opened from disk renders
  normally but cannot fetch anything.
- **Translated PDFs** do not carry the upload's active content. The single-view export
  edits the uploaded file, so before saving it drops `/OpenAction` scripts, every `/AA`
  (additional actions), document-level JavaScript, XFA forms, embedded files and file
  attachments, and link or outline actions that run JavaScript, launch programs, submit
  or import form data, open other files (`GoToR`/`GoToE`) or play rich media. Internal
  links, URI links and the outline stay. The side-by-side export is a new document and
  never had any.
- **External images are never loaded automatically.** The server renderer only turns
  `images/<file>` and `data:image/(png|jpeg|gif|webp)` into `<img>`; any other source
  (remote URLs, LAN addresses, absolute paths) becomes a click-only link. The SPA also
  replaces non-same-origin image sources with a placeholder before inserting server
  HTML, so a tracking image in a document cannot reveal that it was opened.
- Math is typeset with vendored KaTeX 0.18.10 (GHSA-238p-pmpm-9mq7 is fixed from 0.18.2)
  using `maxSize: 10`, `maxExpand: 1000`, `strict: 'ignore'` and `trust: false`, in the
  app and in standalone exports alike. `frontend/tests/katex-vendor.test.mjs` pins the
  vendored version and the options.
- Static frontend files are served with `Cache-Control: no-cache`, so an upgrade cannot
  leave a browser running a mix of old and new modules.

## Supply chain and image scanning

- CI (`dependency-audit` job) runs `pip-audit` on the backend `uv.lock` (the same CPU set
  the image ships) and on both sidecar locks. `make audit` (`scripts/dependency_audit.sh`)
  runs the same audit locally with the same accepted list. The release workflow refuses
  to push an image while `trivy` reports a HIGH or CRITICAL vulnerability that has a fix.
- Accepted advisories are listed with a reason and a review date: pip-audit IDs in
  `.github/workflows/ci.yml` (mirrored in `scripts/dependency_audit.sh`), image-scan IDs
  in `.github/trivyignore.yaml`. They all come from the model-card pins
  (`transformers==4.57.1`, `torch==2.10.0`) and cover code paths the application does not
  use. The lists expire on 2027-04-01; after that the gates fail until someone reviews
  them again.
- `backend/tests/test_dependency_floor.py` fails if PyMuPDF/MuPDF, Pillow, urllib3 or
  anyio drop below their security-fix versions.
- Base images are pinned by digest and rebuilt with `apt-get upgrade`. The PaddleOCR-VL
  sidecar installs a fully pinned, hash-checked lock; the OvisOCR2 sidecar overlays a
  hash-checked web layer on its digest-pinned vLLM base. GitHub Actions are pinned to full
  commit SHAs. Dependabot proposes backend, action and base-image updates weekly.

## Container hardening

Every service in `docker-compose.yml` runs as uid 1000 with `no-new-privileges` and
`cap_drop: [ALL]`. All four backends have a process limit (1024). The CPU-image backends
(`ocr-cpu`, `ocr-ovis`, `ocr-paddle`) also run with a read-only root filesystem — the
`/data` volume and a 2 GB `/tmp` tmpfs (raise it together with `MAX_UPLOAD_MB`) are the
only writable paths — and the application code inside the image is root-owned and
read-only. `ocr-cuda` and the GPU sidecars do not use a read-only root filesystem yet;
that needs to be validated on a GPU host first.

## Secret response

If a secret is committed, revoke or rotate it first. Removing it from a later commit is
not sufficient; rewrite repository history before sharing the repository further.
