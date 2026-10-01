# Security

## Supported version

Security fixes are applied to the default branch. Pin the documented model revision and
the checked-in lock files when deploying.

## Reporting

Do not open a public issue containing credentials, private PDFs, extracted page text, or
provider responses. Report a vulnerability privately to the repository owner through
GitHub's private vulnerability reporting or another agreed private channel.

## Deployment boundaries

- OCR runs locally with `baidu/Unlimited-OCR`; PDF files, rendered pages, and figures are
  not sent to an OCR API.
- Translation and Q&A may send extracted text to the explicitly configured LLM provider.
  Review the provider and model before enabling these opt-in features.
- The two features use separate credentials. `OPENAI_API_KEY` belongs to the translation
  provider and may point at any OpenAI-compatible base URL; `LLM_OPENAI_API_KEY` is the
  Q&A key and is always sent to the official `https://api.openai.com` host that
  `LLM_OPENAI_BASE_URL` is pinned to. Never reuse a third-party gateway key as
  `LLM_OPENAI_API_KEY` — it would be handed to OpenAI. Q&A is unavailable (advertised as
  `available: false`) rather than falling back to the translation key.
- Keep `.env`, private keys, certificates, job data, and model caches outside Git and the
  Docker build context. The supplied ignore files enforce the common cases.
- The application is unauthenticated. The Compose default binds ports to `0.0.0.0` with
  `ALLOWED_HOSTS=*`, assuming a trusted network (VPN/Tailscale, firewalled LAN). Put
  authentication and TLS at a reverse proxy before exposing beyond that. To restore
  loopback-only operation set **both** values in `.env`: `BIND_HOST=127.0.0.1` (port
  binding) and `ALLOWED_HOSTS=localhost,127.0.0.1` (Host header allowlist). Setting
  `BIND_HOST` alone leaves the wildcard `ALLOWED_HOSTS` in place, so the DNS rebinding
  path stays open.
- Keep `ALLOWED_HOSTS` narrow when clients reach the service via a stable hostname/IP.
  Do not rely on the wildcard default on an untrusted network.
- `POST /api/jobs/{id}/qa` and `POST /api/jobs/{id}/translate` are rate limited per job
  and per client IP over a 60s sliding window, and capped on concurrent execution.
  Requests over a cap are rejected with `429` and a `Retry-After` header. Defaults:
  `QA_RATE_LIMIT_PER_MIN=30`, `QA_MAX_CONCURRENT=4`, `TRANSLATE_RATE_LIMIT_PER_MIN=12`,
  `TRANSLATE_MAX_ACTIVE=4`; a value of 0 or less disables that cap. These bound the cost
  of mistakes and casual abuse of the operator's paid LLM key — they are **not a
  substitute for authentication**, and anyone who can reach the service can still read
  and delete documents. `docker-compose.yml` passes all four variables to every backend
  service, so set them in `.env` to tighten the caps for container deployments too
  (`backend/tests/test_ci_ops_contracts.py` keeps that wiring from regressing).

## Supply chain and image scanning

- CI (`dependency-audit` job) runs `pip-audit` on the backend `uv.lock` (the same CPU set
  the image ships) and on both sidecar locks. The release workflow refuses to push an
  image while `trivy` reports a HIGH or CRITICAL vulnerability that has a fix.
- Accepted advisories are listed with a reason and a review date: pip-audit IDs in
  `.github/workflows/ci.yml`, image-scan IDs in `.github/trivyignore.yaml`. They all come
  from the model-card pins (`transformers==4.57.1`, `torch==2.10.0`) and cover code paths
  the application does not use. The lists expire on 2027-04-01; after that the gates fail
  until someone reviews them again.
- Base images are pinned by digest and rebuilt with `apt-get upgrade`. The PaddleOCR-VL
  sidecar installs a fully pinned, hash-checked lock; the OvisOCR2 sidecar overlays a
  hash-checked web layer on its digest-pinned vLLM base. GitHub Actions are pinned to full
  commit SHAs. Dependabot proposes backend, action and base-image updates weekly.

## Container hardening

Every Compose service runs as uid 1000 with `no-new-privileges` and `cap_drop: [ALL]`.
The CPU-image backends (`ocr-cpu`, `ocr-ovis`, `ocr-paddle`) also run with a read-only
root filesystem — the `/data` volume and a `/tmp` tmpfs are the only writable paths — and
a process limit, and the application code inside the image is root-owned and read-only.
`ocr-cuda` and the GPU sidecars do not use a read-only root filesystem yet; that needs to
be validated on a GPU host first.

## Secret response

If a secret is committed, revoke or rotate it first. Removing it from a later commit is
not sufficient; rewrite repository history before sharing the repository further.
