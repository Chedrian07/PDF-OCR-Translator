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
- Keep `.env`, private keys, certificates, job data, and model caches outside Git and the
  Docker build context. The supplied ignore files enforce the common cases.
- The application is unauthenticated. The Compose default binds ports to `0.0.0.0` with
  `ALLOWED_HOSTS=*`, assuming a trusted network (VPN/Tailscale, firewalled LAN). Put
  authentication and TLS at a reverse proxy before exposing beyond that; set
  `BIND_HOST=127.0.0.1` in `.env` to restore loopback-only binding.
- Keep `ALLOWED_HOSTS` narrow when clients reach the service via a stable hostname/IP.
  Do not rely on the wildcard default on an untrusted network.

## Secret response

If a secret is committed, revoke or rotate it first. Removing it from a later commit is
not sufficient; rewrite repository history before sharing the repository further.
