# Changelog

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
- Identify the release version in the API schema.
- Build and smoke-test CPU Docker images for Linux amd64 and arm64 before publishing to GHCR. Publishing requires successful CI on the exact release commit.

GPU engines remain available through the source Compose profiles; the published container image uses CPU dependencies. Browser and translation E2E checks use local mock LLM providers. Real GPU model inference and external translation providers require deployment-specific validation.
