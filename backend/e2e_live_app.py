"""Opt-in live browser harness; never imported by the production entrypoint.

Only translation settings are read from the repository .env. OCR, storage and
Q&A settings are isolated so historical local configuration cannot select a
different engine or send unrelated requests.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re

from dotenv import dotenv_values

from app.config import _APP_ENV_KEYS
from app.translate.client import OpenAICompatClient
from app.translate.types import TranslateConfig, TranslateError, TranslateOutputTruncated, TranslateTimeout

ROOT = Path(__file__).resolve().parents[1]
_TRANSLATION_KEYS = frozenset(
    key for key in _APP_ENV_KEYS if key.startswith(("OPENAI_", "TRANSLATE_"))
)
_PROVIDER_PREFIX = re.compile(r"^(OPENAI|TRANSLATE|LLM|OLLAMA|ANTHROPIC|AZURE_OPENAI|OPENROUTER|GEMINI|MISTRAL|GROQ|OCR)_")
_SECRETISH = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$|SECRET", re.IGNORECASE)


def live_environment(base, *, dotenv_path: Path, data_dir: Path) -> dict[str, str]:
    """Keep execution/proxy settings; take only registered translation dotenv keys."""
    env = {
        key: value for key, value in base.items()
        if key not in _APP_ENV_KEYS
        and not _PROVIDER_PREFIX.match(key)
        and not _SECRETISH.search(key)
    }
    values = dotenv_values(dotenv_path, encoding="utf-8-sig") if dotenv_path.is_file() else {}
    for key in _TRANSLATION_KEYS:
        value = base.get(key, values.get(key))  # Match normal environment precedence.
        if value is not None:
            env[key] = value
    env.update({
        "DISABLE_DOTENV": "1", "OCR_TEST_MODE": "0",
        "OCR_ENGINE": "unlimited2", "OCR_DEVICE": base.get("E2E_OCR_DEVICE", "cuda"),
        "OCR_SIDECAR_URL": base.get("OCR_SIDECAR_URL", ""),
        "PRELOAD_MODEL": "1", "RENDER_DPI": "200", "PAGES_PER_CHUNK": "2",
        "DATA_DIR": str(data_dir.resolve()), "FRONTEND_DIR": str(ROOT / "frontend"),
        "ALLOWED_HOSTS": "localhost,127.0.0.1", "PDF_WORKER_MODE": "process",
        "LLM_PROVIDER": "openai-responses", "LLM_OPENAI_API_KEY": "",
        "LLM_LOCAL_OPENAI_BASE_URL": "", "LLM_LOCAL_OPENAI_API_KEY": "",
        "OLLAMA_BASE_URL": "http://127.0.0.1:9",
    })
    return env


def probe_provider(cfg: TranslateConfig) -> tuple[dict, int]:
    """One bounded, public fixed-text request; errors never include provider bodies."""
    client = OpenAICompatClient(replace(cfg, max_retries=0, timeout_s=min(cfg.timeout_s, 30)))
    try:
        text = client.complete(
            "Translate the user text into Korean. Return only the translation.",
            "This is a public synthetic translation test.", max_tokens=cfg.max_output_tokens,
        )
        if not any("가" <= char <= "힣" for char in text):
            return {"status": "blocked-provider", "reason": "no-korean-output"}, 409
        return {"status": "ready"}, 200
    except TranslateError as exc:
        status = "blocked-auth" if "인증 실패" in str(exc) else "blocked-provider"
        reason = (
            "authentication" if status == "blocked-auth"
            else "output-limit" if isinstance(exc, TranslateOutputTruncated)
            else "timeout" if isinstance(exc, TranslateTimeout)
            else "connection" if "연결 실패" in str(exc)
            else "provider-error"
        )
        body = {"status": status, "reason": reason}
        http_status = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
        if http_status:
            body["http_status"] = int(http_status[1])
        return body, 409
    finally:
        client.session.close()


def add_provider_probe(app):
    """Place the harness endpoint before the production frontend catch-all mount."""
    from fastapi.responses import JSONResponse

    @app.post("/e2e/live-provider-check", include_in_schema=False)
    def live_provider_check():
        try:
            cfg = TranslateConfig.from_env()
        except TranslateError:
            return JSONResponse({"status": "blocked-config"}, status_code=409)
        body, status = probe_provider(cfg)
        return JSONResponse(body, status_code=status)

    app.router.routes.insert(0, app.router.routes.pop())


def create_live_app():
    data = os.environ.get("E2E_DATA_DIR")
    if not data:
        raise RuntimeError("E2E_DATA_DIR is required for isolated live browser testing")
    env = live_environment(os.environ, dotenv_path=ROOT / ".env", data_dir=Path(data))
    os.environ.clear()
    os.environ.update(env)
    from app.main import create_app

    app = create_app()
    add_provider_probe(app)
    return app
