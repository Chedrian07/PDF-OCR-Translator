"""Unlimited-OCR-2's pinned SGLang OpenAI streaming API adapter."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image

from .base import (
    EngineCapabilities,
    EngineError,
    JobCanceled,
    OCREngine,
    OutputLimitError,
    RepetitiveOutputError,
)
from .postprocess import (
    BlockStream,
    _split_multi_pages,
    save_results_multi,
    save_results_single,
)
from .repetition import SemanticRepetitionDetector

# dill.loads accepts the standard pickle GLOBAL representation. This references
# the installed wheel's official class, without importing CUDA into the web app.
NGRAM_PROCESSOR = json.dumps(
    {
        "callable": (
            b"csglang.srt.sampling.custom_logit_processor\nDeepseekOCRNoRepeatNGramLogitProcessor\n."
        ).hex()
    }
)


class SGLangUnavailable(EngineError):
    transient = True
    retry_same_page = False


class SGLangPageLimit(OutputLimitError):
    limit_label = "MAX_PAGE_OUTPUT_TOKENS"


def page_budget_reached(ids, limit):
    count = 0
    for token in ids:
        if token == 128827:
            count = 0
        elif token != 1:
            count += 1
            if count >= limit:
                return True
    return False


class SGLangEngine(OCREngine):
    name = "unlimited2"
    device = "cuda"
    dtype_name = "bfloat16"

    def __init__(self, settings):
        self._settings = settings
        self.url = settings.sidecar_url or "http://127.0.0.1:10000"
        self._loaded = False
        self._health = None

    @property
    def loaded(self):
        return self._loaded

    def capabilities(self):
        return EngineCapabilities(
            model_id=self._settings.model_id,
            model_revision=self._settings.model_revision,
            provider="sglang",
            runtime="sglang-0.5.20.dev13+g75b9b06b02",
        )

    def load(self):
        s = self._settings
        try:
            with httpx.Client(
                base_url=self.url, trust_env=False, timeout=s.sidecar_health_timeout_s
            ) as client:
                ready = client.get("/health" if self._loaded else "/health_generate")
                ready.raise_for_status()
                response = client.get("/model_info")
                response.raise_for_status()
                info = response.json()
                runtime = client.get("/get_server_info")
                runtime.raise_for_status()
                runtime_info = runtime.json()
                if runtime_info.get("version") != "0.5.20.dev13+g75b9b06b02":
                    raise EngineError(
                        "SGLang runtime version mismatch; install the pinned official wheel"
                    )
            # The launcher passes the resolved immutable snapshot directory. Never
            # advertise the configured revision for an unrelated running server.
            path = info.get("model_path", "").rstrip("/")
            if (
                not path.endswith("/snapshots/" + s.model_revision)
                or "models--baidu--Unlimited-OCR-2/" not in path
            ):
                raise EngineError(
                    "SGLang model identity mismatch; start services/unlimited_ocr2/launch.py"
                )
            self._loaded = True
            self._health = {"status": "ready", "model_revision": s.model_revision}
        except (httpx.HTTPError, ValueError) as exc:
            self._loaded = False
            self._health = {"status": "unavailable"}
            raise SGLangUnavailable("Unlimited-OCR-2 SGLang is not ready") from exc

    def provider_health(self):
        try:
            self.load()
        except EngineError as exc:
            self._loaded = False
            self._health = {"status": "unavailable", "error": str(exc)}
        return self._health

    def wait_until_ready(self, cancel, on_wait=None):
        deadline = time.monotonic() + self._settings.sidecar_model_wait_s
        while not cancel.is_set():
            try:
                self.load()
                return
            except SGLangUnavailable:
                if time.monotonic() >= deadline:
                    raise
                if on_wait:
                    on_wait("Unlimited-OCR-2 SGLang 모델 준비 대기")
                cancel.wait(1)
        raise JobCanceled()

    def payload(self, paths, *, single, rid):
        content = []
        for path in paths:
            mime = (
                "image/jpeg"
                if path.suffix.lower() in (".jpg", ".jpeg")
                else "image/png"
            )
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": "<PAGE><image>" * len(paths) + "Multi page parsing.",
            }
        )
        return {
            "model": "Unlimited-OCR-2",
            "rid": rid,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self._settings.max_length,
            "skip_special_tokens": False,
            "return_output_ids_in_sglext": True,
            "images_config": {"crop_mode": single},
            "custom_logit_processor": NGRAM_PROCESSOR,
            "custom_params": {
                "ngram_size": 35,
                "window_size": 128 if single else 1024,
                "page_token_id": 128827,
                "eos_token_id": 1,
                "max_page_tokens": self._settings.max_page_output_tokens or 20000,
            },
            "stream": True,
        }

    async def _stream(self, payload, sink, cancel, repetition):
        s = self._settings
        timeout = httpx.Timeout(
            s.sidecar_read_timeout_s, connect=s.sidecar_connect_timeout_s
        )
        async with httpx.AsyncClient(
            base_url=self.url, trust_env=False, timeout=timeout
        ) as client:

            async def consume():
                chunks, finish, seen_done, size = [], None, False, 0
                page_capped = False
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    # Bound raw bytes and individual SSE frames before JSON parsing.
                    buffer = b""
                    async for block in response.aiter_bytes():
                        size += len(block)
                        if size > s.sidecar_max_response_mb * 1024 * 1024:
                            raise EngineError(
                                "SGLang response exceeds OCR_SIDECAR_MAX_RESPONSE_MB"
                            )
                        buffer += block
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                seen_done = True
                                continue
                            event = json.loads(data)
                            ids = (event.get("sglext") or {}).get("output_ids") or []
                            cap = s.max_page_output_tokens or 20000
                            if any(page_budget_reached(row, cap) for row in ids):
                                page_capped = True
                            if "error" in event:
                                raise EngineError("SGLang generation failed")
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {}).get("content") or ""
                                if delta:
                                    chunks.append(delta)
                                    if repetition.feed(delta):
                                        raise RepetitiveOutputError(repetition.message)
                                    sink.on_text(delta)
                                finish = choice.get("finish_reason") or finish
                if not seen_done or finish not in ("stop", "length"):
                    raise EngineError(
                        "SGLang stream ended without a valid finish reason"
                    )
                return "".join(chunks), "page_limit" if page_capped else finish

            task = asyncio.create_task(consume())
            deadline = time.monotonic() + s.sidecar_read_timeout_s
            try:
                while not task.done():
                    if cancel.is_set():
                        raise JobCanceled()
                    if time.monotonic() >= deadline:
                        raise SGLangUnavailable("SGLang OCR timeout")
                    await asyncio.wait({task}, timeout=0.1)
                return await task
            finally:
                # Cancel closes the streaming socket; the wheel's document-window
                # finally releases its internal reuse session and prefetched windows.
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                try:
                    await client.post(
                        "/abort_request", json={"rid": payload["rid"]}, timeout=5
                    )
                except httpx.HTTPError:
                    pass  # Socket-disconnect cleanup is still owned by SGLang.

    def _run(self, paths, out_dir, sink, cancel, *, single):
        if cancel.is_set():
            raise JobCanceled()
        self.load()
        sink = BlockStream(sink)
        repetition = SemanticRepetitionDetector(
            max_page_chars=self._settings.max_page_output_chars,
            max_page_tokens=None,
            expected_pages=len(paths),
        )
        payload = self.payload(paths, single=single, rid=uuid.uuid4().hex)
        try:
            raw, finish = asyncio.run(self._stream(payload, sink, cancel, repetition))
        except (httpx.HTTPError, ValueError) as exc:
            raise SGLangUnavailable("SGLang OCR connection/stream failed") from exc
        if cancel.is_set():
            raise JobCanceled()
        sink.end()
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            output = (
                save_results_single(
                    "\n".join(_split_multi_pages(raw)), images[0], str(out_dir)
                )
                if single
                else save_results_multi(raw, images, str(out_dir))
            )
        finally:
            for image in images:
                image.close()
        if finish == "page_limit":
            raise SGLangPageLimit("SGLang per-page output cap reached")
        if finish == "length":
            raise OutputLimitError(
                "SGLang output token limit reached",
                partial_output=None if single else output,
            )
        return output

    def run_multi(
        self, image_paths: list[Path], out_dir: Path, sink, cancel: threading.Event
    ):
        return self._run(image_paths, out_dir, sink, cancel, single=False)

    def run_single(
        self, image_path: Path, out_dir: Path, sink, cancel: threading.Event
    ):
        return self._run([image_path], out_dir, sink, cancel, single=True)
