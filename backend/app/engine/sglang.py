"""Unlimited-OCR-2's pinned SGLang OpenAI streaming API adapter."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import shutil
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .base import (
    EngineCapabilities,
    EngineError,
    JobCanceled,
    NullSink,
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
from .region_rescue import RegionOutput, rescue_page
from .repetition import SemanticRepetitionDetector
from .image_input import normalize_model_image

# dill.loads accepts the standard pickle GLOBAL representation. This references
# the installed wheel's official class, without importing CUDA into the web app.
NGRAM_PROCESSOR = json.dumps(
    {
        "callable": (
            b"csglang.srt.sampling.custom_logit_processor\nDeepseekOCRNoRepeatNGramLogitProcessor\n."
        ).hex()
    }
)


# /api/health가 sidecar 확인을 기다리는 상한(초) — 넘기면 직전 결과를 낸다 (provider_health)
HEALTH_PROBE_WAIT_S = 1.0
# 영역 재처리(run_single_rescue) 예산 — 페이지 한 장당 영역 호출 수·시간 상한. 실측 116쪽 논문의
# 최악 페이지가 29회·114초였다(24회 상한에서는 줄 6개를 비운 채 0.387로 게이트에 걸렸다).
RESCUE_MAX_CALLS = 40
RESCUE_MAX_SECONDS = 240.0
# 영역 상한은 높이에 비례해 줄인다(+50% 여유) — 상한 루프가 작은 영역에서도 페이지 상한(6,144토큰
# ≈25초)까지 돌지 않게. 한 줄 영역도 이 아래로는 줄이지 않는다.
REGION_MIN_TOKENS = 768
REGION_MIN_CHARS = 2_048


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
        # 한 번이라도 준비 확인에 성공했는지 — 그 전의 미준비는 정상 기동 과정(starting),
        # 그 뒤의 미준비는 sidecar 중단·재시작(unavailable)이다 (provider_health)
        self._ever_ready = False
        # /api/health용 sidecar 확인 — 한 번에 하나만 돌고(single-flight) 마지막 결과를 남긴다
        self._probe_lock = threading.Lock()
        self._probe: concurrent.futures.Future | None = None
        self._last_probe: dict | None = None

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

    def _mark_unavailable(self):
        self._loaded = False
        self._health = {"status": "unavailable"}

    def load(self):
        s = self._settings
        try:
            with httpx.Client(
                base_url=self.url, trust_env=False, timeout=s.sidecar_health_timeout_s
            ) as client:
                # In the pinned wheel both /health variants run an inference and
                # wait at least one second. /ready checks startup, pause, shutdown
                # and unhealthy state without scheduling a synthetic generation.
                # Keep the live scheduler query and identity checks on every call.
                ready = client.get("/ready")
                ready.raise_for_status()
                response = client.get("/model_info")
                response.raise_for_status()
                info = response.json()
                runtime = client.get("/server_info")
                runtime.raise_for_status()
                runtime_info = runtime.json()
                if not isinstance(info, dict) or not isinstance(runtime_info, dict):
                    raise ValueError("SGLang identity metadata must be objects")
                if runtime_info.get("version") != "0.5.20.dev13+g75b9b06b02":
                    raise EngineError(
                        "SGLang runtime version mismatch; install the pinned official wheel"
                    )
            # The launcher passes the resolved immutable snapshot directory. Never
            # advertise the configured revision for an unrelated running server.
            path = info.get("model_path", "")
            if not isinstance(path, str):
                raise ValueError("SGLang model path must be a string")
            path = path.rstrip("/")
            if (
                not path.endswith("/snapshots/" + s.model_revision)
                or "models--baidu--Unlimited-OCR-2/" not in path
            ):
                raise EngineError(
                    "SGLang model identity mismatch; start services/unlimited_ocr2/launch.py"
                )
            if (
                runtime_info.get("grammar_backend") != "none"
                or runtime_info.get("disable_outlines_disk_cache") is not True
            ):
                raise EngineError(
                    "SGLang optional grammar/cache must be disabled; "
                    "start services/unlimited_ocr2/launch.py"
                )
            self._loaded = True
            self._ever_ready = True
            self._health = {"status": "ready", "model_revision": s.model_revision}
        except (httpx.HTTPError, ValueError) as exc:
            self._mark_unavailable()
            raise SGLangUnavailable("Unlimited-OCR-2 SGLang is not ready") from exc
        except EngineError:
            self._mark_unavailable()
            raise

    def provider_health(self):
        # /api/health는 sidecar 확인을 HEALTH_PROBE_WAIT_S까지만 기다린다. 정지된 sidecar
        # 컨테이너는 이름 조회(getaddrinfo)가 10초 걸려 실패하는데 이 시간은 httpx 타임아웃
        # 밖이라, 예전에는 health 응답이 매번 10초 걸렸다(컨테이너 HEALTHCHECK 타임아웃 5초도
        # 넘는다). 확인은 데몬 스레드 하나에서만 돌고 — 동시 요청·다음 폴링은 진행 중인 확인을
        # 함께 기다려 멈춘 확인 위에 스레드를 쌓지 않는다 — 늦으면 실패로 단정하지 않고 직전
        # 결과를 낸다(바쁜 sidecar의 느린 응답이 '연결 안 됨' 오경보가 되지 않게). 그 대가로
        # 상태는 최대 확인 한 번만큼 늦게 바뀐다.
        with self._probe_lock:
            probe = self._probe
            if probe is None or probe.done():
                probe = self._probe = concurrent.futures.Future()
                threading.Thread(
                    target=self._run_probe, args=(probe,), name="sglang-health-probe",
                    daemon=True,  # 이름 조회에 멈춘 확인이 프로세스 종료를 붙잡지 않게
                ).start()
        try:
            return probe.result(timeout=HEALTH_PROBE_WAIT_S)
        except concurrent.futures.TimeoutError:
            return self._previous_health()

    def _previous_health(self):
        """확인이 늦을 때 낼 직전 상태 — 끝난 확인이 아직 없으면 마지막 load() 결과로 만든다."""
        if self._last_probe is not None:
            return self._last_probe
        if self._loaded:
            return {"status": "ready", "model_revision": self._settings.model_revision}
        return {
            "status": "unavailable" if self._ever_ready else "starting",
            "error": "Unlimited-OCR-2 SGLang readiness check is still running",
        }

    def _run_probe(self, probe):
        try:
            result = self._probe_now()
        except Exception as exc:  # noqa: BLE001 — api.health가 받아 error 상태로 낸다
            probe.set_exception(exc)
        else:
            self._last_probe = result
            probe.set_result(result)

    def _probe_now(self):
        try:
            self.load()
        except EngineError as exc:
            self._loaded = False
            # 한 번도 준비된 적 없는 일시적 미준비는 모델 다운로드·CUDA 그래프 캡처 중인
            # 정상 기동이다 — 웹은 '로딩 중'으로 두고(대기 상한을 넘기면 프리로드가
            # model_load_error로 올린다), 준비된 뒤 끊긴 sidecar만 '연결 안 됨'으로 띄운다.
            # 둘을 구분하지 않으면 웹은 기동 중을 장애로 띄우거나, 장애를 '로딩 중'으로 숨긴다.
            starting = exc.transient and not self._ever_ready
            self._health = {
                "status": "starting" if starting else "unavailable",
                "error": str(exc),
            }
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

    def payload(self, paths, *, single, rid, max_page_tokens=None):
        content = []
        for path in paths:
            mime = (
                "image/jpeg"
                if path.suffix.lower() in (".jpg", ".jpeg")
                else "image/png"
            )
            with Image.open(path) as source:
                prepared = normalize_model_image(source)
                try:
                    if prepared is source:
                        encoded = path.read_bytes()
                    else:
                        buffer = BytesIO()
                        prepared.save(buffer, format="PNG", compress_level=1)
                        encoded = buffer.getvalue()
                        mime = "image/png"
                finally:
                    if prepared is not source:
                        prepared.close()
            data = base64.b64encode(encoded).decode("ascii")
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
                "max_page_tokens": max_page_tokens
                or self._settings.max_page_output_tokens
                or 20000,
            },
            "stream": True,
        }

    async def _stream(self, payload, sink, cancel, repetition, page_cap=None):
        s = self._settings
        # 서버 page guard(custom_params.max_page_tokens)와 같은 상한으로 잘림을 판정한다 —
        # 영역 재처리가 상한을 줄였는데 여기서 전역 상한을 보면, 서버가 끊은 영역이 정상
        # 종료(stop)로 보여 잘린 출력이 조용히 채택된다.
        cap = page_cap or s.max_page_output_tokens or 20000
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
                            if any(page_budget_reached(row, cap) for row in ids):
                                page_capped = True
                            if "error" in event:
                                raise EngineError("SGLang generation failed")
                            for choice in event.get("choices", []):
                                delta = choice.get("delta", {}).get("content") or ""
                                if delta:
                                    chunks.append(delta)
                                    if repetition.feed(delta):
                                        error = RepetitiveOutputError(repetition.message)
                                        # 영역 재처리는 루프 직전까지 끝낸 블록과 루프 블록의
                                        # 좌표를 이어 읽기에 쓴다(문서 내용이라 메시지엔 없다)
                                        error.partial_text = "".join(chunks)
                                        raise error
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
        except SGLangUnavailable:
            self._mark_unavailable()
            raise
        except (httpx.HTTPError, ValueError) as exc:
            self._mark_unavailable()
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

    supports_region_rescue = True

    def run_single_rescue(
        self, model_image, page_image, out_dir, sink, cancel, on_region=None, line_text=None
    ):
        """반복 생성으로 실패한 페이지를 영역 단위로 다시 읽는다 — 알고리즘은 region_rescue 참조.

        영역 호출은 라이브로 흘리지 않는다: 크롭 좌표의 박스가 라이브 레이아웃에 잘못 그려진다.
        다 읽은 뒤 페이지 좌표로 합친 출력을 한 번에 sink에 넣는다."""
        if cancel.is_set():
            raise JobCanceled()
        self.load()
        s = self._settings
        page_cap = s.max_page_output_tokens or 20000
        work = Path(out_dir) / "regions"
        work.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with Image.open(model_image) as source:
            image = source.convert("RGB")
        calls = 0

        def read_region(crop, rect, _pad):
            nonlocal calls
            if cancel.is_set():
                raise JobCanceled()
            share = min(1.0, rect.height / image.height)
            cap = min(page_cap, max(REGION_MIN_TOKENS, int(page_cap * 1.5 * share)))
            chars = s.max_page_output_chars
            if chars is not None:
                chars = min(chars, max(REGION_MIN_CHARS, int(chars * 1.5 * share)))
            path = work / f"region_{rect.x0}_{rect.y0}_{rect.x1}_{rect.y1}.png"
            crop.save(path, format="PNG", compress_level=1)
            payload = self.payload(
                [path], single=True, rid=uuid.uuid4().hex, max_page_tokens=cap
            )
            repetition = SemanticRepetitionDetector(
                max_page_chars=chars, max_page_tokens=None, expected_pages=1
            )
            try:
                raw, finish = asyncio.run(
                    self._stream(payload, NullSink(), cancel, repetition, page_cap=cap)
                )
                output = RegionOutput(
                    finish not in ("page_limit", "length"), raw,
                    {"page_limit": "cap", "length": "length"}.get(finish, "ok"),
                )
            except RepetitiveOutputError as exc:
                output = RegionOutput(False, getattr(exc, "partial_text", ""), "loop")
            except SGLangUnavailable:
                self._mark_unavailable()
                raise
            except (httpx.HTTPError, ValueError) as exc:
                self._mark_unavailable()
                raise SGLangUnavailable("SGLang OCR connection/stream failed") from exc
            finally:
                path.unlink(missing_ok=True)
            if cancel.is_set():
                raise JobCanceled()
            calls += 1
            if on_region is not None:
                on_region(calls)
            return output

        try:
            result = rescue_page(
                image, read_region, max_calls=RESCUE_MAX_CALLS,
                deadline=lambda: time.monotonic() - started > RESCUE_MAX_SECONDS,
                line_text=line_text,
            )
        finally:
            image.close()
            shutil.rmtree(work, ignore_errors=True)
        body = "\n".join(_split_multi_pages(result.raw))
        with Image.open(page_image) as page:
            stored = page.convert("RGB")
        try:
            result.markdown = save_results_single(body, stored, str(out_dir))
        finally:
            stored.close()
        stream = BlockStream(sink)
        stream.on_text(result.raw)
        stream.end()
        return result
