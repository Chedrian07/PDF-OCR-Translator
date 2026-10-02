"""OpenAI 호환 클라이언트 — Chat Completions·Responses API 양쪽 지원.

로컬 서버(vLLM·Ollama·llama.cpp 등)는 어느 한쪽만 지원하는 경우가 많아 api_mode
"auto"는 첫 호출에 responses를 시도하고 404/405/501이면 chat으로 영구 래치한다.
requests만 쓰며(런타임 기존 의존성), 실제 전송은 _post 한 메서드로 모아 테스트가
그것만 몽키패치하도록 한다.

잘림(truncation) 처리: chat `finish_reason=="length"` / responses
`status=="incomplete"`를 감지하면 같은 요청을 **max_tokens 2배로 1회 재시도**한다
(2026-07-08 합의 정책 ②). thinking 모델은 reasoning 토큰이 같은 예산에서 차감되어
effort 테이블(types.REASONING_MAX_TOKENS)로도 드물게 잘릴 수 있다.
재시도 뒤에도 잘렸으면 **잘린 출력을 반환하지 않고** TranslateOutputTruncated(유닛
단위 거부)를 던진다 — 종전처럼 잘린 출력을 돌려주면 플레이스홀더 없는 산문은 래더에
들어가지도 않고 문단 끝이 빠진 채 채택·캐시됐다(probe:MLX-02). 반복 루프·과도한
길이처럼 재시도해도 같은 결과가 뻔하면 2배 재시도 자체를 생략한다(probe:MLX-05).

스트리밍(TRANSLATE_STREAM, auto = chat 모드): SSE로 받으면 read timeout이 '토큰 사이
정지 시간'이 되어 느리지만 진행 중인 생성은 끊기지 않는다. HTTP 왕복은 헬퍼 스레드에서
돌고 호출 스레드는 0.1초마다 취소를 본다 — 취소되면 즉시 반환하고 소켓을 끊어, 서버
(mlx_lm 등)가 다음 토큰 쓰기에서 끊김을 보고 생성을 멈춘다. 비스트리밍 요청은 서버가
끝까지 생성한 뒤에야 끊김을 알 수 있다(고아 생성, probe:MLX-04). 응답 본문은
TRANSLATE_MAX_RESPONSE_MB를 넘으면 읽기를 멈춘다(라이브러리 버전과 무관한 상한) —
선언된 Content-Length를 먼저 보고, 본문은 비스트리밍·SSE 모두 고정 크기 조각(압축이면
풀린 크기)으로 읽어 청크 전송이 아닌 HTTP/1.0 스트림(mlx_lm.server)에서도 점진적으로 건다.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import ReadTimeoutError

from .masking import is_degenerate_repetition
from .types import (
    TranslateAPIError,
    TranslateCancelled,
    TranslateConfig,
    TranslateEmptyOutput,
    TranslateOutputTruncated,
    TranslateTimeout,
    TranslateUnitRejected,
)

logger = logging.getLogger(__name__)

# 재시도 대상 상태코드 (일시적 오류)
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})
# auto 모드에서 responses → chat 폴백을 유발하는 상태코드
_FALLBACK = frozenset({404, 405, 501})
# 유닛 하나가 결정적으로 거부되는 상태코드 (입력이 너무 길거나 서버가 처리 불가).
# 인증(401/403)·엔드포인트(404)처럼 전역 원인인 코드는 여기 넣지 않는다.
_UNIT_REJECTED = frozenset({400, 413, 422})

# 접속 단계 상한 — 응답 생성이 아무리 길어도 TCP 연결 자체는 10초 안에 되거나 안 된다.
# 스칼라 timeout은 connect에도 read와 같은 값(기본 180s)을 걸어, 엔드포인트가 죽으면
# 워커 하나가 시도당 수십 초(리눅스 SYN 재시도 상한 ~127초)를 붙잡혔다. 재시도 4회 ×
# 백오프까지 더하면 잡 하나가 오류를 알리는 데 500초를 넘긴다 — 동시성을 올릴수록
# 그만큼 워커가 통째로 묶인다. read 타임아웃은 그대로라 정상 응답에는 영향이 없다.
_CONNECT_TIMEOUT_S = 10.0

# 재시도 대기 상한 — 지수 백오프와 Retry-After 헤더 양쪽에 같은 상한을 건다.
_MAX_BACKOFF_S = 30.0

# 잘린 출력이 '쓸 수 없을 만큼 길다'고 보는 최소 길이(문자) — _hopeless_truncation 참조.
_OVERLONG_MIN_CHARS = 2000

# 응답 정지(ReadTimeout) 재시도 상한 — 재시도는 같은 긴 생성의 반복이고 비스트리밍
# 서버는 끊긴 요청도 끝까지 생성한다(실측: 4회 중복, 고아 32,768토큰). 최대 1회.
_MAX_TIMEOUT_RETRIES = 1
# 헬퍼 스레드의 HTTP 왕복을 기다리는 동안 취소를 관측하는 주기(초)
_CANCEL_POLL_S = 0.1
# 본문 수신 조각 크기 — 비스트리밍·SSE 공통. 상한 검사가 이 단위로 점진적으로 걸린다.
_BODY_CHUNK = 64 * 1024
# [DONE] 뒤 본문 끝(청크 종료 표시) 소진 상한 — keep-alive 연결을 풀에 돌려주려고 짧게만
# 읽는다. [DONE] 뒤에도 스트림을 열어 두는 서버에 막히지 않게 시간·바이트를 묶는다.
_DRAIN_TIMEOUT_S = 1.0
_DRAIN_MAX_BYTES = 64 * 1024
# auto 스트리밍을 서버가 거부했다고 보는 상태코드 — 같은 요청을 비스트리밍으로 1회 시도
_STREAM_REJECTED = frozenset({400, 415, 422})

# thinking 출력 표기 — 여는 태그는 선두에서만 의미가 있다(템플릿이 프롬프트 끝에
# `<think>`를 미리 넣는 Qwen3 계열은 content가 여는 태그 없이 '…</think>답'으로 온다).
_THINK_OPEN_RE = re.compile(r"^\s*<think>")
_THINK_CLOSE = "</think>"
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)```\s*$", re.DOTALL)


class _NeedsFallback(Exception):
    """내부용 — auto 모드에서 responses가 미지원일 때 chat 폴백을 신호."""


class _RequestCancelled(TranslateCancelled):
    """내부용 — 요청 전·진행 중에 cancel/abort를 관찰했다(공개 계약은 TranslateCancelled)."""


class _ModeFlight:
    """auto 최초 협상의 결과/오류를 동시 호출자에게 한 번만 공개한다."""

    __slots__ = ("event", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: tuple[type[BaseException], tuple] | None = None


def _normalize_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    # 빈 포트 교정: "https://host:/v1" → "https://host/v1" (사용자 .env 실측)
    url = re.sub(r"^(https?://[^/]+?):(?=/|$)", r"\1", url)
    # origin만 적은 일반적인 OPENAI_BASE_URL도 수용한다. 기존의 /v1 또는
    # 공급자별 커스텀 경로는 절대 바꾸지 않고, path가 정말 비었을 때만 /v1을
    # 붙인다. query/fragment는 그대로 보존한다.
    parts = urlsplit(url)
    if parts.scheme in ("http", "https") and parts.netloc and not parts.path:
        url = urlunsplit((parts.scheme, parts.netloc, "/v1", parts.query, parts.fragment))
    return url


def _endpoint_url(base_url: str, path: str) -> str:
    """base의 query/fragment 앞에 API path를 붙인다 (문자열 이어붙이기 금지)."""
    parts = urlsplit(base_url)
    endpoint_path = f"{parts.path.rstrip('/')}/{path.lstrip('/')}"
    return urlunsplit((parts.scheme, parts.netloc, endpoint_path, parts.query, parts.fragment))


class OpenAICompatClient:
    def __init__(
        self,
        cfg: TranslateConfig,
        *,
        request_semaphore: threading.Semaphore | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.base_url = _normalize_base_url(cfg.base_url)
        self.session = requests.Session()
        self._request_semaphore = request_semaphore
        self._cancel_check = cancel_check
        self._latched: str | None = None  # auto 확정 모드 (인스턴스 수명 동안 유지)
        self._mode_lock = threading.Lock()  # _latched/_mode_flight 상태만 짧게 보호
        self._mode_flight: _ModeFlight | None = None
        self.api_mode_used = "" if cfg.api_mode == "auto" else cfg.api_mode
        # auto 스트리밍 래치 — None=미확인, True=스트림 성공, False=서버가 거부해 비스트리밍
        self._stream_ok: bool | None = None
        # Responses store:false 래치 — False면 서버가 store 파라미터를 거부해 빼고 보낸다
        self._store_ok: bool | None = None

    def set_cancel_check(self, check: Callable[[], bool] | None) -> None:
        """엔진의 cancel+abort predicate를 주입한다 (사용자 제공 client와 호환용 선택 API)."""
        self._cancel_check = check

    def _raise_if_cancelled(self) -> None:
        check = self._cancel_check
        if check is not None and check():
            raise _RequestCancelled("번역 요청이 취소되었습니다")

    def _wait_or_cancel(self, seconds: float) -> None:
        """재시도 backoff를 잘게 기다려 취소 뒤 새 요청이 나가지 않게 한다."""
        wait = max(0.0, float(seconds))
        if self._cancel_check is None:
            time.sleep(wait)
            return
        deadline = time.monotonic() + wait
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    @staticmethod
    def _raise_flight_error(error: tuple[type[BaseException], tuple]) -> None:
        error_type, error_args = error
        try:
            cloned = error_type(*error_args)
        except TypeError as clone_error:
            raise TranslateAPIError("번역 API 초기 모드 협상에 실패했습니다") from clone_error
        raise cloned

    # ── 전송 심(seam) — 테스트는 이 메서드만 몽키패치 ──────────────────
    def _post(self, path: str, payload: dict) -> tuple[int, dict | str, dict]:
        """(status, body(json이면 dict 아니면 str), headers) 반환.

        payload["stream"]이 참이면 SSE를 받아 비스트리밍과 같은 모양의 dict로 조립한다
        (choices[0].message.content·finish_reason·usage) — 파서와 테스트가 한 모양만 본다.
        """
        url = _endpoint_url(self.base_url, path)
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        semaphore = self._request_semaphore
        acquired = False
        if semaphore is not None:
            # 여러 잡의 전역 슬롯을 기다리는 동안 cancel/abort를 확인한다. 무기한
            # acquire 뒤 곧장 전송하면 취소된 잡이 슬롯이 풀린 순간 유료 요청을 보낸다.
            while not semaphore.acquire(timeout=0.1):
                self._raise_if_cancelled()
            acquired = True
        try:
            # acquire 직후 cancel과의 마지막 경쟁창도 닫고 나서만 네트워크로 나간다.
            self._raise_if_cancelled()
            return self._exchange(url, payload, headers)
        finally:
            if semaphore is not None and acquired:
                semaphore.release()

    def _exchange(self, url: str, payload: dict, headers: dict) -> tuple[int, dict | str, dict]:
        """HTTP 왕복 1회. 취소 관측자가 있으면 헬퍼 스레드에서 돌리고 0.1초마다 취소를 본다.

        종전에는 session.post가 응답 끝까지(최대 TRANSLATE_TIMEOUT_S=180s) 워커를 붙잡아
        취소·잡 삭제가 그만큼 늦었고 번역 슬롯도 점유됐다(concurrency-10). 취소되면
        결과를 버리고 즉시 _RequestCancelled를 던지며, 소켓을 끊어 서버 생성도 멈춘다.
        """
        check = self._cancel_check
        abort = threading.Event()
        box: dict = {}
        if check is None:
            return self._transfer(url, payload, headers, abort, box)
        done = threading.Event()

        def _run() -> None:
            try:
                box["result"] = self._transfer(url, payload, headers, abort, box)
            except BaseException as exc:  # noqa: BLE001 — 호출 스레드로 전달
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True, name="translate-http").start()
        while not done.wait(_CANCEL_POLL_S):
            if check():
                abort.set()
                _shutdown_socket(box.get("resp"))
                raise _RequestCancelled("번역 요청이 취소되었습니다")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _transfer(
        self, url: str, payload: dict, headers: dict, abort: threading.Event, box: dict,
    ) -> tuple[int, dict | str, dict]:
        """실제 POST + 본문 수신(스트리밍이면 SSE 조립). 응답 크기 상한을 강제한다."""
        resp = self.session.post(
            url, json=payload, headers=headers, stream=True,
            # (connect, read) — min은 timeout_s를 10초 미만으로 줄인 설정을 존중한다.
            # stream=True라 read는 '바이트 사이 정지 시간' 상한이다.
            timeout=(min(_CONNECT_TIMEOUT_S, self.cfg.timeout_s), self.cfg.timeout_s),
        )
        box["resp"] = resp
        try:
            if abort.is_set():  # 헤더가 오기 전에 취소됐다 — 서버에 끊김을 알린다
                raise _RequestCancelled("번역 요청이 취소되었습니다")
            status = resp.status_code
            hdrs = dict(resp.headers)
            ctype = _header(hdrs, "Content-Type").lower()
            if status == 200 and payload.get("stream") and "text/event-stream" in ctype:
                return status, self._read_sse(resp, hdrs, abort), hdrs
            return status, _decode_body(self._read_body(resp, hdrs, abort)), hdrs
        except requests.exceptions.ConnectionError as exc:
            # stream=True 본문 수신 중 read timeout은 requests가 ConnectionError로 감싼다 —
            # 응답 정지와 연결 끊김을 구분해야 재시도 정책·안내가 맞는다.
            if _is_read_timeout(exc):
                raise requests.exceptions.ReadTimeout(str(exc)) from exc
            raise
        finally:
            resp.close()

    def _cap_bytes(self) -> int:
        return self.cfg.max_response_mb * 1024 * 1024

    def _too_large(self) -> TranslateAPIError:
        return TranslateAPIError(
            f"번역 API 응답이 상한({self.cfg.max_response_mb}MB)을 넘어 읽기를 중단했습니다 — "
            "게이트웨이 이상이 아니라면 TRANSLATE_MAX_RESPONSE_MB를 올리세요"
        )

    def _check_declared(self, hdrs: dict) -> None:
        """선언된 Content-Length가 상한을 넘으면 본문을 읽기 전에 거절한다."""
        declared = _header(hdrs, "Content-Length")
        if declared.isdigit() and int(declared) > self._cap_bytes():
            raise self._too_large()

    def _read_body(self, resp, hdrs: dict, abort: threading.Event) -> bytes:
        cap = self._cap_bytes()
        self._check_declared(hdrs)
        buf = bytearray()
        for chunk in resp.iter_content(_BODY_CHUNK):
            if abort.is_set():
                raise _RequestCancelled("번역 요청이 취소되었습니다")
            buf += chunk
            if len(buf) > cap:
                raise self._too_large()
        return bytes(buf)

    def _read_sse(self, resp, hdrs: dict, abort: threading.Event) -> dict:
        """SSE(chat.completion.chunk) 스트림을 비스트리밍 응답 모양의 dict로 조립한다.

        고정 크기(_BODY_CHUNK)로 읽는다. 종전 iter_content(chunk_size=None)은 청크 전송이
        아닌 응답(HTTP/1.0·Connection: close — mlx_lm.server 형식)을 EOF까지 한 번에 읽고
        urllib3의 압축 해제 상한도 꺼, 끝없이 보내는 서버에는 상한 검사가 본문 전체를
        메모리에 올린 뒤에야 돌았다(translate-1). 파서는 [DONE] 전에는 부분 출력을 쓰지
        않으므로 조각이 묶여 와도 결과·지연은 같다.
        """
        cap = self._cap_bytes()
        self._check_declared(hdrs)
        acc = _StreamAccumulator()
        lines = _LineBuffer()
        received = 0
        chunks = resp.iter_content(_BODY_CHUNK)
        for chunk in chunks:
            if abort.is_set():
                raise _RequestCancelled("번역 요청이 취소되었습니다")
            if not chunk:
                continue
            received += len(chunk)
            if received > cap:
                raise self._too_large()
            for line in lines.feed(chunk):
                if acc.feed(line):
                    _drain_after_done(resp, chunks, abort)
                    return acc.body()
        acc.feed(lines.rest())
        acc.feed(b"")  # 마지막 이벤트 경계
        if not acc.complete:
            raise requests.exceptions.ChunkedEncodingError(
                "번역 API 스트림이 완료 신호 없이 끊겼습니다"
            )
        return acc.body()

    # ── 공개 API ───────────────────────────────────────────────────
    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """번역문을 반환한다. 잘림이 남으면 TranslateOutputTruncated(유닛 단위 거부).

        잘린 출력은 어떤 경로로도 반환하지 않는다 — 호출자가 캐시·게시하지 않게.
        """
        text, truncated = self._complete_once(system, user, max_tokens)
        if not truncated:
            return text  # 빈 응답은 _parse가 TranslateEmptyOutput으로 이미 raise
        # 잘림(finish_reason=length / responses incomplete) — 예산 2배로 1회 재시도.
        if self.cfg.max_tokens_param == "none":
            # max_tokens를 안 보내는 설정이면 재시도는 같은 요청의 반복이다. mlx_lm은
            # 이때 서버 기본 --max-tokens 512에서 자른다(probe:MLX-02).
            logger.warning(
                "번역 API 출력 잘림 — TRANSLATE_MAX_TOKENS_PARAM=none이라 서버 기본 상한에서"
                " 잘렸고 재시도할 수 없습니다"
            )
            raise self._truncated(text, None, "max_tokens 미전송 — 서버 기본 상한")
        hopeless = _hopeless_truncation(text, user)
        if hopeless:
            # 온도 0 greedy 루프는 2배 예산도 끝까지 태운다(실측 8192→16384 연속 루프).
            logger.warning("번역 API 출력 잘림 — %s, 2배 재시도 생략", hopeless)
            raise self._truncated(text, max_tokens, hopeless)
        logger.warning("번역 API 출력 잘림 — max_tokens %d→%d로 1회 재시도", max_tokens, max_tokens * 2)
        try:
            retry_text, retry_truncated = self._complete_once(system, user, max_tokens * 2)
        except TranslateUnitRejected as e:
            # 2배 예산이 서버 상한을 넘어 400 등 — 그 유닛의 원인은 여전히 잘림이다.
            # 연결 실패·5xx 같은 전역 오류는 그대로 전파한다(엔드포인트 문제).
            raise self._truncated(text, max_tokens, f"2배 재시도 거부: {e}") from e
        if not retry_truncated:
            return retry_text
        why = _hopeless_truncation(retry_text, user) or "2배 예산에서도 잘림"
        raise self._truncated(retry_text or text, max_tokens * 2, why)

    def _truncated(self, text: str, budget: int | None, why: str) -> TranslateOutputTruncated:
        if not text:
            # 본문이 한 글자도 없다 = thinking이 예산을 다 썼다 → 서버측 안내
            return TranslateOutputTruncated(_exhausted_message(self.cfg))
        where = f"max_tokens({budget})" if budget else "서버 상한"
        return TranslateOutputTruncated(
            f"번역 API 출력이 {where}에서 잘렸습니다({why}) — 잘린 번역은 쓰지 않습니다"
        )

    def _complete_once(self, system: str, user: str, max_tokens: int) -> tuple[str, bool]:
        """1회 완성 시도 — (텍스트, 잘림 여부) 반환. auto 모드 폴백/래치 담당."""
        if self.cfg.api_mode != "auto":
            return self._send(
                self.cfg.api_mode, system, user, max_tokens, allow_fallback=False,
            )

        # concurrency worker들이 모두 _latched=None을 보고 /responses를 중복 probe하지
        # 않게 첫 capability negotiation을 single-flight한다. 성공 뒤 각 유닛 요청은
        # lock 밖에서 병렬로 흐르고, 최초 probe가 실패하면 그 시점의 동시 대기자 모두
        # 같은 오류를 받아 죽은 endpoint를 직렬로 다시 두드리지 않는다.
        owner = False
        with self._mode_lock:
            flight = self._mode_flight
            if flight is not None and flight.event.is_set() and flight.error is not None:
                self._raise_flight_error(flight.error)
            mode = self._latched
            if mode is None and flight is None:
                flight = self._mode_flight = _ModeFlight()
                owner = True

        if not owner and flight is not None:
            # Event.wait 자체는 cancel을 모르므로 짧게 끊는다. owner는 모든 경로에서
            # result/error를 공개한 뒤 반드시 set한다.
            while not flight.event.wait(0.1):
                self._raise_if_cancelled()
            if flight.error is not None:
                self._raise_flight_error(flight.error)
            with self._mode_lock:
                mode = self._latched
            if mode is None:  # 방어 경로 — 성공 flight는 반드시 mode를 래치한다.
                raise TranslateAPIError("번역 API 초기 모드 협상 결과가 없습니다")
            return self._send(mode, system, user, max_tokens, allow_fallback=False)

        if mode is not None:
            return self._send(mode, system, user, max_tokens, allow_fallback=False)

        try:
            try:
                result = self._send(
                    "responses", system, user, max_tokens, allow_fallback=True,
                )
            except _NeedsFallback:
                with self._mode_lock:
                    self._latched = "chat"
                    self.api_mode_used = "chat"
                result = self._send(
                    "chat", system, user, max_tokens, allow_fallback=False,
                )
            else:
                with self._mode_lock:
                    self._latched = "responses"
                    self.api_mode_used = "responses"
            return result
        except BaseException as exc:
            flight.error = (type(exc), exc.args)
            raise
        finally:
            flight.event.set()
            if flight.error is not None:
                # 이미 flight 참조를 얻은 동시 대기자들은 같은 오류를 받되, 나중의
                # 순차 호출은 새 협상을 허용한다. 일시 500/연결 오류 하나를 client
                # 수명 전체에 영구 래치하면 glossary 실패 뒤 본 번역도 회복할 수 없다.
                with self._mode_lock:
                    if self._mode_flight is flight:
                        self._mode_flight = None

    # ── 내부 ────────────────────────────────────────────────────────

    def _use_stream(self, mode: str) -> bool:
        """이 요청을 SSE로 받을까 — responses 모드는 비스트리밍(이벤트 형식이 다르다)."""
        if mode != "chat" or self.cfg.stream == "off":
            return False
        return self.cfg.stream == "on" or self._stream_ok is not False

    def _build_payload(
        self, mode: str, system: str, user: str, max_tokens: int, *, stream: bool = False,
    ) -> dict:
        cfg = self.cfg
        temp_ok = cfg.temperature != "none"
        if mode == "responses":
            p: dict = {"model": cfg.model, "instructions": system, "input": user}
            if self._store_ok is not False:
                # Responses는 store 생략 시 true — OpenAI는 응답 객체를 30일 보관하고 oMLX는
                # 원문+번역을 SSD에 최근 1000건까지 평문 영속한다(mlx-integration-2).
                # Q&A 클라이언트와 같은 store:false 약속을 번역 경로에도 지킨다.
                p["store"] = False
            if temp_ok:
                p["temperature"] = float(cfg.temperature)
            if cfg.max_tokens_param != "none":
                p["max_output_tokens"] = max_tokens
        else:
            p = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if temp_ok:
                p["temperature"] = float(cfg.temperature)
            if cfg.max_tokens_param == "max_tokens":
                p["max_tokens"] = max_tokens
            elif cfg.max_tokens_param == "max_completion_tokens":
                p["max_completion_tokens"] = max_tokens
            if stream:
                # include_usage: 마지막 청크에 usage를 싣는다(OpenAI·vLLM·mlx_lm 공통).
                # mlx_lm은 stream_options가 있으면 include_usage 키를 반드시 읽는다.
                p["stream"] = True
                p["stream_options"] = {"include_usage": True}
        _apply_reasoning(p, mode, cfg.reasoning, cfg.effective_reasoning_style)
        _merge_extra_body(p, cfg.extra_body_dict)
        return p

    def _send(
        self, mode: str, system: str, user: str, max_tokens: int, allow_fallback: bool
    ) -> tuple[str, bool]:
        path = "responses" if mode == "responses" else "chat/completions"
        payload = self._build_payload(
            mode, system, user, max_tokens, stream=self._use_stream(mode),
        )
        attempt = 0
        timeouts = 0
        stream_probe = False  # auto 스트리밍이 거부돼 같은 요청을 비스트리밍으로 재시도 중
        store_probe = False   # store 파라미터가 거부돼 store 없이 재시도 중
        while True:
            self._raise_if_cancelled()
            try:
                status, body, headers = self._post(path, payload)
            except _RequestCancelled:
                raise
            except requests.exceptions.ReadTimeout as e:
                # 응답 정지 — 스트리밍이면 토큰 사이, 비스트리밍이면 대기열+생성 전체가
                # timeout_s를 넘었다. 재시도는 같은 긴 생성의 반복이라 최대 1회만 한다.
                if timeouts < min(_MAX_TIMEOUT_RETRIES, self.cfg.max_retries):
                    timeouts += 1
                    wait = self._backoff({}, attempt)
                    logger.warning(
                        "번역 API 응답 시간 초과(%gs) — %.1fs 후 재시도 (%d/%d)",
                        self.cfg.timeout_s, wait, timeouts, _MAX_TIMEOUT_RETRIES,
                    )
                    self._wait_or_cancel(wait)
                    continue
                raise TranslateTimeout(
                    f"번역 API 응답 시간 초과 — {self.cfg.timeout_s:g}초 동안 응답이 없었습니다"
                    " (TRANSLATE_TIMEOUT_S, 서버 대기열·생성 속도 확인)"
                ) from e
            except requests.RequestException as e:
                # ConnectionError·Timeout뿐 아니라 본문 수신 중 끊김(ChunkedEncodingError·
                # ContentDecodingError 등 RequestException 계열, ConnectionError 비상속)도
                # 일시적 네트워크 결함이므로 같은 백오프로 재시도한다. HTTP 상태코드 분기는
                # _post가 응답을 반환한 경우(아래)라 이 절과 무관하다.
                if attempt < self.cfg.max_retries:
                    wait = self._backoff({}, attempt)
                    logger.warning(
                        "번역 API 연결 오류(%s) — %.1fs 후 재시도 (%d/%d)",
                        type(e).__name__, wait, attempt + 1, self.cfg.max_retries,
                    )
                    self._wait_or_cancel(wait)
                    attempt += 1
                    continue
                raise TranslateAPIError(f"번역 API 연결 실패: {e}") from e

            if status == 200:
                result = self._parse(mode, body)
                if store_probe:
                    self._store_ok = False
                    logger.warning(
                        "번역 API가 store 파라미터를 거부했습니다 — 이 잡은 store 없이 보냅니다"
                        " (서버의 응답 보관 정책을 확인하세요)"
                    )
                elif "store" in payload:
                    self._store_ok = True
                if stream_probe:
                    self._stream_ok = False
                    logger.warning(
                        "번역 API가 스트리밍 요청을 거부했습니다 — 이 잡은 비스트리밍으로 진행합니다"
                        " (TRANSLATE_STREAM=0으로 고정 가능)"
                    )
                elif payload.get("stream"):
                    self._stream_ok = True
                return result
            if (
                payload.get("stream") and self.cfg.stream == "auto"
                and self._stream_ok is None and status in _STREAM_REJECTED
            ):
                # 스트리밍(stream_options 등)을 모르는 엄격한 서버일 수 있다 — 같은 요청을
                # 비스트리밍으로 1회 보내 보고, 그게 성공하면 이 클라이언트는 비스트리밍으로
                # 래치한다. 비스트리밍도 같은 오류면 원래 오류 처리로 간다(유닛 거부 등).
                payload = self._build_payload(mode, system, user, max_tokens, stream=False)
                stream_probe = True
                continue
            if (
                "store" in payload and self._store_ok is None
                and status in _STREAM_REJECTED and "store" in str(body).lower()
            ):
                # store를 모르는 엄격한 Responses 구현 — 빼고 1회 재시도, 성공하면 래치.
                payload = {k: v for k, v in payload.items() if k != "store"}
                store_probe = True
                continue
            if allow_fallback and status in _FALLBACK:
                raise _NeedsFallback()
            if status in (401, 403):
                raise TranslateAPIError("번역 API 인증 실패 — OPENAI_API_KEY를 확인하세요")
            if status == 404 and mode == "chat":
                raise TranslateAPIError(_not_found_message(self.cfg.model, body))
            if status in _RETRYABLE and attempt < self.cfg.max_retries:
                wait = self._backoff(headers, attempt)
                ra = _header(headers, "Retry-After") or None
                logger.warning(
                    "번역 API HTTP %d — %.1fs 후 재시도 (%d/%d)%s",
                    status, wait, attempt + 1, self.cfg.max_retries,
                    f" (Retry-After: {ra})" if ra is not None else "",
                )
                self._wait_or_cancel(wait)
                attempt += 1
                continue
            # 재시도 불가 4xx는 결정적 거부 — 같은 요청을 다시 보내도 같은 자리에서
            # 죽는다. 엔진이 유닛 단위로 강등(래더 → 원문 유지)할 수 있게 구분한다.
            if status in _UNIT_REJECTED:
                raise TranslateUnitRejected(
                    f"번역 API 오류 (HTTP {status}): {_body_preview(body)}"
                )
            raise TranslateAPIError(f"번역 API 오류 (HTTP {status}): {_body_preview(body)}")

    def _backoff(self, headers: dict, attempt: int) -> float:
        ra = _header(headers, "Retry-After") or None
        if ra is not None:
            try:
                # 상한 없이 따르면 "Retry-After: 3600" 한 줄이 워커를 한 시간 묶어
                # 번역이 멈춘 것처럼 보인다. 지수 백오프와 같은 30초 상한을 적용한다.
                return min(_MAX_BACKOFF_S, max(0.0, float(ra)))
            except (TypeError, ValueError):
                pass
        return min(_MAX_BACKOFF_S, float(3 ** attempt))  # 1 → 3 → 9 → 27 → 30

    def _parse(self, mode: str, body: dict | str) -> tuple[str, bool]:
        """(텍스트, 잘림 여부) 반환. 잘림 = chat finish_reason=="length" /
        responses status=="incomplete" (미제공 서버는 False — 종전과 동일 동작).
        빈 응답은 오류지만, 잘려서 빈 경우(reasoning이 예산 소진)는 재시도 대상이므로
        raise하지 않고 ("", True)로 넘긴다."""
        if not isinstance(body, dict):
            raise TranslateAPIError(f"번역 API 응답 파싱 실패: {_body_preview(body)}")
        try:
            if mode == "responses":
                truncated = body.get("status") == "incomplete"
                ot = body.get("output_text")
                if isinstance(ot, str) and ot.strip():
                    text = ot
                else:
                    text = _parse_responses_output(body.get("output", []))
            else:
                choice = body["choices"][0]
                truncated = choice.get("finish_reason") == "length"
                text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise TranslateAPIError(f"번역 API 응답 파싱 실패: {_body_preview(body)}") from e
        text = _postprocess(text)
        if not text and not truncated:
            # 사고만 내고 끝났거나 빈 문자열 — 같은 프롬프트(온도 0)면 같은 결과라 유닛 단위.
            raise TranslateEmptyOutput("번역 API가 빈 응답을 반환했습니다")
        return text, truncated


class _LineBuffer:
    """받은 바이트 조각을 줄 단위로 나눈다(줄 끝 CR 제거) — 전체 비용이 받은 바이트에 선형.

    종전에는 조각마다 누적 버퍼를 새로 이어 붙이고 처음부터 다시 split해, 개행 없는 줄이
    길어질수록 CPU가 제곱으로 늘었다(8MB에 9초 — 데이터가 계속 와 read timeout도 걸리지
    않는다, translate-8). 새로 붙은 구간에서만 개행을 찾고, 완성된 줄들은 한 번만 자른다.
    """

    __slots__ = ("_buf", "_scan")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._scan = 0  # 이 위치 앞에는 개행이 없다

    def feed(self, chunk: bytes) -> list[bytes]:
        buf = self._buf
        buf += chunk
        end = buf.rfind(b"\n", self._scan)
        if end < 0:
            self._scan = len(buf)
            return []
        complete = bytes(buf[:end])
        del buf[:end + 1]
        self._scan = len(buf)  # 마지막 개행 뒤 조각에는 개행이 없다
        return [line.rstrip(b"\r") for line in complete.split(b"\n")]

    def rest(self) -> bytes:
        """개행으로 끝나지 않은 마지막 줄."""
        return bytes(self._buf).rstrip(b"\r")


class _StreamAccumulator:
    """chat.completion.chunk SSE 이벤트를 모아 비스트리밍 응답 모양으로 만든다.

    delta.content만 본문으로 잇고 reasoning/reasoning_content는 길이만 센다(사고 과정은
    번역문이 아니다). 스트림 중간의 {"error": …}는 서버 오류로 올린다.
    """

    def __init__(self) -> None:
        self.data: list[bytes] = []
        self.parts: list[str] = []
        self.reasoning_chars = 0
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.done = False

    @property
    def complete(self) -> bool:
        return self.done or self.finish_reason is not None

    def feed(self, line: bytes) -> bool:
        """SSE 한 줄을 먹인다. [DONE]을 만나면 True(더 읽을 필요 없음)."""
        if not line:
            self._dispatch()
            return self.done
        if line.startswith(b":"):
            return False  # 주석 — mlx_lm의 prefill keepalive 등
        name, _, value = line.partition(b":")
        if name == b"data":
            self.data.append(value[1:] if value.startswith(b" ") else value)
        return False

    def _dispatch(self) -> None:
        if not self.data:
            return
        text = b"\n".join(self.data).decode("utf-8", "replace").strip()
        self.data = []
        if text == "[DONE]":
            self.done = True
            return
        try:
            event = json.loads(text)
        except ValueError as e:
            raise TranslateAPIError(f"번역 API 스트림 파싱 실패: {text[:200]}") from e
        if not isinstance(event, dict):
            return
        if event.get("error"):
            raise TranslateAPIError(f"번역 API 스트림 오류: {_body_preview(event)}")
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if content:
                    self.parts.append(_content_text(content))
                for key in ("reasoning_content", "reasoning"):
                    if isinstance(delta.get(key), str):
                        self.reasoning_chars += len(delta[key])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def body(self) -> dict:
        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(self.parts)},
                "finish_reason": self.finish_reason,
            }],
            "usage": self.usage or {},
            "stream_stats": {"reasoning_chars": self.reasoning_chars},
        }


def _error_detail(body: dict | str) -> str:
    """오류 본문의 서버 메시지 — JSON {"error": "…"|{"message": "…"}}일 때만."""
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        err = err.get("message") or err.get("code") or ""
    return str(err or body.get("detail") or "")[:300]


def _not_found_message(model: str, body: dict | str) -> str:
    """chat 404 진단 — 경로가 없는 것인지 모델 ID가 틀린 것인지 구분한다(probe:MLX-07).

    종전에는 모든 404를 'OPENAI_BASE_URL이 /v1까지 포함하는지 확인'으로 바꾸고 서버
    본문을 버렸다. mlx_lm은 모르는 모델 ID에 404 + {"error": …}를 내고 그 사이 현재
    모델을 언로드하므로, 사용자는 URL만 고치며 헤맸다. JSON 오류 본문이 있으면 모델
    ID 문제로 안내하고 서버 메시지를 보여 준다. 순수 'Not Found'만 경로 문제로 본다.
    """
    detail = _error_detail(body)
    if detail:
        return (
            f"번역 API가 404를 반환했습니다 — 모델 ID(OPENAI_MODEL/TRANSLATE_MODEL={model})가 "
            "서버의 /v1/models 목록과 다를 수 있습니다(mlx_lm.server는 default_model 또는 "
            f"/v1/models의 id 그대로). 서버 응답: {detail}"
        )
    return "번역 API 엔드포인트 없음 — OPENAI_BASE_URL이 /v1까지 포함하는지 확인하세요"


def _header(headers: dict, name: str) -> str:
    """헤더 값을 대소문자 무관하게 찾는다 — dict(resp.headers)는 서버가 보낸 표기를 그대로
    키로 쓴다. mlx_lm.server는 'Content-type'을 보내 SSE 판정이 빗나갔다(실측)."""
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return ""


def _decode_body(raw: bytes) -> dict | str:
    """비스트리밍 본문 — JSON이면 파싱 결과, 아니면 텍스트(오류 미리보기용)."""
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def _is_read_timeout(exc: BaseException) -> bool:
    """requests가 ConnectionError로 감싼 urllib3 ReadTimeoutError인가."""
    inner = exc.args[0] if exc.args else None
    return isinstance(inner, ReadTimeoutError) or "Read timed out" in str(exc)


def _response_socket(resp):
    """응답이 읽고 있는 소켓 — urllib3 내부 속성에 기대므로 못 찾으면 None."""
    raw = getattr(resp, "raw", None)
    sock = getattr(getattr(raw, "_connection", None), "sock", None)
    if sock is None:
        fp = getattr(getattr(raw, "_fp", None), "fp", None)
        sock = getattr(getattr(fp, "raw", None), "_sock", None)
    return sock


def _shutdown_socket(resp) -> None:
    """진행 중인 응답의 소켓을 끊는다 — 서버는 다음 토큰 쓰기에서 끊김을 보고 생성을 멈춘다.

    다른 스레드가 recv에 막혀 있어도 shutdown(SHUT_RDWR)은 즉시 깨운다(close만으로는
    리눅스에서 깨지 않는다). 연결 반납(close)은 읽던 헬퍼 스레드가 finally에서 한다 —
    여기서 반납하면 다른 워커가 아직 읽히는 중인 연결을 재사용할 수 있다.
    urllib3 내부 속성에 기대므로 찾지 못하면 조용히 넘어간다(헬퍼가 다음 청크에서 끊는다).
    """
    sock = _response_socket(resp)
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _drain_after_done(resp, chunks, abort: threading.Event) -> None:
    """[DONE] 뒤 남은 본문(청크 종료 표시)을 짧게 읽어 연결을 keep-alive 풀로 돌려준다.

    requests는 본문을 끝까지 읽지 않은 응답을 close()에서 소켓째 닫는다. [DONE]에서 바로
    반환하면 종료 청크(0\\r\\n\\r\\n)가 소켓에 남아, 원격 HTTPS에서는 유닛 요청마다 TCP+TLS를
    새로 맺었다(translate-4). 길이가 정해진 keep-alive 본문(chunked·Content-Length)만
    시도하고, [DONE] 뒤에도 스트림을 열어 두는 서버에 막히지 않게 소켓 읽기 시간과 총
    시간·바이트를 묶는다. 다 읽지 못하면 종전처럼 연결을 닫을 뿐이다.
    """
    raw = getattr(resp, "raw", None)
    if raw is None:
        return
    framed = bool(getattr(raw, "chunked", False)) or getattr(raw, "length_remaining", None) is not None
    if not framed or "close" in _header(dict(resp.headers), "Connection").lower():
        return
    sock = _response_socket(resp)
    if sock is None:
        return
    try:
        previous = sock.gettimeout()
        sock.settimeout(_DRAIN_TIMEOUT_S)
    except OSError:
        return
    deadline = time.monotonic() + _DRAIN_TIMEOUT_S
    drained = 0
    try:
        for chunk in chunks:  # 끝까지 돌면 requests가 본문을 다 읽은 것으로 보고 연결을 반납한다
            drained += len(chunk)
            if abort.is_set() or drained > _DRAIN_MAX_BYTES or time.monotonic() > deadline:
                return
    except (requests.RequestException, OSError, Urllib3HTTPError):
        return  # 시간 초과·끊김 — 연결은 close()가 닫는다
    finally:
        try:
            sock.settimeout(previous)
        except OSError:
            pass


def _hopeless_truncation(text: str, user: str) -> str:
    """잘린 출력을 2배 예산으로 다시 받아도 쓸 수 없을 게 뻔한 이유 — 없으면 "".

    본문이 비었으면(thinking이 예산을 다 씀) 2배 예산이면 본문까지 갈 수 있어 재시도한다.
    프롬프트 전체 길이를 쓰는 이유: 클라이언트는 원문 구간을 모른다. 원문 ⊂ 프롬프트라
    '프롬프트의 4배 초과'면 원문 대비 게이트 상한(3–4배)도 반드시 넘는다(보수적).
    짧은 프롬프트는 thinking 뒤 남은 본문 몇 줄만으로도 4배를 넘으므로 하한을 둔다.
    """
    if not text:
        return ""
    if is_degenerate_repetition(text):
        return "반복 루프 감지"
    if len(text) > max(4 * len(user), _OVERLONG_MIN_CHARS):
        return "출력이 입력의 4배를 넘음"
    return ""


def _exhausted_message(cfg: TranslateConfig) -> str:
    """출력이 전부 잘렸을 때의 안내 — 원인은 대개 thinking이 예산을 다 쓴 것이다.

    종전 문구('TRANSLATE_REASONING 예산을 확인하세요')는 off가 이미 설정돼 있어도 같은
    말을 해 원인을 가렸다. mlx_lm·oMLX는 reasoning 필드를 무시하고 thinking을 켜므로
    서버측에서 끄는 방법까지 안내한다.
    """
    style = cfg.effective_reasoning_style
    if not cfg.reasoning:
        how = "TRANSLATE_REASONING=off로 thinking을 끄세요"
    elif cfg.reasoning == "off" and style in ("openrouter", "none"):
        how = (
            f"TRANSLATE_REASONING=off가 이 서버에 전달되지 않았을 수 있습니다(방식: {style}) — "
            "로컬 MLX·vLLM 서버는 TRANSLATE_REASONING_STYLE=chat_template_kwargs를 쓰세요"
        )
    elif cfg.reasoning == "off":
        how = "서버가 thinking 끄기 요청을 따르지 않았습니다"
    else:
        how = f"reasoning effort({cfg.reasoning})를 낮추거나 off로 두세요"
    return (
        "번역 API 출력이 max_tokens에서 전부 잘렸습니다 — 모델의 thinking(reasoning)이 "
        f"출력 예산을 모두 쓴 것으로 보입니다. {how}. 서버에서 끄려면 mlx_lm.server는 "
        "--chat-template-args '{\"enable_thinking\":false}', oMLX·LM Studio는 모델별 "
        "thinking 설정을 끄거나 비-thinking(Instruct) 모델을 쓰세요"
    )


def _apply_reasoning(p: dict, mode: str, reasoning: str, style: str) -> None:
    """TRANSLATE_REASONING 값을 서버 계열별 필드로 싣는다 (types.REASONING_STYLES 참조).

    reasoning이 빈 값이면 아무것도 보내지 않는다 — 구형 서버 호환 기본값(opt-in).
    """
    if not reasoning or style == "none":
        return
    off = reasoning == "off"
    if style == "openrouter":
        p["reasoning"] = {"enabled": False} if off else {"effort": reasoning}
    elif style == "chat_template_kwargs":
        # mlx_lm·oMLX·vLLM·llama.cpp가 채팅 템플릿 인자로 넘긴다. mlx_lm은 Qwen 계열에
        # enable_thinking=True를 기본 주입하므로 off를 반드시 명시해야 꺼진다.
        p["chat_template_kwargs"] = {"enable_thinking": not off}
        if not off:
            # effort 개념이 있는 모델(gpt-oss 등)용 — 모르는 서버·모델은 무시한다.
            if mode == "responses":
                p["reasoning"] = {"effort": reasoning}
            else:
                p["reasoning_effort"] = reasoning
    elif style == "reasoning_effort":
        effort = "none" if off else reasoning
        if mode == "responses":
            p["reasoning"] = {"effort": effort}
        else:
            p["reasoning_effort"] = effort


def _merge_extra_body(p: dict, extra: dict) -> None:
    """TRANSLATE_EXTRA_BODY 병합 — 객체 값은 한 단계 합치고(chat_template_kwargs 등)
    나머지는 덮어쓴다. 클라이언트가 책임지는 키는 설정 단계에서 이미 거부됐다."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(p.get(key), dict):
            p[key] = {**p[key], **value}
        else:
            p[key] = value


def _parse_responses_output(output) -> str:
    """responses output[] 순회 — reasoning은 건너뛰고 message의 output_text/text를 잇는다."""
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") == "reasoning":
            continue
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "".join(parts)


def _strip_think(text: str) -> str:
    """thinking 흔적을 걷어낸 본문.

    세 형태를 처리한다(감사 mlx-integration-6·translate-llm-16, probe:MLX-06):
      * `<think>…</think>답` — 선두 완결 블록(종전 유일 처리 형태)
      * `…</think>답` — 여는 태그가 프롬프트 쪽에 있어 content에는 닫는 태그만 온다.
        reasoning을 분리하지 않는 서버(llama.cpp --reasoning-format none, LM Studio
        분리 끔 등)에서 짧은 영어 독백이 번역문 앞에 붙은 채 게이트를 통과했다.
      * 닫히지 않은 선두 `<think>…` — 사고 도중 잘렸거나 사고만 냈다 → 본문 없음.
    원문의 `<think>` 리터럴은 마스킹이 플레이스홀더로 바꾸므로 출력의 태그는 언제나
    모델의 사고 표기다. 따라서 마지막 닫는 태그 뒤만 본문으로 쓴다.
    """
    if _THINK_CLOSE in text:
        return text.rsplit(_THINK_CLOSE, 1)[1]
    if _THINK_OPEN_RE.match(text):
        return ""
    return text


def _content_text(content) -> str:
    """message.content — 문자열 또는 [{"type":"text","text":…}] 파트 배열."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text", "output_text")
        )
    return ""


def _postprocess(text) -> str:
    """thinking 흔적 제거, 전체 감싼 코드펜스 벗기기, strip."""
    text = _strip_think(_content_text(text)).strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text.strip()


def _body_preview(body: dict | str) -> str:
    s = body if isinstance(body, str) else str(body)
    return s[:200]
