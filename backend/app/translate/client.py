"""OpenAI 호환 클라이언트 — Chat Completions·Responses API 양쪽 지원.

로컬 서버(vLLM·Ollama·llama.cpp 등)는 어느 한쪽만 지원하는 경우가 많아 api_mode
"auto"는 첫 호출에 responses를 시도하고 404/405/501이면 chat으로 영구 래치한다.
requests만 쓰며(런타임 기존 의존성), 실제 전송은 _post 한 메서드로 모아 테스트가
그것만 몽키패치하도록 한다.

잘림(truncation) 처리: chat `finish_reason=="length"` / responses
`status=="incomplete"`를 감지하면 같은 요청을 **max_tokens 2배로 1회 재시도**한다
(2026-07-08 합의 정책 ②). thinking 모델은 reasoning 토큰이 같은 예산에서 차감되어
effort 테이블(types.REASONING_MAX_TOKENS)로도 드물게 잘릴 수 있다 — 잘린 출력을
조용히 반환하면 unmask 실패→래더→원문 유지로 강등되고, 용어집 Pass-0 JSON은
시드로 조용히 강등되던 취약 지점이다.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import requests

from .types import TranslateAPIError, TranslateConfig, TranslateUnitRejected

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

# thinking 출력 표기 — 여는 태그는 선두에서만 의미가 있다(템플릿이 프롬프트 끝에
# `<think>`를 미리 넣는 Qwen3 계열은 content가 여는 태그 없이 '…</think>답'으로 온다).
_THINK_OPEN_RE = re.compile(r"^\s*<think>")
_THINK_CLOSE = "</think>"
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)```\s*$", re.DOTALL)


class _NeedsFallback(Exception):
    """내부용 — auto 모드에서 responses가 미지원일 때 chat 폴백을 신호."""


class _RequestCancelled(TranslateAPIError):
    """내부용 — 새 HTTP 요청을 보내기 전에 cancel/abort를 관찰했다."""


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
        """(status, body(json이면 dict 아니면 str), headers) 반환."""
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
            resp = self.session.post(
                url, json=payload, headers=headers,
                # (connect, read) — min은 timeout_s를 10초 미만으로 줄인 설정을 존중한다.
                timeout=(min(_CONNECT_TIMEOUT_S, self.cfg.timeout_s), self.cfg.timeout_s),
            )
        finally:
            if semaphore is not None and acquired:
                semaphore.release()
        try:
            body: dict | str = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body, dict(resp.headers)

    # ── 공개 API ───────────────────────────────────────────────────
    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        text, truncated = self._complete_once(system, user, max_tokens)
        if not truncated:
            return text  # 빈 응답은 _parse가 이미 raise
        # 잘림(finish_reason=length / responses incomplete) — 예산 2배로 1회 재시도.
        # max_tokens 파라미터를 아예 안 보내는 설정(none)이면 같은 요청의 반복이라 생략.
        if self.cfg.max_tokens_param != "none":
            logger.warning("번역 API 출력 잘림 — max_tokens %d→%d로 1회 재시도", max_tokens, max_tokens * 2)
            try:
                retry_text, _ = self._complete_once(system, user, max_tokens * 2)
            except TranslateAPIError as e:
                # 2배 재시도가 실패해도 잘린 첫 출력이 있으면 버리지 않는다 — 래더가 흡수.
                if not text:
                    raise
                logger.warning("번역 API 잘림 2배 재시도 실패(%s) — 잘린 첫 출력을 사용", type(e).__name__)
                return text
            if retry_text:
                return retry_text  # 여전히 잘렸어도 더 긴 출력 — 래더가 흡수
        if text:
            return text
        raise TranslateAPIError(_exhausted_message(self.cfg))

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

    def _build_payload(self, mode: str, system: str, user: str, max_tokens: int) -> dict:
        cfg = self.cfg
        temp_ok = cfg.temperature != "none"
        if mode == "responses":
            p: dict = {"model": cfg.model, "instructions": system, "input": user}
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
        _apply_reasoning(p, mode, cfg.reasoning, cfg.effective_reasoning_style)
        _merge_extra_body(p, cfg.extra_body_dict)
        return p

    def _send(
        self, mode: str, system: str, user: str, max_tokens: int, allow_fallback: bool
    ) -> tuple[str, bool]:
        path = "responses" if mode == "responses" else "chat/completions"
        payload = self._build_payload(mode, system, user, max_tokens)
        attempt = 0
        while True:
            self._raise_if_cancelled()
            try:
                status, body, headers = self._post(path, payload)
            except _RequestCancelled:
                raise
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
                return self._parse(mode, body)
            if allow_fallback and status in _FALLBACK:
                raise _NeedsFallback()
            if status in (401, 403):
                raise TranslateAPIError("번역 API 인증 실패 — OPENAI_API_KEY를 확인하세요")
            if status == 404 and mode == "chat":
                raise TranslateAPIError(
                    "번역 API 엔드포인트 없음 — OPENAI_BASE_URL이 /v1까지 포함하는지 확인하세요"
                )
            if status in _RETRYABLE and attempt < self.cfg.max_retries:
                wait = self._backoff(headers, attempt)
                ra = headers.get("Retry-After") or headers.get("retry-after")
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
        ra = headers.get("Retry-After") or headers.get("retry-after")
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
            raise TranslateAPIError("번역 API가 빈 응답을 반환했습니다")
        return text, truncated


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
