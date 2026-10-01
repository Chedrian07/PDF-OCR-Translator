"""클라이언트 — chat/responses 파싱·auto 폴백·재시도·오류·후처리·URL 정규화."""

import pytest

from app.translate.client import OpenAICompatClient, _endpoint_url, _normalize_base_url
from app.translate.types import TranslateAPIError, TranslateConfig


def _cfg(**kw) -> TranslateConfig:
    base = dict(
        base_url="https://host/v1", api_key="sk-x", model="m",
        api_mode="auto", max_retries=3, temperature="0", max_tokens_param="max_tokens",
    )
    base.update(kw)
    return TranslateConfig(**base)


def test_base_url_정규화():
    assert _normalize_base_url("https://host:/v1") == "https://host/v1"   # 빈 포트 교정
    assert _normalize_base_url("  https://host/v1/ ") == "https://host/v1"  # strip + 끝 /
    assert _normalize_base_url("https://host:8080/v1") == "https://host:8080/v1"  # 실 포트 보존
    assert _normalize_base_url("https://host") == "https://host/v1"  # bare origin 편의
    assert _normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    # 명시한 공급자별 경로와 query는 추측해서 바꾸지 않는다.
    assert _normalize_base_url("https://host/gateway?tenant=x") == "https://host/gateway?tenant=x"
    assert _endpoint_url("https://host/v1?tenant=x", "responses") == (
        "https://host/v1/responses?tenant=x"
    )


@pytest.mark.parametrize(("timeout_s", "expected"), [
    (180.0, (10.0, 180.0)),
    (5.0, (5.0, 5.0)),
])
def test_post는_connect와_read_timeout을_분리(timeout_s, expected):
    captured = {}

    class Response:
        status_code = 200
        headers = {"X-Test": "yes"}
        text = "unused"

        @staticmethod
        def json():
            return {"output_text": "ok"}

    class Session:
        @staticmethod
        def post(url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

    client = OpenAICompatClient(_cfg(api_mode="responses", timeout_s=timeout_s))
    client.session = Session()
    status, body, headers = client._post("responses", {"input": "x"})

    assert captured["timeout"] == expected
    assert captured["url"] == "https://host/v1/responses"
    assert status == 200 and body == {"output_text": "ok"}
    assert headers["X-Test"] == "yes"


def test_request_semaphore는_여러_잡의_실제_HTTP_동시성을_제한():
    import concurrent.futures as cf
    import threading

    active = 0
    peak = 0
    lock = threading.Lock()

    class Response:
        status_code = 200
        headers = {}
        text = "unused"

        @staticmethod
        def json():
            return {"output_text": "ok"}

    # 두 요청이 실제로 겹치는지 sleep 타이밍에 맡기면 부하가 큰 CI에서 flaky하다.
    # Barrier(2)로 짝을 이루게 하면 슬롯이 2개일 때만 통과한다(1개면 timeout으로 실패).
    pair = threading.Barrier(2, timeout=10)

    class Session:
        @staticmethod
        def post(url, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            pair.wait()
            with lock:
                active -= 1
            return Response()

    slots = threading.BoundedSemaphore(2)
    clients = [
        OpenAICompatClient(_cfg(api_mode="responses"), request_semaphore=slots)
        for _ in range(6)
    ]
    for client in clients:
        client.session = Session()
    with cf.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        results = list(executor.map(
            lambda client: client._post("responses", {"input": "x"})[0], clients,
        ))

    assert results == [200] * len(clients)
    assert peak == 2


def test_request_semaphore_대기중_취소면_HTTP를_보내지_않음():
    import concurrent.futures as cf
    import threading

    entered = threading.Event()
    canceled = threading.Event()
    slots = threading.BoundedSemaphore(1)
    assert slots.acquire(blocking=False)  # 다른 잡이 유일한 전역 슬롯을 점유
    calls = 0

    class ObservedSemaphore:
        def acquire(self, **kwargs):
            entered.set()
            return slots.acquire(**kwargs)

        def release(self):
            slots.release()

    class Session:
        @staticmethod
        def post(url, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("취소된 요청은 session.post에 도달하면 안 된다")

    client = OpenAICompatClient(
        _cfg(api_mode="responses"),
        request_semaphore=ObservedSemaphore(),
        cancel_check=canceled.is_set,
    )
    client.session = Session()
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client._post, "responses", {"input": "x"})
        assert entered.wait(1)
        canceled.set()
        with pytest.raises(TranslateAPIError, match="취소"):
            future.result(timeout=2)
    assert calls == 0
    slots.release()


def test_chat_파싱():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": "안녕하세요"}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "안녕하세요"
    assert c.api_mode_used == "chat"


def test_responses_output_text():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = lambda p, pl: (200, {"output_text": "응답 텍스트"}, {})
    assert c.complete("s", "u", max_tokens=100) == "응답 텍스트"


def test_responses_output_배열_reasoning_스킵():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    body = {"output": [
        {"type": "reasoning", "content": [{"type": "text", "text": "무시"}]},
        {"type": "message", "content": [
            {"type": "output_text", "text": "앞"},
            {"type": "text", "text": "뒤"},
        ]},
    ]}
    c._post = lambda p, pl: (200, body, {})
    assert c.complete("s", "u", max_tokens=100) == "앞뒤"


def test_responses_output_text_빈문자열이면_배열로():
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    body = {"output_text": "   ", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "배열본문"}]},
    ]}
    c._post = lambda p, pl: (200, body, {})
    assert c.complete("s", "u", max_tokens=100) == "배열본문"


def test_auto_404_chat_래치():
    paths = []

    def post(p, pl):
        paths.append(p)
        if p == "responses":
            return (404, "not found", {})
        return (200, {"choices": [{"message": {"content": "챗"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="auto"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "챗"
    assert paths == ["responses", "chat/completions"]
    assert c.api_mode_used == "chat"
    # 이후 호출은 chat 직행 (영구 래치)
    paths.clear()
    c.complete("s", "u", max_tokens=100)
    assert paths == ["chat/completions"]


def test_auto_responses_성공시_래치():
    c = OpenAICompatClient(_cfg(api_mode="auto"))
    c._post = lambda p, pl: (200, {"output_text": "ok"}, {})
    c.complete("s", "u", max_tokens=100)
    assert c.api_mode_used == "responses"


def test_auto_첫_probe는_concurrency에서도_single_flight():
    import concurrent.futures as cf
    import threading
    import time

    workers = 8
    start = threading.Barrier(workers)
    lock = threading.Lock()
    paths = []

    def post(path, payload):
        with lock:
            paths.append(path)
        if path == "responses":
            time.sleep(0.08)  # lock이 없으면 모든 worker가 None을 읽고 함께 probe
            return (404, "not found", {})
        return (200, {"choices": [{"message": {"content": payload["messages"][1]["content"]}}]}, {})

    client = OpenAICompatClient(_cfg(api_mode="auto"))
    client._post = post

    def complete(index):
        start.wait()
        return client.complete("s", f"u{index}", max_tokens=100)

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(complete, range(workers)))

    assert results == [f"u{i}" for i in range(workers)]
    assert paths.count("responses") == 1
    assert paths.count("chat/completions") == workers
    assert client.api_mode_used == "chat"


def test_auto_첫_probe_실패도_동시_대기자에게_single_flight():
    import concurrent.futures as cf
    import threading
    import time

    workers = 8
    start = threading.Barrier(workers)
    lock = threading.Lock()
    calls = 0

    def post(path, payload):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.08)
        return (500, "provider down", {})

    client = OpenAICompatClient(_cfg(api_mode="auto", max_retries=0))
    client._post = post

    def complete(index):
        start.wait()
        with pytest.raises(TranslateAPIError, match="HTTP 500"):
            client.complete("s", f"u{index}", max_tokens=100)

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(complete, range(workers)))

    assert calls == 1
    assert client.api_mode_used == ""


def test_auto_실패_flight는_후속_순차호출의_회복을_막지_않음():
    calls = []

    def post(path, payload):
        calls.append(path)
        if len(calls) == 1:
            return (500, "temporary", {})
        return (200, {"output_text": "회복"}, {})

    client = OpenAICompatClient(_cfg(api_mode="auto", max_retries=0))
    client._post = post
    with pytest.raises(TranslateAPIError, match="HTTP 500"):
        client.complete("s", "first", max_tokens=100)
    assert client.complete("s", "second", max_tokens=100) == "회복"
    assert calls == ["responses", "responses"]
    assert client.api_mode_used == "responses"


def test_429_retry_after_재시도():
    seq = iter([
        (429, "느림", {"Retry-After": "0"}),
        (200, {"choices": [{"message": {"content": "성공"}}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "성공"


def test_401_인증실패_메시지():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (401, "unauthorized", {})
    with pytest.raises(TranslateAPIError, match="인증 실패"):
        c.complete("s", "u", max_tokens=100)


def test_chat_404_엔드포인트_메시지():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (404, "x", {})
    with pytest.raises(TranslateAPIError, match="엔드포인트 없음"):
        c.complete("s", "u", max_tokens=100)


def test_think_스트립_코드펜스_벗기기():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    content = "<think>추론 과정</think>\n```\n최종 번역문\n```"
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": content}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "최종 번역문"


@pytest.mark.parametrize(("content", "expected"), [
    # 템플릿이 <think>를 프롬프트에 미리 넣어 content에는 닫는 태그만 오는 형태
    ("Okay, the user wants Korean.\n</think>\n\n본 논문은 새로운 방법을 제안한다.",
     "본 논문은 새로운 방법을 제안한다."),
    ("<think>\nfirst</think>\n<think>second</think>\n\n최종 번역.", "최종 번역."),
    ("<think></think>\n\n번역문.", "번역문."),
    ("  평범한 번역문.  ", "평범한 번역문."),
])
def test_think_흔적은_마지막_닫는_태그_뒤만_본문으로(content, expected):
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": content}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == expected


def test_닫히지_않은_think는_본문이_없는_것으로_본다():
    """사고 도중 잘린 출력이 영어 독백 그대로 번역문이 되면 안 된다."""
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {
        "content": "<think>\nLet me think about the terminology first"}}]}, {})
    with pytest.raises(TranslateAPIError, match="빈 응답"):
        c.complete("s", "u", max_tokens=100)


def test_reasoning_content_필드는_번역문에_섞이지_않는다():
    """reasoning을 분리하는 서버(mlx_lm·oMLX·vLLM)의 사고 필드는 무시한다."""
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {
        "content": "번역문.", "reasoning_content": "Let me think", "reasoning": "hmm",
    }}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "번역문."


def test_content_파트_배열도_본문으로_잇는다():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": [
        {"type": "text", "text": "앞 "}, {"type": "text", "text": "뒤"},
    ]}}]}, {})
    assert c.complete("s", "u", max_tokens=100) == "앞 뒤"


def test_빈응답_오류():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": "   "}}]}, {})
    with pytest.raises(TranslateAPIError, match="빈 응답"):
        c.complete("s", "u", max_tokens=100)


def test_temperature_max_tokens_생략():
    c = OpenAICompatClient(_cfg(api_mode="chat", temperature="none", max_tokens_param="none"))
    captured = {}

    def post(p, pl):
        captured.update(pl)
        return (200, {"choices": [{"message": {"content": "x"}}]}, {})

    c._post = post
    c.complete("s", "u", max_tokens=100)
    assert "temperature" not in captured
    assert "max_tokens" not in captured and "max_completion_tokens" not in captured


def test_max_completion_tokens_파라미터():
    c = OpenAICompatClient(_cfg(api_mode="chat", max_tokens_param="max_completion_tokens"))
    captured = {}

    def post(p, pl):
        captured.update(pl)
        return (200, {"choices": [{"message": {"content": "x"}}]}, {})

    c._post = post
    c.complete("s", "u", max_tokens=512)
    assert captured["max_completion_tokens"] == 512 and "max_tokens" not in captured


def test_잘림_chat_finish_reason_length_예산2배_재시도():
    """chat 출력이 length로 잘리면 max_tokens 2배로 1회 재시도한다."""
    calls = []

    def post(p, pl):
        calls.append(pl.get("max_tokens"))
        if len(calls) == 1:
            return (200, {"choices": [{"message": {"content": "잘린 절반"},
                                       "finish_reason": "length"}]}, {})
        return (200, {"choices": [{"message": {"content": "완전한 번역"},
                                   "finish_reason": "stop"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "완전한 번역"
    assert calls == [100, 200]


def test_잘림_responses_incomplete_재시도():
    calls = []

    def post(p, pl):
        calls.append(pl.get("max_output_tokens"))
        if len(calls) == 1:
            return (200, {"status": "incomplete", "output_text": "부분"}, {})
        return (200, {"status": "completed", "output_text": "전체 번역"}, {})

    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "전체 번역"
    assert calls == [100, 200]


def test_잘림_재시도도_잘리면_재시도_출력_반환():
    """2배 예산 후에도 잘리면 그 출력을 그대로 쓴다 — 이후는 래더가 흡수."""
    seq = iter([
        (200, {"choices": [{"message": {"content": "A"}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "AB"}, "finish_reason": "length"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "AB"


def test_잘림_재시도_API오류면_잘린_첫출력_반환(caplog):
    """2배 재시도가 TranslateAPIError로 실패해도 잘린 첫 출력이 있으면 그것을 반환 — 래더가 흡수."""
    import logging

    seq = iter([
        (200, {"choices": [{"message": {"content": "잘린 절반"}, "finish_reason": "length"}]}, {}),
        (400, "bad request", {}),  # 2배 재시도 — 비재시도 상태코드로 즉시 TranslateAPIError
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "잘린 절반"
    assert any("2배 재시도 실패" in r.message for r in caplog.records)


def test_잘림_재시도_API오류_첫출력도_비면_예외전파():
    """첫 출력이 비어 있으면(전부 잘림) 재시도 실패 예외를 그대로 전파한다."""
    seq = iter([
        (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, {}),
        (400, "bad request", {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with pytest.raises(TranslateAPIError, match="HTTP 400"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_빈출력_reasoning_예산소진_재시도로_회복():
    """thinking이 예산을 다 먹어 content가 비어도 '빈 응답' 오류 대신 재시도."""
    seq = iter([
        (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "본문"}, "finish_reason": "stop"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    assert c.complete("s", "u", max_tokens=100) == "본문"


def test_잘림_전부_빈출력이면_오류():
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": ""},
                                                "finish_reason": "length"}]}, {})
    with pytest.raises(TranslateAPIError, match="잘렸습니다"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_max_tokens_param_none이면_재시도_안함():
    """max_tokens를 안 보내는 설정에선 재시도해도 같은 요청 — 1회로 끝낸다."""
    calls = []

    def post(p, pl):
        calls.append(1)
        return (200, {"choices": [{"message": {"content": "부분 출력"},
                                   "finish_reason": "length"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat", max_tokens_param="none"))
    c._post = post
    assert c.complete("s", "u", max_tokens=100) == "부분 출력"
    assert len(calls) == 1


def test_재시도_경로_warning_로그(caplog):
    """429 백오프·잘림 2배 재시도가 서버 로그에 warning으로 남는다 — 무기록 재시도 금지.
    본문·API 키는 로그에 남기지 않는다."""
    import logging

    seq = iter([
        (429, "느림", {"Retry-After": "0"}),
        (200, {"choices": [{"message": {"content": "부분"}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "완전"}, "finish_reason": "stop"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "완전"
    msgs = [r.message for r in caplog.records]
    assert any("HTTP 429" in m and "Retry-After" in m for m in msgs)   # 백오프 warning
    assert any("잘림" in m and "100→200" in m for m in msgs)           # 잘림 재시도 warning
    joined = "\n".join(msgs)
    assert "sk-x" not in joined and "부분" not in joined               # 키·본문 무기록


def test_연결오류_재시도_warning_로그(caplog):
    import logging

    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        if len(calls) == 1:
            raise _requests.ConnectionError("boom")
        return (200, {"choices": [{"message": {"content": "성공"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "성공"
    msgs = [r.message for r in caplog.records]
    assert any("연결 오류(ConnectionError)" in m for m in msgs)


def test_연결오류_ChunkedEncodingError_재시도로_회복(caplog):
    """본문 수신 중 끊김(RequestException 계열, ConnectionError 비상속)도 재시도한다."""
    import logging

    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        if len(calls) == 1:
            raise _requests.exceptions.ChunkedEncodingError("본문 수신 중 끊김")
        return (200, {"choices": [{"message": {"content": "성공"}}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=100) == "성공"
    assert len(calls) == 2
    assert any("연결 오류(ChunkedEncodingError)" in r.message for r in caplog.records)


def test_연결오류_계속_실패면_TranslateAPIError_래핑():
    """재시도 예산 소진 시 기존 ConnectionError 경로와 동일하게 TranslateAPIError로 전파."""
    import requests as _requests

    calls = []

    def post(p, pl):
        calls.append(1)
        raise _requests.exceptions.ContentDecodingError("깨진 응답")

    c = OpenAICompatClient(_cfg(api_mode="chat", max_retries=1))
    c._post = post
    c._backoff = lambda headers, attempt: 0.0                          # 테스트 대기 제거
    with pytest.raises(TranslateAPIError, match="연결 실패"):
        c.complete("s", "u", max_tokens=100)
    assert len(calls) == 2                                             # 최초 1 + 재시도 1


def test_reasoning_effort별_max_tokens_예산():
    """effort별 요청 max_tokens 테이블 (사용자 확정값) + xhigh 모드 지원."""
    from app.translate.types import REASONING_MAX_TOKENS, TranslateConfig

    expect = {"": 8192, "off": 8192, "low": 10240, "medium": 20480, "high": 40960, "xhigh": 81920}
    assert REASONING_MAX_TOKENS == expect
    for mode, budget in expect.items():
        cfg = TranslateConfig(base_url="https://h/v1", api_key="", model="m", reasoning=mode)
        assert cfg.max_output_tokens == budget

    # from_env가 xhigh를 허용하고 payload에 effort로 실림. 단일 라벨 호스트("h")는
    # auto가 로컬 서버로 보므로 OpenRouter 방식을 명시해 종전 페이로드를 고정한다.
    cfg = TranslateConfig.from_env({
        "OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
        "TRANSLATE_REASONING": "xhigh", "TRANSLATE_API_MODE": "chat",
        "TRANSLATE_REASONING_STYLE": "openrouter",
    })
    assert cfg.reasoning == "xhigh" and cfg.max_output_tokens == 81920
    from app.translate.client import OpenAICompatClient
    p = OpenAICompatClient(cfg)._build_payload("chat", "s", "u", cfg.max_output_tokens)
    assert p["reasoning"] == {"effort": "xhigh"} and p["max_tokens"] == 81920


# ── reasoning 전달 방식 (TRANSLATE_REASONING_STYLE) ─────────────────────────

@pytest.mark.parametrize(("base_url", "style"), [
    ("http://127.0.0.1:1235/v1", "chat_template_kwargs"),        # oMLX
    ("http://localhost:1234", "chat_template_kwargs"),           # LM Studio
    ("http://[::1]:8080/v1", "chat_template_kwargs"),            # mlx_lm.server
    ("http://host.docker.internal:1235/v1", "chat_template_kwargs"),
    ("http://vllm:8000/v1", "chat_template_kwargs"),             # compose 서비스명
    ("http://192.168.0.10:8000/v1", "chat_template_kwargs"),     # 사설망
    ("https://openrouter.ai/api/v1", "openrouter"),
    ("https://api.openai.com/v1", "reasoning_effort"),
    ("https://gateway.example.com/v1", "openrouter"),            # 알 수 없는 공개 호스트 = 종전
])
def test_auto_reasoning_style은_base_url로_확정(base_url, style):
    from app.translate.types import resolve_reasoning_style

    assert resolve_reasoning_style("auto", base_url) == style
    assert resolve_reasoning_style("none", base_url) == "none"   # 명시값은 그대로


@pytest.mark.parametrize(("style", "mode", "reasoning", "expected"), [
    ("openrouter", "chat", "off", {"reasoning": {"enabled": False}}),
    ("openrouter", "responses", "low", {"reasoning": {"effort": "low"}}),
    ("chat_template_kwargs", "chat", "off", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("chat_template_kwargs", "chat", "high",
     {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "high"}),
    ("chat_template_kwargs", "responses", "low",
     {"chat_template_kwargs": {"enable_thinking": True}, "reasoning": {"effort": "low"}}),
    ("reasoning_effort", "chat", "off", {"reasoning_effort": "none"}),
    ("reasoning_effort", "chat", "medium", {"reasoning_effort": "medium"}),
    ("reasoning_effort", "responses", "off", {"reasoning": {"effort": "none"}}),
    ("none", "chat", "off", {}),
])
def test_reasoning_style별_페이로드(style, mode, reasoning, expected):
    """mlx_lm·oMLX는 reasoning 필드를 무시하고 chat_template_kwargs만 읽는다(probe:MLX-01)."""
    keys = ("reasoning", "chat_template_kwargs", "reasoning_effort")
    c = OpenAICompatClient(_cfg(reasoning=reasoning, reasoning_style=style))
    p = c._build_payload(mode, "s", "u", 100)
    assert {k: p[k] for k in keys if k in p} == expected


def test_reasoning_미설정이면_어떤_방식이든_필드를_보내지_않는다():
    for style in ("auto", "openrouter", "chat_template_kwargs", "reasoning_effort"):
        c = OpenAICompatClient(_cfg(base_url="http://127.0.0.1:1235/v1", reasoning_style=style))
        p = c._build_payload("chat", "s", "u", 100)
        assert not {"reasoning", "chat_template_kwargs", "reasoning_effort"} & set(p)


def test_extra_body는_병합되고_객체값은_한단계_합친다():
    cfg = TranslateConfig.from_env({
        "OPENAI_BASE_URL": "http://127.0.0.1:1235/v1", "OPENAI_MODEL": "m",
        "TRANSLATE_REASONING": "off",
        "TRANSLATE_EXTRA_BODY": '{"chat_template_kwargs": {"thinking_budget": 0},'
                                ' "repetition_penalty": 1.05, "temperature": 0.2}',
    })
    p = OpenAICompatClient(cfg)._build_payload("chat", "s", "u", 100)
    assert p["chat_template_kwargs"] == {"enable_thinking": False, "thinking_budget": 0}
    assert p["repetition_penalty"] == 1.05 and p["temperature"] == 0.2
    assert p["messages"][1]["content"] == "u" and p["max_tokens"] == 100


@pytest.mark.parametrize("value", [
    "[1, 2]", "not json", '{"model": "other"}', '{"stream": true}', '{"store": true}',
    '{"max_tokens": 10}', "{" + '"k": "' + "x" * 5000 + '"}',
])
def test_extra_body_검증(value):
    from app.translate.types import TranslateError

    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m", "TRANSLATE_EXTRA_BODY": value}
    with pytest.raises(TranslateError, match="TRANSLATE_EXTRA_BODY"):
        TranslateConfig.from_env(env)


def test_reasoning_style_검증():
    from app.translate.types import TranslateError

    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
           "TRANSLATE_REASONING_STYLE": "magic"}
    with pytest.raises(TranslateError, match="TRANSLATE_REASONING_STYLE"):
        TranslateConfig.from_env(env)


def test_request_variant는_종전과_같은_요청이면_비어_있다():
    """종전(OpenRouter 방식)과 바이트 동일한 요청이면 캐시 키를 바꾸지 않는다."""
    legacy = _cfg(base_url="https://gateway.example.com/v1", reasoning="off")
    assert legacy.request_variant == ""
    assert _cfg(base_url="http://127.0.0.1:1235/v1").request_variant == ""  # reasoning 미설정
    local_off = _cfg(base_url="http://127.0.0.1:1235/v1", reasoning="off")
    assert local_off.request_variant == "reasoning_style=chat_template_kwargs"
    extra = _cfg(extra_body='{"top_k":20}')
    assert extra.request_variant == 'extra_body={"top_k":20}'


def test_전부_잘림_오류는_서버측_thinking_끄기를_안내():
    c = OpenAICompatClient(_cfg(api_mode="chat", base_url="https://gw.example.com/v1",
                                reasoning="off"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": ""},
                                                "finish_reason": "length"}]}, {})
    with pytest.raises(TranslateAPIError) as exc:
        c.complete("s", "u", max_tokens=100)
    msg = str(exc.value)
    assert "잘렸습니다" in msg and "thinking" in msg
    assert "TRANSLATE_REASONING_STYLE=chat_template_kwargs" in msg
    assert "enable_thinking" in msg


def test_translate_concurrency_default_and_server_cap():
    from app.translate.types import MAX_TRANSLATE_CONCURRENCY, TranslateConfig

    base = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m"}
    assert MAX_TRANSLATE_CONCURRENCY == 8
    assert TranslateConfig.from_env(base).concurrency == 8
    assert TranslateConfig.from_env({**base, "TRANSLATE_CONCURRENCY": "3"}).concurrency == 3
    assert TranslateConfig.from_env({**base, "TRANSLATE_CONCURRENCY": "99"}).concurrency == 8
    assert TranslateConfig.from_env({**base, "TRANSLATE_CONCURRENCY": "0"}).concurrency == 1


@pytest.mark.parametrize(("name", "value"), [
    ("TRANSLATE_CONCURRENCY", "abc"),
    ("TRANSLATE_CONCURRENCY", "1.5"),
    ("TRANSLATE_MAX_RETRIES", "abc"),
    ("TRANSLATE_TIMEOUT_S", "abc"),
    ("TRANSLATE_TIMEOUT_S", "nan"),
    ("TRANSLATE_TIMEOUT_S", "inf"),
    ("TRANSLATE_TIMEOUT_S", "-inf"),
])
def test_invalid_numeric_config_reports_setting_name(name, value):
    from app.translate.types import TranslateError

    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m", name: value}
    with pytest.raises(TranslateError, match=name):
        TranslateConfig.from_env(env)


@pytest.mark.parametrize("value", ["abc", "0,2", "nan", "inf", "-1", "2.5", "-0.1"])
def test_temperature는_none_또는_0에서_2_사이만_허용(value):
    """오타·음수가 '연결 실패'로 위장되던 경로(probe:MLX-10)를 설정 경계에서 막는다."""
    from app.translate.types import TranslateError

    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
           "TRANSLATE_TEMPERATURE": value}
    with pytest.raises(TranslateError, match="TRANSLATE_TEMPERATURE"):
        TranslateConfig.from_env(env)


@pytest.mark.parametrize(("value", "expected"), [
    ("", "0"), ("0", "0"), ("0.7", "0.7"), ("2", "2"), ("NONE", "none"), (" 1.0 ", "1.0"),
])
def test_temperature_유효값은_표기를_보존(value, expected):
    """캐시 키 재료라 유효한 값의 표기는 그대로 둔다(기존 units.json 적중 유지)."""
    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
           "TRANSLATE_TEMPERATURE": value}
    assert TranslateConfig.from_env(env).temperature == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("", True), ("1", True), ("true", True),
    ("0", False), ("false", False), ("False", False), ("OFF", False), ("No", False),
])
def test_translate_context_대소문자_무관하게_끈다(value, expected):
    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m",
           "TRANSLATE_CONTEXT": value}
    assert TranslateConfig.from_env(env).context is expected


def test_numeric_config_defaults_and_clamps_are_preserved():
    env = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m"}
    blank = TranslateConfig.from_env({
        **env, "TRANSLATE_CONCURRENCY": "", "TRANSLATE_TIMEOUT_S": "",
        "TRANSLATE_MAX_RETRIES": "",
    })
    assert (blank.concurrency, blank.timeout_s, blank.max_retries) == (8, 180.0, 3)
    negative = TranslateConfig.from_env({
        **env, "TRANSLATE_CONCURRENCY": "-1", "TRANSLATE_TIMEOUT_S": "-1",
        "TRANSLATE_MAX_RETRIES": "-1",
    })
    assert (negative.concurrency, negative.timeout_s, negative.max_retries) == (1, 5.0, 0)


def test_retry_after_헤더에도_백오프_상한을_적용():
    """"Retry-After: 3600" 한 줄이 워커를 한 시간 묶으면 번역이 멈춘 것처럼 보인다."""
    c = OpenAICompatClient(_cfg())
    assert c._backoff({"Retry-After": "3600"}, 0) == 30.0
    assert c._backoff({"retry-after": "2"}, 0) == 2.0          # 상한 이하면 그대로 따른다
    assert c._backoff({"Retry-After": "-5"}, 0) == 0.0         # 음수 방어
    # HTTP-date 형식은 파싱 실패 → 지수 백오프로 폴백
    assert c._backoff({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 1) == 3.0
    assert c._backoff({}, 5) == 30.0                           # 지수 백오프 상한도 동일


@pytest.mark.parametrize("status", [400, 413, 422])
def test_결정적_4xx는_유닛강등용_예외로_구분(status):
    """엔진이 유닛 하나만 원문 유지로 강등할 수 있도록 재시도 불가 4xx를 구분한다."""
    from app.translate.types import TranslateUnitRejected

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda path, payload: (status, {"error": "context length exceeded"}, {})
    with pytest.raises(TranslateUnitRejected):
        c.complete("s", "u", max_tokens=16)


def test_비결정적_오류는_종전대로_일반_API오류():
    """5xx 소진·인증 실패는 전역 원인 — 유닛 강등 대상이 아니다."""
    from app.translate.types import TranslateUnitRejected

    c = OpenAICompatClient(_cfg(api_mode="chat", max_retries=0))
    c._post = lambda path, payload: (503, "busy", {})
    with pytest.raises(TranslateAPIError) as exc:
        c.complete("s", "u", max_tokens=16)
    assert not isinstance(exc.value, TranslateUnitRejected)

    c2 = OpenAICompatClient(_cfg(api_mode="chat"))
    c2._post = lambda path, payload: (401, "nope", {})
    with pytest.raises(TranslateAPIError) as exc2:
        c2.complete("s", "u", max_tokens=16)
    assert not isinstance(exc2.value, TranslateUnitRejected)
