"""클라이언트 — chat/responses 파싱·auto 폴백·재시도·오류·후처리·URL 정규화."""

import pytest

from app.translate.client import OpenAICompatClient, _endpoint_url, _normalize_base_url
from app.translate.types import TranslateAPIError, TranslateConfig


class _FakeResponse:
    """requests.Response 대역 — 클라이언트는 stream=True로 받아 iter_content로 읽는다."""

    status_code = 200
    headers: dict = {}
    payload = b'{"output_text": "ok"}'

    def iter_content(self, chunk_size=None):
        yield self.payload

    def close(self):
        pass


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

    class Response(_FakeResponse):
        headers = {"X-Test": "yes"}

    class Session:
        @staticmethod
        def post(url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

    client = OpenAICompatClient(_cfg(api_mode="responses", timeout_s=timeout_s))
    client.session = Session()
    status, body, headers = client._post("responses", {"input": "x"})

    assert captured["timeout"] == expected
    assert captured["stream"] is True        # 본문 상한을 강제하려면 직접 읽어야 한다
    assert captured["url"] == "https://host/v1/responses"
    assert status == 200 and body == {"output_text": "ok"}
    assert headers["X-Test"] == "yes"


def test_request_semaphore는_여러_잡의_실제_HTTP_동시성을_제한():
    import concurrent.futures as cf
    import threading

    active = 0
    peak = 0
    lock = threading.Lock()
    Response = _FakeResponse

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


@pytest.mark.parametrize("failure", ["timeout", "http400", "empty"])
def test_auto_협상_owner의_유닛_단위_오류는_대기자에게_복제하지_않는다(failure):
    """owner 유닛 하나의 시간 초과·400·빈 출력을 요청을 보내지도 않은 대기 유닛들에 복제해,
    재개 run에서 최대 7개 유닛이 같은 사유로 원문 유지됐다(translate-7). 전역 원인(5xx·
    연결)만 복제하고, 유닛 단위 거부면 대기자는 각자 자기 요청으로 협상한다."""
    import concurrent.futures as cf
    import threading
    import time

    import requests as _requests

    from app.translate.types import TranslateUnitRejected

    workers = 4
    start = threading.Barrier(workers)
    lock = threading.Lock()
    sent = []

    def post(path, payload):
        user = payload.get("input") or payload["messages"][-1]["content"]
        with lock:
            sent.append(user)
        if user == "HUGE":
            time.sleep(0.3)
            if failure == "timeout":
                raise _requests.exceptions.ReadTimeout("Read timed out")
            if failure == "http400":
                return (400, {"error": {"message": "context length exceeded"}}, {})
            return (200, {"output_text": "", "status": "completed"}, {})
        return (200, {"output_text": f"{user} 번역", "status": "completed"}, {})

    client = OpenAICompatClient(_cfg(api_mode="auto"))
    client._backoff = lambda headers, attempt: 0.0
    client._post = post

    def call(unit):
        start.wait()
        if unit != "HUGE":
            time.sleep(0.05)  # owner가 먼저 협상을 잡게 한다
        try:
            return unit, client.complete("s", unit, max_tokens=10)
        except TranslateUnitRejected as e:
            return unit, type(e).__name__

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        results = dict(executor.map(call, ["HUGE", "a", "b", "c"]))

    assert {u: results[u] for u in "abc"} == {u: f"{u} 번역" for u in "abc"}
    assert results["HUGE"] in ("TranslateTimeout", "TranslateUnitRejected", "TranslateEmptyOutput")
    assert {"a", "b", "c"} <= set(sent)            # 대기자도 자기 요청을 실제로 보냈다
    assert client.api_mode_used == "responses"


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


def test_chat_404_JSON_오류본문이면_모델ID를_안내한다():
    """mlx_lm은 모르는 모델 ID에 404 + {"error": …}를 낸다(probe:MLX-07)."""
    c = OpenAICompatClient(_cfg(api_mode="chat", model="mlx-community/Qwen3.5-0.8B-6bit"))
    c._post = lambda p, pl: (404, {"error": "Cannot find an appropriate cached snapshot"}, {})
    with pytest.raises(TranslateAPIError) as exc:
        c.complete("s", "u", max_tokens=100)
    msg = str(exc.value)
    assert "모델 ID" in msg and "mlx-community/Qwen3.5-0.8B-6bit" in msg
    assert "cached snapshot" in msg and "default_model" in msg

    c2 = OpenAICompatClient(_cfg(api_mode="chat"))
    c2._post = lambda p, pl: (404, {"error": {"message": "The model `m` does not exist",
                                              "code": "model_not_found"}}, {})
    with pytest.raises(TranslateAPIError, match="does not exist"):
        c2.complete("s", "u", max_tokens=100)


def test_responses_요청은_store_false를_보낸다():
    """번역 Responses에 store가 없어 OpenAI 30일 보관·oMLX SSD 영속(mlx-integration-2)."""
    c = OpenAICompatClient(_cfg(api_mode="responses"))
    assert c._build_payload("responses", "s", "u", 10)["store"] is False
    assert "store" not in c._build_payload("chat", "s", "u", 10)


def test_store를_거부하는_서버는_빼고_재시도한_뒤_래치한다(caplog):
    import logging

    sent = []

    def post(path, payload):
        sent.append(dict(payload))
        if "store" in payload:
            return (400, {"error": {"message": "Unknown parameter: 'store'."}}, {})
        return (200, {"output_text": "번역"}, {})

    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = post
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        assert c.complete("s", "u", max_tokens=10) == "번역"
    assert ["store" in p for p in sent] == [True, False] and c._store_ok is False
    assert any("store" in r.message for r in caplog.records)
    assert c.complete("s", "u", max_tokens=10) == "번역"
    assert ["store" in p for p in sent] == [True, False, False]   # 이후 store 없이 직행


def test_store_무관한_400은_재시도하지_않는다():
    from app.translate.types import TranslateUnitRejected

    calls = []

    def post(path, payload):
        calls.append(1)
        return (400, {"error": {"message": "context length exceeded"}}, {})

    c = OpenAICompatClient(_cfg(api_mode="responses"))
    c._post = post
    with pytest.raises(TranslateUnitRejected):
        c.complete("s", "u", max_tokens=10)
    assert calls == [1] and c._store_ok is None


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
    """빈 응답은 같은 프롬프트(온도 0)면 반복되는 유닛 단위 거부다(probe:MLX-03)."""
    from app.translate.types import TranslateEmptyOutput

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: (200, {"choices": [{"message": {"content": "   "}}]}, {})
    with pytest.raises(TranslateEmptyOutput, match="빈 응답"):
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


def test_잘림_재시도도_잘리면_잘린_출력을_반환하지_않는다():
    """2배 예산 후에도 잘리면 유닛 단위 거부(TranslateOutputTruncated)로 올린다.

    종전에는 '래더가 흡수'한다며 잘린 출력을 반환했지만, 플레이스홀더 없는 산문은
    래더에 들어가지도 않아 문단 끝이 빠진 번역이 채택·캐시됐다(probe:MLX-02,
    mlx-integration-3). 잘린 출력은 어떤 경로로도 호출자에게 가지 않는다.
    """
    from app.translate.types import TranslateOutputTruncated, TranslateUnitRejected

    seq = iter([
        (200, {"choices": [{"message": {"content": "A"}, "finish_reason": "length"}]}, {}),
        (200, {"choices": [{"message": {"content": "AB"}, "finish_reason": "length"}]}, {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with pytest.raises(TranslateOutputTruncated, match="잘렸습니다") as exc:
        c.complete("s", "u", max_tokens=100)
    assert isinstance(exc.value, TranslateUnitRejected)   # 엔진의 유닛 강등 경로


def test_잘림_재시도가_4xx면_잘림으로_보고한다():
    """2배 예산이 서버 상한을 넘어 400이 와도 그 유닛의 원인은 잘림이다."""
    from app.translate.types import TranslateOutputTruncated

    seq = iter([
        (200, {"choices": [{"message": {"content": "잘린 절반"}, "finish_reason": "length"}]}, {}),
        (400, "max_tokens too large", {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with pytest.raises(TranslateOutputTruncated, match="2배 재시도 거부"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_재시도_4xx_첫출력도_비면_thinking_안내():
    """첫 출력이 전부 잘렸으면(thinking이 예산 소진) 서버측 끄기 안내를 담는다."""
    from app.translate.types import TranslateOutputTruncated

    seq = iter([
        (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, {}),
        (400, "bad request", {}),
    ])
    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = lambda p, pl: next(seq)
    with pytest.raises(TranslateOutputTruncated, match="thinking"):
        c.complete("s", "u", max_tokens=100)


def test_잘림_재시도가_연결실패면_전역_오류로_전파():
    """연결 실패·5xx 소진은 엔드포인트 문제라 유닛 강등 대상이 아니다."""
    import requests as _requests

    from app.translate.types import TranslateUnitRejected

    calls = []

    def post(p, pl):
        calls.append(1)
        if len(calls) == 1:
            return (200, {"choices": [{"message": {"content": "잘린 절반"},
                                       "finish_reason": "length"}]}, {})
        raise _requests.ConnectionError("down")

    c = OpenAICompatClient(_cfg(api_mode="chat", max_retries=0))
    c._post = post
    with pytest.raises(TranslateAPIError, match="연결 실패") as exc:
        c.complete("s", "u", max_tokens=100)
    assert not isinstance(exc.value, TranslateUnitRejected)


def test_잘린_출력이_반복_루프면_2배_재시도를_생략한다(caplog):
    """온도 0 greedy 루프는 2배 예산도 끝까지 태운다(probe:MLX-05) — 왕복 1회로 끝낸다."""
    import logging

    from app.translate.types import TranslateOutputTruncated

    calls = []

    def post(p, pl):
        calls.append(pl["max_tokens"])
        return (200, {"choices": [{"message": {"content": "불필요한 영역을" + "만" * 400},
                                   "finish_reason": "length"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        with pytest.raises(TranslateOutputTruncated, match="반복 루프"):
            c.complete("s", "u" * 50, max_tokens=8192)
    assert calls == [8192]
    assert any("2배 재시도 생략" in r.message for r in caplog.records)


def test_잘린_출력이_입력보다_지나치게_길면_2배_재시도를_생략한다():
    from app.translate.types import TranslateOutputTruncated

    calls = []
    words = " ".join(f"단어{i}" for i in range(1200))   # 반복 아님 · 입력의 4배 초과

    def post(p, pl):
        calls.append(1)
        return (200, {"choices": [{"message": {"content": words},
                                   "finish_reason": "length"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    c._post = post
    with pytest.raises(TranslateOutputTruncated, match="4배"):
        c.complete("s", "[번역할 원문]\n" + "Short source sentence." * 5, max_tokens=8192)
    assert calls == [1]


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


def test_잘림_max_tokens_param_none이면_재시도_없이_잘림으로_보고(caplog):
    """max_tokens를 안 보내는 설정에선 재시도해도 같은 요청 — 1회로 끝낸다.

    종전에는 잘린 출력을 경고 없이 반환했다. mlx_lm은 이때 서버 기본 512토큰에서
    잘라 '만만만…' 루프가 그대로 캐시됐다(probe:MLX-02) — 이제 잘림으로 보고한다.
    """
    import logging

    from app.translate.types import TranslateOutputTruncated

    calls = []

    def post(p, pl):
        calls.append(1)
        return (200, {"choices": [{"message": {"content": "부분 출력"},
                                   "finish_reason": "length"}]}, {})

    c = OpenAICompatClient(_cfg(api_mode="chat", max_tokens_param="none"))
    c._post = post
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        with pytest.raises(TranslateOutputTruncated, match="서버 상한"):
            c.complete("s", "u", max_tokens=100)
    assert len(calls) == 1
    assert any("TRANSLATE_MAX_TOKENS_PARAM=none" in r.message for r in caplog.records)


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


def test_연결_실패_문구는_URL과_쿼리를_담지_않는다(caplog):
    """requests 예외 문구에는 요청 URL(쿼리 포함)·호스트·포트가 그대로 들어 있어, 무인증
    /translate/state·SSE로 base URL의 쿼리 자격증명과 내부 주소가 노출됐다(security-2)."""
    import logging
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                           # 닫힌 포트 — 연결 거부
    c = OpenAICompatClient(_cfg(base_url=f"http://127.0.0.1:{port}/v1?api-key=sk-SECRET-IN-QUERY",
                                api_mode="chat", max_retries=0))
    with caplog.at_level(logging.WARNING, logger="app.translate.client"):
        with pytest.raises(TranslateAPIError, match="연결 실패") as exc:
            c.complete("s", "u", max_tokens=10)
    msg = str(exc.value)
    assert "연결 거부" in msg                             # 원인은 남긴다
    for leak in ("sk-SECRET", "api-key", "127.0.0.1", str(port), "/v1"):
        assert leak not in msg
    ours = [r.getMessage() for r in caplog.records if r.name == "app.translate.client"]
    assert ours and not any("sk-SECRET" in m for m in ours)   # 로그도 쿼리 값은 가린다


def test_연결_실패_원인은_고정_문구로만_요약한다():
    import socket

    import requests as _requests

    err = _requests.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='llm.internal.corp', port=8443): Max retries exceeded "
        "with url: /v1/chat/completions?api-key=sk-x")
    err.__cause__ = socket.gaierror(8, "nodename nor servname provided, or not known")

    def post(path, payload):
        raise err

    c = OpenAICompatClient(_cfg(api_mode="chat", max_retries=0))
    c._post = post
    with pytest.raises(TranslateAPIError, match="호스트 이름") as exc:
        c.complete("s", "u", max_tokens=10)
    assert "llm.internal.corp" not in str(exc.value) and "sk-x" not in str(exc.value)


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
    # translate-9 — OpenAI 데이터 레지던시 호스트와 사설 이름 규칙(종전엔 전부 openrouter)
    ("https://eu.api.openai.com/v1", "reasoning_effort"),
    ("https://us.api.openai.com/v1", "reasoning_effort"),
    ("http://host.containers.internal:8080/v1", "chat_template_kwargs"),   # Podman
    ("http://gateway.docker.internal:8080/v1", "chat_template_kwargs"),
    ("http://llm.corp.internal/v1", "chat_template_kwargs"),
    ("http://mlx.localhost:8080/v1", "chat_template_kwargs"),
    ("http://studio.lan:1234/v1", "chat_template_kwargs"),
    ("http://nas.home.arpa:8000/v1", "chat_template_kwargs"),
    ("https://notapi.openai.com.example.net/v1", "openrouter"),   # 접미사만 닮은 공개 호스트
    ("https://evil-lan.com/v1", "openrouter"),
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


def test_request_variant는_extra_body_원문_대신_해시만_싣는다():
    """request_variant는 state.json에 기록돼 무인증 /translate/state로 나간다 — 원문 JSON을
    실으면 게이트웨이 키·테넌트 토큰 같은 TRANSLATE_EXTRA_BODY 값이 노출됐다(translate-10).
    캐시 키 재료로는 해시로 충분하다(값이 바뀌면 키도 바뀐다)."""
    secret = _cfg(extra_body='{"metadata":{"tenant_token":"tt-SECRET"},"top_k":20}')
    variant = secret.request_variant
    assert variant.startswith("extra_body=sha256:")
    assert "tt-SECRET" not in variant and "top_k" not in variant
    assert variant == _cfg(extra_body=secret.extra_body).request_variant      # 결정적
    assert variant != _cfg(extra_body='{"top_k":20}').request_variant         # 값마다 다르다


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


# ── 스트리밍·취소·응답 상한 — 실제 HTTP 서버(프로세스 내, 임시 포트)로 검증 ──────
# (translate-llm-4, probe:MLX-04, mlx-integration-5, concurrency-10,
#  gap2-dependency-vuln-reachability-4)

import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import json as _json  # noqa: E402
import pathlib  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
_SRC_PROMPT = "[번역할 원문]\nThe model is fast and the results are good."


def _load_mock():
    spec = importlib.util.spec_from_file_location("_client_mock_llm", _SCRIPTS / "mock_llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def mock_llm():
    mod = _load_mock()
    with _serve(mod.Handler) as base:
        yield mod, base


class _Quiet(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return _json.loads(self.rfile.read(n) or b"{}")

    def _json(self, code: int, obj) -> None:
        raw = _json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _sse_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


def _chunk(content=None, finish=None) -> bytes:
    delta = {} if content is None else {"content": content}
    obj = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {_json.dumps(obj)}\n\n".encode()


def test_chat은_기본으로_SSE_스트리밍을_조립한다(mock_llm):
    mod, base = mock_llm
    c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat"))
    sent = []
    real_post = c.session.post

    def spy(url, **kw):
        sent.append(kw["json"])
        return real_post(url, **kw)

    c.session.post = spy
    out = c.complete("s", _SRC_PROMPT, max_tokens=100)
    assert out == mod._translate("The model is fast and the results are good.")
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert c._stream_ok is True and mod.STATS["stream_chunks"] > 3


def test_mlx_lm식_소문자_헤더와_keepalive_주석도_스트림으로_읽는다():
    """mlx_lm.server는 'Content-type'(소문자 t)을 보내고 prefill 동안 ': keepalive' 주석을
    먼저 쓴다 — 실서버 검증에서 SSE 판정이 빗나가 파싱 실패가 났던 형태."""

    class MlxLike(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b": keepalive 555/590\n\n: keepalive 590/590\n\n")
            self.wfile.write(_chunk("번역") + _chunk("문입니다", "stop") + b"data: [DONE]\n\n")

    with _serve(MlxLike) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat"))
        assert c.complete("s", "u", max_tokens=10) == "번역문입니다"
    assert OpenAICompatClient(_cfg())._backoff({"retry-after": "2"}, 0) == 2.0


def test_responses와_stream_0은_비스트리밍이다(mock_llm):
    _mod, base = mock_llm
    for cfg in (_cfg(base_url=f"{base}/v1", api_mode="responses"),
                _cfg(base_url=f"{base}/v1", api_mode="chat", stream="off")):
        c = OpenAICompatClient(cfg)
        assert c.complete("s", _SRC_PROMPT, max_tokens=100)
        mode = "responses" if cfg.api_mode == "responses" else "chat"
        assert "stream" not in c._build_payload(mode, "s", "u", 10, stream=c._use_stream(mode))


def test_스트림의_finish_reason_length는_잘림으로_처리한다(mock_llm):
    from app.translate.types import TranslateOutputTruncated

    _mod, base = mock_llm
    c = OpenAICompatClient(_cfg(base_url=f"{base}/v1?finish=length", api_mode="chat"))
    with pytest.raises(TranslateOutputTruncated):
        c.complete("s", _SRC_PROMPT, max_tokens=100)


def test_스트림의_reasoning은_본문에_섞이지_않는다(mock_llm):
    mod, base = mock_llm
    c = OpenAICompatClient(_cfg(base_url=f"{base}/v1?reasoning=300", api_mode="chat"))
    out = c.complete("s", _SRC_PROMPT, max_tokens=100)
    assert "생각" not in out
    assert out == mod._translate("The model is fast and the results are good.")


def test_스트리밍_취소는_즉시_반환하고_서버_생성을_멈춘다(mock_llm):
    """취소가 진행 중 HTTP 종료(최대 180초)까지 밀리고 서버는 끊긴 요청도 끝까지
    생성하던 문제(concurrency-10, probe:MLX-04) — 연결을 끊어 둘 다 막는다."""
    from app.translate.client import _RequestCancelled

    mod, base = mock_llm
    cancel = threading.Event()
    c = OpenAICompatClient(
        _cfg(base_url=f"{base}/v1?delay=0.05&chunk=2", api_mode="chat"),
        cancel_check=cancel.is_set,
    )
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(_RequestCancelled):
        c.complete("s", "[번역할 원문]\n" + "The model is fast. " * 60, max_tokens=100)
    assert time.monotonic() - t0 < 1.5
    deadline = time.monotonic() + 3
    while mod.STATS["stream_aborted"] < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert mod.STATS["stream_aborted"] == 1          # 서버가 끊김을 보고 생성을 멈췄다
    sent = mod.STATS["stream_chunks"]
    time.sleep(0.3)
    assert mod.STATS["stream_chunks"] == sent        # 더 이상 생성하지 않는다


def test_비스트리밍_요청도_취소되면_응답을_기다리지_않는다():
    from app.translate.client import _RequestCancelled

    class Slow(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            time.sleep(3)
            self._json(200, {"output_text": "늦은 응답"})

    with _serve(Slow) as base:
        cancel = threading.Event()
        c = OpenAICompatClient(
            _cfg(base_url=f"{base}/v1", api_mode="responses"), cancel_check=cancel.is_set,
        )
        threading.Timer(0.2, cancel.set).start()
        t0 = time.monotonic()
        with pytest.raises(_RequestCancelled):
            c.complete("s", "u", max_tokens=10)
        assert time.monotonic() - t0 < 1.0


def test_응답_본문_상한을_넘으면_읽기를_멈춘다():
    """번역 응답에 애플리케이션 수준 크기 상한이 없었다(gap2-…-4)."""
    big = b"x" * (1024 * 1024 + 10)

    class Declared(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(50 * 1024 * 1024))
            self.end_headers()
            self.wfile.write(b"{}")  # 선언만 크다 — 읽기 전에 거절돼야 한다

    class Endless(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.wfile.write(big)

    class EndlessStream(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self._sse_head()
            with contextlib.suppress(OSError):
                self.wfile.write(b"data: " + big)   # 개행 없는 거대한 한 줄

    for handler, mode in ((Declared, "responses"), (Endless, "responses"),
                          (EndlessStream, "chat")):
        with _serve(handler) as base:
            c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode=mode,
                                        max_response_mb=1, max_retries=0))
            with pytest.raises(TranslateAPIError, match="상한"):
                c.complete("s", "u", max_tokens=10)


def test_길이_없는_HTTP_1_0_SSE도_상한에서_바로_읽기를_멈춘다():
    """mlx_lm.server 형식(HTTP/1.0, 길이·청크 없음) — 종전에는 iter_content(None)이 EOF까지
    통째로 읽은 뒤에야 상한을 검사해, 끝없이 보내는 서버에 메모리가 전송량만큼 늘었다
    (translate-1). 위 EndlessStream은 1MB만 쓰고 닫아 사후 검사로도 통과했다."""
    total = 64 * 1024 * 1024
    line = b": " + b"p" * (64 * 1024 - 3) + b"\n"   # SSE 주석 줄 64KB
    sent = {"n": 0}
    finished = threading.Event()

    class Http10Stream(_Quiet):
        protocol_version = "HTTP/1.0"

        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.end_headers()
            try:
                while sent["n"] < total:
                    self.wfile.write(line)
                    sent["n"] += len(line)
            except OSError:
                pass
            finally:
                finished.set()

    with _serve(Http10Stream) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat",
                                    max_response_mb=1, max_retries=0))
        with pytest.raises(TranslateAPIError, match="상한"):
            c.complete("s", "u", max_tokens=10)
        assert finished.wait(10)
    # 소켓 버퍼 몫의 여유만 허용한다 — 종전에는 64MB를 끝까지 받았다.
    assert sent["n"] < 16 * 1024 * 1024


def test_SSE도_선언된_Content_Length가_상한을_넘으면_읽기_전에_거절한다():
    class DeclaredStream(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(50 * 1024 * 1024))
            self.end_headers()
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.wfile.write(_chunk("번역", "stop") + b"data: [DONE]\n\n")

    with _serve(DeclaredStream) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat",
                                    max_response_mb=1, max_retries=0, timeout_s=5))
        with pytest.raises(TranslateAPIError, match="상한"):
            c.complete("s", "u", max_tokens=10)


def test_압축된_SSE는_풀린_크기로_조금씩_읽어_상한을_건다():
    """iter_content(None)은 urllib3의 압축 해제 상한(max_length)도 꺼, 수십 KB짜리 gzip
    한 덩이가 상한 검사 전에 통째로 풀렸다(255KB → +257MB 실측, translate-1)."""
    import gzip
    import tracemalloc

    bomb = gzip.compress(b"data: " + b"a" * (64 * 1024 * 1024), compresslevel=9)

    class GzipStream(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(bomb)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(bomb)

    with _serve(GzipStream) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat",
                                    max_response_mb=1, max_retries=0))
        tracemalloc.start()
        try:
            with pytest.raises(TranslateAPIError, match="상한"):
                c.complete("s", "u", max_tokens=10)
            _now, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < 16 * 1024 * 1024   # 풀린 64MB를 한 번에 메모리에 올리지 않는다


def test_개행_없는_긴_SSE_줄도_선형_시간에_상한까지_읽는다():
    """종전 파서는 조각마다 누적 버퍼를 복사·재분할해(pending += chunk; split) 개행 없는
    줄이 길어질수록 CPU가 제곱으로 늘었다(8MB에 9초, 데이터가 계속 와 read timeout도
    걸리지 않음, translate-8). 작은 조각으로 상한까지 흘려도 금방 끝나야 한다."""

    class Trickle:
        headers: dict = {}

        def iter_content(self, chunk_size=None):
            yield b"data: "
            piece = b"x" * 256
            while True:
                yield piece

    c = OpenAICompatClient(_cfg(api_mode="chat", max_response_mb=4))
    t0 = time.process_time()
    with pytest.raises(TranslateAPIError, match="상한"):
        c._read_sse(Trickle(), {}, threading.Event())
    assert time.process_time() - t0 < 1.0   # 제곱 시간이면 4MB/256B 조각에 수 초가 걸린다


def test_SSE_줄_분할은_조각_경계와_CRLF에_무관하다():
    events =(_chunk("가나") + _chunk("다", "stop")).replace(b"\n", b"\r\n") + b"data: [DONE]\r\n\r\n"

    class Pieces:
        headers: dict = {}

        def __init__(self, size):
            self.size = size

        def iter_content(self, chunk_size=None):
            for i in range(0, len(events), self.size):
                yield events[i:i + self.size]

    c = OpenAICompatClient(_cfg(api_mode="chat"))
    for size in (1, 2, 3, 7, len(events)):
        body = c._read_sse(Pieces(size), {}, threading.Event())
        assert body["choices"][0]["message"]["content"] == "가나다"
        assert body["choices"][0]["finish_reason"] == "stop"


def _http_chunk(data: bytes) -> bytes:
    return b"%x\r\n%s\r\n" % (len(data), data)


def _keepalive_sse_server(conns: set, *, stall_after_done: float = 0.0, chunked: bool = True):
    """HTTP/1.1 keep-alive SSE — chunked(vLLM·llama.cpp·LM Studio 형식) 또는 Content-Length."""
    parts = (_chunk("안녕"), _chunk("하세요", "stop"), b"data: [DONE]\n\n")

    class KeepAliveSSE(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            conns.add(self.client_address)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if not chunked:
                body = b"".join(parts)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in parts:
                self.wfile.write(_http_chunk(part))
                self.wfile.flush()
            if stall_after_done:
                time.sleep(stall_after_done)       # [DONE] 뒤에도 스트림을 열어 둔다
            with contextlib.suppress(OSError):
                self.wfile.write(b"0\r\n\r\n")

    return KeepAliveSSE


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("helper_thread", [False, True])
def test_길이가_정해진_SSE는_DONE_뒤_본문_끝을_읽어_연결을_재사용한다(helper_thread, chunked):
    """[DONE]에서 바로 반환하면 requests가 덜 읽힌 응답의 소켓을 닫아, 원격 HTTPS에서는
    유닛마다 TCP+TLS를 새로 맺었다(10요청 → 연결 10개, translate-4)."""
    conns: set = set()
    with _serve(_keepalive_sse_server(conns, chunked=chunked)) as base:
        c = OpenAICompatClient(
            _cfg(base_url=f"{base}/v1", api_mode="chat"),
            cancel_check=(lambda: False) if helper_thread else None,
        )
        for _ in range(5):
            assert c.complete("s", "u", max_tokens=10) == "안녕하세요"
    assert len(conns) == 1


def test_DONE_뒤_스트림을_닫지_않는_서버에도_오래_막히지_않는다():
    conns: set = set()
    with _serve(_keepalive_sse_server(conns, stall_after_done=5.0)) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", timeout_s=30))
        t0 = time.monotonic()
        assert c.complete("s", "u", max_tokens=10) == "안녕하세요"
        assert time.monotonic() - t0 < 3.0       # 소진은 짧게 시도하고 포기한다


def test_스트리밍을_거부하는_서버는_비스트리밍으로_래치한다():
    seen = []

    class NoStream(_Quiet):
        def do_POST(self):  # noqa: N802
            body = self._body()
            seen.append(bool(body.get("stream")))
            if body.get("stream"):
                self._json(400, {"error": {"message": "Unrecognized field: stream_options"}})
                return
            self._json(200, {"choices": [{"message": {"content": "번역"},
                                          "finish_reason": "stop"}]})

    with _serve(NoStream) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat"))
        assert c.complete("s", "u", max_tokens=10) == "번역"
        assert c._stream_ok is False and seen == [True, False]
        assert c.complete("s", "u", max_tokens=10) == "번역"
        assert seen == [True, False, False]                 # 이후는 비스트리밍 직행


def test_stream_1이면_거부돼도_폴백하지_않는다():
    from app.translate.types import TranslateUnitRejected

    class NoStream(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self._json(400, {"error": {"message": "stream not supported"}})

    with _serve(NoStream) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", stream="on"))
        with pytest.raises(TranslateUnitRejected, match="HTTP 400"):
            c.complete("s", "u", max_tokens=10)


def test_응답_정지는_1회만_재시도하고_유닛_단위_시간초과로_보고한다():
    """토큰 사이 정지가 timeout_s를 넘으면 ReadTimeout — 같은 긴 생성의 반복이라 재시도는
    1회뿐이고, 소진되면 '연결 실패'(잡 전체) 대신 TranslateTimeout(유닛 단위)이다."""
    from app.translate.types import TranslateTimeout, TranslateUnitRejected

    hits = []

    class Stall(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            hits.append(1)
            self._sse_head()
            with contextlib.suppress(OSError):
                self.wfile.write(_chunk("앞부분"))
                self.wfile.flush()
                time.sleep(2)

    with _serve(Stall) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", timeout_s=0.4))
        c._backoff = lambda headers, attempt: 0.0
        with pytest.raises(TranslateTimeout, match="TRANSLATE_TIMEOUT_S") as exc:
            c.complete("s", "u", max_tokens=10)
    assert isinstance(exc.value, TranslateUnitRejected)
    assert len(hits) == 2                       # 최초 1 + 재시도 1 (max_retries=3이어도)


def test_완료_신호_없이_끊긴_스트림은_연결_오류로_재시도한다():
    hits = []

    class Cut(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            hits.append(1)
            self._sse_head()
            self.wfile.write(_chunk("잘린"))
            if len(hits) > 1:
                self.wfile.write(_chunk("완전한 번역", "stop") + b"data: [DONE]\n\n")

    with _serve(Cut) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", max_retries=1))
        c._backoff = lambda headers, attempt: 0.0
        assert c.complete("s", "u", max_tokens=10) == "잘린완전한 번역"
    assert len(hits) == 2


def test_스트림_중간의_오류_이벤트는_API_오류로_올린다():
    class Err(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            self._sse_head()
            self.wfile.write(_chunk("부분"))
            self.wfile.write(b'data: {"error": {"message": "model crashed"}}\n\n')

    with _serve(Err) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", max_retries=0))
        with pytest.raises(TranslateAPIError, match="스트림 오류"):
            c.complete("s", "u", max_tokens=10)


def _flaky_stream_server(first_events: bytes, hits: list):
    """첫 요청만 스트림 중간 오류(first_events)로 끝나고, 이후 요청은 정상 스트림이다."""

    class Flaky(_Quiet):
        def do_POST(self):  # noqa: N802
            self._body()
            hits.append(1)
            self._sse_head()
            if len(hits) == 1:
                self.wfile.write(_chunk("부분") + first_events)
                return
            self.wfile.write(_chunk("완전한 번역", "stop") + b"data: [DONE]\n\n")

    return Flaky


_OPENROUTER_502 = (
    b'data: {"error": {"code": 502, "message": "Provider returned error"}, '
    b'"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}\n\n'
)


@pytest.mark.parametrize("events", [
    _OPENROUTER_502,                                                        # OpenRouter 형식
    b'data: {"error": {"code": "503", "message": "overloaded"}}\n\n',       # 문자열 코드
    b'data: {"error": {"message": "upstream hiccup"}}\n\n',                 # 코드 없음
    _chunk(None, "error"),                                                  # 오류 객체 없이 finish=error
], ids=["openrouter-502", "string-503", "no-code", "finish-error"])
def test_스트림_중간의_일시_오류_이벤트는_재시도한다(events):
    """OpenRouter·LiteLLM은 스트림 시작 뒤의 공급자 장애를 200 본문 안의 오류 이벤트로
    보낸다 — 비스트리밍이면 502로 와 재시도됐을 일시 장애 한 번에 잡 전체가 실패했다
    (translate-5). finish_reason=error만 오면 부분 출력을 번역문으로 돌려줬다."""
    hits: list = []
    with _serve(_flaky_stream_server(events, hits)) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", max_retries=2))
        c._backoff = lambda headers, attempt: 0.0
        assert c.complete("s", "u", max_tokens=10) == "완전한 번역"
    assert len(hits) == 2


def test_스트림_중간의_입력_거부_오류는_유닛_단위로_올린다():
    from app.translate.types import TranslateUnitRejected

    hits: list = []
    events = (b'data: {"error": {"code": 400, "type": "BadRequestError", '
              b'"message": "context length exceeded"}}\n\n')
    with _serve(_flaky_stream_server(events, hits)) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", max_retries=2))
        with pytest.raises(TranslateUnitRejected, match="스트림 오류"):
            c.complete("s", "u", max_tokens=10)
    assert len(hits) == 1                       # 결정적 거부 — 같은 요청을 반복하지 않는다


def test_스트림_중간의_인증_오류는_전역_오류로_올린다():
    from app.translate.types import TranslateUnitRejected

    hits: list = []
    events = b'data: {"error": {"code": 401, "message": "invalid api key"}}\n\n'
    with _serve(_flaky_stream_server(events, hits)) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat", max_retries=2))
        with pytest.raises(TranslateAPIError, match="인증 실패") as exc:
            c.complete("s", "u", max_tokens=10)
    assert not isinstance(exc.value, TranslateUnitRejected) and len(hits) == 1


def test_2배_재시도_응답이_상한을_넘으면_잡_오류가_아니라_잘림이다():
    """xhigh 예산(81,920토큰)의 2배 재시도 사고 스트림(~53MB)은 기본 상한 32MB를 넘는다 —
    종전에는 상한 오류(TranslateAPIError)가 잘림 처리를 우회해 잡 전체가 실패했다
    (translate-6). 원인은 여전히 잘림이므로 유닛 단위 거부여야 한다."""
    from app.translate.types import TranslateOutputTruncated

    seen = []
    thought = _json.dumps({"choices": [{"index": 0, "delta": {"reasoning": "생각" * 50},
                                        "finish_reason": None}]})
    event = f"data: {thought}\n\n".encode()

    class Thinker(_Quiet):
        def do_POST(self):  # noqa: N802
            budget = self._body()["max_tokens"]
            seen.append(budget)
            self._sse_head()
            with contextlib.suppress(OSError):
                for _ in range(4 if budget <= 100 else 20_000):   # 재시도는 상한(1MB)을 넘는다
                    self.wfile.write(event)
                self.wfile.write(_chunk(None, "length") + b"data: [DONE]\n\n")

    with _serve(Thinker) as base:
        c = OpenAICompatClient(_cfg(base_url=f"{base}/v1", api_mode="chat",
                                    max_response_mb=1, max_retries=0))
        with pytest.raises(TranslateOutputTruncated, match="thinking"):
            c.complete("s", "u", max_tokens=100)
    assert seen == [100, 200]


def test_stream_설정_검증():
    from app.translate.types import TranslateError

    base = {"OPENAI_BASE_URL": "https://h/v1", "OPENAI_MODEL": "m"}
    assert TranslateConfig.from_env(base).stream == "auto"
    assert TranslateConfig.from_env({**base, "TRANSLATE_STREAM": "0"}).stream == "off"
    assert TranslateConfig.from_env({**base, "TRANSLATE_STREAM": "1"}).stream == "on"
    assert TranslateConfig.from_env(base).max_response_mb == 32
    for name, value in (("TRANSLATE_STREAM", "maybe"), ("TRANSLATE_MAX_RESPONSE_MB", "0"),
                        ("TRANSLATE_MAX_RESPONSE_MB", "abc")):
        with pytest.raises(TranslateError, match=name):
            TranslateConfig.from_env({**base, name: value})
