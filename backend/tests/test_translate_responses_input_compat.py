"""Gateways may require canonical Responses input instead of shorthand text."""

import concurrent.futures
import threading

import pytest

from app.translate.client import OpenAICompatClient
from app.translate.types import TranslateConfig, TranslateUnitRejected

ERROR = {"error": {"message": 'Provider exception: {"detail":"Input must be a list"}'}}


def client(**overrides):
    config = dict(
        base_url="https://provider.example/v1", api_key="test-key", model="selected-model",
        api_mode="responses", max_retries=0, reasoning="off", reasoning_style="reasoning_effort",
    )
    config.update(overrides)
    return OpenAICompatClient(TranslateConfig(**config))


def test_structured_input_negotiates_once_without_changing_other_payload(caplog):
    engine = client()
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        return (400, ERROR, {}) if isinstance(payload["input"], str) else (200, {"output_text": "성공"}, {})

    engine._post = post
    source = "Public source\nwith exact spacing."
    assert engine.complete("system instructions", source, max_tokens=100) == "성공"
    assert engine.complete("system instructions", source, max_tokens=100) == "성공"
    assert len(sent) == 3
    assert sent[0]["input"] == source
    assert sent[1]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": source}]}]
    assert sent[2] == sent[1]
    assert {key: value for key, value in sent[0].items() if key != "input"} == {
        key: value for key, value in sent[1].items() if key != "input"
    }
    assert sent[1]["reasoning"] == {"effort": "none"}
    assert engine.responses_input_list
    assert sum("input 목록을 요구합니다" in item.message for item in caplog.records) == 1


def test_repeated_structured_input_rejection_does_not_loop():
    engine = client()
    sent = []
    engine._post = lambda path, payload: (sent.append(payload.copy()) or (400, ERROR, {}))
    with pytest.raises(TranslateUnitRejected):
        engine.complete("system", "public source", max_tokens=100)
    assert len(sent) == 2
    assert isinstance(sent[1]["input"], list)


def test_chat_input_errors_do_not_trigger_responses_negotiation():
    engine = client(api_mode="chat", stream="off")
    sent = []
    engine._post = lambda path, payload: (sent.append(payload.copy()) or (400, ERROR, {}))
    with pytest.raises(TranslateUnitRejected):
        engine.complete("system", "public source", max_tokens=100)
    assert len(sent) == 1 and "input" not in sent[0]
    assert not engine.responses_input_list


def test_structured_input_survives_store_fallback_and_later_payload_rebuilds():
    engine = client()
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        if isinstance(payload["input"], str):
            return 400, ERROR, {}
        if "store" in payload:
            return 400, {"error": {"message": "Unsupported parameter: store"}}, {}
        return 200, {"output_text": "성공"}, {}

    engine._post = post
    assert engine.complete("system", "public source", max_tokens=100) == "성공"
    assert engine.complete("system", "next public source", max_tokens=100) == "성공"
    assert len(sent) == 4
    assert all(isinstance(payload["input"], list) for payload in sent[1:])
    assert ["store" in payload for payload in sent] == [True, True, False, False]


def test_shared_client_structured_input_latch_is_monotonic(caplog):
    engine = client()
    initial_requests = threading.Barrier(4, timeout=5)
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        if isinstance(payload["input"], str):
            initial_requests.wait()
            return 400, ERROR, {}
        return 200, {"output_text": "성공"}, {}

    engine._post = post
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda n: engine.complete("system", str(n), max_tokens=100), range(4))) == ["성공"] * 4
    assert engine.complete("system", "later", max_tokens=100) == "성공"
    assert len(sent) == 9 and isinstance(sent[-1]["input"], list)
    assert sum("input 목록을 요구합니다" in item.message for item in caplog.records) == 1
