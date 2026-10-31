"""Explicit unsupported sampling parameters negotiate once without changing reasoning."""

import concurrent.futures
import json
import threading

import pytest

from app.translate.client import OpenAICompatClient
from app.translate.types import TranslateAPIError, TranslateConfig, TranslateUnitRejected


REJECTION = {
    "error": {
        "message": "Selected model doesn't support temperature=0.0 while reasoning is active. "
        "Only temperature=1 is supported unless reasoning.effort resolves to none."
    }
}


def client(mode="responses", **extra):
    config = dict(
        base_url="https://provider.example/v1", api_key="test-key", model="selected-model",
        api_mode=mode, max_retries=0, reasoning="low", reasoning_style="reasoning_effort",
    )
    config.update(extra)
    return OpenAICompatClient(TranslateConfig(**config))


def success(mode):
    return (
        {"output_text": "성공"} if mode == "responses"
        else {"choices": [{"message": {"content": "성공"}, "finish_reason": "stop"}]}
    )


@pytest.mark.parametrize("mode", ["chat", "responses"])
@pytest.mark.parametrize("rejection", [
    REJECTION,
    {"error": {"message": "Unsupported value: 'temperature' does not support 0 with this model."}},
    {"error": {"param": "temperature", "code": "unsupported_parameter", "message": None}},
])
def test_unsupported_temperature_is_omitted_and_latched(mode, rejection, caplog):
    engine = client(mode)
    sent = []

    def post(path, payload):
        sent.append((path, payload.copy()))
        return (400, rejection, {}) if "temperature" in payload else (200, success(mode), {})

    engine._post = post
    assert engine.complete("system", "first", max_tokens=100) == "성공"
    assert engine.complete("system", "second", max_tokens=100) == "성공"
    assert len(sent) == 3
    assert ["temperature" in payload for _, payload in sent] == [True, False, False]
    reasoning_key = "reasoning" if mode == "responses" else "reasoning_effort"
    expected_reasoning = {"effort": "low"} if mode == "responses" else "low"
    assert all(payload[reasoning_key] == expected_reasoning for _, payload in sent)
    assert all(payload["model"] == "selected-model" for _, payload in sent)
    assert len(set(path for path, _ in sent)) == 1
    assert engine.cfg.temperature == "0" and engine.cfg.reasoning == "low"
    assert sum("temperature를 지원하지 않습니다" in item.message for item in caplog.records) == 1


@pytest.mark.parametrize("mode", ["chat", "responses"])
def test_repeated_temperature_rejection_does_not_loop_or_probe_other_fields(mode):
    engine = client(mode)
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        return 400, REJECTION, {}

    engine._post = post
    with pytest.raises(TranslateUnitRejected):
        engine.complete("system", "text", max_tokens=100)
    assert len(sent) == 2 and "temperature" not in sent[-1]


@pytest.mark.parametrize("message", [
    "Invalid model ID: temperature",
    "Model 'temperature-tuned' does not support tools",
    "temperature must be between 0 and 2",
    "Maximum context length exceeded",
])
def test_other_400_errors_do_not_negotiate_temperature(message):
    engine = client(stream="off")
    sent = []
    engine._post = lambda path, payload: (sent.append(payload.copy()) or (400, {"error": {"message": message}}, {}))
    with pytest.raises(TranslateUnitRejected):
        engine.complete("system", "text", max_tokens=100)
    assert len(sent) == 1 and not engine._temperature_disabled


@pytest.mark.parametrize("mode", ["chat", "responses"])
def test_reported_active_reasoning_must_not_hide_explicit_off(mode):
    engine = client(mode, reasoning="off", reasoning_style="openrouter")
    sent = []
    engine._post = lambda path, payload: (sent.append(payload.copy()) or (400, REJECTION, {}))
    with pytest.raises(TranslateAPIError, match="reasoning=off"):
        engine.complete("system", "text", max_tokens=100)
    assert len(sent) == 1 and not engine._temperature_disabled
    assert sent[0]["reasoning"] == {"enabled": False}


def test_temperature_fallback_keeps_a_compatible_explicit_reasoning_off():
    engine = client(reasoning="off")
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        return (
            (400, {"error": {"message": "Unsupported parameter: temperature"}}, {})
            if "temperature" in payload else (200, success("responses"), {})
        )

    engine._post = post
    assert engine.complete("system", "text", max_tokens=100) == "성공"
    assert all(payload["reasoning"] == {"effort": "none"} for payload in sent)


def test_extra_body_cannot_reintroduce_a_rejected_temperature():
    engine = client(temperature="none", extra_body='{"temperature":0.2}')
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        return (400, REJECTION, {}) if "temperature" in payload else (200, success("responses"), {})

    engine._post = post
    assert engine.complete("system", "first", max_tokens=100) == "성공"
    assert engine.complete("system", "second", max_tokens=100) == "성공"
    assert ["temperature" in payload for payload in sent] == [True, False, False]


def test_shared_client_temperature_latch_is_sticky_across_inflight_requests(caplog):
    engine = client()
    initial_requests = threading.Barrier(4, timeout=5)
    sent = []

    def post(path, payload):
        sent.append(payload.copy())
        if "temperature" in payload:
            initial_requests.wait()
            return 400, REJECTION, {}
        return 200, success("responses"), {}

    engine._post = post
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda n: engine.complete("system", str(n), max_tokens=100), range(4))) == ["성공"] * 4
    assert engine.complete("system", "later", max_tokens=100) == "성공"
    assert len(sent) == 9 and "temperature" not in sent[-1]
    assert sum("temperature를 지원하지 않습니다" in item.message for item in caplog.records) == 1


def test_effective_omission_and_notice_are_persisted_in_translation_state(tmp_path):
    from app.translate.engine import run_translation

    engine = client(context=False)
    (tmp_path / "result.md").write_text("This document explains the model in detail.\n", encoding="utf-8")

    def post(path, payload):
        if "temperature" in payload:
            return 400, REJECTION, {}
        if isinstance(payload["input"], str):
            return 400, {"error": {"message": "Input must be a list"}}, {}
        output = (
            "[]" if "용어집 편집자" in payload.get("instructions", "")
            else "이 문서는 모델을 자세히 설명합니다."
        )
        return 200, {"output_text": output}, {}

    engine._post = post
    result = run_translation(tmp_path, "ko", engine.cfg, client=engine)
    assert result.status == "done"
    for name in ("state.json", "report.json"):
        state = json.loads((tmp_path / "translations/ko" / name).read_text())
        assert state["temperature"] == "0" and state["temperature_omitted"] is True
        assert state["responses_input_list"] is True
        assert all(warning in state["warnings"] for warning in engine.compatibility_warnings)
