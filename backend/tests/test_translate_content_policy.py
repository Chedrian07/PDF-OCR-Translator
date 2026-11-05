"""공급자 콘텐츠 정책 거부는 상태코드와 무관하게 그 유닛의 거부다.

LiteLLM 프록시는 OpenAI 추론 모델의 프롬프트 필터 거부를 HTTP 500으로 감싸 돌려준다(실측 본문
그대로). 상태코드만 보면 일시 장애라 재시도 3회 뒤 잡 전체가 실패했다.
"""

import pytest

from app.translate.client import OpenAICompatClient, _stream_error
from app.translate.types import (
    TranslateAPIError,
    TranslateConfig,
    TranslateContentRejected,
    TranslateUnitRejected,
)

# 2026-11-04 운영 로그의 LiteLLM 응답 본문(HTTP 500)
LITELLM_500 = {
    "error": {
        "message": (
            "litellm.BadRequestError: litellm.ContentPolicyViolationError: "
            "ContentPolicyViolationError: ChatgptException - Invalid prompt: your prompt was "
            "flagged as potentially violating our usage policy. Please try again with a different "
            "prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting\n"
            "model=chatgpt/gpt-6-luna. content_policy_fallback=None. fallbacks=None.\n\n"
            "LiteLLM: model group 'chatgpt/gpt-6-luna' failed with the error above. "
            "No fallback was attempted."
        ),
        "type": "internal_server_error",
        "param": None,
        "code": "500",
    }
}
OPENAI_400 = {
    "error": {
        "message": "Invalid prompt: your prompt was flagged as potentially violating our usage policy.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_prompt",
    }
}
AZURE_400 = {
    "error": {
        "message": "The response was filtered due to the prompt triggering Azure OpenAI's content "
        "management policy.",
        "type": None,
        "param": "prompt",
        "code": "content_filter",
    }
}
SUCCESS = {"output_text": "번역문"}


def client(max_retries=3):
    return OpenAICompatClient(TranslateConfig(
        base_url="https://provider.example/v1", api_key="test-key", model="selected-model",
        api_mode="responses", max_retries=max_retries, temperature="none",
    ))


def scripted(engine, replies):
    sent, waits = [], []

    def post(path, payload):
        sent.append(payload)
        return replies[min(len(sent), len(replies)) - 1]

    engine._post = post
    engine._wait_or_cancel = waits.append
    return sent, waits


@pytest.mark.parametrize("status, body", [(500, LITELLM_500), (400, OPENAI_400), (400, AZURE_400)])
def test_a_policy_rejection_is_a_unit_rejection_after_one_immediate_resend(status, body):
    engine = client()
    sent, waits = scripted(engine, [(status, body, {})])

    with pytest.raises(TranslateContentRejected) as caught:
        engine.complete("system", "flagged source", max_tokens=100)

    assert isinstance(caught.value, TranslateUnitRejected)
    assert len(sent) == 2, "정책 거부는 백오프 재시도 3회가 아니라 1회만 다시 보낸다"
    assert waits == []
    assert "콘텐츠 정책" in str(caught.value)


def test_a_policy_rejection_that_passes_on_the_resend_is_a_translation():
    engine = client()
    sent, waits = scripted(engine, [(500, LITELLM_500, {}), (200, SUCCESS, {})])

    assert engine.complete("system", "sometimes flagged", max_tokens=100) == "번역문"
    assert len(sent) == 2 and waits == []


def test_no_resend_when_retries_are_disabled():
    engine = client(max_retries=0)
    sent, _ = scripted(engine, [(500, LITELLM_500, {})])

    with pytest.raises(TranslateContentRejected):
        engine.complete("system", "flagged source", max_tokens=100)
    assert len(sent) == 1


def test_a_plain_server_error_is_still_a_retried_job_error():
    """본문이 정책 거부라고 말하지 않는 5xx는 종전대로 엔드포인트 장애다."""
    engine = client(max_retries=2)
    body = {"error": {"message": "upstream exploded", "type": "internal_server_error", "code": "500"}}
    sent, waits = scripted(engine, [(500, body, {})])

    with pytest.raises(TranslateAPIError) as caught:
        engine.complete("system", "source", max_tokens=100)

    assert not isinstance(caught.value, TranslateUnitRejected)
    assert len(sent) == 3 and len(waits) == 2


def test_a_policy_error_event_inside_a_stream_is_a_unit_rejection():
    error = _stream_error(LITELLM_500["error"])
    assert isinstance(error, TranslateContentRejected)
