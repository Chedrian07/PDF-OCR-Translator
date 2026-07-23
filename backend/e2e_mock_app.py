"""브라우저 E2E 전용 앱 진입점 — 외부 Q&A 호출을 메모리 mock으로 대체한다.

프로덕션 `app.main:app`에는 테스트 분기나 우회 키를 넣지 않는다. 이 모듈은
frontend/tests/e2e/mock-full-flow.e2e.mjs만 명시적으로 기동한다.
"""

from __future__ import annotations

from app.llm import GenerationResult
from app.main import create_app


class MockOpenAIResponsesRouter:
    def default_model(self, provider: str) -> str:
        del provider
        return "mock-responses"

    async def providers(self) -> dict:
        return {
            "default_provider": "openai-responses",
            "default_reasoning_effort": "low",
            "providers": [{
                "id": "openai-responses",
                "label": "OpenAI Responses (E2E mock)",
                "available": True,
                "remote": True,
                "supports_reasoning_summary": True,
                "models": ["mock-responses"],
                "default_model": "mock-responses",
            }],
        }

    async def ask(
        self, *, question: str, context: str, provider: str | None,
        model: str | None, reasoning_effort: str, reasoning_summary: str,
        thinking: bool,
    ) -> GenerationResult:
        del question, context, provider, reasoning_summary
        return GenerationResult(
            content="모의 Q&A 응답: 현재 페이지의 핵심을 확인하였다.",
            model=model or "mock-responses",
            provider="openai-responses",
            reasoning_effort=reasoning_effort,
            thinking_requested=thinking,
            reasoning_summary="테스트용 요약",
            usage={"input_tokens": 12, "output_tokens": 9},
        )


app = create_app()
app.state.llm_router = MockOpenAIResponsesRouter()
