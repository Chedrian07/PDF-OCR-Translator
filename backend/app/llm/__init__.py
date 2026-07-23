"""Localight에서 이식한 다중 프로바이더 LLM 계층 (OpenAI Responses/Chat + 로컬 Ollama).

번역/Q&A 라우트가 공용으로 쓰는 진입점 — 요청 페이로드 계약은 providers.py,
엔드포인트 allowlist와 라우터 팩토리는 validate.py 참조.
"""

from .providers import (  # noqa: F401
    GenerationResult,
    LlmError,
    LlmRouter,
    ModelInfo,
    OllamaClient,
    OpenAIClient,
)
from .validate import build_router  # noqa: F401

__all__ = [
    "GenerationResult",
    "LlmError",
    "LlmRouter",
    "ModelInfo",
    "OllamaClient",
    "OpenAIClient",
    "build_router",
]
