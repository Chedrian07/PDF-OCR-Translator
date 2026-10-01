from .base import (  # noqa: F401
    EngineError,
    JobCanceled,
    OCREngine,
    OutputLimitError,
    RepetitiveOutputError,
    StreamSink,
)


def __getattr__(name: str):
    """build_engine은 처음 찾을 때 임포트한다(PEP 562).

    registry는 config → llm(httpx 등)을 끌어온다. 패키지 초기화에서 바로 임포트하면
    `app.engine.base`만 필요한 곳 — PDF 워커 프로세스의 충실도 분석·textlayer 추출 — 까지
    설정·LLM 계층을 통째로 싣는다(pipeline/pdf_worker.py: 워커는 가벼운 모듈만 임포트한다).
    `from app.engine import build_engine`은 그대로 동작한다."""
    if name == "build_engine":
        from .registry import build_engine

        return build_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
