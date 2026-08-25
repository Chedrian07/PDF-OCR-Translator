"""페이지 Q&A (Localight 이식) — 요청 스키마 + 페이지 컨텍스트 헬퍼.

라우트는 api.py의 POST /jobs/{job_id}/qa가 담당한다. 컨텍스트는 result.md를
page_separator로 나눈 해당 페이지 텍스트 하나뿐이다(문서 전체 RAG 아님) —
Localight의 프라이버시 계약("답변은 현재 페이지의 텍스트만 사용")을 유지한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Localight app/schemas.py의 어휘 그대로 — 'default'는 프로바이더 페이로드에서 생략됨
ReasoningEffort = Literal[
    "default", "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
ReasoningSummary = Literal["none", "auto", "concise", "detailed"]


class AskRequest(BaseModel):
    """Localight AskRequest 계약 (question 1–2000자, page 1-based)."""

    question: str = Field(min_length=1, max_length=2_000)
    page: int | None = Field(default=None, ge=1)  # None이면 1페이지
    provider: str | None = None                   # None이면 settings.llm_provider
    model: str | None = None                      # None이면 router.default_model(provider)
    # Localight 스키마 기본은 'low'였지만 여기서는 서버 설정(LLM_REASONING_EFFORT)이
    # 기본값의 단일 출처다 — None이면 settings.llm_reasoning_effort로 해석한다.
    reasoning_effort: ReasoningEffort | None = None
    reasoning_summary: ReasoningSummary = "none"
    thinking: bool = True


def get_page_context(job_dir: Path, page: int, page_separator: str) -> tuple[int, str]:
    """result.md를 page_separator로 나눠 (총 페이지 수, 해당 페이지 텍스트)를 돌려준다.

    page는 1-based. /html의 doc-page 분할과 동일한 규칙(구분자 단순 split)이며,
    구분자 잔여물(양끝 개행·공백)은 페이지별로 strip한다. 범위를 벗어나면 텍스트는
    빈 문자열 — 422 판정은 라우트 몫이다. result.md가 없거나 비어 있으면 (0, "").

    "split 인덱스 == 리더의 페이지 번호"는 생산자 쪽에서 강제된다: merge의
    `_clean()`이 페이지 본문에서 구분자와 리터럴로 충돌하는 줄(OCR이 뱉는 `---`
    각주선 등)을 무해화하고, finalize가 "N페이지 = 구분자 N-1개"를 검증한다.
    """
    md_path = Path(job_dir) / "result.md"
    text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    if not text.strip():
        return 0, ""
    pages = [part.strip() for part in text.split(page_separator)]
    if not (1 <= page <= len(pages)):
        return len(pages), ""
    return len(pages), pages[page - 1]
