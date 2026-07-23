"""페이지 Q&A + 프로바이더 카탈로그 API 테스트.

LLM 코어는 app.state.llm_router를 FakeRouter로 교체해 검증한다 — Localight
tests/test_api.py의 FakeLlm 덕타입(default_model / async providers / async ask)
을 따른다. OCR 파이프라인은 conftest의 FakeEngine 기반 client 픽스처 그대로.
"""

from types import SimpleNamespace

from conftest import wait_done

from app.llm import LlmError
from app.llm.providers import GenerationResult

_CATALOG = {
    "default_provider": "openai-responses",
    "default_reasoning_effort": "low",
    "providers": [
        {"id": "openai-responses", "label": "OpenAI Responses", "available": True,
         "remote": True, "supports_reasoning_summary": True,
         "models": ["fake:openai"], "default_model": "fake:openai"},
        {"id": "openai-chat", "label": "OpenAI Chat Completions", "available": True,
         "remote": True, "supports_reasoning_summary": False,
         "models": ["fake:chat"], "default_model": "fake:chat"},
        {"id": "ollama", "label": "Ollama Local", "available": True,
         "remote": False, "supports_reasoning_summary": False,
         "models": ["fake:local"], "default_model": "fake:local"},
    ],
}


class FakeRouter:
    """LlmRouter 덕타입 — api.py가 쓰는 표면만 구현 (Localight FakeLlm 미러)."""

    openai = SimpleNamespace(configured=True)

    def __init__(self, ask_error: str | None = None):
        self.ask_error = ask_error
        self.ask_calls: list[dict] = []

    def default_model(self, provider: str) -> str:
        return "fake:openai" if provider.startswith("openai-") else "fake:local"

    async def providers(self) -> dict:
        return _CATALOG

    async def ask(self, **kwargs) -> GenerationResult:
        if self.ask_error is not None:
            raise LlmError(self.ask_error)
        self.ask_calls.append(kwargs)
        provider = kwargs.get("provider") or "openai-responses"
        return GenerationResult(
            content=f"답변: {kwargs['question']} / {kwargs['context'][:8]}",
            model=kwargs.get("model") or self.default_model(provider),
            provider=provider,
            reasoning_effort=kwargs["reasoning_effort"],
            thinking_requested=kwargs["thinking"],
            reasoning_summary=None,
            usage={"input_tokens": 3},
        )


# ── 헬퍼 ──────────────────────────────────────────────────────────────────
def _done_job(client, sample_pdf) -> str:
    r = client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
        data={"mode": "multi"},
    )
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    return jid


# ── 1. 풀 플로: 업로드→done→질문 (Localight /ask 응답 형태) ────────────────
def test_qa_full_flow(client, sample_pdf):
    fake = FakeRouter()
    client.app.state.llm_router = fake
    jid = _done_job(client, sample_pdf)

    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "핵심은?", "page": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "핵심은?" in body["answer"]
    assert body["provider"] == "openai-responses"     # settings.llm_provider 기본값
    assert body["model"] == "fake:openai"             # router.default_model 해석
    assert body["page"] == 1
    assert body["reasoning_effort"] == "low"          # settings.llm_reasoning_effort 기본값
    assert body["reasoning_summary"] is None
    assert body["thinking_requested"] is True
    assert body["usage"] == {"input_tokens": 3}
    assert body["remote"] is True
    assert body["local_only"] is False

    # 컨텍스트는 '[Page N]' 접두 + 해당 페이지 텍스트만 (단일 페이지 계약)
    call = fake.ask_calls[-1]
    assert call["context"].startswith("[Page 1]\n")
    assert "페이지 1" in call["context"]
    assert "페이지 2" not in call["context"]
    assert call["reasoning_summary"] == "none"

    # page 생략 시 1페이지가 기본
    r2 = client.post(f"/api/jobs/{jid}/qa", json={"question": "첫 페이지?"})
    assert r2.status_code == 200
    assert r2.json()["page"] == 1


# ── 2. 프로바이더 명시 오버라이드 (로컬 경로 local_only) ────────────────────
def test_qa_provider_override_local(client, sample_pdf):
    client.app.state.llm_router = FakeRouter()
    jid = _done_job(client, sample_pdf)

    r = client.post(
        f"/api/jobs/{jid}/qa",
        json={"question": "로컬?", "page": 2, "provider": "ollama",
              "reasoning_effort": "high", "thinking": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "fake:local"
    assert body["page"] == 2
    assert body["reasoning_effort"] == "high"   # 요청값이 서버 기본값보다 우선
    assert body["thinking_requested"] is False
    assert body["remote"] is False
    assert body["local_only"] is True


# ── 3. 409 미완료 잡 (번역과 동일 판정) ────────────────────────────────────
def test_qa_before_done_conflicts(client, sample_pdf):
    client.app.state.llm_router = FakeRouter()
    jid = _done_job(client, sample_pdf)
    # fake_delay=0.0이라 done 전 상태를 실시간으로 잡기 어렵다 — 번역 테스트와
    # 동일하게 비-done 상태를 강제해 결정적으로 검증한다.
    job = client.app.state.store.get(jid)
    old = job.status
    job.status = "running"
    try:
        r = client.post(f"/api/jobs/{jid}/qa", json={"question": "질문"})
        assert r.status_code == 409
        assert "완료" in r.json()["detail"]
    finally:
        job.status = old


# ── 4. 404 없는 잡 ────────────────────────────────────────────────────────
def test_qa_unknown_job(client):
    client.app.state.llm_router = FakeRouter()
    assert client.post("/api/jobs/j_nope/qa", json={"question": "?"}).status_code == 404


# ── 5. 422 페이지 범위 밖 / 스키마 검증 ────────────────────────────────────
def test_qa_page_out_of_range(client, sample_pdf):
    client.app.state.llm_router = FakeRouter()
    jid = _done_job(client, sample_pdf)

    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 99})
    assert r.status_code == 422
    assert "범위" in r.json()["detail"]

    # 스키마 수준 검증: page>=1, question 1–2000자
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 0}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": ""}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "x" * 2001}).status_code == 422


# ── 6. 422 빈 페이지 (범위 안이지만 텍스트 없음) ───────────────────────────
def test_qa_blank_page(client, sample_pdf, settings):
    client.app.state.llm_router = FakeRouter()
    jid = _done_job(client, sample_pdf)
    sep = settings.page_separator
    # 2페이지가 빈 3페이지 문서로 교체 — 분리자 사이가 공백뿐이면 빈 페이지다
    (settings.jobs_dir / jid / "result.md").write_text(
        f"1쪽 텍스트{sep}{sep}3쪽 텍스트", encoding="utf-8"
    )

    ok = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 3})
    assert ok.status_code == 200          # 범위 계산이 3페이지로 유지되는지 확인

    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 2})
    assert r.status_code == 422
    assert "텍스트를 찾지 못했습니다" in r.json()["detail"]


# ── 6b. 선두 빈 페이지 — 페이지 인덱스가 밀리지 않는다 ─────────────────────
def test_qa_leading_blank_page_keeps_indexing(client, sample_pdf, settings):
    """빈 표지 스캔처럼 1쪽이 비어도(merge는 선두 구분자를 보존한다) 2쪽 질문이
    2쪽 텍스트를 받아야 하고, 총 페이지 수도 3으로 유지되어야 한다."""
    fake = FakeRouter()
    client.app.state.llm_router = fake
    jid = _done_job(client, sample_pdf)
    sep = settings.page_separator
    (settings.jobs_dir / jid / "result.md").write_text(
        f"{sep}2쪽 텍스트{sep}3쪽 텍스트\n", encoding="utf-8"
    )

    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 2})
    assert r.status_code == 200, r.text
    assert fake.ask_calls[-1]["context"] == "[Page 2]\n2쪽 텍스트"

    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 3}).status_code == 200
    r1 = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": 1})
    assert r1.status_code == 422              # 범위 안(1-3)이지만 빈 페이지
    assert "텍스트를 찾지 못했습니다" in r1.json()["detail"]


# ── 7. LlmError → 503 메시지 그대로 전달 ──────────────────────────────────
def test_qa_llm_error_maps_to_503(client, sample_pdf):
    client.app.state.llm_router = FakeRouter(
        ask_error="Ollama is not running. Start it with `ollama serve`."
    )
    jid = _done_job(client, sample_pdf)
    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q"})
    assert r.status_code == 503
    assert "Ollama is not running" in r.json()["detail"]


# ── 8. GET /api/providers — 라우터 카탈로그 그대로 ────────────────────────
def test_providers_catalog(client):
    client.app.state.llm_router = FakeRouter()
    r = client.get("/api/providers")
    assert r.status_code == 200
    body = r.json()
    assert body == _CATALOG                     # verbatim 반환 (가공 없음)
    assert {p["id"] for p in body["providers"]} == {
        "openai-responses", "openai-chat", "ollama"
    }


# ── 9. /api/health에 Q&A 필드 (추가만) ────────────────────────────────────
def test_health_reports_qa_fields(client):
    body = client.get("/api/health").json()
    assert body["qa_available"] is True
    assert body["llm_default_provider"] == "openai-responses"
    # 기존 필드는 그대로 (추가만 정책)
    assert body["status"] == "ok"
    assert "translate_available" in body
