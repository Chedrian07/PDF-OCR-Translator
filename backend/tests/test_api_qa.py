"""페이지 Q&A + 프로바이더 카탈로그 API 테스트.

LLM 코어는 app.state.llm_router를 FakeRouter로 교체해 검증한다 — Localight
tests/test_api.py의 FakeLlm 덕타입(default_model / async providers / async ask)
을 따른다. OCR 파이프라인은 conftest의 FakeEngine 기반 client 픽스처 그대로.
"""

import json
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


# ── 6c. result.md 페이지 인덱스 == 레이아웃/리더 페이지 번호 ───────────────
def test_qa_page_index_matches_layout_pages(client, sample_pdf, settings):
    """OCR이 본문에 `---` 각주선을 뱉어도 Q&A 페이지 좌표계가 밀리지 않는다.

    Q&A는 result.md를 page_separator로 split한 인덱스로 페이지를 해석하고
    프런트는 그 값을 리더의 실제 페이지 번호로 프리필한다 — 두 좌표계가
    같다는 것이 merge의 무해화로 보장되어야 한다."""
    from app.pipeline.merge import ChunkResult, IncrementalMerger

    fake = FakeRouter()
    client.app.state.llm_router = fake
    jid = _done_job(client, sample_pdf)
    job_dir = settings.jobs_dir / jid

    # 실제 생산자(IncrementalMerger)로 result.md·layout.json을 다시 만든다 —
    # 1쪽 본문에 구분선(`---`)이 들어 있는 실측 OCR 형태.
    chunk = job_dir / "work" / "qa_chunk"
    (chunk / "images").mkdir(parents=True, exist_ok=True)
    (chunk / "raw_pages.json").write_text(
        json.dumps({"pages": [
            f"<|det|>text [60, 100, 940, 300]<|/det|>{n}쪽 본문" for n in (1, 2, 3)
        ]}, ensure_ascii=False), encoding="utf-8",
    )
    merger = IncrementalMerger(job_dir, settings.page_separator)
    merger.add_chunk(ChunkResult(
        chunk, 1, 3,
        "<PAGE>\n1쪽 본문\n\n---\n\n각주 텍스트\n<PAGE>\n2쪽 본문\n<PAGE>\n3쪽 본문",
    ))
    merger.finalize()
    assert merger.warnings == []

    layout = json.loads((job_dir / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == [1, 2, 3]

    for n in (1, 2, 3):
        r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "page": n})
        assert r.status_code == 200, r.text
        context = fake.ask_calls[-1]["context"]
        assert context.startswith(f"[Page {n}]\n")
        assert f"{n}쪽 본문" in context
        # layout 페이지 N의 블록 텍스트와 같은 페이지를 가리킨다
        blocks = " ".join(b.get("content", "") for b in layout[n - 1]["blocks"])
        assert f"{n}쪽 본문" in blocks
        for other in {1, 2, 3} - {n}:
            assert f"{other}쪽 본문" not in context

    # 4쪽은 존재하지 않는다 — 구분선이 페이지를 늘리지 않았다
    assert client.post(
        f"/api/jobs/{jid}/qa", json={"question": "q", "page": 4}
    ).status_code == 422


# ── 7. LlmError → 503 메시지 그대로 전달 ──────────────────────────────────
def test_qa_llm_error_maps_to_503(client, sample_pdf):
    client.app.state.llm_router = FakeRouter(
        ask_error="Ollama is not running. Start it with `ollama serve`."
    )
    jid = _done_job(client, sample_pdf)
    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q"})
    assert r.status_code == 503
    assert "Ollama is not running" in r.json()["detail"]


# ── 7b. 허용목록 위반은 클라이언트 입력 오류(400) — 업스트림 장애(503)와 구분 ──
def test_qa_model_allowlist_violation_maps_to_400(client, sample_pdf):
    """app/llm/providers.py는 입력 거절과 업스트림 장애를 같은 LlmError로 던진다.
    허용목록 위반은 재시도해도 소용없으므로 400이어야 한다(문구는 providers.py 실물)."""
    jid = _done_job(client, sample_pdf)

    for message in (
        "'gpt-9' is not an allowed openai-responses model. Allowed: fake:openai.",
        "'qwen3:cloud' is not an on-device Ollama model. Cloud models are blocked; "
        "install a local model with `ollama pull qwen3:8b`.",
        "Unknown LLM provider: bogus",
        "Unsupported OpenAI provider: openai-legacy",
    ):
        client.app.state.llm_router = FakeRouter(ask_error=message)
        r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "model": "gpt-9"})
        assert r.status_code == 400, (message, r.status_code)
        assert r.json()["detail"] == message

    # 진짜 업스트림 장애는 여전히 503
    client.app.state.llm_router = FakeRouter(ask_error="OpenAI API request failed: timeout")
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q"}).status_code == 503


def test_qa_unknown_provider_rejected_before_llm_call(client, sample_pdf):
    """카탈로그에 없는 프로바이더는 LLM 호출 전에 400 — 라우터의 기본 모델 해석이
    엉뚱한 프로바이더로 흘러가지 않게 한다."""
    fake = FakeRouter()
    client.app.state.llm_router = fake
    jid = _done_job(client, sample_pdf)

    r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q", "provider": "bogus"})
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]
    assert fake.ask_calls == []


# ── 7c. 남용 방어: 잡·IP 단위 레이트리밋 + 동시 실행 상한 → 429 ─────────────
def test_qa_rate_limit_returns_429(tmp_path, sample_pdf):
    """인증이 없으므로 같은 네트워크의 누구나 유료 키를 소진할 수 있다 — 상한 초과는
    429 + Retry-After, 상한 안의 정상 사용은 그대로 통과한다."""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    settings = Settings(
        engine="fake", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, fake_delay=0.0, frontend_dir=tmp_path / "no-frontend",
        qa_rate_limit_per_min=2,
    )
    with TestClient(create_app(settings)) as client:
        client.app.state.llm_router = FakeRouter()
        jid = _done_job(client, sample_pdf)

        for _ in range(2):
            assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q"}).status_code == 200
        r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q"})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1


def test_qa_concurrency_cap_returns_429(client, sample_pdf):
    """동시에 처리 중인 질문 수에도 상한이 있다 — 초과 요청은 대기 없이 429."""
    # 가드는 첫 요청에서 지연 생성되므로 그 전에 Settings 상한만 낮추면 된다
    client.app.state.settings.qa_max_concurrent = 1
    client.app.state.llm_router = FakeRouter()
    jid = _done_job(client, sample_pdf)
    # 첫 요청이 슬롯을 잡은 상태를 재현 (가드는 첫 요청에서 지연 생성된다)
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q"}).status_code == 200

    import app.api as api_mod

    guard = api_mod._abuse_guard(client.app.state, "qa")
    assert guard.acquire()
    try:
        r = client.post(f"/api/jobs/{jid}/qa", json={"question": "q"})
        assert r.status_code == 429
        assert r.headers["retry-after"] == "5"
    finally:
        guard.release()
    # 슬롯이 반환되면 정상 사용 재개
    assert client.post(f"/api/jobs/{jid}/qa", json={"question": "q"}).status_code == 200


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
    # qa_available은 기본 프로바이더의 구성 여부로 판정된다 — 테스트 Settings에는
    # LLM 키가 없으므로 구성된 라우터(FakeRouter.openai.configured=True)를 주입한다.
    client.app.state.llm_router = FakeRouter()
    body = client.get("/api/health").json()
    assert body["qa_available"] is True
    assert body["llm_default_provider"] == "openai-responses"
    # 기존 필드는 그대로 (추가만 정책)
    assert body["status"] == "ok"
    assert "translate_available" in body


def test_health_qa_available_false_without_key(client):
    """미구성 상태에서도 상수 true를 내보내면 README·문서의 '사용 가능 여부'
    안내가 거짓이 된다 — openai-* 기본 프로바이더는 키 유무를 반영한다."""
    # conftest Settings에는 LLM_OPENAI_API_KEY가 없다(실제 라우터 openai.configured=False)
    assert client.get("/api/health").json()["qa_available"] is False

    client.app.state.llm_router = SimpleNamespace(
        openai=SimpleNamespace(configured=True)
    )
    assert client.get("/api/health").json()["qa_available"] is True

    # 로컬 ollama가 기본이면 데몬 상태를 헬스에서 동기 조회할 수 없어 True 유지
    client.app.state.settings.llm_provider = "ollama"
    assert client.get("/api/health").json()["qa_available"] is True
