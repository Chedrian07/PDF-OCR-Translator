"""클릭재킹 방어 — HTML 응답의 헤더 CSP는 어떤 출처도 이 앱을 프레임에 싣지 못하게 한다.

이 서비스는 인증이 없고(삭제·번역·Q&A 버튼이 곧 권한이다) 기본 바인딩이 0.0.0.0이다.
다른 사이트가 UI를 보이지 않는 iframe에 싣고 클릭을 유도할 수 있었다. frame-ancestors는
<meta> CSP에서는 무시되므로 헤더에만 있다(감사 Phase 1 f2 요청).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import wait_done

REPO = Path(__file__).resolve().parents[2]


def _frame_ancestors(policy: str) -> list[str]:
    for part in policy.split(";"):
        tokens = part.split()
        if tokens and tokens[0] == "frame-ancestors":
            return tokens[1:]
    return []


@pytest.fixture
def spa_client(settings):
    settings.frontend_dir = REPO / "frontend"
    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        yield client


def test_spa_document_cannot_be_framed(spa_client):
    response = spa_client.get("/")
    assert response.status_code == 200
    assert _frame_ancestors(response.headers["content-security-policy"]) == ["'none'"]


def test_api_html_cannot_be_framed(client, sample_pdf):
    jid = client.post(
        "/api/jobs", files={"file": ("a.pdf", sample_pdf, "application/pdf")},
    ).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    for path in ("html", "layout", "document.html"):
        response = client.get(f"/api/jobs/{jid}/{path}")
        assert response.status_code == 200, path
        assert _frame_ancestors(response.headers["content-security-policy"]) == ["'none'"], path
