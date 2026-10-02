"""HTML 응답 보안 헤더(CSP·Referrer-Policy)와 정적 프런트엔드 캐시 정책.

마크다운 속 `![](https://…)`가 그대로 <img>가 되어 문서를 여는 순간 제3자·LAN으로
요청이 나갔다(감사 frontend-3·pipeline-ocr-4·sidecar-5). CSP는 그 심층 방어이고,
리더는 서버 렌더 HTML을 SPA에 innerHTML로 넣으므로 SPA 문서의 정책이 핵심이다.
정적 파일은 Cache-Control 없이 나가 업그레이드 뒤 옛 ES 모듈이 섞였다(frontend-15).
브라우저 실측(Chromium): 외부 이미지 요청 0건, 주입된 onerror 미실행, 테마 부트스트랩
정상 실행 — 여기서는 그 근거가 되는 헤더 계약을 고정한다. 테마 부트스트랩은 같은 출처
파일(frontend/theme-init.js)이라 'self'로 실행되고, 인라인 스크립트가 다시 생기면 해시로만
허용된다.

SPA에는 정책이 두 겹이다 — 서버 헤더(main.py)와 index.html의 meta. 브라우저는 둘 다
적용하므로(교집합) 한쪽만 고치면 다른 쪽이 조용히 막거나, 고친 줄 알았던 완화가 무효가
된다. 그래서 meta가 표현할 수 있는 지시어는 두 층이 같아야 하고, Referrer 정책도 같다.
"""

import base64
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from conftest import wait_done

REPO = Path(__file__).resolve().parents[2]


def _directives(policy: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


class _InlineScripts(HTMLParser):
    """CSP 해시 대조용 — 정규식이 아니라 HTML 파서로 인라인 스크립트 본문(bodies)과
    외부 스크립트 주소(sources)를 뽑는다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.bodies: list[str] = []
        self.sources: list[str] = []
        self._inline = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = next((value for name, value in attrs if name == "src"), None)
            if src is not None:
                self.sources.append(src)
            self._inline = src is None
            self._buf = []

    def handle_data(self, data):
        if self._inline:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            if self._inline:
                self.bodies.append("".join(self._buf))
            self._inline = False


class _MetaPolicies(HTMLParser):
    """index.html의 meta CSP·meta referrer 값을 모은다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.csp: list[str] = []
        self.referrer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        a = dict(attrs)
        if (a.get("http-equiv") or "").lower() == "content-security-policy":
            self.csp.append(a.get("content") or "")
        elif (a.get("name") or "").lower() == "referrer":
            self.referrer.append(a.get("content") or "")


# meta로는 전달되지 않는(브라우저가 무시하는) 지시어 — 헤더에만 둘 수 있다.
_HEADER_ONLY_DIRECTIVES = {"frame-ancestors", "report-uri", "report-to", "sandbox"}
# 없으면 다른 지시어로 대체되는 fetch 지시어(CSP3 대체 사슬). base-uri·form-action처럼
# 사슬이 없는 지시어는 없으면 '제한 없음'이다.
_FALLBACK = {
    "script-src-elem": ("script-src", "default-src"),
    "script-src-attr": ("script-src", "default-src"),
    "style-src-elem": ("style-src", "default-src"),
    "style-src-attr": ("style-src", "default-src"),
    "worker-src": ("child-src", "script-src", "default-src"),
    "frame-src": ("child-src", "default-src"),
    **{
        name: ("default-src",)
        for name in (
            "child-src", "script-src", "style-src", "img-src", "font-src", "connect-src",
            "media-src", "object-src", "manifest-src",
        )
    },
}


def _effective(policy: dict[str, list[str]], directive: str) -> list[str] | None:
    """지시어의 실효 출처 목록(대체 사슬 반영, 순서 무관). None = 제한 없음."""
    for name in (directive, *_FALLBACK.get(directive, ())):
        if name in policy:
            return sorted(policy[name])
    return None


def _sha256_source(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


@pytest.fixture
def spa_client(settings):
    settings.frontend_dir = REPO / "frontend"
    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        yield client


def test_spa_document_csp_blocks_external_images_but_allows_its_own_scripts(spa_client):
    response = spa_client.get("/")
    assert response.status_code == 200
    policy = _directives(response.headers["content-security-policy"])
    assert policy["img-src"] == ["'self'", "data:", "blob:"]      # 외부 이미지 비컨 차단
    assert policy["default-src"] == ["'self'"]                     # 연결·프레임 등도 같은 출처
    assert "'unsafe-inline'" in policy["style-src"]                # 레이아웃 좌표·KaTeX 인라인 스타일
    assert policy["object-src"] == ["'none'"]
    # 인라인 스크립트는 'unsafe-inline' 없이 해시로만 — 주입된 on* 속성·인라인 스크립트는 막힌다
    assert "'unsafe-inline'" not in policy["script-src"]
    parser = _InlineScripts()
    parser.feed((REPO / "frontend" / "index.html").read_text(encoding="utf-8"))
    # 인라인 스크립트는 정확히 그 해시들만 허용된다(지금은 없다 — 해시 출처도 없어야 한다)
    hashes = {src for src in policy["script-src"] if src.startswith("'sha256-")}
    assert hashes == {_sha256_source(body) for body in parser.bodies}
    # 앱 스크립트(테마 부트스트랩 theme-init.js 포함)는 같은 출처 파일이라 'self'로 실행된다
    assert "'self'" in policy["script-src"]
    assert any(src.endswith("theme-init.js") for src in parser.sources)
    for src in parser.sources:
        parts = urlsplit(src)
        assert not parts.scheme and not parts.netloc, src     # 외부 출처 스크립트 없음
    assert response.headers["referrer-policy"] == "same-origin"


def test_spa_meta_csp_and_referrer_agree_with_the_header(spa_client):
    """SPA 문서에는 서버 헤더 CSP와 index.html meta CSP가 함께 걸린다(실효 정책은 교집합).
    둘이 어긋나면 한쪽 수정이 다른 쪽에 조용히 막힌다 — 예: 헤더에만 출처를 더하면 meta가
    막고, 인라인 스크립트를 넣으면 헤더는 해시를 자동으로 더하지만 meta가 막는다. meta가
    표현할 수 있는 모든 지시어(frame-ancestors 등 헤더 전용 제외)의 실효 출처와 Referrer
    정책이 두 층에서 같아야 한다."""
    response = spa_client.get("/")
    header = _directives(response.headers["content-security-policy"])
    meta = _MetaPolicies()
    meta.feed((REPO / "frontend" / "index.html").read_text(encoding="utf-8"))
    assert len(meta.csp) == 1, meta.csp
    declared = _directives(meta.csp[0])
    assert not _HEADER_ONLY_DIRECTIVES & declared.keys()        # meta에 두면 무시될 뿐이다
    names = (header.keys() | declared.keys()) - _HEADER_ONLY_DIRECTIVES
    # 비교가 공허하지 않게 — 핵심 지시어는 두 층 모두 직접 선언한다
    assert {"default-src", "script-src", "style-src", "img-src"} <= header.keys() & declared.keys()
    mismatched = {
        name: {"header": _effective(header, name), "meta": _effective(declared, name)}
        for name in sorted(names)
        if _effective(header, name) != _effective(declared, name)
    }
    assert mismatched == {}
    # Referrer: meta가 헤더를 덮어쓴다 — 값이 다르면 문서화된 헤더 정책이 거짓말이 된다
    assert meta.referrer == [response.headers["referrer-policy"]]


def test_spa_csp_follows_index_html_changes(settings, tmp_path):
    """index.html의 인라인 스크립트가 바뀌면(배포·개발 중 수정) 재시작 없이 해시도 바뀐다 —
    옛 해시가 남으면 새 테마 부트스트랩이 막힌다."""
    from app.main import create_app

    frontend = tmp_path / "frontend"
    frontend.mkdir()
    index = frontend / "index.html"
    index.write_text("<!doctype html><script>var a = 1;</script>", encoding="utf-8")
    settings.frontend_dir = frontend
    with TestClient(create_app(settings)) as client:
        first = client.get("/").headers["content-security-policy"]
        assert _sha256_source("var a = 1;") in first
        index.write_text("<!doctype html><script>var b = 22;</script>", encoding="utf-8")
        second = client.get("/").headers["content-security-policy"]
        assert _sha256_source("var b = 22;") in second
        assert _sha256_source("var a = 1;") not in second


@pytest.mark.parametrize("path", ["/", "/index.html", "/app.js", "/js/core.js", "/styles.css"])
def test_static_frontend_is_revalidated_not_heuristically_cached(spa_client, path):
    """Cache-Control 없이 Last-Modified만 나가면 브라우저가 휴리스틱 신선도로 파일마다
    다른 시점까지 재검증 없이 재사용한다 — 업그레이드 뒤 옛 모듈과 새 모듈이 섞인다."""
    response = spa_client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    revalidated = spa_client.get(path, headers={"if-none-match": response.headers["etag"]})
    assert revalidated.status_code == 304                 # no-cache = 재검증, 대부분 304
    if not path.endswith(".html") and path != "/":
        assert "content-security-policy" not in response.headers   # 문서가 아닌 정적 파일


@pytest.mark.parametrize(("path", "status"), [
    ("/", 200), ("/app.js", 200), ("/js/core.js", 200), ("/styles.css", 200),
    ("/vendor/katex/katex.min.js", 200), ("/api/health", 200), ("/api/no-such-route", 404),
])
def test_every_response_forbids_mime_sniffing(spa_client, path, status):
    """모든 응답에 X-Content-Type-Options: nosniff — 브라우저가 명시한 Content-Type 밖으로 내용을
    추측해 해석하지 않는다(보안 리뷰 관찰: 헤더가 없었다). ES 모듈은 원래 MIME을 엄격히 보므로
    정적 파일의 JS MIME이 맞는지도 함께 고정한다."""
    response = spa_client.get(path)
    assert response.status_code == status
    assert response.headers["x-content-type-options"] == "nosniff"
    if path.endswith(".js"):
        assert response.headers["content-type"].split(";")[0] in (
            "text/javascript", "application/javascript",
        )


def test_api_html_responses_carry_csp_and_keep_route_headers(client, sample_pdf):
    """API가 내보내는 HTML(조각·내려받기 문서)도 같은 리소스 출처 규칙을 받는다.
    document.html은 KaTeX를 인라인으로 품으므로 스크립트는 'unsafe-inline'이다."""
    jid = client.post(
        "/api/jobs", files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    ).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    for path in ("html", "layout", "document.html"):
        response = client.get(f"/api/jobs/{jid}/{path}")
        assert response.status_code == 200, path
        policy = _directives(response.headers["content-security-policy"])
        assert policy["img-src"] == ["'self'", "data:", "blob:"], path
        assert "'unsafe-inline'" in policy["script-src"], path
        assert response.headers["referrer-policy"] == "same-origin"
    preview = client.post(f"/api/jobs/{jid}/render-preview", content=b"![](https://t.example/p.png)")
    assert "img-src 'self' data: blob:" in preview.headers["content-security-policy"]

    # JSON·조건부 캐시 헤더는 라우트가 정한 그대로다(덮어쓰지 않는다)
    manifest = client.get(f"/api/jobs/{jid}/viewer-manifest")
    assert manifest.headers["cache-control"] == "private, no-cache"
    assert "content-security-policy" not in manifest.headers
    assert "cache-control" not in client.get("/api/health").headers


def test_no_html_page_depends_on_a_cdn_script_blocked_by_the_csp(spa_client):
    """FastAPI 자동 문서(/docs Swagger UI·/redoc ReDoc)는 cdn.jsdelivr.net 스크립트와 인라인 초기화
    스크립트로 그려져 HTML CSP(script-src 'self')에 막혀 빈 페이지만 떴다(P4 Docker 재현). 문서
    페이지는 끄고 기계용 스키마(/openapi.json)는 남긴다."""
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
        response = spa_client.get(path)
        assert response.status_code == 404, path
        assert "cdn.jsdelivr.net" not in response.text, path
    schema = spa_client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["paths"]
    # SPA 문서 자체는 외부 출처 스크립트가 없다(위 테스트) — 정책이 막는 HTML이 남지 않았다
    index = spa_client.get("/")
    assert "cdn.jsdelivr.net" not in index.text
