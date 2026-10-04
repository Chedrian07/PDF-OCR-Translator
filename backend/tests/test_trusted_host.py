"""로컬 uvicorn의 LAN/Tailscale 접속과 명시적 Host 제한 회귀 검사."""

from fastapi.testclient import TestClient
import pytest
from starlette.responses import PlainTextResponse

from app.config import Settings
from app.main import create_app
from app.trusted_host import DirectIPTrustedHostMiddleware


def _with_server(app, server):
    async def wrapper(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, "server": server}
        await app(scope, receive, send)

    return wrapper


@pytest.mark.parametrize("server,host", [
    ("192.168.0.2", "192.168.0.2"),
    ("192.168.0.2", "192.168.0.2:8000"),
    ("100.92.205.98", "100.92.205.98:8000"),
    ("::1", "[::1]:8000"),
    ("fd7a:115c:a1e0::763b:cd62", "[fd7a:115c:a1e0::763b:cd62]"),
])
def test_default_hosts_allow_direct_server_ip(server, host):
    async def app(scope, receive, send):
        await PlainTextResponse("ok")(scope, receive, send)

    middleware = DirectIPTrustedHostMiddleware(
        app, allowed_hosts=["localhost", "127.0.0.1"], allow_server_ip=True,
    )
    client = TestClient(_with_server(middleware, (server, 8000)))
    response = client.get("/", headers={"host": host})
    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.parametrize("server,host", [
    (("100.92.205.98", 8000), "100.124.21.32:8000"),  # 클라이언트의 VPN IP
    (("192.168.0.2", 8000), "192.168.0.3:8000"),
    (("192.168.0.2", 8000), "evil.example.com:8000"),
    (("192.168.0.2", 8000), "192.168.0.2.evil.example.com"),
    (("192.168.0.2", 8000), "user@192.168.0.2:8000"),
    (("192.168.0.2", 8000), "192.168.0.2:bad-port"),
    (("fd7a:115c:a1e0::763b:cd62", 8000), "[fd7a:115c:a1e0::763b:cd62"),
    (("fd7a:115c:a1e0::763b:cd62", 8000), "[fd7a:115c:a1e0::763b:cd62]evil"),
    (("0.0.0.0", 8000), "0.0.0.0:8000"),
    (("::", 8000), "[::]:8000"),
    (None, "192.168.0.2:8000"),
    (("testserver", 8000), "192.168.0.2:8000"),
])
def test_direct_ip_allowance_rejects_other_hosts(server, host):
    async def app(scope, receive, send):
        pytest.fail("rejected Host reached the app")

    middleware = DirectIPTrustedHostMiddleware(
        app, allowed_hosts=["localhost", "127.0.0.1"], allow_server_ip=True,
    )
    client = TestClient(_with_server(middleware, server))
    response = client.get("/", headers={
        "host": host,
        "x-forwarded-host": "192.168.0.2:8000",
        "x-forwarded-for": "192.168.0.2",
        "forwarded": "host=192.168.0.2:8000;for=192.168.0.2",
    })
    assert response.status_code == 400
    assert response.text == "Invalid host header"


@pytest.mark.parametrize("configured_hosts,expected", [
    (None, 200),
    ("", 200),
    ("   ", 200),
    ("localhost,127.0.0.1", 400),
    ("localhost,127.0.0.1,100.92.205.98", 200),
    ("*", 200),
])
def test_app_honors_default_and_explicit_host_policy(monkeypatch, tmp_path, configured_hosts, expected):
    monkeypatch.setenv("OCR_ENGINE", "fake")
    monkeypatch.setenv("PRELOAD_MODEL", "0")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    if configured_hosts is None:
        monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    else:
        monkeypatch.setenv("ALLOWED_HOSTS", configured_hosts)
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>PDF OCR</h1>", encoding="utf-8")
    (frontend / "favicon.ico").write_bytes(b"favicon")
    monkeypatch.setenv("FRONTEND_DIR", str(frontend))

    settings = Settings.from_env()
    assert settings.allow_server_ip_host is (configured_hosts is None or not configured_hosts.strip())
    app = create_app(settings)
    with TestClient(_with_server(app, ("100.92.205.98", 8000))) as client:
        for path in ("/", "/favicon.ico", "/api/health"):
            assert client.get(path, headers={"host": "100.92.205.98:8000"}).status_code == expected
        assert client.get("/api/health", headers={"host": "localhost:8000"}).status_code == 200
        if configured_hosts != "*":
            assert client.get("/", headers={"host": "evil.example.com"}).status_code == 400


def test_directly_constructed_settings_keep_explicit_hosts_strict():
    settings = Settings(allowed_hosts=["localhost", "127.0.0.1"])
    assert settings.allow_server_ip_host is False
