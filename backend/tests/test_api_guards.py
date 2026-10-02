"""남용 방어(레이트리밋·동시 실행 상한) 회귀 테스트.

인증이 없는 서비스라 같은 네트워크의 누구나 비용이 드는 라우트를 부를 수 있다.
여기서는 가드 자체의 판정 규칙과, 가드가 붙은 라우트의 HTTP 계약(429 + Retry-After)을
함께 고정한다.
"""

import threading
import time

from conftest import wait_done


def _upload(client, pdf_bytes: bytes):
    return client.post(
        "/api/jobs", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )


# ── api-jobs-9: 여러 키를 원자적으로 판정한다 ─────────────────────────────────
def test_request_rejected_by_its_ip_key_does_not_consume_job_buckets():
    """IP 한도를 다 쓴 클라이언트 X가 계속 보낸 요청은 전부 429인데도 잡 버킷에 기록돼,
    새 클라이언트 Y까지 그 잡에서 429를 받았다(rl_probe 재현). X가 남에게 줄 수 있는
    영향은 자기 IP 한도분뿐이어야 한다."""
    import pytest
    from fastapi import HTTPException

    import app.api as api_mod

    guard = api_mod._AbuseGuard(3, 0)
    for _ in range(3):                                   # X가 잡 A에서 IP 한도 소진
        guard.check_rate(("job:A", "ip:X"))
    for _ in range(5):                                   # X → 잡 B: 전부 IP 키에서 거절
        with pytest.raises(HTTPException) as excinfo:
            guard.check_rate(("job:B", "ip:X"))
        assert excinfo.value.status_code == 429
        assert int(excinfo.value.headers["Retry-After"]) >= 1
    for _ in range(3):                                   # 잡 B 버킷은 비어 있다
        guard.check_rate(("job:B", "ip:Y"))
    with pytest.raises(HTTPException):
        guard.check_rate(("job:B", "ip:Z"))              # 이제 잡 B 자체 한도


def test_weighted_hits_and_retry_after():
    import app.api as api_mod

    limiter = api_mod._SlidingWindowLimiter(10, window=60.0)
    assert limiter.hit_many(("k",), cost=4) is None
    assert limiter.hit_many(("k",), cost=4) is None
    retry = limiter.hit_many(("k",), cost=4)             # 8 + 4 > 10
    assert retry is not None and 1.0 <= retry <= 60.0
    assert limiter.hit_many(("k",), cost=2) is None      # 거절된 몫은 기록되지 않았다
    assert limiter.hit_many(("other",), cost=11) == 60.0  # 상한보다 큰 몫은 창 전체
    assert api_mod._SlidingWindowLimiter(0).hit_many(("k",), cost=99) is None   # 비활성


def test_retry_after_is_rounded_up_so_waiting_that_long_is_enough():
    """api-3: Retry-After를 버림(int)해 실제 남은 창보다 짧게 알렸다 — 0.4초 뒤 두 번째
    요청이 '59'를 받고, 59초를 쉬고 다시 보내면 창이 0.6초 남아 429를 한 번 더 받았다."""
    import pytest
    from fastapi import HTTPException

    import app.api as api_mod

    guard = api_mod._AbuseGuard(1, 0)
    hits = guard.limiter._hits
    guard.check_rate(("k",))
    hits["k"] = [t - 0.4 for t in hits["k"]]              # 첫 요청이 0.4초 전이었다
    with pytest.raises(HTTPException) as excinfo:
        guard.check_rate(("k",))
    retry_after = int(excinfo.value.headers["Retry-After"])
    assert retry_after == 60                              # 남은 59.6초 → 올림
    hits["k"] = [t - retry_after for t in hits["k"]]      # 안내받은 만큼 기다렸다
    guard.check_rate(("k",))                              # 이번에는 통과한다


# ── security-1: /render-preview 레이트리밋·동시 렌더 상한 ───────────────────────
def test_render_preview_rate_limit_is_weighted_by_body_size(client, sample_pdf, monkeypatch):
    """라이브 미리보기의 잦은 작은 요청은 통과하고, 상한 크기 본문을 쏟아붓는 요청은
    잡·IP 버킷을 빠르게 소진해 429 + Retry-After를 받는다."""
    import app.api as api_mod

    monkeypatch.setattr(api_mod, "_PREVIEW_UNITS_PER_MIN", 20)   # 가드는 첫 사용 때 생긴다
    jid = _upload(client, sample_pdf).json()["job_id"]
    url = f"/api/jobs/{jid}/render-preview"
    big = b"x" * api_mod._PREVIEW_MAX_BYTES                # 16단위
    assert client.post(url, content=big).status_code == 200
    busy = client.post(url, content=big)                  # 16 + 16 > 20
    assert busy.status_code == 429, busy.text
    assert int(busy.headers["Retry-After"]) >= 1
    for _ in range(4):                                    # 작은 본문(1단위)은 남은 몫으로 통과
        assert client.post(url, content="# 라이브 페이지".encode()).status_code == 200
    assert client.post(url, content=b"# one more").status_code == 429


def test_render_preview_concurrency_is_capped(client, sample_pdf, monkeypatch):
    """렌더는 GIL을 쥔 CPU 작업이다 — 한 클라이언트가 스레드풀을 렌더로 채우지 못하게
    동시 렌더 수를 묶는다(초과분은 즉시 429, 기다리며 스레드를 물지 않는다)."""
    import app.api as api_mod

    monkeypatch.setattr(api_mod, "_PREVIEW_MAX_CONCURRENT", 1)
    jid = _upload(client, sample_pdf).json()["job_id"]
    wait_done(client, jid)
    entered = threading.Event()
    release = threading.Event()
    real_render = api_mod.render_markdown_html

    def _slow_render(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_render(*args, **kwargs)

    monkeypatch.setattr(api_mod, "render_markdown_html", _slow_render)
    url = f"/api/jobs/{jid}/render-preview"
    first: dict = {}
    thread = threading.Thread(
        target=lambda: first.update(status=client.post(url, content=b"# a").status_code),
        daemon=True,
    )
    thread.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        second = client.post(url, content=b"# b")
        assert second.status_code == 429, second.text
        assert second.headers["Retry-After"] == "1"
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
        thread.join(10)
    assert first["status"] == 200
    assert client.post(url, content=b"# c").status_code == 200   # 슬롯이 반납됐다


# ── api-jobs-10: X-Forwarded-For는 신뢰 프록시가 붙인 것만 믿는다 ─────────────────
def _req(host: str, *xff: str, port: int = 40123):
    """직접 연결 피어(host:port)와 X-Forwarded-For 필드 줄들 — 실제 TCP 피어라 포트는 0이 아니다."""
    from types import SimpleNamespace

    from starlette.datastructures import Headers

    raw = [(b"x-forwarded-for", line.encode("latin-1")) for line in xff]
    return SimpleNamespace(client=SimpleNamespace(host=host, port=port), headers=Headers(raw=raw))


def test_forged_forwarded_for_from_a_direct_client_is_ignored(monkeypatch, caplog):
    """nginx를 두고 TRUSTED_PROXY_HOPS=1을 켰는데 백엔드 포트(compose 기본 0.0.0.0)도 열려
    있으면, 직접 붙은 LAN 클라이언트가 요청마다 XFF를 바꿔 IP 레이트리밋을 무력화했다
    (rl_probe: 192.168.1.66이 위조한 10.9.9.0..4가 키 5개로)."""
    import logging

    import app.api as api_mod

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)       # 기본 = 루프백만
    # 경고는 프로세스당 한 번이다 — 앞서 돈 테스트(예: 아래 CIDR 테스트의 127.0.0.1 피어)가
    # 이미 남겼으면 여기서 다시 나오지 않아 실행 순서에 따라 실패했다. 메모를 비우고 본다.
    monkeypatch.setattr(api_mod, "_trusted_proxy_warned", set())
    with caplog.at_level(logging.WARNING, logger="app.api"):
        keys = {api_mod._client_key(_req("192.168.1.66", f"10.9.9.{i}")) for i in range(5)}
    assert keys == {"192.168.1.66"}                              # 위조 무시 → 피어 IP 한 버킷
    assert any("TRUSTED_PROXY_IPS" in r.getMessage() for r in caplog.records)

    # 같은 호스트의 프록시(루프백)가 붙인 헤더는 믿는다 — 가장 흔한 배치는 설정 없이 동작
    assert api_mod._client_key(_req("127.0.0.1", "203.0.113.7")) == "203.0.113.7"
    assert api_mod._client_key(_req("::1", "203.0.113.8")) == "203.0.113.8"
    assert api_mod._client_key(_req("::ffff:127.0.0.1", "203.0.113.9")) == "203.0.113.9"


def test_trusted_proxy_ips_accepts_addresses_and_cidrs(monkeypatch):
    import app.api as api_mod

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "172.17.0.0/16, 10.1.2.3, not-an-ip")
    assert api_mod._client_key(_req("172.17.0.1", "203.0.113.7")) == "203.0.113.7"   # 도커 브리지
    assert api_mod._client_key(_req("10.1.2.3", "203.0.113.8")) == "203.0.113.8"
    assert api_mod._client_key(_req("127.0.0.1", "203.0.113.9")) == "127.0.0.1"      # 명시하면 기본 대체
    # IP가 아닌 피어(유닉스 소켓 등 로컬 전송)는 원격 클라이언트일 수 없다 — 그 앞 프록시는 믿는다
    assert api_mod._client_key(_req("unknown", "203.0.113.9")) == "203.0.113.9"
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "0")                 # 홉 0이면 목록과 무관하게 무시
    assert api_mod._client_key(_req("172.17.0.1", "203.0.113.7")) == "172.17.0.1"


# ── api-2: uvicorn 기본 ProxyHeadersMiddleware가 이미 소비한 루프백 홉 ───────────────
def _key_behind_uvicorn(peer: str, *xff: str) -> tuple[tuple, str]:
    """uvicorn 기본 구성이 앱을 감싸는 ProxyHeadersMiddleware(127.0.0.1)를 실제로 거쳐
    _client_key를 부른다 → (앱이 본 client, 키). 위 _req는 미들웨어를 건너뛰어, 실배포의
    루프백 피어가 앱에 그대로 보인다고 잘못 가정했다."""
    import asyncio

    from starlette.requests import Request
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    import app.api as api_mod

    seen: dict = {}

    async def _app(scope, receive, send):
        seen["client"] = tuple(scope["client"])
        seen["key"] = api_mod._client_key(Request(scope))

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/", "raw_path": b"/", "query_string": b"", "root_path": "",
        "headers": [(b"x-forwarded-for", line.encode("latin-1")) for line in xff],
        "client": (peer, 51234), "server": ("127.0.0.1", 8000),
    }
    asyncio.run(ProxyHeadersMiddleware(_app, trusted_hosts="127.0.0.1")(scope, None, None))
    return seen["client"], seen["key"]


def test_uvicorn_wraps_the_app_and_rewrites_loopback_peers_by_default(monkeypatch):
    """아래 테스트들의 전제 — `uvicorn app.main:app`(Dockerfile·make dev)은 --no-proxy-headers
    없이 뜨고, 그 기본 구성은 앱을 127.0.0.1 피어를 믿는 ProxyHeadersMiddleware로 감싼다."""
    from uvicorn.config import Config
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    async def _app(scope, receive, send):
        return None

    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    config = Config(app=_app, log_config=None)
    config.load()
    assert isinstance(config.loaded_app, ProxyHeadersMiddleware)
    assert "127.0.0.1" in config.loaded_app.trusted_hosts
    assert "192.168.1.66" not in config.loaded_app.trusted_hosts


def test_hops_count_the_loopback_proxy_that_uvicorn_already_consumed(monkeypatch, caplog):
    """api-2: uvicorn이 루프백 피어의 client를 XFF 항목으로 바꿔 넘기는데 앱이 그것을 직접
    피어로 봐 홉을 두 번 셌다 — CDN + 같은 호스트 nginx(HOPS=2)는 같은 엣지를 거친 모든
    사용자가 엣지 IP 한 버킷을 나눠 썼고, nginx만(HOPS=1)은 실제 클라이언트를 '목록 밖
    피어'로 지목하는 거짓 경고를 남겼다."""
    import logging

    import app.api as api_mod

    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)            # 기본 = 루프백
    monkeypatch.setattr(api_mod, "_trusted_proxy_warned", set())      # 경고는 프로세스당 한 번

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    client, key = _key_behind_uvicorn("127.0.0.1", "203.0.113.5, 198.51.100.10")
    assert client == ("198.51.100.10", 0)        # uvicorn이 루프백 홉을 이미 소비했다
    assert key == "203.0.113.5"
    assert _key_behind_uvicorn("127.0.0.1", "203.0.113.77, 198.51.100.10")[1] == "203.0.113.77"
    # uvicorn이 바꾸지 않은 요청(--no-proxy-headers·::1 피어)과 같은 답이다
    assert api_mod._client_key(_req("127.0.0.1", "203.0.113.5, 198.51.100.10")) == "203.0.113.5"

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    with caplog.at_level(logging.WARNING, logger="app.api"):
        assert _key_behind_uvicorn("127.0.0.1", "6.6.6.6, 192.168.1.50")[1] == "192.168.1.50"
    assert not [r for r in caplog.records if "TRUSTED_PROXY_IPS" in r.getMessage()]


def test_proxy_rewrite_handling_keeps_forged_headers_out(monkeypatch, caplog):
    """서버의 판정을 따르는 것이 위조 경로를 새로 열지 않는다 — uvicorn이 믿지 않는 LAN 피어는
    그대로 피어 IP다. 새 줄을 덧붙이는 프록시(HAProxy option forwardfor) 뒤에서는 클라이언트가
    보낸 첫 줄이 아니라 줄 전체의 체인으로 센다(첫 줄만 읽어 위조 값을 키로 썼다)."""
    import logging

    import app.api as api_mod

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)
    monkeypatch.setattr(api_mod, "_trusted_proxy_warned", set())
    with caplog.at_level(logging.WARNING, logger="app.api"):
        client, key = _key_behind_uvicorn("192.168.1.66", "10.9.9.1")
    assert client == ("192.168.1.66", 51234) and key == "192.168.1.66"
    assert any("TRUSTED_PROXY_IPS" in r.getMessage() for r in caplog.records)

    assert api_mod._client_key(_req("127.0.0.1", "6.6.6.6", "192.168.1.50")) == "192.168.1.50"
    assert _key_behind_uvicorn("127.0.0.1", "6.6.6.6", "192.168.1.50")[1] == "192.168.1.50"
    # 포트 0은 서버가 헤더로 바꾼 표식일 뿐 — XFF가 없으면 그 피어를 그대로 쓴다
    assert api_mod._client_key(_req("192.168.1.66", port=0)) == "192.168.1.66"
