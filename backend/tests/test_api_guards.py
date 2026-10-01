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
def _req(host: str, xff: str | None = None):
    from types import SimpleNamespace

    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)


def test_forged_forwarded_for_from_a_direct_client_is_ignored(monkeypatch, caplog):
    """nginx를 두고 TRUSTED_PROXY_HOPS=1을 켰는데 백엔드 포트(compose 기본 0.0.0.0)도 열려
    있으면, 직접 붙은 LAN 클라이언트가 요청마다 XFF를 바꿔 IP 레이트리밋을 무력화했다
    (rl_probe: 192.168.1.66이 위조한 10.9.9.0..4가 키 5개로)."""
    import logging

    import app.api as api_mod

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)       # 기본 = 루프백만
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
    assert api_mod._client_key(_req("unknown", "203.0.113.9")) == "unknown"
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "0")                 # 홉 0이면 목록과 무관하게 무시
    assert api_mod._client_key(_req("172.17.0.1", "203.0.113.7")) == "172.17.0.1"
