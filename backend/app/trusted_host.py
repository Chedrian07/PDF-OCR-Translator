"""로컬 실행의 LAN/VPN IP 접속을 허용하는 Host 검증."""

from collections.abc import Sequence
from ipaddress import ip_address

from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


def _matches_server_ip(scope: Scope) -> bool:
    """Host의 IP가 실제 요청을 받은 소켓 주소일 때만 허용한다.

    DNS 조회·client 주소·Forwarded 헤더는 쓰지 않는다. 따라서 다른 사설 IP나
    DNS rebinding 도메인도 자동 허용하지 않는다. 0.0.0.0 바인딩에서도 uvicorn은
    scope['server']에 이 연결의 실제 목적지 IP를 넣는다.
    """
    server = scope.get("server")
    if not server:
        return False
    authority = Headers(scope=scope).get("host", "")
    if authority.startswith("["):
        host, closing, suffix = authority[1:].partition("]")
        if not closing or (suffix and not suffix.startswith(":")):
            return False
        port = suffix[1:] if suffix else None
    else:
        host, separator, value = authority.partition(":")
        port = value if separator else None
    if port is not None and (not port.isascii() or not port.isdigit()):
        return False
    try:
        requested_ip = ip_address(host)
        server_ip = ip_address(server[0])
    except ValueError:
        return False
    return requested_ip == server_ip and not server_ip.is_unspecified


class DirectIPTrustedHostMiddleware(TrustedHostMiddleware):
    """명시한 ALLOWED_HOSTS는 그대로, 미설정 기본값에는 서버 IP 접속만 추가."""

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: Sequence[str],
        allow_server_ip: bool = False,
    ) -> None:
        super().__init__(app, allowed_hosts=allowed_hosts)
        self.allow_server_ip = allow_server_ip

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.allow_server_ip
            and scope["type"] in ("http", "websocket")
            and _matches_server_ip(scope)
        ):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)
