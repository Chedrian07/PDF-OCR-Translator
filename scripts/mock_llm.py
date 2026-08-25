"""OpenAI 호환 목 서버 — 실제 과금 없이 번역 파이프라인 전체를 실행하기 위한 하네스.

두 가지 모드를 한 프로세스에서 제공한다:
  - 정상 모드(기본): 입력 텍스트를 결정적으로 "번역"한다. 마스킹 플레이스홀더
    (<m1 .../> 형태)는 **그대로 보존**하고 영문 단어만 한글로 치환하므로,
    복원 단계·layout 정렬·PDF 조판까지 실제 경로가 전부 돈다.
  - 결함 주입 모드: ?fault=refusal|echo|drop_placeholder|http400|http429
    쿼리 또는 FAULT 환경변수로 A-1(출력 검증) 회귀를 실증한다.

Responses API(/v1/responses)와 Chat Completions(/v1/chat/completions)를 모두 지원한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 호출 계측 — D-1(이중 번역) 검증에 쓴다.
STATS = {"calls": 0, "by_text": {}}
_LOCK = threading.Lock()

PLACEHOLDER_RE = re.compile(r"<m\d+\b[^>]*/>")

# 결정적 사전 — 실제 번역기가 아니라 "영문이 한글로 바뀌었다"는 신호를 만드는 것이 목적.
_DICT = {
    "the": "그", "and": "그리고", "of": "의", "in": "에서", "to": "으로",
    "we": "우리는", "our": "우리의", "this": "이", "that": "그", "is": "이다",
    "are": "이다", "for": "위한", "with": "함께", "model": "모델",
    "models": "모델들", "data": "데이터", "results": "결과", "result": "결과",
    "method": "방법", "methods": "방법론", "experiment": "실험",
    "experiments": "실험들", "figure": "그림", "table": "표", "page": "페이지",
    "abstract": "초록", "introduction": "서론", "conclusion": "결론",
    "performance": "성능", "training": "학습", "accuracy": "정확도",
    "language": "언어", "image": "이미지", "text": "텍스트", "document": "문서",
}


def _translate(text: str) -> str:
    """플레이스홀더를 보존한 채 영문 토큰만 한글로 바꾼다."""
    parts = []
    last = 0
    for m in PLACEHOLDER_RE.finditer(text):
        parts.append(("t", text[last:m.start()]))
        parts.append(("p", m.group(0)))
        last = m.end()
    parts.append(("t", text[last:]))

    out = []
    for kind, chunk in parts:
        if kind == "p":
            out.append(chunk)
            continue

        def sub(word: re.Match[str]) -> str:
            w = word.group(0)
            lo = w.lower()
            if lo in _DICT:
                return _DICT[lo]
            # 사전에 없는 영단어는 **결정적 한글 의사단어**로 바꾼다.
            # 실제 한국어 번역문의 한글 밀도를 재현해야 출력 검증 게이트를
            # 현실적으로 통과한다(라틴 문자를 남기면 거부문으로 오인된다).
            n = max(2, min(4, len(w) // 2))
            h = 0
            for ch in lo:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            out_chars = []
            for i in range(n):
                # 가(0xAC00) ~ 힣(0xD7A3) 범위에서 결정적으로 고른다.
                out_chars.append(chr(0xAC00 + (h >> (i * 5)) % 11172))
            return "".join(out_chars)

        out.append(re.sub(r"[A-Za-z]{2,}", sub, chunk))
    return "".join(out)


SOURCE_MARKER = "[번역할 원문]\n"


def _source_only(prompt: str) -> str:
    """프롬프트에서 실제 번역 대상 본문만 뽑는다.

    prompts.py는 [원문 유지]/[용어집]/[직전 문맥] 섹션을 앞에 붙이고 마지막에
    "[번역할 원문]\\n{masked_src}"를 둔다. 실제 LLM은 본문만 번역해 돌려주므로
    목도 동일하게 동작해야 파이프라인(복원·정렬·조판)이 현실적으로 검증된다.
    """
    idx = prompt.rfind(SOURCE_MARKER)
    if idx == -1:
        return prompt
    return prompt[idx + len(SOURCE_MARKER):]


def _payload_text(body: dict) -> str:
    """Responses / Chat 양쪽에서 사용자 입력 텍스트를 뽑는다."""
    if "input" in body:
        inp = body["input"]
        if isinstance(inp, str):
            return inp
        chunks = []
        for item in inp:
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        chunks.append(c["text"])
        return "\n".join(chunks)
    msgs = body.get("messages") or []
    return "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")


def _apply_fault(fault: str, src: str) -> str | None:
    if fault == "refusal":
        return "I cannot translate this text."
    if fault == "refusal_ko":
        return "죄송합니다, 번역할 수 없습니다."
    if fault == "echo":
        return src
    if fault == "summary":
        return "요약입니다."
    if fault == "drop_placeholder":
        return PLACEHOLDER_RE.sub("", _translate(src))
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # 조용히
        pass

    def _send(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/__stats"):
            with _LOCK:
                self._send(200, STATS)
            return
        if self.path.startswith("/__reset"):
            with _LOCK:
                STATS["calls"] = 0
                STATS["by_text"] = {}
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "bad json"}})
            return

        fault = os.environ.get("FAULT", "")
        if "fault=" in self.path:
            fault = self.path.split("fault=", 1)[1].split("&")[0]

        src = _source_only(_payload_text(body))
        with _LOCK:
            STATS["calls"] += 1
            # 중복 계측은 **번역 대상 본문** 기준 — 용어집·문맥 프리픽스는 유닛마다
            # 달라지므로 프롬프트 전체를 키로 쓰면 이중 번역을 놓친다.
            STATS["by_text"][src] = STATS["by_text"].get(src, 0) + 1

        if fault == "http400":
            self._send(400, {"error": {"message": "injected deterministic 400"}})
            return
        if fault == "http429":
            self.send_response(429)
            self.send_header("Retry-After", "86400")  # Retry-After 상한 검증용
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return

        out = _apply_fault(fault, src)
        if out is None:
            out = _translate(src)

        if self.path.startswith("/v1/responses") or "responses" in self.path:
            self._send(200, {
                "id": "resp_mock", "object": "response", "model": body.get("model", "mock"),
                "status": "completed",
                "output": [{
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": out}],
                }],
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            })
            return

        self._send(200, {
            "id": "chatcmpl_mock", "object": "chat.completion", "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })


if __name__ == "__main__":
    port = int(sys.argv[1] if len(sys.argv) > 1 else "8899")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock llm on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
