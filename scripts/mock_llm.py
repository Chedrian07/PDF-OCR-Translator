"""OpenAI 호환 목 서버 — 실제 과금 없이 번역 파이프라인 전체를 실행하기 위한 하네스.

두 가지 모드를 한 프로세스에서 제공한다:
  - 정상 모드(기본): 입력 텍스트를 결정적으로 "번역"한다. 마스킹 플레이스홀더
    (<m1 .../> 형태)는 **그대로 보존**하고 영문 단어만 한글로 치환하므로,
    복원 단계·layout 정렬·PDF 조판까지 실제 경로가 전부 돈다.
  - 결함 주입 모드: ?fault=refusal|refusal_ko|echo|summary|drop_placeholder|paired_tags|http400|http429
    쿼리 또는 FAULT 환경변수로 A-1(출력 검증) 회귀를 실증한다.
    paired_tags는 XML 습관이 있는 소형 모델처럼 자기 닫힘 플레이스홀더를 쌍 태그로 바꾼다
    (감사 translate-llm-10 — 유닛의 첫 태그는 빈 쌍, 나머지는 내용을 감싼 쌍).
    쿼리 경로는 `OPENAI_BASE_URL=http://host:port/v1?fault=echo` 로 쓴다 —
    translate/client.py `_endpoint_url()`이 base의 query를 보존한 채 경로만 이어
    붙이므로(`/v1/chat/completions?fault=echo`) 실제로 도달한다. verify_e2e.py는
    drop_placeholder를 이 경로로 주입해 두 방식이 모두 살아 있음을 매 실행 증명한다.

정상 모드는 **실제 한국어의 길이 압축률을 재현한다**(MOCK_TRANSLATE_RATIO, 기본 0.4).
영→한 번역문은 원문의 0.3~0.5배 길이인데, 목이 길이를 보존하면
translate/masking.py `looks_untranslated()`의 길이비 하한(0.3) 회귀가 하네스에
전혀 잡히지 않는다(목 출력이 하한의 2배라 하한을 0.5까지 올려도 통과한다).
MOCK_TRANSLATE_RATIO=0 으로 두면 예전 길이 보존 동작으로 되돌아간다.

Responses API(/v1/responses)와 Chat Completions(/v1/chat/completions)를 모두 지원한다.
Chat 요청이 `"stream": true`면 실제 서버처럼 SSE(chat.completion.chunk → [DONE])로
보낸다 — 번역 클라이언트의 기본(TRANSLATE_STREAM=auto)이 chat 스트리밍이라 하네스도
그 경로를 돈다. 위 결함 모드는 스트리밍에서도 같은 출력을 낸다.

시나리오 노브(결함 모드와 별개 — pytest가 쓴다. 쿼리 또는 환경변수):
  delay=초 / MOCK_STREAM_DELAY_S   SSE 조각 사이 지연 — 취소 시 서버 생성 중단 검증용
  chunk=N / MOCK_STREAM_CHUNK      SSE 조각당 글자 수(기본 16)
  finish=length / MOCK_FINISH      finish_reason=length(responses는 incomplete) — 잘림 재현
  reasoning=N / MOCK_REASONING_CHARS  본문 앞에 사고 N자를 reasoning_content로 싣는다
계측: GET /__stats의 stream_chunks(보낸 SSE 조각 수)·stream_aborted(클라이언트가 끊어
쓰기가 실패한 스트림 수).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

# 호출 계측 — D-1(이중 번역) 검증에 쓴다. stream_*은 SSE 경로 계측.
STATS = {"calls": 0, "by_text": {}, "stream_chunks": 0, "stream_aborted": 0}
_LOCK = threading.Lock()

# ⚠ 접두 문자는 masking.py `_PLACEHOLDER_RE`와 **같은 집합**이어야 한다
# (m=수식 k=코드 g=이미지 u=URL c=인용 f=참조 t=HTML태그). 예전에는 `<m…>`만 봐서
# ① 정상 모드에서 c/f/u 플레이스홀더의 v 속성이 한글로 뭉개졌고
# ② drop_placeholder 결함이 수식 없는 문서에서 **완전 no-op**이었다
#    (실측: 논문 6페이지 = c 23 · f 5 · u 4 · m 0 → 지울 것이 하나도 없었다).
PLACEHOLDER_RE = re.compile(r"<[mkgucft]\d+\b[^>]*/>")

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


# 압축 모드에서 통째로 지우는 기능어 — 실제 한국어는 관사·전치사·대명사를
# 조사로 흡수하므로 단어 수 자체가 줄어든다. 길이비만 맞추려고 글자를 깎으면
# 재현되지 않는 압축 경로다.
_DROP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "is", "are", "was", "were", "be", "been",
    "to", "in", "on", "at", "for", "with", "that", "this", "these", "those", "as",
    "by", "it", "its", "from", "we", "our", "their", "which", "has", "have", "had",
    "not", "but", "than", "can", "may",
})

# 기본 압축률 — 실측(sample/2504.19874v1.pdf 블록 248개)에서 전체 0.50배,
# 유닛 최솟값 0.36배가 나온다. 게이트 하한 0.3 대비 20% 여유.
_DEFAULT_RATIO = 0.4
# 목 때문에 정상 경로가 깨지면 안 되므로 유닛별 하한을 둔다. 압축 결과가 이보다
# 짧으면 그 조각만 길이 보존 방식으로 되돌린다(짧은 유닛이 전부 기능어인 경우 등).
_MIN_SAFE_RATIO = 0.36


def _target_ratio() -> float:
    """MOCK_TRANSLATE_RATIO — 0 이하이거나 파싱 실패면 길이 보존(예전) 모드."""
    raw = os.environ.get("MOCK_TRANSLATE_RATIO", "")
    if not raw.strip():
        return _DEFAULT_RATIO
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_RATIO


def _hangulize(word: str, n: int) -> str:
    """영단어 → 결정적 한글 의사단어. 라틴 문자를 남기면 거부문으로 오인된다."""
    h = 0
    for ch in word.lower():
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    # 가(0xAC00) ~ 힣(0xD7A3) 범위에서 결정적으로 고른다.
    return "".join(chr(0xAC00 + (h >> (i * 5)) % 11172) for i in range(n))


def _sub_keep(word: re.Match[str]) -> str:
    """길이 보존 치환(예전 동작) — 영단어 하나를 한글 2~4자로."""
    w = word.group(0)
    return _DICT.get(w.lower()) or _hangulize(w, max(2, min(4, len(w) // 2)))


def _translate_chunk(chunk: str, ratio: float) -> str:
    """플레이스홀더가 없는 텍스트 조각 하나를 "번역"한다."""
    if ratio <= 0:
        return re.sub(r"[A-Za-z]{2,}", _sub_keep, chunk)

    def sub_compress(word: re.Match[str]) -> str:
        w = word.group(0)
        lo = w.lower()
        if lo in _DROP_WORDS:
            return ""
        return _DICT.get(lo) or _hangulize(w, max(1, round(len(w) * ratio)))

    out = re.sub(r"[A-Za-z]{2,}", sub_compress, chunk)
    # 기능어를 지운 자리에 남은 연속 공백을 접는다(줄바꿈은 건드리지 않는다 —
    # 마크다운 구조와 layout 정렬이 줄 단위로 걸려 있다).
    return re.sub(r"[^\S\n]{2,}", " ", out)


def _translate(text: str, ratio: float | None = None) -> str:
    """플레이스홀더를 보존한 채 영문 토큰만 한글로 바꾼다."""
    r = _target_ratio() if ratio is None else ratio
    parts = []
    last = 0
    for m in PLACEHOLDER_RE.finditer(text):
        parts.append(("t", text[last:m.start()]))
        parts.append(("p", m.group(0)))
        last = m.end()
    parts.append(("t", text[last:]))

    body = "".join(chunk for kind, chunk in parts if kind == "t")
    out = [chunk if kind == "p" else _translate_chunk(chunk, r) for kind, chunk in parts]
    if r > 0:
        # 유닛 단위 하한 — 짧은 유닛이 통째로 기능어라 지나치게 짧아지면 정상 경로가
        # 게이트에 걸린다(목 때문에 파이프라인이 깨지면 안 된다). 그때만 길이 보존.
        got = sum(len(t) for t, (kind, _) in zip(out, parts) if kind == "t")
        if body.strip() and got < _MIN_SAFE_RATIO * len(body):
            out = [chunk if kind == "p" else re.sub(r"[A-Za-z]{2,}", _sub_keep, chunk)
                   for kind, chunk in parts]
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


# 쌍 태그 결함용 — 자기 닫힘 플레이스홀더의 id·속성
_PLACEHOLDER_PARTS_RE = re.compile(r"<([mkgucft]\d+)\b([^>]*?)\s*/>")
_V_ATTR_RE = re.compile(r'\bv="([^"]*)"')


def _pair_placeholders(text: str) -> str:
    """자기 닫힘 플레이스홀더를 쌍 태그로 바꾼다 (translate-llm-10).

    유닛의 첫 태그는 빈 쌍(`<m1 v="x"></m1>` — 자기 닫힘과 같으니 채택돼야 한다), 나머지는
    미리보기를 번역해 감싼 쌍(`<f2 v="Figure 2">그림 2</f2>` — 닫는 태그와 'Figure 2그림 2'
    같은 중복 내용이 산출물에 박히면 안 된다)으로 만든다."""
    seen = [0]

    def sub(m: re.Match[str]) -> str:
        seen[0] += 1
        pid, attrs = m.group(1), m.group(2)
        if seen[0] == 1:
            return f"<{pid}{attrs}></{pid}>"
        preview = _V_ATTR_RE.search(attrs)
        inner = _translate(preview.group(1)) if preview and preview.group(1) else "내용"
        return f"<{pid}{attrs}>{inner}</{pid}>"

    return _PLACEHOLDER_PARTS_RE.sub(sub, text)


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
    if fault == "paired_tags":
        return _pair_placeholders(_translate(src))
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
                STATS["stream_chunks"] = 0
                STATS["stream_aborted"] = 0
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
        knobs = _scenario(self.path)
        finish = "length" if knobs["finish"] == "length" else "stop"
        reasoning = "생각 중… " * (knobs["reasoning"] // 6 + 1) if knobs["reasoning"] else ""
        reasoning = reasoning[:knobs["reasoning"]]

        if urlsplit(self.path).path.startswith("/v1/responses") or "responses" in self.path:
            self._send(200, {
                "id": "resp_mock", "object": "response", "model": body.get("model", "mock"),
                "status": "incomplete" if finish == "length" else "completed",
                "output": [{
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": out}],
                }],
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            })
            return

        if body.get("stream"):
            self._stream_chat(body, out, finish, reasoning, knobs)
            return

        message = {"role": "assistant", "content": out}
        if reasoning:
            message["reasoning_content"] = reasoning
        self._send(200, {
            "id": "chatcmpl_mock", "object": "chat.completion", "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    def _stream_chat(self, body: dict, out: str, finish: str, reasoning: str, knobs: dict):
        """SSE 응답 — 실제 서버처럼 조각마다 flush하고, 클라이언트가 끊으면 즉시 멈춘다."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # 본문 길이 없음 — 종료로 끝을 알린다
        self.end_headers()
        self.close_connection = True
        model = body.get("model", "mock")
        step = max(1, knobs["chunk"])

        def event(delta: dict, finish_reason=None, **extra) -> bytes:
            obj = {"id": "chatcmpl_mock", "object": "chat.completion.chunk", "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            obj.update(extra)
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

        pieces = [event({"role": "assistant"})]
        pieces += [event({"reasoning_content": reasoning[i:i + step]})
                   for i in range(0, len(reasoning), step)]
        pieces += [event({"content": out[i:i + step]}) for i in range(0, len(out), step)]
        pieces.append(event({}, finish))
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"prompt_tokens": 10, "completion_tokens": len(pieces), "total_tokens": 20}
            pieces.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        pieces.append(b"data: [DONE]\n\n")
        try:
            # prefill 중 keepalive 주석(mlx_lm 형태) — 클라이언트는 무시해야 한다
            self.wfile.write(b": keepalive 1/1\n\n")
            for piece in pieces:
                if knobs["delay"]:
                    time.sleep(knobs["delay"])
                self.wfile.write(piece)
                self.wfile.flush()
                with _LOCK:
                    STATS["stream_chunks"] += 1
        except (BrokenPipeError, ConnectionResetError):
            with _LOCK:
                STATS["stream_aborted"] += 1


def _scenario(path: str) -> dict:
    """시나리오 노브 — 쿼리가 환경변수보다 우선한다(모듈 docstring 참조)."""
    query = parse_qs(urlsplit(path).query)

    def pick(name: str, env: str, default: str) -> str:
        return (query.get(name) or [os.environ.get(env, default)])[0]

    def num(raw: str, default: float) -> float:
        try:
            return float(raw)
        except ValueError:
            return default

    return {
        "delay": max(0.0, num(pick("delay", "MOCK_STREAM_DELAY_S", "0"), 0.0)),
        "chunk": int(num(pick("chunk", "MOCK_STREAM_CHUNK", "16"), 16)),
        "finish": pick("finish", "MOCK_FINISH", ""),
        "reasoning": int(num(pick("reasoning", "MOCK_REASONING_CHARS", "0"), 0)),
    }


if __name__ == "__main__":
    port = int(sys.argv[1] if len(sys.argv) > 1 else "8899")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock llm on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
