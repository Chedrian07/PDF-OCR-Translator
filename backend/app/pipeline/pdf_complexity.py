"""업로드 복잡도 게이트 — 페이지를 렌더하지 않고 MuPDF가 해야 할 그리기 작업량을 센다.

probe_pdf(업로드 검증)가 큐에 넣기 **전에** 부른다(감사 A11: security-2·
gap3-mupdf-analysis-amplification-2). 페이지 수·한 변 길이만 보던 검증을 통과한 두 가지
모양이 렌더·분석을 폭주시켰다.

- **중첩 팬아웃**: Form XObject가 다음 단계를 N번 부르는 것을 k단계 쌓으면 그리기가 N^k로
  펼쳐진다. 3,023바이트 파일(10단계 × 10)이 리프 그리기 10^12회 — 렌더 외삽 수십 일이고,
  그동안 단일 OCR 워커와 GIL이 묶였다. 바이트 수로는 잡히지 않으니 **펼친 호출 수**를 센다.
- **평면 대량 콘텐츠**: XObject 없이 path 10^7개를 그리는 페이지는 get_drawings 한 번에
  ~16GB를 요구해 컨테이너가 OOM-kill됐다. 압축률이 높아 파일은 작을 수 있으니 **압축을
  푼** 콘텐츠 바이트를 센다(상한을 넘는 즉시 멈추므로 압축 폭탄도 끝까지 풀지 않는다).

## 센다

- 콘텐츠 바이트: 페이지 콘텐츠 스트림과, 거기서 도달하는 Form XObject·주석 외형(AP)·
  타일링 패턴·Type3 글리프 스트림의 압축 해제 길이 합. 서로 다른 스트림은 한 번씩만 센다
  (같은 그림을 여러 번 부르는 것은 아래 호출 수의 몫이다).
- 펼친 XObject 호출 수: 콘텐츠의 `Do`마다 그것이 부를 XObject를 그 리소스에서 찾아, Form이면
  그 안의 호출을 곱해 더한다(DAG 메모·포화 덧셈 — 10^12도 즉시 계산된다). 이미지 Do도 한 번의
  그리기로 센다. 주석 외형·패턴·Type3 글리프는 페이지에서 한 번 실행되는 것으로 센다.

## 과소 추정하지 않는다

정확한 비용 모델이 아니라 값싼 **상한** 검사다 — 틀리면 반드시 과대 추정(= 안전한 쪽)이어야
한다. 콘텐츠를 MuPDF와 다르게 토큰화하면 그 차이가 곧 우회로다(감사 isolation-1: `/Fm#E9`처럼
비ASCII 바이트를 이스케이프한 이름을 못 찾아 10^4 체인을 0회로 셌다). 그래서:

- `Do` 후보는 뒤가 토큰 경계인 모든 `Do`다. 단 스트림을 MuPDF 콘텐츠 렉서와 같은 규칙으로
  처음부터 따라 읽을 수 있는 구간에서는 리터럴 문자열(중첩 괄호·백슬래시 이스케이프)·16진
  문자열·주석 안의 `Do`는 연산자가 아니므로 세지 않는다 — 본문 문자열의 'Do' 단어 하나가
  리소스의 가장 비싼 Form을 부르는 것으로 세여 정상 논문이 거부됐다(delta-core-2). 렉서가
  따라갈 수 없는 곳은 예전처럼 전부 센다: 인라인 이미지(BI·ID — MuPDF가 원시 바이트를 읽는다)
  뒤, 앞 콘텐츠 스트림이 문자열·주석·16진 문자열 안에서 끝난 다음 스트림, 렉서 사건 상한을
  넘은 스트림.
- 이름은 MuPDF 렉서와 같은 규칙으로 푼다(`#xx` 이스케이프 — `#00`은 그대로, 콘텐츠 이름 버퍼
  255바이트에서 자름, PyMuPDF가 사전 키를 돌려주는 UTF-8·surrogateescape 문자열). 리소스는
  사전 키를 순회해 찾는다(비ASCII 키를 C 문자열 조회에 넘기지 않는다).
- MuPDF는 Do에 직전 연산자 이후 **처음** 나온 이름을 쓴다(`/A /B Do`는 A). `/이름 Do` 쌍은
  이름 앞을 거슬러 공백·숫자만 지나 연산자(키워드)나 실행 시작에 닿을 때만 확실하다고 본다 —
  사이에 주석·문자열·배열·다른 이름이 끼거나 이전 콘텐츠 스트림에서 피연산자·주석이 이어질 수
  있으면 확실하지 않다. 확실하지 않은 Do와 리소스에서 못 찾은 이름은 그 리소스의 **가장 비싼**
  XObject를 부르는 것으로 센다.
- 순환 Form(MuPDF는 실행 중인 Form을 다시 실행하지 않는다)은 순환을 끊은 결과가 호출 맥락에
  따라 달라지므로 맥락과 무관할 때만 메모한다. 메모할 수 없는 순환이 너무 얽혀 있으면 거부한다.

Type3 글리프 반복·거대 이미지·셰이딩처럼 여기서 세지 않는 비용은 워커 프로세스의 페이지별
시간 상한(pdf_worker)이 받친다.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

# PDF 공백(NUL 포함)과 구분자 밖의 바이트가 '정규 문자'다 — 이름·키워드를 이룬다(MuPDF 렉서와 같다)
_WS = b"\x00\t\n\x0c\r "
_DELIMITERS = b"()<>[]{}/%"
_REGULAR = rb"[^\x00\t\n\x0c\r ()<>\[\]{}/%]"
# `Do` 키워드가 될 수 있는 자리 — 뒤가 토큰 경계인 모든 `Do`(과대 추정)
_DO_KEYWORD = re.compile(rb"Do(?!" + _REGULAR + rb")")
# `/이름 Do` — 이름과 Do 사이에는 PDF 공백만(주석·다른 피연산자가 끼면 쌍으로 보지 않는다)
_DO_PAIR = re.compile(
    rb"/(" + _REGULAR + rb"+)[\x00\t\n\x0c\r ]*Do(?!" + _REGULAR + rb")"
)
_TRAILING_RUN = re.compile(_REGULAR + rb"+\Z")
# 최상위에서 렉서 상태를 바꾸는 바이트: 리터럴 문자열 '(', 주석 '%', 16진 문자열·사전 '<', 그리고
# 인라인 이미지 연산자 BI·ID(키워드 — 앞이 정규 문자면 다른 키워드·이름의 일부다. 숫자 뒤는
# MuPDF가 숫자 토큰을 끊으므로 키워드가 될 수 있다 — 넓게 잡을수록 보수적이다).
_LEX_EVENT = re.compile(
    rb"[(%<]|(?<![^\x00\t\n\x0c\r ()<>\[\]{}/%0-9.+\-])(?:BI|ID)(?!" + _REGULAR + rb")"
)
_STRING_EVENT = re.compile(rb"[()\\]")
_EOL_BYTE = re.compile(rb"[\r\n]")
# 렉서 사건(위 바이트·문자열 안의 괄호·백슬래시) 상한 — 문서 하나에서 이만큼 따라 읽은 뒤의
# 스트림은 렉싱하지 않고 예전처럼 전부 센다(분석 시간을 묶는다, 과대 추정 쪽).
_MAX_LEX_EVENTS = 2_000_000
_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
# 이름과 그 앞 연산자 사이에 올 수 있는 것 — 공백과 숫자 피연산자(이름 피연산자를 바꾸지 않는다)
_WS_OR_NUMBER = _WS + b"0123456789.+-"
# MuPDF 콘텐츠 렉서의 이름 버퍼(256바이트, NUL 포함) — 더 긴 이름은 잘린 채로 리소스를 찾는다
_MAX_NAME_BYTES = 255
# 쌍의 피연산자를 확인할 때 되돌아볼 최대 바이트 — 넘으면 확실하지 않은 것으로 센다
_PAIR_LOOKBACK = 256
# Form 중첩 깊이 상한 — 정상 문서는 한 자릿수다. 그보다 깊은 체인은 단계마다 팬아웃 1로
# 숨겼다가 아래에서 폭발시킬 수 있어 세다 멈추지 않고 거부한다.
MAX_FORM_DEPTH = 64
# 콘텐츠 바이트 검사를 끈 경우(상한 0)에도 호출 수를 세려면 스트림을 풀어야 한다 —
# 압축 폭탄이 메모리를 다 쓰지 않게 이만큼까지만 푼다.
_DECODE_CAP_WHEN_UNLIMITED = 256 * 1024 * 1024
_UNBOUNDED_CALLS = 10**15
# 메모할 수 없는(호출 맥락에 따라 달라지는) 순환 Form을 **다시** 펼치는 작업 상한 — 정상
# 문서는 순환이 없어 (Form, 리소스) 조합마다 한 번씩만 펼친다(처음 펼치는 것은 세지 않는다).
_MAX_REEXPANSIONS = 20_000
_NO_CUT = 1 << 30


class ContentTooComplex(ValueError):
    """업로드 거부 사유 — 사용자 메시지(서버 경로 없음)를 담는다. probe_pdf → API 400."""


@dataclass(frozen=True)
class PageCost:
    page: int            # 1-based
    content_bytes: int
    xobject_calls: int   # 포화값 — 상한+1이면 '상한 초과'


@dataclass(frozen=True)
class _Scan:
    """스트림 하나의 XObject 호출 — 리소스와 무관하게 콘텐츠만 본 결과."""

    size: int                 # 압축 해제 길이
    named: dict[str, int]     # 피연산자가 확실한 Do: MuPDF가 찾을 리소스 이름 → 횟수
    uncertain: int            # 피연산자를 확정하지 못한 Do 후보 수(가장 비싼 XObject로 센다)
    ends_top: bool = False    # 렉서로 끝까지 따라 읽었고 최상위(문자열·주석 밖)에서 끝났다


@dataclass(frozen=True)
class _Lexed:
    """MuPDF 콘텐츠 렉서로 따라 읽은 결과 — [starts[i], stops[i])는 문자열·16진 문자열·주석."""

    starts: list[int]
    stops: list[int]
    stop_at: int      # 여기부터는 따라 읽지 못했다(인라인 이미지·사건 상한) — len(data)면 끝까지
    ends_top: bool    # 끝까지 읽었고 최상위에서 끝났다
    events: int

    def inside(self, pos: int) -> bool:
        """pos가 문자열·16진 문자열·주석 안인가 — 따라 읽은 구간(stop_at 앞)만 판정한다."""
        i = bisect_right(self.starts, pos) - 1
        return i >= 0 and pos < self.stops[i]


def _lex(data: bytes, max_events: int) -> _Lexed:
    """MuPDF 콘텐츠 렉서(pdf_lex)처럼 최상위에서 문자열·16진 문자열·주석 구간을 찾는다.

    - `(`: 리터럴 문자열 — 괄호 중첩을 세고, 백슬래시 뒤 바이트는 무엇이든 글자다(`\\)`는
      닫지 않는다). 짝이 없으면 스트림 끝까지.
    - `%`: 주석 — 다음 `\r`·`\n`까지(없으면 끝까지 — 다음 스트림으로 이어질 수 있다).
    - `<<`는 사전, 그 밖의 `<`는 16진 문자열 — 다음 `>`까지(사이의 어떤 바이트도 끝내지 않는다).
    - BI·ID 키워드: 인라인 이미지 — MuPDF가 렉서 밖에서 원시 바이트를 읽으므로 여기서 멈춘다.
    이름·숫자·키워드는 `(`·`%`·`<`를 담을 수 없으므로(구분자) 최상위 사건만 보면 된다."""
    starts: list[int] = []
    stops: list[int] = []
    n = len(data)
    pos = 0
    events = 0

    def done(stop_at: int, ends_top: bool) -> _Lexed:
        return _Lexed(starts, stops, stop_at, ends_top, events)

    while True:
        found = _LEX_EVENT.search(data, pos)
        if found is None:
            return done(n, True)
        start = found.start()
        events += 1
        if events > max_events:
            return done(start, False)
        byte = data[start]
        if byte == 0x28:  # (
            depth, i = 1, start + 1
            while depth:
                inner = _STRING_EVENT.search(data, i)
                if inner is None:
                    starts.append(start)
                    stops.append(n)
                    return done(n, False)  # 닫히지 않은 문자열 — 다음 스트림으로 이어진다
                events += 1
                if events > max_events:
                    return done(start, False)
                at = inner.start()
                if data[at] == 0x5C:  # 백슬래시 — 다음 바이트는 이스케이프된 글자
                    i = at + 2
                    continue
                depth += 1 if data[at] == 0x28 else -1
                i = at + 1
            starts.append(start)
            stops.append(i)
            pos = i
        elif byte == 0x25:  # %
            eol = _EOL_BYTE.search(data, start + 1)
            starts.append(start)
            if eol is None:
                stops.append(n)
                return done(n, False)  # 주석이 다음 스트림으로 이어질 수 있다
            stops.append(eol.start())
            pos = eol.start()
        elif byte == 0x3C:  # <
            if data[start + 1:start + 2] == b"<":
                pos = start + 2  # 사전 — 안의 토큰은 최상위와 같은 규칙
                continue
            close = data.find(b">", start + 1)
            starts.append(start)
            if close < 0:
                stops.append(n)
                return done(n, False)
            stops.append(close + 1)
            pos = close + 1
        else:  # BI·ID — 인라인 이미지의 원시 바이트는 렉서가 따라갈 수 없다
            return done(start, False)


def _decode_name(raw: bytes) -> str:
    """콘텐츠 스트림의 이름 토큰(`/` 뒤 바이트) → MuPDF가 리소스 사전에서 찾는 키 문자열.

    MuPDF 렉서처럼 두 자리 16진 `#xx`를 바이트로 풀되 `#00`은 그대로 두고, 콘텐츠 렉서의
    이름 버퍼(255바이트)에서 자른다. PyMuPDF는 이름(C 문자열)을 UTF-8·surrogateescape로
    파이썬 문자열화하므로 같은 방식으로 바꿔야 `pdf_to_name()`이 준 사전 키와 비교된다
    (latin-1로 풀면 `/Fm#E9`가 'Fmé'가 되어 키 'Fm\\udce9'와 어긋났다)."""
    def unescape(match: re.Match) -> bytes:
        if match.group(1) == b"00":
            return match.group(0)
        return bytes([int(match.group(1), 16)])

    data = _NAME_ESCAPE.sub(unescape, raw)[:_MAX_NAME_BYTES]
    return data.decode("utf-8", "surrogateescape")


def _first_eol(data: bytes, start: int, end: int) -> int:
    """data[start:end]의 첫 줄바꿈(\\r·\\n) 위치, 없으면 -1."""
    hits = [pos for pos in (data.find(b"\n", start, end), data.find(b"\r", start, end))
            if pos >= 0]
    return min(hits) if hits else -1


def scan_calls(
    data: bytes, *, fresh: bool = True, max_candidates: int = 0, lexed: bool | None = None,
) -> tuple[dict, int]:
    """콘텐츠 바이트의 Do를 (확실한 이름별 횟수, 확실하지 않은 Do 수)로 나눈다 — 순수 함수.

    fresh=False는 실행 중간에 이어지는 스트림(페이지의 두 번째 이후 콘텐츠 스트림)이다 — 앞
    스트림의 피연산자·주석이 이어질 수 있어 스트림 처음은 '실행 시작'이 아니고, 첫 줄은 주석
    안일 수 있다. lexed(기본: fresh)는 스트림 처음의 렉서 상태가 최상위라고 아는가다 — 알면
    문자열·주석 안의 Do를 빼고, 모르면 예전처럼 전부 후보로 센다. max_candidates를 넘는 Do
    후보는 쌍을 따지지 않고 전부 확실하지 않은 것으로 센다(정상 문서는 그만큼 Do를 쓰지 않는다
    — 분석 시간을 묶는다).
    """
    named, uncertain, _lexed = _scan_calls(
        data, fresh=fresh, max_candidates=max_candidates,
        lexed=fresh if lexed is None else lexed, max_events=_MAX_LEX_EVENTS,
    )
    return named, uncertain


def _scan_calls(
    data: bytes, *, fresh: bool, max_candidates: int, lexed: bool, max_events: int,
    want_end: bool = False,
) -> tuple[dict, int, _Lexed | None]:
    """scan_calls 본체 — 렉싱 결과(다음 스트림의 시작 상태·쓴 사건 수)도 돌려준다.

    Do 후보가 없으면 렉싱하지 않는다 — want_end(다음 콘텐츠 스트림이 이 스트림의 끝 상태를
    알아야 한다)일 때만 끝까지 읽는다."""
    total = _DO_KEYWORD.subn(b"", data)[1]
    if max_candidates and total > max_candidates:
        return {}, total, None
    lex = _lex(data, max_events) if lexed and max_events > 0 and (total or want_end) else None
    if lex is not None and lex.starts and total:
        # 따라 읽은 구간의 문자열·16진 문자열·주석 안 'Do'는 연산자가 아니다(본문의 'Do' 단어)
        total -= sum(
            1 for found in _DO_KEYWORD.finditer(data)
            if found.start() < lex.stop_at and lex.inside(found.start())
        )
    if not total:
        return {}, 0, lex
    named: dict[str, int] = {}
    certain = 0
    # 주석 휴리스틱('%' 역탐색)은 렉서가 따라가지 못한 구간(lex.stop_at부터)에서만 쓴다 — 따라
    # 읽은 구간은 주석 구간을 정확히 안다. stop_at은 최상위라 그 자리에 열린 주석은 없다.
    exact_until = lex.stop_at if lex is not None else 0
    has_comment = data.find(b"%", exact_until) >= 0 or (not fresh and lex is None)
    scanned = exact_until
    last_pct = -1 if fresh or lex is not None else -2   # 이어지는 스트림은 처음 앞에 주석이 열려 있을 수 있다
    eol_after_pct = -1
    eol_scanned = exact_until
    for match in _DO_PAIR.finditer(data):
        start = match.start()
        exact = start < exact_until
        if exact:
            if lex.inside(match.end() - 2):
                continue  # 문자열·주석 안의 'Do' — 후보가 아니다(위에서 뺐다)
            if lex.inside(start):
                continue  # 이름이 문자열·주석 안 — 피연산자를 확정할 수 없다
        elif has_comment:
            pct = data.rfind(b"%", scanned, start)
            if pct >= 0:
                last_pct, eol_after_pct, eol_scanned = pct, -1, pct + 1
            scanned = start
        lo = max(0, start - _PAIR_LOOKBACK)
        head = data[lo:start].rstrip(_WS_OR_NUMBER)
        if not head:
            # 공백·숫자만 지나 스트림 처음에 닿았다 — 실행 시작이면 피연산자 스택이 비어 있다
            if lo or not fresh:
                continue
            run_start = 0
        else:
            if head[-1:] in _WS + _DELIMITERS:
                continue  # 문자열·배열·사전·이름 등 — 앞에 다른 이름 피연산자가 있을 수 있다
            run = _TRAILING_RUN.search(head)
            if run is None or (run.start() == 0 and lo):
                continue  # 연산자가 되돌아볼 범위 밖까지 이어진다
            if run.start() and head[run.start() - 1:run.start()] == b"/":
                continue  # 연산자가 아니라 이름이다
            run_start = lo + run.start()
        if exact:
            if lex.inside(run_start):
                continue  # 연산자처럼 보이는 글자가 문자열·주석 안이다
        elif has_comment and last_pct != -1:
            # 연산자 줄(또는 그 앞)에서 열린 주석이 연산자·이름을 숨겼을 수 있다
            if eol_after_pct < 0 and eol_scanned < run_start:
                eol_after_pct = _first_eol(data, max(0, eol_scanned), run_start)
                eol_scanned = run_start
            if not 0 <= eol_after_pct < run_start:
                continue
        name = _decode_name(match.group(1))
        named[name] = named.get(name, 0) + 1
        certain += 1
    return named, max(0, total - certain), lex


class _TooTangled(Exception):
    pass


class _TooDeep(Exception):
    pass


class ComplexityScanner:
    """한 문서의 페이지별 그리기 작업량 — 스트림 해석 결과는 문서 단위로 캐시한다."""

    def __init__(self, fitz, doc, *, max_content_bytes: int, max_xobject_calls: int) -> None:
        self.mu = fitz.mupdf
        self.doc = doc
        self.pdoc = self.mu.pdf_document_from_fz_document(doc.this)
        self.max_bytes = max(0, int(max_content_bytes))
        self.max_calls = max(0, int(max_xobject_calls))
        self._call_cap = self.max_calls + 1 if self.max_calls else _UNBOUNDED_CALLS
        self._decode_cap = self.max_bytes or _DECODE_CAP_WHEN_UNLIMITED
        # (스트림 번호, fresh, 렉서 시작 상태를 아는가, 끝 상태가 필요한가) → 콘텐츠 해석 결과
        self._streams: dict[tuple[int, bool, bool, bool], _Scan] = {}
        # 문서 전체의 렉서 사건 예산 — 다 쓰면 이후 스트림은 렉싱 없이 예전처럼 전부 센다
        self._lex_budget = _MAX_LEX_EVENTS
        # (스트림 번호, 리소스 키) → 한 번 실행 시 펼친 호출 수(호출 맥락과 무관한 것만)
        self._calls: dict[tuple, int] = {}
        # 리소스 키 → {XObject 이름: 객체}, 그중 가장 비싼 호출(맥락과 무관하게 셀 수 있을 때만)
        self._xobjects: dict[tuple, dict] = {}
        self._worst: dict[tuple, int] = {}
        # 실행 중인 Form 스트림 번호 → 그 스택 깊이(순환 절단·메모 판단용)
        self._active: dict[int, int] = {}
        self._expanded: set[tuple] = set()
        self._reexpansions = 0
        # 객체 번호 → (Form인가, 자기 /Resources(없으면 None)) — 순환을 다시 펼칠 때 저수준 조회를 줄인다
        self._forms: dict[int, tuple[bool, tuple | None]] = {}

    # ── PDF 객체 도우미 ─────────────────────────────────────────────────

    def _get(self, obj, key: str):
        """ASCII 고정 키 조회 전용 — 콘텐츠에서 온 이름은 _xobject_map으로 찾는다."""
        if obj is None or not self.mu.pdf_is_dict(obj):
            return None
        value = self.mu.pdf_dict_gets(obj, key)
        return None if self.mu.pdf_is_null(value) else value

    def _items(self, obj):
        if obj is None or not self.mu.pdf_is_dict(obj):
            return
        for i in range(self.mu.pdf_dict_len(obj)):
            yield (
                self.mu.pdf_to_name(self.mu.pdf_dict_get_key(obj, i)),
                self.mu.pdf_dict_get_val(obj, i),
            )

    def _name(self, obj) -> str:
        return self.mu.pdf_to_name(obj) if obj is not None else ""

    def _is_form(self, xobject) -> bool:
        return self._name(self._get(xobject, "Subtype")) == "Form"

    def _res_key(self, resources, owner: int) -> tuple:
        """리소스 사전의 정체성 — 간접 객체면 그 번호, 직접 사전이면 담고 있는 객체 번호."""
        if self.mu.pdf_is_indirect(resources):
            return ("obj", self.mu.pdf_to_num(resources))
        return ("in", owner)

    def _own_resources(self, stream_obj, parent_res, parent_key: tuple) -> tuple:
        """Form·패턴·글리프의 리소스 — 자기 /Resources가 없으면 부모 것을 쓴다(MuPDF와 같다)."""
        own = self._get(stream_obj, "Resources")
        if own is not None and self.mu.pdf_is_dict(own):
            return own, self._res_key(own, self.mu.pdf_to_num(stream_obj))
        return parent_res, parent_key

    def _xobject_map(self, resources, res_key: tuple) -> dict:
        """리소스의 XObject 사전 — 키를 순회해 {pdf_to_name 문자열: 스트림 객체}로 만든다.

        비ASCII 이름을 pdf_dict_gets(C 문자열)로 찾으면 surrogate 문자열에서 TypeError가 나
        페이지 검사가 통째로 건너뛰어졌다 — 파이썬 쪽에서 비교한다."""
        cached = self._xobjects.get(res_key)
        if cached is None:
            cached = {
                name: value for name, value in self._items(self._get(resources, "XObject"))
                if self.mu.pdf_is_stream(value)
            }
            self._xobjects[res_key] = cached
        return cached

    # ── 스트림 해석 ─────────────────────────────────────────────────────

    def _stream(
        self, num: int, fresh: bool = True, lexed: bool = True, want_end: bool = False,
    ) -> _Scan:
        """스트림 하나의 해석 — fresh=False는 페이지의 이어지는 콘텐츠 스트림, lexed는 그 스트림
        처음의 렉서 상태가 최상위라고 아는가(Form·외형·패턴·글리프와 첫 콘텐츠 스트림은 늘 안다),
        want_end는 다음 콘텐츠 스트림을 위해 끝 상태(ends_top)가 필요한가다."""
        key = (num, fresh, lexed, want_end)
        cached = self._streams.get(key)
        if cached is not None:
            return cached
        try:
            stm = self.mu.pdf_open_stream_number(self.pdoc, num)
            size = int(self.mu.fz_skip(stm, self._decode_cap + 1))
        except Exception:  # noqa: BLE001 — 깨진 스트림은 렌더도 건너뛴다
            size = 0
        named: dict[str, int] = {}
        uncertain = 0
        # 빈(또는 읽지 못한) 스트림은 렉서 상태를 바꾸지 않는다
        ends_top = lexed and size == 0
        if 0 < size <= self._decode_cap and (not self.max_bytes or size <= self.max_bytes):
            try:
                data = self.doc.xref_stream(num) or b""
            except Exception:  # noqa: BLE001
                data = b""
            named, uncertain, lex = _scan_calls(
                data[: self._decode_cap], fresh=fresh,
                max_candidates=self.max_calls, lexed=lexed, max_events=self._lex_budget,
                want_end=want_end,
            )
            if lex is not None:
                self._lex_budget = max(0, self._lex_budget - lex.events)
                ends_top = lex.ends_top
        scan = _Scan(size, named, uncertain, ends_top)
        self._streams[key] = scan
        return scan

    def _form_info(self, target) -> tuple[int, bool, tuple | None]:
        """(객체 번호, Form인가, 자기 리소스 (사전, 키) — 없으면 None). 객체 번호로 캐시한다."""
        num = self.mu.pdf_to_num(target)
        info = self._forms.get(num)
        if info is None:
            own = None
            is_form = self._is_form(target)
            if is_form:
                found = self._get(target, "Resources")
                if found is not None and self.mu.pdf_is_dict(found):
                    own = (found, self._res_key(found, num))
            info = (is_form, own)
            self._forms[num] = info
        return num, info[0], info[1]

    def _cost(self, target, resources, res_key: tuple, depth: int) -> tuple[int, int]:
        """XObject 하나를 부르는 비용(호출 1 + Form이면 펼친 내부 호출)과 순환 절단 깊이."""
        num, is_form, own = self._form_info(target)
        if is_form:
            sub_res, sub_key = own if own is not None else (resources, res_key)
            inner, low = self._expand(num, sub_res, sub_key, depth + 1)
            return min(1 + inner, self._call_cap), low
        return 1, _NO_CUT

    def _charge(self, scan: _Scan, resources, res_key: tuple, depth: int) -> tuple[int, int]:
        """스트림을 한 번 실행할 때의 XObject 호출 수(포화)와 부딪힌 순환 절단 중 가장 얕은 깊이."""
        xobjects = self._xobject_map(resources, res_key)
        total, low = 0, _NO_CUT
        uncertain = scan.uncertain
        for name, count in scan.named.items():
            target = xobjects.get(name)
            if target is None:
                uncertain += count  # 못 찾은 이름 — 어느 XObject든 될 수 있다고 본다
                continue
            cost, cut = self._cost(target, resources, res_key, depth)
            low = min(low, cut)
            total = min(total + count * cost, self._call_cap)
            if total >= self._call_cap:
                return total, low
        if uncertain and xobjects:
            worst = self._worst.get(res_key)
            if worst is None:
                worst, worst_low = 0, _NO_CUT
                for target in xobjects.values():
                    cost, cut = self._cost(target, resources, res_key, depth)
                    worst_low = min(worst_low, cut)
                    worst = max(worst, cost)
                    if worst >= self._call_cap:
                        break
                low = min(low, worst_low)
                if worst_low == _NO_CUT:
                    self._worst[res_key] = worst
            total = min(total + uncertain * worst, self._call_cap)
        return total, low

    def _expand(self, num: int, resources, res_key: tuple, depth: int) -> tuple[int, int]:
        """Form 스트림을 한 번 실행할 때의 XObject 호출 수(중첩을 펼친 값, 포화)와 순환 절단 깊이.

        실행 중인 Form을 다시 부르면 MuPDF처럼 끊는다(0). 끊은 결과는 호출 맥락(어느 Form이
        실행 중인가)에 따라 달라지므로, 이 Form보다 얕은(= 바깥 맥락의) Form에서 끊긴 계산은
        메모하지 않는다 — 메모하면 순환 짝을 다른 곳에서 여러 번 부를 때 그 안의 팬아웃을
        빠뜨린다."""
        key = (num, res_key)
        cached = self._calls.get(key)
        if cached is not None:
            return cached, _NO_CUT
        active = self._active.get(num)
        if active is not None:
            return 0, active  # 순환 Form — MuPDF는 실행 중인 XObject를 다시 실행하지 않는다
        if depth > MAX_FORM_DEPTH:
            raise _TooDeep()
        if key in self._expanded:
            self._reexpansions += 1
            if self._reexpansions > _MAX_REEXPANSIONS:
                raise _TooTangled()
        self._expanded.add(key)
        self._active[num] = depth
        try:
            total, low = self._charge(self._stream(num), resources, res_key, depth)
        finally:
            del self._active[num]
        if low >= depth:
            self._calls[key] = total
            low = _NO_CUT
        return total, low

    def _reachable_forms(self, scan: _Scan, resources, res_key: tuple):
        """이 스트림이 부를 수 있는 Form — 확실하지 않은 Do가 있으면 리소스의 모든 Form."""
        xobjects = self._xobject_map(resources, res_key)
        names = list(scan.named)
        if scan.uncertain or any(name not in xobjects for name in names):
            names = list(xobjects)
        for name in names:
            target = xobjects.get(name)
            if target is not None and self._form_info(target)[1]:
                yield target

    # ── 페이지 ──────────────────────────────────────────────────────────

    def _extra_roots(self, resources, res_key: tuple):
        """리소스 사전에서 한 번 실행으로 세는 스트림: 타일링 패턴·Type3 글리프."""
        for _name, pattern in self._items(self._get(resources, "Pattern")):
            if self.mu.pdf_is_stream(pattern):
                yield (pattern, *self._own_resources(pattern, resources, res_key))
        for _name, font in self._items(self._get(resources, "Font")):
            if self._name(self._get(font, "Subtype")) != "Type3":
                continue
            font_res, font_key = self._own_resources(font, resources, res_key)
            for _glyph, proc in self._items(self._get(font, "CharProcs")):
                if self.mu.pdf_is_stream(proc):
                    yield (proc, font_res, font_key)

    def _add(self, total: int, more: int) -> int:
        return min(total + more, self._call_cap)

    def page_cost(self, index: int) -> PageCost:
        mu = self.mu
        page_obj = mu.pdf_lookup_page_obj(self.pdoc, index)
        page_num = mu.pdf_to_num(page_obj)
        resources = mu.pdf_dict_get_inheritable(page_obj, mu.pdf_new_name("Resources"))
        res_key = self._res_key(resources, page_num)

        contents: list = []
        raw_contents = self._get(page_obj, "Contents")
        if raw_contents is not None and mu.pdf_is_array(raw_contents):
            for i in range(mu.pdf_array_len(raw_contents)):
                item = mu.pdf_array_get(raw_contents, i)
                if mu.pdf_is_stream(item):
                    contents.append(item)
        elif raw_contents is not None and mu.pdf_is_stream(raw_contents):
            contents.append(raw_contents)
        appearances: list[tuple] = []  # 주석 외형 스트림 — 페이지와 함께 한 번 그려진다
        annots = self._get(page_obj, "Annots")
        if annots is not None and mu.pdf_is_array(annots):
            for i in range(mu.pdf_array_len(annots)):
                normal = self._get(self._get(mu.pdf_array_get(annots, i), "AP"), "N")
                if normal is None:
                    continue
                streams = (
                    [normal] if mu.pdf_is_stream(normal)
                    else [v for _k, v in self._items(normal) if mu.pdf_is_stream(v)]
                )
                for ap in streams:
                    appearances.append((ap, *self._own_resources(ap, resources, res_key)))

        # 페이지 콘텐츠 스트림들은 MuPDF가 이어서 한 번 실행한다 — 앞 스트림의 피연산자·주석이
        # 다음 스트림으로 이어지므로 두 번째부터는 '실행 시작'이 아니다(fresh=False). 렉서 상태도
        # 이어진다(MuPDF는 공백 하나를 끼워 이어 붙인다) — 앞 스트림이 최상위에서 끝났을 때만
        # 다음 스트림의 문자열·주석을 따라 읽는다.
        calls = 0
        content_scans: list[_Scan] = []
        lexed = True
        for position, stream in enumerate(contents):
            scan = self._stream(
                mu.pdf_to_num(stream), fresh=position == 0, lexed=lexed,
                want_end=position + 1 < len(contents),
            )
            lexed = scan.ends_top
            content_scans.append(scan)
            calls = self._add(calls, self._charge(scan, resources, res_key, 0)[0])
        for ap, ap_res, ap_key in appearances:
            calls = self._add(calls, 1 + self._expand(mu.pdf_to_num(ap), ap_res, ap_key, 1)[0])

        # 도달하는 스트림을 한 번씩 돌며 바이트를 더하고, 처음 보는 리소스 사전의 패턴·Type3
        # 글리프를 한 번 실행으로 더한다.
        content_bytes = 0
        seen_streams: set[int] = set()
        seen_resources: set[tuple] = set()
        queue = deque(
            [(s, resources, res_key, scan) for s, scan in zip(contents, content_scans)]
            + [(ap, ap_res, ap_key, None) for ap, ap_res, ap_key in appearances]
        )
        while queue:
            obj, res, key, scan = queue.popleft()
            if key not in seen_resources:
                seen_resources.add(key)
                for extra, extra_res, extra_key in self._extra_roots(res, key):
                    inner = self._expand(mu.pdf_to_num(extra), extra_res, extra_key, 1)[0]
                    calls = self._add(calls, 1 + inner)
                    queue.append((extra, extra_res, extra_key, None))
            num = mu.pdf_to_num(obj)
            if num in seen_streams:
                if scan is not None:
                    # Contents에 다시 나온 같은 스트림(렉서 시작 상태가 다를 수 있다) — 바이트는
                    # 한 번만 세되, 그 해석이 부를 수 있는 Form은 모두 따라간다
                    for target in self._reachable_forms(scan, res, key):
                        queue.append((target, *self._own_resources(target, res, key), None))
                continue
            seen_streams.add(num)
            if scan is None:
                scan = self._stream(num)
            content_bytes += scan.size
            if self.max_bytes and content_bytes > self.max_bytes:
                break
            for target in self._reachable_forms(scan, res, key):
                queue.append((target, *self._own_resources(target, res, key), None))
        return PageCost(index + 1, content_bytes, calls)

    def check(self, index: int) -> PageCost:
        """페이지 하나를 검사한다 — 상한을 넘으면 ContentTooComplex."""
        try:
            cost = self.page_cost(index)
        except _TooDeep:
            raise ContentTooComplex(
                f"{index + 1}페이지의 Form XObject 중첩이 {MAX_FORM_DEPTH}단계를 넘습니다 — "
                "처리할 수 없는 PDF입니다"
            ) from None
        except _TooTangled:
            raise ContentTooComplex(
                f"{index + 1}페이지의 Form XObject가 서로를 부르는 순환이 너무 얽혀 있습니다 — "
                "처리할 수 없는 PDF입니다"
            ) from None
        if self.max_bytes and cost.content_bytes > self.max_bytes:
            raise ContentTooComplex(
                f"{cost.page}페이지가 그리는 콘텐츠가 너무 큽니다 (압축 해제 "
                f"{cost.content_bytes / 1048576:.0f}MB 초과 — 페이지당 상한 "
                f"{self.max_bytes / 1048576:.0f}MB, PDF_MAX_PAGE_CONTENT_MB). "
                "서버가 안전하게 처리할 수 없는 PDF입니다"
            )
        if self.max_calls and cost.xobject_calls > self.max_calls:
            raise ContentTooComplex(
                f"{cost.page}페이지의 중첩 그리기 호출(Form XObject)이 너무 많습니다 "
                f"(펼치면 {self.max_calls:,}회 초과 — 페이지당 상한, "
                "PDF_MAX_PAGE_XOBJECT_CALLS). 렌더에 비정상적으로 오래 걸리는 PDF입니다"
            )
        return cost
