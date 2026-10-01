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
- 펼친 XObject 호출 수: 콘텐츠의 `/이름 Do`를 그 리소스로 해석해, Form이면 그 안의 호출을
  곱해 더한다(DAG 메모·포화 덧셈 — 10^12도 즉시 계산된다). 이미지 Do도 한 번의 그리기로
  센다. 주석 외형·패턴·Type3 글리프는 페이지에서 한 번 실행되는 것으로 센다.

정확한 비용 모델이 아니라 값싼 상한 검사다 — 문자열·주석 안의 'Do'까지 세는 쪽(과대
추정 = 안전한 쪽)으로 틀린다. Type3 글리프 반복·거대 이미지·셰이딩처럼 여기서 세지 않는
비용은 워커 프로세스의 페이지별 시간 상한(pdf_worker)이 받친다.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

# 콘텐츠 스트림의 XObject 호출 `/이름 Do` — 이름은 PDF 구분자 전까지, 뒤에 이름 문자가
# 이어지면(`Dox`) 연산자가 아니다.
_DO_OPERATOR = re.compile(rb"/([^\s/\[\]()<>{}%]+)\s*Do(?![^\s/\[\]()<>{}%])")
_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
# Form 중첩 깊이 상한 — 정상 문서는 한 자릿수다. 그보다 깊은 체인은 단계마다 팬아웃 1로
# 숨겼다가 아래에서 폭발시킬 수 있어 세다 멈추지 않고 거부한다.
MAX_FORM_DEPTH = 64
# 콘텐츠 바이트 검사를 끈 경우(상한 0)에도 호출 수를 세려면 스트림을 풀어야 한다 —
# 압축 폭탄이 메모리를 다 쓰지 않게 이만큼까지만 푼다.
_DECODE_CAP_WHEN_UNLIMITED = 256 * 1024 * 1024
_UNBOUNDED_CALLS = 10**15


class ContentTooComplex(ValueError):
    """업로드 거부 사유 — 사용자 메시지(서버 경로 없음)를 담는다. probe_pdf → API 400."""


@dataclass(frozen=True)
class PageCost:
    page: int            # 1-based
    content_bytes: int
    xobject_calls: int   # 포화값 — 상한+1이면 '상한 초과'


def _decode_name(raw: bytes) -> str:
    return _NAME_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), raw).decode("latin-1")


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
        # 스트림 번호 → (압축 해제 길이, {XObject 이름: Do 횟수})
        self._streams: dict[int, tuple[int, dict[str, int]]] = {}
        # (스트림 번호, 리소스 키) → 한 번 실행 시 펼친 호출 수
        self._calls: dict[tuple, int] = {}
        self._active: set[int] = set()

    # ── PDF 객체 도우미 ─────────────────────────────────────────────────

    def _get(self, obj, key: str):
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

    # ── 스트림 해석 ─────────────────────────────────────────────────────

    def _stream(self, num: int) -> tuple[int, dict[str, int]]:
        cached = self._streams.get(num)
        if cached is not None:
            return cached
        try:
            stm = self.mu.pdf_open_stream_number(self.pdoc, num)
            size = int(self.mu.fz_skip(stm, self._decode_cap + 1))
        except Exception:  # noqa: BLE001 — 깨진 스트림은 렌더도 건너뛴다
            size = 0
        calls: dict[str, int] = {}
        if 0 < size <= self._decode_cap and (not self.max_bytes or size <= self.max_bytes):
            try:
                data = self.doc.xref_stream(num) or b""
            except Exception:  # noqa: BLE001
                data = b""
            for match in _DO_OPERATOR.finditer(data[: self._decode_cap]):
                name = _decode_name(match.group(1))
                calls[name] = calls.get(name, 0) + 1
        self._streams[num] = (size, calls)
        return size, calls

    def _expand(self, num: int, resources, res_key: tuple, depth: int) -> int:
        """스트림을 한 번 실행할 때 일어나는 XObject 호출 수(중첩을 펼친 값, 포화)."""
        key = (num, res_key)
        cached = self._calls.get(key)
        if cached is not None:
            return cached
        if num in self._active:
            return 0  # 순환 Form — MuPDF는 재귀 XObject를 다시 실행하지 않는다
        if depth > MAX_FORM_DEPTH:
            raise _TooDeep()
        self._active.add(num)
        try:
            _size, calls = self._stream(num)
            xobjects = self._get(resources, "XObject")
            total = 0
            for name, count in calls.items():
                target = self._get(xobjects, name)
                if target is None or not self.mu.pdf_is_stream(target):
                    continue
                total += count
                if self._is_form(target):
                    sub_res, sub_key = self._own_resources(target, resources, res_key)
                    inner = self._expand(self.mu.pdf_to_num(target), sub_res, sub_key, depth + 1)
                    total += count * inner
                if total >= self._call_cap:
                    total = self._call_cap
                    break
        finally:
            self._active.discard(num)
        self._calls[key] = total
        return total

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

        calls = 0
        for stream in contents:
            calls = self._add(calls, self._expand(mu.pdf_to_num(stream), resources, res_key, 0))
        for ap, ap_res, ap_key in appearances:
            calls = self._add(calls, 1 + self._expand(mu.pdf_to_num(ap), ap_res, ap_key, 1))

        # 도달하는 스트림을 한 번씩 돌며 바이트를 더하고, 처음 보는 리소스 사전의 패턴·Type3
        # 글리프를 한 번 실행으로 더한다.
        content_bytes = 0
        seen_streams: set[int] = set()
        seen_resources: set[tuple] = set()
        queue = deque([(s, resources, res_key) for s in contents] + appearances)
        while queue:
            obj, res, key = queue.popleft()
            if key not in seen_resources:
                seen_resources.add(key)
                for extra, extra_res, extra_key in self._extra_roots(res, key):
                    calls = self._add(
                        calls, 1 + self._expand(mu.pdf_to_num(extra), extra_res, extra_key, 1),
                    )
                    queue.append((extra, extra_res, extra_key))
            num = mu.pdf_to_num(obj)
            if num in seen_streams:
                continue
            seen_streams.add(num)
            size, names = self._stream(num)
            content_bytes += size
            if self.max_bytes and content_bytes > self.max_bytes:
                break
            xobjects = self._get(res, "XObject")
            for name in names:
                target = self._get(xobjects, name)
                if target is not None and mu.pdf_is_stream(target) and self._is_form(target):
                    queue.append((target, *self._own_resources(target, res, key)))
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


class _TooDeep(Exception):
    pass
