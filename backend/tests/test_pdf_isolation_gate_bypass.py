"""업로드 복잡도 게이트 우회 — 스캐너가 MuPDF와 다르게 읽어 과소 집계하던 모양들.

게이트는 '과대 추정 = 안전한 쪽'이어야 한다. 콘텐츠를 MuPDF와 다르게 토큰화하면 그 차이가
곧 우회로였다(감사 isolation-1): `/Fm#E9`처럼 비ASCII 바이트를 이스케이프한 이름은 리소스를
찾지 못해(latin-1 해석 + C 문자열 조회의 TypeError) 10^4 체인을 0회로 셌고, 콘텐츠 바이트·
깊이 검사도 함께 꺼졌다. 같은 계열로 `/Bomb 5 Do`·`/Bomb (x) Do`·주석·여러 콘텐츠 스트림에
걸친 피연산자, `/A /B Do`(MuPDF는 A를 쓴다), 순환 Form 메모가 숨긴 팬아웃도 과소 집계됐다.

픽스처는 바이트로 직접 합성한다(이름 바이트를 그대로 통제하려고). 각 우회 모양은 MuPDF가
실제로 그 무거운 Form을 실행한다는 사실(그린 path 수)을 함께 확인해 픽스처 자체를 검증한다.
"""

from __future__ import annotations

import random
import time

import pytest

from app.pipeline.pdf import probe_pdf


def _fitz():
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    return pymupdf


class _RawPdf:
    """객체 바이트를 그대로 넣는 최소 PDF 작성기 — 페이지 1장."""

    def __init__(self) -> None:
        self.objects: dict[int, bytes] = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        }
        self.next = 4  # 3은 페이지

    def add(self, body: bytes) -> int:
        num = self.next
        self.objects[num] = body
        self.next += 1
        return num

    @staticmethod
    def _stream(head: bytes, data: bytes) -> bytes:
        return b"<< " + head + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"

    @staticmethod
    def _xobject_dict(xobjects: dict[bytes, int]) -> bytes:
        return b"<< /XObject << " + b" ".join(
            b"/" + name + b" %d 0 R" % num for name, num in xobjects.items()
        ) + b" >> >>"

    def form(
        self, content: bytes, xobjects: dict[bytes, int] | None = None, *,
        shared: int | None = None,
    ) -> int:
        """Form XObject — 리소스는 자기 XObject 사전, 또는 shared(공유 리소스 객체 번호)."""
        resources = (
            b"%d 0 R" % shared if shared is not None else self._xobject_dict(xobjects or {})
        )
        head = b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources " + resources
        return self.add(self._stream(head, content))

    def page(self, contents: list[bytes], xobjects: dict[bytes, int]) -> None:
        refs = b" ".join(b"%d 0 R" % self.add(self._stream(b"", c)) for c in contents)
        self.objects[3] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources "
            + self._xobject_dict(xobjects) + b" /Contents [" + refs + b"] >>"
        )

    def bytes(self) -> bytes:
        out = b"%PDF-1.7\n"
        offsets = {}
        last = max(self.objects)
        for num in range(1, last + 1):
            offsets[num] = len(out)
            out += b"%d 0 obj\n" % num + self.objects.get(num, b"null") + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (last + 1)
        for num in range(1, last + 1):
            out += b"%010d 00000 n \n" % offsets[num]
        return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
            last + 1, xref,
        )


LEAF = b"0 0 m 1 1 l S\n"  # 실행될 때마다 path 1개 — MuPDF가 실제로 실행한 Form 수를 센다


def _chain(pdf: _RawPdf, levels: int, fanout: int, name: bytes = b"X") -> int:
    """리프부터 levels단계 — 단계마다 아래를 fanout번 부르는 Form 체인(리프 실행 fanout^levels)."""
    below = pdf.form(LEAF)
    for _ in range(levels):
        below = pdf.form((b"q /" + name + b" Do Q\n") * fanout, {name: below})
    return below


def _drawn_paths(data: bytes) -> int:
    fitz = _fitz()
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return len(doc[0].get_drawings())
    finally:
        doc.close()


@pytest.fixture
def small_limits(monkeypatch):
    """작은 픽스처로 상한을 넘기려고 상한을 낮춘다(1MB·1,000회)."""
    monkeypatch.setenv("PDF_MAX_PAGE_CONTENT_MB", "1")
    monkeypatch.setenv("PDF_MAX_PAGE_XOBJECT_CALLS", "1000")


def _probe(tmp_path, data: bytes, name: str = "x.pdf") -> int:
    path = tmp_path / name
    path.write_bytes(data)
    return probe_pdf(path, max_pages=10)


# ── 이름: MuPDF와 같은 문자열로 풀고 사전 키를 순회해 찾는다 ──────────────────

ESCAPED_NAMES = {
    "hex_e9": (b"Fm#E9", b"Fm#E9"),             # UTF-8이 아닌 바이트 → 'Fm\udce9'
    "utf8": (b"Fm#C3#A9", b"Fm#C3#A9"),         # 유효한 UTF-8 → 'Fmé'
    "raw_e9": (b"Fm\xe9", b"Fm\xe9"),           # 이스케이프 없는 날 바이트
    "hex_ff_lower": (b"Fm#ff", b"Fm#FF"),       # 소문자 16진도 같은 바이트
    "nul_kept": (b"#00x", b"#00x"),             # MuPDF는 #00을 풀지 않는다
    "truncated": (b"F" * 300, b"F" * 255),      # 콘텐츠 이름은 255바이트에서 잘린다
}


@pytest.mark.parametrize("label", sorted(ESCAPED_NAMES))
def test_escaped_or_long_names_reach_the_nested_chain(tmp_path, small_limits, label):
    """감사 재현: 같은 10^3 체인이 /Fm1로 부르면 거부되고 /Fm#E9로 부르면 통과했다."""
    content_name, key_name = ESCAPED_NAMES[label]
    pdf = _RawPdf()
    top = _chain(pdf, levels=3, fanout=10)
    pdf.page([b"q /" + content_name + b" Do Q"], {key_name: top})
    data = pdf.bytes()
    assert _drawn_paths(data) == 1000  # MuPDF는 그 이름으로 체인을 실제로 실행한다
    with pytest.raises(ValueError, match="중첩 그리기 호출"):
        _probe(tmp_path, data)


def test_content_bytes_behind_an_escaped_name_are_counted(tmp_path, small_limits):
    """비ASCII 이름 뒤의 2MB Form — 바이트 상한(1MB)도 함께 꺼졌다."""
    pdf = _RawPdf()
    big = pdf.form(LEAF * 150_000)
    pdf.page([b"/Fm#E9 Do"], {b"Fm#E9": big})
    with pytest.raises(ValueError, match="콘텐츠가 너무 큽니다"):
        _probe(tmp_path, pdf.bytes())


def test_depth_limit_applies_through_escaped_names(tmp_path):
    """팬아웃 1로 70단계 — 이름을 못 찾으면 재귀하지 않아 깊이 검사도 건너뛰었다."""
    pdf = _RawPdf()
    top = _chain(pdf, levels=70, fanout=1, name=b"X#E9")
    pdf.page([b"/X#E9 Do"], {b"X#E9": top})
    with pytest.raises(ValueError, match="64단계"):
        _probe(tmp_path, pdf.bytes())


def test_decoded_names_equal_the_keys_mupdf_reports():
    from app.pipeline.pdf_complexity import _decode_name

    fitz = _fitz()
    names = [b"Fm#E9", b"Fm#C3#A9", b"Fm\xe9", b"A#41#42", b"#00x", b"x#0G", b"q#20r", b"F#2Fx"]
    pdf = _RawPdf()
    leaf = pdf.form(LEAF)
    pdf.page([b""], {name: leaf for name in names})
    doc = fitz.open(stream=pdf.bytes(), filetype="pdf")
    try:
        mu = fitz.mupdf
        page = mu.pdf_lookup_page_obj(mu.pdf_document_from_fz_document(doc.this), 0)
        xobjects = mu.pdf_dict_gets(mu.pdf_dict_gets(page, "Resources"), "XObject")
        keys = {
            mu.pdf_to_name(mu.pdf_dict_get_key(xobjects, i))
            for i in range(mu.pdf_dict_len(xobjects))
        }
    finally:
        doc.close()
    assert {_decode_name(name) for name in names} == keys
    assert _decode_name(b"Fm#E9") == "Fm\udce9" and _decode_name(b"#00x") == "#00x"
    assert len(_decode_name(b"F" * 300)) == 255


# ── 피연산자: MuPDF는 직전 연산자 이후 '처음' 나온 이름을 쓴다 ───────────────

OPERAND_TRICKS = {
    "number_between": [b"/Bomb 5 Do"],
    "string_between": [b"/Bomb (x) Do"],
    "dict_between": [b"/Bomb <</A 1>> Do"],
    "array_between": [b"/Bomb [1] Do"],
    "comment_between": [b"/Bomb %c\nDo"],
    "first_name_wins": [b"/Bomb /Cheap Do"],
    "comment_hides_operator": [b"/Bomb % q\n/Cheap Do"],
    "comment_in_string_line": [b"/Bomb (q /Cheap %)\nDo"],
    "split_across_streams": [b"q /Bomb", b"Do Q"],
    "comment_spans_streams": [b"/Bomb %", b"q\n/Cheap Do"],
    "nul_whitespace": [b"/Bomb\x00Do"],
}


@pytest.mark.parametrize("label", sorted(OPERAND_TRICKS))
def test_do_operand_tricks_are_not_undercounted(tmp_path, small_limits, label):
    pdf = _RawPdf()
    bomb = _chain(pdf, levels=3, fanout=10)
    cheap = pdf.form(LEAF)
    pdf.page(OPERAND_TRICKS[label], {b"Bomb": bomb, b"Cheap": cheap})
    data = pdf.bytes()
    assert _drawn_paths(data) == 1000  # MuPDF가 실행하는 것은 Bomb다
    with pytest.raises(ValueError, match="중첩 그리기 호출"):
        _probe(tmp_path, data)


def test_ordinary_do_operands_are_still_counted_exactly():
    """정상 콘텐츠의 `q … cm /이름 Do Q`는 그대로 센다 — 최악 추정으로 부풀리지 않는다."""
    from app.pipeline.pdf_complexity import scan_calls

    body = b"q 1 0 0 1 10 10 cm /Im0 Do Q\nq /Fm0 Do Q /Fm0 Do\nBT (x) Tj ET q\x00/Im0\x00Do Q"
    assert scan_calls(b"/Fm0 Do " + body) == ({"Fm0": 3, "Im0": 2}, 0)
    # 실행 중간에 이어지는 스트림의 첫 줄은 앞 스트림의 주석·피연산자가 이어질 수 있다
    assert scan_calls(b"/Fm0 Do\nq /Fm0 Do", fresh=False) == ({"Fm0": 1}, 1)
    # 문자열·주석 안의 'Do'까지 후보로 센다(과대 추정)
    assert scan_calls(b"(Do) Tj % Do\n") == ({}, 2)


# ── 순환 Form: 끊은 결과를 맥락과 무관하게 메모하면 팬아웃이 숨는다 ──────────────


def test_cycles_cannot_hide_fanout(tmp_path, small_limits):
    """A↔B 순환에서 A가 리프 50개를 부른다. 페이지가 B를 20번 부르면 B마다 A의 50개가 실행된다.

    예전에는 A를 펼치는 중에 'A가 끊긴' B의 값을 메모해, 페이지의 B 20번을 거의 공짜로 셌다
    (MuPDF 1,050회 실행 대 집계 93회)."""
    pdf = _RawPdf()
    leaf = pdf.form(LEAF)
    shared = pdf.add(b"null")
    a = pdf.form(b"/B Do\n" + b"/L Do\n" * 50, shared=shared)
    b = pdf.form(b"/A Do\n", shared=shared)
    pdf.objects[shared] = _RawPdf._xobject_dict({b"A": a, b"B": b, b"L": leaf})
    pdf.page([b"/A Do\n" + b"/B Do\n" * 20], {b"A": a, b"B": b, b"L": leaf})
    data = pdf.bytes()
    assert _drawn_paths(data) == 1050
    with pytest.raises(ValueError, match="중첩 그리기 호출"):
        _probe(tmp_path, data)


def test_tangled_cycles_are_rejected_quickly(tmp_path):
    """서로를 모두 부르는 Form 12개 — 단순 경로가 수억 개라 끝까지 펼치지 않고 거부한다."""
    pdf = _RawPdf()
    count = 12
    first = pdf.next
    shared = first + count
    for i in range(count):
        pdf.form(b"".join(b"q /F%d Do Q\n" % j for j in range(count) if j != i), shared=shared)
    pdf.add(_RawPdf._xobject_dict({b"F%d" % i: first + i for i in range(count)}))
    pdf.page([b"/F0 Do"], {b"F0": first})
    started = time.monotonic()
    with pytest.raises(ValueError, match="순환|중첩 그리기 호출"):
        _probe(tmp_path, pdf.bytes())
    assert time.monotonic() - started < 5


# ── 무작위 콘텐츠: 스캐너 집계 ≥ MuPDF가 실제로 실행한 Form 수 ─────────────────

_FUZZ_NAMES = [b"A", b"B", b"Fm#E9", b"Fm#C3#A9", b"#00x", b"Q#20R", b"F" * 260, b"Im#41"]


def _fuzz_token(rng: random.Random, names: list[bytes]) -> bytes:
    roll = rng.random()
    if roll < 0.25:
        return b"/" + rng.choice(names)
    if roll < 0.40:
        return b"Do"
    if roll < 0.50:
        return rng.choice([b"q", b"Q", b"cm", b"gs", b"BT", b"ET", b"true", b"EI", b"x"])
    if roll < 0.60:
        return rng.choice([b"1", b"-2.5", b"+3", b".5", b"0"])
    if roll < 0.66:
        return b"(" + rng.choice([b"a", b"%", b"q /A Do", b")", b"\\)", b"(n)"]) + b")"
    if roll < 0.72:
        return b"%" + rng.choice([b"", b" q", b" /A Do"]) + rng.choice([b"\n", b"\r", b""])
    if roll < 0.77:
        return b"[" + rng.choice([b"/A", b"(x)", b"1", b""]) + b"]"
    if roll < 0.81:
        return b"<<" + rng.choice([b"/A 1", b"", b"/B /A"]) + b">>"
    if roll < 0.84:
        return b"<" + rng.choice([b"41", b"q", b""]) + b">"
    if roll < 0.86:
        return b"BI /W 1 /H 1 /BPC 8 /CS /G ID " + rng.choice([b"x", b"q /A Do x"]) + b" EI"
    return rng.choice([b" ", b"\n", b"\x00", b"\r\n", b"\t"])


def _fuzz_stream(rng: random.Random, names: list[bytes], tokens: int) -> bytes:
    return b"".join(
        _fuzz_token(rng, names) + rng.choice([b" ", b"\n", b"", b"\x00"]) for _ in range(tokens)
    )


def test_scanner_never_undercounts_mupdf_on_random_content():
    """이름 이스케이프·주석·문자열·배열·사전·인라인 이미지·여러 콘텐츠 스트림·순환을 섞은 무작위
    콘텐츠에서 집계가 MuPDF의 실제 실행 수보다 작으면 그것이 곧 우회로다(예전 스캐너는 약
    4분의 1에서 과소 집계)."""
    from app.pipeline.pdf_complexity import ComplexityScanner

    fitz = _fitz()
    undercounts = []
    for seed in range(400):
        rng = random.Random(seed)
        keys = rng.sample(_FUZZ_NAMES, k=rng.randint(1, 4))
        pool = keys + [b"Nope"]
        pdf = _RawPdf()
        shared = pdf.add(b"null")
        forms = []
        for _ in keys:
            body = LEAF + _fuzz_stream(rng, pool, rng.randint(0, 8))
            forms.append(pdf.form(body, shared=shared))
        mapping = dict(zip(keys, forms))
        pdf.objects[shared] = _RawPdf._xobject_dict(mapping)
        pdf.page(
            [_fuzz_stream(rng, pool, rng.randint(1, 20)) for _ in range(rng.randint(1, 3))],
            mapping,
        )
        data = pdf.bytes()
        drawn = _drawn_paths(data)
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            scanner = ComplexityScanner(
                fitz, doc, max_content_bytes=64 << 20, max_xobject_calls=10**9,
            )
            counted = scanner.page_cost(0).xobject_calls
        finally:
            doc.close()
        if counted < drawn:
            undercounts.append((seed, drawn, counted))
    assert not undercounts, undercounts[:5]
