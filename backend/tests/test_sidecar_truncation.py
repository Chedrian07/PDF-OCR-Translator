"""sidecar 출력 잘림(page.truncated)·로드 재시도·재시작 대기 (감사 sidecar-11·6·7).

잘린 페이지는 원본 PDF 텍스트 레이어와 대조해 잃은 것이 크면 runner의 잘림 복구
(텍스트 레이어 폴백)로 넘기고, 스캔 문서처럼 대조할 수 없거나 충실도가 충분하면 잘린
출력을 경고와 함께 그대로 쓴다 — 텍스트 레이어가 없는 페이지를 플레이스홀더로 만들지 않는다.
"""

import json
import threading
from io import BytesIO

import pytest
from PIL import Image

from app.config import Settings
from app.engine.base import NullSink, OutputLimitError
from app.engine.registry import build_engine
from app.engine.sidecar import SidecarNotReadyError, SidecarOutputTruncated
from app.main import create_app

from tests.conftest import wait_done
from tests.test_sidecar_client import StubSidecar, _health_body, _parse_body

# 텍스트 레이어 정답(충실도 판정 최소 200자)을 넉넉히 넘는 서로 다른 문장들
SENTENCES = [
    "Optical character recognition converts scanned pages into machine readable text.",
    "The pipeline renders every page before sending it to the recognition model.",
    "Figures are cropped from the page image using normalized bounding boxes.",
    "Tables are emitted as HTML so that merged cells survive the conversion.",
    "Mathematical formulas are written in LaTeX and rendered by the viewer.",
    "Each chunk of pages is merged into a single Markdown document at the end.",
    "Translation runs per unit and keeps placeholders for protected tokens.",
    "Quality gates compare the output with the embedded text layer of the file.",
    "Pages that lose most of their content are recovered from that text layer.",
    "Scanned documents have no text layer, so their output is kept as it is.",
]
SIDECAR_WARNING = "출력 토큰 상한(8192)에 도달 — 페이지 끝부분이 잘렸을 수 있습니다"


@pytest.fixture
def stub():
    s = StubSidecar()
    yield s
    s.close()


def _settings(tmp_path, stub, **kw) -> Settings:
    base = dict(
        engine="ovisocr2", device="cpu", data_dir=tmp_path / "data",
        preload_model=False, frontend_dir=tmp_path / "no-frontend",
        sidecar_url=stub.url, sidecar_connect_timeout_s=2.0,
        sidecar_read_timeout_s=10.0, sidecar_health_timeout_s=2.0,
    )
    base.update(kw)
    return Settings(**base)


def _text_pdf(pages: int = 1) -> bytes:
    import pymupdf as fitz

    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(56, 56, 540, 800), "\n".join(SENTENCES), fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf() -> bytes:
    """텍스트 레이어가 없는 스캔 페이지 — 그림만 있다."""
    import pymupdf as fitz

    buf = BytesIO()
    Image.new("RGB", (300, 420), (240, 240, 240)).save(buf, "PNG")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def _job(tmp_path, pdf: bytes, pages: int = 1) -> list:
    """runner 규약의 잡 디렉터리 — {job}/source.pdf + {job}/pages/page_%04d.png."""
    job = tmp_path / "job"
    (job / "pages").mkdir(parents=True)
    (job / "source.pdf").write_bytes(pdf)
    paths = []
    for i in range(1, pages + 1):
        path = job / "pages" / f"page_{i:04d}.png"
        Image.new("RGB", (400, 560), "white").save(path)
        paths.append(path)
    return paths


def _body(markdown: str, truncated: bool = True, figure: bool = True) -> bytes:
    body = _parse_body()
    body["page"]["markdown"] = markdown + ("\n\n[[FIGURE:0]]" if figure else "")
    if not figure:
        body["page"]["blocks"] = []
    body["page"]["warnings"] = [SIDECAR_WARNING] if truncated else []
    if truncated:
        body["page"]["truncated"] = True
    return json.dumps(body).encode()


HEAD_ONLY = SENTENCES[0]                  # 첫 문장만 남기고 잘림 — 충실도 ≪ 0.70
NEARLY_ALL = "\n\n".join(SENTENCES[:-1])  # 마지막 문장만 잘림 — 충실도 ≥ 0.70


def _engine(tmp_path, stub, **kw):
    eng = build_engine(_settings(tmp_path, stub, **kw))
    eng.load()
    return eng


# ── 엔진 단위 판정 ─────────────────────────────────────────────────────────

def test_badly_truncated_page_is_handed_to_the_text_layer_recovery(tmp_path, stub):
    (path,) = _job(tmp_path, _text_pdf())
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub)
    out = tmp_path / "single"
    with pytest.raises(OutputLimitError) as ei:
        eng.run_single(path, out, NullSink(), threading.Event())
    err = ei.value
    assert isinstance(err, SidecarOutputTruncated)
    assert err.retry_same_page is False
    assert "충실도" in str(err) and "< 0.70" in str(err)
    # partial_output은 run_single 반환 형식 그대로 — 산출물은 out_dir에 있다
    assert HEAD_ONLY in err.partial_output and "![](images/0.jpg)" in err.partial_output
    assert (out / "images" / "0.jpg").is_file()
    # 버리는 페이지의 경고(sidecar의 상한 도달 문구)는 잡 경고로 올리지 않는다 —
    # 그 자리는 runner가 텍스트 레이어로 채우고 사유를 남긴다
    assert SIDECAR_WARNING not in eng.drain_warnings()


def test_truncated_scanned_page_keeps_its_output(tmp_path, stub):
    """텍스트 레이어가 없으면 넘겨 봐야 플레이스홀더가 된다 — 잘린 출력이 낫다."""
    (path,) = _job(tmp_path, _scanned_pdf())
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())
    assert HEAD_ONLY in md and "![](images/0.jpg)" in md
    warnings = eng.drain_warnings()
    keep = [i for i, w in enumerate(warnings)
            if "대조할 PDF 텍스트 레이어가 없어 잘린 출력을 그대로" in w]
    assert keep and warnings.index(SIDECAR_WARNING) < keep[0], warnings  # 잘림 → 처리 순서


@pytest.mark.parametrize("failure,reason", [
    ("timeout", "분석 시간 상한 초과 — 판정 생략; 상한은 PDF_PAGE_TIMEOUT_S"),
    ("crash", "분석 중 처리 프로세스가 비정상 종료 — 판정 생략"),
    ("broken_pdf", "원본 PDF 열기 실패"),
])
def test_unjudged_truncated_page_names_the_real_reason(tmp_path, stub, monkeypatch,
                                                       failure, reason):
    """대조를 못 한 것(분석 시간 상한·처리 프로세스 사망·원본 열기 실패)은 '텍스트 레이어가
    없다'가 아니다 — 텍스트 레이어가 멀쩡한 페이지의 경고가 엉뚱한 원인을 적으면 운영자가
    진짜 원인(PDF_PAGE_TIMEOUT_S 등)을 놓친다. 잘린 출력은 그대로 쓴다."""
    from app.pipeline import pdf_worker

    (path,) = _job(tmp_path, b"%PDF-1.7 broken" if failure == "broken_pdf" else _text_pdf())
    errors = {"timeout": pdf_worker.PdfWorkerTimeout(30.0),
              "crash": pdf_worker.PdfWorkerCrashed(-9)}
    if failure in errors:
        def _fail(*a, **k):
            raise errors[failure]

        monkeypatch.setattr(pdf_worker, "run_page", _fail)
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())

    assert HEAD_ONLY in md
    warnings = eng.drain_warnings()
    expected = f"출력 토큰 상한에서 잘린 페이지 — 텍스트 레이어로 판정할 수 없어({reason}) 잘린 출력을 그대로 씁니다"
    assert expected in warnings, warnings
    assert not any("텍스트 레이어가 없어" in w for w in warnings), warnings


def test_truncation_judgment_error_is_not_reported_as_a_missing_text_layer(
    tmp_path, stub, monkeypatch
):
    (path,) = _job(tmp_path, _text_pdf())

    def _boom(*a, **k):
        raise RuntimeError("판정 중 예기치 못한 오류")

    monkeypatch.setattr("app.engine.sidecar._truncated_page_fidelity", _boom)
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())

    assert HEAD_ONLY in md
    warnings = eng.drain_warnings()
    assert any("텍스트 레이어 대조가 실패해(RuntimeError) 잘린 출력을 그대로" in w
               for w in warnings), warnings
    assert not any("텍스트 레이어가 없어" in w for w in warnings), warnings


def test_slightly_truncated_page_keeps_its_structured_output(tmp_path, stub):
    """끝부분만 조금 잘린 페이지를 평문 텍스트 레이어로 바꾸면 표·수식·그림 구조를 잃는다."""
    (path,) = _job(tmp_path, _text_pdf())
    stub.parse_behavior = lambda: (200, _body(NEARLY_ALL))
    eng = _engine(tmp_path, stub)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())
    assert SENTENCES[-2] in md
    warnings = eng.drain_warnings()
    assert SIDECAR_WARNING in warnings
    assert any("기준(0.70) 이상이라 잘린 출력을 그대로" in w for w in warnings), warnings


def test_disabled_fidelity_threshold_keeps_truncated_output(tmp_path, stub):
    (path,) = _job(tmp_path, _text_pdf())
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub, ocr_fidelity_threshold=0.0)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())
    assert HEAD_ONLY in md
    assert any("OCR_FIDELITY_THRESHOLD≤0" in w for w in eng.drain_warnings())


def test_untruncated_page_is_never_judged(tmp_path, stub, monkeypatch):
    """잘림 표시가 없는 페이지(옛 sidecar 포함)는 원본을 열어 보지도 않는다."""
    (path,) = _job(tmp_path, _text_pdf())
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY, truncated=False))
    calls = []
    monkeypatch.setattr("app.engine.sidecar._truncated_page_fidelity",
                        lambda *a, **k: calls.append(a))
    eng = _engine(tmp_path, stub)
    md = eng.run_single(path, tmp_path / "single", NullSink(), threading.Event())
    assert HEAD_ONLY in md and calls == []
    assert not any("잘린" in w for w in eng.drain_warnings())


def test_multi_hand_over_keeps_leading_pages_and_skips_the_same_page_rerun(tmp_path, stub):
    """run_multi의 partial_output은 끝까지 받은 앞 페이지 + 잘린 페이지(`<PAGE>` 구분)다.
    runner가 곧바로 그 페이지를 run_single로 다시 부르면 GPU에 다시 보내지 않는다."""
    p1, p2 = _job(tmp_path, _text_pdf(pages=2), pages=2)
    bodies = iter([_body("\n\n".join(SENTENCES), truncated=False), _body(HEAD_ONLY)])
    stub.parse_behavior = lambda: (200, next(bodies))
    eng = _engine(tmp_path, stub)
    out = tmp_path / "chunk_00"
    with pytest.raises(SidecarOutputTruncated) as ei:
        eng.run_multi([p1, p2], out, NullSink(), threading.Event())
    partial = ei.value.partial_output
    assert partial.count("<PAGE>") == 2 and partial.startswith("<PAGE>\n")
    assert "![](images/page_0_0.jpg)" in partial and HEAD_ONLY in partial
    assert (out / "images" / "page_0_0.jpg").is_file() and (out / "boxes.json").is_file()
    assert len(stub.requests_seen) == 2

    # runner의 페이지 단위 재처리 — 같은 판정을 즉시 낸다(요청 없음)
    with pytest.raises(SidecarOutputTruncated, match="충실도"):
        eng.run_single(p2, tmp_path / "fallback", NullSink(), threading.Event())
    assert len(stub.requests_seen) == 2
    # 표식은 1회용 — 그다음 호출은 다시 sidecar로 간다
    stub.parse_behavior = lambda: (200, _body("\n\n".join(SENTENCES), truncated=False))
    eng.run_single(p2, tmp_path / "again", NullSink(), threading.Event())
    assert len(stub.requests_seen) == 3


def test_replay_mark_does_not_outlive_the_next_chunk(tmp_path, stub):
    p1, p2 = _job(tmp_path, _text_pdf(pages=2), pages=2)
    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    eng = _engine(tmp_path, stub)
    with pytest.raises(SidecarOutputTruncated):
        eng.run_multi([p2], tmp_path / "c0", NullSink(), threading.Event())
    stub.parse_behavior = lambda: (200, _body("\n\n".join(SENTENCES), truncated=False))
    eng.run_multi([p1], tmp_path / "c1", NullSink(), threading.Event())  # 다음 청크
    seen = len(stub.requests_seen)
    eng.run_single(p2, tmp_path / "late", NullSink(), threading.Event())
    assert len(stub.requests_seen) == seen + 1


# ── runner까지 (업로드 → 잡 done) ────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["multi", "per_page"])
def test_job_recovers_a_badly_truncated_page_from_the_text_layer(tmp_path, stub, mode):
    from fastapi.testclient import TestClient

    bodies = iter([_body("\n\n".join(SENTENCES), truncated=False), _body(HEAD_ONLY)])
    stub.parse_behavior = lambda: (200, next(bodies))
    with TestClient(create_app(_settings(tmp_path, stub))) as c:
        r = c.post("/api/jobs", data={"mode": mode},
                   files={"file": ("doc.pdf", BytesIO(_text_pdf(pages=2)), "application/pdf")})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        body = wait_done(c, job_id, timeout=30)
        md = c.get(f"/api/jobs/{job_id}/markdown").text
    assert body["status"] == "done", body
    page1, page2 = md.split("\n\n---\n\n")
    assert "![](images/p0001_0.jpg)" in page1 and SENTENCES[-1] in page1
    # 2쪽은 잘린 OCR 대신 텍스트 레이어 전문 — 마지막 문장까지 있다
    assert "PDF 내장 텍스트 레이어에서 복구" in page2 and SENTENCES[-1] in page2
    assert any("2페이지" in w and "텍스트 레이어로 복구" in w for w in body["warnings"]), body
    assert not any(SIDECAR_WARNING in w for w in body["warnings"]), body["warnings"]
    # 원인은 sidecar의 출력 상한(limit_label)으로 적는다 — sidecar에는 효과가 없는
    # MAX_LENGTH를 가리키지 않는다(multi는 청크 재처리 참고, per_page는 복구 경고)
    recovered = [w for w in body["warnings"] if "텍스트 레이어로 복구" in w]
    assert all("sidecar 출력 토큰 상한 도달(출력 잘림)" in w for w in recovered), recovered
    if mode == "multi":
        assert any(
            "sidecar 출력 토큰 상한 도달로 출력이 잘려 페이지별 재처리" in n for n in body["notices"]
        ), body["notices"]
    assert not any("MAX_LENGTH" in m for m in body["warnings"] + body["notices"]), body
    assert len(stub.requests_seen) == 2  # 같은 페이지를 GPU에서 다시 돌리지 않았다


def test_job_keeps_a_truncated_scanned_page(tmp_path, stub):
    from fastapi.testclient import TestClient

    stub.parse_behavior = lambda: (200, _body(HEAD_ONLY))
    with TestClient(create_app(_settings(tmp_path, stub))) as c:
        r = c.post("/api/jobs", data={"mode": "multi"},
                   files={"file": ("scan.pdf", BytesIO(_scanned_pdf()), "application/pdf")})
        job_id = r.json()["job_id"]
        body = wait_done(c, job_id, timeout=30)
        md = c.get(f"/api/jobs/{job_id}/markdown").text
    assert body["status"] == "done", body
    assert HEAD_ONLY in md and "변환에 실패" not in md
    assert any(SIDECAR_WARNING in w for w in body["warnings"]), body["warnings"]
    assert any("잘린 출력을 그대로" in w for w in body["warnings"]), body["warnings"]


# ── 로드 재시도 · 재시작 대기 (health 추가 필드) ──────────────────────────────

RETRY = {"attempt": 2, "max_attempts": 5, "next_retry_s": 30.0,
         "last_error": "OSError: hub unreachable"}


def test_provider_health_surfaces_retry_and_restart_state(tmp_path, stub):
    stub.health_response = _health_body(model_loaded=False, load_retry=RETRY)
    ph = build_engine(_settings(tmp_path, stub)).provider_health()
    assert ph["load_retry"] == RETRY and ph["restarting"] is False

    stub.health_response = _health_body(model_loaded=False, restarting=True)
    ph = build_engine(_settings(tmp_path, stub)).provider_health()
    assert ph["restarting"] is True and ph["load_retry"] is None

    stub.health_response = _health_body()  # 옛 sidecar — 필드 없음
    ph = build_engine(_settings(tmp_path, stub)).provider_health()
    assert ph["load_retry"] is None and ph["restarting"] is False


def test_load_retry_is_a_wait_with_its_own_message(tmp_path, stub):
    stub.health_response = _health_body(model_loaded=False, load_retry=RETRY)
    eng = build_engine(_settings(tmp_path, stub))
    with pytest.raises(SidecarNotReadyError) as ei:
        eng.load()
    assert ei.value.transient is True
    assert "2/5번째 시도 실패, 30초 뒤 재시도 — 마지막 오류: OSError: hub unreachable" in str(ei.value)
    assert "재시도 대기" in ei.value.note


@pytest.mark.parametrize("status", ["ok", "error"])
def test_restarting_sidecar_is_awaited_even_if_it_reports_an_error(tmp_path, stub, status):
    """엔진 사망 뒤 컨테이너 재시작 중이면 기다리면 풀린다 — 하드 실패로 끝내지 않는다."""
    stub.health_response = _health_body(model_loaded=False, restarting=True, status=status,
                                        load_error="엔진 비정상" if status == "error" else None)
    eng = build_engine(_settings(tmp_path, stub))
    with pytest.raises(SidecarNotReadyError, match="재시작하는 중") as ei:
        eng.load()
    assert "재시작 대기" in ei.value.note


def test_wait_note_follows_the_sidecar_state(tmp_path, stub, monkeypatch):
    """잡 시작 대기의 진행 문구가 첫 로드·로드 재시도·재시작 대기를 구분한다."""
    from app.sidecar.protocol import SidecarHealth

    states = iter(SidecarHealth.model_validate(b) for b in [
        _health_body(model_loaded=False, load_retry=RETRY),
        _health_body(model_loaded=False, restarting=True),
        _health_body(model_loaded=False),
    ])
    final = SidecarHealth.model_validate(_health_body())
    monkeypatch.setattr("app.engine.sidecar._MODEL_WAIT_POLL_S", 0.01)
    eng = build_engine(_settings(tmp_path, stub))
    monkeypatch.setattr(eng._client, "health", lambda: next(states, final))
    notes: list[str] = []
    eng.wait_until_ready(threading.Event(), on_wait=notes.append)
    assert len(notes) == 3
    assert "재시도 대기" in notes[0] and "재시작 대기" in notes[1] and "모델 로딩 대기" in notes[2]
    assert eng.loaded
