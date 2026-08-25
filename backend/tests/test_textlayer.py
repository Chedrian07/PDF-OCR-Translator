"""textlayer 엔진(텍스트 레이어 + Tesseract, Localight 이식) 테스트.

conftest의 settings/client 픽스처는 engine='fake' 고정이라 여기서 동일 구성을
engine='textlayer'로 로컬 구성한다 (conftest는 수정하지 않는다). Tesseract가
설치되지 않은 환경에서도 결정적으로 돌도록 OCR 경로는 모듈 심
(find_tesseract/run_tesseract)을 monkeypatch로 대체한다.
"""

import shutil

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import textlayer as textlayer_mod
from app.engine.registry import build_engine
from app.engine.textlayer import TextLayerEngine, sanitize_text
from app.main import create_app
from conftest import make_pdf_bytes, wait_done


def _settings(tmp_path, **kw) -> Settings:
    # conftest의 settings 픽스처와 동일 구성 + engine='textlayer'.
    # native_text_threshold는 테스트 PDF의 텍스트 양(페이지당 영숫자 ~40자)에
    # 맞춰 낮춘다 — 텍스트 페이지는 결정적으로 네이티브 경로, 빈 페이지는 OCR 경로.
    kw.setdefault("native_text_threshold", 10)
    return Settings(
        engine="textlayer",
        device="cpu",
        data_dir=tmp_path / "data",
        preload_model=False,
        frontend_dir=tmp_path / "no-frontend",  # 정적 마운트 비활성화
        **kw,
    )


def _client(tmp_path, **kw) -> TestClient:
    return TestClient(create_app(_settings(tmp_path, **kw)))


def _upload(client, pdf_bytes: bytes, **data):
    return client.post(
        "/api/jobs",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        data=data,
    )


def make_blank_pdf_bytes(pages: int = 1) -> bytes:
    """텍스트 레이어가 전혀 없는(스캔류) PDF — OCR 경로 강제용."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    data = doc.tobytes()
    doc.close()
    return data


# ── (1) E2E: 텍스트 PDF → 네이티브 텍스트 레이어 경로 ────────────────────


def test_textlayer_e2e_native_text(tmp_path):
    with _client(tmp_path) as client:
        # 로드할 모델이 없다 — health가 시작 즉시 준비 상태를 보고해야 한다
        health = client.get("/api/health").json()
        assert health["engine"] == "textlayer"
        assert health["model_loaded"] is True
        assert health["model_id"] == "pymupdf-textlayer+tesseract"
        assert health["provider"] == "in-process"
        assert health["capabilities"]["layout"] == "full"
        assert health["capabilities"]["stream_granularity"] == "page"
        assert health["capabilities"]["figures"] is False

        r = _upload(client, make_pdf_bytes(pages=3))
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body
        assert body["result"]["has_layout"] is True  # layout.json 기록됨

        md = client.get(f"/api/jobs/{jid}/markdown")
        assert md.status_code == 200
        assert "Sample page 1" in md.text
        assert "Sample page 3" in md.text

        # raw_pages.json det 문법이 layout 뷰 파서(parse_page_blocks)로 소화된다
        layout = client.get(f"/api/jobs/{jid}/layout")
        assert layout.status_code == 200
        assert "Sample page 1" in layout.text


# ── (2) Tesseract 미설치 + 빈 페이지 → 폴백 + 잡당 1회 경고 ──────────────


def test_textlayer_blank_pdf_without_tesseract(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **k: None)
    # 청크=1페이지로 잡아 여러 청크에 걸쳐도 경고가 잡당 1회인지 확인한다
    with _client(tmp_path, pages_per_chunk=1) as client:
        r = _upload(client, make_blank_pdf_bytes(pages=2))
        assert r.status_code == 202, r.text
        body = wait_done(client, r.json()["job_id"])
        assert body["status"] == "done", body
        tesseract_warnings = [w for w in body["warnings"] if "Tesseract" in w]
        assert len(tesseract_warnings) == 1, body["warnings"]

        # 다음 잡에서는 다시 경고가 나온다 (엔진은 잡 간 공유되는 프로세스 전역)
        r2 = _upload(client, make_blank_pdf_bytes(pages=1))
        body2 = wait_done(client, r2.json()["job_id"])
        assert body2["status"] == "done", body2
        assert [w for w in body2["warnings"] if "Tesseract" in w], body2["warnings"]


# ── (2b) 선두 빈 페이지 — 페이지 수 계약(구분자 N-1개)이 유지된다 ─────────


def test_textlayer_blank_first_page_keeps_page_count(tmp_path, monkeypatch):
    """빈 표지 스캔처럼 1쪽이 완전 공백이어도 result.md에서 페이지가 사라지거나
    밀리면 안 된다 (Q&A의 페이지 인덱스·/html 문서 뷰가 result.md 분할에 의존)."""
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **k: None)  # OCR 폴백 차단
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # 1쪽: 텍스트 레이어 없음 (빈 표지)
    for i in (2, 3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Body text page {i}", fontsize=18)
    pdf = doc.tobytes()
    doc.close()

    with _client(tmp_path) as client:
        r = _upload(client, pdf)
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body
        assert body["progress"]["total_pages"] == 3

        md = client.get(f"/api/jobs/{jid}/markdown").text
        pages = [p.strip() for p in md.split("\n\n---\n\n")]
        assert len(pages) == 3, md  # 선두 빈 페이지도 자리(구분자)를 유지한다
        # OCR을 못 돌린 페이지는 완전 공백이 아니라 사유 표식을 갖는다 —
        # 표식이 없으면 전면 스캔 문서가 status=done인 채 빈 문서로 끝난다.
        assert "OCR에 실패" in pages[0] and "Tesseract 미설치" in pages[0]
        assert "Body text page 2" in pages[1]
        assert "Body text page 3" in pages[2]


# ── (3) OCR 경로: Tesseract 러너 출력이 result.md에 실린다 ────────────────


def test_textlayer_ocr_path_uses_tesseract_output(tmp_path, monkeypatch):
    calls: list[tuple[str, str, bytes]] = []

    def _fake_run(executable: str, languages: str, png_bytes: bytes) -> str:
        calls.append((executable, languages, png_bytes))
        return "OCR로 복원된 본문 텍스트입니다"

    monkeypatch.setattr(textlayer_mod, "find_tesseract", lambda: "/fake/tesseract")
    monkeypatch.setattr(textlayer_mod, "run_tesseract", _fake_run)

    with _client(tmp_path) as client:
        r = _upload(client, make_blank_pdf_bytes(pages=1))
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body

        md = client.get(f"/api/jobs/{jid}/markdown").text
        assert "OCR로 복원된 본문 텍스트입니다" in md

        # 이미 렌더된 페이지 PNG 바이트가 설정된 언어로 넘어간다 (Localight 이식 계약)
        assert calls, "tesseract 러너가 호출되지 않음"
        assert calls[0][1] == "eng+kor"
        assert calls[0][2][:8] == b"\x89PNG\r\n\x1a\n"


# ── (3a) OCR 페이지의 레이아웃 raw는 버려진 텍스트 레이어에서 합성되지 않는다 ─


def test_OCR_페이지는_희박한_텍스트레이어_레이아웃을_남기지_않는다(tmp_path, monkeypatch):
    """본문으로 채택되지 않은(임계 미만) 텍스트 레이어로 det 문법을 합성하면
    레이아웃 뷰가 result.md에 없는 문장을 박스로 보여준다."""
    monkeypatch.setattr(textlayer_mod, "find_tesseract", lambda: "/fake/tesseract")
    monkeypatch.setattr(textlayer_mod, "run_tesseract",
                        lambda *a, **k: "OCR로 복원된 본문")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "sparse", fontsize=12)  # 임계 미만 → OCR 경로
    source = tmp_path / "source.pdf"
    doc.save(str(source))
    doc.close()

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    image_path = pages_dir / "page_0001.png"
    src = fitz.open(str(source))
    try:
        src[0].get_pixmap(dpi=72).save(str(image_path))
        engine = TextLayerEngine(_settings(tmp_path, native_text_threshold=1000))
        text, raw = engine._extract_page(src, image_path, tmp_path)
    finally:
        src.close()
    assert text == "OCR로 복원된 본문"
    assert raw == ""  # 버려진 텍스트 레이어의 det 문법을 재사용하지 않는다


# ── (3b) Tesseract 실행 실패는 페이지 단위로 격리된다 ─────────────────────


def test_tesseract_실패는_같은_청크의_정상페이지를_죽이지_않는다(tmp_path, monkeypatch):
    """언어팩 누락(kor.traineddata 없음)·페이지 타임아웃처럼 실행 중 나는 예외가
    청크를 관통하면 8페이지가 통째로 플레이스홀더가 된다 — 페이지 단위 강등."""
    from app.engine.base import EngineError

    def _boom(executable: str, languages: str, png_bytes: bytes) -> str:
        raise EngineError("Tesseract OCR 실패: Error opening data file kor.traineddata")

    monkeypatch.setattr(textlayer_mod, "find_tesseract", lambda: "/fake/tesseract")
    monkeypatch.setattr(textlayer_mod, "run_tesseract", _boom)

    doc = fitz.open()
    for i in (1, 2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Body text page {i}", fontsize=18)
    doc.new_page(width=595, height=842)  # 3쪽: 스캔류 빈 페이지 → OCR 경로에서 실패
    for i in (4, 5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Body text page {i}", fontsize=18)
    pdf = doc.tobytes()
    doc.close()

    with _client(tmp_path) as client:  # 기본 pages_per_chunk=8 → 5쪽이 한 청크
        r = _upload(client, pdf)
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body

        md = client.get(f"/api/jobs/{jid}/markdown").text
        assert "변환에 실패했습니다" not in md  # 청크 전체 플레이스홀더 회귀 방지
        for i in (1, 2, 4, 5):
            assert f"Body text page {i}" in md
        tesseract_warnings = [w for w in body["warnings"] if "Tesseract 실패" in w]
        assert len(tesseract_warnings) == 1, body["warnings"]


# ── (4) 원문 속 리터럴 '<PAGE>'가 페이지 수를 흔들지 않는다 ───────────────


def test_textlayer_literal_page_marker_in_source(tmp_path):
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Marker test page {i + 1}", fontsize=18)
        page.insert_text((72, 120), "The literal token <PAGE> appears here.", fontsize=12)
    pdf = doc.tobytes()
    doc.close()

    with _client(tmp_path) as client:
        r = _upload(client, pdf)
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body
        assert body["progress"]["total_pages"] == 3
        # 마커 과잉/부족 보정 경고가 없어야 한다 (merge.add_chunk의 '페이지 마커' 경고)
        assert not [w for w in body["warnings"] if "페이지 마커" in w], body["warnings"]

        md = client.get(f"/api/jobs/{jid}/markdown").text
        assert "<PAGE>" not in md
        assert "⟨PAGE⟩" in md  # 내용은 시각적 등가 문자로 보존
        assert md.count("\n\n---\n\n") == 2  # 페이지 3장 = 기본 구분자 2개


# ── (4b) /Rotate 페이지: det bbox는 렌더 PNG와 같은 회전 공간이어야 한다 ──


def test_textlayer_rotated_page_bbox_in_rotated_space(tmp_path):
    """get_text('blocks') 좌표는 비회전 공간, rect·렌더 PNG는 /Rotate 반영 공간 —
    rotation_matrix 사상 없이 정규화하면 레이아웃 뷰 박스가 전치되어 어긋난다."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # 세로 612×792 → /Rotate 90 시 rect는 792×612
    page.insert_text((72, 100), "Rotated page text sample", fontsize=18)
    page.set_rotation(90)
    source = tmp_path / "source.pdf"
    doc.save(str(source))
    doc.close()

    engine = TextLayerEngine(_settings(tmp_path))
    src = fitz.open(str(source))
    try:
        blocks = engine._native_blocks(src, tmp_path / "page_0001.png")
    finally:
        src.close()

    assert len(blocks) == 1
    bbox, text = blocks[0]
    assert "Rotated page text sample" in text
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    # 비회전 상단-좌측 텍스트는 회전(90°) 렌더에서 우측 상단의 세로 박스가 된다.
    # (비회전 좌표를 그대로 792×612 rect로 정규화하면 x1≈91로 좌측에 붙는다 — 회귀 판별점)
    assert x1 > 800 and x2 <= 999, bbox
    assert y1 < 200 and y2 > 300, bbox


def test_sanitize_text_fixed_point():
    """중첩 조작으로 스팬 제거 후 드러나는 '<PAGE>'/'<|…|>'도 정화된다."""
    assert sanitize_text("a<|det|>b<|/det|>c") == "abc"
    assert sanitize_text("<<|x|>PAGE>") == "⟨PAGE⟩"          # 스팬 제거가 마커를 조립하는 경우
    assert sanitize_text("<|a<|b|>c|>") == ""                 # 중첩 스팬은 고정점까지 반복 제거
    assert sanitize_text("본문 <PAGE> 텍스트") == "본문 ⟨PAGE⟩ 텍스트"


def test_sanitize_text_pathological_nesting_is_bounded():
    """순수 중첩은 패스당 최심 스팬 1개만 제거된다 — 상한 없는 고정점 반복은
    O(n²)라 조작 PDF 한 장이 단일 워커를 잠근다. 상한 초과 시 '<' 치환으로
    스팬·마커 문법을 무력화하고 즉시 종료해야 한다 (상한 없으면 이 테스트가 멈춘다)."""
    deep = "머리 " + "<|" * 50_000 + "|>" * 50_000 + " 꼬리"
    out = sanitize_text(deep)
    assert textlayer_mod._SPECIAL_SPAN.search(out) is None
    assert "<PAGE>" not in out
    assert out.startswith("머리 ") and out.endswith(" 꼬리")  # 스팬 밖 내용은 보존


# ── (5) registry 등록 ─────────────────────────────────────────────────────


def test_registry_builds_textlayer_engine():
    eng = build_engine(Settings(engine="textlayer", device="cpu", preload_model=False))
    assert isinstance(eng, TextLayerEngine)
    assert eng.name == "textlayer"
    assert eng.device == "cpu"
    assert eng.loaded is True  # 로드할 모델이 없다 — 즉시 준비


def test_registry_invalid_engine_lists_textlayer():
    with pytest.raises(ValueError, match="textlayer"):
        build_engine(Settings(engine="does-not-exist", device="cpu"))


# ── (2c) 전면 스캔 문서: OCR 불가 시 result.md가 무표식 공백으로 끝나지 않는다 ──


def test_scanned_document_without_ocr_is_marked_not_silently_empty(tmp_path, monkeypatch):
    """모든 페이지가 스캔이고 Tesseract가 없으면 예전에는 result.md가 구분자만 남은
    완전 공백이었다 — status=done인 채 빈 문서를 정상 결과로 오인하게 된다."""
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **k: None)
    with _client(tmp_path) as client:
        r = _upload(client, make_blank_pdf_bytes(pages=3))
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body

        md = client.get(f"/api/jobs/{jid}/markdown").text
        pages = [p.strip() for p in md.split("\n\n---\n\n")]
        assert len(pages) == 3, md                      # 페이지 수 계약 불변
        assert all(page.startswith("> ⚠️") for page in pages), md
        assert all("Tesseract 미설치" in page for page in pages), md
        # 잡 경고 채널도 그대로 (표식은 경고를 대체하지 않고 보완한다)
        assert [w for w in body["warnings"] if "Tesseract" in w], body["warnings"]


def test_tesseract_run_failure_marks_the_page(tmp_path, monkeypatch):
    """실행 실패(언어팩 누락 등) 페이지도 본문에 사유가 남는다 — 정상 페이지와
    섞이면 경고 배열만으로는 어느 페이지가 비었는지 알 수 없다."""
    from app.engine.base import EngineError

    def _boom(executable: str, languages: str, png_bytes: bytes) -> str:
        raise EngineError("Tesseract OCR 실패: Error opening data file kor.traineddata")

    monkeypatch.setattr(textlayer_mod, "find_tesseract", lambda: "/fake/tesseract")
    monkeypatch.setattr(textlayer_mod, "run_tesseract", _boom)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Body text page 1", fontsize=18)
    doc.new_page(width=595, height=842)  # 2쪽: 스캔류 → OCR 경로에서 실패
    pdf = doc.tobytes()
    doc.close()

    with _client(tmp_path) as client:
        r = _upload(client, pdf)
        jid = r.json()["job_id"]
        body = wait_done(client, jid)
        assert body["status"] == "done", body
        pages = [
            p.strip()
            for p in client.get(f"/api/jobs/{jid}/markdown").text.split("\n\n---\n\n")
        ]
        assert "Body text page 1" in pages[0]
        assert "> ⚠️" not in pages[0]                   # 정상 페이지는 표식 없음
        assert "Tesseract 실행 실패" in pages[1]
