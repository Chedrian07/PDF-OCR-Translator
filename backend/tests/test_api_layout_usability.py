"""좌표 layout '사용 가능' 판정(artifacts.has_usable_layout)의 API·번역 배선.

figure_only 엔진(OvisOCR2)의 옛 잡은 image 블록만 든 layout.json(과 번역이 만든
layout.ko.json)을 남겼다. 라우트가 파일 존재만 보면 그런 잡이 has_layout으로 보여
document.html이 OCR 텍스트 없는 facsimile로, /pdf?lang=ko가 번역 안 된 원문으로 나가고
리더가 빈 정렬·개요를 약속했다(감사 sidecar-1, Phase 1 c-pipeline). 새 figure_only 잡은
layout.json 자체가 없다 — 두 경우가 같은 응답을 내야 한다.
"""

import json

import pytest

from conftest import wait_done

IMAGE_ONLY = [{
    "page": 1, "width": 595, "height": 842,
    "blocks": [{"type": "image", "bbox": [100, 100, 900, 600], "image": "p0001_0.jpg"}],
}]


def _done_job(client, sample_pdf) -> tuple[str, object]:
    jid = client.post(
        "/api/jobs", files={"file": ("doc.pdf", sample_pdf, "application/pdf")},
    ).json()["job_id"]
    assert wait_done(client, jid)["status"] == "done"
    return jid, client.app.state.store.get(jid)


def _make_layout_unusable(job, kind: str) -> None:
    """kind=image_only: 옛 Ovis 잡(image 블록뿐인 layout·번역 layout),
    kind=absent: 새 figure_only 잡(layout 파일 없음). 둘 다 한국어 번역본은 있다."""
    (job.dir / "result.ko.md").write_text("# 번역된 제목\n\n번역 본문", encoding="utf-8")
    for name in ("layout.json", "layout.ko.json"):
        path = job.dir / name
        if kind == "image_only":
            path.write_text(json.dumps(IMAGE_ONLY), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("kind", ["image_only", "absent"])
def test_unusable_layout_jobs_answer_like_layoutless_jobs(client, sample_pdf, monkeypatch, kind):
    import app.api as api_mod

    jid, job = _done_job(client, sample_pdf)
    _make_layout_unusable(job, kind)
    base = f"/api/jobs/{jid}"

    assert client.get(base).json()["result"]["has_layout"] is False

    for path in ("/layout", "/layout?lang=ko", "/alignment?page=1", "/alignment?page=1&lang=ko",
                 "/outline", "/outline?lang=ko", "/viewer/pages", "/viewer/pages?lang=ko"):
        assert client.get(base + path).status_code == 404, path

    caps = client.get(f"{base}/viewer-manifest?lang=ko").json()["capabilities"]
    assert caps["alignment"] is False and caps["outline"] is False
    assert caps["translated_page_image"] is False
    assert caps["source_page_image"] is True

    # 리더 페이지는 원본 렌더 그대로 — 번역 PDF를 만들 좌표가 없다
    def _no_facsimile(*args, **kwargs):
        raise AssertionError("좌표 텍스트가 없는 잡에 facsimile을 만들려 했다")

    monkeypatch.setattr(api_mod, "_ensure_facsimile_pages", _no_facsimile)
    monkeypatch.setattr(api_mod, "_try_facsimile_pages", _no_facsimile)
    for path in ("/page/1", "/page/1?lang=ko"):
        page = client.get(base + path)
        assert page.status_code == 200, path
        assert page.headers["content-type"] == "image/png"
        assert page.content == (job.dir / "pages" / "page_0001.png").read_bytes()

    # HTML 내보내기는 OCR 텍스트(semantic)로 폴백한다 — 원문 래스터만 든 facsimile이 아니다
    doc = client.get(f"{base}/document.html")
    assert doc.status_code == 200 and "페이지 1" in doc.text
    doc_ko = client.get(f"{base}/document.html?lang=ko")
    assert doc_ko.status_code == 200 and "번역 본문" in doc_ko.text

    pdf = client.get(f"{base}/pdf?lang=ko")
    assert pdf.status_code == 409
    assert "좌표 레이아웃이 없어" in pdf.json()["detail"]


def test_text_layouts_still_enable_every_coordinate_feature(client, sample_pdf):
    """대조군 — 텍스트 블록이 있는 layout은 그대로 좌표 기능을 켠다."""
    jid, job = _done_job(client, sample_pdf)
    base = f"/api/jobs/{jid}"
    assert client.get(base).json()["result"]["has_layout"] is True
    caps = client.get(f"{base}/viewer-manifest").json()["capabilities"]
    assert caps["alignment"] is True and caps["outline"] is True
    for path in ("/layout", "/alignment?page=1", "/outline", "/viewer/pages"):
        assert client.get(base + path).status_code == 200, path


def test_one_text_block_anywhere_keeps_the_layout_usable(client, sample_pdf):
    """판정은 문서 단위다 — 그림 페이지가 섞여 있어도 텍스트 블록이 하나라도 있으면 쓴다."""
    jid, job = _done_job(client, sample_pdf)
    pages = json.loads((job.dir / "layout.json").read_text(encoding="utf-8"))
    for page in pages[1:]:
        page["blocks"] = [b for b in page["blocks"] if b.get("type") == "image"]
    (job.dir / "layout.json").write_text(json.dumps(pages), encoding="utf-8")
    assert client.get(f"/api/jobs/{jid}").json()["result"]["has_layout"] is True
    assert client.get(f"/api/jobs/{jid}/layout").status_code == 200


def test_unusable_layouts_are_not_prewarmed(client, sample_pdf, monkeypatch):
    """/pdf가 409인 잡은 번역 완료 때 PDF를 예열하지 않는다 — 빌더가 매번 실패해
    'PDF 예열 실패' 경고와 traceback만 남겼다."""
    import app.api as api_mod
    from app.pipeline import derived

    started = []
    monkeypatch.setattr(
        derived, "warm_translated_pdf_async", lambda *a, **kw: started.append(a) or True,
    )
    jid, job = _done_job(client, sample_pdf)
    st = client.app.state
    _make_layout_unusable(job, "image_only")
    assert api_mod._warm_export_pdf(st, job, "ko") is False
    assert started == []

    # 텍스트 layout이면 예열한다(대조군)
    jid2, job2 = _done_job(client, sample_pdf)
    (job2.dir / "layout.ko.json").write_bytes((job2.dir / "layout.json").read_bytes())
    assert api_mod._warm_export_pdf(st, job2, "ko") is True
    assert len(started) == 1
