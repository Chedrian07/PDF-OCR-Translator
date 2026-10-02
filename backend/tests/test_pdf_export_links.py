"""번역으로 바꾼 블록 위의 링크도 번역 PDF에 남는다 (실측: 25쪽 논문의 링크 167개 중 17개만 남음).

MuPDF 리댁션은 지우는 영역과 겹친 링크 annotation을 페이지 /Annots에서 뺀다. 내보내기는 같은
annotation 객체를 다시 달고, 링크 아래 원문(인용 번호·URL — 번역 때 마스킹돼 그대로 복원된다)이
번역문에 다시 나오면 클릭 영역을 그 글자로 옮긴다. 모호하면 원래 자리에 둔다.
"""

from __future__ import annotations

import json

import fitz
import pytest

from app.pipeline.pdf_export import build_translated_pdf

from tests.test_pdf_export import _unit_job

SRC_TEXT = "Prior work [12] studies this problem. Code is at https://example.org/turbo for all."
KO_TEXT = "선행 연구 [12]는 이 문제를 다룬다. 코드는 https://example.org/turbo 에 있다."


def _translate_to(job_dir, text: str) -> None:
    path = job_dir / "layout.ko.json"
    pages = json.loads(path.read_text(encoding="utf-8"))
    pages[0]["blocks"][0]["content"] = text
    path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")


def _add_links(job_dir, links: list[dict]) -> list:
    """source.pdf에 링크를 단다 — 'anchor' 키는 원문에서 찾아 그 자리를 'from'으로 쓴다."""
    src = job_dir / "source.pdf"
    staged = job_dir / "linked.pdf"
    doc = fitz.open(src)
    page = doc[0]
    rects = []
    for link in links:
        spec = dict(link)
        anchor = spec.pop("anchor")
        hits = page.search_for(anchor)
        assert len(hits) == 1, (anchor, hits)
        # 회전 쪽에서 insert_link의 'from'은 비회전 좌표(search_for와 같은 공간)다
        spec["from"] = hits[0]
        page.insert_link(spec)
        rects.append(hits[0])
    doc.save(staged)
    doc.close()
    staged.replace(src)
    return rects


def _exported_links(path) -> list[tuple[dict, str]]:
    """번역 PDF 첫 쪽의 (링크, 그 클릭 영역 아래 글자)."""
    doc = fitz.open(path)
    try:
        page = doc[0]
        out = []
        for link in page.get_links():
            # get_links의 'from'은 화면(회전) 좌표 — 글자 추출은 비회전 좌표다
            area = fitz.Rect(link["from"]) * page.derotation_matrix
            out.append((link, " ".join(page.get_textbox(area).split())))
        return out
    finally:
        doc.close()


@pytest.mark.parametrize("rotate", [0, 90])
def test_links_over_replaced_text_stay_and_move_onto_their_text(tmp_path, rotate):
    job_dir = _unit_job(tmp_path, src_text=SRC_TEXT, rotate=rotate)
    _translate_to(job_dir, KO_TEXT)
    _add_links(job_dir, [
        {"anchor": "[12]", "kind": fitz.LINK_GOTO, "page": 0, "to": fitz.Point(0, 0)},
        {"anchor": "https://example.org/turbo", "kind": fitz.LINK_URI,
         "uri": "https://example.org/turbo"},
    ])

    result = build_translated_pdf(job_dir, "ko")
    assert result.replaced == 1

    links = _exported_links(result.path)
    by_kind = {link["kind"]: text for link, text in links}
    assert sorted(by_kind) == [fitz.LINK_GOTO, fitz.LINK_URI], links   # 둘 다 남는다
    assert "12" in by_kind[fitz.LINK_GOTO]                     # 번역문의 [12] 위로 옮겨졌다
    assert "example.org" in by_kind[fitz.LINK_URI]
    uri = next(link for link, _text in links if link["kind"] == fitz.LINK_URI)
    assert uri["uri"] == "https://example.org/turbo"           # 목적지는 원본 그대로


def test_ambiguous_link_text_keeps_the_link_where_it_was(tmp_path):
    """번역문에 같은 번호가 더 많이 나오면(번역된 '정리 3') 어느 것인지 모른다 — 원래 자리."""
    src = "As shown in Lemma 3 the bound holds for every input vector."
    job_dir = _unit_job(tmp_path, src_text=src)
    _translate_to(job_dir, "보조정리 3 에서 보였듯이 상한은 3 개의 모든 입력 벡터에 대해 성립한다.")
    (original,) = _add_links(job_dir, [
        {"anchor": "3", "kind": fitz.LINK_GOTO, "page": 0, "to": fitz.Point(0, 0)},
    ])

    result = build_translated_pdf(job_dir, "ko")

    links = _exported_links(result.path)
    assert len(links) == 1
    assert tuple(fitz.Rect(links[0][0]["from"])) == pytest.approx(tuple(original), abs=0.05)


def test_restored_link_still_loses_a_dangerous_action(tmp_path):
    """되단 링크도 능동 콘텐츠 정리를 받는다 — 번역문 위 Launch 링크가 실행 동작을 되찾지 않는다."""
    job_dir = _unit_job(tmp_path, src_text=SRC_TEXT)
    _translate_to(job_dir, KO_TEXT)
    _add_links(job_dir, [{"anchor": "[12]", "kind": fitz.LINK_LAUNCH, "file": "calc.exe"}])

    result = build_translated_pdf(job_dir, "ko")

    doc = fitz.open(result.path)
    try:
        exported = "\n".join(doc.xref_object(x) for x in range(1, doc.xref_length()))
        kinds = [link["kind"] for link in doc[0].get_links()]
    finally:
        doc.close()
    assert "/Launch" not in exported and "calc.exe" not in exported
    assert fitz.LINK_LAUNCH not in kinds


def test_links_away_from_replaced_text_are_left_alone(tmp_path):
    job_dir = _unit_job(tmp_path, src_text=SRC_TEXT)
    _translate_to(job_dir, KO_TEXT)
    doc = fitz.open(job_dir / "source.pdf")
    far = fitz.Rect(60, 700, 200, 720)                         # 번역 블록과 겹치지 않는 자리
    doc[0].insert_link({"kind": fitz.LINK_URI, "from": far, "uri": "https://example.org/far"})
    doc.saveIncr()
    doc.close()

    result = build_translated_pdf(job_dir, "ko")

    (link, _text), = _exported_links(result.path)
    assert link["uri"] == "https://example.org/far"
    assert tuple(fitz.Rect(link["from"])) == pytest.approx(tuple(far), abs=0.05)
