"""조판 시험(dry-run)은 기하가 같은 빈 페이지에서 — 결과는 원문 페이지에서 시험한 것과 같다.

Shape.insert_textbox는 시험마다 그 페이지의 폰트 리소스 전체(폼 XObject 재귀 포함)를 다시
훑는다(insert_font → CheckFont → get_page_fonts). 논문 페이지는 폰트·폼이 많아 그 스캔이 빌드
시간의 3분의 1을 넘었다(25쪽 cProfile 33s/87s — P4). build는 빌드마다 trial_pages()를 열고,
_plan_shrink_to_fit은 그 안에서 기하가 같은 빈 페이지로 시험한다. 실제 삽입(커밋)은 그대로
원문 페이지다. 25쪽 논문(평문·CropBox 사본) 빌드는 렌더 픽셀·추출 텍스트·리포트가 모두 같았다.
"""

from __future__ import annotations

import pytest

from app.pipeline.pdf_export import _plan_shrink_to_fit, _resolve_font
from app.pipeline.pdf_export import fitting


_BASE14 = ("helv", "tiro", "cour", "hebo", "tibo", "cobo", "heit", "tiit", "coit")


def _heavy_page(doc, *, rotation=0, cropbox=None, mediabox=None):
    """원문 논문 페이지 대역 — 리소스에 폰트가 여럿이다."""
    page = doc.new_page(width=612, height=792)
    if mediabox is not None:
        page.set_mediabox(mediabox)
    for index, fontname in enumerate(_BASE14):
        page.insert_text((72, 60 + index * 12), f"Source line {index}", fontname=fontname,
                         fontsize=8)
    if cropbox is not None:
        page.set_cropbox(cropbox)
    if rotation:
        page.set_rotation(rotation)
    return page


def _font() -> tuple[str, str | None]:
    fontfile, fontname = _resolve_font("")
    return (fontname, fontfile) if fontfile else ("korea", None)


_TEXT = (
    "트랜스포머는 순환 구조 없이 어텐션만으로 문장의 모든 위치를 한 번에 본다. "
    "그래서 병렬화가 쉽고 긴 의존성을 짧은 경로로 잇는다. " * 4
)


def _plan(page):
    import fitz

    fontname, fontfile = _font()
    return _plan_shrink_to_fit(
        page, fitz.Rect(72, 300, 300, 360), _TEXT, 10, fontname, fontfile,
        max_rect=fitz.Rect(72, 300, 300, 600), lineheights=(None, 1.44),
    )


@pytest.mark.parametrize(
    "geometry",
    [
        {},
        {"cropbox": (15, 30, 597, 702)},
        {"cropbox": (20, 40, 580, 700), "rotation": 90},
        {"mediabox": (10, 20, 622, 812)},
        {"mediabox": (-30, -40, 600, 800), "cropbox": (0, 0, 500, 700), "rotation": 270},
    ],
    ids=["plain", "cropbox", "crop-rot90", "offset-mediabox", "offset-crop-rot270"],
)
def test_trial_page_plans_exactly_like_the_source_page(geometry):
    import fitz

    kwargs = {key: fitz.Rect(value) if key != "rotation" else value
              for key, value in geometry.items()}
    doc = fitz.open()
    page = _heavy_page(doc, **kwargs)
    on_source = _plan(page)
    with fitting.trial_pages():
        trial = fitting._trial_page(page)
        assert trial is not page                     # 기하를 복제한 빈 페이지가 쓰인다
        on_trial = _plan(page)
    assert on_source is not None
    assert on_trial == on_source
    doc.close()


def test_shrink_trials_never_rescan_the_source_page_fonts(monkeypatch):
    """빌드 안의 시험은 원문 페이지에 폰트를 넣거나 그 리소스를 훑지 않는다(수정 전에는 시험마다)."""
    import fitz

    doc = fitz.open()
    page = _heavy_page(doc)
    source_xref = page.xref
    calls: list[int] = []
    original = fitz.Page.insert_font

    def counting_insert_font(self, *args, **kwargs):
        calls.append(self.xref if self.parent is doc else -1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "insert_font", counting_insert_font)
    with fitting.trial_pages():
        assert _plan(page) is not None
    assert calls, "시험 조판이 돌지 않았다"
    assert source_xref not in calls, calls
    assert set(calls) == {-1}                        # 전부 빈 시험 문서의 페이지
    doc.close()


def test_outside_a_build_the_source_page_is_used_and_trial_docs_are_closed():
    import fitz

    doc = fitz.open()
    page = _heavy_page(doc)
    assert fitting._trial_page(page) is page        # 빌드 밖(직접 호출·테스트)은 예전 그대로
    with fitting.trial_pages():
        trial = fitting._trial_page(page)
        assert fitting._trial_page(page) is trial   # 같은 기하는 한 장을 재사용
        trial_doc = trial.parent
    assert trial_doc.is_closed
    doc.close()


def test_mixed_page_sizes_keep_only_a_few_trial_docs_open():
    """쪽 크기가 제각각인 문서(폰 스캔 자동 자르기·여러 출처를 합친 PDF)는 기하마다 시험 문서를
    새로 열어 빌드가 끝날 때까지 닫지 않았다 — 100쪽에 시험 문서 82개, 빌드 최대 메모리가
    379MB → 718MB(300쪽 540MB → 1.9GB)로 늘었고 속도 이득도 없었다(delta-pdf-translate-3).
    최근 기하 몇 개만 남기고 밀려난 시험 문서는 바로 닫는다."""
    import fitz

    doc = fitz.open()
    for index in range(20):
        _heavy_page(doc, mediabox=fitz.Rect(0, 0, 600 - index, 780 - index))
    trial_docs = []
    with fitting.trial_pages():
        for page in doc:
            trial = fitting._trial_page(page)
            assert trial is not page
            trial_docs.append(trial.parent)
            open_docs = sum(1 for trial_doc in trial_docs if not trial_doc.is_closed)
            assert open_docs <= fitting._TRIAL_CACHE_MAX, open_docs
        last = doc[len(doc) - 1]
        assert _plan(last) is not None               # 밀어낸 뒤에도 시험은 정상이다
        # 다시 돌아온 기하(최근 것)는 그대로 재사용한다
        assert fitting._trial_page(last).parent is trial_docs[-1]
    assert all(trial_doc.is_closed for trial_doc in trial_docs)
    doc.close()
