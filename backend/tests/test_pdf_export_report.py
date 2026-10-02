"""내보내기 리포트의 페이지 단위 집계 — 경고 표본(50건) 밖의 페이지도 사유가 남는다.

예전 리포트는 경고를 앞 50건만 실어, 25쪽 실행(경고 78건)에서 p13 이후 페이지는 원문을 남긴
사유가 있어도 리포트로 확인할 수 없었고 verify_e2e가 그 표본을 전부로 읽었다(P4 macOS 스위트).
"""

from app.pipeline.pdf_export.report import PdfExportResult


def test_report_keeps_untruncated_page_summaries_beside_the_warning_sample(tmp_path):
    total = PdfExportResult(path=tmp_path / "export.ko.pdf")
    for page in range(1, 26):
        trial = PdfExportResult(path=total.path)        # 페이지 하나의 계획 집계
        trial.keep("no_fit", 2)
        trial.warnings += [f"p{page}: 블록 {i} 교체 생략(공간 부족) — 원문 보존" for i in range(3)]
        total.merge(trial, page=page)
    total.keep("page_source_mismatch", 4, page=30)
    total.warnings.append("p30: 레이아웃이 29쪽 내용과 일치해 이 페이지는 건너뜀")
    total.warnings.append("한글 폰트 파일을 찾지 못해 PyMuPDF 내장 CJK로 대체합니다")

    report = total.report()
    assert report["warning_count"] == 77 and len(report["warnings"]) == 50   # 표본은 그대로
    assert report["warning_pages"] == [*range(1, 26), 30]                    # 페이지는 전부
    assert report["kept_pages"] == [*([page, 2] for page in range(1, 26)), [30, 4]]
    assert sum(count for _page, count in report["kept_pages"]) == report["kept"] == 54
    assert sum(report["kept_reasons"].values()) == report["kept"]


def test_merging_without_a_page_carries_the_page_counts_over(tmp_path):
    inner = PdfExportResult(path=tmp_path / "x.pdf")
    inner.keep("no_fit", 3, page=7)
    inner.keep("unchanged")                      # 페이지를 모르는 보존은 kept에만
    outer = PdfExportResult(path=inner.path)
    outer.merge(inner)
    assert outer.kept == 4
    assert outer.report()["kept_pages"] == [[7, 3]]
