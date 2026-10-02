"""내보내기 결과 리포트와 사용자 대면 오류."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 경고 문구의 페이지 접두사("p12: …") — warning_pages 집계용
_WARNING_PAGE_RE = re.compile(r"p(\d+):")
# report()에 싣는 경고 표본 상한 — 페이지 단위 집계(kept_pages·warning_pages)는 자르지 않는다
_WARNINGS_SAMPLE = 50

# 캐시된 export PDF가 이전 조판 규칙으로 생성됐는지 판별하는 공개 포맷 버전.
# 조판 결과가 달라지는 변경에서는 반드시 올려 기존 잡도 다음 요청 때 재생성한다.
# 7: 폰트 서브셋(20.8MB → 0.94MB) + `\sim` tofu를 ASCII 물결표로 낮춤. 크기가
#    바뀌는 것만으로도 올릴 값어치가 있다 — 올리지 않으면 이미 캐시된 잡은
#    입력이 그대로라 계속 20MB짜리를 내려보낸다.
# 8: 페이지 등록 게이트(대조되지 않는 페이지는 수정하지 않음) + 계획/리댁션
#    장애물 모델 일치(겹침 제거 · "공간 부족" 오판 제거). 조판 결과가 달라진다.
# 9: 피할 수 없는 전폭 내부 띠를 장애물에서 제외 + 윗변이 가려진 상자는 장애물
#    아래에서 시작하는 후보를 추가. 자리가 있는데도 버려지던 번역이 들어간다
#    (실측 no_fit 66 → 25). 조판 결과가 달라진다.
# 10: 스캔(래스터) 원문 픽셀을 바탕색으로 덮고 번역 삽입, baseline 띠 리댁션(이웃
#     보존 줄 유지), 평문화 보정(부등호 사이 문장·글머리표·코드·LaTeX 구조),
#     회전 페이지 한 줄 경로·폰트 백필 좌표계, flow의 원문 장애물·읽기 순서 유지,
#     부분 리스팅 잔여 줄 장애물, 공백 ToUnicode 복원. 조판 결과가 달라진다.
# 11: 스캔 표는 픽셀 격자로 바뀐 셀의 글자만 덮고(경계를 확정 못 하면 보존), 회전 페이지의
#     흐름 배치를 화면 공간에서 계획, 스캔 덮개가 겹친 그림·보존 블록을 피함, 띠·여백 스캔과
#     '이미지 아래 텍스트' 스캔 인식, 좁은 다줄 스캔 단 번역, 퇴화 메트릭 폰트 리댁션.
#     조판 결과가 달라진다(감사 pdf-1~5·8·9).
# 12: 원문 그대로 남은 목록 블록을 다시 조판하지 않음, 교체한 문단의 인라인 수식 선(분수선·
#     근호 윗선)을 함께 지우고 계획의 장애물에서 뺌. 조판 결과가 달라진다.
# 13: `\langle`·`\rangle`을 ⟨⟩(폰트에 없으면 〈〉)로 평문화 — 예전에는 'langle'이 찍혔다.
# 14: TeX 확장 괄호 조각(제어 코드 글리프·세로로 쌓인 span)도 원문으로 지움. 조판 결과가 달라진다.
# 15: 기호 명령 바로 뒤의 감싸개·분수가 명령 이름에 붙지 않음('langley'·'cdotx' → '⟨y'·'·x').
# 16: 줄 위·아래로 튀어나와 OCR bbox와 조금만 겹치는 기호(근호 등)를 맞닿은 블록 하나에 붙여
#     그 블록을 교체할 때 함께 지움 — 번역문 위에 '√'만 남지 않는다. 조판 결과가 달라진다.
# 17: 원본의 능동 콘텐츠(문서 열기 스크립트·/AA·문서 JavaScript·위험한 링크 동작·첨부 파일)를
#     번역 PDF에 싣지 않음(security-3). 캐시된 PDF가 그것을 그대로 내보내지 않게 올린다.
#     같은 버전에서: 그림 영역 안의 두께 0 선(범례 등)을 겹친 캡션이 소유해 지우지 않음
#     (delta-pdf-translate-6), 회전 쪽(/Rotate 90·270)의 분수선·근호 윗선 소유와 튀어나온
#     기호(√) 부착을 화면 좌표로 판정(delta-pdf-translate-2). 조판 결과가 달라진다.
# 18: 리댁션이 걷어낸 링크 annotation(인용·절·URL)을 다시 달고 번역문의 같은 글자 위로 옮김,
#     글자 크기 없는 스캔 블록의 크기를 번역문이 아니라 원문 글자로 추정(쪽 본문 중앙값 상한).
#     캐시된 PDF에 링크가 빠져 있으므로 올린다. 조판 결과가 달라진다.
PDF_EXPORT_FORMAT_VERSION = 18


class PdfExportError(RuntimeError):
    """내보내기 불가(입력 파일 없음/손상). 사용자에게 그대로 보여줄 한국어 메시지."""


@dataclass
class PdfExportResult:
    path: Path
    replaced: int = 0
    kept: int = 0
    relocated: int = 0
    table_cells_replaced: int = 0
    listing_lines_replaced: int = 0
    # 원문이 텍스트가 아니라 래스터 픽셀(스캔 페이지·표 이미지)이라, 번역을 넣기 전에
    # 그 영역을 바탕색으로 덮어 지운 교체 블록 수. 0이 아니면 '교체'가 픽셀 덮기를
    # 동반했다는 뜻이다(덮지 못한 경우는 경고로 남는다).
    raster_blocks_erased: int = 0
    specialist_kept: dict[str, int] = field(default_factory=dict)
    kept_reasons: dict[str, int] = field(default_factory=dict)
    # 페이지(1-base)별 보존 블록 수 — 합은 페이지를 알고 기록한 kept와 같다. 경고는 50건
    # 표본만 리포트에 실리므로, '이 페이지의 번역이 왜 PDF에 없나'는 이 집계로 답한다.
    kept_pages: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def keep(self, reason: str, count: int = 1, page: int | None = None) -> None:
        """보존 블록 수와 사유를 함께 기록한다(page를 알면 페이지별로도).

        경고는 사용자에게 보여줄 만한 이상 징후만 남기므로 `kept`의 일부만
        설명한다. 다음 미번역 신고를 코드 없이 진단하려면 보존된 *모든* 블록의
        사유가 필요하다. `kept_reasons`의 합은 항상 `kept`와 같다.
        """
        self.kept += count
        self.kept_reasons[reason] = self.kept_reasons.get(reason, 0) + count
        if page is not None and count:
            self.kept_pages[page] = self.kept_pages.get(page, 0) + count

    def merge(self, other: "PdfExportResult", page: int | None = None) -> None:
        """다른 집계를 흡수한다 — 계획을 여러 패스 시도할 때 마지막 것만 반영.

        page를 주면 other의 보존 블록 전부를 그 페이지 몫으로 센다(페이지 하나의 계획)."""
        self.replaced += other.replaced
        self.kept += other.kept
        if page is not None and other.kept:
            self.kept_pages[page] = self.kept_pages.get(page, 0) + other.kept
        for key, count in other.kept_pages.items():
            if page is None:
                self.kept_pages[key] = self.kept_pages.get(key, 0) + count
        self.relocated += other.relocated
        self.table_cells_replaced += other.table_cells_replaced
        self.listing_lines_replaced += other.listing_lines_replaced
        self.raster_blocks_erased += other.raster_blocks_erased
        for key, count in other.specialist_kept.items():
            self.specialist_kept[key] = self.specialist_kept.get(key, 0) + count
        for key, count in other.kept_reasons.items():
            self.kept_reasons[key] = self.kept_reasons.get(key, 0) + count
        self.warnings.extend(other.warnings)

    def report(self) -> dict:
        """경로·본문 없이 UI에 안전하게 노출할 ASCII/숫자 중심 생성 리포트."""
        return {
            "format_version": PDF_EXPORT_FORMAT_VERSION,
            "replaced": self.replaced,
            "kept": self.kept,
            "relocated": self.relocated,
            "table_cells_replaced": self.table_cells_replaced,
            # 리스팅·평탄화 표에서 원문 줄·열 좌표에 그대로 조판한 줄 수.
            "listing_lines_replaced": self.listing_lines_replaced,
            # 스캔 픽셀을 바탕색으로 덮은 뒤 번역을 넣은 블록 수.
            "raster_blocks_erased": self.raster_blocks_erased,
            "specialist_kept": dict(sorted(self.specialist_kept.items())),
            # 교체 대상 타입이 아닌 블록(image/equation/algorithm 등)은 애초에
            # kept로 세지 않고 specialist_kept로만 집계한다.
            "kept_reasons": dict(sorted(self.kept_reasons.items())),
            # [[페이지, 보존 블록 수], …] — 페이지 순, 자르지 않는다(dict가 아닌 이유: 사유별
            # 집계 dict와 섞이지 않게). warning_pages는 경고가 하나라도 있는 페이지 전부.
            "kept_pages": [[page, count] for page, count in sorted(self.kept_pages.items())],
            "warning_pages": sorted({
                int(match.group(1))
                for warning in self.warnings
                if (match := _WARNING_PAGE_RE.match(str(warning)))
            }),
            "warning_count": len(self.warnings),
            "warnings": self.warnings[:_WARNINGS_SAMPLE],
        }
