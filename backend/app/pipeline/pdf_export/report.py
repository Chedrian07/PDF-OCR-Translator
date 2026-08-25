"""내보내기 결과 리포트와 사용자 대면 오류."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 캐시된 export PDF가 이전 조판 규칙으로 생성됐는지 판별하는 공개 포맷 버전.
# 조판 결과가 달라지는 변경에서는 반드시 올려 기존 잡도 다음 요청 때 재생성한다.
PDF_EXPORT_FORMAT_VERSION = 6


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
    specialist_kept: dict[str, int] = field(default_factory=dict)
    kept_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def keep(self, reason: str, count: int = 1) -> None:
        """보존 블록 수와 사유를 함께 기록한다.

        경고는 사용자에게 보여줄 만한 이상 징후만 남기므로 `kept`의 일부만
        설명한다. 다음 미번역 신고를 코드 없이 진단하려면 보존된 *모든* 블록의
        사유가 필요하다. `kept_reasons`의 합은 항상 `kept`와 같다.
        """
        self.kept += count
        self.kept_reasons[reason] = self.kept_reasons.get(reason, 0) + count

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
            "specialist_kept": dict(sorted(self.specialist_kept.items())),
            # 교체 대상 타입이 아닌 블록(image/equation/algorithm 등)은 애초에
            # kept로 세지 않고 specialist_kept로만 집계한다.
            "kept_reasons": dict(sorted(self.kept_reasons.items())),
            "warning_count": len(self.warnings),
            "warnings": self.warnings[:50],
        }
