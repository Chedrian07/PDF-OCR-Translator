"""OCR 엔진 인터페이스. 계약: docs/ARCHITECTURE.md §4

엔진 출력 형식(모델 업스트림 save_results 규약과 동일):
- run_multi: `<PAGE>` 마커로 페이지가 구분된 처리 완료 마크다운을 반환.
  figure는 out_dir/images/page_{청크내idx}_{k}.jpg 로 저장되고 마크다운에는
  ![](images/page_{i}_{k}.jpg) 참조가 들어감. 페이지별 레이아웃 오버레이는
  out_dir/result_with_boxes_{i}.jpg
- run_single: 단일 페이지 마크다운 반환. figure는 out_dir/images/{k}.jpg,
  오버레이는 out_dir/result_with_boxes.jpg
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class EngineError(RuntimeError):
    """엔진 실행 실패 (사용자에게 노출 가능한 메시지).

    transient=True이면 "일시적"(재시도/대기하면 해소될 수 있음) 조건이다 — 프리로드가
    이를 하드 실패로 로깅하지 않고, 워커의 준비 대기가 계속 기다린다.

    retry_same_page=False이면 runner가 같은 청크·페이지를 **즉시 다시 요청하지 않고**
    곧바로 페이지 격리(텍스트 레이어 폴백 → 플레이스홀더)로 넘긴다. 같은 입력을 곧장
    다시 보내 봐야 소용없는 실패용이다 — 예: sidecar 읽기 타임아웃(SidecarTimeoutError)은
    provider가 끊긴 요청의 추론을 끝까지 하므로 재요청이 그 뒤에 줄을 서 다시 타임아웃이
    난다. 기본값 True(1회 재시도). runner는 EngineError가 아닌 예외(벤더 RuntimeError·
    OOM 등)도 받으므로 `getattr(error, "retry_same_page", True)`로 읽는다."""

    transient: bool = False
    retry_same_page: bool = True


class RepetitiveOutputError(EngineError):
    """모델 생성이 반복에 빠지거나 페이지별 출력 상한을 넘어 조기 중단됨."""


class OutputLimitError(RepetitiveOutputError):
    """생성이 MAX_LENGTH 같은 총 길이 상한에 닿아 출력이 잘림.

    반복이 아니어도 잘린 출력은 마지막 페이지(들)의 내용이 조용히 빠진 상태다.
    runner는 다른 불안전 생성(RepetitiveOutputError)과 똑같이 취급해 그 출력을
    채택하지 않고 페이지 단위로 복구해야 한다 — 그래서 하위 클래스로 둔다.

    partial_output: run_multi 형식(`<PAGE>` 구분, 크롭·raw_pages.json 등 산출물은
    out_dir에 있음)의 잘린 출력. 엔진이 실어 주면 runner가 **끝까지 생성된 앞 페이지**
    (마지막 세그먼트는 잘린 페이지)를 병합하고 잘린 페이지부터만 다시 처리한다. 모델이
    페이지 마커를 내는 엔진(supports_multi_page)이면 세그먼트 수만 믿지 않고 원본 텍스트
    레이어와 대조해 남길 물리 페이지와 배치를 정한다(IncrementalMerger.completed_prefix —
    잘리기 전에 쪼개거나 건너뛴 페이지가 있으면 그 앞까지만 남긴다).
    None이면(run_single·모름) 청크 전체를 페이지별로 다시 처리한다. 출력 문자열은
    문서 내용이므로 메시지(로그·잡 경고에 남는다)에는 넣지 않는다.

    limit_label: 닿은 상한의 이름 — runner가 잡 경고·참고·라이브 문구에 '<이름> 도달'로
    쓴다. 기본 'MAX_LENGTH'(in-process 엔진의 총 길이 상한). 다른 상한에서 잘리는 엔진은
    하위 클래스에서 바꾼다 — 예: sidecar는 페이지당 출력 토큰 상한이라 MAX_LENGTH를
    가리키면 효과 없는 설정으로 운영자를 이끈다.
    """

    limit_label: str = "MAX_LENGTH"

    def __init__(self, message: str, partial_output: str | None = None) -> None:
        super().__init__(message)
        self.partial_output = partial_output


class JobCanceled(Exception):
    """사용자 취소로 중단됨."""


class StreamSink(Protocol):
    def on_text(self, text: str) -> None:
        """모델이 생성한 텍스트 델타 (SSE token 이벤트로 전달됨)."""
        ...


class NullSink:
    def on_text(self, text: str) -> None:  # pragma: no cover - trivial
        pass


@dataclass(frozen=True)
class EngineCapabilities:
    """엔진 메타데이터·능력 선언 — runner의 청크 크기 결정과 health/Job 메타에 쓰인다.

    기본값은 기존 Unlimited-OCR 의미를 보존한다(하위 호환): 멀티페이지 문맥 지원,
    토큰 단위 스트리밍, 완전한 layout. 페이지 단위 sidecar 엔진은 이를 오버라이드한다.

    - preferred_chunk_size: None이면 settings.pages_per_chunk 사용.
    - stream_granularity: "token"(생성 토큰 델타) | "page"(페이지 완료 시 일괄).
    - layout_capability: "full"(텍스트+figure bbox) | "figure_only" | "none".
    """

    model_id: str = ""
    model_revision: str = ""
    provider: str = "in-process"
    supports_multi_page: bool = True
    preferred_chunk_size: int | None = None
    stream_granularity: str = "token"
    layout_capability: str = "full"
    figure_capability: bool = True


class OCREngine(abc.ABC):
    name: str = "base"
    device: str = "cpu"
    dtype_name: str = "float32"
    # 같은 페이지를 다시 돌려도 결과가 같은 결정적 엔진인가(텍스트 레이어·Tesseract).
    # True면 runner의 충실도 게이트가 단독 재처리를 하지 않는다 — 같은 호출을 반복해
    # 시간만 쓰고 거짓 '충실도 미달' 경고를 남긴다. 생성 모델 엔진은 False(기본).
    deterministic_rerun: bool = False

    @property
    @abc.abstractmethod
    def loaded(self) -> bool: ...

    def capabilities(self) -> EngineCapabilities:
        """기본값 = 기존 Unlimited 의미 — fake/기존 테스트가 깨지지 않는다."""
        return EngineCapabilities()

    def provider_health(self) -> dict | None:
        """외부 provider(sidecar) 상태 — in-process 엔진은 None."""
        return None

    def drain_warnings(self) -> list[str]:
        """직전 실행에서 쌓인 사용자 노출용 경고를 꺼내고 비운다 (기본: 없음).

        runner가 청크마다 호출해 잡 warnings에 합친다 — 정화/절단으로 내용이
        빠졌는데 잡이 조용히 'done'이 되는 것을 막는다. 여기에는 **실제 품질 저하**만
        넣는다(잡 quality.state가 이것으로 'degraded'가 된다)."""
        return []

    def drain_notices(self) -> list[str]:
        """직전 실행에서 쌓인 정보성 메모를 꺼내고 비운다 (기본: 없음).

        내용·품질에는 문제가 없는 처리 경위(예: 일시 중단 뒤 재시도로 정상 처리됨)용이다.
        runner가 drain_warnings와 같은 시점에 잡 notices로 합친다 — warnings에 섞으면
        정상 결과까지 'degraded'로 보인다."""
        return []

    @abc.abstractmethod
    def load(self) -> None:
        """모델/리소스 로드 (멱등·스레드 세이프 — 동시 호출 시 한쪽만 로드, 나머지는 완료 대기).

        프리로드 스레드(main)와 워커 스레드(jobs)가 동시에 호출할 수 있다.
        """

    def wait_until_ready(
        self,
        cancel: "threading.Event",
        on_wait: "Callable[[str], None] | None" = None,
    ) -> None:
        """잡 처리 가능 상태가 될 때까지 블로킹 확보 (워커가 잡 시작 전에 호출).

        기본 구현은 load() 1회 — in-process 엔진은 load()가 모델 적재까지 블로킹하므로
        반환 시점에 곧 사용 가능하다. sidecar 엔진은 이를 오버라이드해 모델이 준비될
        때까지 취소 가능하게 폴링 대기한다(최초 기동의 다운로드·컴파일을 잡 실패로
        만들지 않기 위해). on_wait(note)는 대기 중 진행 문구를 전달하는 콜백."""
        self.load()

    @abc.abstractmethod
    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str: ...

    @abc.abstractmethod
    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str: ...

    def gpu_name(self) -> str | None:
        return None
