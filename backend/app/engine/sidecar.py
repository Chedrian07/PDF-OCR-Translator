"""sidecar 기반 OCR 엔진 (ovisocr2 · paddleocr_vl) — GPU는 sidecar 컨테이너만 사용.

메인 backend는 페이지 이미지를 HTTP로 보내고 normalized 결과를 받아
공통 materializer로 기존 청크 산출물 규약을 재현한다. 스트리밍은 페이지
단위(stream_granularity="page")다 — 가짜 토큰 스트리밍을 만들지 않고,
페이지가 완료될 때 `<PAGE>` 마커와 함께 전체 텍스트를 한 번에 발행한다
(기존 프론트의 `<PAGE>` 진행 계약과 그대로 호환).

단일 GPU 원칙: 이 엔진은 다른 GPU 모델로의 자동 fallback을 하지 않는다.
provider 실패는 명확한 오류로 표면화되고 전환은 사용자가 profile로 결정한다.

출력 토큰 상한에서 끊긴 페이지(`page.truncated`, Ovis finish_reason=length)는 원본 PDF
텍스트 레이어와 대조해, 잘림으로 잃은 것이 크면 `OutputLimitError`로 넘겨 runner의
기존 복구(텍스트 레이어 폴백)를 태우고, 아니면(스캔 문서·대조 불가·충실도 충분) 잘린
출력을 경고와 함께 그대로 쓴다 — 텍스트 레이어가 없는 페이지를 플레이스홀더로 만들지
않기 위해서다(`_truncation_verdict`).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..sidecar.client import (
    SidecarClient,
    SidecarError,
    SidecarTimeoutError,
    SidecarUnavailableError,
)
from ..sidecar.materializer import ChunkMaterializer
from ..sidecar.protocol import FIGURE_PLACEHOLDER_RE, PageResult, sanitize_page
from .base import (
    EngineCapabilities,
    EngineError,
    JobCanceled,
    OCREngine,
    OutputLimitError,
    StreamSink,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..pipeline.fidelity import PageFidelity

logger = logging.getLogger(__name__)

# health 프로브 캐시 TTL — 성공·실패 **둘 다** 캐시한다. 실패를 캐시하지 않으면
# /api/health 폴링(프런트 10초 주기)마다 죽은 sidecar로 연결을 시도해 요청 스레드가
# connect timeout만큼 묶이고, 잡 실행 중에는 페이지마다 같은 대기가 반복된다.
_HEALTH_CACHE_TTL_S = 5.0
_MAX_JOB_WARNINGS = 40
_MODEL_WAIT_POLL_S = 3.0   # wait_until_ready 폴링 간격
_CANCEL_POLL_S = 0.1       # _AnyCancel.wait의 폴링 슬라이스
# runner의 렌더 규약 — {job.dir}/pages/page_%04d.png (1-based 전역 페이지 번호)
_PAGE_IMAGE_RE = re.compile(r"^page_(\d+)\.png$")

# 대기 중 진행 문구(잡 진행 note·잡 경고) — 같은 상태에서는 문구가 바뀌지 않아야 경고가
# 시도마다 쌓이지 않는다. 시도 횟수·남은 시간 같은 세부는 예외 메시지와 health에 싣는다.
_WAIT_NOTE_LOADING = "모델 로딩 대기 중… (최초 기동은 다운로드·컴파일로 수 분 소요)"
_WAIT_NOTE_RETRY = "모델 로드 재시도 대기 중… (일시적 로드 실패 — sidecar가 자동으로 다시 시도)"
_WAIT_NOTE_RESTART = "sidecar 재시작 대기 중… (추론 엔진 복구 — 모델 재로드 뒤 이어서 진행)"


class SidecarNotReadyError(EngineError):
    """sidecar는 응답하지만 모델이 아직 로드 중 — 일시적(대기하면 준비됨).

    note는 대기 중 진행 문구다(첫 로드·로드 재시도·재시작 대기를 구분한다)."""

    transient = True

    def __init__(self, message: str, note: str = _WAIT_NOTE_LOADING) -> None:
        super().__init__(message)
        self.note = note


class SidecarOutputTruncated(OutputLimitError):
    """sidecar가 출력 토큰 상한에서 끊긴 페이지를 보고했고, 잃은 것이 커서 다른 원천으로
    복구해야 한다(`OutputLimitError`라 runner의 잘림 복구 경로를 그대로 탄다).

    partial_output은 이 호출의 반환 형식 그대로다(run_multi면 `<PAGE>` 구분 — 끝까지 받은
    앞 페이지 + 잘린 페이지, 산출물은 out_dir). OutputLimitError 생성자가 partial_output을
    받기 전후 모두 동작하도록 속성으로 붙인다."""

    # 그리디(temperature 0)·같은 이미지·같은 상한이라 같은 페이지를 다시 보내도 같은 곳에서
    # 잘린다 — runner가 같은 페이지를 다시 돌리지 않게 알린다.
    retry_same_page = False
    # runner 경고 문구용 상한 이름 (MAX_LENGTH가 아니라 sidecar의 페이지당 출력 상한이다)
    limit_label = "sidecar 출력 토큰 상한"


class _AnyCancel:
    """여러 취소 신호의 OR — 잡 취소 + 청크 내부 실패(형제 요청 중단)를 합친다."""

    def __init__(self, *signals) -> None:
        self._signals = signals

    def is_set(self) -> bool:
        return any(s.is_set() for s in self._signals)

    def wait(self, timeout: float) -> bool:
        """threading.Event.wait 호환 — 어느 신호든 관측되면 즉시 True.

        wait_until_ready(복귀 대기)가 취소 가능한 슬립으로 쓴다. 여러 신호를
        동시에 기다릴 방법이 없으므로 짧은 슬라이스로 폴링한다."""
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(_CANCEL_POLL_S, remaining))
        return True


def _live_stream_text(page: PageResult) -> str:
    """라이브 뷰용 스트림 표현 — 텍스트(markdown) + figure 그라운딩 토큰.

    페이지 markdown의 `[[FIGURE:n]]` placeholder를 해당 image 블록의 bbox로
    `<|det|>image [x1, y1, x2, y2]<|/det|>` 그라운딩 토큰으로 치환한다. 프론트의
    라이브 파서가 이 토큰으로 원본 이미지 위에 박스를 그리고(왼쪽 패널),
    structurePreview가 나머지 markdown을 렌더한다(RAW·미리보기 패널). 이 표현은
    라이브 뷰 전용이며 저장/병합되는 결과 markdown에는 영향을 주지 않는다.

    figure_only 엔진(Ovis)은 텍스트 bbox가 없으므로 텍스트는 markdown 그대로
    흐르고 figure만 박스가 되며, full-layout 엔진(Paddle)도 동일하게 동작한다.
    """
    boxes: dict[int, tuple[int, int, int, int]] = {
        b.figure_index: b.bbox
        for b in page.blocks
        if b.type == "image" and b.figure_index is not None and b.bbox is not None
    }

    def _repl(m) -> str:
        idx = int(m.group(1))
        bbox = boxes.get(idx)
        if bbox is None:
            return ""  # crop이 폐기된 figure — 박스 없이 제거
        x1, y1, x2, y2 = bbox
        return f"<|det|>image [{x1}, {y1}, {x2}, {y2}]<|/det|>"

    return FIGURE_PLACEHOLDER_RE.sub(_repl, page.markdown)


def _retry_summary(retry: dict) -> str:
    """health.load_retry → '2/5번째 시도 실패, 30초 뒤 재시도 — 마지막 오류: …' (있는 값만)."""
    parts: list[str] = []
    attempt, total = retry.get("attempt"), retry.get("max_attempts")
    if attempt is not None:
        parts.append(f"{attempt}/{total}번째 시도 실패" if total else f"{attempt}번째 시도 실패")
    delay = retry.get("next_retry_s")
    if delay is not None:
        parts.append(f"{delay:.0f}초 뒤 재시도")
    text = ", ".join(parts) or "재시도 대기"
    error = retry.get("last_error")
    if error:
        text += f" — 마지막 오류: {error}"
    return text


def _truncated_page_fidelity(
    image_path: Path, page: PageResult, text_bboxes: bool, should_cancel,
) -> "PageFidelity | None":
    """잘린 페이지 출력을 원본 PDF 텍스트 레이어와 대조한다 — 충실도 게이트와 같은 잣대.

    원본은 runner의 잡 디렉터리 규약({job.dir}/pages/page_%04d.png ↔ {job.dir}/source.pdf,
    textlayer 엔진과 같은 가정)으로 찾는다. 규약 밖 경로면 None(판정 불가).
    텍스트 bbox가 없는 엔진(figure_only)은 페이지 markdown을 본문 블록 하나로 보고,
    그림 블록은 bbox째 넘겨 그림 안 글자를 정답에서 뺀다.
    """
    m = _PAGE_IMAGE_RE.match(image_path.name)
    source = image_path.parent.parent / "source.pdf"
    if m is None or not source.is_file():
        return None
    from ..pipeline.fidelity import evaluate_layout_pages

    blocks: list[dict] = []
    for b in page.blocks:
        if b.type != "image" and not text_bboxes:
            continue
        block: dict = {"type": b.type, "content": b.content}
        if b.bbox is not None:
            block["bbox"] = list(b.bbox)
        blocks.append(block)
    if not text_bboxes:
        blocks.append({"type": "text", "content": FIGURE_PLACEHOLDER_RE.sub(" ", page.markdown)})
    return evaluate_layout_pages(
        source, [{"page": int(m.group(1)), "blocks": blocks}], should_cancel
    )[0]


# fidelity.page_fidelity_blocks가 정답(텍스트 레이어) 글자가 모자랄 때 남기는 사유
_SHORT_TRUTH_REASON = "정답 텍스트 부족"


def _no_text_layer(fid: "PageFidelity") -> bool:
    """대조가 끝까지 돌았고 텍스트 레이어에 글자가 하나도 없었는가 (스캔 페이지).

    시간 상한·처리 프로세스 사망·원본 열기 실패도 truth_chars=0으로 오지만, 그건
    '텍스트 레이어가 없다'가 아니라 '대조하지 못했다'다."""
    return fid.truth_chars == 0 and fid.reason == _SHORT_TRUTH_REASON


def _unjudged_reason(fid: "PageFidelity") -> str:
    """판정하지 못한 사유 — 분석 시간 상한이면 조정할 설정 이름을 붙인다."""
    from ..pipeline.fidelity import ANALYSIS_TIMEOUT_REASON

    if fid.timed_out and fid.reason == ANALYSIS_TIMEOUT_REASON:
        return f"{fid.reason}; 상한은 PDF_PAGE_TIMEOUT_S"
    return fid.reason


@dataclass(frozen=True)
class SidecarSpec:
    default_model_id: str
    layout_capability: str  # "full" | "figure_only"


# 지원 sidecar 엔진 선언 — 새 엔진은 여기와 services/에 추가한다
SIDECAR_SPECS: dict[str, SidecarSpec] = {
    "ovisocr2": SidecarSpec(
        default_model_id="ATH-MaaS/OvisOCR2",
        layout_capability="figure_only",  # 텍스트 bbox 미제공 — 거짓 full 금지
    ),
    "paddleocr_vl": SidecarSpec(
        default_model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        layout_capability="full",
    ),
}


class SidecarEngine(OCREngine):
    def __init__(self, settings: "Settings", name: str) -> None:
        if name not in SIDECAR_SPECS:  # registry가 걸러주지만 방어적으로
            raise ValueError(f"알 수 없는 sidecar 엔진: {name!r}")
        self.name = name
        self.device = "cuda"          # provider가 CUDA에서 실행 (backend 자체는 GPU 미사용)
        self.dtype_name = "bfloat16"
        self._spec = SIDECAR_SPECS[name]
        self._settings = settings
        self._client = SidecarClient(
            settings.sidecar_url,
            engine_name=name,
            connect_timeout_s=settings.sidecar_connect_timeout_s,
            read_timeout_s=settings.sidecar_read_timeout_s,
            health_timeout_s=settings.sidecar_health_timeout_s,
            max_response_mb=settings.sidecar_max_response_mb,
            retries=settings.sidecar_retries,
        )
        self._health_lock = threading.Lock()
        self._last_health = None       # SidecarHealth | None (성공 프로브 결과)
        self._last_health_error = None  # str | None (실패 사유 — 실패도 캐시한다)
        self._last_probe_ts = 0.0
        self._warn_lock = threading.Lock()
        self._warnings: list[str] = []
        self._degraded_noted = False   # status 이상 신고를 이번 잡에 이미 경고로 올렸는지
        self._degraded_refreshed = False  # 성공한 parse 뒤 이상 신고 캐시를 다시 확인했는지
        self._job_key: Path | None = None  # 잡 단위 상태(신고 1회·장애 대기)의 기준 잡
        # 장애 복귀 대기의 공유 데드라인 — 페이지마다 OCR_SIDECAR_MODEL_WAIT_S씩 따로
        # 기다리지 않게 첫 대기 시작 시각을 기준으로 엔진 인스턴스에서 공유한다.
        self._outage_deadline: float | None = None
        # run_multi가 잘림(SidecarOutputTruncated)으로 넘긴 페이지 → 사유. runner는 곧바로
        # 그 페이지를 run_single로 다시 부르는데, 같은 요청은 같은 곳에서 다시 잘리므로
        # sidecar에 보내지 않고 같은 예외로 답한다(1회용 — 다음 run_multi·잡 전환에서 비움).
        self._truncated_replay: dict[Path, str] = {}

    # ── 상태/메타 ──────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        # `model_loaded` 축만 본다. `status` 축은 별개다 — sidecar는 임계 기반
        # 자가 복구형 웨지 신고를 status="error" + model_loaded=True로 낸다.
        # 그걸 미로드로 취급하면 오탐 1건이 모든 잡을 "모델 로드 실패"로 즉시
        # 마감시키고, sidecar는 요청을 못 받아 스스로 복구할 수도 없다.
        # status 이상은 _check_ready가 하드 실패/잡 경고로 나눠 처리한다.
        h = self._last_health
        return h is not None and h.model_loaded

    def _probe_health(self):
        """health 프로브 (성공·실패 모두 TTL 캐시) — /api/health 폴링용.
        반환: (health|None, error|None). 준비 대기 루프는 _check_ready(직접·유형 구분)를 쓴다."""
        now = time.monotonic()
        with self._health_lock:
            if now - self._last_probe_ts < _HEALTH_CACHE_TTL_S and (
                self._last_health is not None or self._last_health_error is not None
            ):
                return self._last_health, self._last_health_error
        health = None
        error: str | None = None
        try:
            health = self._client.health()
        except SidecarError as e:
            error = str(e)[:300]
        with self._health_lock:
            self._last_health = health
            self._last_health_error = error
            self._last_probe_ts = time.monotonic()
        if health is not None:
            self.dtype_name = health.dtype or self.dtype_name
            self.device = health.device or self.device
            self._note_if_degraded(health)
        return health, error

    def _commit_health(self, h) -> None:
        """정상 health 응답을 캐시에 반영 (loaded/gpu_name/device/dtype 갱신)."""
        with self._health_lock:
            self._last_health = h
            self._last_health_error = None
            self._last_probe_ts = time.monotonic()
        self.dtype_name = h.dtype or self.dtype_name
        self.device = h.device or self.device
        self._note_if_degraded(h)

    def _note_if_degraded(self, h) -> None:
        """모델은 로드돼 있는데 status만 이상한 신고를 잡 경고로 올린다 (하드 실패 금지).

        **잡당 1회**만 올린다 — 예전에는 드레인(= 청크, 기본 1페이지)마다 풀려서 200쪽
        잡이면 같은 문장이 페이지마다 쌓여 잡 경고 상한을 채우고 실제 손실 경고를
        밀어냈다. status가 ok로 돌아오면 플래그를 풀어 다음 신고를 다시 알린다."""
        degraded = h.model_loaded and h.status != "ok"
        with self._health_lock:
            if not degraded:
                self._degraded_noted = False
                return
            if self._degraded_noted:
                return
            self._degraded_noted = True
        detail = h.load_error or f"sidecar 상태 이상({h.status})"
        self._note(
            f"sidecar가 이상 상태를 보고했습니다({detail}) — 모델은 로드돼 있어 그대로 "
            "진행합니다. 페이지가 계속 실패하면 sidecar를 재시작하세요."
        )

    def _invalidate_health(self, reason: str) -> None:
        """health 캐시 무효화 — 실패를 관측했으면 stale 캐시를 신뢰하면 안 된다.

        이걸 빼면 loaded가 stale True로 남아 wait_until_ready가 즉시 반환하고
        복귀 대기 자체가 무효화된다."""
        with self._health_lock:
            self._last_health = None
            self._last_health_error = reason[:300]
            self._last_probe_ts = 0.0
            self._degraded_noted = False  # 복귀 후 다시 이상하면 새 경고를 남긴다

    def _check_ready(self, force: bool = False) -> None:
        """준비 상태를 확인하고 미준비 시 **유형별** 예외를 던진다.

        전송 계층 예외를 문자열로 평탄화하지 않고 직접 받아 구분한다:
        - 연결 실패(SidecarUnavailableError)·모델 미로드: 일시적(transient) — 대기하면 풀림
        - **프로토콜/엔진 불일치(SidecarProtocolError)**: 영구 오설정(URL 오배선·버전 불일치)
          — 대기해도 안 풀리므로 하드 실패(EngineError)로 즉시 전파해야 한다.
        - sidecar 자체 로드 실패(status=error 이면서 **model_loaded=False**): 하드 실패
        - status=error인데 **model_loaded=True**: 자가 복구형 웨지 신고(오탐 가능) —
          하드 실패시키지 않고 잡 경고로 올린 뒤 통과시킨다. 진짜 이상이면 parse가
          502로 답해 기존 청크 격리가 받고, 오탐이면 성공 1회로 자동 복구된다.
        준비됐으면 조용히 반환(loaded=True).
        """
        if self.loaded and not force:
            # 캐시가 이미 이상 신고를 담고 있으면(웨지 의심) 이 잡에도 알린다 —
            # 캐시 히트로 조용히 빠져나가면 잡은 신고를 영영 못 본다.
            h = self._last_health
            if h is not None:
                self._note_if_degraded(h)
            return
        try:
            h = self._client.health()
        except SidecarUnavailableError as e:
            # 기동 직후엔 컨테이너가 아직 안 떴을 수 있어 일시적으로 취급
            raise SidecarNotReadyError(f"sidecar에 아직 연결할 수 없습니다: {e}") from e
        except SidecarError as e:
            # 프로토콜/엔진 불일치 등 — 대기 무의미, 하드 실패
            raise EngineError(f"sidecar 통신 오류(대기해도 해소되지 않음): {e}") from e
        self._commit_health(h)
        if not h.model_loaded:
            if h.restarting:
                # 추론 엔진이 죽어 sidecar가 스스로 종료·재기동하는 중 — 컨테이너가 돌아오면
                # 모델을 다시 올린다. status와 무관하게 기다리면 풀리는 상태다.
                raise SidecarNotReadyError(
                    "sidecar가 추론 엔진 복구를 위해 재시작하는 중입니다 — 컨테이너 재기동과 "
                    "모델 재로드 뒤 이어서 진행합니다 (진행: docker compose logs -f)",
                    note=_WAIT_NOTE_RESTART,
                )
            if h.status != "ok":
                # 모델이 없는데 status까지 error — 진짜 로드 실패다 (하드 실패)
                detail = h.load_error or f"sidecar 상태 이상({h.status})"
                raise EngineError(f"sidecar 모델 로드 실패: {detail}")
            if h.load_retry:
                raise SidecarNotReadyError(
                    f"sidecar 모델 로드가 일시적으로 실패해 다시 시도하는 중입니다 "
                    f"({_retry_summary(h.load_retry)})",
                    note=_WAIT_NOTE_RETRY,
                )
            raise SidecarNotReadyError(
                "sidecar가 아직 모델을 로드하지 못했습니다 — 최초 기동은 모델 다운로드·"
                "컴파일로 수 분 걸릴 수 있습니다 (진행: docker compose logs -f)"
            )
        # 모델이 살아 있는데 status만 이상한 경우는 _commit_health → _note_if_degraded가
        # 잡 경고로 올렸다. 여기서는 통과시킨다 (하드 실패 금지).

    def load(self) -> None:
        """준비 상태 1회 확인 (대기 없음). 미준비면 유형별 예외.

        멱등: 이미 loaded면 즉시 반환. 프리로드(main)와 워커가 호출한다 — 워커는
        실제로는 wait_until_ready로 대기하고, load()는 단발 확인/프리로드용이다.
        """
        self._check_ready()

    def wait_until_ready(self, cancel, on_wait=None) -> None:
        """모델이 준비될 때까지 취소 가능하게 폴링 대기 (상한 sidecar_model_wait_s).

        최초 기동의 다운로드·컴파일 창에 업로드해도 잡을 실패시키지 않고 기다린다.
        하드 실패(sidecar 자체 로드 실패)는 대기하지 않고 즉시 전파, 취소 시 JobCanceled.
        워커가 잡 시작 때 부른다 — 잡마다 새 대기 예산으로 시작한다."""
        with self._health_lock:
            self._outage_deadline = None
        if self.loaded:
            return
        self._wait_ready(
            cancel, time.monotonic() + self._settings.sidecar_model_wait_s, on_wait
        )

    def _await_recovery(self, cancel) -> None:
        """잡 도중 관측한 장애에서 복귀를 기다린다 — 대기 예산은 엔진 인스턴스가 공유.

        예전에는 청크 시작의 load()가 대기 없이 health를 한 번만 보고 실패해서, 재기동
        중인 sidecar의 남은 페이지가 수 ms 만에 전부 플레이스홀더가 됐다. 반대로
        페이지마다 sidecar_model_wait_s씩 기다리면 긴 장애에서 페이지 수만큼 곱해진다.
        첫 대기 시작부터 한 번의 예산만 쓰고, 다 쓴 뒤로는 페이지마다 한 번만 확인한다.
        """
        now = time.monotonic()
        with self._health_lock:
            if self._outage_deadline is None:
                self._outage_deadline = now + self._settings.sidecar_model_wait_s
            deadline = self._outage_deadline
        if now >= deadline:
            self._check_ready(force=True)  # 여전히 내려가 있으면 예외 — 대기 없이 실패
        else:
            self._wait_ready(cancel, deadline, self._note)
        with self._health_lock:
            self._outage_deadline = None

    def _wait_ready(self, cancel, deadline: float, on_wait=None) -> None:
        last = ""
        note = _WAIT_NOTE_LOADING
        while True:
            if cancel.is_set():
                raise JobCanceled()
            try:
                self._check_ready(force=True)
                return  # 준비됨
            except SidecarNotReadyError as e:
                last = str(e)
                note = e.note  # 첫 로드·로드 재시도·재시작 대기를 구분해 보인다
            # EngineError(하드 실패)는 여기서 잡지 않고 그대로 전파 — 대기 무의미
            if time.monotonic() >= deadline:
                raise EngineError(
                    f"sidecar 모델이 제한시간({int(self._settings.sidecar_model_wait_s)}초) 내에 "
                    f"준비되지 않았습니다 ({last}). OCR_SIDECAR_MODEL_WAIT_S로 늘리거나 "
                    "docker compose logs로 sidecar 상태를 확인하세요."
                )
            if on_wait is not None:
                on_wait(note)
            cancel.wait(_MODEL_WAIT_POLL_S)  # 취소 가능한 슬립

    def capabilities(self) -> EngineCapabilities:
        h = self._last_health
        return EngineCapabilities(
            model_id=(h.model_id if h else "") or self._spec.default_model_id,
            model_revision=(h.model_revision if h else ""),
            provider="local-sidecar",
            supports_multi_page=False,       # 페이지 단위 모델 — 문맥 공유 없음
            preferred_chunk_size=max(1, self._settings.remote_page_concurrency),
            stream_granularity="page",
            layout_capability=self._spec.layout_capability,
            figure_capability=True,
        )

    def provider_health(self) -> dict | None:
        """/api/health 폴링용 — 성공·실패 모두 TTL 캐시 (죽은 sidecar 폴링이
        요청 스레드를 connect timeout만큼 묶지 않게)."""
        health, error = self._probe_health()
        if health is None:
            return {"status": "unreachable", "error": error}
        return self._health_dict(health)

    # ── 경고 채널 ───────────────────────────────────────────────

    def _note(self, message: str) -> None:
        """사용자 노출용 경고 적재 (페이지 스레드에서도 호출되므로 락 보호).

        **중복은 적재 시점에 접는다.** drain_warnings가 어차피 중복을 1건으로
        접으므로 손실은 없고, 접지 않으면 반복 호출되는 경고 하나가 40칸 예산을
        통째로 먹는다 — 503 복귀 대기의 on_wait=self._note는 3초마다 같은 문구를
        넣어 2분이면 버퍼를 채우고, 그 뒤 같은 청크의 손실 고지(정화로 버려진 표
        등)가 전부 조용히 버려진다.
        """
        with self._warn_lock:
            if message in self._warnings:
                return
            if len(self._warnings) < _MAX_JOB_WARNINGS:
                self._warnings.append(message)

    def drain_warnings(self) -> list[str]:
        with self._warn_lock:
            drained, self._warnings = self._warnings, []
        # 이상 신고 1회 제한은 여기서 풀지 않는다(풀면 청크마다 같은 신고가 쌓인다).
        # 잡이 바뀌면 _begin_job이 푼다 — runner가 잡 시작 때 이전 잡 잔여를 버리는
        # 드레인에 신고가 삼켜져도, 새 잡의 첫 준비 확인이 다시 올린다.
        # 페이지마다 반복되는 동일 경고는 1건으로 접는다 (순서 보존)
        seen: set[str] = set()
        unique: list[str] = []
        for w in drained:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique

    @staticmethod
    def _health_dict(h) -> dict:
        return {
            "status": h.status,
            "runtime": h.runtime,
            "version": h.runtime_version,
            "model_loaded": h.model_loaded,
            "gpu_total_mb": h.gpu_total_mb,
            "gpu_free_mb": h.gpu_free_mb,
            # 미로드의 이유 — 일시적 로드 실패 뒤 재시도 대기(attempt·max_attempts·
            # next_retry_s·last_error) / 추론 엔진 사망 뒤 컨테이너 재시작 대기
            "load_retry": h.load_retry,
            "restarting": h.restarting,
        }

    def gpu_name(self) -> str | None:
        h = self._last_health
        return h.gpu_name if h else None

    # ── 실행 ───────────────────────────────────────────────────

    def _parse_one(
        self, image_path: Path, local_page: int, cancel
    ) -> tuple[PageResult, list[str]]:
        """페이지 1장 파싱 → (정화된 페이지, 정화 경고). 경고 승격은 소비 쪽 몫이다 —
        복구 경로로 넘기는 페이지(잘림)나 앞 페이지 실패로 버려지는 형제 결과의 경고가
        잡 경고에 남지 않게."""
        request_id = f"{uuid.uuid4().hex[:12]}-p{local_page}"
        try:
            resp = self._client.parse_page(
                image_path,
                page_index=local_page,
                request_id=request_id,
                options={},
                cancel=cancel,
            )
        except SidecarTimeoutError:
            # provider는 살아서 그 페이지를 계속 추론 중일 수 있다 — 여기서 재요청하면
            # 같은 페이지를 GPU에서 두 번 돌린다. 상위 runner의 청크 재시도에 맡긴다.
            raise
        except SidecarUnavailableError as e:
            # sidecar 재시작/모델 재로드(HTTP 503·연결 끊김) — 컨테이너가 돌아오길
            # 기다렸다가 이 페이지만 1회 재시도한다. 기다리지 않으면 재기동+모델 로드
            # 시간 동안의 페이지가 전부 플레이스홀더로 확정된다. 대기 예산은 공유한다.
            self._invalidate_health(str(e))
            self._note("sidecar 재시작/모델 재로드 대기 중… (해당 페이지는 복귀 후 재시도)")
            self._await_recovery(cancel)
            resp = self._client.parse_page(
                image_path,
                page_index=local_page,
                request_id=f"{request_id}r",
                options={},
                cancel=cancel,
            )
        self._refresh_degraded_health()
        return sanitize_page(resp.page)

    def _note_page_warnings(self, page: PageResult, sanitize_warnings: list[str]) -> None:
        """정화로 버려진 블록·절단은 사용자에게 알린다 (조용한 내용 손실 방지).

        sidecar가 스스로 보고한 경고(해상도 강등·출력 상한 도달 등)도 함께 승격한다.
        페이지 번호는 붙이지 않는다 — local_page는 청크 내 인덱스라 기본 설정
        (청크=1페이지)에서는 항상 0이다. 전역 페이지 범위는 runner가 붙인다."""
        for w in sanitize_warnings:
            self._note(w)
        for w in page.warnings:
            if w not in sanitize_warnings:
                self._note(w)

    def _truncation_verdict(
        self, image_path: Path, page: PageResult, cancel
    ) -> SidecarOutputTruncated | str:
        """출력 상한에서 끊긴 페이지의 처리 — 복구로 넘길 예외, 또는 그대로 쓸 때의 안내 문구.

        runner의 잘림 복구는 텍스트 레이어로 그 페이지를 대신하고, 텍스트 레이어가 없으면
        플레이스홀더를 넣는다. 그래서 텍스트 레이어가 **믿을 만하고**(충실도 판정 가능)
        잘린 출력의 충실도가 OCR_FIDELITY_THRESHOLD 미만일 때만 넘긴다. 스캔 문서·판정
        불가·충실도 충분(끝부분만 조금 잘림)이면 잘린 출력을 경고와 함께 그대로 쓴다 —
        표·수식·그림 구조가 살아 있는 출력을 평문 텍스트 레이어나 빈 페이지로 바꾸지 않는다.
        """
        threshold = self._settings.ocr_fidelity_threshold
        if threshold > 0:
            failure = ""
            try:
                fid = _truncated_page_fidelity(
                    image_path, page, self._spec.layout_capability == "full", cancel.is_set,
                )
            except JobCanceled:
                raise
            except Exception as e:  # noqa: BLE001 — 판정 실패는 '유지'로 흡수한다
                logger.warning(
                    "잘린 페이지 충실도 판정 실패 (%s: %s) — 잘린 출력을 유지",
                    e.__class__.__name__, str(e)[:200],
                )
                fid = None
                failure = e.__class__.__name__
            if fid is not None and fid.score is not None and fid.score < threshold:
                return SidecarOutputTruncated(
                    "sidecar 출력이 페이지당 출력 토큰 상한에서 잘림 — PDF 텍스트 레이어 대조 "
                    f"충실도 {fid.score:.2f} < {threshold:.2f}"
                )
            if fid is not None and fid.score is not None:
                why = f"PDF 텍스트 레이어 대조 충실도가 {fid.score:.2f}로 기준({threshold:.2f}) 이상이라"
            elif fid is not None and fid.reason and not _no_text_layer(fid):
                # 텍스트 레이어가 없는 게 아니라 대조를 못 했다 — 시간 상한·처리 프로세스 사망·
                # 원본 열기 실패·신뢰할 수 없는 텍스트 레이어. 사유를 그대로 보여야 운영자가
                # 진짜 원인(예: PDF_PAGE_TIMEOUT_S)을 본다.
                why = f"텍스트 레이어로 판정할 수 없어({_unjudged_reason(fid)})"
            elif failure:
                why = f"텍스트 레이어 대조가 실패해({failure})"
            else:
                why = "대조할 PDF 텍스트 레이어가 없어"
        else:
            why = "충실도 판정이 꺼져 있어(OCR_FIDELITY_THRESHOLD≤0)"
        return f"출력 토큰 상한에서 잘린 페이지 — {why} 잘린 출력을 그대로 씁니다"

    def _begin_job(self, image_paths: list[Path]) -> None:
        """잡이 바뀌면 잡 단위 상태(이상 신고 1회·성공 뒤 재확인·장애 대기)를 푼다.

        runner는 늘 {job.dir}/pages/page_%04d.png를 넘긴다 — 그 두 단계 위가 잡 키다
        (textlayer 엔진의 잡당 1회 경고와 같은 방식)."""
        key = image_paths[0].parent.parent if image_paths else None
        with self._health_lock:
            if key == self._job_key:
                return
            self._job_key = key
            self._degraded_noted = False
            self._degraded_refreshed = False
            self._outage_deadline = None
            self._truncated_replay.clear()

    def _refresh_degraded_health(self) -> None:
        """parse가 성공했는데 캐시가 이상 신고를 들고 있으면 잡당 한 번 다시 확인한다.

        sidecar는 성공 1회로 자가 복구 신고를 지우는데 backend 캐시는 그대로라, 복구된
        뒤에도 잡이 계속 degraded로 보였다."""
        h = self._last_health
        if h is None or h.status == "ok":
            return
        with self._health_lock:
            if self._degraded_refreshed:
                return
            self._degraded_refreshed = True
        try:
            fresh = self._client.health()
        except SidecarError:
            return
        self._commit_health(fresh)

    def _ensure_ready(self, cancel) -> None:
        """청크 시작의 준비 확인 — 캐시가 '준비됨'이면 바로, 아니면 복귀를 기다린다."""
        if self.loaded:
            self._check_ready()  # 캐시 히트 — 이상 신고만 이 잡에 올린다
            return
        self._await_recovery(cancel)

    def _run_pages(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
        single: bool,
    ) -> str:
        self._begin_job(image_paths)
        if single:
            replay = self._truncated_replay.pop(image_paths[0], None)
            if replay is not None:
                # 방금 run_multi가 잘림으로 넘긴 페이지를 runner가 페이지 단위로 다시 부른 것 —
                # 같은 요청은 같은 곳에서 다시 잘린다. GPU에 다시 보내지 않고 같은 판정을 낸다.
                raise SidecarOutputTruncated(replay)
        else:
            self._truncated_replay.clear()  # 새 청크 — 이전 청크의 1회용 표식은 무효
        self._ensure_ready(cancel)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 텍스트 bbox가 없는 엔진(figure_only)은 raw_pages.json에 좌표를 싣지 않는다 — 실으면
        # image 블록뿐인 layout.json이 생겨 HTML·PDF 내보내기가 OCR 텍스트를 잃는다.
        # 파일은 페이지마다 빈 원출력으로 남는다(merge가 원출력 개수로 페이지 수를 맞춰 본다).
        mat = ChunkMaterializer(
            out_dir, single=single,
            write_raw=self._spec.layout_capability == "full",
        )
        parts: list[str] = []

        concurrency = min(len(image_paths), max(1, self._settings.remote_page_concurrency))
        if concurrency <= 1:
            pages = self._iter_serial(image_paths, cancel)
        else:
            pages = self._iter_concurrent(image_paths, cancel, concurrency)

        try:
            for local_page, (page, sanitize_warnings) in pages:
                image_path = image_paths[local_page]
                verdict = (
                    self._truncation_verdict(image_path, page, cancel) if page.truncated else None
                )
                if isinstance(verdict, SidecarOutputTruncated):
                    self._hand_over_truncated(verdict, mat, parts, page, image_path,
                                              local_page, single)
                self._note_page_warnings(page, sanitize_warnings)
                if verdict is not None:
                    self._note(verdict)  # 잘린 출력을 그대로 쓰는 이유
                # 취소 이후 도착한 결과는 병합하지 않는다 (여기 도달 전에 JobCanceled 전파)
                # 라이브 뷰용 스트림은 **그라운딩 토큰 표현**으로 발행한다(처리된 md와 별개):
                # figure는 <|det|>image [bbox]<|/det|>로 내보내 왼쪽 원본+레이아웃 패널의
                # 실시간 박스 오버레이가 그려지게 하고, 텍스트는 마크다운 그대로 흘려
                # RAW/미리보기 패널이 채워지게 한다 (Unlimited의 라이브 경험과 동일).
                live = _live_stream_text(page)
                if single:
                    sink.on_text(live)
                else:
                    sink.on_text("<PAGE>\n")
                    sink.on_text(live + "\n")
                # 이 페이지 토큰을 즉시 flush한다 — 동시성>1의 다중 페이지 청크에서 다음
                # 페이지의 <PAGE>가 유발하는 progress(current_page+1)보다 **먼저** 와이어에
                # 실리게 해, 라이브 박스가 다음 페이지로 오귀속되는 것을 막는다.
                # (StreamSink는 flush를 요구하지 않으므로 있는 경우에만 호출)
                flush = getattr(sink, "flush", None)
                if callable(flush):
                    flush()
                # 반환/병합용은 처리된 마크다운(![](images/…))을 그대로 유지한다
                md = mat.add_page(page, image_path, local_page)
                parts.append(md)
        finally:
            # 소비가 예외로 끊겨도 동시 요청 생성기의 정리(형제 요청 중단·executor 종료)를
            # 지금 돌린다 — 생성기가 yield에 멈춘 채면 그 정리는 GC 때까지 미뤄진다.
            pages.close()
        for w in mat.warnings:
            self._note(w)
        mat.finalize()
        if single:
            return parts[0] if parts else ""
        return "<PAGE>\n" + "\n<PAGE>\n".join(parts)

    def _hand_over_truncated(
        self,
        error: SidecarOutputTruncated,
        mat: ChunkMaterializer,
        parts: list[str],
        page: PageResult,
        image_path: Path,
        local_page: int,
        single: bool,
    ) -> None:
        """잘린 페이지를 runner의 잘림 복구로 넘긴다 — 산출물을 확정하고 예외를 올린다.

        앞 페이지들과 잘린 페이지까지 materialize·finalize해 partial_output을 이 호출의
        반환 형식 그대로 만든다(run_multi면 runner가 끝까지 받은 앞 페이지를 살리고 잘린
        페이지부터 다시 처리한다). 버리는 페이지의 경고는 올리지 않는다 — 그 자리는
        runner가 텍스트 레이어로 채우고 사유를 남긴다. 라이브 스트림에도 내보내지 않는다.
        """
        before = len(mat.warnings)
        md = mat.add_page(page, image_path, local_page)
        del mat.warnings[before:]
        for w in mat.warnings:
            self._note(w)
        mat.finalize()
        if single:
            error.partial_output = md
        else:
            error.partial_output = "<PAGE>\n" + "\n<PAGE>\n".join([*parts, md])
            self._truncated_replay[image_path] = str(error)
        raise error

    def _iter_serial(self, image_paths: list[Path], cancel: threading.Event):
        """(local_page, (정화된 페이지, 정화 경고))를 순서대로 낸다."""
        for local_page, path in enumerate(image_paths):
            if cancel.is_set():
                raise JobCanceled()
            yield local_page, self._parse_one(path, local_page, cancel)

    def _iter_concurrent(
        self, image_paths: list[Path], cancel: threading.Event, concurrency: int
    ):
        """페이지를 동시 요청하되 **순서대로** 소비한다 (SSE·병합 순서 보존).

        OCR_REMOTE_PAGE_CONCURRENCY>1일 때만. sidecar가 자체 큐로 직렬화하더라도
        요청 파이프라이닝으로 왕복 지연을 숨긴다.

        중단(취소·페이지 실패) 시 내부 stop 신호를 잡 취소와 OR로 묶어 형제 요청의
        대기를 즉시 푼다. 다만 **이미 전송된 요청은 sidecar에서 완주**한다 —
        추론 중에는 응답 헤더가 없어 연결을 실제로 끊을 수단이 없기 때문이다
        (docs/OCR_ENGINE_PROTOCOL.md §취소 의미론과 한계). 재시도는 그 뒤에 줄을 선다.
        """
        stop = threading.Event()
        signal = _AnyCancel(cancel, stop)
        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="sidecar-page")
        try:
            futures = [
                executor.submit(self._parse_one, path, i, signal)
                for i, path in enumerate(image_paths)
            ]
            for i, fut in enumerate(futures):
                if cancel.is_set():
                    raise JobCanceled()
                yield i, fut.result()
        finally:
            stop.set()  # 미완료 형제 요청의 연결을 끊는다 (다음 시도의 큐를 비움)
            executor.shutdown(wait=False, cancel_futures=True)

    def run_multi(
        self,
        image_paths: list[Path],
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        return self._run_pages(image_paths, out_dir, sink, cancel, single=False)

    def run_single(
        self,
        image_path: Path,
        out_dir: Path,
        sink: StreamSink,
        cancel: threading.Event,
    ) -> str:
        return self._run_pages([image_path], out_dir, sink, cancel, single=True)
