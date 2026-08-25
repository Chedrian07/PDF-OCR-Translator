"""번역 오케스트레이터 — 공개 진입점 run_translation().

api.py는 이 함수만 안다. 워커 스레드에서 블로킹 호출되며, 진행률은
progress 콜백으로, 중단은 cancel 이벤트로 통신한다 (OCR 워커와 같은 패턴).

상태 전이: state.json을 이 함수가 직접 기록한다 —
  running(current/total 갱신) → done | error(message) | canceled
호출자는 예외를 SSE error 이벤트로만 중계하면 된다.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import re

from . import prompts
from .client import OpenAICompatClient
from .flight import SingleFlight
from .glossary import Glossary, build_glossary
from .masking import (
    _TOKEN_RE,
    mask,
    sanitize_translation,
    should_skip,
    unmask,
    untranslated_reason,
)
from .segment import (
    apply_layout,
    assemble_markdown,
    layout_line_sources,
    layout_units,
    reconcile_markdown_with_layout,
    reference_rule_mismatch,
    split_markdown,
)
from .types import (
    PROMPT_V,
    TranslateAPIError,
    TranslateConfig,
    TranslateError,
    TranslateResult,
    TranslateUnitRejected,
    cache_key,
)

logger = logging.getLogger(__name__)

# 문장 경계 — 종결부호 뒤 공백. 분할 지점·분할 가능 판정에 함께 쓴다.
_SENT_BOUND_RE = re.compile(r"(?<=[.!?…])\s+")
# 구조 유닛 감지 — 줄 시작이 표(|)·목록(-, *)·인용(>)·번호목록(숫자.)·펜스(```)인 줄.
_STRUCT_LINE_RE = re.compile(r"^\s*(?:[|>*-]|\d+\.|```)")


def _splittable(src: str) -> bool:
    """문장 분할 재시도 대상인가 — 여러 문장이고 구조 유닛(표·목록·인용·펜스)이 아니면 True."""
    if not _SENT_BOUND_RE.search(src):
        return False  # 문장 경계가 없으면 나눌 수 없다
    return not any(_STRUCT_LINE_RE.match(ln) for ln in src.split("\n"))


def _split_two(src: str) -> tuple[str, str] | None:
    """문장 경계 중 중앙에 가장 가까운 지점에서 2분할. (앞, 뒤) 또는 None(경계 없음/한쪽 공백).

    마스킹 토큰(여러 줄 $$ 수식 등) **내부**에 떨어지는 경계는 후보에서 제외한다 —
    토큰 한가운데를 자르면 반쪽의 짝 잃은 $$/$가 mask()에 잡히지 않아 원시 LaTeX가
    모델에 노출되고, sanitize가 잔여 $$를 지워 '조용한 수식 훼손'이 된다(무손실 위반).
    모든 경계가 토큰 내부면 None — 호출자가 분할을 포기하고 원문 유지로 귀결된다.
    """
    spans = [m.span() for m in _TOKEN_RE.finditer(src)]
    bounds = [
        b for b in (m.end() for m in _SENT_BOUND_RE.finditer(src))
        if not any(s < b < e for s, e in spans)
    ]
    if not bounds:
        return None
    mid = len(src) / 2
    cut = min(bounds, key=lambda b: abs(b - mid))
    left, right = src[:cut].rstrip(), src[cut:].lstrip()
    if not left or not right:
        return None
    return left, right


# HTML 표 유닛 — 문장 분할 대신 행(</tr>) 경계에서 자른다. 초대형 표(실측 6.4KB,
# 플레이스홀더 수십 개)는 한 번에 번역·repair가 모두 실패하는 유일한 유형이었다.
_TABLE_ROW_END_RE = re.compile(r"</tr\s*>", re.I)


def _is_table_unit(src: str) -> bool:
    return src.lstrip().lower().startswith("<table")


def _split_table(src: str) -> tuple[str, str] | None:
    """`</tr>` 경계 중 중앙에 가장 가까운 지점에서 2분할. 재결합은 단순 이어붙임 —
    행 사이 공백은 HTML 렌더에 무의미하므로 구조가 정확히 보존된다."""
    bounds = [m.end() for m in _TABLE_ROW_END_RE.finditer(src)]
    if len(bounds) < 2:
        return None  # 행이 하나뿐이면 나눠도 의미 없음
    mid = len(src) / 2
    # 마지막 </tr> 뒤에서 자르면 오른쪽이 </table>뿐이 된다 — 마지막 경계는 제외
    cut = min(bounds[:-1], key=lambda b: abs(b - mid))
    left, right = src[:cut], src[cut:]
    if not left.strip() or not right.strip():
        return None
    return left, right


def _fully_covered(src: str, covered: set[str]) -> bool:
    """유닛의 비어있지 않은 모든 줄이 layout 매핑 대상인가 (reconcile과 같은 strip 규칙)."""
    lines = [ln.strip() for ln in src.split("\n") if ln.strip()]
    return bool(lines) and all(ln in covered for ln in lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1))


def _zero_stats() -> dict:
    # kept_reason: 래더가 소진돼 원문 유지로 떨어질 때의 최초 패스 실패 사유
    return {"retried": 0, "repaired": 0, "split": 0, "sanitized": 0, "kept_reason": ""}


@dataclass
class _TranslationRun:
    """한 번역 run의 실행 상태 — 종전 run_translation 클로저(nonlocal)의 승격판.

    중첩 def가 공유하던 cfg·client·용어집·컨텍스트·통계·캐시·게이트 집계를 필드로
    올려 단계별로 단위 테스트할 수 있게 한다. 갱신 시점과 순서는 종전과 동일하다 —
    report.json의 숫자는 이 이동으로 달라지지 않는다.
    """

    job_dir: Path
    lang: str
    cfg: TranslateConfig
    page_separator: str = "\n\n---\n\n"
    progress: Callable[[int, int], None] | None = None
    cancel: threading.Event | None = None
    force: bool = False  # True면 유닛 캐시를 읽지 않고 전부 재번역 (쓰기는 함)
    client: object | None = None  # 테스트 주입용 — None이면 cfg로 OpenAICompatClient 생성

    # ── 진행률·집계 ──────────────────────────────────────────────────────────
    total: int = 0
    done: int = 0
    skipped: int = 0
    translated_n: int = 0
    cached_n: int = 0
    retried_n: int = 0    # 최초 패스 실패로 래더(repair/분할)에 진입한 유닛 수
    repaired_n: int = 0   # repair 패스로 복구된 유닛 수
    split_n: int = 0      # 문장 분할로 복구된 유닛 수
    sanitized_n: int = 0  # sanitize 치환 총건수 (모든 complete 출력 경로 합산)
    kept_original: list[str] = field(default_factory=list)
    # 관측용 사유별 집계 — "왜 이 문단이 영어로 남았나"를 report.json에서 구분한다.
    skip_reasons: dict[str, int] = field(default_factory=dict)   # references / already-korean / non-linguistic / identifier
    kept_reasons: dict[str, int] = field(default_factory=dict)   # gate-rejected / placeholder-mismatch / empty-output / api-rejected
    # 출력 게이트가 어떤 규칙으로 몇 번 거부했는가 (scaffold / refusal / hangul-ratio /
    # length-ratio). kept_reasons는 "래더까지 소진돼 영어로 남은" 최종 결과만 세므로,
    # 래더가 흡수한 거부(=추가 API 왕복 비용)와 오탐의 원인 규칙이 보이지 않았다.
    # 오탐 파도(임계값 회귀)와 진짜 공급자 고장을 이 분포로 구분한다.
    gate_reasons: dict[str, int] = field(default_factory=dict)
    gate_lock: threading.Lock = field(default_factory=threading.Lock)  # worker 스레드에서 증가한다
    ref_rule: dict = field(default_factory=dict)  # md·layout 참고문헌 규칙 불일치

    # ── 실행 자원 ────────────────────────────────────────────────────────────
    # 내부 abort — 유닛의 치명 오류(step-0 API 오류)가 전파되기 시작하면 set된다.
    # 이미 실행을 시작한 유닛이 다음 API 호출 전에 빠져나오게 해, 죽은 엔드포인트로
    # 남은 큐가 드레인되는 것(유닛 수 × 백오프)을 막는다. 사용자 취소(cancel)와 별개.
    abort: threading.Event = field(default_factory=threading.Event)
    # 이 run에서 API 왕복이 한 번이라도 성공했는가 = 엔드포인트·인증·설정은 정상이라는 신호.
    # step-0의 결정적 4xx를 유닛 강등으로 흡수해도 되는지 판단하는 데 쓴다.
    # **캐시 적중은 세우지 않는다** — API를 타지 않아 엔드포인트 건강의 증거가 아니다.
    # (세우면 모델명 오타로 전 요청이 400인 상태에서도 신규 유닛이 전부 kept로 강등돼
    #  잡이 조용히 done 된다.) 재개 run(이전 캐시 존재)에서 이게 0인 경우의 처리는
    #  _may_degrade·_verify_endpoint_health 참조.
    progressed: threading.Event = field(default_factory=threading.Event)
    # progressed가 아직 없는 상태에서 "판정을 run 끝으로 미룬 채" 강등한 4xx 건수와
    # 대표 메시지. 이전 run의 캐시가 있는 재개 run에서만 미룬다(_may_degrade 참조).
    unverified_rejects: int = 0     # gate_lock으로 보호 (worker 스레드에서 증가)
    unverified_msg: str = ""
    flights: SingleFlight = field(default_factory=SingleFlight)

    # ── 문서 상태 (execute 단계에서 채워진다) ────────────────────────────────
    md_text: str = ""
    layout_pages: list | None = None
    md_units: list = field(default_factory=list)
    lay_units: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    deferred: list = field(default_factory=list)
    # 직전 유닛 꼬리 컨텍스트 (같은 소스 내에서만). 실제 프롬프트 입력이므로
    # 캐시 키에도 넣어 같은 문장이 다른 문맥의 번역을 잘못 공유하지 않게 한다.
    context_map: dict[str, str] = field(default_factory=dict)
    glossary: Glossary | None = None
    results: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    tdir: Path = field(init=False)
    upath: Path = field(init=False)   # 유닛 캐시 파일 (translations/{lang}/units.json)
    started: str = field(init=False)

    def __post_init__(self) -> None:
        self.job_dir = Path(self.job_dir)
        self.tdir = self.job_dir / "translations" / self.lang
        self.tdir.mkdir(parents=True, exist_ok=True)  # error 상태도 기록할 수 있도록 선행 생성
        self.upath = self.tdir / "units.json"
        self.started = _now()

    # ── 상태·취소 ────────────────────────────────────────────────────────────

    def write_state(self, status: str, current: int, total_: int, error: str | None = None) -> None:
        cfg = self.cfg
        mode = getattr(self.client, "api_mode_used", "") or (cfg.api_mode if cfg.api_mode != "auto" else "")
        try:
            _atomic_write_json(self.tdir / "state.json", {
                "lang": self.lang,
                "status": status,
                "current": current,
                "total": total_,
                "error": error,
                "model": cfg.model,
                "api_mode": mode,
                "prompt_v": PROMPT_V,
                "context": cfg.context,
                # 캐시 키 재료 — translate_eval이 units.json 키를 재현할 때 쓴다
                "temperature": cfg.temperature,
                "reasoning": cfg.reasoning,
                "started_at": self.started,
                "finished_at": _now() if status in ("done", "error", "canceled") else None,
            })
        except FileNotFoundError:
            pass  # 잡 삭제 경합(DELETE가 디렉터리 제거) — 상태 기록은 best-effort

    def _cancel_set(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _halted(self) -> bool:
        return self.abort.is_set() or self._cancel_set()

    def flush_cache(self) -> None:
        snapshot = self.flights.snapshot()
        try:
            _atomic_write_json(self.upath, snapshot)
        except FileNotFoundError:
            pass  # 잡 삭제 경합 — 캐시는 잡과 함께 사라진다

    # ── 유닛 번역 ────────────────────────────────────────────────────────────

    def _max_toks(self, masked: str) -> int:
        # reasoning effort별 고정 예산 (types.REASONING_MAX_TOKENS, 사용자 확정) —
        # thinking 토큰이 같은 예산에서 차감되므로 길이 공식 대신 모드 상한을 그대로 쓴다.
        # 미사용 토큰은 과금되지 않고, 상한은 폭주 방지용이다.
        return self.cfg.max_output_tokens

    def _run_pass(self, prompt: str, max_toks: int, mapping: dict) -> tuple[str, list, list, int, str]:
        """complete → sanitize → unmask 한 번. (복원문, missing, dup, 치환건수, 정리된_원출력)."""
        raw = self.client.complete(prompts.SYSTEM_TRANSLATE, prompt, max_tokens=max_toks)
        # API 왕복 하나가 성공했다 = 엔드포인트·인증·설정은 정상. step-0의 결정적
        # 4xx를 유닛 단위로 강등해도 되는지 판단하는 신호(워커 안에서 즉시 세운다).
        self.progressed.set()
        clean, sc = sanitize_translation(raw)
        restored, missing, dup = unmask(clean, mapping)
        return restored, missing, dup, sc, clean

    def _accepted(self, src: str, restored: str, missing: list, dup: list, mapping: dict) -> bool:
        """유닛 출력 채택 판정 — 플레이스홀더 정합 + 비어있지 않음 + 출력 측 검증.

        거부문·요약·영문 echo는 플레이스홀더가 온전해도 통과시키지 않는다.
        거절된 출력은 기존 래더(repair→분할)가 흡수하고, 다 소진되면 "kept"로
        떨어져 report.json에 남는다. publish는 status=="translated"일
        때만 호출되므로 units.json 오염도 함께 차단된다.

        게이트 거부는 사유별로 집계한다 — 래더가 흡수해 최종적으로 번역에
        성공한 건도 세므로 "오탐이 얼마나 비쌌나"가 report.json에 남는다.
        """
        if missing or dup or not restored.strip():
            return False
        reason = untranslated_reason(src, restored, mapping)
        if reason:
            with self.gate_lock:
                self.gate_reasons[reason] = self.gate_reasons.get(reason, 0) + 1
            return False
        return True

    def _translate_fragment(
        self, src, pairs, first, ctx, stats, keep=None, unit_kind: str = "",
    ) -> str | None:
        """분할된 반쪽 하나를 독립 번역 — mask→complete→sanitize→unmask + repair 1회(추가 분할 없음).

        성공 시 복원문, 실패 시 None. sanitize 건수만 stats에 누적한다. 반쪽 단계의
        API 오류는 무손실 원칙상 치명적이지 않으므로 None 처리(원 유닛 원문 유지로 귀결).
        취소·abort는 API 호출 사이에서 확인한다 — 거대 표 래더가 취소 후에도 수 분간
        호출을 이어가지 않게(응답성)."""
        if self._halted():
            return None
        masked, mapping = mask(src)
        max_toks = self._max_toks(masked)
        prompt = prompts.build_unit_prompt(
            masked,
            pairs,
            first,
            context_tail=ctx,
            keep_terms=keep,
            unit_kind=unit_kind,
        )
        try:
            restored, missing, dup, sc, clean = self._run_pass(prompt, max_toks, mapping)
        except TranslateError:
            return None
        stats["sanitized"] += sc
        if self._accepted(src, restored, missing, dup, mapping):
            return restored
        if self._halted():
            return None
        try:
            rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
            r_restored, r_missing, r_dup, r_sc, _ = self._run_pass(rprompt, max_toks, mapping)
            stats["sanitized"] += r_sc
            if self._accepted(src, r_restored, r_missing, r_dup, mapping):
                return r_restored
        except TranslateError:
            pass
        return None

    def _unit_key(self, u):
        """유닛의 마스킹·용어집 파생값과 캐시 키 (유닛당 ~1.5ms — API 왕복의 0.04%)."""
        cfg = self.cfg
        masked, mapping = mask(u.src)
        pairs, first = self.glossary.for_unit(u.src, u.id)
        keep = self.glossary.keep_terms(u.src)
        ctx = self.context_map.get(u.id) if cfg.context else None
        # keep(A 원형)도 출력 정책을 바꾸므로 캐시 키에 포함 — (k, k) 쌍으로 해시
        key = cache_key(
            masked,
            cfg.model,
            pairs + first + [(k, k) for k in keep],
            original_src=u.src,
            unit_kind=u.kind,
            context_tail=ctx,
            temperature=cfg.temperature,
            reasoning=cfg.reasoning,
        )
        return masked, mapping, pairs, first, keep, key

    def _may_degrade(self, exc: TranslateUnitRejected) -> bool:
        """step-0의 결정적 4xx를 유닛 강등(래더 → 원문 유지)으로 흡수해도 되는가.

        증거는 두 축이고 성격이 다르다.
        - progressed: **이번 run**의 API 왕복 성공 = 엔드포인트·인증·설정이 지금 정상.
          이게 있으면 4xx는 그 유닛의 결함이므로 즉시 강등한다.
        - 이전 run이 남긴 유닛 캐시(flights.prior_keys): 캐시 적중은 API를 타지 않아
          엔드포인트 건강의 증거가 **아니지만**, 이 잡이 같은 설정으로 이미 번역됐다는
          이력이다. 취소·오류 후 재개(캐시 워밍)에서는 대부분이 캐시 적중이라
          progressed가 끝까지 안 설 수 있고, 그때 신규 유닛 하나의 4xx로 잡 전체를
          하드 실패시키면 부분 캐시 보존(재개 비용 절감)이 무의미해진다.
          그래서 여기서는 전파하지 않고 **판정만 run 끝으로 미룬다**
          (_verify_endpoint_health: 이전 캐시가 실제로 적중했는지로 확정).

        캐시가 아예 없는 콜드 run은 이전 성공 이력이 전혀 없으므로 종전대로 즉시
        전파한다 — 죽은 엔드포인트에서 남은 큐가 드레인되는 것도 함께 막는다.
        (재개 run에서 판정을 미루면 최악의 경우 남은 큐가 드레인되지만, 결정적 4xx는
         _RETRYABLE이 아니라 백오프 없이 즉시 실패하므로 비용은 유닛 수 × 왕복 1회
         수준이다. 반대로 즉시 전파하면 캐시 적중 순서에 따라 판정이 흔들린다.)
        """
        if self.progressed.is_set():
            return True
        if not self.flights.prior_keys:
            return False
        with self.gate_lock:
            self.unverified_rejects += 1
            if not self.unverified_msg:
                self.unverified_msg = str(exc)
        return True

    def translate_unit(self, u, precomputed=None):
        stats = _zero_stats()
        # 취소·abort 응답성: 이미 디스패치된 유닛도 API 호출·래더 단계 사이에서
        # 취소(사용자)·abort(치명 오류 전파)를 확인하고 "canceled" 센티널로 조기
        # 반환한다 (결과 미반영, kept 오염 없음).
        if self._halted():
            return u, u.src, "canceled", "", stats
        masked, mapping, pairs, first, keep, key = precomputed or self._unit_key(u)
        if not self.force:
            hit, cached_text = self.flights.read(key)
            if hit:
                # 캐시 적중은 API를 타지 않으므로 progressed를 세우지 않는다 —
                # 죽은 엔드포인트에서도 신규 유닛이 조용히 강등되는 것을 막는다.
                # (이전 run 캐시의 적중은 flights.prior_hits로 따로 계측돼
                #  _verify_endpoint_health의 재개 판정에 쓰인다.)
                return u, cached_text, "cached", key, stats
        ctx = self.context_map.get(u.id) if self.cfg.context else None
        max_toks = self._max_toks(masked)
        prompt = prompts.build_unit_prompt(
            masked,
            pairs,
            first,
            context_tail=ctx,
            keep_terms=keep,
            unit_kind=u.kind,
        )

        # 0) 최초 패스 — complete→sanitize→unmask. 태그 완전하고 출력 검증을
        #    통과하면 즉시 성공.
        #    유닛 하나만 결정적으로 거부되는 재시도 불가 4xx(초대형 표·초장문이
        #    서버 n_ctx를 넘겨 400 등)는 잡 전체를 죽이는 대신 래더로 넘겨 강등한다.
        #    단 이 run에서 성공한 유닛이 아직 없으면 엔드포인트·설정 자체 문제일 수
        #    있으므로 강등 허용 여부는 _may_degrade가 판정한다.
        api_rejected = False
        try:
            restored, missing, dup, sc, clean = self._run_pass(prompt, max_toks, mapping)
        except TranslateUnitRejected as e:
            if not self._may_degrade(e):
                raise
            logger.warning("번역 유닛 최초 패스 API 거부 — 래더로 강등: %s", u.id)
            api_rejected = True
            restored, missing, dup, sc, clean = "", [], [], 0, ""
        stats["sanitized"] += sc
        if not api_rejected and self._accepted(u.src, restored, missing, dup, mapping):
            return u, restored, "translated", key, stats

        # 여기부터 신뢰도 래더 — 태그 누락·중복, 빈 출력, 출력 검증 실패, API 거부
        stats["retried"] = 1
        # 래더가 끝내 실패해 원문 유지로 떨어질 때 보고할 사유(최초 패스 기준).
        if api_rejected:
            stats["kept_reason"] = "api-rejected"
        elif missing or dup:
            stats["kept_reason"] = "placeholder-mismatch"
        elif not restored.strip():
            stats["kept_reason"] = "empty-output"
        else:
            stats["kept_reason"] = "gate-rejected"
        if self._halted():
            return u, u.src, "canceled", key, stats

        # 1) repair 패스 — 원문(태그 포함)+깨진 번역문을 주고 태그만 바로잡게 한다.
        #    API 거부는 고칠 깨진 출력 자체가 없으므로 건너뛰고 바로 분할로 간다.
        if not api_rejected:
            try:
                rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
                r_restored, r_missing, r_dup, r_sc, _ = self._run_pass(rprompt, max_toks, mapping)
                stats["sanitized"] += r_sc
                if self._accepted(u.src, r_restored, r_missing, r_dup, mapping):
                    stats["repaired"] = 1
                    return u, r_restored, "translated", key, stats
            except TranslateError:
                pass
        if self._halted():
            return u, u.src, "canceled", key, stats

        # 2a) HTML 표 유닛 — </tr> 행 경계 분할 (깊이 2 = 최대 4분할).
        #     초대형 표는 반쪽도 실패할 수 있어 재귀 한 단계를 더 허용한다.
        if _is_table_unit(u.src):
            def _table_part(src: str, depth: int) -> str | None:
                got = self._translate_fragment(
                    src, pairs, first, None, stats, keep, u.kind,
                )
                if got is not None or depth <= 0:
                    return got
                sub = _split_table(src)
                if sub is None:
                    return None
                a = _table_part(sub[0], depth - 1)
                if a is None:
                    return None
                b = _table_part(sub[1], depth - 1)
                return None if b is None else a + b

            ts = _split_table(u.src)
            if ts is not None:
                left = _table_part(ts[0], 1)
                right = _table_part(ts[1], 1) if left is not None else None
                if left is not None and right is not None:
                    stats["split"] = 1
                    return u, left + right, "translated", key, stats

        # 2b) 문장 분할 재시도(깊이 1) — 여러 문장·비구조 유닛만. 양쪽 성공 시 " "로 결합.
        elif _splittable(u.src):
            halves = _split_two(u.src)
            if halves is not None:
                left_src, right_src = halves
                left = self._translate_fragment(
                    left_src, pairs, first, ctx, stats, keep, u.kind,
                )
                if left is not None:
                    # 뒷반 컨텍스트: 앞반 src의 꼬리 200자 (컨텍스트 비활성 시 생략)
                    right_ctx = left_src[-200:] if self.cfg.context else None
                    right = self._translate_fragment(
                        right_src, pairs, first, right_ctx, stats, keep, u.kind,
                    )
                    if right is not None:
                        stats["split"] = 1
                        return u, left + " " + right, "translated", key, stats

        # 3) 최종 실패 → 원문 유지 (무손실 원칙 불변). 단, 래더 실패가 취소로 인한
        #    조기 반환(None) 때문이면 kept_original 통계를 오염시키지 않는다.
        if self._halted():
            return u, u.src, "canceled", key, stats
        return u, u.src, "kept", key, stats

    def translate_unit_shared(self, u):
        """중복 유닛 single-flight 래퍼. 선점 스레드가 번역을 끝내면 그 결과를
        같이 쓰고(=cached), 선점 스레드가 실패·취소로 끝나면 오늘과 동일하게
        같은 outcome/예외를 공유한다. force 경로는 캐시를 쓰지 않으므로 제외한다."""
        if self.force:
            return self.translate_unit(u)
        if self._halted():
            return u, u.src, "canceled", "", _zero_stats()
        precomputed = self._unit_key(u)
        key = precomputed[5]
        hit, cached_text = self.flights.read(key)
        if hit:
            # translate_unit과 동일 — 캐시 적중은 엔드포인트 건강의 증거가 아니다.
            return u, cached_text, "cached", key, _zero_stats()
        flight, owner = self.flights.acquire(key)
        if not owner:
            if not flight.wait_for(self._halted):
                return u, u.src, "canceled", key, _zero_stats()
            if flight.error is not None:
                flight.reraise()
            hit, cached_text = self.flights.read(key)
            if hit:
                return u, cached_text, "cached", key, _zero_stats()
            if flight.result is not None:
                _owner_u, text, status, result_key, _stats = flight.result
                if status == "kept":
                    # 재시도 통계는 0(추가 왕복 없음)이지만 사유는 선점 결과를 공유한다.
                    shared = _zero_stats()
                    shared["kept_reason"] = (_stats or {}).get("kept_reason", "")
                    return u, u.src, "kept", result_key, shared
                if status == "canceled":
                    return u, u.src, "canceled", result_key, _zero_stats()
                return u, text, "cached", result_key, _zero_stats()
            # 방어 경로 — event는 result/error 중 하나를 공개한 뒤에만 set한다.
            raise RuntimeError("동일 번역 유닛의 공유 결과가 없습니다")
        try:
            result = self.translate_unit(u, precomputed)
            if result[2] == "translated":
                self.flights.publish(result[3], result[1])  # 대기 중인 중복에게 즉시 공개
            flight.result = result
            return result
        except BaseException as exc:
            flight.capture(exc)
            raise
        finally:
            flight.event.set()

    # ── 디스패치·후처리 ──────────────────────────────────────────────────────

    def _canceled_result(self) -> TranslateResult:
        self.write_state("canceled", self.done, self.total)
        return TranslateResult(
            status="canceled", total=self.total, translated=self.translated_n, cached=self.cached_n,
            kept_original=self.kept_original, skipped=self.skipped,
            api_mode=getattr(self.client, "api_mode_used", "") or self.cfg.api_mode,
        )

    def _dispatch(self, units: list) -> bool:
        """유닛 목록을 병렬 번역해 results·통계에 반영. 취소를 만나면 False.

        2단 패스(1차 targets → reconcile 폴백 시 deferred)가 같은 코드를 쓴다.
        """
        stopped = False
        with cf.ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
            futures = {}
            for u in units:
                if self._cancel_set():
                    stopped = True
                    break
                futures[ex.submit(self.translate_unit_shared, u)] = u

            if not stopped:
                for fut in cf.as_completed(futures):
                    if self._cancel_set():
                        stopped = True
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        u, text, status, key, stats = fut.result()
                    except BaseException:
                        # 유닛 오류(step-0 API 오류 등) 전파 — 남은 futures를 취소하고
                        # abort를 알린다. 이것 없이는 executor 종료(shutdown wait=True)가
                        # 남은 큐 전체를 드레인해(죽은 엔드포인트 × 유닛별 백오프) 수 분
                        # 뒤에야 error 상태가 기록된다. 원래 예외는 그대로 재전파.
                        self.abort.set()
                        for f in futures:
                            f.cancel()
                        raise
                    if status == "canceled":
                        # 유닛이 래더 도중 취소를 감지하고 조기 반환 — 결과 미반영
                        stopped = True
                        for f in futures:
                            f.cancel()
                        break
                    self.results[u.id] = text
                    self.retried_n += stats["retried"]
                    self.repaired_n += stats["repaired"]
                    self.split_n += stats["split"]
                    self.sanitized_n += stats["sanitized"]
                    if status == "cached":
                        self.cached_n += 1
                    elif status == "translated":
                        self.translated_n += 1
                        # non-force owner는 waiter 공개 전에 이미 캐시에 썼다.
                        # force 경로만 shared wrapper를 우회하므로 여기서 저장한다.
                        if self.force:
                            self.flights.publish(key, text)
                    elif status == "kept":
                        self.kept_original.append(u.id)
                        kr = stats.get("kept_reason") or "unknown"
                        self.kept_reasons[kr] = self.kept_reasons.get(kr, 0) + 1
                        # 식별자만 기록 — 문서 원문·번역문 내용은 로그에 남기지 않는다
                        logger.warning("번역 유닛 원문 유지: %s (lang=%s, unit=%s)", self.job_dir.name, self.lang, u.id)
                    self.done += 1
                    if self.progress is not None:
                        self.progress(self.done, self.total)
                    if self.done % 10 == 0:
                        self.flush_cache()
                        # SSE 없이 state 폴링으로 보는 클라이언트(프런트 폴백)용 진행률
                        self.write_state("running", self.done, self.total)
        return not stopped

    def _verify_endpoint_health(self) -> None:
        """미뤄둔 4xx 강등을 확정하거나, 엔드포인트 고장으로 보고 잡을 실패시킨다.

        _may_degrade가 미룬 건은 "이번 run의 API 성공이 0"인 상태에서 강등된 것들이다.
        dispatch가 끝난 지금은 이전 캐시가 실제로 적중했는지(prior_hits)를 순서에
        무관하게 알 수 있다.
        - 적중 있음: 같은 모델·프롬프트·샘플링(캐시 키 재료)으로 만든 번역을 이번 run이
          재사용했다 = 설정 자체는 유효하다. 강등을 확정하되, API 성공이 0건이라는
          사실은 경고로 남긴다(엔드포인트가 이제 막 죽었을 수도 있다).
        - 적중 없음: 모델명 오타처럼 키가 전부 달라졌거나 처음부터 죽은 엔드포인트다.
          전 유닛이 kept로 조용히 done 되지 않도록 잡 전체를 실패시킨다.
        """
        if not self.unverified_rejects or self.progressed.is_set():
            return
        n, msg = self.unverified_rejects, self.unverified_msg
        self.unverified_rejects = 0  # 2차 패스에서 다시 판정 (경고 중복 방지)
        if not self.flights.prior_hits:
            raise TranslateAPIError(
                f"{msg} — 이번 실행에서 성공한 API 호출이 하나도 없습니다"
                f" (유닛 {n}개 연속 거부). 모델명·엔드포인트 설정을 확인하세요."
            )
        warn = (
            f"이번 실행에서 성공한 API 호출이 없습니다 — 신규 유닛 {n}개가 API 거부로"
            " 원문 유지되었고 나머지는 기존 캐시를 재사용했습니다."
            " 엔드포인트·모델 설정이 여전히 유효한지 확인하세요."
        )
        self.warnings.append(warn)
        logger.warning("번역 API 성공 0건: %s (lang=%s) — %s", self.job_dir.name, self.lang, warn)

    def _sweep_degenerate(self) -> None:
        # ── 문서 단위 축퇴(degenerate) 출력 방어 ──────────────────────────────
        # 유닛별 검증(looks_untranslated)은 "이 출력이 이 원문의 번역인가"만 본다.
        # 공급자가 고장 나 **모든 유닛에 같은 문자열**을 돌려주는 실패 모드(캔드 응답·
        # 루프·잘못 설정된 게이트웨이)는 유닛 하나만 보면 정상처럼 보여 통과한다.
        # 서로 다른 원문 여럿이 한 출력으로 수렴하면 그건 번역이 아니므로 원문을 지킨다.
        # 원문은 **전체 유닛**에서 찾는다. targets는 deferred 선별로 줄어들어 있어
        # 여기서 만들면 조회 실패분이 서로 다른 원문처럼 세어져 오탐이 난다.
        results = self.results
        src_by_id = {u.id: u.src for u in (*self.md_units, *self.lay_units)}
        by_output: dict[str, set[str]] = {}
        for uid, text in results.items():
            norm = " ".join(text.split())
            src = src_by_id.get(uid)
            if norm and src is not None:
                by_output.setdefault(norm, set()).add(src)
        for norm, srcs in by_output.items():
            # 서로 다른 원문 3개 이상이 같은 출력 → 축퇴. 원문이 실제로 같은 유닛
            # (반복되는 표 헤더 등)이 같은 번역을 받는 것은 정상이므로 원문 기준으로 센다.
            if len(srcs) < 3:
                continue
            degenerate = [uid for uid, text in results.items() if " ".join(text.split()) == norm]
            for uid in degenerate:
                results.pop(uid, None)
                if uid not in self.kept_original:
                    self.kept_original.append(uid)
                    self.kept_reasons["degenerate-output"] = self.kept_reasons.get("degenerate-output", 0) + 1
                    self.translated_n = max(0, self.translated_n - 1)
            # 캐시도 함께 비운다 — 축퇴 출력이 units.json에 남으면 공급자를 고친 뒤
            # force 없이 재실행해도 같은 손실이 그대로 재사용된다.
            self.flights.purge(lambda v, _norm=norm: " ".join(v.split()) == _norm)
            logger.warning(
                "번역 축퇴 출력 감지: %s (lang=%s, 원문 %d종이 동일 출력 → 원문 유지)",
                self.job_dir.name, self.lang, len(srcs),
            )
        self.flush_cache()

    # 조립 — 번역된 유닛만 교체(나머지 원문 보존)
    def _assemble_md(self) -> str:
        md_trans = {u.id: self.results[u.id] for u in self.md_units if u.id in self.results}
        return assemble_markdown(self.md_text, self.page_separator, md_trans)

    # ── 실행 단계 ────────────────────────────────────────────────────────────

    def _load_source(self) -> None:
        # POST 직후 SSE 접속(404 방지)을 위해 유닛 집계 전이라도 즉시 running을 남긴다.
        self.write_state("running", 0, 0)
        result_md = self.job_dir / "result.md"
        if not result_md.is_file():
            raise TranslateError("번역할 결과가 없습니다 — 변환이 완료된 잡인지 확인하세요")
        self.md_text = result_md.read_text(encoding="utf-8")

        layout_path = self.job_dir / "layout.json"
        self.layout_pages = None
        if layout_path.is_file():
            try:
                loaded = json.loads(layout_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self.layout_pages = loaded
            except Exception:
                self.layout_pages = None

    def _select_units(self) -> None:
        # 유닛 분리 — md(문서 순서) + layout
        self.md_units = split_markdown(self.md_text, self.page_separator)
        self.lay_units = layout_units(self.layout_pages) if self.layout_pages else []
        all_units = self.md_units + self.lay_units

        targets = []
        self.skipped = 0
        for u in all_units:
            # 유닛 자체 사유(references)가 우선, 없으면 게이트 판정 사유를 그대로 쓴다.
            reason = u.skip_reason or should_skip(u.src)
            if reason:
                self.skipped += 1
                self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
            else:
                targets.append(u)
        self.targets = targets
        self.total = len(targets)

        # md(heading 스윕)와 layout(ref_text) 참고문헌 규칙의 불일치 관측 — 같은 영역이
        # result.{lang}.md와 PDF에서 서로 다르게 처리되는 사례를 리포트로 드러낸다.
        self.ref_rule = reference_rule_mismatch(self.md_units, self.lay_units)

        # 2단 패스 — reconcile이 성공하면 md 유닛 번역은 전량 폐기되고 layout 번역이
        # 단일 기준이 된다(LLM 왕복의 절반이 낭비). 그래서 **모든 줄이 layout 블록에
        # 커버되는** md 유닛만 1차에서 빼두고(deferred), reconcile이 폴백을 돌려준
        # 경우에만 2차로 번역해 무손실 계약을 지킨다. 부분만 걸치는 md 유닛(다중 줄
        # 블록·표·수식 줄)은 매핑에 안 걸려 원문이 남으므로 지금처럼 1차에서 번역한다.
        self.deferred = []
        if self.layout_pages is not None and self.lay_units:
            lay_target_srcs = {u.src.strip() for u in self.targets if u.id.startswith("lay:")}
            covered = layout_line_sources(self.layout_pages) & lay_target_srcs
            if covered:
                remaining = []
                for u in self.targets:
                    if u.id.startswith("md:") and _fully_covered(u.src, covered):
                        self.deferred.append(u)
                    else:
                        remaining.append(u)
                self.targets = remaining
                self.total = len(remaining)

        self.context_map = {}
        if self.cfg.context:
            for seq in (self.md_units, self.lay_units):
                prev = None
                for u in seq:
                    if prev is not None:
                        self.context_map[u.id] = prev.src[-200:]
                    prev = u

    def _prepare_run(self) -> None:
        cfg = self.cfg
        if self.client is None:
            self.client = OpenAICompatClient(cfg)
        # OpenAICompatClient는 전역 HTTP 슬롯 대기·재시도 backoff·auto 협상 중에도
        # 사용자 cancel과 내부 abort를 확인한다. 테스트/사용자 정의 client에는 이
        # 선택 API가 없어도 기존 프로토콜(complete)만으로 그대로 동작한다.
        set_cancel_check = getattr(self.client, "set_cancel_check", None)
        if callable(set_cancel_check):
            set_cancel_check(self._halted)
        self.write_state("running", 0, self.total)
        logger.info(
            "번역 시작: %s (lang=%s, 유닛 %d개, 건너뜀 %d)",
            self.job_dir.name, self.lang, self.total, self.skipped,
        )

        # 용어집 — 있으면 로드(캐시 안정), 없거나 force면 빌드 후 저장
        gpath = self.tdir / "glossary.json"
        if gpath.is_file() and not self.force:
            glossary = Glossary.load(gpath)
        else:
            glossary = build_glossary(self.md_text, self.md_units, self.client, cfg)
            glossary.save(gpath)
        self.glossary = glossary

        # 유닛 캐시 (dict: cache_key → 번역문)
        cache: dict[str, str] = {}
        if self.upath.is_file():
            try:
                loaded = json.loads(self.upath.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cache = loaded
            except Exception:
                cache = {}
        self.flights = SingleFlight(cache)

        self.warnings = list(glossary.warnings)
        ref_rule = self.ref_rule
        if ref_rule.get("md_only") or ref_rule.get("layout_only"):
            self.warnings.append(
                "참고문헌 규칙 불일치 — 같은 원문이 한쪽 경로에서만 원문 유지됩니다"
                f" (md만 유지 {ref_rule.get('md_only', 0)}건,"
                f" layout(PDF)만 유지 {ref_rule.get('layout_only', 0)}건)"
            )
            logger.info(
                "참고문헌 규칙 불일치: %s (lang=%s, md_only=%d, layout_only=%d)",
                self.job_dir.name, self.lang, ref_rule.get("md_only", 0), ref_rule.get("layout_only", 0),
            )
        self.results = {}
        self.kept_original = []
        self.translated_n = 0
        self.cached_n = 0
        self.retried_n = 0
        self.repaired_n = 0
        self.split_n = 0
        self.sanitized_n = 0

    def _write_outputs(self) -> TranslateResult | None:
        """조립·reconcile·2차 패스 후 산출물 기록. 2차 패스에서 취소되면 결과를 돌려준다."""
        assembled = self._assemble_md()
        new_pages = None
        if self.layout_pages is not None:
            lay_trans = {u.id: self.results[u.id] for u in self.lay_units if u.id in self.results}
            new_pages = apply_layout(self.layout_pages, lay_trans)
            reconciled = reconcile_markdown_with_layout(
                self.md_text,
                assembled,
                self.layout_pages,
                new_pages,
                self.page_separator,
            )
            # 폴백이면 reconcile은 인자로 받은 assembled를 그대로 돌려준다.
            if reconciled is assembled and self.deferred:
                # layout 번역이 md를 덮지 못했다 — 1차에서 미룬 md 유닛을 지금 번역해
                # 무손실 계약을 지킨다(2단 패스). SSE는 total 증가를 허용한다.
                self.total += len(self.deferred)
                self.write_state("running", self.done, self.total)
                logger.info("reconcile 폴백 — 지연 md 유닛 %d개 2차 번역", len(self.deferred))
                canceled = False
                try:
                    canceled = not self._dispatch(self.deferred)
                    if not canceled:
                        # 2차 패스 유닛도 같은 축퇴 방어를 받아야 한다 —
                        # 1차 스윕은 이 dispatch 이전에 끝났다.
                        self._sweep_degenerate()
                finally:
                    self.flush_cache()
                if canceled:
                    return self._canceled_result()
                self._verify_endpoint_health()  # 2차 패스의 미뤄둔 4xx도 같은 기준으로 확정
                assembled = self._assemble_md()
            else:
                assembled = reconciled
            _atomic_write(
                self.job_dir / f"layout.{self.lang}.json",
                json.dumps(new_pages, ensure_ascii=False),
            )
        _atomic_write(self.job_dir / f"result.{self.lang}.md", assembled)
        return None

    def _finish(self) -> TranslateResult:
        cfg = self.cfg
        prior_cache_n = self.flights.prior_count
        prior_hits = self.flights.prior_hits
        # 캐시 전량 무효화 안내 — PROMPT_V·모델·샘플링을 바꾸면 units.json 키가 전부
        # 달라져 기존 잡이 조용히 전량 재번역된다(그만큼 API 비용·시간이 든다).
        # 종전에는 이 사실이 어디에도 남지 않아 운영자가 청구서를 보고서야 알았다.
        if prior_cache_n and not prior_hits and not self.force and self.total:
            msg = (
                f"기존 캐시 {prior_cache_n}건이 하나도 적중하지 않아 유닛 {self.total}개를 "
                f"전량 재번역했습니다 — 프롬프트 버전(PROMPT_V={PROMPT_V})·모델"
                f"({cfg.model})·샘플링 설정 중 하나가 바뀌면 units.json 키가 전부 달라집니다. "
                "업그레이드 직후라면 정상이며, 잡마다 1회만 발생합니다."
            )
            self.warnings.append(msg)
            logger.warning("번역 캐시 전량 무효: %s (lang=%s) — %s", self.job_dir.name, self.lang, msg)

        api_mode = getattr(self.client, "api_mode_used", "") or cfg.api_mode
        _atomic_write_json(self.tdir / "report.json", {
            "kept_original": self.kept_original,
            "retried": self.retried_n,
            "repaired": self.repaired_n,
            "split": self.split_n,
            "sanitized": self.sanitized_n,
            "skipped": self.skipped,
            # 사유별 내역 — 총합(skipped/kept_original)만으로는 "왜 영어로 남았나"를
            # 구분할 수 없어 사용자·프런트가 원인을 알 방법이 없었다.
            "skip_reasons": self.skip_reasons,
            "kept_reasons": self.kept_reasons,
            # 출력 게이트가 규칙별로 몇 번 거부했나 — 래더가 흡수한 건까지 포함한다.
            # gate_reasons 합 ≫ kept_reasons["gate-rejected"]이면 오탐이 비용만
            # 태우고 있다는 신호다(임계값 회귀 감시 지표).
            "gate_reasons": self.gate_reasons,
            # 이번 run 시작 시점의 units.json 항목 수와 그중 실제 재사용된 건수 —
            # cache_prior>0인데 cache_reused==0이면 PROMPT_V·모델·샘플링 변경으로
            # 캐시가 전량 무효화돼 전량 재번역(비용)이 발생한 것이다.
            "cache_prior": prior_cache_n,
            "cache_reused": len(prior_hits),
            "reference_rule": self.ref_rule,
            "cached": self.cached_n,
            "translated": self.translated_n,
            "api_mode": api_mode,
            "warnings": self.warnings,
        })
        self.write_state("done", self.total, self.total)
        logger.info(
            "번역 완료: %s (lang=%s, 번역 %d·캐시 %d·원문유지 %d)",
            self.job_dir.name, self.lang, self.translated_n, self.cached_n, len(self.kept_original),
        )
        return TranslateResult(
            status="done", total=self.total, translated=self.translated_n, cached=self.cached_n,
            kept_original=self.kept_original, skipped=self.skipped, api_mode=api_mode,
        )

    def execute(self) -> TranslateResult:
        self._load_source()
        self._select_units()
        self._prepare_run()

        # 취소: 디스패치 전 선체크
        if self._cancel_set():
            self.flush_cache()
            return self._canceled_result()

        canceled = False
        try:
            canceled = not self._dispatch(self.targets)
        finally:
            # 정상·오류·취소 모든 경로에서 부분 캐시 보존 — 오류 전파 시에도 마지막
            # 주기 flush(10유닛) 이후 완료된 유닛의 번역이 유실되지 않게 한다.
            self.flush_cache()

        if canceled:
            return self._canceled_result()

        # 산출물을 쓰기 전에 판정한다 — 실패로 결론나면 부분 번역이 result.{lang}.md로
        # 나가지 않고 state는 error가 된다(캐시는 위 finally에서 이미 보존됐다).
        self._verify_endpoint_health()

        self._sweep_degenerate()

        early = self._write_outputs()
        if early is not None:
            return early
        return self._finish()


def run_translation(
    job_dir: Path,
    lang: str,
    cfg: TranslateConfig,
    *,
    page_separator: str = "\n\n---\n\n",
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    force: bool = False,  # True면 유닛 캐시를 읽지 않고 전부 재번역 (쓰기는 함)
    client=None,  # 테스트 주입용 — None이면 cfg로 OpenAICompatClient 생성
) -> TranslateResult:
    """job_dir의 result.md(+ 있으면 layout.json)를 lang으로 번역한다.

    산출물·캐시·상태 파일 계약은 types.py 모듈 docstring 참조.
    실패 시 TranslateError(사용자 표시용 한국어 메시지)를 던지고
    state.json에 error를 남긴다. 취소 시 부분 캐시는 보존된다.
    """
    run = _TranslationRun(
        job_dir=Path(job_dir),
        lang=lang,
        cfg=cfg,
        page_separator=page_separator,
        progress=progress,
        cancel=cancel,
        force=force,
        client=client,
    )
    try:
        return run.execute()
    except TranslateError as e:
        # 전역 HTTP 슬롯/재시도 대기에서 cancel을 관찰한 client도 TranslateError
        # 계열로 빠져나온다. 사용자가 이미 취소한 잡을 error로 덮어쓰지 않는다.
        if run._cancel_set():
            return run._canceled_result()
        run.write_state("error", run.done, run.total, error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 — 사용자용 메시지로 감싸 재발생
        logger.exception("번역 중 오류: %s (lang=%s)", run.job_dir.name, lang)
        run.write_state("error", run.done, run.total, error=f"번역 중 오류: {e}")
        raise TranslateError(f"번역 중 오류: {e}") from e
