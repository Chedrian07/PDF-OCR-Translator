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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import re

from . import prompts
from .client import OpenAICompatClient
from .glossary import Glossary, build_glossary
from .masking import (
    _TOKEN_RE,
    looks_untranslated,
    mask,
    sanitize_translation,
    should_skip,
    unmask,
)
from .segment import (
    apply_layout,
    assemble_markdown,
    layout_line_sources,
    layout_units,
    reconcile_markdown_with_layout,
    split_markdown,
)
from .types import (
    PROMPT_V,
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
    job_dir = Path(job_dir)
    tdir = job_dir / "translations" / lang
    tdir.mkdir(parents=True, exist_ok=True)  # error 상태도 기록할 수 있도록 선행 생성
    started = _now()
    total = 0
    done = 0
    skipped = 0
    translated_n = 0
    cached_n = 0
    kept_original: list[str] = []

    def write_state(status: str, current: int, total_: int, error: str | None = None) -> None:
        mode = getattr(client, "api_mode_used", "") or (cfg.api_mode if cfg.api_mode != "auto" else "")
        try:
            _atomic_write_json(tdir / "state.json", {
                "lang": lang,
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
                "started_at": started,
                "finished_at": _now() if status in ("done", "error", "canceled") else None,
            })
        except FileNotFoundError:
            pass  # 잡 삭제 경합(DELETE가 디렉터리 제거) — 상태 기록은 best-effort

    def _cancel_set() -> bool:
        return cancel is not None and cancel.is_set()

    # 내부 abort — 유닛의 치명 오류(step-0 API 오류)가 전파되기 시작하면 set된다.
    # 이미 실행을 시작한 유닛이 다음 API 호출 전에 빠져나오게 해, 죽은 엔드포인트로
    # 남은 큐가 드레인되는 것(유닛 수 × 백오프)을 막는다. 사용자 취소(cancel)와 별개.
    abort = threading.Event()

    # 이 run에서 유닛 하나라도 성공했는가 = 엔드포인트·인증·설정은 정상이라는 신호.
    # step-0의 결정적 4xx를 유닛 강등으로 흡수해도 되는지 판단하는 데 쓴다.
    progressed = threading.Event()

    def _halted() -> bool:
        return abort.is_set() or _cancel_set()

    try:
        # POST 직후 SSE 접속(404 방지)을 위해 유닛 집계 전이라도 즉시 running을 남긴다.
        write_state("running", 0, 0)
        result_md = job_dir / "result.md"
        if not result_md.is_file():
            raise TranslateError("번역할 결과가 없습니다 — 변환이 완료된 잡인지 확인하세요")
        md_text = result_md.read_text(encoding="utf-8")

        layout_path = job_dir / "layout.json"
        layout_pages = None
        if layout_path.is_file():
            try:
                loaded = json.loads(layout_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    layout_pages = loaded
            except Exception:
                layout_pages = None

        # 유닛 분리 — md(문서 순서) + layout
        md_units = split_markdown(md_text, page_separator)
        lay_units = layout_units(layout_pages) if layout_pages else []
        all_units = md_units + lay_units

        targets = []
        skipped = 0
        for u in all_units:
            if u.skip_reason or should_skip(u.src):
                skipped += 1
            else:
                targets.append(u)
        total = len(targets)

        # 2단 패스 — reconcile이 성공하면 md 유닛 번역은 전량 폐기되고 layout 번역이
        # 단일 기준이 된다(LLM 왕복의 절반이 낭비). 그래서 **모든 줄이 layout 블록에
        # 커버되는** md 유닛만 1차에서 빼두고(deferred), reconcile이 폴백을 돌려준
        # 경우에만 2차로 번역해 무손실 계약을 지킨다. 부분만 걸치는 md 유닛(다중 줄
        # 블록·표·수식 줄)은 매핑에 안 걸려 원문이 남으므로 지금처럼 1차에서 번역한다.
        deferred: list = []
        if layout_pages is not None and lay_units:
            lay_target_srcs = {u.src.strip() for u in targets if u.id.startswith("lay:")}
            covered = layout_line_sources(layout_pages) & lay_target_srcs
            if covered:
                remaining = []
                for u in targets:
                    if u.id.startswith("md:") and _fully_covered(u.src, covered):
                        deferred.append(u)
                    else:
                        remaining.append(u)
                targets = remaining
                total = len(targets)

        # 직전 유닛 꼬리 컨텍스트 (같은 소스 내에서만). 실제 프롬프트 입력이므로
        # 캐시 키에도 넣어 같은 문장이 다른 문맥의 번역을 잘못 공유하지 않게 한다.
        context_map: dict[str, str] = {}
        if cfg.context:
            for seq in (md_units, lay_units):
                prev = None
                for u in seq:
                    if prev is not None:
                        context_map[u.id] = prev.src[-200:]
                    prev = u

        if client is None:
            client = OpenAICompatClient(cfg)
        # OpenAICompatClient는 전역 HTTP 슬롯 대기·재시도 backoff·auto 협상 중에도
        # 사용자 cancel과 내부 abort를 확인한다. 테스트/사용자 정의 client에는 이
        # 선택 API가 없어도 기존 프로토콜(complete)만으로 그대로 동작한다.
        set_cancel_check = getattr(client, "set_cancel_check", None)
        if callable(set_cancel_check):
            set_cancel_check(_halted)
        write_state("running", 0, total)
        logger.info("번역 시작: %s (lang=%s, 유닛 %d개, 건너뜀 %d)", job_dir.name, lang, total, skipped)

        # 용어집 — 있으면 로드(캐시 안정), 없거나 force면 빌드 후 저장
        gpath = tdir / "glossary.json"
        if gpath.is_file() and not force:
            glossary = Glossary.load(gpath)
        else:
            glossary = build_glossary(md_text, md_units, client, cfg)
            glossary.save(gpath)

        # 유닛 캐시 (dict: cache_key → 번역문)
        upath = tdir / "units.json"
        cache: dict[str, str] = {}
        if upath.is_file():
            try:
                loaded = json.loads(upath.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cache = loaded
            except Exception:
                cache = {}

        # 같은 캐시 키를 두 스레드가 동시에 API로 보내지 않기 위한 single-flight.
        # 지금까지 중복 유닛 제거는 "먼저 끝난 유닛이 메인 루프에서 캐시에 기록되고,
        # 늦게 시작한 중복이 그걸 읽는" 레이스에 기대고 있었다 — 동시성을 올릴수록
        # 그 레이스가 덜 맞아 같은 문단을 두 번 번역하게 된다(문서별 중복률 4~16% 실측).
        class _Flight:
            __slots__ = ("event", "result", "error")

            def __init__(self) -> None:
                self.event = threading.Event()
                self.result = None
                self.error: tuple[type[BaseException], tuple] | None = None

        # 완료 flight도 이 run 동안 보존한다. translated는 영속 cache로, kept는
        # 메모리 outcome으로 후속 중복에 재사용해 실패한 같은 문단을 다시 두드리지 않는다.
        inflight: dict[str, _Flight] = {}
        inflight_lock = threading.Lock()
        cache_lock = threading.Lock()

        def read_cache(key: str) -> tuple[bool, str]:
            """공유 캐시의 단일 키를 원자적으로 읽는다 (빈 문자열도 값으로 구분)."""
            with cache_lock:
                if key in cache:
                    return True, cache[key]
            return False, ""

        def publish_cache(key: str, value: str) -> None:
            with cache_lock:
                cache[key] = value

        def flush_cache() -> None:
            # worker가 single-flight 결과를 공개하는 동안 json.dumps(cache)가 dict를
            # 순회하면 "dictionary changed size"로 잡 전체가 실패할 수 있다. 잠금은
            # 메모리 스냅샷까지만 잡고 디스크 I/O 동안에는 worker를 막지 않는다.
            with cache_lock:
                snapshot = dict(cache)
            try:
                _atomic_write_json(upath, snapshot)
            except FileNotFoundError:
                pass  # 잡 삭제 경합 — 캐시는 잡과 함께 사라진다

        warnings = list(glossary.warnings)
        results: dict[str, str] = {}
        kept_original: list[str] = []
        translated_n = 0
        cached_n = 0
        retried_n = 0    # 최초 패스 실패로 래더(repair/분할)에 진입한 유닛 수
        repaired_n = 0   # repair 패스로 복구된 유닛 수
        split_n = 0      # 문장 분할로 복구된 유닛 수
        sanitized_n = 0  # sanitize 치환 총건수 (모든 complete 출력 경로 합산)

        def _max_toks(masked: str) -> int:
            # reasoning effort별 고정 예산 (types.REASONING_MAX_TOKENS, 사용자 확정) —
            # thinking 토큰이 같은 예산에서 차감되므로 길이 공식 대신 모드 상한을 그대로 쓴다.
            # 미사용 토큰은 과금되지 않고, 상한은 폭주 방지용이다.
            return cfg.max_output_tokens

        def _run_pass(prompt: str, max_toks: int, mapping: dict) -> tuple[str, list, list, int, str]:
            """complete → sanitize → unmask 한 번. (복원문, missing, dup, 치환건수, 정리된_원출력)."""
            raw = client.complete(prompts.SYSTEM_TRANSLATE, prompt, max_tokens=max_toks)
            # API 왕복 하나가 성공했다 = 엔드포인트·인증·설정은 정상. step-0의 결정적
            # 4xx를 유닛 단위로 강등해도 되는지 판단하는 신호(워커 안에서 즉시 세운다).
            progressed.set()
            clean, sc = sanitize_translation(raw)
            restored, missing, dup = unmask(clean, mapping)
            return restored, missing, dup, sc, clean

        def _accepted(src: str, restored: str, missing: list, dup: list, mapping: dict) -> bool:
            """유닛 출력 채택 판정 — 플레이스홀더 정합 + 비어있지 않음 + 출력 측 검증.

            거부문·요약·영문 echo는 플레이스홀더가 온전해도 통과시키지 않는다.
            거절된 출력은 기존 래더(repair→분할)가 흡수하고, 다 소진되면 "kept"로
            떨어져 report.json에 남는다. publish_cache는 status=="translated"일
            때만 호출되므로 units.json 오염도 함께 차단된다.
            """
            if missing or dup or not restored.strip():
                return False
            return not looks_untranslated(src, restored, mapping)

        def _translate_fragment(
            src, pairs, first, ctx, stats, keep=None, unit_kind: str = "",
        ) -> str | None:
            """분할된 반쪽 하나를 독립 번역 — mask→complete→sanitize→unmask + repair 1회(추가 분할 없음).

            성공 시 복원문, 실패 시 None. sanitize 건수만 stats에 누적한다. 반쪽 단계의
            API 오류는 무손실 원칙상 치명적이지 않으므로 None 처리(원 유닛 원문 유지로 귀결).
            취소·abort는 API 호출 사이에서 확인한다 — 거대 표 래더가 취소 후에도 수 분간
            호출을 이어가지 않게(응답성)."""
            if _halted():
                return None
            masked, mapping = mask(src)
            max_toks = _max_toks(masked)
            prompt = prompts.build_unit_prompt(
                masked,
                pairs,
                first,
                context_tail=ctx,
                keep_terms=keep,
                unit_kind=unit_kind,
            )
            try:
                restored, missing, dup, sc, clean = _run_pass(prompt, max_toks, mapping)
            except TranslateError:
                return None
            stats["sanitized"] += sc
            if _accepted(src, restored, missing, dup, mapping):
                return restored
            if _halted():
                return None
            try:
                rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
                r_restored, r_missing, r_dup, r_sc, _ = _run_pass(rprompt, max_toks, mapping)
                stats["sanitized"] += r_sc
                if _accepted(src, r_restored, r_missing, r_dup, mapping):
                    return r_restored
            except TranslateError:
                pass
            return None

        def _zero_stats() -> dict:
            return {"retried": 0, "repaired": 0, "split": 0, "sanitized": 0}

        def _unit_key(u):
            """유닛의 마스킹·용어집 파생값과 캐시 키 (유닛당 ~1.5ms — API 왕복의 0.04%)."""
            masked, mapping = mask(u.src)
            pairs, first = glossary.for_unit(u.src, u.id)
            keep = glossary.keep_terms(u.src)
            ctx = context_map.get(u.id) if cfg.context else None
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

        def translate_unit(u, precomputed=None):
            stats = _zero_stats()
            # 취소·abort 응답성: 이미 디스패치된 유닛도 API 호출·래더 단계 사이에서
            # 취소(사용자)·abort(치명 오류 전파)를 확인하고 "canceled" 센티널로 조기
            # 반환한다 (결과 미반영, kept 오염 없음).
            if _halted():
                return u, u.src, "canceled", "", stats
            masked, mapping, pairs, first, keep, key = precomputed or _unit_key(u)
            if not force:
                hit, cached_text = read_cache(key)
                if hit:
                    progressed.set()  # 캐시 적중도 "이 유닛은 이미 해결됨" 신호
                    return u, cached_text, "cached", key, stats
            ctx = context_map.get(u.id) if cfg.context else None
            max_toks = _max_toks(masked)
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
            #    단 이 run에서 성공한 유닛이 아직 없으면 엔드포인트·설정 자체 문제이므로
            #    종전대로 전파한다 — 전 유닛이 kept로 조용히 done 되는 회귀 방지.
            api_rejected = False
            try:
                restored, missing, dup, sc, clean = _run_pass(prompt, max_toks, mapping)
            except TranslateUnitRejected:
                if not progressed.is_set():
                    raise
                logger.warning("번역 유닛 최초 패스 API 거부 — 래더로 강등: %s", u.id)
                api_rejected = True
                restored, missing, dup, sc, clean = "", [], [], 0, ""
            stats["sanitized"] += sc
            if not api_rejected and _accepted(u.src, restored, missing, dup, mapping):
                return u, restored, "translated", key, stats

            # 여기부터 신뢰도 래더 — 태그 누락·중복, 빈 출력, 출력 검증 실패, API 거부
            stats["retried"] = 1
            if _halted():
                return u, u.src, "canceled", key, stats

            # 1) repair 패스 — 원문(태그 포함)+깨진 번역문을 주고 태그만 바로잡게 한다.
            #    API 거부는 고칠 깨진 출력 자체가 없으므로 건너뛰고 바로 분할로 간다.
            if not api_rejected:
                try:
                    rprompt = prompts.build_repair_prompt(masked, clean, missing + dup)
                    r_restored, r_missing, r_dup, r_sc, _ = _run_pass(rprompt, max_toks, mapping)
                    stats["sanitized"] += r_sc
                    if _accepted(u.src, r_restored, r_missing, r_dup, mapping):
                        stats["repaired"] = 1
                        return u, r_restored, "translated", key, stats
                except TranslateError:
                    pass
            if _halted():
                return u, u.src, "canceled", key, stats

            # 2a) HTML 표 유닛 — </tr> 행 경계 분할 (깊이 2 = 최대 4분할).
            #     초대형 표는 반쪽도 실패할 수 있어 재귀 한 단계를 더 허용한다.
            if _is_table_unit(u.src):
                def _table_part(src: str, depth: int) -> str | None:
                    got = _translate_fragment(
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
                    left = _translate_fragment(
                        left_src, pairs, first, ctx, stats, keep, u.kind,
                    )
                    if left is not None:
                        # 뒷반 컨텍스트: 앞반 src의 꼬리 200자 (컨텍스트 비활성 시 생략)
                        right_ctx = left_src[-200:] if cfg.context else None
                        right = _translate_fragment(
                            right_src, pairs, first, right_ctx, stats, keep, u.kind,
                        )
                        if right is not None:
                            stats["split"] = 1
                            return u, left + " " + right, "translated", key, stats

            # 3) 최종 실패 → 원문 유지 (무손실 원칙 불변). 단, 래더 실패가 취소로 인한
            #    조기 반환(None) 때문이면 kept_original 통계를 오염시키지 않는다.
            if _halted():
                return u, u.src, "canceled", key, stats
            return u, u.src, "kept", key, stats

        def translate_unit_shared(u):
            """중복 유닛 single-flight 래퍼. 선점 스레드가 번역을 끝내면 그 결과를
            같이 쓰고(=cached), 선점 스레드가 실패·취소로 끝나면 오늘과 동일하게
            같은 outcome/예외를 공유한다. force 경로는 캐시를 쓰지 않으므로 제외한다."""
            if force:
                return translate_unit(u)
            if _halted():
                return u, u.src, "canceled", "", _zero_stats()
            precomputed = _unit_key(u)
            key = precomputed[5]
            owner = False
            hit, cached_text = read_cache(key)
            if hit:
                progressed.set()
                return u, cached_text, "cached", key, _zero_stats()
            with inflight_lock:
                flight = inflight.get(key)
                if flight is None:
                    flight = inflight[key] = _Flight()
                    owner = True
            if not owner:
                # 선점 스레드 완료까지 대기 — 취소 응답성을 위해 0.5초씩 끊어 확인한다.
                # 한 유닛은 재시도·repair·분할로 cfg.timeout_s보다 오래 걸릴 수 있다.
                # 여기서 임의 deadline을 두면 정상 owner가 도는 중 같은 API를 중복 호출한다.
                # owner는 모든 종료 경로의 finally에서 event를 set하므로 별도 상한이 필요 없다.
                while not flight.event.wait(0.5):
                    if _halted():
                        return u, u.src, "canceled", key, _zero_stats()
                if flight.error is not None:
                    error_type, error_args = flight.error
                    try:
                        cloned_error = error_type(*error_args)
                    except TypeError as clone_error:
                        raise RuntimeError("동일 번역 유닛 처리 중 오류가 발생했습니다") from clone_error
                    raise cloned_error
                hit, cached_text = read_cache(key)
                if hit:
                    return u, cached_text, "cached", key, _zero_stats()
                if flight.result is not None:
                    _owner_u, text, status, result_key, _stats = flight.result
                    if status == "kept":
                        return u, u.src, "kept", result_key, _zero_stats()
                    if status == "canceled":
                        return u, u.src, "canceled", result_key, _zero_stats()
                    return u, text, "cached", result_key, _zero_stats()
                # 방어 경로 — event는 result/error 중 하나를 공개한 뒤에만 set한다.
                raise RuntimeError("동일 번역 유닛의 공유 결과가 없습니다")
            try:
                result = translate_unit(u, precomputed)
                if result[2] == "translated":
                    publish_cache(result[3], result[1])  # 대기 중인 중복에게 즉시 공개
                flight.result = result
                return result
            except BaseException as exc:
                flight.error = (type(exc), exc.args)
                raise
            finally:
                flight.event.set()

        def _canceled_result() -> TranslateResult:
            write_state("canceled", done, total)
            return TranslateResult(
                status="canceled", total=total, translated=translated_n, cached=cached_n,
                kept_original=kept_original, skipped=skipped,
                api_mode=getattr(client, "api_mode_used", "") or cfg.api_mode,
            )

        def _dispatch(units: list) -> bool:
            """유닛 목록을 병렬 번역해 results·통계에 반영. 취소를 만나면 False.

            2단 패스(1차 targets → reconcile 폴백 시 deferred)가 같은 코드를 쓴다.
            """
            nonlocal done, translated_n, cached_n, retried_n, repaired_n, split_n, sanitized_n
            stopped = False
            with cf.ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
                futures = {}
                for u in units:
                    if _cancel_set():
                        stopped = True
                        break
                    futures[ex.submit(translate_unit_shared, u)] = u

                if not stopped:
                    for fut in cf.as_completed(futures):
                        if _cancel_set():
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
                            abort.set()
                            for f in futures:
                                f.cancel()
                            raise
                        if status == "canceled":
                            # 유닛이 래더 도중 취소를 감지하고 조기 반환 — 결과 미반영
                            stopped = True
                            for f in futures:
                                f.cancel()
                            break
                        results[u.id] = text
                        retried_n += stats["retried"]
                        repaired_n += stats["repaired"]
                        split_n += stats["split"]
                        sanitized_n += stats["sanitized"]
                        if status == "cached":
                            cached_n += 1
                        elif status == "translated":
                            translated_n += 1
                            # non-force owner는 waiter 공개 전에 이미 캐시에 썼다.
                            # force 경로만 shared wrapper를 우회하므로 여기서 저장한다.
                            if force:
                                publish_cache(key, text)
                        elif status == "kept":
                            kept_original.append(u.id)
                            # 식별자만 기록 — 문서 원문·번역문 내용은 로그에 남기지 않는다
                            logger.warning("번역 유닛 원문 유지: %s (lang=%s, unit=%s)", job_dir.name, lang, u.id)
                        done += 1
                        if progress is not None:
                            progress(done, total)
                        if done % 10 == 0:
                            flush_cache()
                            # SSE 없이 state 폴링으로 보는 클라이언트(프런트 폴백)용 진행률
                            write_state("running", done, total)
            return not stopped

        # 취소: 디스패치 전 선체크
        if _cancel_set():
            flush_cache()
            return _canceled_result()

        canceled = False
        try:
            canceled = not _dispatch(targets)
        finally:
            # 정상·오류·취소 모든 경로에서 부분 캐시 보존 — 오류 전파 시에도 마지막
            # 주기 flush(10유닛) 이후 완료된 유닛의 번역이 유실되지 않게 한다.
            flush_cache()

        if canceled:
            return _canceled_result()

        # 조립 — 번역된 유닛만 교체(나머지 원문 보존)
        def _assemble_md() -> str:
            md_trans = {u.id: results[u.id] for u in md_units if u.id in results}
            return assemble_markdown(md_text, page_separator, md_trans)

        assembled = _assemble_md()
        new_pages = None
        if layout_pages is not None:
            lay_trans = {u.id: results[u.id] for u in lay_units if u.id in results}
            new_pages = apply_layout(layout_pages, lay_trans)
            reconciled = reconcile_markdown_with_layout(
                md_text,
                assembled,
                layout_pages,
                new_pages,
                page_separator,
            )
            # 폴백이면 reconcile은 인자로 받은 assembled를 그대로 돌려준다.
            if reconciled is assembled and deferred:
                # layout 번역이 md를 덮지 못했다 — 1차에서 미룬 md 유닛을 지금 번역해
                # 무손실 계약을 지킨다(2단 패스). SSE는 total 증가를 허용한다.
                total += len(deferred)
                write_state("running", done, total)
                logger.info("reconcile 폴백 — 지연 md 유닛 %d개 2차 번역", len(deferred))
                try:
                    canceled = not _dispatch(deferred)
                finally:
                    flush_cache()
                if canceled:
                    return _canceled_result()
                assembled = _assemble_md()
            else:
                assembled = reconciled
            _atomic_write(job_dir / f"layout.{lang}.json", json.dumps(new_pages, ensure_ascii=False))
        _atomic_write(job_dir / f"result.{lang}.md", assembled)

        api_mode = getattr(client, "api_mode_used", "") or cfg.api_mode
        _atomic_write_json(tdir / "report.json", {
            "kept_original": kept_original,
            "retried": retried_n,
            "repaired": repaired_n,
            "split": split_n,
            "sanitized": sanitized_n,
            "skipped": skipped,
            "cached": cached_n,
            "translated": translated_n,
            "api_mode": api_mode,
            "warnings": warnings,
        })
        write_state("done", total, total)
        logger.info(
            "번역 완료: %s (lang=%s, 번역 %d·캐시 %d·원문유지 %d)",
            job_dir.name, lang, translated_n, cached_n, len(kept_original),
        )
        return TranslateResult(
            status="done", total=total, translated=translated_n, cached=cached_n,
            kept_original=kept_original, skipped=skipped, api_mode=api_mode,
        )

    except TranslateError as e:
        # 전역 HTTP 슬롯/재시도 대기에서 cancel을 관찰한 client도 TranslateError
        # 계열로 빠져나온다. 사용자가 이미 취소한 잡을 error로 덮어쓰지 않는다.
        if _cancel_set():
            write_state("canceled", done, total)
            return TranslateResult(
                status="canceled", total=total, translated=translated_n, cached=cached_n,
                kept_original=kept_original, skipped=skipped,
                api_mode=getattr(client, "api_mode_used", "") or cfg.api_mode,
            )
        write_state("error", done, total, error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 — 사용자용 메시지로 감싸 재발생
        logger.exception("번역 중 오류: %s (lang=%s)", job_dir.name, lang)
        write_state("error", done, total, error=f"번역 중 오류: {e}")
        raise TranslateError(f"번역 중 오류: {e}") from e
