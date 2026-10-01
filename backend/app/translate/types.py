"""번역 파이프라인 공용 계약. 문서: docs/ARCHITECTURE.md §번역

디렉터리 계약 (job_dir 기준, lang은 BCP-47 소문자 예: "ko"):
  translations/{lang}/state.json    진행 상태 — 아래 write_state() 스키마
  translations/{lang}/glossary.json 문서 용어집 [{"src","ko","policy","first_unit"}]
  translations/{lang}/units.json    유닛 캐시 {cache_key: 번역문}
  translations/{lang}/report.json   품질 리포트 {"kept_original":[...],"retried":n,"skipped":n,...}
  result.{lang}.md                  번역된 마크다운 — page_separator 구조·페이지 수 보존
  layout.{lang}.json                blocks[].content만 교체된 layout.json (그 외 필드 동일)

설계 불변식:
  * 플레이스홀더(<m1 v="…"/> 형식) 복원 실패 유닛은 **원문 유지** — 내용을 잃지 않는다.
  * 캐시 키에 model·PROMPT_V·용어집 부분집합이 들어가 설정 변경 시 자동 재번역된다.
  * 이 패키지는 OCR 엔진/torch에 의존하지 않는다 (requests + 표준 라이브러리만).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

# 프롬프트/마스킹 규칙 개정 시 올린다 → 캐시 키가 바뀌어 자동 재번역 (fonts_v 패턴)
# v2: 문체 few-shot 예시 추가 (4B급 모델 합쇼체 이탈 실측 → 예시로 0건)
# v3: 신뢰도 래더(sanitize+repair+분할) + 규칙5 인명 음차 금지 강화 → kept_original 감소
# v4: [원문 유지] 유닛별 A-용어 목록(MSE 풀어쓰기 실측 차단) + 용어 매칭 수식 제외 + 스톱워드
# v5: 제목 유닛의 의미·정보량 보존 지시(짧은 UI 라벨식 축약 방지)
# v6: 출력 측 검증(masking.looks_untranslated) 도입 + 인라인 수식 통화 오인 수정
#     → 거부문·요약·영문 echo가 이미 캐시된 units.json을 강제로 무효화한다
#
# **운영자 안내 — PROMPT_V를 올리면 기존 units.json은 전부 무효가 된다.**
# cache_key에 PROMPT_V·model·temperature·reasoning이 들어가므로, 값 하나만 바뀌어도
# 기존 잡의 캐시는 단 한 건도 적중하지 않고 유닛 전량이 다시 API를 탄다(비용·시간).
# 배포 전에 알아둘 것:
#   * 잡당 1회만 발생한다 — 재번역 결과가 새 키로 다시 캐시된다.
#   * 무효화가 실제로 일어나면 엔진이 report.json의 warnings에 남기고
#     `번역 캐시 전량 무효` WARNING을 로그로 낸다. cached==0 · cache_prior>0이 지표다.
#   * 새 버전이 이전보다 나쁘면 되돌릴 곳은 이 상수다 — 값을 내리면 옛 캐시가 그대로
#     다시 적중하므로 롤백에 재번역 비용이 들지 않는다(옛 units.json을 지우지 말 것).
PROMPT_V = "6"

SUPPORTED_LANGS = ("ko",)
MAX_TRANSLATE_CONCURRENCY = 8

# TRANSLATE_REASONING effort별 요청당 max_tokens 예산 (사용자 확정, 2026-07-08).
# thinking 모델은 reasoning 토큰이 같은 예산에서 차감되므로 effort에 비례해 키운다.
REASONING_MAX_TOKENS = {
    "": 8192,        # 파라미터 미전송 (reasoning 여부 모름 — off와 동일 예산)
    "off": 8192,
    "low": 10240,
    "medium": 20480,
    "high": 40960,
    "xhigh": 81920,
}

# TRANSLATE_REASONING 값을 서버에 **어떤 필드로** 전달할지 (TRANSLATE_REASONING_STYLE).
# reasoning을 끄는 표준 파라미터가 없어 서버 계열마다 다르다:
#   openrouter           reasoning:{enabled:false} / reasoning:{effort} (종전 유일 방식)
#   chat_template_kwargs chat_template_kwargs:{enable_thinking:false} — mlx_lm.server·oMLX·
#                        vLLM·llama.cpp·SGLang. mlx_lm은 Qwen 계열에 thinking을 **기본 주입**
#                        하고 reasoning 필드는 읽지 않아, 종전 off는 무효였다(실측: 2문장
#                        유닛이 8192+16384 토큰을 사고에 쓰고 100초 뒤 잡 실패).
#   reasoning_effort     OpenAI 공식 — chat은 최상위 reasoning_effort, responses는
#                        reasoning.effort. off는 "none"으로 보낸다.
#   none                 어떤 reasoning 필드도 보내지 않는다(엄격한 게이트웨이용).
#   auto                 base URL로 고른다 — 루프백·사설·도커 내부 → chat_template_kwargs,
#                        openrouter.ai → openrouter, api.openai.com → reasoning_effort,
#                        그 외 공개 호스트 → openrouter(종전 동작 유지).
REASONING_STYLES = ("auto", "openrouter", "chat_template_kwargs", "reasoning_effort", "none")

# TRANSLATE_EXTRA_BODY가 덮어쓸 수 없는 키 — 클라이언트가 구조를 책임지는 필드다
# (요청 본문·스트리밍 파서·잘림 재시도 예산·store:false 프라이버시 약속).
EXTRA_BODY_RESERVED = frozenset({
    "model", "messages", "input", "instructions", "stream", "stream_options", "n",
    "max_tokens", "max_completion_tokens", "max_output_tokens", "store",
})
_EXTRA_BODY_MAX_CHARS = 4096


def _host_is_local(host: str) -> bool:
    """루프백·사설망·도커 내부·mDNS·단일 라벨(도커 서비스명) 호스트인가."""
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # ollama·vllm·omlx 같은 compose 서비스명
    return not ip.is_global


def resolve_reasoning_style(style: str, base_url: str) -> str:
    """auto를 base URL 기준의 실제 전달 방식으로 확정한다(그 외 값은 그대로)."""
    if style != "auto":
        return style
    host = (urlsplit(base_url.strip()).hostname or "").lower().rstrip(".")
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    if host == "api.openai.com":
        return "reasoning_effort"
    if _host_is_local(host):
        return "chat_template_kwargs"
    return "openrouter"


def _env_extra_body(env) -> str:
    """TRANSLATE_EXTRA_BODY — 모든 요청 본문에 병합할 JSON 객체(정규화 문자열로 보관).

    서버별 비표준 파라미터(chat_template_kwargs의 추가 키, repetition_penalty 등)를
    코드 수정 없이 전달하는 탈출구다. 객체가 아니거나 너무 크거나 클라이언트가
    책임지는 키(EXTRA_BODY_RESERVED)를 건드리면 기동 시 바로 거부한다.
    """
    raw = (env.get("TRANSLATE_EXTRA_BODY") or "").strip()
    if not raw:
        return ""
    if len(raw) > _EXTRA_BODY_MAX_CHARS:
        raise TranslateError(
            f"TRANSLATE_EXTRA_BODY가 너무 깁니다({len(raw)}자 > {_EXTRA_BODY_MAX_CHARS}자)"
        )
    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise TranslateError(f"TRANSLATE_EXTRA_BODY는 JSON 객체여야 합니다 ({e.msg})") from e
    if not isinstance(obj, dict):
        raise TranslateError("TRANSLATE_EXTRA_BODY는 JSON 객체({...})여야 합니다")
    reserved = sorted(k for k in obj if k in EXTRA_BODY_RESERVED)
    if reserved:
        raise TranslateError(
            "TRANSLATE_EXTRA_BODY는 다음 키를 덮어쓸 수 없습니다: " + ", ".join(reserved)
        )
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class TranslateError(RuntimeError):
    """번역 실패 — message는 사용자에게 그대로 보여줄 수 있는 한국어 문장."""


class TranslateAPIError(TranslateError):
    """업스트림 API 오류 (상태코드·본문 요약 포함)."""


class TranslateUnitRejected(TranslateAPIError):
    """재시도해도 같은 결과인 4xx 거부 (400·413·422 등 — 429/408 제외).

    거대 병합 표·초장문 문단 하나가 모델 컨텍스트를 넘긴 경우가 대표적이다.
    엔진은 이 오류만 유닛 단위로 강등(래더 → 원문 유지)한다. 연결 실패·5xx 등
    비결정적 오류는 종전대로 잡 전체 실패로 전파한다.
    """


class TranslateOutputTruncated(TranslateUnitRejected):
    """출력이 max_tokens에서 잘렸다(2배 재시도 후에도, 또는 재시도가 무의미).

    잘린 번역은 절대 반환·캐시하지 않는다 — 종전에는 '래더가 흡수'한다며 반환했지만
    플레이스홀더 없는 산문은 래더에 들어가지도 않아 문단 끝이 조용히 사라진 채
    캐시됐다. 유닛 단위 거부라 엔진이 분할(반쪽은 예산 안에 든다)로 보내고, 끝내
    실패하면 kept_reason 'truncated'로 원문을 유지한다.
    """


class TranslateEmptyOutput(TranslateUnitRejected):
    """200 응답인데 본문이 비었다(사고만 내고 끝났거나 빈 문자열) — 유닛 단위 거부."""


class TranslateTimeout(TranslateUnitRejected):
    """응답이 TRANSLATE_TIMEOUT_S 동안 한 바이트도 오지 않았다(재시도 소진).

    스트리밍에서는 토큰 사이 정지 시간이 상한이라, 이 오류는 그 유닛의 생성이 멈췄거나
    서버 대기열이 막혔다는 뜻이다. 이번 실행에서 성공한 호출이 있으면 엔진이 유닛
    단위로 강등해 분할(짧은 생성)로 회복을 시도한다 — 종전에는 '연결 실패'로 잡 전체가
    실패했다.
    """


def _clean(v: str | None) -> str:
    """env 값 정리 — 공백/따옴표 제거 (.env를 셸/compose 밖에서 읽었을 때 대비)."""
    return (v or "").strip().strip("'\"").strip()


def _env_int(env, name: str, default: int) -> int:
    try:
        return int(_clean(env.get(name)) or default)
    except ValueError as e:
        raise TranslateError(f"{name}는 정수여야 합니다") from e


def _env_float(env, name: str, default: float) -> float:
    try:
        value = float(_clean(env.get(name)) or default)
    except ValueError as e:
        raise TranslateError(f"{name}는 유한한 숫자여야 합니다") from e
    if not math.isfinite(value):
        raise TranslateError(f"{name}는 유한한 숫자여야 합니다")
    return value


# 샘플링 온도 허용 범위 — OpenAI 호환 API 공통 상한(2)과 하한(0).
_TEMPERATURE_MAX = 2.0


def _env_temperature(env) -> str:
    """TRANSLATE_TEMPERATURE — "none"(파라미터 생략) 또는 0–2의 유한한 숫자.

    종전에는 소문자화만 하고 요청을 만들 때 float()을 불렀다. 'abc'·'0,2'는 잡마다
    '번역 중 오류: could not convert…'로, 'nan'은 JSON NaN이 되어 '연결 실패'로,
    음수는 mlx_lm이 응답 없이 연결을 끊어 재시도 뒤 '연결 실패'로 끝났다(실측).
    그 사이 health는 translate_available:true를 광고했다. 설정 경계에서 거른다.
    캐시 키 재료이므로 유효한 값의 표기는 바꾸지 않는다(기존 units.json 호환).
    """
    raw = (_clean(env.get("TRANSLATE_TEMPERATURE")) or "0").lower()
    if raw == "none":
        return raw
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and 0.0 <= value <= _TEMPERATURE_MAX):
        raise TranslateError(
            "TRANSLATE_TEMPERATURE는 none 또는 0–2 사이의 숫자여야 합니다"
        )
    return raw


_FALSE_WORDS = ("0", "false", "no", "off")


def _env_flag(env, name: str, default: bool) -> bool:
    """불리언 env — 0/false/no/off(대소문자 무관)만 거짓. 빈 값은 기본값.

    TRANSLATE_CONTEXT=False·OFF처럼 대문자로 끈 설정이 종전 비교(소문자 튜플과
    대소문자 구분 비교)에서 참으로 읽혀 문맥이 계속 실렸다.
    """
    raw = _clean(env.get(name)).lower()
    if not raw:
        return default
    return raw not in _FALSE_WORDS


@dataclass(frozen=True)
class TranslateConfig:
    base_url: str
    api_key: str
    model: str
    api_mode: str = "auto"  # auto | chat | responses
    concurrency: int = MAX_TRANSLATE_CONCURRENCY
    timeout_s: float = 180.0
    max_retries: int = 3
    temperature: str = "0"  # "none"이면 파라미터 생략
    max_tokens_param: str = "max_tokens"  # max_tokens | max_completion_tokens | none
    context: bool = True  # 직전 유닛 꼬리를 참고 컨텍스트로 프롬프트에 포함
    # reasoning 모델 제어: "" = 파라미터 미전송(호환 기본), off | low|medium|high|xhigh.
    # 어떤 필드로 보낼지는 reasoning_style이 정한다(REASONING_STYLES 주석 참조).
    # 실측(qwen3.7-plus): off가 유닛당 37s→1.7s, 출력 토큰 ~1/40 — 번역엔 reasoning 불필요.
    reasoning: str = ""
    reasoning_style: str = "auto"  # REASONING_STYLES — auto는 base URL로 확정
    # 모든 요청 본문에 병합할 JSON 객체(정규화 문자열, "" = 없음) — TRANSLATE_EXTRA_BODY
    extra_body: str = ""

    @property
    def max_output_tokens(self) -> int:
        """요청당 max_tokens 예산 — reasoning effort별 고정 테이블 (사용자 확정값).

        thinking 토큰이 출력 예산에서 차감되므로 effort가 높을수록 예산을 키운다.
        미사용 토큰은 과금되지 않으므로 상한은 폭주 방지용이다."""
        return REASONING_MAX_TOKENS.get(self.reasoning, REASONING_MAX_TOKENS[""])

    @property
    def effective_reasoning_style(self) -> str:
        """실제로 쓰는 reasoning 전달 방식 (auto를 base URL로 확정한 값)."""
        return resolve_reasoning_style(self.reasoning_style, self.base_url)

    @property
    def extra_body_dict(self) -> dict:
        return json.loads(self.extra_body) if self.extra_body else {}

    @property
    def request_variant(self) -> str:
        """캐시 키 재료 — 종전 키(model·temperature·reasoning) 밖에서 요청을 바꾸는 설정.

        종전과 **바이트 동일한 요청**이면 빈 문자열이다: reasoning 미설정이거나 openrouter
        방식(종전 유일 방식)이고 extra body가 없으면 기존 units.json이 그대로 적중한다.
        off를 chat_template_kwargs로 보내면 실제로 thinking이 꺼져 출력이 달라지므로
        키가 바뀌어야 한다 — 종전 off가 무효였던 서버의 캐시를 재사용하지 않는다.
        """
        parts = []
        style = self.effective_reasoning_style
        if self.reasoning and style != "openrouter":
            parts.append(f"reasoning_style={style}")
        if self.extra_body:
            parts.append(f"extra_body={self.extra_body}")
        return ";".join(parts)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "TranslateConfig":
        e = os.environ if env is None else env
        base_url = _clean(e.get("OPENAI_BASE_URL"))
        model = _clean(e.get("TRANSLATE_MODEL")) or _clean(e.get("OPENAI_MODEL"))
        if not base_url or not model:
            raise TranslateError(
                "번역 프로바이더가 설정되지 않았습니다 — .env에 OPENAI_BASE_URL과 "
                "OPENAI_MODEL을 지정하세요 (.env.example 참조)"
            )
        mode = (_clean(e.get("TRANSLATE_API_MODE")) or "auto").lower()
        if mode not in ("auto", "chat", "responses"):
            raise TranslateError("TRANSLATE_API_MODE는 auto|chat|responses 중 하나여야 합니다")
        mt_param = (_clean(e.get("TRANSLATE_MAX_TOKENS_PARAM")) or "max_tokens").lower()
        if mt_param not in ("max_tokens", "max_completion_tokens", "none"):
            raise TranslateError(
                "TRANSLATE_MAX_TOKENS_PARAM은 max_tokens|max_completion_tokens|none 중 하나여야 합니다"
            )
        reasoning = (_clean(e.get("TRANSLATE_REASONING")) or "").lower()
        if reasoning not in REASONING_MAX_TOKENS:
            raise TranslateError("TRANSLATE_REASONING은 off|low|medium|high|xhigh 또는 빈 값이어야 합니다")
        style = (_clean(e.get("TRANSLATE_REASONING_STYLE")) or "auto").lower()
        if style not in REASONING_STYLES:
            raise TranslateError(
                "TRANSLATE_REASONING_STYLE은 " + "|".join(REASONING_STYLES) + " 중 하나여야 합니다"
            )
        return cls(
            base_url=base_url,
            api_key=_clean(e.get("OPENAI_API_KEY")),
            model=model,
            api_mode=mode,
            # 한 잡의 worker 상한은 8. 잘못 큰 값을 넣어도 설정 경계에서
            # 1..8로 고정한다. API 서버는 별도의 프로세스 전역 semaphore로
            # 여러 잡을 합친 실제 HTTP 요청 수도 같은 상한 안에 둔다.
            concurrency=min(
                MAX_TRANSLATE_CONCURRENCY,
                max(1, _env_int(e, "TRANSLATE_CONCURRENCY", MAX_TRANSLATE_CONCURRENCY)),
            ),
            timeout_s=max(5.0, _env_float(e, "TRANSLATE_TIMEOUT_S", 180.0)),
            max_retries=max(0, _env_int(e, "TRANSLATE_MAX_RETRIES", 3)),
            temperature=_env_temperature(e),
            max_tokens_param=mt_param,
            context=_env_flag(e, "TRANSLATE_CONTEXT", True),
            reasoning=reasoning,
            reasoning_style=style,
            extra_body=_env_extra_body(e),
        )


@dataclass
class TranslateResult:
    """run_translation 반환값 — report.json에도 같은 내용이 남는다."""

    status: str  # done | canceled
    total: int = 0  # 번역 대상 유닛 수 (skip 제외)
    translated: int = 0  # 이번 실행에서 API로 번역된 유닛
    cached: int = 0  # 캐시 적중 유닛
    kept_original: list[str] = field(default_factory=list)  # 복원 실패 → 원문 유지 유닛 id
    skipped: int = 0  # 정책상 번역 제외 (references·수식뿐인 블록 등)
    api_mode: str = ""  # 실제 사용된 모드 (auto가 확정된 결과)


def cache_key(
    masked_src: str,
    model: str,
    glossary_pairs: list[tuple[str, str]],
    *,
    original_src: str,
    unit_kind: str,
    context_tail: str | None,
    temperature: str = "",
    reasoning: str = "",
    request_variant: str = "",
) -> str:
    """유닛 캐시 키 — 원문·마스킹문·종류·모델·프롬프트·용어집·샘플링에 민감.

    용어집이 바뀌면 영향받는 유닛만 자연 무효화된다. glossary_pairs는
    (src, ko) 튜플 목록이며 순서 무관하도록 정렬해 해시한다.

    마스킹 플레이스홀더의 ``v`` 미리보기는 의도적으로 짧다. 따라서 masked_src만
    해시하면 앞부분이 같은 긴 수식/URL이 충돌해 다른 유닛의 *복원 완료 텍스트*를
    재사용할 수 있다. 원문 전체를 함께 넣어 그 데이터 훼손 경로를 막는다. title은
    본문과 다른 프롬프트 정책을 쓰므로 unit_kind도 반드시 키 재료에 포함한다.
    직전 문맥이 프롬프트에 들어가는 유닛은 context_tail까지 포함해 같은 문장이
    다른 문맥에서 서로의 번역을 강제로 재사용하지 않게 한다.
    temperature·reasoning도 출력을 바꾸는 요청 파라미터이므로 키에 넣는다 —
    reasoning을 off→high로 올린 뒤 재개해도 이전 설정의 번역이 재사용되던 문제.
    request_variant(TranslateConfig.request_variant — reasoning 전달 방식·extra body)는
    **비어 있지 않을 때만** 해시에 넣는다. 종전과 같은 요청이면 키가 그대로라 기존
    units.json이 계속 적중한다.
    """
    h = hashlib.sha256()
    h.update(PROMPT_V.encode())
    h.update(b"\x1f")
    h.update(model.encode())
    h.update(b"\x1f")
    for s, k in sorted(glossary_pairs):
        h.update(s.encode())
        h.update(b"\x1e")
        h.update(k.encode())
        h.update(b"\x1f")
    h.update(masked_src.encode())
    h.update(b"\x1f")
    h.update(original_src.encode())
    h.update(b"\x1f")
    h.update(unit_kind.encode())
    h.update(b"\x1f")
    h.update((context_tail or "").encode())
    h.update(b"\x1f")
    h.update(temperature.encode())
    h.update(b"\x1f")
    h.update(reasoning.encode())
    if request_variant:
        h.update(b"\x1f")
        h.update(request_variant.encode())
    return h.hexdigest()
