"""환경변수 기반 설정. 계약: docs/ARCHITECTURE.md §7"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from .llm.validate import local_openai_url, local_url, openai_url

logger = logging.getLogger(__name__)

_DEFAULT_REVISION = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"
_DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1"
# 렌더 dpi 허용 범위 — 업로드의 요청별 dpi 검증(api.py)과 기본값 RENDER_DPI 검증이
# 같은 범위를 쓴다. 어긋나면 dpi를 안 보내는 클라이언트의 모든 업로드가 400이 된다.
RENDER_DPI_MIN = 72
RENDER_DPI_MAX = 400


def _find_dotenv() -> Path | None:
    """자동 탐색 — 실행 cwd → 저장소 루트 순서로 처음 찾은 .env 하나.

    서로 다른 비밀 파일을 합치지 않도록 하나만 읽는다. 저장소 밖(상위 디렉터리)은
    보지 않는다 — 예전 final/ 워크스페이스 배치의 잔재(parents[3])가 레포 .env가
    없을 때 무관한 프로젝트의 키·엔드포인트를 조용히 채택해, 문서 원문이 엉뚱한
    API로 나갈 수 있었다.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for base in dict.fromkeys((Path.cwd(), repo_root)):
        cand = base / ".env"
        if cand.is_file():
            return cand
    return None


def load_dotenv_file(path: Path | None = None) -> None:
    """로컬 실행용 .env 주입 — **이미 설정된 키는 건드리지 않는다**.

    docker-compose는 .env를 읽어 environment로 넘기지만(그 값이 우선 유지됨),
    로컬 실행(macOS Metal 등)은 아무도 .env를 읽지 않아 번역 프로바이더가
    503("프로바이더 미설정")으로 떨어졌다 — CPU/CUDA/Metal 범용성 결함 수정.

    파싱은 python-dotenv(dotenv_values)에 맡긴다 — docker compose와 같은 규칙이다
    (compose v5로 대조 확인). 따옴표 없는 값은 '공백+#'부터 인라인 주석, 따옴표
    값은 짝이 맞는 따옴표까지(안쪽 '#' 보존, 큰따옴표 안은 \\n 등 이스케이프 해석),
    `export ` 접두사, CRLF, BOM을 처리한다. 예전 자체 파서는 주석까지 값에 넣어
    .env.example 줄의 '#'만 지우면 OCR_FAST_DECODE=1이 False로 뒤집히고
    OCR_DEVICE=metal이 기동에 실패했다. 이름만 있고 '='가 없는 줄은 건너뛴다.
    ⚠ `KEY=   # 설명`처럼 값 없이 주석만 두면 compose와 똑같이 주석이 값이 된다 —
    그래서 .env.example은 설명을 별도 줄에 둔다.

    자동 탐색(path 미지정)은 실행 cwd와 저장소 루트만 본다(_find_dotenv). 실제로
    읽은 경로는 INFO로 남긴다(값은 절대 남기지 않는다). `DISABLE_DOTENV`가 참 값
    (1/true/yes/on)이면 자동 탐색을 끈다 — 테스트·E2E 하네스가 개발자의 실제
    .env(실키)를 프로세스 환경에 주입하지 않게 하는 스위치다. path를 명시한
    호출은 이 스위치와 무관하게 그 파일을 읽는다.
    """
    if path is None:
        if _env_bool("DISABLE_DOTENV", False):
            return
        path = _find_dotenv()
    if path is None or not path.is_file():
        return
    # utf-8-sig: 편집기가 붙인 BOM이 첫 키 이름에 섞이지 않게 한다
    values = dotenv_values(path, encoding="utf-8-sig")
    applied = kept = 0
    for key, value in values.items():
        if value is None:
            continue
        if key in os.environ:
            kept += 1  # 실제 환경변수(compose·셸 주입)가 이긴다
            continue
        os.environ[key] = value
        applied += 1
    logger.info(
        ".env 로드: %s (새로 적용 %d개, 이미 설정돼 유지 %d개)", path, applied, kept,
    )


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_raw(name: str) -> str | None:
    """빈 값(공백뿐인 값 포함)은 미설정으로 본다 — compose는 선택 키를 빈 문자열로 넘긴다."""
    v = os.environ.get(name)
    return v if v is not None and v.strip() else None


def _check_range(
    name: str, raw: str, value: float, lo: float | None, hi: float | None,
) -> None:
    if lo is not None and hi is not None:
        if not lo <= value <= hi:
            raise ValueError(f"{name}={raw!r}: {lo}–{hi} 범위여야 합니다")
    elif lo is not None and value < lo:
        raise ValueError(f"{name}={raw!r}: {lo} 이상이어야 합니다")
    elif hi is not None and value > hi:
        raise ValueError(f"{name}={raw!r}: {hi} 이하여야 합니다")


def _env_int(
    name: str, default: int, *, lo: int | None = None, hi: int | None = None,
) -> int:
    """정수 env — 비정수·범위 밖 값은 **변수명을 담은** ValueError로 기동 시 실패한다.

    예전에는 `invalid literal for int()`만 남아 어느 키가 틀렸는지 traceback을 읽어야
    했고, RENDER_DPI=600·MAX_UPLOAD_MB=0처럼 범위 밖 값은 기동은 되지만 모든 업로드를
    400/413으로 만들어 요청 쪽 문제로 보였다.
    """
    raw = _env_raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: 정수여야 합니다") from None
    _check_range(name, raw, value, lo, hi)
    return value


def _env_float(
    name: str, default: float, *,
    lo: float | None = None, hi: float | None = None, positive: bool = False,
) -> float:
    """실수 env — NaN·무한대는 어느 노브에서도 의미가 없어 거부한다(변수명 포함).

    positive=True는 0보다 커야 하는 값(타임아웃)이다. requests는 0·음수·NaN
    타임아웃에 RequestException이 아닌 ValueError를 던져, 기동은 정상인데 잡의
    모든 페이지가 실패했다.
    """
    raw = _env_raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: 숫자여야 합니다") from None
    if not math.isfinite(value):
        raise ValueError(f"{name}={raw!r}: 유한한 숫자여야 합니다")
    if positive and value <= 0:
        raise ValueError(f"{name}={raw!r}: 0보다 커야 합니다")
    _check_range(name, raw, value, lo, hi)
    return value


def _env_limit(name: str, default: int) -> int | None:
    """상한형 env — 0 이하는 비활성(None)으로 매핑한다.

    감지기(SemanticRepetitionDetector)는 상한이 1 미만이면 ValueError를 던지므로,
    여기서 매핑하지 않으면 설정 실수가 잡 실행 시점의 혼란스러운 오류로 발현된다."""
    v = _env_int(name, default)
    return v if v > 0 else None


def _env_int_or_warn(name: str, default: int) -> int:
    """상한형 env — 오타(비정수)로 서비스가 500이 되지 않게 기본값으로 강등한다."""
    try:
        return _env_int(name, default)
    except ValueError:
        logger.warning("%s 값이 정수가 아닙니다 — 기본값 %d 사용", name, default)
        return default


def _split_hosts(v: str) -> list[str]:
    return [h.strip() for h in v.split(",") if h.strip()]


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    """콤마 구분 env → 튜플 (항목 양끝 공백 제거, 빈 항목 무시)."""
    return tuple(item.strip() for item in os.environ.get(name, default).split(",") if item.strip())


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """허용값 검증형 env — 잘못된 값은 기동 시점(ValueError)에 즉시 실패한다."""
    v = os.environ.get(name, default)
    if v not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return v


def _env_unescaped(name: str, default: str) -> str:
    """백슬래시 이스케이프(`\\n`·`\\t`·`\\uXXXX` 등)를 해석하는 문자열 env.

    예전 `encode().decode("unicode_escape")`는 UTF-8 바이트를 Latin-1로 읽어 한글·
    전각 대시가 'í\\x8e\\x98…'처럼 깨졌다. Latin-1 밖 문자를 먼저 `\\uXXXX`로 바꿔
    두면 unicode_escape가 원래 문자로 되돌리고, 기존 ASCII 이스케이프 해석은 그대로다.
    큰따옴표 .env 값처럼 이미 풀린 실제 개행도 그대로 남는다.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return raw.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{name}={raw!r}: 백슬래시 이스케이프를 해석할 수 없습니다 ({e.reason})"
        ) from None


def _translate_global_concurrency() -> int:
    """서버 전체 번역 HTTP 상한(1..8).

    새 전역 키가 없으면 기존 TRANSLATE_CONCURRENCY를 따라 단일 설정만 쓰던 배포도
    여러 잡 합계까지 같은 값으로 제한한다. 기존 키가 잘못된 경우 앱 기동 자체는
    유지하고 TranslateConfig의 503 검증에 맡긴다. 새 키의 오타는 명시 설정이므로
    기동 시 바로 드러낸다.
    """
    raw = os.environ.get("TRANSLATE_GLOBAL_CONCURRENCY")
    # Compose는 선택 키를 빈 문자열로 명시 전달한다. 빈 값은 '미설정'과 같게
    # 취급해야 아래의 잡당 상한 fallback 계약이 컨테이너에서도 유지된다.
    if raw is not None and raw.strip():
        return min(8, max(1, _env_int("TRANSLATE_GLOBAL_CONCURRENCY", 8)))
    try:
        return min(8, max(1, int(os.environ.get("TRANSLATE_CONCURRENCY") or 8)))
    except ValueError:
        return 8


def _local_openai_settings() -> dict:
    """local-openai(Q&A) 설정 — Settings의 llm_local_openai_* 필드 값.

    base URL이 비면 미구성(공급자 목록에 광고하지 않음)이다. 주소는 루프백·
    host.docker.internal만 허용하고(local_openai_url), 모델 없이 주소만 둔 설정은 기동
    시점에 거부한다. 키는 LLM_LOCAL_OPENAI_API_KEY만 읽는다 — 번역용 OPENAI_API_KEY나
    Q&A용 LLM_OPENAI_API_KEY로 폴백하면 그 키가 로컬 서버 로그 등으로 새어 나간다.
    """
    raw = os.environ.get("LLM_LOCAL_OPENAI_BASE_URL", "").strip()
    model = os.environ.get("LLM_LOCAL_OPENAI_MODEL", "").strip()
    models = _env_csv("LLM_LOCAL_OPENAI_MODELS", "")
    key = os.environ.get("LLM_LOCAL_OPENAI_API_KEY", "").strip()
    if raw and not model:
        raise ValueError(
            "LLM_LOCAL_OPENAI_MODEL이 필요합니다 — LLM_LOCAL_OPENAI_BASE_URL을 설정하면 "
            "서버의 /v1/models id(mlx_lm.server는 default_model)를 함께 지정하세요"
        )
    return {
        "llm_local_openai_base_url": local_openai_url(raw) if raw else "",
        "llm_local_openai_model": model,
        "llm_local_openai_models": models,
        "llm_local_openai_api_key": key,
    }


def _llm_provider() -> str:
    """LLM_PROVIDER — local-openai는 LLM_LOCAL_OPENAI_BASE_URL이 있을 때만 기본값이 될 수 있다."""
    provider = _env_choice(
        "LLM_PROVIDER", "openai-responses",
        ("openai-responses", "openai-chat", "ollama", "local-openai"),
    )
    if provider == "local-openai" and not os.environ.get("LLM_LOCAL_OPENAI_BASE_URL", "").strip():
        raise ValueError("LLM_PROVIDER=local-openai에는 LLM_LOCAL_OPENAI_BASE_URL이 필요합니다")
    return provider


@dataclass
class Settings:
    device: str = "cpu"                 # cpu | cuda | metal (mps는 metal의 별칭)
    dtype: str = "auto"                 # auto | bfloat16 | float16 | float32
    engine: str = "unlimited"           # unlimited | fake | textlayer | ovisocr2 | paddleocr_vl (registry.VALID_ENGINES)
    model_id: str = "baidu/Unlimited-OCR"
    model_revision: str = _DEFAULT_REVISION
    preload_model: bool = True
    data_dir: Path = field(default_factory=lambda: Path("data"))
    frontend_dir: Path | None = None    # None이면 리포 상대 경로에서 탐색
    render_dpi: int = 200
    pages_per_chunk: int = 8
    # 페이지 OCR 충실도 게이트 — born-digital PDF의 텍스트 레이어와 대조해 임계값
    # 미만인 페이지만 단독 재실행한다(pipeline/fidelity.py 참조). 0 이하 = 비활성.
    # 기본 0.70의 근거(46쪽 실측, 열화 4쪽이 확인된 실행): 열화 페이지 최고 0.494 vs
    # **전 페이지 중** 비열화 최저 0.813. 유효 구간의 가운데다.
    ocr_fidelity_threshold: float = 0.70
    # 재시도 상한 — 문서 페이지 수 대비 비율. 스캔·손상 문서에서 전 페이지 재실행으로
    # 런타임이 폭발하지 않게 막는다. 짧은 문서도 복구할 수 있게 최소 2쪽은 허용한다.
    ocr_fidelity_max_retry_ratio: float = 0.2
    max_pages: int = 200
    max_upload_mb: int = 100
    max_length: int = 32768
    max_page_output_chars: int | None = 16_384  # 페이지별 decoded 문자 hard limit (env 0 이하=비활성)
    max_page_output_tokens: int | None = 6_144  # 페이지별 생성 토큰 hard limit (env 0 이하=비활성)
    page_separator: str = "\n\n---\n\n"
    cpu_threads: int = 0                # 0=torch 기본값 (CPU 백엔드 전용)
    fast_decode: bool = True            # 커스텀 그리디 디코드 루프 (0이면 HF generate 폴백)
    decode_block: int = 8               # fast_decode의 호스트 동기화 배칭 크기(토큰)
    fake_delay: float = 0.02            # FakeEngine 페이지당 지연(초)
    job_ttl_days: int = 0               # 터미널 잡(done/error/canceled) 자동 GC 보존 일수 — 0=비활성(opt-in)
    # ── sidecar 엔진 (OCR_ENGINE=ovisocr2|paddleocr_vl) 공용 클라이언트 설정 ──
    sidecar_url: str = ""               # 예: http://ovisocr2:8080 — sidecar 엔진 선택 시 필수
    sidecar_connect_timeout_s: float = 10.0
    sidecar_read_timeout_s: float = 600.0
    sidecar_health_timeout_s: float = 5.0
    sidecar_max_response_mb: int = 20   # /v1/parse 응답 크기 상한 (response bomb 방어)
    sidecar_retries: int = 1            # 연결 수립 실패 시 재시도 횟수 (읽기 중 실패는 runner 재시도 몫)
    remote_page_concurrency: int = 1    # sidecar 페이지 동시 요청 수(=sidecar 엔진의 청크 크기)
    # 잡이 sidecar 모델 준비를 기다리는 상한(초). 최초 기동은 모델 다운로드 + vLLM
    # 그래프 컴파일로 수 분 걸릴 수 있어 넉넉히 잡는다 — 이 시간 안에 업로드하면
    # 잡이 실패하지 않고 대기했다가 처리된다(취소 가능).
    sidecar_model_wait_s: float = 900.0
    # Host 헤더 화이트리스트 (DNS rebinding 방어) — 포트는 비교 시 무시됨 (localhost:8000 → localhost)
    allowed_hosts: list[str] = field(default_factory=lambda: _split_hosts(_DEFAULT_ALLOWED_HOSTS))
    # ── Localight 통합: 페이지 텍스트 추출 + LLM 프로바이더 ──
    ocr_languages: str = "eng+kor"          # tesseract -l 인자 (textlayer 엔진이 소비)
    native_text_threshold: int = 120        # auto 추출이 텍스트 레이어를 신뢰할 최소 영숫자 수 (0 이상)
    ollama_base_url: str = "http://127.0.0.1:11434"  # local_url 검증 — 온디바이스 allowlist 전용
    ollama_model: str = "qwen3:8b"          # 기본 로컬 Ollama 모델
    llm_provider: str = "openai-responses"  # openai-responses | openai-chat | ollama
    llm_reasoning_effort: str = "low"       # default|none|minimal|low|medium|high|xhigh|max
    llm_openai_base_url: str = "https://api.openai.com/v1"  # openai_url 검증 — 공식 호스트 고정
    llm_openai_responses_models: tuple[str, ...] = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
    llm_openai_chat_models: tuple[str, ...] = ("chat-latest", "gpt-5.6-luna", "gpt-5.6-terra")
    llm_openai_responses_model: str = "gpt-5.6-luna"  # Responses 기본 모델
    llm_openai_chat_model: str = "chat-latest"        # Chat Completions 기본 모델
    # 로컬 OpenAI 호환 서버(local-openai: oMLX·LM Studio·mlx_lm.server) — 루프백 전용.
    # 빈 base URL = 미구성. 키는 전용 LLM_LOCAL_OPENAI_API_KEY만 쓴다(폴백 없음).
    llm_local_openai_base_url: str = ""
    llm_local_openai_model: str = ""
    llm_local_openai_models: tuple[str, ...] = ()
    llm_local_openai_api_key: str = ""
    # 번역 서브시스템(TranslateConfig.from_env)과 공유하는 키 — 여기서는 읽기만 한다
    openai_api_key: str = ""
    # Q&A 전용 OpenAI 키 — 번역 키와 분리(공유 금지). llm_openai_base_url이 공식
    # api.openai.com으로 고정돼 있으므로, OpenRouter·로컬 게이트웨이용 OPENAI_API_KEY를
    # 폴백으로 재사용하면 그 키가 제3자에게 전송된다. 미설정 시 openai-* 공급자는
    # available:false로 광고된다.
    llm_openai_api_key: str = ""
    # 여러 번역 잡을 합친 실제 upstream HTTP 요청 상한. 기본은 잡당 상한과 같은 8.
    translate_global_concurrency: int = 8
    # 번역 PDF 내보내기용 한글 폰트 파일 경로 — 빈 값이면 시스템 폰트 → 내장 CJK 폴백
    pdf_export_font: str = ""
    # ── 남용 방어 상한 (api.py의 _AbuseGuard가 소비) — 0 이하면 해당 상한 비활성 ──
    qa_rate_limit_per_min: int = 30      # 잡·IP 단위 Q&A 요청 수/분
    qa_max_concurrent: int = 4           # 동시에 처리 중인 Q&A 수
    translate_rate_limit_per_min: int = 12   # 잡·IP 단위 번역 시작 요청 수/분
    translate_max_active: int = 4            # 동시에 도는 번역 스레드 수

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()  # 로컬 실행(Metal 등)에서도 .env의 번역/OCR 설정이 잡히게
        frontend = os.environ.get("FRONTEND_DIR")
        device = os.environ.get("OCR_DEVICE", "cpu").strip().lower()
        return cls(
            device="metal" if device == "mps" else device,
            dtype=os.environ.get("OCR_DTYPE", "auto").strip().lower(),
            engine=os.environ.get("OCR_ENGINE", "unlimited").strip().lower(),
            model_id=os.environ.get("MODEL_ID", "baidu/Unlimited-OCR"),
            # `or` 폴백: compose가 ${MODEL_REVISION:-}로 **빈 문자열**을 넘겨도
            # 기본 고정 SHA가 무력화되지 않게 (sidecar들의 _env_str 패턴과 동일)
            model_revision=os.environ.get("MODEL_REVISION") or _DEFAULT_REVISION,
            preload_model=_env_bool("PRELOAD_MODEL", True),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            frontend_dir=Path(frontend) if frontend else None,
            # 범위 검증 — 범위 밖 값은 기동은 되지만 나중에 모든 업로드·페이지를
            # 실패시킨다(요청 쪽 문제로 보임). 기동 시 변수명과 함께 바로 실패한다.
            render_dpi=_env_int("RENDER_DPI", 200, lo=RENDER_DPI_MIN, hi=RENDER_DPI_MAX),
            pages_per_chunk=_env_int("PAGES_PER_CHUNK", 8, lo=1),
            ocr_fidelity_threshold=_env_float("OCR_FIDELITY_THRESHOLD", 0.70),
            ocr_fidelity_max_retry_ratio=_env_float(
                "OCR_FIDELITY_MAX_RETRY_RATIO", 0.2
            ),
            max_pages=_env_int("MAX_PAGES", 200, lo=1),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 100, lo=1),
            max_length=_env_int("MAX_LENGTH", 32768, lo=1),
            max_page_output_chars=_env_limit("MAX_PAGE_OUTPUT_CHARS", 16_384),
            max_page_output_tokens=_env_limit("MAX_PAGE_OUTPUT_TOKENS", 6_144),
            page_separator=_env_unescaped("PAGE_SEPARATOR", "\n\n---\n\n"),
            cpu_threads=_env_int("OCR_CPU_THREADS", 0),
            fast_decode=_env_bool("OCR_FAST_DECODE", True),
            decode_block=_env_int("OCR_DECODE_BLOCK", 8, lo=1),
            fake_delay=_env_float("FAKE_DELAY", 0.02, lo=0),
            job_ttl_days=_env_int("JOB_TTL_DAYS", 0, lo=0),
            allowed_hosts=_split_hosts(os.environ.get("ALLOWED_HOSTS") or _DEFAULT_ALLOWED_HOSTS),
            sidecar_url=os.environ.get("OCR_SIDECAR_URL", "").strip().rstrip("/"),
            sidecar_connect_timeout_s=_env_float(
                "OCR_SIDECAR_CONNECT_TIMEOUT_S", 10.0, positive=True,
            ),
            sidecar_read_timeout_s=_env_float(
                "OCR_SIDECAR_READ_TIMEOUT_S", 600.0, positive=True,
            ),
            sidecar_health_timeout_s=_env_float(
                "OCR_SIDECAR_HEALTH_TIMEOUT_S", 5.0, positive=True,
            ),
            sidecar_max_response_mb=max(1, _env_int("OCR_SIDECAR_MAX_RESPONSE_MB", 20)),
            sidecar_retries=max(0, _env_int("OCR_SIDECAR_RETRIES", 1)),
            remote_page_concurrency=max(1, _env_int("OCR_REMOTE_PAGE_CONCURRENCY", 1)),
            sidecar_model_wait_s=max(0.0, _env_float("OCR_SIDECAR_MODEL_WAIT_S", 900.0)),
            ocr_languages=os.environ.get("OCR_LANGUAGES", "eng+kor"),
            native_text_threshold=max(0, _env_int("NATIVE_TEXT_THRESHOLD", 120)),
            ollama_base_url=local_url(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
            llm_provider=_llm_provider(),
            llm_reasoning_effort=_env_choice(
                "LLM_REASONING_EFFORT",
                "low",
                ("default", "none", "minimal", "low", "medium", "high", "xhigh", "max"),
            ),
            llm_openai_base_url=openai_url(
                os.environ.get("LLM_OPENAI_BASE_URL", "https://api.openai.com/v1")
            ),
            llm_openai_responses_models=_env_csv(
                "LLM_OPENAI_RESPONSES_MODELS", "gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol"
            ),
            llm_openai_chat_models=_env_csv(
                "LLM_OPENAI_CHAT_MODELS", "chat-latest,gpt-5.6-luna,gpt-5.6-terra"
            ),
            llm_openai_responses_model=os.environ.get("LLM_OPENAI_RESPONSES_MODEL", "gpt-5.6-luna"),
            llm_openai_chat_model=os.environ.get("LLM_OPENAI_CHAT_MODEL", "chat-latest"),
            **_local_openai_settings(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            # OPENAI_API_KEY 폴백을 두지 않는다 — 번역 키가 api.openai.com으로 새는 경로
            llm_openai_api_key=os.environ.get("LLM_OPENAI_API_KEY", "").strip(),
            translate_global_concurrency=_translate_global_concurrency(),
            pdf_export_font=os.environ.get("PDF_EXPORT_FONT", "").strip(),
            # 오타는 기본값으로 강등한다 — 운영 중 남용 방어가 500을 내면 안 된다
            qa_rate_limit_per_min=_env_int_or_warn("QA_RATE_LIMIT_PER_MIN", 30),
            qa_max_concurrent=_env_int_or_warn("QA_MAX_CONCURRENT", 4),
            translate_rate_limit_per_min=_env_int_or_warn("TRANSLATE_RATE_LIMIT_PER_MIN", 12),
            translate_max_active=_env_int_or_warn("TRANSLATE_MAX_ACTIVE", 4),
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolve_frontend_dir(self) -> Path | None:
        if self.frontend_dir is not None:
            return self.frontend_dir if self.frontend_dir.is_dir() else None
        candidate = Path(__file__).resolve().parents[2] / "frontend"
        return candidate if candidate.is_dir() else None
