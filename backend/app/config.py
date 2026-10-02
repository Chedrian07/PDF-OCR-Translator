"""환경변수 기반 설정. 계약: docs/ARCHITECTURE.md §7"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import threading
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


# ── 알려진 env 키 레지스트리 ───────────────────────────────────────────────
# .env의 모르는 키를 알리기 위한 목록이다(load_dotenv_file). 예: 루트 .env의
# REASONING_EFFORT는 어떤 코드도 읽지 않아, 번역 reasoning을 껐다고 믿은 설정이 조용히
# 무시됐다(감사 translate-llm-13·infra-docs-7·mlx-integration-8).
# 키 이름을 하나씩 따옴표로 감싸지 않고 공백 구분 블록으로 둔다 — tests/test_ci_ops_contracts의
# env 키 스캐너는 이 파일의 "KEY" 문자열을 '코드가 읽는 키'로 세므로, 레지스트리가 읽기로
# 잘못 집계되면 하네스 키까지 .env.example·compose 스레딩을 요구받는다.
# tests/test_config_env_registry.py가 코드·.env.example·compose·하네스와 양방향으로 대조한다.

# 앱 코드가 읽는 운영 키 (config·translate/types·api·derived·pdf_worker·엔진·벤더·native_ops)
_APP_ENV_KEYS = frozenset("""
    OCR_DEVICE OCR_DTYPE OCR_MLX_QUANT_BITS OCR_ENGINE MODEL_ID MODEL_REVISION PRELOAD_MODEL
    DATA_DIR FRONTEND_DIR RENDER_DPI PAGES_PER_CHUNK MAX_PAGES MAX_UPLOAD_MB MAX_LENGTH
    MAX_PAGE_OUTPUT_CHARS MAX_PAGE_OUTPUT_TOKENS PAGE_SEPARATOR JOB_TTL_DAYS ALLOWED_HOSTS
    OCR_FIDELITY_THRESHOLD OCR_FIDELITY_MAX_RETRY_RATIO OCR_CPU_THREADS OCR_FAST_DECODE
    OCR_DECODE_BLOCK OCR_CUDA_GRAPHS OCR_MOE_FUSED OCR_MOE_FAST OCR_SDPA OCR_NGRAM_HOST
    PYTORCH_ENABLE_MPS_FALLBACK FAKE_DELAY DISABLE_DOTENV
    OCR_SIDECAR_URL OCR_SIDECAR_CONNECT_TIMEOUT_S OCR_SIDECAR_READ_TIMEOUT_S
    OCR_SIDECAR_HEALTH_TIMEOUT_S OCR_SIDECAR_MAX_RESPONSE_MB OCR_SIDECAR_RETRIES
    OCR_SIDECAR_MODEL_WAIT_S OCR_REMOTE_PAGE_CONCURRENCY OCR_LANGUAGES NATIVE_TEXT_THRESHOLD
    OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL TRANSLATE_MODEL TRANSLATE_API_MODE
    TRANSLATE_CONCURRENCY TRANSLATE_GLOBAL_CONCURRENCY TRANSLATE_TIMEOUT_S TRANSLATE_STREAM
    TRANSLATE_MAX_RESPONSE_MB TRANSLATE_MAX_RETRIES TRANSLATE_TEMPERATURE
    TRANSLATE_MAX_TOKENS_PARAM TRANSLATE_REASONING TRANSLATE_REASONING_STYLE
    TRANSLATE_EXTRA_BODY TRANSLATE_CONTEXT
    LLM_PROVIDER LLM_REASONING_EFFORT LLM_OPENAI_API_KEY LLM_OPENAI_BASE_URL
    LLM_OPENAI_RESPONSES_MODELS LLM_OPENAI_CHAT_MODELS LLM_OPENAI_RESPONSES_MODEL
    LLM_OPENAI_CHAT_MODEL LLM_LOCAL_OPENAI_BASE_URL LLM_LOCAL_OPENAI_MODEL
    LLM_LOCAL_OPENAI_MODELS LLM_LOCAL_OPENAI_API_KEY OLLAMA_BASE_URL OLLAMA_MODEL
    QA_RATE_LIMIT_PER_MIN QA_MAX_CONCURRENT TRANSLATE_RATE_LIMIT_PER_MIN TRANSLATE_MAX_ACTIVE
    TRUSTED_PROXY_HOPS TRUSTED_PROXY_IPS
    PDF_EXPORT_FONT PDF_EXPORT_MAX_CONCURRENT PDF_EXPORT_QUEUE_TIMEOUT_S PDF_EXPORT_WARM_WAIT_S
    PDF_WORKER_MODE PDF_PAGE_TIMEOUT_S PDF_EXPORT_BUILD_TIMEOUT_S PDF_WORKER_MEM_LIMIT_MB
    PDF_MAX_PAGE_CONTENT_MB PDF_MAX_PAGE_XOBJECT_CALLS
""".split())
# docker compose가 같은 .env에서 읽는 배포 키 (sidecar 이미지 설정·메모리 상한·바인딩).
# 오버레이(compose.ollama.yaml)의 ${…}도 포함한다 — 그 주석이 '.env로 올릴 것'이라 안내한다.
# HF_HUB_CACHE·HF_HUB_OFFLINE은 .env.example이 로컬 실행용으로 안내하는 허브 캐시 키다(앱이 아니라
# huggingface_hub가 읽는다 — HF_TOKEN과 같은 처지).
_DEPLOY_ENV_KEYS = frozenset("""
    BIND_HOST GPU_DEVICE HF_TOKEN HF_HUB_CACHE HF_HUB_OFFLINE CUDA_LAUNCH_BLOCKING
    OCR_CPU_MEM_LIMIT OCR_CUDA_MEM_LIMIT OCR_WEB_MEM_LIMIT OVIS_MEM_LIMIT PADDLE_MEM_LIMIT
    OLLAMA_MEM_LIMIT
    OVIS_MODEL_ID OVIS_MODEL_REVISION OVIS_DTYPE OVIS_GPU_MEMORY_UTILIZATION OVIS_MAX_MODEL_LEN
    OVIS_MAX_OUTPUT_TOKENS OVIS_MAX_NUM_SEQS OVIS_MIN_PIXELS OVIS_MAX_PIXELS
    OVIS_GDN_PREFILL_BACKEND OVIS_MAX_UPLOAD_MB
    PADDLEOCR_MODEL_ID PADDLEOCR_MODEL_REVISION PADDLEOCR_DEVICE PADDLEOCR_MIN_PIXELS
    PADDLEOCR_MAX_PIXELS PADDLEOCR_MAX_UPLOAD_MB
""".split())
# 문서화된 테스트·하네스 스위치 (opt-in 실기기 테스트·E2E·모의 LLM)
_HARNESS_ENV_KEYS = frozenset("""
    OCR_MPS_TESTS OCR_MLX_REAL_TESTS E2E_MOCK_PORT E2E_BACKEND_PORT E2E_BASE_URL E2E_PDF
    E2E_TIMEOUT_S E2E_VERIFY_MOCK_LLM MOCK_STREAM_DELAY_S MOCK_STREAM_CHUNK MOCK_FINISH
    MOCK_REASONING_CHARS MOCK_TRANSLATE_RATIO FAULT
""".split())
KNOWN_ENV_KEYS: frozenset[str] = _APP_ENV_KEYS | _DEPLOY_ENV_KEYS | _HARNESS_ENV_KEYS

# 다른 도구가 읽는 키 — 같은 .env에 두는 일이 흔하다(HF 캐시·torch·프록시·compose). 경고하지 않는다.
# MLX_(libmlx 런타임 노브 — Apple Silicon 기본 OCR 엔진이 같은 프로세스에서 읽는다)·MTL_·
# METAL_(Metal 디버그)·OBJC_(fork 안전성), TESSDATA_PREFIX(textlayer의 tesseract가 상속).
_FOREIGN_ENV_PREFIXES = tuple("""
    HF_ HUGGINGFACE_ HUGGING_FACE_ TRANSFORMERS_ TOKENIZERS_ TORCH_ PYTORCH_ CUDA_ NVIDIA_
    NCCL_ OMP_ MKL_ OPENBLAS_ KMP_ COMPOSE_ DOCKER_ BUILDKIT_ UV_ PIP_ PYTHON LC_ SSL_
    MLX_ MTL_ METAL_ OBJC_
""".split())
_FOREIGN_ENV_KEYS = frozenset("""
    TZ LANG LANGUAGE HOME PATH USER SHELL TERM TMPDIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY TESSDATA_PREFIX
""".split())
# 흔한 혼동의 명시 안내 — difflib가 엉뚱한 키를 고르는 경우(OPENAI_API_BASE → OPENAI_API_KEY)나
# 뜻이 둘로 갈리는 경우. 키를 dict(...) 키워드로 적는 것도 위와 같은 이유(스캐너)다.
_ENV_KEY_HINTS = dict(
    REASONING_EFFORT=(
        "번역은 TRANSLATE_REASONING(off|low|medium|high|xhigh — max 없음), "
        "Q&A는 LLM_REASONING_EFFORT(max 허용)를 쓰세요"
    ),
    OPENAI_API_BASE="번역 엔드포인트 주소는 OPENAI_BASE_URL입니다",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
# 이미 로그로 남긴 안내(키마다 문구가 하나다 — 프로세스 단위). from_env가 다시 불려도
# 같은 WARNING을 반복하지 않는다(반환값·health에는 매번 실린다).
_LOGGED_DOTENV_WARNINGS: set[str] = set()
_LOGGED_LOCK = threading.Lock()
_MAX_CONFIG_WARNINGS = 20


def _is_foreign_env_key(key: str) -> bool:
    upper = key.upper()
    if f"OCR_{upper}" in KNOWN_ENV_KEYS:
        # 이 앱의 키에서 OCR_ 접두사만 빠뜨린 이름(MLX_QUANT_BITS → OCR_MLX_QUANT_BITS,
        # CUDA_GRAPHS → OCR_CUDA_GRAPHS)은 다른 도구의 접두사와 겹쳐도 오타로 안내한다.
        return False
    return upper in _FOREIGN_ENV_KEYS or upper.startswith(_FOREIGN_ENV_PREFIXES)


def _unknown_key_warning(key: str) -> str:
    """모르는 .env 키 → 짧은 한국어 안내 (키 이름만 — 값은 절대 담지 않는다)."""
    head = f"{key}: 이 앱이 읽지 않는 .env 키"
    hint = _ENV_KEY_HINTS.get(key)
    if hint:
        return f"{head} — {hint}"
    close = difflib.get_close_matches(key, sorted(KNOWN_ENV_KEYS), n=2, cutoff=0.75)
    if close:
        return f"{head} — {' 또는 '.join(close)}의 오타인가요?"
    return f"{head} (다른 도구용이면 무시해도 됩니다)"


def unknown_dotenv_key_warnings(values: dict[str, str | None]) -> list[str]:
    """dotenv_values 결과에서 이 앱이 모르는 키의 안내문 목록(키 이름 순서 그대로).

    값이 없는 줄('=' 없는 줄 — 붙여 넣다 남은 토큰일 수 있다)과 환경변수 이름 꼴이 아닌
    키는 보지 않는다 — 안내가 로그·/api/health로 나가므로 비밀이 섞일 여지를 두지 않는다."""
    out: list[str] = []
    for key, value in values.items():
        if value is None or not _ENV_NAME.fullmatch(key or ""):
            continue
        if key in KNOWN_ENV_KEYS or _is_foreign_env_key(key):
            continue
        out.append(_unknown_key_warning(key))
    return out[:_MAX_CONFIG_WARNINGS]


def shadowed_dotenv_key_warnings(path: Path) -> list[str]:
    """값이 있는 줄 뒤에 같은 키의 **빈 줄**이 다시 나온 키 — dotenv는 마지막 줄이 이겨 빈 값
    (미설정)이 적용된다. README의 번역 블록을 .env.example 위쪽에 붙여 넣으면 뒤쪽의 빈
    OPENAI_* 줄이 이겨 번역이 '프로바이더 미설정'이 됐는데 아무 안내가 없었다(fresh-user-5).
    키 이름만 담는다 — 값은 로그·/api/health로 내보내지 않는다."""
    from dotenv.parser import parse_stream

    with_value: set[str] = set()
    shadowed: list[str] = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for binding in parse_stream(stream):
                key, value = binding.key, binding.value
                if not key or value is None or not _ENV_NAME.fullmatch(key):
                    continue
                if value.strip():
                    with_value.add(key)
                elif key in with_value and key not in shadowed:
                    shadowed.append(key)
    except (OSError, UnicodeDecodeError):
        return []
    return [
        f"{key}: 값이 있는 줄 뒤에 같은 키의 빈 줄이 다시 나와 빈 값(미설정)이 적용됩니다 — "
        "뒤쪽 줄을 지우거나 주석 처리하세요"
        for key in shadowed
    ][:_MAX_CONFIG_WARNINGS]


def load_dotenv_file(path: Path | None = None) -> list[str]:
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

    반환: 값이 있던 키를 뒤쪽의 빈 줄이 덮어쓴 안내(shadowed_dotenv_key_warnings)와 이 앱이
    읽지 않는 키의 안내문 목록(KNOWN_ENV_KEYS 밖이면서 다른 도구의 키도
    아닌 것 — 키 이름과 오타 후보·별칭 안내만, 값은 담지 않는다). 키마다 프로세스에서 한
    번만 WARNING으로 남기고, Settings.config_warnings → /api/health의 config_warnings로
    보인다. 예전에는 모르는 키를 경고 없이 환경에 넣어 오타·다른 이름의 설정이 조용히
    무시됐다(REASONING_EFFORT — 번역은 TRANSLATE_REASONING을 읽는다).
    """
    if path is None:
        if _env_bool("DISABLE_DOTENV", False):
            return []
        path = _find_dotenv()
    if path is None or not path.is_file():
        return []
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
    warnings = (shadowed_dotenv_key_warnings(path) + unknown_dotenv_key_warnings(values))[
        :_MAX_CONFIG_WARNINGS
    ]
    with _LOGGED_LOCK:
        fresh = [w for w in warnings if w not in _LOGGED_DOTENV_WARNINGS]
        _LOGGED_DOTENV_WARNINGS.update(fresh)
    for warning in fresh:
        logger.warning(".env 설정 확인 (%s): %s", path.name, warning)
    return warnings


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


def _env_int_choice(name: str, default: int, allowed: tuple[int, ...], why: str = "") -> int:
    """허용 목록형 정수 env — 비정수·목록 밖 값은 **변수명을 담은** ValueError로 기동 시 실패한다.

    범위(lo/hi)로는 표현할 수 없는 이산 값(예: 양자화 비트 0|8)용이다. 목록 밖 값을
    조용히 기본값으로 바꾸면 운영자는 설정이 먹었다고 믿는다."""
    value = _env_int(name, default)
    if value not in allowed:
        raise ValueError(
            f"{name}={_env_raw(name)!r}: {' 또는 '.join(map(str, allowed))}만 지원합니다{why}"
        )
    return value


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
    # auto | cpu | cuda | metal | mlx (mps는 metal의 별칭, registry.VALID_DEVICES).
    # 직접 생성(테스트·스크립트)은 하드웨어 탐지 없이 결정적인 cpu, 운영 진입점 from_env는 auto.
    device: str = "cpu"
    dtype: str = "auto"                 # auto | bfloat16 | float16 | float32
    # MLX 엔진(device=mlx) 디코더 인메모리 양자화 비트 — 0=없음(dtype 그대로) | 8.
    # 4비트는 감사 스파이크에서 숫자 오인식(2504→2304)이 측정돼 받지 않는다.
    mlx_quant_bits: int = 0
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
    # 페이지별 출력 내용 문자 hard limit — 레이아웃 태그·HTML 표 태그 제외 (env 0 이하=비활성)
    max_page_output_chars: int | None = 16_384
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
    llm_provider: str = "openai-responses"  # openai-responses | openai-chat | ollama | local-openai
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
    # .env의 모르는 키 안내(load_dotenv_file) — /api/health의 config_warnings로 보인다.
    # 직접 생성(테스트·스크립트)은 .env를 읽지 않으므로 빈 값이다.
    config_warnings: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        # 로컬 실행(Metal 등)에서도 .env의 번역/OCR 설정이 잡히게 — 모르는 키 안내는 health로.
        # (None을 돌려주는 대역 — 예전 시그니처 — 도 받는다)
        dotenv_warnings = tuple(load_dotenv_file() or ())
        frontend = os.environ.get("FRONTEND_DIR")
        # 미설정(빈 값 포함) = auto: unlimited 엔진이 mlx → cuda → metal → cpu 중 쓸 수 있는
        # 첫 디바이스를 고른다(registry.resolve_auto_device). compose는 backend 서비스마다
        # OCR_DEVICE를 명시하고 Dockerfile에는 기본값이 없다 — CPU 이미지는 auto여도 cpu다.
        # 바뀌는 것은 로컬 실행뿐: Apple Silicon의 `make dev`가 조용히 CPU fp32로 돌던 함정
        # 대신 MLX(없으면 torch MPS)를 쓴다. .env의 OCR_DEVICE는 그대로 존중된다.
        device = (_env_raw("OCR_DEVICE") or "auto").strip().lower()
        return cls(
            device="metal" if device == "mps" else device,
            dtype=os.environ.get("OCR_DTYPE", "auto").strip().lower(),
            mlx_quant_bits=_env_int_choice(
                "OCR_MLX_QUANT_BITS", 0, (0, 8),
                " (0=양자화 없음, 8=MLX 디코더 8비트 — 4비트는 숫자 오인식으로 미지원)",
            ),
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
            config_warnings=dotenv_warnings,
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
