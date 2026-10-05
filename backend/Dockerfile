# 빌드 컨텍스트: 리포 루트 (docker-compose.yml 참조)
# ARG TORCH_VARIANT=cpu | cu129  → pyproject의 extra 선택
#
# CUDA 이미지도 python:3.12-slim 기반이다: torch cu129 휠이 CUDA 런타임을
# 동봉하므로 호스트 드라이버(+ nvidia container runtime)만 있으면 된다.

ARG TORCH_VARIANT=cpu

# ── 베이스 이미지 고정 ─────────────────────────────────────────
# 태그는 다시 밀린다 — digest(멀티 아키텍처 인덱스)까지 고정해 같은 커밋이 같은 OS·
# 툴체인으로 빌드되게 한다(오프라인 tar.gz와 레지스트리 이미지의 내용 일치).
# 갱신: `docker buildx imagetools inspect <이미지:태그>`의 Digest로 교체하고 릴리스
# 노트에 남긴다. digest 고정만 하면 OS 보안 패치가 멈추므로 런타임 단계의
# apt-get upgrade와 반드시 함께 간다(Dependabot docker 생태계가 갱신 PR을 연다).
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

# ── 1단계: 의존성 + 네이티브 모듈 빌드 ─────────────────────────
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS builder
COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# UV_COMPILE_BYTECODE: /venv를 빌드 시점에 .pyc로 컴파일한다 — 런타임 사용자(app)는 root
#   소유 /venv에 __pycache__를 못 써서, 미리 컴파일하지 않으면 기동·OOM 재기동마다
#   torch·transformers를 메모리에서 다시 컴파일했다.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

ARG TORCH_VARIANT
WORKDIR /src/backend
COPY backend/pyproject.toml backend/uv.lock ./
# --locked(≠ --frozen): lock이 pyproject와 어긋나면 **빌드에서 즉시 실패**한다.
# --frozen은 실패하지 않고 lock에 없는 의존성을 빠뜨린 채 이미지를 만들어,
# 컨테이너 기동 시 ImportError로만 드러난다.
RUN uv sync --locked --no-dev --no-install-project --extra "${TORCH_VARIANT}"

COPY native /src/native
RUN uv pip install -p /venv/bin/python /src/native

# ── 2단계: 런타임 ──────────────────────────────────────────────
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

# tesseract-ocr(+eng/kor): textlayer 엔진의 스캔 페이지 OCR 폴백 (OCR_LANGUAGES=eng+kor)
# fonts-noto-cjk: 번역 PDF 내보내기(pdf_export._SYSTEM_FONT_CANDIDATES)의
#   Noto Serif/Sans CJK — 없으면 PyMuPDF 내장 비임베드 Dotum 폴백으로
#   전각 조판(라틴 자간 파손)이 된다
# fontconfig: 정적 폰트 후보가 전부 없을 때의 fc-list 런타임 탐색
#   (pdf_export._fontconfig_candidates)용
# apt-get upgrade: 베이스 스냅샷 이후 나온 Debian 보안 업데이트를 받는다 — 0.1.0
#   이미지에는 수정판이 이미 있던 libexpat1 HIGH 6건이 그대로 남아 있었다.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
        fonts-noto-cjk fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home app \
    && mkdir -p /data/hf && chown -R app:app /data

# 코드는 root 소유·읽기 전용으로 둔다 — 실행 사용자(app)가 쓸 수 있는 곳은 /data
# (볼륨)뿐이다. 예전 --chown=app:app은 파일 쓰기 취약점 하나로 무인증 사용자 전원에게
# 서빙되는 app.js(지속형 XSS)나 app/*.py(재시작 뒤에도 남는 코드 실행)를 바꿀 수 있게 했다.
COPY --from=builder /venv /venv
COPY backend/app /srv/backend/app
COPY frontend /srv/frontend
# 권한 정규화: 빌드 컨텍스트의 권한(개발자 umask 077이면 600)을 그대로 받으면 app이
# 코드를 못 읽는다 — 모두 읽기 가능·쓰기는 root만으로 맞춘다. 앱 코드도 미리 컴파일한다
# (/venv와 같은 이유 — 실행 사용자는 __pycache__를 못 쓴다).
RUN chmod -R a+rX,go-w /srv/backend/app /srv/frontend \
    && /venv/bin/python -m compileall -q /srv/backend/app

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    HF_HOME=/data/hf \
    FRONTEND_DIR=/srv/frontend

USER app
WORKDIR /srv/backend
EXPOSE 8000

# 모델 다운로드(~6.7GB) + 로드가 첫 기동에 포함될 수 있어 start-period를 길게 둔다
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

# --timeout-graceful-shutdown: 열린 SSE(잡·번역 진행 스트림)는 스스로 끝나지 않아, 상한이
# 없으면 `docker compose stop/restart`가 'Waiting for connections to close'에서 멈췄다가
# 10초 유예 뒤 SIGKILL로 죽고 lifespan 종료(워커 정지·잡 저장소 락 해제)가 돌지 않았다.
# 5초 뒤 남은 연결을 끊고 lifespan 종료를 마친다(Docker 기본 유예 10초 안).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "5"]
