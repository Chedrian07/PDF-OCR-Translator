# 빌드 컨텍스트: 리포 루트 (docker-compose.yml 참조)
# ARG TORCH_VARIANT=cpu | cu129  → pyproject의 extra 선택
#
# CUDA 이미지도 python:3.12-slim 기반이다: torch cu129 휠이 CUDA 런타임을
# 동봉하므로 호스트 드라이버(+ nvidia container runtime)만 있으면 된다.

ARG TORCH_VARIANT=cpu

# ── 1단계: 의존성 + 네이티브 모듈 빌드 ─────────────────────────
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/venv \
    UV_PYTHON_DOWNLOADS=never

ARG TORCH_VARIANT
WORKDIR /src/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra "${TORCH_VARIANT}"

COPY native /src/native
RUN uv pip install -p /venv/bin/python /src/native

# ── 2단계: 런타임 ──────────────────────────────────────────────
FROM python:3.12-slim-bookworm

# tesseract-ocr(+eng/kor): textlayer 엔진의 스캔 페이지 OCR 폴백 (OCR_LANGUAGES=eng+kor)
# fonts-noto-cjk: 번역 PDF 내보내기(pdf_export._SYSTEM_FONT_CANDIDATES)의
#   Noto Serif/Sans CJK — 없으면 PyMuPDF 내장 비임베드 Dotum 폴백으로
#   전각 조판(라틴 자간 파손)이 된다
# fontconfig: 정적 폰트 후보가 전부 없을 때의 fc-list 런타임 탐색
#   (pdf_export._fontconfig_candidates)용
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
        fonts-noto-cjk fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home app \
    && mkdir -p /data/hf && chown -R app:app /data

COPY --from=builder /venv /venv
COPY --chown=app:app backend/app /srv/backend/app
COPY --chown=app:app frontend /srv/frontend

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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
