# PaddleOCR-VL-1.6 CUDA sidecar — RTX 5070 Ti (Blackwell, sm_120) 대상
#
# 공식 NVIDIA Blackwell 가이드의 wheel 경로를 따른다
# (https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html):
#   1) paddlepaddle-gpu (공식 cu129 인덱스 — 가이드 최소 3.2.1, 여기서는 3.3.1 고정)
#   2) paddleocr[doc-parser] 3.6.x (PaddleOCR-VL-1.6 동봉 버전)
# 공식 sm120 Docker 이미지(ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:
# paddleocr3.6-nvidia-gpu-sm120 계열)도 대안이다 — 접근성이 낮아(중국 레지스트리)
# 재현 가능한 자체 빌드를 기본으로 한다 (docs/PADDLEOCR_VL_BLACKWELL_5070TI.md).
#
# 요구: 호스트 NVIDIA driver가 CUDA 12.9+ 지원, nvidia container toolkit.
# 대상 플랫폼: linux/amd64 (requirements.lock의 paddlepaddle-gpu 휠이 x86_64 전용).
# HF 토큰은 빌드 레이어에 넣지 않는다 — compose가 runtime env로만 전달.

# 베이스는 digest로 고정한다 — 같은 태그를 다시 빌드해도 OS 레이어가 바뀌지 않는다.
# 갱신: `docker buildx imagetools inspect python:3.12-slim-bookworm`의 Digest로 교체한다
# (backend/Dockerfile과 같은 값을 쓴다). 고정만 하면 OS 보안 패치가 멈추므로 아래
# apt-get upgrade와 반드시 함께 간다.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 curl ccache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/paddleocr_vl

# PaddlePaddle GPU + PaddleOCR + sidecar 의존성 — 전이 의존성까지 해시로 고정한 실제
# 잠금(requirements.lock, 원천은 requirements.in)에서만 설치한다:
#   --require-hashes : 해시가 다른 아티팩트는 설치 실패(미러·CDN 변조 차단)
#   --no-deps        : lock 밖 패키지를 끌어오지 않는다(누락은 import 오류로 즉시 드러남)
#   --only-binary    : 빌드 시점 소스 빌드 금지(빌드 의존성은 해시 검증 밖이다)
# paddlepaddle-gpu는 PyPI에 없는 cu129 빌드라 lock이 공식 CDN 휠 URL을 직접 고정한다.
# pip 캐시 마운트: lock 한 줄만 바뀌어도 ~6GB(paddle·CUDA 휠)를 다시 받지 않게 한다.
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes --no-deps --only-binary :all: -r requirements.lock

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    HOME=/home/app \
    PADDLE_PDX_MODEL_SOURCE=huggingface

# 비루트 실행 — 이 컨테이너는 사용자가 올린 임의 PDF의 렌더 이미지를 파서에 먹이는
# 쪽이라 backend보다 신뢰 경계가 바깥이다. uid는 backend/Dockerfile과 동일한 1000.
# ⚠ PaddleX 캐시는 $HOME/.paddlex이므로 root가 아닌 /home/app/.paddlex가 된다 —
#   docker-compose.yml의 paddle-x-cache 마운트 경로도 함께 바뀌었다.
# ⚠ 볼륨 소유권: 비어 있는 named volume은 이 chown 결과(uid 1000)를 물려받지만,
#   이미 root 소유로 채워진 기존 볼륨은 자동으로 바뀌지 않는다. 한 번만 — 볼륨 이름을
#   추측하지 말고 compose로 실행한다(실제 이름은 `<PROJECT>_paddle-hf-cache`처럼 접두사가
#   붙어 `docker run -v paddle-hf-cache:…`는 빈 볼륨을 새로 만들고 exit 0으로 끝난다):
#     docker compose --profile paddle run --rm --no-deps --user 0 --cap-add CHOWN \
#       --cap-add DAC_OVERRIDE --entrypoint chown paddleocr-vl \
#       -R 1000:1000 /data/hf /home/app/.paddlex
#   (compose가 cap_drop: [ALL]을 걸어 두므로 root라도 이 실행에만 두 캡을 돌려준다)
RUN useradd --uid 1000 --home-dir /home/app --create-home app \
    && mkdir -p /data/hf /home/app/.paddlex \
    && chown -R 1000:1000 /data /home/app /srv/paddleocr_vl
USER 1000

EXPOSE 8080

# 최초 기동은 모델 다운로드(~1GB + layout 모델)가 포함될 수 있어 start-period를 길게
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=5 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
