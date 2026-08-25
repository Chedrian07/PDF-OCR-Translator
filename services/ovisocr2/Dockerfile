# OvisOCR2 CUDA sidecar — RTX 5070 Ti (Blackwell, sm_120) 대상
#
# 베이스: vLLM 공식 이미지 v0.22.1 (모델 카드 공식 권장 버전), cu129 태그.
#   - Blackwell은 CUDA >= 12.8 필요 — cu129 태그가 sm_120 커널을 포함한다.
#   - 이미지에 torch(cu129)·vllm==0.22.1·fastapi·uvicorn·pillow가 동봉되어 있어
#     추가 의존성은 requirements.lock의 최소 항목뿐이다.
#   - `latest` 태그 금지 — 버전 고정 (docs/OVISOCR2_CUDA_5070TI.md).
#
# HF 토큰은 빌드 레이어에 절대 넣지 않는다 — compose가 runtime env로만 전달.

FROM vllm/vllm-openai:v0.22.1-cu129

WORKDIR /srv/ovisocr2

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    HOME=/home/app

# 비루트 실행 — 이 컨테이너는 사용자가 올린 임의 PDF의 렌더 이미지를 파서에 먹이는
# 쪽이라 backend보다 신뢰 경계가 바깥이다. uid는 backend/Dockerfile과 동일한 1000.
# ⚠ 볼륨 소유권: 이미지 안에서 /data/hf를 미리 만들고 chown해야 Docker가 **비어 있는**
#   named volume을 초기화할 때 uid 1000 소유로 만든다. 이미 root 소유 내용이 들어 있는
#   기존 볼륨(ovis-hf-cache)은 자동으로 바뀌지 않으므로 한 번만 손봐야 한다:
#     docker run --rm -v ovis-hf-cache:/v alpine chown -R 1000:1000 /v
#   (모델을 다시 받아도 되면 `docker volume rm ovis-hf-cache`로 대체 가능)
# 베이스(vLLM) 이미지가 이미 uid 1000을 갖고 있을 수 있어 생성은 조건부로,
# USER는 이름이 아닌 uid로 지정한다. HOME은 위 ENV가 passwd 항목보다 우선한다.
RUN if ! getent passwd 1000 >/dev/null; then \
        useradd --uid 1000 --home-dir /home/app --create-home app; \
    fi \
    && mkdir -p /data/hf /home/app \
    && chown -R 1000:1000 /data /home/app /srv/ovisocr2
USER 1000

EXPOSE 8080

# 최초 기동은 모델 다운로드(~1.7GB)+로드가 포함될 수 있어 start-period를 길게 둔다.
# health는 로드 전에도 200(model_loaded=false)을 반환하므로 여기서는 연결성만 본다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=5 \
    CMD python3 -c "import urllib.request,sys; r=urllib.request.urlopen('http://localhost:8080/health', timeout=4); sys.exit(0 if r.status==200 else 1)" || exit 1

# 베이스 이미지의 OpenAI 서버 entrypoint를 비우고 sidecar 앱으로 대체
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
