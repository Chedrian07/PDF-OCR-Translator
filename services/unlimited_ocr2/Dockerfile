FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3
COPY --from=uv /uv /bin/uv
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libnuma1 libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
COPY services/unlimited_ocr2/requirements.lock /srv/requirements.lock
RUN uv venv /venv && uv pip sync --python /venv/bin/python --require-hashes /srv/requirements.lock
COPY services/unlimited_ocr2/launch.py /srv/launch.py
RUN useradd --uid 1000 --create-home app && mkdir -p /data/hf && chown -R app:app /data
# The pinned nvcc wheel supplies the toolkit; DeepGEMM discovers it through CUDA_HOME.
ENV CUDA_HOME=/venv/lib/python3.12/site-packages/nvidia/cu13
# SGLang JIT links -L$CUDA_HOME/lib64 -lcudart; PyPI supplies lib/libcudart.so.13.
RUN ln -s lib "$CUDA_HOME/lib64" \
    && ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so" \
    && echo "$CUDA_HOME/lib" > /etc/ld.so.conf.d/uocr-cuda.conf \
    && ldconfig
ENV PATH="/venv/bin:/venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
    HF_HOME=/data/hf PYTHONUNBUFFERED=1 SGLANG_HOST=0.0.0.0
USER app
EXPOSE 10000
CMD ["python", "/srv/launch.py"]
