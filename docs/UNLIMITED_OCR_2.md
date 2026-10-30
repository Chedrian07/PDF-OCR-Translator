# Unlimited-OCR-2 운영 및 검증

이 문서는 v2 전환 이후 OCR 설치·운영의 기준이다. 과거 v1/Ovis/Paddle 측정값은
새 모델의 성능·품질 근거로 사용하지 않는다.

## 고정 배포물

- 모델: `baidu/Unlimited-OCR-2`
- 코드·토크나이저·safetensors revision: `c14071b549f4fcab10909f3c4acc6a2edb366947`
- CUDA: 공식 `sglang-0.5.20.dev13+g75b9b06b02` wheel
- wheel SHA-256: `576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea`
- Python 3.12. nvcc wheel의 toolkit을 `CUDA_HOME`으로 지정하고 JIT 링크용 `lib64`/`libcudart.so` 경로를 이미지에서 구성한다.
- 이미지 digest와 전체 Python 의존성 해시는
  `services/unlimited_ocr2/Dockerfile`, `requirements.lock`에 고정했다.
- 공식 wheel의 torch 2.13 / torchaudio 2.11 의존성 충돌은 `overrides.txt`로
  torch를 공식 OCR 버전에 유지한다. 오디오 추론은 이 서비스에서 사용하지 않는다.

출처: [공식 모델 배포](https://huggingface.co/baidu/Unlimited-OCR-2/tree/c14071b549f4fcab10909f3c4acc6a2edb366947).
원격 코드를 실행하는 대신 CPU·Metal은 저장소의 검토 가능한 벤더 코드를 읽는다.
SGLang은 고정 wheel의 모델 구현을 사용한다.

## 실행

```bash
cp .env.example .env
# CPU
make setup
docker compose up -d --build ocr-cpu
# CUDA (웹 :8001, 내부 SGLang :10000)
docker compose --profile cuda up -d --build unlimited-ocr2 ocr-cuda
# Apple Silicon: MLX 기본, 명시적 torch Metal도 동일 모델
make setup-mlx
make dev
OCR_DEVICE=metal make dev
```

CUDA를 로컬에서 실행할 때는 웹과 모델 런타임을 분리한다.

```bash
uv venv .venv-sglang --python 3.12
uv pip sync --python .venv-sglang/bin/python --require-hashes services/unlimited_ocr2/requirements.lock
.venv-sglang/bin/python services/unlimited_ocr2/launch.py
# 다른 터미널
cd backend
DISABLE_DOTENV=1 OCR_ENGINE=unlimited2 OCR_DEVICE=cuda \
  OCR_SIDECAR_URL=http://127.0.0.1:10000 uv run uvicorn app.main:app --port 8000
```

로컬 SGLang은 기본적으로 루프백에만 바인딩합니다. Compose의 모델 포트는 내부 서비스 전용이며 호스트에 공개하지 않습니다.

16GB VRAM 기본값은 동시 요청 1, 정적 메모리 비율 0.65, prefill 2048 토큰이다.
`SGLANG_CONCURRENCY`, `SGLANG_MEM_FRACTION`을 실제 여유 VRAM에 맞춰 조정한다.
기본 양자화는 없으며 multi는 1024 전역 뷰, single은 공식 1024/640 타일링을 쓴다.
Compose의 `hf-cache-v2`는 이전 볼륨과 분리된다. 로컬 Hub 캐시도 모델 저장소 ID로 분리된다.

`OCR_ENGINE=unlimited`, `ovisocr2`, `paddleocr_vl`, `textlayer`는 기동 오류다.
구형 MoE/fast-decode/Tesseract 설정도 제거하고 `.env.example`로 옮겨야 한다.
`fake`는 `OCR_TEST_MODE=1`을 명시하는 테스트 하네스 전용이다.
기존 업로드·조회·SSE·다운로드 URL은 유지된다. health 및 작업 메타데이터의
`model_revision`, `runtime`으로 실제 사용 경로를 확인한다.

## 추론 및 실패 동작

공식 `<PAGE><image>` 반복 프롬프트, page2d RoPE, SR-SWA 라우팅,
35-gram 억제(멀티 1024 / 단일 128 창)를 사용한다. `<BLOCK>`/`<LEVELn>`/`<FORK>`
출력은 기존 좌표·그림·Markdown 계약으로 변환한다. 토큰 스트림에서도 분할된 헤더를 버퍼링한다.

각 문서는 독립 요청/캐시를 사용한다. 취소·시간 초과에는 스트림을 닫고 해당 request ID만
abort한다. SGLang의 내부 문서 창 세션은 wheel의 disconnect/finally 경로가 회수한다.
CPU·Metal은 직렬화된 공식 생성 루프가 DASH 상태를 복원하며, MLX 캐시는 요청마다 새로 만든다.

출력 길이 제한은 `OutputLimitError`로 페이지별 재처리에 연결한다. 원본 텍스트 레이어는
충실도 비교와 페이지 정렬에만 사용한다. 실패한 모델 출력은 원본 텍스트/Tesseract로
대체하지 않는다. 실패 페이지는 경고와 자리표시자를 유지하며 전부 실패하면 작업이 실패한다.

## WSL NVIDIA 진단

이번 환경에서는 `/usr/lib/wsl/lib/libnvidia-ml.so.1`의 ELF 재배치 중 SIGSEGV가 발생했다.
`/dev/dxg`는 존재했고 Docker NVIDIA runtime도 등록되어 있었다. 설치된 드라이버 디렉터리의
라이브러리를 **해당 프로세스에만** 지정하면 GPU 조회와 CUDA 실행이 복구됐다.

```bash
# 실제 설치된 nv*.inf 디렉터리를 확인한 뒤 사용한다. 시스템 파일을 덮어쓰지 않는다.
LD_LIBRARY_PATH=/usr/lib/wsl/drivers/<installed-nvidia-driver>:/usr/lib/wsl/lib nvidia-smi
```

## 검증 명령

```bash
cd backend
uv run --extra cpu pytest
cd ../frontend
npm test
npm run test:e2e-mock
E2E_BASE_URL=http://127.0.0.1:8001 npm run test:e2e
```

실모델 비교는 `scripts/benchmark_unlimited2.py`로 PDF를 같은 DPI로 렌더하고 준비 실행
1회 후 3회를 측정한다. JSON에는 입력·래스터 해시, 실제 GPU, OCR 시간 중앙값,
첫 텍스트 시간, 페이지 처리량, VRAM 최대값, 페이지 마커와 그림 수를 기록한다.
`--baseline`은 변경 전 체크아웃의 `PYTHONPATH`와 v1 CUDA 환경에서만 사용한다.
품질 평가는 동일 평가셋의 정답·표 구조·수식·읽기 순서·좌표와 함께 수행해야 한다.
단순 구조 개수나 텍스트 레이어 유사도만으로 품질 비퇴행을 판정하지 않는다.

Apple Silicon 실제 추론은 사용자가 수행하기로 했다. 다음 명령은 실가중치를 사용한다.

```bash
cd backend
OCR_MPS_TESTS=1 uv run --extra metal --extra mlx pytest tests/test_mps_contract.py
OCR_DEVICE=mlx DISABLE_DOTENV=1 uv run --extra metal --extra mlx uvicorn app.main:app
# 별도 터미널에서 E2E_BASE_URL=http://127.0.0.1:8000 npm --prefix frontend run test:e2e
```

MLX sparse 연산은 `tests/test_unlimited2_sparse.py`가 torch reference와 실제 배열 연산을
비교한다. Linux에서는 선택적으로 `mlx[cpu]==0.32.3`을 설치해 실행할 수 있다.
이 수치 검증은 Apple GPU 실검증을 대신하지 않는다.
