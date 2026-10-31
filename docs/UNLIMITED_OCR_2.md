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

## WSL GPU 상태

2026-10-31 재검증에서는 기본 `nvidia-smi`, Docker `--gpus`, 표준 Compose의
`gpus: all`로 실제 CUDA 추론이 성공했다. 드라이버 경로·장치를 수동 지정할 필요가 없다.

2026-10-30에는 WSL NVML 라이브러리 로딩에서 SIGSEGV가 발생해 검증 프로세스에만
설치된 드라이버 디렉터리를 지정했다. 충돌 위치는 확인했지만 그 상태의 근본 원인은
확정하지 못했다. 시스템 드라이버 파일은 수정하지 않았다. 그 우회는 과거 진단 기록이다.

## 검증 명령

```bash
cd backend
uv run --extra cpu pytest
cd ../frontend
npm test
npm run test:e2e-mock
E2E_BASE_URL=http://127.0.0.1:8001 npm run test:e2e
```

실제 번역 공급자까지 쓰는 옵트인 브라우저 검증은 다음 명령이다. `.env`의 등록된
`OPENAI_`/`TRANSLATE_` 설정만 격리된 테스트 백엔드에 전달하며, 공개 합성 문서를 쓴다.
인증 실패는 `blocked-auth`, 종료 코드 2로 기록하고 모의 공급자로 대체하지 않는다.

```bash
OCR_SIDECAR_URL=http://127.0.0.1:10000 npm --prefix frontend run test:e2e-live
```

모델 포트는 Compose 내부 전용이므로, 이 호스트 하네스는 로컬 SGLang 주소나 해당
호스트에서 접근 가능한 컨테이너 주소가 필요하다. 실제 번역 성공 분기는 유효한 공급자
인증 설정이 있어야 실행된다.

수식 구분자가 없는 출력은 제한된 문법으로만 조판을 복구한다. 원 모델 출력은 보존하며
코드·URL·기존 수식은 바꾸지 않는다. 흐린 흰 종이의 모델 입력에는 제한적 명암 확장을,
직사각형 그림에는 네 변의 픽셀 근거에 따른 경계 정제를 적용한다. 원본 이미지와 해상도는
유지한다. [최신 검증 기록](UNLIMITED_OCR_2_VALIDATION.md)에 범위와 실제 지표를 남겼다.

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

## 실모델 CI 선택 실행

일반 push/PR CI는 모델 없는 회귀와 브라우저 하네스를 사용한다. GitHub Actions의
수동 실행에서 `run_real_ocr=true`를 선택한 경우에만 실모델 브라우저 작업을 실행한다.
러너 라벨은 `self-hosted`, `linux`, `x64`, `ocr-gpu`이며 다음 준비가 필요하다.

- Docker Compose와 정상 NVIDIA Container Toolkit/CUDA GPU 접근.
- 기본 설정 기준 VRAM 16GB 및 모델·빌드 캐시를 담을 충분한 RAM/디스크.
- Chromium 실행에 필요한 OS 라이브러리와 Noto Sans/Serif CJK 폰트.

작업은 호스트 패키지나 드라이버를 수정하지 않는다. 실행/재시도 ID별로 고유한 Compose
프로젝트를 만들어 모델만 기동하고, `/ready`를 최대900초 기다린 뒤 실제 CUDA OCR과
모의 번역/Q&A 브라우저 검사를 실행한다. 종료 시 해당 프로젝트의 컨테이너·볼륨만 정리한다.
실제 유료 번역 공급자의 성공 검증은 별도 `test:e2e-live` 하네스에서 수행한다.

## 번역 공급자 호환 처리

공급자가 HTTP400으로 temperature 미지원을 명시하면 같은 모델·추론 설정을 유지하고
해당 필드를 한 번만 생략 재시도한다. Responses의 문자열 input 축약을 거절하고 메시지
목록을 요구하면 동일한 instructions/원문을 정식 user 메시지 목록으로 한 번만 재시도한다.
성공한 호환 설정은 클라이언트에 기억하고 상태/리포트의 `temperature_omitted`,
`responses_input_list`와 안내문에 기록한다. 명시적인 reasoning=off가 지켜지지 않는 경우는
조용히 무시하지 않고 설정 오류로 알린다. 사용자 `.env`나 선택 모델을 자동 변경하지 않는다.
