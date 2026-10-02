# PDF OCR Translator — PDF 그대로 읽고 번역하기

> 이 리포는 **Unlimited-OCR(PDF→Markdown)** 와 **Localight(로컬 우선 논문 리더)** 를
> 병합한 워크스페이스입니다 — textlayer 엔진과 페이지 Q&A('질문' 탭)가 Localight에서 이식되었습니다.

[![CI](https://github.com/Chedrian07/PDF-OCR-Translator/actions/workflows/ci.yml/badge.svg)](https://github.com/Chedrian07/PDF-OCR-Translator/actions/workflows/ci.yml)

CI는 push/PR마다 backend pytest(+커버리지 아티팩트) · backend×네이티브 통합 · ruff(backend·services) ·
frontend `node --test` · native 패리티 · sidecar 테스트 · 의존성 감사(pip-audit) · CPU 이미지
빌드와 하드닝 스모크(amd64·arm64) · 실 PDF 전 구간 하네스(`verify-e2e`)를 실행합니다. 러너 시간이
큰 hermetic 브라우저 E2E(`e2e-mock` — mock LLM + FakeEngine)는 PR을 막지 않고 main
push·nightly(18:00 UTC)·수동 실행에서만 돕니다 — 릴리스는 태그 커밋의 push CI 성공을 요구하므로
사실상 릴리스 게이트입니다. 배지 URL은 이 리포의 원격
(`Chedrian07/PDF-OCR-Translator`) 기준입니다 — 포크·이전했다면 두 URL의 경로를 바꾸세요.

웹에서 PDF를 업로드하면 로컬 비전-언어 OCR 모델로 **이미지(figure)까지 추출된
Markdown**을 만들어 주는 셀프호스팅 서비스입니다. 기본 엔진은
[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)(3.3B MoE, MIT)이고,
단일 RTX 5070 Ti(16GB) 기준으로 **엔진을 선택**할 수 있습니다:

| 엔진 | 선택 기준 | 실행 |
|---|---|---|
| **Unlimited-OCR** (기본) | 멀티페이지 문맥 · 실시간 토큰 스트리밍 | `docker compose up -d --build ocr-cuda` → :8001 |
| **OvisOCR2** (0.9B, Apache-2.0) | 속도(25쪽 실문서 3.3 s/p) · figure bbox | `docker compose --profile ovis up -d --build ovisocr2 ocr-ovis` → :8002 |
| **PaddleOCR-VL-1.6** (0.9B, Apache-2.0) | **한국어 정확도** · 표/수식 · 완전한 layout | `docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle` → :8003 |
| **textlayer** (모델 불필요, Localight 이식) | **텍스트 PDF 최적** · CPU 전용 · 즉시 시작 | `OCR_ENGINE=textlayer docker compose up -d --build` → :8000 |

> **Apple Silicon Mac**에서는 같은 Unlimited-OCR을 **in-process MLX 엔진**으로 로컬
> 실행합니다 — `make setup-mlx && make dev` (§Apple Silicon). torch MPS는 폴백입니다.

> RTX 5070 Ti 실측 비교(속도·VRAM·한국어 오독 사례)와 Apple Silicon(M4 Max) 실측은
> [docs/OCR_BENCHMARK.md](docs/OCR_BENCHMARK.md) 참조 — 한국어 문서는
> PaddleOCR-VL, 영문·속도 우선은 OvisOCR2가 유리했습니다.

> **textlayer**는 모델 다운로드 없이 PDF 내장 텍스트 레이어를 우선 추출하고,
> 텍스트가 부족한(스캔) 페이지만 Tesseract OCR로 폴백하는 경량 엔진입니다
> (레이아웃 뷰 지원 · figure 크롭 없음). 로컬(uv) 실행 시에는 Tesseract 설치가
> 필요합니다 — macOS: `brew install tesseract tesseract-lang`. Docker 이미지에는
> `eng+kor` 데이터가 포함되어 있습니다.

⚠ **한 시점에 GPU 스택 하나만** 기동하세요 (단일 16GB GPU — VRAM 경쟁 시 OOM).
신규 엔진은 GPU 전용 sidecar 컨테이너로 격리되어 메인 backend의 Python 환경을
오염시키지 않습니다. 상세: [docs/CUDA_5070TI_MULTI_OCR_PLAN.md](docs/CUDA_5070TI_MULTI_OCR_PLAN.md) ·
[docs/OVISOCR2_CUDA_5070TI.md](docs/OVISOCR2_CUDA_5070TI.md) ·
[docs/PADDLEOCR_VL_BLACKWELL_5070TI.md](docs/PADDLEOCR_VL_BLACKWELL_5070TI.md)

- **PDF 속 이미지 완벽 처리**: 모델의 그라운딩 박스(`<|ref|>image<|/ref|><|det|>…`)로
  figure를 원본에서 크롭해 `images/`에 저장하고 마크다운에 `![](images/…)`로 연결
- **GIF 스타일 3-패널 라이브 뷰**: 변환 중 ① 원본 페이지 위 실시간 레이아웃
  박스 오버레이(그라운딩 좌표) ② RAW OUTPUT 토큰 스트림 ③ 실시간 렌더 미리보기가
  동시에 흐르고, STOP 버튼으로 중단해도 부분 결과가 보존됨
  (공식 데모 GIF의 long-horizon 파싱 경험 재현). 최초 연결이 늦거나 SSE가 자동
  재연결돼도 서버의 실행 중 토큰 replay로 세 패널을 같은 원문에서 재동기화함
- **전체 화면 논문 뷰어**: 완료 결과에서 `논문 뷰어 열기`를 누르면 페이지
  썸네일·목차 / 원문 PDF 좌표면 / 원문 위치와 연결된 번역문을 3열로 표시합니다.
  `?viewer=1&page=N&lang=ko#잡ID` 딥링크, 키보드 페이지 이동, 패널 접기,
  원문-번역 블록 1:1 강조를 지원하며 좌측 기준면은 항상 원문 PDF로 고정됩니다.
- **디바이스 백엔드**: CPU / CUDA / Apple Silicon(**MLX 기본**, torch MPS 폴백) —
  `OCR_DEVICE`를 비워 두면(auto) 쓸 수 있는 가장 빠른 디바이스를 고릅니다
- **한국어 번역 (선택)**: 변환 결과를 OpenAI 호환 API로 한국어 번역 — 로컬 MLX 서버
  (oMLX·LM Studio·mlx_lm.server)도 지원. §한국어 번역 참조
- **원커맨드 배포**: `docker compose up` 하나로 끝 (Apple Silicon GPU는 로컬 실행 — 아래 참조)

## 0.1.0 릴리스 이미지

> **주의 — 0.1.0은 이 README가 설명하는 보안 수정 이전의 이미지입니다.** PyMuPDF 1.28.2
> (MuPDF CVE-2026-3308)·Pillow 12.3.0 갱신, PyMuPDF 워커 프로세스 격리, 업로드 복잡도
> 게이트(압축·중첩 XObject 폭탄 거부)가 없어, 악의적인 PDF 한 장으로 서버가 멈추거나 MuPDF
> 취약점에 닿을 수 있습니다. MLX 엔진·`local-openai` Q&A 공급자·번역 스트리밍
> (`TRANSLATE_STREAM`·`TRANSLATE_REASONING_STYLE`)도 없습니다. 출처를 모르는 PDF를 다룬다면
> 아래 §빠른 시작 (Docker)의 소스 빌드(`docker compose up -d --build`)를 쓰거나 다음 릴리스를
> 기다리세요.

[v0.1.0 릴리스](https://github.com/Chedrian07/PDF-OCR-Translator/releases/tag/v0.1.0)의
CPU 이미지는 Linux amd64/arm64를 지원합니다. 모델 없이 PDF 내장 텍스트와
Tesseract를 사용하는 textlayer 엔진으로 바로 시작할 수 있습니다:

```bash
docker run -d --name pdf-ocr-translator --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e OCR_ENGINE=textlayer -e ALLOWED_HOSTS=localhost,127.0.0.1 \
  -v pdf-ocr-translator-data:/data \
  ghcr.io/chedrian07/pdf-ocr-translator:0.1.0-cpu
# → http://localhost:8000
```

릴리스 첨부파일의 `pdf-ocr-translator-0.1.0-cpu-amd64.tar.gz` 또는
`pdf-ocr-translator-0.1.0-cpu-arm64.tar.gz`를 받아 오프라인으로 설치할 수도 있습니다.
GHCR 이미지 접근 권한이 없어도 이 방법을 사용할 수 있습니다. 다운로드한
아키텍처의 이름을 사용해 이미지를 불러오고 위 실행 명령의 공통 태그를 붙입니다:

```bash
docker load -i pdf-ocr-translator-0.1.0-cpu-amd64.tar.gz
docker tag ghcr.io/chedrian07/pdf-ocr-translator:0.1.0-cpu-amd64 \
  ghcr.io/chedrian07/pdf-ocr-translator:0.1.0-cpu
# arm64를 받았다면 두 명령의 amd64를 arm64로 바꿉니다.
```

번역·Q&A를 활성화하려면 실행 명령에 `--env-file .env`를 추가합니다. `.env`의
설정 예시는 아래 한국어 번역·페이지 Q&A 절을 참조하세요 — 단 `LLM_PROVIDER=local-openai`·
`LLM_LOCAL_OPENAI_*`·`TRANSLATE_STREAM`·`TRANSLATE_REASONING_STYLE`·`TRANSLATE_EXTRA_BODY`는
0.1.0 이후 설정입니다(0.1.0은 `LLM_PROVIDER=local-openai`면 설정 오류로 기동하지 못해 재시작을
반복하고, 나머지 키는 무시합니다). `docker run --env-file`은
따옴표와 줄 끝 주석을 값에 그대로 넣으므로 `KEY=값` 형식으로만 적습니다
(`.env.example`은 설명을 별도 줄에 둡니다). CPU Unlimited-OCR을
사용하려면 `OCR_ENGINE=unlimited`로 바꿉니다. CUDA·sidecar 배포는 아래
소스 빌드용 Compose 프로필을 사용합니다.

버전별 변경사항은 [CHANGELOG.md](CHANGELOG.md)에 있습니다.

### 릴리스 절차 (다음 릴리스부터 — 0.1.0에는 적용되지 않음)

릴리스 워크플로는 같은 커밋의 CI 통과를 확인하고(진행 중이면 최대 90분 기다린다) 각
아키텍처 이미지에서 실제 PDF 업로드→OCR→Markdown/ZIP 다운로드를 검증한 뒤, `trivy`
게이트(수정판이 있는 HIGH/CRITICAL이면 중단 — 수용 목록 `.github/trivyignore.yaml`,
2027-04-01 만료)를 통과한 이미지만 배포합니다. 0.1.0은 이 게이트를 도입하기 전에
배포됐습니다. 오프라인 설치 tarball과 `SHA256SUMS`는 워크플로가 릴리스에 직접
첨부합니다(릴리스가 없으면 초안을 만든다). 베이스 이미지 digest는
`docker buildx imagetools inspect`로 갱신하고, Dependabot이 갱신 PR을 엽니다.

## 빠른 시작 (Docker)

```bash
# CPU (기본 서비스 — 클론 직후 .env 없이 이 한 줄로 기동)
docker compose up -d --build
# → http://localhost:8000

# CUDA (NVIDIA GPU + nvidia container toolkit 필요)
docker compose up -d --build ocr-cuda   # 서비스명 지정 → cuda 프로필 자동 활성화
# → http://localhost:8001

# 신규 CUDA 엔진 (RTX 5070 Ti — GPU 스택은 한 번에 하나만!)
python scripts/check_cuda_environment.py               # preflight (드라이버/sm_120/docker GPU)
docker compose --profile ovis up -d --build ovisocr2 ocr-ovis   # → http://localhost:8002
docker compose stop ovisocr2 ocr-ovis                          # 전환 전 반드시 정지
docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle  # → http://localhost:8003
```

- 최초 실행 시 모델 가중치(~6.7GB)를 `hf-cache` 볼륨에 1회 다운로드합니다
  (진행 상황: `docker compose logs -f`). CPU/CUDA 서비스가 캐시를 공유합니다.
- 모델 로딩 여부는 헤더 배지 또는 `GET /api/health`의 `model_loaded`로 확인.
  같은 응답의 `worker_alive`(잡 처리 스레드 생존) · `worker_job_id`·`worker_progress_age_s`
  (실행 중 잡과 마지막 진행 뒤 경과 초 — 잡이 있는데 계속 늘면 멈춘 것) ·
  `translate_available` · `qa_available`로 나머지 구성 상태도 볼 수 있습니다.
- 한국어 번역·페이지 Q&A를 쓰려면 `cp .env.example .env` 후 키를 설정합니다 — 번역은
  `OPENAI_API_KEY`, Q&A는 **별도의** `LLM_OPENAI_API_KEY`입니다(`.env.example`에서
  주석 처리돼 있으니 `#`을 지우고 값을 넣으세요). 아래 §한국어 번역 · §페이지 Q&A 참조.
- backend 서비스(`ocr-cpu`·`ocr-cuda`·`ocr-ovis`·`ocr-paddle`)는 잡 저장소(`ocr-data`
  볼륨)를 공유하므로 **한 번에 하나만** 뜹니다. 두 번째 backend는 실행 중인 잡을
  망가뜨리지 않도록 기동을 거부하니, 스택을 바꿀 때는 먼저 `docker compose stop ocr-cpu`
  처럼 떠 있는 backend를 멈추세요. 로컬(uv) 실행도 같은 `DATA_DIR`로 서버를 둘 띄우거나
  `--workers`를 2 이상으로 줄 수 없습니다.
- `.env` 값은 `docker-compose.yml`의 `environment:`에 적힌 키로만 컨테이너에 들어가고, `.env`
  파일 자체는 컨테이너에 없습니다. 그래서 `/api/health`의 `config_warnings`는 Docker에서 늘 빈
  목록이고, 이 앱이 읽지 않는 키(예: `REASONING_EFFORT`)는 경고 없이 버려집니다. `.env`를
  고쳤으면 아래 명령으로 키 이름을 점검하세요 — 떠 있는 backend의 판정 코드를 그대로 쓰고 값은
  출력하지 않습니다(다른 스택이면 `ocr-cpu`를 떠 있는 backend 서비스명으로 바꿉니다 — sidecar
  `ovisocr2`·`paddleocr-vl` 이미지에는 이 코드가 없습니다).

```bash
docker compose exec -T ocr-cpu python -c '
import sys
from dotenv import dotenv_values
from app.config import unknown_dotenv_key_warnings as check
print("\n".join(check(dotenv_values(stream=sys.stdin))) or "모르는 키 없음")' < .env
```

Docker 없이 로컬(uv)로 바로 시작할 수도 있습니다:

```bash
make setup-mlx && make dev    # Apple Silicon — MLX 엔진 (아래 §Apple Silicon)
make setup && make dev        # Linux — torch CPU. http://127.0.0.1:8000
make dev-textlayer            # 모델 없이 textlayer 엔진으로 기동
```

Intel Mac: 고정한 torch 2.10에 macOS x86_64 휠이 없어 `make setup`(Unlimited-OCR 엔진)은
설치되지 않습니다 — Docker CPU 이미지(`docker compose up -d --build`)나, 의존성만 받는
`cd backend && uv sync` 뒤 `make dev-textlayer`(torch 불필요)를 쓰세요.

전체 타깃은 §Makefile에 있습니다. 로컬 실행은 실행 디렉터리, 없으면 리포 루트의 `.env`
하나를 직접 읽습니다 — 이미 설정된 환경변수가 우선이고, 이 앱이 읽지 않는 키는 기동 로그와
`/api/health`의 `config_warnings`로 알려 줍니다(키 이름만, 값은 남기지 않는다).
예: `REASONING_EFFORT`는 어떤 코드도 읽지 않습니다 — 번역은 `TRANSLATE_REASONING`,
Q&A는 `LLM_REASONING_EFFORT`로 바꾸세요.

### 모델 없이 UI/파이프라인만 체험

```bash
OCR_ENGINE=fake docker compose up -d --build
```

## macOS · Apple Silicon (MLX 기본 · torch MPS 폴백)

Docker(맥의 Linux VM)에는 GPU 패스스루가 없어 Metal을 쓸 수 없습니다. Apple Silicon
Mac(macOS 14+, arm64)에서는 로컬(uv)로 실행하세요. 먼저 필요한 것:

- **uv** — `brew install uv` 또는 [docs.astral.sh/uv](https://docs.astral.sh/uv/) 설치 스크립트
  (없으면 `make setup-mlx`가 `uv: command not found`로 멈춥니다). Python 3.12는 uv가 받습니다.
- **Xcode Command Line Tools**(`xcode-select --install`) — C++ 가속 모듈(`native/`) 빌드용.

```bash
make setup-mlx    # = cd backend && uv sync --extra metal --extra mlx && uv pip install ../native
make dev          # http://127.0.0.1:8000 — OCR_DEVICE=auto → mlx
make dev PORT=8010   # 8000을 다른 서버가 쓰고 있으면 포트만 바꾼다(dev-metal·dev-textlayer도 같다)
```

- **기본 엔진은 in-process MLX**입니다(`backend/app/vendor/unlimited_ocr_mlx` — mlx-vlm 0.7.4의
  Unlimited-OCR 모델 코드를 mlx 단독 의존으로 옮긴 것, MIT). mlx-vlm·torch에 의존하지 않고,
  torch 경로와 **같은 고정 스냅샷**(`ee63731b`)을 변환 없이 읽습니다(HF 캐시도 공유).
  멀티페이지 청크·토큰 스트리밍·레이아웃·figure·충실도 게이트 등 계약은 torch 엔진과 같습니다.
- 첫 기동은 모델(~6.7GB)을 HF 허브 캐시(`HF_HUB_CACHE`, 기본 `$HF_HOME/hub` =
  `~/.cache/huggingface/hub`)의 `models--baidu--Unlimited-OCR`에 받습니다. 다른 곳에 받아 둔
  캐시를 쓰려면 셸이나 `.env`에 `HF_HUB_CACHE=<models--…가 바로 아래 있는 디렉터리>`를 둡니다 —
  `HF_HOME`만 바꾸면 `<HF_HOME>/hub`를 찾아 다시 받습니다. 다운로드를 막으려면
  `HF_HUB_OFFLINE=1`(캐시에 없으면 모델 로드 실패로 알립니다).
  기동 로그의 `OCR_DEVICE=auto → mlx`와 헤더의 `MLX · <칩 이름>` 배지로 확인합니다.
- **`OCR_DEVICE`** (비우면 `auto`): `auto` | `mlx` | `metal`(=`mps`, torch MPS) | `cpu` |
  `cuda`. `auto`는 unlimited 엔진에만 적용되며 mlx → cuda → metal → cpu 순으로 쓸 수 있는
  첫 디바이스를 고릅니다. `make dev OCR_DEVICE=metal`처럼 바꿀 수 있고, `.env`의
  `OCR_DEVICE`도 그대로 존중됩니다. Apple Silicon에서 `cpu`를 고르면 MLX보다 수십 배 느리다는
  경고가 기동 로그에 남습니다.
- **`OCR_MLX_QUANT_BITS`**: `0`(기본, bf16 그대로) | `8`(디코더만 인메모리 8비트 — 파라미터
  6.7→3.9 GB, 약 1.4배 빠름, 재현율 동일). 다른 값은 기동 시 변수명과 함께 실패합니다
  (4비트는 숫자를 잘못 읽어 미지원).
- **torch MPS 폴백**: `make dev-metal`(= `OCR_DEVICE=metal`). mlx가 설치돼 있지 않으면
  `auto`도 MPS를 고릅니다. 결과 비교·우회용입니다.
- ⚠ `make setup`(cpu extra)이나 `uv sync --extra metal` 단독은 mlx와 C++ 모듈을 지웁니다 —
  Apple Silicon에서는 `make setup-mlx`(=`make setup-metal`)를 쓰세요.
- 청크가 끝날 때마다 MLX/MPS 캐시를 비워 유니파이드 메모리를 돌려줍니다. torch MPS 경로는
  잡마다 ObjC 오토릴리스 풀을 비워 긴 세션의 메모리 증가도 막습니다.
- 로컬 번역 서버도 같은 GPU를 씁니다 — OCR과 번역을 동시에 돌리면 둘 다 느려집니다(§한국어 번역).

### 성능 실측 — Apple M4 Max (조용한 머신, 2026-10-02)

다른 측정·빌드 없이 순차로 잰 값입니다(load average 1.8–4.5 — 데스크톱 상주 프로세스만, 앱
엔진을 프로세스 안에서 직접 호출). 측정 조건·전체 표:
[docs/OCR_BENCHMARK.md §Apple Silicon 실측](docs/OCR_BENCHMARK.md).

| 경로 | 측정 | 값 |
|---|---|---|
| MLX bf16 (기본) | 8쪽 청크 OCR | 3.79–3.80 s/쪽 · 288 tok/s · 청크당 TTFT 1.02 s |
| MLX bf16 | 25쪽 논문 잡 전체 (렌더·충실도 게이트·병합 포함) | 96.3 s (3.85 s/쪽, 충실도 재처리 0건) |
| MLX 8비트 (`OCR_MLX_QUANT_BITS=8`) | 8쪽 청크 OCR · 25쪽 잡 전체 | 2.60–2.62 s/쪽 · 427–430 tok/s · 67.0 s (2.68 s/쪽) |
| MLX | 단독 1쪽 (per_page·충실도 재처리) | 2.9–4.5 s (TTFT 0.62 s · 294 tok/s) |
| MLX | 모델 준비 (로드 + 워밍업) | 웜 1.8–2.0 s · 외장 볼륨 콜드 8.8 s |
| MLX | 메모리 | 파라미터 6.67 GB (8비트 3.92 GB) · 8쪽 청크 피크 8.27 GB (8비트 5.52 GB) · 프로세스 phys_footprint 잡 사이 약 7,300 MB, 최고 약 11,600 MB(단독 1쪽 크롭 경로) |
| MLX | 한 서버 프로세스에서 8쪽 잡 5회 연속 | 잡당 31.0 s · RSS 첫 잡 뒤 +320 MB, 이후 4잡 합계 +5 MB |
| MLX | 번역 PDF 빌드가 동시에 돌 때 디코드 | 290 tok/s (빌드 없음 288 tok/s · 예전 서버 내 빌드: 약 70 tok/s) |
| 내보내기 | 25쪽 번역 PDF 빌드 (워커 프로세스) | 8.2 s (0.33 s/쪽) |
| torch MPS 폴백 | 8쪽 청크 OCR · 25쪽 잡 전체 | 9.03–9.13 s/쪽 · 119 tok/s · 231.8 s (9.27 s/쪽) — 개선 전(a9a4400) 34.0 s/쪽 |

품질: 텍스트 레이어 재현율은 MLX bf16·8비트·torch MPS가 모두 0.9748(8쪽 청크)로 같고, 25쪽
전체는 0.9386–0.9388입니다. MLX bf16 출력은 MPS와 토큰 단위로 같지 않으므로(bf16 반올림 차이)
비교는 유사도·재현율로 합니다. `PAGES_PER_CHUNK`는 4·12쪽 청크가 같은 품질에서 10% 넘게 빠르지
않아(최대 3.4%) 기본 8을 유지합니다.

## 보안

이 서비스는 **인증이 없습니다** — 접근 가능한 사람은 누구나 문서 열람·삭제·변환·(설정 시)
유료 번역 트리거가 가능합니다. CSRF 방어도 없으므로 같은 브라우저에서 연 다른 웹페이지가
이 인스턴스로 요청을 보낼 수 있습니다. 아래 신뢰 네트워크·인증 프록시 전제가 그대로 적용됩니다.

compose 기본값은 **외부 노출**입니다: 포트를 `0.0.0.0`에 바인딩하고
`ALLOWED_HOSTS` 기본이 `*`(모든 Host 허용)입니다. 이 기본값은 **신뢰
네트워크**(VPN/Tailscale, 방화벽 뒤 홈랩)를 전제로 합니다 — 공개 인터넷에
노출한다면 **반드시 인증을 제공하는 리버스 프록시**(예: nginx + basic auth)
뒤에 두세요.

로컬 전용으로 되돌리려면 `.env`에:

1. `BIND_HOST=127.0.0.1` — 포트를 루프백에만 바인딩
2. `ALLOWED_HOSTS=localhost,127.0.0.1` — Host 헤더 화이트리스트 복원
   (DNS rebinding 방어 — 도메인/IP로 접속한다면 그 값을 목록에 추가.
   포트는 비교 시 무시됨). compose가 컨테이너로 전달합니다.

⚠ **둘 다** 설정하세요. `BIND_HOST`만 바꾸면 compose 기본 `ALLOWED_HOSTS=*`가 그대로
남아 DNS rebinding 경로가 열려 있습니다.

⚠ **Docker가 게시한 포트는 `ufw`·`firewalld` 같은 호스트 방화벽 규칙을 거치지 않습니다**
(Docker가 자체 포워딩 규칙을 그 앞에 넣는다). "방화벽 뒤"는 네트워크 방화벽, `DOCKER-USER`
체인 규칙, 또는 밖에서 닿지 않는 `BIND_HOST`(루프백·VPN 인터페이스 주소)를 뜻합니다 —
호스트에서 `ufw deny 8000`만 해 두고 공인 IP 서버에서 기본값으로 띄우면 그대로 노출됩니다.

### 리버스 프록시 뒤에 둘 때

`TRUSTED_PROXY_HOPS`에 앞단 프록시 수를, `TRUSTED_PROXY_IPS`에 그 프록시의 IP·CIDR을
넣습니다(비우면 루프백만). `X-Forwarded-For`는 직접 연결한 피어가 이 목록에 있을 때만
믿고, 목록 밖 피어(백엔드 포트에 바로 붙은 LAN 클라이언트)의 헤더는 위조로 보고 그 피어
IP로 레이트리밋합니다. 호스트의 프록시가 Docker 게시 포트로 붙으면 피어가 브리지
게이트웨이(예: `172.17.0.1`)로 보이므로 그 주소를 넣고, `BIND_HOST=127.0.0.1`로 백엔드
포트도 닫으세요.

### 남용 방어 (레이트리밋) — 인증의 대체가 아님

Q&A·번역 엔드포인트에는 잡·클라이언트 IP 단위 슬라이딩 윈도우(60초) 레이트리밋과
동시 실행 상한이 있습니다. 초과하면 `429`와 `Retry-After` 헤더로 거절합니다.

| 환경변수 | 기본 | 적용 대상 | 초과 시 |
|---|---|---|---|
| `QA_RATE_LIMIT_PER_MIN` | 30 | `POST /api/jobs/{id}/qa` 분당 요청 | 429 + `Retry-After`(남은 창 초) |
| `QA_MAX_CONCURRENT` | 4 | 동시에 처리 중인 질문 수 | 429 + `Retry-After: 5` |
| `TRANSLATE_RATE_LIMIT_PER_MIN` | 12 | `POST /api/jobs/{id}/translate` 분당 요청 | 429 + `Retry-After`(남은 창 초) |
| `TRANSLATE_MAX_ACTIVE` | 4 | 동시에 실행 중인 번역 잡 수 | 429 + `Retry-After: 30` |

0 이하로 두면 해당 상한이 비활성화되고, 정수가 아닌 값은 경고 로그와 함께 기본값으로
강등됩니다. 기본값은 1인 로컬 사용을 방해하지 않는 수준입니다. 이 밖에 라이브 미리보기
렌더(`POST /render-preview`, 본문 256 KiB 상한·분당 상한·동시 4건)와 SSE 구독(잡당 8개·
전체 64개, 넘으면 503 + `Retry-After: 5`)에도 고정 상한이 있습니다.

이 상한은 **실수와 경미한 남용**(운영자의 유료 LLM 키 소진, 200페이지 번역 반복
트리거)의 비용 상한일 뿐 **인증의 대체가 아닙니다** — 여전히 누구나 문서를 열람·삭제할
수 있습니다. 신뢰 네트워크 밖에 두려면 인증 리버스 프록시가 필요합니다.

네 변수는 `docker-compose.yml`의 모든 backend 서비스에 전달되므로 `.env`에서
조정할 수 있습니다.

### 브라우저·업로드 방어

- **CSP**: 모든 HTML 응답에 CSP 헤더(외부 스크립트·이미지·폰트·연결 차단,
  `frame-ancestors 'none'`으로 다른 사이트의 iframe 삽입 금지)와 `Referrer-Policy: same-origin`이
  붙고, `index.html`의 meta CSP도 같은 정책입니다. 내려받은 `document.html`은 파일 하나로
  열리되 자체 meta CSP(`default-src 'none'`)로 어떤 외부 요청도 내지 않습니다.
- **외부 이미지 차단**: OCR 텍스트 속 `![](https://…)`·LAN 주소 이미지는 자동으로 불러오지
  않고 클릭해야 열리는 링크로 바뀝니다 — 문서를 연 사실·IP가 제3자에게 새지 않습니다.
- **수식**: 로컬 KaTeX 0.18.10(GHSA-238p-pmpm-9mq7 수정판)을 크기·매크로 확장 상한과 함께 씁니다.
- **악성 PDF 격리**: PDF 렌더·분석·번역 PDF 빌드는 서버 밖 워커 프로세스에서 페이지·빌드별
  시간 상한(`PDF_PAGE_TIMEOUT_S` 60초·`PDF_EXPORT_BUILD_TIMEOUT_S` 900초)을 두고 돌며, 넘기거나
  죽은 워커만 정리됩니다 — 페이지 하나가 서버·다른 잡을 멈추지 못합니다. 업로드 때는
  렌더 없이 페이지 작업량을 재서 중첩 XObject 폭탄·압축 폭탄을 400으로 거부합니다
  (`PDF_MAX_PAGE_CONTENT_MB` 64·`PDF_MAX_PAGE_XOBJECT_CALLS` 2,000,000 — 정상 논문 표본은
  페이지당 최대 0.3 MB·129회).
  워커는 API 키·토큰 같은 환경변수를 지우고 시작합니다.

자세한 정책: [SECURITY.md](SECURITY.md) · [docs/ARCHITECTURE.md §14·§18](docs/ARCHITECTURE.md).

### 컨테이너 하드닝

`docker-compose.yml`의 모든 서비스는 비루트(uid 1000)·`no-new-privileges`·`cap_drop: [ALL]`로
돌고, backend 4개는 프로세스 수 상한(1024)이 있습니다. CPU 이미지 backend(`ocr-cpu`·`ocr-ovis`·`ocr-paddle`)는
루트 파일시스템이 읽기 전용이라 쓰기는 `/data` 볼륨과 `/tmp` tmpfs(2 GB — `MAX_UPLOAD_MB`를
크게 올리면 함께 올린다)뿐입니다. `ocr-cuda`와 GPU sidecar는 아직 읽기 전용이 아닙니다(GPU
호스트 검증 필요). 이미지 안 앱 코드는 root 소유·읽기 전용이고 바이트코드를 미리 컴파일하며,
uvicorn은 종료 시 열린 SSE를 5초 안에 정리합니다(`--timeout-graceful-shutdown 5`). 예외: 선택
overlay `compose.ollama.yaml`의 `ollama`는 `no-new-privileges`만 걸려 있고 공식 이미지 그대로
root·Docker 기본 캡으로 돕니다(포트는 Docker 내부 네트워크에만 열립니다).

### 원격 접속 (Tailscale) — HTTPS 권장

`http://<tailscale IP>:8001` 직접 접속도 동작하지만, 비보안(http) origin에서는
최신 브라우저가 **파일 다운로드를 "안전하지 않음"으로 차단**하고(다운로드
트레이에서 수동 "보관" 필요) 클립보드 복사도 제한됩니다. Tailscale 내장
HTTPS를 쓰면 전부 해결됩니다 (tailnet 전용 — 인터넷에 노출되지 않음):

```bash
# 1회: 관리자 콘솔(https://login.tailscale.com/admin/dns)에서
#      MagicDNS + "HTTPS Certificates" 활성화
sudo tailscale serve --bg --https=443 http://127.0.0.1:8001
# → https://<노드명>.<tailnet>.ts.net 로 접속 (정식 인증서, 재부팅에도 유지)
```

**트러블슈팅 — 작은 응답(health)은 되는데 페이지/다운로드가 멈출 때**: 터널
경로 MTU 블랙홀 가능성이 높습니다. `tailscale ping --size 1250 <peer>` 는
되는데 `--size 1280` 이 실패하면, tailscaled에 `TS_DEBUG_MTU=1200` 환경변수를
주고(systemd drop-in) 재시작 + tailscale0에 TCP MSS 클램핑(iptables mangle
FORWARD, `--clamp-mss-to-pmtu` 양방향)을 적용하세요. `tailscale serve` 경로는
tun이 아닌 tailscaled 내부 netstack을 타므로 **tailscaled 재시작까지 해야**
적용됩니다 (2026-08 WSL2 호스트 실측).

## 한국어 번역

변환이 끝난 문서(`result.md` + 레이아웃)를 OpenAI 호환 API로 한국어 번역해
번역본 미리보기/레이아웃/다운로드를 제공합니다 (수식·이미지·표는 마스킹으로 보존).

```bash
cp .env.example .env   # 키 설정 후 docker compose up -d 로 재기동
```

로컬(uv — `make dev`·`make dev-metal`)로 띄웠다면 `.env`를 고친 뒤 서버를 Ctrl+C로 끄고 다시
실행하세요 — `.env`는 기동할 때 한 번만 읽고, `--reload`는 `backend/`의 파이썬 파일만 지켜봐
`.env` 변경으로는 재시작하지 않습니다. 예시 블록을 `.env.example`을 복사한 `.env`에 붙여 넣을 때는
같은 키의 빈 줄이 뒤에 남지 않게 하세요(뒤쪽 줄이 이깁니다 — 남아 있으면 `/api/health`의
`config_warnings`가 그 키를 알려 줍니다).

`.env`에 아래 값을 설정하면 활성화됩니다:

- `OPENAI_BASE_URL` — OpenAI 호환 base URL. `https://host`처럼 origin만 쓰면
  `/v1`을 자동 보완하며, 명시한 버전/게이트웨이 경로는 그대로 사용합니다.
- `OPENAI_API_KEY` — 로컬 서버는 생략 가능
- `OPENAI_MODEL` — 번역에 쓸 모델 ID (서버 `/v1/models`가 보고하는 id 그대로)

미설정이어도 나머지 기능은 그대로 동작합니다 — 번역 요청 시에만 503과 함께
"번역 프로바이더가 설정되지 않았습니다" 안내가 표시됩니다.
동시성/재시도/reasoning 등 세부 옵션과 파이프라인 설계는
[docs/ARCHITECTURE.md §13](docs/ARCHITECTURE.md#13-한국어-번역-translation) 참조.
잡 하나의 번역 요청 동시성은 기본 8이며 `TRANSLATE_CONCURRENCY`를 1–8 범위에서
조정할 수 있습니다. 여러 잡을 합친 실제 upstream HTTP 요청 상한은
`TRANSLATE_GLOBAL_CONCURRENCY`이며, 생략하면 잡당 값과 같은 수를 사용합니다.
로컬 vLLM/Ollama나 429가 잦은 공급자는 두 값을 4 이하로 낮추는 편이 안전합니다.

- **스트리밍 기본**: chat 모드 요청은 SSE로 스트리밍합니다(`TRANSLATE_STREAM=auto`). 그래서
  `TRANSLATE_TIMEOUT_S`는 '토큰 사이 정지 시간' 상한이 되어 느린 로컬 생성도 끊지 않고, 번역을
  취소하면 연결을 끊어 서버의 생성까지 멈춥니다. 서버가 스트리밍을 거부하면(400/415/422)
  자동으로 비스트리밍으로 바꾸고, `TRANSLATE_STREAM=0`이면 예전 방식입니다.
- **reasoning 제어**: `TRANSLATE_REASONING`(`off`|`low`|`medium`|`high`|`xhigh`)을 어떤 필드로
  보낼지는 `TRANSLATE_REASONING_STYLE`(기본 `auto`)이 base URL로 고릅니다 — 루프백·사설 IP·
  단일 라벨·`localhost`·`*.localhost`·`*.local`·`*.lan`·`*.home.arpa`·`*.internal`
  (`host.docker.internal`·Podman `host.containers.internal` 포함) → `chat_template_kwargs`
  (`enable_thinking`), `openrouter.ai` → `reasoning`, `api.openai.com`·`*.api.openai.com`
  (eu./us. 데이터 레지던시) → `reasoning_effort`. 서버 전용 파라미터는
  `TRANSLATE_EXTRA_BODY`(JSON 객체)로 덧붙입니다.
- **잘린 출력은 쓰지 않습니다**: 응답이 출력 상한에서 잘리거나 비었거나 시간 초과면 그 문단은
  분할 재시도를 거쳐 끝내 원문으로 남고(캐시하지 않음), 사유가 `report.json`에 남습니다.
  번역 품질 경고는 결과 화면의 흐린 접이식 '번역 참고 사항 N건' 목록에 보입니다.

### 로컬 MLX 서버로 번역 (oMLX · LM Studio · mlx_lm.server)

Apple Silicon에서는 번역 모델도 로컬로 돌릴 수 있습니다. 세 서버 모두 OpenAI 호환
`/v1/chat/completions`를 냅니다:

| 서버 | base URL (네이티브 실행) | `OPENAI_MODEL` |
|---|---|---|
| oMLX | `http://127.0.0.1:1235/v1` | 서버 `/v1/models`가 보고하는 id |
| LM Studio | `http://127.0.0.1:1234/v1` | 서버 `/v1/models`가 보고하는 id |
| mlx_lm.server | `http://127.0.0.1:8080/v1` | `default_model` (= `--model`로 띄운 모델) |

mlx_lm.server는 이 리포에 들어 있지 않습니다 — 별도 도구 환경에 설치해 띄웁니다:

```bash
uv tool install mlx-lm      # 별도 환경(~/.local/bin/mlx_lm.server). backend/.venv에 넣지 마세요
mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-8bit --port 8080   # 첫 실행에 약 4.8GB 다운로드
curl http://127.0.0.1:8080/v1/models                                  # 떠 있는지 확인
```

`backend/.venv`에 `uv pip install mlx-lm`을 하면 mlx-lm이 transformers 5.x를 끌어와 고정한
4.57.1을 바꿉니다 — torch MPS 폴백과 `.venv/bin/python`으로 부르는 테스트가 import 단계에서
깨지고, 다음 `make setup-mlx`(uv sync)는 mlx-lm을 다시 지웁니다.

- Docker 컨테이너의 backend에서 호스트 서버로 갈 때는 `127.0.0.1` 대신 `host.docker.internal`.
- 모델 id가 틀리면 서버가 404를 내고, 오류 문구가 모델 id를 확인하라고 알려 줍니다.
- **thinking 끄기**: `TRANSLATE_REASONING=off` — `TRANSLATE_REASONING_STYLE=auto`(기본)가 루프백·
  사설 IP·단일 라벨·`localhost`·`*.localhost`·`*.local`·`*.lan`·`*.home.arpa`·`*.internal`
  (`host.docker.internal`·Podman `host.containers.internal` 포함) 주소를 보고
  `chat_template_kwargs: {"enable_thinking": false}`로
  전달합니다(공개 호스트 이름으로 띄웠다면 `TRANSLATE_REASONING_STYLE=chat_template_kwargs`로
  고정). 실측(mlx_lm.server + Qwen3.5-0.8B): 유닛당 0.18초·reasoning 0자 — 예전 방식은
  5.8초 동안 4,075자를 생각하다 잘렸습니다. 서버가 이 인자를 무시하면
  서버 쪽에서 끕니다: `mlx_lm.server --chat-template-args '{"enable_thinking":false}'`,
  oMLX·LM Studio는 모델별 thinking 설정 또는 비-thinking(Instruct) 모델.
- `TRANSLATE_API_MODE=chat` — 로컬 서버(mlx_lm)에는 `/v1/responses`가 없어 auto 모드는 잡마다
  404 확인 요청을 한 번 보냅니다.
- `TRANSLATE_MAX_TOKENS_PARAM=none`은 쓰지 마세요 — mlx_lm은 `max_tokens`가 없으면 512토큰에서
  자릅니다.
- 동시성: 요청을 모아 처리하는(연속 배칭) 서버는 8이 맞습니다(실측 mlx_lm 0.32: 동시 1/4/8/16
  요청에 합계 208/503/808/878 tok/s). 요청을 하나씩 처리하는 서버는
  `TRANSLATE_CONCURRENCY`·`TRANSLATE_GLOBAL_CONCURRENCY`를 1–2로 낮추세요.
- 모델: 0.8B급은 출력 게이트를 자주 통과하지 못해(프롬프트 echo·반복) 원문이 많이 남습니다 —
  4B 이상의 비-thinking Instruct 모델을 권장합니다.
- OCR(MLX)과 로컬 번역 모델은 같은 GPU를 씁니다. 큰 문서는 OCR이 끝난 뒤 번역하세요.

`.env` 예시 (mlx_lm.server 기준 — 다른 서버는 base URL과 모델 id만 바꾼다):

```dotenv
# 로컬 mlx_lm.server — 네이티브(uv) 실행 기준.
# Docker 컨테이너의 backend에서는 127.0.0.1 대신 host.docker.internal을 쓴다.
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
# 서버에 키를 걸지 않았으면 비워 둔다
OPENAI_API_KEY=
# mlx_lm.server는 default_model(= --model로 띄운 모델).
# oMLX·LM Studio는 서버 /v1/models가 보고하는 id를 그대로 쓴다.
OPENAI_MODEL=default_model
# 로컬 서버에는 /v1/responses가 없다 — 잡마다 한 번 하던 404 확인 요청을 없앤다
TRANSLATE_API_MODE=chat
# thinking 끄기 — 루프백 주소면 chat_template_kwargs enable_thinking=false로 전달된다
TRANSLATE_REASONING=off
# reasoning 전달 방식 — auto는 루프백·사설 IP·단일 라벨·localhost·*.localhost·*.local·*.lan·
# *.home.arpa·*.internal(host.docker.internal·host.containers.internal 포함)이면 chat_template_kwargs
TRANSLATE_REASONING_STYLE=auto
# max_tokens를 꼭 보낸다 (none이면 mlx_lm이 512토큰에서 자른다)
TRANSLATE_MAX_TOKENS_PARAM=max_tokens
# 요청을 모아 처리하는(연속 배칭) 서버는 8, 하나씩 처리하는 서버는 1–2
TRANSLATE_CONCURRENCY=8
TRANSLATE_GLOBAL_CONCURRENCY=8
# chat은 기본 스트리밍이라 이 값은 '토큰 사이 정지 시간' 상한(초)이다
TRANSLATE_TIMEOUT_S=180
# 페이지 Q&A도 같은 서버로 (루프백·host.docker.internal 주소만 허용)
LLM_LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
LLM_LOCAL_OPENAI_MODEL=default_model
# '질문' 탭의 기본 공급자로 쓰려면
LLM_PROVIDER=local-openai
```

## 읽기 뷰 (기본 화면)

변환이 완료된 논문은 **'읽기' 탭**이 기본으로 열립니다. 왼쪽은 언어 전환과
무관하게 원본 PDF 전 페이지를 하나의 연속 스크롤 면으로 유지하고, 오른쪽도 같은
페이지 순서의 원문 또는 한국어 번역 블록을 연속 읽기 레일로 보여 줍니다.

- 마우스/트랙패드 스크롤만으로 다음 페이지를 계속 읽을 수 있습니다. ◀/▶, 페이지
  번호, 썸네일, 문서 개요, 키보드 ←/→는 해당 페이지로 즉시 이동하는 보조 탐색입니다.
- **연동/개별** 토글로 원문과 번역문을 같은 OCR 문단에 맞춰 양방향 스크롤하거나
  서로 따로 볼 수 있습니다. 마지막 읽던 페이지는 잡별로 저장되어 다음 열기에 복원됩니다.
- 긴 논문은 전체 페이지 자리를 먼저 잡되 현재 주변 이미지만 지연 로드해 메모리와
  네트워크 사용량을 제한합니다. 완료 잡 목록의 책 아이콘으로 뷰어를 바로 열 수 있습니다.
- 확대/축소와 너비 맞춤: 60–220%, 브라우저에 저장
- 번역 전이면 **'한국어로 읽기'** 버튼으로 곧바로 전체 번역을 시작할 수 있고,
  완료되면 자동으로 한국어 레일로 전환됩니다 ([원문|한국어] 토글로 언제든 대조)
- 한국어 보기는 번역본이 정말 없을 때(404/409)만 원문으로 돌아갑니다. 서버가 번역 PDF를
  만드는 중이면(503 + `Retry-After`) '… 준비 중… N초 뒤 다시 시도합니다 (k/4)'를 보이며 최대
  4번 기다리고, 네트워크 오류·그 밖의 5xx는 한국어 보기를 유지한 채 [다시 시도]·[원문 보기]를
  띄웁니다.
- 오른쪽 문단을 가리키거나 누르면 같은 OCR 블록의 원문 bbox가 왼쪽에서 강조되고,
  왼쪽 bbox를 누르면 대응 번역문으로 자동 이동합니다
- OCR `title` 블록으로 만든 문서 개요에서 섹션을 눌러 페이지로 이동
- 페이지 요약, 선택 문장 설명, 하이라이트, 인용 저장을 연구 레일에서 실행
- **하이라이트·인용 저장**: 이 브라우저(localStorage)에 잡별로 남고, [선택 문장 도구] 아래
  목록에서 페이지로 이동·삭제·Markdown 복사/내보내기(`<파일명>.notes.md`)를 할 수 있습니다.
  잡을 삭제하면 함께 지워집니다. 잡당 200개, 브라우저당 50개 잡까지 보관합니다(오래 손대지
  않은 잡부터 정리). 수식을 가로지른 하이라이트는 새로고침 뒤 목록에만 남을 수 있습니다.
  같은 논문을 여러 탭에서 열어도 서로의 메모를 덮어쓰지 않고 다른 탭의 변경이 바로 목록에
  반영됩니다. 저장 공간이 차면 가장 오래 손대지 않은 다른 문서의 메모부터 지우고 저장 안내에
  알립니다.
- **지원 브라우저**: 최신 Chrome·Edge·Firefox, Safari 16 이상.
- 'AI 질문'과 선택 설명은 기존 질문 탭으로 현재 페이지·문장을 그대로 전달

## 결과 화면 — 품질 경고와 참고

- **주의/참고**: 변환 결과 헤더의 '주의 N건' 칩을 누르면 실제 품질 저하(텍스트 레이어로
  복구한 페이지, 실패 플레이스홀더, 렌더 시간 상한을 넘어 흰 페이지로 대체한 페이지 등) 목록이
  펼쳐집니다. 내용에 문제가 없는 처리 경위(페이지별 재처리로 복구됨, 충실도 재처리 채택,
  페이지 단위 엔진 안내 등)는 흐린 '참고 N건'으로 따로 보입니다 — 경고 없이 참고만 있으면 칩도
  '참고 N건'입니다. 'N페이지' 언급은 읽기 뷰의 그 페이지로 가는 링크이고, 잡 목록에는
  '주의 N' 배지가 붙습니다. API: 잡의 `warnings`(품질 저하)와 `notices`(정보성)·
  `started_at`·`finished_at`.
- **번역 참고 사항 N건**: 마지막 번역의 경고(캐시 전량 무효화·용어집 판정 실패 등)를 결과
  툴바 아래 흐린 접이식 목록으로 보여 줍니다.
- **잡 목록**: 최신 50건을 5초마다 갱신하고, 더 오래된 잡은 사이드바 아래 '더 보기 (50/132)'로
  50건씩 이어 받습니다. 다른 탭에서 지운 잡은 다음 갱신 때 닫히며 '열려 있던 작업이
  삭제되었습니다.'를 알립니다. 대기 중인 잡을 취소하면 즉시 '취소됨'으로 마감됩니다.
- **상태 배지**: health를 30초마다(로딩·실패 중에는 10초, 숨긴 탭에서는 멈춤) 확인해 디바이스
  (`MLX · M4 Max`처럼 칩 이름 포함)·'모델 로드 실패'(사유는 툴팁)·'작업 처리기 중지됨'을
  표시합니다. 바뀐 배지만 갱신하므로 스크린리더가 같은 배지를 폴링마다 다시 읽지 않습니다.

## 내보내기

완역본을 두 형식으로 내려받을 수 있습니다 (결과 화면의 다운로드 줄):

| 형식 | 버튼 | 내용 |
| --- | --- | --- |
| HTML | `원본 HTML` / `한국어 HTML` | **PDF facsimile 단일 파일** — 완성 페이지 PNG를 인라인하고 OCR 블록을 검색·복사용 투명 텍스트 레이어로 보존 |
| PDF | `원문·한국어 PDF` | **원문·번역 대조 PDF** — 한 장의 가로 스프레드에 원본 페이지(왼쪽)와 레이아웃 보존 한국어판(오른쪽)을 나란히 배치. 일반 텍스트와 구조가 안정적인 표 셀을 교체하며, 독립 수식·그림·참고문헌·세로쓰기는 원본 유지 |

기존 `/layout.html` 주소는 같은 파일을 중복 생성하지 않고 정식
`/document.html` 내보내기로 리다이렉트합니다. 내려받은 HTML은 모든 자원(KaTeX·폰트·페이지
이미지)을 품은 파일 하나라 디스크에서 그대로 열리고, 자체 CSP로 외부 요청을 하나도 내지
않습니다.

PDF 내보내기 UI는 `GET /api/jobs/{id}/pdf?lang=ko&view=dual`로 원문·한국어
대조본을 만듭니다. 기존 단일 한국어판 API(`view=single`, 기본값)도 호환성을 위해
유지합니다. 한국어 번역이 완료된 잡에서 활성화되며, 좌표 텍스트 레이아웃이 없는 잡
(OvisOCR2 같은 그림 전용 엔진, textlayer 엔진으로 처리한 전면 스캔 문서)은 PDF 대신 HTML
내보내기(의미 기반 문서)를 안내합니다. 원본 PDF span의 serif/sans 계열과 실측 글자 크기를 블록마다
추출해 대응시킵니다. 배포 기준인 Docker 이미지는 `fonts-noto-cjk`와 `fontconfig`를
기본 설치하므로 별도 설정 없이 본문·절 제목에는 Noto Serif CJK, sans 대표 제목에는
Noto Sans CJK를 임베드합니다. macOS의 Apple 계열 폰트는 네이티브 로컬 실행에서만
쓰는 조건부 후보이며 컨테이너 런타임 의존성이 아닙니다. 한 줄 제목·목록은 원본 baseline과
크기를 직접 보존하고, 여러 줄 본문만 충돌 없는 범위에서 행간·줄바꿈을 맞춥니다.
글꼴이 없으면 PyMuPDF 내장 CJK로 폴백하며 `PDF_EXPORT_FONT`로
원하는 폰트 파일을 지정할 수 있습니다. 원문과 번역 레이아웃의 블록 대응을 먼저
검증하고, 번역문이 최소 크기에도 들어가지 않는 블록은 원문을 지우지 않고 보존합니다.
텍스트 리댁션은 이미지·벡터 그래픽을 건드리지 않고 글자 줄마다 기준선 띠만 지워 이웃 줄을
보존하며, 원본 PDF 텍스트 레이어의 실측 폰트 크기를 자동으로 사용합니다. 스캔·이미지
페이지는 원문 픽셀 위에 한국어를 겹쳐 찍지 않고, 블록 영역을 그 페이지의 바탕색으로 덮은 뒤
번역을 넣습니다(이미지 자체는 다시 인코딩하지 않는다). 스캔 표는 픽셀에서 열·행 경계를 찾아
바뀐 셀의 글자만 덮고, 경계를 확정하지 못하면 원문 표를 보존합니다. 회전된 페이지도 화면
방향 그대로 조판합니다. 공간이 부족하면 이웃 블록 전의 빈 영역까지만 확장하고, 다운로드 뒤
번역·표 셀·재배치·원문 보존·스캔 원문 지움 개수를 토스트로 알려주며, 결과 화면 아래 흐린 접이식
'PDF 생성 리포트 · 주의 N건' 목록에 원문 보존 사유(예: '원문 줄 위치 정렬 실패(그 줄만
원문)')와 'N페이지: …' 주의 문장을 보여 줍니다(`GET /api/jobs/{id}/pdf/report`).
참고문헌은 저자·학술지·URL의 서지 형식과 원문 조판을 그대로 보존합니다.
한국어 HTML은 이 번역 PDF를 같은 DPI로 렌더한 페이지를 기준면으로 사용하므로
HTML과 PDF의 제목·단·그림·수식 위치가 서로 달라지지 않습니다.
읽기 패널의 흐름형 Markdown도 원문 줄과 레이아웃 블록이 대응하면 같은 블록
번역을 재사용하므로, PDF·문서 개요·오른쪽 텍스트의 제목과 용어가 서로 달라지지
않습니다.

번역 PDF 빌드는 서버 밖 워커 프로세스에서 돌아(`PDF_EXPORT_MAX_CONCURRENT`개가 실제로
병렬) OCR을 느리게 하지 않습니다. 빌드가 몰리면 다운로드가 503 + `Retry-After`를 받는데,
UI는 최대 4번까지 기다렸다가 다시 받습니다. 빌드가 `PDF_EXPORT_BUILD_TIMEOUT_S`(900초)를
넘거나 입력이 손상되면 409와 사유가 돌아옵니다.

Compose에서 사용자 폰트를 쓸 때 `PDF_EXPORT_FONT`는 호스트 경로가 아니라
**컨테이너 내부 절대경로**여야 합니다. 사용하는 backend 서비스에 읽기 전용으로
마운트한 뒤 같은 경로를 `.env`에 지정합니다(기본 Noto를 쓸 때는 모두 생략).

```yaml
# compose.font.yaml — 실제로 기동할 backend 서비스에 동일하게 적용
services:
  ocr-cpu:
    volumes:
      - /absolute/host/path/custom.otf:/fonts/custom.otf:ro
```

```dotenv
PDF_EXPORT_FONT=/fonts/custom.otf
```

```bash
docker compose -f docker-compose.yml -f compose.font.yaml up -d --build ocr-cpu
```

## 페이지 Q&A (AI에게 묻기)

변환된 문서를 보면서 결과 화면의 **'질문' 탭**으로 현재 페이지 내용에 대해
AI에게 물을 수 있습니다 (Localight에서 이식). 공급자는 넷 중 선택
(`LLM_PROVIDER`, 기본 `openai-responses`):

| 공급자 | 엔드포인트 | Reasoning/Thinking |
| --- | --- | --- |
| `openai-responses` | `/v1/responses` | `reasoning.effort`, 선택적 `reasoning.summary` |
| `openai-chat` | `/v1/chat/completions` | `reasoning_effort` |
| `ollama` | `/api/chat` | `think: false/true/low/medium/high` |
| `local-openai` | 로컬 OpenAI 호환 서버의 `/v1/chat/completions` | `chat_template_kwargs.enable_thinking` |

- API: `GET /api/providers`(공급자·모델 목록) · `POST /api/jobs/{id}/qa`(페이지 질의).
  활성 여부는 `GET /api/health`의 `qa_available`·`llm_default_provider`로 확인합니다.
- OpenAI 인증은 **Q&A 전용 `LLM_OPENAI_API_KEY`**를 씁니다 — 번역용 `OPENAI_API_KEY`는
  폴백으로도 재사용하지 않습니다. `LLM_OPENAI_BASE_URL`이 공식 `https://api.openai.com`
  호스트로 고정돼 있어, OpenRouter·로컬 게이트웨이용 번역 키를 공유하면 그 키가 무관한
  제3자(OpenAI)로 전송되기 때문입니다. `.env`에 `LLM_OPENAI_API_KEY=`를 채우고
  `docker compose up -d`로 재기동하면 됩니다(compose가 네 backend 서비스 모두에 전달 — 로컬
  `make dev`는 Ctrl+C 뒤 다시 실행).
  미설정이면 `openai-*` 공급자는 `/api/providers`에서 `available:false`,
  `/api/health`의 `qa_available`도 false가 되고 UI가 키 설정을 안내합니다 — Ollama
  공급자는 이 키 없이 동작합니다.
- **`local-openai`**: oMLX·LM Studio·mlx_lm.server 같은 로컬 OpenAI 호환 서버로 묻습니다.
  `LLM_LOCAL_OPENAI_BASE_URL`(루프백 `127.0.0.1`·`localhost`·`::1` 또는
  `host.docker.internal`만 허용 — 그 밖은 기동 실패)과 `LLM_LOCAL_OPENAI_MODEL`(서버
  `/v1/models`의 id, 주소를 두면 필수 — mlx_lm.server는 `default_model`. 그 서버의 `/v1/models`는
  `--model`로 띄운 모델 대신 HF 캐시의 모델 전부를 보이는데, 그중 다른 id를 고르면 서버가 지금
  모델을 내리고 그 모델로 갈아 끼워 번역과 번갈아 쓰일 때마다 다시 올린다)을 설정하면 목록에
  나타나고, 서버가 떠 있는지는 `/models`로 확인합니다. `LLM_LOCAL_OPENAI_MODELS`는 추가 선택지, 키는 전용
  `LLM_LOCAL_OPENAI_API_KEY`만 씁니다(다른 키로 폴백하지 않음). 서버가 꺼져 있으면 UI가
  서버를 켜고 두 설정을 확인하라고 안내합니다. 위 §로컬 MLX 서버의 `.env` 예시 참조.
- 지원 effort는 모델마다 다르므로 API가 거부하는 조합은 UI 오류로 그대로 안내합니다.

**프라이버시 약속** (Localight에서 승계):

- 외부로는 **현재 페이지에서 추출된 텍스트만** 전송합니다 — 원본 PDF·페이지 이미지는 전송하지 않습니다.
- OpenAI 요청은 `store: false`로 생성합니다(번역의 Responses 요청도 같다).
- Ollama의 `:cloud`/`remote_host` 모델은 목록에서 제외하고 호출도 차단합니다.
  Ollama 주소는 루프백·`host.docker.internal`·`ollama`(overlay 컨테이너)만 허용됩니다.
- 원시 chain-of-thought는 표시하거나 저장하지 않습니다 — Responses의 reasoning
  summary만 선택적으로 표시합니다.

### Ollama로 완전 로컬 Q&A (Docker)

```bash
make docker-up-ollama    # ocr-cpu + ollama 컨테이너 (overlay: compose.ollama.yaml)
make docker-pull-model   # 기본 qwen3:8b — `MODEL=… make docker-pull-model`로 변경
make docker-down-ollama  # overlay 스택 정리
```

호스트에서 직접 Ollama를 돌리는 경우(macOS는 Metal 가속 때문에 보통 더 빠름)
overlay 없이 기본 compose로 충분합니다 — 컨테이너는 `OLLAMA_BASE_URL` 기본값
`http://host.docker.internal:11434`로 호스트 Ollama에 접근합니다.
컨테이너 Ollama의 포트(11434)는 호스트에 공개되지 않습니다.

## E2E 스모크 테스트

```bash
cd backend && uv run python ../scripts/make_sample_pdf.py ../sample/sample.pdf && cd ..
./scripts/smoke_e2e.sh                      # CPU (8000)
./scripts/smoke_e2e.sh http://localhost:8001  # CUDA (8001)

# 신규 엔진 실 GPU smoke (RTX 5070 Ti — 해당 프로필 기동 후)
cd backend
uv run python ../scripts/smoke_ovisocr2_5070ti.py        # ovis 프로필 (:8002)
uv run python ../scripts/smoke_paddleocr_vl_5070ti.py    # paddle 프로필 (:8003)

# 엔진 비교 벤치마크 (스택을 하나씩 띄워 순차 실행 — benchmark_docs/README.md)
uv run python ../scripts/benchmark_ocr_engines.py \
  --endpoint ovis=http://127.0.0.1:8002 --input ../benchmark_docs/ --out ../bench_out/
```

벤치마크는 모델 준비를 기다리고 첫 문서를 `--warmup`(기본 1회)으로 기록에서 빼며, 잡 상태를
0.25초 간격으로 폴링해 처리 시간(`process_s`)으로 s/page를 잽니다(예전 2초 폴링 값은 짧은
문서에서 믿을 수 없었다).

### 전 구간 검증 하네스 (외부 API 없이)

`smoke_e2e.sh`는 업로드→OCR→마크다운/zip에서 끝납니다. 그 뒤의 번역 · 레이아웃 보존
PDF 내보내기 · 뷰어 계약 · Q&A 키 분리 · 워커 복원력까지 한 번에 점검하려면
(`make setup` 이후 — Apple Silicon은 `make setup-mlx`. 하네스는 textlayer 엔진만 쓰지만
`make setup`은 mlx와 C++ 모듈을 지운다):

```bash
make verify-e2e                          # 기본 sample/2504.19874v1.pdf 전체
VERIFY_ARGS="--pages 4" make verify-e2e  # 앞 4페이지만 (빠른 확인)
```

`scripts/verify_e2e.py`가 빈 포트를 골라 mock LLM(`scripts/mock_llm.py`)과
`OCR_ENGINE=textlayer` 백엔드를 직접 띄우므로 **모델 다운로드도 유료 키도 필요
없고** 개발 서버(8000)와 충돌하지 않습니다. 셸에 내보낸 앱 설정·자격증명·프록시 변수는
자식 프로세스에 넘기지 않습니다. 포트를 고정해야 하면 `--mock-port`·`--api-port`를 줍니다
(서로 달라야 한다). 서버 로그·내보낸 PDF 등 산출물은 `tmp/verify-e2e/`에 남습니다
(`--work DIR`로 변경, `--skip faults,worker,cropbox`로 단계 생략). 단언은 89개이고,
M4 Max에서 `--pages 4`는 약 2분(그중 약 1분은 429 백오프 실증), 25쪽 전체는 약 6분입니다.

브라우저까지 포함한 hermetic 전체 흐름(mock OpenAI + FakeEngine)은
`make e2e-mock`입니다 — playwright chromium을 처음 한 번 내려받습니다. 포트를 고정하려면
`E2E_MOCK_PORT`·`E2E_BACKEND_PORT`를 줍니다.

## 로컬 개발 (uv)

```bash
# 백엔드 (Python 3.12, torch CPU)
cd backend
uv sync --extra cpu          # CUDA: --extra cu129 · Apple Silicon: --extra metal --extra mlx (= make setup-mlx)
uv pip install ../native     # C++ 가속 모듈 (선택 — 없어도 동작)
uv run pytest                # 유닛/통합 테스트 (FakeEngine, 모델 불필요)
uv run uvicorn app.main:app --reload   # http://localhost:8000 (권장: make dev)

# 네이티브 모듈 단독 테스트
cd native && uv venv --python 3.12 .venv \
  && uv pip install -p .venv/bin/python -e . pytest numpy \
  && .venv/bin/python -m pytest tests/ -v

# 프론트엔드 테스트 (Node 22 필요 — 의존성 설치 불필요, 리포 루트에서 실행)
npm test --prefix frontend
```

기기 의존 테스트는 기본 스위트에서 건너뛰고, 명시적으로 켭니다: `make test-mps`
(`OCR_MPS_TESTS=1` — torch MPS 계약, torch·macOS 업그레이드 전후)와 `make test-mlx-real`
(`OCR_MLX_REAL_TESTS=1` — MLX 실가중치 패리티, 고정 스냅샷이 로컬 HF 캐시에 있어야 하며
`make dev`를 한 번 띄우면 받아진다. `.env`의 `HF_HOME`·`HF_HUB_CACHE`도 `make dev`처럼 반영하고,
가중치가 없으면 pytest 전에 사유를 보이고 종료코드 2로 멈춘다 — `scripts/require_hf_snapshot.py`.
fp32 MLX·torch 모델을 차례로 올려 최고 메모리가 약 23GiB라 32GB 이상 Mac에서 돌린다).
둘 다 건너뛴 테스트의 사유를 보여 줍니다(`-rs`).

환경변수 전체 목록: [docs/ARCHITECTURE.md §7](docs/ARCHITECTURE.md) —
`OCR_DEVICE`(auto/mlx/metal/cpu/cuda), `OCR_DTYPE`, `OCR_MLX_QUANT_BITS`,
`OCR_ENGINE`(unlimited/fake/textlayer/ovisocr2/paddleocr_vl), `OCR_SIDECAR_URL`,
`PAGES_PER_CHUNK`, `RENDER_DPI`, `MAX_UPLOAD_MB`, `PDF_PAGE_TIMEOUT_S` 등.

## Makefile

리포 루트의 `make` 타깃 (uv·npm 기반 — 위 명령들의 축약):

| 타깃 | 동작 |
|---|---|
| `make setup` | backend 의존성 설치 (`uv sync --extra cpu`) — Apple Silicon에서는 mlx·C++ 모듈을 지우므로 아래를 쓴다 |
| `make setup-mlx` | **Apple Silicon 권장** — MLX 엔진 + torch MPS 폴백 + C++ 모듈 (`uv sync --extra metal --extra mlx && uv pip install ../native`) |
| `make setup-metal` | `setup-mlx`와 같다 (예전 metal 단독 sync는 mlx·C++ 모듈을 지웠다) |
| `make setup-native` | C++ 가속 모듈 설치 (선택) |
| `make dev` | 개발 서버 — `127.0.0.1:8000`(`make dev PORT=8010`으로 변경), `--reload`, 디바이스 자동 선택(`make dev OCR_DEVICE=cpu`처럼 지정 가능) |
| `make dev-metal` | torch MPS(Metal) 폴백으로 개발 서버 — MLX와 결과·속도 비교용 |
| `make dev-textlayer` | `OCR_ENGINE=textlayer`로 개발 서버 (모델 불필요) |
| `make test` | 핵심 로컬 검사 — backend pytest · ruff · frontend 테스트 |
| `make test-mps` | Apple Silicon torch MPS 계약 테스트 (`OCR_MPS_TESTS=1`) — torch·macOS 업그레이드 전후 |
| `make test-mlx-real` | MLX 실가중치 패리티 테스트 (`OCR_MLX_REAL_TESTS=1`) — mlx 업그레이드·MLX 포팅 수정·스냅샷 갱신 전후, 스냅샷이 없으면 종료코드 2 |
| `make coverage` | backend 커버리지 (`pytest-cov`를 `--with`로 임시 설치 — `uv.lock` 무변경) |
| `make audit` | 의존성 취약점 감사 — CI `dependency-audit` 잡과 같은 pip-audit·수용 목록 (네트워크 필요) |
| `make e2e` | `./scripts/smoke_e2e.sh` (기동된 백엔드 필요) |
| `make verify-e2e` | 실 PDF 전 구간 점검 — 업로드→OCR→번역→PDF→뷰어→보안 (서버를 직접 띄움, 외부 API 없음) |
| `make e2e-mock` | hermetic 브라우저 E2E — mock OpenAI + FakeEngine (playwright chromium 설치) |
| `make docker-up` / `make docker-down` | CPU 스택(ocr-cpu) 기동/정리 |
| `make docker-up-ollama` / `make docker-down-ollama` | ocr-cpu + Ollama 컨테이너 (overlay) |
| `make docker-pull-model` | 컨테이너 Ollama 모델 다운로드 (`MODEL=qwen3:8b` 기본) |

GPU 스택(cuda/ovis/paddle)은 compose 프로필 함정(서비스명 명시 필수) 때문에
make로 감싸지 않습니다 — 위 §빠른 시작의 blessed 명령을 그대로 사용하세요.
개발 서버(`dev`·`dev-metal`·`dev-textlayer`)는 Dockerfile과 같은
`--timeout-graceful-shutdown 5`로 떠서 열린 SSE가 종료·리로드를 붙잡지 않습니다.

## 동작 방식

```
PDF 업로드 → 업로드 검증(페이지 수·크기·작업량 게이트 — 검증 워커 프로세스)
          → 페이지 PNG 렌더(기본 200dpi — OCR 워커 프로세스, 페이지마다 시간 상한)
          → infer_multi()가 8페이지 청크 단위 one-shot 파싱 (<PAGE> 마커로 페이지 구분)
          → 충실도 게이트: 원본 텍스트 레이어와 대조해 청크가 놓친 페이지만 단독 재처리
          → figure 크롭(images/) · 레이아웃 오버레이(layout/) · 참조 재작성
          → 페이지 병합 result.md → 미리보기/다운로드(.md, .zip)
```

- 모델 코드는 `backend/app/vendor/unlimited_ocr/`에 **벤더링**되어 있습니다
  (revision 고정, `trust_remote_code` 불필요). 업스트림은 CUDA 전용이라
  CPU 지원 패치 + `eval()` 보안 패치를 적용했습니다 — 내역:
  [PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md)
- Apple Silicon의 MLX 엔진은 같은 모델의 MLX 포팅(`backend/app/vendor/unlimited_ocr_mlx/`)을
  씁니다 — 로컬 패치 M1–M7(CLIP 활성함수·원샷 프리필·위치 임베딩 리샘플 등) 내역:
  [PROVENANCE.md](backend/app/vendor/unlimited_ocr_mlx/PROVENANCE.md)
- `per_page` 모드(요청 옵션)는 페이지별 gundam 프리셋(1024/640/crop)으로 처리합니다.
- **실패 격리**: 청크가 실패하면(출력 상한 `MAX_LENGTH` 도달 포함) 끝까지 생성된 앞 페이지는
  지키고 나머지만 페이지별로 다시 처리하며, 그래도 안 되면 PDF 텍스트 레이어 → 실패
  플레이스홀더 순으로 메웁니다. 기본 설정(`MAX_LENGTH` 32,768 < 8쪽 최악 길이)에서는 기동 시
  안내가 INFO로 한 번 남습니다 — 잘린 청크는 끝까지 생성된 앞 페이지(텍스트 레이어가 있으면
  원본 본문과 대조해 확인한 페이지)만 지키고 잘린 페이지부터 페이지별로 다시 처리합니다.
  대조할 텍스트 레이어가 없는 스캔은 모델의 페이지 마커를 그대로 믿습니다.
- **PyMuPDF 격리**: PDF 렌더·분석·내보내기는 서버 밖 워커 프로세스(ocr 1개 · export
  `PDF_EXPORT_MAX_CONCURRENT`개 · 업로드 검증 2개)에서 돕니다. MuPDF는 GIL을 쥔 채 돌기
  때문에 서버 안에서 돌리면 번역 PDF 빌드 동안 OCR 디코드가 멎었습니다(M4 Max 실측 MLX
  약 290 → 70 tok/s).
- **수식 렌더링**: 모델의 `\(…\)`/`\[…\]` LaTeX를 렌더 레이어에서 정규화해
  (mdit-py-plugins dollarmath) 로컬 벤더링된 **KaTeX**(`frontend/vendor/katex/`,
  외부 CDN 없음)로 타이포셋합니다. 다운로드되는 `result.md`에는 원본 LaTeX가
  그대로 유지됩니다.
- **렌더 충실도**: figure는 그라운딩 bbox로 계산한 **원본 페이지 대비 상대
  폭**으로 표시되고(좁으면 센터링), 최종 미리보기는 페이지별
  `<section class="doc-page">`로 구분됩니다. 결과 탭의 **레이아웃** 뷰는
  전 블록의 좌표로 다단 배치까지 근사 재구성합니다 (best-effort — 텍스트
  리플로우/검색은 마크다운 뷰 담당). 이 모든 변형은 렌더 레이어 전용이며
  `result.md`는 순수 마크다운으로 유지됩니다.
- C++ 모듈(`native/`)은 토큰 생성 핫패스(no-repeat-ngram)를 가속합니다.
  없으면 순수 파이썬 폴백으로 동일하게 동작합니다.
- no-repeat-ngram 검사는 디바이스별 최적 경로를 탑니다: CUDA/MPS는 **GPU 상주
  torch 구현**(토큰마다 발생하던 시퀀스 D2H 복사·동기화 제거 — MPS는 창 길이를 고정해
  길이마다 새 Metal 그래프를 만들지 않는다), MLX는 같은 의미론의 MLX 구현, CPU는 마지막
  window 토큰만 슬라이스해 C++/파이썬으로 스캔 — 모든 구현이 레퍼런스와
  패리티 테스트로 검증됩니다. 참고: batch=1 자기회귀 디코드 특성상 GPU
  사용률은 원래 낮습니다(HF generate 루프가 지배) — 대량 처리 스루풋이
  필요하면 모델이 공식 지원하는 vLLM/SGLang 서빙을 고려하세요.
- CPU 스레드 수는 `OCR_CPU_THREADS`, CUDA GPU 선택은 `GPU_DEVICE`(compose)로
  제어합니다.

## 디바이스 백엔드 현황

| 백엔드 | 상태 | 비고 |
|---|---|---|
| CPU | ✅ | 기본 float32 (`OCR_DTYPE=bfloat16` 가능) |
| CUDA | ✅ | bf16, torch 2.10 cu129 (sm_89·sm_120 확인) |
| MLX | ✅ | **Apple Silicon 기본**, `OCR_DEVICE=mlx`(또는 auto) — in-process MLX, bf16(선택 8비트), macOS 14+ arm64 로컬 실행 전용 (Docker 불가) |
| Metal | ✅ | torch MPS 폴백, `OCR_DEVICE=metal`(별칭 `mps`) — bf16, macOS 14+, Apple Silicon 로컬 실행 전용 (Docker 불가) |

`OCR_DEVICE`를 비우면(`auto`) unlimited 엔진이 mlx → cuda → metal → cpu 순으로 고르고
기동 로그와 `/api/health`의 `device`·`dtype`(예: `bfloat16+q8`)에 결과를 남깁니다. compose는
서비스마다 `cpu`/`cuda`를 고정하므로 컨테이너 동작은 그대로입니다.

## 문서

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 아키텍처, REST/SSE API 계약, 설계 결정
- [docs/OCR_ENGINE_PROTOCOL.md](docs/OCR_ENGINE_PROTOCOL.md) — backend ↔ GPU sidecar 프로토콜
- [docs/OCR_BENCHMARK.md](docs/OCR_BENCHMARK.md) — 엔진 비교·Apple Silicon 실측
- [SECURITY.md](SECURITY.md) — 노출·자격증명·브라우저·업로드 방어 정책
- [CHANGELOG.md](CHANGELOG.md) — 버전별 변경사항
- [backend/app/vendor/unlimited_ocr/PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md) — torch 벤더링/패치 내역
- [backend/app/vendor/unlimited_ocr_mlx/PROVENANCE.md](backend/app/vendor/unlimited_ocr_mlx/PROVENANCE.md) — MLX 포팅 출처/패치·실측 내역

## 라이선스

프로젝트 코드는 [MIT](LICENSE)입니다. 벤더링된 코드는 각자의 라이선스를 따릅니다:

- 모델 가중치·벤더링된 모델 코드 — Baidu MIT
  ([backend/app/vendor/unlimited_ocr/LICENSE](backend/app/vendor/unlimited_ocr/LICENSE))
- MLX 포팅 모델 코드 — mlx-vlm MIT
  ([backend/app/vendor/unlimited_ocr_mlx/LICENSE](backend/app/vendor/unlimited_ocr_mlx/LICENSE))
- KaTeX — MIT ([frontend/vendor/katex/LICENSE](frontend/vendor/katex/LICENSE))
