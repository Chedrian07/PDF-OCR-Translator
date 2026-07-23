# PDF OCR Translator — PDF 그대로 읽고 번역하기

> 이 리포는 **Unlimited-OCR(PDF→Markdown)** 와 **Localight(로컬 우선 논문 리더)** 를
> 병합한 워크스페이스입니다 — textlayer 엔진과 페이지 Q&A('질문' 탭)가 Localight에서 이식되었습니다.

<!-- CI 배지 — 병합 리포의 원격 주소 확정 후 <OWNER>/<REPO>를 실제 경로로 교체:
[![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml) -->
CI는 push/PR마다 backend pytest·ruff·frontend·native 패리티 테스트를 실행합니다.

웹에서 PDF를 업로드하면 로컬 비전-언어 OCR 모델로 **이미지(figure)까지 추출된
Markdown**을 만들어 주는 셀프호스팅 서비스입니다. 기본 엔진은
[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)(3.3B MoE, MIT)이고,
단일 RTX 5070 Ti(16GB) 기준으로 **엔진을 선택**할 수 있습니다:

| 엔진 | 선택 기준 | 실행 |
|---|---|---|
| **Unlimited-OCR** (기본) | 멀티페이지 문맥 · 실시간 토큰 스트리밍 | `docker compose up -d --build ocr-cuda` → :8001 |
| **OvisOCR2** (0.9B, Apache-2.0) | 속도(워밍업 후 ~1.1s/p) · figure bbox | `docker compose --profile ovis up -d --build ovisocr2 ocr-ovis` → :8002 |
| **PaddleOCR-VL-1.6** (0.9B, Apache-2.0) | **한국어 정확도** · 표/수식 · 완전한 layout | `docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle` → :8003 |
| **textlayer** (모델 불필요, Localight 이식) | **텍스트 PDF 최적** · CPU 전용 · 즉시 시작 | `OCR_ENGINE=textlayer docker compose up -d --build` → :8000 |

> RTX 5070 Ti 실측 비교(속도·VRAM·한국어 오독 사례)는
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
- **디바이스 백엔드**: CPU / CUDA / Metal(Apple Silicon, torch MPS) 지원
- **한국어 번역 (선택)**: 변환 결과를 OpenAI 호환 API로 한국어 번역 — §한국어 번역 참조
- **원커맨드 배포**: `docker compose up` 하나로 끝 (Metal은 로컬 실행 — 아래 참조)

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
- 한국어 번역 기능을 쓰려면 `cp .env.example .env` 후 API 키를 설정 — 아래 §한국어 번역 참조.

Docker 없이 로컬(uv)로 바로 시작할 수도 있습니다:

```bash
make setup && make dev        # http://127.0.0.1:8000 — 전체 타깃은 §Makefile
make dev-textlayer            # 모델 없이 textlayer 엔진으로 기동
```

### 모델 없이 UI/파이프라인만 체험

```bash
OCR_ENGINE=fake docker compose up -d --build
```

## 보안

이 서비스는 **인증이 없습니다** — 접근 가능한 사람은 누구나 문서 열람·삭제·변환·(설정 시)
유료 번역 트리거가 가능합니다. 그래서 compose는 기본적으로 **루프백(127.0.0.1)에만**
포트를 바인딩하고, `ALLOWED_HOSTS`(기본 `localhost,127.0.0.1`) 밖의 Host 헤더는
400으로 거부합니다(DNS rebinding 방어).

LAN이나 공개 네트워크에 노출하려면 **반드시 인증을 제공하는 리버스 프록시**
(예: nginx + basic auth, Tailscale/VPN) 뒤에 두세요. 그 후:

1. `docker-compose.yml`의 ports를 `"8000:8000"`으로 변경 (또는 프록시만 컨테이너에 접근)
2. 접속에 쓸 호스트명/IP를 `.env`의 `ALLOWED_HOSTS`에 추가 — 예:
   `ALLOWED_HOSTS=localhost,127.0.0.1,ocr.example.com` (포트는 비교 시 무시됨).
   compose가 컨테이너로 전달합니다.

## 한국어 번역

변환이 끝난 문서(`result.md` + 레이아웃)를 OpenAI 호환 API로 한국어 번역해
번역본 미리보기/레이아웃/다운로드를 제공합니다 (수식·이미지·표는 마스킹으로 보존).

```bash
cp .env.example .env   # 키 설정 후 docker compose up -d 로 재기동
```

`.env`에 아래 값을 설정하면 활성화됩니다:

- `OPENAI_BASE_URL` — OpenAI 호환 base URL. `https://host`처럼 origin만 쓰면
  `/v1`을 자동 보완하며, 명시한 버전/게이트웨이 경로는 그대로 사용합니다.
- `OPENAI_API_KEY` — 로컬 서버는 생략 가능
- `OPENAI_MODEL` — 번역에 쓸 모델 ID

미설정이어도 나머지 기능은 그대로 동작합니다 — 번역 요청 시에만 503과 함께
"번역 프로바이더가 설정되지 않았습니다" 안내가 표시됩니다.
동시성/재시도/reasoning 등 세부 옵션과 파이프라인 설계는
[docs/ARCHITECTURE.md §13](docs/ARCHITECTURE.md#13-한국어-번역-translation) 참조.
번역 요청 동시성은 기본 8이며 `TRANSLATE_CONCURRENCY`를 1–8 범위에서 조정할 수
있습니다. 8을 넘는 값은 API 서버 상한 보호를 위해 자동으로 8로 제한됩니다.

## 읽기 뷰 (기본 화면)

변환이 완료된 논문은 **'읽기' 탭**이 기본으로 열립니다. 왼쪽은 언어 전환과
무관하게 원본 PDF를 좌표 기준면으로 유지하고, 오른쪽은 현재 페이지의 원문 또는
한국어 번역 블록을 읽기 레일로 보여 줍니다.

- 페이지 이동: ◀/▶ 버튼, 페이지 번호 입력, 키보드 ←/→
- 확대/축소와 너비 맞춤: 60–220%, 브라우저에 저장
- 번역 전이면 **'한국어로 읽기'** 버튼으로 곧바로 전체 번역을 시작할 수 있고,
  완료되면 자동으로 한국어 레일로 전환됩니다 ([원문|한국어] 토글로 언제든 대조)
- 오른쪽 문단을 가리키거나 누르면 같은 OCR 블록의 원문 bbox가 왼쪽에서 강조되고,
  왼쪽 bbox를 누르면 대응 번역문으로 자동 이동합니다
- OCR `title` 블록으로 만든 문서 개요에서 섹션을 눌러 페이지로 이동
- 페이지 요약, 선택 문장 설명, 세션 하이라이트, 인용 저장을 연구 레일에서 실행
- 'AI 질문'과 선택 설명은 기존 질문 탭으로 현재 페이지·문장을 그대로 전달

## 내보내기

완역본을 두 형식으로 내려받을 수 있습니다 (결과 화면의 다운로드 줄):

| 형식 | 버튼 | 내용 |
| --- | --- | --- |
| HTML | `원본 HTML` / `한국어 HTML` | **PDF facsimile 단일 파일** — 완성 페이지 PNG를 인라인하고 OCR 블록을 검색·복사용 투명 텍스트 레이어로 보존 |
| PDF | `한국어 PDF` | **레이아웃 보존 번역 PDF** — 일반 텍스트와 구조가 안정적인 표 셀을 교체. 독립 수식·그림·참고문헌·세로쓰기는 원본 유지 |

기존 `/layout.html` 주소는 같은 파일을 중복 생성하지 않고 정식
`/document.html` 내보내기로 리다이렉트합니다.

PDF 내보내기(`GET /api/jobs/{id}/pdf?lang=ko`)는 한국어 번역이 완료된 잡에서
활성화되며, 좌표 레이아웃이 없는 figure_only 엔진(OvisOCR2)에서는 HTML
내보내기를 안내합니다. 원본 PDF span의 serif/sans 계열과 실측 글자 크기를 블록마다
추출해, 본문·절 제목은 시스템 명조(macOS AppleMyungjo, Linux Noto Serif CJK),
sans 대표 제목은 시스템 고딕으로 대응시킵니다. 한 줄 제목·목록은 원본 baseline과
크기를 직접 보존하고, 여러 줄 본문만 충돌 없는 범위에서 행간·줄바꿈을 맞춥니다.
글꼴이 없으면 PyMuPDF 내장 CJK로 폴백하며 `PDF_EXPORT_FONT`로
원하는 폰트 파일을 지정할 수 있습니다. 원문과 번역 레이아웃의 블록 대응을 먼저
검증하고, 번역문이 최소 크기에도 들어가지 않는 블록은 원문을 지우지 않고 보존합니다.
텍스트 리댁션은 이미지·벡터 그래픽을 건드리지 않으며 원본 PDF 텍스트 레이어의
실측 폰트 크기를 자동으로 사용합니다. 공간이 부족하면 이웃 블록 전의 빈 영역까지만
확장하고, 다운로드 뒤 번역·표 셀·재배치·원문 보존 개수를 토스트로 알려줍니다.
참고문헌은 저자·학술지·URL의 서지 형식과 원문 조판을 그대로 보존합니다.
한국어 HTML은 이 번역 PDF를 같은 DPI로 렌더한 페이지를 기준면으로 사용하므로
HTML과 PDF의 제목·단·그림·수식 위치가 서로 달라지지 않습니다.
읽기 패널의 흐름형 Markdown도 원문 줄과 레이아웃 블록이 대응하면 같은 블록
번역을 재사용하므로, PDF·문서 개요·오른쪽 텍스트의 제목과 용어가 서로 달라지지
않습니다.

## 페이지 Q&A (AI에게 묻기)

변환된 문서를 보면서 결과 화면의 **'질문' 탭**으로 현재 페이지 내용에 대해
AI에게 물을 수 있습니다 (Localight에서 이식). 공급자는 셋 중 선택
(`LLM_PROVIDER`, 기본 `openai-responses`):

| 공급자 | 엔드포인트 | Reasoning/Thinking |
| --- | --- | --- |
| `openai-responses` | `/v1/responses` | `reasoning.effort`, 선택적 `reasoning.summary` |
| `openai-chat` | `/v1/chat/completions` | `reasoning_effort` |
| `ollama` | `/api/chat` | `think: false/true/low/medium/high` |

- API: `GET /api/providers`(공급자·모델 목록) · `POST /api/jobs/{id}/qa`(페이지 질의).
  활성 여부는 `GET /api/health`의 `qa_available`·`llm_default_provider`로 확인합니다.
- OpenAI 인증은 번역과 **같은 `OPENAI_API_KEY`를 공유**합니다.
  `LLM_OPENAI_BASE_URL`은 공식 `https://api.openai.com` 호스트만 허용됩니다.
- 지원 effort는 모델마다 다르므로 API가 거부하는 조합은 UI 오류로 그대로 안내합니다.

**프라이버시 약속** (Localight에서 승계):

- 외부로는 **현재 페이지에서 추출된 텍스트만** 전송합니다 — 원본 PDF·페이지 이미지는 전송하지 않습니다.
- OpenAI 요청은 `store: false`로 생성합니다.
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

## macOS에서 Metal(MPS)로 실행

Docker(맥의 Linux VM)에서는 GPU 패스스루가 없어 Metal을 쓸 수 없습니다.
Apple Silicon Mac에서는 로컬(uv)로 실행하세요:

```bash
cd backend
uv sync --extra metal        # macOS arm64 torch 휠 (MPS 내장)
uv pip install ../native     # 선택 — C++ 가속
OCR_DEVICE=metal uv run uvicorn app.main:app   # http://localhost:8000
```

- dtype은 `auto`면 bfloat16(macOS 14+), 미지원 조합이면 float32로 자동 폴백
- 첫 청크는 Metal 셰이더 컴파일 때문에 이후 청크보다 느릴 수 있습니다
- 청크가 끝날 때마다 `torch.mps.empty_cache()`로 유니파이드 메모리를 반환합니다

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

## 로컬 개발 (uv)

```bash
# 백엔드 (Python 3.12, torch CPU)
cd backend
uv sync --extra cpu          # CUDA: --extra cu129 · macOS Metal: --extra metal
uv pip install ../native     # C++ 가속 모듈 (선택 — 없어도 동작)
uv run pytest                # 유닛/통합 테스트 (FakeEngine, 모델 불필요)
uv run uvicorn app.main:app --reload   # http://localhost:8000

# 네이티브 모듈 단독 테스트
cd native && uv venv --python 3.12 .venv \
  && uv pip install -p .venv/bin/python -e . pytest numpy \
  && .venv/bin/python -m pytest tests/ -v

# 프론트엔드 테스트 (Node 22 필요 — 의존성 설치 불필요, 리포 루트에서 실행)
npm test --prefix frontend
```

환경변수 전체 목록: [docs/ARCHITECTURE.md §7](docs/ARCHITECTURE.md) —
`OCR_DEVICE`(cpu/cuda/metal), `OCR_DTYPE`,
`OCR_ENGINE`(unlimited/fake/textlayer/ovisocr2/paddleocr_vl), `OCR_SIDECAR_URL`,
`PAGES_PER_CHUNK`, `RENDER_DPI`, `MAX_UPLOAD_MB` 등.

## Makefile

리포 루트의 `make` 타깃 (uv·npm 기반 — 위 명령들의 축약):

| 타깃 | 동작 |
|---|---|
| `make setup` | backend 의존성 설치 (`uv sync --extra cpu`) |
| `make setup-metal` | macOS Metal용 설치 (`--extra metal`) |
| `make setup-native` | C++ 가속 모듈 설치 (선택) |
| `make dev` | 개발 서버 — `127.0.0.1:8000`, `--reload` |
| `make dev-textlayer` | `OCR_ENGINE=textlayer`로 개발 서버 (모델 불필요) |
| `make test` | CI와 동일 — backend pytest · ruff · frontend 테스트 |
| `make e2e` | `./scripts/smoke_e2e.sh` (기동된 백엔드 필요) |
| `make docker-up` / `make docker-down` | CPU 스택(ocr-cpu) 기동/정리 |
| `make docker-up-ollama` / `make docker-down-ollama` | ocr-cpu + Ollama 컨테이너 (overlay) |
| `make docker-pull-model` | 컨테이너 Ollama 모델 다운로드 (`MODEL=qwen3:8b` 기본) |

GPU 스택(cuda/ovis/paddle)은 compose 프로필 함정(서비스명 명시 필수) 때문에
make로 감싸지 않습니다 — 위 §빠른 시작의 blessed 명령을 그대로 사용하세요.

## 동작 방식

```
PDF 업로드 → pymupdf로 페이지 PNG 렌더(기본 200dpi)
          → infer_multi()가 8페이지 청크 단위 one-shot 파싱 (<PAGE> 마커로 페이지 구분)
          → figure 크롭(images/) · 레이아웃 오버레이(layout/) · 참조 재작성
          → 페이지 병합 result.md → 미리보기/다운로드(.md, .zip)
```

- 모델 코드는 `backend/app/vendor/unlimited_ocr/`에 **벤더링**되어 있습니다
  (revision 고정, `trust_remote_code` 불필요). 업스트림은 CUDA 전용이라
  CPU 지원 패치 + `eval()` 보안 패치를 적용했습니다 — 내역:
  [PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md)
- `per_page` 모드(요청 옵션)는 페이지별 gundam 프리셋(1024/640/crop)으로 처리합니다.
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
  torch 구현**(토큰마다 발생하던 시퀀스 D2H 복사·동기화 제거), CPU는 마지막
  window 토큰만 슬라이스해 C++/파이썬으로 스캔 — 세 구현 모두 레퍼런스와
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
| Metal | ✅ | torch MPS, `OCR_DEVICE=metal`(별칭 `mps`) — bf16, Apple Silicon 로컬 실행 전용 (Docker 불가) |

## 문서

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 아키텍처, REST/SSE API 계약, 설계 결정
- [backend/app/vendor/unlimited_ocr/PROVENANCE.md](backend/app/vendor/unlimited_ocr/PROVENANCE.md) — 벤더링/패치 내역

## 라이선스

프로젝트 코드는 [MIT](LICENSE)입니다. 벤더링된 코드는 각자의 라이선스를 따릅니다:

- 모델 가중치·벤더링된 모델 코드 — Baidu MIT
  ([backend/app/vendor/unlimited_ocr/LICENSE](backend/app/vendor/unlimited_ocr/LICENSE))
- KaTeX — MIT ([frontend/vendor/katex/LICENSE](frontend/vendor/katex/LICENSE))
