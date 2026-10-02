# Unlimited-OCR PDF → Markdown 변환 서비스 — 아키텍처 & API 계약

> 이 문서는 본 프로젝트의 **단일 진실 공급원(SSOT)** 입니다.
> 백엔드/프론트엔드/네이티브 모듈은 모두 이 문서의 계약을 따릅니다.

## 1. 개요

웹에서 PDF를 업로드하면 [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
(3.3B MoE VLM, DeepSeek-OCR 계열, MIT)로 파싱하여 **이미지(figure)까지 포함된 Markdown**으로
변환해 주는 셀프호스팅 애플리케이션.

- 디바이스 백엔드: **CPU**, **CUDA**, Apple Silicon — **MLX**(in-process MLX 포팅, 기본)와
  **Metal**(torch MPS, 폴백). `OCR_DEVICE=auto`(미설정 기본)가 쓸 수 있는 가장 빠른 것을 고른다(§6)
- 배포: `docker compose up` 한 번으로 실행 (CPU 기본, GPU는 `docker compose up ocr-cuda`).
  Apple Silicon GPU(MLX·Metal)는 컨테이너 GPU 패스스루가 없어 **로컬(uv) 실행 전용**
- 신뢰할 수 없는 PDF를 다루는 PyMuPDF 작업은 서버 밖 워커 프로세스에서 시간 상한과 함께 돈다(§18)
- 개발 스택: Python 3.12 (uv 관리) + C++17 (pybind11 네이티브 모듈)
- **멀티 엔진 (RTX 5070 Ti 단일 GPU)**: `OCR_ENGINE=ovisocr2|paddleocr_vl`은
  GPU 전용 sidecar 컨테이너(`services/`)와 HTTP로 통신한다 — backend 프로세스는
  GPU 미사용, 한 시점에 GPU 모델 하나만 활성. 계약:
  [OCR_ENGINE_PROTOCOL.md](OCR_ENGINE_PROTOCOL.md), 계획/근거:
  [CUDA_5070TI_MULTI_OCR_PLAN.md](CUDA_5070TI_MULTI_OCR_PLAN.md)

## 2. 모델 사용 방식 (리서치 결과 요약)

| 항목 | 내용 |
|---|---|
| 모델 | `baidu/Unlimited-OCR`, revision `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead` 고정 |
| 로딩 | 벤더링된 모델 코드(`backend/app/vendor/unlimited_ocr/`)의 `UnlimitedOCRForCausalLM.from_pretrained()` — `trust_remote_code` 불필요. 커밋 해시로 고정한 리비전은 `local_files_only`로 HF 캐시에서 먼저 읽고, 캐시에 없거나 불완전할 때만 Hub에 묻는다(`app/engine/hf_snapshot.py` — 예전에는 캐시가 완전해도 로드마다 선택 파일 HEAD 6회·API 1회로 huggingface.co에 접속했다). 캐시만 보는 시도가 **어떤 예외로든** 실패하면 Hub 호출로 한 번 다시 한다 — 중단된 첫 다운로드(tokenizer.json 누락)는 OSError가 아니라 ImportError(protobuf)·ValueError(trust_remote_code)로 끝나기 때문이다. MLX 경로가 캐시 디렉터리를 바로 쓰는 완전성 판정은 config.json·tokenizer_config.json·토크나이저 본체(tokenizer.json 또는 tokenizer.model)·인덱스의 샤드 전부다. 브랜치·태그 리비전은 갱신을 받도록 예전처럼 Hub에 묻는다 |
| 단일 이미지 | `model.infer(tokenizer, prompt='<image>document parsing.', ...)` — gundam(1024/640/crop) 또는 base(1024/1024) |
| PDF/멀티페이지 | `model.infer_multi(tokenizer, prompt='<image>Multi page parsing.', image_files=[...], image_size=1024, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024, save_results=True)` |
| 페이지 구분 | 출력 텍스트에 `<PAGE>` 마커 |
| 이미지(figure) 추출 | 모델이 `<\|ref\|>image<\|/ref\|><\|det\|>[[x1,y1,x2,y2]]<\|/det\|>` (0–999 정규화 좌표) 출력 → 원본 페이지에서 크롭하여 `{out}/images/page_{i}_{k}.jpg` 저장, 마크다운에는 `![](images/page_{i}_{k}.jpg)` 치환 |
| 레이아웃 시각화 | 페이지별 `result_with_boxes_{i}.jpg` 저장 (GIF 데모의 박스 오버레이) |
| 고정 의존성 | 모델 README 기준 — torch==2.10.0, torchvision==0.25.0, transformers==4.57.1 등. 벤더 모델 코드가 이 API에 묶여 고정하며, 남은 pip-audit 권고는 수용한 잔여 위험이다(근거는 `backend/pyproject.toml` 고정 옆 주석) |
| 보안 고정 | 모델 수치와 무관한 I/O 라이브러리는 README 고정에서 이탈 — pymupdf==1.28.2(동봉 MuPDF의 CVE-2026-3308 수정), pillow==12.3.0. 전이 의존 urllib3≥2.8.0·anyio≥4.14.2와 함께 `backend/tests/test_dependency_floor.py`가 하한을 지킨다 |
| MLX extra | `mlx==0.32.3`(darwin-arm64 마커, macOS 14+ 휠) — in-process MLX OCR 엔진(§6)의 런타임. torch와 겹치지 않아 `uv sync --extra metal --extra mlx`(= `make setup-mlx`)로 함께 설치한다. metal 단독·mlx 단독 sync는 서로를(그리고 `uv pip`로 넣은 C++ 모듈을) 지운다 |
| MLX 포팅 | `backend/app/vendor/unlimited_ocr_mlx/` — mlx-vlm **0.7.4**의 Unlimited-OCR 모델 코드를 벤더링(MIT). mlx-vlm(transformers≥5.14 요구)·torch에 의존하지 않고 transformers 4.57.1은 토크나이저에만 쓴다. torch 경로와 **같은 스냅샷**을 변환 없이 strict 로드한다 |
| CUDA 휠 | cu129 (README 테스트 환경 CUDA 12.9, Blackwell sm_120 포함) |
| flash-attn | 선택 사항 (미설치 시 eager attention) — 본 프로젝트는 미사용 |

### 벤더링 패치 (backend/app/vendor/unlimited_ocr/)
업스트림 코드는 `.cuda()` 및 `torch.autocast("cuda")`가 하드코딩되어 CPU에서 동작 불가.
다음 최소 패치를 적용하며, 전체 내역은 `PROVENANCE.md`에 기록:
1. `.cuda()` → 모델 파라미터 디바이스 기준 `.to(dev)`
2. `torch.autocast("cuda", bfloat16)` → 디바이스/디타입 조건부 autocast
3. `masked_scatter_`의 마스크 디바이스 수정 (modeling_unlimitedocr.py:582)
4. 이미지 텐서 `.to(torch.bfloat16)` 하드코딩 → 모델 dtype 추종
5. `infer`/`infer_multi`에 `streamer=None`, `stopping_criteria=None` 파라미터 추가 (SSE 스트리밍/취소용, 기본값이면 업스트림과 동일 동작)
6. 이미지 임베딩 주입을 `masked_scatter_` → bool 인덱싱 대입으로 교체 (torch 2.10 MPS의
   브로드캐스트 마스크 버그 회피 — CPU/CUDA 결과 동일, PROVENANCE P11)
7. `_autocast_ctx`는 `mps`에서 항상 no-op — MPS autocast(bf16)의 로짓 오염 회피 (PROVENANCE P12)
8. 디코드 가속 — P16(SDPA, CUDA 기본·MPS는 eager 유지 — M4 Max 재측정에서 붕괴는 없고
   15–28% 빨랐지만 출력이 비트 동일하지 않아 `OCR_SDPA=1` 옵트인), P17(융합 MoE — **CUDA·MPS
   기본**, 엔진이 `.to(device)` **전에** `prebuild_fused_moe`로 expert 가중치를 한 번 스택해
   뷰로 재지정한다. `OCR_MOE_FUSED=0`이면 프리빌드까지 건너뛰어 legacy 완전 복원), P18(MPS 단일
   토큰 패스트패스 — 이제 `OCR_MOE_FUSED=0`일 때의 MPS 폴백, 레이어당 동기화가 남아 실측 1.20배),
   P20(디코드 링 슬롯 — eager는 호스트 int + 슬라이스 `copy_`, CUDA Graph 캡처 때만 텐서 슬롯.
   MPS의 `index_copy_`가 KV 길이에 비례해 8쪽 청크 디코드를 34 tok/s로 떨어뜨리던 것을 되돌렸다)
9. P22 — 모델 bbox를 픽셀로 바꾼 뒤 **페이지 안으로 clamp**하고 퇴화 상자는 건너뛰되 크롭 번호는
   소비한다(마크다운 참조·boxes.json 정렬 유지). 거대·음수 좌표가 페이지보다 큰 크롭이나 Pillow
   좌표 산술 오버플로에 닿지 않게 하는 보안 패치이며 MLX 포팅도 같은 규칙을 쓴다. 좌표를 못 읽은
   image 매치(쉼표 누락·이름·정수 자릿수 상한)와 상자 목록이 아닌 det 값(문자열·빈 목록·None)·
   기형 평평 목록도 image 매치당 번호 1개를 쓰고, image 판정은 마크다운 치환과 같은 기준
   (`_is_image_ref`, 공백 라벨 포함)이다. layout.json의 `crop_index`도 같은 규칙으로 센다
   (`pipeline/layout.py::_image_slots`) — 레이아웃 뷰·이동 페이지 재크롭이 벤더 파일과 같은 번호를 쓴다
10. P23 — `infer_multi`의 페이지 분할은 첫 `<PAGE>` 앞이 **공백일 때만** 버린다(마커 0개면 전체가
    1쪽) — 업스트림은 늘 버려 선행 마커를 생략한 출력의 1쪽이 사라졌다. `merge.split_pages`와 같은 규칙
    (MLX 포팅도 같은 규칙)

### MLX 포팅 패치 (backend/app/vendor/unlimited_ocr_mlx/)

mlx-vlm 0.7.4 원본 대비 로컬 패치(전체 내역·sha256·측정은 그 디렉터리의 `PROVENANCE.md`,
소스에서는 `grep -n "local patch M"`):

| # | 내용 | 이유 |
|---|---|---|
| M1 | CLIP MLP를 `quick_gelu`, LayerNorm eps 1e-5로 | mlx-vlm 버그 — 원본(torch)과 달라 CLIP 출력 상대오차 0.309, 1쪽 det 블록 19 → 11 |
| M2 | 원샷 프리필(`prefill_length = P`) | mlx-vlm chunked prefill은 P-1을 기록해 마지막 프롬프트 토큰이 링 캐시에서 밀려났다(8쪽 출력이 토큰 105에서 갈림) |
| M3 | 생성 모듈 — GPU no-repeat-ngram, 토큰 콜백, 토큰마다 취소·반복 확인, `hit_max_length` | 엔진의 스트리밍·취소·반복 감지·`OutputLimitError` 계약 |
| M4 | torch 없는 후처리(`ast.literal_eval`만 — P9) + torch P22(bbox clamp, 쓸 수 없는 image 상자·좌표를 못 읽은 image ref·상자 목록이 아닌 det 값도 크롭 번호 소비)·P23(첫 `<PAGE>` 앞 보존) | 산출물(마크다운·크롭·boxes.json·raw_pages.json)이 torch 흐름과 바이트 동일 — tests/test_mlx_postprocess.py가 모서리·무작위 입력으로 대조 |
| M5 | 인메모리 8비트 양자화 — 디코더만(group 64), 8 이외 비트 거부 | 처리량 약 1.44배·재현율 동일, 4비트는 숫자 오인식(2504 → 2304) |
| M6 | **기각** — MoE 게이트 fp32 계산 | 측정상 스파이크·MPS 출력에서 오히려 멀어짐 |
| M7 | gundam 크롭의 위치 임베딩 리샘플을 torch `F.interpolate`와 같은 가중치로 | mlx-vlm이 CLIP 채널 축을 리샘플하고 SAM rel_pos 좌표가 틀렸다 |

M1·M7은 mlx-vlm 업스트림에 알릴 만한 버그다. bf16 MLX 출력은 torch MPS bf16과 토큰 단위로
같지 않다(1쪽 첫 분기 122번째 토큰 — MPS의 fp16·bf16끼리도 같은 지점에서 갈린다). fp32에서는
torch CPU와 로짓 상대오차 1e-5 수준으로 맞으므로, 엔진 간 회귀 판정은 토큰 일치가 아니라
유사도·재현율로 한다.

## 3. 디렉터리 구조

```
├── docker-compose.yml          # ocr-cpu(기본)·ocr-cuda(cuda)·ocr-ovis+ovisocr2(ovis)·ocr-paddle+paddleocr-vl(paddle)
├── compose.ollama.yaml         # 선택 overlay — Ollama 컨테이너 추가 (§8)
├── Makefile                    # setup(-mlx/-metal)/dev(-metal/-textlayer)/test(-mps/-mlx-real)/audit/e2e*/verify-e2e/docker-*
├── .env.example                # 환경변수 템플릿 — 실제 키는 .env에 (커밋되지 않음)
├── README.md · SECURITY.md · CHANGELOG.md  # 사용법 / 보안·노출 정책 (§14와 정합) / 변경 이력
├── docs/
│   ├── ARCHITECTURE.md         # 이 문서 (SSOT)
│   ├── OCR_ENGINE_PROTOCOL.md  # sidecar 프로토콜 v1 계약
│   └── CUDA_5070TI_MULTI_OCR_PLAN.md · OVISOCR2_CUDA_5070TI.md
│       · PADDLEOCR_VL_BLACKWELL_5070TI.md · OCR_BENCHMARK.md · AUDIT/ROADMAP 문서
├── .github/                    # workflows/(ci·release) · dependabot.yml · trivyignore.yaml (§11)
├── backend/
│   ├── pyproject.toml          # uv 프로젝트, extras: cpu / cu129 / metal / mlx
│   ├── uv.lock                 # CI는 `uv sync --locked`로 lock 드리프트를 실패시킨다 (§11)
│   ├── Dockerfile              # ARG TORCH_VARIANT=cpu|cu129, digest 고정 베이스, tesseract 포함, 비루트(uid 1000)
│   ├── e2e_mock_app.py         # 브라우저 E2E 전용 진입점 (Q&A 라우터만 메모리 mock으로 교체)
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 팩토리 + TrustedHost + 보안 헤더(CSP) + 정적 프론트엔드 서빙
│   │   ├── config.py           # 환경변수 설정(Settings) + .env 로더 + 알려진 env 키 레지스트리
│   │   ├── api.py              # REST + SSE 라우트 + 남용 방어(레이트리밋·동시 상한, §5)
│   │   ├── jobs.py             # Job/JobStore + 단일 워커 큐 + SSE 브로커 + TTL GC (§15)
│   │   ├── owner_lock.py       # 잡 저장소 단일 소유자 flock (§4)
│   │   ├── qa.py               # 페이지 텍스트 컨텍스트 추출 (result.md 페이지 인덱스, §17)
│   │   ├── native_ops.py       # uocr_native 로더 + 순수 파이썬 폴백
│   │   ├── engine/
│   │   │   ├── base.py         # OCREngine 프로토콜 + EngineCapabilities + 오류 계약(OutputLimitError 등)
│   │   │   ├── registry.py     # 디바이스·엔진 선택 (auto 해석 · unlimited/fake/textlayer/ovisocr2/paddleocr_vl)
│   │   │   ├── unlimited.py    # 실모델 엔진 — torch (cpu/cuda/metal, 벤더링 코드 사용)
│   │   │   ├── unlimited_mlx.py  # 실모델 엔진 — MLX (Apple Silicon, vendor/unlimited_ocr_mlx 사용)
│   │   │   ├── fast_decode.py  # 커스텀 그리디 디코드 루프 (OCR_FAST_DECODE)
│   │   │   ├── objc_pool.py    # ObjC 오토릴리스 풀 (Metal 메모리 누적 방지, darwin 외 no-op)
│   │   │   ├── repetition.py   # 의미 반복·페이지 출력 폭주 감지 (StoppingCriteria)
│   │   │   ├── textlayer.py    # 텍스트 레이어 우선 + Tesseract 폴백 엔진 (§16)
│   │   │   ├── sidecar.py      # sidecar 엔진 (HTTP client + materializer 연결)
│   │   │   └── fake.py         # 테스트/데모용 가짜 엔진 (torch 불필요)
│   │   ├── sidecar/            # sidecar 공통 계층 (모델 독립)
│   │   │   ├── protocol.py     # 프로토콜 v1 스키마 + sanitize (OCR_ENGINE_PROTOCOL.md)
│   │   │   ├── client.py       # 동기 HTTP client (타임아웃/상한/취소/재시도)
│   │   │   └── materializer.py # normalized 결과 → 기존 청크 산출물 규약
│   │   ├── pipeline/
│   │   │   ├── pdf.py          # PDF → 페이지 PNG (pymupdf) · 업로드 검증(probe_pdf) · 텍스트 레이어 복구
│   │   │   ├── pdf_worker.py   # PyMuPDF 격리 워커 프로세스 풀 (ocr·export·probe, §18)
│   │   │   ├── pdf_complexity.py  # 업로드 복잡도 게이트 — 렌더 없이 페이지 작업량 측정 (§18)
│   │   │   ├── runner.py       # 잡 실행 오케스트레이션(렌더→청크 OCR→병합) + 실패 격리
│   │   │   ├── fidelity.py     # 페이지 OCR 충실도 게이트(원본 텍스트 레이어 = 정답)
│   │   │   ├── merge.py        # <PAGE> 분리, 슬롯 정렬, figure 리넘버링, result.md 페이지 경계 계약 (§4)
│   │   │   ├── reading_order.py  # 텍스트 레이어 블록의 읽기 순서(다단 감지)·조각 병합 (§16)
│   │   │   ├── layout.py       # raw_pages.json → layout.json 블록 파싱 + 단독 HTML 내보내기
│   │   │   ├── artifacts.py    # 잡 산출물 경로 · layout 사용 가능 판정(has_usable_layout)
│   │   │   ├── pdf_fonts.py    # 원본 텍스트 레이어의 실측 폰트 크기·굵기 주입
│   │   │   ├── render.py       # markdown → HTML (markdown-it-py) + document.html
│   │   │   ├── derived.py      # 파생 산출물(페이지 raster·export) 락 + 빌드 스탬프 + 전역 빌드 상한 (§5)
│   │   │   └── pdf_export/     # 레이아웃 보존 번역 PDF (단일/대조) — §5 /pdf. **패키지**
│   │   │       ├── __init__.py # 공개 API(build_translated_pdf 등) — 외부는 여기만 임포트
│   │   │       ├── build.py    # 페이지 순회·리댁션·삽입 오케스트레이션
│   │   │       ├── fitting.py  # 조판 dry-run(행간→축소 사다리) · 확장 공간 탐색
│   │   │       ├── spans.py · text.py · tables.py · raster_tables.py · geometry.py · fonts.py  # raster_tables = 스캔 표 픽셀 격자
│   │   │       └── models.py · constants.py · report.py  # report.py = PDF_EXPORT_FORMAT_VERSION
│   │   ├── translate/          # 번역 코어 (OCR 엔진·torch 무관) — §13
│   │   │   ├── engine.py       # run_translation (2단 패스·래더·캐시·state.json)
│   │   │   ├── client.py       # OpenAICompatClient (chat/responses 협상·재시도)
│   │   │   ├── segment.py      # md/layout → 유닛 분해·재조립·reconcile
│   │   │   ├── masking.py      # 플레이스홀더 마스킹/복원 + 출력 검증(looks_untranslated)
│   │   │   ├── glossary.py     # 문서 용어집, prompts.py # 프롬프트 SSOT
│   │   │   ├── flight.py       # SingleFlight — 같은 cache key 요청 합치기
│   │   │   ├── types.py        # TranslateConfig · cache_key · PROMPT_V · reasoning 전달 방식
│   │   │   └── data/seed_ko.json
│   │   ├── llm/                # Q&A 공급자 계층 — providers.py + validate.py (§17)
│   │   └── vendor/
│   │       ├── unlimited_ocr/      # 벤더링 torch 모델 코드 (Baidu MIT) + PROVENANCE.md (P1–P23)
│   │       └── unlimited_ocr_mlx/  # MLX 포팅 (mlx-vlm 0.7.4 기반, MIT) + PROVENANCE.md (M1–M7)
│   ├── tools/translate_eval.py # 번역 품질 평가 CLI
│   └── tests/
├── services/                   # GPU sidecar 컨테이너 (비루트 uid 1000)
│   ├── ovisocr2/               # app/(main·model·parser·config·lifecycle) + Dockerfile
│   │                           #  + requirements.in/.lock(해시 고정 웹 계층 덧씌움) + tests
│   └── paddleocr_vl/           # app/(main·model·adapter·config·lifecycle) + Dockerfile
│                               #  + requirements.in/.lock(전이 의존성 해시 잠금) + tests
├── native/                     # C++ pybind11 모듈 (uocr_native)
│   ├── pyproject.toml          # scikit-build-core
│   ├── CMakeLists.txt
│   ├── src/uocr_native.cpp
│   └── tests/test_parity.py
├── frontend/                   # 정적 SPA (빌드스텝/외부 의존성 0)
│   ├── index.html · styles.css · layout-fit.js · theme-init.js · katex-guard.js  # theme-init = CSP 아래 테마 부트스트랩, katex-guard = 내려받는 HTML의 KaTeX 크기 가드
│   ├── app.js                  # **진입점만**(372줄) — 부트스트랩 + 모듈 배선. 로직 없음
│   ├── js/                     # ES module 17개(약 8,080줄) — 실제 로직은 전부 여기 (§10)
│   │   ├── core.js · state.js · api.js · sse.js · ui.js · constants.js
│   │   ├── upload.js · jobs.js · live.js · results.js · tabs.js · health.js
│   │   └── translate.js · qa.js · viewer.js · reader.js · notes.js
│   ├── vendor/katex/           # 로컬 번들 KaTeX 0.18.10 (외부 CDN 금지 — §10)
│   └── tests/                  # node --test 단위(+helpers/fake-dom) + tests/e2e/(ui, mock-full-flow)
└── scripts/
    ├── make_sample_pdf.py      # 텍스트+표+차트이미지 포함 샘플 PDF 생성
    ├── smoke_e2e.sh            # compose 기동 → 업로드 → 결과 검증
    ├── smoke_image.sh          # 이미지 하드닝 스모크 (ci docker-image · release 공용)
    ├── verify_e2e.py           # 실 PDF 전 구간 검증 하네스 (= make verify-e2e, §11)
    ├── mock_llm.py             # OpenAI 호환 목 서버 (SSE 스트리밍 · 결함 주입 모드 포함)
    ├── dependency_audit.sh     # = make audit — CI dependency-audit 잡과 같은 pip-audit
    ├── benchmark_ocr_engines.py · _smoke_common.py · check_cuda_environment.py
    └── smoke_ovisocr2_5070ti.py · smoke_paddleocr_vl_5070ti.py
```

## 4. 처리 파이프라인

```
업로드(PDF) ──► 업로드 검증(probe 워커: 페이지 수·크기·작업량 게이트, §18) ──► JobStore(queued)
            ──► 워커(단일 스레드)
  1. render : 페이지마다 ocr 워커 프로세스에서 PNG (RENDER_DPI, 기본 200) → pages/page_%04d.png
              (페이지당 PDF_PAGE_TIMEOUT_S — 넘긴 페이지는 흰 페이지 + 경고, 한 잡에서 3번이면 중단)
  2. ocr    : PAGES_PER_CHUNK(기본 8)개씩 infer_multi() 호출 (torch·MLX 공통 계약)
              - 각 청크는 work/chunk_%02d/ 를 output_path로 사용
              - 커스텀 streamer가 토큰 델타를 SSE 큐로 전달
              - StoppingCriteria로 취소(cancel), rolling 반복, 페이지별 문자·토큰 상한 지원
              - 실패·출력 상한(MAX_LENGTH)은 페이지 단위로 복구한다 — 아래 §청크 실패 복구
              - **충실도 게이트**: 청크를 병합한 직후 페이지마다 원본 PDF 텍스트 레이어와
                대조하고, 임계값(`OCR_FIDELITY_THRESHOLD`, 기본 0.70) 미만이면 **그 페이지만**
                `infer()`로 다시 돌린다. 아래 §충실도 게이트 참조.
  3. merge  : 청크 산출물 병합
              - <PAGE> 마커 분리 → 페이지 단위 마크다운
              - 모델 페이지 → 물리 페이지 슬롯 정렬(assign_slots — 놓치거나 쪼갠 페이지 보정).
                정렬은 마커 수가 청크 페이지 수와 다르거나, raw_pages.json이 있는데 그 길이가 다를
                때만 돈다(원출력을 남기지 않는 엔진의 빈 목록은 불일치로 세지 않는다). 대조 결과가
                위치 그대로면 경고를 남기지 않는다
              - work/chunk_*/images/page_{i}_{k}.jpg → images/p{글로벌페이지:04d}_{k}.jpg 리네임
                (정렬로 옮겨진 페이지는 p{슬롯}_x{k}_{j}.jpg — 자기 슬롯의 raster에서 다시 크롭)
              - 마크다운 내 ![](images/page_{i}_{k}.jpg) 참조를 새 경로로 재작성
              - result_with_boxes_{i}.jpg → layout/page_{글로벌:04d}.jpg
              - 페이지 사이 PAGE_SEPARATOR(기본 "\n\n---\n\n")로 join → result.md
  4. done   : meta.json 갱신(warnings·notices·finished_at), SSE done 이벤트
```

#### 청크 실패 복구와 출력 상한 (`pipeline/runner.py`)

청크는 서로 격리된다 — 한 청크가 죽어도(OOM·벤더 예외·출력 상한) 잡 전체를 죽이지 않는다.

1. **1회 재시도**: 실패한 엔진 호출은 캐시를 비운 뒤 한 번 더 부른다. 예외는 요약 문자열만
   남기고 traceback 프레임을 놓는다(`_detach`) — 예외를 쥔 채로는 실패한 시도의 KV·활성화
   텐서가 재시도 내내 남아 같은 OOM이 재발했다. 반복 감지(`RepetitiveOutputError`)와
   `retry_same_page=False`인 예외(sidecar 읽기 타임아웃·잘린 페이지 — 같은 입력을 곧장 다시
   보내 봐야 소용없다)는 재시도하지 않는다. 페이지 단위 엔진(sidecar)의 여러 쪽 청크도
   통째로 다시 보내지 않는다(정상 페이지까지 GPU에서 다시 추론하게 된다). 이어지는 페이지별
   복구에서도 sidecar 엔진은 그 청크에서 이미 끝난 형제 페이지의 결과를 다시 쓰고, 읽기
   타임아웃이 난 페이지는 다시 보내지 않고 텍스트 레이어로 넘긴다.
2. **복구 전 해제**: 실패한 청크의 예외 프레임을 놓고 캐시를 비운 뒤에 페이지별 복구를 시작한다
   — 8쪽 시도의 텐서를 쥔 채 단독 재처리를 돌리면 '1쪽씩이면 통과'할 OOM이 재발했다.
3. **여러 쪽 청크 → 페이지별 복구**: 페이지마다 단독 실행(1회 재시도) → PDF 텍스트 레이어 →
   실패 플레이스홀더. 반복·출력 상한뿐 아니라 일반 예외(8쪽 prefill OOM 등)도 이 길로 온다 —
   예전에는 재시도까지 실패하면 청크 전 페이지가 플레이스홀더였다.
4. **1쪽 청크**(per_page 모드, sidecar 기본 구성)는 같은 페이지의 다른 OCR 호출 대신 텍스트
   레이어부터 쓴다.
5. **출력 상한(MAX_LENGTH)**: 생성이 총 길이 상한에서 EOS 없이 끝나면 엔진(torch·MLX 모두)이
   `OutputLimitError(message, partial_output)`를 낸다. `partial_output`은 run_multi 형식의 잘린
   출력이고(크롭·raw_pages.json은 out_dir에 있다), runner는 **끝까지 생성된 앞 페이지**만
   병합하고 잘린 페이지부터 다시 처리한다. 지킬 앞 페이지는 세그먼트 수로 세지 않는다 —
   `IncrementalMerger.completed_prefix`가 잘린 마지막 세그먼트까지 모든 세그먼트를 원본 텍스트
   레이어와 대조해(align_model_pages + assign_slots), 잘린 세그먼트의 물리 슬롯(또는 세그먼트가
   없는 첫 슬롯)까지를 지키고, 그 배치(`ChunkResult.alignment`, SlotAlignment)로 병합한다 —
   잘리기 전에 모델이 쪼갠 페이지는 한 슬롯으로 합쳐진다. 근거가 약하고 위치 해석과 어긋나면
   아무것도 지키지 않는다. 텍스트 레이어가 없는 스캔과 페이지 단위 엔진은 위치 그대로다.
   페이지가 옮겨졌으면 라이브 스트림을 청크 시작으로 되돌리고 지킨 페이지를 병합 결과로 다시
   내보낸다(`merge.keep_leading_pages`는 지킨 세그먼트 수로 자른다). 실측(M4 Max, MLX): 4쪽
   청크를 `MAX_LENGTH=4000`으로 자르면 앞 3쪽 유지 + 단독 1회, 15.3초. 잡 참고 문구는
   '<상한 이름> 도달로 출력이 잘려 페이지별 재처리(끝까지 생성된 앞 N쪽은 유지)'이고(N은 대조로
   확인한 쪽 수), 상한 이름은 엔진이 `OutputLimitError.limit_label`로
   알린다(in-process 엔진 `MAX_LENGTH`, sidecar 'sidecar 출력 토큰 상한'). torch에서
   `OCR_FAST_DECODE=0`이어도 HF generate 래퍼를 주입해 잘림을 판정한다.
6. **취소된 청크**: 엔진이 취소 시점까지의 부분 출력을 돌려주면 그만큼만 조용히 병합하고
   경고 1건을 남긴 뒤 끝낸다 — 마커 보정·충실도 게이트를 돌리면 진행률이 청크 끝으로 뛰고
   엉뚱한 품질 경고가 남았다. 페이지별 복구 중에 취소되면 그 페이지도 생성된 부분까지만
   병합하고 `N페이지: 취소로 중단된 페이지 — 생성된 부분까지만 병합했습니다`를 남긴다.
   충실도 재처리 중의 취소는 판정·채택·메모 없이 그 페이지를 병합본으로 되돌리고 끝낸다
   (원본 대조 판정도 취소를 보고 멈춘다).

**길이 예산 안내**: multi 청크의 최악 길이는 `쪽 수 × (MAX_PAGE_OUTPUT_TOKENS + 273) + 5`다
(1024px 전역 뷰의 이미지 토큰 273/쪽 + 프롬프트 텍스트 5토큰 — 실토크나이저로 1·2·4·8쪽 =
278·551·1097·2189 실측). 기본값은 `8 × (6,144 + 273) + 5 = 51,341 > MAX_LENGTH 32,768`이라
늘 넘는다. 잘린 청크는 끝까지 생성된 앞 페이지(텍스트 레이어가 있으면 원본 본문과 대조해
확인한 페이지)만 지키고 잘린 페이지부터 페이지별로 다시 처리한다(위 5번 — 시간이 더 든다).
대조할 텍스트 레이어가 없는 스캔은 모델의 페이지 마커를 그대로 믿는다. 그래서 잡마다 내던
WARNING을 없애고 **기동 시 INFO 한 번**(`runner.chunk_length_budget_note`)으로 바꿨다 — 재처리를 줄이려면
`MAX_LENGTH`를 늘리거나 `PAGES_PER_CHUNK`를 줄인다. 토큰을 스트리밍하는 unlimited 엔진의 multi
청크에만 해당한다. 페이지별 상한은 `MAX_PAGE_OUTPUT_CHARS`(내용 문자만 — `<|ref|>`/`<|det|>`
레이아웃 태그와 HTML 표 태그는 세지 않는다)와 하드 상한 `MAX_PAGE_OUTPUT_TOKENS`다.

**엔진 계약** (`engine/base.py`): `OutputLimitError(message, partial_output=None)` +
`limit_label`, `EngineError.retry_same_page`(기본 True), `OCREngine.deterministic_rerun`(기본
False — textlayer는 True라 충실도 재처리를 하지 않는다), `drain_warnings()`(품질 저하)·
`drain_notices()`(정보성 메모) 훅.

#### 잡 메시지 — 경고(warnings)와 참고(notices)

meta.json과 `GET /api/jobs`·`/api/jobs/{id}`의 메시지는 두 목록으로 나뉜다:

- **`warnings` = 실제 품질 저하**: 실패 플레이스홀더, 텍스트 레이어 복구, 재처리 뒤에도 기준
  미달·재처리 실패, 재처리 예산 소진으로 건너뛴 페이지, 게이트 실패, 취소 청크의 부분 병합,
  마커 보정, 렌더 폴백(흰 페이지), 페이지 경계 불일치, layout 파싱 실패, 엔진 `drain_warnings`.
- **`notices` = 정보성**: 페이지 단위 엔진 안내, 청크의 페이지별 복구 경위(MAX_LENGTH·반복·
  청크 실패 — 페이지가 살아나면 손실 없음), 충실도 재처리 '채택'·'측정 한계' 메모, 게이트 회로
  차단, 시간 상한으로 건너뛴 충실도 검사, 엔진 `drain_notices`(페이지 범위를 붙인다).
- viewer-manifest의 `quality.state`는 warnings만으로 정한다(`degraded`/`ok`) — 예전에는 복구에
  성공한 잡과 sidecar 엔진의 모든 잡이 'degraded'였다. `quality.notice_count`가 따로 있다.
- notices 이전의 옛 meta는 `jobs.LEGACY_NOTICE_MARKERS` 문구로 메모리에서만 갈라 읽고 다시
  쓰지 않는다. runner의 해당 문구를 바꾸면 이 목록도 함께 고친다(`tests/test_job_notices.py`가
  실제 runner 출력과 대조한다).
- 잡 시각: `started_at`·`finished_at`(UTC ISO, 초 단위). `null`은 아직 아님·알 수 없음이다 —
  대기 중 취소된 잡은 `started_at`이, 서버 재시작으로 중단된 잡은 `finished_at`이 없고(멈춘
  시각을 모른다), 옛 meta에는 둘 다 없다.

#### result.md 페이지 인덱스 계약 (코드로 강제)

`result.md`는 **N페이지 = 구분자 N-1개**를 만족해야 한다 — `qa.get_page_context`,
`render_document_html`의 doc-page 분할, 번역 조립이 전부 이 `split(PAGE_SEPARATOR)`에
의존하므로, 어긋나면 리더/레이아웃/Q&A가 서로 다른 페이지를 가리키면서도 조용히
"확신에 찬 오답"을 낸다. 두 단계로 보장한다 (`pipeline/merge.py`):

- **무해화**: 페이지 본문에 구분자 코어와 **리터럴로 같은 줄**이 나오면 렌더 결과는
  같고 리터럴 일치만 깨지는 형태로 바꾼다. 기본 구분자에서는 페이지 본문의
  `---` 한 줄이 **`***`로 저장된다**(둘 다 마크다운 수평선이라 화면은 동일).
  구분자가 요구하는 빈 줄 패딩까지 갖춘 줄만 대상이라 setext 제목의 밑줄
  (`제목` 다음 줄의 `---`)은 건드리지 않는다. 구분선이 아닌 코어는 후행 공백을 붙인다.
- **검증**: `finalize()`가 실제 분할 수와 페이지 수를 비교해 어긋나면
  `result.md 페이지 경계 불일치: …` 경고를 meta.json `warnings`에 남긴다.

### 충실도 게이트 — 청크가 놓친 페이지를 골라 다시 읽는다

멀티페이지 추론은 처리량을 위해 8쪽을 한 컨텍스트에 넣는다. 그런데 모델
(`baidu/Unlimited-OCR`)의 언어부는 **`sliding_window=128`의 12층 MoE**라 그 범위에
걸친 구조 기록을 유지할 구조적 수단이 없다. 46쪽 논문 실측:

| 페이지 | 프로덕션(8쪽 청크) | 단독 `infer()` | 단독(gundam 타일) |
|---|---|---|---|
| p34 | **0자** | 0.945 | 0.955 |
| p39 | **0자** | 0.972 | 0.992 |
| p38 | 0.409 | 0.971 | 0.988 |

**모델이 그 페이지를 못 읽는 게 아니라 청크 안에서 놓친다.** 해상도 문제도 아니다 —
타일 모드가 벌어 주는 건 0.01~0.02뿐이고, 청크에서 빼내는 것이 0.5~0.97을 번다.

born-digital PDF에서는 PyMuPDF가 뽑는 텍스트가 **공짜 정답**이다. `pipeline/fidelity.py`가
페이지별로 대조한다:

* **지표 = 정답 잔존율 × 과생성 페널티**, 문자 bigram 다중집합 기준.
  `containment(정답, 후보) × min(1, 1.3·len(정답)/len(후보))`
  · 전역 순서에 둔감해야 한다 — 모델은 표를 `<table>` HTML로 내고 셀 순서가 PDF 읽기
    순서와 다르다. 순서 민감 지표(difflib·LCS)는 정상 표를 오탐한다(실측 p18: 0.292).
  · **대칭 지표(Dice)를 쓰지 않는다** — 게이트가 잡아야 하는 건 *손실*인데 Dice는
    "빠졌다"와 "더 썼다"를 한 숫자에 섞어 둘 다 둔감해진다(실측: 표 33% 유실에서
    Dice 0.702로 임계 통과, containment 0.587로 탐지).
  · **단어가 아니라 bigram**을 센다 — 단어 토큰은 띄어쓰기 없는 중국어·일본어에서 무너진다.
  · 과생성 페널티는 중복 전사를 잡는다(실측 p37: 인쇄 36·37쪽을 한 페이지에 전사,
    containment 0.974 → 페널티 적용 0.494).
* **HTML 태그는 이름 화이트리스트로만 벗긴다.** 범용 `<[^>]*>`는 반대로 **정답을
  파괴한다** — 코드의 `#include <AK/Debug.h>`·`Vector<Component>`가 태그로 오인되고,
  줄을 이어 붙인 뒤 적용하면 한 줄의 `<`가 수십 줄 뒤의 `>`까지 삼킨다
  (실측 p33: 정답 2,532자 → 866자, 46쪽 합계 2,877자 증발).
* **정답에서 그림 영역을 뺀다.** `image`뿐 아니라 **`chart`**도 포함해야 한다 — 실측
  이 논문은 image 12개 + chart 7개이고, chart로만 이루어진 p25는 전사가 완벽한데도
  0.729로 떨어졌다(임계 0.70과 간격 0.029).
* **정답이 200자 미만이면 판정하지 않는다.** 표지·백지·스캔 PDF는 게이트를 건너뛴다
  (판정 불가는 실패가 아니다).
* **글자와 숫자만 비교한다.** 정답·후보 모두 NFKC → HTML 태그(화이트리스트) 제거 → LaTeX
  명령 제거(그리스 문자는 글리프로) → 글자·숫자만 남기고 casefold한다. 정답과 후보에서
  똑같은 블록을 뺀다(불신하는 수식·그림 블록은 양쪽 모두에서). 목차 점선·글리프 대 LaTeX
  표기의 오탐이 사라졌다(실측 Metal 출력: 0.465 → 1.000, 0.656 → 0.988 — 정상 페이지 최저가
  0.656 → 0.978로 올랐고, 통째 유실(0.0)·중복 전사(0.65)는 그대로 잡힌다). 기호·구두점만의
  부분 손실은 의도적으로 보지 않는다.
* **믿을 수 없는 텍스트 레이어는 판정하지 않는다** — PUA·U+FFFD·제어 문자가 20% 이상이거나
  글자·숫자가 30% 미만인 페이지(깨진 폰트 매핑)는 '텍스트 레이어 신뢰 불가'로 판정 불가다.
* 페이지 그래픽(이미지·벡터 경로)은 페이지당 한 번만 추출한다. 분석은 ocr 워커 프로세스에서
  페이지 단위 작업으로 돌고(§18), 시간 상한을 넘은 페이지는 그 검사만 건너뛰고 참고를 남긴다.
  취소는 페이지마다·벡터 경로 4,096개마다 확인한다.

실측 분리 마진(46쪽, 열화 4쪽이 확인된 실행): 열화 최고 **0.494** vs 비열화 최저
**0.813** → 임계값 0.70이 유효 구간의 가운데다.

복구 절차(`runner._repair_low_fidelity_pages`):

1. 청크를 정상 경로로 병합한 **뒤** 그 청크의 페이지들을 판정한다. 멀티 청크를 쪼개
   재사용할 수는 없다 — `_move_chunk_files`가 파일을 꺼내 가고, multi의 local→global
   매핑이 `chunk.start_page`에 묶여 있다.
2. 열화 페이지가 있으면 **첫 열화 페이지**로 라이브 스트림을 되감고, 그 뒤 정상
   페이지는 병합된 마크다운으로 다시 내보낸다(`sink.emit_page`).
3. 열화 페이지는 `run_single`로 다시 돌리고 **같은 잣대로 재판정**한다.
   `_FIDELITY_ACCEPT_MARGIN`(0.05) 이상 개선됐을 때만 `merger.replace_page()`로 채택한다.
   개선되지 않거나 재처리가 터지면 원래 결과를 지킨다 — **게이트는 손해를 끼치면 안 된다.**
4. 재처리 상한은 문서 페이지 수의 `OCR_FIDELITY_MAX_RETRY_RATIO`(기본 0.2, 최소 2쪽).
   상한에 걸려 건너뛴 페이지는 경고로 남긴다(조용한 절단 금지). 예산은 **통째로 유실된
   페이지**(OCR 텍스트가 정답의 10% 미만)에 먼저, 그다음 점수가 낮은 순으로 쓴다.
5. **회로 차단**: 재처리 3번이 하나도 채택되지 않으면 그 문서에서는 부분 열화 재처리를
   멈춘다(통째 유실 페이지는 계속 재처리). 점수 변화가 0.01 이하면 '측정 한계로 판단'으로
   적는다 — 품질 문제가 아니므로 채택·측정 한계 메모는 참고(notices)이고, 개선되지 않은
   기준 미달·재처리 실패만 경고다.

게이트가 **아예 돌지 않는** 조건 — 원리상 도움이 될 수 없거나 해가 되는 경우다:

* `job.mode == "per_page"`, 임계값 ≤ 0, 원본 PDF 부재
* 결정적 엔진(`OCREngine.deterministic_rerun` — textlayer): 같은 페이지를 다시 돌려도
  결과가 같아 시간만 쓰고 거짓 '충실도 미달' 경고를 남긴다
* `layout_capability != "full"` — 텍스트 bbox를 안 주는 엔진(sidecar의 `figure_only`)은
  블록 내용이 비어 **전 페이지가 0.00**으로 나온다. 전량 오탐이다.
* `not supports_multi_page` 또는 `preferred_chunk_size == 1` — 이미 페이지 단위로 도는
  엔진은 재처리가 **같은 호출**이라 개선 여지가 없고 중복 추론만 늘린다.
* 페이지의 정답 텍스트가 200자 미만(스캔 PDF·표지·백지)

그리고 게이트 자체의 실패는 잡을 죽이지 않는다 — 복구는 **선택적 개선**이므로 IO
오류가 나면 경고만 남기고 원래 결과로 계속한다. 취소만 그대로 전파한다.

두 가지 방어가 더 있다:

* **모델의 그림 분류를 원본으로 검증한다.** 청크에서 페이지를 놓친 모델이 빈 출력 대신
  전면 `image` 블록 하나를 내면(판독 실패 영역을 그림으로 부르는 것은 흔한 동작)
  정답이 통째로 가려져 `truth_chars=0` → 판정 불가 → **게이트가 존재 이유인 바로 그
  실패에서 침묵한다.** 그래서 원본 PDF가 그 영역에 실제로 그린 면적이 10% 미만이면
  그 분류를 믿지 않는다(실측: 이 논문의 그림·차트 19개는 전부 통과).
* **재발행 페이지의 grounding을 살린다.** 되감은 뒤 정상 페이지를 병합된 마크다운으로
  다시 흘리면 `<|det|>` 태그가 없어 그 페이지들의 라이브 레이아웃 박스가 사라진다.
  `layout.blocks_to_raw()`로 grounding을 복원해 내보낸다.

이 경로는 `BrokerSink`가 **모든 페이지**에 되감기 마크를 남긴다는 데 의존한다. 예전에는
`set_chunk`만 마크를 남겨 청크의 첫 페이지에만 지점이 있었고, 청크 한가운데 페이지의
`rewind_to`는 아무 이벤트도 내지 않는 무음 no-op이었다 — 그 상태에서 `emit_page`를
부르면 세그먼트가 하나 늘어 "k번째 `<PAGE>` == 페이지 k" 불변식이 깨진다.

`layout.json`도 같은 이유로 청크의 **모든** 페이지를 채운다 — raw_pages.json이 없거나
페이지 수가 모자라면 빈 블록 페이지로 메워 result.md의 페이지와 1:1을 유지한다.
텍스트 블록을 한 번도 받지 못한 잡은 아예 `layout.json`을 만들지 않는다 — figure_only
엔진(OvisOCR2)은 `raw_pages.json`에 좌표를 싣지 않고(materializer `write_raw` — 페이지마다 빈
원출력만 남겨 merge가 원출력 개수로 페이지 수를 맞춰 본다), 전면 스캔을 처리한
textlayer 잡도 layout이 생기지 않는다. 판정은 파일 존재가 아니라
`artifacts.has_usable_layout(job_dir, lang)` — **image 아닌 블록이 하나 이상**인 layout만
좌표 기능에 쓴다(파일 mtime·크기로 캐시, 1024개). 이전 버전이 만든 image 블록뿐인 layout(옛
OvisOCR2 잡)도 같은 규칙으로 '레이아웃 없음'이다: `has_layout=false`, 좌표 라우트 404,
`/pdf` 409, `document.html`은 의미 기반 HTML, PDF 예열 생략, 번역은 그 layout을 무시하고
`layout.{lang}.json`을 쓰지 않는다(§5).

**병합 정렬** (`merge.py`): 모델이 청크 안에서 페이지를 건너뛰거나 하나를 둘로 쪼개면 k번째
`<PAGE>`가 k번째 물리 페이지가 아니다. 페이지마다 원본 텍스트 레이어와 글자·숫자 정규화로
대조해 `assign_slots`가 단조 증가로 슬롯을 채우고(남은 페이지는 앞뒤 기준점의 탐침 점수로
붙인다), 일치가 50% 미만이면 위치 순서대로 둔다. 옮겨진 페이지의 그림은 자기 슬롯의 페이지
이미지에서 다시 크롭하고(`p{start+slot}_x{k}_{j}.jpg`), 다른 페이지 이미지로 잘린 벤더 크롭과
오버레이는 버린다 — `replace_page`가 이웃 페이지의 그림을 지우거나 덮지 않는다. 한 페이지의
그라운딩을 파싱하지 못하면 그 페이지의 layout 블록만 비우고 경고를 남긴 채 마크다운은
지킨다(다른 페이지는 영향 없음).

- `mode=per_page`일 때는 2단계가 페이지당 `infer()`(gundam) 호출로 대체된다
  (`ngram_window=128`). 이미지 프리픽스는 페이지 디렉터리로 격리 후 동일하게 병합.
- 워커는 프로세스당 1개(모델 메모리 때문). 잡은 FIFO.
- **워커 복원력(잡 단위 예외 방벽)**: 워커 루프의 잡 처리 전체가 `try/finally`로 감싸여
  있어 `execute_job`이나 마감 경로(`store.save`의 OSError 등)에서 예외가 새어 나와도
  **워커 스레드가 죽지 않는다**. 죽으면 이후 제출되는 모든 잡이 영구 `queued`로 남고
  프로세스 재시작 외에 복구 수단이 없다. `cancel_events` 정리는 이 `finally`로
  일원화한다(모든 종료 경로 공통). 워커 생존 여부는 `/api/health`의 `worker_alive`로 노출된다.
- **메타 기록은 best-effort**: `JobStore.save()`는 `OSError`(ENOSPC·권한 등)를 잡아
  경고 로그만 남긴다 — 디스크 문제가 오류 마감 경로나 워커 스레드를 죽이지 않게.
  `FileNotFoundError`는 삭제 경합이라 로그도 남기지 않는다.
- **재시작 시 잔여물 정리**: `load_existing`이 중단된 잡을 복원해 상태를 바꿀 때
  `work/`를 함께 지운다 — runner의 `finally`가 돌지 못하고 죽은 잡은 다시 실행되지
  않아 어느 경로에서도 정리되지 않았다. `meta.json` 없는 `j_<12hex>` 디렉터리(업로드
  도중 죽은 잔해)는 기동 때 지운다.
- **대기 잡은 재시작을 넘긴다**: 제출된(`submitted`) 대기 잡 중 `source.pdf`가 `%PDF-`로
  시작하는 것은 재시작 뒤 원래 큐 순서대로 다시 제출한다 — 제출 때 `meta.json`에 남긴
  `submit_order`(ns 단위, 스토어 안에서 단조 증가)로 정렬하고, 그 값이 없는 구버전 meta는
  `created_at`을 쓴다. 예전에는 초 단위 `created_at` + 무작위 잡 ID로 정렬해 같은 초에 올린
  잡의 순서가 재시작 때 뒤바뀌었다. `Worker.stop()` 뒤에는 워커가 대기 잡을 새로 맡지 않으므로
  종료 도중 시작돼 중단으로 마감되는 잡도 없다. 실행 중이던 잡은 크래시 루프를 피하려고 종전대로
  error로 마감한다(멈춘 시각을 몰라 `finished_at`은 비운다).
- **페이지 구분자는 잡에 고정**: `meta.json`의 `page_separator`가 그 잡의 `result.md`를
  조립한 값이다. `/html`·document.html 폴백·Q&A·번역이 모두 이 값으로 페이지를 나누므로
  `PAGE_SEPARATOR`를 나중에 바꿔도 옛 잡의 페이지 경계가 깨지지 않는다(옛 잡에는 기동 때
  현재 값을 한 번 고정하고, meta mtime은 보존해 TTL 시계는 그대로다).
- **워커 진행 관측**: 워커는 실행 중 잡과 그 잡의 마지막 진행 시각(시작·끝, 그 잡 채널의
  진행·토큰·대기 알림)을 기록한다 → `/api/health`의 `worker_job_id`·
  `worker_last_progress_at`·`worker_progress_age_s`. 잡이 있는데 경과 초가 계속 늘면 멈춘 것이다.
  모델 로드 실패는 프리로드든 워커의 잡 시작 로드든 `model_load_error`에 남고, 재시도가
  성공하면 지워진다.
- **Metal 메모리**: 실제 디바이스가 `metal`(torch MPS)이면 워커가 잡마다 ObjC 오토릴리스
  풀로 감싼다(디코드 스텝·생성 구간·재시도 전 캐시 반환도 각각 풀 안). 끝나지 않는 워커
  스레드에는 런루프 풀이 없어, 풀이 없으면 autorelease된 MPS 임시 객체가 프로세스 수명 내내
  쌓였다(실측: 384토큰 실행당 RSS +10.6 → 약 1.5 MB). MLX는 감싸지 않는다 — 실측(M4 Max,
  8쪽 실가중치 잡 5회 연속): RSS 2,627 → 2,647 MB, MLX 활성 메모리 6,363 MB 고정으로 잡별
  누적이 없었다.
- **잡 저장소 단일 소유자 락** (`owner_lock.py`): 위 정리는 "이 프로세스가 유일한
  소유자"일 때만 안전하다. 같은 `{DATA_DIR}/jobs`를 쓰는 두 번째 백엔드의
  `load_existing`은 먼저 뜬 쪽에서 **실행 중인** 잡을 error로 덮고 `work/`를 지웠다
  (`make dev` 중의 `make test`, 같은 `ocr-data`를 쓰는 compose backend 동시 기동,
  `--workers N`·`WEB_CONCURRENCY`). 그래서 `create_app()`이 `load_existing` **전에**
  `{DATA_DIR}/jobs/.owner.lock`에 배타 `flock`을 비차단으로 잡고 앱 수명 동안 쥔다.
  살아 있는 다른 소유자(다른 프로세스, 또는 같은 프로세스의 아직 살아 있는 앱)가
  있으면 디렉터리를 짚는 `JobsDirInUseError`로 기동을 거부한다. lifespan 종료·앱 조립
  실패 시 즉시 놓고, 버려진 앱은 수거될 때 놓는다(충돌 시 `gc.collect()` 후 1회 재시도).
  프로세스가 죽으면 커널이 회수하므로 stale 락 정리는 없다. 락 파일은 지우지 않는다
  (쥔 채 지우면 다음 소유자가 새 inode를 잠가 공존한다). `fcntl`이 없거나 flock을
  지원하지 않는 파일시스템에서는 경고 후 보호 없이 진행한다.
  **같은 DATA_DIR에는 백엔드 프로세스가 하나만** 뜬다(§8).
- **`app` 지연 생성**: `app.main`의 `app`은 PEP 562 지연 속성이다 — `uvicorn
  app.main:app`·`from app.main import app`이 처음 찾을 때만 기본 앱을 만든다.
  모듈 import만으로는 `.env`를 읽거나 `DATA_DIR`을 건드리지 않는다(예전에는 pytest
  수집만으로 개발 서버의 실행 중 잡이 error로 덮였다).
- **sidecar 재시작/모델 재로드 대기**: sidecar 엔진은 페이지 요청이 `SidecarUnavailableError`
  (HTTP 503·연결 끊김)로 실패하면 health 캐시를 무효화하고 `wait_until_ready()`로
  컨테이너 복귀를 기다린 뒤 **그 페이지만 1회 재시도**한다. 재요청도 503·연결 끊김이면
  `SidecarRestartLoopError`(`retry_same_page=False`)로 runner 재시도 없이 페이지 격리로 넘기고,
  페이지별 복구가 그 페이지를 한 번 더 보낼 때는 또 내려가도 기다려 다시 보내지 않는다. 복귀
  경위는 잡 참고(notices) 한 줄로만 남는다(엔진 `drain_notices`). 기다리지 않으면 재기동 +
  모델 로드 시간 동안의 페이지가 전부 플레이스홀더로 확정된다. 대기 예산
  (`OCR_SIDECAR_MODEL_WAIT_S`)은 장애 한 번에 하나다 — 그 장애의 모든 페이지가 같은 데드라인을
  공유한다. 대기 문구는 health의 `load_retry`·`restarting`으로 첫 로드·로드 재시도·재시작을
  구분한다(OCR_ENGINE_PROTOCOL.md). 반면 `SidecarTimeoutError`는 provider가 아직 그 페이지를
  추론 중일 수 있어 다시 보내지 않는다(`retry_same_page=False` — 같은 페이지를 GPU에서 두 번
  돌리게 된다). runner도 재시도하지 않고 곧바로 페이지 격리(텍스트 레이어 → 플레이스홀더)로
  넘긴다. 출력 토큰 상한에서 잘린 페이지(`page.truncated`)의 처리도 프로토콜 문서에 있다.
  `loaded`는 **`model_loaded` 축만** 본다. sidecar의 `status` 축은 별개 신호라
  `_check_ready`가 나눠 처리한다: `model_loaded=False`+`status!=ok`는 진짜 로드 실패라
  하드 실패(`EngineError`), `model_loaded=True`+`status!=ok`는 임계 기반 **자가 복구형
  웨지 신고**(오탐 가능)이므로 잡 경고로만 올리고 통과시킨다. 후자를 하드 실패로 만들면
  오탐 1건이 모든 잡을 "모델 로드 실패"로 즉시 마감하고, sidecar는 요청을 못 받아 스스로
  복구할 수도 없다(HEALTHCHECK는 200이라 재시작도 안 걸린다). 진짜 이상이면 parse가
  502로 답해 기존 청크 격리가 받고, 오탐이면 성공 1회로 자동 복구된다.
- **실패 격리**: 렌더에서 한 페이지가 깨지거나 페이지 시간 상한(`PDF_PAGE_TIMEOUT_S`)을 넘으면
  흰색 페이지로 대체하고 경고를 남긴 뒤 계속한다(한 잡에서 렌더 상한 초과가 3번이면 잡을
  끝낸다 — §18). OCR 실패·출력 상한은 위 §청크 실패 복구의 순서(단독 재처리 → 원본 PDF의 내장
  텍스트 레이어 → 실패 플레이스홀더)로 메운다. 텍스트 레이어 복구는 읽기 순서 블록마다
  이스케이프한 문단(코드 블록 아님)이라 렌더 그대로 보이고 번역 대상이 되며, 2단 문서의 두
  단이 섞이지 않는다(§16). 내역은 meta.json `warnings`·`notices`에 남으며 전 청크/전 페이지가
  끝내 복구되지 못한 경우만 error, 취소는 그대로 전파한다. 렌더 중 취소는 그 페이지를 그리던
  워커 프로세스를 바로 끝낸다(예전에는 페이지 사이에서만 확인했다).

### 잡 디렉터리 레이아웃 (`{DATA_DIR}/jobs/{job_id}/`)

```
source.pdf                  # 업로드 원본
meta.json                   # 상태/진행/파라미터·warnings·notices·started_at/finished_at·
                            #   page_separator·submitted (재시작 시 복원)
pages/page_0001.png ...     # 렌더된 입력 페이지 (1-based)
work/chunk_00/ ...          # 모델 원시 출력 (실행 중에만 존재 — 터미널 마감 시 자동 삭제, §15)
result.md                   # 최종 병합 마크다운
layout.json                 # 페이지별 블록(type/bbox/content) — 텍스트 블록이 있을 때만
images/p0001_0.jpg ...      # figure 크롭 (글로벌 페이지 번호, 1-based)
layout/page_0001.jpg ...    # 레이아웃 박스 오버레이
```

번역·내보내기 산출물(`translations/{lang}/`, `result.{lang}.md`, `layout.{lang}.json`,
`export.{lang}.*`, `rendered/{lang}/`, `archive.zip`)은 §5·§13에 있다.

## 5. REST / SSE API 계약 (v1)

모든 경로는 `/api` 프리픽스. 프론트엔드는 같은 오리진에서 서빙되므로 CORS 불필요.

**상태코드 계약 (재시도 여부)** — 클라이언트(프런트 `core.langFetchVerdict` 등)는 이 구분에
기대어 동작한다:

| 응답 | 뜻 | 클라이언트 |
|---|---|---|
| **503 + `Retry-After`** | 일시적 과부하 — 빌드 대기열 초과·예열 대기 초과, SSE 구독 상한, 번역 시작의 자원 부족, 업로드 검증 워커 포화 | 기다렸다 다시 시도 (프런트는 1–60초로 묶어 최대 4번) |
| **503 (Retry-After 없음)** | 설정 문제 — 번역·Q&A 프로바이더 미설정 등 | 재시도 무의미 |
| **404 / 409** | 없음 / 지금 상태로는 불가 — 번역본 없음, 좌표 레이아웃 없음, 미완료 잡, 내보내기 불가(입력 손상·빌드 시간 상한 초과) | 대체 경로(원문 보기·HTML 내보내기) |
| **429 + `Retry-After`** | 레이트리밋·동시 상한 (아래) | 남은 시간 뒤 |

### 남용 방어 — 429 + Retry-After (QA·translate)

이 서비스는 인증이 없고 compose 기본 바인딩이 `0.0.0.0`이다(§8·§14). 같은 네트워크의
누구나 `POST /qa`로 운영자의 유료 LLM 키를 소진하거나 200페이지 번역을 반복 트리거할 수
있으므로, 인증을 새로 만드는 대신 **비용이 드는 두 라우트에만** 프로세스 내
슬라이딩 윈도우(60초) 레이트리밋 + 동시 실행 상한을 둔다 (`api.py::_AbuseGuard`).

| 라우트 | 레이트리밋 키 | 상한 초과 응답 |
|---|---|---|
| `POST /jobs/{id}/qa` | `qa:job:{id}` · `qa:ip:{client}` | 429 + `Retry-After`(남은 윈도우 초, 올림) |
| `POST /jobs/{id}/qa` (동시 실행) | 전역 슬롯 | 429 + `Retry-After: 5` |
| `POST /jobs/{id}/translate` | `translate:job:{id}` · `translate:ip:{client}` | 429 + `Retry-After`(남은 윈도우 초, 올림) |
| `POST /jobs/{id}/translate` (동시 번역 수) | 실행 중 번역 태스크 수 | 429 + `Retry-After: 30` |

- 조정 변수(§7): `QA_RATE_LIMIT_PER_MIN`(30)·`QA_MAX_CONCURRENT`(4)·
  `TRANSLATE_RATE_LIMIT_PER_MIN`(12)·`TRANSLATE_MAX_ACTIVE`(4). **0 이하면 해당 상한 비활성**이며,
  정수가 아닌 값은 500 대신 기본값으로 강등하고 경고 로그만 남긴다.
- 가드는 앱 상태에 지연 생성되고 상한은 프로세스 단위다(외부 저장소·인증 없음).
  기본값은 1인 로컬 사용을 방해하지 않는 수준으로 잡혀 있다.
- 한 요청의 키들(잡·IP)은 **원자적으로** 판정한다(`hit_many`) — 어느 키든 넘치면 아무
  것도 기록하지 않고 가장 긴 `Retry-After`를 돌려준다. 예전에는 IP 한도로 거절된 요청도 잡
  버킷을 소모해, 한 클라이언트가 모든 잡의 번역·Q&A를 잠글 수 있었다.
- 클라이언트 IP: `TRUSTED_PROXY_HOPS>0`이어도 `X-Forwarded-For`는 직접 연결 피어가
  `TRUSTED_PROXY_IPS`(IP·CIDR, 비우면 루프백 `127.0.0.0/8`·`::1`)에 있을 때만 믿는다. IP가
  아닌 피어(유닉스 소켓·프로세스 내 하네스)는 로컬 전송이라 믿는다. 목록 밖 피어는 헤더를
  위조로 보고 피어 IP로 레이트리밋한다(한 번 경고 로그). uvicorn 기본 프록시 헤더 처리
  (`FORWARDED_ALLOW_IPS`, 기본 127.0.0.1)가 이미 client를 XFF 항목으로 바꾼 요청(포트 0)은 그
  서버가 믿은 홉으로 보고 같은 홉 수를 헤더 체인 전체에 적용한다 — `--no-proxy-headers` 여부와
  무관하게 같은 키다. `X-Forwarded-For` 필드 줄이 여럿이면 순서대로 잇는다(RFC 9110).
  레이트리밋 429의 `Retry-After`는 남은 윈도우를 올림한 초다 — 그만큼 기다리면 통과한다.
- 같은 가드가 **`POST /render-preview`**도 묶는다: 크기 가중 비용(16 KiB당 1단위, 잡·IP
  키마다 분당 600단위)과 동시 렌더 4건. 넘으면 429 + `Retry-After`(동시 상한은 1초).

### GET /api/health
```json
{
  "status": "ok",
  "engine": "unlimited",            // unlimited | fake | textlayer | ovisocr2 | paddleocr_vl
  "device": "cuda",                 // cpu | cuda | metal | mlx — auto를 푼 실제 디바이스
  "dtype": "bfloat16",              // MLX 8비트면 "bfloat16+q8"
  "model_id": "baidu/Unlimited-OCR",
  "model_loaded": true,             // false면 첫 잡에서 로딩
  "model_load_error": null,         // model_loaded=false일 때만 마지막 로드 실패 사유
                                    //  (프리로드든 워커의 잡 시작 로드든 — 재시도 성공 시 지움)
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",  // cpu면 null, Apple GPU는 칩 이름("Apple M4 Max")
  "native_ops": true,               // C++ 모듈 사용 여부
  "worker_alive": true,             // OCR 워커 스레드 생존 여부 — false면 잡이 영원히 queued
  "worker_job_id": "j_…",           // 워커가 실행 중인 잡 (없으면 null)
  "worker_last_progress_at": "2026-10-02T03:00:00+00:00",  // 그 잡의 마지막 진행(시작·끝·
                                    //  진행·토큰·대기 알림) UTC 시각
  "worker_progress_age_s": 1.2,     // 그로부터 지난 초 — 잡이 있는데 계속 늘면 멈춘 것
  "pdf_workers": {                  // PyMuPDF 격리 워커 풀 (§18)
    "mode": "process",              // process | inline(테스트·디버그)
    "pools": {"ocr": {"size": 1, "workers": 1, "in_use": 0, "tasks": 81, "timeouts": 0,
                      "crashes": 0, "canceled": 0, "spawned": 1, "recycled": 0,
                      "rejected_busy": 0}}   // export·probe도 같은 모양 (처음 쓴 풀만)
  },
  "max_upload_mb": 100,             // 업로드 상한 (MAX_UPLOAD_MB 그대로)
  "translate_available": true,      // 번역 프로바이더 설정 여부 — false면 POST /translate가 503
  "qa_available": true,             // 기본 LLM 공급자의 **실제 구성 여부** (상수 아님 — §17)
                                    //  openai-*: LLM_OPENAI_API_KEY 유무로 판정
                                    //  ollama  : 동기 조회가 불가해 true, 실시간 가용성은 /api/providers
  "llm_default_provider": "openai-responses",  // 기본 LLM 공급자 (LLM_PROVIDER)
  // ── 멀티 엔진 확장 필드 (추가만 — 기존 필드 의미 불변) ──
  "model_revision": "ee63731b…",    // 엔진이 모르는 경우 null
  "provider": "in-process",         // in-process | local-sidecar
  "capabilities": {
    "multi_page_context": true,     // 페이지 단위 엔진(ovis/paddle)은 false
    "stream_granularity": "token",  // token | page — 프론트 라이브 뷰 안내에 사용
    "layout": "full",               // full | figure_only | none — 레이아웃 탭 안내
    "figures": true
  },
  "provider_health": null,          // sidecar 엔진만: {status, runtime, version,
                                    //  model_loaded, gpu_total_mb, gpu_free_mb,
                                    //  load_retry, restarting} (OCR_ENGINE_PROTOCOL.md)
                                    //  sidecar가 죽어도 health 자체는 200 —
                                    //  {status:"unreachable", error:"…"}로 구분
  "config_warnings": []             // 앱이 직접 읽은 .env(로컬 uv 실행)에서 이 앱이 읽지 않는
                                    //  키의 짧은 안내(키 이름·오타 후보만, 값 없음 — §7).
                                    //  컨테이너에는 .env가 없어 Docker에서는 늘 빈 목록
}
```

### POST /api/jobs — PDF 업로드
- `multipart/form-data`: `file`(필수, PDF), `mode`(`multi`|`per_page`, 기본 `multi`),
  `dpi`(72–400, 기본 200). 페이지 상한은 서버의 `MAX_PAGES`로 일괄 적용한다.
- 202 → `{"job_id": "j_1a2b3c4d5e6f", "status": "queued"}`
- 400(비PDF/손상/복잡도 게이트 초과), 413(MAX_UPLOAD_MB 초과),
  **503 + `Retry-After: 5`**(업로드 검증 워커가 모두 다른 업로드를 검사 중 — 파일 문제가 아니다).
  검증 중이거나 워커를 기다리는 업로드는 동시에 4건까지이고, 넘으면 곧바로 503이다. 빈 워커는
  min(`PDF_PAGE_TIMEOUT_S`, 5초)까지만 기다린다 — 예전에는 넘친 업로드가 공유 스레드를 60초씩
  쥐어 `/api/health`까지 늦어졌다.
- 검증(`probe_pdf`)은 probe 워커 프로세스에서 `PDF_PAGE_TIMEOUT_S` 안에 돈다: 페이지 수
  (`MAX_PAGES`)·한 변 길이와 **업로드 복잡도 게이트**(렌더 없이 페이지별 압축 해제 콘텐츠
  바이트·펼친 XObject 호출 수를 센다 — §18). 상한 초과·검증 시간 초과·워커 비정상 종료는
  사유를 담은 400이다(`PDF 검증이 시간 상한(…)을 넘었습니다` 등). 손상 PDF는 고정 문구
  `PDF를 열 수 없습니다 — 손상되었거나 지원하지 않는 형식입니다`이고 서버 경로를 싣지 않는다
  (원래 MuPDF 오류는 서버 로그에만).
- **본문 상한은 라우트 진입 전(ASGI)에서 끊는다** — `UploadBodyLimitMiddleware`가
  본문을 가질 수 있는 메서드(POST/PUT/PATCH)를 (경로 패턴 → 상한) 표로 판정한다:
  `/api/jobs` = `MAX_UPLOAD_MB` + 64KiB(멀티파트 봉투 여유),
  `/api/jobs/{id}/render-preview` = 256KiB(라우트 내부 상한과 동일 값),
  **그 외 = 64KiB 기본**. 표에 없는 새 POST 라우트도 기본 상한으로 자동 보호된다 —
  예전처럼 `/api/jobs`만 검사하면 `POST /jobs/{id}/qa`가 잡 존재 확인 이전에 무제한
  본문을 메모리에 적재한다(실측 uvicorn: 80MB 본문 1건에 RSS +422MB·동시 4건 +1.4GB,
  422 응답이 80MB 원문 반향 → 상한 적용 후 RSS +0MB·62바이트 413).

### GET /api/jobs?limit=50&before=<job_id> — 잡 목록 (최신순)
```json
{"jobs": [ { …GET /api/jobs/{id}와 동일 키… } ], "has_more": true, "total": 132}
```
- 선택 쿼리: `limit`(1–500, 기본 50), `before`=<잡 ID>(그 잡 다음부터 — '더 보기' 커서).
  기본 응답은 예전과 같은 최신 50건이고 `has_more`(뒤에 더 있는가)·`total`(전체 잡 수)이
  덧붙는다. 범위 밖 `limit`, 없는 커서 잡(삭제됨)은 422 — 프런트는 처음부터 다시 받는다.
- **주의**: 목록은 `include_files=False`로 직렬화한다 — `result`의
  `images`/`layouts`/`pages` 배열은 **항상 빈 배열**이다(키는 유지). 폴링마다
  잡×디렉터리를 전수 스캔하던 비용을 없앤 것으로, 실제 파일 URL이 필요하면
  단건 `GET /api/jobs/{id}`를 쓴다. `has_layout`은 잡마다 layout 사용 가능 판정을 하지만
  파일 버전(mtime·크기)별 캐시라 폴링마다 다시 파싱하지 않는다.

### GET /api/jobs/{id} — 상태
```json
{
  "job_id": "j_1a2b3c4d5e6f",
  "filename": "sample.pdf",
  "status": "running",              // queued|running|done|error|canceled
  "mode": "multi",
  "created_at": "2026-07-06T10:00:00+00:00",
  "queue_position": 2,              // 선택 — status=queued일 때만: 대기열 위치(1-base, 생성 순서)
  "progress": {
    "phase": "ocr",                 // loading|render|ocr|merge (loading=sidecar 모델 준비 대기)
    "current_page": 3,              // 1-based, 처리 중/완료된 페이지
    "total_pages": 12,
    "chunk": 1, "total_chunks": 2
  },
  "error": null,
  "warnings": [],                   // 실제 품질 저하 (§4 잡 메시지)
  "notices": [],                    // 정보성 처리 경위 (옛 meta는 문구로 갈라 읽는다)
  "started_at": "2026-07-06T10:00:03+00:00",   // 실행 시작 (대기 중 취소·옛 잡은 null)
  "finished_at": null,              // 종료 (실행 중·재시작으로 중단·옛 잡은 null)
  // ── 엔진/모델 메타 (추가 필드 — 실행 시작 시 확정, 구버전 잡은 null) ──
  "engine": "unlimited",
  "model_id": "baidu/Unlimited-OCR",
  "model_revision": "ee63731b…",
  "provider": "in-process",
  "result": {                       // status=done일 때만
    "markdown_url": "/api/jobs/{id}/markdown",
    "html_url": "/api/jobs/{id}/html",
    "archive_url": "/api/jobs/{id}/archive",
    "viewer_manifest_url": "/api/jobs/{id}/viewer-manifest",
    "images": ["/api/jobs/{id}/files/images/p0001_0.jpg"],
    "layouts": ["/api/jobs/{id}/files/layout/page_0001.jpg"],
    "pages": ["/api/jobs/{id}/files/pages/page_0001.png"],
    "has_layout": true              // 텍스트 블록이 있는 layout인가(has_usable_layout) —
                                    //  false면 /layout·/alignment·/outline·/viewer/pages 404, /pdf 409
  }
}
```
- `queue_position`은 **워커 큐 제출 순번(submit_seq)** 기준이다. 제출은 업로드 본문
  수신 완료 직후라 생성 순서와 어긋날 수 있어(큰 파일을 먼저 올려도 작은 파일이 먼저
  제출된다) 생성 순서 대신 실제 처리 순서를 반영한다.

### GET /api/jobs/{id}/events — SSE
- `Content-Type: text/event-stream`, `retry: 3000`, 15초마다 `: ping` 주석
- 접속 시 현재 상태 스냅샷(progress) 1회 즉시 발행, 종료 잡이면 done/error 즉시 발행
- **구독자 상한**: 채널(잡·번역)당 8개, 프로세스 전체 64개(`jobs.EventBroker`). 넘으면 스트림을
  열기 전에 **503 + `Retry-After: 5`**(`/translate/events`도 같다). 검사 뒤 경합으로 넘친
  연결은 재시도 안내만 보내고 닫는다. 예전에는 상한이 없어 연결 수에 비례해 OCR 워커의
  publish가 느려지고 큐 메모리가 늘었다.
- **서버 종료**: lifespan이 SIGINT/SIGTERM을 잡아 `ShutdownSignal`을 세우면 두 SSE 루프가
  다음 폴(≤1초)에 끝난다 — 실측: 열린 SSE가 있는 실행 중 잡에서 SIGTERM 후 0.75초에 종료
  (예전 13.8초, `docker stop`은 매번 SIGKILL). uvicorn도 `--timeout-graceful-shutdown 5`로 뜬다.
  신호로 끝나는데 OCR 추론(잡 실행 또는 모델 프리로드)이 데몬 스레드에서 아직 돌면, 앱 정리(PDF
  워커 풀·소유 락)를 마친 뒤 마지막 atexit 처리기가 로그만 비우고 `os._exit`한다(SIGTERM 0,
  그 밖에는 128+신호). 그대로 두면 인터프리터 종료의 C++ 정적 소멸자(OpenMP·oneDNN 스레드 풀)가
  커널을 도는 스레드 밑에서 부서져 `std::terminate` → abort였다 — 컨테이너 PID 1(uvicorn이 다시
  올린 SIGTERM을 커널이 무시)에서 OCR 중 `docker stop`이 exit 133. 진행 중이던 잡은 다음 기동이
  '서버 재시작으로 중단'으로 마감한다.
- 잡 디렉터리가 사라지면(삭제) 루프가 끝난다.
- 이벤트:
  - `event: progress` `data: {"phase":"ocr","current_page":3,"total_pages":12,"chunk":1,"total_chunks":2,"status":"running"}`
    — `current_page`의 의미는 phase에 따라 다름: `loading`=sidecar 모델 준비 대기
    (페이지 미정, `note` 필드에 안내 문구), `render`=래스터화된 페이지 수,
    `ocr`=파싱 중 페이지(청크 시작 시 점프, `<PAGE>` 마커마다 증가), `merge`=총 페이지.
    **레이아웃 박스의 페이지 추적은 반드시 `phase==="ocr"`인 이벤트만 사용할 것**
    - `phase==="loading"`(sidecar 엔진만): 최초 기동의 모델 로딩을 기다리는 동안
      `{"phase":"loading","status":"queued","note":"모델 로딩 대기 중…"}`를 주기 발행.
      잡은 실패하지 않고 대기하며, 준비되면 자동으로 render/ocr로 진행한다.
  - `event: token`    `data: {"text":"델타 텍스트"}`   ← 모델 생성 토큰 실시간 (GIF 스타일)

  - `event: replay`   `data: {"text":"누적 토큰", "truncated":false,
    "current_page":3, "total_pages":12}` ← 업로드 응답과 최초 EventSource 연결 사이,
    또는 브라우저 자동 재연결 동안 생성된 토큰 복구. 구독 등록과 히스토리 스냅샷은
    브로커 락에서 원자적으로 수행돼 경계 토큰이 replay/신규 token 중 정확히 한 번
    전달된다. 실행 중 잡별 최대 8MiB를 보관하고 done/error 시 즉시 폐기하며,
    상한 초과 시 `truncated:true`로 전체 재구축을 생략한다(완료 산출물은 영향 없음).

    유실 대비: 느린 구독자에게서 token을 버려야 했다면 그 연결에 표식을 남기고,
    다음 폴 때 누적 원문 `replay`를 다시 보내 재동기화한다(조용한 유실 금지 —
    토큰 하나가 빠지면 `<PAGE>` 마커나 `<|det|>` 절반이 사라져 페이지 귀속이
    영구히 어긋난다).

    **토큰 스트림 문법 (실캡처로 확정, frontend/tests/fixtures/*.sse.txt):**
    **불변식 — 스트림의 k번째 `<PAGE>` 세그먼트 == 글로벌 페이지 k.** 이 프레이밍은
    모델이 아니라 **서버(`runner.BrokerSink`)가 보장한다**:
    - multi(`run_multi`): 모델이 페이지마다 마커를 낸다 → 통과시키며 카운트.
      청크 페이지 수를 넘는 초과 마커는 스트림에서 제거하고, 모자라면 청크 끝에서
      채운다 (merge의 result.md 보정 — 초과분 병합·부족분 빈 페이지 — 과 같은 기준).
    - single(`run_single`: per_page 모드·재처리 폴백): 모델은 마커를 내지 않는다 →
      서버가 페이지 시작에 하나 주입하고 모델이 흘린 마커는 제거한다.
    - 합성 페이지(텍스트 레이어 복구·실패 플레이스홀더): `BrokerSink.emit_page`가
      같은 프레이밍으로 스트림에 넣는다 (건너뛰면 이후 전 페이지가 한 칸씩 밀린다).

    `청크k 스트림 = <PAGE> + page(start_k) 내용 + <PAGE> + page(start_k+1) 내용 + …`
    청크 시작 직전에 `progress(phase=ocr, current_page=start_k, chunk=k)`가 먼저 발행되므로,
    **각 청크의 첫 마커는 이미 선언된 페이지의 재확인(no-op)** 이고 이후 마커만 +1이다.
    페이지 k+1의 선언은 **페이지 k의 마지막 토큰을 내보낸 뒤에** 발행된다 — 순서가
    뒤집히면 페이지 끝 블록의 박스가 다음 페이지 오버레이에 그려진다.
    블록 문법: `<|det|>label [x1,y1,x2,y2]<|/det|>텍스트…` (label: title/text/table/equation/
    image/page_number 등, 좌표 0–999 정규화) 또는 `<|ref|>label<|/ref|><|det|>[[…]]<|/det|>`.
    표는 블록 텍스트 안에 HTML `<table>`로 온다.
  - `event: reset`    `data: {"from_page": 17, "reason": "반복/출력 상한 감지 — 페이지별 재처리"}`
    ← **서버가 이미 보낸 출력을 폐기하고 `from_page`부터 다시 처리한다.** 클라이언트는
    누적 원문을 `from_page` 마커 지점까지 잘라내고(`truncateRawToPage`) 페이지 상태머신·
    박스·확정 프리뷰 캐시·RAW 패널을 그 지점으로 되돌린다(`live.js applyStreamReset`).
    발생 경로: 반복/출력 상한 감지 후 페이지별 재처리, 엔진 1회 재시도,
    PDF 텍스트 레이어 복구, 실패 플레이스홀더 삽입. 진행률(`current_page`)도 함께
    `from_page - 1`로 되돌아간다 — 실제로 그 페이지들을 다시 처리하기 때문이다.
    재연결 히스토리도 같은 지점까지 잘리므로 replay가 폐기분을 다시 싣지 않는다.
    클라이언트가 그 마커를 받은 적이 없으면(늦은 접속) 원문을 자르지 않고 안내만 남긴다.
  - `event: done`     `data: {"markdown_url":"...","archive_url":"..."}`
  - `event: error`    `data: {"message":"..."}`  (취소 시 `"canceled": true` 포함 — 대기 중 잡의
    취소는 POST /cancel 응답과 함께 즉시 발행된다). **삭제**는 실행 중·대기 중 모두
    `{"message":"삭제된 작업입니다","canceled":true,"deleted":true}`로 끝난다 — 프런트는
    `deleted:true`를 보면 그 잡의 화면·뷰어·구독을 닫고 목록 줄·읽던 위치·노트를 지운다.
    삭제 요청과 거의 동시에 잡이 done/error로 끝나면 구독자는 그 종료 이벤트를 받는다
    (프런트는 다음 전체 목록 폴에서 사라진 잡을 닫는다).

### GET /api/jobs/{id}/markdown
- `text/markdown; charset=utf-8`. 실행 중이면 완료된 청크까지의 부분 결과 + `X-Partial: true`

### GET /api/jobs/{id}/html
- 최종(또는 부분) 마크다운을 서버에서 HTML 프래그먼트로 렌더 (markdown-it-py, GFM 테이블 지원)
- `<img src="images/...">` → `src="/api/jobs/{id}/files/images/..."`로 재작성됨

### GET /api/jobs/{id}/layout
- **PDF facsimile 레이아웃 뷰**: 벤더 P14의 raw_pages.json →
  pipeline/layout.py가 파싱한 layout.json(페이지별 type/bbox 0–999/content)을
  사용하되, 화면은 원본/번역 PDF 페이지 이미지를 기준면으로 표시한다. OCR
  블록은 같은 좌표의 투명 텍스트 레이어로 남아 검색·선택·복사가 가능하다.
  페이지 이미지를 만들 수 없을 때만 좌표 텍스트 재조판으로 폴백한다.
  layout.json이 없거나 **텍스트 블록이 없으면**(image 블록뿐인 옛 figure_only 잡·전면 스캔)
  404 `이 잡에는 좌표 텍스트 레이아웃이 없습니다 (그림 전용 엔진·스캔 문서) — 텍스트
  보기(/html)를 사용하세요` (프론트는 탭에서 안내 문구 표시). `/alignment`·`/outline`·
  `/viewer/pages`도 같은 판정(`artifacts.has_usable_layout`)과 같은 404를 쓴다.
  `?lang=` 뷰는 번역 PDF raster를 유발하므로 `/document.html`과 함께
  **503 + `Retry-After`**(전역 빌드 대기열 초과, §5 `/pdf`)를 낼 수 있다.

### GET /api/jobs/{id}/document.html?lang=ko — 단독 HTML 내보내기
- 좌표 layout을 쓸 수 있으면 **facsimile**(페이지 PNG 인라인 + 투명 텍스트 레이어), 아니면
  **의미 기반** HTML(텍스트 보기와 같은 렌더)이다. 둘 다 KaTeX CSS/JS·woff2 폰트·크롭·페이지
  PNG를 data: URI로 품은 파일 하나이고, `<meta charset>` 바로 뒤에 meta CSP
  `default-src 'none'; img-src data: blob:; font-src data:; style-src 'unsafe-inline';
  script-src 'unsafe-inline'`과 `<meta name="referrer" content="no-referrer">`를 둔다 —
  디스크에서 열어도 정상 렌더되고, OCR 텍스트 속 추적·LAN 이미지를 포함한 어떤 외부 요청도
  나가지 않는다(Chromium은 img-src 위반만 기록). KaTeX 옵션은 앱과 같다(maxSize 10,
  maxExpand 1000, strict 'ignore', trust false — `test_layout`이 `constants.js`와 동기화를 고정).
- 레거시 `/layout.html`은 이 경로로 307 리다이렉트한다.

### GET /api/jobs/{id}/page/{page}?lang=ko
- 리더용 최종 페이지 PNG. 원문은 `pages/`, 번역은 `export.{lang}.pdf`를 잡 DPI로
  렌더한 `rendered/{lang}/` 캐시를 반환한다(캐시 마커는 PDF 크기·mtime·DPI·페이지 수).
- 상태코드: **400** 미지원 lang · **404** 번역본 없음/페이지 번호가 layout에 없음/
  이미지 파일 없음 · **409** 내보내기 불가(`PdfExportError` — 입력 누락·손상·빌드 시간 상한
  초과·빌드 워커 비정상 종료) · **503** 전역 빌드 대기열 초과(`Retry-After` 동반 — §5 `/pdf`의
  전역 빌드 상한). 요청한 쪽의 layout(원문은 `layout.json`, 번역은 `layout.{lang}.json`)에 텍스트
  블록이 없으면 원본 `pages/` PNG로 폴백한다. 번역 페이지 raster는 export 워커 풀에서 만든다(§18).

### GET /api/jobs/{id}/outline?lang=ko
- layout의 `title` 블록을 페이지·레벨·텍스트 목록으로 반환한다. 쓸 수 있는 layout이 없으면 404.

### GET /api/jobs/{id}/alignment?page=N&lang=ko
- 원문 bbox와 같은 인덱스의 원문/번역 블록을 연결한다. 번역 페이지·블록 수,
  type, bbox 대응이 어긋나면 잘못된 매핑을 내보내지 않고 409를 반환한다. 쓸 수 있는
  layout이 없으면 404.

### GET /api/jobs/{id}/viewer-manifest?lang=ko
- 전체 화면 논문 뷰어의 부트스트랩 계약. schema/artifact revision, 페이지 수,
  원문 이미지·번역 이미지·alignment·outline capability, 품질 경고와 URL 템플릿을
  작은 JSON으로 반환한다.
- 좌측 뷰어 기준면은 `source_page_template`을 사용해 항상 원문으로 고정한다.
  `translated_page_image`는 번역 layout을 쓸 수 있고 번역 PDF raster 캐시가 실제 준비된
  경우에만 true다. `alignment`·`outline` capability도 파일 존재가 아니라
  `has_usable_layout`(텍스트 블록이 있는 layout)으로 정한다.
- `quality`: `{state, warning_count, warnings(앞 20건), notice_count}` — `state`는 warnings가
  있으면 `degraded`, 없으면 `ok`다. 정보성 notices는 세지 않는다(§4 잡 메시지).
- `Cache-Control: private, no-cache`, `ETag`, `Vary: Authorization`을 제공하며
  일치하는 `If-None-Match`에는 304로 응답한다(`W/` 접두, 콤마 목록, `*` 모두 처리).

### GET /api/jobs/{id}/viewer/pages?start=N&limit=4&lang=ko&include=alignment
- 긴 문서의 인접 페이지 메타/좌표를 한 번의 layout 파싱으로 반환하는 제한 배치.
  `limit`은 1–16이며, 각 item은 원문 이미지 URL과 선택적 alignment를 포함한다.
  잘못된 범위/include는 422, 원문-번역 대응 불변식 위반은 409, 쓸 수 있는 layout이 없으면 404다.
- `ETag`를 보내고 `If-None-Match`가 맞으면 **layout을 파싱하기 전에** 304로 답한다
  (예전에는 재검증 때마다 layout JSON 두 개를 전부 다시 파싱했다).

### GET /api/jobs/{id}/files/{path}
- 잡 디렉터리 하위 정적 파일. 허용 디렉터리는 `pages/`·`images/`·`layout/`·`rendered/`.
- **경로 규칙**: 요청 경로를 `resolve(strict=True)`로 **먼저 정규화**(상위참조 해석 +
  심볼릭 링크 추적)한 뒤, 그 결과가 허용 디렉터리 하위인지 검사한다. 첫 세그먼트만
  allowlist와 비교하면 `pages/../source.pdf`처럼 잡 디렉터리 안의 임의 파일(원본 업로드
  PDF·meta.json·translations/**)이 서빙되므로, 검사 순서 자체가 계약이다.
- 허용 밖 경로·존재하지 않는 파일·디렉터리는 모두 404(존재 여부를 구분해 흘리지 않는다).

### GET /api/jobs/{id}/archive
- `result.md` + `meta.json` + `result.*.md`(번역본) + `images/`를 담은 zip. 미완료 시 409
- 다운로드 파일명은 **`{원본이름}.markdown.zip`**(`Path(job.filename).stem` 기준,
  비면 `result`). 잡 디렉터리에는 `archive.zip`으로 캐시되며, 번역 완료 시 무효화돼
  다음 요청에서 번역본까지 담아 재생성된다.
- 캐시 신선도는 **내용 서명**이다 — 멤버마다 (이름, inode, 크기, mtime_ns)를 해시해 zip
  주석 `uocr-archive:v1:<hash>`로 남기고, 요청 때 같은 서명이 아니면 다시 만든다. 만드는 도중
  입력이 바뀌면 같은 요청에서 다시 만든다(최대 3회). 번역 완료 경로는 `done`을 발행하기
  **전에** archive·PDF 캐시를 무효화하고 예열을 시작한다(예전에는 번역 직후 ZIP이 옛 번역본을
  줄 수 있었다). 확인과 전송 사이에 파일이 사라지면 500이 아니라 404다.

### GET /api/jobs/{id}/pdf?lang=ko[&view=dual]
- 기본 `view=single`은 **레이아웃 보존 번역 PDF** (`{원본이름}.ko.pdf`, application/pdf)를 반환한다.
  `layout.json`(원문)과 `layout.{lang}.json`(번역)을 블록 단위로 비교해
  **내용이 실제로 바뀐 텍스트 블록만** 원본 PDF에서 리댁션(텍스트만 제거,
  이미지·그래픽 보존) 후 같은 자리에 번역 텍스트를 삽입한다
  (`pipeline/pdf_export/` 패키지 — 진입점 `build_translated_pdf`, §3).
  행·열 구조가 같은 HTML 표는 PDF 텍스트 검색으로
  셀 격자를 추정해 **셀별 번역**하고 벡터 격자선을 보존한다(최대 500셀).
  독립 수식·그림·참고문헌과 세로쓰기 블록은 원본 글리프를 유지하며, 일반
  텍스트 안의 단순 LaTeX 위첨자는 읽을 수 있는 평문(`mc^{2}`→`mc²`)으로 낮춘다.
- **표 격자 신뢰도 게이트**: 셀 격자 추정은 `(셀 사각형, grid_trusted)`를 함께 낸다.
  텍스트 레이어가 있는 표인데 원문 셀 검색이 **절반도 맞지 않으면**(`관측×2 < 텍스트 셀 수`)
  균등 분할 격자는 실제 열 폭과 무관하므로 — 번역문이 엉뚱한 셀에 찍히고 인접 셀 원문이
  리댁션된다 — 그 표는 **교체하지 않고 원문을 보존**하고
  `p{n}: 표 셀 격자 추정 실패(원문 검색 불일치) — 원문 표 보존` 경고를 남긴다.
  원문이 스캔 픽셀인 표(래스터 원문 블록)는 이 검색 격자를 쓰지 않는다 — 바뀐 셀을 바탕색으로
  덮으므로 균등 격자는 이웃 셀 글자와 괘선을 지운다. `pdf_export/raster_tables.py`가 표 영역을
  144dpi로 렌더해 긴 어두운 런(괘선)을 떼어 낸 잉크 투영으로 실제 행·열 경계를 찾고(글자 있는
  셀마다 잉크, 서로 다른 셀 경계를 가로지르는 잉크 없음, 고른 경계가 단어 간격보다 뚜렷이 넓음,
  잉크 폭/글자 수 검사), 바뀐 셀의 글자 잉크 상자만 괘선 안쪽으로 덮은 뒤 원문 baseline에
  번역을 넣는다(회전 페이지 포함). 확정하지 못하면 덮지 않고
  `p{n}: 스캔 표의 열·행 경계를 픽셀에서 확정하지 못함 — 원문 표 보존`(사유
  `table_grid_untrusted`)을 남긴다.
- **무손실 가드**: 원문/번역 레이아웃의 페이지·블록 수와 각 블록의
  `type`/`bbox`/`image` 대응을 문서 수정 전에 전부 검증한다. 번역 텍스트는
  PyMuPDF `Shape`로 축소 조판을 dry-run한 뒤 들어가는 블록만 리댁션하며, 최소
  크기에도 들어가지 않으면 번역을 생략하고 원문을 보존해 경고로 남긴다.
  1차 리댁션은 `images=NONE`, `graphics=NONE`, `text=REMOVE`를 명시해 블록 안의
  그림·밑줄·차트 선을 제거하지 않는다. 확장된 삽입 사각형이 아니라 **실제 원문
  bbox만** 리댁션해 인접 원문 글리프가 함께 사라지지 않게 한다. 원래 bbox에
  들어가지 않으면 같은 단의 다음 블록·표·그림·푸터 앞까지만 아래 빈 영역을
  사용하며, 그래도 부족하면 원문을 보존한다. 빈 `list` 컨테이너는 장애물에서
  제외하고, y 경계가 맞닿는 다음 문단은 확장 공간으로 오인하지 않아 번역 블록
  중첩을 막는다.
  **피할 수 없는 장애물은 장애물이 아니다.** 큰 도형(코드 상자 테두리·배경 채움)을
  테두리 띠로 낮출 때 생기는 가로 띠가 본문 블록 **안쪽**에 놓이면 어떤 줄 배치로도
  피할 수 없어 자리가 남는데도 `no_fit`이 된다(실측 p5: 338–438pt 상자 안의
  y=343.0–344.5 전폭 띠). 원문 글자가 그 띠를 가로질러 인쇄돼 있으므로 장식으로 보고
  제외한다. 반대로 상자 **윗변만** 2~3pt 덮는 장애물(그림 범례 조각 등)에는 장애물
  아래에서 시작하는 상자 후보를 하나 더 시도한다. 두 규칙으로 실측 `no_fit` 66 → 26,
  교체 203 → 243.
- **이모지 2차 리댁션(예외적 IMAGE_REMOVE)**: macOS Quartz 산출 PDF는 컬러 이모지를
  (보이지 않는 텍스트 글리프 + 이미지 XObject) 이중으로 기록해 텍스트 리댁션만으로는
  이미지가 번역문 위에 남는다. 그래서 1차 리댁션 뒤 **좁은 조건을 모두 만족하는 소형
  래스터 인스턴스만** `images=IMAGE_REMOVE, graphics=NONE, text=NONE`의 별도 pass로
  지운다 — ① 교체 사각형에 완전히 포함(1pt 허용 오차) ② 면적이 그 사각형의 25% 이하
  ③ **긴 변이 `max(2×블록 폰트 크기, 20pt)` 이하** ④ 종횡비 0.5–2.0.
  ③의 절대 크기 상한이 없으면 넓은 블록 안의 인라인 그림·로고까지 지워지고,
  블록 rect 전체로 `IMAGE_REMOVE`를 걸면 걸친 그림에 흰 구멍이 난다.
- **인라인 수식 선(분수선·근호 윗선)**: TeX 계열 PDF의 이 선은 글리프가 아니라 얇은 가로
  path라 `graphics=NONE` 텍스트 리댁션 뒤 번역문 옆에 남았고, 고정 장애물이라 자기 문단의
  자리를 막아 번역을 가독성 하한 아래로 축소시켰다. 두께 ≤ 1.5pt·길이 1–160pt인 가로 선
  하나로 된 path가 내용 있는 교체 가능 블록 **하나**에만 담기면(1pt 여유, 빈 `list` 컨테이너
  제외, 그림 영역에 걸치지 않음, 끝점에 짧은 세로 선이 닿는 상자·괄호 변 제외) 그 블록이
  소유한다(`_PageVisuals.inline_rules`). 소유된 선은 원문 span처럼 블록이 남을 때만 장애물이고,
  블록이 flow로 통째로 교체되면 선 자신의 사각형으로 `graphics=REMOVE_IF_COVERED`(채움 없음)
  별도 pass로 지운다. 줄 단위 리스팅은 남는 줄이 있을 수 있어 선을 지우지 않고 모두 피한다.
  실측 25쪽 논문: 축소·공간 부족 경고 7 → 1건(포맷 버전 12).
- **CropBox 가드**: 아래 빈 영역 확장의 페이지 하단 기준선은 `page.mediabox`가 아니라
  `page.rect × derotation_matrix`(정규화)로 얻은 **CropBox 기준 표시 영역**이다.
  mediabox는 CropBox를 무시한 PDF 원좌표라 CropBox≠MediaBox 문서에서 실제 페이지
  하단보다 아래를 기준으로 삼고, 회전 페이지에서는 derotation 없이 쓰면 가로/세로가
  뒤바뀐다. 블록 bbox와 같은 비회전 내부 좌표계로 맞춘 값만 쓴다.
- **스캔·이미지 페이지**: 원문이 래스터 픽셀인 블록 — 보이는 원문 span의 중심이 안에 없고,
  스캔 배경 — 래스터 한 장이 페이지의 85% 이상이거나, 맞닿은 래스터 덩어리(가로 띠·타일)가
  85% 이상이거나, 50% 이상이면서 레이아웃 그림이 아니고 안에 보이는 글자가 없는 덩어리(여백
  있는 스캔) — 와 50% 이상 겹치거나 래스터 안에 80% 이상 들어간 표 — 은
  텍스트 리댁션으로 지울 수 없다. 예전에는 영어 픽셀 위에 한국어를 겹쳐 찍었다. 이제 원래
  영역을 그 페이지의 **가장 흔한 바탕색**(50dpi 렌더에서 한 번 표본, 어두우면 흰색)으로 덮는
  리댁션(`images=NONE` — 이미지 픽셀을 다시 인코딩하지 않는다. `IMAGE_PIXELS`는 300dpi JPEG
  스캔 페이지를 Flate로 다시 써 0.92 → 1.73 MB로 키워 기각) 뒤 번역을 넣는다. 원문을 남긴
  래스터 블록은 영역 전체가 장애물이고, 레이아웃의 그림 블록 안쪽은 건드리지 않는다. 투명
  OCR 텍스트 레이어(알파 0 등 — Acrobat·ABBYY·ocrmypdf)는 '보이지 않는 원문'으로 본다. 나중에
  그린 불투명 이미지에 가려진 보이는 모드 텍스트(이미지 아래 텍스트 스캔 — `get_bboxlog` 그리기
  순서, 소프트 마스크 이미지 제외)도 보이지 않는 원문으로 본다. 덮개는 레이아웃 그림·표 블록과
  교체되지 않는 블록 영역(0.8pt 여유 안쪽)을 빼고 덮는다. 내용이 빈 컨테이너(빈 list)는 지키지도
  래스터 원문으로 세지도 않는다. 덮을 곳이 남지 않으면 원문 픽셀이 이웃 그림·보존 블록과 겹쳐
  덮지 못했다는 경고를 남긴다. 세로로 길고 좁은 래스터 블록 중 한 줄짜리(종횡비 6 이상·12자
  이상·줄바꿈 없음·글자 수×폭×0.45 ≤ 높이×1.5)만 여백 도장처럼 세로쓰기로 보고 원문을 두며 그
  사실을 경고로 남긴다 — 좁은 다줄 단(신문 단)은 덮고 번역한다. 리포트 키 `raster_blocks_erased`.
- **리댁션은 줄마다 기준선 띠**: span bbox 대신 MuPDF가 지우는 글리프 상자(폰트 상하단에서
  위아래 10%를 뺀 것)의 가운데 띠(최대 0.5em)를 줄 방향을 따라 지운다 — 10pt/12pt 행간에서
  예전 사각형이 다음 줄을 지우던 문제가 사라졌다. 블록·리스팅·표 셀 경로가 함께 쓴다.
  폰트의 원래 ascender−descender가 0.5em 미만인 퇴화 메트릭 span(PyMuPDF가 bbox를 1em으로
  늘려 보고)은 예전처럼 span bbox(+0.25pt)로 지운다. 원문 span은 `rawdict`로 읽어 글자 원점을
  본다 — 기준선이 0.3em 넘게 다른 글자를 쌓은 span(TeX 확장 괄호: CMEX10 막대 조각)은 조각마다의
  띠를 가장 낮은·높은 원점까지 이어 지운다(가운데 띠 하나는 가운데 조각만 지웠다). 제어 코드
  글리프('\x0c'·'\r' 등 기호 폰트의 확장 조각)만 든 span도 원문이다 — 진짜 공백류만 버린다
  (예전 `str.strip()`이 버려 소유 블록 없이 남은 막대가 번역문에 겹쳤다).
- **줄 밖으로 튀어나온 기호**: 근호(√)·큰 괄호·악센트처럼 줄 위·아래로 튀어나온 1–2글자 span은
  줄에 딱 맞는 OCR bbox와의 겹침이 35%에 못 미치고 가운데도 bbox 밖이라 어느 블록 것도 아니었다
  — 그 블록을 번역으로 바꾸면 기호만 남아 번역문 위에 찍혔다(실앱 25쪽 논문 10쪽 CMSY8 '√',
  겹침 32.5%). 가로로 쓴 줄의 그런 span이 **다른 블록과는 전혀 닿지 않고** 가로 80% 이상 그 블록
  안·세로 25% 이상 겹치면 그 블록에 붙인다(`spans._edge_symbol_owner`, `_SourceSpan.edge`). 붙인
  기호는 블록을 교체할 때 함께 지우고 블록이 남을 때만 장애물이며, 줄 구조 판단(시각 줄 수·리스팅
  정렬·굵은 접두)에는 쓰지 않는다(포맷 버전 16).
- **텍스트 평탄화**: 태그 제거는 `layout.HTML_TAG_RE` 화이트리스트(`p < 0.05 … n > 30`·`<think>`가
  살아남는다), 모델 특수 토큰(`<|…|>`) 보존, 원문이 목록이면 `• `, 강조 기호는 단어 경계에서만
  벗기고 코드 span은 보호, LaTeX 구조 변환(`\frac`→a/b, `\sqrt`, `\mathbb`, 악센트, 간격 명령,
  `\|`→‖, `\langle`·`\rangle`→⟨⟩(폰트에 없으면 〈〉), 그리스 문자·화살표 등 KS X 1001 글리프
  범위). 기호 명령 바로 뒤의 감싸개·분수(`\langle\boldsymbol{y}`·`\cdot\frac{x}{y}`)는 먼저 바뀐
  글자가 명령 이름에 붙지 않게 경계 문자를 끼웠다가 지운다(예전 'langley'·'cdotx/y').
- **회전 페이지**(`/Rotate 90/180/270`): 폰트 실측 주입, 한 줄 조판, 여러 줄 흐름 배치(같은 단
  묶기·아래로 자라기)가 모두 화면 좌표에서 계산하고 페이지 좌표로 되돌린다 — 흐름 배치는 화면
  크기의 회전 없는 임시 페이지에서 dry-run한다(textbox는 rotate=90·270에서 상자 높이를 줄 폭으로
  써 줄바꿈이 같다; `ENRICH_VERSION` 6).
- **흐름 배치 보수화**: '피할 수 없는 내부 장애물' 예외는 벡터 도형 띠에만 적용(원문 span·
  이미지·배치된 텍스트는 절대 가지치지 않는다), 압축 재배치에 읽기 순서 하한, 부분 리스팅이
  남긴 행은 장애물로 유지, 정렬하지 못한 리스팅 줄은 그 줄만 원문으로 두는 보존 사유
  `listing_line_unaligned`, 패스 상한에 닿으면 아무것도 지워졌다고 가정하지 않고 다시 계획.
  표는 검색 클립마다 TextPage 하나(9셀 표: 10개 → 1개), 가로 괘선은 페이지 드로잉 1회에서 읽는다.
- **기타**: 공백·NBSP가 같은 글리프인 폰트의 ToUnicode를 `<0020>`으로 고쳐 복사한 공백이 NBSP로
  나오지 않게 하고(OCR 레이어 스캔 표본 1,995 → 0), 퇴화 det 좌표는 블록만 건너뛰되 크롭 번호는
  소비한다. 보수적인 장애물 모델 때문에 일부 예전 겹쳐 찍기가 `no_fit`(원문 보존)으로 바뀌었고,
  25쪽 표본 빌드는 약 12% 느려졌다(가짜 번역 95 → 108초).
- `view=dual`은 UI의 기본 내보내기다. 같은 번호의 원본 페이지를 왼쪽, 위 단일
  번역 PDF 페이지를 오른쪽에 원래 크기로 붙이고 중앙에 1pt 선을 그린다. 따라서
  A4 세로 원본은 A3 가로 대조 페이지가 되며, 래스터화하지 않아 벡터·그림·텍스트
  선택성을 유지한다.
- 폰트: 원본 span의 실측 크기·굵기·정렬과 `font_style=serif|sans`를 추출한다.
  serif 블록은 시스템 한글 명조(macOS AppleMyungjo, Linux Noto Serif CJK),
  sans 블록은 시스템 한글 고딕에 대응하고 PyMuPDF 내장 CJK(`korea`)로 폴백한다.
  한 줄 번역은 CJK textbox 높이 판정 때문에 축소하지 않고 원문 baseline에 직접
  삽입하며, 여러 줄 본문만 자연 행간→조밀 행간→폰트 축소 순서로 dry-run한다.
  `PDF_EXPORT_FONT`를 지정하면 명시 폰트를 우선한다. PDF 요청 자체가 이 메타를
  읽으므로 사용자가 먼저 레이아웃 탭을 열지 않아도 원본 타이포를 기준으로 조판한다.
  폰트 메트릭 객체(`fitz.Font`)는 `lru_cache(maxsize=8)`로 재사용한다 — 조판
  dry-run이 블록마다 폰트 파일을 다시 파싱하던 비용 제거(결과 불변).
  축소 조판 dry-run(`Shape.insert_textbox`)은 빌드마다 연 `trial_pages()` 안에서 원문 페이지와
  기하(MediaBox·CropBox·변환 행렬·회전)가 같은 빈 페이지로 돈다(`fitting._trial_page` — 기하를
  그대로 복제하지 못하면 원문 페이지). 시험마다 `insert_font`가 그 페이지의 폰트 리소스 전체를
  다시 훑는데(폼 XObject 재귀 포함), 논문 페이지에서는 그 스캔이 빌드 시간의 3분의 1을 넘었다.
  25쪽 논문 빌드 40→26초, CropBox 스트레스 사본 200→144초이고 렌더 픽셀·텍스트·리포트는 같다.
- 상태코드: 400 미지원 lang **또는 미지원 `view`**(single|dual 외) · 404 번역본 없음 ·
  409 미완료 잡, 좌표 레이아웃을 쓸 수 없음(원문·번역 layout에 텍스트 블록이 없음 —
  figure_only 엔진 등, document.html 사용 안내), 내보내기 불가(`PdfExportError` — 입력 누락·
  손상·layout 블록 불일치·MuPDF 오류·빌드 시간 상한 초과·빌드 워커 비정상 종료·삭제된 잡) ·
  **503 빌드 대기열 초과**(`PdfExportBusyError`, `Retry-After` 동반 — 아래 전역 상한).
  예전에는 `PdfExportError`가 500이라 프록시가 본문을 가로채 사용자가 원인 문구를 잃었다.
  `build_translated_pdf`는 페이지 처리·저장의 어떤 예외든 예외 클래스 이름을 담은
  `PdfExportError`로 바꾸고(트레이스백은 서버 로그 WARNING), `build_dual_pdf`는 출력
  디렉터리를 만들지 않는다 — 잡 디렉터리가 사라졌으면 `삭제된 작업입니다 …`로 실패한다.
- **빌드는 export 워커 프로세스에서 돈다**(§18): 빌드 한 건의 상한은
  `PDF_EXPORT_BUILD_TIMEOUT_S`(기본 900초, 0 이하=없음). 넘기거나 워커가 죽으면
  `번역 PDF 생성이 시간 상한(…)을 넘어 중단했습니다 … (PDF_EXPORT_BUILD_TIMEOUT_S)` /
  `… 처리 프로세스가 비정상 종료했습니다 …` 409이고 그 워커만 정리된다. 잡 락과 캐시 판정은
  서버 프로세스에 남는다.
- **전역 빌드 상한 (503 + `Retry-After`)**: 잡 단위 락은 *같은 잡*의 중복 빌드만 막는다.
  `pipeline/derived.py::export_build_slot`이 프로세스 전역 세마포어
  (`PDF_EXPORT_MAX_CONCURRENT`, 기본 **2**)로 빌드 수를 묶고, 슬롯을
  `PDF_EXPORT_QUEUE_TIMEOUT_S`(기본 **30초**) 안에 못 얻으면 매달리는 대신
  **503 + `Retry-After`**로 거절한다(재시도하면 성공할 수 있는 일시적 과부하 —
  입력 누락·손상인 `PdfExportError`와 구분된다). 슬롯은 **실제 빌드에서만** 잡고
  캐시 적중 경로에서는 잡지 않는다. `0` 이하로 두면 슬롯 상한 비활성(export 워커 풀은
  최대 min(8, CPU)개라 그 이상은 워커를 기다린다). 같은 값이 export 워커 수라 빌드 N개가
  **실제로 병렬**이다 — 예전 스레드 빌드는 GIL 때문에 코어 하나를 나눠 썼다(가속비 1.00).
  같은 슬롯을 쓰는 라우트는 이 `/pdf` 외에 `/document.html`·`/layout`·`/page/{n}`
  (번역 페이지 raster가 export를 유발한다)까지 넷이며, 모두 같은 503을 낼 수 있다.
  다만 **같은 (job, lang)의 예열 빌드가 도는 중**이라면 그 기다림은 줄서기가 아니라
  사용자가 원하는 바로 그 PDF가 만들어지는 시간이다. 이 경우에만 대기 상한을
  `PDF_EXPORT_WARM_WAIT_S`(기본 **180초**)로 바꾼다 — 30초로 묶으면 46쪽 문서
  (실측 빌드 67~75s)에서 번역 직후 첫 다운로드가 **항상** 503이었다. 한 요청 안의 여러
  단계(단일 → 대조, facsimile)는 **하나의 대기 예산**을 쓰고, 안쪽 호출이 더 큰 예산을 열면
  이미 기다린 시간을 빼고 넓힌다 — 총 대기는 둘 중 큰 상한으로 묶인다.
  ⚠ 업그레이드 직후 `PDF_EXPORT_FORMAT_VERSION`이 오르면 전 캐시가 한꺼번에
  무효화돼 이 폭주가 **실제로** 일어난다(§15.1).
- **예열(prewarm)**: 번역 완료 직후 PDF를 미리 빌드한다. 예열은 `N-1`개 크기의 예열 전용
  슬롯을 비차단으로 먼저 잡아야 해서 사용자 클릭용 빌드 슬롯이 늘 하나 남는다 —
  `PDF_EXPORT_MAX_CONCURRENT=1`이면 예열하지 않는다(첫 클릭이 빌드). 예열 중에 다시 들어온
  예열 요청은 dirty 표시로 같은 스레드가 다시 돈다(최대 4회).
- 응답 헤더: `X-UOCR-PDF-Replaced`, `-Preserved`, `-Relocated`, `-Table-Cells`,
  `-Specialist-Preserved`, `-Warnings`. 모두 숫자만 담아 원문·경고 본문이 프록시
  메타데이터로 새지 않으며, 프런트 다운로드 토스트가 이를 요약한다. 보존 사유·주의 문장은
  아래 `/pdf/report`로만 나간다.
- 캐시: 단일판 `job.dir/export.{lang}.pdf` + `export.{lang}.report.json`, 대조판
  `export.{lang}.dual.pdf`. 단일판의 유효 조건은 **빌드 스탬프 == 현재값**이다 —
  `export.{lang}.font.txt`에 JSON `{"v":2,"font":<폰트 id>,"inputs":{파일:[inode,크기,
  mtime_ns]}}`(source.pdf·layout.json·layout.{lang}.json)를 빌드 **전에** 잰 값으로 남긴다.
  빌드 도중 입력이 바뀌면 그 결과는 유효로 인정되지 않고 다시 예열된다(facsimile 경로처럼 같은
  잡 락을 바깥에서 쥐고 있으면 가장 바깥 락을 놓은 뒤에 예열을 시작한다. 예전 mtime 비교는
  '입력 교체 뒤 쓰인 옛 출력'을 신선하다고 봤다). 폰트 id에는 명시 폰트(경로·크기·mtime),
  시스템 CJK 폰트 후보 목록의 다이제스트, fontTools 유무가 들어간다(30초 TTL) —
  `fonts-noto-cjk`나 fontTools를 설치하면 폴백 폰트로 만든 PDF가 무효화된다. 대조판은
  원본·단일판보다 오래되면 재생성한다(mtime 규칙, 같은 잡 락 아래 직렬). 번역 완료 시 함께
  무효화한다. 리포트의 `format_version`이 현행 `PDF_EXPORT_FORMAT_VERSION`
  (`pipeline/pdf_export/report.py`, 현재 **11**)과 다르면 캐시를 무시하고 재생성한다 —
  내보내기 동작이 바뀐 사이클에서는 기존 export 캐시가 전부 한 번 재생성된다(§15.1).
- 레이아웃 폰트 백필(`ENRICH_VERSION`이 오른 layout을 처음 읽을 때 실측 폰트를 다시 주입)은
  산출물(layout[.lang].json)마다 하나뿐인 백그라운드 스레드가 export 워커 풀에서 돌린다. 요청은
  그 백필이 시작된 뒤 최대 2초만 기다리고(빈 풀의 보통 문서는 그 안에 끝나 실측 메타로 답한다),
  넘으면 지금 layout(폴백 휴리스틱)으로 답한다 — 빌드가 export 워커를 모두 쥐고 있어도 리더
  라우트가 멈추지 않는다. 백필 스레드는 빈 워커를 상한 없이 기다려 빌드 뒤에 끝낸다. `/page/{n}`은
  기다리지 않는다. 이미 최신이면 다시 쓰지도 예열하지도 않고, 기다리는 사이 파일이 바뀌었거나
  종료 중이면 저장하지 않는다.

### GET /api/jobs/{id}/pdf/report?lang=ko
- 마지막 번역 PDF 빌드의 생성 리포트(`export.{lang}.report.json`)를
  `{"job_id","lang", …report}`로 반환한다 — `format_version`, `replaced`, `kept`, `relocated`,
  `table_cells_replaced`, `listing_lines_replaced`, `raster_blocks_erased`, `specialist_kept`,
  `kept_reasons`, `kept_pages`(`[[페이지, 보존 블록 수], …]` — 자르지 않음), `warning_pages`(경고가
  있는 페이지 전부), `warning_count`, `warnings`(앞 50건 표본). 경로·문서 본문은 없다.
  `verify_e2e`는 '사유 없이 번역이 사라진 페이지' 판정에 `kept_pages`를 쓴다 — 경고 표본으로
  읽으면 51번째 이후 경고의 페이지가 무성 유실로 보였고, 무관한 경고 하나가 그 페이지의 유실을
  모두 설명된 것으로 만들었다.
- 단일·대조 PDF는 같은 번역 PDF 빌드에서 나오므로 리포트는 하나다. 프런트는 다운로드 뒤 이
  JSON으로 '스캔 원문 N개 블록 지움'을 포함한 토스트와 'PDF 생성 리포트 · 주의 N건' 목록을
  그린다.
- 404 `PDF 생성 리포트가 없습니다 — 번역 PDF를 먼저 내보내세요`(빌드 전, 리포트 손상, 번역
  갱신으로 지워진 뒤) · 400 미지원 lang.

### POST /api/jobs/{id}/cancel
- 실행/대기 중 잡을 **삭제 없이** 중단. 잡은 `canceled` 상태로 남고
  완료된 청크까지의 부분 결과는 /markdown 등에서 계속 접근 가능
- 202 `{"job_id","status"}`:
  - 워커가 아직 맡지 않은 **대기 잡**은 같은 락 안에서 즉시 마감한다(`JobStore.try_cancel_queued`) —
    `{"status":"canceled"}`와 함께 종료 SSE(`error {canceled:true}`)를 발행하고 대기열 자리를
    비운다. 워커는 그 잡을 건너뛴다. 예전에는 앞 잡이 끝날 때까지 '취소 중…'으로 남았다.
  - 실행 중(모델 로딩 대기 포함) 잡은 `{"status":"canceling"}` — 워커가 다음 확인 지점에서
    마감한다. 렌더 중이면 그 페이지를 그리던 워커 프로세스를 바로 끝낸다(§18).
  - 이미 종료된 잡이면 현재 status를 그대로 반환한다.

### POST /api/jobs/{id}/render-preview
- 요청 본문(text/plain, **≤256KiB**)의 마크다운을 /html과 동일한 안전 렌더러로
  HTML 프래그먼트 렌더 (라이브 미리보기용 — 프론트가 정리한 스트림 텍스트를 debounce 전송).
  예전 2MB 상한은 인증 없는 요청 한 건에 수십 초짜리 렌더를 허용했다 — 렌더러의 수식·코드
  스캔은 이제 선형 시간이다(§14).
- 413(256KiB 초과) · 429 + `Retry-After`(크기 가중 레이트리밋·동시 4건 — §5 남용 방어).
  프런트는 429를 실패로 세지 않고 `Retry-After`(1–300초, 없으면 30초)만큼 쉰다.

### DELETE /api/jobs/{id}
- 실행 중이면 취소(cancel) 후 삭제, 완료면 디렉터리 삭제. 204
- 구독자에게 `error {"message":"삭제된 작업입니다","canceled":true,"deleted":true}`를
  발행한다(실행 중 잡은 runner가, 그 밖은 삭제 전에 API가). TTL GC도 같은 경로
  (`JobStore.delete_dir`)라 삭제 요청 표시를 세운다.
- 이 잡의 **실행 중 번역 스레드에도 cancel을 전파**한다 — 삭제된 디렉터리에
  유료 API 호출·파일 기록이 계속되지 않게. 번역 엔진과 파생 빌드는 잡 디렉터리를 **다시
  만들지 않는다**(`parents=True` 금지) — 빌드 도중 삭제됐으면 되살린 디렉터리를 지우고
  실패한다. 번역은 `작업 디렉터리가 없습니다 — 삭제된 작업은 번역할 수 없습니다`로 끝난다.

## 6. 디바이스 백엔드

| 백엔드 | 상태 | 선택 방법 | 비고 |
|---|---|---|---|
| CPU | ✅ 구현 | `OCR_DEVICE=cpu` | 기본 dtype float32 (`OCR_DTYPE`로 변경 가능) |
| CUDA | ✅ 구현 | `OCR_DEVICE=cuda` | bf16, cu129 휠, sm_89/sm_120 확인 |
| MLX | ✅ 구현 | `OCR_DEVICE=mlx` | **Apple Silicon 기본**(auto의 1순위). in-process MLX 포팅(`engine/unlimited_mlx.py`). macOS 14+ arm64 로컬 실행 전용 — Docker 불가. `make setup-mlx` |
| Metal | ✅ 구현 | `OCR_DEVICE=metal` (별칭 `mps`) | torch MPS 폴백. Apple Silicon 로컬 실행 전용 — Docker 불가. `make setup-mlx`(metal extra 포함)·`make dev-metal` |

`app/engine/registry.py`가 단일 진입점: 디바이스/엔진 이름 검증(`VALID_DEVICES` =
auto·cpu·cuda·metal·mlx) 후 엔진 생성. CUDA/MPS/MLX 가용성 검증은 `load()` 시점(= 프리로드
스레드/첫 잡)에 수행되어 실패 사유가 `/api/health`의 `model_load_error`로 노출된다.

### 디바이스 자동 선택 (`OCR_DEVICE=auto`)

- `Settings.from_env()`는 `OCR_DEVICE`가 없거나 비면 **`auto`**다(`Settings()` 데이터클래스
  기본은 테스트·스크립트가 결정적이도록 `cpu`). 예전에는 Apple Silicon의 `make dev`가 조용히
  CPU fp32로 돌았다.
- auto는 **unlimited 엔진에만** 적용된다: `resolve_auto_device()`가 mlx(macOS arm64 + mlx 임포트
  가능 + Metal 사용 가능 — 플랫폼을 먼저 봐서 Linux·Docker는 mlx를 임포트하지 않는다) → cuda
  (`torch.cuda.is_available()`) → metal(torch MPS) → cpu 순으로 처음 쓸 수 있는 것을 고르고
  INFO로 남긴다(`OCR_DEVICE=auto → mlx (…)`). 엔진은 디바이스를 푼 설정 사본을 받으므로
  health의 `device`·`dtype`은 실제 값이다(`app.state.settings.device`는 `auto` 그대로 —
  디바이스가 필요하면 `engine.device`를 본다). torch로 넘어가면(Linux, mlx 없는 Mac) 판정을
  위해 기동 때 torch를 동기로 임포트한다(약 1–2초).
- fake·textlayer·sidecar 엔진은 하드웨어를 조사하지 않는다(fake는 `cpu`로 표기).
- unlimited 엔진을 Apple Silicon에서 **명시적으로 `cpu`**로 두면 MLX보다 수십 배 느리다는
  WARNING을 기동 로그에 남긴다.
- **명시적 `mlx`**를 Apple Silicon 밖이나 mlx 없이 쓰면 엔진은 만들어지지만 `load()`가 한국어
  사유와 설치 안내(`cd backend && uv sync --extra metal --extra mlx (= make setup-mlx)`)를 담은
  `EngineError`를 낸다 — health의 `model_load_error`. metal·cuda가 없는 환경과 같은 규칙이다.
- 컨테이너는 영향이 없다: compose는 backend 서비스마다 `OCR_DEVICE`를 고정하고
  (ocr-cpu·ocr-ovis·ocr-paddle = cpu, ocr-cuda = cuda), Dockerfile에는 `OCR_DEVICE`가 없어 CPU
  이미지는 auto여도 cpu다. 부작용 하나: compose 없이 `docker run --gpus all`로 cuda 이미지를
  띄우면 이제 cuda를 쓴다(예전 cpu).

### MLX 엔진 (`engine/unlimited_mlx.py`)

- 계약은 torch 엔진과 같다 — 이름 `unlimited`, 같은 프롬프트·해상도·no-repeat-ngram(35, 창
  1024/128)·`MAX_LENGTH`, 같은 산출물 이름 규약, 같은 capability(멀티페이지 문맥·토큰
  스트리밍·full layout·figure — 테스트가 고정). 충실도 게이트·청크 복구도 그대로 탄다.
- dtype: `OCR_DTYPE=auto` → bfloat16(float16·float32도 가능), `OCR_MLX_QUANT_BITS=8`이면
  디코더만 인메모리 8비트(group 64, affine — SAM·CLIP·projector·MoE 게이트는 그대로)로
  health `dtype`은 `bfloat16+q8`. 0·8 밖의 값은 기동 시 실패한다.
- `load()`: 가용성 확인 → 고정 스냅샷 해석 — 커밋 고정 리비전이 캐시에 완전하면(설정·토크나이저
  설정·인덱스의 샤드 전부) 그 디렉터리를 바로 쓰고(Hub 조회 0회), 아니면
  `huggingface_hub.snapshot_download`로 받는다(`HF_HOME`·`HF_HUB_CACHE`·`HF_HUB_OFFLINE` 존중,
  torch 경로와 같은 캐시) → strict 로드(키·모양이
  정확히 같아야 한다) → `warmup()` 1회(약 0.45초 — 두 모드의 Metal 커널을 미리 JIT, 없으면 첫
  실행 TTFT가 약 4배). 멱등·스레드 세이프.
- 스트리밍: HF `TextStreamer`와 같은 텍스트를 같은 단위로 낸다. 바이트 수준 토크나이저면
  접두 메모 디코더가 줄마다 통째로 다시 디코드하던 O(L²)를 없앤다(6,000토큰 한 줄 1.56 →
  0.018초). EOS 문자열은 torch처럼 `\n`.
- 취소·반복 감지는 토큰마다 확인한다(멈출 때 버리는 GPU 스텝 최대 1개). 취소는 부분 출력을
  돌려주고, 반복은 `RepetitiveOutputError`, `MAX_LENGTH`는 `OutputLimitError`(multi는
  `partial_output` 포함)다. 실행이 끝나면(예외 포함) `mx.clear_cache()`, 실행 락이 GPU 사용을
  직렬화한다. 실행마다 INFO 한 줄(프롬프트·생성 토큰·TTFT·tok/s·종료 사유).
- 공유 노브: `MAX_LENGTH`, `MAX_PAGE_OUTPUT_*`, `PAGES_PER_CHUNK`, `MODEL_ID`/`MODEL_REVISION`,
  `OCR_FIDELITY_*`, `OCR_DTYPE`. **무시하는 torch 전용 노브**: `OCR_FAST_DECODE`,
  `OCR_DECODE_BLOCK`, `OCR_CPU_THREADS`, `OCR_MOE_*`, `OCR_CUDA_GRAPHS`, `OCR_SDPA`,
  `OCR_NGRAM_HOST`.
- MLX 코드는 torch를 임포트하지 않지만 transformers의 `AutoTokenizer` 임포트가 torch가 설치돼
  있으면 함께 올린다(약 1.2초·메모리). GPU를 기다리는 동안 GIL을 놓아 이벤트 루프가 막히지 않는다.
- gundam 모드는 torch처럼 크롭 전부(최대 32타일)를 SAM 배치 하나로 인코딩한다 — 아주 긴 이미지는
  메모리가 작은 Mac에서 순간적으로 튈 수 있다.
- 메모리: 파라미터 bf16 6.67 GB(8비트 3.92 GB), 8쪽 청크 피크 약 8.3 GB(8비트 5.5 GB). 생성 중
  256토큰마다·실행 뒤 캐시를 비워 활성 메모리가 파라미터 크기로 돌아온다. 프로세스 phys_footprint는
  잡 사이 약 7,300 MB이고 최고치(약 11,600 MB)는 위 gundam 크롭 경로에서 나온다 — 로드 직후 워밍업도
  이 경로를 돈다. 잡 단위 ObjC 풀은 쓰지 않는다(§4 — 실측상 누적 없음). 속도 실측은
  OCR_BENCHMARK.md(M4 Max, 조용한 머신, 2026-10-02: bf16 8쪽 청크 3.8 s/쪽, 25쪽 잡 3.85 s/쪽).
- `mlx==0.32.3` 고정. `mx.fast.rope`·`scaled_dot_product_attention`·`gather_mm`/`gather_qmm`·
  `nn.quantize`에 기대므로, 올릴 때는 패리티 테스트와 `make test-mlx-real`을 다시 돌린다.

### Metal(MPS) 구현 노트

- 사용자 노출 디바이스명은 `metal`, torch 디바이스는 `mps`
  (`engine/unlimited.py`의 `torch_device_name()`이 매핑, health에는 `metal`로 표기)
- torch 2.10 MPS는 **macOS 14.0+**가 필요하다 — 쓸 수 없으면 설치된 torch·macOS 버전을 담은
  안내로 실패한다. dtype `auto`: bf16 텐서 할당 프로브 성공 시 bfloat16, 실패(사실상 도달하지
  않는 프로브 실패) 시 float32 폴백. `OCR_DTYPE=float16`도 선택 가능(bf16 권장)
- 벤더 코드는 P1/P4 패치(`_autocast_ctx`, 파라미터 디바이스 추종)로 디바이스 중립.
  단, torch 2.10.0 MPS의 조용한 버그 2건을 회피하는 패치가 추가로 필요했다 (PROVENANCE.md):
  - **P11**: 브로드캐스트 마스크 `masked_scatter_` 오동작 → 이미지 임베딩 미주입 → 빈 출력
  - **P12**: `torch.autocast("mps", bf16)`가 로짓 오염 → 반복 루프. MPS에서는 autocast 미사용
    (가중치가 bf16이라 성능 동일)
- **디코드 성능**: MPS 디코드는 연산량이 아니라 커널 디스패치/호스트 동기화에 바운드된다
  (감사 프로파일: 동기화 대기가 CPU 시간의 69%). 기본 경로:
  - P17 융합 MoE(**MPS 기본**, 로드 시 프리빌드) — 레이어당 GPU→CPU 동기화 0회. `OCR_MOE_FUSED=0`이면
    P18(MoE 단일 토큰 패스트패스 — 레이어당 동기화 1회가 남아 1.20배)로 돌아간다.
  - P20 eager 링 슬롯은 호스트 int + 슬라이스 `copy_`(MPS의 `index_copy_`는 KV 길이에 비례해 느렸다).
  - no-repeat-ngram 프로세서는 MPS에서 **고정 길이 창**(`static_shape=True` — 창이 차기 전에도
    음수 센티널로 왼쪽을 채운다)이라 길이마다 새 MPSGraph를 컴파일하지 않는다(실측: 새 길이 200개에
    RSS +584 → +0.8 MB, 새 프롬프트 길이 묶음의 첫 실행 비용 +3–12초 → +0.1–0.3초 — 그래서 로드 시
    워밍업 생성은 두지 않는다).
  - P19(rotary 스텝 캐시)와 `fast_decode.py`의 명시적 `position_ids`.
  - 측정(M4 Max, 조용한 머신, 2026-10-02 — 개선 전은 a9a4400 감사 값): 1쪽 캡 384토큰 50.6 →
    99–101 tok/s, 8쪽 청크 무제한 34.0 → 9.0–9.1 s/쪽(119 tok/s), `OCR_MOE_FUSED=0`은 16.4 s/쪽
    (65 tok/s), 전부 토큰 동일(OCR_BENCHMARK.md). 융합 경로는 P18 대비 토큰 동일
    실측이지만 비트 동일을 보장하지는 않는다 — `OCR_MOE_FUSED=0`이 P18을 정확히 복원한다.
  - `OCR_SDPA=1`(P16)은 옵트인: M4 Max에서 붕괴 없이 15–28% 빨랐지만 출력이 비트 동일하지 않다.
  - 청크 크기에 따른 차이는 작다(캡 기준 1쪽 약 100, 8쪽 약 96 tok/s; 25쪽 잡은 4쪽 청크 234.6 s ·
    8쪽 231.8 s). 2026-10-02 재측정에서 4·12쪽 청크가 같은 품질로 10% 넘게 빠르지 않아(MLX도 최대
    3.4%) `PAGES_PER_CHUNK` 기본 8을 유지한다 — 디바이스별 기본값도 두지 않는다(OCR_BENCHMARK.md).
- `PYTORCH_ENABLE_MPS_FALLBACK=1`을 macOS에서 torch 엔진 모듈(`engine/unlimited.py`) 임포트 때
  `setdefault` — 미구현 op는 CPU 폴백 (안전망). torch는 이 값을 첫 `import torch` 때만 읽으므로
  auto 판정(registry의 torch 조회 — `unlimited_mlx`가 이 모듈을 먼저 임포트한다)보다 먼저 둔다.
  그보다 먼저 torch를 올린 호출자에서는 적용되지 않아 metal 엔진 생성 때 경고한다. 운영자가 둔
  값(0 포함)은 덮어쓰지 않는다
- **메모리**: 청크(infer 호출) 종료마다 `torch.mps.empty_cache()`로 유니파이드 메모리를 반환한다.
  그와 별개로 CPU/ObjC 힙 — MPS가 autorelease로 넘기는 임시 객체 — 은 끝나지 않는 워커 스레드에서
  회수 지점이 없어 생성 토큰당 약 25 KB씩 쌓였다. 디코드 스텝·생성 구간·모델 로드·재시도 전 캐시
  반환을 각각 ObjC 오토릴리스 풀(`engine/objc_pool.py`)로 감싸고, 워커가 잡마다 한 번 더 감싼다
  (§4). 실측: 웜 384토큰 실행당 RSS +9.5–12.9 → +0.0–3.5 MB, 첫 실행 +1.4 → +0.29 GB. 로드 직후
  드라이버 메모리 약 7.1 GB, 첫 실행 피크 약 12.0 GB. 웜 실행 뒤
  `torch.mps.driver_allocated_memory`에 보이는 +1.6 GB는 Metal 내부의 회수 가능 캐시라
  phys_footprint에 잡히지 않고 늘지 않는다.
- 메모리 상한 조정이 필요하면 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (torch 문서 참조 — 기본값 권장)
- `gpu_name`은 `sysctl machdep.cpu.brand_string` (예: "Apple M4 Max")

## 7. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OCR_DEVICE` | `auto` | `auto`\|`cpu`\|`cuda`\|`metal`\|`mlx` (`mps`는 `metal`의 별칭). 미설정·빈 값 = `auto` — unlimited 엔진이 mlx → cuda → metal → cpu 중 처음 쓸 수 있는 것(§6). `Settings()` 직접 생성 기본은 `cpu`. compose는 서비스마다 고정 |
| `OCR_DTYPE` | `auto` | `auto`(cuda·mlx→bf16, metal→bf16 또는 fp32 폴백, cpu→fp32)\|`bfloat16`\|`float16`\|`float32` |
| `OCR_MLX_QUANT_BITS` | `0` | (MLX 엔진) `0`=dtype 그대로 \| `8`=디코더만 인메모리 8비트(group 64 — health dtype `bfloat16+q8`). 그 밖의 값은 기동 실패(4비트는 숫자 오인식으로 미지원). 컨테이너에는 Metal이 없어 compose에 전달하지 않는다 |
| `OCR_ENGINE` | `unlimited` | `unlimited`\|`fake`\|`textlayer`(§16)\|`ovisocr2`\|`paddleocr_vl` (sidecar 둘은 `OCR_SIDECAR_URL` 필수) |
| `OCR_SIDECAR_URL` | (없음) | sidecar 엔진의 base URL — compose 프로필(ovis/paddle)이 자동 설정 |
| `OCR_SIDECAR_CONNECT_TIMEOUT_S` | `10` | sidecar 연결 타임아웃(초) — 유한한 양수 |
| `OCR_SIDECAR_READ_TIMEOUT_S` | `600` | 페이지 1장 추론 대기 상한(초) — 유한한 양수 |
| `OCR_SIDECAR_HEALTH_TIMEOUT_S` | `5` | sidecar health 대기(초) — 유한한 양수 |
| `OCR_SIDECAR_MAX_RESPONSE_MB` | `20` | 응답 크기 상한 (response bomb 방어) |
| `OCR_SIDECAR_RETRIES` | `1` | 연결 수립 실패 재시도 횟수 (그 외 재시도는 runner 몫) |
| `OCR_REMOTE_PAGE_CONCURRENCY` | `1` | sidecar 페이지 동시 요청 수 = sidecar 엔진의 청크 크기 (16GB 단일 GPU는 1 권장) |
| `OCR_SIDECAR_MODEL_WAIT_S` | `900` | 잡이 sidecar 모델 준비를 기다리는 상한(초) — 최초 기동 창에 업로드해도 실패 대신 대기(취소 가능). 음수는 0으로 보정 |
| `MODEL_ID` | `baidu/Unlimited-OCR` | HF 모델 ID |
| `MODEL_REVISION` | `ee63731b…` | HF revision 고정 (README의 검증 커밋) |
| `PRELOAD_MODEL` | `1` | 기동 시 모델 로드 (0이면 첫 잡에서 lazy) |
| `DATA_DIR` | `data` (Docker `/data`) | 잡 저장소 루트 (`{DATA_DIR}/jobs`). config.py 기본은 상대 경로 `data`, Dockerfile ENV가 `/data`로 덮는다. 백엔드 프로세스 하나만 소유한다 — `{DATA_DIR}/jobs/.owner.lock` (§4 단일 소유자 락) |
| `HF_HOME` | `/data/hf` | HF 캐시 (Dockerfile ENV + compose 볼륨) |
| `RENDER_DPI` | `200` | 요청별 `dpi`로 오버라이드 가능. 72–400 (요청별 `dpi` 검증과 같은 범위) |
| `PAGES_PER_CHUNK` | `8` | infer_multi 청크 크기 (1 이상) |
| `MAX_PAGES` | `200` | 페이지 상한 (1 이상) |
| `MAX_UPLOAD_MB` | `100` | 업로드 상한 (1 이상) |
| `MAX_LENGTH` | `32768` | 생성 총 길이 상한 (1 이상) |
| `MAX_PAGE_OUTPUT_CHARS` | `16384` | `<PAGE>` 기준 페이지별 출력 **내용** 문자 hard limit — 레이아웃 태그(`<\|ref\|>`/`<\|det\|>` 블록)·HTML 표 태그는 세지 않는다(태그가 델타 사이에 걸치면 닫힐 때까지 최대 512자 보류). single에도 동일 적용, unlimited 엔진 전용, 0 이하=비활성 |
| `MAX_PAGE_OUTPUT_TOKENS` | `6144` | 페이지별 생성 토큰 hard limit (fast decode는 최대 한 block만큼 정지 지연, 0 이하=비활성). 마크업만 반복하는 폭주는 이 상한이 멈춘다 |
| `PAGE_SEPARATOR` | `\n\n---\n\n` | 병합 시 페이지 구분자. 백슬래시 이스케이프(`\n`·`\t`·`\uXXXX`)를 해석하고 한글 등 비ASCII는 그대로 보존 |
| `OCR_CPU_THREADS` | `0` | CPU 백엔드 torch 스레드 수 (0=torch 기본) |
| `OCR_FAST_DECODE` | `1` | 커스텀 그리디 디코드 루프(cpu/cuda/mps 공용, 호스트 동기화 블록 배칭). `0`이면 HF generate 폴백 |
| `OCR_DECODE_BLOCK` | `8` | fast decode의 동기화 배칭 크기(토큰, 1 이상) — EOS를 블록 경계에서 확인 |
| `OCR_MOE_FAST` | (미설정) | MoE 단일 토큰 디코드 패스트패스(벤더 P18) 강제 on/off (`1`/`0`). 미설정 시 MPS에서만 on — 지금은 P17이 디코드를 받지 않을 때(`OCR_MOE_FUSED=0`)의 MPS 폴백, 결과 비트 동일 |
| `OCR_MOE_FUSED` | (미설정) | MoE 융합 디코드(벤더 P17, 기본 **CUDA·MPS에서 on**, 로드 시 프리빌드) 킬스위치 — `0`이면 프리빌드까지 건너뛰어 legacy 경로 완전 복원(MPS는 P18) |
| `OCR_SDPA` | (미설정) | 디코더 어텐션 SDPA 융합 커널(벤더 P16) 강제 on/off. 미설정 시 CUDA에서만 on. MPS는 옵트인 — M4 Max에서 15–28% 빨랐지만 출력이 비트 동일하지 않다 |
| `OCR_NGRAM_HOST` | (미설정) | `1`이면 GPU/MPS에서도 no-repeat-ngram 배닝을 호스트(C++/파이썬) 티어로 강제 (절연 레버, `native_ops.py`) |
| `FAKE_DELAY` | `0.02` | FakeEngine 페이지당 지연(초, 0 이상) — 테스트/데모 전용 |
| `FRONTEND_DIR` | (미설정) | 정적 프론트엔드 경로 오버라이드 — 미설정이면 리포 상대 경로에서 탐색 |
| `OPENAI_BASE_URL` | (없음) | 번역 프로바이더 base URL. bare origin(`https://host`)이면 `/v1`을 자동 보완하고, 명시 경로는 그대로 사용. 미설정 시 번역 기능만 비활성(503) |
| `OPENAI_API_KEY` | (없음) | **번역 전용** 프로바이더 API 키 (로컬 서버는 생략 가능). Q&A는 이 키를 쓰지 않는다 → `LLM_OPENAI_API_KEY` |
| `OPENAI_MODEL` | (없음) | 번역 모델 ID. `OPENAI_BASE_URL`과 함께 있어야 번역 활성화 |
| `TRANSLATE_MODEL` | `OPENAI_MODEL` | 번역 전용 모델 오버라이드 |
| `TRANSLATE_API_MODE` | `auto` | `auto`\|`chat`\|`responses` (auto: responses 시도 → 미지원 시 chat) |
| `TRANSLATE_CONCURRENCY` | `8` | 잡당 동시 번역 요청 수 (1–8) |
| `TRANSLATE_GLOBAL_CONCURRENCY` | `TRANSLATE_CONCURRENCY` | 여러 잡을 합친 프로세스 전체 실제 번역 HTTP 상한 (1–8) |
| `TRANSLATE_TIMEOUT_S` | `180` | 응답 읽기 타임아웃(초). 연결은 `min(10초, 이 값)`으로 별도 제한. 스트리밍(chat 기본)에서는 **토큰 사이 정지 시간** 상한, 비스트리밍(responses·`TRANSLATE_STREAM=0`)에서는 대기열+생성 전체 시간 상한. 읽기 타임아웃은 1번만 다시 시도하고, 그래도 넘으면 그 유닛을 분할 래더로 넘긴다(끝내 실패하면 `timeout` 사유로 원문을 둔다) |
| `TRANSLATE_MAX_RETRIES` | `3` | 연결 오류와 408/429/500/502/503/504 재시도 횟수 |
| `TRANSLATE_TEMPERATURE` | `0` | `none`이면 temperature 파라미터 자체 생략 |
| `TRANSLATE_MAX_TOKENS_PARAM` | `max_tokens` | `max_tokens`\|`max_completion_tokens`\|`none` — `none`은 잘림 2배 재시도를 끄고 경고를 남긴다(mlx_lm은 `max_tokens`가 없으면 512토큰에서 자른다) |
| `TRANSLATE_STREAM` | `auto` | `auto`\|`1`\|`0` — auto는 chat 모드에서 SSE 스트리밍. 서버가 첫 스트리밍 요청을 400/415/422로 거부하면 비스트리밍으로 바꿔 고정한다. 스트리밍 중 취소는 소켓을 끊어 서버 생성까지 멈춘다 |
| `TRANSLATE_MAX_RESPONSE_MB` | `32` | 번역 응답 본문 상한(MB, 1–1024) — 선언된 Content-Length와 실제로 읽은 바이트(비스트리밍 본문·SSE 스트림 모두)를 센다. 64KB 조각마다 점진 적용(압축은 풀린 크기); 잘림 뒤 2배 재시도의 상한 초과는 잡 오류가 아니라 유닛 단위 잘림 |
| `TRANSLATE_REASONING_STYLE` | `auto` | `TRANSLATE_REASONING`을 보낼 필드: `auto`\|`openrouter`\|`chat_template_kwargs`\|`reasoning_effort`\|`none`. auto는 base URL로 고른다 — 루프백·사설 IP·단일 라벨·`localhost`·`*.localhost`·`*.local`·`*.lan`·`*.home.arpa`·`*.internal`(`host.docker.internal`·`host.containers.internal` 포함) → `chat_template_kwargs`(`enable_thinking`), `openrouter.ai` → `openrouter`(`reasoning.enabled/effort`), `api.openai.com`·`*.api.openai.com` → `reasoning_effort`(off=`none`), 그 밖 공개 호스트 → `openrouter`(종전) (§13) |
| `TRANSLATE_EXTRA_BODY` | (빈 값) | 모든 번역 요청 본문에 병합할 JSON 객체(4,096자 이하, 객체 값은 한 단계 병합). `model`·`messages`·`input`·`instructions`·`stream`·`stream_options`·`n`·`max_tokens`·`max_completion_tokens`·`max_output_tokens`·`store`는 덮어쓸 수 없다 |
| `OCR_CUDA_GRAPHS` | (CUDA on) | 디코드 스텝 CUDA Graph 캡처·리플레이 — 커널 launch 갭 제거. `0`으로 비활성. 실측(8p): 191s→57s, sm 33%→98% |
| `TRANSLATE_REASONING` | (미전송) | reasoning 모델 제어: `off`\|`low`\|`medium`\|`high`\|`xhigh`(`max` 없음 — Q&A의 `LLM_REASONING_EFFORT`와 다르다). 전달 필드는 `TRANSLATE_REASONING_STYLE`. reasoning 모델은 `off` 권장 — 실측 유닛당 37s→1.7s, 출력 토큰 ~1/40. effort별 요청 max_tokens: 8192/10240/20480/40960/81920 (미설정=8192) |
| `TRANSLATE_CONTEXT` | `1` | 직전 유닛 꼬리를 번역 문맥으로 제공. `0`이면 비활성 |
| `QA_RATE_LIMIT_PER_MIN` | `30` | `POST /qa`의 잡·IP별 60초 윈도우 상한 (0 이하=비활성, §5) |
| `QA_MAX_CONCURRENT` | `4` | 동시 처리 중인 Q&A 요청 수 상한 — 초과 시 429 `Retry-After: 5` |
| `TRANSLATE_RATE_LIMIT_PER_MIN` | `12` | `POST /translate`의 잡·IP별 60초 윈도우 상한 (0 이하=비활성) |
| `TRANSLATE_MAX_ACTIVE` | `4` | 동시에 실행 중인 번역 태스크 수 상한 — 초과 시 429 `Retry-After: 30` |
| `TRUSTED_PROXY_HOPS` | `0` | 앱 앞단의 **신뢰 프록시 홉 수**. `0`=`X-Forwarded-For` 완전 무시(기본, 위조 방어). 리버스 프록시 뒤에 둘 때만 홉 수를 넣는다 — 그렇지 않으면 위 레이트리밋 키가 프록시 IP 하나로 붕괴한다. 정수가 아니면 경고 후 `0` (§5·§14) |
| `TRUSTED_PROXY_IPS` | (빈 값 = 루프백) | `X-Forwarded-For`를 믿을 직접 연결 피어의 IP·CIDR 목록(콤마). `TRUSTED_PROXY_HOPS>0`일 때만 쓰인다. 목록 밖 피어의 헤더는 위조로 보고 피어 IP로 레이트리밋(한 번 경고). IP가 아닌 피어(유닉스 소켓)는 로컬이라 믿는다. Docker: 호스트 프록시가 게시 포트로 붙으면 브리지 게이트웨이(예: `172.17.0.1`) (§5) |
| `PDF_EXPORT_MAX_CONCURRENT` | `2` | 서로 다른 잡의 PDF 빌드·래스터 **프로세스 전역** 동시 실행 상한이자 **export 워커 프로세스 수** — 빌드 N개가 실제로 병렬이다(§18). 예열은 N-1개 슬롯만 쓰므로 `1`이면 예열하지 않는다. `0` 이하=슬롯 상한 비활성(워커는 최대 min(8, CPU)개). 캐시 적중 경로는 슬롯을 잡지 않는다 (§5 `/pdf`) |
| `PDF_EXPORT_QUEUE_TIMEOUT_S` | `30` | 위 슬롯 대기 상한(초) — 초과 시 매달리는 대신 503 + `Retry-After` |
| `OCR_FIDELITY_THRESHOLD` | `0.70` | 페이지 OCR 충실도 게이트 임계값. 원본 PDF 텍스트 레이어와 대조해 이 값 미만인 페이지만 단독 재실행한다. `0` 이하=비활성. 텍스트 레이어가 없으면 자동 건너뜀 (§4 충실도 게이트). sidecar의 잘린 페이지(`truncated`)를 텍스트 레이어로 바꿀지도 이 값으로 정한다 |
| `OCR_FIDELITY_MAX_RETRY_RATIO` | `0.2` | 위 재실행의 상한(문서 페이지 수 대비 비율, 최소 2쪽). 상한에 걸려 건너뛴 페이지는 잡 경고로 남는다 |
| `PDF_EXPORT_WARM_WAIT_S` | `180` | 같은 잡의 **예열 빌드**가 도는 동안 들어온 클릭의 대기 상한(초). 예열이 끝나면 캐시 적중이므로 일반 대기열 상한과 분리한다 (§5 `/pdf`) |
| `PDF_PAGE_TIMEOUT_S` | `60` | 페이지 한 장의 MuPDF 작업(입력 렌더·충실도 분석·텍스트 레이어 추출·facsimile 래스터)과 업로드 검증 한 건의 상한(초). 넘긴 페이지는 흰 페이지(경고)로 대체하거나 그 분석을 건너뛴다. 렌더 초과가 한 잡에서 3번이면 잡 종료. `0` 이하=없음(권장하지 않음) (§18) |
| `PDF_EXPORT_BUILD_TIMEOUT_S` | `900` | 번역·대조 PDF 빌드 한 건의 상한(초). 넘기면 그 다운로드는 409, 빌드 워커만 정리된다. `0` 이하=없음 (§18) |
| `PDF_WORKER_MEM_LIMIT_MB` | `0` | PDF 워커 프로세스당 가상 메모리 상한(MB, `RLIMIT_AS` — Linux만). `0`=끔. 끄더라도 Linux 워커는 `oom_score_adj=1000` (§18) |
| `PDF_MAX_PAGE_CONTENT_MB` | `64` | 업로드 게이트 — 페이지가 그리게 하는 콘텐츠(압축 해제 기준)의 페이지당 상한(MB). 넘으면 400. `0`=검사 끔 (§18) |
| `PDF_MAX_PAGE_XOBJECT_CALLS` | `2000000` | 업로드 게이트 — Form XObject 중첩을 펼친 페이지당 그리기 호출 상한. 넘으면 400. `0`=검사 끔 (§18) |
| `GPU_DEVICE` | `0` | (compose) CUDA_VISIBLE_DEVICES로 전달 — 두 번째 GPU는 `1` |
| `HOST`/`PORT` | `0.0.0.0`/`8000` | 컨테이너 내부 uvicorn 바인드 (Dockerfile CMD 고정값) |
| `BIND_HOST` | `0.0.0.0` | (compose) 호스트 쪽 포트 바인딩 주소 — **기본은 외부 노출**. 루프백 전용으로 되돌리려면 `127.0.0.1` (§8·§14) |
| `ALLOWED_HOSTS` | config.py `localhost,127.0.0.1` / **compose `*`** | Host 헤더 화이트리스트(콤마 구분) — DNS rebinding 방어, 포트는 비교 시 무시. compose는 외부 노출 기본과 정합을 위해 `*`(모든 Host 허용)을 넘긴다 (§14) |
| `OCR_CPU_MEM_LIMIT` / `OCR_CUDA_MEM_LIMIT` / `OCR_WEB_MEM_LIMIT` | `24g` / `16g` / `8g` | (compose) backend 서비스별 메모리 상한 (§8) |
| `OVIS_MEM_LIMIT` / `PADDLE_MEM_LIMIT` | `24g` / `24g` | (compose) sidecar 컨테이너 메모리 상한 |
| `OLLAMA_MEM_LIMIT` | `16g` | (compose.ollama.yaml) ollama 컨테이너 메모리 상한 — 큰 모델이면 .env로 올린다 |
| `OVIS_MAX_UPLOAD_MB` / `PADDLEOCR_MAX_UPLOAD_MB` | `128` / `128` | (compose → sidecar) `/v1/parse` 페이지 이미지 업로드 상한 |
| `OVIS_*` / `PADDLEOCR_*` | (sidecar 문서) | (compose → sidecar) 모델 ID·revision·dtype·VRAM·픽셀 상한·디바이스 — `OVIS_MODEL_ID`·`OVIS_MODEL_REVISION`·`OVIS_DTYPE`·`OVIS_GPU_MEMORY_UTILIZATION`·`OVIS_MAX_MODEL_LEN`·`OVIS_MAX_OUTPUT_TOKENS`·`OVIS_MAX_NUM_SEQS`·`OVIS_MIN_PIXELS`·`OVIS_MAX_PIXELS`·`OVIS_GDN_PREFILL_BACKEND`, `PADDLEOCR_MODEL_ID`·`PADDLEOCR_MODEL_REVISION`·`PADDLEOCR_DEVICE`·`PADDLEOCR_MIN_PIXELS`·`PADDLEOCR_MAX_PIXELS` (OVISOCR2_CUDA_5070TI.md·PADDLEOCR_VL_BLACKWELL_5070TI.md·`.env.example`) |
| `HF_TOKEN` | (빈 값) | (compose) 프라이빗 미러용 Hugging Face 토큰 — 런타임 env로만 전달(빌드 레이어 미포함). PDF 워커는 기동 때 지운다 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (macOS에서 torch 엔진 모듈이 첫 torch 임포트 전에 `setdefault`) | torch MPS 미구현 op의 CPU 폴백 — 안전망(§6) |
| `CUDA_LAUNCH_BLOCKING` | (빈 값) | (compose → ocr-cpu·ocr-cuda) CUDA 디버깅용 동기 실행 — 운영에서는 비워 둔다 |
| `JOB_TTL_DAYS` | `0` | 터미널 잡(done/error/canceled) 자동 GC 보존 일수(0 이상) — `0`=비활성(기본, opt-in). 시작 시 1회 + 6시간 주기 (§15) |
| `OCR_LANGUAGES` | `eng+kor` | (textlayer) Tesseract 언어 조합 — `tesseract -l` 인자 (§16) |
| `NATIVE_TEXT_THRESHOLD` | `120` | (textlayer) 텍스트 레이어를 신뢰할 페이지당 최소 영숫자 수 — 미만이면 Tesseract 폴백 (§16) |
| `LLM_PROVIDER` | `openai-responses` | (Q&A) 기본 LLM 공급자: `openai-responses`\|`openai-chat`\|`ollama`\|`local-openai` (§17). `local-openai`는 `LLM_LOCAL_OPENAI_BASE_URL` 없이는 기동 실패 |
| `LLM_REASONING_EFFORT` | `low` | (Q&A) 기본 reasoning effort: `default`\|`none`\|`minimal`\|`low`\|`medium`\|`high`\|`xhigh`\|`max` |
| `LLM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | (Q&A) 공식 `api.openai.com` 호스트만 허용 — 그 외 값은 기동 시 즉시 실패 (§17.3) |
| `LLM_OPENAI_API_KEY` | (없음) | **(Q&A 전용 OpenAI 키 — 번역용 `OPENAI_API_KEY`와 분리, 폴백 없음.)** 위 base URL이 공식 호스트로 고정돼 있어, 제3자 게이트웨이용 `OPENAI_API_KEY`를 재사용하면 그 키가 `api.openai.com`으로 전송된다. 미설정 시 `openai-*` 공급자는 `available:false`이고 `qa_available:false` (§17.3) |
| `LLM_OPENAI_RESPONSES_MODELS` | `gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol` | (Q&A) Responses 모델 선택지 (csv) |
| `LLM_OPENAI_CHAT_MODELS` | `chat-latest,gpt-5.6-luna,gpt-5.6-terra` | (Q&A) Chat Completions 모델 선택지 (csv) |
| `LLM_OPENAI_RESPONSES_MODEL` | `gpt-5.6-luna` | (Q&A) Responses 기본 모델 |
| `LLM_OPENAI_CHAT_MODEL` | `chat-latest` | (Q&A) Chat Completions 기본 모델 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | (Q&A) 로컬 Ollama 주소 — 루프백·`host.docker.internal`·`ollama`만 허용, 그 외 기동 시 즉시 실패 (§17.3). compose 컨테이너 기본값은 `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` | (Q&A) 기본 로컬 Ollama 모델 (`:cloud`/`remote_host` 모델은 차단) |
| `LLM_LOCAL_OPENAI_BASE_URL` | (빈 값 = 미사용) | (Q&A `local-openai`) 로컬 OpenAI 호환 서버(oMLX·LM Studio·mlx_lm.server) 주소 — `127.0.0.1`·`localhost`·`::1`·`host.docker.internal`의 http(s)만 허용, 그 외 기동 실패 (§17.3) |
| `LLM_LOCAL_OPENAI_MODEL` | (빈 값) | (Q&A `local-openai`) 기본 모델 — 서버 `/v1/models`의 id(mlx_lm.server는 `default_model`). 주소를 두면 필수 |
| `LLM_LOCAL_OPENAI_MODELS` | (빈 값) | (Q&A `local-openai`) 추가로 고를 수 있는 모델(csv) — 그 밖의 모델 요청은 400 |
| `LLM_LOCAL_OPENAI_API_KEY` | (빈 값) | (Q&A `local-openai`) 서버에 키를 걸었을 때만. 이 키만 쓴다 — `OPENAI_API_KEY`·`LLM_OPENAI_API_KEY`로 폴백하지 않는다 |
| `PDF_EXPORT_FONT` | (빈 값) | 번역 PDF 내보내기용 한글 폰트 파일 경로 — 비우면 시스템 폰트 → 내장 CJK 폴백 (§5 /pdf) |

- **기동 시 검증**: 숫자 노브는 `Settings.from_env()`에서 검증한다. 정수·숫자가 아니거나
  위 표의 범위 밖이면(NaN·무한대 포함) 변수명과 값을 담은 `ValueError`로 **기동 시**
  실패한다 — 예전에는 기동은 되고 나중에 모든 업로드·페이지가 실패해 요청 쪽 문제로
  보였다(`RENDER_DPI=600` → dpi 없는 업로드 전부 400, sidecar 타임아웃 `0` → 전 페이지
  실패). 빈 값은 미설정과 같다(compose가 선택 키를 빈 문자열로 넘긴다). 예외: 남용 방어
  4종(`QA_*`·`TRANSLATE_RATE_LIMIT_PER_MIN`·`TRANSLATE_MAX_ACTIVE`)은 오타를 경고 후
  기본값으로 강등한다 — 운영 중 방어 설정 실수가 기동 실패·500이 되면 안 된다.
- **로컬 `.env` 로딩** (`load_dotenv_file`): 로컬(uv) 실행은 `.env`를 읽어 줄 주체가 없어
  `Settings.from_env()`가 직접 읽는다. 실행 cwd → 저장소 루트 순서로 처음 찾은 `.env`
  **하나만** 읽고, 이미 설정된 환경변수(셸·compose 주입)는 덮지 않는다. 저장소 밖
  상위 디렉터리는 보지 않는다(무관한 프로젝트의 키를 채택하지 않게). 파싱은
  python-dotenv로 docker compose와 대부분 같은 규칙이다 — 따옴표 없는 값은 '공백+`#`'부터
  주석, 따옴표 안의 `#`은 값, `export ` 접두사·CRLF·BOM 허용, 중복 키는 마지막 값.
  (차이: python-dotenv는 작은따옴표 안의 `${VAR}`도 치환하고 맨 `$VAR`는 그대로 두며
  `${VAR:?err}` 형식이 없다.) ⚠ `KEY=   # 설명`처럼
  값 없이 주석만 두면 compose처럼 주석이 값이 되므로 `.env.example`은 설명을 별도
  줄에 둔다(어느 줄을 주석 해제해도 유효한 값). 읽은 경로는 INFO로 남기고 값은
  남기지 않는다. `DISABLE_DOTENV=1`이면 자동 탐색을 끈다 — pytest(conftest)와 E2E
  하네스가 개발자의 실키를 프로세스에 주입하지 않게 쓰는 스위치다(운영 노브 아님).
- **모르는 `.env` 키 안내**: `.env`의 키가 `config.KNOWN_ENV_KEYS`(앱·배포·하네스 키
  레지스트리)에 없고 다른 도구의 키(`HF_`·`TORCH_`·`PYTORCH_`·`CUDA_`·`COMPOSE_`·`UV_` 등
  접두, 프록시, `TZ` 등)도 아니면 키마다 프로세스에서 한 번 WARNING으로 남기고(키 이름만 —
  값은 남기지 않는다, 최대 20개) `/api/health`의 `config_warnings`에 싣는다. 안내에는 오타
  후보(difflib, cutoff 0.75)와 별칭이 붙는다 — `REASONING_EFFORT` → 번역은
  `TRANSLATE_REASONING`(off|low|medium|high|xhigh — max 없음), Q&A는
  `LLM_REASONING_EFFORT`(max 허용), `OPENAI_API_BASE` → `OPENAI_BASE_URL`. 다른 도구가 읽는
  키라면 무시해도 된다(안내 문구도 그렇게 말한다). 예전에는 이런 키가 조용히 무시돼
  `REASONING_EFFORT`로 번역 reasoning을 껐다고 믿은 설정이 효과가 없었다. 대상은 위 로컬
  `.env` 로딩이 읽은 파일뿐이다 — 이미지에는 `.env`가 없고(`.dockerignore`) compose도
  `env_file` 없이 `environment:`에 적은 키만 넘기므로, Docker에서는 `config_warnings`가 늘
  비어 있고 그 밖의 키는 경고 없이 버려진다. compose 배포는 README §빠른 시작 (Docker)의
  `docker compose exec -T ocr-cpu python -c '…' < .env`로 호스트 `.env`를 같은 판정
  (`config.unknown_dotenv_key_warnings`)에 돌린다(키 이름만 출력).
- **키 레지스트리 계약**: 코드가 읽는 키, `.env.example`의 키 줄, compose `${…}` 키는 모두
  `config._APP_ENV_KEYS`·`_DEPLOY_ENV_KEYS`·`_HARNESS_ENV_KEYS` 중 하나에 있어야 한다
  (`tests/test_config_env_registry.py`가 양방향 대조). 이 블록은 공백 구분 문자열로 둔다 —
  따옴표 리터럴로 바꾸면 `tests/test_ci_ops_contracts.py`의 스캐너가 '코드가 읽는 키'로 세어
  하네스 키까지 `.env.example`·compose 스레딩을 요구한다. 새 운영 키는 `.env.example`(키 줄 +
  들여쓴 설명 줄 — 인라인 주석 금지)과 compose backend 4개 스레딩, 또는 `SERVICE_SCOPED`·
  `NOT_OPERATOR_KNOBS` 등록(이유와 함께)이 필요하다.
- **테스트·하네스 전용 스위치**(운영 노브 아님 — `.env.example`에 없다): `OCR_MPS_TESTS`
  (`make test-mps`), `OCR_MLX_REAL_TESTS`(`make test-mlx-real`), `E2E_MOCK_PORT`·
  `E2E_BACKEND_PORT`(mock 브라우저 E2E 포트), `E2E_BASE_URL`·`E2E_PDF`·`E2E_TIMEOUT_S`·
  `E2E_VERIFY_MOCK_LLM`(`ui.e2e.mjs`), `MOCK_STREAM_DELAY_S`·`MOCK_STREAM_CHUNK`·`MOCK_FINISH`·
  `MOCK_REASONING_CHARS`·`MOCK_TRANSLATE_RATIO`·`FAULT`(`scripts/mock_llm.py`), `DISABLE_DOTENV`,
  `PDF_WORKER_MODE`(`process`|`inline` — inline은 PyMuPDF 격리·시간 상한을 끄고 기동 WARNING을
  남긴다. conftest가 inline으로 켜고 verify_e2e 하네스는 지운다).

## 8. docker-compose

- `ocr-cpu`: 프로필 없음(기본), 포트 **8000**, `OCR_DEVICE=cpu`
- `ocr-cuda`: 프로필 `cuda`, 포트 **8001**, `OCR_DEVICE=cuda`, `gpus: all`
- `ocr-ovis`+`ovisocr2`: 프로필 `ovis`, backend 포트 **8002**(GPU 미사용) +
  OvisOCR2 sidecar(GPU, 내부 expose 8080만) — `services/ovisocr2/`
- `ocr-paddle`+`paddleocr-vl`: 프로필 `paddle`, backend 포트 **8003** +
  PaddleOCR-VL sidecar(GPU) — `services/paddleocr_vl/`
- **단일 GPU 원칙**: cuda/ovis/paddle 스택은 동시 기동 금지 (16GB VRAM 경쟁).
  sidecar **모델 캐시** 볼륨은 런타임(vLLM/PaddleX)이 달라 스택별로 분리 유지
  (`ovis-hf-cache`, `paddle-hf-cache`, `paddle-x-cache`)
- ⚠ **포트는 기본적으로 `0.0.0.0`에 바인딩된다** — 전 backend 서비스가
  `"${BIND_HOST:-0.0.0.0}:<호스트포트>:8000"`이고 `ALLOWED_HOSTS` 기본도 `*`다.
  즉 **무인증 서비스가 기본값에서 LAN/네트워크에 노출**된다. 신뢰 네트워크
  (VPN/Tailscale, 방화벽 뒤 홈랩)를 전제로 한 기본값이며, 루프백 전용으로 되돌리려면
  `.env`에 `BIND_HOST=127.0.0.1` + `ALLOWED_HOSTS=localhost,127.0.0.1`을 넣는다
  (compose가 컨테이너로 전달한다). 자세한 내용은 §14 · README §보안 · SECURITY.md.
  ⚠ Docker가 게시한 포트는 `ufw`·`firewalld` 같은 호스트 방화벽 규칙을 거치지 않는다
  (Docker가 자체 포워딩 규칙을 그 앞에 넣는다) — '방화벽 뒤'는 네트워크 방화벽, `DOCKER-USER`
  체인 규칙, 또는 밖에서 닿지 않는 `BIND_HOST`(루프백·VPN 인터페이스 주소)를 뜻한다.
  이 기본값(0.0.0.0·`ALLOWED_HOSTS=*`)은 의도된 결정이며 `test_ci_ops_contracts`가 고정한다.
  CSRF 방어도 두지 않는다(인증 없는 서비스와 같은 신뢰 네트워크 전제).
- **공유 볼륨**: `hf-cache`(모델 가중치 ~6.7GB, 최초 1회 다운로드)와
  `ocr-data`(잡 결과)를 **네 backend 서비스가 모두 공유**한다 — 엔진(스택)을 바꿔도
  잡 이력이 남는다. ⚠ 그래서 backend는 **한 번에 하나만** 뜬다: 두 번째 backend는
  잡 저장소 단일 소유자 락(§4)에 막혀 기동을 거부하고(`다른 백엔드가 이미 이 잡
  디렉터리를 사용 중입니다`), `restart: unless-stopped`로 재시작을 반복한다. 스택을
  바꿀 때는 먼저 떠 있는 backend를 `docker compose stop ocr-cpu`처럼 멈춘다.
  과거의 `ocr-ovis-data`/`ocr-paddle-data`는 더 이상 참조되지 않으며,
  그 안의 잡을 살리려면 한 번만 `ocr-data`로 복사한 뒤 볼륨을 지운다(compose 주석에 절차 있음).
- ⚠ **볼륨 명령에는 프로젝트 접두사가 필요하다**: compose 볼륨의 실제 이름은
  `<PROJECT>_ocr-data`처럼 접두사가 붙는다(PROJECT 기본값 = 이 디렉터리 이름,
  `docker compose config --format json`의 `volumes[].name`으로 확인). 접두사를 빠뜨린
  `docker run -v <없는이름>:...`은 **빈 볼륨을 새로 만들고 exit 0**을 내므로 마이그레이션이
  조용한 no-op이 된다(`--mount type=volume`도 동일하게 자동 생성한다 — 실측 확인).
  그래서 위 절차와 Dockerfile의 `chown` 절차는 모두 (1) 이름 확인 →
  (2) `docker volume inspect ... || exit 1` 존재 가드 → (3) 실행(원본은 `--mount ...,readonly`)
  순으로 돌려야 한다. 존재 가드가 유일한 안전장치다. 복사 단계는 **GNU coreutils 이미지**
  (`debian:stable-slim`)로 돌린다 — alpine의 busybox `cp -n`은 대상 디렉터리가 이미 있으면
  트리 전체를 조용히 건너뛰고 exit 0을 낸다(실측: 0바이트 복사).
- `ocr-cpu`는 프로필이 없어 `docker compose up` = CPU 서비스만 기동 (.env 불필요)
- GPU: `docker compose up -d ocr-cuda` — 서비스명을 명시하면 cuda 프로필이 자동 활성화
- **하드닝**: 전 서비스 `security_opt: no-new-privileges:true` + `cap_drop: [ALL]`(비루트라
  쓰는 캡이 없다 — 경계 집합도 비운다). backend 4개는 `deploy.resources.limits.pids: 1024`
  (fork 폭주 차단 — 실측 최대 PID: textlayer 8, unlimited CPU 8스레드 60). CPU 이미지 backend
  (`ocr-cpu`·`ocr-ovis`·`ocr-paddle`)는 `read_only: true` + `tmpfs /tmp`(`rw,noexec,nosuid,nodev,
  size=2g`)라 쓰기는 `/data` 볼륨과 `/tmp`뿐이다(docker diff 실측 — 침해된 프로세스가 컨테이너
  레이어에 페이로드를 남기지 못한다). `/tmp`에는 업로드 스풀·폰트 서브셋·PDF 워커 임시
  디렉터리가 생기므로 `MAX_UPLOAD_MB`를 크게 올리면 size도 함께 올린다(메모리 상한에
  포함된다). `ocr-cuda`와 GPU sidecar는 아직 read-only가 아니다(드라이버 JIT·컴파일 캐시 쓰기를
  GPU 호스트에서 검증한 뒤 적용). 4개 backend 서비스는 `extra_hosts:
  host.docker.internal:host-gateway` — Linux에서도 호스트 Ollama(§17)·로컬 OpenAI 호환 서버에
  접근할 수 있게 한다. `backend/Dockerfile`: digest 고정 베이스 + `apt-get upgrade`, 앱 코드는
  `--chown` 없이 복사해 **root 소유·읽기 전용**(예전에는 실행 사용자가 쓸 수 있었다), 바이트코드
  미리 컴파일, uvicorn `--timeout-graceful-shutdown 5`(열린 SSE가 `docker stop`을 10초 SIGKILL로
  끌고 가지 않게 — 실측 10.3초/exit 137 → 6.4초/exit 0). `compose.ollama.yaml`의 Ollama는
  `0.35.0@digest`로 고정한다(root로 돌며 캡은 건드리지 않았다).
  두 sidecar 이미지도 **비루트(uid 1000)로 실행**한다 — 사용자가 올린 임의 PDF의 렌더
  이미지를 파서에 먹이는 쪽이라 신뢰 경계가 backend보다 바깥이다. PaddleX 캐시는
  `$HOME` 기준이라 마운트 경로가 `/home/app/.paddlex`로 바뀌었고, **이미 root 소유로
  채워진 기존 캐시 볼륨은 한 번 소유권을 바꿔야 한다** — 볼륨 이름을 추측하지 말고 compose로
  실행한다(`cap_drop: [ALL]` 때문에 그 실행에만 캡을 돌려준다):
  `docker compose --profile ovis run --rm --no-deps --user 0 --cap-add CHOWN --cap-add
  DAC_OVERRIDE --entrypoint chown ovisocr2 -R 1000:1000 /data/hf` (paddle은 `--profile paddle …
  paddleocr-vl -R 1000:1000 /data/hf /home/app/.paddlex` — OCR_ENGINE_PROTOCOL.md §운영).
- **로그 로테이션·리소스 상한**: 전 서비스가 YAML 앵커 `x-logging`으로 json-file
  `max-size 10m × max-file 3`(서비스당 최대 30MB)을 쓴다 — `restart: unless-stopped`와
  겹쳐 장기 구동 호스트의 디스크를 조용히 채우던 문제 차단. 메모리 상한은
  `deploy.resources.limits.memory`로 서비스별로 걸고 `.env`로 조정한다(§7).
  CPU는 상한 대신 `OCR_CPU_THREADS`로 조절한다.
- **compose 스레딩**: `.env`는 `.dockerignore`로 이미지·컨테이너 안에 없으므로,
  compose `environment`에 명시하지 않은 키는 컨테이너에서 조용히 무시된다.
  `LLM_OPENAI_API_KEY`·`PAGE_SEPARATOR`는 backend 4개 서비스
  (`ocr-cpu`/`ocr-cuda`/`ocr-ovis`/`ocr-paddle`) **전부**에 있다.
  남용 방어 4종(`QA_RATE_LIMIT_PER_MIN`·`QA_MAX_CONCURRENT`·
  `TRANSLATE_RATE_LIMIT_PER_MIN`·`TRANSLATE_MAX_ACTIVE`)과 `TRUSTED_PROXY_HOPS`·
  `TRUSTED_PROXY_IPS`, 내보내기 전역 상한(`PDF_EXPORT_MAX_CONCURRENT`·
  `PDF_EXPORT_QUEUE_TIMEOUT_S`·`PDF_EXPORT_WARM_WAIT_S`), PDF 워커 상한·업로드 게이트
  (`PDF_PAGE_TIMEOUT_S`·`PDF_EXPORT_BUILD_TIMEOUT_S`·`PDF_WORKER_MEM_LIMIT_MB`·
  `PDF_MAX_PAGE_CONTENT_MB`·`PDF_MAX_PAGE_XOBJECT_CALLS`), 번역 노브(`TRANSLATE_STREAM`·
  `TRANSLATE_MAX_RESPONSE_MB`·`TRANSLATE_REASONING_STYLE`·`TRANSLATE_EXTRA_BODY` 등),
  `LLM_LOCAL_OPENAI_*`도 마찬가지로 4개 전부에 있다 — 소비처(`api.py`·`pipeline/derived.py`·
  `pipeline/pdf_worker.py`·번역·Q&A)가 엔진과 무관하게 모든 backend에서 돈다.
  `OCR_MLX_QUANT_BITS`는 어느 서비스에도 없다 — MLX는 macOS 호스트 전용이다. ⚠ 이 중 하나라도 빠지면 운영자는 `.env`로
  **조였다고 믿는데 컨테이너는 코드 기본값을 쓴다**(유료 LLM 키 소진 방어에서
  가장 위험한 실패 형태). `tests/test_ci_ops_contracts.py`가 "코드가 읽는 env 키는
  `.env.example`에 있고, 엔진 무관 키는 4개 서비스 전부에 있다"를 고정한다.
  `MAX_LENGTH`는 일부러 `ocr-cpu`·`ocr-cuda`에만 둔다 — 소비처가
  `engine/unlimited.py`(로컬 생성 상한) 하나뿐이라 sidecar 스택에서는 아무 효과가
  없다. 반대로 `PAGE_SEPARATOR`는 **엔진과 무관**하다(`merge.py`의 result.md 조립,
  `render.py`의 doc-page 분할, `qa.py`의 페이지 컨텍스트가 전부 이 값으로 split).
  ⚠ 노브를 추가할 때는 "어느 코드가 읽는가"를 먼저 보고 그 코드를 도는 서비스에만
  넣는다 — 전부에 복붙하면 소비되지 않는 값이 문서와 함께 굳어진다.
- **Ollama 컨테이너(선택)는 overlay로만**: `compose.ollama.yaml`은 프로필 없는
  `ollama` 서비스를 추가하고 ocr-cpu의 `OLLAMA_BASE_URL`을 `http://ollama:11434`로
  덮어쓴다. 본 파일에 병합하면 기본 경로(`docker compose up`)에서도 함께 기동돼
  버리므로 반드시 overlay 파일로 유지한다:
  `docker compose -f docker-compose.yml -f compose.ollama.yaml up -d --build ocr-cpu ollama`
  (= `make docker-up-ollama`). Ollama 포트(11434)는 내부 `expose`만 — 호스트 미공개.

## 9. C++ 네이티브 모듈 (`native/`, 모듈명 `uocr_native`)

목적: 토큰 생성 핫패스(no-repeat-ngram 배닝)의 C++ 가속.
**없어도 앱은 동작해야 한다** — `app/native_ops.py`가 임포트 실패 시 순수 파이썬 폴백 사용.

⚠ **앱이 실제로 쓰는 함수는 `banned_ngram_tokens` 하나뿐이다.** `crop_regions`는
`native/tests/test_parity.py` 말고는 호출자가 없다 — 앱의 figure 크롭은 두 개의
서로 다른 **파이썬** 경로로 일어난다: sidecar 스택은 `sidecar/materializer.py`의
PIL `Image.crop`, Unlimited 엔진은 벤더 코드
`vendor/unlimited_ocr/modeling_unlimitedocr.py`의 PIL `crop`. 두 경로 모두
`uocr_native`를 import하지 않는다. 아래 9.2는 **모듈의 계약 명세**이지 앱 가속
경로의 서술이 아니다(잘못 읽으면 "네이티브를 깔면 크롭이 빨라진다"는 오해가 된다).

### 9.1 `banned_ngram_tokens(sequence, ngram_size, window) -> ndarray[int64]`
- 입력: `sequence` 1-D `int64` C-contiguous ndarray (지금까지 생성된 토큰열),
  `ngram_size >= 1`, `window >= 1`
- 의미론 (아래 파이썬 레퍼런스와 **완전 동일**해야 함, 반환은 오름차순 유니크):

```python
def banned_ngram_tokens_ref(sequence: list[int], ngram_size: int, window: int) -> list[int]:
    if len(sequence) < ngram_size:
        return []
    search_start = max(0, len(sequence) - window)
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return []
    current_prefix = tuple(sequence[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for idx in range(search_start, search_end):
        ngram = sequence[idx:idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(ngram[-1])
    return sorted(banned)
```

### 9.2 `crop_regions(image, boxes) -> list[ndarray | None]`
- 입력: `image` HxWx3 `uint8` C-contiguous, `boxes` Nx4 `int64` (x1,y1,x2,y2 — **0–999 정규화**)
- 각 박스에 대해 `x1p=int(x1/999*W)`, `y1p=int(y1/999*H)` … (파이썬 `int()` 절삭과 동일),
  `x2p=min(x2p,W)`, `y2p=min(y2p,H)`, `x1p=max(x1p,0)`, `y1p=max(y1p,0)`
- `x2p<=x1p or y2p<=y1p`면 해당 항목 `None`, 아니면 `(y2p-y1p, x2p-x1p, 3)` uint8 크롭 반환
- 반환 리스트 길이는 항상 N (박스와 1:1)

### 9.3 빌드/테스트
- scikit-build-core + pybind11 + CMake(C++17, `-O3`), Python 3.12
- `native/tests/test_parity.py`: 랜덤 케이스에서 레퍼런스와 완전 일치 검증 (경계: 빈 시퀀스,
  window > len, ngram_size=1, 좌표 0/999, 퇴화 박스)
- **빌드 의존성은 `==`로 정확 고정**한다(`native/pyproject.toml`). `backend/Dockerfile`이
  이미지 빌드 시점에 `uv pip install /src/native`로 PEP 517 격리 빌드를 돌리므로,
  상한 없는 `>=`면 같은 커밋이 날마다 다른 툴체인으로 컴파일된다. 실측(2026-08):
  기존 `scikit-build-core>=0.10`·`pybind11>=2.12`가 **1.0.3·3.1.0**으로 해석됐다
  (둘 다 메이저 2번 건너뜀). 올릴 때는 핀을 고치고 패리티 테스트를 통과시킨 뒤 커밋한다.
- CI에서 이 모듈이 걸리는 곳은 **두 잡**이다: `native`(모듈 자체 패리티) +
  `backend-native`(모듈이 설치된 상태의 backend 스위트, §11.2).

## 10. 프론트엔드 (frontend/, 정적 SPA)

- **외부 네트워크 리소스 0** (CDN/폰트/트래커 금지), 빌드 스텝 없음, 바닐라 JS(ES modules)
- **지원 브라우저**: 최신 Chromium·Firefox, Safari 16.0 이상 — 파싱 단계에서 깨지는 문법(정규식
  lookbehind 등)은 쓰지 않는다(`tests/browser-compat.test.mjs`가 막는다). lookbehind 하나로
  Safari 16.0–16.3에서 화면 전체가 비었다
- ⚠ **`app.js`에는 로직이 없다**: 진입점(372줄 — 부트스트랩 `init()` + 테스트가 쓰는
  공개 심볼 재노출)일 뿐이고, 실제 구현은 `frontend/js/` **17개 모듈(약 8,080줄)**에
  있다. 브라우저 네이티브 ES 모듈이라 번들러가 없으므로 임포트 그래프가 곧 구조다.
  | 모듈 | 역할 |
  |---|---|
  | `constants.js` · `state.js` · `ui.js` | UI 상수(KaTeX 옵션 포함) · 전역 DOM/상태 핸들 · DOM 헬퍼·토스트·테마·안전한 HTML 주입 |
  | `api.js` · `sse.js` | `/api` fetch 래퍼(타임아웃·503 재시도) · SSE 구독/재연결·폴백 폴링 |
  | `core.js` | **순수 함수 코어**(스트림 파싱·진행률·라벨·URL 조립·응답 판정·목록 페이징·노트·리포트 정리) — 노드 단위 테스트 대상 |
  | `upload.js` · `jobs.js` · `health.js` | 드롭존/검증 · 잡 목록(키 기반 렌더·'더 보기')·라우팅·주의/참고 칩 · health 배지 |
  | `live.js` · `results.js` · `tabs.js` | 3-패널 라이브 뷰 · 완료 화면 · 결과 탭 |
  | `translate.js` · `qa.js` | 번역 실행/상태/경고 요약(429 재시도 잠금 포함) · 페이지 Q&A |
  | `viewer.js` · `reader.js` · `notes.js` | 전체화면 뷰어 부트스트랩 · 연속 스크롤 리더(원문↔번역 동기화·PDF 생성 리포트) · 리더 노트 저장소 |
- `theme-init.js`는 `<head>`에서 동기로 도는 테마 부트스트랩이다 — CSP `script-src 'self'`
  아래 인라인 스크립트를 두지 않기 위해 파일로 뺐다.
- 한국어 UI, 다크/라이트 자동(`prefers-color-scheme`) + 수동 토글(localStorage)
- 구성:
  - 헤더: "PDF OCR Translator — PDF → HTML · 한국어", `/api/health` 기반 디바이스/엔진 배지
    (`MLX · M4 Max`처럼 칩 이름 포함). health는 정상이면 30초, 로딩·실패·sidecar 오류·워커
    중지·조회 실패면 10초마다 다시 묻고 숨긴 탭에서는 멈춘다(보이면 즉시). `model_load_error`는
    '모델 로드 실패'(사유는 툴팁), `worker_alive=false`는 '작업 처리기 중지됨' 배지다. 배지는
    자리(slot)마다 바뀐 곳만 갱신한다 — 같은 응답의 재폴링은 aria-live 배지 영역과 업로드 안내
    (role=alert)를 다시 쓰지 않아 스크린리더가 폴링마다 다시 읽지 않고, 상태가 바뀌면 한 번 알린다.
  - 좌측: PDF 드롭존(+파일선택, 확장자/크기 검증) · 옵션(mode, dpi) · 잡 목록. 목록은 최신 50건을
    5초마다 갱신하고(직렬화 — 진행 중이면 건너뜀), 서버가 `has_more`를 주면 '더 보기 (50/132)'로
    `?before=<마지막 id>` 50건씩 이어 받는다. 넓힌 창은 5초 폴에서도 유지된다(500건 넘게는 커서로
    이어 받음), 커서 잡이 지워져 422면 처음부터 다시 받는다. 행은 `job_id`로 키를 두고 바뀐 필드만
    갱신해 포커스·2단계 삭제 상태가 폴링을 넘긴다. 경고가 있는 잡에는 '주의 N' 배지.
  - 메인(활성 잡, **공식 데모 GIF 재현 3-패널 라이브 뷰**):
    1. 원본+레이아웃 — 현재 페이지 이미지 위에 스트림의 `<|det|>label [x1,y1,x2,y2]<|/det|>`
       (0–999 정규화) 좌표로 컬러 박스를 실시간 오버레이, `<PAGE>` 마커로 페이지 자동 전환
    2. RAW OUTPUT — SSE `token` 델타 모노스페이스 append (자동 스크롤, 청크 경계 holdback)
    3. 실시간 미리보기 — 정리된 스트림 텍스트를 600ms debounce로
       `POST /render-preview`(본문 256 KiB 상한)에 보내 렌더된 HTML 표시. 밀린 페이지는 3개씩
       동시에 렌더하고 순서대로 반영한다(실패하면 받은 페이지는 두고 다음 주기에 실패 페이지부터).
       429는 실패로 세지 않고 `Retry-After`(델타 초·HTTP 날짜, 1–300초, 없으면 30초)만큼 쉰다 —
       연속 5번 실패하면 멈추는 규칙은 다른 오류에만 적용된다.
    - 실행 중 STOP(정지) 버튼 → `POST /cancel` (부분 결과 보존) · 진행 바(phase + 페이지 n/N).
      대기 중 잡의 202 `{status:'canceled'}`는 즉시 마감한다(GET 한 번).
    - 완료 시 탭 [읽기] [미리보기] [Markdown] [레이아웃] [감지 박스] [원본 페이지] [질문],
      다운로드 [원본 HTML] [한국어 HTML] [원문·한국어 PDF] [Markdown] [전체 ZIP] · 삭제
    - 헤더의 '주의 N건' 칩(경고 없이 참고만 있으면 흐린 '참고 N건')을 펼치면 `warnings`, 흐린
      참고 목록에 `notices`가 보인다. 'N페이지'·'3–5페이지' 언급은 읽기 뷰 링크다. 전체 화면
      뷰어가 열려 있는 동안 뷰어 밖의 모든 것(번역 참고 사항·PDF 생성 리포트 접이식 목록 포함,
      `#toast` 제외)이 inert·aria-hidden이다 — 뷰어에서 body까지 조상마다의 형제를 고른다.
    - 결과 툴바 아래 흐린 접이식 목록 두 개: '번역 참고 사항 N건'(`/translate/state`의
      `warnings`)과 'PDF 생성 리포트 · 주의 N건'(PDF 다운로드 뒤 `/pdf/report` — 보존 사유를
      한국어로, 예: `listing_line_unaligned` → '원문 줄 위치 정렬 실패(그 줄만 원문)', 주의
      문장은 'N페이지: …', 50건 초과분은 개수 안내). 다운로드 토스트는 '스캔 원문 N개 블록
      지움'까지 요약한다.
  - 미리보기 탭은 `/api/jobs/{id}/html` 응답을 주입 (클라이언트 md 렌더러 불필요)
  - SSE 폴백: 첫 연결이 비-200(프록시 502·404 등)으로 바로 닫히면 즉시, 연결 중 오류는 2번째에
    1초 상태 폴링으로 바꾸고 10초→20초→30초(상한 — 이후 30초마다) 간격으로 SSE 재승격을
    시도한다. **폴백은 상태만 폴링한다** — 라이브 세 패널은 재승격 때 `replay`로 복구된다(부분
    markdown을 주기적으로 읽지 않는다). 번역 SSE는 강등 조건만 같다 — `/translate/state`를
    1.5초 간격으로 폴링하고 재승격하지 않는다(상태가 running인 동안 계속 — 완료·오류·취소를
    보면 멈춘다).
  - 삭제: SSE `error {deleted:true}`를 받으면 그 잡의 화면·뷰어·구독을 닫고 목록 줄·읽던 위치·
    노트를 지운다. 다른 탭·API로 지워졌을 때만 '열려 있던 작업이 삭제되었습니다.' 토스트를 띄운다.
    완료된 잡이 다른 곳에서 지워지면 다음 **전체** 목록 폴(`has_more=false`)에서 사라진 것을
    보고 GET 404로 확인한 뒤 닫는다.
- **한국어 보기 응답 판정**(`core.langFetchVerdict`): 404/409만 '번역 없음'으로 원문 보기로
  돌아간다. 503(+`Retry-After`)은 '… 준비 중… N초 뒤 다시 시도합니다 (k/4)'를 보이며 최대 4번,
  1–60초씩 기다린다(Retry-After가 없으면 5/10/20/30초). 네트워크 오류·그 밖의 5xx는 한국어
  보기를 유지하고 [다시 시도]·[원문 보기]를 띄운다. PDF 다운로드도 503을 최대 4번 기다린다.
  같은 (탭, 잡, 언어) 요청은 진행 중 하나로 합치고, 아티팩트 요청은 시도당 240초, 폴링은 20초
  타임아웃이다(멈춘 요청 하나가 폴링을 막지 않게).
- **안전한 HTML 주입**: 서버 HTML(미리보기·레이아웃·라이브 미리보기 조각·리더 레일)은 inert
  `<template>`에서 파싱한 뒤, 같은 출처·`data:image`·`blob:`이 아닌 이미지 출처(`src`·`srcset`·
  `poster`·`<source>`)를 '[외부 이미지 차단됨: alt]' 자리표시로 바꾸고 넣는다 — 서버 렌더러의
  차단(§14)과 이중 방어다. 레이아웃·미리보기 이미지는 `loading=lazy`·`decoding=async`.
- **수식**: 로컬 KaTeX **0.18.10**(GHSA-238p-pmpm-9mq7은 0.18.2에서 수정). 모든 `katex.render`가
  `constants.katexOptions`(`throwOnError:false, maxSize:10, maxExpand:1000, strict:'ignore',
  trust:false`)를 쓴다 — `\rule{2000em}` 같은 거대 박스와 무한 매크로를 막는다. KaTeX maxSize는
  양수 크기만 묶는다. 모든 조판은 `ui.renderMath`를 거쳐 크기 인자(`\raisebox{-4000em}`·
  `\rule[-3000em]`·`\kern-5000em`·`\\[-900em]` …)를 같은 단위의 ±10em으로 묶고
  (`core.clampTexSizes`), 그래도 조판 결과 style에 100em(`KATEX_MAX_BOX_EM`)을 넘는 길이가 있으면
  (매크로로 만든 크기·`\arraystretch` 등) 원문 TeX 글자로 보인다(`data-math-fallback="oversized"`).
  내려받는 standalone HTML(document·layout)은 ES 모듈을 쓸 수 없어 같은 규칙을 옮긴 클래식
  스크립트 `katex-guard.js`를 KaTeX 번들 뒤에 인라인해 조판한다(`layout._katex_inline_bundle` —
  가드 파일이 없으면 KaTeX를 싣지 않아 원문 LaTeX로 둔다). `tests/katex-guard.test.mjs`가 두
  구현의 크기 묶기·style 상한·옵션을 대조한다.
  `tests/katex-vendor.test.mjs`가 VERSION·번들 일치와 옵션 사용을 고정한다.
- **리더 노트**(`notes.js`): 하이라이트·인용은 localStorage `uocr-reader-notes-<잡 id>`
  (`{v, updated, items}`, 읽을 때마다 검증)에 잡당 200개, 브라우저당 50개 잡(오래 손대지 않은 잡부터
  정리)까지 남는다. 저장소는 같은 잡을 연 다른 탭과 공유한다 — 저장·삭제는 매번 최신 저장
  목록 위에 적용하고(read-modify-write), 다른 탭의 변경은 `storage` 이벤트로 목록과 하이라이트에
  반영한다. 용량 초과(QuotaExceededError)면 가장 오래 갱신하지 않은 다른 잡의 노트부터 지우고
  다시 저장하며, 저장 토스트가 몇 개 문서의 노트를 지웠는지 알린다. 그 밖의 실패(사생활 보호
  모드·저장소 차단)는 '저장됨' 대신 오류 토스트다. 목록은 노트 id로 증분 렌더해, 삭제 뒤 키보드
  포커스가 이웃 줄(목록이 비면 [선택 문장 도구] 요약)로 간다. 레일이나 선택이 든 페이지를 다시
  그리면(언어 전환 등) 선택을 지우고, 노트의 언어는 문장을 고른 레일의 언어다. [선택 문장
  도구] 안 목록에서 페이지 이동·삭제·Markdown 복사·내보내기(`<이름>.notes.md`). 하이라이트는 레일을
  다시 그린 뒤(언어 전환·정렬 도착·새로고침) 공백 무시 검색으로 다시 칠하고, 카드·페이지를 넘는
  선택은 텍스트 노드 조각마다 `<mark>`로 감싼다(DOM 복제·KaTeX 분할 없음). 수식을 가로지른
  하이라이트는 새로고침 뒤 목록에만 남을 수 있다.
- 성능: token append는 rAF 배칭. 리더 레일은 레일이 다시 만들어질 때(잡·언어·페이지 수 변경)만
  전체를 다시 그리고 그 밖에는 아직 안 그린 섹션만 채운다. 레이아웃 탭이 보이게 될 때와
  `document.fonts.ready` 뒤 레이아웃 맞춤을 다시 돌린다.
- 테스트: `tests/*.test.mjs`(node --test — `helpers/fake-dom.mjs`·`reader-setup.mjs`로 jsdom 없이
  `js/*.js` 런타임을 돌린다) + `tests/e2e/`(`ui.e2e.mjs` 실서버, `mock-full-flow.e2e.mjs` hermetic).
  mock E2E는 모든 브라우저 컨텍스트의 잡히지 않은 예외(pageerror·weberror)·콘솔 오류·HTTP ≥400을
  관문으로 삼고, 시나리오가 일부러 주입한 실패(502·503·429와 CSP 탐침 — URL로 맞춤)만 허용한다
  (`freshContext(options, label, allow)`). `tests/browser-compat.test.mjs`는 정규식 lookbehind를
  막는다.

## 11. 테스트 전략

- `backend/tests/` (FakeEngine, torch 불필요 — CI/로컬 빠른 실행):
  - merge 로직(리넘버링/참조 재작성/`<PAGE>` 분리/페이지 경계 계약) 단위 테스트
  - API 플로우: 업로드→상태→SSE→markdown/html/zip→삭제 (httpx + TestClient),
    번역·Q&A 라우트, 레이트리밋, 잡 GC
  - pdf.py 렌더 테스트(생성 PDF), render.py img src 재작성,
    `test_pdf_export*.py`(레이아웃 보존·시각 안전성), translate/sidecar/llm 계약
  - 테스트 격리(`conftest.py`): `DISABLE_DOTENV=1`(개발자 `.env`의 실키를 읽지 않는다)과 세션
    임시 `DATA_DIR`을 앱 임포트 전에 강제하고, `PDF_WORKER_MODE=inline`으로 PyMuPDF 작업을
    호출 스레드에서 돌린다(많은 테스트가 pymupdf 내부를 monkeypatch한다). 격리 자체는
    `pdf_worker_processes` 픽스처(그 테스트만 process 모드, 앞뒤로 풀 종료)와
    `test_pdf_worker.py`·`test_pdf_complexity_gate.py`·`test_pdf_isolation.py`가 검증한다.
    테스트는 `create_app()`을 인자 없이 부르지 않는다(`Settings(data_dir=tmp_path/…)`).
  - MLX 테스트(`test_mlx_*.py`)는 mlx가 없으면 건너뛰고, 전처리·후처리 패리티는 Linux CI에서도
    돈다. 문서·계약 드리프트 검사: `test_ci_ops_contracts.py`(env 키·compose·Dockerfile·워크플로·
    SECURITY.md), `test_config_env_registry.py`(키 레지스트리), `test_dependency_floor.py`
    (보안 하한·레거시 `import fitz` 금지), `test_security_headers.py`(CSP 두 층 동기화).
- **기기 의존 opt-in 테스트**(기본 스위트에서는 건너뛴다 — `-rs`로 사유가 보인다):
  - `make test-mps` = `OCR_MPS_TESTS=1` `tests/test_mps_contract.py`·`test_objc_pool.py` — 실제
    MPS에서 P11·정적 ngram·P18·P17 뷰·ObjC 풀 회수를 확인한다. torch·macOS 업그레이드 전후에
    돌린다(M4 Max 약 4초).
  - `make test-mlx-real` = `OCR_MLX_REAL_TESTS=1` `tests/test_mlx_model_parity.py` — 실가중치
    bf16 그리디 첫 64토큰 고정과 torch CPU fp32 로짓 비교. 고정 스냅샷이 로컬 HF 캐시
    (`HF_HOME`, 기본 `~/.cache/huggingface`)에 있어야 한다(`local_files_only` — `make dev`를 한
    번 띄우면 받아진다). 테스트는 `.env`를 읽지 않으므로(`DISABLE_DOTENV=1`) 앞단
    `scripts/require_hf_snapshot.py`가 `make dev`와 같은 `.env`(cwd → 저장소 루트)에서 HF 캐시 위치
    키(`HF_HOME`·`HF_HUB_CACHE`·`HUGGINGFACE_HUB_CACHE`·`XDG_CACHE_HOME`, 셸 값 우선)만 넘기고,
    코드 기본 스냅샷의 가중치(`*.safetensors`)가 없으면 pytest를 돌리지 않고 종료코드 2로 멈춘다.
    mlx 업그레이드·MLX 포팅 수정·스냅샷 갱신 전후에 돌린다. fp32 비교는 MLX 로짓을 먼저 구하고
    MLX 모델·버퍼 캐시를 내린 뒤 torch 모델을 올려, 파일 전체 최고 메모리가 약 23GiB다(M4 Max
    `/usr/bin/time -l` 실측 — 두 fp32 사본을 함께 들던 예전에는 44GiB, 16–36GB Mac에서 스왑·종료).
    32GB 이상 Mac에서 돌린다(약 15초).
  - 두 타깃은 `uv run` 대신 `backend/.venv/bin/python`을 직접 부른다(extra·C++ 모듈이 기본
    동기화 밖이라).
- `services/{ovisocr2,paddleocr_vl}/tests/`: sidecar 파서·어댑터·수명주기(로드 재시도·엔진 사망
  재시작) (stdlib만 — 모델·CUDA 불필요) + HTTP 계층(`test_api.py` — CI는 서비스
  `requirements.lock`을 제약(`-c`)으로 걸어 웹 계층을 배포와 같은 버전·해시로 설치)
- `native/tests/`: C++ ↔ 파이썬 레퍼런스 패리티
- `frontend/tests/`: `node --test`(replay·reader-scroll·busy-retry·job-list·pdf-report·
  translate-warnings 등 — `helpers/fake-dom.mjs`로 jsdom 없이 런타임 검증) + `tests/e2e/`
  (`ui.e2e.mjs` = 실서버 대상, `mock-full-flow.e2e.mjs` = hermetic playwright — 포트는
  `E2E_MOCK_PORT`·`E2E_BACKEND_PORT`로 고정 가능. CSP 아래 테마·외부 이미지 차단·내려받은
  한국어 HTML을 디스크에서 열어 meta CSP 확인·'더 보기'·번역 참고 사항·PDF 리포트까지 본다)
- 실모델 E2E: `scripts/smoke_e2e.sh` — compose 기동 후 샘플 PDF 변환, figure 파일 존재 검증
  (`capabilities.figures=true`인데 figure가 하나도 없으면 실패). `scripts/smoke_image.sh IMAGE
  [EXPECT_VERSION]`은 compose와 같은 하드닝으로 이미지를 띄워 health·네이티브·버전·비루트·
  root 소유 코드를 본 뒤 smoke_e2e를 돌린다(CI `docker-image`·release 공용). Paddle GPU smoke는
  입력 텍스트 레이어에 한글이 있는데(또는 `--expect-korean`) 출력에 없으면 실패한다.

### 11.1 전 구간 검증 하네스 (`make verify-e2e`)

`scripts/verify_e2e.py`는 smoke가 끝나는 지점(업로드→OCR→markdown/zip) **이후**를
실제 서버를 띄워 검증한다. 외부 API·GPU·모델 다운로드가 필요 없다:

- `OCR_ENGINE=textlayer`로 backend를 띄우고, 번역 프로바이더로는 같은 하네스가 띄운
  `scripts/mock_llm.py`(OpenAI 호환 목 서버)를 가리킨다. 목 서버는 마스킹
  플레이스홀더를 보존한 채 결정적으로 "번역"하므로 복원·layout 정렬·PDF 조판까지
  실제 경로가 전부 돈다.
  `?fault=refusal|refusal_ko|echo|summary|drop_placeholder|paired_tags|http400|http429`
  쿼리(또는 `FAULT` 환경변수)로 **결함 주입**도 한다. 쿼리 경로는
  `OPENAI_BASE_URL=…/v1?fault=echo` 형태로 쓴다 — `client._endpoint_url()`이 base의
  query를 보존하므로 실제로 도달한다. 하네스는 `drop_placeholder`를 **쿼리 경로로**
  주입해 두 방식이 모두 살아 있음을 매 실행 증명한다(문서만의 주장이 되지 않게).
  쿼리 전용 노브 `?transfer=chunked`는 vLLM·llama.cpp·LM Studio처럼 HTTP/1.1 keep-alive
  `Transfer-Encoding: chunked` SSE로 답한다(기본은 mlx_lm.server처럼 길이 없는
  `Connection: close`). `/__stats`의 `connections`가 POST를 보낸 서로 다른 TCP 연결 수를 센다(연결 재사용 확인).
  ⚠ 목의 플레이스홀더 정규식 접두 집합은 `masking.py`와 반드시 같아야 한다
  (`[mkgucft]`). 예전에는 `<m…>`(수식)만 봐서 수식이 없는 문서에서
  `drop_placeholder`가 완전 no-op이었다(실측: 논문 6페이지 = c 23·f 5·u 4·m 0).
- 단계: 서버 기동/health → 업로드·OCR(`verify_ocr`) → 번역(`verify_translation`) →
  PDF 내보내기(`verify_pdf_export`) → CropBox(`verify_cropbox`) → 뷰어 계약
  (`verify_viewer`) → 보안(`verify_security` — `/files` 경로 탈출 등) →
  번역 결함 주입(`verify_translation_faults`) → 워커 복원력(`verify_worker_resilience`).
  `check()` 단언 89개를 세고 마지막에 `통과 N / 실패 M`을 출력한다(실패가 있으면 종료코드 1 —
  `--pages 4`와 25쪽 전체가 같은 89개). M4 Max 실측: `--pages 4` 약 2분(그중 약 61초가 429
  백오프 실증이다), 25쪽 전체 약 6분.
- **D-1(같은 유닛 이중 번역)**: 목이 센 원문별 LLM 호출 수를 하네스가 `result.md`·`layout.json`·
  `layout.{lang}.json`·`report.json`으로 **다시 세운 번역 계획**과 대조한다 — layout이 덮어 지연된
  md 유닛은 1차에서 호출되지 않아야 하고, 2차 패스 유닛 수는 엔진 로그와 같아야 하며, 원문별
  호출 수는 그 원문을 가진 유닛 수 이하여야 한다. 같은 문장이 다른 자리에 나오는 것은 정상
  이다(25쪽에 21종) — 예전의 '중복 원문 0건' 단언은 25쪽 전체에서 늘 실패했다.
- 옵션: `--pdf PATH`(기본 `sample/2504.19874v1.pdf`) · `--pages N`(앞 N페이지만) ·
  `--skip faults,worker,cropbox` · `--work DIR`(기본 `tmp/verify-e2e`) ·
  `--mock-port`·`--api-port`(기본 0 = 빈 포트 자동, 둘은 달라야 한다).
  포트는 매 실행마다 빈 포트를 잡아 개발 서버(8000)와 충돌하지 않는다.
- **상속 환경 정리**: 자식 프로세스 환경은 `scrubbed_env()` 위에 만든다 — `.env.example`에
  문서화된 앱 노브 전부, `DATA_DIR`·`FRONTEND_DIR`·`FAKE_DELAY`·`DISABLE_DOTENV`·
  `PDF_WORKER_MODE`, 자격증명처럼 보이는 이름(`…KEY`·`…TOKEN`·`…SECRET`·`PASSWORD` 등), LLM
  공급자 접두, `HTTP(S)/ALL/NO_PROXY`를 지우고 `LLM_PROVIDER=openai-responses`·
  `OLLAMA_BASE_URL=http://127.0.0.1:9`를 고정한다. 셸에 실키·프록시가 있어도 외부 호출이
  없고, `PDF_WORKER_MODE=inline`이 새어 들어와 격리가 꺼지지 않는다. 기동 중 실패하면 이미
  띄운 자식 프로세스도 정리한다(고아 목이 다음 실행의 결함 주입을 끄던 문제).
  `make verify-e2e VERIFY_ARGS="--pages 4"` 형태로 인자를 넘긴다.
- **작업 디렉터리는 포트처럼 자동으로 갈라지지 않는다.** `api.log`·`data-main`·
  `fault-*`는 고정 이름이고 각 단계가 시작할 때 `rmtree`한다 — 같은 `--work`로 두
  실행이 겹치면 뒤 실행이 앞 실행의 잡을 지운다. 그래서 `--work`에 배타 락
  (`.harness.lock`, pid 기록)을 걸고, 살아 있는 실행이 잡고 있으면 "다른 `--work`를
  쓰라"는 메시지와 함께 **종료코드 2**로 즉시 멈춘다. 죽은 프로세스가 남긴 락은
  stale로 보고 인수한다.
- 결함 주입 단계는 `refusal`·`refusal_ko`·`echo`·`summary`·`drop_placeholder`·`paired_tags`와
  HTTP 오류 2종(`http400` 결정적 4xx / `http429` 재시도성)을 모두 돌린다. `paired_tags`는 XML
  습관이 있는 소형 모델처럼 자기 닫힘 플레이스홀더를 쌍 태그로 바꾸는 결함이다 — 하네스는
  닫는 태그·플레이스홀더 잔여물이 없고 빈 쌍이 채택되지 않았는지 본다(25쪽 기준 30–60초 추가).
  429는 `Retry-After: 86400`을 함께 보내므로 상한(`_MAX_BACKOFF_S=30`)이 없으면
  워커가 하루 묶인다 — 하네스는 벽시계로 **20초 이상 300초 미만**을 단언해 상·하한을
  동시에 지킨다(하한이 없으면 "재시도를 아예 안 하는" 퇴화도 통과한다).
  `drop_placeholder`는 `kept_original` 기록 여부에 더해 **보호 토큰 수**
  (`masking.mask()`로 원문/번역본을 같은 자로 잰다)와 잔여 `<c1`류 태그를 본다.

### 11.2 CI 잡 구성 (`.github/workflows/ci.yml`)

| 잡 | 내용 |
|---|---|
| `backend` | `uv sync --locked --extra cpu` → Noto CJK 설치 → `pytest --cov`(term+xml), coverage.xml 아티팩트 업로드. **네이티브 미설치 = 순수 파이썬 폴백 경로** |
| `backend-native` | 같은 스위트를 `uv pip install ../native` 후 `.venv/bin/python -m pytest`로 다시 돌린다 — **프로덕션 컨테이너가 실제로 쓰는 조합**. `uocr-native`는 backend 의존성 그래프 밖이라 이 잡이 없으면 `HAVE_NATIVE=True` 경로가 CI에서 0회 실행된다(`test_native_ops.py`의 패리티 검사가 통째로 skip). 설치가 조용히 되돌려지는 것을 막으려고 pytest 앞에 `HAVE_NATIVE` 단언 스텝을 둔다 |
| `lint` | `ruff check . ../services` — sidecar는 backend uv.lock에 고정된 ruff를 재사용 |
| `dependency-audit` | backend `uv.lock`(배포 이미지와 같은 cpu extra)을 `uv export`로 풀어 `pip-audit==2.10.1 --strict`로 검사(torch `+cpu` 같은 로컬 버전 표기는 떼고 감사 — 안 떼면 조용히 빠진다). 수용 권고 9건(torch 2·transformers 7)은 이유 주석과 함께 `--ignore-vuln`, **2027-04-01**이 지나면 잡이 실패한다. 두 sidecar의 `requirements.lock`도 감사하고 `paddlepaddle==3.3.1`은 OSV로 본다. `make audit`(`scripts/dependency_audit.sh`)이 같은 검사를 로컬에서 돌린다(수용 목록·기한·pip-audit 버전이 같은지 계약 테스트가 본다) |
| `frontend` | `node --test` 단위 테스트 |
| `native` | C++ 빌드 + 패리티 pytest (`native/tests` — 모듈 자체의 의미론) |
| `sidecar` | matrix(`ovisocr2`,`paddleocr_vl`) 파서/어댑터·수명주기 pytest + HTTP 계층(서비스 `requirements.lock`을 제약(`-c`)으로 걸어 웹 계층을 배포와 같은 버전으로 설치 — fastapi만 고정하면 starlette가 PyPI 최신으로 풀린다. 엔진 사망 503·폼 필드 상한) |
| `docker-image` | matrix(amd64·arm64) CPU 이미지 빌드(push 없음) → `scripts/smoke_image.sh`(compose와 같은 하드닝 아래 textlayer 전 구간). GHA 캐시는 main(push·nightly·수동)에서만 쓴다 — PR·태그 캐시는 다른 ref에서 복원되지 않아 쿼터만 먹는다 |
| `verify-e2e` | §11.1 하네스를 `--pages 6`으로 실행 (업로드→OCR→번역→PDF→뷰어→보안→결함 주입→워커 복원력). 실패 시 `api.log`·내보낸 PDF 아티팩트 업로드 |
| `e2e-mock` | mock OpenAI + FakeEngine 백엔드 hermetic 브라우저 E2E. 러너 시간이 커서 **PR에서는 돌지 않고** main push·nightly(`schedule: 0 18 * * *`)·`workflow_dispatch`에서 실행, 실패 시 스크린샷 아티팩트 업로드. 릴리스가 태그 커밋의 push CI 성공을 요구하므로 **사실상 릴리스 게이트**다 |

- 액션은 전부 **전체 커밋 SHA**로 고정하고(버전은 주석) checkout은 `persist-credentials: false`다.
  Dependabot(`.github/dependabot.yml`)이 backend uv·액션·베이스 이미지·compose 이미지 갱신을 주 1회
  묶어서 연다 — transformers·torch·torchvision(모델 카드 고정)과 vLLM 베이스, sidecar
  lock(날짜 창·CDN URL로 다시 만들어야 한다)은 받지 않는다.
- **릴리스**(`release.yml`): 태그 커밋의 CI가 끝날 때까지 최대 90분 기다리고, 아키텍처별 이미지를
  빌드·스모크한 뒤 push **전에** `trivy`(0.75.0, digest 고정)로 수정판이 있는 HIGH/CRITICAL을
  막는다(수용 목록 `.github/trivyignore.yaml` — purl로 현재 고정 버전에만, 2027-04-01 만료).
  `release-assets` 잡이 아키텍처별 `.sha256`을 확인하고 `SHA256SUMS`를 써서 오프라인 tarball과
  함께 릴리스에 첨부한다(릴리스가 없으면 초안 생성). GHA 캐시는 main 캐시를 읽기만 한다.
- **`--locked`가 계약**: lock 드리프트(pyproject만 고치고 uv.lock 커밋 누락)를 CI에서
  즉시 실패시킨다. 없으면 uv가 조용히 재잠금해 통과시키고, Dockerfile의 `--frozen`은
  의존성을 빠뜨린 채 빌드해 컨테이너 기동 시 ImportError로 드러난다.
- 커버리지는 게이트가 아니라 관측용이다 — `pytest-cov`는 lock을 건드리지 않도록
  `uv run --with`로 임시 설치한다(`make coverage`도 동일).
- **`backend-native`·`verify-e2e`에서는 `uv run`을 쓰지 않는다**: `uv run`은 환경을
  `uv.lock`에 맞춰 재동기화한다. `uocr-native`는 lock 밖 패키지라 정리 대상이 될지가
  uv 버전·설정에 달려 있어, 의존하지 않고 `uv sync` 뒤 `.venv/bin/python`을 직접
  부른다(하네스가 `_python()`으로 고르는 인터프리터와 같다). 진짜 가드는
  pytest 앞의 `HAVE_NATIVE` 단언 스텝이다 — 설치가 되돌려지면 거기서 붉어진다.

## 12. 로드맵

1. ~~Metal 백엔드~~ — 완료 (§6 참조)
2. ~~sidecar 기반 멀티 엔진(OvisOCR2/PaddleOCR-VL)~~ — 완료 (§1·§8, OvisOCR2 sidecar는
   vLLM 서빙을 사용). **남은 항목**: Unlimited-OCR 자체를 vLLM/SGLang로 서빙하는 옵션
   (모델 repo가 공식 지원 — 대량 처리용)
3. ~~textlayer 엔진(모델 없이 CPU 즉시 동작)~~ — 완료 (§16)
4. ~~한국어 번역 + 레이아웃 보존 PDF 내보내기~~ — 완료 (§13·§5 /pdf)
5. ~~페이지 Q&A(LLM 공급자 레이어)~~ — 완료 (§17)
6. 동시 워커 (GPU 멀티 인스턴스 / 페이지 병렬) — **미완료**. 워커는 여전히
   프로세스당 1개이며(`main.py`가 `Worker`를 하나만 만든다) 잡은 FIFO다.
7. ~~Apple Silicon in-process MLX 엔진~~ — 완료 (§2·§6, `OCR_DEVICE=auto`의 Apple 기본. torch
   MPS는 폴백). 성능 기준선은 조용한 머신에서 순차로 다시 쟀다(2026-10-02, OCR_BENCHMARK.md).
8. ~~PyMuPDF 프로세스 격리 + 업로드 복잡도 게이트~~ — 완료 (§18). **남은 항목**: 큰 내보내기
   빌드를 페이지 범위로 나눠 여러 워커에서 병렬로 만들기, 쉬는 워커 회수, 업로드 검증 전용의
   더 짧은 시간 상한.

## 13. 한국어 번역 (Translation)

OCR로 얻은 **데이터 레이어**(`result.md` + `layout.json`)를 OpenAI 호환 API로 번역해
`result.{lang}.md` / `layout.{lang}.json`을 만들고, 공통 facsimile 페이지 모델로
번역본 미리보기/HTML/PDF를 제공한다. 지원 언어: `ko` (`SUPPORTED_LANGS`).

### 13.1 파이프라인 개요

```
변환 완료(done) 잡 ──► POST /translate ──► 번역 데몬 스레드(잡·lang별, OCR 워커와 별개로 병렬)
  run_translation(job_dir, lang, cfg, *, page_separator, progress, cancel, force)
    1. result.md(+layout.json)를 번역 유닛으로 분해, 마스킹(<m1 .../> 플레이스홀더)
       (layout.json에 텍스트 블록이 없으면 — image 블록뿐 — layout 없이 md만 번역하고
        layout.{lang}.json을 쓰지 않는다. 잡 디렉터리가 사라졌으면 다시 만들지 않고 실패)
    1b. 2단 패스 준비 — layout 블록 번역으로 완전히 덮이는 md 유닛은 1차에서 제외(deferred)
    2. 유닛 캐시(units.json)·용어집(glossary.json) 활용해 API 호출 (Chat/Responses)
       - chat은 기본 SSE 스트리밍(TRANSLATE_STREAM), reasoning 필드는 서버 계열별
         (TRANSLATE_REASONING_STYLE), Responses 요청은 store:false, 응답은 TRANSLATE_MAX_RESPONSE_MB 상한
       - 잡당 최대 8 worker, 프로세스 전역 HTTP 세마포어 기본 8
         (`TRANSLATE_GLOBAL_CONCURRENCY`로 1–8 조정)
       - 같은 cache key는 single-flight로 결과·오류를 공유해 중복 과금/재시도를 차단
       - auto 모드의 Responses→Chat capability probe도 동시 최초 호출끼리는 single-flight.
         초기 협상이 일시 오류로 실패하면 후속 순차 호출은 재협상 가능. owner 유닛 고유의
         거부(400·413·422·빈 출력·잘림·시간 초과)는 대기자에게 복제하지 않고, 대기자는 각자
         자기 요청으로 협상한다(연결·인증·5xx 같은 전역 원인만 공유)
       - `title` 유닛은 의미·정보량을 유지하고 UI 라벨식 축약을 금지한다
       - 출력 측 검증 게이트(_accepted)를 통과한 유닛만 채택·캐시된다 (§13.4)
    3. 플레이스홀더 복원 → **유닛 단위 정렬**: layout 블록 번역으로 완전히 덮이는 md 유닛은
       그 번역을 쓴다(PDF·개요·읽기 텍스트의 제목/용어 SSOT, ref_text는 양쪽 모두 원문 유지)
    3b. 덮개가 깨진 deferred 유닛(그 layout 블록이 원문으로 남는 등) → 2차 번역(total 증가) 후 재조립
    4. result.{lang}.md / layout.{lang}.json 기록
    5. state.json에 running(current/total) → done|error|canceled 기록, report.json 저장
```

#### 2단 패스 (deferred md 유닛)

layout 블록이 md 유닛을 완전히 덮으면 md 유닛 번역은 버려지고 layout 번역이 단일 기준이
된다 — 그 왕복은 낭비다. 그래서 정렬은 **유닛 단위**다(`segment.layout_line_map` +
`map_unit_lines`): md 유닛은 비어 있지 않은 모든 줄이 layout 블록으로 덮이거나 유닛 전체가
layout 블록 하나(여러 줄 블록 포함)와 같을 때만 layout 번역을 쓰고, 아니면 자기 번역을 쓴다.
덮이는 md 유닛은 1차 디스패치에서 빼두고(`deferred`), 덮개 쪽 layout 블록이 실패해 원문으로
남으면(kept) 그 블록은 덮개에서 빠지므로 deferred 쌍둥이가 2차로 번역된다 — 무손실 계약.
쪽번호 같은 보존 블록의 줄은 deferral을 막지 않는다. 예전 줄 단위 reconcile(70% 대응률
문턱)은 부분만 덮인 유닛에 영어 줄을 남겼다. 25쪽 하네스 잡에서 LLM 호출은 810 → 483,
중복 원문은 365 → 39로 줄었다. 2차 진입 시 `total`이 늘어나므로 SSE progress의 `total`은
**증가할 수 있다**(단조 감소는 없음).

- 번역 코어(`app/translate/`)는 **OCR 엔진·torch에 의존하지 않는다**(requests + 표준 라이브러리).
  품질 평가 CLI `tools/translate_eval.py --judge`는 번역과 같은 출력 예산(`cfg.max_output_tokens`)을
  쓰고 채점 실패 사유(`failed_reasons`)를 출력한다.
- layout 블록 번역은 위 유닛 단위 규칙으로만 `result.{lang}.md`에 재사용한다(문서 전체의
  대응률 문턱은 없다). 중복 원문의 번역이 서로 다르면 보수적으로 정렬 대상에서 제외한다.
- API 레이어는 `run_translation`만 안다. 진행률은 `progress(current,total)` 콜백,
  중단은 `threading.Event` cancel로 통신(OCR 워커와 동일 패턴).
- `/html?lang=ko`는 흐름형 읽기 텍스트를 제공한다. `/document.html`,
  `/layout?lang=ko`, `/page/{n}?lang=ko`는 동일한 번역 PDF 페이지를 기준면으로
  사용한다. 정식 standalone 내보내기는 `/document.html` 하나이며, 구버전
  `/layout.html`은 이 경로로 307 리다이렉트한다.
- 읽기 탭은 왼쪽에 전 페이지 `/page/{n}` 자리를 연속으로 쌓고 현재 페이지 ±2만
  이미지/좌표를 hydrate한다. 오른쪽도 전 페이지 레일을 연속으로 유지하며
  `/viewer/pages?start=N&limit=L&include=alignment` 배치 응답(구버전은 단건
  `/alignment?page={n}` 폴백)의 블록 인덱스와 bbox로 양방향 스크롤을 맞춘다.
  따라서 번역 PDF 재조판 결과가 아니라 OCR 원문의 실제 위치를 항상 가리킨다.
  페이지 점프·줌·패널 접기·창 리사이즈 때는 `{page,fraction}` 앵커를 새 높이에
  다시 매핑하고, 잡별 마지막 페이지와 연동 설정을 localStorage에 보존한다.

### 13.2 파일 계약 (`{job_dir}/`)

```
translations/{lang}/state.json     진행 상태 (아래 스키마)
translations/{lang}/glossary.json  문서 용어집 [{"src","ko","policy","first_unit","first_unit_lay"}]
                                   (first_unit = md 순서, first_unit_lay = layout 순서의 첫 등장)
translations/{lang}/glossary.incomplete  용어집 LLM 판정이 실패한 채 저장됐다는 표식 — 다음 실행이 다시 판정
translations/{lang}/units.json     유닛 캐시 {cache_key: 번역문}
translations/{lang}/report.json    품질 리포트 (아래 키 — GET /translate/report가 그대로 노출)
result.{lang}.md                   번역 마크다운 — page_separator 구조·페이지 수 보존
layout.{lang}.json                 blocks[].content만 교체된 layout.json (그 외 필드 동일)
```

`state.json` 스키마(엔진이 기록):
```json
{
  "lang": "ko", "status": "running",   // running|done|error|canceled
  "current": 3, "total": 12,
  "error": null, "model": "gpt-4o-mini", "api_mode": "chat",
  "prompt_v": "6", "context": true,   // prompt_v = types.PROMPT_V 현재값 (§13.4)
  "reasoning_style": "chat_template_kwargs",  // auto를 푼 실제 reasoning 전달 방식
  "request_variant": "",           // 요청 모양이 종전과 다를 때만 값(캐시 키 재료)
  "started_at": "…", "finished_at": null
}
```

`report.json` 스키마(엔진이 `_finish`에서 1회 기록):
```json
{
  "kept_original": ["u12", "…"],      // 원문 유지 유닛 id (§13.4)
  "retried": 0, "repaired": 0, "split": 0, "sanitized": 0, "skipped": 4,
  "skip_reasons":  {"references": 3, "non-linguistic": 1},  // 왜 번역 대상에서 빠졌나
  "kept_reasons":  {"gate-rejected": 1},                    // 왜 원문이 그대로 남았나
  "gate_reasons":  {"refusal": 1, "hangul-ratio": 2},       // 출력 게이트 규칙별 거부 횟수
  "cache_prior": 120, "cache_reused": 0,                  // 전량 재번역 감지 (§15.1)
  "cache_rejected": 0,                // 예전 캐시 중 지금 게이트를 통과하지 못해 다시 번역한 수
  "reference_rule": "…", "cached": 0, "translated": 8,
  "api_mode": "chat", "warnings": ["…"]
}
```
- `skip_reasons`(입력 측 `should_skip`·segment의 references 표시 —
  `references`\|`non-linguistic`\|`already-korean`\|`identifier`)·
  `kept_reasons`(래더 소진 후 원문 유지 —
  `gate-rejected`\|`placeholder-mismatch`\|`empty-output`\|`truncated`\|`timeout`\|`api-rejected`\|
  `degenerate-output`)·
  `gate_reasons`(출력 측 검증 게이트가 거부한 규칙, §13.4 —
  `refusal`\|`scaffold`\|`repetition`\|`label-sentence`\|`number-mismatch`\|`hangul-ratio`\|
  `echo`\|`length-ratio`)는
  **"왜 이 문단이 영어 그대로인가"**의 사유별 집계다.
  총합(`skipped`/`kept_original`)만으로는 원인을 구분할 수 없어서 추가됐다.
  `gate_reasons` 합이 `kept_reasons["gate-rejected"]`보다 **훨씬 크면** 게이트 오탐이
  래더 왕복 비용만 태우고 있다는 신호다(임계값 회귀 감시 지표).
- `cache_prior > 0`인데 `cache_reused == 0`이면 `PROMPT_V`·모델·샘플링 변경으로 캐시가
  전량 무효화돼 유료 API 전량 재호출이 일어난 것이다 — 같은 내용이 `warnings`에도 남는다(§15.1).
- `reference_rule`(`{md_only, layout_only, sample_units}`)은 같은 원문 줄이 result.{lang}.md와
  PDF 중 **한쪽에서만** 참고문헌으로 원문 유지되는 블록 수다. 실제 판정으로 센다 — md 쪽은
  유닛의 실제 건너뜀 사유(heading 스윕 또는 `should_skip` 내용 판정)와, layout 줄에 전부 덮여
  layout 번역·보존을 그대로 받는 md 유닛(deferred)은 갈라질 수 없으므로 빼고 센다. 예전에는
  heading 표시만 봐서 제목 표기(#) 없는 Unlimited-OCR result.md마다 두 산출물이 같아도
  "참고문헌 규칙 불일치" 경고가 났다(실측 25쪽 논문: layout만 유지 66건, 실제 차이 0).

### 13.3 REST / SSE 계약 (번역)

- **POST /api/jobs/{id}/translate** — body `{"lang":"ko","force":false}` (기본 `lang="ko"`).
  - `400` 지원하지 않는 언어 / `409` 변환이 완료된 잡만 번역 가능 /
    `429` 잡·IP 레이트리밋 또는 동시 실행 상한 초과(`Retry-After` 동반 — §5) /
    `503` 프로바이더 미설정(detail=사유, `Retry-After` 없음 — 기다려도 안 된다) /
    `503 + Retry-After: 5` 디스크·스레드 자원 부족으로 시작 실패(재시도 대상 — 실패는 state에도 남긴다)
  - 검사 순서: lang → 잡 상태(409) → 레이트리밋(429) → 프로바이더 구성(503) →
    동시 실행 상한(429, `translate_lock` 안)
  - 이미 실행 중 → `200 {"status":"running"}`; state가 `done`이고 `force` 아님 → `200 {"status":"done"}`
  - 그 외 → 데몬 스레드 시작 후 `202 {"job_id","lang","status":"running"}`
  - 성공 시 `archive.zip` 캐시를 삭제해 다음 `/archive`가 `result.{lang}.md`까지 담아 재생성
- **GET /api/jobs/{id}/translate/state?lang=ko** — `state.json` 없으면 `200 {"status":"none","lang"}`.
  있으면 내용 반환하되 **stale 조정** 적용(§13.5).
  - `report.json`이 있으면 사유별 집계 **`skip_reasons`·`kept_reasons`·`reference_rule`**를
    응답에 덧붙인다(§13.2) — 상태 폴링 한 번으로 "왜 원문이 그대로인가"를 알 수 있게.
    `gate_reasons`는 진단용이라 state에 붙이지 않고 아래 `/translate/report`로만 노출한다.
  - 리포트 경고(용어집 LLM 판정 실패·참고문헌 규칙 불일치·캐시 전량 무효 등)는 **`warnings`**
    (문자열 목록, 최대 50건·건당 1,000자)로 덧붙인다 — 경고가 없으면 키가 없다. 사유 집계와
    같이 **마지막으로 완료된 번역**의 것이다. 프런트는 '번역 참고 사항 N건' 목록으로 보인다.
- **GET /api/jobs/{id}/translate/report?lang=ko** — `translations/{lang}/report.json`을
  `{"job_id","lang", …report}`로 그대로 반환한다(§13.2의 전 키 — `kept_original` 유닛 id,
  `skip_reasons`/`kept_reasons`/`gate_reasons`, `cache_prior`/`cache_reused`, `warnings`).
  `400` 미지원 lang / `404` 리포트 없음("먼저 번역을 실행하세요").
- **POST /api/jobs/{id}/translate/cancel?lang=ko** — 실행 중이면 `202 {"status":"canceling"}`,
  아니면 현재 상태 반환.
- **GET /api/jobs/{id}/translate/events?lang=ko** — `/events`와 동일 SSE 패턴
  (`retry:3000`, 15초 `: ping`, 구독자 상한 초과 시 503 + `Retry-After: 5`). 브로커 채널 키
  `"{id}:translate:{lang}"`.
  스냅샷: `done`→`done` 1회 후 종료 / `error`·`canceled`→`error` 후 종료 /
  `running`→`progress` 스냅샷 후 구독 루프 / `none`→`404`.
  - `event: progress` `{"phase":"translate","lang":"ko","current":3,"total":12,"status":"running"}`
  - `event: done` `{"phase":"translate","lang":"ko","markdown_url":"…?lang=ko","html_url":"…?lang=ko","layout_url":"…?lang=ko","counts":{"total","translated","cached","skipped","kept_original"}}`
  - `event: error` `{"message":"…","canceled":false}` (취소 시 `message:"번역이 취소되었습니다"`, `canceled:true`)
- **기존 라우트의 `?lang=` 쿼리** (미지정이면 원본 동작 그대로, 지원 외 언어는 400):
  - `/markdown?lang=ko`·`/html?lang=ko` → `result.{lang}.md` 사용(없으면 404 "한국어 번역본이
    없습니다 — 먼저 번역을 실행하세요"), `X-Partial` 헤더 없음.
  - `/layout?lang=ko` → `layout.{lang}.json` 로드(없으면 동일 404).
  - `/layout.html?lang=ko` → `/document.html?lang=ko`로 307 리다이렉트(레거시 호환).
  - `/alignment?page=1&lang=ko` → 원문/번역 페이지·블록 수, type, bbox 불변식을
    검증한 뒤 `{id,index,type,bbox,source,target,translated}` 배열 반환. 대응이
    손상된 번역 레이아웃은 잘못 표시하지 않고 409.
  - `/archive` → `result.md`와 함께 `result.*.md`(예: `result.ko.md`)를 zip에 포함.

### 13.4 불변식

- **플레이스홀더 100% 복원**: 수식·이미지·표 등은 `<m1 v="…"/>`로 마스킹 후 번역, 복원한다.
- **출력 측 검증 게이트**: 유닛 채택 조건은 "플레이스홀더 정합(missing/dup 없음) +
  비어 있지 않음 + `masking.looks_untranslated(src, out, mapping)`가 False"다.
  입력 측 `should_skip()`과 대칭인 게이트로, **플레이스홀더가 온전해도** 모델 거부문
  ("I cannot translate…")·한 줄 요약·영문 echo가 문단을 통째로 대체하는 것을 막는다.
  판정 순서: ① 출력에만 있는 거부문 패턴 → 즉시 거절 ② 마스킹 후 잔여 영단어가
  2개 미만이면 면제(고유명사·짧은 라벨) ③ 복원된 불변 토큰을 뺀 뒤 한글 비율 <15%면
  거절 ④ 길이비가 범위(마스킹 있으면 0.2–4.0, 없으면 0.3–3.0) 밖이면 거절.
  오탐은 래더 왕복 비용만 늘리지만 미탐은 내용 손실이므로 보수적으로 잡혀 있다.
  추가 규칙: 프롬프트 스캐폴딩 echo(`scaffold`), 반복 루프(`repetition` — 짧은 원문 면제보다
  먼저 본다), 1–2단어 라벨이 합쇼체 문장으로 바뀜(`label-sentence`), 80자 이하 원문의 4자리 이상
  숫자가 전부 사라짐(`number-mismatch`), 한국어 메타 문장으로 감싼 원문 echo — 8단어 이상
  원문의 단어 80% 이상이 같은 순서로 연속해 남음(`echo`, 한글 비율 규칙 뒤에 본다 — 실번역 쌍은
  최대 39%), 짧은 원문 면제에도 출력 길이 상한. 원문이 번역에 관한
  글이면 '번역…수 없' 같은 한국어 문장은 거부문으로 보지 않는다. 예전 코드가 캐시한 출력도
  지금 게이트로 다시 판정해 떨어지면 다시 번역한다(`cache_rejected`).
- **퇴화 출력 소거**: 같은 정규화 출력이 서로 다른 원문 3개 이상에서 나오면(예: 소형 모델의
  '요약입니다.') 퇴화로 보고 버린다. 한 번 퇴화로 판정된 출력은 **실행 전체**에서 기억해 2차
  패스에서도 퇴화다. 게이트가 거부한 출력과 분할 조각 출력도 증거로 세고, 이어 붙인 분할 결과도
  유닛 전체 기준으로 게이트를 다시 통과해야 한다.
- **원문 유지 폴백(kept_original 강등)**: 플레이스홀더 복원 실패나 위 검증 거절 유닛은
  기존 래더(repair → 문장 분할)로 흡수하고, 래더까지 소진되면 **원문을 그대로 둔다**
  (내용 손실 금지). 해당 유닛 id는 `report.json`의 `kept_original`에 남는다.
  캐시 기록(`publish_cache`)은 `status=="translated"`일 때만 호출되므로 거절된 출력이
  `units.json`을 오염시키지 않는다. 취소로 조기 반환된 유닛은 이 통계에 포함되지 않는다.
- **유닛 단위 4xx 강등**: 재시도해도 같은 결과인 4xx(`TranslateUnitRejected` — 400·413·422 등,
  429/408 제외)는 잡 전체를 죽이지 않고 래더로 넘긴다. 단 이 실행에서 아직 성공한
  유닛이 하나도 없으면 엔드포인트·설정 자체 문제이므로 종전대로 전파한다
  (전 유닛이 kept로 조용히 done 되는 회귀 방지).
- **캐시 키 구성**: `sha256(PROMPT_V ∥ model ∥ 정렬된 용어집쌍 ∥ 마스킹된 원문
  ∥ 원문 전체 ∥ 블록 종류 ∥ 직전 문맥 ∥ temperature ∥ reasoning [∥ request_variant])`.
  `request_variant`(reasoning 전달 방식·`TRANSLATE_EXTRA_BODY`)는 요청 모양이 종전과 다를 때만
  넣는다 — 종전과 같은 요청이면 키가 그대로라 기존 `units.json`이 계속 적중한다.
  `TRANSLATE_EXTRA_BODY`는 원문 JSON이 아니라 정규화한 값의 다이제스트
  `extra_body=sha256:<앞 16자리>`로만 싣는다 — `request_variant`는 인증 없는
  `/translate/state`가 내주는 state.json에 남기 때문이다.
  모델·프롬프트 버전·해당 유닛 용어집·원문·제목/본문 정책·샘플링 설정이 바뀌면
  영향받는 유닛만 자동 재번역된다. 짧은 placeholder 미리보기 충돌도 원문 전체로 분리하고,
  같은 문장도 직전 문맥이 다르면 별도 번역한다. `TRANSLATE_REASONING`을 off→high로
  올린 뒤 재개해도 이전 설정의 번역이 재사용되지 않는다.
- **PROMPT_V 상승 = 캐시 전면 무효화**: `PROMPT_V`가 캐시 키의 첫 재료이므로 값이 오르면
  기존 `units.json`은 전부 미스가 된다. 현재 값은 **`"6"`**(v6 = 출력 측 검증 도입 +
  인라인 수식 통화 오인 수정)이라 **v5 이전에 만들어진 유닛 캐시는 이번 사이클에
  무효화된다** — 이미 캐시된 거부문·echo를 강제로 다시 번역시키는 것이 목적이다.
  `report.json`/`state.json`의 `prompt_v` 필드로 어떤 버전으로 만든 결과인지 확인할 수 있다.
- **잘린 출력은 채택하지 않는다**: `client.complete()`는 잘린 텍스트를 돌려주지 않는다. chat
  `finish_reason=="length"` / Responses `status=="incomplete"`를 감지하면 같은 요청을 **max_tokens
  2배로 1회 재시도**하고(thinking 토큰이 예산을 소진하는 경우 대비), 그래도 잘리면
  `TranslateOutputTruncated`를 낸다. 반복 루프·지나치게 긴 출력(프롬프트의 4배와 2,000자 중 큰
  값 초과)·`TRANSLATE_MAX_TOKENS_PARAM=none`(경고 로그)은 2배 재시도를 하지 않는다. 빈 출력은
  `TranslateEmptyOutput`, 읽기 타임아웃(1회 재시도 뒤)은 `TranslateTimeout` — 셋 다
  `TranslateUnitRejected` 하위라 유닛 단위로 분할 래더를 타고, 끝내 실패하면 `truncated`·
  `empty-output`·`timeout` 사유로 원문을 두며 **캐시하지 않는다**. 단 마지막 API 성공 이후 최초
  패스가 시간 초과로 끝난 유닛이 max(2, `TRANSLATE_CONCURRENCY`)개에 닿으면 엔드포인트가 멈춘
  것으로 보고 `번역 API가 응답하지 않습니다` 오류로 잡을 실패시킨다(API 성공 한 번이면 집계가
  0으로 돌아간다). 유닛 하나가 반쪽까지 멈추는 것은 그 유닛만 timeout 사유로 원문 유지한다.
  2배 재시도의 응답이 `TRANSLATE_MAX_RESPONSE_MB`를 넘으면 잡 오류가 아니라 그 유닛의 잘림이다
  (첫 시도의 상한 초과는 여전히 잡 오류 — 게이트웨이 보호). 첫 성공 전에 잘린 유닛이
  나오면(콜드 실행) 잡은 thinking을 끄라는 안내와 함께 실패한다(로컬 서버면 `--chat-template-args`
  ·모델별 설정까지 안내). repair(태그만 바로잡기) 패스는 태그가 빠지거나 겹쳤고 출력이 루프도
  과다 길이도 아닐 때만 돈다.
- **스트리밍**(chat 기본, `TRANSLATE_STREAM`): HTTP 교환은 도우미 스레드가 하고 워커는 0.1초마다
  취소를 확인한다 — 취소면 소켓을 끊어 서버 생성까지 멈춘다(실측 mlx_lm: 취소 1.52초 뒤 반환,
  서버는 BrokenPipe 뒤 토큰을 더 내지 않음). usage·finish_reason은 스트림에서 읽고 `: keepalive`
  주석은 무시한다. 서버가 첫 스트리밍 요청을 400/415/422로 거부하면 비스트리밍으로 다시 보내고
  그 클라이언트에서 고정한다. 응답 본문은 `TRANSLATE_MAX_RESPONSE_MB`로 묶는다(선언 길이와 실제
  바이트 모두 — 끝나지 않는 SSE 한 줄도). 헤더는 대소문자를 가리지 않는다(mlx_lm은
  `Content-type`을 보낸다).
  - 본문은 비스트리밍·SSE 모두 64KB 고정 조각으로 읽고 SSE도 선언된 Content-Length를 먼저 본다
    — 길이 없는 HTTP/1.0 스트림(mlx_lm.server)·거대 청크·gzip 응답에도 상한이 점진적으로
    걸린다. SSE 줄 분할은 받은 바이트에 선형이다.
  - `[DONE]` 뒤 길이가 정해진 keep-alive 응답(chunked·Content-Length, `Connection: close` 아님)은
    종료 표시를 1초·64KB 안에서 읽어 연결을 풀에 돌려준다(못 읽으면 종전처럼 닫는다).
  - **폭주 차단**: 본문(`_postprocess` 뒤 — 분리되지 않은 `<think>…`는 세지 않는다)이 프롬프트의
    2배(하한 2,000자)를 넘으면 1,000자마다 반복 루프(`is_degenerate_repetition`, 프롬프트의 반복
    대비)와 과다 길이(프롬프트의 4배·2,000자 중 큰 값 초과)를 확인한다. 걸리면 `[DONE]`·
    max_tokens를 기다리지 않고 `TranslateOutputTruncated`(2배 재시도 없음)로 끝내고 연결을 닫아
    서버 생성도 멈춘다 — 그 출력은 끝까지 받아도 출력 게이트가 거부한다(루프는 늘기만 하고, 4배
    초과는 길이비 상한 밖). 예전에는 로컬 0.8B 번역에서 요청 415건 중 29건이 8192토큰을 끝까지
    태워 전체 요청 시간의 75%를 썼다(P4 실앱). 콜드 실행 규칙(첫 성공 전 잘림은 잡 실패)은 같다.
    과다 길이 기준은 답이 사고와 **구분될 때만** 쓴다 — content에 `</think>`가 왔거나(그 뒤가 답)
    reasoning이 별도 필드로 온 스트림이다. 템플릿이 `<think>`를 프롬프트 끝에 미리 넣는 thinking
    모델(Qwen3·QwQ·R1 distill)을 reasoning 분리 없이 서빙하면(llama.cpp `--reasoning-format none`,
    LM Studio 분리 끔, vLLM parser 없음) content가 여는 태그 없이 사고로 시작해 `</think>` 전에는
    답과 가를 수 없다 — 그런 출력은 반복 루프만 보고, 길이는 완료 뒤 길이비 게이트가 판정한다
    (사고를 답으로 세어 끊던 때는 그 구성의 콜드 런 잡이 전부 실패했다).
  - 스트림 중간 `{"error": …}` 이벤트와 `finish_reason: "error"`는 상태코드 정책을 따른다 —
    5xx·408·429·코드 없음은 5xx처럼 백오프 재시도, 400·413·422 또는 invalid_request·
    context_length 유형은 유닛 거부, 401·403은 인증 오류. 부분 출력은 번역문으로 쓰지 않는다.
  - 연결 실패 문구는 원인 요약만 담고(`번역 API 연결 실패(연결 거부) — OPENAI_BASE_URL과 …`)
    URL·호스트·쿼리는 쿼리를 가린 서버 로그에만 남긴다. 엔진이 쓰는 state.json `error`는
    500자 상한이다.
- **Responses `store:false`**: 번역의 Responses 요청도 `store:false`를 싣는다. 서버가 `store`를
  이유로 400/422를 내면 한 번 빼고 다시 보내 그 클라이언트에서 고정하고 경고를 남긴다.
- **think 정리**: 마지막 `</think>` 뒤만 남긴다(여는 태그가 없어도). 닫히지 않은 선행
  `<think>`는 본문 없음(잘림·빈 출력). `reasoning`·`reasoning_content` 필드는 무시한다. 출력
  맨 앞에 프롬프트의 `[번역할 원문]` 줄이 메아리치면 한 줄 지운다.
- **용어집**: 판정 호출이 실패하면 `glossary.json`과 `glossary.incomplete` 표식을 함께 저장하고
  경고한다 — 다음 실행이 다시 판정하고 성공하면 표식을 지운다. 판정 중 취소면 `glossary.json`을
  남기지 않는다. 첫 등장 병기는 md 순서(`first_unit`)와 layout 순서(`first_unit_lay`)를 따로
  계산해 layout 잡도 `[첫 등장 병기]`를 정확히 한 번 보낸다. recall·attention 같은 시드 용어는
  일반 관용구에 강제하지 않는다. 저장은 원자적이고 깨진 파일은 다시 만든다.
- **설정 검증**: `TRANSLATE_TEMPERATURE`는 `none` 또는 0–2, `TRANSLATE_CONTEXT`는
  `False`/`OFF`도 받는다. 모델 id가 틀려 404 + JSON 오류 본문이 오면 서버 문구와 함께 모델 id를
  확인하라고 안내한다.
- **취소 응답성**: cancel은 유닛 디스패치 사이뿐 아니라 **래더 단계(최초→repair→분할)
  사이에서도** 확인된다 — 거대 표 래더(유닛당 수 분)가 취소 후에도 이어지지 않는다.
  취소로 조기 반환된 유닛은 kept_original 통계에 포함되지 않는다.
- **엔진이 상태의 단일 기록자**: `state.json`은 `run_translation`이 직접 쓴다.
  API 스레드는 SSE 이벤트 중계와 레지스트리 정리만 담당한다. 예외: 번역 스레드가 엔진 밖에서
  실패했는데 state가 아직 `running`이면 API가 실제 사유로 마감한다(엔진이 이미 쓴 error·canceled는
  덮지 않는다).
- **삭제된 잡**: 번역 엔진은 잡 디렉터리를 다시 만들지 않는다 — `translations/`·`translations/{lang}`만
  `parents` 없이 만들고, 잡 디렉터리가 없으면 `작업 디렉터리가 없습니다 — 삭제된 작업은 번역할 수
  없습니다`로 끝난다(예전에는 삭제 도중의 번역이 빈 잡 디렉터리를 되살렸다).

### 13.5 stale-running 조정

서버가 재시작되면 진행 중이던 번역 스레드는 사라지지만 `state.json`은 `running`으로 남는다.
`translate_tasks` 레지스트리(키 `(job_id, lang)`)에 태스크가 없는데 `state.json`이 `running`이면,
`/translate/state`·`/translate/events` 응답 시 **error로 원자적 재기록**한다
(메시지: "서버가 재시작되어 번역이 중단되었습니다 — 다시 실행하세요"). OCR 잡 복원(§JobStore.
load_existing)과 같은 사상 — 좀비 running을 사용자에게 보이지 않게 한다.

## 14. 보안

- **인증 없음**: 모든 REST/SSE 엔드포인트가 무인증 — 접근 가능하면 전 문서 열람
  (`GET /api/jobs`)·삭제·변환·(설정 시) 유료 번역/Q&A 트리거가 가능하다.
- ⚠ **compose 기본값은 노출이다** (커밋 `3be81c8` 이후): 포트는
  `${BIND_HOST:-0.0.0.0}`에 바인딩되고 `ALLOWED_HOSTS` 기본은 `*`다. 따라서
  **기본 신뢰 경계는 "로컬 머신"이 아니라 "서버가 붙어 있는 네트워크"** 이며,
  VPN/Tailscale이나 방화벽 뒤 홈랩 같은 신뢰 네트워크를 전제로 한다. 공개
  인터넷에 노출한다면 **반드시 인증을 제공하는 리버스 프록시**(nginx basic auth 등)
  뒤에 두어야 한다. README §보안 · SECURITY.md와 같은 정책이다. Docker 게시 포트는
  `ufw`·`firewalld` 규칙을 거치지 않으므로 '방화벽 뒤'는 네트워크 방화벽·`DOCKER-USER` 체인·
  밖에서 닿지 않는 `BIND_HOST`를 뜻한다(§8). CSRF 방어도 두지 않는다 — 이 기본값들은 의도된
  결정이고 0.0.0.0·`*` 기본은 `test_ci_ops_contracts`가 고정한다.
- **로컬 전용으로 되돌리기(opt-in 하드닝)** — `.env`에:
  1. `BIND_HOST=127.0.0.1` — 포트를 루프백에만 바인딩
  2. `ALLOWED_HOSTS=localhost,127.0.0.1` — Host 헤더 화이트리스트 복원
     (도메인/IP로 접속한다면 그 값을 목록에 추가)
- **Host 헤더 화이트리스트**: `TrustedHostMiddleware`가 `ALLOWED_HOSTS` 밖의 Host를
  400으로 거부 — 악성 웹페이지가 DNS rebinding으로 same-origin을 획득해 인스턴스의
  문서를 읽어가는 것을 차단한다. `config.py`의 **코드 기본값은 `localhost,127.0.0.1`**
  이지만(로컬 `make dev` 실행에 적용) compose는 위 노출 정책에 맞춰 `*`를 넘긴다 —
  와일드카드가 들어오면 `main.py`가 기동 시 경고 로그를 남긴다.
  Starlette는 포트를 떼고 비교하므로 `localhost:8000`도 통과하고, 컨테이너 내부
  healthcheck(`curl http://localhost:8000/api/health`)도 동작한다.
  테스트는 conftest가 테스트 프로세스의 기본 화이트리스트에만 `testserver`를 추가한다.
- **비용 남용 방어**: 인증이 없는 대신 유료 경로(`/qa`·`/translate`)에 잡·IP 단위
  레이트리밋과 동시 실행 상한을 두고 초과분을 429 + `Retry-After`로 거절한다 (§5).
  `/render-preview`도 크기 가중 레이트리밋·동시 4건·본문 256 KiB로 묶고, SSE 구독은 잡당 8·전체
  64로 묶는다. 리버스 프록시 뒤에서는 `TRUSTED_PROXY_IPS`에 있는 피어의 `X-Forwarded-For`만
  믿는다(§5).
- **HTTP 보안 헤더** (`main.SecurityHeadersMiddleware`, 순수 ASGI): 모든 `text/html` 응답에 CSP와
  `Referrer-Policy: same-origin`. 공통 지시어 `default-src 'self'`, `style-src 'self'
  'unsafe-inline'`(레이아웃 좌표·KaTeX의 인라인 style 속성), `img-src 'self' data: blob:`,
  `font-src 'self' data:`, `object-src 'none'`, `base-uri 'self'`, `form-action 'self'`,
  **`frame-ancestors 'none'`**(다른 사이트가 이 인증 없는 UI를 보이지 않는 iframe에 싣고 삭제·번역
  버튼 클릭을 유도하는 클릭재킹 차단 — meta CSP에서는 무시되므로 헤더에만 있다). SPA 문서는
  `script-src 'self'` + `index.html` 인라인 스크립트의 sha256 해시(지금은 없다 — 테마 부트스트랩은
  `theme-init.js`, 파일이 바뀌면 해시를 다시 계산), API가 내보내는 HTML(`document.html` — 인라인
  KaTeX)은 `script-src 'self' 'unsafe-inline'`. 정적 프론트엔드(`/api` 밖)는
  `Cache-Control: no-cache` — 업그레이드 뒤 브라우저가 옛 ES 모듈과 새 모듈을 섞어 쓰지 않게
  ETag로 재검증한다(라우트가 정한 헤더는 덮지 않는다).
- **SPA meta CSP**: `index.html`의 `<meta http-equiv="Content-Security-Policy">`와
  `<meta name="referrer" content="same-origin">`는 meta가 표현할 수 있는 모든 지시어에서 헤더와
  **같다**(`connect-src 'self'`는 `default-src`와 같은 값) — 실효 정책은 헤더 정책 그대로이고
  `frame-ancestors`만 헤더 전용이다. 두 층은 함께 고친다(`tests/test_security_headers.py`가
  어긋나면 실패한다). 예전 meta의 `font-src 'self'`·`no-referrer`는 헤더와 달라 실효 정책이
  교집합이었다 — 지금 실효 리퍼러는 same-origin이며 다른 출처로는 여전히 Referer가 가지 않는다.
- **단독 내보내기 CSP**: 내려받는 `document.html`(facsimile·의미 기반 둘 다)은 `<meta charset>`
  바로 뒤에 `default-src 'none'; img-src data: blob:; font-src data:; style-src 'unsafe-inline';
  script-src 'unsafe-inline'` meta CSP와 `no-referrer`를 둔다 — 모든 자원이 인라인이라
  디스크에서 열어도 정상 렌더되고 어떤 외부 요청도 나가지 않는다(§5).
- **외부 이미지 차단**: 렌더러(`pipeline/render.py`)는 `images/<파일>`(하위 경로·`..` 없음)과
  `data:image/(png|jpeg|gif|webp)`만 `<img>`로 만들고, 그 밖(http(s)·`//호스트`·LAN 주소·같은
  출처 절대 경로·경로 탈출)은 클릭해야 열리는 링크 `<a class="blocked-image" rel="noopener
  noreferrer nofollow">`로 바꾼다 — OCR·텍스트 레이어 마크다운의 `![](https://…)`가 문서를 여는
  순간 열람 사실·IP·인스턴스 주소를 흘리고 내부망 GET을 유도하던 경로(텍스트 레이어의 보이지 않는
  `render_mode=3` 비콘 포함). SPA도 넣기 전에 외부 이미지 출처를 자리표시로 바꾼다(§10).
- **렌더러 선형 시간**: 수식·코드 펜스 스캔이 선형이다(다음 이스케이프 안 된 여는 기호나 빈 줄에서
  멈추고, `\\[2pt]` 같은 이스케이프는 구분자가 아니다) — 예전 2차 정규식은 20 KiB에 0.56초였고
  `/render-preview` 한 번이 GIL을 쥔 채 서버 전체를 멈출 수 있었다. 마스크 복원은 한 번에 하고
  NUL은 U+FFFD로 바꿔 재귀·팽창이 없다.
- **수식(KaTeX 0.18.10)**: GHSA-238p-pmpm-9mq7(0.18.2에서 수정)을 포함한 판으로 올렸고, 앱과
  단독 내보내기 모두 `maxSize 10`·`maxExpand 1000`·`strict 'ignore'`·`trust false`로 부른다(§10).
- **신뢰할 수 없는 PDF**: MuPDF 작업은 서버 밖 워커 프로세스에서 시간 상한과 함께 돌고, 업로드는
  렌더 없이 작업량을 재는 복잡도 게이트를 지난다(§18). 손상 PDF는 서버 경로 없는 고정 문구로
  거부하고(원래 오류는 서버 로그에만), Pillow 압축 폭탄 상한은 앱의 최대 렌더 픽셀 + 5%
  (5,250만 픽셀)로 낮춘다(프로세스 전역 — 낮추기만 하고 올리지 않는다).
- **모델 출력은 비신뢰**: 벤더 P8/P9가 `eval()`을 없앴고(좌표는 `ast.literal_eval`), P22가 bbox를
  페이지 안으로 자른다(MLX 포팅 M4도 같다). sidecar 응답은 스키마 검증·정화를 거친다
  (OCR_ENGINE_PROTOCOL.md).
- **공급망**: pip-audit(`dependency-audit` 잡·`make audit`)·trivy(릴리스 push 전)·Dependabot,
  digest 고정 베이스 + `apt-get upgrade`, 해시 고정 sidecar lock, SHA 고정 액션(§11.2).
  `tests/test_dependency_floor.py`가 MuPDF/PyMuPDF·Pillow·urllib3·anyio 보안 하한을 지킨다.
  수용한 잔여 권고(torch 2.10.0·transformers 4.57.1 — 모델 카드 고정)는 이유와 2027-04-01 기한을
  달고 있다(SECURITY.md).
- **설정 노출**: `/api/health`의 `config_warnings`는 이 앱이 읽지 않는 `.env` 키의 **이름**만
  싣는다(값은 로그에도 남기지 않는다).
- **자격증명 분리**: Q&A는 `LLM_OPENAI_API_KEY`만 읽고 번역용 `OPENAI_API_KEY`로
  폴백하지 않는다 (§17.3·§17.4) — 제3자 게이트웨이 키가 `api.openai.com`으로
  전송되던 유출 경로를 막는다.
- **경로 탈출 방어**: `/files/{path}`는 정규화(resolve) **이후에** allowlist를
  적용해 `pages/../source.pdf` 류로 잡 디렉터리 안의 임의 파일을 읽는 우회를 막는다 (§5).

## 15. 운영 — 디스크 보존 정책

- **work/ 즉시 정리**: 잡이 터미널 상태(done/error/canceled)로 마감되면 `work/`
  (모델 원시 출력)를 삭제한다 — 필요 산출물은 병합(add_chunk) 시점에 이미
  `images/`·`layout/`·`result.md`·`layout.json`으로 이동/기록돼 있고, work/에는
  boxes.json·raw_pages.json·실패 청크 잔여물만 남아 잡마다 축적됐다.
- **잡 TTL GC (`JOB_TTL_DAYS`, 기본 `0`=비활성)**: 로컬 도구에서 사용자 데이터
  자동 삭제는 opt-in — 기본값에서는 잡이 무기한 보존되며 UI/`DELETE /api/jobs/{id}`로
  수동 정리한다. N>0으로 설정하면 서버 시작 시 1회 + 6시간마다, 터미널 상태이고
  meta.json mtime이 N일보다 오래된 잡을 DELETE 엔드포인트와 같은 경로
  (`JobStore.delete_dir`)로 제거한다(삭제마다 `logger.info` 1줄).
  queued/running 잡과 번역 스레드가 실행 중인 잡은 절대 삭제되지 않는다.
- **⚠ `output/` 추적 해제는 다른 클론에서 "파일 삭제"로 나타난다**: `fd478ce`가
  일회성 변환 산출물 40MB(6개 파일)를 `.gitignore` + `git rm -r --cached output/`으로
  뺐다. `--cached`는 **인덱스에서만** 지우지만 커밋에는 삭제로 기록되므로, 파일이
  남는 것은 명령을 실행한 작업 트리뿐이고 그 커밋을 pull 하는 다른 클론에서는
  `output/`이 통째로 사라진다. 필요하면 `git checkout fd478ce^ -- output/`으로
  파일만 되살린다(추적은 되지 않는다). `output/`은 재생성 가능한 산출물이므로
  보관이 필요하면 리포 밖에 둔다. 앞으로 추적 중인 디렉터리를 무시 목록에 넣을
  때는 커밋 메시지에 "다른 클론에서는 삭제된다"를 반드시 적을 것.

### 15.1 업그레이드 노트 — 버전 상수 상향 = 캐시 전량 무효화

이 리포에는 산출물 캐시를 무효화하는 **버전 상수가 셋** 있다. 오르면 기존 배포를 올린 뒤
잡마다 1회씩 아래가 실제로 일어난다. **코드 변경 없이 조용히 일어나므로**, 모르면 "왜 갑자기
느리고 청구서가 늘었나"가 된다. 0.1.0 이후 사이클에서는 `PDF_EXPORT_FORMAT_VERSION`(→ 16)과
`ENRICH_VERSION`(→ 6)이 올랐고 `PROMPT_V`는 그대로다(아래 '이번 업그레이드').

| 상수 | 위치 | 현재 값 | 상향 시 무효화되는 것 | 비용 |
|---|---|---|---|---|
| `PROMPT_V` | `translate/types.py` | `"6"` | `translations/{lang}/units.json` **전체**(캐시 키 첫 재료) | **유료 API 전량 재호출** |
| `PDF_EXPORT_FORMAT_VERSION` | `pipeline/pdf_export/report.py` | `16` | `export.{lang}.pdf`·`.dual.pdf`·`.report.json` | CPU (실측 9.4s/16p — 지금은 export 워커 프로세스) |
| `ENRICH_VERSION` | `pipeline/pdf_fonts.py` | `6` | `layout.json`의 폰트 메타(`fonts_v`) → 뒤이어 export 캐시 | CPU (재조판) |

- **가장 비싼 것은 번역이다**: 모델·샘플링(`TRANSLATE_TEMPERATURE`·`TRANSLATE_REASONING`)을
  바꿔도 같은 일이 일어난다(§13.4). 재번역은 **번역을 다시 실행할 때만** 일어나므로,
  이미 있는 `result.ko.md`를 그대로 두면 비용은 0이다 — 예산을 확인하기 전에는 기존
  잡의 번역을 재실행하지 말 것.
- **확인 방법**: 재번역이 일어났다면 `report.json`에 `cache_prior > 0 && cache_reused == 0`이
  남고 `warnings`에 "기존 캐시 N건이 하나도 적중하지 않아 유닛 M개를 전량 재번역했습니다"가
  들어간다. 서버 로그에도 같은 줄이 `WARNING`으로 남는다. `state.json`/`report.json`의
  `prompt_v`로 어떤 프롬프트 버전으로 만든 결과인지 구분한다.
- **PDF/레이아웃 쪽은 CPU만 든다**: 다만 업그레이드 직후 사용자가 여러 잡을 동시에 열면
  export 빌드가 한꺼번에 몰린다. 그래서 전역 상한(`PDF_EXPORT_MAX_CONCURRENT`, 기본 2)이
  있고, 대기가 `PDF_EXPORT_QUEUE_TIMEOUT_S`(기본 30초)를 넘으면 503 + `Retry-After`가 나간다
  (§5 `/pdf`). 업그레이드 직후 잠깐 503이 보이는 것은 **설계된 동작**이며, 코어가 넉넉하면
  상한을 올려 흡수한다. 이 값은 이제 export **워커 프로세스 수**라 빌드가 코어를 실제로 나눠
  쓴다(예전 스레드 빌드는 GIL 때문에 몇 개를 띄워도 코어 하나였다) — 작은 서버에서는 OCR과
  코어를 다투므로 1–2로 둔다(1이면 예열하지 않는다).
- **권장 절차**: ① 이미지 갱신 전 `data/jobs`(=`ocr-data` 볼륨) 백업 → ② 올린 뒤 잡 하나로
  번역을 재실행해 `report.json`의 `cache_reused`로 비용 규모를 실측 → ③ 여럿이 쓰는 서버라면
  `TRANSLATE_RATE_LIMIT_PER_MIN`·`TRANSLATE_MAX_ACTIVE`를 임시로 낮춰 동시 재번역을 묶는다.
- **롤백의 비대칭**: `units.json`은 키→번역문 **사전**이라 옛 항목이 지워지지 않는다 —
  구버전으로 되돌리면 옛 키가 다시 맞아 재번역이 일어나지 않는다(신·구 항목이 함께 쌓인다).
  반면 `export.{lang}.pdf`·`layout.json`은 **덮어쓰는 단일 산출물**이라 버전을 오갈 때마다
  매번 재생성된다(CPU만).

#### 이번 업그레이드(0.1.0 → 현재)에서 운영자가 보게 되는 것

- **내보내기 캐시 1회 재생성**: `PDF_EXPORT_FORMAT_VERSION` 16과 새 빌드 스탬프(§5 `/pdf` —
  옛 `export.{lang}.font.txt`는 스탬프가 아니라 무효), `archive.zip`의 내용 서명 때문에 잡마다
  번역 PDF·대조 PDF·ZIP을 한 번씩 다시 만든다. `ENRICH_VERSION` 6이라 layout 폰트 메타도 잡마다
  한 번 다시 주입된다(회전 페이지 수정).
- **옛 meta**: `page_separator`가 없는 잡에는 기동 때 현재 값을 한 번 고정한다(meta mtime 보존 —
  TTL 시계 불변). `notices`가 없는 옛 잡의 처리 경위 메모는 메모리에서 참고로 갈라 읽는다(§4).
  옛 잡에는 `started_at`·`finished_at`이 없다.
- **옛 OvisOCR2 잡**(image 블록뿐인 layout.json): 이제 '레이아웃 없음'이다 — 좌표 라우트 404,
  `/pdf` 409, `has_layout=false`, `document.html`은 의미 기반. 다시 번역해도 `layout.ko.json`을
  쓰지 않고, 남아 있는 옛 파일은 사용 가능 판정에서 무시된다.
- **번역 캐시**: `PROMPT_V`는 그대로라 전량 재번역은 없다. 캐시 키에 `request_variant`가 들어가는
  것은 reasoning 전달 방식이나 `TRANSLATE_EXTRA_BODY`가 종전과 다를 때뿐이다(로컬 서버에
  reasoning을 설정한 잡은 의도대로 다시 번역된다). `TRANSLATE_EXTRA_BODY`를 쓴 잡은 키 재료가
  원문에서 다이제스트로 바뀌어 한 번 다시 번역되고, `*.api.openai.com`이나 `*.localhost`·
  `*.lan`·`*.home.arpa`·`*.internal` 호스트 뒤에서 `TRANSLATE_REASONING`을 설정한 잡도 auto
  전달 방식 판정이 바뀌어 한 번 다시 번역된다. 마스킹 규칙·용어집 첫 등장 계산이 바뀐 일부
  유닛과, 예전 코드가 캐시했지만 지금 게이트를 통과하지 못하는 출력(`cache_rejected`)은 다시
  번역된다. `result.ko.md`는 부분만 덮이던 유닛이 이제 자기 번역을 써서 내용이 달라질 수 있다.
- **디바이스 기본값**: 로컬(uv) 실행의 `OCR_DEVICE` 기본이 `auto`다 — Apple Silicon은 MLX(없으면
  MPS), cu129 extra를 깐 Linux는 CUDA. compose는 서비스마다 고정이라 그대로다.
- **번역은 기본 스트리밍**(chat 모드) — 스트리밍을 못 다루는 게이트웨이면 `TRANSLATE_STREAM=0`.
- **`.env` 확인**: 로컬(uv) 실행은 이 앱이 읽지 않는 키를 기동 로그·`/api/health`의
  `config_warnings`로 알린다. Docker는 컨테이너에 `.env`가 없어 이 목록이 늘 비어 있으니 README
  §빠른 시작 (Docker)의 점검 명령을 쓴다. `REASONING_EFFORT`는 번역 `TRANSLATE_REASONING` 또는
  Q&A `LLM_REASONING_EFFORT`로 바꾼다.
- **프록시 배포**: `TRUSTED_PROXY_HOPS>0`인데 프록시가 루프백이 아니면 `TRUSTED_PROXY_IPS`가
  필요하다(없으면 레이트리밋이 프록시 IP 하나로 붕괴하고 한 번 경고).
- **`PDF_EXPORT_MAX_CONCURRENT=1`이면 예열하지 않는다.** 업로드 복잡도 게이트가 아주 조밀한
  도면·지도 페이지를 400으로 거부하면 사유에 적힌 설정을 올리거나 0으로 끈다(§18).
- **시작 직후 첫 잡 목록 폴**은 잡마다 layout.json을 한 번 파싱한다(그 뒤로는 파일 버전별
  캐시). 메타데이터가 없는 `j_<12hex>` 디렉터리는 기동 때 지워진다.
- **더 엄격한 기동 검증**: 범위 밖 숫자 노브(예: `PAGES_PER_CHUNK=0`·`OCR_DECODE_BLOCK=0`·
  음수 `JOB_TTL_DAYS`)는 예전처럼 조용히 보정되지 않고 기동 시 실패한다(§7).

## 16. textlayer 엔진 (Localight 통합)

`OCR_ENGINE=textlayer` — **모델 다운로드 없이 CPU만으로 즉시 동작**하는 경량 엔진.
Localight의 "텍스트 레이어 우선 + OCR 폴백" 추출기를 기존 엔진 계약
(`app/engine/base.py`의 OCREngine)에 맞춰 이식했다.

### 16.1 계약

- **텍스트 레이어 우선**: 페이지별로 pymupdf 내장 텍스트 레이어를 먼저 추출한다.
  추출 영숫자 수가 `NATIVE_TEXT_THRESHOLD`(기본 `120`) 미만인 페이지만
  **Tesseract OCR**(`OCR_LANGUAGES`, 기본 `eng+kor`)로 폴백한다 — 텍스트 PDF는
  OCR 오류 없이 원문 그대로, 스캔 페이지만 OCR을 거친다. Tesseract 출력에는 bbox가 없어
  OCR한 페이지는 레이아웃 블록이 없다 — 전면 스캔 문서는 `layout.json`이 생기지 않는다(§4).
- **읽기 순서** (`pipeline/reading_order.py`, 텍스트 레이어 복구 경로와 공용): `sort=True`의 좌표
  정렬 대신 **내용 스트림 순서**에서 시작해 다단을 감지한다 — 가로지르는 블록 높이가 가장 작은
  x를 단 사이 홈으로 보고(양쪽 각 20% 이상, 가로지름 25% 이하, O(n log n)) 가로지르는 블록을 띠
  경계로 삼아 띠마다 왼쪽 단 → 오른쪽 단으로 읽는다(3단 이상은 재귀). 세로 여백 도장은 맨 뒤로
  보낸다. 같은 줄 조각과 같은 문단의 다음 줄 이어짐을 합친다(글자 크기·마침 구두점·목록·알고리즘
  항목·표 행·가운데 제목 규칙으로 막는다). 왼쪽·오른쪽 블록이 행마다 짧은 블록으로 마주 보는
  띠(줄바꿈되는 셀의 표, 부캡션 그림 격자 — 양쪽 블록의 60% 이상이 같은 행 짝을 두고 그 블록이
  3.5줄 이하)는 단으로 가르지 않고 내용 순서(행 순서)를 지킨다. 2504.19874v1: 블록 969 → 673,
  글자 손실 0(페이지별 글자 다중집합 대조). PyMuPDF 1.28의 문단 감지가 제목·수식을 더 잘게 쪼개는 것도 여기서
  다시 붙는다.
- **결정적 재실행**: `deterministic_rerun = True` — 같은 페이지를 다시 돌려도 결과가 같으므로
  충실도 게이트가 재처리하지 않는다(§4).
- 텍스트 레이어 추출은 ocr 워커 프로세스에서 페이지 단위로 돈다(§18). 시간 상한을 넘거나
  워커가 죽은 페이지는 `… 텍스트 레이어 추출을 건너뛰었습니다 (…) — 이 페이지는 OCR 경로로
  처리합니다` 경고와 함께 Tesseract 경로로 간다.
- **raw_pages.json 합성**: 블록 좌표(0–999 정규화)를 담은 raw_pages.json을 벤더
  P14와 같은 스키마로 합성한다 → 기존 pipeline/layout.py 경로로 **레이아웃 뷰
  (`GET /layout`)가 그대로 동작**한다.
- **figure 크롭 없음**: capabilities는 `figures: false` — `images/` 산출물이 없고
  마크다운에 이미지 참조도 넣지 않는다 (health의 `capabilities`로 프론트가 안내).
- 모델 로딩이 없어 `model_loaded`는 즉시 `true` — GPU·HF 캐시·sidecar 불필요.
- **Tesseract 런타임**: Docker 이미지에 `tesseract-ocr` + `eng`/`kor` 데이터 포함
  (backend/Dockerfile 2단계 apt). 로컬(uv) 실행은 별도 설치 —
  macOS: `brew install tesseract tesseract-lang`.

## 17. LLM 공급자 레이어 + 페이지 Q&A (Localight 통합)

결과 화면의 **'질문' 탭**에서 현재 페이지 내용에 대해 AI에게 질문한다.
Localight의 공급자 계층을 `app/llm/`(providers.py + validate.py)로 이식했다 —
httpx만 사용하는 자립 모듈(app.* 임포트 없음, lazy import 원칙과 무관하게 torch
의존성 0)로, 계약은 `tests/test_llm_providers.py`가 고정한다.

### 17.1 공급자 (`LLM_PROVIDER`)

| 공급자 | 엔드포인트 | Reasoning/Thinking |
| --- | --- | --- |
| `openai-responses` | `{LLM_OPENAI_BASE_URL}/responses` | thinking이고 effort≠`default`일 때만 중첩 `reasoning {effort, summary}` |
| `openai-chat` | `{LLM_OPENAI_BASE_URL}/chat/completions` | 최상위 `reasoning_effort` + system 프롬프트는 role `developer` (중첩 reasoning 객체 금지) |
| `ollama` | `{OLLAMA_BASE_URL}/api/chat` | `think` 매핑: thinking=False→`false`, effort∈{low,medium,high}→effort, 그 외 `true` |
| `local-openai` | `{LLM_LOCAL_OPENAI_BASE_URL}/chat/completions` | `chat_template_kwargs.enable_thinking`(+ thinking이면 `reasoning_effort`), role `system`, `max_tokens` 8192. think 태그·reasoning 필드는 버린다. `finish_reason=length`면 잘린 답·원시 사고를 돌려주지 않고 503(`answer was cut off at max_tokens` — thinking 끄기 안내) |

`local-openai`(`LocalOpenAIClient`)는 oMLX·LM Studio·mlx_lm.server 같은 **로컬** OpenAI 호환
서버용이다. `LLM_LOCAL_OPENAI_BASE_URL`을 설정해야 구성되고(그때 `LLM_LOCAL_OPENAI_MODEL` 필수),
`/api/providers`에는 구성됐을 때만 나오며 `available`은 실시간 `GET /models` 확인이다.
`LLM_LOCAL_OPENAI_MODELS`는 추가 허용 모델(그 밖은 400), 키는 `LLM_LOCAL_OPENAI_API_KEY`만 쓴다.
`LLM_PROVIDER=local-openai`는 base URL 없이 기동하면 실패한다. `qa_available`은
`LlmRouter.configured(provider)`를 따르고, `GenerationResult.remote`는 공급자 집합으로 정한다.
프런트는 공급자를 늘 명시해 보내므로 `POST /qa`가 `local-openai`를 받는다. 서버가 꺼져 있으면
UI가 서버(oMLX·LM Studio·mlx_lm.server)를 켜고 `LLM_LOCAL_OPENAI_BASE_URL`·
`LLM_LOCAL_OPENAI_MODEL`을 확인하라고 안내한다 — 오류 문구는 URL이 아니라 설정 이름을 적는다.
프런트의 첫 공급자는 저장값 → 서버 기본(`LLM_PROVIDER`) → 그 밖의 순이되 `available`인 공급자를
먼저 고른다(`core.pickQaProvider` — `LLM_PROVIDER`를 비운 배포에서 키 없는 `openai-responses`를
골라 쓸 수 있는 `local-openai`를 두고 첫 질문이 막혔다). Thinking 토글은 사용자가 고른 적이
없으면 원격 공급자만 켜고 로컬(`remote: false` — Ollama·local-openai)은 끈다(`core.qaThinkingDefault`
— 로컬 사고 모델은 8192토큰 예산을 사고로 다 써 503이 났다). 고른 값은 localStorage에 남아 그대로다.

### 17.2 REST 계약 (Q&A)

- **GET /api/providers** — 공급자 카탈로그. 공급자별
  `{id, label, available, remote, supports_reasoning_summary, models, default_model}`.
  OpenAI 계열은 **`LLM_OPENAI_API_KEY`** 유무로, Ollama는 로컬 데몬의 모델 목록
  (`:cloud`/`remote_host` 필터링 후)으로 `available`을 판정한다.
- **POST /api/jobs/{id}/qa** — **현재 페이지에서 추출된 텍스트만** 문맥으로 질의.
  공급자·모델·effort·thinking을 요청별로 오버라이드할 수 있고, 미지정 시
  `LLM_PROVIDER`/`LLM_REASONING_EFFORT`와 공급자별 기본 모델을 쓴다.
  상태코드: 404 잡 없음 / 409 미완료 잡 / 422 페이지 범위 밖·빈 페이지 /
  400 지원하지 않는 프로바이더·허용목록 밖 모델 / **429 레이트리밋·동시 실행 상한
  초과(`Retry-After` 동반 — §5)** / 503 공급자 미구성 또는 업스트림 LLM 장애.
- **GET /api/health** 추가 필드(추가만 — 기존 필드 의미 불변): `qa_available`
  (기본 공급자의 실제 구성 여부를 반영 — 상수 true가 아니다), `llm_default_provider` (§5 참조).

### 17.3 보안 (Localight 프라이버시 약속 승계)

- **전송 최소화**: 외부로는 현재 페이지에서 추출된 텍스트만 전송 — 원본 PDF·페이지
  이미지는 전송하지 않는다.
- **store:false**: OpenAI 두 경로(/responses, /chat/completions) 모두 `store: false`.
- **URL allowlist** (`app/llm/validate.py` — `Settings.from_env()`가 호출하므로
  잘못된 값은 기동 시점에 즉시 실패):
  - `openai_url`: `LLM_OPENAI_BASE_URL`은 `https://api.openai.com` 호스트만 허용.
  - `local_url`: `OLLAMA_BASE_URL`은 루프백(`127.0.0.1`/`localhost`/`::1`)·
    `host.docker.internal`·`ollama`(compose overlay의 컨테이너)만 허용.
    자격증명(userinfo) 포함 URL은 거부.
  - `local_openai_url`: `LLM_LOCAL_OPENAI_BASE_URL`은 정확히 `127.0.0.1`·`localhost`·`::1`·
    `host.docker.internal`의 http(s)만 허용 — userinfo, 쿼리·프래그먼트, 잘못된 포트, IPv4 매핑·
    그 밖의 IPv6, 10진·8진·16진·축약 IP 표기, 끝 점, DNS 이름은 거부한다. 리다이렉트는 따라가지
    않는다.
- **local-openai 전용 키**: `LLM_LOCAL_OPENAI_API_KEY`만 보낸다. 번역 `OPENAI_API_KEY`나 Q&A
  `LLM_OPENAI_API_KEY`로 폴백하면 그 키가 로컬 서버 로그 등으로 샌다.
- **Q&A 전용 키 (`LLM_OPENAI_API_KEY`) — 번역 키와 분리, 폴백 없음**:
  `LLM_OPENAI_BASE_URL`이 공식 `api.openai.com`으로 고정돼 있으므로, OpenRouter·로컬
  게이트웨이용 `OPENAI_API_KEY`를 폴백으로 재사용하면 **그 키가 제3자에게 전송된다**
  (자격증명 유출). `build_router()`는 `settings.llm_openai_api_key`만 읽고,
  미설정이면 `openai-*` 공급자는 `available:false`·`qa_available:false`로 광고되며
  호출 시 `LLM_OPENAI_API_KEY is not configured…` 오류를 낸다.
  compose는 4개 backend 서비스 모두에 이 키를 스레딩한다(§8).
- **`:cloud`/`remote_host` 모델 이중 차단**: 이런 모델은 프롬프트를 외부로 보낼 수
  있으므로 `models()` 목록에서 필터링하고 `generate()`에서 재검증해 호출도 막는다.
- **OpenAI 모델 허용목록 강제**: 요청 `model`은 `LLM_OPENAI_*_MODELS`(+해당 공급자의
  기본 모델)에 없으면 업스트림 호출 전에 거절한다(→ 400). 그러지 않으면 허용목록이
  표시용에 그친다 — Ollama의 온디바이스 재검증과 같은 자리에서 막는다.
- **chain-of-thought 비노출**: 원시 thinking은 표시·저장하지 않는다 — Responses의
  reasoning `summary_text`(요약)만 선택적으로 표면화한다.

### 17.4 번역 서브시스템과의 관계

- 번역(§13)은 계속 `app/translate/client.py`의 **`OpenAICompatClient`**를 쓴다 —
  `OPENAI_BASE_URL`/`OPENAI_MODEL`/`TRANSLATE_*` 설정과 동작은 그대로이며 Q&A
  레이어와 독립이다 (`OPENAI_BASE_URL`은 §17.3 allowlist의 검증 대상이 아니다).
- **키는 분리돼 있다**: 번역은 `OPENAI_API_KEY`, Q&A(OpenAI 공급자)는
  `LLM_OPENAI_API_KEY`를 읽으며 **서로 폴백하지 않는다**. 번역용 키는 임의
  게이트웨이(`OPENAI_BASE_URL`)를 가리킬 수 있는 반면 Q&A는 항상 공식
  `api.openai.com`으로 나가므로, 공유하면 그 키가 무관한 제3자에게 전송된다 (§17.3).
  둘 다 쓰려면 `.env`에 두 키를 각각 넣는다.
- **번역을 Ollama로 돌리기**: 번역 코어는 버전 경로 포함 base URL이 필요하므로
  OpenAI 호환 `/v1` 경로를 지정한다 — 로컬: `OPENAI_BASE_URL=http://localhost:11434/v1`
  + `OPENAI_MODEL=qwen3:8b`, 컨테이너: `http://ollama:11434/v1`
  (compose.ollama.yaml overlay의 주석 예시 참조).

## 18. PyMuPDF 프로세스 격리 + 업로드 복잡도 게이트

### 18.1 왜 별도 프로세스인가

MuPDF C 호출은 GIL을 쥔 채 돌고 중간에 멈출 지점이 없다. 서버 프로세스 안에서 돌리면:

- 수 KB짜리 중첩 Form XObject PDF 하나가 단일 OCR 워커를 영구히 점거하고, GIL 때문에
  `/api/health`·취소까지 멈췄다 — 재시작만이 복구 수단이었다.
- 평면 path 10^7개 문서의 분석(`get_drawings`)이 컨테이너 메모리 상한을 넘겨 서버째 OOM-kill되고,
  대기 중이던 다른 사용자의 잡까지 error가 됐다.
- 번역 PDF 빌드 동안 같은 프로세스의 OCR 디코드가 굶었다 — MLX·torch 디코드 루프도 토큰마다
  GIL이 필요하다. 실측(M4 Max, MLX bf16, 16쪽): 서버 안 빌드와 겹치면 디코드 70 tok/s·OCR
  275.6초, 워커 프로세스로 옮긴 뒤 빌드가 OCR 내내 돌아도 291 tok/s·64.7초(빌드 없음 289 tok/s·
  65.8초). 서버 프로세스의 순수 파이썬 루프는 빌드 스레드가 있으면 0.53–0.57배로 느려졌다(워커
  빌드면 0.99배).
- 빌드 스레드 N개는 코어 하나를 나눠 쓸 뿐이었고(가속비 1.00), 한 프로세스에서 MuPDF를 여러
  스레드로 쓰는 것은 업스트림 비지원이다(손상 PDF 2스레드에서 SIGSEGV 재현).

### 18.2 워커 풀 (`pipeline/pdf_worker.py`)

spawn 방식 상주 워커 프로세스가 이름 붙은 작업(`'모듈:함수'`)을 실행한다. 인자·결과는 피클로
오간다. 풀 셋, 모두 처음 쓸 때 띄운다:

| 풀 | 워커 수 | 작업 |
|---|---|---|
| `ocr` | 1 | OCR 입력 페이지 렌더, 충실도 분석·재처리 채점, 텍스트 레이어 복구, 페이지 정렬 텍스트, 병합 때 폰트 실측 주입, textlayer 엔진 추출 |
| `export` | `PDF_EXPORT_MAX_CONCURRENT`(0 이하면 min(8, CPU)) | 번역·대조 PDF 빌드, facsimile 래스터, API 레이아웃 폰트 백필(백그라운드 스레드, `pool_scope('export')`) |
| `probe` | 2 | 업로드 검증(`probe_pdf` + 복잡도 게이트) |

- 풀을 셋으로 나눈 이유: 업로드가 15분짜리 빌드나 적대적 OCR 페이지 뒤에 줄서지 않게 한다.
  빈 워커를 기다리다 넘치면 `PdfWorkerBusy` — 업로드는 503 + `Retry-After: 5`(§5).
- 앱 lifespan 종료와 atexit에서 풀을 멈춘다(쉬는 워커는 정상 종료, 작업 중이면 종료). 워커는
  작업 200건을 넘기거나 최대 RSS가 1.5 GB를 넘으면 반납 때 교체한다. 최대 RSS는 Linux에서
  워커 자신의 `/proc/self/status` VmHWM이고, 없으면 `ru_maxrss`를 쓰되 워커 기동 때 값을 넘은
  경우만 센다 — Linux `ru_maxrss`는 부모의 최댓값을 물려받아, 큰 서버 프로세스 아래서는 워커가
  작업마다 다시 떴다. 쉬는 워커를 회수하지는 않아
  서버 외에 최대 5개 정도(+ 자원 추적 프로세스)가 남는다(각 약 60–320 MB).
- **시간 상한·장애**: 상한 초과·워커 비정상 종료(SIGSEGV·OOM-kill 포함)·취소면 **그 워커만**
  끝낸다(SIGTERM, 1초 뒤 SIGKILL). 다음 호출이 새 워커를 띄우므로 서버 프로세스와 다른 잡은
  영향이 없다. 오류 타입: `PdfWorkerTimeout`, `PdfPageQuarantined`(`PdfWorkerTimeout` 하위),
  `PdfWorkerCrashed`, `PdfWorkerCanceled`, `PdfWorkerBusy`, `PdfWorkerRemoteError`.
- **페이지 단위 작업**: 페이지마다 따로 작업이라 페이지마다 자기 시간 상한(`PDF_PAGE_TIMEOUT_S`,
  기본 60초)을 갖는다(25쪽 잡이면 작업 약 81개 — 작업당 수 ms의 프로세스 통신이 더해질 뿐
  verify_e2e 시간은 그대로다). 워커는 마지막으로 연 문서를 (경로, inode, 크기, mtime) 키로 캐시하고
  30초 쉬거나 작업 오류가 나면 닫는다.
- **격리(quarantine)**: 상한을 넘기거나 워커를 죽인 (파일, 페이지)는 서버 프로세스가 기억해 같은
  잡의 다음 단계가 상한을 또 기다리지 않고 곧바로 건너뛴다 — 충실도는 그 페이지를 `timed_out`
  으로 표시하고 참고를 남기며, 정렬은 빈 텍스트로, 폰트 주입은 그 페이지만 스탬프하고, 텍스트
  복구는 None이다. 이 기억은 서버를 재시작하면 사라진다.
- **렌더 규칙**: 페이지 렌더가 상한을 넘으면 흰 페이지 + 경고
  `{k}/{n}페이지가 렌더 시간 상한({t}초)을 넘어 흰 페이지로 대체했습니다 (…) (PDF_PAGE_TIMEOUT_S)`.
  한 문서에서 세 번이면 잡을 끝낸다(`페이지 N개의 렌더가 시간 상한(…)을 넘어 처리를 중단했습니다`).
  facsimile에서는 `PdfExportError`가 돼 좌표 텍스트 재조판으로 폴백한다. 렌더 중 취소·삭제는
  그 페이지를 그리던 워커를 바로 끝낸다(`render_pdf_pages(…, should_cancel=)`).
- **빌드**: 번역·대조 PDF 빌드는 export 워커에서 `PDF_EXPORT_BUILD_TIMEOUT_S`(기본 900초) 안에 돈다
  (실측 약 0.5초/쪽). 시간 초과·비정상 종료는 `PdfExportError`(409)다. 잡 락과 캐시 판정은 서버
  프로세스에 남는다(§5 `/pdf`). 실서버 확인: OCR·백필·빌드·대조 빌드·facsimile·document.html
  내내 서버 프로세스는 `_mupdf`를 한 번도 로드하지 않았다(lsof).
- **워커 안에서는 inline**: 워커 안에서 다시 이 모듈을 거치는 호출(빌드 안의 폰트 주입 등)은 새
  프로세스를 띄우지 않고 그 자리에서 돈다.
- **실행 모드** `PDF_WORKER_MODE`: `process`(기본) | `inline`(호출 스레드에서 바로 실행 — 상한·
  격리 없음, 기동 WARNING). 운영 노브가 아니다 — 테스트 세션이 inline으로 켜고(많은 테스트가
  pymupdf 내부를 monkeypatch한다) verify_e2e 하네스는 지운다. 기동 INFO 한 줄이 모드·풀 크기·
  상한·게이트 값을 남긴다.

### 18.3 워커 프로세스

- 이 모듈과 작업 모듈만 임포트한다(pdf·fidelity·pdf_fonts·pdf_export·engine.textlayer — torch·
  mlx·app.main·설정/LLM 계층·httpx는 끌어오지 않는다. `app.engine`은 `build_engine`을 PEP 562로
  지연 로드한다). 부모의 `__main__`도 다시 실행하지 않는다 — spawn 준비 데이터에서 그 항목을
  뺀다(CPython 내부 `multiprocessing.spawn.get_preparation_data`를 감싼다. 사라지면 기본 동작으로
  돌아가 런처 모듈(uvicorn 등)을 함께 올릴 뿐 동작은 한다).
- SIGINT는 무시하고(개발 서버 Ctrl+C는 부모가 정리) 로그는 프로세스 이름(`pdf-ocr-1` 등)을 붙여
  stderr로 낸다.
- **자격 증명 제거**: 이름에 `KEY`·`TOKEN`·`SECRET`·`PASSWORD`·`CREDENTIAL`이 든 환경 변수(번역·
  Q&A API 키, HF 토큰)를 기동 즉시 `os.environ`에서 지운다 — 워커가 띄우는 자식 프로세스에는
  키가 가지 않는다. 다만 커널이 보관하는 처음 환경 블록(`/proc/self/environ`)과 같은 사용자가
  읽을 수 있는 서버 프로세스의 환경에는 남으므로, 워커가 장악됐다면 키는 노출된 것으로 보고
  교체한다.
- **임시 파일**: 부모가 워커마다 `mkdtemp`로 `$TMPDIR/pdfocr-worker-<풀>-<서버 pid>-<임의>`(0700)를
  새로 만들어 넘기고, 워커는 자기 소유의 0700 디렉터리만 쓴다(예측 가능한 이름을 남이 먼저
  만들어 두는 경로 차단). 종료당한 워커의 디렉터리는 부모가 지우고, 이전 서버가 SIGKILL로 남긴
  것은 첫 풀 생성 때 쓸어 낸다 — 우리 소유의 실제 디렉터리만 지우고 심볼릭 링크는 따라가지 않는다.
- **자폭 알람**: 작업마다 `SIGALRM`을 상한 + 10초로 건다 — 기본 동작이 커널의 프로세스 종료라
  GIL이 필요 없어, 서버가 SIGKILL로 사라져도 적대적 작업이 CPU를 영원히 태우지 않는다
  (`SIGALRM`이 있는 플랫폼에서만).
- **Linux 전용**: `/proc/self/oom_score_adj`를 1000으로 올려 메모리 압박 시 커널이 서버 대신
  워커를 고르게 하고, `PDF_WORKER_MEM_LIMIT_MB`>0이면 `RLIMIT_AS`도 건다(macOS 커널은
  `RLIMIT_AS`를 강제하지 않아 건너뛴다). x86_64 이미지를 에뮬레이션(Rosetta)으로 돌리면
  `RLIMIT_AS`가 에뮬레이터의 가상 메모리 예약(~1.5–1.8GB)까지 세어 워커가 기동 직후 죽으므로 이
  상한을 쓰지 않는다(Apple Silicon은 arm64 이미지).
- **원인 가르기**: 업로드 검증 중 워커가 죽으면 같은 풀에서 빈 작업(`_selftest_task` — PyMuPDF
  임포트·빈 문서)을 한 번 돌려 본다. 그것까지 죽으면 입력이 아니라 환경 문제라
  `PdfWorkerUnavailable` → **500**(서버 설정 문제 안내, 오류 로그)이고, 통과하면 예전처럼 그 PDF를
  **400**(손상·미지원)으로 거부한다. 예전에는 기동 직후 죽는 워커 환경에서 모든 업로드가 사용자
  파일 탓의 400이었다.
- 워커는 **자원·장애 격리 경계이지 권한 샌드박스가 아니다** — 같은 사용자·같은 파일 시스템이고,
  서버는 워커가 돌려준 피클 결과를 신뢰한다.
- 두 스레드가 같은 워커를 동시에 멈춰도 안전하다(종료·`close` 경합 수정).

### 18.4 업로드 복잡도 게이트 (`pipeline/pdf_complexity.py`)

`probe_pdf`(probe 워커, `PDF_PAGE_TIMEOUT_S` 안)가 큐에 넣기 **전에** 페이지마다 렌더 없이 잰다:

- **콘텐츠 바이트**: 페이지 콘텐츠 스트림과 거기서 도달하는 Form XObject·주석 외형(AP)·타일링
  패턴·Type3 글리프 스트림의 **압축 해제** 길이 합(서로 다른 스트림은 한 번씩). 스트리밍으로 풀다가
  상한에서 멈춰 압축 폭탄을 끝까지 풀지 않는다. 상한 `PDF_MAX_PAGE_CONTENT_MB`(기본 64).
- **펼친 XObject 호출 수**: 콘텐츠의 `/이름 Do`를 리소스로 해석해 Form이면 그 안의 호출을 곱해
  더한다(DAG 메모·포화 덧셈 — 10^12도 즉시 계산). 이미지 Do도 한 번의 그리기로 센다. 상한
  `PDF_MAX_PAGE_XOBJECT_CALLS`(기본 2,000,000). Form 중첩이 64단계를 넘으면 거부한다.
- **MuPDF와 같은 이름 해석**: `#xx` 이스케이프를 풀되 `#00`은 그대로, 콘텐츠 이름은 255바이트에서
  자르고, 리소스 사전 키는 PyMuPDF가 돌려주는 UTF-8·surrogateescape 문자열을 파이썬에서 순회해
  맞춘다 — 예전에는 비ASCII 이름(`/Fm#E9`)의 콘텐츠·리소스 디코딩이 달라 호출도 바이트도 0으로
  셌다. `/이름 Do` 짝은 연산자 뒤 첫 이름 피연산자이고 주석·문자열이 끼지 않으며 2번째 이후
  콘텐츠 스트림의 이어진 줄이 아닐 때만 확실하다고 본다. 확실하지 않은 Do와 리소스에서 못 찾은
  이름은 그 리소스 사전에서 가장 비싼 XObject로 센다(과소 계수 0 — 20,000개 무작위 문서를
  MuPDF 실제 그리기 수와 대조).
- **순환 Form**: MuPDF는 실행 중인 Form을 다시 실행하지 않는다. 순환을 끊은 결과는 호출 맥락에
  따라 달라지므로 맥락과 무관할 때만 메모하고, 메모할 수 없는 순환이 너무 얽혀 있으면
  `N페이지의 Form XObject가 서로를 부르는 순환이 너무 얽혀 있습니다 — 처리할 수 없는 PDF입니다`
  (400)로 거부한다.
- 넘으면 설정 이름을 담은 한국어 사유로 400이다. 각 상한은 0이면 그 검사를 끈다(호출 수만 세려면
  스트림을 풀어야 하므로 콘텐츠 검사를 꺼도 256 MB까지만 푼다).
- **세지 않는 비용**: Type3 글리프 반복, 타일링 반복, 거대 이미지·셰이딩 디코드 — 페이지 시간
  상한이 받친다. 문자열·주석 안의 `Do`까지 세는 쪽(과대 추정 = 안전한 쪽)으로 틀린다.

보정(M4 Max 실측):

| 입력 | 페이지당 크기 | 페이지당 호출 | 결과 |
|---|---|---|---|
| 리포 표본 PDF | 최대 0.31 MB | 최대 129 | 통과 |
| 중첩 폭탄 bomb_3 / bomb_4 / bomb_5 | — | 10^3 / 10^4 / 10^5 | 통과, 렌더 0.006–0.39초 |
| 중첩 폭탄 bomb_12 (3,023바이트, 10단계 × 10) | — | 1.1×10^12 | 1 ms 안에 거부 |
| 평면 path 10^6개 | 33 MB | — | 통과, 렌더 1.5초 · 벡터 추출 3.4초/1.66 GB |
| 마커 산점도 10^6개 | — | 10^6 | 통과, 렌더 3.6초 |
| 1 GB flate 폭탄 | — | — | 35 ms에 거부, 추가 메모리 없음 |
| 감사의 손상 PDF들 | — | — | 통과하고 정상 완료 |

실서버(fake 엔진)에서 bomb_12는 10 ms, 1 GB flate 폭탄은 50 ms에 400이었다. 게이트를 끄고 페이지
상한을 10초로 둔 실험에서 정상-폭탄-정상 문서는 10.3초에 done(2쪽은 흰 페이지 + 경고), 폭탄
한 쪽 문서는 10.0초에 error였고 health는 내내 1–2 ms로 답했다.

**오탐 가능성**: 페이지당 압축 해제 콘텐츠 64 MB나 펼친 호출 200만 회를 넘는 정상 페이지(아주
조밀한 CAD 도면·지도)는 400으로 거부된다 — 사유에 적힌 설정을 올리거나 0으로 끈다.
