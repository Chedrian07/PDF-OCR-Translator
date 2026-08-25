# Unlimited-OCR PDF → Markdown 변환 서비스 — 아키텍처 & API 계약

> 이 문서는 본 프로젝트의 **단일 진실 공급원(SSOT)** 입니다.
> 백엔드/프론트엔드/네이티브 모듈은 모두 이 문서의 계약을 따릅니다.

## 1. 개요

웹에서 PDF를 업로드하면 [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)
(3.3B MoE VLM, DeepSeek-OCR 계열, MIT)로 파싱하여 **이미지(figure)까지 포함된 Markdown**으로
변환해 주는 셀프호스팅 애플리케이션.

- 디바이스 백엔드: **CPU**, **CUDA**, **Metal**(torch MPS, Apple Silicon) — 모두 구현 완료
- 배포: `docker compose up` 한 번으로 실행 (CPU 기본, GPU는 `docker compose up ocr-cuda`).
  Metal은 컨테이너 GPU 패스스루가 없어 **로컬(uv) 실행 전용**
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
| 로딩 | 벤더링된 모델 코드(`backend/app/vendor/unlimited_ocr/`)의 `UnlimitedOCRForCausalLM.from_pretrained()` — `trust_remote_code` 불필요 |
| 단일 이미지 | `model.infer(tokenizer, prompt='<image>document parsing.', ...)` — gundam(1024/640/crop) 또는 base(1024/1024) |
| PDF/멀티페이지 | `model.infer_multi(tokenizer, prompt='<image>Multi page parsing.', image_files=[...], image_size=1024, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024, save_results=True)` |
| 페이지 구분 | 출력 텍스트에 `<PAGE>` 마커 |
| 이미지(figure) 추출 | 모델이 `<|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` (0–999 정규화 좌표) 출력 → 원본 페이지에서 크롭하여 `{out}/images/page_{i}_{k}.jpg` 저장, 마크다운에는 `![](images/page_{i}_{k}.jpg)` 치환 |
| 레이아웃 시각화 | 페이지별 `result_with_boxes_{i}.jpg` 저장 (GIF 데모의 박스 오버레이) |
| 고정 의존성 | torch==2.10.0, torchvision==0.25.0, transformers==4.57.1, pymupdf==1.27.2.2 등 (모델 README 기준) |
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

## 3. 디렉터리 구조

```
├── docker-compose.yml          # ocr-cpu(기본)·ocr-cuda(cuda)·ocr-ovis+ovisocr2(ovis)·ocr-paddle+paddleocr-vl(paddle)
├── compose.ollama.yaml         # 선택 overlay — Ollama 컨테이너 추가 (§8)
├── Makefile                    # setup/dev/test/coverage/e2e/e2e-mock/verify-e2e/docker-*
├── .env.example                # 환경변수 템플릿 — 실제 키는 .env에 (커밋되지 않음)
├── README.md · SECURITY.md     # 사용법 / 보안·노출 정책 (§14와 정합)
├── docs/
│   ├── ARCHITECTURE.md         # 이 문서 (SSOT)
│   ├── OCR_ENGINE_PROTOCOL.md  # sidecar 프로토콜 v1 계약
│   └── CUDA_5070TI_MULTI_OCR_PLAN.md · OVISOCR2_CUDA_5070TI.md
│       · PADDLEOCR_VL_BLACKWELL_5070TI.md · OCR_BENCHMARK.md · AUDIT/ROADMAP 문서
├── backend/
│   ├── pyproject.toml          # uv 프로젝트, extras: cpu / cu129 / metal
│   ├── uv.lock                 # CI는 `uv sync --locked`로 lock 드리프트를 실패시킨다 (§11)
│   ├── Dockerfile              # ARG TORCH_VARIANT=cpu|cu129, tesseract 포함, 비루트(uid 1000)
│   ├── e2e_mock_app.py         # 브라우저 E2E 전용 진입점 (Q&A 라우터만 메모리 mock으로 교체)
│   ├── app/
│   │   ├── main.py             # FastAPI 앱 팩토리 + TrustedHost + 정적 프론트엔드 서빙
│   │   ├── config.py           # 환경변수 설정(Settings) + .env 로더
│   │   ├── api.py              # REST + SSE 라우트 + 남용 방어(레이트리밋·동시 상한, §5)
│   │   ├── jobs.py             # Job/JobStore + 단일 워커 큐 + SSE 브로커 + TTL GC (§15)
│   │   ├── qa.py               # 페이지 텍스트 컨텍스트 추출 (result.md 페이지 인덱스, §17)
│   │   ├── native_ops.py       # uocr_native 로더 + 순수 파이썬 폴백
│   │   ├── engine/
│   │   │   ├── base.py         # OCREngine 프로토콜 + EngineCapabilities
│   │   │   ├── registry.py     # 디바이스·엔진 선택 (unlimited/fake/textlayer/ovisocr2/paddleocr_vl)
│   │   │   ├── unlimited.py    # 실모델 엔진 (벤더링 코드 사용)
│   │   │   ├── fast_decode.py  # 커스텀 그리디 디코드 루프 (OCR_FAST_DECODE)
│   │   │   ├── repetition.py   # 의미 반복·페이지 출력 폭주 감지 (StoppingCriteria)
│   │   │   ├── textlayer.py    # 텍스트 레이어 우선 + Tesseract 폴백 엔진 (§16)
│   │   │   ├── sidecar.py      # sidecar 엔진 (HTTP client + materializer 연결)
│   │   │   └── fake.py         # 테스트/데모용 가짜 엔진 (torch 불필요)
│   │   ├── sidecar/            # sidecar 공통 계층 (모델 독립)
│   │   │   ├── protocol.py     # 프로토콜 v1 스키마 + sanitize (OCR_ENGINE_PROTOCOL.md)
│   │   │   ├── client.py       # 동기 HTTP client (타임아웃/상한/취소/재시도)
│   │   │   └── materializer.py # normalized 결과 → 기존 청크 산출물 규약
│   │   ├── pipeline/
│   │   │   ├── pdf.py          # PDF → 페이지 PNG (pymupdf)
│   │   │   ├── runner.py       # 잡 실행 오케스트레이션(렌더→청크 OCR→병합) + 실패 격리
│   │   │   ├── merge.py        # <PAGE> 분리, figure 리넘버링, result.md 페이지 경계 계약 (§4)
│   │   │   ├── layout.py       # raw_pages.json → layout.json 블록 파싱
│   │   │   ├── pdf_fonts.py    # 원본 텍스트 레이어의 실측 폰트 크기·굵기 주입
│   │   │   ├── render.py       # markdown → HTML (markdown-it-py) + document.html
│   │   │   └── pdf_export.py   # 레이아웃 보존 번역 PDF (단일/대조) — §5 /pdf
│   │   ├── translate/          # 번역 코어 (OCR 엔진·torch 무관) — §13
│   │   │   ├── engine.py       # run_translation (2단 패스·래더·캐시·state.json)
│   │   │   ├── client.py       # OpenAICompatClient (chat/responses 협상·재시도)
│   │   │   ├── segment.py      # md/layout → 유닛 분해·재조립·reconcile
│   │   │   ├── masking.py      # 플레이스홀더 마스킹/복원 + 출력 검증(looks_untranslated)
│   │   │   ├── glossary.py     # 문서 용어집, prompts.py # 프롬프트 SSOT
│   │   │   ├── types.py        # TranslateConfig · cache_key · PROMPT_V
│   │   │   └── data/seed_ko.json
│   │   ├── llm/                # Q&A 공급자 계층 — providers.py + validate.py (§17)
│   │   └── vendor/unlimited_ocr/   # 벤더링 모델 코드 (MIT) + PROVENANCE.md
│   ├── tools/translate_eval.py # 번역 품질 평가 CLI
│   └── tests/
├── services/                   # GPU sidecar 컨테이너 (비루트 uid 1000)
│   ├── ovisocr2/               # app/(main·model·parser·config) + Dockerfile + tests
│   └── paddleocr_vl/           # app/(main·model·adapter·config) + Dockerfile + tests
├── native/                     # C++ pybind11 모듈 (uocr_native)
│   ├── pyproject.toml          # scikit-build-core
│   ├── CMakeLists.txt
│   ├── src/uocr_native.cpp
│   └── tests/test_parity.py
├── frontend/                   # 정적 SPA (빌드스텝/외부 의존성 0)
│   ├── index.html · styles.css · app.js · layout-fit.js
│   ├── vendor/katex/           # 로컬 번들 (외부 CDN 금지 — §10)
│   └── tests/                  # node --test 단위 + tests/e2e/(ui, mock-full-flow)
└── scripts/
    ├── make_sample_pdf.py      # 텍스트+표+차트이미지 포함 샘플 PDF 생성
    ├── smoke_e2e.sh            # compose 기동 → 업로드 → 결과 검증
    ├── verify_e2e.py           # 실 PDF 전 구간 검증 하네스 (= make verify-e2e, §11)
    ├── mock_llm.py             # OpenAI 호환 목 서버 (결함 주입 모드 포함)
    ├── benchmark_ocr_engines.py · check_cuda_environment.py
    └── smoke_ovisocr2_5070ti.py · smoke_paddleocr_vl_5070ti.py
```

## 4. 처리 파이프라인

```
업로드(PDF) ──► JobStore(queued) ──► 워커(단일 스레드)
  1. render : pymupdf로 페이지별 PNG (RENDER_DPI, 기본 200) → pages/page_%04d.png
  2. ocr    : PAGES_PER_CHUNK(기본 8)개씩 infer_multi() 호출
              - 각 청크는 work/chunk_%02d/ 를 output_path로 사용
              - 커스텀 streamer가 토큰 델타를 SSE 큐로 전달
              - StoppingCriteria로 취소(cancel), rolling 반복, 페이지별 문자·토큰 상한 지원
  3. merge  : 청크 산출물 병합
              - <PAGE> 마커 분리 → 페이지 단위 마크다운
              - work/chunk_*/images/page_{i}_{k}.jpg → images/p{글로벌페이지:04d}_{k}.jpg 리네임
              - 마크다운 내 ![](images/page_{i}_{k}.jpg) 참조를 새 경로로 재작성
              - result_with_boxes_{i}.jpg → layout/page_{글로벌:04d}.jpg
              - 페이지 사이 PAGE_SEPARATOR(기본 "\n\n---\n\n")로 join → result.md
  4. done   : meta.json 갱신, SSE done 이벤트
```

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

`layout.json`도 같은 이유로 청크의 **모든** 페이지를 채운다 — raw_pages.json이 없거나
페이지 수가 모자라면 빈 블록 페이지로 메워 result.md의 페이지와 1:1을 유지한다.
좌표 데이터를 한 번도 받지 못한 잡(figure_only 엔진)은 아예 `layout.json`을 만들지
않아 `has_layout=false`로 남는다.

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
  않아 어느 경로에서도 정리되지 않았다.
- **sidecar 재시작/모델 재로드 대기**: sidecar 엔진은 페이지 요청이 `SidecarUnavailableError`
  (HTTP 503·연결 끊김)로 실패하면 health 캐시를 무효화하고 `wait_until_ready()`로
  컨테이너 복귀를 기다린 뒤 **그 페이지만 1회 재시도**한다. 기다리지 않으면 재기동 +
  모델 로드 시간 동안의 페이지가 전부 플레이스홀더로 확정된다. 반면
  `SidecarTimeoutError`는 provider가 아직 그 페이지를 추론 중일 수 있어 여기서
  재요청하지 않고(같은 페이지를 GPU에서 두 번 돌리게 된다) 상위 runner의 청크 재시도에 맡긴다.
  `loaded` 판정도 `status=="ok" and model_loaded`로 좁혀, sidecar가 엔진 사망(wedge)을
  `status="error"`로 신고하면서 `model_loaded=True`를 유지하는 경우를 놓치지 않는다.
- **실패 격리**: 렌더에서 한 페이지가 깨지면 흰색 페이지로 대체하고 계속한다. multi 생성의
  rolling 반복/페이지 상한 초과는 같은 multi를 재시도하지 않고 즉시 페이지별 single로 격리한다.
  single에서도 같은 문제가 나거나 일반 오류 재시도까지 실패하면 원본 PDF의 내장 텍스트 레이어를
  plain-text Markdown으로 복구하고, 텍스트 레이어도 없을 때만 실패 플레이스홀더를 넣는다. 내역은
  meta.json `warnings`에 남으며 전 청크/전 페이지가 끝내 복구되지 못한 경우만 error, 취소는 그대로
  전파한다.

### 잡 디렉터리 레이아웃 (`{DATA_DIR}/jobs/{job_id}/`)

```
source.pdf                  # 업로드 원본
meta.json                   # 상태/진행/파라미터 (재시작 시 복원)
pages/page_0001.png ...     # 렌더된 입력 페이지 (1-based)
work/chunk_00/ ...          # 모델 원시 출력 (실행 중에만 존재 — 터미널 마감 시 자동 삭제, §15)
result.md                   # 최종 병합 마크다운
images/p0001_0.jpg ...      # figure 크롭 (글로벌 페이지 번호, 1-based)
layout/page_0001.jpg ...    # 레이아웃 박스 오버레이
```

## 5. REST / SSE API 계약 (v1)

모든 경로는 `/api` 프리픽스. 프론트엔드는 같은 오리진에서 서빙되므로 CORS 불필요.

### 남용 방어 — 429 + Retry-After (QA·translate)

이 서비스는 인증이 없고 compose 기본 바인딩이 `0.0.0.0`이다(§8·§14). 같은 네트워크의
누구나 `POST /qa`로 운영자의 유료 LLM 키를 소진하거나 200페이지 번역을 반복 트리거할 수
있으므로, 인증을 새로 만드는 대신 **비용이 드는 두 라우트에만** 프로세스 내
슬라이딩 윈도우(60초) 레이트리밋 + 동시 실행 상한을 둔다 (`api.py::_AbuseGuard`).

| 라우트 | 레이트리밋 키 | 상한 초과 응답 |
|---|---|---|
| `POST /jobs/{id}/qa` | `qa:job:{id}` · `qa:ip:{client}` | 429 + `Retry-After`(남은 윈도우 초) |
| `POST /jobs/{id}/qa` (동시 실행) | 전역 슬롯 | 429 + `Retry-After: 5` |
| `POST /jobs/{id}/translate` | `translate:job:{id}` · `translate:ip:{client}` | 429 + `Retry-After`(남은 윈도우 초) |
| `POST /jobs/{id}/translate` (동시 번역 수) | 실행 중 번역 태스크 수 | 429 + `Retry-After: 30` |

- 조정 변수(§7): `QA_RATE_LIMIT_PER_MIN`(30)·`QA_MAX_CONCURRENT`(4)·
  `TRANSLATE_RATE_LIMIT_PER_MIN`(12)·`TRANSLATE_MAX_ACTIVE`(4). **0 이하면 해당 상한 비활성**이며,
  정수가 아닌 값은 500 대신 기본값으로 강등하고 경고 로그만 남긴다.
- 가드는 앱 상태에 지연 생성되고 상한은 프로세스 단위다(외부 저장소·인증 없음).
  기본값은 1인 로컬 사용을 방해하지 않는 수준으로 잡혀 있다.

### GET /api/health
```json
{
  "status": "ok",
  "engine": "unlimited",            // unlimited | fake | textlayer | ovisocr2 | paddleocr_vl
  "device": "cuda",                 // cpu | cuda | metal
  "dtype": "bfloat16",
  "model_id": "baidu/Unlimited-OCR",
  "model_loaded": true,             // false면 첫 잡에서 로딩
  "model_load_error": null,         // model_loaded=false일 때만 마지막 로드 실패 사유
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",  // cpu면 null
  "native_ops": true,               // C++ 모듈 사용 여부
  "worker_alive": true,             // OCR 워커 스레드 생존 여부 — false면 잡이 영원히 queued
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
  "provider_health": null           // sidecar 엔진만: {status, runtime, version,
                                    //  model_loaded, gpu_total_mb, gpu_free_mb}
                                    //  sidecar가 죽어도 health 자체는 200 —
                                    //  {status:"unreachable", error:"…"}로 구분
}
```

### POST /api/jobs — PDF 업로드
- `multipart/form-data`: `file`(필수, PDF), `mode`(`multi`|`per_page`, 기본 `multi`),
  `dpi`(72–400, 기본 200). 페이지 상한은 서버의 `MAX_PAGES`로 일괄 적용한다.
- 202 → `{"job_id": "j_1a2b3c4d5e6f", "status": "queued"}`
- 400(비PDF/손상), 413(MAX_UPLOAD_MB 초과)

### GET /api/jobs — 잡 목록 (최신순, 최대 50)
```json
{"jobs": [ { …GET /api/jobs/{id}와 동일 키… } ]}
```
- **주의**: 목록은 `include_files=False`로 직렬화한다 — `result`의
  `images`/`layouts`/`pages` 배열은 **항상 빈 배열**이다(키는 유지). 폴링마다
  잡×디렉터리를 전수 스캔하던 비용을 없앤 것으로, 실제 파일 URL이 필요하면
  단건 `GET /api/jobs/{id}`를 쓴다.

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
  "warnings": [],                   // 실패 격리·품질 경고 누적 (§4)
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
    "has_layout": true              // layout.json 유무 — false면 /layout·/pdf가 404/409
  }
}
```
- `queue_position`은 **워커 큐 제출 순번(submit_seq)** 기준이다. 제출은 업로드 본문
  수신 완료 직후라 생성 순서와 어긋날 수 있어(큰 파일을 먼저 올려도 작은 파일이 먼저
  제출된다) 생성 순서 대신 실제 처리 순서를 반영한다.

### GET /api/jobs/{id}/events — SSE
- `Content-Type: text/event-stream`, `retry: 3000`, 15초마다 `: ping` 주석
- 접속 시 현재 상태 스냅샷(progress) 1회 즉시 발행, 종료 잡이면 done/error 즉시 발행
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

    **토큰 스트림 문법 (실캡처로 확정, frontend/tests/fixtures/*.sse.txt):**
    각 청크의 스트림은 `<PAGE>` 마커로 **시작**한다 — 마커는 "지금 시작하는 페이지의 선언"이다.
    `청크k 스트림 = <PAGE> + page(start_k) 내용 + <PAGE> + page(start_k+1) 내용 + …`
    청크 시작 직전에 `progress(phase=ocr, current_page=start_k, chunk=k)`가 먼저 발행되므로,
    **각 청크의 첫 마커는 이미 선언된 페이지의 재확인(no-op)** 이고 이후 마커만 +1이다.
    블록 문법: `<|det|>label [x1,y1,x2,y2]<|/det|>텍스트…` (label: title/text/table/equation/
    image/page_number 등, 좌표 0–999 정규화) 또는 `<|ref|>label<|/ref|><|det|>[[…]]<|/det|>`.
    표는 블록 텍스트 안에 HTML `<table>`로 온다.
  - `event: done`     `data: {"markdown_url":"...","archive_url":"..."}`
  - `event: error`    `data: {"message":"..."}`  (취소 시 `"canceled": true` 포함)

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
  layout.json 없으면 404 (프론트는 탭에서 안내 문구 표시)

### GET /api/jobs/{id}/page/{page}?lang=ko
- 리더용 최종 페이지 PNG. 원문은 `pages/`, 번역은 `export.{lang}.pdf`를 잡 DPI로
  렌더한 `rendered/{lang}/` 캐시를 반환한다(캐시 마커는 PDF 크기·mtime·DPI·페이지 수).
- 상태코드: **400** 미지원 lang · **404** 번역본 없음/페이지 번호가 layout에 없음/
  이미지 파일 없음 · **409** 내보내기 불가(`PdfExportError` — 입력 누락·손상).
  `layout.json`(또는 `layout.{lang}.json`)이 없는 잡은 원본 `pages/` PNG로 폴백한다.

### GET /api/jobs/{id}/outline?lang=ko
- layout의 `title` 블록을 페이지·레벨·텍스트 목록으로 반환한다.

### GET /api/jobs/{id}/alignment?page=N&lang=ko
- 원문 bbox와 같은 인덱스의 원문/번역 블록을 연결한다. 번역 페이지·블록 수,
  type, bbox 대응이 어긋나면 잘못된 매핑을 내보내지 않고 409를 반환한다.

### GET /api/jobs/{id}/viewer-manifest?lang=ko
- 전체 화면 논문 뷰어의 부트스트랩 계약. schema/artifact revision, 페이지 수,
  원문 이미지·번역 이미지·alignment·outline capability, 품질 경고와 URL 템플릿을
  작은 JSON으로 반환한다.
- 좌측 뷰어 기준면은 `source_page_template`을 사용해 항상 원문으로 고정한다.
  `translated_page_image`는 번역 PDF raster 캐시가 실제 준비된 경우에만 true다.
- `Cache-Control: private, no-cache`, `ETag`, `Vary: Authorization`을 제공하며
  일치하는 `If-None-Match`에는 304로 응답한다.

### GET /api/jobs/{id}/viewer/pages?start=N&limit=4&lang=ko&include=alignment
- 긴 문서의 인접 페이지 메타/좌표를 한 번의 layout 파싱으로 반환하는 제한 배치.
  `limit`은 1–16이며, 각 item은 원문 이미지 URL과 선택적 alignment를 포함한다.
  잘못된 범위/include는 422, 원문-번역 대응 불변식 위반은 409다.

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

### GET /api/jobs/{id}/pdf?lang=ko[&view=dual]
- 기본 `view=single`은 **레이아웃 보존 번역 PDF** (`{원본이름}.ko.pdf`, application/pdf)를 반환한다.
  `layout.json`(원문)과 `layout.{lang}.json`(번역)을 블록 단위로 비교해
  **내용이 실제로 바뀐 텍스트 블록만** 원본 PDF에서 리댁션(텍스트만 제거,
  이미지·그래픽 보존) 후 같은 자리에 번역 텍스트를 삽입한다
  (`pipeline/pdf_export.py`). 행·열 구조가 같은 HTML 표는 PDF 텍스트 검색으로
  셀 격자를 추정해 **셀별 번역**하고 벡터 격자선을 보존한다(최대 500셀).
  독립 수식·그림·참고문헌과 세로쓰기 블록은 원본 글리프를 유지하며, 일반
  텍스트 안의 단순 LaTeX 위첨자는 읽을 수 있는 평문(`mc^{2}`→`mc²`)으로 낮춘다.
- **표 격자 신뢰도 게이트**: 셀 격자 추정은 `(셀 사각형, grid_trusted)`를 함께 낸다.
  텍스트 레이어가 있는 표인데 원문 셀 검색이 **절반도 맞지 않으면**(`관측×2 < 텍스트 셀 수`)
  균등 분할 격자는 실제 열 폭과 무관하므로 — 번역문이 엉뚱한 셀에 찍히고 인접 셀 원문이
  리댁션된다 — 그 표는 **교체하지 않고 원문을 보존**하고
  `p{n}: 표 셀 격자 추정 실패(원문 검색 불일치) — 원문 표 보존` 경고를 남긴다.
  텍스트 레이어가 없는 스캔 표는 지울 원문이 없어 균등 격자를 그대로 신뢰한다.
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
- **이모지 2차 리댁션(예외적 IMAGE_REMOVE)**: macOS Quartz 산출 PDF는 컬러 이모지를
  (보이지 않는 텍스트 글리프 + 이미지 XObject) 이중으로 기록해 텍스트 리댁션만으로는
  이미지가 번역문 위에 남는다. 그래서 1차 리댁션 뒤 **좁은 조건을 모두 만족하는 소형
  래스터 인스턴스만** `images=IMAGE_REMOVE, graphics=NONE, text=NONE`의 별도 pass로
  지운다 — ① 교체 사각형에 완전히 포함(1pt 허용 오차) ② 면적이 그 사각형의 25% 이하
  ③ **긴 변이 `max(2×블록 폰트 크기, 20pt)` 이하** ④ 종횡비 0.5–2.0.
  ③의 절대 크기 상한이 없으면 넓은 블록 안의 인라인 그림·로고까지 지워지고,
  블록 rect 전체로 `IMAGE_REMOVE`를 걸면 걸친 그림에 흰 구멍이 난다.
- **CropBox 가드**: 아래 빈 영역 확장의 페이지 하단 기준선은 `page.mediabox`가 아니라
  `page.rect × derotation_matrix`(정규화)로 얻은 **CropBox 기준 표시 영역**이다.
  mediabox는 CropBox를 무시한 PDF 원좌표라 CropBox≠MediaBox 문서에서 실제 페이지
  하단보다 아래를 기준으로 삼고, 회전 페이지에서는 derotation 없이 쓰면 가로/세로가
  뒤바뀐다. 블록 bbox와 같은 비회전 내부 좌표계로 맞춘 값만 쓴다.
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
- 상태코드: 400 미지원 lang **또는 미지원 `view`**(single|dual 외) · 404 번역본 없음 ·
  409 미완료 잡 또는 좌표 레이아웃 없음(figure_only 엔진 — document.html 사용 안내) ·
  500 내보내기 실패(`PdfExportError`).
- 응답 헤더: `X-UOCR-PDF-Replaced`, `-Preserved`, `-Relocated`, `-Table-Cells`,
  `-Specialist-Preserved`, `-Warnings`. 모두 숫자만 담아 원문·경고 본문이 프록시
  메타데이터로 새지 않으며, 프런트 다운로드 토스트가 이를 요약한다.
- 캐시: 단일판 `job.dir/export.{lang}.pdf` + `export.{lang}.report.json`, 대조판
  `export.{lang}.dual.pdf` — 단일판은 `layout.{lang}.json`보다 오래되면 재생성하고,
  대조판은 원본·단일판보다 오래되면 재생성한다. 번역 완료 시 함께 무효화한다.
  리포트의 `format_version`이 현행 `PDF_EXPORT_FORMAT_VERSION`(현재 **4**)과 다르면
  캐시를 무시하고 재생성한다 — 내보내기 동작이 바뀐 사이클에서는 기존 export 캐시가
  전부 한 번 재생성된다.

### POST /api/jobs/{id}/cancel
- 실행/대기 중 잡을 **삭제 없이** 중단. 202 `{"job_id","status":"canceling"}`
  (이미 종료된 잡이면 현재 status 반환). 잡은 `canceled` 상태로 남고
  완료된 청크까지의 부분 결과는 /markdown 등에서 계속 접근 가능

### POST /api/jobs/{id}/render-preview
- 요청 본문(text/plain, ≤2MB)의 마크다운을 /html과 동일한 안전 렌더러로
  HTML 프래그먼트 렌더 (라이브 미리보기용 — 프론트가 정리한 스트림 텍스트를 debounce 전송)

### DELETE /api/jobs/{id}
- 실행 중이면 취소(cancel) 후 삭제, 완료면 디렉터리 삭제. 204
- 이 잡의 **실행 중 번역 스레드에도 cancel을 전파**한다 — 삭제된 디렉터리에
  유료 API 호출·파일 기록이 계속되지 않게 (번역 엔진의 state/캐시 기록은
  삭제 경합 시 FileNotFoundError를 무시하는 best-effort)

## 6. 디바이스 백엔드

| 백엔드 | 상태 | 선택 방법 | 비고 |
|---|---|---|---|
| CPU | ✅ 구현 | `OCR_DEVICE=cpu` | 기본 dtype float32 (`OCR_DTYPE`로 변경 가능) |
| CUDA | ✅ 구현 | `OCR_DEVICE=cuda` | bf16, cu129 휠, sm_89/sm_120 확인 |
| Metal | ✅ 구현 | `OCR_DEVICE=metal` (별칭 `mps`) | torch MPS. Apple Silicon 로컬 실행 전용 — Docker 불가. `uv sync --extra metal` |

`app/engine/registry.py`가 단일 진입점: 디바이스/엔진 이름 검증 후 엔진 생성.
CUDA/MPS 가용성 검증은 `UnlimitedEngine.load()` 시점(= 프리로드 스레드/첫 잡)에 수행되어
실패 사유가 `/api/health`의 `model_load_error`로 노출된다.

### Metal(MPS) 구현 노트

- 사용자 노출 디바이스명은 `metal`, torch 디바이스는 `mps`
  (`engine/unlimited.py`의 `torch_device_name()`이 매핑, health에는 `metal`로 표기)
- dtype `auto`: bf16 텐서 할당 프로브 성공 시 bfloat16(macOS 14+), 실패 시 float32 폴백.
  `OCR_DTYPE=float16`도 선택 가능(구형 macOS에서 속도용 — bf16 권장)
- 벤더 코드는 P1/P4 패치(`_autocast_ctx`, 파라미터 디바이스 추종)로 디바이스 중립.
  단, torch 2.10.0 MPS의 조용한 버그 2건을 회피하는 패치가 추가로 필요했다 (PROVENANCE.md):
  - **P11**: 브로드캐스트 마스크 `masked_scatter_` 오동작 → 이미지 임베딩 미주입 → 빈 출력
  - **P12**: `torch.autocast("mps", bf16)`가 로짓 오염 → 반복 루프. MPS에서는 autocast 미사용
    (가중치가 bf16이라 성능 동일)
- **디코드 성능**: MPS 디코드는 연산량이 아니라 커널 디스패치/호스트 동기화에 바운드된다.
  토큰당 오버헤드를 3중으로 줄였다 — 벤더 P18(MoE 단일 토큰 패스트패스, `OCR_MOE_FAST`로
  강제 on/off·기본 MPS 전용)·P19(rotary cos/sin 스텝 캐시로 스텝당 계산을 레이어 수회→1회)와
  `fast_decode.py`의 명시적 `position_ids`(attention_mask 제거 → prepare_inputs가 토큰마다
  마스크 전체에 돌리던 cumsum 커널 체인 제거). 셋 다 결과 비트 동일, 원본(비패스트패스) 경로 불변
- `PYTORCH_ENABLE_MPS_FALLBACK=1`을 엔진 생성 시 `setdefault` — 미구현 op는 CPU 폴백 (안전망)
- 청크(infer 호출) 종료마다 `torch.mps.empty_cache()` — 유니파이드 메모리 반환으로
  장문서 잡의 시스템 메모리 압박 완화
- 첫 청크는 Metal 셰이더 컴파일로 이후보다 느림 (정상)
- 메모리 상한 조정이 필요하면 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (torch 문서 참조 — 기본값 권장)
- `gpu_name`은 `sysctl machdep.cpu.brand_string` (예: "Apple M4 Max")

## 7. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OCR_DEVICE` | `cpu` | `cpu`\|`cuda`\|`metal` (`mps`는 `metal`의 별칭) |
| `OCR_DTYPE` | `auto` | `auto`(cuda→bf16, metal→bf16 또는 fp32 폴백, cpu→fp32)\|`bfloat16`\|`float16`\|`float32` |
| `OCR_ENGINE` | `unlimited` | `unlimited`\|`fake`\|`textlayer`(§16)\|`ovisocr2`\|`paddleocr_vl` (sidecar 둘은 `OCR_SIDECAR_URL` 필수) |
| `OCR_SIDECAR_URL` | (없음) | sidecar 엔진의 base URL — compose 프로필(ovis/paddle)이 자동 설정 |
| `OCR_SIDECAR_CONNECT_TIMEOUT_S` | `10` | sidecar 연결 타임아웃(초) |
| `OCR_SIDECAR_READ_TIMEOUT_S` | `600` | 페이지 1장 추론 대기 상한(초) |
| `OCR_SIDECAR_HEALTH_TIMEOUT_S` | `5` | sidecar health 대기(초) |
| `OCR_SIDECAR_MAX_RESPONSE_MB` | `20` | 응답 크기 상한 (response bomb 방어) |
| `OCR_SIDECAR_RETRIES` | `1` | 연결 수립 실패 재시도 횟수 (그 외 재시도는 runner 몫) |
| `OCR_REMOTE_PAGE_CONCURRENCY` | `1` | sidecar 페이지 동시 요청 수 = sidecar 엔진의 청크 크기 (16GB 단일 GPU는 1 권장) |
| `OCR_SIDECAR_MODEL_WAIT_S` | `900` | 잡이 sidecar 모델 준비를 기다리는 상한(초) — 최초 기동 창에 업로드해도 실패 대신 대기(취소 가능) |
| `MODEL_ID` | `baidu/Unlimited-OCR` | HF 모델 ID |
| `MODEL_REVISION` | `ee63731b…` | HF revision 고정 (README의 검증 커밋) |
| `PRELOAD_MODEL` | `1` | 기동 시 모델 로드 (0이면 첫 잡에서 lazy) |
| `DATA_DIR` | `data` (Docker `/data`) | 잡 저장소 루트 (`{DATA_DIR}/jobs`). config.py 기본은 상대 경로 `data`, Dockerfile ENV가 `/data`로 덮는다 |
| `HF_HOME` | `/data/hf` | HF 캐시 (Dockerfile ENV + compose 볼륨) |
| `RENDER_DPI` | `200` | 요청별 `dpi`로 오버라이드 가능 |
| `PAGES_PER_CHUNK` | `8` | infer_multi 청크 크기 |
| `MAX_PAGES` | `200` | 페이지 상한 |
| `MAX_UPLOAD_MB` | `100` | 업로드 상한 |
| `MAX_LENGTH` | `32768` | 생성 총 길이 상한 |
| `MAX_PAGE_OUTPUT_CHARS` | `16384` | `<PAGE>` 기준 페이지별 decoded 출력 문자 hard limit (single에도 동일 적용, 0 이하=비활성) |
| `MAX_PAGE_OUTPUT_TOKENS` | `6144` | 페이지별 생성 토큰 hard limit (fast decode는 최대 한 block만큼 정지 지연, 0 이하=비활성) |
| `PAGE_SEPARATOR` | `\n\n---\n\n` | 병합 시 페이지 구분자 |
| `OCR_CPU_THREADS` | `0` | CPU 백엔드 torch 스레드 수 (0=torch 기본) |
| `OCR_FAST_DECODE` | `1` | 커스텀 그리디 디코드 루프(cpu/cuda/mps 공용, 호스트 동기화 블록 배칭). `0`이면 HF generate 폴백 |
| `OCR_DECODE_BLOCK` | `8` | fast decode의 동기화 배칭 크기(토큰) — EOS를 블록 경계에서 확인 |
| `OCR_MOE_FAST` | (미설정) | MoE 단일 토큰 디코드 패스트패스 강제 on/off (`1`/`0`). 미설정 시 **MPS에서만 on** — 벤더 P18, 결과 비트 동일 |
| `OCR_MOE_FUSED` | (미설정) | CUDA MoE 융합 디코드(벤더 P17, 기본 **CUDA에서 on**) 킬스위치 — `0`이면 legacy 경로 완전 복원 |
| `OCR_NGRAM_HOST` | (미설정) | `1`이면 GPU/MPS에서도 no-repeat-ngram 배닝을 호스트(C++/파이썬) 티어로 강제 (절연 레버, `native_ops.py`) |
| `FAKE_DELAY` | `0.02` | FakeEngine 페이지당 지연(초) — 테스트/데모 전용 |
| `FRONTEND_DIR` | (미설정) | 정적 프론트엔드 경로 오버라이드 — 미설정이면 리포 상대 경로에서 탐색 |
| `OPENAI_BASE_URL` | (없음) | 번역 프로바이더 base URL. bare origin(`https://host`)이면 `/v1`을 자동 보완하고, 명시 경로는 그대로 사용. 미설정 시 번역 기능만 비활성(503) |
| `OPENAI_API_KEY` | (없음) | **번역 전용** 프로바이더 API 키 (로컬 서버는 생략 가능). Q&A는 이 키를 쓰지 않는다 → `LLM_OPENAI_API_KEY` |
| `OPENAI_MODEL` | (없음) | 번역 모델 ID. `OPENAI_BASE_URL`과 함께 있어야 번역 활성화 |
| `TRANSLATE_MODEL` | `OPENAI_MODEL` | 번역 전용 모델 오버라이드 |
| `TRANSLATE_API_MODE` | `auto` | `auto`\|`chat`\|`responses` (auto: responses 시도 → 미지원 시 chat) |
| `TRANSLATE_CONCURRENCY` | `8` | 잡당 동시 번역 요청 수 (1–8) |
| `TRANSLATE_GLOBAL_CONCURRENCY` | `TRANSLATE_CONCURRENCY` | 여러 잡을 합친 프로세스 전체 실제 번역 HTTP 상한 (1–8) |
| `TRANSLATE_TIMEOUT_S` | `180` | 응답 읽기 타임아웃(초). 연결은 `min(10초, 이 값)`으로 별도 제한 |
| `TRANSLATE_MAX_RETRIES` | `3` | 연결 오류와 408/429/500/502/503/504 재시도 횟수 |
| `TRANSLATE_TEMPERATURE` | `0` | `none`이면 temperature 파라미터 자체 생략 |
| `TRANSLATE_MAX_TOKENS_PARAM` | `max_tokens` | `max_tokens`\|`max_completion_tokens`\|`none` |
| `OCR_CUDA_GRAPHS` | (CUDA on) | 디코드 스텝 CUDA Graph 캡처·리플레이 — 커널 launch 갭 제거. `0`으로 비활성. 실측(8p): 191s→57s, sm 33%→98% |
| `TRANSLATE_REASONING` | (미전송) | reasoning 모델 제어: `off`\|`low`\|`medium`\|`high`\|`xhigh`. reasoning 모델은 `off` 권장 — 실측 유닛당 37s→1.7s, 출력 토큰 ~1/40. effort별 요청 max_tokens: 8192/10240/20480/40960/81920 (미설정=8192) |
| `TRANSLATE_CONTEXT` | `1` | 직전 유닛 꼬리를 번역 문맥으로 제공. `0`이면 비활성 |
| `QA_RATE_LIMIT_PER_MIN` | `30` | `POST /qa`의 잡·IP별 60초 윈도우 상한 (0 이하=비활성, §5) |
| `QA_MAX_CONCURRENT` | `4` | 동시 처리 중인 Q&A 요청 수 상한 — 초과 시 429 `Retry-After: 5` |
| `TRANSLATE_RATE_LIMIT_PER_MIN` | `12` | `POST /translate`의 잡·IP별 60초 윈도우 상한 (0 이하=비활성) |
| `TRANSLATE_MAX_ACTIVE` | `4` | 동시에 실행 중인 번역 태스크 수 상한 — 초과 시 429 `Retry-After: 30` |
| `GPU_DEVICE` | `0` | (compose) CUDA_VISIBLE_DEVICES로 전달 — 두 번째 GPU는 `1` |
| `HOST`/`PORT` | `0.0.0.0`/`8000` | 컨테이너 내부 uvicorn 바인드 (Dockerfile CMD 고정값) |
| `BIND_HOST` | `0.0.0.0` | (compose) 호스트 쪽 포트 바인딩 주소 — **기본은 외부 노출**. 루프백 전용으로 되돌리려면 `127.0.0.1` (§8·§14) |
| `ALLOWED_HOSTS` | config.py `localhost,127.0.0.1` / **compose `*`** | Host 헤더 화이트리스트(콤마 구분) — DNS rebinding 방어, 포트는 비교 시 무시. compose는 외부 노출 기본과 정합을 위해 `*`(모든 Host 허용)을 넘긴다 (§14) |
| `OCR_CPU_MEM_LIMIT` / `OCR_CUDA_MEM_LIMIT` / `OCR_WEB_MEM_LIMIT` | `24g` / `16g` / `8g` | (compose) backend 서비스별 메모리 상한 (§8) |
| `OVIS_MEM_LIMIT` / `PADDLE_MEM_LIMIT` | `24g` / `24g` | (compose) sidecar 컨테이너 메모리 상한 |
| `JOB_TTL_DAYS` | `0` | 터미널 잡(done/error/canceled) 자동 GC 보존 일수 — `0`=비활성(기본, opt-in). 시작 시 1회 + 6시간 주기 (§15) |
| `OCR_LANGUAGES` | `eng+kor` | (textlayer) Tesseract 언어 조합 — `tesseract -l` 인자 (§16) |
| `NATIVE_TEXT_THRESHOLD` | `120` | (textlayer) 텍스트 레이어를 신뢰할 페이지당 최소 영숫자 수 — 미만이면 Tesseract 폴백 (§16) |
| `LLM_PROVIDER` | `openai-responses` | (Q&A) 기본 LLM 공급자: `openai-responses`\|`openai-chat`\|`ollama` (§17) |
| `LLM_REASONING_EFFORT` | `low` | (Q&A) 기본 reasoning effort: `default`\|`none`\|`minimal`\|`low`\|`medium`\|`high`\|`xhigh`\|`max` |
| `LLM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | (Q&A) 공식 `api.openai.com` 호스트만 허용 — 그 외 값은 기동 시 즉시 실패 (§17.3) |
| `LLM_OPENAI_API_KEY` | (없음) | **(Q&A 전용 OpenAI 키 — 번역용 `OPENAI_API_KEY`와 분리, 폴백 없음.)** 위 base URL이 공식 호스트로 고정돼 있어, 제3자 게이트웨이용 `OPENAI_API_KEY`를 재사용하면 그 키가 `api.openai.com`으로 전송된다. 미설정 시 `openai-*` 공급자는 `available:false`이고 `qa_available:false` (§17.3) |
| `LLM_OPENAI_RESPONSES_MODELS` | `gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol` | (Q&A) Responses 모델 선택지 (csv) |
| `LLM_OPENAI_CHAT_MODELS` | `chat-latest,gpt-5.6-luna,gpt-5.6-terra` | (Q&A) Chat Completions 모델 선택지 (csv) |
| `LLM_OPENAI_RESPONSES_MODEL` | `gpt-5.6-luna` | (Q&A) Responses 기본 모델 |
| `LLM_OPENAI_CHAT_MODEL` | `chat-latest` | (Q&A) Chat Completions 기본 모델 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | (Q&A) 로컬 Ollama 주소 — 루프백·`host.docker.internal`·`ollama`만 허용, 그 외 기동 시 즉시 실패 (§17.3). compose 컨테이너 기본값은 `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` | (Q&A) 기본 로컬 Ollama 모델 (`:cloud`/`remote_host` 모델은 차단) |
| `PDF_EXPORT_FONT` | (빈 값) | 번역 PDF 내보내기용 한글 폰트 파일 경로 — 비우면 시스템 폰트 → 내장 CJK 폴백 (§5 /pdf) |

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
- **공유 볼륨**: `hf-cache`(모델 가중치 ~6.7GB, 최초 1회 다운로드)와
  `ocr-data`(잡 결과)를 **네 backend 서비스가 모두 공유**한다 — 엔진(스택)을 바꿔도
  잡 이력이 남는다. 과거의 `ocr-ovis-data`/`ocr-paddle-data`는 더 이상 참조되지 않으며,
  그 안의 잡을 살리려면 한 번만 `ocr-data`로 복사한 뒤 볼륨을 지운다(compose 주석에 명령 있음).
- `ocr-cpu`는 프로필이 없어 `docker compose up` = CPU 서비스만 기동 (.env 불필요)
- GPU: `docker compose up -d ocr-cuda` — 서비스명을 명시하면 cuda 프로필이 자동 활성화
- **하드닝**: 전 서비스 `security_opt: no-new-privileges:true`. 4개 backend 서비스
  (ocr-cpu/ocr-cuda/ocr-ovis/ocr-paddle)는 `extra_hosts: host.docker.internal:host-gateway`
  — Linux에서도 호스트 Ollama(§17) 접근이 가능하게 한다.
  두 sidecar 이미지도 **비루트(uid 1000)로 실행**한다 — 사용자가 올린 임의 PDF의 렌더
  이미지를 파서에 먹이는 쪽이라 신뢰 경계가 backend보다 바깥이다. PaddleX 캐시는
  `$HOME` 기준이라 마운트 경로가 `/home/app/.paddlex`로 바뀌었고, **이미 root 소유로
  채워진 기존 캐시 볼륨은 한 번 `chown -R 1000:1000` 해야 한다**(각 Dockerfile 주석에 명령 있음).
- **로그 로테이션·리소스 상한**: 전 서비스가 YAML 앵커 `x-logging`으로 json-file
  `max-size 10m × max-file 3`(서비스당 최대 30MB)을 쓴다 — `restart: unless-stopped`와
  겹쳐 장기 구동 호스트의 디스크를 조용히 채우던 문제 차단. 메모리 상한은
  `deploy.resources.limits.memory`로 서비스별로 걸고 `.env`로 조정한다(§7).
  CPU는 상한 대신 `OCR_CPU_THREADS`로 조절한다.
- **compose 스레딩**: `.env`는 `.dockerignore`로 이미지·컨테이너 안에 없으므로,
  compose `environment`에 명시하지 않은 키는 컨테이너에서 조용히 무시된다.
  이번 사이클에 `MAX_LENGTH`·`PAGE_SEPARATOR`·`LLM_OPENAI_API_KEY`가 4개 backend
  서비스 전부에 추가됐다.
- **Ollama 컨테이너(선택)는 overlay로만**: `compose.ollama.yaml`은 프로필 없는
  `ollama` 서비스를 추가하고 ocr-cpu의 `OLLAMA_BASE_URL`을 `http://ollama:11434`로
  덮어쓴다. 본 파일에 병합하면 기본 경로(`docker compose up`)에서도 함께 기동돼
  버리므로 반드시 overlay 파일로 유지한다:
  `docker compose -f docker-compose.yml -f compose.ollama.yaml up -d --build ocr-cpu ollama`
  (= `make docker-up-ollama`). Ollama 포트(11434)는 내부 `expose`만 — 호스트 미공개.

## 9. C++ 네이티브 모듈 (`native/`, 모듈명 `uocr_native`)

목적: 토큰 생성 핫패스(no-repeat-ngram 배닝)와 figure 크롭의 C++ 가속.
**없어도 앱은 동작해야 한다** — `app/native_ops.py`가 임포트 실패 시 순수 파이썬 폴백 사용.

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

## 10. 프론트엔드 (frontend/, 정적 SPA)

- **외부 네트워크 리소스 0** (CDN/폰트/트래커 금지), 빌드 스텝 없음, 바닐라 JS(ES modules)
- 한국어 UI, 다크/라이트 자동(`prefers-color-scheme`) + 수동 토글(localStorage)
- 구성:
  - 헤더: 앱명 "Unlimited-OCR — PDF → Markdown", `/api/health` 기반 디바이스/엔진 배지
  - 좌측: PDF 드롭존(+파일선택, 확장자/크기 검증) · 옵션(mode, dpi) · 잡 히스토리(5초 폴링)
  - 메인(활성 잡, **공식 데모 GIF 재현 3-패널 라이브 뷰**):
    1. 원본+레이아웃 — 현재 페이지 이미지 위에 스트림의 `<|det|>label [x1,y1,x2,y2]<|/det|>`
       (0–999 정규화) 좌표로 컬러 박스를 실시간 오버레이, `<PAGE>` 마커로 페이지 자동 전환
    2. RAW OUTPUT — SSE `token` 델타 모노스페이스 append (자동 스크롤, 청크 경계 holdback)
    3. 실시간 미리보기 — 정리된 스트림 텍스트를 600ms debounce로
       `POST /render-preview`에 보내 렌더된 HTML 표시
    - 실행 중 STOP(정지) 버튼 → `POST /cancel` (부분 결과 보존) · 진행 바(phase + 페이지 n/N)
    - 완료 시 탭 [미리보기(HTML)] [Markdown] [레이아웃] [원본 페이지] · [.md] [.zip] 다운로드 · 삭제
  - 미리보기 탭은 `/api/jobs/{id}/html` 응답을 주입 (클라이언트 md 렌더러 불필요)
  - SSE 불가 환경 폴백: 1초 상태 폴링(+부분 markdown 주기 조회)
- 성능: token append는 rAF 배칭, 히스토리 50개 제한

## 11. 테스트 전략

- `backend/tests/` (FakeEngine, torch 불필요 — CI/로컬 빠른 실행):
  - merge 로직(리넘버링/참조 재작성/`<PAGE>` 분리/페이지 경계 계약) 단위 테스트
  - API 플로우: 업로드→상태→SSE→markdown/html/zip→삭제 (httpx + TestClient),
    번역·Q&A 라우트, 레이트리밋, 잡 GC
  - pdf.py 렌더 테스트(생성 PDF), render.py img src 재작성,
    `test_pdf_export*.py`(레이아웃 보존·시각 안전성), translate/sidecar/llm 계약
- `services/{ovisocr2,paddleocr_vl}/tests/`: sidecar 파서·어댑터 (stdlib만 — 모델·CUDA 불필요)
- `native/tests/`: C++ ↔ 파이썬 레퍼런스 패리티
- `frontend/tests/`: `node --test`(replay·reader-scroll 등) + `tests/e2e/`
  (`ui.e2e.mjs` = 실서버 대상, `mock-full-flow.e2e.mjs` = hermetic playwright)
- 실모델 E2E: `scripts/smoke_e2e.sh` — compose 기동 후 샘플 PDF 변환, figure 파일 존재 검증

### 11.1 전 구간 검증 하네스 (`make verify-e2e`)

`scripts/verify_e2e.py`는 smoke가 끝나는 지점(업로드→OCR→markdown/zip) **이후**를
실제 서버를 띄워 검증한다. 외부 API·GPU·모델 다운로드가 필요 없다:

- `OCR_ENGINE=textlayer`로 backend를 띄우고, 번역 프로바이더로는 같은 하네스가 띄운
  `scripts/mock_llm.py`(OpenAI 호환 목 서버)를 가리킨다. 목 서버는 마스킹
  플레이스홀더를 보존한 채 결정적으로 "번역"하므로 복원·layout 정렬·PDF 조판까지
  실제 경로가 전부 돈다. `?fault=refusal|echo|drop_placeholder|http400|http429`
  쿼리(또는 `FAULT` 환경변수)로 **결함 주입**도 한다.
- 단계: 서버 기동/health → 업로드·OCR(`verify_ocr`) → 번역(`verify_translation`) →
  PDF 내보내기(`verify_pdf_export`) → CropBox(`verify_cropbox`) → 뷰어 계약
  (`verify_viewer`) → 보안(`verify_security` — `/files` 경로 탈출 등) →
  번역 결함 주입(`verify_translation_faults`) → 워커 복원력(`verify_worker_resilience`).
  총 60개 안팎의 `check()` 단언을 세고 마지막에 `통과 N / 실패 M`을 출력한다
  (실패가 있으면 종료코드 1).
- 옵션: `--pdf PATH`(기본 `sample/2504.19874v1.pdf`) · `--pages N`(앞 N페이지만) ·
  `--skip faults,worker,cropbox` · `--work DIR`(기본 `tmp/verify-e2e`).
  포트는 매 실행마다 빈 포트를 잡아 개발 서버(8000)와 충돌하지 않는다.
  `make verify-e2e VERIFY_ARGS="--pages 4"` 형태로 인자를 넘긴다.

### 11.2 CI 잡 구성 (`.github/workflows/ci.yml`)

| 잡 | 내용 |
|---|---|
| `backend` | `uv sync --locked --extra cpu` → Noto CJK 설치 → `pytest --cov`(term+xml), coverage.xml 아티팩트 업로드 |
| `lint` | `ruff check . ../services` — sidecar는 backend uv.lock에 고정된 ruff를 재사용 |
| `frontend` | `node --test` 단위 테스트 |
| `native` | C++ 빌드 + 패리티 pytest |
| `sidecar` | matrix(`ovisocr2`,`paddleocr_vl`) 파서/어댑터 pytest |
| `e2e-mock` | mock OpenAI + FakeEngine 백엔드 hermetic 브라우저 E2E. 러너 시간이 커서 **PR에서는 돌지 않고** nightly(`schedule: 0 18 * * *`)·`workflow_dispatch`에서만 실행, 실패 시 스크린샷 아티팩트 업로드 |

- **`--locked`가 계약**: lock 드리프트(pyproject만 고치고 uv.lock 커밋 누락)를 CI에서
  즉시 실패시킨다. 없으면 uv가 조용히 재잠금해 통과시키고, Dockerfile의 `--frozen`은
  의존성을 빠뜨린 채 빌드해 컨테이너 기동 시 ImportError로 드러난다.
- 커버리지는 게이트가 아니라 관측용이다 — `pytest-cov`는 lock을 건드리지 않도록
  `uv run --with`로 임시 설치한다(`make coverage`도 동일).

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

## 13. 한국어 번역 (Translation)

OCR로 얻은 **데이터 레이어**(`result.md` + `layout.json`)를 OpenAI 호환 API로 번역해
`result.{lang}.md` / `layout.{lang}.json`을 만들고, 공통 facsimile 페이지 모델로
번역본 미리보기/HTML/PDF를 제공한다. 지원 언어: `ko` (`SUPPORTED_LANGS`).

### 13.1 파이프라인 개요

```
변환 완료(done) 잡 ──► POST /translate ──► 번역 데몬 스레드(잡·lang별, OCR 워커와 별개로 병렬)
  run_translation(job_dir, lang, cfg, *, page_separator, progress, cancel, force)
    1. result.md(+layout.json)를 번역 유닛으로 분해, 마스킹(<m1 .../> 플레이스홀더)
    1b. 2단 패스 준비 — 모든 줄이 layout 블록에 완전히 커버되는 md 유닛은 1차에서 제외(deferred)
    2. 유닛 캐시(units.json)·용어집(glossary.json) 활용해 API 호출 (Chat/Responses)
       - 잡당 최대 8 worker, 프로세스 전역 HTTP 세마포어 기본 8
         (`TRANSLATE_GLOBAL_CONCURRENCY`로 1–8 조정)
       - 같은 cache key는 single-flight로 결과·오류를 공유해 중복 과금/재시도를 차단
       - auto 모드의 Responses→Chat capability probe도 동시 최초 호출끼리는 single-flight.
         초기 협상이 일시 오류로 실패하면 후속 순차 호출은 재협상 가능
       - `title` 유닛은 의미·정보량을 유지하고 UI 라벨식 축약을 금지한다
       - 출력 측 검증 게이트(_accepted)를 통과한 유닛만 채택·캐시된다 (§13.4)
    3. 플레이스홀더 복원 → layout 블록과 정확히 대응하는 Markdown 줄은 동일 번역으로 정렬
       (PDF·개요·읽기 텍스트의 제목/용어 SSOT, ref_text는 양쪽 모두 원문 유지)
    3b. reconcile이 폴백이면 → deferred 유닛을 2차 번역(total 증가) 후 재조립
    4. result.{lang}.md / layout.{lang}.json 기록
    5. state.json에 running(current/total) → done|error|canceled 기록, report.json 저장
```

#### 2단 패스 (deferred md 유닛)

reconcile이 성공하면 md 유닛 번역은 전량 폐기되고 layout 번역이 단일 기준이 된다 —
LLM 왕복의 절반이 낭비였다. 그래서 **비어 있지 않은 모든 줄이 layout 블록 원문에
커버되는** md 유닛만 1차 디스패치에서 빼두고(`deferred`), `reconcile_markdown_with_layout`이
**폴백을 돌려준 경우에만** 2차로 번역해 무손실 계약을 지킨다. 부분만 걸치는 md 유닛
(다중 줄 블록·표·수식 줄)은 매핑에 안 걸려 원문이 남으므로 종전대로 1차에서 번역한다.
2차 진입 시 `total`이 늘어나므로 SSE progress의 `total`은 **증가할 수 있다**(단조 감소는 없음).

- 번역 코어(`app/translate/`)는 **OCR 엔진·torch에 의존하지 않는다**(requests + 표준 라이브러리).
- 원본 Markdown의 비어 있지 않은 줄 중 70% 이상이 layout 블록과 정확히 대응하면
  `layout.{lang}.json`의 블록 번역을 `result.{lang}.md`에도 재사용한다. 대응률이
  낮은 비정형 Markdown은 독립 Markdown 번역을 유지하며, 중복 원문의 번역이 서로
  다르거나 여러 줄인 블록은 보수적으로 정렬 대상에서 제외한다.
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
translations/{lang}/glossary.json  문서 용어집 [{"src","ko","policy","first_unit"}]
translations/{lang}/units.json     유닛 캐시 {cache_key: 번역문}
translations/{lang}/report.json    품질 리포트 {"kept_original":[...],"retried":n,...}
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
  "started_at": "…", "finished_at": null
}
```

### 13.3 REST / SSE 계약 (번역)

- **POST /api/jobs/{id}/translate** — body `{"lang":"ko","force":false}` (기본 `lang="ko"`).
  - `400` 지원하지 않는 언어 / `409` 변환이 완료된 잡만 번역 가능 /
    `429` 잡·IP 레이트리밋 또는 동시 실행 상한 초과(`Retry-After` 동반 — §5) /
    `503` 프로바이더 미설정(detail=사유)
  - 검사 순서: lang → 잡 상태(409) → 레이트리밋(429) → 프로바이더 구성(503) →
    동시 실행 상한(429, `translate_lock` 안)
  - 이미 실행 중 → `200 {"status":"running"}`; state가 `done`이고 `force` 아님 → `200 {"status":"done"}`
  - 그 외 → 데몬 스레드 시작 후 `202 {"job_id","lang","status":"running"}`
  - 성공 시 `archive.zip` 캐시를 삭제해 다음 `/archive`가 `result.{lang}.md`까지 담아 재생성
- **GET /api/jobs/{id}/translate/state?lang=ko** — `state.json` 없으면 `200 {"status":"none","lang"}`.
  있으면 내용 반환하되 **stale 조정** 적용(§13.5).
- **POST /api/jobs/{id}/translate/cancel?lang=ko** — 실행 중이면 `202 {"status":"canceling"}`,
  아니면 현재 상태 반환.
- **GET /api/jobs/{id}/translate/events?lang=ko** — `/events`와 동일 SSE 패턴
  (`retry:3000`, 15초 `: ping`). 브로커 채널 키 `"{id}:translate:{lang}"`.
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
  ∥ 원문 전체 ∥ 블록 종류 ∥ 직전 문맥 ∥ temperature ∥ reasoning)`.
  모델·프롬프트 버전·해당 유닛 용어집·원문·제목/본문 정책·샘플링 설정이 바뀌면
  영향받는 유닛만 자동 재번역된다. 짧은 placeholder 미리보기 충돌도 원문 전체로 분리하고,
  같은 문장도 직전 문맥이 다르면 별도 번역한다. `TRANSLATE_REASONING`을 off→high로
  올린 뒤 재개해도 이전 설정의 번역이 재사용되지 않는다.
- **PROMPT_V 상승 = 캐시 전면 무효화**: `PROMPT_V`가 캐시 키의 첫 재료이므로 값이 오르면
  기존 `units.json`은 전부 미스가 된다. 현재 값은 **`"6"`**(v6 = 출력 측 검증 도입 +
  인라인 수식 통화 오인 수정)이라 **v5 이전에 만들어진 유닛 캐시는 이번 사이클에
  무효화된다** — 이미 캐시된 거부문·echo를 강제로 다시 번역시키는 것이 목적이다.
  `report.json`/`state.json`의 `prompt_v` 필드로 어떤 버전으로 만든 결과인지 확인할 수 있다.
- **잘림 감지 재시도**: chat `finish_reason=="length"` / Responses `status=="incomplete"`를
  감지하면 같은 요청을 **max_tokens 2배로 1회 재시도**한다 (thinking 토큰이 예산을
  소진하는 경우 대비). `TRANSLATE_MAX_TOKENS_PARAM=none`이면 같은 요청의 반복이라 생략.
- **취소 응답성**: cancel은 유닛 디스패치 사이뿐 아니라 **래더 단계(최초→repair→분할)
  사이에서도** 확인된다 — 거대 표 래더(유닛당 수 분)가 취소 후에도 이어지지 않는다.
  취소로 조기 반환된 유닛은 kept_original 통계에 포함되지 않는다.
- **엔진이 상태의 단일 기록자**: `state.json`은 `run_translation`이 직접 쓴다.
  API 스레드는 SSE 이벤트 중계와 레지스트리 정리만 담당한다.

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
  뒤에 두어야 한다. README §보안 · SECURITY.md와 같은 정책이다.
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

## 16. textlayer 엔진 (Localight 통합)

`OCR_ENGINE=textlayer` — **모델 다운로드 없이 CPU만으로 즉시 동작**하는 경량 엔진.
Localight의 "텍스트 레이어 우선 + OCR 폴백" 추출기를 기존 엔진 계약
(`app/engine/base.py`의 OCREngine)에 맞춰 이식했다.

### 16.1 계약

- **텍스트 레이어 우선**: 페이지별로 pymupdf 내장 텍스트 레이어를 먼저 추출한다.
  추출 영숫자 수가 `NATIVE_TEXT_THRESHOLD`(기본 `120`) 미만인 페이지만
  **Tesseract OCR**(`OCR_LANGUAGES`, 기본 `eng+kor`)로 폴백한다 — 텍스트 PDF는
  OCR 오류 없이 원문 그대로, 스캔 페이지만 OCR을 거친다.
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
