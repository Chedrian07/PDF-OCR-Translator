# OvisOCR2 sidecar — RTX 5070 Ti (CUDA, Blackwell)

## 역할

페이지 단위 **정밀 문서 파싱 + figure bbox** 엔진. 한 번의 생성으로 Markdown·
표 HTML·LaTeX·figure bbox(`<img src="images/bbox_l_t_r_b.jpg" />`, [0,1000))를
출력하는 0.9B 경량 모델이다. 멀티페이지 문맥·토큰 스트리밍이 필요하면
Unlimited-OCR을, 한국어 레이아웃 중심이면 PaddleOCR-VL을 선택한다.

## 고정값 (2026-07-20 공식 소스 재확인)

| 항목 | 값 |
|---|---|
| 모델 ID | `ATH-MaaS/OvisOCR2` |
| 모델 revision | `65c619d374b55d4152e85150fc1b003700bc1f0c` (코드 기본값 — env로 오버라이드 가능) |
| 라이선스 | Apache-2.0 |
| 파라미터 | 852.9M (BF16 safetensors 단일 파일, ~1.7GB) |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (GDN 하이브리드 어텐션) |
| runtime | vLLM **0.22.1** (모델 카드 공식 권장 — `pip install "vllm==0.22.1"`) |
| Docker base | `vllm/vllm-openai:v0.22.1-cu129@sha256:e1668bce…` (Blackwell=CUDA≥12.8, cu129 태그 + digest 고정) |
| 웹 계층 | `requirements.lock` 덧씌움(fastapi 0.139.0·starlette 1.3.1·python-multipart 0.0.32·uvicorn 0.50.2 등 7개, 해시 고정) — 아래 §의존성 잠금 |
| dtype | bfloat16 |
| 프롬프트 | 모델 카드 공식 OCR 프롬프트 원문 (`services/ovisocr2/app/model.py::OFFICIAL_PROMPT`) |

### 의존성 잠금 정책

- `services/ovisocr2/requirements.in`이 원천, `requirements.lock`이 해시까지 고정한 출력이다
  (다시 만드는 `uv pip compile` 명령은 lock 머리말). 베이스 vLLM 이미지의 torch·vllm·
  pydantic·pillow는 digest 고정이 재현성을 맡고, lock은 웹 계층만 덧씌운다.
- Dockerfile은 `pip install --no-cache-dir --require-hashes --no-deps --only-binary :all:
  -r requirements.lock`으로만 설치한다 — 해시 불일치·소스 빌드는 빌드 실패이고, `--no-deps`라
  베이스 패키지는 건드리지 않는다. 덧씌운 웹 계층이 베이스와 맞물리는지 빌드가
  `python3 -c "import app.main"`으로 확인한다.
- 웹 계층 버전이 backend와 같아 폼 파서 상한도 같다(필드 1000개 초과 → 400).
- 전부 순수 파이썬 휠이라 linux/amd64·arm64 베이스 모두에 맞는다. CI `dependency-audit`
  잡이 이 lock을 pip-audit로 검사한다. Dependabot은 vLLM 베이스를 갱신하지 않는다
  (GPU 호스트 재검증이 필요해서) — 올릴 때는 `docker buildx imagetools inspect
  vllm/vllm-openai:<태그>`의 digest로 바꾸고 smoke를 다시 돌린다.

## RTX 5070 Ti 16GB VRAM 정책 (기본값 근거)

| 설정 | 기본값 | 근거 |
|---|---|---|
| `OVIS_GPU_MEMORY_UTILIZATION` | 0.80 | 데스크톱/WSL 디스플레이·런타임 몫 확보. 0.92 초과는 설정 거부 |
| `OVIS_MAX_MODEL_LEN` | 24576 | max_pixels 2880²에서 비전 토큰 ~8K + 출력 8192 + 프롬프트 여유. 하이브리드 어텐션이라 KV 비용 미미 |
| `OVIS_MAX_OUTPUT_TOKENS` | 8192 | 페이지별 생성 상한 (폭주 방지) |
| `OVIS_MAX_NUM_SEQS` | 1 | 단일 GPU 예측 가능성 — 동시 시퀀스 없음 |
| `OVIS_MIN_PIXELS` / `OVIS_MAX_PIXELS` | 448² / 2880² | 모델 카드 공식 값 |
| `OVIS_GDN_PREFILL_BACKEND` | `triton` | **sm_120 필수** — FlashInfer GDN 경로는 데이터센터 아키텍처 게이트. 모델 카드 공식 예제와 동일 |

## 실행

```bash
docker compose --profile ovis up -d --build ovisocr2 ocr-ovis   # sidecar(GPU) + backend(:8002)
# 진행 로그(모델 다운로드 ~1.7GB 포함):
docker compose --profile ovis logs -f ovisocr2
```

health 확인:

```bash
curl -s http://127.0.0.1:8002/api/health | python3 -m json.tool   # backend 관점
docker compose --profile ovis exec ovisocr2 \
  python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8080/health').read().decode())"
```

smoke test (실 GPU):

```bash
cd backend && uv run python ../scripts/smoke_ovisocr2_5070ti.py
```

종료 / 전환:

```bash
docker compose stop ovisocr2 ocr-ovis         # 대상만 정지 (⚠ paddle/cuda와 동시 기동 금지)
docker compose --profile ovis down            # 볼륨 외 전체 정리 (ocr-cpu도 함께 내려감)
```

캐시 삭제 (모델 재다운로드):

```bash
# 실제 볼륨 이름에는 compose 프로젝트 접두사가 붙는다(기본 = 이 디렉터리 이름).
# 이름을 추측하지 말고 확인한다:
docker compose --profile ovis config --format json | python3 -c \
  "import json,sys;print(json.load(sys.stdin)['volumes']['ovis-hf-cache']['name'])"
docker volume rm <위에서 확인한 이름>      # 예: <PROJECT>_ovis-hf-cache
```

⚠ **비루트 실행(uid 1000)으로 전환됨** (`services/ovisocr2/Dockerfile`). 비어 있는
볼륨은 이미지의 chown 결과를 물려받지만, **이미 root 소유로 채워진 기존 캐시 볼륨은
자동으로 바뀌지 않는다** — 업그레이드 후 캐시 쓰기가 실패하면 한 번만 실행한다
(또는 위 `docker volume rm`으로 지우고 다시 받는다). compose가 `cap_drop: [ALL]`을
걸어 두므로 이 실행에만 두 캡을 돌려준다:

```bash
docker compose --profile ovis run --rm --no-deps --user 0 --cap-add CHOWN \
  --cap-add DAC_OVERRIDE --entrypoint chown ovisocr2 -R 1000:1000 /data/hf
```

예전 안내의 `docker run -v <접두사 없는 이름>:… alpine chown …`은 존재하지 않는 볼륨을
새로 만들고 exit 0으로 끝나 아무것도 고치지 못했다 — compose로 실행하면 compose가 실제
볼륨을 마운트한다.

compose는 이 sidecar에 로그 로테이션(json-file 10MB×3)과 호스트 RAM 상한
`OVIS_MEM_LIMIT`(기본 24g)을 건다 — VRAM과 무관한 안전판이며 호스트 RAM이 작으면
`.env`로 낮춘다.

## 출력 파싱 (strict)

`services/ovisocr2/app/parser.py` — 다음 형식만 figure로 인정:

```
<img src="images/bbox_{l}_{t}_{r}_{b}.jpg" />     (공백·slash 생략 변형만 허용)
```

- 좌표는 각각 ≤4자리 정수, [0,1000] 범위(1000은 999로 clamp), x1<x2·y1<y2,
  최소 변 2, 중복 제거, 페이지당 64개 상한
- 그 밖의 모든 `<img …>` 태그(외부 URL·경로 탈출·비정상 속성·닫히지 않은 태그)는
  **제거**되고 warning으로 기록
- 유효 태그는 순서대로 `[[FIGURE:n]]` placeholder로 치환 — 파일명은 절대 모델
  출력에서 오지 않는다
- 문서 본문에 리터럴로 실린 `[[FIGURE:`는 유효 태그 **사이의 텍스트 조각마다**
  `&#91;&#91;FIGURE:`로 이스케이프한다(렌더하면 같은 글자, placeholder 정규식에는 안 걸림 —
  `\[`는 수식 구분자라 쓰지 않는다). 그림 자리 납치를 막는다(OCR_ENGINE_PROTOCOL.md §markdown 규약)
- 중첩·닫히지 않은 `<img>` 같은 비정상 출력에서도 유효 태그의 placeholder는 제자리에 남는다
  (예전 닫히지 않은 태그 정리가 placeholder까지 지워 그림이 페이지 끝으로 밀렸다)
- 반복 suffix 정리: 모델 카드의 `_clean_truncated_repeats` 알고리즘을 의미론
  그대로 구현 (독립 unit test: `services/ovisocr2/tests/test_parser.py`)

## 출력 상한 · 로드 재시도 · 자가 재시작

- **출력 상한(`finish_reason=length`)**: 페이지가 `OVIS_MAX_OUTPUT_TOKENS`(기본 8192)에서
  끊기면 warning과 함께 `page.truncated=true`를 보낸다. backend는 PDF 텍스트 레이어와
  대조해 잃은 것이 크면(충실도 < `OCR_FIDELITY_THRESHOLD`) 그 페이지를 텍스트 레이어로
  복구하고, 아니면 잘린 출력을 경고와 함께 쓴다 — 같은 페이지를 GPU에 다시 보내지 않는다.
  잡 경고에는 'sidecar 출력 토큰 상한 도달'로 적힌다(`MAX_LENGTH`는 이 엔진과 무관).
- **로드 재시도**: 첫 기동의 다운로드·로드가 일시적으로 실패하면 15/30/60/120초 뒤 다시
  시도한다(최대 5회). 그동안 health는 `status:ok`·`model_loaded:false`·`load_retry:{…}`라
  잡은 실패하지 않고 `모델 로드 재시도 대기 중…`으로 기다린다. CUDA 가드·설치 누락·
  HTTP 401/403/404는 곧바로 `load_error`다.
- **엔진 사망 → 자가 재시작**: vLLM EngineCore가 죽으면(`EngineDead`·`EngineCore` 시그니처)
  그 요청에 503을 돌려주고 health에 `restarting:true`를 올린 뒤 1.5초 뒤 종료 코드 3으로
  끝난다. compose `restart: unless-stopped`가 컨테이너를 다시 띄우고, backend는
  `sidecar 재시작 대기 중…`으로 기다렸다가 그 페이지만 다시 보낸다. 예전에는 웨지 신고만
  하고 복구 경로가 없어 컨테이너를 손으로 재시작해야 했다.

## OOM 완화 순서

1. `OCR_REMOTE_PAGE_CONCURRENCY=1` 확인 (기본값)
2. `OVIS_MAX_PIXELS` 감소 (예: `4194304`=2048² — sidecar도 OOM 시 자동 1회 강등)
3. `OVIS_MAX_OUTPUT_TOKENS` 감소 (예: 4096)
4. `OVIS_MAX_MODEL_LEN` 감소 (max_pixels를 줄였다면 함께: 픽셀/1024 ≈ 비전 토큰)
5. `OVIS_GPU_MEMORY_UTILIZATION` 조정 (0.75 → 0.70)
6. (해당 없음 — 이 sidecar는 layout detector가 없다)
7. 마지막 수단: 더 작은 입력(RENDER_DPI 150) 또는 다른 엔진 선택

CPU offload는 사용하지 않는다.

## 문제 해결

| 증상 | 확인 |
|---|---|
| health `status:error` | `docker compose --profile ovis logs ovisocr2` — load_error 필드에 요약 |
| `엔진 서버 연결 안 됨` 배지 | sidecar 컨테이너 기동/healthcheck 상태 (`docker compose ps`) |
| 첫 잡이 `아직 모델을 로드하지` 오류 | 최초 다운로드가 `OCR_SIDECAR_MODEL_WAIT_S`(900초) 안에 끝나지 않음 — 로그로 진행 확인 후 재시도하거나 값을 늘린다 |
| 잡이 `모델 로드 재시도 대기 중…`에 머묾 | 일시적 로드 실패 뒤 백오프 중 — `/api/health`의 `provider_health.load_retry`(시도 수·다음 재시도·마지막 오류) |
| 컨테이너가 종료 코드 3으로 재시작됨 | 추론 엔진 사망 뒤 자가 재시작(위 §자가 재시작) — 반복되면 로그의 원인(대개 CUDA OOM)을 보고 OOM 완화 순서대로 줄인다 |
| `출력 토큰 상한`이 들어간 경고(`…잘린 출력을 그대로 씁니다` / `sidecar 출력 토큰 상한 도달…`) | 페이지가 `OVIS_MAX_OUTPUT_TOKENS`에서 끊김 — 표가 아주 긴 페이지면 값을 올리고 `OVIS_MAX_MODEL_LEN`도 함께 본다 |
| Triton/GDN 관련 크래시 | `OVIS_GDN_PREFILL_BACKEND=triton` 유지 확인 (sm_120에서 flashinfer 불가) |
| 응답 시간 초과 | 페이지 해상도↓(`RENDER_DPI`) 또는 `OCR_SIDECAR_READ_TIMEOUT_S`↑ |
| 캐시/HF 다운로드 `Permission denied` | 비루트(uid 1000) 전환 전에 만들어진 볼륨 소유권 — 위 §캐시 삭제의 chown 1회 |

## Known limitations

- **layout_capability = figure_only**: 텍스트 블록 bbox를 제공하지 않는다. 그래서
  이 엔진의 청크는 좌표 없는 `raw_pages.json`(페이지마다 빈 원출력 — merge의 페이지 수 대조용)만
  남기고 잡은 `layout.json`을 만들지 않아 `has_layout=false`다 —
  좌표 기능(`/layout`·`/alignment`·`/outline`·`/viewer/pages`)은 404, 번역 PDF(`/pdf`)는
  409이고, 읽기·내보내기는 텍스트 보기(`/html`)와 의미 기반 `document.html`을 쓴다.
  이전 버전에서 만든 OvisOCR2 잡(image 블록만 있는 `layout.json`)도 같은 규칙으로
  다룬다(`artifacts.has_usable_layout` — 텍스트 블록이 하나 이상인 layout만 좌표로 쓴다).
- 스트리밍은 페이지 단위 — 토큰 델타 스트림 없음 (가짜 스트리밍 미구현이 의도).
- 취소 시 진행 중인 페이지의 GPU 추론은 완주 후 폐기된다 (프로토콜 문서 §취소).
- **한국어 본문 정확도가 PaddleOCR-VL보다 낮다(실측)**: 자체 한국어 샘플에서
  "혼용된→훈련된", "한글 자모 ㄱㄴㄷ→한국 자료 717", "보존→보준" 오독과 한자
  간체 혼입이 확인됐다. 한국어 중심 문서는 PaddleOCR-VL을 권장한다
  (반대로 표 셀 숫자는 Ovis가 정확했다 — docs/OCR_BENCHMARK.md 대조표).
- 첫 요청 컴파일 비용(~40초/2페이지)이 크다 — 짧은 문서를 가끔 처리하는 용도라면
  체감이 나쁠 수 있다(컨테이너를 계속 띄워 두면 해소).

## 검증 상태

- 구현·파서 fixture 테스트(28)·backend 통합 테스트·`docker compose config`: **완료**
- **RTX 5070 Ti 실 runtime 검증 완료 (2026-07-20, `scripts/smoke_ovisocr2_5070ti.py` exit 0)**:
  - 환경: WSL2 · driver 591.86 · `vllm/vllm-openai:v0.22.1-cu129` ·
    revision `65c619d3…` 고정 로드 확인 (`gdn_prefill_backend=triton`)
  - 모델 로드: 가중치 1.72GiB, health `gpu=NVIDIA GeForce RTX 5070 Ti`
  - 샘플 PDF(2페이지, 이미지 2·표 1): figure crop 2/2 · 표 1 · layout 오버레이 생성,
    실패 페이지 0
  - **성능은 콜드/웜을 분리해야 한다**: 컨테이너 기동 후 **첫 요청**은 vLLM 그래프
    컴파일로 2페이지에 **40~43초**(실측 42.8s)가 걸린다. 벤치마크·체감 성능을 논할 때
    첫 요청 수치를 쓰면 안 된다. ⚠ 당시 기록한 정상 상태 '2페이지 2.2초(1.1초/페이지)'는
    벤치마크가 잡 상태를 **2초 간격**으로 폴링하던 시절 값이라 해상도(±2초)보다 작아
    믿을 수 없다. 처리 속도는 아래 25쪽 실문서 결과(3.3초/페이지)를 기준으로 삼는다
    (폴링 0.25초·처리 시간 기준으로 바뀐 벤치마크로 재측정 예정 — docs/OCR_BENCHMARK.md)
  - **peak VRAM 12,969MB / 16,302MB** — `OVIS_GPU_MEMORY_UTILIZATION=0.80` 예산 내, OOM 없음
  - 잡 메타에 "페이지 단위 모델" 안내 정상 기록 (지금은 품질 경고가 아니라 참고 — `notices`)
- **실문서 확장 검증 (2026-07-21)**: 실제 arxiv 논문 2504.19874v1(25p, 2단, 수식
  밀집)을 82s(3.3s/p)에 처리 — 2단 저자 블록 읽기순서·display/inline LaTeX·Lemma
  구조 정확, figure 27개 추출, 실패 0. 스캔 시뮬 PDF(텍스트 레이어 0)도 정확히
  OCR. `OCR_REMOTE_PAGE_CONCURRENCY` 1↔4 출력 바이트 동일(순서 보존 확인).
- 참고: 이 머신은 5070 Ti + 4060 Ti 2-GPU 구성. compose가 GPU 서비스 셋
  (ocr-cuda·ovisocr2·paddleocr-vl)에 `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 기본
  설정하므로 `GPU_DEVICE` 인덱스는 nvidia-smi(PCI) 순서와 결정적으로 일치한다 —
  이 호스트에서는 0 = 5070 Ti (단일 GPU 원칙은 불변 — TP=1, 분산 없음).
  기동 후 sidecar `/health`의 `gpu_name`으로 의도한 카드인지 확인할 것.
