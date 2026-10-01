# PaddleOCR-VL-1.6 sidecar — RTX 5070 Ti (NVIDIA Blackwell)

## 역할

**한국어·다국어 + 완전한 layout** 엔진. layout detection(PP-DocLayout 계열) +
0.9B VL 인식기의 2단 파이프라인으로 페이지의 블록(제목/본문/표/수식/차트/도장/
각주/머리글…)·읽기 순서·bbox·Markdown을 함께 낸다. 시리즈 공식 문서 기준
109개 언어(한국어 명시) 지원.

## 고정값 (2026-07-20 공식 소스 재확인)

| 항목 | 값 |
|---|---|
| 모델 ID | `PaddlePaddle/PaddleOCR-VL-1.6` (파이프라인명 `PaddleOCR-VL-1.6-0.9B`) |
| 모델 revision | `66317acc4c9fc17bd154591ce650735cd2855f3e` (코드 기본값 — 기동 시 `snapshot_download`로 캐시 선점) |
| 라이선스 | Apache-2.0 |
| 파라미터 | 0.9B (BF16, ~959MB) + layout detector |
| paddlepaddle-gpu | **3.3.1** (공식 cu129 CDN 휠 URL + 해시 고정 — Blackwell 가이드 최소 3.2.1) |
| paddleocr | **3.6.0** (`paddleocr[doc-parser]` — VL-1.6 동봉 버전, paddlex 3.6.1) |
| Docker base | `python:3.12-slim-bookworm@sha256:392307d2…`(backend와 같은 digest) + `apt-get upgrade` + 아래 해시 잠금 (자체 빌드 — 아래 §설치 경로) |
| 플랫폼 | **linux/amd64 전용** (paddlepaddle-gpu 휠이 x86_64 전용) |
| dtype | bfloat16 (공식 기본) |

## Blackwell 설치 경로 (공식 가이드 기준)

공식 [PaddleOCR-VL NVIDIA Blackwell 가이드]가 제시하는 두 경로 중 **wheel 경로**를
기본으로 채택했다. 가이드의 명령은 다음과 같다:

```
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -m pip install "paddleocr[doc-parser]==3.6.0"
```

이미지는 이 명령을 그대로 돌리지 않고 같은 버전을 **해시 잠금**으로 설치한다
(`services/paddleocr_vl/Dockerfile`):

- `requirements.in`이 원천(최상위 고정·보안 하한), `requirements.lock`이 전이 의존성 116개를
  해시까지 고정한 실제 잠금이다. Dockerfile은
  `pip install --require-hashes --no-deps --only-binary :all: -r requirements.lock`으로만
  설치한다 — 해시 불일치(미러·CDN 변조)·lock 밖 패키지·소스 빌드는 빌드 실패다.
- `paddlepaddle-gpu`는 PyPI에 cu129 빌드가 없다(2.6.2까지만). cu129 인덱스를 추가 인덱스로
  걸면 그 인덱스가 미러로 서빙하는 numpy·pillow 등까지 출처가 섞이므로, 인덱스가 가리키는
  공식 CDN 휠 URL과 해시를 lock에 직접 고정한다. 그래서 이미지는 linux/amd64 전용이다
  (예전 인덱스 설치는 aarch64도 허용했다).
- 날짜 창: RTX 5070 Ti 실측 검증 이미지의 빌드 시점(`--exclude-newer 2026-07-27`,
  huggingface_hub 1.24.0)으로 전이 버전을 맞춘다. 그 뒤에 나온 보안 수정판만 예외로
  올린다(urllib3 2.8.0). 다시 만드는 명령은 lock 머리말에 있다.
- 웹 계층(fastapi 0.139.0·starlette 1.3.1·python-multipart 0.0.32·uvicorn 0.50.2)은
  backend와 같은 버전이라 폼 파서 상한도 같다(필드 1000개 초과 → 400). CI
  `dependency-audit` 잡이 lock을 pip-audit로, `paddlepaddle==3.3.1`을 OSV로 검사한다.
  Dependabot은 이 lock을 같은 방식으로 다시 만들 수 없어 대상에서 뺐다 — 갱신은 사람이 한다.
- ⚠ 이 잠금은 2026-07-27 해석을 재현한 것이지 검증 이미지의 `pip freeze`가 아니다
  (protobuf 7.35.1·numpy 2.3.5 포함). 다시 빌드한 이미지는 GPU 호스트에서
  `scripts/smoke_paddleocr_vl_5070ti.py`로 확인한다.

- 가이드는 RTX 5090/5080/**5070 Ti**/5070/5060(Ti)/5050을 대상 목록에 두지만
  **공식 검증 장비는 RTX 5070**이다 — 5070 Ti에서는 반드시
  `scripts/smoke_paddleocr_vl_5070ti.py`로 실측 검증한다.
- 요구 조건: **CUDA 12.9+를 지원하는 NVIDIA 드라이버**, nvidia-container-toolkit.
- 대안: 공식 sm120 Docker 이미지
  (`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:<버전>-nvidia-gpu-sm120`)
  — 중국 레지스트리 접근성 문제로 자체 빌드를 기본으로 한다. `latest` 태그 금지.
- ⚠ `paddleocr install_genai_server_deps`(vLLM 서버 경로)의 사전 빌드 휠은 CUDA
  12.6 대상이라 Blackwell에서 쓰지 않는다 — in-process transformers/paddle 경로 사용.

### revision 고정의 잔여 위험

`PaddleOCRVL` 생성자는 HF revision 인자를 받지 않는다. sidecar는 기동 시
`huggingface_hub.snapshot_download(repo, revision=고정 SHA)`로 캐시를 선점하고
`PADDLE_PDX_MODEL_SOURCE=huggingface`로 HF 경로를 강제한다. 캐시가 비어 있고
스냅샷 다운로드가 실패하면 paddlex가 다른 스냅샷을 받을 수 있다 — health의
`model_revision`은 **의도된 고정값**이며, 엄밀한 검증은 스냅샷 로그로 확인한다.

## 디바이스 정책

- `PADDLEOCR_DEVICE=gpu:0` (기본): layout detector + VL 모두 GPU.
  0.9B BF16 + layout 모델은 16GB에서 여유가 크다 (커뮤니티 실측 vLLM 기준 ~3.3GB,
  in-process 경로는 그보다 높지만 16GB 내 — smoke test의 peak VRAM으로 실측).
- **컴포넌트별 디바이스 분리(layout=CPU, VL=GPU)는 공식 in-process 파이프라인이
  지원하지 않는다.** 공식 분리 경로는 `device="cpu"` + 별도 genai-server인데,
  이는 컨테이너 2개·CUDA 12.6 휠 문제로 이번 범위에서 제외 (VRAM 여유가 커서
  실익도 없음). 따라서 `PADDLEOCR_LAYOUT_DEVICE` 같은 변수는 **의도적으로 없다**.
- 전체 CPU 실행이 필요하면 `PADDLEOCR_DEVICE=cpu` (느림 — 비상용).

## 실행

```bash
docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle   # sidecar(GPU) + backend(:8003)
docker compose --profile paddle logs -f paddleocr-vl
```

health / smoke / 종료:

```bash
curl -s http://127.0.0.1:8003/api/health | python3 -m json.tool
cd backend && uv run python ../scripts/smoke_paddleocr_vl_5070ti.py   # 한국어 문서는 --pdf 지정
docker compose stop paddleocr-vl ocr-paddle    # 대상만 정지 (⚠ ovis/cuda와 동시 기동 금지)
docker compose --profile paddle down           # 전체 정리 (ocr-cpu도 함께 내려감)
```

캐시 삭제:

```bash
# 실제 볼륨 이름에는 compose 프로젝트 접두사가 붙는다(기본 = 이 디렉터리 이름) — 확인 후 지운다
docker compose --profile paddle config --format json | python3 -c \
  "import json,sys;v=json.load(sys.stdin)['volumes'];print(v['paddle-hf-cache']['name'],v['paddle-x-cache']['name'])"
docker volume rm <위에서 확인한 두 이름>   # 예: <PROJECT>_paddle-hf-cache <PROJECT>_paddle-x-cache
```

⚠ **비루트 실행(uid 1000)으로 전환됨** (`services/paddleocr_vl/Dockerfile`).
PaddleX 캐시는 `$HOME` 기준이라 컨테이너 마운트 지점이 `/root/.paddlex` →
**`/home/app/.paddlex`** 로 바뀌었다(볼륨 이름 `paddle-x-cache`는 그대로). 비어 있는
볼륨은 이미지의 chown 결과를 물려받지만, **이미 root 소유로 채워진 기존 볼륨은
자동으로 바뀌지 않는다** — 업그레이드 후 캐시 쓰기가 실패하면 한 번만 실행한다.
compose가 `cap_drop: [ALL]`을 걸어 두므로 이 실행에만 두 캡을 돌려준다:

```bash
docker compose --profile paddle run --rm --no-deps --user 0 --cap-add CHOWN \
  --cap-add DAC_OVERRIDE --entrypoint chown paddleocr-vl \
  -R 1000:1000 /data/hf /home/app/.paddlex
```

예전 안내의 `docker run -v <접두사 없는 이름>:… alpine chown …`은 존재하지 않는 볼륨을
새로 만들고 exit 0으로 끝나 아무것도 고치지 못했다.

compose는 이 sidecar에 로그 로테이션(json-file 10MB×3)과 호스트 RAM 상한
`PADDLE_MEM_LIMIT`(기본 24g)을 건다 — VRAM과 무관한 안전판이며 호스트 RAM이 작으면
`.env`로 낮춘다.

## 결과 스키마 → 프로토콜 변환

`services/paddleocr_vl/app/adapter.py`가 공식 결과(`res.json`)의
`parsing_res_list[].block_bbox(픽셀)/block_label/block_content/block_order`를
내부 프로토콜로 변환한다:

- bbox: 픽셀 → [0,999] 정규화 (2%+2px 초과 이상치는 폐기+warning)
- 라벨 → 정규화 타입 (`doc_title/paragraph_title→title`, `chart/seal→image`,
  `number→page_number` 등)
- markdown 재조립: `block_order` 순. 공식 기본과 동일하게 `number/footnote/
  header(+image)/footer(+image)/aside_text`는 markdown 제외·블록 보존
- 수식은 구분자 없으면 `\[ … \]`로 감싼다. 표 HTML·셀 내 줄바꿈·한글 음절/자모/
  한자/영문 혼용은 무변형 보존 (`tests/test_adapter.py`가 고정)
- 공식 markdown dict(base64 이미지 포함)는 **사용하지 않는다** — 이미지 바이너리
  금지 원칙. figure는 bbox로 backend가 원본 페이지에서 직접 crop
- 문서 본문에 리터럴로 실린 `[[FIGURE:`는 **조립한 페이지 markdown에서만**
  `&#91;&#91;FIGURE:`로 이스케이프한다(렌더하면 같은 글자, placeholder로는 안 읽힘).
  블록 `content`는 리터럴 그대로 둔다 — placeholder로 읽히는 것은 페이지 markdown뿐이다
  (OCR_ENGINE_PROTOCOL.md §markdown 규약)
- fixture: `services/paddleocr_vl/tests/fixtures/official_page.json` — 실 GPU에서
  스키마 드리프트 발견 시 실측 결과로 교체하고 어댑터를 함께 갱신할 것

## 로드 재시도 · 고착 CUDA 오류 자가 재시작

- **로드 재시도**: 첫 기동의 HF 스냅샷 다운로드·파이프라인 로드가 일시적으로 실패하면
  15/30/60/120초 뒤 다시 시도한다(최대 5회). 그동안 health는 `status:ok`·
  `model_loaded:false`·`load_retry:{…}`라 잡은 `모델 로드 재시도 대기 중…`으로 기다린다.
  CUDA 가드·설치 누락(`ImportError`)·HTTP 401/403/404는 곧바로 `load_error`다.
- **웨지 신고**: 연속 추론 실패가 임계를 넘으면 모델은 유지한 채 health를
  `status:"error"`로 바꾸고, 다음 성공에서 스스로 푼다(OvisOCR2와 같은 규칙). 오탐일 수
  있어 backend는 `model_loaded:true`인 이 신고로 잡을 실패시키지 않고 잡마다 한 번
  경고로만 남긴다.
- **고착(sticky) CUDA 오류**: illegal memory access·illegal instruction처럼 CUDA 컨텍스트가
  망가지는 오류가 나면 같은 프로세스의 이후 GPU 작업이 전부 실패한다. 그 요청에 503을
  돌려주고 health에 `restarting:true`를 올린 뒤 1.5초 뒤 종료 코드 3으로 끝난다 — compose
  `restart: unless-stopped`가 컨테이너를 다시 띄우고, backend는 `sidecar 재시작 대기 중…`으로
  기다렸다가 그 페이지만 다시 보낸다.

## OOM 완화 순서

1. `OCR_REMOTE_PAGE_CONCURRENCY=1` 확인 (기본값)
2. `PADDLEOCR_MAX_PIXELS` 설정/감소 (예: `4194304` — sidecar도 OOM 시 자동 1회 강등)
3. (출력 토큰 상한 — 공식 파이프라인 노출 옵션 없음, 해당 없음)
4. `RENDER_DPI` 감소 (backend 측 입력 축소)
5. (gpu_memory_utilization — paddle in-process 경로에는 해당 옵션 없음)
6. layout GPU 해제 = `PADDLEOCR_DEVICE=cpu` (전체 CPU — 최후에만)
7. 더 작은 엔진(OvisOCR2) 선택

## 문제 해결

| 증상 | 확인 |
|---|---|
| health `status:error` | `docker compose --profile paddle logs paddleocr-vl` (paddle import/CUDA 오류가 흔함) |
| `The GPU architecture is not supported` 류 | 드라이버가 CUDA 12.9+ 지원인지, lock의 `paddlepaddle-gpu`가 cu129 CDN 휠인지 확인 |
| arm64 호스트에서 빌드 실패 | 정상 — lock의 paddlepaddle-gpu 휠이 x86_64 전용이라 이 이미지는 linux/amd64만 빌드된다 |
| 빌드가 해시 불일치로 실패 | lock을 손으로 고치지 말고 `requirements.in`을 고친 뒤 lock 머리말의 명령으로 다시 만든다 |
| 모델 다운로드 실패 | `PADDLE_PDX_MODEL_SOURCE=huggingface`, HF_TOKEN(프라이빗 미러 시), 네트워크 — 일시적 실패는 자동 재시도(`/api/health`의 `provider_health.load_retry`) |
| 컨테이너가 종료 코드 3으로 재시작됨 | 고착 CUDA 오류 뒤 자가 재시작(위 §고착 CUDA 오류) — 반복되면 로그의 원인을 보고 OOM 완화 순서대로 줄인다 |
| 캐시 쓰기 `Permission denied` | 비루트(uid 1000) 전환 전에 만들어진 볼륨 소유권 — 위 §캐시 삭제의 chown 1회 |
| 한글 깨짐/누락 | smoke를 `--pdf 한국어문서.pdf`로 실행해 재현(텍스트 레이어가 없는 스캔본은 `--expect-korean`) — 입력에 한글이 있는데 출력에 없으면 smoke가 실패한다. adapter는 무변형 보존이므로 모델/렌더 단 확인 |

## 실행 스레드 정책 (중요 — 실측 기반)

`PaddleModel`은 파이프라인 **생성·워밍업·추론·캐시 해제를 전용 스레드 하나
(`max_workers=1` 실행기)** 에서만 수행한다. 이유는 실측 회귀다:

> 파이프라인을 만든 스레드에서 추론하면 정상인데, **같은 객체를 FastAPI 요청
> 스레드에서 호출하면** 파이프라인의 `vlm` 워커가 static graph 모드로 올라와
> `RuntimeError: Exception from the 'vlm' worker: int(Tensor) is not supported in
> static graph mode`로 **모든** 추론이 실패했다 (2026-07-20 RTX 5070 Ti, paddleocr
> 3.6.0 / paddlepaddle-gpu 3.3.1). 소유 스레드로 고정한 뒤 4연속 실행 모두 정상.

따라서 다음을 지켜야 한다 (수정 시 회귀 위험):

- paddle API를 요청 경로에서 직접 호출하지 않는다 — health의 GPU 이름/총량은
  로드 시 1회 수집한 캐시값이고, 가용량만 `nvidia-smi`로 읽는다.
- 로드 직후 소유 스레드에서 워밍업 추론을 1회 수행해 `vlm` 워커 생성 시점을
  통제한다(첫 사용자 요청 지연도 함께 제거된다).
- 이 실행기가 곧 직렬화 장치라 별도 추론 락은 두지 않는다 (단일 GPU 정책과 일치).

## Known limitations

- 공식 검증 GPU는 RTX 5070 — 5070 Ti는 목록 포함이지만 자체 smoke 필수.
- 페이지 단위 스트리밍 (토큰 델타 없음).
- 취소 시 진행 중 페이지의 추론은 완주 후 폐기.
- chart 인식(`use_chart_recognition`) 등 부가 옵션은 기본 비활성 (공식 기본값).

## 검증 상태

- 구현·어댑터 fixture 테스트(20)·backend 통합 테스트·`docker compose config`: **완료**
- **RTX 5070 Ti 실 runtime 검증 완료 (2026-07-20, `scripts/smoke_paddleocr_vl_5070ti.py` exit 0)**:
  - 환경: WSL2 · driver 591.86 · paddlepaddle-gpu 3.3.1(cu129) · paddleocr 3.6.0 ·
    HF 스냅샷이 고정 revision `66317acc…`로 내려오는 것을 로그로 확인
  - health: `gpu=NVIDIA GeForce RTX 5070 Ti`, `gpu_total_mb=16302`, `gpu_free_mb` 정상 보고
  - 영문 샘플 2페이지: **6.8s (3.4s/페이지)**, figure 2 · 표 1 · layout 정상
  - **한국어 문서 1페이지: 8.1~12.1s**, 한글 음절·자모(ㄱㄴㄷ)·한자·영문 혼용·제목
    계층·표 HTML·figure crop·각주 모두 보존 확인 (3회 연속 재현)
  - **한국어 정확도는 OvisOCR2보다 명확히 우수**(같은 입력 대조): Ovis가
    "혼용된→훈련된", "한글 자모 ㄱㄴㄷ→한국 자료 717"로 오독한 구간을 정확히
    인식했다. 반면 속도는 워밍업 후 Ovis가 2~6배 빠르다 — docs/OCR_BENCHMARK.md
  - **peak VRAM 8,285MB / 16,302MB** — layout+VL을 모두 GPU에 올린 기본 설정에서 OOM 없음
  - 실측으로 발견해 수정한 항목: `vlm` 워커 스레드 이슈(위 §실행 스레드 정책),
    `paddle.device.cuda.mem_get_info` 부재(→ nvidia-smi 조회), 미지 라벨
    `figure_title`·`display_formula`(→ LABEL_MAP 추가)
- **실문서 확장 검증 (2026-07-21)**: 실제 arxiv 논문 2504.19874v1(25p, 2단, 수식
  밀집)을 441s(**17.6s/p** — 밀집 학술 텍스트에서 Ovis의 5배 느림)에 처리, 저자·
  초록·2단 읽기순서·인라인 수식 정확, figure 14 추출, 실패 0. 스캔 시뮬 PDF는
  제목·표·수식 구조는 OCR하나 열화 심한 CJK 줄은 오독(스캔 견고성 한계).
  `OCR_REMOTE_PAGE_CONCURRENCY` 1↔4 출력 바이트 동일(순서 보존 확인).
- 실측으로 보강한 라벨: `figure_title`·`table_title`·`chart_title`·`display_formula`·
  `formula_number`·`reference_content`·`algorithm` → 매핑 추가. **미지 라벨은 내용
  손실이 아니라 경고 로그일 뿐**(content는 markdown에 text로 보존).

[PaddleOCR-VL NVIDIA Blackwell 가이드]: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html
