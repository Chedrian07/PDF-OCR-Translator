# sidecar 내부 프로토콜 v1 — backend ↔ CUDA OCR sidecar

메인 backend(`app/sidecar/`)와 모델 sidecar(`services/ovisocr2`,
`services/paddleocr_vl`)가 공유하는 HTTP 계약. 스키마 사본이 양쪽에 존재하므로
(독립 배포 단위 — 코드 공유 없음) 변경 시 **양쪽 + 이 문서 + protocol_version**을
함께 갱신한다.

## 설계 원칙

- **모델 출력은 비신뢰 입력**: sidecar 응답에는 파일 경로·이미지 바이너리·
  외부 URL·secret이 없다. figure는 `[[FIGURE:n]]` placeholder와 bbox만 전달하고,
  crop 파일 생성·파일명 결정은 전적으로 메인 backend(materializer)가 한다.
- **bbox는 [0,999] 정규화 정수** (x1<x2, y1<y2). 경미한 초과만 backend가 clamp,
  심각한 이상은 블록 폐기 + warning. NaN/문자열/무한대/거대값은 스키마 거부.
- **페이지 단위**: 한 `/v1/parse` 요청 = 페이지 이미지 1장. 멀티페이지 문맥 없음.
- sidecar는 GPU를 독점하고, backend 프로세스는 GPU를 사용하지 않는다.

## GET /health

모델 로딩 전에도 200을 반환한다 — 가용성은 `model_loaded`/`status`로 구분.

```json
{
  "status": "ok",                      // "ok" | "error"(로드 실패 — load_error 참조)
  "protocol_version": 1,
  "engine": "ovisocr2",                // "ovisocr2" | "paddleocr_vl"
  "model_id": "ATH-MaaS/OvisOCR2",
  "model_revision": "65c619d374b55d4152e85150fc1b003700bc1f0c",
  "runtime": "vllm",                   // "vllm" | "paddleocr"
  "runtime_version": "0.22.1",
  "device": "cuda",
  "dtype": "bfloat16",
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "gpu_total_mb": 16384,
  "gpu_free_mb": 12000,
  "model_loaded": true,
  "load_error": null
}
```

backend는 `protocol_version==1`과 `engine`이 설정된 엔진과 일치하는지 검증한다
(불일치 = `OCR_SIDECAR_URL` 오배선 — 명확한 오류).

## POST /v1/parse (multipart/form-data)

| 필드 | 타입 | 설명 |
|---|---|---|
| `file` | file | 페이지 이미지 1장 (PNG/JPEG, 기본 상한 **128MB**(`OVIS_MAX_UPLOAD_MB`/`PADDLEOCR_MAX_UPLOAD_MB`)·60M픽셀) |
| `page_index` | int | 청크 내 로컬 페이지 인덱스 (에코백용) |
| `request_id` | str | 로그 상관관계용 (내용 로깅 금지) |
| `options` | JSON str | 제한된 스키마 — 미지 키는 422. 허용: `max_pixels`, (ovis) `max_output_tokens` |

응답 200:

```json
{
  "protocol_version": 1,
  "engine": "ovisocr2",
  "model_id": "ATH-MaaS/OvisOCR2",
  "model_revision": "…",
  "page": {
    "page_index": 0,
    "markdown": "# 제목\n\n[[FIGURE:0]]\n\n본문…",
    "blocks": [
      {"type": "image", "bbox": [100, 200, 800, 700], "content": "",
       "order": 4, "figure_index": 0, "confidence": null}
    ],
    "provider_raw": "(디버그용 원문 — 100k자 상한)",
    "warnings": ["GPU 메모리 부족으로 해상도 강등(...)"]
  },
  "timings": {"preprocess_ms": 12.0, "inference_ms": 1830.5, "postprocess_ms": 3.1}
}
```

오류: `503`(모델 미로드 — detail에 사유. backend는 이를 **일시적**으로 보고 대기 후
재시도한다 — 아래 §재시작·모델 재로드 참조) · `422`(요청/옵션 스키마 위반) ·
`400/413`(이미지 이상/크기) · `502`(추론 실패 — OOM 1회 강등 재시도 후에도 실패).

### block type (정규화 어휘)

`title` `text` `table` `formula` `image` `header` `footer` `footnote`
`page_number` `unknown`. 프로바이더 라벨 별칭(`figure/chart/seal→image`,
`equation→formula`, `doc_title/paragraph_title→title`, `number→page_number` 등)은
backend `protocol.TYPE_ALIASES`가 정규화한다. figure/image는 내부적으로 `image`
하나로 통일된다.

### markdown 규약

- figure 자리는 `[[FIGURE:n]]`만 허용 (`n` = 0–999, 페이지당 최대 64개).
  최종 `![](images/…)` 치환은 backend materializer의 몫.
- `[[FIGURE:n]]`은 **앱이 소유한 참조 문법**이다(`<PAGE>`와 동일). 살아남은 image
  블록 index당 **첫 placeholder 하나만** 유효하고, 대응 crop이 없거나 중복된
  placeholder(문서 본문에 리터럴로 실려 모델이 그대로 옮겨 적은 경우 포함)는
  `protocol.sanitize_page`가 제거하고 warning을 남긴다 — figure 위치 납치·중복
  삽입 방지.
- 표는 HTML `<table>…</table>`, 수식은 LaTeX(`\(…\)`/`\[…\]`/`$$`), 나머지는
  표준 Markdown. `<|…|>` 특수 토큰 패턴은 backend가 제거한다.

## backend 클라이언트 정책 (`app/sidecar/client.py`)

| 항목 | 값(환경변수) |
|---|---|
| 연결/읽기/health 타임아웃 | `OCR_SIDECAR_CONNECT_TIMEOUT_S`=10 / `OCR_SIDECAR_READ_TIMEOUT_S`=600 / `OCR_SIDECAR_HEALTH_TIMEOUT_S`=5 |
| 응답 크기 상한 | `OCR_SIDECAR_MAX_RESPONSE_MB`=20 (스트리밍 계수, 초과 즉시 중단) |
| 재시도 | 연결 **수립 실패에만** `OCR_SIDECAR_RETRIES`=1회. 그 외는 runner의 청크 1회 재시도가 담당 (이중 재시도 없음) |
| 페이지 동시성 | `OCR_REMOTE_PAGE_CONCURRENCY`=1 (= sidecar 엔진의 청크 크기) |

**페이지 동시성 검증 (실측 2026-07-21, RTX 5070 Ti)**: `OCR_REMOTE_PAGE_CONCURRENCY`를
1과 4로 두고 같은 문서(실제 논문 14p·4p)를 처리한 결과 **markdown이 바이트 단위로
동일**했다(Ovis·Paddle 양쪽) — `_iter_concurrent`의 순서 보존이 확인됐다. 다만
sidecar가 추론을 내부에서 직렬화하므로(Ovis `max_num_seqs=1`+락, Paddle 소유
스레드) **속도 이득은 없다**(c1 48.3s vs c4 48.7s). 동시성>1은 "안전하지만 무익"
하며 기본값 1이 옳다.

오류 분류:

| 예외 | 조건 | 성질 |
|---|---|---|
| `SidecarUnavailableError` | 연결 실패 · **HTTP 503**(재시작/모델 재로드 중) | `transient=True` — 대기하면 풀린다 |
| `SidecarTimeoutError` | 읽기 응답 시간 초과 (`SidecarUnavailableError` 서브클래스) | provider는 살아서 그 페이지를 계속 추론 중일 수 있다 — **엔진의 복귀-대기 재시도에서 제외**(같은 페이지 즉시 재요청 = GPU 중복 추론). 상위 runner의 청크 재시도만 담당 |
| `SidecarError` | 5xx 추론 실패(503 제외) | 하드 실패 |
| `SidecarProtocolError` | 스키마/크기/버전 위반, 4xx 요청 거부 — malformed provider response | 하드 실패(대기 무의미) |

넷 모두 `EngineError` 서브클래스라 runner의 페이지 격리·placeholder·내장 텍스트
fallback 경로를 그대로 탄다.

## 취소 의미론과 한계

**보장되는 것**: 취소 후 도착한 결과는 **절대 병합되지 않는다.** backend는 요청
전·후는 물론 대기 중에도 0.2초 주기로 cancel을 확인하고, 취소가 관측되면 즉시
`JobCanceled`를 올려 호출자를 풀어 준다 — 사용자 관점의 응답성은 즉각적이다.

**보장되지 않는 것 (실측 확인된 한계)**:

- 추론이 진행 중인 동안에는 아직 **응답 헤더가 오지 않아** 클라이언트가 잡고
  있는 `Response` 객체가 없다(`stream=True`는 헤더 수신 시점에 반환하는데,
  sidecar의 `/v1/parse`는 추론이 끝나야 헤더를 보낸다). 이 구간에서 backend가
  하는 세션 교체(`Session.close()`)는 urllib3 계약상 **in-flight 연결에 영향이
  없다**("This will not affect in-flight connections"). 즉 소켓은 살아 있고
  헬퍼 스레드는 read timeout까지 남는다.
- sidecar 내부에서 이미 시작된 추론(vLLM generate / paddle predict)은 그 페이지가
  끝날 때까지 GPU에서 계속된다. 추론은 sidecar 내부에서 직렬화되므로 다음 요청
  전에는 반드시 끝나지만, **취소 직후 GPU가 즉시 비지는 않는다.**
- 따라서 취소는 "잡을 즉시 멈추고 부분 결과를 보존"하는 의미이지, "GPU 작업을
  즉시 회수"하는 의미가 아니다. 즉시 회수가 필요하면 sidecar 컨테이너 재시작이
  유일한 수단이다.
- `OCR_REMOTE_PAGE_CONCURRENCY>1`에서 한 페이지가 실패하면 형제 요청에도 중단
  신호를 보내지만, 위와 같은 이유로 **이미 전송된 요청은 sidecar에서 완주**한다
  (다음 재시도가 그 뒤에 줄을 선다 — 동시성을 올릴 때 감안할 것).

## 라이브 뷰 스트리밍 (sidecar 엔진)

sidecar 엔진은 페이지 단위라 토큰 스트림이 없지만, **라이브 3-패널 뷰는 그대로
동작한다**. 페이지가 완료되면 SidecarEngine이 그 페이지를 **그라운딩 토큰 표현**으로
sink에 발행한다(`_live_stream_text`):

- figure는 `<|det|>image [x1, y1, x2, y2]<|/det|>`로 → 왼쪽 "원본+레이아웃" 패널의
  실시간 박스 오버레이가 그려진다 (Unlimited와 동일한 파서 재사용).
- 텍스트·표·수식은 markdown 그대로 흘러 → RAW·미리보기 패널이 채워진다.

이 스트림 표현은 **라이브 뷰 전용**이며, 저장/병합되는 결과 markdown(`![](images/…)`)과
분리되어 있다. 텍스트가 페이지 단위로 도착하는 것은 모델 특성상 불가피하며
(sub-page 토큰 스트림 아님), 프론트는 "페이지 단위 갱신" 칩으로 이를 명시한다.

## 최초 기동 — 모델 로딩 대기

sidecar의 첫 모델 로드는 다운로드 + (Ovis) vLLM 컴파일로 수 분 걸린다. 이 창에
업로드된 잡은 **실패하지 않고 대기**한다:

- 워커가 `SidecarEngine.wait_until_ready(cancel, on_wait)`로 취소 가능하게 폴링
  (상한 `OCR_SIDECAR_MODEL_WAIT_S`=900s). 대기 중 `phase:"loading"` 진행 이벤트 발행.
- sidecar가 준비되면 자동으로 진행. 사용자가 취소하면 즉시 중단(JobCanceled).
- **하드 실패 구분**: sidecar가 `status:"error"`(예: CUDA 가드 트립, `load_error` 포함)를
  보고하면 대기하지 않고 즉시 잡 오류로 표면화한다(대기해도 안 풀리므로).
- 프리로드(`main.create_app`)는 `transient=True` 예외를 traceback 없이 info로만
  남긴다 — sidecar가 아직 준비 중인 것은 정상적인 기동 과정이다.

### 잡 도중 재시작·모델 재로드 (HTTP 503)

`restart: unless-stopped`로 sidecar가 재기동되면 그 창의 페이지 요청은 연결 실패
또는 **HTTP 503**을 받는다. 이를 바로 실패로 확정하면 재기동+모델 로드 시간 동안의
페이지가 **전부 플레이스홀더**가 된다. 그래서 `SidecarEngine._parse_one`은:

1. `SidecarUnavailableError`(연결 실패·503)를 잡아 health 캐시를 무효화하고
   (stale `loaded=True`가 남으면 대기가 즉시 반환돼 무효화된다),
2. `"sidecar 재시작/모델 재로드 대기 중… (해당 페이지는 복귀 후 재시도)"` 경고를
   잡에 적재한 뒤,
3. `wait_until_ready`(취소 가능, 상한 `OCR_SIDECAR_MODEL_WAIT_S`)로 복귀를 기다리고,
4. **그 페이지만 1회** 재요청한다(`request_id`에 `r` 접미사). 재요청도 실패하면
   그대로 전파해 기존 페이지 격리 경로를 탄다.

예외: `SidecarTimeoutError`(읽기 타임아웃)는 이 경로에서 제외된다 — provider가
그 페이지를 아직 추론 중일 수 있어 즉시 재요청하면 GPU에서 같은 페이지를 두 번
돌린다. 하드 실패(`status:"error"`·프로토콜 불일치)는 대기 없이 즉시 전파된다.

## 운영 — sidecar 컨테이너

- **비루트 실행 (uid 1000)**: 두 sidecar 이미지는 `USER 1000`으로 돈다
  (`services/*/Dockerfile`). 사용자가 올린 임의 PDF의 렌더 이미지를 파서에 먹이는
  쪽이라 backend보다 신뢰 경계가 바깥이고, uid는 backend 이미지와 동일하다.
  `no-new-privileges:true`는 그대로 유지된다.
- **볼륨 소유권 (1회 마이그레이션)**: 비어 있는 named volume은 이미지의 chown 결과
  (uid 1000)를 물려받지만, **이미 root 소유로 채워진 기존 캐시 볼륨은 자동으로
  바뀌지 않는다**. 한 번만 손봐야 한다(또는 볼륨을 지우고 모델 재다운로드):

  ```bash
  docker run --rm -v ovis-hf-cache:/v alpine chown -R 1000:1000 /v
  docker run --rm -v paddle-hf-cache:/a -v paddle-x-cache:/b alpine chown -R 1000:1000 /a /b
  ```

- PaddleX 캐시 경로가 `$HOME` 기준이라 `/root/.paddlex` → **`/home/app/.paddlex`**로
  바뀌었다(compose의 `paddle-x-cache` 마운트도 함께 변경됨).
- **로그 로테이션·메모리 상한**: compose가 전 서비스에 json-file 10MB×3 로테이션을
  걸고, sidecar에는 `OVIS_MEM_LIMIT`/`PADDLE_MEM_LIMIT`(기본 24g) 메모리 상한을 둔다
  (가중치는 VRAM이고 이 상한은 호스트 RAM 안전판이다).
- 잡 데이터 볼륨은 네 backend가 `ocr-data`/`hf-cache`를 공유한다(엔진을 바꿔도 잡
  이력 유지). sidecar 모델 캐시만 런타임별로 분리 유지한다.

## 파이프라인 연결 (backend 내부)

```
ParseResponse ─ protocol.sanitize_page (clamp/폐기/상한/특수토큰 제거)
             ─ materializer.ChunkMaterializer
                 ├ figure crop → images/{k}.jpg | images/page_{local}_{k}.jpg
                 ├ boxes.json (픽셀 crop 좌표 + 페이지 크기 — 벤더 P13 계약)
                 ├ result_with_boxes[_{local}].jpg (타입별 색 오버레이)
                 ├ raw_pages.json — 기존 layout.py 문법을 normalized block에서 합성
                 │   (inline det만 사용: 문서 순서 == image crop_index 순서 보장)
                 └ [[FIGURE:n]] → ![](images/…) 치환
             → 기존 IncrementalMerger (무수정)
```
