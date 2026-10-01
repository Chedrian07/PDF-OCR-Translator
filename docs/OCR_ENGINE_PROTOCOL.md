# sidecar 내부 프로토콜 v1 — backend ↔ CUDA OCR sidecar

메인 backend(`app/sidecar/`)와 모델 sidecar(`services/ovisocr2`,
`services/paddleocr_vl`)가 공유하는 HTTP 계약. 스키마 사본이 양쪽에 존재하므로
(독립 배포 단위 — 코드 공유 없음) 변경 시 **양쪽 + 이 문서**를 함께 갱신한다.

- **깨지는 변경**(필수 필드 추가·삭제, 기존 필드의 의미 변경)은 `protocol_version`을 올린다.
- **추가 필드**는 올리지 않는다. 양쪽 스키마가 모르는 키를 무시하고(`extra="ignore"`),
  새 backend는 필드가 없으면 기본값(= 예전 동작)으로 읽으므로 옛 sidecar·옛 backend와
  그대로 맞물린다. 아래 `load_retry`·`restarting`(health)과 `truncated`(page)가 이
  방식으로 들어왔다.

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
  "load_error": null,
  "load_retry": null,                  // 추가 필드 — 일시적 로드 실패 뒤 재시도 대기 중이면
                                       //  {"attempt":1,"max_attempts":5,"next_retry_s":15.0,"last_error":"…"}
  "restarting": false                  // 추가 필드 — 추론 엔진이 죽어 컨테이너 재시작을 기다리는 중
}
```

backend는 `protocol_version==1`과 `engine`이 설정된 엔진과 일치하는지 검증한다
(불일치 = `OCR_SIDECAR_URL` 오배선 — 명확한 오류).

`model_loaded=false`이면 backend는 아래 조합으로 기다릴지 실패할지를 정한다
(`engine/sidecar.py::_check_ready`):

| health | 뜻 | backend 동작 · 잡 대기 문구 |
|---|---|---|
| `status:"ok"`, `load_retry:null`, `restarting:false` | 첫 로드 중(다운로드·컴파일) | 대기 — `모델 로딩 대기 중…` |
| `status:"ok"`, `load_retry:{…}` | 일시적 로드 실패 뒤 재시도 대기 | 대기 — `모델 로드 재시도 대기 중…` |
| `restarting:true` (**`status`와 무관**) | 추론 엔진 사망 → 프로세스 종료·재기동 예정 | 대기 — `sidecar 재시작 대기 중…` |
| `status:"error"`, `restarting:false` | 진짜 로드 실패(`load_error`) | 대기 없이 잡 오류 |

- `load_retry`는 `attempt`·`max_attempts`(0 이상 정수), `next_retry_s`(유한한 0 이상 수),
  `last_error`(문자열, backend가 300자로 자른다)만 받는다. 형식이 이상한 값은 버린다 —
  health 전체를 스키마 위반으로 만들면 기다려야 할 상태가 하드 실패가 된다.
  `restarting`과 page의 `truncated`는 JSON `true`만 참으로 읽는다.
- 두 필드는 `/api/health`의 `provider_health`에 그대로 실린다
  (`{status, runtime, version, model_loaded, gpu_total_mb, gpu_free_mb, load_retry, restarting}`).

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
    "warnings": ["GPU 메모리 부족으로 해상도 강등(...)"],
    "truncated": false                 // 추가 필드 — 출력 토큰 상한에서 끊긴 페이지
  },
  "timings": {"preprocess_ms": 12.0, "inference_ms": 1830.5, "postprocess_ms": 3.1}
}
```

오류: `503`(모델 미로드, 또는 추론 엔진이 죽어 재시작 예정 — detail에 사유. backend는
이를 **일시적**으로 보고 대기 후 그 페이지만 재시도한다 — 아래 §잡 도중 재시작 참조) ·
`422`(요청/옵션 스키마 위반) · `400/413`(이미지 이상/크기, 폼 필드 1000개 초과) ·
`502`(추론 실패 — OOM 1회 강등 재시도 후에도 실패).

### 잘린 페이지 (`truncated`)

OvisOCR2는 vLLM `finish_reason=length`(페이지당 `OVIS_MAX_OUTPUT_TOKENS` 도달)일 때
`truncated:true`와 warning을 함께 보낸다. PaddleOCR-VL과 옛 sidecar는 필드를 보내지 않는다
(= `false`, 정상 처리). backend(`SidecarEngine._truncation_verdict`)는 잘린 페이지를
충실도 게이트와 같은 지표로 PDF 텍스트 레이어와 대조한다:

- 텍스트 레이어가 **믿을 만하고** 점수가 `OCR_FIDELITY_THRESHOLD`(기본 0.70) 미만이면
  `SidecarOutputTruncated`(`OutputLimitError`의 하위, `limit_label`='sidecar 출력 토큰
  상한', `retry_same_page=False`)로 넘긴다. runner는 이를 MAX_LENGTH 잘림과 같은 길로
  복구한다 — 1쪽 청크는 텍스트 레이어 폴백(없으면 플레이스홀더), 여러 쪽 청크는 끝까지
  생성된 앞 페이지를 지키고 잘린 페이지부터 다시 처리한다. 같은 페이지를 GPU에 다시
  보내지 않는다(결정적 결과라 같은 잘림이 난다 — 직후 `run_single`은 1회용 재생 표시로
  GPU 호출 없이 같은 예외를 받는다).
- 그 밖(스캔 문서·대조 불가·점수가 기준 이상·임계값 ≤ 0)은 잘린 출력을 그대로 쓰고
  `출력 토큰 상한에서 잘린 페이지 — … 잘린 출력을 그대로 씁니다` 경고를 남긴다. 표·수식·
  그림 구조가 살아 있는 출력을 평문 텍스트 레이어나 빈 페이지로 바꾸지 않기 위해서다.
- 텍스트 레이어 복구는 그림·표 구조를 잃는다 — 임계값을 바꾸면 이 판정도 함께 바뀐다.

### block type (정규화 어휘)

`title` `text` `table` `formula` `image` `header` `footer` `footnote`
`page_number` `unknown`. 프로바이더 라벨 별칭(`figure/chart/seal→image`,
`equation→formula`, `doc_title/paragraph_title→title`, `number→page_number` 등)은
backend `protocol.TYPE_ALIASES`가 정규화한다. figure/image는 내부적으로 `image`
하나로 통일된다.

### markdown 규약

- figure 자리는 `[[FIGURE:n]]`만 허용 (`n` = 0–999, 페이지당 최대 64개).
  최종 `![](images/…)` 치환은 backend materializer의 몫.
- `[[FIGURE:n]]`은 **앱이 소유한 참조 문법**이다(`<PAGE>`와 동일). 문서 본문에 리터럴로
  실린 `[[FIGURE:`(모델이 그대로 옮겨 적은 글자)는 **sidecar가 이스케이프해서** 보낸다:
  `&#91;&#91;FIGURE:` — 숫자 문자 참조라 렌더하면 `[[FIGURE:`로 보이지만 placeholder
  정규식에는 걸리지 않는다. `\[`로 이스케이프하면 렌더러가 디스플레이 수식(`\[ … \]`)으로
  읽으므로 쓰지 않는다.
  - OvisOCR2(`parser.py`): 엉뚱한 `<img>` 태그를 지운 뒤, 유효한 figure 태그 **사이의
    텍스트 조각마다** 이스케이프한다(유효 태그가 만든 placeholder는 건드리지 않는다).
  - PaddleOCR-VL(`adapter.py`): 조립한 페이지 markdown에서만 이스케이프한다. 블록
    `content`는 리터럴을 그대로 둔다 — placeholder로 읽히는 것은 페이지 markdown뿐이다.
- backend(`protocol.sanitize_page`)는 제어 문법(`<PAGE>`·`<|…|>`)을 placeholder **사이
  조각마다** 따로 지운다. 통째로 지우면 `[<PAGE>[FIGURE:0]]`처럼 지운 자리 양쪽이 이어져
  placeholder가 새로 생긴다 — 그래도 지운 뒤 생긴 것은 문서 글자이므로 리터럴
  `&#91;&#91;FIGURE:n]]`로 되돌린다.
- 방어층: 살아남은 image 블록 index에 대응 crop이 없는 placeholder는 제거하고, 같은
  index가 여러 번 나오면 **한 줄을 혼자 차지한 것**(실제 figure 자리, 없으면 첫 것)만
  살리고 나머지는 리터럴 표기로 바꾼 뒤 warning을 남긴다 — figure 위치 납치·중복 삽입 방지.
- 표는 HTML `<table>…</table>`, 수식은 LaTeX(`\(…\)`/`\[…\]`/`$$`), 나머지는
  표준 Markdown. `<|…|>` 특수 토큰 패턴은 backend가 제거한다.

## backend 클라이언트 정책 (`app/sidecar/client.py`)

| 항목 | 값(환경변수) |
|---|---|
| 연결/읽기/health 타임아웃 | `OCR_SIDECAR_CONNECT_TIMEOUT_S`=10 / `OCR_SIDECAR_READ_TIMEOUT_S`=600 / `OCR_SIDECAR_HEALTH_TIMEOUT_S`=5 |
| 응답 크기 상한 | `OCR_SIDECAR_MAX_RESPONSE_MB`=20 (스트리밍 계수, 초과 즉시 중단) |
| 재시도 | 연결 **수립 실패에만** client가 `OCR_SIDECAR_RETRIES`=1회. 읽기 타임아웃은 재시도하지 않는다(아래 표). 그 외 실패는 runner가 같은 청크를 1회 재시도한다 (이중 재시도 없음) |
| 페이지 동시성 | `OCR_REMOTE_PAGE_CONCURRENCY`=1 (= sidecar 엔진의 청크 크기) |

**페이지 동시성 검증 (실측 2026-07-21, RTX 5070 Ti)**: `OCR_REMOTE_PAGE_CONCURRENCY`를
1과 4로 두고 같은 문서(실제 논문 14p·4p)를 처리한 결과 **markdown이 바이트 단위로
동일**했다(Ovis·Paddle 양쪽) — `_iter_concurrent`의 순서 보존이 확인됐다. 다만
sidecar가 추론을 내부에서 직렬화하므로(Ovis `max_num_seqs=1`+락, Paddle 소유
스레드) **속도 이득은 없다**(c1 48.3s vs c4 48.7s). 기본값 1이 옳다. 동시성>1에서도
페이지 실패는 그 페이지로 격리된다 — runner는 페이지 단위 엔진의 여러 쪽 청크를
통째로 다시 보내지 않고(정상 페이지까지 GPU에서 다시 추론하게 된다) 곧바로 페이지별
처리로 내린다. 앞 페이지 실패로 버려진 형제 페이지의 sidecar 경고는 잡 경고로 올리지
않는다 — 경고는 페이지 결과를 **소비할 때** 승격되고, 그 페이지들은 다시 처리돼 자기
경고를 남긴다.

오류 분류:

| 예외 | 조건 | 성질 |
|---|---|---|
| `SidecarUnavailableError` | 연결 실패 · **HTTP 503**(재시작/모델 재로드 중) | `transient=True` — 대기하면 풀린다 |
| `SidecarTimeoutError` | 읽기 응답 시간 초과 (`SidecarUnavailableError` 서브클래스) | provider는 살아서 그 페이지를 계속 추론 중일 수 있다 — 같은 페이지를 곧장 다시 보내면 그 뒤에 줄을 서 또 타임아웃이 난다. `retry_same_page=False`라 엔진의 복귀-대기 재시도도, runner의 청크 재시도도 하지 않고 **곧바로 페이지 격리**(텍스트 레이어 → placeholder)로 간다 |
| `SidecarOutputTruncated` | `truncated:true` 페이지가 텍스트 레이어 대조에서 기준 미달 (위 §잘린 페이지) | `OutputLimitError` 하위·`retry_same_page=False` — runner의 잘림 복구 |
| `SidecarError` | 5xx 추론 실패(503 제외) | 하드 실패 |
| `SidecarProtocolError` | 스키마/크기/버전 위반, 4xx 요청 거부 — malformed provider response | 하드 실패(대기 무의미) |

모두 `EngineError` 서브클래스라 runner의 페이지 격리·placeholder·내장 텍스트
fallback 경로를 그대로 탄다(`retry_same_page` 계약은 `engine/base.py::EngineError`).

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
  (상한 `OCR_SIDECAR_MODEL_WAIT_S`=900s). 대기 중 `phase:"loading"` 진행 이벤트를
  발행하고, `note`가 대기 이유를 구분한다(`모델 로딩 대기 중…` / `모델 로드 재시도 대기
  중…` / `sidecar 재시작 대기 중…` — 위 §GET /health 표).
- sidecar가 준비되면 자동으로 진행. 사용자가 취소하면 즉시 중단(JobCanceled).
- **sidecar 쪽 로드 재시도** (`services/*/app/lifecycle.py::supervise_load`): 첫 기동의
  HF 다운로드가 네트워크 순단·5xx·rate limit으로 한 번 실패해도 굳지 않는다. 일시적
  실패는 15/30/60/120초 간격으로 최대 5회까지 다시 시도하고(대기 합 225초 — 기본
  `OCR_SIDECAR_MODEL_WAIT_S` 안), 그동안 health는 `status:"ok"`·`model_loaded:false`·
  `load_retry:{…}`라 backend가 기다린다. 재시도해도 안 풀리는 실패 — CUDA 가드
  (`PermanentLoadError`)·`ImportError`·HTTP 4xx(408·429 제외: 토큰·게이트·저장소/revision
  오타) — 와 마지막 시도의 실패만 `load_error`로 고정한다.
- **하드 실패 구분**: sidecar가 `status:"error"`(예: CUDA 가드 트립, `load_error` 포함)를
  보고하면 대기하지 않고 즉시 잡 오류로 표면화한다(대기해도 안 풀리므로). 단
  `restarting:true`면 `status`와 무관하게 기다린다.
- 프리로드(`main.create_app`)는 `transient=True` 예외를 traceback 없이 info로만
  남긴다 — sidecar가 아직 준비 중인 것은 정상적인 기동 과정이다.
- 모델은 살아 있는데 `status`만 이상한 경우(임계 기반 자가 복구형 웨지 신고)는 잡을
  실패시키지 않고 **잡마다 한 번** 경고로만 남긴다. 성공한 parse 뒤 health를 한 번 다시
  확인해 신고가 풀렸는지 본다.

### 잡 도중 재시작·모델 재로드 (HTTP 503)

`restart: unless-stopped`로 sidecar가 재기동되면 그 창의 페이지 요청은 연결 실패
또는 **HTTP 503**을 받는다. 이를 바로 실패로 확정하면 재기동+모델 로드 시간 동안의
페이지가 **전부 플레이스홀더**가 된다. 그래서 `SidecarEngine._parse_one`은:

1. `SidecarUnavailableError`(연결 실패·503)를 잡아 health 캐시를 무효화하고
   (stale `loaded=True`가 남으면 대기가 즉시 반환돼 무효화된다),
2. `"sidecar 재시작/모델 재로드 대기 중… (해당 페이지는 복귀 후 재시도)"` 경고를
   잡에 적재한 뒤,
3. 취소 가능하게 복귀를 기다리고(`_await_recovery`),
4. **그 페이지만 1회** 재요청한다(`request_id`에 `r` 접미사). 재요청도 실패하면
   그대로 전파해 기존 페이지 격리 경로를 탄다.

대기 예산은 **장애 한 번에 하나**다: 첫 대기가 시작될 때 `OCR_SIDECAR_MODEL_WAIT_S`
데드라인을 잡고 그 장애의 모든 페이지가 공유한다(페이지마다 900초씩 곱해지지 않는다).
예산을 다 쓴 뒤로는 페이지마다 health를 한 번만 확인하고 실패시킨다. 복귀하면 예산은
다음 장애를 위해 풀린다.

예외: `SidecarTimeoutError`(읽기 타임아웃)는 이 경로에서 제외된다 — provider가
그 페이지를 아직 추론 중일 수 있어 즉시 재요청하면 GPU에서 같은 페이지를 두 번
돌린다. 하드 실패(`status:"error"`·프로토콜 불일치)는 대기 없이 즉시 전파된다.

**sidecar 쪽 자가 재시작** (`lifecycle.schedule_restart`): 추론 엔진이 프로세스째 죽으면
(Ovis — vLLM `EngineDead`/`EngineCore` 사망 시그니처, Paddle — illegal memory access 같은
고착 CUDA 오류) 같은 프로세스 안에서는 회복할 수 없다. sidecar는 그 요청에 503을 돌려주고
health에 `restarting:true`를 올린 뒤, 1.5초 뒤 `os._exit(3)`으로 끝나 compose의
`restart: unless-stopped`가 컨테이너를 다시 띄우게 한다. 재시작이 예약된 뒤의 실패는
`load_error`를 세우지 않는다(backend가 하드 실패로 오인하지 않게). 시그니처는 부분 문자열
일치라 오탐이면 컨테이너가 한 번 재시작될 뿐이다. 일반 추론 실패는 종전대로 502다.
Docker 밖에서 돌리면 프로세스는 그냥 종료되고 backend는 `OCR_SIDECAR_MODEL_WAIT_S`까지
기다린다.

## 운영 — sidecar 컨테이너

- **비루트 실행 (uid 1000)**: 두 sidecar 이미지는 `USER 1000`으로 돈다
  (`services/*/Dockerfile`). 사용자가 올린 임의 PDF의 렌더 이미지를 파서에 먹이는
  쪽이라 backend보다 신뢰 경계가 바깥이고, uid는 backend 이미지와 동일하다.
  `no-new-privileges:true`와 `cap_drop: [ALL]`이 함께 걸린다.
- **볼륨 소유권 (1회 마이그레이션)**: 비어 있는 named volume은 이미지의 chown 결과
  (uid 1000)를 물려받지만, **이미 root 소유로 채워진 기존 캐시 볼륨은 자동으로
  바뀌지 않는다**. 한 번만 손봐야 한다(또는 볼륨을 지우고 모델 재다운로드).
  볼륨 이름을 추측하지 말고 compose로 실행한다 — 실제 이름에는 프로젝트 접두사가
  붙어(`<PROJECT>_ovis-hf-cache`) `docker run -v ovis-hf-cache:…`는 엉뚱한 빈 볼륨을
  새로 만들고 exit 0으로 끝난다. compose가 `cap_drop: [ALL]`을 걸어 두므로 root라도
  이 실행에만 두 캡을 돌려준다:

  ```bash
  docker compose --profile ovis run --rm --no-deps --user 0 --cap-add CHOWN \
    --cap-add DAC_OVERRIDE --entrypoint chown ovisocr2 -R 1000:1000 /data/hf
  docker compose --profile paddle run --rm --no-deps --user 0 --cap-add CHOWN \
    --cap-add DAC_OVERRIDE --entrypoint chown paddleocr-vl \
    -R 1000:1000 /data/hf /home/app/.paddlex
  ```

- PaddleX 캐시 경로가 `$HOME` 기준이라 `/root/.paddlex` → **`/home/app/.paddlex`**로
  바뀌었다(compose의 `paddle-x-cache` 마운트도 함께 변경됨).
- **의존성 잠금**: 각 sidecar의 `requirements.in`이 원천이고 `requirements.lock`이 해시까지
  고정한 출력이다(다시 만드는 명령은 lock 머리말). Dockerfile은
  `pip install --require-hashes --no-deps --only-binary :all: -r requirements.lock`으로만
  설치한다 — 해시 불일치·lock 밖 패키지·소스 빌드는 빌드 실패다.
  - PaddleOCR-VL: 전이 의존성까지 고정한 실제 잠금이다. RTX 5070 Ti 검증 이미지의 빌드
    시점(`--exclude-newer 2026-07-27`)으로 버전을 맞추고 보안 수정판(urllib3 2.8.0)만
    예외로 올렸다. `paddlepaddle-gpu`는 PyPI에 없는 cu129 빌드라 공식 CDN 휠 URL과 해시를
    직접 고정한다(추가 인덱스 없음) — 그래서 이 이미지는 **linux/amd64 전용**이다.
  - OvisOCR2: digest로 고정한 vLLM 이미지 위에 웹 계층(fastapi·starlette·python-multipart·
    uvicorn·anyio·h11·annotated-doc)만 `--no-deps`로 덧씌운다. 빌드가 `import app.main`으로
    베이스와 맞물리는지 확인한다.
  - 두 sidecar의 웹 계층은 backend와 같은 버전(fastapi 0.139.0·starlette 1.3.1·
    python-multipart 0.0.32·uvicorn 0.50.2)이라 폼 파서 상한도 같다 — 필드 1000개를 넘는
    폼은 파싱 전에 400이다(예전 starlette 0.41은 수십만 필드를 다 파싱했다).
  - `services/*/uv.lock`은 없다(지웠고 gitignore). CI `dependency-audit` 잡이 두 lock을
    pip-audit로 검사한다.
- **로그 로테이션·메모리 상한**: compose가 전 서비스에 json-file 10MB×3 로테이션을
  걸고, sidecar에는 `OVIS_MEM_LIMIT`/`PADDLE_MEM_LIMIT`(기본 24g) 메모리 상한을 둔다
  (가중치는 VRAM이고 이 상한은 호스트 RAM 안전판이다).
- 잡 데이터 볼륨은 네 backend가 `ocr-data`/`hf-cache`를 공유한다(엔진을 바꿔도 잡
  이력 유지). sidecar 모델 캐시만 런타임별로 분리 유지한다.

## 파이프라인 연결 (backend 내부)

```
ParseResponse ─ protocol.sanitize_page (clamp/폐기/상한/특수토큰 제거·placeholder 정리)
             ─ (truncated) _truncation_verdict → 그대로 쓰기(경고) | SidecarOutputTruncated
             ─ materializer.ChunkMaterializer
                 ├ figure crop → images/{k}.jpg | images/page_{local}_{k}.jpg
                 ├ boxes.json (픽셀 crop 좌표 + 페이지 크기 — 벤더 P13 계약)
                 ├ result_with_boxes[_{local}].jpg (타입별 색 오버레이)
                 ├ raw_pages.json — 기존 layout.py 문법을 normalized block에서 합성
                 │   (inline det만 사용: 문서 순서 == image crop_index 순서 보장)
                 │   layout=full 엔진만 쓴다(write_raw). figure_only(OvisOCR2)는 쓰지 않아
                 │   layout.json이 생기지 않는다 → has_layout=false, 좌표 기능 대신 /html
                 └ [[FIGURE:n]] → ![](images/…) 치환
             → 기존 IncrementalMerger (무수정)
```

sidecar 경고(정화로 버린 블록·절단·sidecar가 보고한 강등 등)는 페이지 결과를 병합에
**소비할 때** 잡 경고로 승격한다(위 §backend 클라이언트 정책).
