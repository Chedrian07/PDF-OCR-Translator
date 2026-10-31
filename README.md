# PDF OCR Translator — Unlimited-OCR-2

PDF 업로드 → OCR 스트리밍 → 읽기 뷰·한국어 번역 → HTML/Markdown/ZIP/PDF 내보내기.
모든 제품 OCR 경로는 **baidu/Unlimited-OCR-2**를 사용합니다. CUDA는 공식 전용 SGLang,
CPU·Metal은 torch, Apple Silicon 기본 경로는 MLX입니다.

## 빠른 시작 (Docker)

```bash
cp .env.example .env
# CPU: http://localhost:8000
docker compose up -d --build ocr-cpu
# CUDA: http://localhost:8001
docker compose --profile cuda up -d --build unlimited-ocr2 ocr-cuda
# Apple Silicon
make setup-mlx
make dev
```

호스트 `.env`의 오타를 확인하려면 값 대신 키 이름만 검사합니다.

```bash
docker compose exec -T ocr-cpu python -c 'import sys; from dotenv import dotenv_values; from app.config import unknown_dotenv_key_warnings; print("\n".join(unknown_dotenv_key_warnings(dotenv_values(stream=sys.stdin))))' < .env
```

모델 revision은 `c14071b549f4fcab10909f3c4acc6a2edb366947`로 고정합니다.
기존 엔진 선택·모델 ID·fast-decode 설정은 제거해야 합니다. 자동 구형 모델 복귀나
텍스트 레이어/Tesseract 대체 OCR은 없습니다.
설치·환경변수 이전·VRAM 설정·플랫폼 검증은 [운영 문서](docs/UNLIMITED_OCR_2.md)를 보세요.
과거 릴리스 이미지에는 v1 모델이 들어 있으므로 이번 전환은 소스에서 새로 빌드합니다.

PDF → Markdown ZIP 작업도 지원합니다. 업로드에서 해당 워크플로를 선택하면 페이지별
Markdown과 그림을 ZIP으로 내려받습니다. 기존 작업 API와 결과 다운로드 경로는 유지됩니다.

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

## 개발 및 검증

```bash
make setup
make dev
make test
npm --prefix frontend run test:e2e-mock
E2E_BASE_URL=http://127.0.0.1:8001 npm --prefix frontend run test:e2e
```

실제 설정된 번역 공급자까지 검증하려면 `npm --prefix frontend run test:e2e-live`를 사용합니다.
인증 오류는 성공으로 세지 않고 별도로 기록합니다.

실모델 측정은 `scripts/benchmark_unlimited2.py`를 사용합니다. 모델 없는 테스트는
실모델 품질·성능 검증을 대신하지 않습니다. Apple Silicon 실기기 검증은 사용자 측에서 진행합니다.

## 문서

- [Unlimited-OCR-2 설치·운영·검증](docs/UNLIMITED_OCR_2.md)
- [전환 검증 기록](docs/UNLIMITED_OCR_2_VALIDATION.md)
- [아키텍처](docs/ARCHITECTURE.md)
- [네이티브 모듈](native/README.md)

## 라이선스

MIT. 벤더 코드의 라이선스와 출처는 각 `backend/app/vendor` 디렉터리를 확인하세요.
