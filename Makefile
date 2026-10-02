# Unlimited-OCR + Localight 병합 워크스페이스 — 루트 Makefile (uv·npm 기반)
#
# GPU 스택은 compose 프로필 함정 때문에 make 타깃으로 감싸지 않는다 —
# docker-compose.yml 헤더의 blessed 명령을 그대로 쓸 것 (서비스명 반드시 명시):
#   CUDA:          docker compose up -d --build ocr-cuda                               → :8001
#   OvisOCR2:      docker compose --profile ovis up -d --build ovisocr2 ocr-ovis       → :8002
#   PaddleOCR-VL:  docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle → :8003
#   전환(정지):    docker compose stop ovisocr2 ocr-ovis
#                  (`--profile … down`은 프로필 없는 ocr-cpu까지 지우므로 전체 정리용으로만)

.PHONY: setup setup-mlx setup-metal setup-native dev dev-metal dev-textlayer test test-mps \
	test-mlx-real coverage audit e2e e2e-mock verify-e2e \
	docker-up docker-down docker-up-ollama docker-pull-model docker-down-ollama

setup:            ## backend 의존성 설치 (torch CPU)
	cd backend && uv sync --extra cpu

# Apple Silicon: MLX 엔진(기본) + torch MPS(폴백) + C++ 모듈. uv sync는 고른 extra만 남기고
# 나머지(다른 extra·uv pip로 넣은 native)를 지운다 — metal 단독 sync는 mlx와 native를,
# mlx 단독 sync는 torch를 지웠다. 그래서 둘을 함께 고르고 native를 다시 넣는다.
setup-mlx:        ## macOS Apple Silicon용 — MLX 엔진 + torch MPS 폴백 + C++ 모듈
	cd backend && uv sync --extra metal --extra mlx && uv pip install ../native

setup-metal: setup-mlx ## setup-mlx와 같다 (torch MPS만 쓰려면 make dev-metal)

setup-native:     ## C++ 가속 모듈 설치 (선택 — 없어도 순수 파이썬 폴백으로 동작)
	cd backend && uv pip install ../native

# OCR_DEVICE는 넘기지 않는다 — 미설정이면 코드 기본 auto(mlx → cuda → metal → cpu, 결정은
# 기동 로그·/api/health)라 Apple Silicon은 MLX를 쓴다. 여기서 값을 박으면 .env의 OCR_DEVICE가
# 무시된다(.env는 이미 있는 환경변수를 덮지 않는다). 지정: make dev OCR_DEVICE=cpu
# --timeout-graceful-shutdown 5: 열린 SSE 스트림·keep-alive 연결이 종료(Ctrl+C·--reload 재시작)를
# 무기한 붙잡지 않게 한다 — Dockerfile의 uvicorn과 같은 값.
# PORT: 개발 서버 포트(기본 8000) — 8000을 다른 서버가 쓰면 make dev PORT=8010
PORT ?= 8000

dev:              ## 로컬 개발 서버 — http://127.0.0.1:$(PORT) (디바이스 자동 선택, 기본 8000)
	cd backend && uv run uvicorn app.main:app --reload --host 127.0.0.1 --port $(PORT) \
		--timeout-graceful-shutdown 5

dev-metal:        ## torch MPS(Metal) 폴백으로 개발 서버 — MLX와 결과·속도 비교용
	cd backend && OCR_DEVICE=metal uv run uvicorn app.main:app --reload --host 127.0.0.1 --port $(PORT) \
		--timeout-graceful-shutdown 5

dev-textlayer:    ## 모델 다운로드 없이 textlayer 엔진으로 개발 서버
	cd backend && OCR_ENGINE=textlayer uv run uvicorn app.main:app --reload --host 127.0.0.1 --port $(PORT) \
		--timeout-graceful-shutdown 5

test:             ## 핵심 로컬 3종 — backend pytest · ruff · frontend (node --test)
	cd backend && uv run pytest
	cd backend && uv run --only-group dev ruff check .
	npm test --prefix frontend

# 기기 의존 opt-in 테스트(기본 스위트에서는 건너뛴다). extra(metal·mlx)와 uv pip로 넣은 C++ 모듈은
# uv.lock 기본 동기화 밖이라 uv run 대신 venv 인터프리터를 직접 부른다(CI backend-native 잡과 같은
# 이유). -rs: 조건이 안 맞아 건너뛴 테스트는 사유를 보인다 — 조용한 skip을 통과로 읽지 않게.
test-mps:         ## Apple Silicon torch MPS 계약 테스트 — torch·macOS 업그레이드 전후에 돌린다
	cd backend && OCR_MPS_TESTS=1 .venv/bin/python -m pytest -rs tests/test_mps_contract.py tests/test_objc_pool.py

# 코드 기본 고정 스냅샷(baidu/Unlimited-OCR@기본 MODEL_REVISION — .env의 MODEL_ID·MODEL_REVISION은
# 보지 않는다)을 로컬 HF 캐시에서만 읽는다(local_files_only). make dev를 한 번 띄우면(기본
# 프리로드) 받아진다. 테스트는 .env를 읽지 않으므로(conftest가 DISABLE_DOTENV=1) 앞단의
# scripts/require_hf_snapshot.py가 make dev와 같은 .env에서 HF 캐시 위치 키(HF_HOME·HF_HUB_CACHE
# 등 — 셸 값이 이긴다)만 넘기고, 그 캐시에 가중치가 없으면 pytest를 돌리지 않고 사유를 보인 뒤
# 종료코드 2로 멈춘다(예전에는 실가중치 2건이 LocalEntryNotFoundError로 실패해 회귀처럼 보였다).
# make setup-mlx 필요(torch CPU 로짓과 비교). fp32 MLX·torch 모델을 차례로 올려 최고 메모리 약 23GiB
# (M4 Max 실측 — 예전에는 둘을 함께 들어 44GiB), 약 15초. 32GB 이상 Mac에서 돌린다.
test-mlx-real:    ## MLX 실가중치 패리티 테스트 — mlx 업그레이드·MLX 포팅 수정·스냅샷 갱신 전후에 돌린다
	cd backend && OCR_MLX_REAL_TESTS=1 .venv/bin/python ../scripts/require_hf_snapshot.py -- \
		.venv/bin/python -m pytest -rs tests/test_mlx_model_parity.py

coverage:         ## backend 커버리지 (pytest-cov는 --with로 임시 설치 — uv.lock 무변경)
	cd backend && uv run --locked --with pytest-cov pytest --cov --cov-report=term

audit:            ## 의존성 취약점 감사 — CI dependency-audit 잡과 같은 pip-audit (네트워크 필요)
	./scripts/dependency_audit.sh

e2e:              ## 실서버 스모크 (기동된 백엔드 필요 — README §E2E)
	./scripts/smoke_e2e.sh

e2e-mock:         ## hermetic 브라우저 E2E — mock OpenAI + FakeEngine 백엔드를 직접 띄운다
	cd frontend && npm ci && npx playwright install chromium && npm run test:e2e-mock

# 동시에 두 번 돌리려면 작업 디렉터리를 갈라야 한다 — 하네스가 --work에 배타 락을
# 걸고, 겹치면 종료코드 2로 멈춘다(포트와 달리 작업 디렉터리는 자동으로 안 갈린다):
#   make verify-e2e VERIFY_ARGS="--pages 6 --work tmp/verify-e2e-2"
verify-e2e:       ## 실 PDF 전 구간 점검 — 업로드→OCR→번역→PDF→뷰어→보안 (외부 API 없음)
	cd backend && uv run python ../scripts/verify_e2e.py $(VERIFY_ARGS)

docker-up:        ## CPU 스택 기동 (프로필 없는 ocr-cpu만 — .env 불필요)
	docker compose up -d --build

docker-down:      ## 기본(프로필 없는) 서비스 정리
	docker compose down

docker-up-ollama: ## ocr-cpu + Ollama 컨테이너 (overlay: compose.ollama.yaml)
	docker compose -f docker-compose.yml -f compose.ollama.yaml up -d --build ocr-cpu ollama

docker-pull-model: ## Ollama 모델 다운로드 — 기본 qwen3:8b, `MODEL=… make docker-pull-model`로 변경
	docker compose -f docker-compose.yml -f compose.ollama.yaml exec ollama ollama pull $${MODEL:-qwen3:8b}

docker-down-ollama: ## overlay 스택(ocr-cpu + ollama) 정리
	docker compose -f docker-compose.yml -f compose.ollama.yaml down
