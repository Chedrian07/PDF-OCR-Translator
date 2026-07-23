# Unlimited-OCR + Localight 병합 워크스페이스 — 루트 Makefile (uv·npm 기반)
#
# GPU 스택은 compose 프로필 함정 때문에 make 타깃으로 감싸지 않는다 —
# docker-compose.yml 헤더의 blessed 명령을 그대로 쓸 것 (서비스명 반드시 명시):
#   CUDA:          docker compose up -d --build ocr-cuda                               → :8001
#   OvisOCR2:      docker compose --profile ovis up -d --build ovisocr2 ocr-ovis       → :8002
#   PaddleOCR-VL:  docker compose --profile paddle up -d --build paddleocr-vl ocr-paddle → :8003
#   전환(정지):    docker compose stop ovisocr2 ocr-ovis
#                  (`--profile … down`은 프로필 없는 ocr-cpu까지 지우므로 전체 정리용으로만)

.PHONY: setup setup-metal setup-native dev dev-textlayer test e2e \
	docker-up docker-down docker-up-ollama docker-pull-model docker-down-ollama

setup:            ## backend 의존성 설치 (torch CPU)
	cd backend && uv sync --extra cpu

setup-metal:      ## macOS Apple Silicon용 (torch MPS)
	cd backend && uv sync --extra metal

setup-native:     ## C++ 가속 모듈 설치 (선택 — 없어도 순수 파이썬 폴백으로 동작)
	cd backend && uv pip install ../native

dev:              ## 로컬 개발 서버 — http://127.0.0.1:8000
	cd backend && uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

dev-textlayer:    ## 모델 다운로드 없이 textlayer 엔진으로 개발 서버
	cd backend && OCR_ENGINE=textlayer uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test:             ## CI와 동일한 3종 — backend pytest · ruff · frontend (node --test)
	cd backend && uv run pytest
	cd backend && uv run --only-group dev ruff check .
	npm test --prefix frontend

e2e:              ## 실서버 스모크 (기동된 백엔드 필요 — README §E2E)
	./scripts/smoke_e2e.sh

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
