#!/usr/bin/env bash
# 빌드한 CPU 이미지 스모크 — 모델 다운로드 없이(textlayer) 이미지 자체를 검증한다.
# CI(docker-image 잡)와 release(publish-images, push 전)가 같은 검사를 쓴다.
#
#   scripts/smoke_image.sh IMAGE [EXPECT_VERSION]
#     PORT=18000        호스트 루프백 포트
#     OUT_DIR=…         smoke_e2e.sh 산출물 위치 (기본: 임시 디렉터리)
#
# 컨테이너는 docker-compose.yml(ocr-cpu)과 같은 하드닝으로 띄운다 — 읽기 전용 루트FS +
# tmpfs /tmp, cap_drop ALL, no-new-privileges, pids 상한, /data는 볼륨. 이미지가 그 아래서
# 깨지면(루트FS 쓰기 등) 배포 전에 여기서 걸린다.
set -euo pipefail

IMAGE="${1:?usage: smoke_image.sh IMAGE [EXPECT_VERSION]}"
EXPECT_VERSION="${2:-}"
PORT="${PORT:-18000}"
NAME="image-smoke-$$"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
mkdir -p "$OUT_DIR"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "── 컨테이너 로그 (실패)"
    docker logs "$NAME" 2>&1 | tail -80 || true
  fi
  docker rm -f -v "$NAME" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

echo "── 기동: $IMAGE (127.0.0.1:$PORT, compose와 같은 하드닝)"
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --mount type=volume,dst=/data \
  --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 1024 \
  -e OCR_ENGINE=textlayer -e PRELOAD_MODEL=1 \
  -e ALLOWED_HOSTS=localhost,127.0.0.1 \
  "$IMAGE" >/dev/null

ready=false
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" -o "$OUT_DIR/health.json"; then
    ready=true
    break
  fi
  sleep 2
done
[ "$ready" = true ] || { echo "health가 120초 안에 응답하지 않음"; exit 1; }

echo "── 이미지 계약 (엔진·네이티브 모듈·버전·실행 사용자)"
python3 - "$OUT_DIR/health.json" "http://127.0.0.1:$PORT" "$EXPECT_VERSION" <<'PY'
import json, sys, urllib.request
health_path, base, expect_version = sys.argv[1], sys.argv[2], sys.argv[3]
health = json.load(open(health_path))
assert health["engine"] == "textlayer" and health["worker_alive"], health
assert health["native_ops"], "이미지에 네이티브 모듈(uocr_native)이 없다"
schema = json.load(urllib.request.urlopen(f"{base}/openapi.json"))
if expect_version:
    assert schema["info"]["version"] == expect_version, (schema["info"]["version"], expect_version)
print(f"  engine={health['engine']} native_ops={health['native_ops']} "
      f"version={schema['info']['version']}")
PY
# 코드는 root 소유·실행 사용자는 비루트여야 한다(backend/Dockerfile 하드닝)
docker exec "$NAME" sh -c '
  [ "$(id -u)" != 0 ] || { echo "컨테이너가 root로 돈다"; exit 1; }
  [ "$(stat -c %U /srv/backend/app /srv/frontend | sort -u)" = root ] \
    || { echo "앱 코드가 root 소유가 아니다"; exit 1; }
  echo "  uid=$(id -u) · 코드 소유자 root"'

echo "── 업로드→OCR→markdown/zip (scripts/smoke_e2e.sh)"
OUT_DIR="$OUT_DIR" TIMEOUT_SECS="${TIMEOUT_SECS:-120}" \
  "$ROOT/scripts/smoke_e2e.sh" "http://127.0.0.1:$PORT"
echo "── 이미지 스모크 성공 ✔"
