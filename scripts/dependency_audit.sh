#!/usr/bin/env bash
# 의존성 취약점 감사 — CI dependency-audit 잡(.github/workflows/ci.yml)과 같은 검사를
# 로컬에서 돌린다 (= make audit). 네트워크(PyPI·OSV 권고 DB)가 필요하다.
#
#   1) backend uv.lock — 배포 이미지와 같은 cpu extra 집합을 pip-audit --strict로
#   2) sidecar 잠금(services/*/requirements.lock)
#   3) paddlepaddle-gpu CDN 휠 — 같은 버전의 CPU 패키지 이름(paddlepaddle)으로 OSV 조회
#
# 수용한 권고(IGNORES)와 재검토 기한(DEADLINE)은 CI 잡과 **같은 목록**이어야 한다 —
# backend/tests/test_ci_ops_contracts.py가 두 곳의 일치를 강제한다. 바꿀 때는 ci.yml과
# .github/trivyignore.yaml(이미지 쪽 같은 기한·이유)도 함께 고친다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_AUDIT="pip-audit==2.10.1"
DEADLINE="2027-04-01"
IGNORES=(
  # torch 2.10.0 — 벤더 패치(PROVENANCE P11·P12·P16)가 2.10.0 동작에 맞춰 검증돼 고정
  PYSEC-2025-194   # CVE-2025-3000 jit.script 메모리 손상: 벤더 quick_gelu 정의만 script, 업로드 입력 미도달
  PYSEC-2026-139   # CVE-2026-4538 pt2 로딩 역직렬화: 가중치는 safetensors로만 읽는다(.pt/.pt2 없음)
  # transformers 4.57.1 — 벤더 모델 코드가 4.57.1 API에 묶여 고정. 리비전 SHA 고정·신뢰 저장소만
  PYSEC-2025-217   # CVE-2025-14929 X-CLIP 체크포인트 변환 스크립트: 쓰지 않는다
  PYSEC-2025-218   # CVE-2025-14930 GLM4 로딩 경로: GLM4를 로드하지 않는다
  PYSEC-2026-2288  # CVE-2026-1839 Trainer._load_rng_state의 torch.load: Trainer를 쓰지 않는다
  PYSEC-2026-2289  # CVE-2026-4372 config의 원격 attn 구현 로드: 리비전 SHA 고정 저장소만 로드
  PYSEC-2026-2290  # CVE-2026-5241 LightGlue trust_remote_code 우회: LightGlue를 로드하지 않는다
  PYSEC-2026-3929  # CVE-2026-9856 save_pretrained chat_template 경로 탈출: 저장하지 않는다
  CVE-2026-80047   # load_custom_generate 신뢰 확인 전 원격 파일 기록: custom_generate 미사용
)

command -v uv >/dev/null || { echo "uv가 필요합니다 — https://docs.astral.sh/uv/" >&2; exit 2; }

[[ "$(date -u +%F)" < "$DEADLINE" ]] \
  || { echo "수용 권고 재검토 기한($DEADLINE)이 지났다 — 다시 판단하고 ci.yml과 함께 날짜를 옮긴다" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

args=()
for id in "${IGNORES[@]}"; do args+=(--ignore-vuln "$id"); done

echo "== backend (uv.lock · cpu extra)"
(cd "$REPO/backend" && uv export --quiet --frozen --no-hashes --extra cpu --no-emit-project \
  -o "$tmp/req.raw.txt")  # --quiet: -o를 줘도 stdout에 전체 목록을 다시 찍는다
# torch CPU 휠의 로컬 버전(+cpu)은 PyPI에 없어 '감사 불가'로 조용히 빠진다 — 로컬 표기를 떼어
# 같은 릴리스로 감사하고, --strict로 그런 누락 자체를 실패시킨다. (sed -i는 BSD/GNU가 달라 쓰지 않는다)
sed -E 's/^([A-Za-z0-9._-]+==[^ ;+]+)\+[A-Za-z0-9._]+/\1/' "$tmp/req.raw.txt" > "$tmp/req.txt"
uvx "$PIP_AUDIT" -r "$tmp/req.txt" --disable-pip --no-deps --strict "${args[@]}"

for svc in ovisocr2 paddleocr_vl; do
  echo "== services/$svc"
  uvx "$PIP_AUDIT" -r "$REPO/services/$svc/requirements.lock" --disable-pip
done

# paddlepaddle-gpu는 PyPI에 없는 CDN 휠 URL이라 위에서 '감사 불가'로 빠진다. 권고는 같은
# 코드의 CPU 패키지 이름(paddlepaddle)으로 등록되므로 같은 버전을 OSV로 본다.
ver="$(sed -nE 's#.*/paddlepaddle_gpu-([0-9.]+)-.*#\1#p' "$REPO/services/paddleocr_vl/requirements.lock" | head -1)"
[[ -n "$ver" ]] || { echo "requirements.lock에서 paddlepaddle_gpu 버전을 찾지 못했다" >&2; exit 1; }
printf 'paddlepaddle==%s\n' "$ver" > "$tmp/paddle.txt"
echo "== paddlepaddle==$ver (paddlepaddle-gpu, OSV)"
uvx "$PIP_AUDIT" -r "$tmp/paddle.txt" --disable-pip --no-deps -s osv
