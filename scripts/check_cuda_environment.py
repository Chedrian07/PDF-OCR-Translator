#!/usr/bin/env python3
"""RTX 5070 Ti CUDA 환경 preflight — sidecar 스택 기동 전 호환성 점검.

사용:
    python scripts/check_cuda_environment.py
    (torch 정보까지 보려면: cd backend && uv run python ../scripts/check_cuda_environment.py)

표준 라이브러리만 사용한다. torch는 있으면 추가 정보를 출력한다 (없어도 동작).
종료 코드: 0 = GPU 확인됨, 1 = GPU/드라이버 문제.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

OK = "\033[32m✓\033[0m"
WARN = "\033[33m⚠\033[0m"
FAIL = "\033[31m✗\033[0m"


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_nvidia_smi() -> dict | None:
    if not shutil.which("nvidia-smi"):
        print(f"{FAIL} nvidia-smi 없음 — NVIDIA 드라이버가 설치되지 않았습니다")
        return None
    q = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not q:
        print(f"{FAIL} nvidia-smi 질의 실패 — 드라이버/GPU 상태를 확인하세요")
        return None
    gpus = []
    for idx, line in enumerate(q.splitlines()):
        name, total_mb, used_mb, driver, cc = [p.strip() for p in line.split(",")]
        gpus.append({
            "index": idx, "gpu_name": name, "vram_total_mb": int(float(total_mb)),
            "vram_used_mb": int(float(used_mb)), "driver": driver, "compute_cap": cc,
        })
        print(f"{OK} GPU {idx}: {name}")
        print(f"   VRAM: {int(float(total_mb))}MB 총량 / {int(float(used_mb))}MB 사용 중")
        print(f"   드라이버: {driver} · compute capability: {cc}")
    if len(gpus) > 1:
        print(f"{WARN} GPU가 {len(gpus)}개 감지됨 — compose는 CUDA_VISIBLE_DEVICES="
              "${GPU_DEVICE:-0}로 서빙 카드를 선택합니다. compose가 "
              "CUDA_DEVICE_ORDER=PCI_BUS_ID를 설정하므로 인덱스는 위(nvidia-smi) "
              "순서와 일치합니다 — GPU_DEVICE가 의도한 카드인지 확인하세요")
    # 서빙 대상 = sm_120(Blackwell) 행 우선, 없으면 GPU 0 (JSON 스키마 후방 호환)
    target = next((g for g in gpus if g["compute_cap"].split(".")[0] == "12"), gpus[0])
    info = {k: v for k, v in target.items() if k != "index"}
    info["gpu_index"] = target["index"]
    info["gpus"] = gpus

    smi = _run(["nvidia-smi"]) or ""
    # 구형 헤더는 "CUDA Version: 12.x", 신형(예: 610.xx)은 "CUDA UMD Version: 13.x" —
    # 마지막 "Version:" 뒤 값을 취하면 두 형식 모두에서 CUDA 버전이 나온다.
    cuda_line = next(
        (line for line in smi.splitlines() if "CUDA" in line and "Version" in line), "")
    if cuda_line:
        cuda_ver = cuda_line.split("Version:")[-1].strip().rstrip("|").strip()
        info["driver_cuda"] = cuda_ver
        print(f"   드라이버 CUDA 지원: {cuda_ver}")
    return info


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        print(f"{WARN} torch 미설치 프로세스 — BF16/torch 런타임 점검 생략 "
              "(backend venv에서 실행하면 포함: cd backend && uv run python ../scripts/…)")
        return
    print(f"{OK} torch {torch.__version__} (CUDA runtime {torch.version.cuda})")
    if not torch.cuda.is_available():
        print(f"{FAIL} torch.cuda.is_available() == False — 컨테이너 밖이라면 정상일 수 있음")
        return
    cap = torch.cuda.get_device_capability(0)
    print(f"   torch가 본 GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"{OK if bf16 else FAIL} BF16 지원: {bf16}")
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"   현재 가용 VRAM: {free_b // 2**20}MB / {total_b // 2**20}MB")


def check_docker_gpu() -> None:
    if not shutil.which("docker"):
        print(f"{WARN} docker 없음 — compose 배포 대신 로컬 실행만 가능")
        return
    runtimes = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    if runtimes and "nvidia" in runtimes:
        print(f"{OK} docker nvidia runtime 등록됨")
    else:
        print(f"{WARN} docker runtime 목록에서 nvidia 미확인 — "
              "nvidia-container-toolkit 설치/설정 확인 필요 "
              "(gpus: all이 실패하면 sidecar가 CUDA를 못 씁니다)")
    ver = _run(["docker", "compose", "version", "--short"])
    if ver:
        print(f"{OK} docker compose {ver}")


def sidecar_compat_warnings(info: dict) -> None:
    print("\n── sidecar 호환성 ─────────────────────────────")
    cc = info.get("compute_cap", "")
    major = cc.split(".")[0] if cc else ""
    if major == "12":
        print(f"{OK} Blackwell(sm_120) 감지 — CUDA 12.8+ 스택 필요:")
        print("   · OvisOCR2 sidecar: vllm/vllm-openai:v0.22.1-cu129 (충족)")
        print("   · PaddleOCR-VL sidecar: paddlepaddle-gpu cu129 wheel (충족)")
        print("   · cu118/torch<2.7 기반 스택(DeepSeek-OCR-2 공식 경로 등)은 동작 불가")
        if info.get("gpu_index", 0) != 0:
            print(f"{WARN} 서빙 대상 GPU(sm_120)가 인덱스 {info['gpu_index']}입니다 — "
                  f".env에 GPU_DEVICE={info['gpu_index']} 를 설정하세요 (기본 0)")
    elif cc:
        print(f"{WARN} compute capability {cc} — 이 저장소의 sidecar 기본값은 "
              "RTX 5070 Ti(sm_120) 기준입니다. 다른 GPU는 문서의 VRAM 정책을 조정하세요")
    dc = info.get("driver_cuda", "")
    try:
        if dc and float(dc) < 12.9:
            print(f"{WARN} 드라이버 CUDA {dc} < 12.9 — PaddleOCR-VL Blackwell 가이드는 "
                  "CUDA 12.9+ 지원 드라이버를 요구합니다. 드라이버를 업데이트하세요")
    except ValueError:
        pass
    total = info.get("vram_total_mb", 0)
    used = info.get("vram_used_mb", 0)
    if total and total < 15_000:
        print(f"{WARN} VRAM {total}MB < 16GB — 기본값(OVIS_GPU_MEMORY_UTILIZATION=0.80 등)을 낮추세요")
    if used > 2_000:
        print(f"{WARN} 이미 {used}MB 사용 중 — 데스크톱/다른 프로세스가 VRAM을 점유하고 "
              "있습니다. sidecar OOM 시 먼저 이를 확인하세요")
    print(f"{WARN} 한 시점에 GPU 스택 하나만 기동하세요 (ocr-cuda | --profile ovis | --profile paddle)")


def main() -> int:
    print("── GPU / 드라이버 ─────────────────────────────")
    info = check_nvidia_smi()
    print("\n── torch ─────────────────────────────────────")
    check_torch()
    print("\n── docker ────────────────────────────────────")
    check_docker_gpu()
    if info:
        sidecar_compat_warnings(info)
        print("\n" + json.dumps(info, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
