# Unlimited-OCR-2 전환 검증 기록

검증일: 2026-10-30. 변경 전 기준 커밋: `cc97954`.

**구현은 전환했지만 전체 합격 조건은 충족하지 못했다.** 수식 조판·일부 품질 지표가 퇴행했고 실제 번역 공급자는 인증에 실패했다. Apple Silicon 실기기 검증은 사용자가 맡기로 했다. 이 기록은 완료·릴리스 승인 선언이 아니다.

## 성능

RTX 5070 Ti 16GB, WSL2, 같은 PDF·래스터 해시, 200 DPI, 1024 전역 뷰, BF16, 동시성 1, 청크 8쪽. 준비 실행 1회 후 3회 중앙값. v1은 torch 2.10/cu129 자체 최적화 경로, v2는 공식 SGLang wheel의 torch 2.13/cu130 및 full prefill/decode CUDA graphs를 사용했다. 양자화·해상도 축소는 적용하지 않았다.

| 문서 | v1 OCR 초 | v2 OCR 초 | 시간 단축 | v1 → v2 쪽/초 | v1 → v2 최대 VRAM MiB |
|---|---:|---:|---:|---|---|
| en-mixed | 3.500 | 2.408 | 31.2% | 0.571 → 0.830 | 8455 → 11435 |
| ko-report | 3.859 | 2.600 | 32.6% | 0.259 → 0.385 | 8473 → 11435 |
| scan | 4.167 | 2.659 | 36.2% | 0.240 → 0.376 | 8511 → 11473 |
| columns | 6.902 | 2.904 | 57.9% | 0.145 → 0.344 | 8491 → 11473 |
| long-32 | 40.467 | 28.281 | 30.1% | 0.791 → 1.131 | 8791 → 12293 |

VRAM은 100ms 간격 GPU 전체 사용량 표본이며 데스크톱 사용량을 포함한다. 첫 텍스트 델타까지의 시간 중앙값은 다음과 같다. 네트워크·준비 확인·디코딩 버퍼를 포함하므로 모델 내부 첫 토큰 시간과 구분해야 한다.

| 문서 | v1 첫 텍스트 초 | v2 첫 텍스트 초 |
|---|---:|---:|
| en-mixed | 0.381 | 1.219 |
| ko-report | 0.289 | 1.134 |
| scan | 0.370 | 1.171 |
| columns | 0.317 | 1.148 |
| long-32 | 0.917 | 1.881 |

## 품질

아래 CER은 공백·구두점·일부 LaTeX wrapper를 정규화한 진단용 문자 오류율이다. 표는 행·열 구조를 비교했다. 수식 표식 개수는 의미 정확도 평가를 대신하지 않는다. 전체 품질 비퇴행은 **불합격/미충족**이다.

| 문서 | v1 → v2 CER | 그림 수 v1 → v2 / 정답 | 수식 표식 v1 → v2 | 그림 평균 IoU v1 → v2 |
|---|---|---|---|---|
| en-mixed | 0.00% → 0.00% | 2 → 2 / 2 | 1 → 0 | 0.9805 → 0.9737 |
| ko-report | 6.67% → 6.67% | 1 → 1 / 1 | 1 → 1 | 0.9632 → 0.9561 |
| scan | 6.67% → 7.11% | 1 → 1 / 1 | 1 → 1 | 미주석/해당 없음 |
| columns | 0.00% → 0.00% | 0 → 0 / 0 | 0 → 0 | 미주석/해당 없음 |
| long-32 | 5.36% → 0.00% | 22 → 32 / 32 | 15 → 0 | 0.6735 → 0.9752 |

- 모든 후보 실행에서 페이지 마커와 그림 수가 입력/정답 수와 일치했다. 장문 v1 결과의 그림 10개 누락은 v2에서 재현되지 않았다.
- 다단 문서는 두 모델 모두 읽기 순서 표식 16/16개, 표식 쌍 순서 정확도 1.0이다. 비교된 표의 행·열 구조는 유지됐다.
- 영문 수식 `E = mc^2`는 문자는 보존됐지만 v2가 수식 구분자를 내지 않아 KaTeX로 조판되지 않았다. 스캔 CER과 일부 그림 IoU도 악화됐다.
- 영어 두 페이지의 실제 뷰어를 시각 검수했다. 표·그림·페이지 위치 연결은 보존됐으며 수식이 일반 텍스트로 남는 차이를 확인했다. 전체 수식 의미·전체 레이아웃 벤치마크는 미완료다.

## 플랫폼 및 회귀

- CPU: 호스트와 빌드한 CPU Docker 이미지에서 실제 고정 가중치로 `Hello OCR 2` 이미지 추론 성공. Torch FP32, CPU 스레드 4.
- CUDA: RTX 5070 Ti에서 5종 실제 문서 비교 측정 성공. 실제 스트림 첫 출력 후 취소까지 약 34ms. 작은 페이지 토큰 상한은 `MAX_PAGE_OUTPUT_TOKENS` 오류로 감지됐고 다음 정상 문서 추론이 성공했다.
- MLX: Linux MLX CPU 런타임에서 torch SR-SWA와 실제 배열 연산을 대조했다. prefill·페이지 경계 decode·문서 간 상태 격리 테스트 통과. 별도 측정 최대 절대오차 약 1.8e-6. 실제 체크포인트의 활성 파라미터 691개 strict 로딩 확인. Apple GPU 실추론은 사용자 검증 항목이다.
- Metal: 새 공식 torch 경로로 이식했고 CUDA 하드코딩 이미지 주입을 제거했다. Apple GPU에서 실행하지 않았다.
- 프론트엔드 단위 테스트: 381개 통과. 모델 없는 전체 브라우저 E2E 통과.
- 실모델 OCR + 모의 번역/Q&A 전체 브라우저 E2E: 끝까지 실행, 수식 조판 관련 2개 실패. 업로드·스트리밍·표·그림·뷰어·HTML/PDF 다운로드·번역 UI·일시 오류 복구 시나리오는 실행됐다. 모의 번역은 실제 공급자 검증으로 세지 않았다.
- 실제 기존 OpenRouter 번역 설정: 인증 오류. 성공한 실제 번역 E2E는 미검증이다. 키 값은 기록하지 않았다.
- 백엔드: 2015 통과, 34 건너뜀, 1 실패. 실패는 `test_single_line_dry_run_reserves_actual_cjk_span_bounds[shallow]`이며, 동일 환경에서 변경 전 커밋에도 재현됐다. 이번 전환으로 해당 PDF 조판 코드는 변경하지 않았다.
- Ruff 및 Compose 구성 검사 통과. CPU 및 SGLang 이미지 빌드 성공. CPU 이미지의 권한 제한 하네스와 실제 CPU 추론 성공. 수정한 SGLang 이미지도 WSL 장치/드라이버 명시 연결로 RTX 5060 Ti에서 실제 추론에 성공했다.

## 환경 진단

`nvidia-smi` 기본 실행은 SIGSEGV(139). gdb에서 `/usr/lib/wsl/lib/libnvidia-ml.so.1`의 ELF relocation 중 충돌을 확인했다. 설치된 Windows NVIDIA 드라이버 디렉터리를 프로세스 `LD_LIBRARY_PATH` 앞에 두면 host CUDA가 동작했다. 시스템 라이브러리는 수정하지 않았다.

표준 Docker NVIDIA prestart hook도 동일 NVML 로딩 충돌을 보였다. 검증 컨테이너에만 `/dev/dxg`와 WSL 드라이버 디렉터리를 읽기 전용으로 연결하고 `runc`를 사용했을 때 torch CUDA 연산과 SGLang 실제 모델 추론이 성공했다. Compose의 표준 `gpus` 경로는 이 호스트에서 아직 통과하지 못했다.

## 재현 및 증거

- [고정 평가셋·모든 3회 측정값·출력·품질 지표](../benchmark_docs/unlimited2/README.md)
- [실제 뷰어 1쪽](../benchmark_docs/unlimited2/validation/viewer-page1.png), [2쪽](../benchmark_docs/unlimited2/validation/viewer-page2.png)
- [설치·런타임 설정·사용자 Apple Silicon 검증 명령](UNLIMITED_OCR_2.md)

```bash
DISABLE_DOTENV=1 backend/.venv/bin/pytest backend/tests --disable-warnings --tb=short
npm --prefix frontend test
npm --prefix frontend run test:e2e-mock
# 실제 CUDA OCR, 모의 번역/Q&A:
E2E_OCR_ENGINE=unlimited2 E2E_OCR_DEVICE=cuda npm --prefix frontend run test:e2e-mock
# 실제 CUDA 측정:
PYTHONPATH=backend backend/.venv/bin/python scripts/benchmark_unlimited2.py \
  --pdf benchmark_docs/unlimited2/en-mixed.pdf --out /tmp/uocr2-measurements
```
