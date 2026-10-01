# OCR 엔진 벤치마크 — 절차와 후보 조사

## 벤치마크 실행

도구: `scripts/benchmark_ocr_engines.py` (사용법·단일 GPU 순차 실행 절차는
[benchmark_docs/README.md](../benchmark_docs/README.md) 참조).

측정값: 엔진/모델/문서명/페이지 수/전체 시간/초당 페이지/페이지 평균 시간/
Markdown 문자 수/표·수식·figure 수/warning 수/실패 페이지 수/peak VRAM/출력 경로.
ground truth 제공 시에만 normalized edit distance·CER·구조 일치도·figure IoU 추가.
**GT 없이 정확도 점수를 만들지 않는다** — 구조 집계는 존재 확인용이다.

**시간 측정 방식 (2026-10 변경)** — 예전 벤치마크는 잡 상태를 **2초 간격**으로 폴링한
벽시계였다. 몇 초 안에 끝나는 짧은 문서는 2초 단위로 양자화돼 엔진 사이 속도비가 수 배까지
틀릴 수 있었고(아래 2026-07-20 표의 Ovis 1.1 s/page가 그 값이다), 스택을 막 띄운 직후의
첫 문서는 vLLM 컴파일 시간을 포함했다. 지금은:

- 측정 전에 health의 `model_loaded`를 기다리고(`--model-wait`, 기본 1800초),
  endpoint마다 첫 PDF로 `--warmup N`번(기본 1, 0=끔) 돌리고 기록하지 않는다 — 콜드
  스타트(모델 준비·컴파일)를 결과에서 뺀다.
- 0.25초 간격으로 폴링해 `running` 관측 → 완료 관측을 **처리 시간**(`process_s`)으로 잰다.
  `s/page`(`avg_page_s`)는 이 처리 시간을 페이지 수로 나눈 값이다.
- 새 열: `upload_s`(업로드) · `queue_s`(대기열) · `process_s`(처리) ·
  `timing_resolution_s`(폴링 간격 = 오차 상한). 업로드·대기를 포함한 벽시계는 `total_s`로
  따로 남는다. 서버 쪽 잡 시각은 meta·`GET /api/jobs/{id}`의 `started_at`·`finished_at`
  (UTC ISO, 초 단위)으로도 볼 수 있다.

### 실측 (2026-07-20, RTX 5070 Ti 16GB · WSL2 · driver 591.86)

`scripts/benchmark_ocr_engines.py`로 **동일 입력·동일 절차**로 순차 측정한 결과다
(스택을 하나씩 기동 — 단일 GPU). 입력은 저장소에서 재생성 가능한 2종:

```bash
cd backend
uv run python ../scripts/make_sample_pdf.py <dir>/en-mixed.pdf            # 영문 2p (표·차트·수식표기)
uv run python ../scripts/make_sample_pdf.py <dir>/ko-report.pdf --korean  # 한국어 1p (표·수식·figure·각주)
```

| engine | doc | pages | total(s) | s/page | md chars | tables | formulas | figures | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| ovis | en-mixed | 2 | 2.2 | **1.1** | 738 | 1 | 0 | 2 | 13,419MB |
| paddle | en-mixed | 2 | 6.8 | 3.4 | 733 | 1 | 0 | 2 | 8,893MB |
| ovis | ko-report | 1 | 2.1 | **2.1** | 579 | 1 | 2 | 1 | 13,419MB |
| paddle | ko-report | 1 | 12.1 | 12.1 | 587 | 1 | 2 | 1 | 10,079MB |

**⚠ 콜드 스타트를 시간에서 반드시 분리할 것**: OvisOCR2의 **첫 요청**은 vLLM
그래프 컴파일 때문에 2페이지에 40~43초가 걸린다(실측 42.8s). 위 표는 그 이후의
정상 상태 수치다 — 컨테이너를 새로 띄운 직후 한 번 측정하고 "20초/페이지"라고
적으면 20배 틀린 값이 된다(이 문서의 초판이 그 오류를 냈다). 지금은 `--warmup`이
이 첫 요청을 기록에서 뺀다.

**⚠ 위 표는 2초 폴링 시절 값이다**: 전체 시간이 2~12초인 1~2쪽 문서라 `total(s)`·
`s/page`의 해상도가 ±2초다. 특히 Ovis의 1.1 s/page는 폴링 간격보다 작은 값이라 믿을 수
없다 — 처리 속도 비교는 아래 25쪽 실문서 결과(Ovis 3.3 s/p, Paddle 17.6 s/p)를 쓰고,
GPU 호스트에서 새 측정 방식으로 다시 잰다.

### 한국어 텍스트 정확도 (같은 입력, 실측 출력 대조)

| 원문 | OvisOCR2 | PaddleOCR-VL-1.6 |
|---|---|---|
| 혼용된 | **훈련된** ✗ | 혼용된 ✓ |
| 한글 자모 ㄱㄴㄷ | **한국 자료 717** ✗ | 한글 자모 ㄱㄴㄷ ✓ |
| 硏究報告書 | 研究**报**告書 (간체 혼입) | 研究報告書 ✓ |
| 보존 | **보준** ✗ | 보존 ✓ |
| 표 셀 `1,234` | `1,234` ✓ | `1, 2 3 4` (숫자 사이 공백) |

→ **한국어 본문 정확도는 PaddleOCR-VL이 명확히 우수**하고, 표 셀의 숫자 붙임은
OvisOCR2가 정확하다. 두 엔진 모두 제목 계층·표 HTML·figure·각주 구조는 보존한다.
(숫자 공백은 이 합성 PDF의 CJK 폰트 자간 특성일 수 있어 실문서로 재확인 권장.)

### 실문서 검증 (2026-07-21, 저장소 내 실제 arxiv 논문 + 스캔 시뮬)

합성 샘플 외에 **실제 문서 유형**으로 확장 검증했다 (입력은 저장소에 있거나
재생성 가능):

| 문서 | 유형 | 엔진 | 결과 |
|---|---|---|---|
| `sample/2504.19874v1.pdf` (25p) | 영문 논문·**2단**·수식 밀집·긴 PDF | Ovis | done 82s(3.3s/p)·figure 27·실패 0 · **저자 블록 읽기순서·display/inline LaTeX·Lemma 구조 정확** |
| 〃 | 〃 | Paddle | done 441s(**17.6s/p**)·figure 14·실패 0 · 저자·초록·인라인 수식·2단 읽기순서 정확, 미지 라벨 경고 1(algorithm) |
| `sample/unlimited-ocr-paper.pdf` (14p) | 영문 논문·figure | Ovis / Paddle | done·실패 0 · figure 2~3 · 표 4 |
| `scan-ko.pdf` (`make_sample_pdf.py --scan`) | **스캔 시뮬**(텍스트 레이어 0·1.4° 기울임·노이즈·JPEG q55) | Ovis / Paddle | **제목·표·수식 구조는 정확**하게 OCR하지만 열화가 심한 CJK 줄은 오독(예: "ㄱㄴㄷ"→"그늘", "硏究報告書"→"컨혈갑"). 내장 텍스트 fallback이 아니라 진짜 OCR(텍스트 레이어 0) — 스캔 견고성의 한계를 보여준다 |

**다단(2-column) 읽기 순서**: OvisOCR2는 2504.19874의 2단 저자 블록·본문을 읽기
순서대로 평탄화하고 수식을 보존했다(발췌: `$$ D(p_X,B):=\inf\{…\} $$`,
`$I(x;y)=h(x)-h(x|y)$`). 실측 출력은 위 표의 판정 근거다.

**속도 격차(실문서에서 확대)**: 밀집 학술 텍스트에서 Ovis는 3.3s/p인데 Paddle은
블록별 layout+VL 비용으로 **~17s/p**까지 느려진다(14p 244s). 짧은/한국어 문서는
Paddle, 길고 밀집한 영문 문서는 Ovis가 유리하다는 앞의 결론이 실문서에서도 유지된다.

> ground truth가 없어 편집거리·CER은 계산하지 않았다(구조·읽기순서·수식 보존은
> 실측 출력 대조로 확인). 정량 정확도가 필요하면 정답 md를 만들어
> `--ground-truth`로 측정할 것.

## Apple Silicon 실측 — Unlimited-OCR (MLX · torch MPS)

**환경**: Apple M4 Max(16코어, 128 GB) · macOS · mlx 0.32.3 · torch 2.10.0 · bf16 ·
고정 스냅샷 `ee63731b`, 입력은 `sample/2504.19874v1.pdf`(25쪽 2단 논문)와 그 앞 16쪽,
200 dpi. 2026-10 개선 작업 중 측정했고 **상당수는 다른 작업이 GPU·CPU를 함께 쓰던
때의 값**이다(아래 '조건' 열). 정식 기준선은 Phase 4에서 순차로 다시 잰다 — 이 절과
README의 요약표를 함께 갱신할 것.

### MLX 엔진 (`OCR_DEVICE=auto`/`mlx`, Apple Silicon 기본)

| 측정 | 결과 | 조건 |
|---|---|---|
| 8쪽 청크 1개 (벤더 수준, 무제한 생성) | 30.4–31.6 s = **3.80–3.95 s/쪽**, 276–288 tok/s, 8,366토큰, 피크 8.27 GB | 다른 레인이 GPU 사용 중 |
| 25쪽 문서 (8/8/8/1쪽 청크) | **97.9 s = 3.92 s/쪽**, 청크 뒤 활성 메모리 6.67 GB로 복귀 | 〃 |
| 앱 파이프라인 16쪽 (청크 2개) | 청크 30.15 s·33.42 s = **3.97 s/쪽**, 290 tok/s, 청크당 TTFT 1.01 s, 잡 전체 71.2 s(렌더·충실도 게이트·병합 포함 **4.45 s/쪽**) | GPU 단독 |
| 앱 파이프라인 16쪽, 8비트(`OCR_MLX_QUANT_BITS=8`) | 청크 20.76 s·23.1 s = **2.74 s/쪽**, 430 tok/s, 잡 전체 49.7 s | GPU 단독 |
| 8쪽 청크, 8비트 (벤더 수준) | 21.5 s = 2.69 s/쪽, 413 tok/s, 파라미터 3.92 GB·피크 5.52 GB | 다른 레인이 GPU 사용 중 |
| 단독 1쪽 (gundam 12타일, 프롬프트 1,517토큰) | bf16 3.1 s (TTFT 0.65 s, 291 tok/s) · fp32 5.0 s | 〃 |
| per_page 잡 (그림 4쪽) | `run_single` 쪽당 2.8–4.9 s, 298 tok/s | GPU 단독 |
| 1쪽 멀티 (캡 384토큰) | TTFT 0.14 s, 304–308 tok/s (다른 레인과 경합 시 266–281) | — |
| 모델 준비 | 로드 1.4–1.6 s + 워밍업 0.45–0.48 s, 웜 2.2 s · 외장 볼륨 콜드 8.9 s | — |
| 파라미터 메모리 | bf16 6.67 GB · 8비트 3.92 GB | — |
| 생성 중 취소 | `cancel` 후 27 ms 안에 반환(부분 출력 저장 포함) | — |
| MAX_LENGTH 잘림 복구 | 4쪽 청크를 `MAX_LENGTH=4000`으로 잘라 앞 3쪽 유지 + 단독 1회, 15.3 s | — |
| 연속 잡 5회 (8쪽 실가중치, 한 프로세스) | RSS 로드 후 2,302 MB → 2,627/2,629/2,645/2,645/2,647 MB, phys_footprint 6,979 → 7,300–7,321 MB, MLX 활성 메모리 6,363 MB 고정, 피크 ≈7.9 GB, 잡당 28–35 s (첫 잡만 워밍업 +≈325 MB) | — |

- **품질**: 텍스트 레이어 재현율 0.9748로 torch MPS와 같다(8쪽 청크). 8비트도 0.9748,
  bf16 대비 문자 유사도 0.9995. 앱 16쪽의 충실도 평균은 bf16 0.908 · 8비트 0.906.
  4비트는 arXiv 번호 2504를 2304로 읽어 지원하지 않는다.
- **토큰 동일성**: MLX bf16은 torch MPS bf16과 토큰 단위로 같지 않다(1쪽 첫 분기 122번째
  토큰 — MPS의 fp16과 bf16끼리도 같은 지점에서 갈린다). 회귀 판정은 토큰 일치가 아니라
  유사도·재현율로 한다. fp32에서는 torch CPU와 로짓 상대오차 1e-5 수준으로 맞는다
  (`backend/app/vendor/unlimited_ocr_mlx/PROVENANCE.md`).

### PyMuPDF 프로세스 격리의 효과 (MLX bf16, 16쪽 = 8쪽 청크 2개)

번역 PDF 빌드가 같은 서버에서 동시에 돌 때 OCR 디코드 속도(같은 문서·같은 머신):

| 실행 | 디코드 tok/s (청크 1 / 청크 2) | OCR 시간 |
|---|---|---|
| 워커 프로세스, 빌드 없음 | 289.2 / 288.9 | 65.8 s |
| 워커 프로세스, OCR 내내 빌드 실행(56.7 s·51.3 s 두 건) | 291.0 / 290.8 | 64.7 s |
| 예전 방식(서버 프로세스 안, `PDF_WORKER_MODE=inline`), 같은 빌드 | 70.1 / 69.1 | 275.6 s |

inline 실행은 TTFT도 1.02 s → 2.60 s로 늘었다. MuPDF 호출이 GIL을 쥔 채 돌아 디코드
루프가 토큰마다 GIL을 기다린 것이다(ARCHITECTURE §18).

### torch MPS 폴백 (`make dev-metal`, `OCR_DEVICE=metal`)

벤더 P17(융합 MoE 기본)·P20(eager int 링 슬롯)·정적 ngram 창·ObjC 오토릴리스 풀 적용
전후, 같은 입력에서 출력 토큰 동일. 다른 레인이 머신을 쓰던 중(load average 3–6.6)이라
보수적인 값이다:

| 측정 | 적용 전 | 적용 후 |
|---|---|---|
| 1쪽 캡 384토큰 (웜) | 42.5–44.9 tok/s | **91.3–96.2 tok/s** |
| 1쪽 무제한 (801토큰) | 35.4 tok/s | 105.8 tok/s |
| 8쪽 청크 캡 320토큰 | 32.6–34.6 tok/s | 84.5–89.6 tok/s |
| 8쪽 청크 무제한 | 34.0 s/쪽 (272 s) | **9.8 s/쪽** (78.1 s, 112 tok/s) |
| 웜 384토큰 실행당 RSS 증가 | +9.5–12.9 MB | +0.0–3.5 MB |
| 첫 실행 RSS 증가 | +1,405 MB | +290 MB |
| 로드 직후 드라이버 메모리 · 첫 실행 피크 | 7,280 MB · 12.2 GB | 7,112 MB · 12.0 GB |

- 웜 실행 뒤 `torch.mps.driver_allocated_memory`에 보이는 +1.6 GB는 Metal 내부의 회수
  가능 캐시라 phys_footprint에 잡히지 않고 늘지 않는다.
- `OCR_FAST_DECODE=0`(HF generate 폴백)은 토큰 동일, 웜 61 tok/s.
- `OCR_SDPA=1`(P16 옵트인)은 M4 Max에서 15–28% 빠르지만 출력이 비트 동일하지 않다 —
  기본은 eager.
- 같은 8쪽 청크를 MLX는 3.8–3.95 s/쪽에 처리한다 — Apple Silicon 기본이 MLX인 이유다.

## 확정 엔진 요약

| | Unlimited-OCR (유지) | OvisOCR2 | PaddleOCR-VL-1.6 |
|---|---|---|---|
| 모델 | baidu/Unlimited-OCR 3.3B MoE | ATH-MaaS/OvisOCR2 0.9B | PaddlePaddle/PaddleOCR-VL-1.6 0.9B |
| 라이선스 | MIT | Apache-2.0 | Apache-2.0 |
| 실행 | in-process torch(CPU·CUDA·MPS) · in-process MLX(Apple Silicon) | vLLM 0.22.1 sidecar | paddle 3.3.1 sidecar |
| 강점 | 멀티페이지 문맥·토큰 스트리밍 | 페이지 정밀 파싱·figure bbox | 한국어·layout 블록·읽기 순서 |
| layout | full (그라운딩 토큰) | figure_only | full (블록+순서) |
| 16GB 적합성 | 검증됨 (~7GB) | 여유 큼 (util 0.80) | 여유 큼 |

## 후보 조사 (구현 제외 — 2026-07-20 공식 소스 기준)

| 후보 | 크기/라이선스 | CUDA·Blackwell | 16GB BF16 | 한국어 근거 | bbox/layout | 통합 비용 | 판정 |
|---|---|---|---|---|---|---|---|
| `zai-org/GLM-OCR` | 1.33B(safetensors)/MIT | 공식 언급 없음 | 여유 (~2.7GB) | HF 태그에 `ko` (본문 근거·벤치 없음) | 모델 자체 bbox 없음 — 별도 PP-DocLayoutV3 2단 파이프라인 | 높음 (vLLM nightly + transformers git HEAD 요구) | **제외** — OmniDocBench 1위(94.62)지만 bbox가 Paddle layout 의존이라 PaddleOCR-VL과 역할 중복, 스택이 nightly 의존 |
| `baidu/Qianfan-OCR` | 4.74B/Apache-2.0 | A100 벤치만, Blackwell 언급 없음 | **빠듯** (~9.5GB 가중치 + thinking 16K KV + 4K 비전) | 없음 ("192 languages"에 한국어 미명명) | Layout-as-Thought — 좌표 형식 미문서화 | 중간 (trust_remote_code) | **제외** — 16GB 헤드룸 부족, 공식 저비트 경로 없음 |
| `deepseek-ai/DeepSeek-OCR-2` | 3.39B/Apache-2.0 | **공식 스택이 CUDA 11.8 + torch 2.6 + 커스텀 vllm-0.8.5 wheel — sm_120 불가** | 여유 (~6.8GB, 돌릴 수 있다면) | 없음 | grounding 프롬프트 존재하나 출력 형식 미문서화 | 높음 (레거시 고정 스택) | **제외** — Blackwell 공식 경로 부재가 결정적 |
| `rednote-hilab/dots.mocr` | 3.04B/MIT | cu128/cu130 스택 권장 (사실상 Blackwell 가능, 공식 RTX50 언급은 없음) | 여유 (~6.1GB) | 없음 (showcase에 한국어 부재) | **가장 우수** — 페이지 JSON(11개 카테고리+bbox+content) | 중저 (vLLM 0.11+ 통합, trust_remote_code) | **제외** — olmOCR-bench 83.9로 매력적이나 한국어 근거가 전무해 이번 목적(한국어 문서) 대비 이점 불명확. 차기 후보 1순위 |

공통 결론: 네 후보 모두 "공식 Blackwell 지원 + 한국어 근거 + 16GB 여유"를 동시에
만족하지 못한다. OvisOCR2(페이지 파싱·figure)와 PaddleOCR-VL-1.6(한국어·layout)의
역할을 명백히 대체하는 후보가 없어 구현 범위에 추가하지 않았다.
"지원 완료"로 표기된 엔진은 fake/unlimited/textlayer/ovisocr2/paddleocr_vl 5종뿐이다
(unlimited는 torch 구현과 Apple Silicon용 MLX 구현 두 가지 — `OCR_DEVICE`로 고른다).
