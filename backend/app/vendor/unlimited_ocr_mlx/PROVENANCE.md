# 벤더링 출처 및 패치 내역 — Unlimited-OCR MLX 포팅

Apple Silicon in-process OCR 엔진용 baidu/Unlimited-OCR의 MLX 구현이다. 감사
스파이크(2026-10-01, `tmp/audit-2026-10-01` measure.mlx·MLX-01~07)가 mlx-vlm 0.7.4의
`unlimited_ocr`/`deepseekocr` 모델이 고정 HF 스냅샷을 변환 없이 읽고 torch MPS 대비
8.6배 빠르며(8쪽 청크 272 s → 31.7 s) 재현율이 같다(0.9748)는 것을 확인했다. mlx-vlm은
transformers>=5.14를 요구해 backend의 transformers==4.57.1 고정과 해소되지 않으므로(MLX-03)
패키지에 의존하지 않고 필요한 모델 코드만 벤더링했다.

- 업스트림: mlx-vlm **0.7.4** (PyPI 휠, MIT — `LICENSE`, Copyright © 2025 Prince Canuma)
- 모델 가중치: `baidu/Unlimited-OCR` revision `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead`
  (torch 경로와 같은 스냅샷·같은 safetensors를 변환 없이 strict 로드)
- 런타임 의존: `mlx==0.32.3`(backend `mlx` extra), numpy, Pillow, huggingface_hub,
  transformers 4.57.1(토크나이저만). **torch·mlx-vlm을 임포트하지 않는다**
  (tests/test_mlx_runtime.py가 torch·mlx_vlm 임포트를 막고 전 구간을 돌려 고정).
  단, transformers는 torch가 설치돼 있으면 `AutoTokenizer` 임포트 때 torch를 함께 올린다.
- 수치 레퍼런스: torch 벤더 `app/vendor/unlimited_ocr` (P1–P21 적용본, 기준 커밋 77224246 —
  후처리는 P22(det 모양 검사·`_is_image_ref` 포함)·P23까지 들어간 커밋 1958c50 판을 따른다).
  전처리·후처리·infer 흐름은 baidu/Unlimited-OCR `modeling_unlimitedocr.py`(MIT,
  Copyright (c) 2026 Baidu — 전문은 `../unlimited_ocr/LICENSE`)에서 직접 옮겼다.

패치 지점은 소스에서 `grep -n "local patch M" *.py`로 전수 확인할 수 있다.

모듈 이름 주의: 패키지가 내보내는 `generate`·`infer` 함수와 같은 이름의 하위 모듈을 두지
않는다(하위 모듈 임포트가 패키지 속성을 모듈로 덮어 함수 대신 모듈이 나온다) — 그래서
생성 루프는 `generation.py`, infer 흐름은 `inference.py`다(tests/test_mlx_runtime.py가 고정).

## 파일 매핑 (업스트림 원본 sha256)

| 로컬 파일 | 업스트림 원본 | sha256 |
|---|---|---|
| `config.py` | `mlx_vlm/models/deepseekocr/config.py` | `ef40ebd5fab0b3c2e5a7bad0aebe3a719ec177e9d578215ed3c8ccefa0da2f34` |
| | `mlx_vlm/models/unlimited_ocr/config.py` | `7823e1ab0da8ddb98a209cc668fb5020c749175feeb5c26c616f452403155fbc` |
| `sam.py` | `mlx_vlm/models/deepseekocr/sam.py` | `84325d5f831f6206e7a3249055c40b065a8e1293b410b8d94a91916d1f2ede97` |
| `vision.py` | `mlx_vlm/models/deepseekocr/vision.py` | `d131727f0f6884e9ceb25df34f6cd191ec8dc71c3c8a30e91e2264ce26d54528` |
| `language.py` | `mlx_vlm/models/deepseekocr/language.py` | `1c49d8dac0e02386a8fd86ef58276d4970b8cb95665069017a3fbbd7078a2b14` |
| | `mlx_vlm/models/unlimited_ocr/language.py` | `656e03580fa0370d9fcee6d41b8e5b175a482b219a7c30d52c7e8be1c6e70241` |
| | `mlx_vlm/models/mlp.py` (DeepseekMLP) | `e5413aa08a61ee26ab3bf70fac9aeef2ac9c13342e2a4507c2e7cc908a3273e7` |
| `switch_layers.py` | `mlx_vlm/models/switch_layers.py` | `ace00152d1d004e52292d8265af640d8dd69ffa67b8906886b53a99eb21951de` |
| | `mlx_vlm/models/activations.py` (swiglu) | `2ce04422f5670d8c3507f723d33253632e14121f45f2aa8070b08544ea92dfc9` |
| `cache.py` | `mlx_vlm/models/unlimited_ocr/language.py` (RingSlidingKVCache) | `656e03580fa0370d9fcee6d41b8e5b175a482b219a7c30d52c7e8be1c6e70241` |
| | `mlx_vlm/models/cache.py` (KVCache 슬라이스 갱신) | `b715008771817eb4fe05d8591e537bef30b827104950d65eea01a8abf0f2a3b0` |
| `model.py` | `mlx_vlm/models/deepseekocr/deepseekocr.py` | `e2ff4215a55d82a4e8c540837ff1e92bb751b36df2a2561495fa5c3adc112f54` |
| | `mlx_vlm/models/unlimited_ocr/unlimitedocr.py` | `2862f0c2f42c43876c5cc413bede25cb8cd4c3777b596fa00d8c485871b3ab64` |
| `generation.py` | `mlx_vlm/generate/ar.py` (generate_step) | `6f60faefc70bc7dec60e8fdec8f11fe9609eee4966f8240617cbcaf101046e19` |
| `loader.py` | `mlx_vlm/utils.py` (load_model·quantize 흐름) | `2045e18d1a16222d07b157251e4a5108b2b12bf740b885f9682a071c8dc59134` |
| `LICENSE` | `mlx_vlm-0.7.4.dist-info/licenses/LICENSE` | `f74c448746be3376e27dee43e2be9ec9d55f4820cd7466c603a6c7d1fd2d25b8` |
| `processing.py` | torch 벤더 `modeling_unlimitedocr.py` (infer/infer_multi 전처리) | `ed56bc28084da20c1306d92ce57ccc41bf78ae743ae4dc7d0e112f17a55f7b8f` |
| `postprocess.py` | 같은 파일 1958c50 판 (re_match·_is_image_ref·extract_coordinates_and_label·_clamp_box·draw_bounding_boxes·_split_multi_pages·save_results) | `4facb67a63233191810ad30995524b0035a3dab52eacbfb3570fab384f4130da` |
| `inference.py` | 같은 파일 (infer/infer_multi 흐름) + 로컬 `warmup()` | 〃 |
| `ngram.py` | 로컬 (스파이크 `mlx_bench.py` NoRepeatNgramGPU 정리, 원본 `145fb8ee…873a`) | — |
| `resample.py` | 로컬 (M7) | — |

업스트림 sha256은 설치된 휠 파일의 hex 다이제스트이며 휠 RECORD 해시와 일치함을 확인했다.
torch 벤더 sha256은 기준 커밋 77224246의 패치 적용본이다(`postprocess.py` 행만 P22·P23 적용본
1958c50 — `deepencoder.py`
`bae51b96…227c`, `modeling_deepseekv2.py` `f9807cb0…bd35`, 레퍼런스 ngram
`app/native_ops.py` `25506e25…b04a`).

뺀 것: mlx-vlm 프로세서(transformers 5 의존 — torch 레퍼런스를 직접 옮긴 `processing.py`로
대체), 배치·스펙큘러티브 생성, chunked prefill, KV 캐시 양자화, MLA·YaRN 분기(설정이 거부),
SwitchGLU 소배치 가속·오프로드 경로.

## 로컬 패치 목록

| # | 내용 | 이유 | 측정 |
|---|---|---|---|
| M1 | (`vision.py`) CLIP MLP 활성함수 `nn.GELU` → `quick_gelu`(`x*sigmoid(1.702x)`), 층 LayerNorm eps 1e-6 → 1e-5 | torch `deepencoder.py` NoTPFeedForward·`layernorm_epsilon=1e-5`와 다름(MLX-02). 미수정 시 CLIP 출력 상대오차 0.309, 1쪽 det 블록 19 → 11 | 실가중치 fp32 1쪽(teacher-forced 74토큰): SAM 4.1e-5, CLIP 3.9e-5, projector 3.2e-5, 로짓 4.08e-6(KL 1.8e-6, top-4 동일) |
| M2 | (`cache.py`, `generation.py`) 원샷 프리필 — 첫 update가 프롬프트 전체를 받아 `prefill_length = P`를 기록, 버퍼를 P+W로 선할당 | mlx-vlm 기본 chunked prefill은 마지막 프롬프트 토큰을 따로 넣어 P-1을 기록 → 그 토큰이 링에 들어갔다가 축출(8쪽 출력이 토큰 105에서 갈림, MLX-04). torch는 첫 q_len>1 forward 뒤 P를 기록 | 작은 무작위 모델(W=4, 프리필+12스텝, 링 3바퀴): torch 대비 로짓 상대오차 ≤2.5e-6. W를 5로 바꾸면 워밍업 뒤 갈라짐(음성 대조) |
| M3 | (`generation.py`, `ngram.py`) 그리디 생성 모듈 — EOS 포함 종료, 총 길이 `max_length`(+`hit_max_length`), 토큰 콜백 `on_token`, 토큰마다 `should_stop`, GPU no-repeat-ngram(n·창, 프롬프트+생성 기준), `mx.async_eval` 파이프라이닝(멈출 때 버리는 스텝 ≤1, 상한 스텝은 미예약) | mlx-vlm에는 ngram이 없고(반복 루프 방어 부재) 생성기는 잘림을 알리지 않는다. 엔진이 스트리밍·취소·반복 감지·OutputLimitError를 처리할 계약 필요 | ngram: 경계 18건 + 무작위 600건 + 생성 경로 증분 상태 20,557스텝이 레퍼런스와 일치. 작은 모델 그리디 40토큰(ngram 3/16 금지 발동 17스텝)이 torch fast_greedy_decode+앱 프로세서와 동일 |
| M4 | (`postprocess.py`, `inference.py`) torch-free 후처리 이식: `re_match`·`extract_coordinates_and_label`(**ast.literal_eval만**, P9)·`draw_bounding_boxes`(+P13 boxes.json)·`_dump_raw_pages`(P14)·save_results 흐름 + torch 벤더 **P22**(`_clamp_box` 규칙 그대로: 픽셀 좌표 `int(v/999*size)`를 x∈[0,W]·y∈[0,H]로 자르고, 퇴화(`x2<=x1 or y2<=y1`)·숫자 아님(bool 포함)·비유한·4개가 아닌 상자는 크롭·boxes.json·오버레이를 건너뜀. 건너뛴 image 상자와 좌표를 못 읽은 image ref도 번호 `img_idx`를 소비. det 값은 비어 있지 않은 list/tuple일 때만 상자 목록으로 읽고(문자열·빈 목록·0은 좌표를 못 읽은 ref로 번호 1개), 중첩되지 않은 목록은 상자 하나로 감싼다. image 판정은 마크다운 치환과 같은 `_is_image_ref`(공백 라벨 포함)) + **P23**(`_split_multi_pages`: 첫 `<PAGE>` 앞이 공백뿐일 때만 버림 — 마커 0개면 출력 전체가 1쪽) | 후처리가 torch를 임포트하는 모듈에 묶여 있었다(MLX-07). 경계 밖 좌표는 업스트림이 검은 패딩 크롭·경계 밖 bbox를 남겼다. P22의 번호 소비·P23은 torch 벤더에 나중에 들어가 이 포팅에서 빠져 있었다 — 좌표를 못 읽은 image det 뒤 그림이 한 칸 앞 번호 파일로 저장돼 마크다운 앞 자리에 붙었고, 모델이 선행 마커를 생략하면 1쪽이 사라지고 뒤 페이지가 한 칸씩 밀렸다(audit mlx-1·mlx-2·torch-1·torch-2) | 모델 없는 torch infer/infer_multi와 파일·마크다운 동일(tests/test_mlx_postprocess.py — 선행 마커 생략·마커 0개·빈 출력, 쉼표 누락·좌표 3/5개·inf·거대 숫자·bool·비리터럴·문자열·빈 목록·0·기형 평평 목록 image 좌표, `_clamp_box`·`_split_multi_pages`·`draw_bounding_boxes` 무작위 대조). 실가중치 8쪽: MLX infer_multi 산출물(마크다운·figure 크롭 바이트·boxes.json·raw_pages.json)이 같은 토큰을 넣은 torch 흐름과 바이트 동일 |
| M5 | (`loader.py`) 인메모리 8비트 양자화(affine, group 64) — 디코더만(embed_tokens·q/k/v/o·dense·shared·switch_mlp·lm_head). SAM·CLIP·projector·MoE 게이트 제외. 8 이외 비트 수는 거부 | 스파이크(MLX-06): 8비트 처리량 1.44배·품질 동등, 4비트는 arXiv 번호 2504 → 2304 오인식 | 8쪽 청크 21.5 s(2.69 s/쪽), 413 tok/s, 파라미터 3.92 GB·피크 5.52 GB, 텍스트층 재현율 0.9748(bf16과 같음), bf16 대비 문자 유사도 0.9995(MPS 대비 0.9992 — 스파이크 변환본 0.9970) |
| M6 | **기각(미적용)** MoE 게이트 로짓·softmax·가중합을 torch MPS 경로처럼 fp32로 | torch MPS/CPU는 게이트를 fp32 linear로, CUDA autocast는 bf16 linear로 계산한다 | bf16 8쪽: 스파이크 대비 0.9975, MPS 대비 텍스트 유사도 0.9972로 업스트림 수치(0.9993)보다 **멀어졌다**. 1쪽 384토큰은 스파이크와 135번째에서 갈림. 업스트림 수치를 유지한다 |
| M7 | (`resample.py`, `sam.py`, `vision.py`) 위치 임베딩 리샘플을 torch `F.interpolate`와 같은 가중치로: CLIP 16→10·SAM 64→40은 bicubic antialias, SAM 전역 블록 상대 위치표 127→79는 linear(반-픽셀 중심) | gundam 640 크롭에서만 쓰인다. mlx-vlm CLIP은 채널-우선 텐서에 `nn.Upsample`(채널-마지막 규약)을 걸어 (1,640,10,16) 모양 — 채널 축을 리샘플하고 reshape로 뒤섞는다. SAM rel_pos 보간은 `i*scale` 좌표라 torch 대비 상대오차 0.48. (SAM 절대 위치 커널은 torch와 2e-6로 일치했지만 같은 행렬 경로로 통일) | 가중치 행렬 vs torch: bicubic 최대 1.4e-6, linear 2.5e-7. 실가중치 fp32 gundam 1쪽 전체(12타일, P=1517) 로짓 상대오차 6.9e-5·top-5 동일 |

## 의도된 비-비트 동일 지점 (패치 아님)

- RoPE는 `mx.fast.rope`(fp32 내부)다. torch는 cos/sin을 fp32로 계산해 연산 dtype으로
  내린 뒤(bf16이면 bf16 cos/sin) 곱한다. 어텐션 softmax·LayerNorm 반올림·matmul 누적
  순서도 다르다 → fp32는 1e-5 수준 일치, bf16은 토큰 단위로 같지 않다(1쪽 첫 분기
  122번째 — torch MPS fp16과 bf16끼리도 같은 지점에서 갈린다).
- 이미지 분기: torch는 `sum(pixels) != 0`으로 크롭/전역 분기를 고른다. 여기서는 크롭 유무를
  명시적으로 넘긴다(합이 우연히 0이면 torch는 모양 불일치로 실패한다).
- 잘못된 입력: `<image>`가 정확히 1개가 아닌 프롬프트·열 수 없는 이미지는 ValueError
  (torch는 IndexError/AttributeError 또는 텍스트 중복).

## 실가중치 검증 (M4 Max 16코어·128 GB, mlx 0.32.3, 다른 레인이 GPU 동시 사용 중)

| 항목 | 결과 |
|---|---|
| 로드 | bf16 1.4–1.6 s(6.67 GB), fp32 1.3–1.5 s(13.3 GB) |
| 1쪽 멀티 cap 384 (bf16) | 토큰 ids가 스파이크 수정 bf16 실행과 384/384 동일, MPS bf16 레퍼런스와 첫 분기 122(스파이크와 같음). TTFT 0.14 s(콜드 0.61 s), 디코드 304–308 tok/s(경합 시 266–281), 피크 8.26 GB |
| 8쪽 청크 (bf16, 무제한) | 8366토큰, 출력이 스파이크 `p8_mlx_fix_bf16.md`와 **완전 동일**(문자 유사도 1.0000). 30.4–31.6 s = 3.80–3.95 s/쪽, 276–288 tok/s, 피크 8.27 GB, `<PAGE>` 8, det 100, 텍스트층 재현율 0.9748(MPS 0.9748) |
| 25쪽 문서 (8/8/8/1쪽 청크, bf16) | 97.9 s = 3.92 s/쪽(스파이크 98.6 s), 25,999토큰, 모든 청크 EOS·`<PAGE>` 수 정상, 청크 뒤 활성 메모리 6.67 GB로 복귀(누수 없음)·피크 8.27 GB, 텍스트가 스파이크 `doc25_bf16fix.md`와 같다(정규화 유사도 1.0) |
| 단일 gundam 1쪽 (12타일, P=1517) | bf16 3.1 s(TTFT 0.65 s, 291 tok/s)·fp32 5.0 s, 둘 다 675토큰. torch CPU fp32 앱 경로(19.6 s)와 처리된 마크다운·raw_pages·파일 집합이 **동일** |
| 워밍업 `warmup()` | 0.48 s(두 모드) 뒤 멀티 1쪽 TTFT 0.144 s·gundam 12타일 TTFT 0.62 s (워밍업 없이 첫 멀티 실행은 TTFT 0.61 s) |
| fp32 로짓 vs torch CPU fp32 | 멀티(+16토큰) 1.43e-5, gundam(2x1 타일) 7.2e-5, gundam 1쪽 전체 6.9e-5 — 모두 top-5 동일. gundam 1쪽 링 캐시 디코드 12스텝(프리필 포함 13개 로짓) 상대오차 1.0e-5–6.9e-5, 매 스텝 argmax 동일 |

opt-in 실가중치 테스트: `OCR_MLX_REAL_TESTS=1`(tests/test_mlx_model_parity.py — bf16 그리디
첫 64토큰 고정, torch CPU fp32 로짓 비교).

## 업스트림 갱신 시

새 mlx-vlm 버전의 위 파일들과 diff를 떠서 반영하고, M1·M2·M7이 여전히 필요한지(업스트림
수정 여부) 확인한 뒤 이 문서의 sha256·측정값을 갱신한다. 기준선: 작은 모델 패리티
(test_mlx_model_parity)·실가중치 opt-in 테스트·8쪽 청크 출력 비교.
