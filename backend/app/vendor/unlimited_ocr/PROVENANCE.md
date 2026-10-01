# 벤더링 출처 및 패치 내역

- 출처: https://huggingface.co/baidu/Unlimited-OCR
- Revision: `ee63731b6461c8afcdcc7b15352e7d2ffecc2ead` (2026-07-03)
- 가져온 날짜: 2026-07-06
- 라이선스: MIT (동봉된 LICENSE)
- 파일: `modeling_unlimitedocr.py`, `modeling_deepseekv2.py`,
  `configuration_deepseek_v2.py`, `deepencoder.py`, `conversation.py`

업스트림 코드는 CUDA 전용(`.cuda()`/`torch.autocast("cuda")` 하드코딩)이라
CPU 백엔드 지원을 위해 벤더링 후 아래 패치를 적용했다.
수정 대상 파일: **P1–P15는 `modeling_unlimitedocr.py`**, **P16–P20은
`modeling_deepseekv2.py`**, **P21은 `modeling_unlimitedocr.py` + `deepencoder.py`**,
**P22–P23은 `modeling_unlimitedocr.py`**.
`configuration_deepseek_v2.py`·`conversation.py`는 원본 그대로다.
(패치 지점은 소스에서 `grep -n "vendor patch P" *.py`로 전수 확인할 수 있다 —
업스트림 갱신 시 이 목록이 아니라 grep 결과를 진리원으로 삼을 것.)

## 패치 목록

| # | 내용 | 이유 |
|---|---|---|
| P1 | `_autocast_ctx(device, dtype)` 헬퍼 추가 | `torch.autocast("cuda", bf16)` 하드코딩을 디바이스/디타입 인식으로 대체 (fp32면 no-op) |
| P2 | `UnlimitedOCRModel.forward`의 `masked_scatter_` 마스크 `.cuda()` → `.to(inputs_embeds.device)` | CPU/기타 디바이스 동작 |
| P3 | `infer()`/`infer_multi()`에 `streamer=None`, `stopping_criteria=None`, `logits_processor=None` 파라미터 추가 (기본값이면 업스트림과 동일 동작) | SSE 토큰 스트리밍, 잡 취소, C++ 가속 ngram 프로세서 주입 |
| P4 | `infer()`/`infer_multi()` 내부의 모든 `.cuda()` → `.to(모델 파라미터 디바이스)`, autocast → P1 헬퍼 | CPU/CUDA 공용 |
| P5 | 이미지 텐서 `.to(torch.bfloat16)` 하드코딩 → 모델 파라미터 dtype 추종 | fp32(CPU 기본) 로딩 시 dtype 불일치 방지 |
| P6 | 디코드 시 `input_ids.unsqueeze(0).cuda().shape[1]` → `input_ids.shape[0]` | 불필요한 디바이스 전송 제거 (동일 값) |
| P7 | `infer()`의 `save_results` 분기 끝에 `return outputs` 추가 | 업스트림은 파일만 쓰고 None 반환 — 호출자가 처리된 마크다운을 받도록 (다른 분기 동작 불변) |
| P8 | `eval()` 기반 geo 플로팅 분기 비활성화 (`if False and ...`) | **보안**: 모델 출력(=문서 내용에 의해 조작 가능)을 `eval()`에 전달 — PDF 내용을 통한 코드 실행 벡터. 본 앱은 geo 모드 미사용 |
| P9 | det 좌표 파싱 `eval()` → `ast.literal_eval()` | **보안**: 동일 벡터. 리터럴 좌표만 허용, 이상 입력은 기존 try/except로 스킵 |
| P10 | `UnlimitedOCRForCausalLM`에 `GenerationMixin` 명시 상속 추가 | transformers 4.50+는 커스텀 모델에 `generate()`를 자동 제공하지 않음. 업스트림은 trust_remote_code 로딩 시 transformers의 하위호환 심이 자동 부착하지만, 벤더링 직접 로드는 심이 없어 명시 상속 필요 |
| P11 | 이미지 임베딩 주입의 `masked_scatter_`(브로드캐스트 (L,1) 마스크) → bool 인덱싱 대입 (`inputs_embeds[idx][mask] = …`) | **MPS 정합성**: torch 2.10.0 MPS에서 브로드캐스트 마스크 `masked_scatter_`가 조용히 오동작(뷰/expand 무관, 소스 첫 원소만 기록)해 이미지 임베딩이 주입되지 않음 → 모델이 즉시 EOS 출력. CPU/CUDA는 결과 동일 (P2의 마스크 디바이스 수정 의도 포함) |
| P12 | `_autocast_ctx`가 `mps`에서는 항상 nullcontext 반환 | **MPS 정합성**: torch 2.10.0의 `torch.autocast("mps", bf16)`가 로짓을 오염시켜 생성이 반복 루프(`"), ), "` 무한 반복)로 퇴화. 가중치를 bf16으로 로딩하므로 autocast 없이도 bf16 연산 — 성능 손실 없음. CPU/CUDA 경로 불변 |
| P13 | `draw_bounding_boxes`가 figure 크롭 bbox(픽셀)+페이지 크기를 `{output_path}/boxes.json`으로 export | 업스트림은 크롭 후 좌표를 버림 — 렌더 레이어가 원본 페이지 대비 상대 폭으로 figure를 표시하는 데 필요 (마크다운 출력 계약 불변, 파일 추가만) |
| P14 | `infer`/`infer_multi`의 save_results가 치환 전 페이지 원문을 `{output_path}/raw_pages.json`으로 export | 좌표 기반 레이아웃 뷰(전 블록 type/bbox/텍스트)가 필요 — 구조 파싱은 앱 코드(pipeline/layout.py)에서 수행해 벤더 변경을 dump 한 줄로 최소화 (마크다운 계약 불변) |
| P15 | `infer`/`infer_multi`에 `generate_fn=None` 파라미터 추가 — 있으면 `self.generate` 대신 호출 | 커스텀 디코드 루프(app/engine/fast_decode.py — 호스트 동기화 블록 배칭, cpu/cuda/mps 공용) 주입용. 기본값 None이면 업스트림과 동일 동작. HF generate의 토큰당 동기화·기계장치가 측정된 디코드 병목이었음 |
| P16 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention`의 prefill·디코드 어텐션을 SDPA 융합 커널로 — **CUDA 전용 기본**, `OCR_SDPA=1/0`으로 전 디바이스 강제 on/off. 정책은 모듈 함수 `_sdpa_enabled(device_type, env_value)`(forward가 호출당 env 1회 조회) | eager(matmul→fp32 softmax→matmul)의 어텐션 행렬 물질화·다중 디스패치 제거 (CUDA 골든 파리티 검증). **MPS 실측(M1 Max, torch 2.10.0): SDPA fused 커널이 로짓을 오염시켜 디코드가 반복 붕괴**(P12 autocast 오염과 같은 계열) → MPS/CPU는 원본 eager(fp32 softmax) 유지. **M4 Max 재측정(torch 2.10.0, 2026-10 감사)**: 붕괴 미재현(1쪽 801토큰·2쪽 청크 1,680토큰 EOS 정상 종료, 반복 0건), 디코드 1쪽 +15%·8쪽 +17%·융합 MoE 위 +28%, 그러나 출력이 비트 동일하지 않음(bbox ±1~2, 2쪽은 74번째 토큰부터 분할 방식 상이, 8쪽은 2번째 토큰부터 분기) → 칩별 결과가 갈려 **기본값 불변(MPS eager)**, `OCR_SDPA=1`은 옵트인 성능 옵션 |
| P17 | (`modeling_deepseekv2.py`) `DeepseekV2MoE`에 융합 디코드 경로(`_moe_infer_fused`) 추가 — 전 expert의 gate/up/down weight를 `[E, out, in]` 텐서로 `torch.stack`하고 각 expert Linear의 `.weight`를 그 스택의 뷰로 재지정(개별 텐서 참조 해제 → 메모리 중복 없음, 기존 `moe_infer` 루프도 뷰로 계속 동작). 디코드에선 `index_select`+`bmm` 3방으로 라우팅·MLP를 융합. **스택은 로드 시점 1회**: 엔진이 cuda/mps에서 `.to(device)` **전에** 모듈 함수 `prebuild_fused_moe(model, device)`를 호출해 CPU에서 스택 → 디바이스로 1회 전송 → 뷰 재지정(이후 `.to()`는 이미 디바이스에 있는 뷰를 그대로 둔다). 첫 호출 지연 스택(`_build_fused_experts()`)은 프리빌드가 없을 때의 폴백. **CUDA·MPS·디코드(`N==1`)·`ep_size==1` 한정 기본 on**(N>1 프리필 조각은 mm↔bmm 누적차로 근접 argmax 플립 실측 → 제외), `OCR_MOE_FUSED=0/false/no/off`로 킬스위치(프리빌드도 생략 → 완전 legacy 복원, env는 인스턴스별 1회 조회·캐시). 수치: **N=1 bitwise 동일은 CPU에서만 성립**(test_moe_fused.py로 고정). CUDA는 가중치 재스택만으로 cuBLAS 커널 선택이 바뀌어 **legacy 바이트 파리티 불가**(동일 커널 mm 변형 실측) → 채택 게이트: 동일 프로세스 내 결정성(같은 컨테이너 2회 변환 바이트 동일 실측 — 프로세스 재시작 간에는 cuBLASLt 알고리즘 휴리스틱으로 공백/동점 토큰 수준 변동이 legacy 포함 원래 존재)·구조 지표 등가(표/셀/이미지/수식 수 재빌드 간 동일 실측)·킬스위치 완전 복원. MPS는 P18 대비 **토큰 동일 실측**(M4 Max: 1쪽 384·801토큰, 8쪽 청크 320토큰, 감사 실측 8,372토큰)이나 비트 동일을 보장하지는 않는다 | 디코드 병목이 `moe_infer`의 레이어당 GPU→CPU 동기화(`tokens_per_expert.cpu().numpy()`, P18 경로는 `topk_ids[0].tolist()`) + expert별 파이썬 루프의 소형 커널 난사(토큰 1개에 6 expert × 3 linear). 융합 경로는 **CPU 동기화 0회**. 실측 배경: RTX 5070 Ti batch-1 디코드 ~13 tok/s·sm 21%. **M4 Max MPS(2026-10)**: P18이 CPU 시간의 69%를 동기화 대기에 썼고, 융합 경로로 1쪽 캡 384토큰 50.6→89.4 tok/s(1.77x, 감사 실측), P20 int 슬롯과 함께 1쪽 ~96 tok/s·8쪽 청크 캡 320토큰 33~35→91~93 tok/s(재측정), 전부 토큰 동일. 메모리: 첫 디코드 지연 스택(감사 실측)은 MPS 힙 단편화로 empty_cache 뒤 드라이버 메모리 +1.7~3.3GB·첫 실행 피크 15~16GB였다 → 이동 전 프리빌드로 로드 직후 `driver_allocated_memory` 7,280→7,112MB, 첫 실행 피크 12.2→12.0GB, 상주(dirty) 그래픽 메모리 7,490→7,346MB(footprint 도구). 웜 실행 뒤 `driver_allocated_memory`에 보이는 +1.6GB는 Metal 내부 **회수 가능(reclaimable)** 캐시라 phys_footprint에 잡히지 않는다(9,169 vs P18 9,170MB). dtype 캐스트/최종 가중합 순서를 upstream과 동일하게 맞춤(N>1은 mm↔bmm 누적 순서차 fp32 round-off ~2e-7뿐). 발동 조건 밖·`ep_size>1`은 기존 `moe_infer` 그대로 |
| P18 | (`modeling_deepseekv2.py`) `DeepseekV2MoE.moe_infer`에 단일 토큰(seq==1)·`ep_size==1` 디코드 패스트패스 추가 — **MPS 전용 기본**, `OCR_MOE_FAST=1/0`으로 전 디바이스 강제 on/off (P16 게이트 패턴). P17이 발동하지 않을 때 `moe_infer` 안에서 동작 — MPS 디코드는 이제 P17이 먼저 받으므로 P18은 `OCR_MOE_FUSED=0`일 때의 MPS 폴백 | 배치=1 디코드에서 argsort/scatter/cnts 라우팅 기계장치를 제거하고 topk 전문가만 직접 실행. **레이어당 호스트 동기화는 제거되지 않는다**: `tokens_per_expert.cpu()`가 더 작은 `topk_ids[0].tolist()`로 바뀌었을 뿐 MoE 레이어마다 1회(토큰당 11회) 남는다(감사 프로파일: 동기화 대기가 CPU 시간의 69%). P17과 달리 **가중치 재스택 없이** 기존 expert 뷰 루프만 사용 → 전문가 연산과 최종 가중합(`view→type→mul_→sum→type`)의 연산 순서가 원본과 동일해 **결과 비트 동일**(M4 Max 골든 실측: 2p·25p result.md sha256 동일, 합성 벤치 `torch.equal`=True). 이득: 구 기록 '2p 2.0x·25p 1.86x'는 도입 당시 스택 기준이고, **현재 스택(fast_decode on, M4 Max, 2026-10 감사) 실측은 1.20x**(1쪽 캡 384토큰 42.0→50.6 tok/s). 원본(비패스트패스) 경로는 불변 |
| P19 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention`에 rotary cos/sin 스텝 캐시(`_rope_cached`) 추가 — layer 0가 계산해 캐시 객체(past_kv)에 스태시, 레이어 1+는 **동일 텐서를 재사용** | rotary 출력은 position_ids·dtype에만 의존해 한 스텝의 전 레이어가 같은 값을 중복 계산 → 스텝당 rotary 계산을 레이어 수회에서 1회로 축소. **동일 텐서 재사용이라 결과 비트 동일**. layer 0는 키 일치와 무관하게 항상 재계산·갱신하므로 스텝 간 `id()` 재사용 충돌에도 안전(레이어 실행 순서 항상 0→N). 게이트 없이 전 디바이스 적용 |

| P20 | (`modeling_deepseekv2.py`) `SlidingWindowLlamaAttention.forward`의 **디코드 정상상태(링) 분기** 슬롯 상태를 **모드별 단일 표현**으로 관리 — 기본(eager: CPU/MPS 전 구간·CUDA eager)은 파이썬 int(`past_kv._ring_pos` dict) + 슬라이스 `copy_`, **CUDA Graph 모드에서만** 디바이스 상주 0-dim int64 텐서(`past_kv._ring_pos_t` dict) + `index_copy_`·`add_(1).remainder_(W)`. 전환은 모듈 함수 `ring_slots_to_tensor`(fast_decode 그래프 경로가 캡처 직전 1회)·`ring_slots_to_int`(그래프 실패 폴백의 eager 재개 직전)로만 하고 `_ring_tensor_mode` 플래그로 한 시점에 한 표현만 둔다(이중 상태 금지). `_prefill_length`는 캡처 시점 고정 상수라 int dict 유지 | app/engine/fast_decode.py의 **CUDA Graph 디코드 캡처(U2)**는 링 슬롯 인덱싱·갱신이 캡처 안에서 재생 가능한 텐서 연산이어야 함(파이썬 int 갱신은 캡처에 기록되지 않아 리플레이 시 같은 슬롯만 덮어씀). 두 경로의 저장 값은 동일(같은 슬롯에 같은 K/V) → **출력 불변**(int↔텐서 전환을 섞은 실행이 스텝별 비트 동일 — 테스트로 고정). **MPS 영향(2026-10, M4 Max·torch 2.10)**: 최초 P20은 텐서 슬롯을 전 백엔드에 적용했는데, MPS의 `index_copy_`는 KV 길이에 비례하는 비용(호출당 KV 406/679/2317에서 151/196/663µs — 슬라이스 `copy_`는 5~6µs, 토큰당 24회 → 8쪽 청크 토큰당 15.9ms)이라 기본 8쪽 청크 디코드가 34 tok/s로 떨어졌다(audit MPS-1). eager를 int 슬롯으로 되돌린 뒤 실측: 8쪽 청크(프롬프트 2,189) 캡 320토큰 33.2~34.6 → 53.8~54.7 tok/s(+60%), 1쪽 캡 384토큰·8쪽 캡 320토큰 출력 토큰 동일 |

| P21 | 추론 경로의 죽은 계산·중간 텐서 3곳 제거 — (a) (`modeling_unlimitedocr.py`) `forward`에서 `labels is None`이고 `q_len>1`(프리필)이면 `lm_head` 전에 `hidden_states[:, -1:, :]`로 자른다, (b) (`modeling_unlimitedocr.py`) `prepare_inputs_for_generation`의 `cache_position` 계산 삭제(`model_inputs`에 포함되지 않던 죽은 값), (c) (`deepencoder.py`) `SAM` neck 뒤 `net_2` 출력의 불필요한 `clone()` 제거(`net_3`은 입력을 in-place 변경하지 않고 `x2`는 이후 미사용) | (a) 프리필이 (시퀀스 × vocab) fp32 logits를 통째로 실체화해 페이지당 ~1GB급 VRAM 스파이크를 냈다 — 추론(HF generate·fast_decode)은 `[:, -1, :]`만 소비하므로 **출력 불변**, 학습(labels) 경로는 전체 유지. (b) 디코드 스텝마다 버려지는 `torch.arange` 디바이스 할당 제거 — 소비처가 없어 **동작 불변**. (c) 값이 같은 복사본 제거 — **비트 동일** |

| P22 | (`modeling_unlimitedocr.py`) `draw_bounding_boxes`가 모델 좌표를 픽셀로 바꾼 뒤 **이미지 안으로 clamp** — `_clamp_box`: x/y = `int(v/999*size)`, x는 [0, W]·y는 [0, H]로 clamp, clamp 뒤 x2<=x1 또는 y2<=y1이면 퇴화 상자로 건너뜀(숫자가 아니거나(bool 포함)·비유한·4개가 아닌 좌표도 건너뜀). 건너뛴 image 상자와 좌표를 못 읽은 image ref도 크롭 번호(`img_idx`)는 소비 | **보안·견고성**: 모델 출력(PDF 내용으로 유도 가능)의 좌표가 그대로 `Image.crop`/`ImageDraw`로 가면 거대·음수 좌표가 페이지보다 큰 검정 크롭(메모리 폭주)이나 Pillow crop/paste 좌표 산술 오버플로(GHSA-6r8x-57c9-28j4 — Pillow 12.3.0에서 수정, 의존성 상향은 별도)에 닿는다(audit gap2-dependency-vuln-reachability-2). 번호 소비는 기존 crop 실패 시와 같은 규칙 — 마크다운 참조(`images/{prefix}{idx}.jpg`)·boxes.json 정렬 유지. 정상 좌표(0~999)의 결과는 불변. 같은 clamp 규칙을 MLX 엔진 이식이 공유한다 |
| P23 | (`modeling_unlimitedocr.py`) `infer_multi`의 페이지 분할을 `outputs.split('<PAGE>')[1:]` → `_split_multi_pages(outputs)`로 교체 — 첫 마커 앞이 **공백뿐일 때만** 버린다(마커 0개면 출력 전체가 1쪽) | 업스트림은 첫 마커 앞을 항상 버려, 모델이 선행 마커를 생략하면 1쪽 내용이 result·raw_pages.json에서 조용히 사라지고 이후 페이지의 figure 크롭이 한 칸 앞 이미지(`images[page_idx]`)로 잘렸다(audit decode-correctness-7 — 실제 잡 11건에서는 미관측). 앱 `merge.split_pages`와 같은 규칙이라 벤더 산출물과 병합이 어긋나지 않는다(무작위 입력 동치 테스트). 선행 마커가 있는 정상 출력의 결과는 불변 |

업스트림 갱신 시: 새 revision을 받아 이 패치들을 재적용하고 이 문서를 갱신할 것.
