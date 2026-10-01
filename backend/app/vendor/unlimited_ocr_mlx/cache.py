# Vendored from mlx-vlm 0.7.4 (MIT): mlx_vlm/models/unlimited_ocr/language.py
# (RingSlidingKVCache) + mlx_vlm/models/cache.py (KVCache 슬라이스 갱신 패턴).
# 로컬 패치: [local patch M2] 원샷 프리필에서 prefill_length = P를 기록 + 용량 P+W 선할당.
# 출처·패치 내역: PROVENANCE.md
"""Unlimited-OCR R-SWA(링 슬라이딩 윈도) KV 캐시.

레퍼런스 = torch ``SlidingWindowLlamaAttention`` (modeling_deepseekv2.py):

1. 프리필(첫 q_len>1 forward): 프롬프트 KV P개를 그대로 담고 ``_prefill_length=P``를
   기록한다. 이미지 토큰은 전부 프롬프트 안에 있으므로 절대 축출되지 않는다.
2. 워밍업 디코드: 캐시 길이가 P+W가 될 때까지 생성 토큰 KV를 이어 붙인다.
3. 정상상태 디코드: 슬롯 ``P + ring_pos``를 덮어쓰고 ``ring_pos = (ring_pos+1) % W``.
   어텐션은 프롬프트 전체 + 최근 생성 토큰 W개(자기 자신 포함)를 마스크 없이 본다.

RoPE는 캐시에 넣기 **전에** 절대 위치(``offset``)로 적용하므로 위치는 링과 무관하게
계속 증가한다(torch와 동일). 슬롯 순서는 softmax 합에 영향이 없다.
"""

from __future__ import annotations

import mlx.core as mx


class RingSlidingKVCache:
    def __init__(self, window_size: int):
        if not window_size or int(window_size) < 1:
            raise ValueError("window_size(링 창)는 1 이상이어야 합니다")
        self.window_size = int(window_size)
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        # 지금까지 넣은 총 토큰 수 = 다음 토큰의 RoPE 위치 (링 덮어쓰기와 무관하게 증가)
        self.offset = 0
        self.prefill_length: int | None = None
        self._ring_pos = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """keys/values [B, H, L, D]를 넣고 어텐션이 볼 KV를 돌려준다."""
        seq_len = int(keys.shape[2])

        if self.prefill_length is None:
            # [local patch M2] 첫 호출 = 프롬프트 전체를 한 번에 넣는 원샷 프리필.
            # torch는 첫 q_len>1 forward 직후 캐시 길이(=P)를 _prefill_length로 기록한다.
            # mlx-vlm 기본 chunked prefill은 마지막 프롬프트 토큰을 seq_len=1로 따로 넣어
            # P-1을 기록했고, 그 토큰이 링에 들어갔다가 W 스텝 뒤 축출됐다(8쪽 출력이
            # 토큰 105에서 갈림 — 감사 스파이크 MLX-04).
            if seq_len < 2:
                raise ValueError("첫 update는 프롬프트 전체(2토큰 이상) 원샷 프리필이어야 합니다")
            B, H, _, D = keys.shape
            capacity = seq_len + self.window_size  # 이후 재할당 없이 P+W 고정
            self.keys = mx.zeros((B, H, capacity, D), dtype=keys.dtype)
            self.values = mx.zeros((B, H, capacity, values.shape[3]), dtype=values.dtype)
            self.keys[..., :seq_len, :] = keys
            self.values[..., :seq_len, :] = values
            self.prefill_length = seq_len
            self.offset = seq_len
            return self.keys[..., :seq_len, :], self.values[..., :seq_len, :]

        if seq_len != 1:
            raise ValueError("프리필 이후에는 1토큰 디코드만 지원합니다")

        P, W = self.prefill_length, self.window_size
        if self.offset < P + W:
            # 워밍업: 링 영역이 찰 때까지 이어 붙이기 (torch cat-append와 같은 내용)
            prev = self.offset
            self.keys[..., prev : prev + 1, :] = keys
            self.values[..., prev : prev + 1, :] = values
            self.offset += 1
            if self.offset == P + W:
                self._ring_pos = 0
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

        # 정상상태: 가장 오래된 생성 토큰 슬롯을 덮어쓴다
        slot = P + self._ring_pos
        self.keys[..., slot : slot + 1, :] = keys
        self.values[..., slot : slot + 1, :] = values
        self._ring_pos = (self._ring_pos + 1) % W
        self.offset += 1
        return self.keys, self.values

    def make_mask(self, n_queries: int):
        """프리필은 인과 마스크, 디코드(q=1)는 마스크 없음 (torch와 동일)."""
        return "causal" if n_queries > 1 else None

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes
