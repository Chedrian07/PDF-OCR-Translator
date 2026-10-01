import functools
import random

import pytest

from app import native_ops


def _brute(seq, n, w):
    return native_ops.banned_ngram_tokens_py(seq, n, w)


CASES = [
    ([], 3, 10),
    ([1, 2], 3, 10),
    ([1, 2, 3], 3, 10),
    ([1, 2, 3, 1, 2, 4, 1, 2], 3, 100),
    ([5] * 40, 1, 8),
    (list(range(10)) * 3, 2, 7),
]


@pytest.mark.parametrize("seq,n,w", CASES)
def test_fallback_reference_cases(seq, n, w):
    out = native_ops.banned_ngram_tokens(seq, n, w)
    assert out == _brute(seq, n, w)


def test_fallback_semantics_repeat_blocking():
    # "1 2 3" 이 윈도우 안에 있고 현재 프리픽스가 "1 2"면 3이 금지된다
    assert 3 in _brute([1, 2, 3, 9, 9, 1, 2], 3, 100)
    # 윈도우 밖이면 금지되지 않는다
    assert _brute([1, 2, 3] + [9] * 50 + [1, 2], 3, 4) == []


@pytest.mark.skipif(not native_ops.HAVE_NATIVE, reason="uocr_native 미설치")
def test_native_matches_python_randomized():
    rng = random.Random(42)
    for _ in range(100):
        seq = [rng.randrange(6) for _ in range(rng.randrange(0, 200))]
        n = rng.randrange(1, 6)
        w = rng.randrange(1, 64)
        assert native_ops.banned_ngram_tokens(seq, n, w) == _brute(seq, n, w)


# ── 디바이스별 로짓 프로세서 패리티 (torch 필요 — cpu 텐서로 로직 검증) ──
# "…[static]"은 MPS 기본 티어(static_shape=True: 창 길이 고정 + 음수 센티널 패딩).

PROCESSOR_NAMES = [
    "TorchSlidingWindowNoRepeatNgram",
    "TorchSlidingWindowNoRepeatNgram[static]",
    "HostSlidingWindowNoRepeatNgram",
]


def _proc_factory(proc_name):
    if proc_name.endswith("[static]"):
        cls = getattr(native_ops, proc_name.removesuffix("[static]"))
        return functools.partial(cls, static_shape=True)
    return getattr(native_ops, proc_name)


def _processor_banned(proc_cls, seq, n, w, vocab):
    torch = pytest.importorskip("torch")
    ids = torch.tensor([list(seq)], dtype=torch.long)
    scores = torch.zeros(1, vocab)
    proc_cls(n, w)(ids, scores)
    return sorted(i for i in range(vocab) if scores[0, i] == float("-inf"))


@pytest.mark.parametrize("proc_name", PROCESSOR_NAMES)
@pytest.mark.parametrize("seq,n,w", CASES)
def test_processors_reference_cases(proc_name, seq, n, w):
    proc_cls = _proc_factory(proc_name)
    vocab = 16
    assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w)


@pytest.mark.parametrize("proc_name", PROCESSOR_NAMES)
def test_processors_randomized_parity(proc_name):
    proc_cls = _proc_factory(proc_name)
    rng = random.Random(7)
    vocab = 8
    for _ in range(200):
        seq = [rng.randrange(vocab) for _ in range(rng.randrange(0, 160))]
        n = rng.randrange(1, 6)
        w = rng.randrange(1, 48)
        assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w), (seq, n, w)


@pytest.mark.parametrize("proc_name", PROCESSOR_NAMES)
def test_processors_window_slice_boundaries(proc_name):
    # 슬라이스 동치가 위험한 경계: L == w, L == w±1, L == n, w < n
    proc_cls = _proc_factory(proc_name)
    vocab = 6
    rng = random.Random(11)
    for L, n, w in [(64, 3, 64), (65, 3, 64), (63, 3, 64), (5, 5, 64), (40, 4, 2), (40, 1, 8)]:
        for _ in range(20):
            seq = [rng.randrange(vocab) for _ in range(L)]
            assert _processor_banned(proc_cls, seq, n, w, vocab) == _brute(seq, n, w), (L, n, w)


def test_processor_production_shape():
    # 실사용 파라미터 (n=35, window=1024) 대형 시퀀스
    rng = random.Random(3)
    vocab = 24
    seq = [rng.randrange(vocab) for _ in range(3000)]
    # 반복 유도: 마지막 34개 프리픽스를 윈도우 중간에 복제
    seq[1500:1534] = seq[-34:]
    ref = _brute(seq, 35, 1024)
    for name in PROCESSOR_NAMES:
        assert _processor_banned(_proc_factory(name), seq, 35, 1024, vocab) == ref, name
    # 창이 덜 찬 구간(프롬프트 < window — 1쪽 멀티 청크의 첫 ~750토큰)도 동일
    short = seq[:600]
    short[300:334] = short[-34:]
    ref_short = _brute(short, 35, 1024)
    for name in PROCESSOR_NAMES:
        assert _processor_banned(_proc_factory(name), short, 35, 1024, vocab) == ref_short, name


def test_make_processor_tiers():
    torch_proc = native_ops.make_ngram_logits_processor(35, 1024, "cuda")
    assert isinstance(torch_proc[0], native_ops.TorchSlidingWindowNoRepeatNgram)
    assert torch_proc[0].static_shape is False  # CUDA 커널은 shape 무관 — 동적 슬라이스 유지
    mps_proc = native_ops.make_ngram_logits_processor(35, 1024, "mps")
    assert isinstance(mps_proc[0], native_ops.TorchSlidingWindowNoRepeatNgram)
    assert mps_proc[0].static_shape is True  # MPS는 shape별 그래프 컴파일 — 창 길이 고정
    cpu_proc = native_ops.make_ngram_logits_processor(35, 128, "cpu")
    assert isinstance(cpu_proc[0], native_ops.HostSlidingWindowNoRepeatNgram)


def test_static_shape_window_length_never_changes():
    """MPS 티어의 밴 검사 창은 시퀀스 길이와 무관하게 항상 [window] — 길이가 토큰마다
    바뀌면 MPS가 새 그래프를 컴파일·캐시해 RSS가 길이당 ~2.9MB씩 영구히 늘었다
    (audit gap1-metal-real-e2e-4). 동적 티어(CUDA)는 기존처럼 min(L, window)."""
    torch = pytest.importorskip("torch")
    static = native_ops.TorchSlidingWindowNoRepeatNgram(4, 16, static_shape=True)
    dynamic = native_ops.TorchSlidingWindowNoRepeatNgram(4, 16)
    for length in range(1, 40):
        seq = torch.arange(length, dtype=torch.long)
        seg = static._segment(seq)
        pads = max(0, 16 - length)
        assert seg.shape == (16,), length
        assert seg[pads:].tolist() == seq[-16:].tolist()  # 실제 토큰은 끝에 그대로
        assert (seg[:pads] < 0).all()  # 앞은 음수 센티널
        assert len(set(seg[:pads].tolist())) == pads  # 센티널끼리도 서로 다름
        assert dynamic._segment(seq).shape == (min(length, 16),)


def test_static_shape_unigram_never_bans_from_padding():
    """n=1이면 모든 창이 '일치'하므로 센티널 창을 무효 처리하지 않으면 음수 인덱스가
    다른 토큰을 밴한다 — 레퍼런스와 같은 집합만 밴해야 한다."""
    proc = functools.partial(native_ops.TorchSlidingWindowNoRepeatNgram, static_shape=True)
    for seq in ([3], [5, 0], [1, 1, 2]):
        assert _processor_banned(proc, seq, 1, 8, 6) == _brute(seq, 1, 8), seq


# ── GraphSlidingWindowNoRepeatNgram (CUDA Graph용 정적 shape) — CPU 텐서로 로직 검증 ──

def _graph_ngram(n, w):
    torch = pytest.importorskip("torch")
    return native_ops.GraphSlidingWindowNoRepeatNgram(n, w, torch.device("cpu"))


def _graph_banned(g, vocab):
    """step()이 밴한 토큰 집합 (scores=0 기준)."""
    torch = pytest.importorskip("torch")
    scores = g.step(torch.zeros(1, vocab))
    return sorted(i for i in range(vocab) if scores[0, i] == float("-inf"))


def test_graph_ngram_parity_200_token_greedy_walk():
    """무작위 로짓 200스텝 그리디: Host 프로세서(전체 시퀀스 유지) vs
    GraphNgram(prime+step/push) — 선택 토큰 시퀀스가 완전 동일해야 한다."""
    torch = pytest.importorskip("torch")
    rng = random.Random(1234)
    vocab = 12
    for n, w in [(3, 16), (2, 8), (35, 64), (1, 8), (4, 200)]:
        prefix = [rng.randrange(vocab) for _ in range(rng.randrange(0, 40))]
        gen = torch.Generator().manual_seed(n * 1000 + w)
        logits_seq = [torch.randn(vocab, generator=gen) for _ in range(200)]

        # 레퍼런스: 전체 시퀀스를 들고 Host 프로세서로 밴
        host = native_ops.HostSlidingWindowNoRepeatNgram(n, w)
        seq = list(prefix)
        host_out = []
        for logits in logits_seq:
            ids = torch.tensor([seq], dtype=torch.long)
            scores = host(ids, logits.clone().unsqueeze(0))
            nxt = int(scores.argmax(dim=-1))
            seq.append(nxt)
            host_out.append(nxt)

        # 그래프용: prime 후 step/push만으로 동일 워크
        g = _graph_ngram(n, w)
        g.prime(torch.tensor(prefix, dtype=torch.long))
        graph_out = []
        for logits in logits_seq:
            scores = g.step(logits.clone().unsqueeze(0))
            nxt = int(scores.argmax(dim=-1))
            g.push(torch.tensor(nxt))
            graph_out.append(nxt)

        assert graph_out == host_out, (n, w, len(prefix))


def test_graph_ngram_noop_until_ngram_size_reached():
    """count < ngram_size 동안은 밴 없음(no-op) — 마스크 산술 분기 검증."""
    torch = pytest.importorskip("torch")
    n, w, vocab = 4, 8, 6
    g = _graph_ngram(n, w)
    g.prime(torch.tensor([1, 2], dtype=torch.long))  # count=2 < n → no-op
    assert _graph_banned(g, vocab) == []
    g.push(torch.tensor(3))  # count=3 < n → 여전히 no-op
    assert _graph_banned(g, vocab) == []
    g.push(torch.tensor(1))  # count=4 = n → [1,2,3,1]: 레퍼런스와 동일해야 함
    assert _graph_banned(g, vocab) == _brute([1, 2, 3, 1], n, w)


def test_graph_ngram_window_smaller_than_ngram_is_noop():
    """window < ngram_size면 ngram 자체가 불가 — 상수 no-op 분기."""
    torch = pytest.importorskip("torch")
    g = _graph_ngram(5, 3)
    g.prime(torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long))
    assert _graph_banned(g, 4) == []


def test_graph_ngram_ring_wraparound_matches_reference():
    """window(8)의 여러 바퀴(50스텝)를 돌며 매 스텝 밴 집합이 레퍼런스와 동일."""
    torch = pytest.importorskip("torch")
    rng = random.Random(5)
    n, w, vocab = 3, 8, 6
    g = _graph_ngram(n, w)
    seq = [rng.randrange(vocab) for _ in range(4)]
    g.prime(torch.tensor(seq, dtype=torch.long))
    for i in range(50):
        assert _graph_banned(g, vocab) == _brute(seq, n, w), (i, seq)
        tok = rng.randrange(vocab)
        g.push(torch.tensor(tok))
        seq.append(tok)


def test_graph_ngram_prime_then_recent():
    """prime 후 recent(k)가 마지막 k개 토큰(오래된→최신)을 정확히 돌려준다 —
    그래프 리플레이 루프의 블록 회수(D2H) 정합의 근거."""
    torch = pytest.importorskip("torch")
    g = _graph_ngram(3, 8)
    g.prime(torch.tensor([1, 2, 3, 4, 5], dtype=torch.long))
    assert g.recent(3).tolist() == [3, 4, 5]
    assert g.recent(5).tolist() == [1, 2, 3, 4, 5]
    g.push(torch.tensor(6))
    g.push(torch.tensor(7))
    assert g.recent(4).tolist() == [4, 5, 6, 7]
    # 링 용량 초과 push 후에도 최신 k개 유지 (랩어라운드)
    for t in [8, 9, 10, 11, 12]:
        g.push(torch.tensor(t))
    assert g.recent(4).tolist() == [9, 10, 11, 12]
    # 링 용량(w=4)보다 긴 프라임: 마지막 w개만 보존
    g2 = _graph_ngram(3, 4)
    g2.prime(torch.tensor(list(range(10)), dtype=torch.long))
    assert g2.recent(4).tolist() == [6, 7, 8, 9]
    assert g2.recent(2).tolist() == [8, 9]


def test_ngram_host_강제_레버(monkeypatch):
    """OCR_NGRAM_HOST=1이면 GPU/MPS에서도 호스트 티어 — MPS scatter 이슈 절연용."""
    from app.native_ops import (
        HostSlidingWindowNoRepeatNgram,
        TorchSlidingWindowNoRepeatNgram,
        make_ngram_logits_processor,
    )

    monkeypatch.delenv("OCR_NGRAM_HOST", raising=False)
    assert isinstance(make_ngram_logits_processor(30, 128, "mps")[0], TorchSlidingWindowNoRepeatNgram)
    monkeypatch.setenv("OCR_NGRAM_HOST", "1")
    assert isinstance(make_ngram_logits_processor(30, 128, "mps")[0], HostSlidingWindowNoRepeatNgram)
    assert isinstance(make_ngram_logits_processor(30, 128, "cuda")[0], HostSlidingWindowNoRepeatNgram)


def test_ngram_host_레버_그래프_비허용_마커(monkeypatch):
    """강제 호스트 프로세서는 graph_capable=False — CUDA 그래프 경로가 GraphNgram으로
    바꿔치기해 절연을 조용히 무효화하는 함정 방지. 기본 경로는 마커 없이 그래프 허용."""
    from app.native_ops import make_ngram_logits_processor

    monkeypatch.setenv("OCR_NGRAM_HOST", "1")
    for dev in ("cuda", "mps", "cpu"):
        assert make_ngram_logits_processor(30, 128, dev)[0].graph_capable is False, dev

    monkeypatch.delenv("OCR_NGRAM_HOST", raising=False)
    for dev in ("cuda", "mps", "cpu"):
        proc = make_ngram_logits_processor(30, 128, dev)[0]
        assert getattr(proc, "graph_capable", True) is True, dev  # 마커 없음 = 그래프 허용
