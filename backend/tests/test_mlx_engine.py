"""MLX in-process 엔진(UnlimitedMLXEngine) 계약 — torch UnlimitedEngine과 같은 의미.

- 스트리밍: sink로 나간 텍스트가 HF TextStreamer(skip_special_tokens=False)와 호출 단위까지
  같고(EOS 문자열 → '\\n'), 이어 붙이면 생성 원문과 같다. 접두 메모 디코드는 바이트 수준
  디코더에서만 켜진다.
- 산출물: 반환 마크다운 = result.md, 파일 이름·배치가 torch 규약(engine/base.py)과 같다.
- 취소는 토큰 1개 안에 멈추고 부분 출력을 돌려준다. 반복·페이지 예산은 RepetitiveOutputError,
  MAX_LENGTH 잘림은 OutputLimitError(runner의 페이지 단위 복구 대상) — multi는 잘린 출력을
  partial_output으로 실어 끝까지 생성된 앞 페이지를 살린다.
- load()는 동시 호출에도 한 번만 돌고, 로드한 스레드와 다른 스레드에서 실행해도 결과가 같다.
- create_app → 업로드 → done까지 이 엔진으로 돈다(잘림 → 앞 페이지 유지·페이지 단위 복구 포함).

모델은 두 가지다: 토큰을 대본대로 내는 가짜 모델(내용·파일·취소 검사)과 작은 무작위 가중치
MLX 모델(실제 비전·프리필·링 캐시 디코드 경로). 토크나이저는 이 파일에서 만드는 바이트 단위
BPE다(id = 2 + 바이트, 실토크나이저와 같은 ByteLevel 디코더) — 다바이트 문자가 토큰 사이에서
쪼개지는 경우가 매번 생긴다. mlx가 필요한 검사는 mlx가 없으면(Linux CI) 건너뛴다.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.engine import unlimited_mlx as um
from app.engine.base import EngineError, OutputLimitError, RepetitiveOutputError
from app.engine.repetition import SemanticRepetitionDetector
from app.engine.unlimited_mlx import UnlimitedMLXEngine
from app.vendor.unlimited_ocr_mlx import IMAGE_TOKEN_ID, STOP_STR, mlx_status

_MLX_OK, _MLX_WHY = mlx_status()
needs_mlx = pytest.mark.skipif(not _MLX_OK, reason=_MLX_WHY)

BACKEND = Path(__file__).resolve().parents[1]
BOS_TEXT = "<｜begin▁of▁sentence｜>"
EOS = 1
VOCAB = 129280  # 실제 어휘 크기 — 이미지 토큰(128815)·ngram 금지 슬롯이 로짓 범위 안에 있게
TOKENS_PER_PAGE = 273  # 1024px 전역 뷰: (16 + 1) × 16 + 1


# ── 테스트 토크나이저 (바이트 단위 BPE) ───────────────────────────


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 ByteLevel 바이트 → 문자 표 (tokenizers의 ByteLevel 디코더가 되돌리는 표)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _byte_tokenizer(*, clean_up: bool = False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    b2u = _bytes_to_unicode()
    vocab = {BOS_TEXT: 0, STOP_STR: EOS}
    for b in range(256):
        vocab[b2u[b]] = 2 + b
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token=BOS_TEXT, eos_token=STOP_STR,
        clean_up_tokenization_spaces=clean_up,
    )


@pytest.fixture(scope="module")
def tok():
    t = _byte_tokenizer()
    assert t.eos_token_id == EOS and t.decode([EOS]) == STOP_STR
    return t


def _enc(text: str) -> list[int]:
    """테스트 토크나이저의 encode와 같다 (id = 2 + UTF-8 바이트)."""
    return [2 + b for b in text.encode("utf-8")]


class RecSink:
    def __init__(self, on_first=None) -> None:
        self.parts: list[str] = []
        self._on_first = on_first

    def on_text(self, text: str) -> None:
        self.parts.append(text)
        if self._on_first is not None and len(self.parts) == 1:
            self._on_first()

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _reference_calls(tokenizer, ids) -> list[tuple[str, bool]]:
    """HF TextStreamer에 토큰을 하나씩 넣었을 때의 on_finalized_text 호출들."""
    from transformers import TextStreamer

    class Rec(TextStreamer):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            self.calls: list[tuple[str, bool]] = []

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            self.calls.append((text, stream_end))

    s = Rec(tokenizer, skip_prompt=False, skip_special_tokens=False)
    for i in ids:
        s.put(np.array([i]))
    s.end()
    return s.calls


def _memo_calls(tokenizer, ids) -> list[tuple[str, bool]]:
    """같은 TextStreamer가 접두 메모 디코더를 쓸 때의 호출들."""
    from transformers import TextStreamer

    class Rec(TextStreamer):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            self.calls: list[tuple[str, bool]] = []

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            self.calls.append((text, stream_end))

    s = Rec(um._PrefixMemoDecoder(tokenizer), skip_prompt=False, skip_special_tokens=False)
    for i in ids:
        s.put(np.array([i]))
    s.end()
    return s.calls


MIXED_TEXT = (
    "<PAGE><|det|>title [155, 133, 844, 185]<|/det|>TurboQuant: Online Vector Quantization\n"
    "<|det|>text [180, 213, 361, 268]<|/det|>한국어 문단입니다. 中文没有空格的长行中文测试 日本語のテキスト "
    "😀🎉 ½ ∑∫ é́ <table><tr><td>셀</td><td>값 1,234</td></tr></table>\n\n"
    "$$\\sum_{i=1}^{n} x_i^2$$  trailing   spaces   \n"
)


# ── 스트리밍 (mlx 불필요) ───────────────────────────────────────


def test_prefix_memo_streams_exactly_like_textstreamer(tok):
    """바이트 토큰이 다바이트 문자를 쪼개는 모든 경우에 TextStreamer와 호출 단위까지 같다."""
    rng = random.Random(1234)
    cases = [_enc(MIXED_TEXT) + [EOS], _enc(MIXED_TEXT * 3), [EOS], []]
    for _ in range(150):
        n = rng.randrange(1, 300)
        ids = [2 + rng.randrange(256) for _ in range(n)]  # 무효 UTF-8 조각 포함
        for _ in range(rng.randrange(0, 4)):
            ids.insert(rng.randrange(len(ids) + 1), rng.choice([EOS, 2 + 10, 2 + 32]))
        cases.append(ids)
    for ids in cases:
        assert _memo_calls(tok, ids) == _reference_calls(tok, ids), ids


def test_prefix_memo_matches_the_real_tokenizer_when_cached():
    """고정 리비전 실토크나이저(로컬 HF 캐시에 있을 때만) — 실제 BPE 병합·특수 토큰에서도 같다."""
    from transformers import AutoTokenizer

    s = Settings()
    try:
        real = AutoTokenizer.from_pretrained(
            s.model_id, revision=s.model_revision, local_files_only=True
        )
    except Exception as e:  # noqa: BLE001 — 캐시 없음(CI)
        pytest.skip(f"실토크나이저 캐시 없음: {type(e).__name__}")
    assert um._decode_is_char_additive(real)
    rng = random.Random(7)
    vocab = len(real)  # 실토크나이저의 len()은 추가 토큰 때문에 느리다 — 한 번만
    ids = real.encode(MIXED_TEXT * 4, add_special_tokens=False) + [real.eos_token_id]
    cases = [ids]
    for _ in range(40):
        cases.append([rng.randrange(vocab) for _ in range(rng.randrange(1, 400))])
    for case in cases:
        assert _memo_calls(real, case) == _reference_calls(real, case)


def test_memo_is_used_only_for_char_additive_decoders(tok):
    """공백 정리 규칙이나 Metaspace(첫 토큰 앞 공백 제거) 디코더는 조각 디코드가 전체와
    달라진다 — 그런 토크나이저에서는 메모 없이 전체 디코드로 TextStreamer와 같게 낸다."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    assert um._decode_is_char_additive(tok)
    assert not um._decode_is_char_additive(_byte_tokenizer(clean_up=True))

    tk = Tokenizer(models.WordLevel({"▁hello": 0, "▁world": 1, "<unk>": 2}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Metaspace()
    tk.decoder = decoders.Metaspace()
    meta = PreTrainedTokenizerFast(tokenizer_object=tk, clean_up_tokenization_spaces=False)
    assert not um._decode_is_char_additive(meta)
    ids = [0, 1, 1, 0]
    # 메모를 강제로 쓰면 실제로 달라진다 — 가드가 의미 있는 이유
    assert _memo_calls(meta, ids) != _reference_calls(meta, ids)
    sink = RecSink()
    det = SemanticRepetitionDetector(max_page_chars=None, max_page_tokens=None)
    streamer = um.make_sink_streamer(meta, sink, det, eos_text="</s>")
    for i in ids:
        streamer.put(np.array([i]))
    streamer.end()
    assert sink.parts == [t for t, _ in _reference_calls(meta, ids) if t]


class _RecDetector:
    """감지기 호출 순서 기록 (feed 텍스트 → feed_tokens 개수)."""

    detected = False

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def feed(self, text, *, stream_end=False):
        self.events.append(("text", text, stream_end))
        return False

    def feed_tokens(self, count):
        self.events.append(("tokens", count))
        return False


def test_sink_streamer_maps_eos_to_newline_and_feeds_text_before_tokens(tok):
    ids = _enc("Hello world 한글\nnext line") + [EOS]
    sink, det = RecSink(), _RecDetector()
    streamer = um.make_sink_streamer(tok, sink, det, eos_text=STOP_STR)
    for i in ids:
        streamer.put(np.array([i]))
    streamer.end()
    assert sink.text == "Hello world 한글\nnext line\n"
    assert STOP_STR not in sink.text
    # 토큰마다 정확히 1개씩, 디코드된 텍스트를 먼저 본 뒤에 센다 (torch _SinkStreamer와 같은 순서)
    assert sum(e[1] for e in det.events if e[0] == "tokens") == len(ids)
    for i, event in enumerate(det.events):
        if event[0] == "text" and not event[2]:
            assert det.events[i + 1][0] == "tokens"
    # EOS 문자열은 디코드 시점엔 개행이 아니라 'line<EOS>' 꼬리가 end()에서 플러시된다
    assert det.events[-1] == ("text", "line\n", True)


# ── 생성·설정 (mlx 불필요) ──────────────────────────────────────


def test_engine_construction_needs_no_mlx_torch_or_transformers():
    code = (
        "import sys\n"
        "from app.config import Settings\n"
        "from app.engine.unlimited_mlx import UnlimitedMLXEngine\n"
        "e = UnlimitedMLXEngine(Settings(engine='unlimited', device='mlx', preload_model=False))\n"
        "assert e.device == 'mlx' and e.name == 'unlimited' and not e.loaded\n"
        "print(sorted(m for m in ('mlx', 'mlx.core', 'torch', 'transformers') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True,
        timeout=120, check=True,
    )
    assert out.stdout.strip() == "[]"


def test_capabilities_match_the_torch_engine():
    from app.engine.unlimited import UnlimitedEngine

    s = Settings(engine="unlimited", device="mlx", preload_model=False)
    torch_caps = UnlimitedEngine(Settings(engine="unlimited", device="cpu")).capabilities()
    assert UnlimitedMLXEngine(s).capabilities() == torch_caps


@pytest.mark.parametrize(
    "dtype,bits,name",
    [("auto", 0, "bfloat16"), ("auto", 8, "bfloat16+q8"), ("bfloat16", 0, "bfloat16"),
     ("float32", 0, "float32"), ("float16", 8, "float16+q8")],
)
def test_dtype_name_reports_dtype_and_quantization(dtype, bits, name):
    eng = UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx", dtype=dtype, mlx_quant_bits=bits))
    assert eng.dtype_name == name


def test_invalid_dtype_or_quant_bits_fail_at_construction():
    with pytest.raises(ValueError, match="OCR_DTYPE"):
        UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx", dtype="int8"))
    with pytest.raises(ValueError, match="OCR_MLX_QUANT_BITS"):
        UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx", mlx_quant_bits=4))


@needs_mlx
def test_engine_dtype_and_quant_tables_match_the_vendor_loader():
    from app.vendor.unlimited_ocr_mlx.loader import DTYPES, SUPPORTED_QUANT_BITS

    assert set(um.MLX_DTYPES) == set(DTYPES)
    assert set(um.MLX_QUANT_BITS) - {0} == set(SUPPORTED_QUANT_BITS)


def test_load_on_non_apple_silicon_fails_with_a_korean_reason(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(um.platform, "machine", lambda: "x86_64")
    eng = UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx"))
    with pytest.raises(EngineError, match="Apple Silicon") as info:
        eng.load()
    assert "linux/x86_64" in str(info.value) and "OCR_DEVICE=auto" in str(info.value)
    assert not eng.loaded


def test_load_without_mlx_points_to_the_install_command(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(um.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    eng = UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx"))
    with pytest.raises(EngineError, match="uv sync --extra metal --extra mlx"):
        eng.load()


# ── 대본 모델 (mlx 필요) ────────────────────────────────────────

MULTI_MARK = _enc("Multi")


def _contains(seq: list[int], sub: list[int]) -> bool:
    n = len(sub)
    return any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def page_text(i: int) -> str:
    """멀티 출력 1쪽 — 페이지마다 좌표·문구가 달라 35-그램 금지가 대본을 바꾸지 않는다."""
    return (
        f"<PAGE><|det|>title [{100 + i}, {50 + i}, {900 - i}, {90 + i}]<|/det|>Heading {i + 1}\n"
        f"<|det|>image [{100 + 7 * i}, {120 + 3 * i}, {500 + 5 * i}, {420 + 2 * i}]<|/det|>\n"
        f"<|det|>text [{90 + i}, {450 + i}, {910 - i}, {600 + i}]<|/det|>Body {i + 1} 한글 中文 😀 text.\n"
    )


SINGLE_TEXT = (
    "<|det|>image [150, 150, 550, 450]<|/det|>\n"
    "<|det|>text [100, 480, 900, 620]<|/det|>Single page body recovered alone.\n"
)


def endless_text(n_words: int = 4000, seed: int = 3, *, marker: bool = True) -> str:
    """EOS 없이 이어지는 무반복 본문 (잘림 검사용 — 행 템플릿·롤링 반복이 없다)."""
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "theta", "zeta", "lambda"]
    body = " ".join(f"{rng.choice(words)}{rng.randrange(10**6)}" for _ in range(n_words))
    return ("<PAGE>" if marker else "") + body


def template_rows(n: int = 200) -> str:
    """숫자만 다른 행(행 템플릿 루프). 번호가 모두 달라 35-그램 금지에 걸리지 않는다 —
    숫자 사이 고정 문구(24바이트)가 35보다 짧아 모든 35-그램이 번호 하나를 통째로 품는다."""
    return "".join(f"Entry {100000 + 7919 * i} value alpha beta\n" for i in range(n))


def _assert_ngram_free(text: str, n: int = 35) -> None:
    """대본에 같은 n-그램이 두 번 나오지 않는다 = 엔진의 no-repeat-ngram이 대본을 안 바꾼다."""
    ids = _enc(text)
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    assert len(set(grams)) == len(grams), "대본에 반복 35-그램 — ngram 금지가 출력을 바꾼다"


def test_scripts_are_free_of_repeated_35_grams():
    for text in (page_text(0) + page_text(1), SINGLE_TEXT, endless_text(),
                 endless_text(marker=False), "<PAGE>" + template_rows()):
        _assert_ngram_free(text)


class _ScriptedLM:
    def __init__(self, owner: "ScriptedModel") -> None:
        self.owner = owner
        self.k = 0

    def __call__(self, inputs, inputs_embeds=None, cache=None, last_only=False):
        import mlx.core as mx

        o = self.owner
        if inputs_embeds is not None:  # 프리필 — 대본을 처음부터
            self.k = 0
        else:
            o.decode_calls += 1
        tok_id = o.script[self.k] if self.k < len(o.script) else EOS
        self.k += 1
        return (mx.arange(VOCAB) == tok_id).astype(mx.float32)[None, None, :] * 10.0


class ScriptedModel:
    """vendor generate 계약(make_cache·encode_images·get_input_embeddings·language_model)만
    갖춘 가짜 모델. 프롬프트(멀티/단일, 페이지 수)를 보고 대본 토큰을 고른다."""

    def __init__(self, multi=None, single=None) -> None:
        self.language_model = _ScriptedLM(self)
        self._multi = multi or (lambda pages: "".join(page_text(i) for i in range(pages)))
        self._single = single or (lambda: SINGLE_TEXT)
        self.script: list[int] = []
        self.prompts: list[tuple[str, int]] = []
        self.decode_calls = 0

    def make_cache(self):
        return []

    def encode_images(self, global_views, crops=None, spatial_crop=(1, 1)):
        import mlx.core as mx

        return mx.zeros((1, 4))

    def get_input_embeddings(self, input_ids, images_seq_mask, image_features):
        import mlx.core as mx

        ids = [int(x) for x in input_ids]
        if _contains(ids, MULTI_MARK):
            pages = ids.count(IMAGE_TOKEN_ID) // TOKENS_PER_PAGE
            mode, text = "multi", self._multi(pages)
        else:
            pages, mode, text = 1, "single", self._single()
        self.prompts.append((mode, pages))
        self.script = (_enc(text) + [EOS]) if text is not None else []
        self.decode_calls = 0
        return mx.zeros((1, len(ids), 4))


def _page_images(tmp_path: Path, n: int = 2, size=(600, 800)) -> list[Path]:
    paths = []
    for i in range(n):
        im = Image.new("RGB", size, (255, 255, 255))
        im.paste((60 + 40 * i, 90, 200), (100, 150, 320, 380))
        p = tmp_path / f"page_{i + 1:04d}.png"
        im.save(p)
        paths.append(p)
    return paths


def _engine(monkeypatch, model, tokenizer, **settings_kw) -> UnlimitedMLXEngine:
    s = Settings(engine="unlimited", device="mlx", preload_model=False, **settings_kw)
    eng = UnlimitedMLXEngine(s)
    monkeypatch.setattr(eng, "_load_weights", lambda: (model, tokenizer))
    monkeypatch.setattr(eng, "_warmup", lambda m, t: None)
    return eng


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


@needs_mlx
def test_run_multi_streams_what_it_generates_and_writes_torch_named_artifacts(
    monkeypatch, tmp_path, tok
):
    model = ScriptedModel()
    eng = _engine(monkeypatch, model, tok)
    sink, out = RecSink(), tmp_path / "chunk_00"
    md = eng.run_multi(_page_images(tmp_path), out, sink, threading.Event())

    raw = page_text(0) + page_text(1)
    # 스트림 = 생성 원문 + EOS 자리의 '\n', 그리고 TextStreamer와 같은 단위
    assert sink.text == raw + "\n"
    ids = _enc(raw) + [EOS]
    ref = [t.replace(STOP_STR, "\n") for t, _ in _reference_calls(tok, ids) if t]
    assert sink.parts == ref
    assert model.prompts == [("multi", 2)]
    gen = eng.last_generation
    assert gen.token_ids == ids and gen.finish_reason == "eos"

    # torch 규약: images/page_{청크내idx}_{k}.jpg, result_with_boxes_{i}.jpg, P13/P14 메타
    assert _files(out) == [
        "boxes.json", "images/page_0_0.jpg", "images/page_1_0.jpg", "raw_pages.json",
        "result.md", "result_with_boxes_0.jpg", "result_with_boxes_1.jpg",
    ]
    assert md == (out / "result.md").read_text(encoding="utf-8")
    assert md.startswith("<PAGE>\n") and md.count("<PAGE>") == 2
    assert "![](images/page_0_0.jpg)" in md and "![](images/page_1_0.jpg)" in md
    assert "<|det|>" not in md and "Body 2 한글 中文 😀 text." in md
    pages = json.loads((out / "raw_pages.json").read_text(encoding="utf-8"))["pages"]
    assert pages == [p.strip() for p in sink.text.split("<PAGE>")[1:]]
    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    assert set(boxes) == {"page_0_0.jpg", "page_1_0.jpg"}
    assert boxes["page_0_0.jpg"]["image_width"] == 600


@needs_mlx
def test_run_single_writes_torch_named_artifacts(monkeypatch, tmp_path, tok):
    model = ScriptedModel()
    eng = _engine(monkeypatch, model, tok)
    sink, out = RecSink(), tmp_path / "page_0003"
    md = eng.run_single(_page_images(tmp_path, 1)[0], out, sink, threading.Event())
    assert model.prompts == [("single", 1)]
    assert sink.text == SINGLE_TEXT + "\n"
    assert _files(out) == [
        "boxes.json", "images/0.jpg", "raw_pages.json", "result.md", "result_with_boxes.jpg",
    ]
    assert md == (out / "result.md").read_text(encoding="utf-8")
    assert md == "![](images/0.jpg)\n\nSingle page body recovered alone."


@needs_mlx
def test_output_limit_raises_output_limit_error_for_page_recovery(monkeypatch, tmp_path, tok):
    model = ScriptedModel(multi=lambda pages: endless_text())
    eng = _engine(monkeypatch, model, tok, max_length=2 * TOKENS_PER_PAGE + 200)
    sink = RecSink()
    with pytest.raises(OutputLimitError, match="MAX_LENGTH") as info:
        eng.run_multi(_page_images(tmp_path), tmp_path / "out", sink, threading.Event())
    assert isinstance(info.value, RepetitiveOutputError)  # runner 복구 경로(하위 클래스)
    gen = eng.last_generation
    assert gen.hit_max_length and gen.prompt_length + len(gen.token_ids) == eng._settings.max_length
    assert EOS not in gen.token_ids
    # 잘린 출력은 run_multi 형식(= 저장된 result.md)으로 실려 runner가 앞 페이지를 살릴 수 있다
    partial = info.value.partial_output
    assert partial == (tmp_path / "out" / "result.md").read_text(encoding="utf-8")
    assert partial.startswith("<PAGE>\n")
    words = ("alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "theta", "zeta")
    assert any(word in partial for word in words)
    assert not any(word in str(info.value) for word in words)  # 문서 내용은 메시지에 없다


@needs_mlx
def test_single_output_limit_carries_no_partial_output(monkeypatch, tmp_path, tok):
    """single은 한 쪽이라 살릴 앞 페이지가 없다 — partial_output 없이 페이지 복구로 간다."""
    from app.engine.unlimited import SINGLE_PROMPT
    from app.vendor.unlimited_ocr_mlx import prepare_single

    image = _page_images(tmp_path, 1)[0]
    prompt_len = prepare_single(tok, SINGLE_PROMPT, str(image), base_size=1024,
                                image_size=640, crop_mode=True).prompt_length
    model = ScriptedModel(single=lambda: endless_text(marker=False))
    eng = _engine(monkeypatch, model, tok, max_length=prompt_len + 100)
    with pytest.raises(OutputLimitError) as info:
        eng.run_single(image, tmp_path / "out", RecSink(), threading.Event())
    assert eng.last_generation.hit_max_length
    assert info.value.partial_output is None


@needs_mlx
def test_cancel_stops_within_one_token_and_returns_partial_output(monkeypatch, tmp_path, tok):
    cancel = threading.Event()
    model = ScriptedModel(multi=lambda pages: endless_text())
    eng = _engine(monkeypatch, model, tok)
    sink = RecSink(on_first=cancel.set)  # 첫 단어가 확정되는 토큰에서 취소
    md = eng.run_multi(_page_images(tmp_path), tmp_path / "out", sink, cancel)

    gen = eng.last_generation
    first_space = _enc(endless_text()).index(2 + ord(" "))
    assert gen.finish_reason == "stop"
    assert len(gen.token_ids) == first_space + 1  # 취소를 본 바로 그 토큰에서 멈춘다
    assert model.decode_calls <= len(gen.token_ids)  # 버린 GPU 스텝 ≤ 1
    assert isinstance(md, str)  # 예외 없이 부분 출력 (병합 뒤 취소 처리는 runner 몫)
    assert (tmp_path / "out" / "result.md").is_file()


@needs_mlx
def test_canceled_output_limit_returns_partial_instead_of_raising(monkeypatch, tmp_path, tok):
    """취소가 이긴다 — 상한에 닿은 실행도 취소됐으면 부분 출력을 돌려준다 (torch와 같은 규칙)."""
    from app.engine.unlimited import MULTI_PROMPT
    from app.vendor.unlimited_ocr_mlx import prepare_multi

    images = _page_images(tmp_path)
    prompt_len = prepare_multi(tok, MULTI_PROMPT, [str(p) for p in images], image_size=1024).prompt_length
    first_space = _enc(endless_text()).index(2 + ord(" "))
    cancel = threading.Event()
    model = ScriptedModel(multi=lambda pages: endless_text())
    # 마지막 허용 토큰이 첫 공백 — 그 토큰에서 첫 단어가 sink로 나가며 취소된다
    eng = _engine(monkeypatch, model, tok, max_length=prompt_len + first_space + 1)
    md = eng.run_multi(images, tmp_path / "out", RecSink(on_first=cancel.set), cancel)
    assert isinstance(md, str) and cancel.is_set()
    assert eng.last_generation.hit_max_length


@needs_mlx
def test_page_budget_raises_repetitive_output_and_stops_generation(monkeypatch, tmp_path, tok):
    model = ScriptedModel(single=lambda: endless_text(marker=False))  # single은 <PAGE>를 내지 않는다
    eng = _engine(monkeypatch, model, tok, max_page_output_tokens=50)
    with pytest.raises(RepetitiveOutputError) as info:
        eng.run_single(_page_images(tmp_path, 1)[0], tmp_path / "out", RecSink(), threading.Event())
    assert not isinstance(info.value, OutputLimitError)
    gen = eng.last_generation
    assert gen.finish_reason == "stop"
    assert len(gen.token_ids) == 51  # 예산을 넘긴 토큰에서 바로 멈춘다
    assert model.decode_calls <= len(gen.token_ids)


@needs_mlx
def test_semantic_repetition_is_detected_like_the_torch_engine(monkeypatch, tmp_path, tok):
    """숫자만 다른 행이 반복되면(행 템플릿) 생성을 멈추고 RepetitiveOutputError."""
    rows = template_rows()
    model = ScriptedModel(multi=lambda pages: "<PAGE>" + rows)
    eng = _engine(monkeypatch, model, tok)
    with pytest.raises(RepetitiveOutputError, match="반복"):
        eng.run_multi(_page_images(tmp_path), tmp_path / "out", RecSink(), threading.Event())
    assert eng.last_generation.finish_reason == "stop"
    assert len(eng.last_generation.token_ids) < len(_enc(rows))


@needs_mlx
def test_sink_errors_propagate_and_release_the_device_cache(monkeypatch, tmp_path, tok):
    import mlx.core as mx

    calls = []
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(1))

    def boom(_text):
        raise RuntimeError("sink failed")

    sink = RecSink()
    sink.on_text = boom
    eng = _engine(monkeypatch, ScriptedModel(), tok)
    with pytest.raises(RuntimeError, match="sink failed"):
        eng.run_multi(_page_images(tmp_path), tmp_path / "out", sink, threading.Event())
    assert calls  # 실패해도 버퍼 캐시는 반환한다
    calls.clear()
    eng.run_single(_page_images(tmp_path, 1)[0], tmp_path / "ok", RecSink(), threading.Event())
    assert calls


@needs_mlx
def test_failure_before_generation_propagates_without_stale_stats(monkeypatch, tmp_path, tok):
    eng = _engine(monkeypatch, ScriptedModel(), tok)
    eng.run_single(_page_images(tmp_path, 1)[0], tmp_path / "ok", RecSink(), threading.Event())
    assert eng.last_generation is not None
    with pytest.raises(ValueError, match="이미지를 열 수 없습니다"):
        eng.run_single(tmp_path / "missing.png", tmp_path / "bad", RecSink(), threading.Event())
    assert eng.last_generation is None  # 이전 실행의 통계가 남지 않는다


@needs_mlx
def test_load_is_idempotent_under_concurrent_callers(monkeypatch, tok):
    loads, warmups = [], []
    model = ScriptedModel()

    def slow_load():
        loads.append(threading.get_ident())
        time.sleep(0.2)  # 두 번째 스레드가 로딩 도중에 들어오게
        return model, tok

    eng = UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx", preload_model=False))
    monkeypatch.setattr(eng, "_load_weights", slow_load)
    monkeypatch.setattr(eng, "_warmup", lambda m, t: warmups.append(1))
    errors: list[Exception] = []

    def _load():
        try:
            eng.load()
        except Exception as e:  # noqa: BLE001 — 스레드 안 예외를 본문으로 전달
            errors.append(e)

    threads = [threading.Thread(target=_load) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors and len(loads) == 1 and len(warmups) == 1
    assert eng.loaded and eng._eos_text == STOP_STR


# ── 작은 무작위 가중치 MLX 모델 (실제 비전·디코더 경로) ─────────────────

TINY_CONFIG = {
    "model_type": "unlimited-ocr",
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "topk_method": "greedy",
    "use_mla": False,
    "sliding_window_size": 8,
    "projector_config": {"projector_type": "linear", "input_dim": 256, "n_embed": 64},
    "vision_config": {
        "width": {
            "sam_vit_b": {
                "width": 32, "layers": 2, "heads": 2, "global_attn_indexes": [1],
                "downsample_channels": [64, 128],
            },
            "clip-l-14-224": {"width": 128, "layers": 1, "heads": 2, "image_size": 224, "patch_size": 14},
        }
    },
}


@pytest.fixture(scope="module")
def tiny_model():
    if not _MLX_OK:
        pytest.skip(_MLX_WHY)
    import mlx.core as mx

    from app.vendor.unlimited_ocr_mlx import Model, ModelConfig

    mx.random.seed(11)
    model = Model(ModelConfig.from_dict(TINY_CONFIG))
    # lm_head를 바이트 토큰 행(id 2..257)에만 연다 — 무작위 가중치가 테스트 토크나이저 어휘
    # 밖 id나 EOS를 내지 않는다(그 밖 행은 0 → 로짓 0, 바이트 행 최댓값은 양수).
    head = model.language_model.lm_head
    w = np.zeros(head.weight.shape, dtype=np.float32)
    w[2:258] = np.random.default_rng(5).normal(0, 4.0, (256, w.shape[1]))
    head.weight = mx.array(w)
    mx.eval(model.parameters())
    return model


def _tiny_engine(monkeypatch, model, tokenizer, **kw) -> UnlimitedMLXEngine:
    eng = UnlimitedMLXEngine(Settings(engine="unlimited", device="mlx", preload_model=False, **kw))
    monkeypatch.setattr(eng, "_load_weights", lambda: (model, tokenizer))
    return eng  # 워밍업은 실제로 돈다 (vendor warmup: 두 모드 짧은 생성)


def _tiny_run(eng, image: Path, out: Path):
    """실행 1회 → (생성 토큰, 스트림). 바이트만 내는 무작위 모델은 EOS가 없어 상한에서 끝난다."""
    sink = RecSink()
    with pytest.raises(OutputLimitError):
        eng.run_single(image, out, sink, threading.Event())
    return list(eng.last_generation.token_ids), sink.text


@needs_mlx
def test_tiny_model_streams_its_generated_tokens(monkeypatch, tmp_path, tiny_model, tok):
    eng = _tiny_engine(monkeypatch, tiny_model, tok, max_length=1600)
    image = _page_images(tmp_path, 1, size=(1280, 640))[0]  # 2x1 타일 gundam 경로 포함
    ids, streamed = _tiny_run(eng, image, tmp_path / "a")
    assert ids and all(2 <= i < 258 for i in ids)
    assert streamed == tok.decode(ids)  # EOS가 없으니 스트림 = 생성 원문 디코드
    assert eng.last_generation.prompt_length + len(ids) == 1600


@needs_mlx
def test_tiny_model_runs_identically_off_the_loading_thread(monkeypatch, tmp_path, tiny_model, tok):
    """프리로드 스레드에서 load()하고 워커 스레드에서 실행해도(동시 실행 포함) 같은 결과."""
    eng = _tiny_engine(monkeypatch, tiny_model, tok, max_length=1560)
    image = _page_images(tmp_path, 1, size=(1280, 640))[0]
    loader = threading.Thread(target=eng.load)
    loader.start()
    loader.join(timeout=120)
    assert eng.loaded

    expected = _tiny_run(eng, image, tmp_path / "main")
    results: dict[str, tuple] = {}
    errors: list[Exception] = []

    def work(name: str) -> None:
        try:
            results[name] = _tiny_run(eng, image, tmp_path / name)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    workers = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
    assert not errors
    assert results == {"w0": expected, "w1": expected}


@needs_mlx
def test_tiny_model_cancel_returns_promptly(monkeypatch, tmp_path, tiny_model, tok):
    eng = _tiny_engine(
        monkeypatch, tiny_model, tok, max_length=20000,
        max_page_output_chars=None, max_page_output_tokens=None,
    )
    eng.load()
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    t0 = time.perf_counter()
    try:
        eng.run_multi(_page_images(tmp_path, 1), tmp_path / "out", RecSink(), cancel)
    finally:
        timer.cancel()
    assert time.perf_counter() - t0 < 10
    gen = eng.last_generation
    assert gen.finish_reason == "stop" and gen.prompt_length + len(gen.token_ids) < 20000


# ── 앱 파이프라인 (create_app → 업로드 → done) ──────────────────────


@pytest.fixture
def mlx_app(monkeypatch, tmp_path, tok):
    """registry가 고른 실제 UnlimitedMLXEngine — 가중치 로드만 대본 모델로 바꾼다."""
    if not _MLX_OK:
        pytest.skip(_MLX_WHY)
    from fastapi.testclient import TestClient

    from app.main import create_app

    model = ScriptedModel()
    monkeypatch.setattr(UnlimitedMLXEngine, "_load_weights", lambda self: (model, tok))
    real_warmup = UnlimitedMLXEngine._warmup

    def warmup_then_forget(self, m, t):
        real_warmup(self, m, t)  # 실제 vendor 워밍업(두 모드 짧은 생성)을 거친다
        model.prompts.clear()  # 잡의 호출만 남긴다

    monkeypatch.setattr(UnlimitedMLXEngine, "_warmup", warmup_then_forget)

    def make(**kw):
        settings = Settings(
            engine="unlimited", device="mlx", data_dir=tmp_path / "data",
            frontend_dir=tmp_path / "no-frontend", preload_model=False, render_dpi=72, **kw,
        )
        return TestClient(create_app(settings)), model, settings

    return make


def _upload_and_wait(client, pdf: bytes) -> dict:
    from conftest import wait_done

    r = client.post(
        "/api/jobs", files={"file": ("doc.pdf", pdf, "application/pdf")}, data={"mode": "multi"}
    )
    assert r.status_code == 202, r.text
    return wait_done(client, r.json()["job_id"], timeout=60)


@needs_mlx
def test_pipeline_runs_a_job_to_done_on_the_mlx_engine(mlx_app):
    from conftest import make_pdf_bytes

    client, model, settings = mlx_app()
    with client:
        health = client.get("/api/health").json()
        assert (health["engine"], health["device"], health["dtype"]) == ("unlimited", "mlx", "bfloat16")
        assert health["capabilities"]["stream_granularity"] == "token"
        job = _upload_and_wait(client, make_pdf_bytes(pages=2))
        assert job["status"] == "done", job
        md = client.get(f"/api/jobs/{job['job_id']}/markdown").text
    assert model.prompts == [("multi", 2)]
    assert "Heading 1" in md and "Body 2 한글 中文 😀 text." in md
    assert md.count("![](images/") == 2  # 두 쪽의 figure가 병합 이미지로 연결됐다
    # raw_pages.json(P14) → layout.json: 쪽마다 title·image·text 블록
    layout = json.loads((settings.jobs_dir / job["job_id"] / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == [1, 2]
    for page in layout:
        assert {"title", "image", "text"} <= {b.get("type") for b in page["blocks"]}


@needs_mlx
def test_pipeline_recovers_a_truncated_chunk_page_by_page(mlx_app):
    """multi가 MAX_LENGTH에서 잘리면(OutputLimitError) 버리고 페이지별 single로 복구한다."""
    from conftest import make_pdf_bytes

    # 단일(gundam) 프롬프트는 72 dpi 쪽이 2×3 타일이라 921토큰 — single 대본은 상한 안에 든다
    client, model, _settings = mlx_app(max_length=1100)
    model._multi = lambda pages: endless_text()
    with client:
        job = _upload_and_wait(client, make_pdf_bytes(pages=2))
        md = client.get(f"/api/jobs/{job['job_id']}/markdown").text
    assert job["status"] == "done", job
    assert [m for m, _ in model.prompts] == ["multi", "single", "single"]
    # runner는 잘림(OutputLimitError)을 반복 감지와 구분해 'MAX_LENGTH 도달'로 알리고,
    # 엔진이 붙인 상세(MAX_LENGTH=1100 …)를 그대로 잇는다. 페이지별로 모두 살렸으니
    # 품질 저하(warnings)가 아니라 처리 경위(notices)다.
    assert any(
        "MAX_LENGTH 도달" in n and "페이지별 재처리" in n and "MAX_LENGTH=1100" in n
        for n in job["notices"]
    ), job["notices"]
    assert job["warnings"] == []
    assert md.count("Single page body recovered alone.") == 2
    assert "alpha" not in md and "omega" not in md  # 잘린 multi 출력은 채택하지 않았다


@needs_mlx
def test_pipeline_keeps_the_completed_pages_of_a_truncated_chunk(mlx_app):
    """multi가 2쪽 중간에서 잘리면 끝까지 생성된 1쪽은 multi 결과를 지키고 2쪽만 다시
    처리한다 — 엔진이 OutputLimitError.partial_output에 잘린 출력을 싣기 때문이다.
    (예전에는 실어 주지 않아 청크 전체를 페이지별로 다시 돌렸다.)"""
    from conftest import make_pdf_bytes

    # 충실도 게이트는 끈다 — 대본 본문이 원본 PDF 텍스트와 달라 1쪽을 재처리하게 된다
    client, model, settings = mlx_app(max_length=1100, ocr_fidelity_threshold=0.0)
    model._multi = lambda pages: page_text(0) + endless_text()
    with client:
        job = _upload_and_wait(client, make_pdf_bytes(pages=2))
        md = client.get(f"/api/jobs/{job['job_id']}/markdown").text
    assert job["status"] == "done", job
    assert [m for m, _ in model.prompts] == ["multi", "single"]  # 2쪽만 단독 재처리
    page1, page2 = md.split("\n\n---\n\n")
    assert "Heading 1" in page1 and "Body 1 한글 中文 😀 text." in page1
    assert "![](images/p0001_0.jpg)" in page1  # 살린 페이지의 multi 크롭도 그대로 병합됐다
    assert page2.strip() == "![](images/p0002_0.jpg)\n\nSingle page body recovered alone."
    assert "alpha" not in md and "omega" not in md  # 잘린 2쪽 multi 출력은 버렸다
    assert any("MAX_LENGTH 도달" in n and "앞 1쪽은 유지" in n for n in job["notices"]), job
    assert job["warnings"] == []
    layout = json.loads((settings.jobs_dir / job["job_id"] / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == [1, 2]


@needs_mlx
@pytest.mark.parametrize("pages", [2, 1], ids=["leading-marker-omitted", "no-marker-at-all"])
def test_pipeline_keeps_page_one_when_the_model_omits_the_leading_marker(mlx_app, pages):
    """[P23] 모델이 첫 <PAGE>를 생략해도(1쪽 청크면 마커 0개) 1쪽 내용·그림이 제자리에 남는다.

    예전 MLX 후처리는 첫 마커 앞을 버려 1쪽이 사라지고 뒤 페이지가 한 칸씩 당겨진 채
    (2쪽 청크는 '빈 페이지로 보정' 경고, 1쪽 청크는 경고 없이 빈 쪽) done으로 끝났다 —
    뒤 페이지 그림은 앞 페이지 래스터에서 잘렸다(audit mlx-1)."""
    from conftest import make_pdf_bytes

    # 충실도 게이트는 끈다 — 대본 본문이 원본 PDF 텍스트와 달라 재처리로 결과를 덮는다
    client, model, settings = mlx_app(ocr_fidelity_threshold=0.0)
    model._multi = lambda n: "".join(page_text(i) for i in range(n)).removeprefix("<PAGE>")
    with client:
        job = _upload_and_wait(client, make_pdf_bytes(pages=pages))
        md = client.get(f"/api/jobs/{job['job_id']}/markdown").text
    assert job["status"] == "done", job
    assert model.prompts == [("multi", pages)]  # 재처리 없이 multi 출력 그대로
    assert job["warnings"] == []
    parts = md.split("\n\n---\n\n")
    assert len(parts) == pages
    boxes = json.loads(
        (settings.jobs_dir / job["job_id"] / "images" / "boxes.json").read_text(encoding="utf-8")
    )
    for i, part in enumerate(parts):
        assert f"Heading {i + 1}" in part and f"Body {i + 1} 한글 中文 😀 text." in part
        name = f"p{i + 1:04d}_0.jpg"
        assert f"![](images/{name})" in part
        # 그 쪽 대본의 image 상자(page_text(i))로 잘렸다 — 다른 쪽 좌표가 아니다
        box = boxes[name]
        assert box["x2"] == int((500 + 5 * i) / 999 * box["image_width"])
