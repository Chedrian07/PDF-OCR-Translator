"""폰트 서브셋이 내보내기 PDF를 작게 만들되 조판을 바꾸지 않는지 고정한다.

`pdf_export/subset.py`는 임베드 폰트를 실제로 쓰는 글리프만 남겨 다운로드
크기를 20배 이상 줄인다. 그 대가로 **조판이 달라질 수 있는** 위험을 지므로,
여기서 동치 기준을 실행 가능한 형태로 못박는다.

동치 = 리포트(replaced/kept/relocated·경고)가 같고, 공백을 무시한 추출
텍스트가 같고, 추출 텍스트에 U+0000이 없다. 재조판된 블록 안에서 줄바꿈이
한 글자 움직이는 것은 동치로 본다(subset.py 모듈 문서 참조).
"""

import json

import pytest

from app.pipeline.pdf_export import build_translated_pdf
from app.pipeline.pdf_export import build as build_mod
from app.pipeline.pdf_export.subset import (
    drawable_charset,
    subset_font,
    subset_font_files,
)
from app.pipeline.pdf_export.fonts import _resolve_font
from app.pipeline.pdf import quiet_fitz

pytest.importorskip("fontTools", reason="fontTools 없으면 전체 폰트 임베드로 폴백한다")

KO_TEXT = "한국어 번역 텍스트 블록입니다"
# 서브셋이 실제로 일어나려면 후보 폰트가 _SUBSET_MIN_BYTES(2MB)를 넘어야 한다.
_MIN_FONT_BYTES = 2 * 1024 * 1024


def _korean_font() -> str:
    fontfile, _name = _resolve_font("")
    if not fontfile:
        pytest.skip("한글 폰트가 없는 환경 — 서브셋 경로가 동작하지 않는다")
    from pathlib import Path

    if Path(fontfile).stat().st_size < _MIN_FONT_BYTES:
        pytest.skip("후보 폰트가 서브셋 최소 크기 미만이라 서브셋을 건너뛴다")
    return fontfile


# ── 문자 집합 닫힘 ────────────────────────────────────────────────────────
def test_charset_covers_pipeline_generated_characters():
    """레이아웃에 없지만 조판기가 만들어내는 문자까지 집합에 들어와야 한다.

    U+00A0은 `_protect_trailing_words`가 직접 넣는다. 빠지면 화면은 같은데
    추출 텍스트에 U+0000이 박혀 복사·검색이 깨진다(실측 회귀).
    """
    charset = drawable_charset([{"blocks": [{"content": "가나다"}]}])
    assert " " in charset, "NBSP가 빠지면 줄바꿈 이음쇠가 NUL로 추출된다"
    for char in "가나다":
        assert char in charset
    # 치환표의 치역 — LaTeX 명령·첨자·기호 폴백이 만들어내는 글자.
    for char in ("α", "×", "⁰", "₁", "-"):
        assert char in charset
    # 조판에 쓰이지 않는 제어문자는 넣지 않는다.
    for char in ("\n", "\r", "\t"):
        assert char not in charset


def test_charset_is_deterministic():
    payload = [{"blocks": [{"content": "나가다"}]}]
    assert drawable_charset(payload) == drawable_charset(payload)


# ── 서브셋 파일 ───────────────────────────────────────────────────────────
def test_subset_font_shrinks_and_preserves_glyph_coverage(tmp_path):
    fontfile = _korean_font()
    charset = drawable_charset([{"blocks": [{"content": KO_TEXT}]}])
    out = subset_font(fontfile, charset, tmp_path)
    assert out is not None, "서브셋이 조용히 포기하면 다운로드 크기가 그대로다"

    from pathlib import Path

    assert Path(out).stat().st_size < Path(fontfile).stat().st_size

    fitz = quiet_fitz()
    original = fitz.Font(fontfile=fontfile)
    subset = fitz.Font(fontfile=out)
    for char in charset:
        assert bool(original.has_glyph(ord(char))) == bool(subset.has_glyph(ord(char))), (
            f"글리프 유무가 달라지면 조판기가 다른 글자를 찍는다: {char!r}"
        )
        # 폭이 달라지면 줄바꿈·맞춤 계산이 통째로 어긋난다.
        assert original.glyph_advance(ord(char)) == pytest.approx(
            subset.glyph_advance(ord(char))
        )


def test_subset_font_files_reuses_one_subset_per_source(tmp_path):
    fontfile = _korean_font()
    charset = drawable_charset([{"blocks": [{"content": KO_TEXT}]}])
    mapping = subset_font_files([fontfile, fontfile, None], charset, tmp_path)
    assert list(mapping) == [fontfile], "같은 폰트를 두 번 서브셋하면 빌드가 느려진다"


def test_subset_font_skips_small_fonts(tmp_path):
    small = tmp_path / "tiny.ttf"
    small.write_bytes(b"not really a font")
    assert subset_font(str(small), "abc", tmp_path) is None


def test_subset_font_survives_a_corrupt_font(tmp_path):
    """폰트가 깨져 있어도 내보내기는 전체 폰트 경로로 계속돼야 한다."""
    broken = tmp_path / "broken.ttf"
    broken.write_bytes(b"\x00" * (_MIN_FONT_BYTES + 1))
    assert subset_font(str(broken), "abc", tmp_path) is None


def test_missing_fonttools_falls_back_to_full_font(tmp_path, monkeypatch):
    """fontTools가 없는 배포에서도 내보내기는 예전처럼 동작한다."""
    import builtins

    real_import = builtins.__import__

    def _no_fonttools(name, *args, **kwargs):
        if name.startswith("fontTools"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_fonttools)
    assert subset_font(_korean_font(), "abc", tmp_path) is None


# ── 빌드 동치 ─────────────────────────────────────────────────────────────
# 줄바꿈이 실제로 일어나고 `_protect_trailing_words`가 U+00A0을 넣을 만큼 길게.
# 짧은 한 줄짜리 블록으로는 NBSP 회귀가 재현되지 않는다.
_LONG_KO = (
    "본 기술 보고서는 양자화된 대규모 언어 모델의 정확도를 회복하는 학습 방법을 "
    "제안하며, 다단계 사후 학습 파이프라인을 거친 모델에서도 안정적으로 동작함을 "
    "여러 벤치마크에서 확인하였다."
)
_SRC_EN = (
    "This technical report presents a training method that recovers the accuracy "
    "of quantized large language models across several benchmarks."
)


def _subset_job(tmp_path, name: str):
    """source.pdf의 글리프 위치와 레이아웃 bbox가 맞는 잡 디렉터리를 만든다."""
    fitz = quiet_fitz()
    job_dir = tmp_path / name
    job_dir.mkdir()
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(60, 85, 535, 300), _SRC_EN, fontsize=11)
    doc.save(job_dir / "source.pdf")
    doc.close()

    blocks = [{
        "type": "text", "bbox": [100, 100, 900, 355],
        "content": _SRC_EN, "fs": 2.0,
    }]
    orig = [
        {"page": n, "width": 1000, "height": 1414, "blocks": json.loads(json.dumps(blocks))}
        for n in (1, 2)
    ]
    trans = json.loads(json.dumps(orig))
    for page in trans:
        page["blocks"][0]["content"] = _LONG_KO
    (job_dir / "layout.json").write_text(json.dumps(orig), encoding="utf-8")
    (job_dir / "layout.ko.json").write_text(
        json.dumps(trans, ensure_ascii=False), encoding="utf-8")
    (job_dir / "result.ko.md").write_text(_LONG_KO, encoding="utf-8")
    return job_dir


def _build_text_and_report(job_dir, *, subsetting: bool, monkeypatch):
    if not subsetting:
        monkeypatch.setattr(
            build_mod, "_subset_export_fonts",
            lambda fonts, charset, out_dir: fonts,
        )
    result = build_translated_pdf(job_dir, "ko", fontfile="")
    fitz = quiet_fitz()
    doc = fitz.open(result.path)
    try:
        text = "\n".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()
    return text, result.report(), result.path.stat().st_size


def test_subset_build_matches_full_font_build(tmp_path, monkeypatch):
    """서브셋 빌드와 전체 폰트 빌드가 같은 문서를 만드는지 — 핵심 회귀 방지."""
    _korean_font()
    full_dir = _subset_job(tmp_path, "full")
    subset_dir = _subset_job(tmp_path, "subset")

    with monkeypatch.context() as m:
        full_text, full_report, full_size = _build_text_and_report(
            full_dir, subsetting=False, monkeypatch=m,
        )
    subset_text, subset_report, subset_size = _build_text_and_report(
        subset_dir, subsetting=True, monkeypatch=monkeypatch,
    )

    assert "\x00" not in subset_text, (
        "추출 텍스트의 NUL은 서브셋이 그려진 글리프의 유니코드 매핑을 잃었다는 뜻"
    )
    assert "".join(subset_text.split()) == "".join(full_text.split()), (
        "공백을 무시한 본문이 달라지면 서브셋이 다른 글자를 찍은 것"
    )
    assert subset_report == full_report, (
        "블록 교체·보존·이동 결정이 폰트 서브셋으로 달라지면 안 된다"
    )
    assert subset_size < full_size, "서브셋이 파일을 줄이지 못하면 의미가 없다"
