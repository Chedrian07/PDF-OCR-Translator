import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.pipeline.layout import (
    estimate_font_size_cqw,
    parse_page_blocks,
    render_layout_html,
    render_layout_standalone,
)
from app.pipeline.merge import ChunkResult, IncrementalMerger
from app.pipeline.render import text_with_math_html

# 실제 frontend 디렉터리 (repo/frontend) — layout-fit.js / KaTeX 자산 존재 확인용.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

RAW = (
    "<|det|>title [100, 50, 800, 100]<|/det|>문서 제목\n"
    "<|det|>text [100, 120, 900, 300]<|/det|>첫 단락 텍스트<|/ref|> 잔여토큰 포함\n"
    "<|det|>image [150, 320, 700, 600]<|/det|>\n"
    "<|det|>table [100, 620, 900, 800]<|/det|><table><tr><td>a</td></tr></table>"
)


def test_parse_page_blocks_document_order_and_types():
    blocks = parse_page_blocks(RAW)
    assert [b["type"] for b in blocks] == ["title", "text", "image", "table"]
    assert blocks[0]["bbox"] == [100, 50, 800, 100]
    assert blocks[0]["content"] == "문서 제목"
    assert "잔여토큰" in blocks[1]["content"] and "<|/ref|>" not in blocks[1]["content"]
    assert blocks[2]["crop_index"] == 0 and blocks[2]["content"] == ""
    assert "<table>" in blocks[3]["content"]


def test_parse_crop_index_matches_vendor_order():
    # 벤더 re_match: ref류 매치 전체가 det류보다 먼저 인덱싱된다 —
    # 문서상 det 이미지가 먼저 나와도 ref 이미지가 crop 0이어야 한다.
    raw = (
        "<|det|>image [10, 10, 100, 100]<|/det|>\n"
        "<|ref|>image<|/ref|><|det|>[[200, 200, 400, 400]]<|/det|>\n"
    )
    blocks = parse_page_blocks(raw)
    by_bbox = {tuple(b["bbox"]): b for b in blocks if b["type"] == "image"}
    assert by_bbox[(200, 200, 400, 400)]["crop_index"] == 0  # ref류 먼저
    assert by_bbox[(10, 10, 100, 100)]["crop_index"] == 1


def test_parse_ref_multibox_and_inner_det_dedup():
    raw = "<|ref|>image<|/ref|><|det|>[[10, 10, 50, 50], [60, 60, 90, 90]]<|/det|>"
    blocks = parse_page_blocks(raw)
    assert len(blocks) == 2  # 내부 det 태그가 중복 파싱되지 않음
    assert [b["crop_index"] for b in blocks] == [0, 1]


def test_render_layout_html_positions_and_escaping():
    pages = [{
        "page": 2, "width": 1000, "height": 1500,
        "blocks": [
            {"type": "title", "bbox": [0, 0, 999, 99], "content": "<script>x</script>"},
            {"type": "table", "bbox": [0, 100, 999, 300], "content": "<table><tr><td>a</td></tr></table>"},
            {"type": "image", "bbox": [100, 400, 600, 800], "content": "", "image": "p0002_0.jpg"},
            {"type": "bad type!", "bbox": [0, 0, 10, 10], "content": "x"},
            {"type": "text", "bbox": [1, 2, 3], "content": "무시됨"},  # 비정상 bbox
        ],
    }]
    html = render_layout_html(pages, "/api/jobs/j_x/files")
    assert 'data-page="2"' in html
    assert "padding-top:150.00%" in html  # 1500/1000
    assert "left:0.00%;top:0.00%" in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "<table><tr><td>a</td></tr></table>" in html  # 표는 화이트리스트 복원
    assert 'src="/api/jobs/j_x/files/images/p0002_0.jpg"' in html
    assert "layout-text" in html and "bad type!" not in html  # 타입 새니타이즈
    assert "무시됨" not in html


def test_text_with_math_html():
    out = text_with_math_html("질량은 \\( E = mc^2 \\) 이고 <b>태그</b>는 이스케이프.")
    assert '<span class="math-inline">E = mc^2</span>' in out
    assert "&lt;b&gt;" in out and "<b>" not in out
    out2 = text_with_math_html("\\[\nx^2 + y^2\n\\] 끝")
    assert '<span class="math-display">x^2 + y^2</span>' in out2
    assert text_with_math_html("수식 없음") == "수식 없음"


def test_layout_blocks_render_math_spans():
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [0, 0, 500, 100], "content": "본문 \\( a^2 \\) 수식"},
        {"type": "equation", "bbox": [0, 200, 900, 300], "content": "\\[ D = \\mathbb{E}[x] \\]"},
    ]}]
    html = render_layout_html(pages, "/b")
    assert '<span class="math-inline">a^2</span>' in html
    assert '<span class="math-display">D = \\mathbb{E}[x]</span>' in html
    assert "\\(" not in html and "\\[" not in html


# ── 면적 기반 폰트 크기 추정 (cqw) ─────────────────────────────────────
def test_estimate_font_size_calibration():
    # 진실 앵커(2504.19874v1.pdf 실측): 612×792pt 페이지 본문 10.9pt = 1.78cqw.
    # 그 박스는 원본 타이포로 ~1180 ASCII자를 담는다 → 재현 케이스
    # (bbox(60,100,940,280)·A4 비율·1180자)에서 fs가 1.6–2.0cqw여야 한다.
    fs = estimate_font_size_cqw((60, 100, 940, 280), "x" * 1180, 1.414)
    assert fs is not None
    assert 1.6 <= fs <= 2.0, fs


def test_estimate_cjk_smaller_than_ascii():
    # 같은 글자수라도 CJK는 가중치(1.0)가 ASCII(0.5)보다 커서 더 작은 fs가 나온다.
    box = (60, 100, 940, 280)
    ascii_fs = estimate_font_size_cqw(box, "x" * 300, 1.414)
    cjk_fs = estimate_font_size_cqw(box, "가" * 300, 1.414)
    assert ascii_fs is not None and cjk_fs is not None
    assert cjk_fs < ascii_fs


def test_estimate_single_line_title_cap():
    # 얕은 박스의 짧은 제목 — 면적 모델은 크게 잡지만 단일 줄 상한(h/1.25)이 눌러야 함.
    bbox = (100, 50, 900, 75)
    fs = estimate_font_size_cqw(bbox, "Title", 1.414)
    h = (75 - 50) / 999 * 100 * 1.414
    cap = h / 1.25
    assert fs is not None
    assert abs(fs - cap) < 0.02, (fs, cap)
    assert cap < 3.6  # 클램프가 아니라 '상한'이 작동함을 보장


def test_estimate_clamps_hold_at_extremes():
    # 상한: 큰 박스 + 극소 글자수 → 3.6 클램프
    hi = estimate_font_size_cqw((100, 100, 200, 900), "xx", 1.414)
    assert hi == 3.6
    # 하한: 큰 박스 + 초대량 글자수 → 0.8 클램프
    lo = estimate_font_size_cqw((0, 0, 999, 999), "가" * 50000, 1.414)
    assert lo == 0.8


def test_estimate_empty_and_none_safe():
    box = (60, 100, 940, 280)
    assert estimate_font_size_cqw(box, "", 1.414) is None
    assert estimate_font_size_cqw(box, "   ", 1.414) is None
    assert estimate_font_size_cqw(box, "<table></table>", 1.414) is None  # 태그만 → 빈 텍스트
    assert estimate_font_size_cqw(None, "본문", 1.414) is None
    assert estimate_font_size_cqw((1, 2, 3), "본문", 1.414) is None  # 비정상 bbox


def test_render_layout_html_font_size_cqw_text_not_image():
    pages = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 텍스트 예시 " * 30},
        {"type": "image", "bbox": [100, 400, 600, 800], "content": "", "image": "p0001_0.jpg"},
    ]}]
    html = render_layout_html(pages, "/b")
    text_div = re.search(r'<div class="layout-block layout-text"[^>]*>', html).group(0)
    assert "font-size:" in text_div and "cqw" in text_div
    assert "line-height:1.22" in text_div
    # 이미지 블록엔 폰트 크기 인라인이 없어야 함
    img_tag = re.search(r'<img class="layout-block layout-image"[^>]*>', html).group(0)
    assert "cqw" not in img_tag and "font-size:" not in img_tag


def test_render_layout_html_fs_precedence():
    # 실측 block["fs"]가 있으면 휴리스틱을 무시하고 그 값을 그대로 쓴다([0.6,6.0] 클램프).
    pages = [{"page": 1, "width": 1000, "height": 1414, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 " * 40, "fs": 1.78},
        {"type": "text", "bbox": [60, 300, 940, 480], "content": "굵게 " * 40, "fs": 2.5, "bold": True},
        {"type": "title", "bbox": [60, 500, 940, 560], "content": "제목", "fs": 3.0, "bold": True},
        {"type": "text", "bbox": [60, 600, 940, 700], "content": "과대", "fs": 99.0},
        {"type": "text", "bbox": [60, 720, 940, 900], "content": "폴백 텍스트 예시 " * 30},  # fs 없음
    ]}]
    html = render_layout_html(pages, "/b")
    divs = re.findall(r'<div class="layout-block layout-\w+"[^>]*style="([^"]*)"', html)
    assert "font-size:1.78cqw;line-height:1.22;" in divs[0]
    assert "font-weight:600;" not in divs[0]  # bold 아님
    # 볼드 실측 블록: 굵게 (제목 아님)
    assert "font-size:2.50cqw;line-height:1.22;" in divs[1] and "font-weight:600;" in divs[1]
    # 제목은 실측 bold라도 font-weight 인라인 안 함 (CSS가 이미 굵게)
    assert "font-size:3.00cqw;line-height:1.22;" in divs[2] and "font-weight:600;" not in divs[2]
    # 클램프 상한 6.0
    assert "font-size:6.00cqw;line-height:1.22;" in divs[3]
    # fs 없는 블록은 휴리스틱 폴백 — cqw는 있되 1.78/2.50 등 실측값과 다름
    assert "cqw" in divs[4] and "font-size:1.78cqw" not in divs[4]


def test_layout_standalone_includes_fitter_after_typeset(tmp_path):
    (tmp_path / "images").mkdir()
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "본문 텍스트 " * 40},
    ]}]
    html = render_layout_standalone(pages, tmp_path, "문서", FRONTEND_DIR)
    assert "window.uocrFitLayout" in html          # fitter 정의 인라인됨
    assert "uocrFitLayout(document)" in html        # 문서 전체에 호출
    assert html.index("window.uocrFitLayout") < html.index("uocrFitLayout(document)")
    assert "cqw" in html                            # 서버가 심은 면적 기반 폰트 크기
    assert "white-space: pre-wrap" in html          # 문서 타이포(줄바꿈 보존)
    assert "container-type: inline-size" in html    # cqw 기준 컨테이너
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        # 순서: KaTeX 타이포셋 → uocrFitLayout(document)
        assert html.index("katex.render") < html.index("uocrFitLayout(document)")


def test_layout_standalone_self_contained(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 900, 80], "content": "제목 \\( x \\)"},
        {"type": "image", "bbox": [100, 100, 800, 600], "content": "", "image": "p0001_0.jpg"},
        {"type": "image", "bbox": [100, 700, 300, 900], "content": "", "image": "missing.jpg"},
    ]}]
    html = render_layout_standalone(pages, tmp_path, "테스트 문서", FRONTEND_DIR)
    assert html.startswith("<!doctype html>")
    assert "<title>테스트 문서</title>" in html
    assert "data:image/jpeg;base64," in html          # 크롭 인라인
    assert 'src="data:,"' in html                      # 결측 크롭 폴백
    assert '<span class="math-inline">x</span>' in html
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        assert "katex" in html and "data:font/woff2;base64," in html  # KaTeX 자립 인라인
    # 외부 참조 없음 (자립성)
    assert 'src="http' not in html and 'href="http' not in html


def test_layout_standalone_facsimile_uses_full_page_and_transparent_text(tmp_path):
    (tmp_path / "images").mkdir()
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0001.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "title", "bbox": [100, 50, 900, 90], "content": "How to Read a Paper"},
        {"type": "image", "bbox": [100, 100, 500, 500], "image": "crop.jpg", "content": ""},
    ]}]

    html = render_layout_standalone(
        pages,
        tmp_path,
        "HowtoReadPaper",
        FRONTEND_DIR,
        pages_dir=pages_dir,
        facsimile=True,
    )
    assert "data:image/png;base64," in html
    assert "layout-page-image" in html and "facsimile-canvas" in html
    assert "facsimile-text-block" in html
    assert "How to Read a Paper" in html
    assert "crop.jpg" not in html  # 완성 페이지에 이미 포함 — 중복 크롭 금지
    assert '<h1 class="facsimile-document-title">HowtoReadPaper</h1>' in html


def test_document_standalone_self_contained(tmp_path):
    """문서 뷰 standalone(document.html) — figure_only 엔진에서도 동작하는 내보내기.
    이미지 base64 인라인·결측 폴백·서버 참조 제거·제목 이스케이프·lang 부여."""
    from app.pipeline.layout import render_document_standalone

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    inner = (
        '<section class="doc-page" data-page="1">\n'
        "<h1>제목</h1>\n<p>본문 문단</p>\n"
        '<p><img src="/api/jobs/j_x/files/images/p0001_0.jpg" alt="" '
        'style="width:31.7%;height:auto;"></p>\n'
        '<p><img src="/api/jobs/j_x/files/images/missing.jpg" alt=""></p>\n'
        '<p><span class="math-inline">\\tau^{2}</span></p>\n</section>'
    )
    html = render_document_standalone(inner, tmp_path, "테스트 <문서>", FRONTEND_DIR)
    assert html.startswith("<!doctype html>")
    assert "<title>테스트 &lt;문서&gt;</title>" in html   # 제목 이스케이프
    assert "data:image/jpeg;base64," in html               # 이미지 인라인
    assert 'src="data:,"' in html                          # 결측 이미지 폴백
    assert "/api/jobs/" not in html                        # 서버 참조 없는 완전 자립
    assert 'style="width:31.7%;height:auto;"' in html      # figure 상대폭 스타일 보존
    assert '<span class="math-inline">\\tau^{2}</span>' in html
    if (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        assert "katex" in html and "data:font/woff2;base64," in html
    assert 'src="http' not in html and 'href="http' not in html
    # lang: 원본엔 없음 / ko엔 html·main 모두 부여 (keep-all 타이포 스코프)
    assert "<html>" in html and "<main>" in html
    ko = render_document_standalone(inner, tmp_path, "문서", FRONTEND_DIR, lang="ko")
    assert '<html lang="ko">' in ko and '<main lang="ko">' in ko


def test_document_standalone_no_traversal(tmp_path):
    """이미지 파일명 캡처는 슬래시 배제 — `../` 류는 매치 자체가 안 돼 원문 유지
    (존재하지 않는 서버 경로로 남을 뿐, 파일 시스템 접근 없음)."""
    from app.pipeline.layout import render_document_standalone

    (tmp_path / "images").mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    inner = '<img src="/api/jobs/j_x/files/images/../secret.txt" alt="">'
    html = render_document_standalone(inner, tmp_path, "t", None)
    assert "secret" not in html.replace("secret.txt", "")  # 파일 내용 미포함
    assert "data:image/jpeg;base64," not in html


def test_vertical_blocks_render_writing_mode_class():
    pages = [{"page": 1, "width": 612, "height": 792, "blocks": [
        {"type": "text", "bbox": [10, 300, 40, 900], "content": "arXiv:1908.07836v1 [cs.CL]",
         "fs": 1.47, "vertical": "up"},
        {"type": "text", "bbox": [60, 100, 940, 280], "content": "일반 본문", "fs": 1.78},
    ]}]
    html = render_layout_html(pages, "/b")
    assert "layout-vertical-up" in html
    # 일반 블록에는 세로 클래스가 붙지 않는다
    assert html.count("layout-vertical-") == 1


def test_vertical_geometric_fallback_without_text_layer():
    # 텍스트 레이어 없는(fs 미주입) 스캔 PDF: 극단적으로 좁고 긴 텍스트 박스는 세로 간주
    tall = {"type": "text", "bbox": [10, 200, 40, 900], "content": "arXiv:1908.07836v1 [cs.CL] 16 Aug 2019"}
    normal = {"type": "text", "bbox": [60, 100, 940, 280], "content": "일반 본문 " * 20}
    html = render_layout_html([{"page": 1, "width": 612, "height": 792, "blocks": [tall, normal]}], "/b")
    assert "layout-vertical-up" in html
    assert html.count("layout-vertical-") == 1
    # fs가 실측된 좁은 박스는 (세로 플래그 없이는) 폴백을 타지 않는다
    html2 = render_layout_html([{"page": 1, "width": 612, "height": 792, "blocks": [
        {**tall, "fs": 1.47},
    ]}], "/b")
    assert "layout-vertical-" not in html2


def test_merge_ingests_layout_json(tmp_path):
    (tmp_path / "pages").mkdir()
    m = IncrementalMerger(tmp_path, "\n\n---\n\n")
    c = tmp_path / "work" / "chunk_00"
    (c / "images").mkdir(parents=True)
    (c / "images" / "page_0_0.jpg").write_bytes(b"jpg")
    (c / "raw_pages.json").write_text(json.dumps({"pages": [RAW]}), encoding="utf-8")
    m.add_chunk(ChunkResult(c, 5, 1, "<PAGE>\n본문 ![](images/page_0_0.jpg)"))

    saved = json.loads((tmp_path / "layout.json").read_text(encoding="utf-8"))
    assert saved[0]["page"] == 5
    img_blocks = [b for b in saved[0]["blocks"] if b["type"] == "image"]
    assert img_blocks[0]["image"] == "p0005_0.jpg"  # 글로벌 이미지명 매핑
    assert "crop_index" not in img_blocks[0]


def test_merge_without_raw_pages_is_fine(tmp_path):
    m = IncrementalMerger(tmp_path, "\n\n---\n\n")
    c = tmp_path / "work" / "chunk_00"
    c.mkdir(parents=True)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\n텍스트만"))
    assert not (tmp_path / "layout.json").exists()


def test_merge_recovery_page_kept_in_layout(tmp_path):
    """raw_pages.json이 없는 복구 페이지(텍스트 레이어 폴백)도 layout.json에 남는다.

    빠지면 facsimile document.html 내보내기에서 그 페이지가 통째로 사라지고
    layout 페이지 번호와 result.md 페이지 인덱스가 어긋난다."""
    (tmp_path / "pages").mkdir()
    m = IncrementalMerger(tmp_path, "\n\n---\n\n")
    c0 = tmp_path / "work" / "chunk_00"
    (c0 / "images").mkdir(parents=True)
    (c0 / "raw_pages.json").write_text(json.dumps({"pages": [RAW]}), encoding="utf-8")
    m.add_chunk(ChunkResult(c0, 1, 1, "<PAGE>\n1쪽"))
    # 2쪽: single OCR 실패 후 PDF 내장 텍스트로 복구 — 산출물이 없는 빈 디렉터리
    c1 = tmp_path / "work" / "recover_02"
    c1.mkdir(parents=True)
    m.add_chunk(ChunkResult(c1, 2, 1, "2쪽 복구 텍스트", single=True))
    c2 = tmp_path / "work" / "chunk_02"
    (c2 / "images").mkdir(parents=True)
    (c2 / "raw_pages.json").write_text(json.dumps({"pages": [RAW]}), encoding="utf-8")
    m.add_chunk(ChunkResult(c2, 3, 1, "<PAGE>\n3쪽"))
    out = m.finalize()

    saved = json.loads((tmp_path / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in saved] == [1, 2, 3]
    assert saved[1]["blocks"] == []                          # 좌표는 없지만 페이지는 존재
    # (b) layout 페이지 N ↔ result.md split 인덱스 N 이 1:1
    assert len(saved) == len(out.split("\n\n---\n\n")) == len(m.pages_md)


def test_ref_block_matches_vendor_without_length_caps():
    """벤더 re_match(`(.*?)`)가 잡는 긴 det 페이로드/라벨을 우리도 잡아야 한다.

    놓치면 블록이 통째로 사라지고 좌표 문자열이 본문으로 새며, image 블록이면
    벤더가 저장한 크롭과 crop_index가 어긋난다."""
    boxes = ", ".join(f"[{i}, {i}, {i + 5}, {i + 5}]" for i in range(30))  # 400자 초과
    raw = (
        f"<|ref|>image<|/ref|><|det|>[{boxes}]<|/det|>\n"
        "<|det|>text [100, 120, 900, 300]<|/det|>본문"
    )
    blocks = parse_page_blocks(raw)
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == 30
    assert [b["crop_index"] for b in image_blocks] == list(range(30))
    text_block = [b for b in blocks if b["type"] == "text"][0]
    assert text_block["content"] == "본문"          # 좌표 문자열이 본문으로 새지 않음

    long_label = "figure_caption_" + "x" * 60      # 40자 상한 초과 라벨
    blocks2 = parse_page_blocks(f"<|ref|>{long_label}<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>내용")
    assert len(blocks2) == 1 and blocks2[0]["bbox"] == [1, 2, 3, 4]


def test_ref_block_does_not_swallow_next_block_on_missing_close():
    """닫는 태그가 빠져도 다음 ref 블록까지 삼키지 않는다(tempered-dot 경계)."""
    raw = (
        "<|ref|>image<|/ref|><|det|>[[1, 2, 3, 4]]\n"     # <|/det|> 결손
        "<|ref|>text<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>본문"
    )
    blocks = parse_page_blocks(raw)
    assert [b["type"] for b in blocks] == ["text"]
    assert blocks[0]["bbox"] == [10, 20, 30, 40]


def test_facsimile_falls_back_when_page_images_too_large(tmp_path, monkeypatch):
    """페이지 PNG 총량이 상한을 넘으면 base64 인라인을 포기하고 좌표 렌더로 폴백한다
    (전 페이지를 메모리에 쌓다 워커가 OOM 나는 것을 막는다)."""
    from app.pipeline import layout as layout_mod

    (tmp_path / "images").mkdir()
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for n in (1, 2):
        (pages_dir / f"page_{n:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 4096)
    pages = [
        {"page": n, "width": 612, "height": 792, "blocks": [
            {"type": "text", "bbox": [100, 50, 900, 90], "content": f"{n}쪽 제목"},
        ]}
        for n in (1, 2)
    ]
    monkeypatch.setattr(layout_mod, "_FACSIMILE_MAX_TOTAL_BYTES", 4096)

    html = layout_mod.render_layout_standalone(
        pages, tmp_path, "큰문서", None, pages_dir=pages_dir, facsimile=True,
    )
    assert "data:image/png;base64," not in html          # 페이지 PNG 인라인 없음
    assert 'class="layout-canvas facsimile-canvas"' not in html  # 백지 페이지 방지
    assert "1쪽 제목" in html and "2쪽 제목" in html      # 내용은 보존

    # 상한 안이면 그대로 facsimile
    monkeypatch.setattr(layout_mod, "_FACSIMILE_MAX_TOTAL_BYTES", 64 * 1024 * 1024)
    ok = layout_mod.render_layout_standalone(
        pages, tmp_path, "큰문서", None, pages_dir=pages_dir, facsimile=True,
    )
    assert "data:image/png;base64," in ok
    assert 'class="layout-canvas facsimile-canvas"' in ok


def test_degenerate_huge_coordinates_do_not_fail_the_page():
    """퇴화한 det 좌표(수천 자리 숫자)가 와도 parse_page_blocks는 예외를 던지지 않는다.

    int()의 문자열 변환 상한(4300자리)을 넘는 숫자는 ValueError를 냈고, merge가 이를
    막지 않아 이미 끝난 앞 청크까지 포함한 잡 전체가 error로 끝났다. 벤더도 같은
    페이로드를 literal_eval 실패로 건너뛰므로 그 박스는 블록 없이 버린다.
    """
    raw = (
        "<|det|>text [100, 120, 900, 300]<|/det|>정상 본문\n"
        "<|det|>table [" + "7" * 4400 + ", 1, 2, 3]<|/det|>퇴화 좌표\n"
        "<|det|>title [10, 20, 900, 60]<|/det|>뒤 제목"
    )
    blocks = parse_page_blocks(raw)
    assert [(b["type"], b["content"]) for b in blocks] == [
        ("text", "정상 본문"), ("title", "뒤 제목"),
    ]


def test_absurd_image_box_keeps_vendor_crop_numbering():
    """쓸 수 없는 image 박스는 블록으로 만들지 않되 crop 번호는 센다.

    벤더는 literal_eval에 성공한 페이로드의 박스마다 크롭 번호를 쓴다(좌표가 이상해도 —
    clamp 뒤 퇴화한 상자도). 박스를 지우며 번호까지 당기면 뒤 그림이 엉뚱한 크롭 파일을
    가리킨다. literal_eval이 실패하는 페이로드는 크롭 파일은 없지만 벤더 P22가 번호 1개를
    소비한다(마크다운 참조와 정렬).
    """
    raw = (
        "<|ref|>image<|/ref|><|det|>"
        "[[1, 2, 3, 4], [5, " + "9" * 20 + ", 7, 8], [10, 20, 30, 40]]<|/det|>\n"
        "<|ref|>image<|/ref|><|det|>[[0, " + "9" * 4400 + ", 1, 2]]<|/det|>\n"
        "<|det|>image [50, 50, 60, 60]<|/det|>"
    )
    blocks = parse_page_blocks(raw)
    assert [(b["bbox"], b["crop_index"]) for b in blocks] == [
        ([1, 2, 3, 4], 0),
        ([10, 20, 30, 40], 2),
        ([50, 50, 60, 60], 4),
    ]


_GOOD_IMAGE_DET = "<|det|>image [100, 450, 800, 900]<|/det|>"
# 쓸 수 없는 image 매치 — 실제 모델의 inline 문법과 ref 문법 (tests/test_mlx_postprocess.py와 같은 표)
_UNUSABLE_IMAGE_DETS = {
    "missing-comma": "<|det|>image [100, 120 500, 420]<|/det|>",  # literal_eval 실패
    "three-coords": "<|det|>image [100, 120, 500]<|/det|>",
    "five-coords": "<|det|>image [100, 120, 500, 600, 700]<|/det|>",
    "inf": "<|det|>image [1e400, 0, 500, 500]<|/det|>",
    "float-overflow": "<|det|>image [" + "9" * 400 + ", 0, 500, 500]<|/det|>",
    "int-digit-limit": "<|det|>image [" + "9" * 5000 + ", 0, 500, 500]<|/det|>",
    "bool": "<|det|>image [True, 0, 500, 500]<|/det|>",  # bool은 좌표가 아니다
    "inverted": "<|det|>image [500, 500, 100, 100]<|/det|>",
    "flat-strings": "<|det|>image ['x', 0, 500, 500]<|/det|>",  # 기형 평평 목록 — 상자 하나
    "ref-non-literal": "<|ref|>image<|/ref|><|det|>[[a, b, c, d]]<|/det|>",
    "ref-three-coords": "<|ref|>image<|/ref|><|det|>[[0, 0, 999]]<|/det|>",
    "ref-name-coord": "<|ref|>image<|/ref|><|det|>[[0, 0, abc, 1]]<|/det|>",
    "ref-empty-list": "<|ref|>image<|/ref|><|det|>[]<|/det|>",  # 상자 목록이 아닌 값
    "ref-string-literal": "<|ref|>image<|/ref|><|det|>'abcd'<|/det|>",
    "ref-none": "<|ref|>image<|/ref|><|det|>None<|/det|>",
    "ref-zero": "<|ref|>image<|/ref|><|det|>0<|/det|>",
}


@pytest.mark.parametrize("bad", list(_UNUSABLE_IMAGE_DETS.values()), ids=list(_UNUSABLE_IMAGE_DETS))
def test_unusable_image_payload_takes_exactly_one_vendor_crop_number(bad):
    """쓸 수 없는 image 매치 하나는 벤더 P22처럼 크롭 번호 정확히 1개 — 뒤 그림은 1번 파일.

    예전 `_quads`는 숫자를 4개씩 묶어 세어, 좌표 3개·bool·비리터럴·빈 목록·문자열 뒤 그림을
    0번(앞 그림 파일)으로, 쉼표 누락·좌표 5개·inf·뒤집힌 상자는 벤더가 저장하지 않은 크롭의
    블록으로 냈다(감사 mlx-2·torch-2 — layout_vs_vendor 재현)."""
    blocks = parse_page_blocks(f"{bad}\nfig A\n{_GOOD_IMAGE_DET}\nfig B")
    assert [(b["bbox"], b["crop_index"]) for b in blocks if "crop_index" in b] == [
        ([100, 450, 800, 900], 1),
    ]


def test_image_bbox_is_clamped_like_the_vendor_crop():
    """음수·999 초과 좌표는 벤더 crop처럼 0–999로 자른다 — 예전에는 부호를 버려 -50이 50이 됐다."""
    blocks = parse_page_blocks(
        "<|det|>image [-50, 100, 500, 600]<|/det|>\n<|det|>image [0, 0, 1200, 999]<|/det|>"
    )
    assert [(b["bbox"], b["crop_index"]) for b in blocks] == [
        ([0, 100, 500, 600], 0),
        ([0, 0, 999, 999], 1),
    ]


def _random_vendor_page(rng) -> str:
    """라벨·문법·좌표가 뒤섞인 한 페이지 원문 — 정상·경계 밖·퇴화·해석 불가가 섞인다."""
    odd = ["-50", "1200", "1e400", "9" * 400, "9" * 5000, "True", "'x'", "a", "None", "2.5"]

    def flat(k):
        if k == 4 and rng.random() < 0.6:  # 정상 상자
            x1, y1 = rng.randrange(0, 800), rng.randrange(0, 800)
            atoms = [str(v) for v in (x1, y1, x1 + rng.randrange(1, 400), y1 + rng.randrange(1, 400))]
        else:  # 이상 원소가 섞인 좌표
            atoms = [rng.choice(odd) if rng.random() < 0.3 else str(rng.randrange(0, 1000)) for _ in range(k)]
        sep = ", " if rng.random() < 0.9 else " "  # 가끔 쉼표 누락
        return "[" + sep.join(atoms) + "]"

    blocks = []
    for _ in range(rng.randrange(1, 7)):
        label = rng.choice(["image", "image", "image", "text", "title", "table"])
        k = rng.choice([4, 4, 4, 3, 5, 0])
        if rng.random() < 0.5:
            blocks.append(f"<|det|>{label} {flat(k)}<|/det|>caption {rng.randrange(100)}")
        else:
            boxes = ", ".join(flat(rng.choice([4, 4, 3])) for _ in range(rng.randrange(0, 3)))
            payload = rng.choice([f"[{boxes}]", flat(k), "[[a, b, c, d]]", "'abcd'", "", "None", "0"])
            blocks.append(f"<|ref|>{label}<|/ref|><|det|>{payload}<|/det|>body {rng.randrange(100)}")
    return "\n".join(blocks)


def test_crop_indices_name_the_files_the_vendor_saves(tmp_path):
    """무작위 페이지에서 parse_page_blocks의 crop_index가 벤더가 실제로 저장한 크롭 파일과 같다.

    torch 없는 MLX 후처리(torch 벤더 P22와 바이트 동일 — tests/test_mlx_postprocess.py)의
    draw_bounding_boxes로 대조한다. 정수 좌표만 있는 페이지는 집합이 같아야 하고, 소수 좌표가
    섞이면 레이아웃이 픽셀 퇴화를 모르므로 '레이아웃 ⊆ 저장 파일'만 요구한다(없는 파일을
    가리키지 않는다)."""
    import random

    from PIL import Image

    from app.vendor.unlimited_ocr_mlx import postprocess as vendor

    rng = random.Random(2222)
    image = Image.new("RGB", (1000, 1000), (200, 210, 220))  # 999px 이상 — 정수 상자는 퇴화하지 않는다
    for i in range(300):
        text = _random_vendor_page(rng)
        out = tmp_path / f"p{i}"
        (out / "images").mkdir(parents=True)
        vendor.draw_bounding_boxes(image, vendor.re_match(text)[0], str(out))
        saved = sorted(int(f.stem) for f in (out / "images").glob("*.jpg"))
        emitted = sorted(b["crop_index"] for b in parse_page_blocks(text) if "crop_index" in b)
        assert set(emitted) <= set(saved), text
        if "2.5" not in text:
            assert emitted == saved, text


# ── standalone 내려받기 파일의 CSP·KaTeX 옵션 ─────────────────────────────────
class _HeadScan(HTMLParser):
    """standalone HTML의 meta·자원 참조를 순서대로 모은다 — CSP 대조용."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.order: list[str] = []          # 'csp' · 'referrer' · 'style' · 'script' (등장 순)
        self.csp: list[str] = []
        self.referrer: list[str] = []
        self.img_src: list[str] = []
        self.loaders: list[str] = []        # link·iframe·object·embed·source 등 외부 로더
        self.styles: list[str] = []
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and (a.get("http-equiv") or "").lower() == "content-security-policy":
            self.order.append("csp")
            self.csp.append(a.get("content") or "")
        elif tag == "meta" and (a.get("name") or "").lower() == "referrer":
            self.order.append("referrer")
            self.referrer.append(a.get("content") or "")
        elif tag in ("style", "script"):
            self.order.append(tag)
            self._in_style = tag == "style"
        elif tag == "img":
            self.img_src.append(a.get("src") or "")
        elif tag in ("link", "iframe", "object", "embed", "source", "video", "audio"):
            self.loaders.append(tag)

    def handle_data(self, data):
        if self._in_style:
            self.styles.append(data)

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False


def _scan(html: str) -> _HeadScan:
    scan = _HeadScan()
    scan.feed(html)
    return scan


def _csp_directives(policy: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


def _standalone_samples(tmp_path):
    """두 standalone 렌더러의 대표 출력 — 크롭·페이지 이미지·수식·KaTeX 인라인 포함."""
    from app.pipeline.layout import render_document_standalone

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p0001_0.jpg").write_bytes(b"\xff\xd8fakejpg")
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "page_0001.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 900, 80], "content": "제목 \\( x \\)"},
        {"type": "image", "bbox": [100, 100, 800, 600], "content": "", "image": "p0001_0.jpg"},
    ]}]
    inner = (
        '<section class="doc-page" data-page="1"><p>본문</p>'
        '<p><img src="/api/jobs/j_x/files/images/p0001_0.jpg" alt=""></p>'
        '<p><span class="math-inline">\\tau^{2}</span></p></section>'
    )
    return {
        "layout": render_layout_standalone(pages, tmp_path, "문서", FRONTEND_DIR, lang="ko"),
        "facsimile": render_layout_standalone(
            pages, tmp_path, "문서", FRONTEND_DIR, pages_dir=pages_dir, facsimile=True,
        ),
        "layout-no-assets": render_layout_standalone(pages, tmp_path, "문서", None),
        "document": render_document_standalone(inner, tmp_path, "문서", FRONTEND_DIR, lang="ko"),
        "document-no-assets": render_document_standalone(inner, tmp_path, "문서", None),
    }


def test_standalone_exports_lock_out_external_origins_with_meta_csp(tmp_path):
    """디스크에서 여는 내려받기 파일은 서버 CSP 헤더를 받지 못한다 — 업로드 PDF 본문이
    파일을 연 사람의 IP·열람 사실을 바깥 주소로 흘리지 못하게 meta CSP로 모든 바깥 출처를
    막는다. 필요한 자원은 전부 data:·인라인이므로 렌더는 그대로다(Chromium 실측: 수식·
    크롭·페이지 이미지 정상, 주입한 바깥 이미지는 img-src 위반으로 차단)."""
    from app.pipeline.layout import STANDALONE_CSP

    assert _csp_directives(STANDALONE_CSP) == {
        "default-src": ["'none'"],
        "img-src": ["data:", "blob:"],
        "font-src": ["data:"],
        "style-src": ["'unsafe-inline'"],
        "script-src": ["'unsafe-inline'"],
    }
    katex_vendored = (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file()
    for name, html in _standalone_samples(tmp_path).items():
        scan = _scan(html)
        assert scan.csp == [STANDALONE_CSP], name
        assert scan.referrer == ["no-referrer"], name            # 링크 클릭에 출처 주소를 싣지 않는다
        # meta CSP는 그 뒤에 오는 <style>·<script>에만 적용된다 — 맨 앞이어야 한다
        assert scan.order[:2] == ["csp", "referrer"], (name, scan.order)
        # 정책이 허용하는 자원만 쓴다: 이미지는 data:, 폰트(CSS url())는 data:, 외부 로더 없음
        assert scan.img_src and all(src.startswith("data:") for src in scan.img_src), name
        urls = re.findall(r"url\(\s*['\"]?([^'\")]+)", "".join(scan.styles))
        assert all(url.startswith("data:") for url in urls), (name, urls[:3])
        assert scan.loaders == [], name
        if katex_vendored and not name.endswith("no-assets"):
            # KaTeX 스크립트·woff2 폰트도 정책 안(인라인·data:)에서 실린다
            assert "script" in scan.order and "data:font/woff2;base64," in html, name


def test_standalone_typesets_math_through_the_shared_size_guard(tmp_path):
    """내려받기 파일의 수식도 앱과 같은 가드(frontend/katex-guard.js)를 거쳐 조판한다.

    옵션(maxSize·maxExpand·trust·strict)·크기 인자 묶기·조판 결과 상한은 그 클래식 스크립트
    하나에 있고 frontend/tests/katex-guard.test.mjs가 앱 구현(constants.katexOptions·
    core.clampTexSizes·katexStyleOversized)과 대조한다. 예전 인라인 조판은 KaTeX를 직접 불러
    \\raisebox{-4000em}{x} 하나로 문단이 7만 px가 됐다(감사 frontend-4 — Chromium 실측)."""
    from app.pipeline.layout import _TYPESET_JS, render_document_standalone

    assert "katex.render" not in _TYPESET_JS  # 가드를 건너뛰는 직접 호출 금지
    assert "uocrKatexGuard.typesetMath(document)" in _TYPESET_JS
    guard = (FRONTEND_DIR / "katex-guard.js").read_text(encoding="utf-8")
    assert "root.uocrKatexGuard = {" in guard
    if not (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        return
    pages = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [0, 0, 900, 80], "content": "\\( x \\)"},
    ]}]
    katex_js = (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").read_text(encoding="utf-8")
    for html in (
        render_layout_standalone(pages, tmp_path, "t", FRONTEND_DIR),
        render_document_standalone("<p>x</p>", tmp_path, "t", FRONTEND_DIR),
    ):
        katex_at = html.index(f"<script>{katex_js}</script>")
        guard_at = html.index(f"<script>{guard}</script>")
        typeset_at = html.index(_TYPESET_JS)
        assert katex_at < guard_at < typeset_at  # KaTeX 번들 → 가드 → DOMContentLoaded 조판
        assert "katex.render(" not in html[typeset_at:]  # 가드 뒤에 직접 조판하는 스크립트가 없다


def test_standalone_without_the_guard_leaves_math_as_tex(tmp_path):
    """가드 파일이 없으면 상한 없는 조판 대신 KaTeX를 아예 싣지 않는다 — 수식은 원문 LaTeX."""
    import shutil

    from app.pipeline.layout import _katex_inline_bundle

    if not (FRONTEND_DIR / "vendor" / "katex" / "katex.min.js").is_file():
        return
    partial = tmp_path / "frontend"
    shutil.copytree(FRONTEND_DIR / "vendor", partial / "vendor")
    assert _katex_inline_bundle(str(partial)) == ""
    shutil.copy(FRONTEND_DIR / "katex-guard.js", partial / "katex-guard.js")
    _katex_inline_bundle.cache_clear()
    assert "uocrKatexGuard" in _katex_inline_bundle(str(partial))
    _katex_inline_bundle.cache_clear()
