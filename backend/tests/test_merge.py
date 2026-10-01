import json
from pathlib import Path

from app.pipeline.merge import ChunkResult, IncrementalMerger, split_pages

SEP = "\n\n---\n\n"


def _touch(p: Path, data: bytes = b"jpg") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _mk_multi_chunk(root: Path, name: str, num_pages: int, images_per_page: int = 1) -> Path:
    d = root / "work" / name
    for i in range(num_pages):
        for k in range(images_per_page):
            _touch(d / "images" / f"page_{i}_{k}.jpg")
        _touch(d / f"result_with_boxes_{i}.jpg")
    return d


def test_split_pages():
    assert split_pages("<PAGE>\nA\n<PAGE>\nB") == ["A", "B"]
    assert split_pages("A only") == ["A only"]
    assert split_pages("") == []
    assert split_pages("<PAGE>") == [""]


def test_multi_chunk_renumbering(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c0 = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    m.add_chunk(ChunkResult(c0, 1, 2, "<PAGE>\nA ![](images/page_0_0.jpg)\n<PAGE>\nB ![](images/page_1_0.jpg)"))
    c1 = _mk_multi_chunk(tmp_path, "chunk_01", 1)
    m.add_chunk(ChunkResult(c1, 3, 1, "<PAGE>\nC ![](images/page_0_0.jpg)"))
    out = m.finalize()

    assert "A ![](images/p0001_0.jpg)" in out
    assert "B ![](images/p0002_0.jpg)" in out
    assert "C ![](images/p0003_0.jpg)" in out
    assert out.count("---") == 2
    for name in ("p0001_0.jpg", "p0002_0.jpg", "p0003_0.jpg"):
        assert (tmp_path / "images" / name).is_file()
    for name in ("page_0001.jpg", "page_0002.jpg", "page_0003.jpg"):
        assert (tmp_path / "layout" / name).is_file()
    assert (tmp_path / "result.md").read_text(encoding="utf-8") == out
    assert m.warnings == []


def test_marker_count_mismatch_pads_and_warns(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(c, 1, 3, "<PAGE>\nonly one page"))
    assert len(m.pages_md) == 3
    assert m.pages_md[1] == "" and m.pages_md[2] == ""
    assert len(m.warnings) == 1


def test_marker_count_excess_merges_tail(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 1)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\nA\n<PAGE>\nB\n<PAGE>\nC"))
    assert len(m.pages_md) == 1
    assert "A" in m.pages_md[0] and "C" in m.pages_md[0]
    assert len(m.warnings) == 1


def test_single_mode_merge(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    d = tmp_path / "work" / "chunk_04"
    _touch(d / "images" / "0.jpg")
    _touch(d / "images" / "1.jpg")
    _touch(d / "result_with_boxes.jpg")
    md = "P5 ![](images/0.jpg) and ![](images/1.jpg)"
    m.add_chunk(ChunkResult(d, 5, 1, md, single=True))
    out = m.finalize()
    assert "![](images/p0005_0.jpg)" in out and "![](images/p0005_1.jpg)" in out
    assert (tmp_path / "images" / "p0005_0.jpg").is_file()
    assert (tmp_path / "images" / "p0005_1.jpg").is_file()
    assert (tmp_path / "layout" / "page_0005.jpg").is_file()


def test_figure_boxes_merged_with_global_names(tmp_path):
    import json

    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    (c / "boxes.json").write_text(json.dumps({
        "page_0_0.jpg": {"x1": 10, "y1": 20, "x2": 410, "y2": 320, "image_width": 1000, "image_height": 1400},
        "page_1_0.jpg": {"x1": 0, "y1": 0, "x2": 950, "y2": 500, "image_width": 1000, "image_height": 1400},
    }), encoding="utf-8")
    m.add_chunk(ChunkResult(c, 3, 2, "<PAGE>\n![](images/page_0_0.jpg)\n<PAGE>\n![](images/page_1_0.jpg)"))

    saved = json.loads((tmp_path / "images" / "boxes.json").read_text(encoding="utf-8"))
    assert saved["p0003_0.jpg"]["x2"] == 410
    assert saved["p0004_0.jpg"]["image_width"] == 1000
    assert m.figure_boxes == saved


def test_missing_boxes_json_is_fine(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 1)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\n![](images/page_0_0.jpg)"))
    assert not (tmp_path / "images" / "boxes.json").exists()
    assert m.figure_boxes == {}


def test_blank_edge_pages_keep_separators(tmp_path):
    """선두/말미 빈 페이지(스캔 문서의 빈 표지 등)가 result.md에서 사라지지 않는다.

    전체 strip()이 구분자의 공백 절반을 먹으면 페이지 수 계약(N페이지 =
    구분자 N-1개)이 깨져 Q&A·번역·/html 문서 뷰의 페이지 인덱스가 밀린다."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(c, 1, 3, "<PAGE>\n\n<PAGE>\np2text\n<PAGE>\n"))
    out = m.finalize()

    assert m.warnings == []                       # 마커 수는 정확 — 보정 경고 없음
    assert [p.strip() for p in out.split(SEP)] == ["", "p2text", ""]
    assert (tmp_path / "result.md").read_text(encoding="utf-8") == out


def test_page_body_hr_does_not_shift_page_boundaries(tmp_path):
    """페이지 본문의 `---` 줄(OCR 각주선)이 페이지 경계로 오인되지 않는다.

    무해화가 없으면 result.md의 split 인덱스가 밀려 Q&A·문서 뷰가 경고 없이
    엉뚱한 페이지를 보게 된다 (N페이지 = 구분자 N-1개 계약 위반)."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(
        c, 1, 3,
        "<PAGE>\n1쪽 본문\n\n---\n\n각주 내용\n"      # 본문 안 구분선
        "<PAGE>\n2쪽 본문\n\n---\n"                    # 페이지 끝 구분선
        "<PAGE>\n3쪽 본문",
    ))
    out = m.finalize()

    assert m.warnings == []
    segments = out.split(SEP)
    assert len(segments) == 3                       # (a) 분할 수 == 실제 페이지 수
    assert "1쪽 본문" in segments[0] and "각주 내용" in segments[0]
    assert "2쪽 본문" in segments[1]
    assert segments[2].strip() == "3쪽 본문"
    assert "***" in segments[0]                     # 구분선은 동등한 마크다운 수평선으로
    assert "---" not in segments[0]


def test_setext_heading_underline_is_not_neutralized(tmp_path):
    """setext 제목 밑줄(`제목` 바로 다음 줄의 `---`)은 구분자와 충돌할 수 없다 —
    빈 줄 패딩이 없으므로 그대로 둬야 제목이 문단+수평선으로 깨지지 않는다."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    m.add_chunk(ChunkResult(c, 1, 2, "<PAGE>\n제목\n---\n본문\n<PAGE>\n2쪽"))
    out = m.finalize()

    assert m.warnings == []
    segments = out.split(SEP)
    assert len(segments) == 2
    assert "제목\n---\n본문" in segments[0]


def test_neutralization_skipped_for_non_hr_separator(tmp_path):
    """구분선이 아닌 커스텀 구분자도 리터럴 일치만 깨고 텍스트는 보존한다."""
    sep = "\n\n@@PAGE@@\n\n"
    m = IncrementalMerger(tmp_path, sep)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    m.add_chunk(ChunkResult(c, 1, 2, "<PAGE>\nA\n\n@@PAGE@@\n\ntail\n<PAGE>\nB"))
    out = m.finalize()

    assert m.warnings == []
    segments = out.split(sep)
    assert len(segments) == 2
    assert "A" in segments[0] and "tail" in segments[0] and "@@PAGE@@" in segments[0]


def test_special_tokens_stripped(tmp_path):
    m = IncrementalMerger(tmp_path, SEP)
    d = tmp_path / "work" / "chunk_00"
    d.mkdir(parents=True)
    m.add_chunk(ChunkResult(d, 1, 1, "<PAGE>\nkeep <|ref|>text<|/ref|><|det|>[[1,2,3,4]]<|/det|> this"))
    out = m.finalize()
    assert "<|ref|>" not in out and "<|det|>" not in out
    assert "keep" in out and "this" in out


def test_code_fence_hr_is_not_rewritten_but_still_breaks_separator(tmp_path):
    """코드펜스 안의 `---`는 문자 치환(`***`) 대상이 아니다 — 펜스 안은 렌더 결과가
    곧 원문이라 YAML 문서 구분자·구분선 예제가 조용히 깨진다(포터빌리티 계약).
    동시에 페이지 경계 불변식(N페이지 = 구분자 N-1개)은 그대로 지켜야 한다."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 2)
    m.add_chunk(ChunkResult(
        c, 1, 2,
        "<PAGE>\n설명:\n\n```yaml\nname: x\n\n---\n\nname: y\n```\n\n본문 각주선\n\n---\n\n각주\n"
        "<PAGE>\n2쪽 본문",
    ))
    out = m.finalize()

    assert m.warnings == []
    segments = out.split(SEP)
    assert len(segments) == 2                      # (a) 페이지 경계 불변식 유지
    assert "***" not in segments[0].split("```")[1]  # (b) 펜스 안은 문자 변조 없음
    assert "\n--- \n" in segments[0]               # 펜스 안은 후행 공백으로만 무해화
    assert "***" in segments[0].split("```")[2]    # (c) 펜스 밖 각주선은 기존대로 치환
    assert "2쪽 본문" in segments[1]


def test_code_fence_close_reenables_neutralization(tmp_path):
    """펜스가 닫힌 뒤의 `---`는 다시 일반 구분선 무해화 경로를 탄다 (상태 누수 방지)."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 1)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\n~~~\ncode\n~~~\n\n---\n\ntail"))
    out = m.finalize()

    assert m.warnings == []
    assert len(out.split(SEP)) == 1
    assert "***" in out


def test_render_warnings_are_inherited_as_job_warnings(tmp_path):
    """렌더 단계의 흰 페이지 대체 고지를 잡 경고로 승계한다 — 승계하지 않으면
    빈 결과가 quality.state='ok'로 남아 정상 변환으로 오인된다."""
    import json

    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    (pages / "render_warnings.json").write_text(
        json.dumps(["2/3페이지 렌더에 실패해 흰 페이지로 대체했습니다 (1, 3)"]),
        encoding="utf-8",
    )
    m = IncrementalMerger(tmp_path, SEP)
    assert m.warnings == ["2/3페이지 렌더에 실패해 흰 페이지로 대체했습니다 (1, 3)"]

    c = _mk_multi_chunk(tmp_path, "chunk_00", 1)
    m.add_chunk(ChunkResult(c, 1, 1, "<PAGE>\n본문"))
    m.finalize()
    assert m.warnings[0].startswith("2/3페이지 렌더에 실패")


def test_render_warnings_absent_or_broken_is_silent(tmp_path):
    """파일이 없거나 깨졌으면 경고 없이 진행한다 (부가 채널이 잡을 못 죽이게)."""
    assert IncrementalMerger(tmp_path, SEP).warnings == []
    pages = tmp_path / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "render_warnings.json").write_text("{not json", encoding="utf-8")
    assert IncrementalMerger(tmp_path, SEP).warnings == []


def test_excess_pages_do_not_invade_the_next_chunk_namespace(tmp_path):
    """마커 초과 생성분의 이미지가 다음 청크의 글로벌 페이지 이름을 침범하지 않는다.

    실측: 46페이지 잡의 1청크가 8페이지에 마커 9개를 냈다. 초과분 로컬 8이
    `start_page + 8` = 다음 청크의 첫 페이지로 계산돼 같은 파일명을 쓰고, 다음
    청크가 그대로 덮어써 그 페이지에 엉뚱한 그림이 표시됐다."""
    m = IncrementalMerger(tmp_path, SEP)
    # 2페이지 청크인데 모델이 3페이지를 냈다 (로컬 2가 초과분)
    c0 = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(
        c0, 1, 2,
        "<PAGE>\nA ![](images/page_0_0.jpg)"
        "\n<PAGE>\nB ![](images/page_1_0.jpg)"
        "\n<PAGE>\nC ![](images/page_2_0.jpg)",
    ))
    c1 = _mk_multi_chunk(tmp_path, "chunk_01", 1)
    m.add_chunk(ChunkResult(c1, 3, 1, "<PAGE>\nD ![](images/page_0_0.jpg)", ))
    out = m.finalize()

    assert len(m.pages_md) == 3, "result.md 페이지 수는 청크 계약대로 2+1"
    # 초과분(C)은 마지막 페이지(2)로 접히고, 파일명도 접두사로 구분된다
    assert "C ![](images/p0002_x2_0.jpg)" in out
    assert (tmp_path / "images" / "p0002_x2_0.jpg").is_file()
    # 3페이지는 온전히 다음 청크의 것이다 (덮어쓰기 없음)
    assert "D ![](images/p0003_0.jpg)" in out
    assert (tmp_path / "images" / "p0003_0.jpg").read_bytes() == b"jpg"
    # 초과분 레이아웃 오버레이는 마지막 페이지의 오버레이를 덮지 않는다
    assert (tmp_path / "layout" / "page_0002.jpg").is_file()
    assert (tmp_path / "layout" / "page_0003.jpg").is_file()
    assert not (tmp_path / "layout" / "page_0004.jpg").exists()


# ── 모델 페이지 → 물리 페이지 정합 ────────────────────────────────────────

def test_align_model_pages_handles_a_split_in_the_middle():
    """모델이 3페이지 중 2페이지를 둘로 쪼개면 그 뒤가 밀린다 — 위치로 잡아야 한다.

    실측: 46p 논문에서 layout 6개 페이지가 한 칸씩 밀려 PDF 23쪽의 프로젝트명
    29개가 리댁션으로 삭제되고 24쪽 캡션이 그 자리에 찍혔다."""
    from app.pipeline.merge import align_model_pages

    page_texts = [
        "alphaalphaalphaalphaalphaalphaalphaalphaalphaalpha",
        "bravobravobravobravobravobravobravobravobravobravo",
        "charliecharliecharliecharliecharliecharliecharlie",
    ]
    # 모델이 2페이지를 앞뒤로 쪼갰다 → 4개
    model = [page_texts[0], page_texts[1][:25], page_texts[1][25:], page_texts[2]]
    mapping = align_model_pages(model, page_texts)
    assert mapping[0] == 0
    assert mapping[3] == 2, f"마지막 페이지가 밀렸다: {mapping}"
    assert all(m in (None, 1) for m in mapping[1:3]), mapping


def test_align_model_pages_handles_a_skipped_page():
    """모델이 가운데 페이지를 통째로 건너뛰어도 뒤 페이지는 제자리를 찾는다."""
    from app.pipeline.merge import align_model_pages

    page_texts = [
        "alphaalphaalphaalphaalphaalphaalphaalphaalphaalpha",
        "bravobravobravobravobravobravobravobravobravobravo",
        "charliecharliecharliecharliecharliecharliecharlie",
    ]
    mapping = align_model_pages([page_texts[0], page_texts[2]], page_texts)
    assert mapping == [0, 2], mapping


def test_align_model_pages_is_a_noop_when_it_cannot_corroborate():
    """대조할 근거가 없으면(스캔 PDF 등) 아무 것도 매칭하지 않는다 — 호출자는
    기존 위치 기반 동작으로 안전하게 되돌아간다."""
    from app.pipeline.merge import align_model_pages

    assert align_model_pages(["짧다", "짧다"], ["", ""]) == [None, None]
    assert align_model_pages([], ["abc"]) == []


def test_place_by_alignment_never_drops_content():
    """매칭되지 않은 모델 페이지는 버리지 않고 앞 페이지에 붙인다."""
    from app.pipeline.merge import place_by_alignment

    out = place_by_alignment(["A", "extra", "B"], [0, None, 1], 2, "\n")
    assert out == ["A\nextra", "B"], out
    # 물리 페이지 수는 정확히 지킨다 (result.md 페이지 계약)
    assert len(place_by_alignment(["A"], [0], 3, "\n")) == 3


def test_misaligned_chunk_is_repositioned_against_the_source_pdf(tmp_path):
    """실제 정합 경로 — 모델이 가운데 페이지를 건너뛰어도 layout/markdown 모두
    올바른 물리 페이지에 놓인다."""
    import fitz

    doc = fitz.open()
    marks = ["ALPHAPAGEONE", "BRAVOPAGETWO", "CHARLIEPAGETHREE"]
    for m in marks:
        pg = doc.new_page()
        # 실제 페이지 분량의 텍스트 레이어 — 정합은 원본 본문과의 대조로 이뤄진다
        for line in range(12):
            pg.insert_text((60, 120 + line * 18), " ".join([m] * 5), fontsize=11)
    job = tmp_path / "job"
    job.mkdir()
    doc.save(str(job / "source.pdf"))
    doc.close()

    m = IncrementalMerger(job, SEP)
    c = _mk_multi_chunk(job, "chunk_00", 3)
    # 모델이 2페이지를 건너뛰었다: 마커 2개 (기대 3)
    body = [" ".join([marks[0]] * 60), " ".join([marks[2]] * 60)]
    (c / "raw_pages.json").write_text(
        json.dumps({"pages": [f"<|det|>text [0,0,999,99]<|/det|>{b}" for b in body]}),
        encoding="utf-8",
    )
    m.add_chunk(ChunkResult(c, 1, 3, f"<PAGE>\n{body[0]}\n<PAGE>\n{body[1]}"))
    m.finalize()

    assert len(m.pages_md) == 3
    assert marks[0] in m.pages_md[0]
    assert m.pages_md[1] == "", f"건너뛴 페이지가 비어 있어야 한다: {m.pages_md[1][:40]!r}"
    assert marks[2] in m.pages_md[2], "3페이지 내용이 2페이지 자리로 밀렸다"

    layout = json.loads((job / "layout.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in layout] == [1, 2, 3]
    assert marks[0] in layout[0]["blocks"][0]["content"]
    assert layout[1]["blocks"] == []
    assert marks[2] in layout[2]["blocks"][0]["content"], "layout이 한 칸 밀렸다"


def _color_of(path) -> str:
    """JPEG 손실을 감안해 가장 강한 채널 이름으로 색을 판별한다."""
    from PIL import Image

    with Image.open(path) as im:
        r, g, b = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
    return max((r, "red"), (g, "green"), (b, "blue"))[1]


def test_realigned_pages_name_and_crop_figures_by_the_physical_slot(tmp_path):
    """정렬이 페이지를 옮기면 그림은 **물리 슬롯**의 이름·래스터를 따라야 한다.

    예전 기대값(모델 인덱스로 이름 짓기)은 감사에서 결함으로 확인된 동작이었다: 모델이
    2쪽을 건너뛰면 3쪽 그림이 `p0002_0.jpg`로 저장되고, 충실도 게이트가 빈 2쪽을 단독
    재처리해 교체(replace_page → `p0002_*` 삭제)하는 순간 3쪽 그림이 지워지고 2쪽 그림으로
    덮였다. 게다가 벤더는 그 그림을 **입력 래스터 2(2쪽)**에서 잘랐다 — 3쪽 래스터에서
    다시 잘라야 한다.
    """
    import fitz
    from PIL import Image

    job = tmp_path / "job"
    (job / "pages").mkdir(parents=True)
    marks = ["ALPHAPAGEONE", "BRAVOPAGETWO", "CHARLIEPAGETHREE"]
    colors = [(220, 30, 30), (30, 200, 30), (30, 30, 220)]
    names = ["red", "green", "blue"]
    doc = fitz.open()
    for i, mark in enumerate(marks):
        pg = doc.new_page()
        for line in range(12):
            pg.insert_text((60, 120 + line * 18), " ".join([mark] * 5), fontsize=11)
        Image.new("RGB", (300, 400), colors[i]).save(job / "pages" / f"page_{i + 1:04d}.png")
    doc.save(str(job / "source.pdf"))
    doc.close()

    m = IncrementalMerger(job, SEP)
    chunk_dir = job / "work" / "chunk_00"
    (chunk_dir / "images").mkdir(parents=True)
    # 모델이 2쪽을 건너뛰었다(마커 2개). 벤더는 모델 페이지 k의 그림을 입력 래스터 k에서
    # 잘랐다 — 모델 1(=3쪽 내용)의 크롭은 2쪽(초록) 래스터에서 나왔다.
    Image.new("RGB", (60, 60), colors[0]).save(chunk_dir / "images" / "page_0_0.jpg")
    Image.new("RGB", (60, 60), colors[1]).save(chunk_dir / "images" / "page_1_0.jpg")
    for k in (0, 1):
        Image.new("RGB", (300, 400), colors[k]).save(chunk_dir / f"result_with_boxes_{k}.jpg")
    body = [" ".join([marks[0]] * 60), " ".join([marks[2]] * 60)]
    raw = [f"<|det|>text [0,0,999,99]<|/det|>{b}\n<|det|>image [100, 300, 900, 900]<|/det|>"
           for b in body]
    (chunk_dir / "raw_pages.json").write_text(json.dumps({"pages": raw}), encoding="utf-8")
    m.add_chunk(ChunkResult(
        chunk_dir, 1, 3,
        f"<PAGE>\n{body[0]}\n![](images/page_0_0.jpg)\n<PAGE>\n{body[1]}\n![](images/page_1_0.jpg)",
    ))

    assert "![](images/p0001_0.jpg)" in m.pages_md[0]
    assert m.pages_md[1] == ""
    assert "![](images/p0003_0.jpg)" in m.pages_md[2]
    on_disk = {f.name for f in (job / "images").iterdir()} - {"boxes.json"}
    assert on_disk == {"p0001_0.jpg", "p0003_0.jpg"}            # 2쪽 이름의 그림은 없다
    assert _color_of(job / "images" / "p0001_0.jpg") == names[0]
    assert _color_of(job / "images" / "p0003_0.jpg") == names[2]  # 3쪽 래스터에서 다시 잘랐다
    assert m.figure_boxes["p0003_0.jpg"]["image_width"] == 300
    layout = json.loads((job / "layout.json").read_text(encoding="utf-8"))
    referenced = {b["image"] for p in layout for b in p["blocks"] if b.get("image")}
    assert referenced == on_disk
    # 엉뚱한 래스터에 그려진 오버레이는 옮겨진 페이지의 것으로 쓰지 않는다
    assert (job / "layout" / "page_0001.jpg").is_file()
    assert not (job / "layout" / "page_0003.jpg").exists()

    # 게이트가 빈 2쪽을 단독 재처리해 채택해도 3쪽 그림은 그대로다
    single = job / "work" / "fidelity" / "page_0002"
    (single / "images").mkdir(parents=True)
    Image.new("RGB", (60, 60), colors[1]).save(single / "images" / "0.jpg")
    (single / "raw_pages.json").write_text(json.dumps({"pages": [
        f"<|det|>text [0,0,999,99]<|/det|>{' '.join([marks[1]] * 60)}\n"
        "<|det|>image [100, 300, 900, 900]<|/det|>"
    ]}), encoding="utf-8")
    assert m.replace_page(2, ChunkResult(single, 2, 1, "B\n![](images/0.jpg)", single=True))
    assert _color_of(job / "images" / "p0002_0.jpg") == names[1]
    assert _color_of(job / "images" / "p0003_0.jpg") == names[2]
    assert "![](images/p0003_0.jpg)" in m.pages_md[2]


def test_two_model_pages_in_one_slot_get_distinct_crop_names(tmp_path):
    """한 슬롯에 모델 페이지 둘이 합쳐지면(쪼개진 페이지) 크롭 번호가 둘 다 0부터라 이름이
    겹친다 — 두 번째부터 `x{k}_` 접두사로 구분하고 layout도 모델 페이지별로 매긴다."""
    import fitz

    job = tmp_path / "job"
    job.mkdir()
    marks = ["ALPHAPAGEONE", "BRAVOPAGETWO"]
    doc = fitz.open()
    for mark in marks:
        pg = doc.new_page()
        for line in range(12):
            pg.insert_text((60, 120 + line * 18), " ".join([mark] * 5), fontsize=11)
    doc.save(str(job / "source.pdf"))
    doc.close()

    m = IncrementalMerger(job, SEP)
    c = _mk_multi_chunk(job, "chunk_00", 3)  # page_0_0, page_1_0, page_2_0
    two = " ".join([marks[1]] * 60)
    model = [" ".join([marks[0]] * 60), two[: len(two) // 2], two[len(two) // 2 :]]
    raw = [f"<|det|>text [0,0,999,99]<|/det|>{t}\n<|det|>image [100, 300, 900, 900]<|/det|>"
           for t in model]
    (c / "raw_pages.json").write_text(json.dumps({"pages": raw}), encoding="utf-8")
    m.add_chunk(ChunkResult(
        c, 1, 2,
        "<PAGE>\n" + "\n<PAGE>\n".join(
            f"{t}\n![](images/page_{k}_0.jpg)" for k, t in enumerate(model)
        ),
    ))

    on_disk = {f.name for f in (job / "images").iterdir()} - {"boxes.json"}
    layout = json.loads((job / "layout.json").read_text(encoding="utf-8"))
    referenced = {b["image"] for p in layout for b in p["blocks"] if b.get("image")}
    assert on_disk == referenced == {"p0001_0.jpg", "p0002_0.jpg", "p0002_x2_0.jpg"}
    assert "![](images/p0002_0.jpg)" in m.pages_md[1]
    assert "![](images/p0002_x2_0.jpg)" in m.pages_md[1]


def test_boxes_json_is_rewritten_when_the_last_figure_disappears(tmp_path):
    """마지막 그림이 사라져도 boxes.json을 갱신해야 한다 — 옛 항목이 남으면
    레이아웃 뷰가 없는 크롭을 그리려 한다."""
    import json as _json

    from app.pipeline.merge import IncrementalMerger

    job_dir = tmp_path / "job"
    merger = IncrementalMerger(job_dir, "\n\n---\n\n")
    merger.images_dir.mkdir(parents=True, exist_ok=True)
    merger.figure_boxes["p0001_0.jpg"] = {"bbox": [0, 0, 1, 1]}
    merger._write_boxes()
    assert _json.loads((merger.images_dir / "boxes.json").read_text())

    merger.figure_boxes.clear()
    merger._write_boxes()
    assert _json.loads((merger.images_dir / "boxes.json").read_text()) == {}


def test_quiet_add_chunk_places_pages_without_marker_warnings(tmp_path):
    """취소로 끊긴 부분 출력은 마커가 모자란 게 당연하다 — 배치·보정은 그대로 하되
    '모델이 페이지를 놓쳤다'로 읽히는 경고는 남기지 않는다."""
    m = IncrementalMerger(tmp_path, SEP)
    c = _mk_multi_chunk(tmp_path, "chunk_00", 3)
    m.add_chunk(ChunkResult(c, 1, 3, "<PAGE>\nonly one page"), warn=False)
    assert m.pages_md == ["only one page", "", ""]
    assert m.warnings == []


def test_keep_leading_pages_drops_truncated_and_unstarted_artifacts(tmp_path):
    from app.pipeline.merge import keep_leading_pages

    c = _mk_multi_chunk(tmp_path, "chunk_00", 4, images_per_page=2)
    (c / "raw_pages.json").write_text(
        json.dumps({"pages": ["r0", "r1", "r2-truncated"]}), encoding="utf-8"
    )
    keep_leading_pages(c, 2)

    assert sorted(f.name for f in (c / "images").iterdir()) == [
        "page_0_0.jpg", "page_0_1.jpg", "page_1_0.jpg", "page_1_1.jpg",
    ]
    assert sorted(f.name for f in c.glob("result_with_boxes_*.jpg")) == [
        "result_with_boxes_0.jpg", "result_with_boxes_1.jpg",
    ]
    assert json.loads((c / "raw_pages.json").read_text(encoding="utf-8")) == {
        "pages": ["r0", "r1"]
    }
    # 원출력이 보존 페이지 수보다 짧으면 빈 원출력으로 채워 개수를 맞춘다
    keep_leading_pages(c, 3)
    assert json.loads((c / "raw_pages.json").read_text(encoding="utf-8"))["pages"] == [
        "r0", "r1", "",
    ]


# ── 정합: 표·수식·그림 페이지, 갭 채우기, 약한 근거 ─────────────────────────


def test_assign_slots_fills_single_gaps_monotonically():
    from app.pipeline.merge import assign_slots

    assert assign_slots([0, None, 2], 3) == [0, 1, 2]           # 가운데 한 칸
    assert assign_slots([None, 1, 2], 3) == [0, 1, 2]           # 앞 끝
    assert assign_slots([0, 1, None], 3) == [0, 1, 2]           # 뒤 끝
    assert assign_slots([0, None, None, 3], 4) == [0, 1, 2, 3]  # 같은 수의 연속 갭
    # 미매칭 수와 빈 슬롯 수가 다르면 앞 매칭 페이지에 붙는다(내용 손실 없음)
    assert assign_slots([0, None, 1], 2) == [0, 0, 1]


def test_assign_slots_uses_scores_to_attach_a_split_half_forward():
    """둘로 쪼개진 페이지의 앞 절반(미매칭)은 점수가 더 높은 다음 페이지에 붙는다."""
    from app.pipeline.merge import assign_slots

    scores = {(1, 0): 0.0, (1, 1): 0.25}  # 모델 1은 슬롯 1(다음)과 더 닮았다
    got = assign_slots([0, None, 1], 2, score=lambda k, s: scores.get((k, s), 0.0))
    assert got == [0, 1, 1]
    # 단조성: 앞쪽 미매칭이 뒤로 가면 그 뒤의 미매칭도 뒤로 간다
    scores = {(1, 1): 0.3, (2, 0): 0.9}
    got = assign_slots([0, None, None, 1], 2, score=lambda k, s: scores.get((k, s), 0.0))
    assert got in ([0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1])
    assert got == sorted(got)


def _pdf_with_pages(path, page_lines):
    import fitz

    doc = fitz.open()
    for lines in page_lines:
        page = doc.new_page(width=595, height=842)
        y = 72
        for line in lines:
            page.insert_text((60, y), line, fontsize=10)
            y += 16
    doc.save(str(path))
    doc.close()


_PROSE = [
    "The quick survey of vector quantization methods covers scalar and product codes.",
    "We evaluate distortion rates on synthetic and real embedding datasets carefully.",
    "Results show the online algorithm matches the information theoretic lower bound.",
    "Further analysis considers inner product estimation and nearest neighbour search.",
] * 3


def test_table_page_is_aligned_after_markup_is_stripped(tmp_path):
    """표 페이지의 `<table><tr><td>`가 정규화에 남아 프로브가 실패하면, 표 페이지가 앞
    페이지에 붙고 제자리가 빈다(마커가 하나 더 나온 청크 — 위치 기반이었으면 맞았다)."""
    job = tmp_path / "job"
    job.mkdir()
    cells = [("Method", "Bits", "Recall"), ("TurboQuant", "3.5", "0.997"),
             ("KIVI", "3", "0.981"), ("SnapKV", "16", "0.858"), ("PolarQuant", "3.9", "0.995")]
    table_lines = ["   ".join(row) for row in cells] * 3
    _pdf_with_pages(job / "source.pdf", [
        ["PAGE ONE " + line for line in _PROSE],
        table_lines,
        ["PAGE THREE " + line for line in _PROSE],
    ])
    table_md = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells * 3
    ) + "</table>"
    third = " ".join("PAGE THREE " + line for line in _PROSE)
    model = [
        " ".join("PAGE ONE " + line for line in _PROSE),
        table_md,
        third[: len(third) // 2],
        third[len(third) // 2 :],
    ]
    m = IncrementalMerger(job, SEP)
    c = _mk_multi_chunk(job, "chunk_00", 3)
    m.add_chunk(ChunkResult(c, 1, 3, "<PAGE>\n" + "\n<PAGE>\n".join(model)))

    assert "PAGE ONE" in m.pages_md[0] and "TurboQuant" not in m.pages_md[0]
    assert "TurboQuant" in m.pages_md[1]
    assert m.pages_md[2].count("PAGE THREE") == len(_PROSE)


def test_latex_formula_page_is_aligned_to_its_glyph_text(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    glyph = [f"For vector x{i} with ∥x∥2 = 1 the bound E[⟨y, x˜⟩] ≤ √3π · 1/4b holds "
             f"for every index {i} in the sequence" for i in range(12)]
    latex = " ".join(
        rf"For vector \(x_{{{i}}}\) with \(\|x\|_{{2}} = 1\) the bound "
        rf"\(\mathbb{{E}}[\langle y, \tilde{{x}} \rangle] \leq \sqrt{{3}}\pi \cdot 1/4^{{b}}\) "
        f"holds for every index {i} in the sequence" for i in range(12)
    )
    _pdf_with_pages(job / "source.pdf", [
        ["PAGE ONE " + line for line in _PROSE], glyph, ["PAGE THREE " + line for line in _PROSE],
    ])
    m = IncrementalMerger(job, SEP)
    c = _mk_multi_chunk(job, "chunk_00", 3)
    # 모델이 3쪽을 건너뛰었다(마커 2개) — 수식 페이지가 매칭돼야 1·2쪽이 제자리다
    m.add_chunk(ChunkResult(
        c, 1, 3,
        "<PAGE>\n" + " ".join("PAGE ONE " + line for line in _PROSE) + "\n<PAGE>\n" + latex,
    ))
    assert "PAGE ONE" in m.pages_md[0]
    assert "holds for every index" in m.pages_md[1]
    assert m.pages_md[2] == ""


def test_weak_alignment_evidence_falls_back_to_positional_placement(tmp_path):
    """대조가 거의 맞지 않으면(텍스트 레이어가 모델 출력과 다른 문서) 근거 없는
    재배치 대신 위치 기반으로 둔다."""
    job = tmp_path / "job"
    job.mkdir()
    _pdf_with_pages(job / "source.pdf", [
        ["PAGE ONE " + line for line in _PROSE],
        ["unrelated text layer number two " * 3] * 8,
        ["unrelated text layer number three " * 3] * 8,
        ["unrelated text layer number four " * 3] * 8,
    ])
    model = [
        "completely different transcription alpha " * 10,
        "completely different transcription beta " * 10,
        " ".join("PAGE ONE " + line for line in _PROSE),   # 1/3만 매칭(물리 1쪽)
    ]
    m = IncrementalMerger(job, SEP)
    c = _mk_multi_chunk(job, "chunk_00", 4)
    m.add_chunk(ChunkResult(c, 1, 4, "<PAGE>\n" + "\n<PAGE>\n".join(model)))

    # 위치 기반: 모델 순서 그대로, 모자란 끝 페이지는 빈 페이지
    assert "alpha" in m.pages_md[0] and "beta" in m.pages_md[1]
    assert "PAGE ONE" in m.pages_md[2] and m.pages_md[3] == ""
    assert any("빈 페이지로 보정" in w for w in m.warnings), m.warnings
