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


def test_realigned_pages_name_crops_by_the_model_page_not_the_slot(tmp_path):
    """정렬이 페이지를 옮기면 layout의 크롭 이름도 같은 기준을 따라야 한다.

    크롭 파일은 벤더가 **모델 페이지 인덱스**로 저장하고(`page_{k}_{j}.jpg`)
    `_move_chunk_files`·`_rewrite_refs`도 k로 이름을 짓는다. `_ingest_layout`만
    물리 슬롯으로 이름을 지으면 layout이 **없는 파일**을 가리키고 실제 파일은
    고아가 된다(레이아웃 뷰에서 그림이 통째로 빠진다).
    """
    import json as _json

    from app.pipeline.merge import ChunkResult, IncrementalMerger

    job_dir = tmp_path / "job"
    (job_dir / "pages").mkdir(parents=True)
    merger = IncrementalMerger(job_dir, "\n\n---\n\n")

    chunk_dir = job_dir / "work" / "chunk_00"
    (chunk_dir / "images").mkdir(parents=True)
    # 모델 페이지 0·1의 크롭. 벤더 규약은 page_{모델인덱스}_{크롭}.jpg
    (chunk_dir / "images" / "page_0_0.jpg").write_bytes(b"crop0")
    (chunk_dir / "images" / "page_1_0.jpg").write_bytes(b"crop1")
    (chunk_dir / "raw_pages.json").write_text(
        _json.dumps({"pages": [
            "<|det|>image [10, 10, 500, 500]<|/det|>",
            "<|det|>image [10, 10, 500, 500]<|/det|>",
        ]}), encoding="utf-8",
    )
    chunk = ChunkResult(chunk_dir, 1, 3, "a\n\n<PAGE>\n\nb")
    merger._move_chunk_files(chunk)
    # 정렬 결과: 모델 0 → 슬롯 0, 모델 1 → 슬롯 2 (가운데 페이지는 유실)
    merger._ingest_layout(chunk, ["<|det|>image [10, 10, 500, 500]<|/det|>", "",
                                  "<|det|>image [10, 10, 500, 500]<|/det|>"],
                          [0, None, 1])

    on_disk = {f.name for f in (job_dir / "images").iterdir()}
    referenced = {
        b["image"]
        for page in merger.layout_pages
        for b in page["blocks"]
        if b.get("image")
    }
    assert referenced <= on_disk, f"없는 파일을 참조한다: {referenced - on_disk}"
    assert on_disk <= referenced | {"boxes.json"}, f"고아 파일: {on_disk - referenced}"
    # 3쪽 슬롯의 그림은 모델 1쪽의 크롭(p0002_0.jpg)이어야 한다
    page3 = [p for p in merger.layout_pages if p["page"] == 3][0]
    assert page3["blocks"][0]["image"] == "p0002_0.jpg", page3["blocks"][0]


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
