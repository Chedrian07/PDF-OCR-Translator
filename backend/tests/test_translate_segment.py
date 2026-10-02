"""세그먼트 — identity 골든·references 스킵·hr 새니타이즈·layout 불변."""

from app.translate.segment import (
    apply_layout,
    assemble_markdown,
    layout_line_map,
    layout_units,
    map_unit_lines,
    split_markdown,
)

SEP = "\n\n---\n\n"

# 표·펜스·heading·인용·references 포함 3페이지 합성 (merge.py 스타일: strip + 후행 \n)
MD = (
    "# Introduction\n\n"
    "This is the opening paragraph with a citation [1] and $x^2$ math.\n\n"
    "| Name | Score |\n|------|-------|\n| A | 0.9 |\n| B | 0.8 |"
    + SEP +
    "## Method\n\n"
    "We describe the approach below.\n\n"
    "```python\ndef f(x):\n    return x + 1\n```\n\n"
    "> An indented quote block.\n\n"
    "- first bullet\n- second bullet"
    + SEP +
    "## References\n\n"
    "[1] Author A. A paper title. Venue, 2020.\n\n"
    "[2] Author B. Another title. Venue, 2021.\n"
)


def test_identity_골든_바이트_동일():
    """전 유닛을 src 그대로 매핑하면 조립 결과가 원본과 바이트 동일해야 한다."""
    units = split_markdown(MD, SEP)
    translations = {u.id: u.src for u in units}
    assert assemble_markdown(MD, SEP, translations) == MD


# 같은 페이지에서 목록 뒤에 문단이 오는 문서(논문 대부분) — markdown-it은 목록 토큰의 줄 범위에
# 뒤따르는 빈 줄까지 넣는다. 예전에는 그 범위를 통째로 번역문(앞뒤 빈 줄을 다듬은 것)으로 바꿔
# 구분 빈 줄이 사라졌고, 다음 문단이 마지막 목록 항목에 흡수돼 한국어 미리보기·flow·Markdown
# 다운로드에서 <li> 안으로 들어갔다(감사 fresh-user-1).
_LIST_THEN_PARAGRAPH = [
    "Intro paragraph.\n\n- first bullet\n- second bullet\n\nEnd of sample document.\n",
    "Intro.\n\n1. one\n2. two\n\nNext para.\n",
    "- loose a\n\n- loose b\n\n\nAfter two blank lines.",
    "1. Introduction\n\nBody text right after a numbered heading.",
    "- a\n- b\n" + SEP + "- c\n- d\n\nPara on page two.\n",
]


def test_identity_골든_목록_뒤_문단도_바이트_동일():
    for md in _LIST_THEN_PARAGRAPH:
        units = split_markdown(md, SEP)
        assert assemble_markdown(md, SEP, {u.id: u.src for u in units}) == md, repr(md)


def test_번역된_목록_뒤_문단이_목록_항목에_흡수되지_않는다():
    from app.pipeline.render import render_markdown_html

    md = "Intro paragraph.\n\n- first bullet\n- second bullet\n\nEnd of sample document.\n"
    units = {u.kind: u for u in split_markdown(md, SEP)}
    out = assemble_markdown(md, SEP, {
        units["list"].id: "- 첫째 항목\n- 둘째 항목\n",   # 모델은 끝 빈 줄을 흔히 빼거나 더한다
        "md:0:2": "샘플 문서의 끝.",
    })
    assert out == "Intro paragraph.\n\n- 첫째 항목\n- 둘째 항목\n\n샘플 문서의 끝.\n"
    html = render_markdown_html(out, "/api/jobs/x/files")
    assert "<li>둘째 항목</li>" in html and "<p>샘플 문서의 끝.</p>" in html, html


def test_유닛_종류와_id():
    units = split_markdown(MD, SEP)
    kinds = {u.id: u.kind for u in units}
    assert kinds["md:0:0"] == "heading"
    assert kinds["md:0:2"] == "table"
    assert kinds["md:1:2"] == "fence"
    assert kinds["md:1:3"] == "blockquote"
    assert kinds["md:1:4"] == "list"


def test_references_스킵():
    units = split_markdown(MD, SEP)
    by_id = {u.id: u for u in units}
    # References heading과 그 뒤 항목 전부 skip
    assert by_id["md:2:0"].skip_reason == "references"
    assert by_id["md:2:1"].skip_reason == "references"
    assert by_id["md:2:2"].skip_reason == "references"
    # 그 외 페이지는 스킵되지 않음
    assert by_id["md:0:1"].skip_reason == ""
    assert by_id["md:1:1"].skip_reason == ""


def test_references_구간_상위heading에서_종료():
    """references(h2) 하위 소제목(h3)은 계속 스킵, 다음 동급/상위 heading(h1)에서 해제."""
    md = (
        "## References\n\n[1] a paper.\n\n### Sub note\n\nstill in refs.\n\n"
        "# Appendix Data\n\nback to translatable content."
    )
    units = split_markdown(md, SEP)
    by_id = {u.id: u for u in units}
    assert by_id["md:0:0"].skip_reason == "references"   # ## References
    assert by_id["md:0:1"].skip_reason == "references"   # [1] a paper
    assert by_id["md:0:2"].skip_reason == "references"   # ### Sub note (하위)
    assert by_id["md:0:3"].skip_reason == "references"   # still in refs
    # "# Appendix Data"는 acknowledg/references 패턴이 아니고 h1(상위) → 해제
    assert by_id["md:0:4"].skip_reason == ""
    assert by_id["md:0:5"].skip_reason == ""


def test_hr_새니타이즈_페이지수_불변():
    """번역문에 '---' 줄이 생겨도 ⸻로 치환되어 페이지 수가 유지된다."""
    units = split_markdown(MD, SEP)
    trans = {}
    for u in units:
        if u.kind == "paragraph" and not u.skip_reason:
            trans[u.id] = "번역 첫 줄\n---\n번역 둘째 줄"
        else:
            trans[u.id] = u.src
    out = assemble_markdown(MD, SEP, trans)
    assert len(out.split(SEP)) == 3       # 페이지 수 보존
    assert "⸻" in out                     # hr → ⸻ 치환
    assert "\n---\n다시" not in out


def test_페이지구분자_유발_유닛_원문유지():
    # 기본 '---' 구분자는 새니타이즈가 중화하므로, 새니타이즈가 못 잡는 커스텀
    # 구분자로 유닛 단위 선방어(page_separator in new_text → 원문 유지)를 검증한다.
    sep = "\n\n@@@\n\n"
    md = "# A\n\nfirst page paragraph." + sep + "# B\n\nsecond page paragraph."
    units = split_markdown(md, sep)
    trans = {u.id: u.src for u in units}
    para = next(u for u in units if u.kind == "paragraph")
    trans[para.id] = "오염 시도" + sep + "뒷부분"   # 구분자 유발
    out = assemble_markdown(md, sep, trans)
    assert len(out.split(sep)) == 2        # 페이지 수 보존
    assert para.src in out                 # 해당 유닛은 원문 유지


def test_layout_units_필터():
    pages = [{
        "page": 3, "width": 1000, "height": 1400, "blocks": [
            {"type": "text", "bbox": [0, 0, 999, 50], "content": "translate me"},
            {"type": "image", "bbox": [0, 60, 999, 400], "content": "", "image": "p0003_0.jpg"},
            {"type": "text", "bbox": [0, 410, 999, 450], "content": "   "},  # 빈 content
        ],
    }]
    units = layout_units(pages)
    assert [u.id for u in units] == ["lay:3:0"]  # 이미지·빈 content 제외
    assert units[0].page == 3


def test_layout_reference_unit은_원문_유지():
    pages = [{"page": 1, "blocks": [
        {"type": "ref_text", "content": "[1] Author, Paper, https://example.com"},
    ]}]
    unit = layout_units(pages)[0]
    assert unit.skip_reason == "references"


def test_apply_layout_content외_필드_불변():
    pages = [{
        "page": 1, "width": 612, "height": 792, "fonts_v": "2", "blocks": [
            {"type": "title", "bbox": [0, 0, 999, 80], "content": "Title", "fs": 2.5, "bold": True},
            {"type": "text", "bbox": [10, 300, 40, 900], "content": "vertical", "fs": 1.47,
             "vertical": "up"},
            {"type": "image", "bbox": [0, 100, 500, 300], "content": "", "image": "p0001_0.jpg"},
        ],
    }]
    units = layout_units(pages)
    out = apply_layout(pages, {u.id: "번역:" + u.src for u in units})
    b = out[0]["blocks"]
    assert b[0]["content"] == "번역:Title" and b[0]["fs"] == 2.5 and b[0]["bold"] is True
    assert b[1]["content"] == "번역:vertical" and b[1]["vertical"] == "up"
    assert out[0]["fonts_v"] == "2"
    # 이미지 블록·원본은 손대지 않음
    assert b[2]["content"] == "" and b[2]["image"] == "p0001_0.jpg"
    assert pages[0]["blocks"][0]["content"] == "Title"  # 원본 불변(deep copy)


def test_markdown과_layout_번역을_단일_표기로_정렬():
    """md 유닛의 모든 줄이 layout 블록과 대응하면 layout 번역을 그대로 쓴다."""
    md = "How to Read a Paper\nBody sentence.\nAnother sentence.\n\n---\n\n[1] Author, Paper."
    source = [
        {"page": 1, "blocks": [
            {"type": "title", "content": "How to Read a Paper"},
            {"type": "text", "content": "Body sentence."},
            {"type": "text", "content": "Another sentence."},
        ]},
        {"page": 2, "blocks": [
            {"type": "ref_text", "content": "[1] Author, Paper."},
        ]},
    ]
    translated = [
        {"page": 1, "blocks": [
            {"type": "title", "content": "논문을 읽는 방법"},
            {"type": "text", "content": "본문 문장."},
            {"type": "text", "content": "다른 문장."},
        ]},
        {"page": 2, "blocks": [
            {"type": "ref_text", "content": "[1] 저자, 논문."},
        ]},
    ]
    mapping = layout_line_map(source, translated)
    assert "[1] Author, Paper." not in mapping            # ref_text는 매핑하지 않는다
    units = split_markdown(md, "\n\n---\n\n")
    assert map_unit_lines(units[0].src, mapping) == "논문을 읽는 방법\n본문 문장.\n다른 문장."
    assert map_unit_lines(units[1].src, mapping) is None


def test_한_줄이라도_매핑이_없으면_유닛은_자기_번역을_쓴다():
    """종전 줄 단위 reconcile은 매핑 안 된 줄을 원문(영어)으로 남겼다(translate-llm-1)."""
    source = [{"page": 1, "blocks": [{"type": "text", "content": "Matched."}]}]
    translated = [{"page": 1, "blocks": [{"type": "text", "content": "일치."}]}]
    mapping = layout_line_map(source, translated)
    assert map_unit_lines("Unmatched one.\nMatched.", mapping) is None
    assert map_unit_lines("  Matched.  ", mapping) == "  일치.  "   # 줄 앞뒤 공백 보존
    assert map_unit_lines("", mapping) is None


def test_번역에_실패한_블록은_매핑하지_않는다():
    """원문이 그대로 남은(번역 실패) 블록이 md 줄을 영어로 '매핑'하면 안 된다."""
    source = [{"page": 3, "blocks": [
        {"type": "text", "content": "Failed sentence."},
        {"type": "page_number", "content": "7"},
        {"type": "text", "content": "Translated sentence."},
    ]}]
    translated = [{"page": 3, "blocks": [
        {"type": "text", "content": "Failed sentence."},
        {"type": "page_number", "content": "7"},
        {"type": "text", "content": "번역된 문장."},
    ]}]
    final = {"lay:3:1", "lay:3:2"}                      # 보존(쪽 번호) + 번역 성공
    mapping = layout_line_map(source, translated, final)
    assert mapping == {"7": "7", "Translated sentence.": "번역된 문장."}
    assert map_unit_lines("7\nTranslated sentence.", mapping) == "7\n번역된 문장."


def test_같은_원문의_상충_번역은_매핑하지_않는다():
    source = [{"page": 1, "blocks": [
        {"type": "text", "content": "Same line."}, {"type": "text", "content": "Same line."},
    ]}]
    translated = [{"page": 1, "blocks": [
        {"type": "text", "content": "같은 줄."}, {"type": "text", "content": "동일한 줄."},
    ]}]
    assert layout_line_map(source, translated) == {}
    assert layout_line_map("깨진 값", translated) == {}


def test_layout_line_sources_는_reconcile과_같은_필터를_쓴다():
    """엔진의 2단 패스가 md 유닛 지연 여부를 판단하는 집합 — ref_text·다중 줄·
    중복 원문은 layout_line_map이 매핑하지 않으므로 여기서도 제외한다."""
    from app.translate.segment import layout_line_sources

    pages = [
        {"page": 1, "blocks": [
            {"type": "text", "content": "Single line block."},
            {"type": "text", "content": "Multi line\nblock here."},
            {"type": "ref_text", "content": "[1] Author, Paper."},
            {"type": "text", "content": "   "},
            {"type": "image", "content": "", "image": "p1.jpg"},
        ]},
        {"page": 2, "blocks": [
            {"type": "text", "content": "Duplicated line."},
            {"type": "text", "content": "Duplicated line."},
        ]},
    ]
    assert layout_line_sources(pages) == {"Single line block."}
    assert layout_line_sources("깨진 값") == set()


# ── 참고문헌 규칙 불일치 관측 (md heading 스윕 vs layout ref_text) ──────────

def test_layout만_ref_text면_불일치로_집계된다():
    """같은 원문이 PDF(layout)에선 원문 유지, result.ko.md에선 번역되는 사례."""
    from app.translate.segment import reference_rule_mismatch

    md_units = split_markdown(
        "# Introduction\n\nBody sentence here.\n\n[1] Author A. A paper title. Venue, 2020.\n",
        SEP,
    )
    lay_units = layout_units([
        {"page": 1, "blocks": [
            {"type": "text", "content": "Body sentence here."},
            {"type": "ref_text", "content": "[1] Author A. A paper title. Venue, 2020."},
        ]},
    ])
    got = reference_rule_mismatch(md_units, lay_units)
    assert got["layout_only"] == 1 and got["md_only"] == 0
    assert got["sample_units"] == ["lay:1:1"]


def test_md만_references_heading이면_불일치로_집계된다():
    """반대 방향 — md는 heading 스윕으로 건너뛰는데 layout은 text라 번역된다."""
    from app.translate.segment import reference_rule_mismatch

    md_units = split_markdown(
        "## References\n\n[1] Author A. A paper title. Venue, 2020.\n", SEP,
    )
    lay_units = layout_units([
        {"page": 1, "blocks": [
            {"type": "text", "content": "[1] Author A. A paper title. Venue, 2020."},
        ]},
    ])
    got = reference_rule_mismatch(md_units, lay_units)
    assert got["md_only"] == 1 and got["layout_only"] == 0


def test_두_규칙이_일치하면_불일치가_0이다():
    from app.translate.segment import reference_rule_mismatch

    md_units = split_markdown(
        "## References\n\n[1] Author A. A paper title. Venue, 2020.\n", SEP,
    )
    lay_units = layout_units([
        {"page": 1, "blocks": [
            {"type": "ref_text", "content": "[1] Author A. A paper title. Venue, 2020."},
        ]},
    ])
    got = reference_rule_mismatch(md_units, lay_units)
    assert got == {"md_only": 0, "layout_only": 0, "sample_units": []}


def test_실제_판정을_넘기면_내용_판정과_deferred_유닛은_불일치가_아니다():
    """엔진은 실제 건너뜀 사유(내용 판정 포함)와 layout 결과를 받는 md 유닛을 넘긴다.

    heading 표시만 보면 제목 표기 없는 result.md의 참고문헌 목록(내용 판정으로 건너뜀)과
    layout 보존을 그대로 받는 md 유닛이 모두 '번역 대상'으로 세여 오경보가 났다. 쪽 번호처럼
    다른 사유로 건너뛰는 layout 블록도 '번역 대상'이 아니다.
    """
    from app.translate.segment import reference_rule_mismatch

    ref1 = "[1] Author A. A paper title. Venue, 2020."
    ref2 = "[2] Author B. Another paper title. Journal, 2021."
    md_units = split_markdown(f"Body sentence here.\n\n{ref1}\n\n{ref2}\n7\n", SEP)
    lay_units = layout_units([
        {"page": 1, "blocks": [
            {"type": "ref_text", "content": ref1},
            {"type": "ref_text", "content": ref2},
            {"type": "page_number", "content": "7"},
        ]},
    ])
    static = reference_rule_mismatch(md_units, lay_units)
    assert static["layout_only"] == 2  # 예전 규칙: heading이 없으니 md 유닛 모두 '번역 대상'

    ids = [u.id for u in md_units]
    reasons = {ids[1]: "", ids[2]: "references", "lay:1:2": "non-linguistic"}
    got = reference_rule_mismatch(
        md_units, lay_units, reasons=reasons, deferred={ids[1]},
    )
    assert got == {"md_only": 0, "layout_only": 0, "sample_units": []}

    # deferred가 아니라 스스로 번역하는 md 유닛이면 여전히 불일치다
    got = reference_rule_mismatch(md_units, lay_units, reasons=reasons)
    assert got["layout_only"] == 1 and got["md_only"] == 0


def test_불일치_표본은_블록당_한_건만_센다():
    """줄이 많은 블록 하나가 집계를 부풀리지 않는다."""
    from app.translate.segment import reference_rule_mismatch

    md_units = split_markdown("Line one here.\n\nLine two here.\n", SEP)
    lay_units = layout_units([
        {"page": 1, "blocks": [
            {"type": "ref_text", "content": "Line one here.\nLine two here."},
        ]},
    ])
    got = reference_rule_mismatch(md_units, lay_units)
    assert got["layout_only"] == 1


def test_유닛_전체가_여러_줄_블록과_같으면_그_번역을_통째로_쓴다():
    """textlayer 잡의 저자 블록·여러 줄 문단 — 줄 매핑으로는 덮이지 않아 두 번 번역됐다."""
    from app.translate.segment import layout_line_sources

    block = "Amir Zandieh\nGoogle Research\nzandieh@google.com"
    source = [{"page": 1, "blocks": [{"type": "text", "content": block}]}]
    translated = [{"page": 1, "blocks": [{"type": "text", "content":
                                          "Amir Zandieh\n구글 리서치\nzandieh@google.com"}]}]
    assert block not in layout_line_sources(source)                   # 줄 매핑 후보는 아님
    assert block in layout_line_sources(source, multiline=True)       # 유닛 매핑 후보
    mapping = layout_line_map(source, translated)
    assert map_unit_lines(block, mapping) == "Amir Zandieh\n구글 리서치\nzandieh@google.com"
    assert map_unit_lines("Amir Zandieh\nGoogle Research", mapping) is None   # 일부만 같으면 안 됨


def test_한_줄_원문의_여러_줄_번역은_줄_매핑에_쓰지_않는다():
    source = [{"page": 1, "blocks": [{"type": "text", "content": "One line."}]}]
    translated = [{"page": 1, "blocks": [{"type": "text", "content": "한\n줄"}]}]
    assert layout_line_map(source, translated) == {}
