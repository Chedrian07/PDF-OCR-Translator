"""코드·CLI 트랜스크립트 블록은 번역하지 않는다 — 의도적 보존.

논문 부록의 에이전트 트랜스크립트·소스 리스팅은 알파벳이 많아 기존
non-linguistic 규칙(문자 비율 0.3)을 그대로 통과했다. 그 결과 왕복 한 번을 쓰고,
출력 게이트가 거부하고, 래더를 소진한 뒤 "미번역"으로 집계됐다. 더 나쁜 경우는
게이트를 통과해 **명령이 실제로 번역되는 것**이다 — 실측:
    `$ condensation`            → `$ 응축`            (동작하지 않는 명령)
    `-rw-r--r-- … May 6 10:26`  → `… 5월 6일 10:26`   (트랜스크립트 훼손)

오탐(산문을 건너뜀)은 재시도도 경고도 없이 영문으로 굳으므로 미탐보다 나쁘다.
그래서 산문 줄이 하나라도 있으면 건너뛰지 않는다.
"""

import pytest

from app.translate.masking import looks_like_code, should_skip
from app.translate.segment import apply_layout

CODE = [
    "$ ls -l /workspace/src-vul/wasmtime/fuzz/fuzz_targets | grep cranelift",
    "$ condensation",
    "[Step 76/100]",
    "[Step: 42/52]",
    "use cranelift_codegen::ir::Signature;",
    "import struct",
    "def find_sos(data):",
    "-rw-r--r-- 1 root root 7691 Apr 30 03:27 cranelift-icache.rs",
    "data[sos+4] = 2",
    "first = data[sos+5:sos+7]",
    '{"task_id":"arvo:56150","exit_code":10}',
    "total 2214212",
    "ls: cannot access '/workspace/src-vul/fuzz/corpus': No such file or directory",
]

PROSE = [
    "AI agents have significant potential to reshape cybersecurity, making a thorough "
    "assessment of their capabilities critical.",
    "We apply various automated and manual filters to improve CyberGym's quality:",
    "# Deep Learning",
    "## 3.1 Preliminaries",
    "Table 1: Comparing CyberGym with existing cybersecurity benchmarks for AI agents.",
    # OCR이 산문 줄 앞에 프롬프트 기호를 잘못 붙인 실측 사례
    "$ Since there are no pre-built binaries, it is likely that the binary must be "
    "built from source using the provided script.",
    # 필드 라벨 뒤의 커밋 메시지 — 로그 레벨 패턴으로 삼키면 안 된다 (실측 p19)
    "MESSAGE: Merge pull request #6222 from JacobBarthelmeh/alerts. don't try to send "
    "an alert to a disconnected peer",
    # 명령이 섞였어도 산문이 있으면 번역 대상
    "[Step 42/69]\n# None of the simple PoCs triggered a crash. The next thing to try "
    "is a larger input that covers more of the parser.",
]


@pytest.mark.parametrize("text", CODE)
def test_code_and_transcripts_are_detected(text):
    assert looks_like_code(text) == "code", text


@pytest.mark.parametrize("text", PROSE)
def test_prose_is_never_detected_as_code(text):
    """오탐은 재시도도 경고도 없이 영문으로 굳는다 — 절대 걸리면 안 된다."""
    assert looks_like_code(text) == "", text


def test_korean_bearing_text_is_left_to_the_existing_rules():
    """한글이 섞인 원문은 코드 판정 대상이 아니다 — 이미 번역됐거나 한국어 문서다.

    (판정 자체를 하지 않으므로 `[단계 81/100]` 같은 한글 마커는 잡히지 않는다.
    원문이 영어인 통상 입력에서는 발생하지 않는 경로다.)
    """
    assert looks_like_code("[단계 81/100]") == ""
    assert looks_like_code("$ ls 를 실행한다") == ""


def test_identifier_lists_are_detected():
    ids = "ada-url, alembic, apache-httpd, arduinojson, args, arrow, assimp, avahi, binutils"
    assert looks_like_code(ids) == "identifier-list"
    # 쉼표로 이어진 **문장**은 식별자 나열이 아니다
    assert looks_like_code(
        "First, we collect the data, then we filter it, and finally we evaluate, "
        "which gives us the numbers, as shown, in Table 1, for each model, per task"
    ) == ""


def test_should_skip_reports_the_decision_not_a_failure():
    assert should_skip("$ condensation") == "code"
    assert should_skip("[Step 76/100]") == "code"
    # 기존 사유는 그대로 — 인라인 수식은 셸 프롬프트가 아니다
    assert should_skip("$E = mc^2$") == "non-linguistic"
    assert should_skip("이미 한국어로 된 문장입니다. 번역할 필요가 없습니다.") == "already-korean"


def test_preserved_reason_reaches_the_layout_artifact():
    """PDF 내보내기가 '실패'와 '의도적 보존'을 구분할 수 있어야 한다."""
    pages = [{
        "page": 1, "width": 100, "height": 100,
        "blocks": [
            {"type": "text", "bbox": [0, 0, 999, 99], "content": "$ ls -l /workspace"},
            {"type": "text", "bbox": [0, 100, 999, 199], "content": "Some prose here."},
        ],
    }]
    out = apply_layout(pages, {"lay:1:1": "여기 산문이 있다."}, {"lay:1:0": "code"})
    assert out[0]["blocks"][0]["preserved"] == "code"
    assert out[0]["blocks"][0]["content"] == "$ ls -l /workspace"  # 원문 그대로
    assert "preserved" not in out[0]["blocks"][1]
    assert out[0]["blocks"][1]["content"] == "여기 산문이 있다."
    # 원본 pages는 변형되지 않는다 (deep copy 계약)
    assert "preserved" not in pages[0]["blocks"][0]


# ── 혼재 블록: 줄 단위 불변 토큰 ──────────────────────────────────────────
# 부록 트랜스크립트는 명령과 산문이 한 유닛에 섞여 있어 블록 단위 skip이 (오탐을
# 피하려고 올바르게) 거부한다. 그러면 명령까지 번역돼 재현 불가능해진다 — 실측
# result.ko.md: `$ condensation`→`$ 응축`, `May 6 10:26`→`5월 6일 10:26`,
# `[Step 85/100]`→`[단계 85/100]`. 줄 단위로 묶어 산문만 번역되게 한다.

INVARIANT_LINES = [
    "$ condensation",
    "$ ls -l /workspace/src-vul | grep cranelift",
    "> make -j8 fuzz",
    "[Step 85/100]",
    "[Step: 42/52]",
]


@pytest.mark.parametrize("line", INVARIANT_LINES)
def test_command_lines_are_masked_as_invariant(line):
    from app.translate.masking import mask

    masked, mapping = mask(line)
    assert list(mapping.values()) == [line], (masked, mapping)


@pytest.mark.parametrize("text,keep", [
    ("-rw-r--r-- 1 root root 723 May 6 10:26 README.md", "723 May 6 10:26"),
    # OCR이 권한·링크수를 뭉갠 실측 형태도 잡아야 한다
    ("rv-r--r-- i root root 4502 Apr 24 08:52 error.txt", "4502 Apr 24 08:52"),
])
def test_ls_long_timestamps_are_masked(text, keep):
    from app.translate.masking import mask

    _masked, mapping = mask(text)
    assert keep in mapping.values(), mapping


def test_prose_dates_are_not_masked():
    """산문 속 날짜는 번역돼야 한다 — 크기·파일명이 없으면 ls -l이 아니다."""
    from app.translate.masking import mask

    text = "We released the dataset in May 2024 and it has been widely used since."
    masked, mapping = mask(text)
    assert mapping == {} and masked == text


def test_mixed_transcript_keeps_commands_and_still_translates_the_prose():
    """산문이 섞인 블록은 건너뛰지 않되, 명령 줄만 불변으로 보호한다."""
    from app.translate.masking import mask

    text = (
        "[Step 42/69]\n"
        "# None of the simple PoCs triggered a crash. The next thing to try is a "
        "larger input that covers more of the parser.\n"
        "$ ls -l /workspace"
    )
    assert looks_like_code(text) == ""          # 산문이 있으니 번역 대상
    masked, mapping = mask(text)
    assert sorted(mapping.values()) == ["$ ls -l /workspace", "[Step 42/69]"]
    assert "None of the simple PoCs" in masked  # 산문은 그대로 모델에 간다


def test_command_with_natural_language_argument_splits_at_the_command_name():
    """`$ think "긴 영어 문장"`은 명령 이름만 보호하고 인자는 번역시킨다.

    줄 전체를 불변으로 묶으면 그 문장이 번역면에 영어로 남고(원문·번역 혼재),
    줄 전체를 번역하면 명령 이름까지 번역된다(`$ condense` → `$ 응축`).
    """
    from app.translate.masking import mask, unmask

    line = '$ think "There is no evidence that any dissector calls this function."'
    masked, mapping = mask(line)
    assert list(mapping.values()) == ["$ think"]
    assert "There is no evidence that any dissector" in masked
    assert unmask(masked, mapping)[0] == line


def test_ocr_mislabelled_prose_behind_a_prompt_is_still_translated():
    """OCR이 산문 앞에 프롬프트를 잘못 붙여도 문장은 번역 대상으로 남는다."""
    from app.translate.masking import mask

    line = "$ Since there are no pre-built binaries, the binary must be built from source."
    masked, mapping = mask(line)
    assert list(mapping.values()) == ["$ Since"]
    assert "there are no pre-built binaries" in masked


def test_markdown_blockquote_is_translated_but_keeps_its_marker():
    """`>`는 셸 연속 프롬프트이자 마크다운 인용 기호 — 기호만 보호한다."""
    from app.translate.masking import mask, unmask

    line = "> The rest of the paper is organized as follows: we describe the method."
    masked, mapping = mask(line)
    assert list(mapping.values()) == [">"]
    assert "The rest of the paper is organized" in masked
    assert unmask(masked, mapping)[0] == line


def test_short_continuation_prompt_stays_verbatim():
    """산문으로 볼 수 없는 짧은 연속 프롬프트 줄은 통째로 원문 유지."""
    from app.translate.masking import mask

    line = "> simulate the vulnerable version:"
    _masked, mapping = mask(line)
    assert list(mapping.values()) == [line]


def test_tokens_inside_a_split_tail_are_still_masked():
    """잘라낸 자연어 꼬리도 재스캔해 URL·참조를 보호한다."""
    from app.translate.masking import mask, unmask

    line = "> But note that the results in Table 3 came from https://example.org/x."
    masked, mapping = mask(line)
    assert set(mapping.values()) >= {">", "Table 3", "https://example.org/x."}
    assert unmask(masked, mapping)[0] == line


def test_split_does_not_recurse_into_another_prompt_line():
    """꼬리 재스캔이 k3 분기를 다시 타지 않는다(무한 재귀·중복 분할 방지)."""
    from app.translate.masking import mask, unmask

    line = "$ echo the following command is not a real prompt: $ rm -rf /tmp/x"
    masked, mapping = mask(line)
    assert unmask(masked, mapping)[0] == line
