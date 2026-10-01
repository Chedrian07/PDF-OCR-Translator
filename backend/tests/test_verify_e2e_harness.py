"""E2E 하네스(scripts/verify_e2e.py · scripts/mock_llm.py) 판정 로직 회귀 테스트.

하네스는 서버를 띄워야 돌지만 **판정 로직 자체는 순수 함수**다. 하네스가 거짓 안심을
주는 회귀(공허한 비교식, 길이 보존 목, 페이지 단위 유실 미탐)를 유닛 수준에서 잡는다.
"""

import importlib.util
import json
import pathlib
import re
import sys

import pytest

from app.translate.masking import looks_untranslated
from app.translate.prompts import build_repair_prompt, build_unit_prompt

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_harness_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


verify_e2e = _load("verify_e2e")
mock_llm = _load("mock_llm")


# ───────────────────────── 스캐폴딩 누출 탐지 ─────────────────────────

def test_scaffolding_markers_read_from_prompts_module():
    """문구를 하드코딩하면 prompts.py가 바뀔 때 검사만 조용히 죽는다."""
    markers = verify_e2e.scaffolding_markers()
    assert "[번역할 원문" in markers
    # repair 프롬프트 문구가 포함돼야 한다 (echo 결함에서 실제로 새던 경로).
    assert any("수정할 번역문" in m for m in markers)


@pytest.mark.parametrize("prompt", [
    build_unit_prompt("Source text here.", [("model", "모델")], [("token", "토큰")],
                      context_tail="앞 문단", keep_terms=["BERT"], unit_kind="title"),
    build_repair_prompt("Source <m1/> text.", "번역문 일부", ["<m1/>"]),
])
def test_scaffolding_hits_catches_real_prompt_leak(prompt):
    assert verify_e2e.scaffolding_hits(prompt)


def test_scaffolding_hits_clean_translation():
    clean = "본 논문은 새로운 방법을 제안한다. 우리는 이를 세 개의 데이터셋에서 평가하였다.\n\n표 2에 결과를 정리하였다."
    assert verify_e2e.scaffolding_hits(clean) == []


def test_echo_growth_shape_is_detectable():
    """줄 수 단언 — repair 스캐폴딩 25줄이 박히면 부풀어 오르는 것을 잡는다."""
    src = "\n".join(f"line {i}" for i in range(108))
    polluted = src + "\n" + "\n".join(["[수정할 번역문]"] * 25 + ["잔여"] * 37)
    assert (polluted.count("\n") + 1) > (src.count("\n") + 1) * 1.15 + 3
    assert verify_e2e.scaffolding_hits(polluted)


# ───────────────── 번역은 있는데 PDF가 버린 페이지 (신고 결함) ─────────────────

def _page(text: str) -> dict:
    return {"blocks": [{"content": text}]}


KO = "본 논문은 새로운 방법을 제안한다. 우리는 이를 세 개의 데이터셋에서 평가하였다."
EN = "This paper proposes a new method. We evaluate it on three datasets."


def test_dropped_translation_pages_flags_untranslated_pdf_page():
    pages = [_page(KO), _page(KO), _page(KO)]
    pdf = [KO, EN, KO + " " + EN * 2]  # p2 완전 유실, p3 부분 유실
    assert verify_e2e.dropped_translation_pages(pages, pdf) == [
        (2, 1.0, 0.0),
        (3, 1.0, 0.23),
    ]


def test_dropped_translation_pages_ignores_by_design_original_pages():
    """참고문헌·수치 표처럼 번역면 자체가 원문인 페이지는 결함이 아니다."""
    pages = [_page(EN)]
    assert verify_e2e.dropped_translation_pages(pages, [EN]) == []


def test_dropped_translation_pages_accepts_faithful_export():
    pages = [_page(KO), _page(KO)]
    assert verify_e2e.dropped_translation_pages(pages, [KO, KO + " Table 2"]) == []


def test_kept_reason_summary_groups_by_reason():
    report = {
        "replaced": 165,
        "kept": 58,
        "specialist_kept": {"vertical": 1, "table": 2},
        "warning_count": 3,
        "warnings": [
            "p6: 블록 1 교체 생략(공간 부족) — 원문 보존",
            "p15: 블록 4 교체 생략(공간 부족) — 원문 보존",
            "p3: 그림 위 텍스트 — 원문 보존",
        ],
    }
    summary = verify_e2e.kept_reason_summary(report)
    assert summary["warning: 블록 N 교체 생략(공간 부족) — 원문 보존"] == 2
    assert summary["warning: 그림 위 텍스트 — 원문 보존"] == 1
    assert summary["specialist_kept.table"] == 2


def test_kept_reason_summary_absorbs_new_reason_fields():
    """fitting 그룹이 사유 집계 필드를 추가해도 이름을 고정하지 않고 흡수한다."""
    summary = verify_e2e.kept_reason_summary({"kept_reasons": {"multiline_cell": 15}})
    assert summary["kept_reasons.multiline_cell"] == 15


# ───────────────────── 목 LLM 압축률 (게이트 하한 회귀 탐지) ─────────────────────

PARA = (
    "We propose a new evaluation method for long-context language models. "
    "The results in Table 2 show that our approach improves accuracy on three datasets, "
    "and the training cost remains comparable to the baseline model."
)


def test_mock_compresses_like_real_korean():
    out = mock_llm._translate(PARA, ratio=0.4)
    ratio = len(out) / len(PARA)
    # 실제 한국어는 영어의 0.3~0.5배. 길이 보존 목이면 이 단언이 깨진다.
    assert 0.3 < ratio < 0.6, ratio
    assert ratio < len(mock_llm._translate(PARA, ratio=0.0)) / len(PARA)


def test_mock_output_still_passes_current_output_gate():
    """압축 목이 정상 경로를 깨면 안 된다 — 현행 하한(0.3)은 통과해야 한다."""
    out = mock_llm._translate(PARA, ratio=0.4)
    assert not looks_untranslated(PARA, out, {})


def test_mock_output_would_fail_a_raised_length_floor():
    """길이비 하한을 0.55로 올리는 회귀가 하네스에서 '실패로' 드러나는 근거."""
    out = mock_llm._translate(PARA, ratio=0.4)
    assert len(out) < 0.55 * len(PARA)


def test_mock_preserves_placeholders_and_line_structure():
    src = "# Introduction\n\nThe model <m1 v=\"E=mc^2\"/> is fast.\n\n- first item\n- second item"
    out = mock_llm._translate(src, ratio=0.4)
    assert '<m1 v="E=mc^2"/>' in out
    assert out.count("\n") == src.count("\n")
    assert out.startswith("# ")


def test_mock_short_function_word_unit_falls_back_to_length_preserving():
    """전부 기능어인 짧은 유닛까지 압축하면 목이 파이프라인을 깨뜨린다."""
    out = mock_llm._translate("The a of and", ratio=0.4)
    assert len(out) >= mock_llm._MIN_SAFE_RATIO * len("The a of and")
    assert any("가" <= c <= "힣" for c in out)


def test_mock_ratio_env_zero_restores_legacy_behaviour(monkeypatch):
    monkeypatch.setenv("MOCK_TRANSLATE_RATIO", "0")
    assert mock_llm._target_ratio() == 0.0
    assert mock_llm._translate(PARA) == mock_llm._translate(PARA, ratio=0.0)
    monkeypatch.setenv("MOCK_TRANSLATE_RATIO", "nonsense")
    assert mock_llm._target_ratio() == mock_llm._DEFAULT_RATIO


def test_fault_expectations_are_substring_checks_not_equality():
    """6,491자 파일 전체를 6자와 == 비교하던 공허한 검사의 회귀 방지."""
    for fault, (_, needles) in verify_e2e._FAULT_EXPECT.items():
        injected = mock_llm._apply_fault(fault, "Some source text.")
        assert injected is not None, fault
        if not needles:
            continue
        # 손상 출력이 큰 문서 안에 묻혀 있어도 잡혀야 한다.
        haystack = ("정상 문단입니다.\n" * 200) + injected + ("\n다른 문단입니다." * 200)
        assert any(n in haystack.lower() or n in haystack for n in needles), fault


def test_layout_pages_accepts_both_shapes():
    assert verify_e2e.layout_pages([{"blocks": []}]) == [{"blocks": []}]
    assert verify_e2e.layout_pages(json.loads('{"pages": [1]}')) == [1]


# ───────────────────── 목 SSE 스트리밍 (번역 클라이언트 기본 경로) ─────────────────────

def _serve_mock():
    import threading
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_llm.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    return srv


@pytest.mark.parametrize("fault", ["", "refusal", "refusal_ko", "echo", "summary",
                                   "drop_placeholder", "paired_tags"])
def test_mock_stream_and_plain_responses_carry_the_same_output(fault):
    """클라이언트 기본이 chat 스트리밍이 됐다 — 결함 모드가 SSE에서도 같은 출력을 내야
    하네스의 결함 주입이 스트리밍 경로에서도 유효하다."""
    from app.translate.client import OpenAICompatClient
    from app.translate.types import TranslateConfig

    srv = _serve_mock()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        if fault:
            base += f"?fault={fault}"
        prompt = '[번역할 원문]\nThe model <m1 v="E=mc^2"/> is fast and the results are good.'
        outs = []
        for stream in ("on", "off"):
            cfg = TranslateConfig(base_url=base, api_key="k", model="m", api_mode="chat",
                                  stream=stream, max_retries=0)
            outs.append(OpenAICompatClient(cfg).complete("s", prompt, max_tokens=100))
        assert outs[0] == outs[1]
        assert mock_llm.STATS["stream_chunks"] > 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_mock_stream_ends_with_done_and_usage():
    import json as _json
    import urllib.request

    srv = _serve_mock()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions?chunk=4"
        body = {"model": "m", "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "[번역할 원문]\nThe model is fast."}]}
        req = urllib.request.Request(url, data=_json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.headers["Content-Type"] == "text/event-stream"
            raw = resp.read().decode()
    finally:
        srv.shutdown()
        srv.server_close()
    events = [e for e in raw.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    usage = _json.loads(events[-2][6:])
    assert usage["choices"] == [] and usage["usage"]["total_tokens"] > 0
    finish = _json.loads(events[-3][6:])["choices"][0]["finish_reason"]
    assert finish == "stop"


# ───────────────── D-1: 원문별 호출 수 ≤ 번역 계획 (같은 유닛 이중 번역) ─────────────────

def _lay(*contents: str, page: int = 1) -> list:
    return [{"page": page, "blocks": [
        {"type": "text", "bbox": [50, 100 + 60 * i, 900, 150 + 60 * i], "content": c}
        for i, c in enumerate(contents)
    ]}]


SAME = "Same sentence about language models."
OTHER = "Unique paragraph about the training data."


def test_d1_allows_the_same_text_once_per_unit_position():
    """위치(직전 문맥)가 다른 같은 원문은 캐시 키가 달라 각자 번역된다 — 중복 0을 요구하면
    25쪽 실행이 정당한 중복 21종으로 늘 실패했다."""
    md = f"{SAME}\n\n{OTHER}\n\n---\n\n{SAME}"
    allowed, known, pending = verify_e2e.d1_unit_plan(md, None)
    assert allowed == {SAME: 2, OTHER: 1} and known == {SAME, OTHER} and pending == 0
    over, unplanned, other = verify_e2e.d1_findings(
        {SAME: 2, OTHER: 1, "[논문 개요]\n용어집 프롬프트": 1}, allowed, known)
    assert over == {} and unplanned == {} and other == 1
    over, _, _ = verify_e2e.d1_findings({OTHER: 2}, allowed, known)
    assert over == {OTHER: (2, 1)}  # 같은 유닛을 두 번 번역했다


def test_d1_md_unit_wholly_covered_by_a_layout_block_is_not_planned():
    """md 유닛이 layout 블록 하나와 같으면 그 블록의 번역을 쓴다 — 둘 다 호출하면 이중 번역."""
    allowed, known, pending = verify_e2e.d1_unit_plan(SAME, _lay(SAME))
    assert allowed == {SAME: 1} and pending == 0  # layout 유닛 몫 하나뿐
    over, _, _ = verify_e2e.d1_findings({SAME: 2}, allowed, known)
    assert over == {SAME: (2, 1)}


LINE1 = "First line of the covered paragraph."
LINE2 = "Second line of the covered paragraph."


def test_d1_md_unit_covered_line_by_line_is_deferred():
    allowed, known, pending = verify_e2e.d1_unit_plan(f"{LINE1}\n{LINE2}", _lay(LINE1, LINE2))
    assert allowed == {LINE1: 1, LINE2: 1} and pending == 0
    md_text = f"{LINE1}\n{LINE2}"
    assert md_text in known
    _, unplanned, _ = verify_e2e.d1_findings({LINE1: 1, LINE2: 1, md_text: 1}, allowed, known)
    assert unplanned == {md_text: 1}  # 지연돼야 할 md 유닛을 1차에서 번역했다


def test_d1_second_pass_is_planned_when_a_covering_block_was_kept():
    """덮는 layout 블록의 번역이 실패(kept)하면 지연된 md 유닛은 2차에서 번역된다 — 정당하다."""
    md_text = f"{LINE1}\n{LINE2}"
    translated = [{"page": 1, "blocks": [{"content": LINE1}, {"content": "둘째 줄을 옮겼다."}]}]
    allowed, _, pending = verify_e2e.d1_unit_plan(
        md_text, _lay(LINE1, LINE2), translated_pages=translated, kept_original=["lay:1:0"],
    )
    assert pending == 1 and allowed[md_text] == 1
    # 둘 다 번역됐으면 2차 패스는 없다
    translated = [{"page": 1, "blocks": [{"content": "첫 줄을 옮겼다."},
                                         {"content": "둘째 줄을 옮겼다."}]}]
    allowed, _, pending = verify_e2e.d1_unit_plan(
        md_text, _lay(LINE1, LINE2), translated_pages=translated,
    )
    assert pending == 0 and md_text not in allowed


def test_d1_skipped_units_are_known_but_never_planned():
    md = "## References\n\n[1] A. Author. A title of a paper. Journal, 2020.\n\n2504.19874"
    allowed, known, _ = verify_e2e.d1_unit_plan(md, _lay("12"))
    assert allowed == {}
    assert "## References" in known and "12" in known
    _, unplanned, _ = verify_e2e.d1_findings({"## References": 1}, allowed, known)
    assert unplanned == {"## References": 1}


def test_d1_counts_masked_text_like_the_mock_sees_it():
    """목은 프롬프트의 '[번역할 원문]' 뒤 = 마스킹된 원문으로 센다."""
    from app.translate.masking import mask

    src = "As shown in Figure 2, the loss drops quickly [12]."
    allowed, _, _ = verify_e2e.d1_unit_plan(src, None)
    assert allowed == {mask(src)[0]: 1}
    prompt = build_unit_prompt(mask(src)[0], [], [], context_tail="previous paragraph")
    assert mock_llm._source_only(prompt) in allowed


def test_d1_plan_for_job_reads_the_pinned_separator_and_reports(tmp_path):
    jd = tmp_path / "j_1"
    (jd / "translations" / "ko").mkdir(parents=True)
    sep = "\n\n<<<page>>>\n\n"
    (jd / "meta.json").write_text(json.dumps({"page_separator": sep}), encoding="utf-8")
    (jd / "result.md").write_text(f"{SAME}{sep}{SAME}", encoding="utf-8")
    allowed, _, pending = verify_e2e.d1_plan_for_job(jd, "ko")
    assert allowed == {SAME: 2} and pending == 0  # 잡에 고정된 구분자로 페이지를 나눈다

    (jd / "result.md").write_text(f"{LINE1}\n{LINE2}", encoding="utf-8")
    (jd / "layout.json").write_text(json.dumps(_lay(LINE1, LINE2)), encoding="utf-8")
    (jd / "layout.ko.json").write_text(json.dumps(
        [{"page": 1, "blocks": [{"content": LINE1}, {"content": "둘째 줄을 옮겼다."}]}]
    ), encoding="utf-8")
    (jd / "translations" / "ko" / "report.json").write_text(
        json.dumps({"kept_original": ["lay:1:0"]}), encoding="utf-8")
    _, _, pending = verify_e2e.d1_plan_for_job(jd, "ko")
    assert pending == 1


def test_second_pass_log_line_matches_the_engine_wording():
    """D-1은 엔진 로그의 2차 패스 수와 하네스 계산을 대조한다 — 문구가 바뀌면 조용히 0이 된다."""
    engine_src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "translate"
                  / "engine.py").read_text(encoding="utf-8")
    template = re.search(r'"(layout으로 덮이지 않는 지연 md 유닛 %d개 2차 번역)"', engine_src)
    assert template, "엔진의 2차 패스 로그 문구가 바뀌었다 — 하네스 정규식도 함께 고친다"
    assert verify_e2e._SECOND_PASS_LOG_RE.fullmatch(template.group(1) % 3)


# ───────────────── 결함 모드: 쌍 태그 (translate-llm-10) ─────────────────

def test_paired_tags_fault_pairs_placeholders_like_an_xml_habit():
    from app.translate.masking import mask, unmask

    masked, mapping = mask("As shown in Figure 2 and Table 3, the loss $L$ drops [12].")
    out = mock_llm._apply_fault("paired_tags", masked)
    assert '<f1 v="Figure 2"></f1>' in out              # 첫 태그 — 빈 쌍
    assert "</f2>" in out and "</m3>" in out             # 나머지 — 내용을 감싼 쌍
    _restored, missing, dup = unmask(out, mapping, masked)
    assert not missing and "</f2>" in dup                # 엔진은 내용 쌍을 거부한다
    single, smap = mask("See [12] for the details of the proposed method.")
    restored, missing, dup = unmask(mock_llm._apply_fault("paired_tags", single), smap, single)
    assert not missing and not dup and "[12]" in restored  # 빈 쌍은 자기 닫힘으로 접힌다


# ───────────────── 포트 고정 (정해진 포트 범위에서 돌리기) ─────────────────

def test_port_options_reject_a_collision_before_starting_anything():
    import subprocess

    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_e2e.py"), "--mock-port", "18999",
         "--api-port", "18999", "--work", "/nonexistent/should-not-be-created"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 2 and "달라야" in r.stderr
    assert not pathlib.Path("/nonexistent/should-not-be-created").exists()
