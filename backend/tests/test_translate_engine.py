"""엔진 — 미니 잡 디렉터리에서 조립·캐시·재시도·취소·상태 전이 검증.

torch/OCR 없이 requests+표준 라이브러리만으로 도는지도 함께 확인한다(스텁 클라이언트).
"""

import json
import re
import threading

import pytest

from app.translate.engine import run_translation
from app.translate.masking import mask, unmask
from app.translate.prompts import build_unit_prompt
from app.translate.types import TranslateConfig, TranslateResult

SEP = "\n\n---\n\n"

# 영문 단어만 같은 길이의 한글로 바꾸는 치환 — 플레이스홀더는 첫 그룹으로 먼저 잡아 보존.
_KO_ECHO_RE = re.compile(r"(<[mkgucft]\d+\b[^>]*>)|([A-Za-z]+)")


def koreanize(text: str) -> str:
    """구조는 그대로 두고 영문 단어만 한글로 바꾼다 (가짜 client 공용 스텁).

    엔진의 출력 측 검증(masking.looks_untranslated)이 영문 echo를 거부하므로
    "원문 구조를 그대로 되돌리되 한국어로 번역된 것처럼 보이는" 스텁이 필요하다.
    길이비 1.0·한글 비율 높음으로 검증을 통과하면서 마스킹 왕복·조립 불변식은
    종전 EchoClient와 똑같이 검증된다.
    """
    return _KO_ECHO_RE.sub(lambda m: m.group(1) or _ko_word(m.group(2)), text)


def _ko_word(word: str) -> str:
    """영단어 → 길이가 같은 **결정적이고 단어마다 다른** 한글.

    종전에는 `"가" * len(word)`였는데, 길이만 같으면 서로 다른 원문이 완전히 같은
    출력으로 수렴한다(실측: 길이가 같은 토큰만 다른 30개 문장 → 출력 30개 전부 동일).
    엔진의 문서 단위 축퇴 방어가 이를 정당하게 잡아내므로, 스텁이 실제 번역기처럼
    "다른 입력 → 다른 출력"을 지키게 한다. 길이는 그대로라 길이비 단언은 불변이다.
    """
    h = 0
    for ch in word.lower():
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return "".join(chr(0xAC00 + ((h >> (i * 3)) + i) % 11172) for i in range(len(word)))


def ko_expected(md_text: str) -> str:
    """EchoClient로 번역했을 때의 result.ko.md 기대값 — 원문에 같은 규칙을 적용."""
    masked, mapping = mask(md_text)
    restored, _missing, _dup = unmask(koreanize(masked), mapping)
    return restored


RESULT_MD = (
    "# Deep Learning\n\n"
    "We train a model with loss $L = \\sum_i x_i$ over the dataset.\n\n"
    "---\n\n"
    "## Results\n\n"
    "The accuracy improved on the benchmark dataset.\n"
)

LAYOUT = [
    {"page": 1, "width": 1000, "height": 1400, "fonts_v": "2", "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Deep Learning", "fs": 2.5, "bold": True},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "We train a model over data.", "fs": 1.78},
        {"type": "image", "bbox": [0, 320, 500, 700], "content": "", "image": "p0001_0.jpg"},
    ]},
    {"page": 2, "width": 1000, "height": 1400, "blocks": [
        {"type": "title", "bbox": [0, 0, 999, 80], "content": "Results", "fs": 2.5},
        {"type": "text", "bbox": [0, 100, 999, 300], "content": "The accuracy improved a lot."},
    ]},
]


def _marker(user: str) -> str | None:
    tag = "[번역할 원문]\n"
    return user.split(tag, 1)[1] if tag in user else None


class EchoClient:
    """[번역할 원문] 섹션의 구조를 그대로 되돌리되 영문만 한글로 바꿔 반환한다.

    마스킹 왕복은 원문과 동일하게 검증되고, 출력 측 검증(looks_untranslated)도
    통과한다 — 종전의 순수 echo는 '영문 그대로'라 이제 번역 실패로 취급된다."""

    def __init__(self):
        self.calls = 0
        self.unit_calls = 0  # 유닛 번역 호출만 (용어집 프롬프트 제외)
        self.api_mode_used = "chat"
        self._count_lock = threading.Lock()

    def _count(self, is_unit: bool) -> None:
        """동시 worker에서도 안전하게 센다 (하위 스텁이 super()로 재사용)."""
        with self._count_lock:
            self.calls += 1
            if is_unit:
                self.unit_calls += 1

    def complete(self, system, user, *, max_tokens):
        src = _marker(user)
        self._count(src is not None)
        if src is None:
            return ""  # 용어집 프롬프트 → 빈 응답(시드 폴백)
        return koreanize(src)


class MarkerClient(EchoClient):
    """각 줄 앞에 § 를 붙임 — 번역 반영 확인용(플레이스홀더는 보존)."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        return "\n".join("§" + koreanize(ln) for ln in src.split("\n"))


class FaultyClient(EchoClient):
    """플레이스홀더를 떨어뜨림 → 래더(repair·분할) 실패 시 원문 유지되어야 함.

    repair 프롬프트엔 [번역할 원문] 마커가 없어 _marker가 None → 빈 응답(repair도 실패)."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", koreanize(src))


def _repair_src(user: str) -> str | None:
    """repair 프롬프트에서 마스킹 원문을 되뽑는다 (테스트용). 헤더는 한 줄 가정."""
    if "[수정할 번역문]" not in user:
        return None
    head = user.split("[수정할 번역문]", 1)[0]          # "[원문 ...]\n{masked}\n\n"
    parts = head.split("\n", 1)                          # 첫 줄(헤더) 분리
    return parts[1].strip() if len(parts) > 1 else ""


class RepairClient(EchoClient):
    """최초 패스엔 태그 소실, repair 패스엔 원문(태그 포함) 복원 → step1에서 복구."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        rsrc = _repair_src(user)
        if rsrc is not None:
            return koreanize(rsrc)                        # repair: 태그 그대로 살려 반환
        src = _marker(user)
        if src is None:
            return ""
        # 최초: 태그 전부 소실
        return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", koreanize(src))


class SplitClient(EchoClient):
    """태그 2개↑면 전부 소실(전체·repair 실패), 1개↓면 보존(반쪽 성공) → step2 분할 복구."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        if _repair_src(user) is not None:
            return ""                                     # repair 실패 유도
        src = _marker(user)
        if src is None:
            return ""
        tags = re.findall(r"<[mkgucft]\d+\b[^>]*>", src)
        if len(tags) >= 2:
            return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", koreanize(src))  # 여러 태그 → 소실
        return koreanize(src)                             # 0~1개 → 보존(성공)


class DelimiterClient(EchoClient):
    """태그는 보존하되 모델 발명 딜리미터·<PAGE>를 섞음 → sanitize가 걷어내야 함."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        return koreanize(src) + " \\(추가\\) $$없음$$ <PAGE>"


class TableSplitClient(EchoClient):
    """태그 20개↑ 유닛은 소실(전체·repair 실패), 미만은 보존 — 초대형 HTML 표가
    </tr> 행 경계 분할(2a)로만 복구되는 실측 시나리오(6.4KB 표) 재현."""

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        if _repair_src(user) is not None:
            return ""  # repair 실패 유도
        src = _marker(user)
        if src is None:
            return ""
        tags = re.findall(r"<[mkgucft]\d+\b[^>]*>", src)
        if len(tags) >= 20:
            return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", koreanize(src))
        return koreanize(src)


@pytest.fixture
def cfg() -> TranslateConfig:
    return TranslateConfig(
        base_url="https://host/v1", api_key="k", model="test-model",
        api_mode="chat", concurrency=2, temperature="0", max_tokens_param="max_tokens",
        context=False,  # 결정성 위해 컨텍스트 비활성(캐시 키엔 무영향)
    )


@pytest.fixture
def job(tmp_path):
    (tmp_path / "result.md").write_text(RESULT_MD, encoding="utf-8")
    (tmp_path / "layout.json").write_text(json.dumps(LAYOUT, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _state(job) -> dict:
    return json.loads((job / "translations" / "ko" / "state.json").read_text(encoding="utf-8"))


def _report(job) -> dict:
    return json.loads((job / "translations" / "ko" / "report.json").read_text(encoding="utf-8"))


def _run_md(tmp_path, cfg, md, client):
    """layout 없이 result.md만 두고 번역 → (result, report, result.ko.md 텍스트)."""
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    res = run_translation(tmp_path, "ko", cfg, client=client)
    return res, _report(tmp_path), (tmp_path / "result.ko.md").read_text(encoding="utf-8")


def test_title_prompt_prevents_ui_label_style_abbreviation():
    prompt = build_unit_prompt(
        "How to Read a Paper",
        [],
        [],
        unit_kind="title",
    )
    assert "[블록 유형 — 제목]" in prompt
    assert "지나치게 짧은 명사구" in prompt
    assert prompt.endswith("[번역할 원문]\nHow to Read a Paper")


def test_echo_결과_구조동일_및_레이아웃_필드보존(job, cfg):
    res = run_translation(job, "ko", cfg, client=EchoClient())
    assert isinstance(res, TranslateResult) and res.status == "done"

    # result.ko.md == 원문에 같은 치환을 적용한 결과 (구조·수식·페이지 구분자 바이트 동일)
    assert (job / "result.ko.md").read_text(encoding="utf-8") == ko_expected(RESULT_MD)

    # layout.ko.json: content만 번역되고 그 외 필드는 원본과 완전 동일
    out = json.loads((job / "layout.ko.json").read_text(encoding="utf-8"))
    expected_layout = json.loads(json.dumps(LAYOUT))
    for page in expected_layout:
        for block in page["blocks"]:
            if "image" not in block:
                block["content"] = koreanize(block["content"])
    assert out == expected_layout

    st = _state(job)
    assert st["status"] == "done" and st["current"] == st["total"] == res.total
    from app.translate.types import PROMPT_V
    assert st["model"] == "test-model" and st["prompt_v"] == PROMPT_V
    assert res.kept_original == [] and res.translated > 0


def test_marker_md와_layout에_반영_필드보존(job, cfg):
    res = run_translation(job, "ko", cfg, client=MarkerClient())
    assert res.status == "done"

    md = (job / "result.ko.md").read_text(encoding="utf-8")
    assert "§" in md
    assert len(md.split(SEP)) == 2                      # 페이지 수 보존
    assert md.count("$L = \\sum_i x_i$") == 1           # 플레이스홀더 복원됨

    out = json.loads((job / "layout.ko.json").read_text(encoding="utf-8"))
    title = out[0]["blocks"][0]
    assert title["content"] == "§" + koreanize("Deep Learning")
    assert title["bbox"] == [0, 0, 999, 80] and title["fs"] == 2.5 and title["bold"] is True
    assert out[0]["fonts_v"] == "2"
    # 이미지 블록은 손대지 않음
    assert out[0]["blocks"][2]["content"] == "" and out[0]["blocks"][2]["image"] == "p0001_0.jpg"


def test_캐시_2회차_호출없음(job, cfg):
    run_translation(job, "ko", cfg, client=EchoClient())
    echo2 = EchoClient()
    res2 = run_translation(job, "ko", cfg, client=echo2)
    assert echo2.calls == 0                    # 용어집 로드 + 전 유닛 캐시 적중
    assert res2.cached == res2.total and res2.translated == 0
    assert (job / "result.ko.md").read_text(encoding="utf-8") == ko_expected(RESULT_MD)


def test_faulty_재시도후_원문유지(job, cfg):
    faulty = FaultyClient()
    res = run_translation(job, "ko", cfg, client=faulty)
    assert res.status == "done"
    # 수식 플레이스홀더가 있는 유닛(md:0:1)은 복원 실패 → 원문 유지
    assert "md:0:1" in res.kept_original
    # retried는 report.json에 기록된다 (TranslateResult엔 없음)
    report = json.loads((job / "translations" / "ko" / "report.json").read_text(encoding="utf-8"))
    assert report["retried"] >= 1
    assert report["kept_original"] == res.kept_original
    assert "md:0:1" in report["kept_original"]
    # 결과 md에는 원문 수식이 그대로 남아있어야 함
    assert "$L = \\sum_i x_i$" in (job / "result.ko.md").read_text(encoding="utf-8")


def test_취소_사전set_canceled(job, cfg):
    ev = threading.Event()
    ev.set()
    res = run_translation(job, "ko", cfg, client=EchoClient(), cancel=ev)
    assert res.status == "canceled"
    assert not (job / "result.ko.md").exists()   # 조립 전에 중단
    assert _state(job)["status"] == "canceled"


def test_취소_래더단계_사이_감지_repair_호출없음(tmp_path, cfg):
    """최초 패스 직후 cancel이 set되면 repair/분할 호출 없이 canceled로 끝난다 —
    거대 표 래더(유닛당 수 분)가 취소 후에도 이어지던 응답성 문제의 회귀 방지."""
    ev = threading.Event()

    class CancelAfterFirst(EchoClient):
        """최초 패스에서 태그를 소실시키며 cancel을 set — 래더 진입 직전 취소."""

        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            src = _marker(user)
            if src is None:
                return ""
            ev.set()
            return re.sub(r"<[mkgucft]\d+\b[^>]*>", "", koreanize(src))

    md = "The loss $L$ is minimized during training.\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    client = CancelAfterFirst()
    res = run_translation(tmp_path, "ko", cfg, client=client, cancel=ev)
    assert res.status == "canceled"
    assert client.calls == 1                     # repair(2번째 호출)에 진입하지 않음
    assert res.kept_original == []               # 취소는 kept 통계를 오염시키지 않는다


def test_잡삭제_경합시_예외없이_canceled(job, cfg):
    """번역 도중 잡 디렉터리가 삭제(DELETE 경합)돼도 예외 없이 canceled로 끝난다 —
    state/캐시 기록은 FileNotFoundError를 무시(best-effort)."""
    import shutil

    ev = threading.Event()

    class DeletingClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is not None and not ev.is_set():
                shutil.rmtree(job, ignore_errors=True)  # DELETE 경합 재현
                ev.set()                                # api.delete_job의 cancel 전파 재현
            return super().complete(system, user, max_tokens=max_tokens)

    res = run_translation(job, "ko", cfg, client=DeletingClient(), cancel=ev)
    assert res.status == "canceled"
    assert not job.exists()  # 상태·캐시 기록이 지워진 잡 디렉터리를 되살리지 않는다


def test_force_재번역(job, cfg):
    run_translation(job, "ko", cfg, client=EchoClient())
    forced = EchoClient()
    res = run_translation(job, "ko", cfg, client=forced, force=True)
    assert forced.calls > 0                       # 캐시 무시하고 재번역
    assert res.translated == res.total and res.cached == 0
    assert (job / "result.ko.md").read_text(encoding="utf-8") == ko_expected(RESULT_MD)


def test_result_md_없으면_에러(tmp_path, cfg):
    from app.translate.types import TranslateError
    with pytest.raises(TranslateError, match="번역할 결과가 없습니다"):
        run_translation(tmp_path, "ko", cfg, client=EchoClient())
    assert _state(tmp_path)["status"] == "error"


def test_progress_콜백(job, cfg):
    seen = []
    run_translation(job, "ko", cfg, client=EchoClient(), progress=lambda c, t: seen.append((c, t)))
    assert seen and seen[-1][0] == seen[-1][1]     # 마지막 current == total
    assert all(t == seen[-1][1] for _, t in seen)


def test_layout_없어도_동작(tmp_path, cfg):
    (tmp_path / "result.md").write_text(RESULT_MD, encoding="utf-8")
    res = run_translation(tmp_path, "ko", cfg, client=EchoClient())
    assert res.status == "done"
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(RESULT_MD)
    assert not (tmp_path / "layout.ko.json").exists()


def test_그림_블록뿐인_layout은_없는_것과_같다(tmp_path, cfg):
    """figure_only 엔진의 옛 잡(image 블록뿐인 layout.json)은 layout 없는 잡처럼 result.md만
    번역한다 — layout.ko.json을 만들면 그 잡이 좌표 내보내기 대상처럼 보였다
    (artifacts.has_usable_layout과 같은 기준)."""
    (tmp_path / "result.md").write_text(RESULT_MD, encoding="utf-8")
    image_only = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "image", "bbox": [0, 320, 500, 700], "content": "", "image": "p0001_0.jpg"},
    ]}]
    (tmp_path / "layout.json").write_text(json.dumps(image_only), encoding="utf-8")
    res = run_translation(tmp_path, "ko", cfg, client=EchoClient())
    assert res.status == "done"
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(RESULT_MD)
    assert not (tmp_path / "layout.ko.json").exists()


def test_삭제된_잡_디렉터리를_되살리지_않는다(tmp_path, cfg):
    """DELETE·TTL GC와 겹친 번역이 parents=True로 잡 디렉터리를 되살려 meta.json 없는
    고아(translations/만 든 디렉터리)를 남겼다 — 없으면 만들지 않고 실패한다."""
    from app.translate.types import TranslateError

    gone = tmp_path / "j_0123456789ab"
    with pytest.raises(TranslateError, match="삭제된"):
        run_translation(gone, "ko", cfg, client=EchoClient())
    assert not gone.exists()

    # 잡 디렉터리는 있고 translations/가 아직 없으면 정상적으로 만든다
    alive = tmp_path / "j_aaaaaaaaaaaa"
    alive.mkdir()
    (alive / "result.md").write_text(RESULT_MD, encoding="utf-8")
    assert run_translation(alive, "ko", cfg, client=EchoClient()).status == "done"
    assert (alive / "translations" / "ko" / "state.json").is_file()


def test_실행_중_잡이_삭제되면_디렉터리를_되살리지_않고_실패한다(tmp_path, cfg):
    """번역 도중 잡이 지워져도(rmtree) 상태·캐시·산출물 기록이 디렉터리를 다시 만들지 않는다."""
    import shutil

    from app.translate.types import TranslateError

    job_dir = tmp_path / "j_bbbbbbbbbbbb"
    job_dir.mkdir()
    (job_dir / "result.md").write_text(RESULT_MD, encoding="utf-8")

    class DeletingClient(EchoClient):
        def complete(self, *args, **kwargs):
            shutil.rmtree(job_dir, ignore_errors=True)
            return super().complete(*args, **kwargs)

    with pytest.raises(TranslateError):
        run_translation(job_dir, "ko", cfg, client=DeletingClient())
    assert not job_dir.exists()


# ── 신뢰도 래더 (kept_original → 0) ─────────────────────────────────────────

def test_래더_repair로_복구_kept0(tmp_path, cfg):
    """(a) 최초 태그 소실 → repair 패스 성공 → translated, report.repaired==1, kept 0."""
    md = "# Title\n\nThe loss $L$ is minimized during the training here.\n"
    res, report, _ = _run_md(tmp_path, cfg, md, RepairClient())
    assert res.status == "done"
    assert res.kept_original == []
    assert report["repaired"] == 1 and report["retried"] >= 1 and report["split"] == 0


def test_래더_분할로_복구_kept0(tmp_path, cfg):
    """(b) 전체 실패(태그 2개 소실)·repair 실패 → 문장 분할로 반쪽씩 성공 → split==1."""
    md = "The value $a$ is here. The value $b$ is there.\n"
    res, report, md_out = _run_md(tmp_path, cfg, md, SplitClient())
    assert res.status == "done"
    assert res.kept_original == []
    assert report["split"] == 1 and report["retried"] >= 1
    assert "$a$" in md_out and "$b$" in md_out  # 두 수식 모두 복원


def test_래더_구조유닛_분할안함_원문유지(tmp_path, cfg):
    """(c) 표(구조 유닛)는 문장 경계가 있어도 분할하지 않는다 → kept_original."""
    md = (
        "| Description column | Result value here |\n"
        "| --- | --- |\n"
        "| First sentence. Second $E$ sentence here | plain data row |\n"
    )
    res, report, md_out = _run_md(tmp_path, cfg, md, FaultyClient())
    assert res.status == "done"
    assert "md:0:0" in res.kept_original      # 표 유닛 원문 유지
    assert report["split"] == 0               # 분할 시도 자체 없음
    assert "$E$" in md_out                    # 원문 수식 보존


def test_래더_발명딜리미터_sanitize_제거(tmp_path, cfg):
    """(d) 출력에 \\(x\\)·$$·<PAGE> 섞여도 최종 결과엔 딜리미터 없음 + sanitized>0."""
    md = "The final result $R$ is good enough.\n"
    res, report, md_out = _run_md(tmp_path, cfg, md, DelimiterClient())
    assert res.status == "done"
    assert res.kept_original == []            # 태그 보존돼 최초 패스 성공
    assert report["sanitized"] > 0
    for delim in ("\\(", "\\)", "\\[", "\\]", "$$", "<PAGE>"):
        assert delim not in md_out
    assert "$R$" in md_out                    # 실제 인라인 수식은 마스킹으로 보호돼 살아남음


def test_kept_original_확정시_warning_로그(tmp_path, cfg, caplog):
    """유닛이 최종 원문 유지로 확정되면 warning 1줄 — 유닛 id 등 식별자만 남고
    문서 원문 내용은 로그에 없어야 한다."""
    import logging

    md = "The loss $L$ is minimized during the training here.\n"
    with caplog.at_level(logging.WARNING, logger="app.translate.engine"):
        res, _, _ = _run_md(tmp_path, cfg, md, FaultyClient())
    assert res.kept_original == ["md:0:0"]
    warned = [r.message for r in caplog.records if "원문 유지" in r.message]
    assert len(warned) == 1 and "md:0:0" in warned[0]
    assert "loss" not in warned[0]            # 원문 내용 무기록


# ── 실패 경로 뒷정리 (2026-07-14 감사: 분할의 수식 훼손·오류 시 큐 드레인) ────

def test_split_two_수식블록_내부경계는_안자름():
    """$$ 블록 내부 문장 경계(x_i. 뒤)는 분할 후보에서 제외 — 토큰 한가운데를 자르면
    반쪽의 짝 잃은 $$가 mask()에 안 잡혀 원시 LaTeX 노출→sanitize 삭제(조용한 훼손)."""
    from app.translate.engine import _split_two
    from app.translate.masking import mask

    src = (
        "The estimator follows. $$\n"
        "\\hat{\\theta} = \\arg\\min_i x_i. \\text{Then the bound holds.}\n"
        "$$ This completes the proof of the theorem."
    )
    halves = _split_two(src)
    assert halves is not None                    # 토큰 밖 경계(follows. 뒤)에서는 분할 가능
    for half in halves:
        assert half.count("$$") % 2 == 0         # $$ 짝 보존 — 토큰 내부를 자르지 않았다
    math_half = next(h for h in halves if "$$" in h)
    masked, mapping = mask(math_half)
    assert mapping                               # 반쪽 mask()가 수식을 온전히 토큰으로 회수
    assert any(v.startswith("$$") and v.endswith("$$") for v in mapping.values())
    # 플레이스홀더 태그(v 미리보기 포함) 밖에는 원시 LaTeX가 남지 않는다
    import re
    residual = re.sub(r"<[mkgucft]\d+\b[^>]*>", "", masked)
    assert "$$" not in residual and "\\hat" not in residual


def test_split_two_경계가_전부_토큰내부면_분할포기():
    """문장 경계가 전부 $$ 블록 안이면 None — 호출자가 분할을 포기하고 kept 경로."""
    from app.translate.engine import _split_two

    src = "Consider $$\na = b. \\text{Then. } c = d\n$$ QED"
    assert _split_two(src) is None


def test_래더_분할이_수식블록을_훼손하지_않음(tmp_path, cfg):
    """검증자 재현: $$ 블록 내부에 문장 경계가 있는 유닛 — 종전엔 분할이 수식 한가운데를
    잘라 sanitize가 잔여 $$를 지우고 '번역 성공'으로 위장됐다. 수정 후엔 안전한 경계에서만
    자르고, 수식 반쪽 실패 시 원문 유지로 귀결돼 $$ 블록이 그대로 남는다(무손실)."""
    md = (
        "The estimator follows. $$\n"
        "\\hat{\\theta} = \\arg\\min_i x_i. \\text{Then the bound holds.}\n"
        "$$ This completes the proof of the theorem.\n"
    )
    res, report, md_out = _run_md(tmp_path, cfg, md, FaultyClient())
    assert res.status == "done"
    assert "md:0:0" in res.kept_original         # 수식 반쪽 실패 → 무손실 원문 유지
    assert md_out.count("$$") == 2               # $$ 블록 보존 — 조용한 삭제 없음
    assert "\\hat{\\theta}" in md_out
    assert report["split"] == 0                  # 분할 '성공'으로 위장되지 않는다


def test_step0_API오류시_큐드레인_방지_및_부분캐시_flush(tmp_path, cfg):
    """step-0 API 오류는 잡 전체 실패로 전파(기존 계약)하되, 남은 futures 취소 + abort로
    큐 드레인(죽은 엔드포인트 × 유닛별 백오프)을 막는다. flush_cache는 finally라 오류
    전에 완료된 유닛 번역도 캐시에 보존된다."""
    import time

    from app.translate.types import TranslateAPIError, TranslateError

    class DeadEndpointClient(EchoClient):
        """유닛 호출 4회는 에코 성공, 이후 전부 API 오류(왕복 50ms) — 죽은 엔드포인트."""

        def __init__(self):
            super().__init__()
            self.unit_calls = 0
            self._lock = threading.Lock()

        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            with self._lock:
                self.calls += 1
                if src is None:
                    return ""                    # 용어집 프롬프트 → 시드 폴백
                self.unit_calls += 1
                n = self.unit_calls
            if n <= 4:
                return koreanize(src)
            time.sleep(0.05)                     # 네트워크 왕복 재현 — 메인 스레드가 abort할 틈
            raise TranslateAPIError("번역 API 연결 실패: dead endpoint")

    md = SEP.join(
        f"Paragraph number {i} explains the training procedure in detail." for i in range(30)
    ) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")

    client = DeadEndpointClient()
    with pytest.raises(TranslateError):
        run_translation(tmp_path, "ko", cfg, client=client)  # cfg.concurrency == 2

    # 드레인 방지 — 남은 큐(30유닛)가 죽은 엔드포인트를 전부 두드리지 않는다 (종전 30회)
    assert client.unit_calls < 15
    assert _state(tmp_path)["status"] == "error"
    # finally flush — 오류 전에 완료된 4유닛의 번역이 유닛 캐시에 남아있다
    units = json.loads(
        (tmp_path / "translations" / "ko" / "units.json").read_text(encoding="utf-8")
    )
    assert len(units) == 4


def test_초대형_표유닛은_행경계_분할로_복구(tmp_path, cfg):
    """HTML 표는 문장 분할 대상이 아니다 — </tr> 행 경계 분할(2a)이 잡아야 한다.
    전체(26태그)·repair 실패 → 반쪽(13태그) 성공 → 이어붙임이 원 구조와 동일."""
    import json as _json

    rows = "".join(f"<tr><td>row {i} data</td><td>value {i}</td></tr>" for i in range(4))
    md = f"# Title\n\n<table>{rows}</table>\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")

    res = run_translation(tmp_path, "ko", cfg, client=TableSplitClient())
    assert res.kept_original == []
    # Echo 기반이라 성공 경로는 원문 복원 — 표 구조가 바이트 그대로 살아야 한다
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)
    rep = _json.loads((tmp_path / "translations" / "ko" / "report.json").read_text(encoding="utf-8"))
    assert rep["split"] == 1


def test_중복_유닛은_한_번만_API를_탄다_single_flight(tmp_path, cfg):
    """같은 문단이 문서에 두 번 나오면 API 왕복도 한 번이어야 한다.

    예전에는 '먼저 끝난 유닛이 메인 루프에서 캐시에 기록되고 늦게 시작한 중복이
    그걸 읽는' 레이스에 기대고 있어, 동시성을 올릴수록 같은 문단을 두 번 번역했다.
    두 워커가 확실히 겹치도록 응답을 지연시켜 그 창을 강제로 연다."""
    from dataclasses import replace

    import time as _time

    class SlowEcho(EchoClient):
        def complete(self, system, user, *, max_tokens):
            if _marker(user) is not None:
                # waiter의 옛 deadline(아래 timeout_s=0.05)보다 길게 붙잡는다.
                # deadline이 있으면 0.5초 첫 poll 뒤 같은 원문을 다시 API로 보낸다.
                _time.sleep(0.65)
            return super().complete(system, user, max_tokens=max_tokens)

    dup = "The accuracy improved a lot on every benchmark we tried."
    unique = "A separate paragraph explains the evaluation protocol in detail."
    # 첫 두 target을 같은 키로 둬 concurrency=2의 두 worker가 owner/waiter로
    # 반드시 겹치게 한다. 제목 사이에 끼우면 실행 순서에 따라 cache hit만 검증된다.
    md = f"{dup}\n\n{dup}\n\n{unique}\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")

    client = SlowEcho()
    res = run_translation(
        tmp_path, "ko", replace(cfg, timeout_s=0.05), client=client,
    )

    assert res.status == "done"
    assert res.cached == 1, f"중복 유닛이 캐시로 처리되지 않았다 (cached={res.cached})"
    # 고유 유닛은 2개 — 중복 본문이 두 번 호출되면 3이 된다.
    assert client.unit_calls == 2, f"중복 유닛이 API를 두 번 탔다 (calls={client.unit_calls})"
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_마스킹_미리보기가_같은_다른_수식은_캐시를_공유하지_않음(tmp_path, cfg):
    """긴 수식의 첫 12자가 같아 masked source가 동일해도 원문 전체를
    키에 포함해야 다른 복원 결과를 single-flight/cache로 재사용하지 않는다.

    종전 키는 두 문단을 같게 봐 API를 1회만 호출하고, 두 번째의
    ``+2`` 수식을 첫 번째의 ``+1``로 바꾸는 실제 데이터 훼손을 일으켰다.
    """
    class FormulaEcho:
        api_mode_used = "chat"

        def __init__(self):
            self.unit_calls = 0

        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is None:
                return ""
            self.unit_calls += 1
            return koreanize(src)

    first = "This method carefully uses $abcdefghijklm+1$ in every evaluation."
    second = "This method carefully uses $abcdefghijklm+2$ in every evaluation."
    md = f"{first}\n\n{second}\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    client = FormulaEcho()

    res = run_translation(tmp_path, "ko", cfg, client=client)

    assert res.status == "done"
    assert res.translated == 2 and res.cached == 0
    assert client.unit_calls == 2
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_같은_문장이라도_직전_문맥이_다르면_캐시를_공유하지_않음(tmp_path, cfg):
    """context_tail은 실제 프롬프트 입력이므로 single-flight/cache 키에 포함한다."""
    from dataclasses import replace

    class ContextEcho:
        api_mode_used = "chat"

        def __init__(self):
            self.unit_calls = 0

        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is None:
                return ""
            self.unit_calls += 1
            return koreanize(src)

    repeated = "This ambiguous result should be interpreted using its preceding context."
    md = (
        "The first experiment concerns image classification accuracy.\n\n"
        f"{repeated}\n\n"
        "The second experiment concerns language-model calibration error.\n\n"
        f"{repeated}\n"
    )
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    client = ContextEcho()

    res = run_translation(
        tmp_path, "ko", replace(cfg, context=True), client=client,
    )

    assert res.status == "done"
    assert res.translated == 4 and res.cached == 0
    assert client.unit_calls == 4
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_single_flight_owner_API오류도_waiter가_중복호출하지_않음(tmp_path, cfg):
    """owner의 치명 오류는 waiter에게 공유되어 같은 죽은 endpoint를 다시 치지 않는다."""
    import time as _time

    from app.translate.types import TranslateAPIError, TranslateError

    class FailingClient(EchoClient):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.unit_calls = 0

        def complete(self, system, user, *, max_tokens):
            if _marker(user) is None:
                return ""  # glossary seed 폴백
            with self.lock:
                self.unit_calls += 1
            _time.sleep(0.05)  # 두 번째 worker가 flight waiter가 될 틈
            raise TranslateAPIError("dead endpoint")

    dup = "The same paragraph reaches a temporarily unavailable translation endpoint."
    (tmp_path / "result.md").write_text(f"{dup}\n\n{dup}\n", encoding="utf-8")
    client = FailingClient()

    with pytest.raises(TranslateError, match="dead endpoint"):
        run_translation(tmp_path, "ko", cfg, client=client)

    assert client.unit_calls == 1, "waiter가 owner 오류 뒤 같은 API 요청을 다시 시작했다"


def test_single_flight_cache_flush가_worker_publish와_경쟁하지_않음(
    tmp_path, cfg, monkeypatch,
):
    """주기 units.json 저장은 live dict가 아닌 잠금 아래 snapshot을 직렬화한다."""
    from dataclasses import replace
    import time as _time

    import app.translate.engine as engine

    release = threading.Event()
    triggered = threading.Event()

    class Coordinated(EchoClient):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.unit_calls = 0

        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is None:
                return ""
            with self.lock:
                self.unit_calls += 1
                n = self.unit_calls
            if n > 10:
                assert release.wait(3), "periodic units.json flush가 오지 않았다"
            return koreanize(src)

    original_write = engine._atomic_write_json

    def coordinated_write(path, obj):
        if path.name == "units.json" and len(obj) >= 10 and not triggered.is_set():
            triggered.set()
            iterator = iter(obj)
            next(iterator)
            initial = len(obj)
            release.set()  # 나머지 worker의 cache publish를 flush 도중 겹치게 한다
            deadline = _time.monotonic() + 0.2
            while len(obj) == initial and _time.monotonic() < deadline:
                _time.sleep(0.005)
            # live cache였다면 worker의 key 추가 뒤 RuntimeError, snapshot이면 안정적.
            list(iterator)
        original_write(path, obj)

    monkeypatch.setattr(engine, "_atomic_write_json", coordinated_write)
    words = [f"token{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(30)]
    md = SEP.join(
        f"This {word} describes a unique translation procedure carefully."
        for word in words
    )
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    client = Coordinated()
    try:
        res = run_translation(
            tmp_path, "ko", replace(cfg, concurrency=8), client=client,
        )
    finally:
        release.set()  # assertion/error 경로에서도 대기 worker를 회수한다

    assert triggered.is_set()
    assert res.status == "done" and res.translated == 30
    assert client.unit_calls == 30
    units = json.loads(
        (tmp_path / "translations" / "ko" / "units.json").read_text(encoding="utf-8")
    )
    assert len(units) == 30


def test_single_flight_대기중_취소는_추가_API를_시작하지_않음(tmp_path, cfg):
    """선점 스레드를 기다리는 중복 유닛은 취소 뒤 별도 API 요청을 시작하지 않는다."""
    import threading as _th
    import time as _time

    ev = _th.Event()

    class BlockingEcho(EchoClient):
        def complete(self, system, user, *, max_tokens):
            if _marker(user) is not None:
                ev.set()          # 유닛 번역 진입을 알리고
                _time.sleep(1.2)  # 대기자가 취소를 만나도록 붙잡아 둔다
            return super().complete(system, user, max_tokens=max_tokens)

    dup = "The accuracy improved a lot on every benchmark we tried."
    md = f"{dup}\n\n{dup}\n\nA final unique paragraph remains.\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")

    cancel = _th.Event()
    stop = _th.Timer(0.3, cancel.set)
    stop.start()
    started = _time.monotonic()
    client = BlockingEcho()
    res = run_translation(tmp_path, "ko", cfg, client=client, cancel=cancel)
    stop.cancel()

    assert res.status == "canceled"
    assert ev.is_set(), "owner가 API에 진입하기 전에 테스트가 취소됐다"
    assert client.unit_calls == 1, "대기 중복 유닛이 별도 API 요청을 시작했다"
    # requests의 이미 진행 중인 owner 호출 자체는 중단할 수 없지만, waiter는 새 호출을
    # 만들지 않고 executor가 owner를 회수한 뒤 제한 시간 안에 canceled로 끝난다.
    assert _time.monotonic() - started < 5


# ── 출력 측 검증 (거부문·요약·영문 echo가 문단을 대체하는 것 차단) ──────────

@pytest.mark.parametrize("refusal", [
    "I cannot translate this text.",          # 영문 거부문 → 한글 비율 검사
    "죄송합니다, 번역할 수 없습니다.",         # 한국어 거부문 → 길이비 검사
])
def test_모델_거부문은_원문유지되고_캐시에_남지_않는다(tmp_path, cfg, refusal):
    """플레이스홀더가 온전해도 거부문은 성공으로 치지 않는다.

    종전엔 report.json이 translated=1·kept_original=[]로 정상 완료를 보고했고,
    거부문이 units.json에 캐시돼 모델을 고쳐도 force 없이는 손실이 재사용됐다."""

    class RefusingClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            if _marker(user) is None and _repair_src(user) is None:
                return ""                     # 용어집 프롬프트 → 시드 폴백
            return refusal

    md = "The accuracy improved on every benchmark dataset that we carefully evaluated.\n"
    res, report, md_out = _run_md(tmp_path, cfg, md, RefusingClient())

    assert res.status == "done"
    assert res.kept_original == ["md:0:0"] and res.translated == 0
    assert report["kept_original"] == ["md:0:0"] and report["retried"] >= 1
    assert md_out == md                       # 무손실 — 원문 그대로 남는다
    units = json.loads(
        (tmp_path / "translations" / "ko" / "units.json").read_text(encoding="utf-8")
    )
    assert units == {}                        # 거부문이 캐시를 오염시키지 않는다


def test_고유명사만_있는_짧은_유닛은_출력검증에_걸리지_않는다(tmp_path, cfg):
    """입력 측 should_skip과 대칭 — 영단어 2개 미만이면 원문 유지가 정상이라 통과."""
    md = "# Adam\n\nThe optimizer converges quickly on this benchmark task.\n"

    class PartialEcho(EchoClient):
        """제목은 원문 그대로, 본문만 한글로 — 제목이 kept로 떨어지면 안 된다."""

        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            src = _marker(user)
            if src is None:
                return ""
            return src if src.strip() == "# Adam" else koreanize(src)

    res, _report_json, md_out = _run_md(tmp_path, cfg, md, PartialEcho())
    assert res.kept_original == []
    assert "# Adam" in md_out


# ── 2단 패스 (md 유닛 + layout 유닛 이중 번역 제거) ─────────────────────────

def _layout_of(lines, page=1):
    return [{"page": page, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [0, i * 100, 999, i * 100 + 80], "content": ln}
        for i, ln in enumerate(lines)
    ]}]


def test_md줄이_layout과_일치하면_유닛당_한_번만_번역한다(tmp_path, cfg):
    """reconcile이 성공하면 md 유닛 번역은 전량 폐기된다 — 아예 호출하지 않는다."""
    lines = [
        "The accuracy improved on every benchmark dataset here.",
        "We trained the model with a very small learning rate.",
        "The evaluation protocol follows the earlier published work.",
    ]
    md = "\n\n".join(lines) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    (tmp_path / "layout.json").write_text(
        json.dumps(_layout_of(lines), ensure_ascii=False), encoding="utf-8")

    client = EchoClient()
    res = run_translation(tmp_path, "ko", cfg, client=client)

    assert res.status == "done"
    # layout 유닛 3회뿐. 종전엔 md 유닛 3회가 더 나가고 그 결과는 통째로 버려졌다.
    assert client.unit_calls == len(lines)
    assert res.total == len(lines)
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_커버리지가_낮아도_layout으로_덮이는_md유닛은_다시_번역하지_않는다(tmp_path, cfg):
    """유닛 단위 reconcile — 종전에는 문서 커버리지(1/5 < 0.7)가 낮으면 덮이는 유닛까지
    2차 패스로 다시 번역했다(같은 문장 이중 과금). 유닛의 모든 줄이 layout 번역으로
    덮이면 커버리지와 무관하게 그 번역을 쓴다(translate-llm-1)."""
    covered_line = "The accuracy improved on every benchmark dataset here."
    others = [
        "We trained the model with a very small learning rate.",
        "The evaluation protocol follows the earlier published work.",
        "Each configuration was repeated with three random seeds.",
        "The remaining sections describe the ablation experiments.",
    ]
    md = "\n\n".join([covered_line, *others]) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    (tmp_path / "layout.json").write_text(
        json.dumps(_layout_of([covered_line]), ensure_ascii=False), encoding="utf-8")

    client = EchoClient()
    res = run_translation(tmp_path, "ko", cfg, client=client)

    assert res.status == "done"
    # md 4 + layout 1 — 덮이는 md 유닛은 layout 번역을 재사용하므로 2차 패스가 없다
    assert res.total == 5 and res.kept_original == []
    assert client.unit_calls == 5
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_layout_쌍둥이가_번역에_실패한_지연_유닛은_2차_패스로_번역한다(tmp_path, cfg):
    """layout 블록이 번역에 실패하면 그 원문을 md에 '매핑'하지 않고 md 유닛을 번역한다."""
    lines = [
        "The accuracy improved on every benchmark dataset here.",
        "We trained the model with a very small learning rate.",
    ]

    class TitleEcho(EchoClient):
        """제목 블록 프롬프트에는 영문을 그대로 돌려준다(게이트 거부 → layout 실패)."""

        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is not None and "[블록 유형 — 제목]" in user:
                self._count(True)
                return src
            return super().complete(system, user, max_tokens=max_tokens)

    md = "\n\n".join(lines) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    lay = _layout_of(lines)
    lay[0]["blocks"][0]["type"] = "title"
    (tmp_path / "layout.json").write_text(json.dumps(lay, ensure_ascii=False), encoding="utf-8")

    res = run_translation(tmp_path, "ko", cfg, client=TitleEcho())

    assert res.kept_original == ["lay:1:0"]                  # PDF 쪽은 실패를 숨기지 않는다
    assert res.total == 3                                    # lay 2 + 2차 md 1
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_유닛_단위_reconcile은_영어_줄을_남기지_않는다(tmp_path, cfg):
    """한 줄 문단 8개 + 2줄 목록 — 종전에는 커버리지 0.8로 reconcile이 성공해 목록
    두 줄(여러 줄 layout 블록이라 매핑 불가)이 영어로 남았다(translate-llm-1)."""
    paras = [f"Paragraph number {w} explains one more detail of the method." for w in (
        "one", "two", "three", "four", "five", "six", "seven", "eight")]
    items = ["- first item describes the encoder", "- second item describes the decoder"]
    md = "\n\n".join([*paras, "\n".join(items)]) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    lay = _layout_of([*paras, "\n".join(items)])
    (tmp_path / "layout.json").write_text(json.dumps(lay, ensure_ascii=False), encoding="utf-8")

    res = run_translation(tmp_path, "ko", cfg, client=EchoClient())

    out = (tmp_path / "result.ko.md").read_text(encoding="utf-8")
    english = [ln for ln in out.splitlines() if ln.strip() and not re.search(r"[가-힣]", ln)]
    assert english == [], english
    assert out == ko_expected(md) and res.kept_original == []


def test_쪽번호_블록이_붙은_md문단도_지연된다(tmp_path, cfg):
    """vendor result.md는 블록을 단일 개행으로 이어 쪽 번호 줄이 문단에 붙는다. 쪽 번호
    layout 블록은 번역 대상이 아니라서 종전에는 그 문단을 '덮이지 않음'으로 보고 1차에서
    한 번 더 번역했다(버린 번역이 layout 번역량의 6~71%)."""
    body = "The accuracy improved on every benchmark dataset here."
    md = f"3\n{body}\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    lay = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "page_number", "bbox": [0, 0, 50, 20], "content": "3"},
        {"type": "text", "bbox": [0, 100, 999, 180], "content": body},
    ]}]
    (tmp_path / "layout.json").write_text(json.dumps(lay, ensure_ascii=False), encoding="utf-8")

    client = EchoClient()
    res = run_translation(tmp_path, "ko", cfg, client=client)

    assert client.unit_calls == 1 and res.total == 1         # layout 본문 블록만 번역
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


# ── step-0 결정적 4xx 강등 ──────────────────────────────────────────────────

def test_성공유닛_이후_결정적_4xx는_해당_유닛만_원문유지(tmp_path, cfg):
    """초대형 유닛 하나가 HTTP 400을 받아도 문서 전체를 실패시키지 않는다 —
    래더로 강등돼 kept_original에 남고 나머지는 정상 번역된다."""
    from dataclasses import replace

    from app.translate.types import TranslateUnitRejected

    class RejectOneClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            src = _marker(user)
            if src is None:
                return ""
            if "oversized" in src:
                raise TranslateUnitRejected("번역 API 오류 (HTTP 400): context length")
            return koreanize(src)

    md = (
        "The first paragraph explains the training procedure in detail.\n\n"
        "This oversized merged table exceeds the server context window.\n\n"
        "The last paragraph summarizes the evaluation protocol briefly.\n"
    )
    # concurrency=1 — 거부 유닛보다 앞선 유닛이 반드시 먼저 API를 탄다(결정성)
    res, report, md_out = _run_md(
        tmp_path, replace(cfg, concurrency=1), md, RejectOneClient(),
    )

    assert res.status == "done"
    assert res.kept_original == ["md:0:1"] and res.translated == 2
    assert report["retried"] == 1
    assert "This oversized merged table exceeds the server context window." in md_out
    assert _state(tmp_path)["status"] == "done"


def test_첫_유닛부터_4xx면_종전대로_잡_전체_실패(tmp_path, cfg):
    """전역 원인(잘못된 payload·모델명)일 때 전 유닛이 kept로 조용히 done 되면 안 된다."""
    from dataclasses import replace

    from app.translate.types import TranslateError, TranslateUnitRejected

    class AlwaysRejectClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            if _marker(user) is None:
                return ""
            raise TranslateUnitRejected("번역 API 오류 (HTTP 400): bad request")

    md = "The first paragraph explains the training procedure in detail.\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    with pytest.raises(TranslateError):
        run_translation(
            tmp_path, "ko", replace(cfg, concurrency=1), client=AlwaysRejectClient(),
        )
    assert _state(tmp_path)["status"] == "error"


# ── 캐시 키에 샘플링 설정 포함 ─────────────────────────────────────────────

def test_reasoning_설정이_바뀌면_캐시를_재사용하지_않는다(job, cfg):
    """reasoning/temperature는 출력을 바꾸는 요청 파라미터다 — 캐시 키 재료."""
    from dataclasses import replace

    run_translation(job, "ko", cfg, client=EchoClient())

    same = EchoClient()
    run_translation(job, "ko", cfg, client=same)
    assert same.calls == 0                       # 같은 설정 → 전 유닛 캐시 적중

    changed = EchoClient()
    res = run_translation(job, "ko", replace(cfg, reasoning="high"), client=changed)
    assert changed.calls > 0 and res.cached == 0 and res.translated == res.total

    hot = EchoClient()
    res2 = run_translation(job, "ko", replace(cfg, temperature="0.7"), client=hot)
    assert hot.calls > 0 and res2.cached == 0


def test_reasoning_전달방식이_요청을_바꿀_때만_캐시를_무효화한다(job, cfg):
    """종전 off(OpenRouter 방식)가 무시되던 로컬 서버에서 thinking이 실제로 꺼지면
    출력이 달라진다 — 그 캐시는 재사용하지 않는다. 같은 요청이면 키는 그대로다."""
    from dataclasses import replace

    legacy = replace(cfg, reasoning="off", reasoning_style="openrouter")
    run_translation(job, "ko", legacy, client=EchoClient())
    assert _state(job)["reasoning_style"] == "openrouter"
    assert _state(job)["request_variant"] == ""

    # 공개 게이트웨이의 auto는 openrouter로 확정 = 종전과 같은 요청 → 전량 적중
    # (fixture의 단일 라벨 호스트 "host"는 auto가 로컬로 보므로 공개 호스트를 쓴다)
    same = EchoClient()
    gateway = replace(legacy, reasoning_style="auto", base_url="https://gateway.example.com/v1")
    run_translation(job, "ko", gateway, client=same)
    assert same.calls == 0

    # chat_template_kwargs로 실제 끄기 → 요청이 달라졌으니 다시 번역한다
    local = EchoClient()
    res = run_translation(
        job, "ko", replace(legacy, reasoning_style="chat_template_kwargs"), client=local,
    )
    assert local.calls > 0 and res.cached == 0
    assert _state(job)["reasoning_style"] == "chat_template_kwargs"
    assert _state(job)["request_variant"] == "reasoning_style=chat_template_kwargs"


# ── 관측: 사유별 skip/kept 집계 + 참고문헌 규칙 불일치 ──────────────────────

def test_report에_skip_사유별_집계가_남는다(tmp_path, cfg):
    """총합 하나로 뭉개진 skipped로는 '왜 영어로 남았나'를 알 수 없다 —
    references / already-korean / non-linguistic을 구분해 남긴다."""
    md = (
        "The accuracy improved on the benchmark dataset that we evaluated.\n\n"
        "이 문단은 이미 한국어라 번역 대상이 아니다.\n\n"
        "1234 5678 90\n\n"
        "## References\n\n"
        "[1] Author A. A paper title. Venue, 2020.\n"
    )
    res, report, _md_out = _run_md(tmp_path, cfg, md, EchoClient())

    assert res.status == "done"
    reasons = report["skip_reasons"]
    assert reasons["references"] == 2          # heading + 서지 항목
    assert reasons["already-korean"] == 1
    assert reasons["non-linguistic"] == 1
    assert sum(reasons.values()) == report["skipped"] == res.skipped


def test_report에_kept_사유가_구분된다(tmp_path, cfg):
    """게이트 거부(gate-rejected)와 플레이스홀더 정합 실패를 구분한다."""

    class RefusingClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            if _marker(user) is None and _repair_src(user) is None:
                return ""
            return "I cannot translate this text."

    md = "The accuracy improved on every benchmark dataset that we evaluated.\n"
    _res, report, _out = _run_md(tmp_path, cfg, md, RefusingClient())
    assert report["kept_original"] == ["md:0:0"]
    assert report["kept_reasons"] == {"gate-rejected": 1}


def test_플레이스홀더_소실은_gate_rejected와_다르게_집계된다(tmp_path, cfg):
    md = "We train a model with loss $L = \\sum_i x_i$ over the dataset.\n"
    _res, report, _out = _run_md(tmp_path, cfg, md, FaultyClient())
    assert report["kept_original"] == ["md:0:0"]
    assert report["kept_reasons"] == {"placeholder-mismatch": 1}


def test_빈_출력은_empty_output으로_집계된다(tmp_path, cfg):
    class SilentClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            return ""

    md = "The accuracy improved on every benchmark dataset that we evaluated.\n"
    _res, report, _out = _run_md(tmp_path, cfg, md, SilentClient())
    assert report["kept_reasons"] == {"empty-output": 1}


def test_결정적_4xx_강등은_api_rejected로_집계된다(tmp_path, cfg):
    from dataclasses import replace

    from app.translate.types import TranslateUnitRejected

    class RejectOneClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            src = _marker(user)
            if src is None:
                return ""
            if "oversized" in src:
                raise TranslateUnitRejected("번역 API 오류 (HTTP 400): context length")
            return koreanize(src)

    md = (
        "The first paragraph explains the training procedure in detail.\n\n"
        "This oversized merged table exceeds the server context window.\n"
    )
    _res, report, _out = _run_md(
        tmp_path, replace(cfg, concurrency=1), md, RejectOneClient(),
    )
    assert report["kept_reasons"] == {"api-rejected": 1}


def test_참고문헌_규칙_불일치가_경고와_집계로_남는다(tmp_path, cfg):
    """layout은 ref_text로 원문 유지, md는 heading 스윕 밖이라 번역 — 같은 원문이
    PDF와 result.ko.md에서 갈라진다. 정책은 그대로 두고 관측만 한다."""
    md = (
        "The accuracy improved on the benchmark dataset that we evaluated.\n\n"
        "[1] Author A. A paper title. Venue, 2020.\n"
    )
    layout = [{"page": 1, "width": 1000, "height": 1400, "blocks": [
        {"type": "text", "bbox": [0, 0, 999, 80],
         "content": "The accuracy improved on the benchmark dataset that we evaluated."},
        {"type": "ref_text", "bbox": [0, 100, 999, 160],
         "content": "[1] Author A. A paper title. Venue, 2020."},
    ]}]
    (tmp_path / "layout.json").write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
    _res, report, _out = _run_md(tmp_path, cfg, md, EchoClient())

    assert report["reference_rule"]["layout_only"] == 1
    assert report["reference_rule"]["md_only"] == 0
    assert report["reference_rule"]["sample_units"] == ["lay:1:1"]
    assert any("참고문헌 규칙 불일치" in w for w in report["warnings"])


def test_모든_유닛에_같은_출력을_주는_공급자는_축퇴로_원문_유지(tmp_path, cfg):
    """문서 단위 축퇴 방어 — 유닛별 검증이 구조적으로 볼 수 없는 실패 모드.

    공급자가 고장 나 모든 요청에 같은 캔드 응답을 돌려주면, 짧은 유닛은 길이비·
    한글 비율 검사를 통과해 정상 번역으로 채택된다(실측: fault=summary 실행에서
    '요약입니다'가 result.ko.md에 잔존). 서로 다른 원문 여럿이 한 출력으로
    수렴하는 것 자체를 신호로 삼아 원문을 지킨다.
    """
    class CannedClient:
        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            return "요약입니다."

    md = "\n\n---\n\n".join(
        "\n\n".join([
            "Abstract", "Introduction", "Related Work", "Conclusion",
        ]) for _ in range(1)
    )
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.md").write_text(md, encoding="utf-8")

    client = CannedClient()
    res = run_translation(job, "ko", cfg, client=client, page_separator="\n\n---\n\n")

    out = (job / "result.ko.md").read_text(encoding="utf-8")
    assert "요약입니다" not in out, "축퇴 출력이 산출물에 반영되면 안 된다"
    assert res.kept_original, "축퇴 유닛은 kept_original로 관측돼야 한다"
    units = job / "translations/ko/units.json"
    if units.is_file():
        import json
        assert "요약입니다." not in json.loads(units.read_text()).values(), "캐시 오염 금지"


# 2차(deferred) 패스 축퇴 방어에서 공용으로 쓰는 캔드 응답.
_CANNED = "이 문단의 요약입니다."


class _CannedClient:
    """모든 요청(용어집 포함)에 같은 문자열을 돌려주는 고장 난 공급자."""

    def __init__(self):
        self.calls = 0

    def complete(self, *a, **k):
        self.calls += 1
        return _CANNED


def _deferred_degenerate_job(job):
    """reconcile 폴백 → 2차 패스가 반드시 도는 잡을 만든다.

    covered 6줄만 layout 블록으로 두고 md는 그 6줄(2줄짜리 문단 3개)과 layout이
    모르는 6줄을 함께 담는다. 매칭률 6/12 = 0.5 < min_coverage(0.7)이라 reconcile은
    인자로 받은 assembled를 그대로 돌려주고(폴백), 1차에서 미뤄둔 md 유닛 3개가
    2차 패스로 내려간다. 반환값은 (deferred 유닛 id, 1차 유닛 수).
    """
    covered = [
        "The model reached higher accuracy.",
        "We repeated every run three times.",
        "Each dataset used the same split.",
        "The baseline stayed clearly behind.",
        "Our tuning changed only one factor.",
        "The final table lists every score.",
    ]
    uncovered = [
        "Training used a very small rate.",
        "The ablation removed one module.",
        "We logged every metric per epoch.",
        "The appendix holds extra figures.",
        "Reviewers asked for more details.",
        "Future work explores larger data.",
    ]
    paragraphs = ["\n".join(covered[i:i + 2]) for i in range(0, len(covered), 2)]
    md = "\n\n".join([*paragraphs, *uncovered]) + "\n"
    job.mkdir(exist_ok=True)
    (job / "result.md").write_text(md, encoding="utf-8")
    (job / "layout.json").write_text(
        json.dumps(_layout_of(covered), ensure_ascii=False), encoding="utf-8")
    # 2줄짜리 md 문단 3개가 deferred, 1차는 layout 6 + 미커버 md 6 = 12유닛.
    return ["md:0:0", "md:0:1", "md:0:2"], len(covered) + len(uncovered)


def test_2차패스_deferred_유닛도_축퇴_방어를_받는다(tmp_path, cfg):
    """축퇴 스윕은 1차 dispatch와 2차(deferred) 패스 **양쪽** 뒤에 필요하다.

    1차 스윕은 deferred dispatch 이전에 끝나므로, 2차 호출이 없으면 지연된 md
    유닛의 캔드 응답만 방어를 통과해 result.{lang}.md에 그대로 실린다(1차 유닛은
    전부 원문 유지된 채 지연 유닛만 '요약'으로 대체되는 최악의 혼합 산출물).
    엔진의 2차 `_sweep_degenerate()` 호출을 지우면 이 테스트는 실패한다.
    """
    job = tmp_path / "job"
    deferred_ids, first_pass_units = _deferred_degenerate_job(job)

    client = _CannedClient()
    res = run_translation(job, "ko", cfg, client=client)

    # 2차 패스가 실제로 돌았다 — total이 deferred 수만큼 늘어난다.
    assert res.total == first_pass_units + len(deferred_ids)
    out = (job / "result.ko.md").read_text(encoding="utf-8")
    assert _CANNED not in out, "2차 패스 축퇴 출력이 산출물에 실리면 안 된다"
    for uid in deferred_ids:
        assert uid in res.kept_original, f"지연 유닛 {uid}이 축퇴로 원문 유지돼야 한다"
    report = _report(job)
    # 1차 12 + 2차 3 = 15유닛 전부 축퇴 사유로 집계된다.
    assert report["kept_reasons"]["degenerate-output"] == first_pass_units + len(deferred_ids)
    cache = json.loads(
        (job / "translations" / "ko" / "units.json").read_text(encoding="utf-8"))
    assert _CANNED not in cache.values(), "2차 축퇴 출력이 캐시에 남으면 안 된다"


class AlwaysRejectClient(EchoClient):
    """유닛 번역 요청마다 재시도 불가 4xx — 죽은 엔드포인트·모델명 오타 재현."""

    def complete(self, system, user, *, max_tokens):
        from app.translate.types import TranslateUnitRejected

        self.calls += 1
        if _marker(user) is None:
            return ""  # 용어집 프롬프트
        raise TranslateUnitRejected("번역 API 오류 (HTTP 400): unknown model")


_WARM_MD = (
    "The first paragraph explains the training procedure in detail.\n\n"
    "The last paragraph summarizes the evaluation protocol briefly.\n"
)
_NEW_PARA = "A newly appended paragraph describes the ablation study setup."


def test_적중하지_않는_캐시만_있으면_결정적_4xx는_잡_전체_실패(tmp_path, cfg):
    """progressed의 의미는 "이 run에서 엔드포인트가 실제로 동작했다"이다.

    캐시 적중은 API를 한 번도 타지 않으므로 엔드포인트 건강의 증거가 될 수 없다.
    캐시 파일이 있다는 사실만으로 강등을 허용하면, 모델명 오타로 전 요청이 400을
    받는 죽은 엔드포인트에서도 전 유닛이 kept_original로 강등돼 잡이 조용히 done
    된다(사용자는 영문 그대로인 산출물을 받는다). 모델명이 바뀌면 캐시 키가 전부
    달라져 **한 건도 적중하지 않으므로**, 이전 run의 성공 이력도 없는 것과 같다.
    """
    from dataclasses import replace

    from app.translate.types import TranslateError

    seq = replace(cfg, concurrency=1)
    _run_md(tmp_path, seq, _WARM_MD, EchoClient())  # units.json 적재 (model=test-model)

    # 모델명 오타 — units.json은 그대로지만 캐시 키가 전부 달라져 적중이 0이다.
    typo = replace(seq, model="test-modle")
    with pytest.raises(TranslateError):
        run_translation(tmp_path, "ko", typo, client=AlwaysRejectClient())
    assert _state(tmp_path)["status"] == "error"


def test_워밍된_재개에서_신규_유닛_4xx는_그_유닛만_원문유지(tmp_path, cfg):
    """취소·오류 후 재개(캐시 워밍)에서는 대부분이 캐시 적중이라 progressed가 서지
    않는다. 그 상태에서 신규 유닛 하나가 결정적 4xx를 받았다고 잡 전체를 하드
    실패시키면, 복구 수단이 전량 재번역뿐이라 부분 캐시 보존의 목적이 무너진다.

    이전 run의 캐시가 **이번 설정으로 실제 적중**했다면 설정 자체는 유효하므로
    그 유닛만 강등하고 done으로 끝낸다 — 단 API 성공 0건은 경고로 남는다.
    """
    from dataclasses import replace

    seq = replace(cfg, concurrency=1)
    _run_md(tmp_path, seq, _WARM_MD, EchoClient())  # 정상 run — 유닛 2개 캐시

    # 같은 설정으로 재개하되 문단 하나가 추가됐고, 그 신규 유닛만 API를 탄다.
    grown = _WARM_MD + "\n" + _NEW_PARA + "\n"
    (tmp_path / "result.md").write_text(grown, encoding="utf-8")
    dead = AlwaysRejectClient()
    res = run_translation(tmp_path, "ko", seq, client=dead)

    assert res.status == "done" and _state(tmp_path)["status"] == "done"
    assert len(res.kept_original) == 1 and res.cached == 2
    report = _report(tmp_path)
    assert report["kept_reasons"] == {"api-rejected": 1}
    assert report["cache_reused"] == 2
    assert any("성공한 API 호출이 없습니다" in w for w in report["warnings"])
    # 무손실 — 강등된 신규 문단은 원문 그대로 남고 캐시 적중분은 번역돼 있다.
    out = (tmp_path / "result.ko.md").read_text(encoding="utf-8")
    assert _NEW_PARA in out
    assert "The first paragraph" not in out


def test_콜드_캐시에서_전_요청_4xx는_즉시_잡_전체_실패(tmp_path, cfg):
    """이전 캐시가 아예 없으면 성공 이력이 전혀 없다 — 종전대로 빠르게 전파한다."""
    from dataclasses import replace

    from app.translate.types import TranslateError

    (tmp_path / "result.md").write_text(_WARM_MD + "\n" + _NEW_PARA + "\n", encoding="utf-8")
    with pytest.raises(TranslateError):
        run_translation(tmp_path, "ko", replace(cfg, concurrency=1), client=AlwaysRejectClient())
    assert _state(tmp_path)["status"] == "error"


def test_게이트_거부가_규칙별로_집계된다(tmp_path, cfg):
    """게이트 오탐은 조용하다 — 래더 왕복만 늘리고 소진되면 문단이 영어로 남는다.

    kept_reasons는 최종 결과(gate-rejected 1건)만 세므로 "어떤 규칙이 몇 번 걸었나"
    "래더가 흡수한 거부는 몇 건인가"가 보이지 않았다. gate_reasons가 그 분포다.
    """
    class EnglishEchoClient(EchoClient):
        """번역 프롬프트든 repair 프롬프트든 영문 원문을 그대로 되돌려준다."""

        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            src = _marker(user)
            if src is not None:
                return src
            head = "[원문 — 아래 꺾쇠 태그가 정답이다]\n"
            if head in user:
                return user.split(head, 1)[1].split("\n\n", 1)[0]
            return ""

    md = (
        "The accuracy improved on the benchmark dataset that we evaluated. "
        "The second sentence describes the ablation study in more detail.\n"
    )
    _res, report, _out = _run_md(tmp_path, cfg, md, EnglishEchoClient())

    assert report["kept_reasons"] == {"gate-rejected": 1}
    assert report["gate_reasons"].get("hangul-ratio", 0) >= 1
    # 래더(repair·분할)가 흡수한 거부까지 세므로 최종 kept 1건보다 많다 = 오탐 비용
    assert sum(report["gate_reasons"].values()) > 1


def test_거부문_출력은_refusal_사유로_집계된다(tmp_path, cfg):
    class RefusingClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            if _marker(user) is None:
                return ""
            return "죄송합니다, 이 텍스트는 번역할 수 없습니다."

    md = "The accuracy improved on the benchmark dataset that we evaluated.\n"
    _res, report, _out = _run_md(tmp_path, cfg, md, RefusingClient())
    assert report["gate_reasons"].get("refusal", 0) >= 1
    assert report["kept_reasons"] == {"gate-rejected": 1}


def test_정상_번역이면_게이트_집계가_비어있다(tmp_path, cfg):
    md = "The accuracy improved on the benchmark dataset that we evaluated.\n"
    _res, report, _out = _run_md(tmp_path, cfg, md, EchoClient())
    assert report["gate_reasons"] == {}


def test_캐시_전량_무효화는_경고와_지표로_남는다(job, cfg):
    """PROMPT_V·모델·샘플링을 바꾸면 units.json이 통째로 무효가 되어 전량 재번역된다.
    비용이 드는 사건인데 종전에는 어디에도 기록이 없었다."""
    from dataclasses import replace

    run_translation(job, "ko", cfg, client=EchoClient())
    warm = _report(job)
    assert warm["cache_prior"] == 0                     # 첫 run — 기존 캐시 없음
    assert not any("전량 재번역" in w for w in warm["warnings"])

    run_translation(job, "ko", cfg, client=EchoClient())
    hit = _report(job)
    assert hit["cache_prior"] > 0 and hit["cache_reused"] == hit["cache_prior"]
    assert not any("전량 재번역" in w for w in hit["warnings"])   # 전 유닛 적중 → 무경고

    # 프롬프트 개정과 같은 효과 — 캐시 키 재료가 바뀌면 기존 키는 하나도 안 맞는다.
    # 판정 기준은 cached가 아니라 cache_reused다: 문서 내 중복 유닛의 single-flight
    # 재사용도 cached를 올려, 실 논문에서는 전량 무효인데도 cached==2였다(실측).
    run_translation(job, "ko", replace(cfg, reasoning="high"), client=EchoClient())
    invalidated = _report(job)
    assert invalidated["cache_reused"] == 0 and invalidated["cache_prior"] > 0
    assert any("전량 재번역" in w and "PROMPT_V" in w for w in invalidated["warnings"])


# ── 잘림·루프 출력 (probe:MLX-02/03/05/09, mlx-integration-3/4) ─────────────

_LONG_PARA = (
    "The first half of this long paragraph explains the training procedure in detail. "
    "The second half of this long paragraph summarizes the evaluation protocol briefly."
)
_SHORT_PARA = "A short paragraph describes the dataset."


class _TruncatingClient(EchoClient):
    """원문이 120자를 넘으면 출력이 잘린다(TranslateOutputTruncated) — 반쪽은 예산 안에 든다."""

    def __init__(self, limit=120):
        super().__init__()
        self.limit = limit
        self.seen: list[str] = []

    def complete(self, system, user, *, max_tokens):
        from app.translate.types import TranslateOutputTruncated

        self.calls += 1
        src = _marker(user)
        if src is None:
            return ""
        self.seen.append(src)
        if len(src) > self.limit:
            raise TranslateOutputTruncated("번역 API 출력이 max_tokens(8192)에서 잘렸습니다")
        return koreanize(src)


def test_잘린_유닛은_분할로_복구되고_잘린_출력은_캐시되지_않는다(tmp_path, cfg):
    from dataclasses import replace

    md = _SHORT_PARA + "\n\n" + _LONG_PARA + "\n"
    client = _TruncatingClient()
    res, report, out = _run_md(tmp_path, replace(cfg, concurrency=1), md, client)

    assert res.status == "done" and res.kept_original == []
    assert report["split"] == 1 and report["retried"] == 1
    assert out == ko_expected(md)
    # 잘림은 repair 대상이 아니다 — 최초 1 + 반쪽 2 (repair 없음)
    assert [len(s) > 120 for s in client.seen] == [False, True, False, False]


def test_분할도_잘리면_truncated_사유로_원문을_유지한다(tmp_path, cfg):
    from dataclasses import replace

    md = _SHORT_PARA + "\n\n" + _LONG_PARA + "\n"
    client = _TruncatingClient(limit=60)   # 반쪽(80자대)도 잘린다
    res, report, out = _run_md(tmp_path, replace(cfg, concurrency=1), md, client)

    assert res.status == "done" and res.kept_original == ["md:0:1"]
    assert report["kept_reasons"] == {"truncated": 1}
    assert _LONG_PARA in out                                   # 무손실 — 원문 유지
    cache = json.loads((tmp_path / "translations/ko/units.json").read_text(encoding="utf-8"))
    assert len(cache) == 1                                     # 짧은 문단만 캐시


def test_성공_전_첫_유닛이_잘리면_잡_오류로_원인을_알린다(tmp_path, cfg):
    """thinking이 모든 유닛의 예산을 태우는 설정 오류 — 전 유닛을 kept로 조용히
    done 시키지 않고, 첫 잘림에서 안내 문구와 함께 실패한다(종전 정책 유지)."""
    from dataclasses import replace

    from app.translate.types import TranslateError

    (tmp_path / "result.md").write_text(_LONG_PARA + "\n", encoding="utf-8")
    with pytest.raises(TranslateError, match="잘렸습니다"):
        run_translation(tmp_path, "ko", replace(cfg, concurrency=1), client=_TruncatingClient())
    assert "잘렸습니다" in _state(tmp_path)["error"]


def test_태그가_멀쩡한_게이트_거부는_repair_없이_분할로_간다(tmp_path, cfg):
    """빈 태그 목록 repair('태그만 바로잡아')는 echo·거부문을 고칠 수 없다(probe:MLX-09)."""
    prompts_seen = []

    class EchoFirstClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            prompts_seen.append(user)
            src = _marker(user)
            if src is None:
                return ""
            return src if len(src) > 120 else koreanize(src)   # 전체는 영문 echo, 반쪽은 번역

    md = _LONG_PARA + "\n"
    res, report, out = _run_md(tmp_path, cfg, md, EchoFirstClient())
    assert res.kept_original == [] and report["split"] == 1 and report["repaired"] == 0
    assert not any("[수정할 번역문]" in p for p in prompts_seen)
    assert out == ko_expected(md)


def test_루프_출력은_repair_없이_거부되고_캐시되지_않는다(tmp_path, cfg):
    """태그가 빠진 루프 출력에 repair를 보내면 같은 루프를 재생산한다(mlx-integration-4)."""
    prompts_seen = []
    loop = "손실을 최소화한다" + "만" * 300

    class LoopClient(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            prompts_seen.append(user)
            return "" if _marker(user) is None else loop

    md = "We minimize the loss $L$ during training.\n"
    res, report, out = _run_md(tmp_path, cfg, md, LoopClient())
    assert res.kept_original == ["md:0:0"]
    assert report["kept_reasons"] == {"placeholder-mismatch": 1}
    assert not any("[수정할 번역문]" in p for p in prompts_seen)
    assert loop not in out
    cache = json.loads((tmp_path / "translations/ko/units.json").read_text(encoding="utf-8"))
    assert loop not in cache.values()


def test_성공_이후_응답_정지는_유닛_단위로_강등돼_분할로_복구된다(tmp_path, cfg):
    """느린 로컬 유닛 하나의 타임아웃이 잡 전체 실패로 번지던 경로(translate-llm-4)."""
    from dataclasses import replace

    from app.translate.types import TranslateTimeout

    class StallingClient(_TruncatingClient):
        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is not None and len(src) > self.limit:
                self.calls += 1
                raise TranslateTimeout("번역 API 응답 시간 초과 — 180초 동안 응답이 없었습니다")
            return super().complete(system, user, max_tokens=max_tokens)

    md = _SHORT_PARA + "\n\n" + _LONG_PARA + "\n"
    res, report, out = _run_md(tmp_path, replace(cfg, concurrency=1), md, StallingClient())
    assert res.status == "done" and res.kept_original == [] and report["split"] == 1
    assert out == ko_expected(md)

    other = tmp_path / "b"
    other.mkdir()
    stuck = StallingClient(limit=60)                    # 반쪽도 멈추면 timeout 사유로 유지
    _res2, report2, out2 = _run_md(other, replace(cfg, concurrency=1), md, stuck)
    assert report2["kept_reasons"] == {"timeout": 1}
    assert _LONG_PARA in out2


class _HangAfterClient(EchoClient):
    """유닛 요청 ok번은 정상 번역, 그 뒤로는 slow(src)가 참인 요청만 응답 정지한다.

    slow를 주지 않으면 이후 모든 요청이 멈춘다 — 연결은 받지만 응답하지 않는 서버(생성
    스레드 교착·과부하)를 재현한다."""

    def __init__(self, ok=2, slow=None):
        super().__init__()
        self.ok = ok
        self.slow = slow or (lambda src: True)
        self.unit_requests = 0

    def complete(self, system, user, *, max_tokens):
        import time

        from app.translate.types import TranslateTimeout

        src = _marker(user)
        if src is None:
            return ""
        with self._count_lock:
            self.unit_requests += 1
            n = self.unit_requests
        if n > self.ok and self.slow(src):
            time.sleep(0.05)                     # 기다린 끝의 시간 초과 — 메인 스레드가 abort할 틈
            raise TranslateTimeout("번역 API 응답 시간 초과 — 180초 동안 응답이 없었습니다")
        return koreanize(src)


def test_성공_뒤_응답을_멈춘_엔드포인트는_잡을_실패시킨다(tmp_path, cfg):
    """첫 성공 뒤 응답이 멈춘 서버에서 유닛마다 시간 초과를 태운 뒤 영어 원문으로 done이
    되던 경로(translate-2 — 기본값이면 유닛당 ~12분, 200유닛 문서는 ~5시간). 성공 없이
    이어지는 시간 초과가 임계를 넘으면 잡 오류로 알린다."""
    from app.translate.types import TranslateAPIError

    paras = [
        f"Paragraph number {i} explains the method in detail. It also describes the results."
        for i in range(12)
    ]
    (tmp_path / "result.md").write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    client = _HangAfterClient(ok=2)
    with pytest.raises(TranslateAPIError, match="응답하지 않습니다"):
        run_translation(tmp_path, "ko", cfg, client=client)
    state = _state(tmp_path)
    assert state["status"] == "error" and "TRANSLATE_TIMEOUT_S" in state["error"]
    # 종전에는 남은 유닛 10개가 최초 패스+분할 반쪽까지 모두 시간 초과를 태웠다(22회).
    assert client.unit_requests < 10
    assert not (tmp_path / "result.ko.md").exists()


def test_성공이_끼어드는_간헐적_시간_초과는_잡을_실패시키지_않는다(tmp_path, cfg):
    from dataclasses import replace

    paras = [f"Paragraph number {i} describes one measured result." for i in range(6)]
    slow = {paras[1], paras[3]}                        # 한 문장 — 분할 없이 바로 원문 유지
    (tmp_path / "result.md").write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    client = _HangAfterClient(ok=0, slow=lambda src: src in slow)
    res = run_translation(tmp_path, "ko", replace(cfg, concurrency=1), client=client)
    assert res.status == "done" and sorted(res.kept_original) == ["md:0:1", "md:0:3"]
    assert _report(tmp_path)["kept_reasons"] == {"timeout": 2}


# ── 축퇴 스윕의 패스 간 기억 (Phase-0 replay: 2차 패스 짧은 지연 유닛 유출) ──────

def test_1차에서_축퇴로_판정된_출력은_2차_패스에서도_축퇴다(tmp_path, cfg):
    """replay(/tmp/pdfocr-deps/replay, MuPDF 1.27 OCR 산출물)의 축소판.

    1차 패스에서 layout 유닛 5개가 같은 캔드 응답을 받아 축퇴로 지워진다. 그 뒤 2차
    패스에서 지연됐던 짧은 md 유닛 2개('Abstract'·'Keywords')가 같은 응답을 받는데,
    종전 스윕은 2차 결과만 보고 '원문 2종'이라 통과시켜 result.ko.md로 샜다.
    """
    canned = "요약"

    class Canned(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            return canned

    longs = [
        "The proposed quantizer compresses every vector with a random rotation first.",
        "We then apply an optimal scalar quantizer independently to each coordinate.",
        "Experiments on large language models confirm the theoretical distortion bound.",
        "The method is data oblivious and therefore suitable for online applications.",
    ]
    md = "\n\n".join(["Abstract", "Keywords", *longs]) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    lay = _layout_of(["Abstract", "Keywords", "Introduction", "Related Work", "Conclusion"])
    (tmp_path / "layout.json").write_text(json.dumps(lay, ensure_ascii=False), encoding="utf-8")

    res = run_translation(tmp_path, "ko", cfg, client=Canned())

    out = (tmp_path / "result.ko.md").read_text(encoding="utf-8")
    assert canned not in out, "2차 패스의 캔드 응답이 산출물에 실리면 안 된다"
    assert {"md:0:0", "md:0:1"} <= set(res.kept_original)
    report = _report(tmp_path)
    assert report["kept_reasons"]["degenerate-output"] == 7     # 1차 lay 5 + 2차 md 2
    cache = json.loads((tmp_path / "translations/ko/units.json").read_text(encoding="utf-8"))
    assert canned not in cache.values()


def test_같은_라벨의_표기_변형이_같은_번역으로_수렴하는_것은_축퇴가_아니다(tmp_path, cfg):
    """대소문자·복수형만 다른 짧은 라벨 3종이 같은 정답으로 수렴하면 정상이다 —
    종전에는 축퇴로 원문 유지되고 캐시가 지워져 매 실행 재과금됐다(translate-llm-15)."""
    labels = ["Training epoch", "Training epochs", "training epoch"]

    class Labeler(EchoClient):
        def complete(self, system, user, *, max_tokens):
            self.calls += 1
            return "" if _marker(user) is None else "학습 에포크"

    res, report, out = _run_md(tmp_path, cfg, "\n\n".join(labels) + "\n", Labeler())
    assert res.kept_original == [] and "degenerate-output" not in report["kept_reasons"]
    assert out.count("학습 에포크") == 3
    again = Labeler()
    run_translation(tmp_path, "ko", cfg, client=again)
    assert again.calls == 0                                  # 캐시가 지워지지 않았다


def test_layout_잡에서도_D정책_첫_등장_병기가_프롬프트에_실린다(tmp_path, cfg):
    """종전에는 first_unit이 md 유닛 id라 lay:* 유닛과 일치하지 않고, md 유닛은 layout으로
    덮여 지연되므로 [첫 등장 병기]가 든 프롬프트가 0건이었다(translate-llm-12)."""
    lines = [
        "We adopt sparse attention for long documents in this work.",
        "The sparse attention pattern reduces the memory footprint.",
    ]
    prompts_seen = []

    class Recorder(EchoClient):
        def complete(self, system, user, *, max_tokens):
            prompts_seen.append(user)
            return super().complete(system, user, max_tokens=max_tokens)

    md = "\n\n".join(lines) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    (tmp_path / "layout.json").write_text(
        json.dumps(_layout_of(lines), ensure_ascii=False), encoding="utf-8")
    run_translation(tmp_path, "ko", cfg, client=Recorder())

    first = [p for p in prompts_seen if "[첫 등장 병기" in p]
    assert len(first) == 1 and lines[0] in first[0]
    saved = json.loads((tmp_path / "translations/ko/glossary.json").read_text(encoding="utf-8"))
    entry = next(e for e in saved if e["src"] == "sparse attention")
    assert entry["first_unit_lay"] == "lay:1:0"


# ── 용어집 실패·취소·손상 (translate-llm-6, concurrency-9, translate-llm-7) ─────

# 대문자 후보가 3번 나와 용어집 LLM 판정이 실제로 호출되는 문서
_GLOSSARY_MD = (
    "We describe the Foo Bar encoder that maps every token to a vector.\n\n"
    "We train the Foo Bar encoder with a contrastive objective.\n\n"
    "We evaluate the Foo Bar encoder on three public benchmarks.\n"
)


class _GlossaryClient(EchoClient):
    """용어집 프롬프트에 대한 응답을 바꿀 수 있는 스텁 — 유닛은 정상 번역."""

    def __init__(self, glossary_reply):
        super().__init__()
        self.glossary_reply = glossary_reply
        self.glossary_calls = 0

    def complete(self, system, user, *, max_tokens):
        if _marker(user) is None:
            self.glossary_calls += 1
            reply = self.glossary_reply
            if isinstance(reply, BaseException):
                raise reply
            return reply
        return super().complete(system, user, max_tokens=max_tokens)


def test_용어집_LLM_실패는_다음_실행에서_다시_판정한다(tmp_path, cfg):
    from app.translate.types import TranslateAPIError

    down = _GlossaryClient(TranslateAPIError("번역 API 연결 실패: refused"))
    _res, report, _ = _run_md(tmp_path, cfg, _GLOSSARY_MD, down)
    tdir = tmp_path / "translations/ko"
    assert (tdir / "glossary.json").is_file() and (tdir / "glossary.incomplete").is_file()
    assert any("다음 실행에서 다시 판정" in w for w in report["warnings"])

    ok = _GlossaryClient('[{"src": "Foo Bar", "ko": "푸 바", "policy": "D"}]')
    _res2, report2, _ = _run_md(tmp_path, cfg, _GLOSSARY_MD, ok)
    assert ok.glossary_calls == 1                         # 시드에 굳지 않고 다시 판정
    assert not (tdir / "glossary.incomplete").exists()
    saved = json.loads((tdir / "glossary.json").read_text(encoding="utf-8"))
    assert any(e["src"] == "foo bar" for e in saved)
    assert not any("다음 실행에서 다시 판정" in w for w in report2["warnings"])

    again = _GlossaryClient("[]")
    run_translation(tmp_path, "ko", cfg, client=again)
    assert again.glossary_calls == 0 and again.unit_calls == 0   # 성공한 용어집·캐시 재사용


def test_용어집_단계의_취소는_시드_용어집을_남기지_않는다(tmp_path, cfg):
    from app.translate.client import _RequestCancelled

    cancel = threading.Event()

    class CancelInGlossary(_GlossaryClient):
        def complete(self, system, user, *, max_tokens):
            if _marker(user) is None:
                cancel.set()
                raise _RequestCancelled("번역 요청이 취소되었습니다")
            return super().complete(system, user, max_tokens=max_tokens)

    (tmp_path / "result.md").write_text(_GLOSSARY_MD, encoding="utf-8")
    res = run_translation(tmp_path, "ko", cfg, client=CancelInGlossary("[]"), cancel=cancel)
    assert res.status == "canceled"
    assert not (tmp_path / "translations/ko/glossary.json").exists()


def test_손상된_glossary_json은_다시_만들고_번역을_계속한다(tmp_path, cfg):
    _run_md(tmp_path, cfg, _GLOSSARY_MD, _GlossaryClient("[]"))
    gpath = tmp_path / "translations/ko/glossary.json"
    gpath.write_text('[{"src": "foo', encoding="utf-8")      # 기록 도중 잘린 파일

    client = _GlossaryClient("[]")
    res, report, _ = _run_md(tmp_path, cfg, _GLOSSARY_MD, client)
    assert res.status == "done" and client.glossary_calls == 1
    assert any("손상" in w for w in report["warnings"])
    assert isinstance(json.loads(gpath.read_text(encoding="utf-8")), list)
    assert not list(gpath.parent.glob("*.tmp"))               # 원자적 기록 — 임시 파일 없음


def test_용어집_호출_성공은_콜드_run의_첫_4xx를_유닛_강등으로_흡수한다(tmp_path, cfg):
    """콜드 run에서 빠르게 돌아온 400 하나가 정상 엔드포인트에서도 잡 전체를 실패시키던
    경쟁(translate-llm-7) — 같은 모델로 성공한 용어집 호출이 건강의 증거다."""
    from dataclasses import replace

    from app.translate.types import TranslateUnitRejected

    class Oversized(_GlossaryClient):
        def complete(self, system, user, *, max_tokens):
            src = _marker(user)
            if src is not None and "maps every token" in src:
                self.calls += 1
                raise TranslateUnitRejected("번역 API 오류 (HTTP 400): context length")
            return super().complete(system, user, max_tokens=max_tokens)

    # concurrency=1 — 거부 유닛이 첫 번째로 API를 탄다(유닛 성공 이력 0)
    res, report, _ = _run_md(tmp_path, replace(cfg, concurrency=1), _GLOSSARY_MD, Oversized("[]"))
    assert res.status == "done" and res.kept_original == ["md:0:0"]
    assert report["kept_reasons"] == {"api-rejected": 1}


def test_이전_실행이_캐시한_루프_출력은_재검증에서_걸러진다(tmp_path, cfg):
    """게이트 강화 전에 채택·캐시된 손상(probe:MLX-02의 '…만만만' 루프)이 force 없이도
    재사용되지 않는다 — 그 유닛만 다시 번역하고 나머지 캐시는 그대로 쓴다."""
    md = (
        "The first paragraph explains the training procedure in detail.\n\n"
        "The second paragraph removes the unnecessary regions before attention.\n"
    )
    _run_md(tmp_path, cfg, md, EchoClient())
    upath = tmp_path / "translations/ko/units.json"
    cache = json.loads(upath.read_text(encoding="utf-8"))
    looped_key = next(k for k, v in cache.items() if v == koreanize(
        "The second paragraph removes the unnecessary regions before attention."))
    cache[looped_key] = "두 번째 문단은 주의 이전에 불필요한 영역을" + "만" * 40
    upath.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    again = EchoClient()
    res, report, out = _run_md(tmp_path, cfg, md, again)
    assert again.unit_calls == 1 and res.cached == 1 and res.translated == 1
    assert report["cache_rejected"] == 1 and report["cache_reused"] == 1
    assert "만만만" not in out and out == ko_expected(md)


# ── 분할 결합 출력 속의 캔드 응답 (verify_e2e 25쪽 실측) ───────────────────────

class _CannedEverywhere(EchoClient):
    def __init__(self, canned):
        super().__init__()
        self.canned = canned

    def complete(self, system, user, *, max_tokens):
        self.calls += 1
        return "[]" if _marker(user) is None else self.canned


def test_분할_반쪽이_축퇴_출력이면_결합_출력도_원문_유지(tmp_path, cfg):
    """반쪽마다 같은 캔드 응답 → 결합 문자열은 다른 유닛 출력과 달라 스윕을 빠져나갔다."""
    canned = "요약된 내용"
    split_src = "The estimator is unbiased. Its variance shrinks with d."
    md = "\n\n".join(["Introduction", "Related Work", "Conclusion", split_src]) + "\n"
    res, report, out = _run_md(tmp_path, cfg, md, _CannedEverywhere(canned))
    assert canned not in out
    assert "md:0:3" in res.kept_original and report["split"] == 1
    assert report["kept_reasons"]["degenerate-output"] == 4
    cache = json.loads((tmp_path / "translations/ko/units.json").read_text(encoding="utf-8"))
    assert not any(canned in v for v in cache.values())


def test_분할_결합_출력도_유닛_전체_기준_게이트를_통과해야_한다(tmp_path, cfg):
    """반쪽 기준으로는 통과해도 이은 결과가 원문 전체에 비해 지나치게 짧으면 버린다."""
    canned = "요약된 내용"
    src = "The estimator stays unbiased for every input. Its variance shrinks with the dimension d."
    res, report, out = _run_md(tmp_path, cfg, src + "\n", _CannedEverywhere(canned))
    assert res.kept_original == ["md:0:0"] and report["split"] == 0
    assert canned not in out and src in out


def test_여러_줄_layout_블록과_같은_md유닛은_한_번만_번역한다(tmp_path, cfg):
    """md 문단(여러 줄)이 layout 블록 하나와 통째로 같으면 layout 번역을 재사용한다 —
    verify_e2e 25쪽 실측에서 같은 원문 중복 호출이 365종(810회 중)이었다."""
    blocks = [
        "Amir Zandieh\nGoogle Research\nzandieh at google dot com",
        "We study online vector quantization\nwith near optimal distortion rates.",
    ]
    md = "\n\n".join(blocks) + "\n"
    (tmp_path / "result.md").write_text(md, encoding="utf-8")
    lay = _layout_of(blocks)
    (tmp_path / "layout.json").write_text(json.dumps(lay, ensure_ascii=False), encoding="utf-8")

    client = EchoClient()
    res = run_translation(tmp_path, "ko", cfg, client=client)
    assert client.unit_calls == len(blocks) and res.total == len(blocks)
    assert (tmp_path / "result.ko.md").read_text(encoding="utf-8") == ko_expected(md)


def test_거부된_출력도_축퇴_증거로_센다(tmp_path, cfg):
    """같은 캔드 응답이 긴 문단들에서는 게이트에 거부되고 짧은 조각 둘만 통과하면, 통과분만
    세는 스윕('원문 3종')은 그 둘을 놓쳤다(verify_e2e 4쪽 fault=summary 실측)."""
    canned = "요약입니다."
    md = "\n\n".join([
        "(inner-prod error)\nDprod := E",
        "(unbiased inner-prod)\nE\nQ",
        "The first long paragraph explains the training procedure and the data split in detail.",
        "The second long paragraph summarizes the evaluation protocol and the reported metrics.",
    ]) + "\n"
    res, report, out = _run_md(tmp_path, cfg, md, _CannedEverywhere(canned))
    assert canned not in out
    assert report["kept_reasons"].get("degenerate-output") == 2
    assert not json.loads((tmp_path / "translations/ko/units.json").read_text(encoding="utf-8"))
