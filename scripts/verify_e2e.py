"""실제 PDF 기반 전 구간 기능 점검 하네스.

scripts/smoke_e2e.sh 는 업로드→OCR→markdown/zip 에서 끝난다. 이 스크립트는 그 뒤의
번역 · 레이아웃 보존 PDF 내보내기 · 뷰어 계약 · Q&A 키 분리 · 워커 복원력까지
**실제 서버를 띄워** 검증한다. 모델 다운로드 없이 돌도록 OCR_ENGINE=textlayer 를 쓰며,
샘플 PDF(arXiv 논문)는 텍스트 레이어가 충분해 tesseract 없이 전 페이지가 처리된다.

사용 (= make verify-e2e):
    uv run --project backend python scripts/verify_e2e.py [--pdf PATH] [--pages N]

산출물(서버 로그·내보낸 PDF 등)은 기본으로 리포의 tmp/verify-e2e/ 에 남는다
(.gitignore 대상). `--work DIR` 로 옮길 수 있다. 포트는 매 실행마다 비어 있는
포트를 잡으므로 개발 서버(8000)와 충돌하지 않는다.

⚠ **작업 디렉터리는 포트와 달리 자동으로 갈라지지 않는다.** api.log·data-main·
fault-* 는 전부 고정 이름이고 각 단계가 시작할 때 rmtree 한다. 같은 --work 로 두
실행이 겹치면 뒤 실행이 앞 실행의 잡 데이터를 지워 앞쪽이 알 수 없는 이유로 깨진다.
그래서 --work 에 배타 락(.harness.lock)을 걸고, 이미 살아 있는 실행이 잡고 있으면
"다른 --work 를 쓰라"는 메시지와 함께 즉시 종료한다(종료코드 2).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 하드코딩 금지 — 이 파일 위치(scripts/)에서 리포 루트를 유도한다.
REPO = Path(__file__).resolve().parents[1]
MOCK_SERVER = Path(__file__).resolve().with_name("mock_llm.py")

# main()에서 실제 값으로 채운다 (작업 디렉터리·포트는 실행 시점에 정해진다).
WORK = REPO / "tmp" / "verify-e2e"
MOCK_PORT = 0
API_PORT = 0
BASE = ""
MOCK = ""


def _free_port() -> int:
    """OS가 비어 있다고 알려주는 포트를 하나 잡는다 (고정 포트 충돌 회피)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _python() -> str:
    """백엔드 venv 인터프리터 — uv run으로 들어왔으면 현재 인터프리터가 곧 그것이다."""
    venv = REPO / "backend" / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


LOCK_NAME = ".harness.lock"


def _pid_alive(pid: int) -> bool:
    """해당 pid가 살아 있는가 (신호 0 — 실제로 보내지 않고 존재만 확인)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 다른 사용자 소유지만 살아 있다
    return True


def acquire_work_lock(work: Path) -> Path:
    """--work 디렉터리에 배타 락을 건다. 이미 살아 있는 실행이 잡고 있으면 SystemExit.

    포트만 격리하고 작업 디렉터리는 공유하던 것이 결함이었다 — data-main·fault-* 는
    각 단계 시작 시 rmtree 되므로 동시 실행이 서로의 잡을 지웠다. 죽은 프로세스가
    남긴 락(정전·kill -9)은 stale로 보고 인수한다.
    """
    work.mkdir(parents=True, exist_ok=True)
    lock = work / LOCK_NAME
    payload = f"{os.getpid()} {time.time():.0f}\n".encode()
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                owner = int(lock.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                owner = -1
            if owner != os.getpid() and _pid_alive(owner):
                # 종료코드 2 = "단언 실패(1)가 아니라 실행 자체를 못 했다" — 스크립트를
                # 감싸는 쪽이 재시도할지 보고할지 구분할 수 있어야 한다.
                print(
                    f"작업 디렉터리를 다른 하네스 실행(pid {owner})이 쓰고 있습니다: {work}\n"
                    f"동시에 돌리려면 --work 로 다른 디렉터리를 주세요 "
                    f"(예: --work {work}-2).",
                    file=sys.stderr, flush=True,
                )
                raise SystemExit(2) from None
            # stale 락 — 소유 프로세스가 이미 죽었다. 지우고 한 번 더 시도한다.
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        return lock
    raise SystemExit(f"작업 디렉터리 락을 잡지 못했습니다: {lock}")


def release_work_lock(lock: Path) -> None:
    """내가 잡은 락만 해제한다 (인수당한 경우 남의 락을 지우지 않게)."""
    try:
        if int(lock.read_text().split()[0]) == os.getpid():
            lock.unlink(missing_ok=True)
    except (OSError, ValueError, IndexError):
        pass

PASS: list[str] = []
FAIL: list[str] = []
INFO: list[str] = []


def layout_pages(obj) -> list:
    """layout.json은 페이지 리스트다. 혹시 모를 dict 래핑도 함께 받아준다."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("pages", [])
    return []


# ─────────────────── 공유 판정 로직 (정상 경로·결함 주입 공통) ───────────────────

# prompts.py import 실패 시에만 쓰는 최소 목록. 정상 경로에서는 실제 문구를 뽑아온다.
_SCAFFOLD_FALLBACK = (
    "[번역할 원문", "[직전 문맥", "[용어집", "[원문 유지", "[첫 등장 병기",
    "[블록 유형", "[수정할 번역문", "다음 꺾쇠 태그가",
)
_SENTINEL = "ZQXSENTINELZQX"
_scaffold_cache: tuple[str, ...] = ()


def scaffolding_markers() -> tuple[str, ...]:
    """번역 결과에 새면 안 되는 프롬프트 스캐폴딩 문구.

    문구를 하네스에 하드코딩하면 prompts.py가 바뀔 때 검사만 조용히 무력화된다.
    실제 빌더를 호출해 센티널이 아닌 줄 = 스캐폴딩으로 간주하고 뽑아낸다.
    **repair 프롬프트도 포함한다** — fault=echo 실행에서 "[수정할 번역문]" 등
    repair 스캐폴딩 25줄이 result.ko.md에 그대로 박힌 사례가 실측됐다.
    """
    global _scaffold_cache
    if _scaffold_cache:
        return _scaffold_cache
    markers: set[str] = set()
    try:
        if str(REPO / "backend") not in sys.path:
            sys.path.insert(0, str(REPO / "backend"))
        from app.translate.prompts import build_repair_prompt, build_unit_prompt

        probes = [
            build_unit_prompt(
                f"{_SENTINEL}SRC",
                [(f"{_SENTINEL}A", f"{_SENTINEL}B")],
                [(f"{_SENTINEL}C", f"{_SENTINEL}D")],
                context_tail=f"{_SENTINEL}CTX",
                keep_terms=[f"{_SENTINEL}K"],
                unit_kind="title",
            ),
            build_repair_prompt(f"{_SENTINEL}SRC", f"{_SENTINEL}OUT", [f"<{_SENTINEL}1/>"]),
        ]
        for probe in probes:
            for line in probe.splitlines():
                line = line.strip()
                if not line or _SENTINEL in line:
                    continue  # 번역 대상 본문·용어 목록은 스캐폴딩이 아니다
                if line.startswith("["):
                    # 대괄호 헤더는 "—" 앞까지만 쓴다(부제가 바뀌어도 계속 잡히게).
                    head = re.split(r"[—\]]", line[1:], maxsplit=1)[0].strip()
                    if head:
                        markers.add("[" + head)
                elif len(line) >= 12:
                    markers.add(line[:24])
    except Exception as e:  # noqa: BLE001 — 하네스: import 실패해도 검사는 남긴다
        info(f"prompts.py에서 스캐폴딩 문구를 읽지 못함 ({type(e).__name__}) — 기본 목록 사용")
    markers.update(_SCAFFOLD_FALLBACK)
    _scaffold_cache = tuple(sorted(markers))
    return _scaffold_cache


def scaffolding_hits(text: str) -> list[str]:
    """text에 들어 있는 스캐폴딩 문구 목록(부분일치)."""
    return [m for m in scaffolding_markers() if m in text]


def hangul_latin(text: str) -> tuple[int, int]:
    hangul = sum("가" <= c <= "힣" for c in text)
    latin = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)
    return hangul, latin


def hangul_share(text: str) -> float:
    """한글 / (한글+라틴) — 수식·숫자·공백은 분모에서 빼 표·수식 페이지 오탐을 막는다."""
    hangul, latin = hangul_latin(text)
    return hangul / (hangul + latin) if (hangul + latin) else 0.0


def dropped_translation_pages(
    layout_ko_pages: list,
    pdf_page_texts: list[str],
    *,
    translated_floor: float = 0.6,
    retention: float = 0.5,
) -> list[tuple[int, float, float]]:
    """번역은 존재하는데 PDF가 버린 페이지 목록 → [(페이지번호, layout비, pdf비)].

    사용자 신고("한글로 번역 안 되는 문단")의 실제 정체다: layout.ko.json은 전 페이지가
    한글 91~99%인데 export.ko.pdf는 p16이 0%, p15가 32%였다(조판 실패 → 원문 보존).
    번역 자체가 없는 페이지(참고문헌 등 원문 유지 설계)는 translated_floor로 걸러낸다.
    """
    bad = []
    for idx, page in enumerate(layout_ko_pages):
        if idx >= len(pdf_page_texts):
            break
        text = "".join(b.get("content") or "" for b in page.get("blocks", []))
        lay = hangul_share(text)
        if lay < translated_floor:
            continue  # 애초에 번역면이 원문 유지 — PDF가 버린 게 아니다
        pdf = hangul_share(pdf_page_texts[idx])
        if pdf < lay * retention:
            bad.append((idx + 1, round(lay, 2), round(pdf, 2)))
    return bad


def kept_reason_summary(report: dict) -> dict[str, int]:
    """export.{lang}.report.json의 보존 사유별 집계.

    fitting 그룹이 사유 집계 필드를 추가하는 중이라 **필드 이름을 고정하지 않는다** —
    int 값을 가진 dict 필드는 전부 접두사와 함께 합치고, warnings 문자열은
    페이지·블록 번호를 지운 사유 문구로 묶는다.
    """
    counts: dict[str, int] = {}
    for key, val in report.items():
        if isinstance(val, dict) and val and all(isinstance(v, int) for v in val.values()):
            for k, v in val.items():
                counts[f"{key}.{k}"] = counts.get(f"{key}.{k}", 0) + v
    for warn in report.get("warnings") or []:
        reason = re.sub(r"^p\d+:\s*", "", str(warn))
        reason = re.sub(r"블록\s*\d+", "블록 N", reason)
        counts[f"warning: {reason}"] = counts.get(f"warning: {reason}", 0) + 1
    return counts


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(f"{name}" + (f" — {detail}" if detail else ""))
    print(("  PASS  " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""), flush=True)
    return ok


def info(msg: str) -> None:
    INFO.append(msg)
    print("  ....  " + msg, flush=True)


def req(method: str, path: str, *, body=None, headers=None, raw=False, timeout=120):
    url = path if path.startswith("http") else BASE + path
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload, dict(resp.headers)
            try:
                return resp.status, json.loads(payload or b"{}"), dict(resp.headers)
            except json.JSONDecodeError:
                return resp.status, payload.decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        payload = e.read()
        if raw:
            return e.code, payload, dict(e.headers)
        try:
            return e.code, json.loads(payload or b"{}"), dict(e.headers)
        except json.JSONDecodeError:
            return e.code, payload.decode("utf-8", "replace"), dict(e.headers)


def post_pdf(pdf: Path, dpi: int = 150) -> tuple[int, dict]:
    """POST /api/jobs — 상태 코드를 그대로 돌려준다(거부 경로 검증용)."""
    boundary = "----uocrverify"
    parts = []
    for key, val in (("mode", "multi"), ("dpi", str(dpi))):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{pdf.name}\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
    )
    parts.append(pdf.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    payload = b"".join(parts)
    status, js, _ = req("POST", "/api/jobs", body=payload,
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return status, js


def upload(pdf: Path, dpi: int = 150) -> str:
    status, js = post_pdf(pdf, dpi)
    if status != 202:
        raise SystemExit(f"업로드 실패 {status}: {js}")
    return js["job_id"]


def wait_job(job_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        _, js, _ = req("GET", f"/api/jobs/{job_id}")
        last = js
        if js.get("status") in ("done", "error", "canceled"):
            return js
        time.sleep(1.5)
    raise SystemExit(f"잡 타임아웃: {last}")


def wait_translate(job_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        _, js, _ = req("GET", f"/api/jobs/{job_id}/translate/state?lang=ko")
        last = js
        if js.get("status") in ("done", "error", "canceled"):
            return js
        time.sleep(1.0)
    raise SystemExit(f"번역 타임아웃: {last}")


def wait_http(url: str, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


class Servers:
    def __init__(self, data_dir: Path, env_extra: dict[str, str]):
        self.data_dir = data_dir
        self.env_extra = env_extra
        self.mock = None
        self.api = None
        self.api_log = None

    def __enter__(self):
        py = _python()
        # FAULT는 **목 서버**가 읽는 결함 주입 스위치다 — 백엔드 env로만 넣으면
        # 목에 닿지 않아 [8] 결함 주입 단계가 조용히 무력화된다. 목에도 전달한다.
        mock_env = dict(os.environ)
        mock_env["FAULT"] = self.env_extra.get("FAULT", "")
        # 실제 한국어의 길이 압축률을 재현시킨다(0=길이 보존 옛 동작). 목이 길이를
        # 보존하면 looks_untranslated()의 길이비 하한 회귀가 하네스에 안 잡힌다.
        mock_env.setdefault("MOCK_TRANSLATE_RATIO", "0.4")
        self.mock = subprocess.Popen(
            [py, str(MOCK_SERVER), str(MOCK_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=mock_env,
        )
        if not wait_http(f"{MOCK}/__stats"):
            raise SystemExit("목 LLM 기동 실패")

        env = dict(os.environ)
        # .env 자동 로딩이 실키를 끌어오지 않도록 명시적으로 덮어쓴다.
        env.update({
            "OCR_ENGINE": "textlayer",
            "OCR_DEVICE": "cpu",
            "PRELOAD_MODEL": "0",
            "DATA_DIR": str(self.data_dir),
            "ALLOWED_HOSTS": "localhost,127.0.0.1",
            "OPENAI_BASE_URL": f"{MOCK}/v1",
            "OPENAI_API_KEY": "sk-mock-translation-key",
            "OPENAI_MODEL": "mock-model",
            "TRANSLATE_API_MODE": "chat",
            "TRANSLATE_CONCURRENCY": "4",
            "TRANSLATE_TIMEOUT_S": "60",
            "TRANSLATE_MAX_RETRIES": "1",
            "PYTHONUNBUFFERED": "1",
        })
        env.update(self.env_extra)
        self.api_log = open(WORK / "api.log", "w")
        self.api = subprocess.Popen(
            [py, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
            cwd=str(REPO / "backend"), env=env,
            stdout=self.api_log, stderr=subprocess.STDOUT,
        )
        if not wait_http(f"{BASE}/api/health", timeout=120):
            print((WORK / "api.log").read_text()[-4000:])
            raise SystemExit("백엔드 기동 실패")
        return self

    def __exit__(self, *exc):
        for p in (self.api, self.mock):
            if p and p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        if self.api_log:
            self.api_log.close()


# ─────────────────────────── 개별 검증 ───────────────────────────

def verify_ocr(job_id: str, job: dict, data_dir: Path, expect_pages: int) -> Path:
    print("\n[2] OCR 산출물")
    jd = data_dir / "jobs" / job_id
    check("잡이 done으로 끝남", job.get("status") == "done", str(job.get("error")))
    md = jd / "result.md"
    check("result.md 생성", md.is_file())
    text = md.read_text(encoding="utf-8") if md.is_file() else ""
    check("result.md 내용이 비어있지 않음", len(text) > 500, f"{len(text)}자")
    lay = jd / "layout.json"
    check("layout.json 생성", lay.is_file())
    if lay.is_file():
        pages = layout_pages(json.loads(lay.read_text()))
        check("layout 페이지 수 == 원본 페이지 수", len(pages) == expect_pages,
              f"layout={len(pages)} pdf={expect_pages}")
        nblocks = sum(len(p.get("blocks", [])) for p in pages)
        check("layout 블록이 충분히 추출됨", nblocks > expect_pages * 3, f"{nblocks} blocks")
    seps = text.count("\n\n---\n\n")
    check("result.md 페이지 구분자 수 == 페이지-1 (Q&A 페이지 인덱스 계약)",
          seps == expect_pages - 1, f"구분자 {seps}, 기대 {expect_pages - 1}")
    return jd


def verify_translation(job_id: str, jd: Path, expect_pages: int) -> None:
    print("\n[3] 번역 파이프라인")
    req("GET", f"{MOCK}/__reset")
    status, js, _ = req("POST", f"/api/jobs/{job_id}/translate", body={"lang": "ko"})
    check("POST /translate 202", status == 202, f"{status} {js}")
    st = wait_translate(job_id)
    check("번역 state=done", st.get("status") == "done", json.dumps(st, ensure_ascii=False)[:300])

    _, stats, _ = req("GET", f"{MOCK}/__stats")
    calls = stats["calls"]
    dupes = {k: v for k, v in stats["by_text"].items() if v > 1}
    # D-1은 문서 모양에 따라 분기한다. reconcile이 성공하면 md 유닛 번역이 폐기되므로
    # md 유닛은 애초에 1차에서 빠져야 하고(중복 0), 폴백이면 md 번역이 실제로
    # result.ko.md에 쓰이므로 중복은 낭비가 아니다. 로그로 어느 분기인지 판정한다.
    log_text = (WORK / "api.log").read_text(errors="replace") if (WORK / "api.log").is_file() else ""
    fell_back = "reconcile 폴백" in log_text
    info(f"LLM 호출 {calls}회, 동일 원문 중복 {len(dupes)}종, reconcile={'폴백' if fell_back else '성공'}")
    if fell_back:
        deferred_logged = "지연 md 유닛" in log_text
        check("D-1(폴백 분기): 지연된 md 유닛이 2차 패스로 실제 번역됨", deferred_logged,
              "2차 패스 로그 확인됨" if deferred_logged else "폴백인데 2차 패스 로그가 없음")
        check("D-1(폴백 분기): 중복 번역분이 result.ko.md에 실제로 반영됨 (낭비 아님)",
              (jd / "result.ko.md").is_file()
              and sum("가" <= c <= "힣" for c in (jd / "result.ko.md").read_text()) > 200)
    else:
        check("D-1(성공 분기): 동일 원문에 대한 중복 LLM 호출 없음", len(dupes) == 0,
              f"중복 {len(dupes)}종 (예: {list(dupes)[:1]})")

    ko = jd / "result.ko.md"
    check("result.ko.md 생성", ko.is_file())
    if ko.is_file():
        kt = ko.read_text(encoding="utf-8")
        hangul = sum("가" <= c <= "힣" for c in kt)
        check("번역문에 한글이 실제로 들어있음", hangul > 200, f"한글 {hangul}자")
        check("번역본 페이지 구분자 수 보존", kt.count("\n\n---\n\n") == expect_pages - 1,
              f"{kt.count(chr(10)+chr(10)+'---'+chr(10)+chr(10))}")
    kl = jd / "layout.ko.json"
    check("layout.ko.json 생성", kl.is_file())
    if kl.is_file() and (jd / "layout.json").is_file():
        pa = layout_pages(json.loads((jd / "layout.json").read_text()))
        pb = layout_pages(json.loads(kl.read_text()))
        check("원문/번역 layout 페이지 수 동일", len(pa) == len(pb), f"{len(pa)} vs {len(pb)}")
        same_blocks = all(len(x.get("blocks", [])) == len(y.get("blocks", []))
                          for x, y in zip(pa, pb))
        check("원문/번역 layout 블록 수 전 페이지 동일", same_blocks)
    src_md = jd / "result.md"
    if ko.is_file() and src_md.is_file():
        ratio = len(ko.read_text(encoding="utf-8")) / max(1, len(src_md.read_text(encoding="utf-8")))
        # 목이 실제 한국어 압축률(원문의 0.3~0.5배)을 내고 있는지 — 길이 보존 목이면
        # 아래 kept_original 단언이 길이비 게이트 회귀를 못 잡는다.
        check("번역문 길이가 실제 한국어 압축률 범위 (목 압축 모드 동작 확인)",
              0.25 <= ratio <= 0.75, f"ko/원문 길이비 {ratio:.2f}")

    rep = jd / "translations/ko/report.json"
    if rep.is_file():
        r = json.loads(rep.read_text())
        kept = len(r.get("kept_original", []))
        done = int(r.get("translated") or 0) + int(r.get("cached") or 0)
        info(f"report: kept_original={kept} translated={done} retried={r.get('retried')}")
        # 정상 경로에서 출력 게이트가 정상 번역을 원문 보존으로 떨구면 무성 손실이다.
        # 목이 하한(0.3)의 1.2배 근처를 내므로, 하한을 올리는 회귀는 여기서 드러난다.
        check("정상 경로에서 출력 검증 게이트가 정상 번역을 버리지 않음 (길이비 하한 회귀 탐지)",
              kept <= max(2, 0.02 * (kept + done)),
              f"kept_original={kept} / 번역 {done} (예: {r.get('kept_original', [])[:2]})")


def verify_pdf_export(job_id: str, jd: Path, expect_pages: int) -> None:
    print("\n[4] 레이아웃 보존 번역 PDF")
    import pymupdf

    status, payload, hdrs = req("GET", f"/api/jobs/{job_id}/pdf?lang=ko", raw=True, timeout=600)
    if not check("GET /pdf?lang=ko 200", status == 200, f"{status}"):
        return
    out = WORK / "export.ko.pdf"
    out.write_bytes(payload)
    # HTTP 헤더 이름은 대소문자를 가리지 않는다 — 서버/프록시에 따라 표기가 달라지므로
    # 소문자로 접어서 찾는다(대소문자 그대로 비교하면 조용히 빈 줄만 찍힌다).
    uocr = {k: v for k, v in hdrs.items() if k.lower().startswith("x-uocr")}
    info("내보내기 헤더: " + (", ".join(f"{k}={v}" for k, v in sorted(uocr.items())) or "(없음)"))

    doc = pymupdf.open(out)
    check("번역 PDF 페이지 수 보존", doc.page_count == expect_pages,
          f"{doc.page_count} vs {expect_pages}")

    total_hangul = 0
    outside = []
    tofu = 0
    page_texts: list[str] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        txt = page.get_text()
        page_texts.append(txt)
        total_hangul += sum("가" <= c <= "힣" for c in txt)
        tofu += txt.count("�") + txt.count("□")
        pr = page.rect
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                # 1pt 여유 — 조판 반올림 허용
                if y1 > pr.y1 + 1 or y0 < pr.y0 - 1 or x1 > pr.x1 + 1 or x0 < pr.x0 - 1:
                    outside.append((pno + 1, [round(v, 1) for v in line["bbox"]]))
    check("번역 PDF에 한글이 실제로 조판됨", total_hangul > 200, f"한글 {total_hangul}자")
    check("A-2: 모든 텍스트 줄이 페이지 표시 영역 안에 있음", not outside,
          f"이탈 {len(outside)}줄 (예: {outside[:2]})")
    check("폰트 폴백 tofu 없음", tofu == 0, f"tofu {tofu}자")
    doc.close()

    verify_pdf_reflects_translation(jd, page_texts, "ko")

    status, payload, hdrs = req("GET", f"/api/jobs/{job_id}/pdf?lang=ko&view=dual",
                                raw=True, timeout=600)
    if check("GET /pdf?view=dual 200", status == 200, f"{status}"):
        dual = WORK / "export.ko.dual.pdf"
        dual.write_bytes(payload)
        d = pymupdf.open(dual)
        check("대조판 페이지 수 == 원본", d.page_count == expect_pages, f"{d.page_count}")
        if d.page_count:
            w, h = d[0].rect.width, d[0].rect.height
            check("대조판이 가로로 2배 폭", w > h, f"{round(w)}x{round(h)}")
        d.close()


def verify_pdf_reflects_translation(jd: Path, page_texts: list[str], lang: str) -> None:
    """번역이 존재하는데 PDF가 버린 페이지를 페이지 단위로 대조한다.

    사용자 신고("한글로 번역 안 되는 문단")의 실제 정체 — layout.ko.json은 전 페이지가
    한글 91~99%인데 export.ko.pdf는 p16 0%, p15 32%였다. 총합(한글 200자 이상)만 보면
    전 페이지가 통째로 원문으로 남아도 통과한다. 이 단언 하나가 그 결함 유형을 잡는다.
    """
    kl = jd / f"layout.{lang}.json"
    if not kl.is_file():
        check(f"layout.{lang}.json 존재 (PDF 반영 대조용)", False)
        return
    pages = layout_pages(json.loads(kl.read_text(encoding="utf-8")))
    dropped = dropped_translation_pages(pages, page_texts)
    translated = [i + 1 for i, p in enumerate(pages)
                  if hangul_share("".join(b.get("content") or "" for b in p.get("blocks", [])))
                  >= 0.6]
    info(f"번역면 한글 페이지 {len(translated)}/{len(pages)}, PDF 반영 실패 {len(dropped)}")

    # 유실을 **설명된 것**과 **조용한 것**으로 가른다. 조판 한계로 원문을 지킨 경우는
    # export report의 kept_reasons/경고가 그 페이지를 설명한다(예: 의사코드 리스팅을
    # 리플로우하면 줄 구조가 깨지므로 보존하는 편이 옳다). 잡아야 할 회귀는
    # **아무 사유도 남기지 않고 사라지는** 유실이다 — 그건 사용자가 원인을 알 수 없다.
    rep_path = jd / f"export.{lang}.report.json"
    explained_pages: set[int] = set()
    if rep_path.is_file():
        rep_json = json.loads(rep_path.read_text(encoding="utf-8"))
        for warn in rep_json.get("warnings") or []:
            m = re.match(r"p(\d+):", str(warn))
            if m:
                explained_pages.add(int(m.group(1)))
    silent = [d for d in dropped if d[0] not in explained_pages]
    if dropped and not silent:
        info(f"유실 {len(dropped)}페이지는 전부 export report에 사유가 남아 있다 "
             f"(조판 한계): {[d[0] for d in dropped]}")
    check(f"사유 없이 번역이 사라진 페이지 없음 (layout.{lang}.json ↔ export PDF)",
          not silent,
          f"무성 유실 {len(silent)}페이지 (페이지, layout비, pdf비)={silent[:6]}")

    rep = jd / f"export.{lang}.report.json"
    if rep.is_file():
        report = json.loads(rep.read_text(encoding="utf-8"))
        info(f"export report: replaced={report.get('replaced')} kept={report.get('kept')} "
             f"relocated={report.get('relocated')} warning_count={report.get('warning_count')}")
        summary = kept_reason_summary(report)
        for reason, n in sorted(summary.items(), key=lambda kv: -kv[1])[:10]:
            info(f"  보존 사유 {n:>3}건 — {reason}")
        if (report.get("warning_count") or 0) > len(report.get("warnings") or []):
            info("  (warnings는 상위 50건만 저장됨 — 집계는 표본이다)")


def verify_cropbox(jd: Path, expect_pages: int) -> None:
    """A-2 실증 — 실제 논문 PDF에 CropBox를 씌워 하단 가드를 강제로 태운다."""
    print("\n[5] A-2 CropBox≠MediaBox 실증 (실제 논문 사본)")
    import pymupdf

    from app.pipeline.pdf_export import build_translated_pdf

    src = jd / "source.pdf"
    if not src.is_file():
        check("source.pdf 존재", False)
        return
    cropped = WORK / "cropped-source.pdf"
    doc = pymupdf.open(src)
    for page in doc:
        r = page.rect
        # 하단 90pt·상단 30pt를 표시 영역에서 제외 — 저널 후처리본에서 흔한 형태
        page.set_cropbox(pymupdf.Rect(r.x0 + 15, r.y0 + 30, r.x1 - 15, r.y1 - 90))
    doc.save(cropped)
    doc.close()

    work = WORK / "cropbox-job"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy(cropped, work / "source.pdf")
    for name in ("layout.json", "layout.ko.json", "result.md", "result.ko.md"):
        if (jd / name).is_file():
            shutil.copy(jd / name, work / name)

    try:
        report = build_translated_pdf(work, "ko")
    except Exception as e:  # noqa: BLE001 — 하네스
        check("CropBox PDF 내보내기 성공", False, f"{type(e).__name__}: {e}")
        return
    check("CropBox PDF 내보내기 성공", True,
          f"replaced={getattr(report, 'replaced', '?')} kept={getattr(report, 'kept', '?')}")

    out = work / "export.ko.pdf"
    if not out.is_file():
        check("CropBox export.ko.pdf 생성", False)
        return
    d = pymupdf.open(out)
    outside = []
    extracted = 0
    for pno in range(d.page_count):
        page = d[pno]
        extracted += len(page.get_text().strip())
        pr = page.rect
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                if y1 > pr.y1 + 1 or y0 < pr.y0 - 1:
                    outside.append((pno + 1, round(y1, 1), round(pr.y1, 1)))
    d.close()
    check("A-2: CropBox 문서에서도 모든 줄이 표시 영역 안", not outside,
          f"이탈 {len(outside)}줄 (예: {outside[:3]})")
    check("A-2: CropBox 문서에서 텍스트가 정상 추출됨", extracted > 500, f"{extracted}자")


def verify_viewer(job_id: str, expect_pages: int) -> None:
    print("\n[6] 뷰어/리더 계약")
    status, man, hdrs = req("GET", f"/api/jobs/{job_id}/viewer-manifest?lang=ko")
    if check("GET /viewer-manifest 200", status == 200, f"{status}"):
        doc = man.get("document", {})
        check("manifest 페이지 수 일치", doc.get("page_count") == expect_pages,
              f"{doc.get('page_count')}")
        check("manifest capabilities.alignment 활성",
              man.get("capabilities", {}).get("alignment") is True,
              json.dumps(man.get("capabilities", {}), ensure_ascii=False))
        info(f"manifest quality={json.dumps(man.get('quality', {}), ensure_ascii=False)[:200]}")
        check("manifest ETag 제공", "etag" in {k.lower() for k in hdrs})
        etag = hdrs.get("etag") or hdrs.get("ETag")
        if etag:
            s2, _, _ = req("GET", f"/api/jobs/{job_id}/viewer-manifest?lang=ko",
                           headers={"If-None-Match": etag})
            check("If-None-Match → 304", s2 == 304, f"{s2}")

    status, out, _ = req("GET", f"/api/jobs/{job_id}/outline?lang=ko")
    check("GET /outline 200", status == 200, f"{status}")

    status, al, _ = req("GET", f"/api/jobs/{job_id}/alignment?page=1&lang=ko")
    check("GET /alignment?page=1 200 (409면 대응 불변식 위반)", status == 200, f"{status}")
    if status == 200:
        blocks = al.get("blocks", [])
        check("alignment 블록이 원문/번역 쌍을 제공", bool(blocks), f"{len(blocks)} blocks")
        # 프롬프트 스캐폴딩이 번역 결과로 새면 안 된다 (마스킹/복원 계약).
        leaked = [(b["id"], scaffolding_hits(b.get("target") or "")) for b in blocks]
        leaked = [x for x in leaked if x[1]]
        check("프롬프트 스캐폴딩이 번역 결과로 새지 않음", not leaked,
              f"오염 {len(leaked)}블록 (예: {leaked[:2]})")

    status, vp, _ = req("GET", f"/api/jobs/{job_id}/viewer/pages?start=1&limit=4&lang=ko&include=alignment")
    if check("GET /viewer/pages 200", status == 200, f"{status}"):
        items = vp.get("items", [])
        check("viewer/pages 배치 크기", 1 <= len(items) <= 4, f"{len(items)}")

    status, _, _ = req("GET", f"/api/jobs/{job_id}/viewer/pages?start=1&limit=99&lang=ko")
    check("viewer/pages limit 경계 422", status == 422, f"{status}")

    status, payload, _ = req("GET", f"/api/jobs/{job_id}/page/1?lang=ko", raw=True, timeout=300)
    check("GET /page/1?lang=ko 200 (번역 facsimile 래스터)", status == 200, f"{status}")
    if status == 200:
        check("페이지 PNG가 유효", payload[:8] == b"\x89PNG\r\n\x1a\n", f"{payload[:8]!r}")


def verify_security(job_id: str) -> None:
    print("\n[7] 보안 — C-1 키 분리 · 경로 탈출")
    status, prov, _ = req("GET", "/api/providers")
    info(f"/api/providers → {json.dumps(prov, ensure_ascii=False)[:300]}")

    # C-1: LLM_OPENAI_API_KEY 미설정 상태에서 번역 키가 Q&A로 새면 안 된다.
    status, js, _ = req("POST", f"/api/jobs/{job_id}/qa",
                        body={"page": 1, "question": "이 페이지의 주제는?",
                              "provider": "openai-responses"})
    check("C-1: LLM_OPENAI_API_KEY 미설정 시 Q&A가 번역 키로 실행되지 않음",
          status in (400, 422, 503), f"status={status} {str(js)[:200]}")

    # 경로 탈출 — 잡 디렉터리 밖/원본 PDF 서빙 금지
    for probe in ("../../../etc/passwd", "..%2f..%2fsource.pdf", "source.pdf",
                  "pages/../source.pdf", "translations/ko/units.json"):
        status, _, _ = req("GET", f"/api/jobs/{job_id}/files/{probe}", raw=True)
        check(f"/files 경로 차단: {probe}", status in (400, 403, 404), f"status={status}")

    # 호스트 화이트리스트
    status, _, _ = req("GET", "/api/health", headers={"Host": "evil.example.com"})
    check("ALLOWED_HOSTS 위반 Host 거절", status == 400, f"status={status}")


# fault별 "손상 출력이 그대로 반영됐다"는 증거 문구. mock_llm._apply_fault()가 내는
# 문자열에서 왔다 — **부분일치**로 본다. 예전 코드는 6,491자 파일 전체를
# `body.strip() == "요약입니다."` 로 비교해 절대 참이 될 수 없었다(공허한 검사).
_FAULT_EXPECT: dict[str, tuple[str, tuple[str, ...]]] = {
    "refusal": ("영문 거부문", ("cannot translate",)),
    "refusal_ko": ("한국어 거부문", ("죄송합니다", "번역할 수 없습니다")),
    "echo": ("원문 echo", ()),  # echo는 문구가 아니라 구조(줄 수·라틴 문자)로 잡는다
    "summary": ("한 줄 요약", ("요약입니다",)),
    # 목이 수식 플레이스홀더(<mN …/>)만 지운 응답을 낸다. 문구가 아니라 **보호 토큰
    # 수**로 잡는다 — unmask가 missing을 보고하지 않고 통과시키면 수식·코드가
    # 산출물에서 조용히 증발한다(문서가 주장하던 커버리지의 실제 대상).
    "drop_placeholder": ("플레이스홀더 유실", ()),
}

# 이 fault들은 FAULT 환경변수 대신 **OPENAI_BASE_URL의 `?fault=` 쿼리**로 주입한다.
# 문서(ARCHITECTURE §11.1)가 두 경로를 모두 주장하는데 하네스는 env만 썼다 —
# client._endpoint_url()이 base의 query를 보존하므로 쿼리 경로도 실제로 살아 있고,
# 여기서 한 번은 그 경로로 돌려 문서가 사실임을 실행으로 증명한다.
_FAULT_VIA_URL = frozenset({"drop_placeholder"})


def protected_token_count(text: str) -> int:
    """masking.mask()가 보호하는 비언어 토큰 수 (수식·코드·이미지·URL·인용·참조).

    하네스가 문구를 흉내 내지 않고 **실제 마스커**를 부른다 — 토큰 규칙이 바뀌어도
    원문/번역본을 같은 자로 재므로 비교가 계속 유효하다.
    """
    from app.translate.masking import mask

    return len(mask(text)[1])


def verify_translation_faults(pdf: Path, data_dir: Path) -> None:
    """A-1 실증 — 거부문/echo/요약/플레이스홀더 유실/HTTP 오류를 주입하고 무성 손실을 본다.

    mock_llm.py가 제공하는 결함 모드는 **전부** 여기서 돈다(tests/test_ci_ops_contracts.py
    가 목록 일치를 강제한다). 코드만 있고 한 번도 실행되지 않는 결함 모드는
    문서상의 커버리지만 부풀리고 아무것도 지키지 않는다.
    """
    print("\n[8] A-1 번역 결함 주입 (거부문 · echo · 플레이스홀더 유실 · HTTP 오류)")
    info(f"스캐폴딩 마커 {len(scaffolding_markers())}종: "
         + ", ".join(scaffolding_markers())[:200])
    for fault, (label, _) in _FAULT_EXPECT.items():
        dd = data_dir.parent / f"fault-{fault}"
        if dd.exists():
            shutil.rmtree(dd)
        if fault in _FAULT_VIA_URL:
            env_extra = {"OPENAI_BASE_URL": f"{MOCK}/v1?fault={fault}",
                         "TRANSLATE_MAX_RETRIES": "0"}
        else:
            env_extra = {"FAULT": fault, "TRANSLATE_MAX_RETRIES": "0"}
        with Servers(dd, env_extra):
            job_id = upload(pdf, dpi=100)
            job = wait_job(job_id)
            if job.get("status") != "done":
                check(f"A-1[{label}]: 사전 OCR 성공", False, str(job.get("error")))
                continue
            req("POST", f"/api/jobs/{job_id}/translate", body={"lang": "ko"})
            st = wait_translate(job_id)
            jd = dd / "jobs" / job_id
            rep = jd / "translations/ko/report.json"
            units = jd / "translations/ko/units.json"
            kept = []
            if rep.is_file():
                kept = json.loads(rep.read_text()).get("kept_original", [])
            n_units = len(json.loads(units.read_text())) if units.is_file() else 0
            ko = (jd / "result.ko.md")
            body = ko.read_text(encoding="utf-8") if ko.is_file() else ""
            src = (jd / "result.md").read_text(encoding="utf-8") if (jd / "result.md").is_file() else ""

            # (a) 이 fault가 내는 손상 문구가 산출물에 남았는가 — 부분일치.
            hits = [p for p in _FAULT_EXPECT[fault][1] if p in body.lower() or p in body]
            check(f"A-1[{label}]: 손상 출력이 result.ko.md에 반영되지 않음", not hits,
                  f"오염 문구 {hits} state={st.get('status')}")

            # (b) 모든 fault 공통 — 프롬프트/repair 스캐폴딩 누출.
            hits = scaffolding_hits(body)
            check(f"A-1[{label}]: 프롬프트·repair 스캐폴딩이 result.ko.md에 새지 않음",
                  not hits, f"누출 {hits[:4]}")

            # (c) echo 전용 구조 단언 — 원문 echo면 산출물이 원문과 같은 규모여야 한다.
            #     실측: 원문 108줄 → 170줄로 불어나며 repair 스캐폴딩 25줄이 박혔는데
            #     기존 검사는 통과했다. 문구가 바뀌어도 이 단언은 남는다.
            if fault == "echo" and src:
                src_lines, ko_lines = src.count("\n") + 1, body.count("\n") + 1
                src_latin, ko_latin = hangul_latin(src)[1], hangul_latin(body)[1]
                check(f"A-1[{label}]: 산출물 줄 수가 원문 대비 부풀지 않음",
                      ko_lines <= src_lines * 1.15 + 3, f"원문 {src_lines}줄 → {ko_lines}줄")
                check(f"A-1[{label}]: 라틴 문자 수가 원문 대비 부풀지 않음",
                      ko_latin <= src_latin * 1.15 + 50,
                      f"원문 {src_latin}자 → {ko_latin}자")

            # (d) drop_placeholder 전용 — 수식·코드가 조용히 증발하지 않았는가.
            #     플레이스홀더가 사라진 응답을 그대로 채택하면 그 유닛의 보호 토큰이
            #     통째로 없어진다. 원문 유지로 강등되면 1:1로 남는다.
            if fault == "drop_placeholder" and src:
                src_tok, ko_tok = protected_token_count(src), protected_token_count(body)
                check(f"A-1[{label}]: 보호 토큰(수식·코드·인용)이 산출물에서 증발하지 않음",
                      ko_tok >= src_tok * 0.9,
                      f"원문 {src_tok}개 → 번역본 {ko_tok}개")
                # 복원 실패 잔여물(`<m1` 류)이 사용자 문서에 그대로 남으면 안 된다.
                residual = re.findall(r"<[mkgucft]\d+", body)
                check(f"A-1[{label}]: 플레이스홀더 잔여물이 result.ko.md에 남지 않음",
                      not residual, f"잔여 {len(residual)}개 (예: {residual[:3]})")

            check(f"A-1[{label}]: kept_original에 기록되어 관측 가능", len(kept) > 0,
                  f"kept={len(kept)} units_cached={n_units}")

    # 결정적 4xx / 재시도성 429 — 잡 전체가 죽지 않고 상태로 마감되는지.
    # 429는 Retry-After: 86400 을 함께 보낸다. 클라이언트가 상한(_MAX_BACKOFF_S)을
    # 걸지 않고 헤더를 그대로 따르면 워커가 하루 묶여 "번역이 멈춘 것처럼" 보인다 —
    # 그래서 벽시계 시간까지 함께 잰다(단언 없이 지나가면 관측되지 않는 결함이다).
    for fault, label, extra in (
        ("http400", "결정적 400", {"TRANSLATE_MAX_RETRIES": "0"}),
        # 429는 재시도를 1회 허용해야 백오프 경로가 실제로 돈다. 동시성 1로 낮춰
        # 30초(_MAX_BACKOFF_S) 대기를 **한 번만** 치른다 — 기본 4면 라운드가 겹쳐
        # 하네스가 60초 이상 늘어난다(실측).
        ("http429", "재시도성 429", {"TRANSLATE_MAX_RETRIES": "1",
                                     "TRANSLATE_CONCURRENCY": "1"}),
    ):
        dd = data_dir.parent / f"fault-{fault}"
        if dd.exists():
            shutil.rmtree(dd)
        with Servers(dd, {"FAULT": fault, **extra}):
            job_id = upload(pdf, dpi=100)
            if wait_job(job_id).get("status") != "done":
                check(f"A-1[{label}]: 사전 OCR 성공", False)
                continue
            t0 = time.time()
            req("POST", f"/api/jobs/{job_id}/translate", body={"lang": "ko"})
            st = wait_translate(job_id, timeout=600)
            elapsed = time.time() - t0
            info(f"{label} 주입 시 번역 state={st.get('status')} "
                 f"{elapsed:.0f}초 error={str(st.get('error'))[:120]}")
            check(f"{label}가 서버를 죽이지 않고 상태로 마감됨",
                  st.get("status") in ("error", "done"), f"{st.get('status')}")
            if fault == "http429":
                # 양쪽 경계를 다 본다. 하한(20초)이 없으면 "429를 재시도조차 안 하게"
                # 퇴화해도 통과하고, 상한(300초)이 없으면 Retry-After: 86400을 그대로
                # 따라 워커가 하루 묶여도 통과한다.
                check("429 재시도 백오프가 Retry-After 상한(30초) 안에서 실제로 일어남",
                      20 <= elapsed < 300, f"{elapsed:.0f}초 (기대 20~300초)")
            s, health, _ = req("GET", "/api/health")
            check(f"{label} 뒤에도 백엔드가 살아 있음", s == 200, f"status={s}")


def verify_worker_resilience(pdf: Path, data_dir: Path) -> None:
    """B-1 실증 — 손상 업로드가 있어도 서버·워커가 살아 다음 잡을 처리하는지.

    백엔드에 테스트 전용 예외 주입 훅은 없다. 있는 척(존재하지 않는 env 이름을
    넘기는 식)하면 1번 잡이 그냥 성공해 이 단계가 조용히 무력화되므로, 공개 API로
    관측 가능한 경로만 쓴다 — 헤더는 %PDF-라 5바이트 검사를 통과하지만 본문이
    깨져 파싱에서 거부되는 파일을 올리고, 그 뒤 정상 잡이 끝까지 가는지 본다.
    """
    print("\n[9] B-1 워커 복원력 (손상 업로드 거부 후 후속 잡 처리)")
    dd = data_dir.parent / "worker-resilience"
    if dd.exists():
        shutil.rmtree(dd)
    broken = WORK / "broken-header-only.pdf"
    broken.write_bytes(b"%PDF-1.7\n" + b"\x00 not a real pdf body \x00" * 64)
    with Servers(dd, {}):
        status, js = post_pdf(broken, dpi=100)
        check("B-1: 손상 PDF는 5xx가 아니라 4xx로 거부됨",
              400 <= status < 500, f"status={status} {str(js)[:120]}")
        second = upload(pdf, dpi=100)
        j2 = wait_job(second, timeout=300)
        check("B-1: 앞 잡이 실패해도 워커가 살아 다음 잡을 완료함",
              j2.get("status") == "done", f"2번 잡 status={j2.get('status')}")
        s, health, _ = req("GET", "/api/health")
        if isinstance(health, dict) and "worker_alive" in health:
            check("health.worker_alive 노출 및 True", health.get("worker_alive") is True,
                  str(health.get("worker_alive")))
        else:
            info("health에 worker_alive 필드 없음 (선택 항목)")


def main() -> int:
    global WORK, MOCK_PORT, API_PORT, BASE, MOCK

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(REPO / "sample/2504.19874v1.pdf"))
    ap.add_argument("--pages", type=int, default=0, help="앞 N페이지만 사용 (0=전체)")
    ap.add_argument("--skip", default="", help="건너뛸 단계 쉼표 구분 (faults,worker,cropbox)")
    ap.add_argument("--work", default=str(REPO / "tmp" / "verify-e2e"),
                    help="산출물/서버 로그 디렉터리 (기본 <repo>/tmp/verify-e2e). "
                         "동시 실행하려면 실행마다 다른 값을 줘야 한다 — 배타 락이 걸린다")
    args = ap.parse_args()

    WORK = Path(args.work).resolve()
    MOCK_PORT = _free_port()
    API_PORT = _free_port()
    BASE = f"http://127.0.0.1:{API_PORT}"
    MOCK = f"http://127.0.0.1:{MOCK_PORT}"

    # 포트는 매번 갈리지만 작업 디렉터리는 안 갈린다 — 동시 실행이 서로의 잡 데이터를
    # rmtree 하지 못하도록 여기서 배타 락을 잡는다 (finally에서 해제).
    lock = acquire_work_lock(WORK)
    try:
        return _run(args)
    finally:
        release_work_lock(lock)


def _run(args) -> int:
    sys.path.insert(0, str(REPO / "backend"))
    import pymupdf

    src = Path(args.pdf)
    if not src.is_file():
        print(f"대상 PDF가 없습니다: {src}", file=sys.stderr)
        return 2
    pdf = src
    if args.pages:
        pdf = WORK / f"trimmed-{args.pages}p.pdf"
        d = pymupdf.open(src)
        d.select(list(range(min(args.pages, d.page_count))))
        d.save(pdf)
        d.close()
    n_pages = pymupdf.open(pdf).page_count
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print(f"대상 PDF: {pdf} ({n_pages}페이지)")
    data_dir = WORK / "data-main"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    print("\n[1] 서버 기동 + 업로드/OCR")
    with Servers(data_dir, {}):
        s, health, _ = req("GET", "/api/health")
        check("GET /api/health 200", s == 200, json.dumps(health, ensure_ascii=False)[:200])
        check("엔진이 textlayer", health.get("engine") == "textlayer", str(health.get("engine")))
        check("번역 프로바이더 활성", health.get("translate_available") is True)

        job_id = upload(pdf, dpi=150)
        job = wait_job(job_id)
        jd = verify_ocr(job_id, job, data_dir, n_pages)
        verify_translation(job_id, jd, n_pages)
        verify_pdf_export(job_id, jd, n_pages)
        verify_viewer(job_id, n_pages)
        verify_security(job_id)

    if "cropbox" not in skip:
        verify_cropbox(data_dir / "jobs" / job_id, n_pages)
    if "faults" not in skip:
        verify_translation_faults(pdf, data_dir)
    if "worker" not in skip:
        verify_worker_resilience(pdf, data_dir)

    print("\n" + "=" * 70)
    print(f"통과 {len(PASS)} / 실패 {len(FAIL)}")
    if FAIL:
        print("\n실패 항목:")
        for f in FAIL:
            print("  ✗ " + f)
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
