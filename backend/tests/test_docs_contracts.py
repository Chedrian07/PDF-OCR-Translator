"""운영 문서 ↔ 코드·배포 구성 대조 회귀 테스트 (감사 api-5·infra-docs-2·infra-docs-3).

운영 안내가 틀려도 다른 테스트는 깨지지 않는다 — 운영자는 그 안내를 믿고 확인을 건너뛴다.
여기서는 문서의 주장을 그 주장이 기대는 코드·compose 구성에서 다시 계산해 대조한다.
ARCHITECTURE의 SSE 폴백 설명은 frontend/tests/docs-sse-fallback.test.mjs가 맡는다.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def _yaml(name: str) -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML 없음 — compose 대조 생략")
    return yaml.safe_load(_read(name))


def _section(text: str, heading: str) -> str:
    """heading 줄부터 같은 깊이 이하의 다음 제목 직전까지 — 코드 블록 안의 '# …' 줄은 제목이 아니다."""
    lines = text.splitlines()
    start = lines.index(heading)
    depth = len(heading) - len(heading.lstrip("#"))
    fenced = False
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(rf"#{{1,{depth}}} ", lines[i]):
            return "\n".join(lines[start:i])
    return "\n".join(lines[start:])


# ── api-5: 컨테이너에는 .env가 없다 ──────────────────────────────────────────
# config_warnings는 앱이 직접 읽은 .env에서만 나온다(config.load_dotenv_file). 이미지에는
# .env가 없고(.dockerignore) compose도 넘기지 않아(env_file 없음) Docker에서는 늘 빈 목록인데,
# README Docker 절이 이를 '.env 키 점검' 수단으로 안내해 거짓 안심을 줬다.

_HOST_DOTENV_CHECK = re.compile(r"docker compose exec -T (\S+) python -c '([^']+)' < \.env")


def test_docker_quickstart_checks_dotenv_keys_from_the_host():
    compose = _yaml("docker-compose.yml")
    ignored = set(_read(".dockerignore").splitlines())
    assert ignored & {".env", "**/.env"}, "전제가 바뀌었다: 이미지에 .env가 들어간다"
    assert not [s for s, spec in compose["services"].items() if "env_file" in spec], (
        "전제가 바뀌었다: compose가 .env를 컨테이너에 넘긴다"
    )

    docker = _section(_read("README.md"), "## 빠른 시작 (Docker)")
    for chunk in re.split(r"\n\s*\n|\n(?=- )", docker):
        flat = " ".join(chunk.split())
        if "config_warnings" in flat:
            assert "빈 목록" in flat or "로컬" in flat, (
                f"Docker 절이 config_warnings를 컨테이너의 .env 점검 수단처럼 안내한다:\n{chunk}"
            )

    m = _HOST_DOTENV_CHECK.search(docker)
    assert m, "Docker 절에 호스트 .env 점검 명령(docker compose exec -T <서비스> python -c '…' < .env)이 없다"
    assert m.group(1) in compose["services"], m.group(1)
    # 안내한 스니펫이 지금 코드로 실제로 돈다 — 모르는 키의 이름만 내고 값·아는 키·다른 도구 키는 내지 않는다
    sample = "REASONING_EFFORT=off\nOPENAI_API_KEY=sk-docs-contract-secret\nOCR_DEVICE=cpu\nHF_HOME=/x\n"
    proc = subprocess.run(
        [sys.executable, "-c", m.group(2)], input=sample, cwd=REPO / "backend",
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert [line.split(":", 1)[0] for line in proc.stdout.splitlines()] == ["REASONING_EFFORT"], proc.stdout
    assert "sk-docs-contract-secret" not in proc.stdout + proc.stderr


# ── infra-docs-2: 하드닝 보증의 범위 ─────────────────────────────────────────
# README가 '모든 compose 서비스'를 비루트·cap_drop ALL이라고 적었지만, 선택 overlay
# compose.ollama.yaml의 ollama는 이미지 기본(root)·Docker 기본 캡으로 돈다.


def test_readme_hardening_scope_matches_the_ollama_overlay():
    ollama = _yaml("compose.ollama.yaml")["services"]["ollama"]
    user = str(ollama.get("user", "")).split(":", 1)[0]
    hardened = ollama.get("cap_drop") == ["ALL"] and user not in ("", "0", "root")
    section = " ".join(_section(_read("README.md"), "### 컨테이너 하드닝").split())
    names_exception = re.search(r"compose\.ollama\.yaml[^.]*root", section) is not None
    if hardened:
        assert not names_exception, "ollama overlay가 하드닝됐다 — README의 root 예외 문구를 지운다"
    else:
        assert "모든 compose 서비스" not in section, "선택 overlay(ollama)까지 포함하는 보증이다"
        assert names_exception, "README 하드닝 절에 compose.ollama.yaml 예외(root·기본 캡)가 없다"


# ── infra-docs-3: .env.example의 MoE 스위치 설명 ↔ 벤더 기본 디바이스 ────────
# P17 융합 MoE가 MPS 기본이 됐는데 .env.example은 'CUDA에서 on'으로 남아, torch MPS 폴백
# 사용자가 OCR_MOE_FUSED=0 킬스위치(P18 복원)를 CUDA 전용으로 읽었다.

_VENDOR_MOE = "backend/app/vendor/unlimited_ocr/modeling_deepseekv2.py"


def _env_example_block(key: str) -> str:
    """'# KEY=…' 줄과 이어지는 들여쓴 설명 줄('#   …')을 한 줄로 편 것."""
    lines = _read(".env.example").splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(rf"# ?{key}=", line)]
    assert starts, f".env.example에 {key} 줄이 없다"
    block = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith("#  "):
            break
        block.append(line)
    return " ".join(" ".join(block).split())


def test_env_example_moe_switches_name_the_vendor_default_devices():
    vendor = _read(_VENDOR_MOE)
    fused = re.search(r"^_FUSED_DEVICE_TYPES = \(([^)]*)\)", vendor, re.MULTILINE)
    fast = re.search(r'def _moe_fast_enabled\(device_type\):.*?return device_type == "(\w+)"', vendor, re.DOTALL)
    assert fused and fast, "벤더의 MoE 기본 디바이스 표기를 못 찾았다 — 이 테스트의 파서를 갱신한다"
    fused_devices = [d.upper() for d in re.findall(r'"(\w+)"', fused.group(1))]
    assert fused_devices, fused.group(0)

    fused_doc = _env_example_block("OCR_MOE_FUSED")
    missing = [d for d in fused_devices if d not in fused_doc]
    assert not missing, f"OCR_MOE_FUSED 설명에 P17 기본 디바이스 {missing}가 없다: {fused_doc}"

    fast_doc = _env_example_block("OCR_MOE_FAST")
    assert fast.group(1).upper() in fast_doc, f"OCR_MOE_FAST 설명에 P18 기본 디바이스가 없다: {fast_doc}"
    # P18은 이제 P17이 받지 않을 때의 폴백이다 — 그 관계가 빠지면 FAST=1이 기본 경로를 바꾼다고 읽힌다
    assert "OCR_MOE_FUSED=0" in fast_doc, fast_doc


# ── docs-final-3: 캐시 버전 상수의 '현재 값'은 한 곳(§15.1 표)에서 코드와 같다 ─────────
# §5 /pdf 절이 PDF_EXPORT_FORMAT_VERSION을 '현재 **11**'로 따로 적어, 표만 16으로 고친 갱신이
# 그 줄을 놓쳤다 — 같은 SSOT 안에서 11과 16이 갈렸다.


def test_architecture_cache_version_table_matches_the_code():
    from app.pipeline.pdf_export.report import PDF_EXPORT_FORMAT_VERSION
    from app.pipeline.pdf_fonts import ENRICH_VERSION
    from app.translate.types import PROMPT_V

    code = {
        "PDF_EXPORT_FORMAT_VERSION": str(PDF_EXPORT_FORMAT_VERSION),
        "ENRICH_VERSION": str(ENRICH_VERSION),
        "PROMPT_V": str(PROMPT_V),
    }
    arch = _read("docs/ARCHITECTURE.md")
    table = {
        m.group(1): m.group(2)
        for m in re.finditer(r'^\| `(\w+)` \| `[^`]+` \| `"?([^`"]+)"?` \|', arch, re.M)
    }
    assert {name: table.get(name) for name in code} == code
    # 다른 절은 현재 값을 따로 적지 않는다(표를 가리킨다) — 따로 적으면 다음 상향 때 갈린다
    stale = re.findall(r"`(PDF_EXPORT_FORMAT_VERSION|ENRICH_VERSION|PROMPT_V)`[^\n]{0,80}현재 \*\*", arch)
    assert not stale, stale
    upgrade = re.search(r"`PDF_EXPORT_FORMAT_VERSION`\(→ (\d+)\)", arch)
    assert upgrade and upgrade.group(1) == code["PDF_EXPORT_FORMAT_VERSION"]
    changelog = re.search(r"`PDF_EXPORT_FORMAT_VERSION` is now (\d+)", _read("CHANGELOG.md"))
    assert changelog and changelog.group(1) == code["PDF_EXPORT_FORMAT_VERSION"]
