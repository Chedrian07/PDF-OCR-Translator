#!/usr/bin/env python3
"""make test-mlx-real 앞단 — make dev와 같은 HF 캐시에서 고정 스냅샷을 찾고, 있을 때만 명령을 실행한다.

사용 (backend/에서 — Makefile의 test-mlx-real):
    .venv/bin/python ../scripts/require_hf_snapshot.py [--dotenv PATH] -- CMD [ARG …]

- 캐시 위치: 테스트 프로세스는 .env를 읽지 않는다(tests/conftest.py가 DISABLE_DOTENV=1 — 실키
  격리). 그래서 HF 캐시 위치를 .env에만 둔 개발자는 make dev가 받은 스냅샷을 두고 테스트가 기본
  ~/.cache/huggingface를 뒤졌다. make dev(app.config.load_dotenv_file)와 같은 파일(실행 cwd →
  저장소 루트의 첫 .env, DISABLE_DOTENV면 읽지 않음)·같은 파서(python-dotenv)·같은 우선순위(셸
  값이 이긴다)로 읽되, 허브 캐시 위치를 정하는 키(CACHE_KEYS)만 넘긴다 — HF_TOKEN·번역 키 같은
  나머지는 테스트 프로세스에 넣지 않는다.
- 스냅샷: 테스트가 쓰는 코드 기본값(Settings() — .env·셸의 MODEL_ID·MODEL_REVISION과 무관)의
  가중치가 그 캐시에 없으면 사유·해결법을 보이고 명령을 실행하지 않은 채 종료코드 2로 끝난다.
  예전에는 실가중치 테스트가 LocalEntryNotFoundError로 실패해 회귀처럼 보였다(감사 mlx-3·
  infra-docs-4). 2 = 검증 실패(1)가 아니라 검증을 시작하지 못함 — verify_e2e와 같은 구분이다.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
# huggingface_hub·transformers가 허브 캐시 위치를 정할 때 보는 키 (우선순위: HF_HUB_CACHE >
# HUGGINGFACE_HUB_CACHE(옛 이름) > HF_HOME/hub > XDG_CACHE_HOME/huggingface/hub)
CACHE_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")
EXIT_NO_SNAPSHOT = 2


def _import_backend() -> None:
    """backend/app을 import할 수 있게 — cwd와 무관하게 이 리포의 backend를 본다."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


def dotenv_cache_env(dotenv: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    """dotenv에서 environ에 아직 없는 캐시 위치 키만 — load_dotenv_file과 같은 파서·우선순위."""
    if dotenv is None or not dotenv.is_file():
        return {}
    from dotenv import dotenv_values

    values = dotenv_values(dotenv, encoding="utf-8-sig")  # BOM — load_dotenv_file과 같다
    return {
        key: value
        for key in CACHE_KEYS
        if (value := values.get(key)) is not None and key not in environ
    }


def _auto_dotenv(environ: Mapping[str, str]) -> Path | None:
    """make dev가 읽는 바로 그 .env — DISABLE_DOTENV(참 값)면 자동 탐색을 끈다."""
    if environ.get("DISABLE_DOTENV", "").strip().lower() in ("1", "true", "yes", "on"):
        return None
    _import_backend()
    from app.config import _find_dotenv  # 실행 cwd → 저장소 루트 (load_dotenv_file과 같은 탐색)

    return _find_dotenv()


def find_snapshot() -> tuple[str, str, Path | None]:
    """(model_id, revision, 가중치가 있는 스냅샷 경로 또는 None).

    huggingface_hub는 import 시점에 캐시 위치를 읽는다 — 캐시 env를 정한 **뒤에** 부른다."""
    _import_backend()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from app.config import Settings

    s = Settings()  # 테스트(_settings())와 같은 코드 기본값
    try:
        snap = Path(snapshot_download(s.model_id, revision=s.model_revision, local_files_only=True))
    except LocalEntryNotFoundError:
        return s.model_id, s.model_revision, None
    # 스냅샷 폴더만 있고 가중치가 없으면(*.json만 받은 부분 스냅샷) 로더가 실패한다 — 없는 것으로 본다
    return s.model_id, s.model_revision, snap if any(snap.glob("*.safetensors")) else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="require_hf_snapshot.py",
        usage="%(prog)s [--dotenv PATH] -- CMD [ARG …]",
        description="고정 스냅샷이 HF 캐시에 있을 때만, 그 캐시 위치를 넘겨 CMD를 실행한다.",
    )
    ap.add_argument("--dotenv", type=Path, help="읽을 .env (기본: make dev와 같은 자동 탐색)")
    if "--" not in argv:
        ap.error("실행할 명령이 없다 — '-- CMD …'로 준다")
    cut = argv.index("--")
    args, cmd = ap.parse_args(argv[:cut]), argv[cut + 1:]
    if not cmd:
        ap.error("실행할 명령이 없다 — '-- CMD …'로 준다")

    dotenv = args.dotenv if args.dotenv is not None else _auto_dotenv(os.environ)
    forwarded = dotenv_cache_env(dotenv, os.environ)
    os.environ.update(forwarded)
    if forwarded:
        print(f"[require_hf_snapshot] {dotenv}의 HF 캐시 위치를 넘긴다: "
              f"{', '.join(sorted(forwarded))}", file=sys.stderr)

    model_id, revision, snap = find_snapshot()
    if snap is None:
        from huggingface_hub import constants

        print(
            f"[require_hf_snapshot] 고정 스냅샷 {model_id}@{revision[:12]}의 가중치가 HF 캐시"
            f"({constants.HF_HUB_CACHE})에 없다 — 명령을 실행하지 않는다.\n"
            "  · make dev를 한 번 띄우면(기본 프리로드) 받아진다.\n"
            "  · 다른 위치에 받아 뒀다면 HF_HOME·HF_HUB_CACHE를 셸에 export하거나 .env에 둔다.",
            file=sys.stderr,
        )
        return EXIT_NO_SNAPSHOT
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(cmd[0], cmd)  # 돌아오지 않는다 — 명령의 종료코드가 그대로 make에 간다


if __name__ == "__main__":
    sys.exit(main())
