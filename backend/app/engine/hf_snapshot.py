"""고정 커밋 HF 스냅샷은 로컬 캐시에서 먼저 읽는다 (torch·MLX 엔진 공용).

from_pretrained·snapshot_download는 캐시가 완전해도 로드마다 Hub에 묻는다 — 업스트림에 없는
선택 파일(generation_config.json·custom_generate/generate.py 등)을 HEAD로 확인하고
(.no_exist 표식을 매번 새로 쓴다), snapshot_download는 리비전 정보를 API로 조회한다. 그래서
자체 호스팅 서버가 모델을 올릴 때마다 huggingface.co에 접속했고, 오프라인·에어갭 호스트는
기동마다 네트워크 시도(프로브당 최대 10초)를 기다렸다(P4 Docker 재현 — compose는
HF_HUB_OFFLINE을 넘기지도 않는다).

커밋 해시(40자 hex)로 고정한 리비전은 내용이 바뀌지 않으므로 캐시에 완전히 있으면 네트워크 없이
쓴다. 없거나 불완전하면 예전처럼 Hub에서 받는다(첫 기동·부분 다운로드 복구). 브랜치·태그
리비전(main 등)은 갱신을 받아야 하므로 로컬 우선을 쓰지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")


def is_pinned_revision(revision: str | None) -> bool:
    """리비전이 커밋 해시(40자 소문자 hex)인가 — 브랜치·태그·짧은 해시는 아니다."""
    return bool(revision) and _COMMIT_REVISION.fullmatch(revision) is not None


def pretrained_local_first(
    load: Callable[..., Any], model_id: str, revision: str | None, **kwargs: Any,
) -> Any:
    """`load(model_id, revision=…, **kwargs)`(from_pretrained) — 고정 리비전은 캐시만 먼저 본다.

    transformers는 캐시에 필요한 파일이 하나라도 없으면 local_files_only에서 OSError를 낸다
    (huggingface_hub의 LocalEntryNotFoundError도 OSError다). 그때만 Hub에 묻는 호출로 다시 한다."""
    if is_pinned_revision(revision):
        try:
            return load(model_id, revision=revision, local_files_only=True, **kwargs)
        except OSError as error:
            logger.info(
                "HF 캐시에 고정 스냅샷 %s@%s가 완전하지 않아 Hub에서 받습니다: %s",
                model_id, str(revision)[:8], str(error)[:200],
            )
    return load(model_id, revision=revision, **kwargs)


def complete_local_snapshot(model_id: str, revision: str | None) -> Path | None:
    """고정 리비전의 스냅샷이 캐시에 **완전히** 있으면 그 경로, 아니면 None(→ Hub 경로).

    완전 = config.json·tokenizer_config.json과, 가중치 인덱스가 가리키는 샤드 전부(인덱스가
    없으면 *.safetensors 하나 이상). snapshot_download(local_files_only)는 스냅샷 폴더만 있으면
    파일이 빠져 있어도 돌려주므로 여기서 확인한다 — 중단된 다운로드는 예전처럼 Hub가 채운다."""
    if not is_pinned_revision(revision):
        return None
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception:  # noqa: BLE001 — LocalEntryNotFoundError 등: 캐시에 없음
        return None
    if not all((snap / name).is_file() for name in ("config.json", "tokenizer_config.json")):
        return None
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    else:
        shards = {path.name for path in snap.glob("*.safetensors")}
    if not shards or not all(isinstance(name, str) and (snap / name).is_file() for name in shards):
        return None
    return snap
