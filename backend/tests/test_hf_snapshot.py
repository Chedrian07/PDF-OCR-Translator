"""고정 커밋 HF 스냅샷 로컬 우선 로딩 (app.engine.hf_snapshot).

예전에는 캐시가 완전해도 모델을 올릴 때마다 Hub에 물었다 — torch 경로는 업스트림에 없는 선택
파일 6개를 HEAD로·채팅 템플릿 목록을 API로(7회), MLX 경로는 리비전 정보를 API로(1회) 조회했다
(가짜 HF_ENDPOINT로 센 실측, P4). 고정 리비전이 캐시에 완전하면 이제 0회다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine import hf_snapshot

PINNED = "ee63731b6461c8afcdcc7b15352e7d2ffecc2ead"


@pytest.mark.parametrize(
    ("revision", "pinned"),
    [
        (PINNED, True),
        ("main", False),
        ("v1.0", False),
        (PINNED[:12], False),            # 짧은 해시는 브랜치·태그와 구분할 수 없다
        (PINNED.upper(), False),
        ("", False),
        (None, False),
    ],
)
def test_only_full_commit_hashes_count_as_pinned(revision, pinned):
    assert hf_snapshot.is_pinned_revision(revision) is pinned


class _Recorder:
    def __init__(self, fail_local: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_local = fail_local

    def __call__(self, model_id, **kwargs):
        self.calls.append({"model_id": model_id, **kwargs})
        if kwargs.get("local_files_only") and self.fail_local:
            raise OSError("We couldn't connect to 'https://huggingface.co' … cached files")
        return f"loaded:{model_id}"


def test_pinned_revision_is_read_from_the_cache_without_asking_the_hub():
    load = _Recorder()
    out = hf_snapshot.pretrained_local_first(load, "baidu/Unlimited-OCR", PINNED, dtype="bf16")
    assert out == "loaded:baidu/Unlimited-OCR"
    assert load.calls == [{
        "model_id": "baidu/Unlimited-OCR", "revision": PINNED, "local_files_only": True,
        "dtype": "bf16",
    }]


def test_incomplete_cache_falls_back_to_the_hub_once():
    load = _Recorder(fail_local=True)
    assert hf_snapshot.pretrained_local_first(load, "m", PINNED, dtype="x") == "loaded:m"
    assert load.calls == [
        {"model_id": "m", "revision": PINNED, "local_files_only": True, "dtype": "x"},
        {"model_id": "m", "revision": PINNED, "dtype": "x"},
    ]


def test_branch_revisions_still_ask_the_hub_for_updates():
    load = _Recorder()
    hf_snapshot.pretrained_local_first(load, "m", "main")
    assert load.calls == [{"model_id": "m", "revision": "main"}]


def _snapshot_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    snap = tmp_path / "snapshots" / PINNED
    snap.mkdir(parents=True)
    for name, body in files.items():
        (snap / name).write_text(body, encoding="utf-8")
    return snap


_INDEX = json.dumps({"weight_map": {"a.w": "model-00001-of-000002.safetensors",
                                    "b.w": "model-00002-of-000002.safetensors"}})
_COMPLETE = {
    "config.json": "{}", "tokenizer_config.json": "{}", "tokenizer.json": "{}",
    "model.safetensors.index.json": _INDEX,
    "model-00001-of-000002.safetensors": "w1", "model-00002-of-000002.safetensors": "w2",
}


def _fake_snapshot_download(monkeypatch, result):
    import huggingface_hub

    calls: list[dict] = []

    def _download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        if isinstance(result, BaseException):
            raise result
        return str(result)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
    return calls


def test_complete_cached_snapshot_is_used_as_a_local_directory(tmp_path, monkeypatch):
    snap = _snapshot_dir(tmp_path, _COMPLETE)
    calls = _fake_snapshot_download(monkeypatch, snap)
    assert hf_snapshot.complete_local_snapshot("baidu/Unlimited-OCR", PINNED) == snap
    assert calls == [{"repo_id": "baidu/Unlimited-OCR", "revision": PINNED, "local_files_only": True}]


@pytest.mark.parametrize(
    "missing",
    ["config.json", "tokenizer_config.json", "model-00002-of-000002.safetensors"],
)
def test_partial_snapshot_goes_back_to_the_hub(tmp_path, monkeypatch, missing):
    """중단된 다운로드(샤드·설정 누락)는 예전처럼 Hub 경로가 채운다 — 로컬로 고집하지 않는다."""
    files = {k: v for k, v in _COMPLETE.items() if k != missing}
    _fake_snapshot_download(monkeypatch, _snapshot_dir(tmp_path, files))
    assert hf_snapshot.complete_local_snapshot("m", PINNED) is None


def test_unindexed_snapshot_needs_at_least_one_weight_file(tmp_path, monkeypatch):
    files = {"config.json": "{}", "tokenizer_config.json": "{}"}
    snap = _snapshot_dir(tmp_path, files)
    _fake_snapshot_download(monkeypatch, snap)
    assert hf_snapshot.complete_local_snapshot("m", PINNED) is None
    (snap / "model.safetensors").write_text("w", encoding="utf-8")
    assert hf_snapshot.complete_local_snapshot("m", PINNED) == snap


def test_corrupt_index_or_missing_cache_means_no_local_snapshot(tmp_path, monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    files = {**_COMPLETE, "model.safetensors.index.json": "{not json"}
    _fake_snapshot_download(monkeypatch, _snapshot_dir(tmp_path, files))
    assert hf_snapshot.complete_local_snapshot("m", PINNED) is None

    calls = _fake_snapshot_download(monkeypatch, LocalEntryNotFoundError("not cached"))
    assert hf_snapshot.complete_local_snapshot("m", PINNED) is None
    assert len(calls) == 1


def test_branch_revision_never_resolves_a_local_snapshot(tmp_path, monkeypatch):
    calls = _fake_snapshot_download(monkeypatch, _snapshot_dir(tmp_path, _COMPLETE))
    assert hf_snapshot.complete_local_snapshot("m", "main") is None
    assert calls == []                      # Hub·캐시 어느 쪽도 조회하지 않고 기존 경로로
