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
    def __init__(self, fail_local: bool | BaseException = False) -> None:
        self.calls: list[dict] = []
        self.fail_local = fail_local

    def __call__(self, model_id, **kwargs):
        self.calls.append({"model_id": model_id, **kwargs})
        if kwargs.get("local_files_only") and self.fail_local:
            if isinstance(self.fail_local, BaseException):
                raise self.fail_local
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


@pytest.mark.parametrize(
    "error",
    [
        # tokenizer.json이 빠진 부분 캐시 — AutoTokenizer가 느린 토크나이저 변환으로 넘어간다
        ImportError("LlamaConverter requires the protobuf library but it was not found"),
        # tokenizer_config.json이 빠진 부분 캐시 — config.json의 auto_map 때문에 원격 코드를 묻는다
        ValueError("The repository contains custom code which must be executed"),
        RuntimeError("anything else the cache-only attempt can raise"),
    ],
    ids=["import-error", "value-error", "runtime-error"],
)
def test_any_cache_only_failure_falls_back_to_the_hub_once(error):
    """OSError만 폴백하던 때는 부분 캐시의 ImportError·ValueError가 그대로 올라가 Hub가 빠진
    파일을 채우지 못했다 — 기동마다 같은 오류로 엔진 로드가 영구 실패했다(delta-core-1)."""
    load = _Recorder(fail_local=error)
    assert hf_snapshot.pretrained_local_first(load, "m", PINNED, dtype="x") == "loaded:m"
    assert load.calls == [
        {"model_id": "m", "revision": PINNED, "local_files_only": True, "dtype": "x"},
        {"model_id": "m", "revision": PINNED, "dtype": "x"},
    ]


def test_a_real_error_still_surfaces_from_the_hub_call():
    """폴백은 한 번뿐이다 — Hub 호출도 실패하면 그 예외가 그대로 올라간다(삼키지 않는다)."""
    calls: list[bool] = []

    def load(model_id, **kwargs):
        calls.append(bool(kwargs.get("local_files_only")))
        raise ValueError(f"broken weights ({'cache' if kwargs.get('local_files_only') else 'hub'})")

    with pytest.raises(ValueError, match=r"broken weights \(hub\)"):
        hf_snapshot.pretrained_local_first(load, "m", PINNED)
    assert calls == [True, False]


def _hf_cache(root: Path, files: dict[str, str], no_exist: tuple[str, ...] = ()) -> Path:
    """HF 허브 캐시 배치(models--org--tok/snapshots/<rev>/…) — 중단된 다운로드 모양을 만든다.

    .no_exist/<rev>/<파일>은 Hub가 '그 파일은 없다'고 답했다는 표식이다(실제 중단 상태와 같게)."""
    repo = root / "models--org--tok"
    snap = repo / "snapshots" / PINNED
    snap.mkdir(parents=True)
    for name, body in files.items():
        (snap / name).write_text(body, encoding="utf-8")
    marks = repo / ".no_exist" / PINNED
    marks.mkdir(parents=True)
    for name in no_exist:
        (marks / name).write_text("", encoding="utf-8")
    return root


def _tiny_tokenizer_json() -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok.to_str()


def _cache_then_hub_loader(attempts: list[str]):
    """캐시 시도는 진짜 AutoTokenizer로, Hub 시도는 실행하지 않고(네트워크 없음) 도달만 기록한다."""
    from transformers import AutoTokenizer

    def load(model_id, **kwargs):
        if kwargs.get("local_files_only"):
            attempts.append("cache")
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        attempts.append("hub")
        return "hub-tokenizer"

    return load


@pytest.fixture
def _no_remote_code_prompt(monkeypatch):
    """원격 코드 확인 프롬프트(input, 15초 대기) 대신 바로 ValueError — 서버 스레드와 같은 결과."""
    pytest.importorskip("transformers")
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "TIME_OUT_REMOTE_CODE", 0)


def test_complete_tiny_tokenizer_cache_loads_without_the_hub(tmp_path, _no_remote_code_prompt):
    """양성 대조 — 아래 부분 캐시 테스트의 캐시 배치가 맞다는 것(완전하면 캐시에서 끝난다)."""
    cache = _hf_cache(tmp_path, {
        "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
        "tokenizer.json": _tiny_tokenizer_json(),
    })
    attempts: list[str] = []
    tok = hf_snapshot.pretrained_local_first(
        _cache_then_hub_loader(attempts), "org/tok", PINNED, cache_dir=str(cache),
    )
    assert attempts == ["cache"]
    assert tok("hello world")["input_ids"] == [1, 2]


@pytest.mark.parametrize(
    ("files", "no_exist"),
    [
        # 중단된 첫 다운로드: tokenizer_config.json만 받았고 tokenizer.json은 .incomplete로 남았다
        ({"tokenizer_config.json": json.dumps({"tokenizer_class": "LlamaTokenizerFast"})},
         ("tokenizer.model",)),
        # tokenizer_config.json도 없고 config.json만 있다(auto_map → 원격 코드 확인)
        ({"config.json": json.dumps({"model_type": "unlimited-ocr-test",
                                     "auto_map": {"AutoConfig": "modeling_x.XConfig"}})},
         ()),
    ],
    ids=["tokenizer-json-missing", "tokenizer-config-missing"],
)
def test_partial_tokenizer_cache_falls_back_to_the_hub_with_the_real_auto_tokenizer(
    tmp_path, _no_remote_code_prompt, files, no_exist,
):
    """진짜 AutoTokenizer는 이런 부분 캐시에서 OSError가 아닌 예외(ImportError(protobuf)·
    ValueError(trust_remote_code))를 낸다 — 그래도 Hub 호출이 빠진 파일을 채우러 가야 한다."""
    cache = _hf_cache(tmp_path, files, no_exist)
    attempts: list[str] = []
    out = hf_snapshot.pretrained_local_first(
        _cache_then_hub_loader(attempts), "org/tok", PINNED, cache_dir=str(cache),
    )
    assert out == "hub-tokenizer"
    assert attempts == ["cache", "hub"]


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
    ["config.json", "tokenizer_config.json", "tokenizer.json",
     "model-00002-of-000002.safetensors"],
)
def test_partial_snapshot_goes_back_to_the_hub(tmp_path, monkeypatch, missing):
    """중단된 다운로드(샤드·설정·토크나이저 누락)는 예전처럼 Hub 경로가 채운다 — 로컬로 고집하지
    않는다. tokenizer.json이 빠진 캐시를 완전하다고 보면 MLX 로더가 6.7GB 가중치를 다 올린 뒤
    토크나이저에서 ImportError(protobuf)로 실패했다(delta-core-1)."""
    files = {k: v for k, v in _COMPLETE.items() if k != missing}
    _fake_snapshot_download(monkeypatch, _snapshot_dir(tmp_path, files))
    assert hf_snapshot.complete_local_snapshot("m", PINNED) is None


def test_sentencepiece_tokenizer_model_also_completes_the_snapshot(tmp_path, monkeypatch):
    """토크나이저 본체는 tokenizer.json 또는 tokenizer.model(sentencepiece 원본) 하나면 된다."""
    files = {k: v for k, v in _COMPLETE.items() if k != "tokenizer.json"}
    snap = _snapshot_dir(tmp_path, {**files, "tokenizer.model": "spm"})
    _fake_snapshot_download(monkeypatch, snap)
    assert hf_snapshot.complete_local_snapshot("m", PINNED) == snap


def test_unindexed_snapshot_needs_at_least_one_weight_file(tmp_path, monkeypatch):
    files = {"config.json": "{}", "tokenizer_config.json": "{}", "tokenizer.json": "{}"}
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
