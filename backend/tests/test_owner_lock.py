"""잡 디렉터리 단일 소유자 락 — 같은 DATA_DIR로 두 번째 백엔드가 뜨면 기동을 거부한다.

나중에 뜬 백엔드의 load_existing()이 먼저 뜬 쪽에서 실행 중인 잡을 '서버 재시작으로
중단'(error)으로 덮고 work/를 지웠다(make dev 중 make test, compose 스택 동시 기동).
감사: api-jobs-1, infra-docs-2
"""

import errno
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.owner_lock as owner_lock_mod
from app.main import create_app
from app.owner_lock import LOCK_FILE_NAME, JobsDirInUseError, acquire_jobs_dir_lock


def _running_job_on_disk(jobs_dir, job_id: str = "j_live"):
    """다른 백엔드에서 실행 중인 잡의 디스크 흔적 (meta=running + work/ 중간 산출물)."""
    job_dir = jobs_dir / job_id
    (job_dir / "work" / "chunk_00").mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "id": job_id, "filename": "live.pdf", "mode": "multi", "dpi": 200,
        "status": "running", "created_at": "2026-10-01T00:00:00+00:00",
        "progress": {}, "error": None, "warnings": [],
    }), encoding="utf-8")
    return job_dir


def test_살아있는_앱의_잡_디렉터리로는_두번째_앱이_기동을_거부한다(settings):
    with TestClient(create_app(settings)):
        job_dir = _running_job_on_disk(settings.jobs_dir)

        with pytest.raises(JobsDirInUseError) as excinfo:
            create_app(settings)

        assert str(settings.jobs_dir.resolve()) in str(excinfo.value)  # 디렉터리를 짚는다
        # 거부는 load_existing 이전이다 — 실행 중 잡의 meta와 work/가 그대로 남는다
        meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "running"
        assert (job_dir / "work" / "chunk_00").is_dir()


def test_종료된_앱의_잡_디렉터리는_다음_앱이_이어받는다(settings):
    first = create_app(settings)
    with TestClient(first):
        assert first.state.owner_lock.held
    assert not first.state.owner_lock.held  # lifespan 종료와 함께 놓았다

    second = create_app(settings)  # 같은 디렉터리로 재시작 — 거부되지 않는다
    with TestClient(second) as client:
        assert client.get("/api/jobs").json()["jobs"] == []  # 락 파일은 잡이 아니다
    # 락 파일은 지우지 않는다 — 쥔 채로 지우면 다음 소유자가 새 inode를 잠가 공존한다
    assert (settings.jobs_dir / LOCK_FILE_NAME).is_file()


def test_조립에_실패한_앱은_락을_남기지_않는다(settings):
    settings.device = "bogus"
    with pytest.raises(ValueError, match="OCR_DEVICE"):
        create_app(settings)

    settings.device = "cpu"
    with TestClient(create_app(settings)):
        pass


def test_시작하지_않고_버린_앱은_수거되며_락을_놓는다(settings):
    create_app(settings)  # lifespan 없이 버린다 — 참조 순환이면 즉시 해제되지 않는다
    with TestClient(create_app(settings)):
        pass


def test_다른_프로세스가_쥔_잡_디렉터리는_거부하고_그_프로세스가_끝나면_이어받는다(settings):
    settings.jobs_dir.mkdir(parents=True)
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import fcntl, os, sys
            fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print("locked", flush=True)
            sys.stdin.read()
        """), str(settings.jobs_dir / LOCK_FILE_NAME)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(JobsDirInUseError):
            create_app(settings)
    finally:
        holder.stdin.close()  # EOF → 소유 프로세스 종료
        holder.wait(timeout=10)
        holder.stdout.close()

    # 소유 프로세스가 끝나면 커널이 락을 회수한다 — stale 락 정리 없이 바로 기동
    with TestClient(create_app(settings)):
        pass


def test_release는_멱등이고_놓은_뒤_바로_다시_잡을_수_있다(tmp_path):
    lock = acquire_jobs_dir_lock(tmp_path)
    assert lock.held
    with pytest.raises(JobsDirInUseError):
        acquire_jobs_dir_lock(tmp_path)

    lock.release()
    lock.release()
    assert not lock.held

    again = acquire_jobs_dir_lock(tmp_path)
    assert again.held
    again.release()


def test_다른_uid가_만든_쓰기_불가_락_파일로도_보호한다(tmp_path):
    """root로 돈 컨테이너 등이 남긴 0444 락 파일 — flock은 읽기 전용 fd로도 잡힌다."""
    lock_file = tmp_path / LOCK_FILE_NAME
    lock_file.touch()
    lock_file.chmod(0o444)

    lock = acquire_jobs_dir_lock(tmp_path)
    try:
        assert lock.held
        with pytest.raises(JobsDirInUseError):
            acquire_jobs_dir_lock(tmp_path)
    finally:
        lock.release()
        lock_file.chmod(0o644)


def test_fcntl이_없는_플랫폼은_보호_없이_기존_동작(settings, monkeypatch):
    monkeypatch.setattr(owner_lock_mod, "fcntl", None)

    first = create_app(settings)
    second = create_app(settings)  # 거부하지 않는다 (Windows 등)

    assert not first.state.owner_lock.held
    assert not second.state.owner_lock.held


def test_flock을_지원하지_않는_파일시스템은_경고하고_진행한다(tmp_path, monkeypatch, caplog):
    def unsupported(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=unsupported)
    monkeypatch.setattr(owner_lock_mod, "fcntl", fake_fcntl)

    lock = acquire_jobs_dir_lock(tmp_path)

    assert not lock.held
    assert any("flock" in r.getMessage() for r in caplog.records)
