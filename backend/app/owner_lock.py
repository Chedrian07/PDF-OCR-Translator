"""잡 저장소(`{DATA_DIR}/jobs`) 단일 소유자 락.

같은 jobs 디렉터리를 두 백엔드가 동시에 쓰면, 나중에 뜬 쪽의
`JobStore.load_existing()`이 먼저 뜬 쪽에서 **실행 중인** 잡을 '서버 재시작으로
중단'(error)으로 덮고 work/를 지운다. `make dev` 중에 돌린 `make test`, 같은
ocr-data 볼륨을 쓰는 compose 스택 동시 기동, `uvicorn --workers N`·WEB_CONCURRENCY가
모두 이 경로다. 그래서 디스크 상태를 바꾸기 전에 `{jobs}/.owner.lock`에 배타
flock을 **비차단**으로 잡아 앱 수명 동안 쥐고, 살아 있는 소유자가 이미 있으면
기동을 거부한다.

- flock은 열린 파일 기술(open file description)에 묶인다. 같은 프로세스라도 파일을
  다시 열어 잠그면 충돌하므로, 한 프로세스에서 앱을 여러 번 만드는 테스트에서도
  '살아 있는 두 앱'이 잡힌다. 닫힌 앱(lifespan 종료)은 release()로 즉시 놓는다.
- 프로세스가 죽으면 커널이 락을 회수한다 — stale 락을 정리할 필요가 없다. 같은
  호스트 커널을 쓰는 컨테이너끼리(named volume)도 동작한다.
- 락 파일은 지우지 않는다. 쥔 채로 지우면 다음 프로세스가 새 inode를 잠가 두
  소유자가 공존한다.
- fcntl이 없는 플랫폼(Windows)에서는 no-op이다. 락 파일을 열 수 없거나 flock을
  지원하지 않는 파일시스템이면 경고만 남기고 보호 없이 진행한다(기존 동작 유지).
"""

from __future__ import annotations

import gc
import logging
import os
import weakref
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None

logger = logging.getLogger(__name__)

LOCK_FILE_NAME = ".owner.lock"


class JobsDirInUseError(RuntimeError):
    """살아 있는 다른 백엔드(또는 같은 프로세스의 다른 앱)가 같은 잡 디렉터리를 소유 중이다."""


def _unlock_and_close(fd: int) -> None:
    # 닫기 전에 명시적으로 푼다 — fork로 fd를 물려받은 자식이 있어도 락이 그 자식의
    # 수명에 묶이지 않게(flock은 fd를 공유하는 모든 사본이 닫혀야 풀린다).
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        os.close(fd)


class JobsDirLock:
    """잡 디렉터리 소유권 핸들.

    release()는 멱등이다. 명시적으로 놓지 않은 핸들도 수거될 때 풀린다 —
    lifespan 없이 만들었다가 버린 앱이 락을 영구히 쥐고 있지 않게.
    """

    def __init__(self, path: Path, fd: int | None = None) -> None:
        self.path = path
        self._finalizer = (
            weakref.finalize(self, _unlock_and_close, fd) if fd is not None else None
        )

    @property
    def held(self) -> bool:
        """실제로 flock을 쥐고 있는지 (보호 없이 진행한 경우와 놓은 뒤에는 False)."""
        return self._finalizer is not None and self._finalizer.alive

    def release(self) -> None:
        if self._finalizer is not None:
            self._finalizer()  # 두 번째 호출부터는 no-op


def _flock_exclusive(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # 같은 프로세스에서 이미 버려졌지만 아직 수거되지 않은 앱(참조 순환)이 쥐고
        # 있을 수 있다 — 한 번 수거하고 다시 시도한다. 살아 있는 소유자는 그대로 남는다.
        gc.collect()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_jobs_dir_lock(jobs_dir: Path) -> JobsDirLock:
    """jobs_dir의 단일 소유권을 비차단으로 잡는다.

    살아 있는 다른 소유자가 있으면 JobsDirInUseError — 호출자는 디스크 상태를 바꾸기
    전에(load_existing 이전에) 불러야 한다.
    """
    path = jobs_dir / LOCK_FILE_NAME
    if fcntl is None:  # pragma: no cover — Windows
        return JobsDirLock(path)
    try:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except PermissionError:
            # 다른 uid(예: root로 돈 컨테이너)가 만든 락 파일 — flock은 읽기 전용
            # fd로도 잡히므로 보호를 포기하지 않는다.
            fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger.warning(
            "잡 디렉터리 소유 락 파일을 열 수 없어 단일 소유자 보호 없이 진행합니다: %s (%s)",
            path, e,
        )
        return JobsDirLock(path)
    try:
        _flock_exclusive(fd)
    except BlockingIOError:
        os.close(fd)
        raise JobsDirInUseError(
            f"다른 백엔드가 이미 이 잡 디렉터리를 사용 중입니다: {jobs_dir.resolve()} "
            "— 같은 DATA_DIR로 서버를 하나 더 띄우면 실행 중인 잡이 '서버 재시작으로 중단'으로 "
            "덮이고 작업 파일(work/)이 지워집니다. 먼저 떠 있는 서버를 종료하거나 DATA_DIR을 "
            "분리하세요(uvicorn --workers·WEB_CONCURRENCY 2 이상도 같은 충돌입니다)."
        ) from None
    except OSError as e:
        os.close(fd)
        logger.warning(
            "이 파일시스템에서는 flock을 쓸 수 없어 단일 소유자 보호 없이 진행합니다: %s (%s)",
            path, e,
        )
        return JobsDirLock(path)
    return JobsDirLock(path, fd)
