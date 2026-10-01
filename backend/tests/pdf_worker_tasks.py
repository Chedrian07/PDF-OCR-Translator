"""PDF 워커 풀 테스트용 보조 작업 — 워커 프로세스가 'pdf_worker_tasks:함수'로 임포트한다.

pytest가 tests/를 sys.path에 넣고, spawn 워커는 부모의 sys.path를 물려받는다.
파일 이름이 test_로 시작하지 않아 수집되지 않는다.
"""

from __future__ import annotations

import os
import signal
import sys
import time


def echo(value):
    return value


def pid() -> int:
    return os.getpid()


def sleep(seconds: float) -> int:
    time.sleep(seconds)
    return os.getpid()


def exit_now(code: int) -> None:
    os._exit(code)


def segfault() -> None:
    """MuPDF 크래시(SIGSEGV)와 같은 종료 — 파이썬 예외가 아니라 프로세스가 죽는다."""
    os.kill(os.getpid(), signal.SIGSEGV)
    time.sleep(5)


def raise_value_error(message: str) -> None:
    raise ValueError(message)


class _NeedsTwoArgs(Exception):
    """피클 왕복이 깨지는 예외 — 재구성 시 인자 하나로 __init__을 불러 TypeError가 난다."""

    def __init__(self, a, b):
        super().__init__(f"{a}/{b}")


def raise_unpicklable() -> None:
    raise _NeedsTwoArgs("left", "right")


def return_unpicklable():
    return lambda: None


def loaded_modules() -> list[str]:
    return sorted(sys.modules)


def alarm_remaining() -> float:
    """이 작업에 걸린 SIGALRM 자가 종료 타이머의 남은 시간(초)."""
    return signal.getitimer(signal.ITIMER_REAL)[0]


def rlimit_as() -> tuple[int, int]:
    import resource

    return resource.getrlimit(resource.RLIMIT_AS)


def oom_score_adj() -> str | None:
    try:
        with open("/proc/self/oom_score_adj", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def in_worker() -> bool:
    from app.pipeline import pdf_worker

    return pdf_worker.in_worker()


def sees_cancel(should_cancel=None) -> bool:
    """inline 모드의 cancel_kwarg 전달 확인용."""
    return should_cancel is not None and bool(should_cancel())


def cached_document_identity(pdf_path) -> tuple[int, int]:
    """(문서 객체 id, 페이지 수) — 같은 워커의 연속 작업이 열린 문서를 재사용하는지 확인용."""
    from app.pipeline import pdf_worker

    with pdf_worker.open_document(pdf_path) as doc:
        return id(doc), doc.page_count


def sleep_page(pdf_path, page_index: int, seconds: float = 30) -> int:
    """페이지 작업 꼴(run_page)의 느린 작업 — 페이지 격리 메모 확인용."""
    time.sleep(seconds)
    return page_index


# 워커가 실제로 실행하는 PDF 작업 모듈 — 이것들을 다 실어도 워커가 가벼워야 한다
TASK_MODULES = (
    "app.pipeline.pdf",
    "app.pipeline.fidelity",
    "app.pipeline.pdf_fonts",
    "app.pipeline.reading_order",
    "app.pipeline.pdf_export.build",
    "app.engine.textlayer",
)


def import_task_modules() -> list[str]:
    import importlib

    for name in TASK_MODULES:
        importlib.import_module(name)
    return sorted(sys.modules)


def temp_dir_then_sleep(report_path, seconds: float = 30) -> str:
    """tempfile로 임시 디렉터리를 만들고(폰트 서브셋처럼) 경로를 남긴 뒤 오래 잔다."""
    import tempfile
    from pathlib import Path

    made = tempfile.mkdtemp(prefix="uocr-font-")
    Path(report_path).write_text(made, encoding="utf-8")
    time.sleep(seconds)
    return made


def env_names() -> list[str]:
    import os

    return sorted(os.environ)
