import atexit
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ── 테스트 격리: app 패키지를 import하기 **전에** 고정한다 ──
# 개발자의 실제 .env(실키)가 Settings.from_env()의 자동 탐색으로 테스트 프로세스
# (와 자식 프로세스) 환경에 주입되지 않게 끄고, 기본 DATA_DIR(backend/data — 개발
# 서버의 실잡 저장소)을 어떤 경로로도 건드리지 않도록 세션 전용 임시 디렉터리로
# 돌린다. 이미 설정돼 있어도 덮어쓴다 — 테스트가 실데이터를 고를 이유는 없다.
os.environ["DISABLE_DOTENV"] = "1"
# PyMuPDF 작업은 기본적으로 테스트 프로세스 안에서(inline) 실행한다 — 많은 테스트가
# pymupdf·pdf_export 내부를 monkeypatch하는데, 워커 프로세스에는 그 패치가 닿지 않는다.
# 프로세스 격리(시간 상한·비정상 종료·취소)는 pdf_worker_processes 픽스처를 쓰는 전용
# 테스트가 실제 워커로 검증한다(app/pipeline/pdf_worker.py).
os.environ["PDF_WORKER_MODE"] = "inline"
_SESSION_DATA_DIR = tempfile.mkdtemp(prefix="pdfocr-pytest-data-")
os.environ["DATA_DIR"] = _SESSION_DATA_DIR
atexit.register(shutil.rmtree, _SESSION_DATA_DIR, ignore_errors=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config as _config  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

# TestClient는 Host: testserver로 요청한다 — 프로덕션 기본값(ALLOWED_HOSTS)은 그대로 두고
# 테스트 프로세스의 기본 화이트리스트에만 추가한다 (Settings를 직접 생성하는 테스트 포함).
_config._DEFAULT_ALLOWED_HOSTS += ",testserver"


def make_pdf_bytes(pages: int = 3, with_image: bool = True) -> bytes:
    import fitz
    from PIL import Image, ImageDraw

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), f"Sample page {i + 1}", fontsize=24)
        page.insert_text((72, 120), "Unlimited-OCR pipeline test document.", fontsize=12)
        if with_image and i == 0:
            img = Image.new("RGB", (240, 140), (245, 245, 245))
            d = ImageDraw.Draw(img)
            for bi, h in enumerate((40, 90, 60, 110)):
                x = 20 + bi * 55
                d.rectangle((x, 130 - h, x + 40, 130), fill=(70, 90, 200))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            page.insert_image(fitz.Rect(72, 200, 372, 375), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf_bytes()


@pytest.fixture
def pdf_worker_processes(monkeypatch):
    """이 테스트에서만 PyMuPDF 작업을 실제 spawn 워커 프로세스로 격리한다.

    앞뒤로 풀을 닫아 다른 테스트의 워커·카운터가 섞이지 않게 한다. 워커는 테스트
    프로세스의 sys.path를 물려받으므로 tests/pdf_worker_tasks.py의 보조 작업도 실행할 수 있다.
    """
    from app.pipeline import pdf_worker

    pdf_worker.shutdown_pools()
    monkeypatch.setenv("PDF_WORKER_MODE", "process")
    yield pdf_worker
    pdf_worker.shutdown_pools()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        engine="fake",
        device="cpu",
        data_dir=tmp_path / "data",
        preload_model=False,
        fake_delay=0.0,
        frontend_dir=tmp_path / "no-frontend",  # 정적 마운트 비활성화
    )


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id: str, timeout: float = 15.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "error", "canceled"):
            return body
        time.sleep(0.03)
    raise AssertionError(f"잡이 제한시간 내에 끝나지 않음: {client.get(f'/api/jobs/{job_id}').json()}")
