"""import·테스트 격리 계약 — app.main import만으로는 아무 부작용이 없어야 한다.

예전에는 app/main.py 끝의 `app = create_app()`이 import 시점에 돌았다. conftest가
create_app을 import하는 것만으로 개발자의 실제 .env(실키)가 테스트 프로세스에
주입됐고, 개발 서버가 쓰는 DATA_DIR(backend/data/jobs)에 load_existing()이 돌아
실행 중 잡을 error로 덮고 work/를 지웠다.
감사: tests-baseline-1, infra-docs-8, api-jobs-1
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_테스트_세션은_실제_dotenv와_개발_DATA_DIR을_쓰지_않는다():
    """conftest가 app import 전에 격리를 건다 — 지우면 위 회귀가 조용히 돌아온다."""
    assert os.environ.get("DISABLE_DOTENV") == "1"
    data_dir = Path(os.environ["DATA_DIR"]).resolve()
    dev_store = (BACKEND / "data").resolve()  # make dev·uvicorn 기본 DATA_DIR(실잡 저장소)
    assert not data_dir.is_relative_to(dev_store)
    assert not dev_store.is_relative_to(data_dir)


def test_app_main은_import만으로_앱을_만들지_않고_uvicorn이_찾을_때_만든다(tmp_path):
    data_dir = tmp_path / "data"
    probe = textwrap.dedent("""
        import os
        import app.main as m

        assert "app" not in vars(m), "import만으로 기본 앱이 만들어졌다"
        assert not os.path.exists(os.environ["DATA_DIR"]), "import만으로 DATA_DIR을 건드렸다"

        from uvicorn.importer import import_from_string

        built = import_from_string("app.main:app")  # uvicorn app.main:app과 같은 경로
        assert built is m.app is import_from_string("app.main:app"), "앱은 한 번만 만든다"
        assert built.state.owner_lock.held, "기본 앱도 잡 디렉터리 소유 락을 쥔다"
    """)
    env = {
        **os.environ,
        "PYTHONPATH": str(BACKEND),
        "DATA_DIR": str(data_dir),
        "OCR_ENGINE": "fake",
        "OCR_DEVICE": "cpu",
        "PRELOAD_MODEL": "0",
    }

    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert (data_dir / "jobs").is_dir()  # 속성 접근 시점에야 만들어졌다
