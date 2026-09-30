"""The deployed API must identify the same version as the release package."""

import tomllib
from pathlib import Path

from app import __version__


def test_openapi_reports_package_version(client):
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    assert __version__ == metadata["project"]["version"]
    assert client.get("/openapi.json").json()["info"]["version"] == __version__
