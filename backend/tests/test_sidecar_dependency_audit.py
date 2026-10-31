"""The custom wheel and narrowly accepted advisory must never become silent skips."""

import hashlib
import importlib.util
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "services/unlimited_ocr2"
spec = importlib.util.spec_from_file_location(
    "_test_sidecar_audit", REPO / "scripts/audit_unlimited_ocr2.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def policy():
    return json.loads((SERVICE / "audit-policy.json").read_text())


def test_exception_expires_and_is_scoped_to_one_exact_sidecar_package(policy):
    assert policy["expires"] == "2026-11-30"
    assert policy["package"] == "diskcache" and policy["version"] == "5.6.3"
    assert audit.validate_policy(policy, ["diskcache==5.6.3"], date(2026, 11, 29))
    for expired in (date(2026, 11, 30), date(2026, 12, 1)):
        with pytest.raises(ValueError, match="expired"):
            audit.validate_policy(policy, ["diskcache==5.6.3"], expired)


@pytest.mark.parametrize(
    "locked", [[], ["diskcache==5.6.2"], ["diskcache==5.6.3", "diskcache==5.6.2"]]
)
def test_exception_cannot_follow_a_different_or_duplicate_dependency(policy, locked):
    with pytest.raises(ValueError, match="only to the reviewed"):
        audit.validate_policy(policy, locked, date(2026, 10, 31))


def test_exception_cannot_ignore_unrelated_advisories(policy):
    policy["advisories"].append("CVE-unrelated")
    with pytest.raises(ValueError, match="only to the reviewed"):
        audit.validate_policy(policy, ["diskcache==5.6.3"], date(2026, 10, 31))


def test_custom_wheel_pin_matches_canonical_input_and_launcher_revision():
    locked = audit.requirements((SERVICE / "requirements.lock").read_text())
    url, digest, version = audit.wheel_reference(
        locked,
        (SERVICE / "requirements.in").read_text(),
        (SERVICE / "launch.py").read_text(),
    )
    assert version == "0.5.20.dev13+g75b9b06b02"
    assert digest == "576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea"
    assert "c14071b549f4fcab10909f3c4acc6a2edb366947/wheel/" in url


def test_custom_wheel_lock_cannot_substitute_another_url_or_hash():
    locked = audit.requirements((SERVICE / "requirements.lock").read_text())
    canonical = (SERVICE / "requirements.in").read_text()
    launcher = (SERVICE / "launch.py").read_text()
    for replacement in ("http://huggingface.co", "https://other.example"):
        changed = [
            item.replace("https://huggingface.co", replacement) for item in locked
        ]
        with pytest.raises(ValueError):
            audit.wheel_reference(
                changed,
                canonical.replace("https://huggingface.co", replacement),
                launcher,
            )
    changed = [
        item.replace("--hash=sha256:576c", "--hash=sha256:0000") for item in locked
    ]
    with pytest.raises(ValueError):
        audit.wheel_reference(changed, canonical, launcher)


@pytest.mark.parametrize("change", ["digest", "version", "both"])
def test_diskcache_acceptance_does_not_follow_a_consistently_relocked_custom_wheel(
    policy, change
):
    locked_text = (SERVICE / "requirements.lock").read_text()
    canonical = (SERVICE / "requirements.in").read_text()
    launcher = (SERVICE / "launch.py").read_text()
    old_revision = "c14071b549f4fcab10909f3c4acc6a2edb366947"
    locked_text = locked_text.replace(old_revision, "2" * 40)
    canonical = canonical.replace(old_revision, "2" * 40)
    launcher = launcher.replace(old_revision, "2" * 40)
    if change in ("digest", "both"):
        old = policy["reviewed_wheel"]["sha256"]
        locked_text = locked_text.replace(old, "1" * 64)
        canonical = canonical.replace(old, "1" * 64)
    if change in ("version", "both"):
        old = policy["reviewed_wheel"]["version"].replace("+", "%2B")
        locked_text = locked_text.replace(old, "0.5.21")
        canonical = canonical.replace(old, "0.5.21")
    _, digest, version = audit.wheel_reference(
        audit.requirements(locked_text), canonical, launcher
    )
    with pytest.raises(ValueError, match="re-review"):
        audit.validate_reviewed_wheel(policy, digest, version)


def test_reviewed_wheel_policy_requires_both_identity_fields(policy):
    wheel = policy["reviewed_wheel"]
    audit.validate_reviewed_wheel(policy, wheel["sha256"], wheel["version"])
    del policy["reviewed_wheel"]["sha256"]
    with pytest.raises(ValueError, match="re-review"):
        audit.validate_reviewed_wheel(policy, wheel.get("sha256", ""), wheel["version"])


@pytest.mark.parametrize(
    "name,version",
    [("sglang", "0.5.20.dev13+g75b9b06b02"), ("other", "1.0"), ("sglang", "0.5.9")],
)
def test_custom_wheel_metadata_identity_and_bytes_are_both_verified(
    tmp_path, name, version
):
    wheel = tmp_path / "custom.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "sglang.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n"
        )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    expected = "0.5.20.dev13+g75b9b06b02"
    if name == "sglang" and version == expected:
        assert audit.verified_wheel_version(wheel, digest, expected) == expected
    else:
        with pytest.raises(ValueError, match="METADATA"):
            audit.verified_wheel_version(wheel, digest, expected)
    with pytest.raises(ValueError, match="SHA-256"):
        audit.verified_wheel_version(wheel, "0" * 64, expected)


def test_custom_and_index_audits_are_strict_and_no_blanket_ignores_are_used(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        audit.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wheel")
    )
    monkeypatch.setattr(
        audit, "verified_wheel_version", lambda path, digest, version: version
    )
    monkeypatch.setattr(
        audit, "validate_policy", lambda *a: sorted(audit.DISKCACHE_ADVISORIES)
    )

    def run(command, **kwargs):
        requirements = Path(command[command.index("-r") + 1]).read_text()
        calls.append((command, requirements))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audit.subprocess, "run", run)
    assert audit.main() == 0
    assert len(calls) == 2
    for command, _ in calls:
        assert (
            "--strict" in command
            and "--disable-pip" in command
            and "--no-deps" in command
        )
        assert "--ignore-unfixed" not in command
    index, custom = calls
    assert " @ " not in index[1] and "--hash=sha256:" in index[1]
    assert custom[1] == "sglang==0.5.20.dev13+g75b9b06b02\n"
    assert custom[0][custom[0].index("--vulnerability-service") + 1] == "osv"
    assert "--ignore-vuln" not in custom[0]
    assert {
        index[0][i + 1] for i, value in enumerate(index[0]) if value == "--ignore-vuln"
    } == audit.DISKCACHE_ADVISORIES
    backend_script = (REPO / "scripts/dependency_audit.sh").read_text()
    assert (
        "PYSEC-2026-2447" not in backend_script
        and "CVE-2025-69872" not in backend_script
    )
