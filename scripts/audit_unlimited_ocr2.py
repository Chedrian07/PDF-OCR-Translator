"""Strictly audit the sidecar lock and the hash-verified custom SGLang wheel.

PyPI cannot audit URL requirements. Keep its hash checks for every index package,
and query OSV for the custom wheel's exact, independently verified METADATA version.
No numerical dependency is installed or modified by this script.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from email.parser import BytesParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "services/unlimited_ocr2"
PIP_AUDIT = "pip-audit==2.10.1"
DISKCACHE_ADVISORIES = frozenset(
    {"PYSEC-2026-2447", "CVE-2025-69872", "GHSA-w8v5-vhqr-4h9v"}
)


def requirements(text: str) -> list[str]:
    result, pending = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pending.append(line.removesuffix("\\").rstrip())
        if not line.endswith("\\"):
            result.append(" ".join(pending))
            pending = []
    if pending:
        raise ValueError("Unfinished locked requirement")
    return result


def validate_policy(
    policy: dict, locked: list[str], today: date | None = None
) -> list[str]:
    today = today or datetime.now(timezone.utc).date()
    if today >= date.fromisoformat(policy["expires"]):
        raise ValueError(
            "DiskCache acceptance expired; review the upstream fix and OCR-only mitigation"
        )
    cache_requirements = [
        item
        for item in locked
        if re.match(r"diskcache(?:==|\s*@)", item, re.IGNORECASE)
    ]
    if (
        policy.get("package") != "diskcache"
        or policy.get("version") != "5.6.3"
        or set(policy.get("advisories", [])) != DISKCACHE_ADVISORIES
        or len(cache_requirements) != 1
        or not re.match(r"diskcache==5\.6\.3(?:\s|$)", cache_requirements[0])
    ):
        raise ValueError(
            "DiskCache acceptance applies only to the reviewed 5.6.3 advisory"
        )
    return sorted(DISKCACHE_ADVISORIES)


def wheel_reference(
    locked: list[str], canonical: str, launcher: str
) -> tuple[str, str, str]:
    direct = [item for item in locked if " @ " in item]
    if len(direct) != 1:
        raise ValueError("Expected exactly the reviewed SGLang URL requirement")
    match = re.fullmatch(
        r"sglang @ (\S+) ((?:--hash=sha256:[0-9a-f]{64}\s*)+)", direct[0]
    )
    expected = requirements(canonical)
    if (
        not match
        or len(expected) != 1
        or expected[0] != "sglang[fastokens] @ " + match[1]
    ):
        raise ValueError("Custom wheel lock differs from requirements.in")
    url = urllib.parse.urlsplit(match[1])
    wheel = re.fullmatch(
        r"/baidu/Unlimited-OCR-2/resolve/([0-9a-f]{40})/wheel/sglang-(.+)-py3-none-any\.whl",
        urllib.parse.unquote(url.path),
    )
    fragment = re.fullmatch(r"sha256=([0-9a-f]{64})", url.fragment)
    revisions = [
        ast.literal_eval(node.value)
        for node in ast.parse(launcher).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REVISION"
            for target in node.targets
        )
    ]
    if (
        url.scheme != "https"
        or url.netloc != "huggingface.co"
        or url.query
        or not wheel
        or not fragment
        or revisions != [wheel[1]]
        or set(re.findall(r"--hash=sha256:([0-9a-f]{64})", match[2])) != {fragment[1]}
    ):
        raise ValueError(
            "Custom wheel must retain the official immutable URL and SHA-256 pin"
        )
    return urllib.parse.urlunsplit(url._replace(fragment="")), fragment[1], wheel[2]


def validate_reviewed_wheel(policy: dict, digest: str, version: str) -> None:
    if policy.get("reviewed_wheel") != {"version": version, "sha256": digest}:
        raise ValueError(
            "Custom wheel changed; re-review the DiskCache reachability acceptance"
        )


def verified_wheel_version(path: Path, digest: str, expected: str) -> str:
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("Custom wheel SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        metadata = [
            item
            for item in archive.infolist()
            if item.filename.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1 or metadata[0].file_size > 1024 * 1024:
            raise ValueError("Custom wheel must contain one bounded METADATA record")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
    if message.get_all("Name") != ["sglang"] or message.get_all("Version") != [
        expected
    ]:
        raise ValueError("Custom wheel METADATA does not match its pinned identity")
    return expected


def main() -> int:
    locked = requirements((SERVICE / "requirements.lock").read_text())
    policy = json.loads((SERVICE / "audit-policy.json").read_text())
    accepted = validate_policy(policy, locked)
    url, digest, expected = wheel_reference(
        locked,
        (SERVICE / "requirements.in").read_text(),
        (SERVICE / "launch.py").read_text(),
    )
    validate_reviewed_wheel(policy, digest, expected)
    with tempfile.TemporaryDirectory(prefix="uocr2-audit-") as directory:
        directory = Path(directory)
        wheel = directory / "sglang.whl"
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            wheel.open("wb") as output,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise ValueError(
                        "Custom wheel exceeds the reviewed 64 MiB download limit"
                    )
                output.write(chunk)
        version = verified_wheel_version(wheel, digest, expected)
        index_requirements = directory / "index.txt"
        index_requirements.write_text(
            "\n".join(item for item in locked if " @ " not in item) + "\n"
        )
        custom_requirements = directory / "custom.txt"
        custom_requirements.write_text(f"sglang=={version}\n")
        base = ["uvx", PIP_AUDIT, "--disable-pip", "--no-deps", "--strict"]
        exceptions = [
            value for advisory in accepted for value in ("--ignore-vuln", advisory)
        ]
        print(
            f"DiskCache 5.6.3 acceptance expires {policy['expires']}: {policy['reason']}",
            flush=True,
        )
        index_result = subprocess.run(
            [*base, "-r", str(index_requirements), *exceptions], check=False
        )
        print(
            f"Auditing hash-verified custom wheel: sglang=={version} (OSV)", flush=True
        )
        custom_result = subprocess.run(
            [
                *base,
                "-r",
                str(custom_requirements),
                "--vulnerability-service",
                "osv",
            ],
            check=False,
        )
        return 1 if index_result.returncode or custom_result.returncode else 0


if __name__ == "__main__":
    raise SystemExit(main())
