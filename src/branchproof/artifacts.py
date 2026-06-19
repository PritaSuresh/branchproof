"""Durable, separate outputs for each invocation. No result replaces an earlier run."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .sandbox import IMAGE


def new_run_directory(parent: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{stamp}-{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    fingerprint = hashlib.sha256()
    for module in sorted(source.glob("*.py")):
        fingerprint.update(module.name.encode() + b"\0" + module.read_bytes())
    (path / "metadata.json").write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "harness_sha256": fingerprint.hexdigest(),
        "schema_version": 2,
        "sandbox_image": IMAGE.get(),
    }, indent=2) + "\n")
    return path
