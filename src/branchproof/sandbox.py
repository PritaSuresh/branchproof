"""Disposable Docker execution, with copied inputs and no host mounts.

The container protects the host. It cannot prove that hostile tests tell the
truth about their own outcomes. Treat images as trusted dependencies.
"""

from __future__ import annotations

import base64
import json
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from .workspace import repository_files

IMAGE: ContextVar[str | None] = ContextVar("test_image", default=None)
MAX_INPUT_BYTES = 32 * 1024 * 1024

# Executed using the image's interpreter, outside the copied repository. Test
# output is captured separately, so printed fake summaries are not our protocol.
WORKER = r'''
import base64, hashlib, json, os, pathlib, subprocess, sys
payload = json.load(sys.stdin)
root = pathlib.Path('/tmp/workspace')
root.mkdir()
for name, encoded in payload['files'].items():
    target = (root / name).resolve()
    if root not in target.parents:
        raise ValueError('invalid archive path')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(encoded))
def hashes():
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and not p.is_symlink()
            and not any(x in {'.pytest_cache', '__pycache__'} for x in p.parts)
            and not p.name.startswith('.coverage')}
before = hashes()
report = pathlib.Path('/tmp/outcomes.xml')
command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:randomly',
           '--no-header', '-o', 'addopts=', '-o', 'cache_dir=/tmp/pytest-cache',
           '--junitxml=' + str(report), '-o', 'junit_family=xunit2']
command += payload['extra']
if payload['targets']:
    command += ['--', *payload['targets']]
environment = {'PATH': os.environ['PATH'], 'HOME': '/tmp', 'LANG': 'C.UTF-8',
               'PYTHONPATH': str(root) + ':' + str(root / 'src'),
               'PYTHONDONTWRITEBYTECODE': '1'}
environment.update(payload['environment'])
try:
    run = subprocess.run(command, cwd=root, env=environment,
                         capture_output=True, text=True, timeout=payload['timeout'])
    result = {'returncode': run.returncode, 'stdout': run.stdout[-100000:],
              'stderr': run.stderr[-10000:], 'timed_out': False}
except subprocess.TimeoutExpired:
    result = {'returncode': -1, 'stdout': '', 'stderr': 'timed out', 'timed_out': True}
result['xml'] = report.read_text() if report.exists() else ''
result['files_changed'] = before != hashes()
coverage = pathlib.Path('/tmp/hardener-coverage.json')
if '--cov-report=json:/tmp/hardener-coverage.json' in payload['extra']:
    try:
        if coverage.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('coverage report exceeds 16 MiB')
        result['coverage_report'] = json.loads(coverage.read_text())
    except (OSError, ValueError):
        result['coverage_report'] = None
print(json.dumps(result))
'''


@contextmanager
def execution_image(image: str | None):
    token = IMAGE.set(image)
    try:
        yield
    finally:
        IMAGE.reset(token)


def container_command(image: str, name: str) -> list[str]:
    if not image or image.startswith("-"):
        raise ValueError("A valid sandbox image is required.")
    return [
        "docker", "run", "--rm", "--pull=never", "--name", name, "--interactive",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=1g", "--cpus=2",
        "--user=65534:65534", "--ulimit=nofile=1024:1024",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=512m,mode=1777",
        "--entrypoint=python", image, "-I", "-c", WORKER,
    ]


def resolve_image(image: str) -> str:
    """Pin a mutable local tag to its immutable image ID for the whole run."""
    if not image or image.startswith("-"):
        raise ValueError("Invalid sandbox image.")
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=15,
    )
    identity = result.stdout.strip()
    if result.returncode or not identity.startswith("sha256:"):
        raise ValueError("Sandbox image unavailable. Start Docker and build docker/Dockerfile. "
                         "There is no automatic fallback to host execution.")
    return identity


def run_container(root: Path, targets: list[str], extra: list[str], timeout: int,
                  environment: dict[str, str] | None = None) -> dict:
    image = IMAGE.get()
    if image is None:
        raise ValueError("No sandbox image selected.")
    files = {}
    size = 0
    for path in repository_files(root):
        relative = path.relative_to(root)
        if path.is_symlink() or any(
            part == ".git" or part.startswith(".env") for part in relative.parts
        ):
            continue
        size += path.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise ValueError("Repository exceeds the sandbox's 32 MiB input limit.")
        files[relative.as_posix()] = base64.b64encode(path.read_bytes()).decode()
    payload = json.dumps({"files": files, "targets": targets, "extra": extra,
                          "timeout": timeout, "environment": environment or {}})
    name = "hardener-" + uuid4().hex
    try:
        completed = subprocess.run(
            container_command(image, name), input=payload, text=True,
            capture_output=True, timeout=timeout + 20,
        )
        if completed.returncode:
            raise RuntimeError("Sandbox execution failed: " + completed.stderr[-1500:])
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "sandbox timed out",
                "timed_out": True, "xml": "", "files_changed": False}
    finally:
        # Remove precisely this invocation, including children after a timeout.
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)
